"""
Combined PubMed pipeline: fetch abstracts (NCBI), Alpaca dataset (gpt-4o-mini), Pinecone RAG ingest.

Run from project root:
  python -m backend.scripts.generate_dataset [--phase fetch|alpaca|pinecone] [--dry-run] [--resume]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Paths ---
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "backend" / "data"
RAW_PATH = DATA_DIR / "raw_abstracts.json"
DATASET_PATH = DATA_DIR / "peptide_dataset.json"
CHECKPOINT_PATH = DATA_DIR / "peptide_dataset_checkpoint.json"

# --- Constants (plan) ---
PEPTIDES: list[str] = [
    "BPC-157",
    "TB-500",
    "Thymosin Beta-4",
    "Semax",
    "Selank",
    "GHK-Cu",
    "Ipamorelin",
    "CJC-1295",
    "Epithalon",
    "PT-141",
    "DSIP",
    "Hexarelin",
    "GHRP-6",
    "Melanotan",
    "Tesamorelin",
    "Sermorelin",
    "AOD-9604",
    "IGF-1 LR3",
    "MGF",
    "Kisspeptin",  # 20th peptide (balances 20 × 100 Alpaca target)
]

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
RATE_SLEEP = 0.34
FETCH_CAP_FULL = 250
FETCH_CAP_DRY = 3
ALPACA_FULL = 100
ALPACA_DRY = 3
CHECKPOINT_EVERY = 100
CHUNK_TOKENS = 200
TEACHER_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
# Pinecone metadata size limit per field — truncate stored chunk text (re-run --phase pinecone to backfill).
PINECONE_METADATA_TEXT_MAX = 32000

ALPACA_RELEVANCE_KEYWORDS: tuple[str, ...] = (
    "mechanism",
    "pathway",
    "receptor",
    "signaling",
    "therapeutic",
    "treatment",
    "healing",
    "repair",
    "anti-inflammatory",
    "angiogenesis",
    "cytokine",
    "dosing",
    "administration",
    "pharmacokinetics",
    "efficacy",
    "safety",
    "contraindication",
    "toxicity",
    "neuroprotective",
    "regenerative",
    "synthesis",
    "inhibit",
    "upregulate",
    "downregulate",
    "modulate",
    "activate",
    "binding",
    "clinical",
    "preclinical",
    "trial",
)

PEPTIDE_ALIASES: dict[str, list[str]] = {
    "BPC-157": ["bpc-157", "bpc 157", "body protection compound"],
    "TB-500": ["tb-500", "tb500", "thymosin beta-4 fragment"],
    "Thymosin Beta-4": ["thymosin beta-4", "thymosin β4", "tβ4", "tmsb4"],
    "Semax": ["semax", "acth 4-10", "semax peptide"],
    "Selank": ["selank", "selank peptide", "tuftsin analog"],
    "GHK-Cu": ["ghk-cu", "ghk copper", "glycyl-l-histidyl"],
    "Ipamorelin": ["ipamorelin", "ipamorelin peptide"],
    "CJC-1295": ["cjc-1295", "cjc1295", "ghrh analog"],
    "Epithalon": ["epithalon", "epitalon", "epithalone", "aedg peptide"],
    "PT-141": ["pt-141", "bremelanotide", "melanocortin receptor"],
    "DSIP": ["delta sleep-inducing peptide", "delta sleep peptide", "dsip peptide"],
    "Hexarelin": ["hexarelin", "examorelin", "hexarelin peptide"],
    "GHRP-6": ["ghrp-6", "ghrp6", "growth hormone releasing peptide 6"],
    "Melanotan": ["melanotan", "melanotan ii", "melanotan-2", "mt-ii"],
    "Tesamorelin": ["tesamorelin", "tesamorelin peptide"],
    "Sermorelin": ["sermorelin", "sermorelin acetate", "ghrh 1-29"],
    "AOD-9604": ["aod-9604", "aod9604", "anti-obesity drug 9604"],
    "IGF-1 LR3": ["igf-1 lr3", "igf1-lr3", "insulin-like growth factor"],
    "MGF": ["mechano growth factor", "mechano-growth factor", "igf-1ec"],
    "Kisspeptin": ["kisspeptin", "kisspeptin-10", "kp-10", "kiss1", "gpr54"],
}


def _strip_ns(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def parse_pubmed_fetch_xml(content: bytes) -> list[tuple[str, str]]:
    """Return list of (pmid, abstract_text) from efetch XML."""
    root = ET.fromstring(content)
    results: list[tuple[str, str]] = []
    for article in root.iter():
        if _strip_ns(article.tag) != "PubmedArticle":
            continue
        pmid: str | None = None
        abstract_chunks: list[str] = []
        for el in article.iter():
            tag = _strip_ns(el.tag)
            if tag == "PMID" and el.text:
                pmid = el.text.strip()
            elif tag == "AbstractText":
                chunk = ""
                if el.text:
                    chunk += el.text
                for child in el:
                    if child.text:
                        chunk += child.text
                    if child.tail:
                        chunk += child.tail
                chunk = chunk.strip()
                if chunk:
                    abstract_chunks.append(chunk)
        if pmid and abstract_chunks:
            results.append((pmid, " ".join(abstract_chunks)))
    return results


def ncbi_get(session: requests.Session, url: str) -> requests.Response:
    time.sleep(RATE_SLEEP)
    r = session.get(url, timeout=120)
    r.raise_for_status()
    return r


def phase_fetch(
    session: requests.Session,
    email: str,
    dry_run: bool,
) -> list[dict[str, Any]]:
    from tqdm import tqdm

    cap = FETCH_CAP_DRY if dry_run else FETCH_CAP_FULL
    tool = "peptiscout_dataset"
    records: list[dict[str, Any]] = []
    seen_pair: set[tuple[str, str]] = set()

    for peptide in tqdm(PEPTIDES, desc="Phase 1: peptides"):
        term = urllib.parse.quote_plus(peptide)
        search_url = (
            f"{NCBI_BASE}/esearch.fcgi?db=pubmed&term={term}&retmax={cap}"
            f"&retmode=json&tool={tool}&email={urllib.parse.quote_plus(email)}"
        )
        r = ncbi_get(session, search_url)
        data = r.json()
        idlist = data.get("esearchresult", {}).get("idlist", [])
        if not idlist:
            logger.warning("No PMIDs for peptide query: %s", peptide)
            continue

        # Batch efetch (max ~200 IDs per request)
        batch_size = 200
        for i in range(0, len(idlist), batch_size):
            batch = idlist[i : i + batch_size]
            ids = ",".join(batch)
            fetch_url = (
                f"{NCBI_BASE}/efetch.fcgi?db=pubmed&id={ids}&rettype=abstract&retmode=xml"
                f"&tool={tool}&email={urllib.parse.quote_plus(email)}"
            )
            fr = ncbi_get(session, fetch_url)
            for pmid, text in parse_pubmed_fetch_xml(fr.content):
                key = (peptide, pmid)
                if key in seen_pair:
                    continue
                seen_pair.add(key)
                records.append({"pmid": pmid, "text": text, "peptide": peptide})

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "records": records}
    RAW_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %s (%d records)", RAW_PATH, len(records))
    return records


def load_raw_records() -> list[dict[str, Any]]:
    if not RAW_PATH.is_file():
        logger.error(
            "Missing %s — run: python -m backend.scripts.generate_dataset --phase fetch",
            RAW_PATH,
        )
        sys.exit(1)
    data = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    return data.get("records", data) if isinstance(data, dict) else data


OUTPUT_SECTIONS = ("[MOA]:", "[Co-factors]:", "[Benefits]:", "[Risks]:")


def validate_alpaca_output(output: str) -> bool:
    o = output.strip()
    return all(marker in o for marker in OUTPUT_SECTIONS)


def normalize_teacher_output(output: Any) -> str:
    """If the model nests sections under output as an object, flatten to one string."""
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        d = output

        def norm_key(k: str) -> str:
            return "".join(c for c in k.lower() if c.isalnum())

        norm_index = {norm_key(str(k)): v for k, v in d.items()}

        def pick(candidates: tuple[str, ...]) -> str:
            for k in candidates:
                if k in d:
                    v = d[k]
                    return str(v).strip() if v is not None else ""
            for k in candidates:
                nk = norm_key(k)
                if nk in norm_index:
                    v = norm_index[nk]
                    return str(v).strip() if v is not None else ""
            return ""

        moa = pick(("MOA", "moa", "[MOA]"))
        cof = pick(("Co-factors", "Co_factors", "cofactors", "[Co-factors]"))
        ben = pick(("Benefits", "benefits", "[Benefits]"))
        ris = pick(("Risks", "risks", "[Risks]"))
        return f"[MOA]: {moa} [Co-factors]: {cof} [Benefits]: {ben} [Risks]: {ris}"
    return str(output)


def call_teacher(client: Any, abstract_text: str, peptide: str) -> dict[str, Any] | None:
    system = (
        "You convert biomedical abstracts into Alpaca training data. "
        "Reply with a single JSON object only, no markdown. "
        'The "output" value must be one plain string, not a nested JSON object.'
    )
    user = f"""Convert this PubMed abstract into an Alpaca instruction-response pair.

Peptide context (search term): {peptide}

Abstract:
{abstract_text[:12000]}

The JSON must have keys: "instruction", "input", "output".
- "instruction": a question about MOA, co-factors, benefits, or risks for the peptide.
- "input": use "".
- "output": MUST follow exactly this structure (four labeled lines/blocks):
[MOA]: {{pathway-level mechanism with specific pathway names, cytokines, receptors}}
[Co-factors]: {{required synergistic compounds}}
[Benefits]: {{known therapeutic benefits and use cases}}
[Risks]: {{contraindications, dangers, safety flags, FDA status}}
The output field must be a plain string, not a JSON object. Format it exactly like this: [MOA]: text here [Co-factors]: text here [Benefits]: text here [Risks]: text here — all as one continuous string value.

Return only valid JSON."""

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=TEACHER_MODEL,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.3,
            )
            raw = resp.choices[0].message.content or ""
            obj = json.loads(raw)
            if not all(k in obj for k in ("instruction", "input", "output")):
                continue
            out = normalize_teacher_output(obj["output"])
            if validate_alpaca_output(out):
                return {
                    "instruction": str(obj["instruction"]),
                    "input": str(obj.get("input", "")),
                    "output": out,
                    "_peptide": peptide,
                    "_pmid": None,
                }
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug("Teacher parse attempt %s: %s", attempt + 1, e)
            time.sleep(1.0)
    return None


def abstract_key(peptide: str, pmid: str) -> str:
    return f"{peptide}|{pmid}"


def phase_alpaca(
    client: Any,
    dry_run: bool,
    resume: bool,
) -> list[dict[str, Any]]:
    from tqdm import tqdm

    records = load_raw_records()
    logger.info("Phase 2A: starting Alpaca generation (%d raw records)", len(records))
    target_per = ALPACA_DRY if dry_run else ALPACA_FULL

    # Group records by peptide (dedupe pmid within peptide)
    by_peptide: dict[str, list[dict[str, Any]]] = {p: [] for p in PEPTIDES}
    seen: dict[str, set[str]] = {p: set() for p in PEPTIDES}
    for rec in records:
        p = rec.get("peptide")
        pmid = str(rec.get("pmid", ""))
        if p not in by_peptide or not pmid:
            continue
        if pmid in seen[p]:
            continue
        seen[p].add(pmid)
        by_peptide[p].append(rec)

    done_keys: set[str] = set()
    rows: list[dict[str, Any]] = []
    global_success = 0

    if resume and CHECKPOINT_PATH.is_file():
        ck = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        rows = ck.get("rows", [])
        done_keys = set(ck.get("done_keys", []))
        global_success = ck.get("global_success_count", len(rows))
        logger.info("Resume: loaded %d rows, %d keys", len(rows), len(done_keys))
    elif resume:
        logger.warning("--resume but no checkpoint at %s; starting fresh", CHECKPOINT_PATH)

    def save_checkpoint() -> None:
        ck_data = {
            "rows": rows,
            "done_keys": list(done_keys),
            "global_success_count": global_success,
        }
        CHECKPOINT_PATH.write_text(json.dumps(ck_data, ensure_ascii=False, indent=2), encoding="utf-8")
        # mirror dataset without internal fields for training file
        clean = []
        for r in rows:
            c = {k: v for k, v in r.items() if not k.startswith("_")}
            clean.append(c)
        DATASET_PATH.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")

    for peptide in tqdm(PEPTIDES, desc="Phase 2A: peptides"):
        skipped_no_peptide_mention = 0
        skipped_off_topic = 0
        count = sum(
            1
            for r in rows
            if r.get("_meta", {}).get("peptide") == peptide
        )
        if count >= target_per:
            continue

        for rec in tqdm(
            by_peptide.get(peptide, []),
            desc=f"  {peptide[:20]}",
            leave=False,
        ):
            if count >= target_per:
                break
            pmid = str(rec.get("pmid", ""))
            key = abstract_key(peptide, pmid)
            if key in done_keys:
                continue
            text = rec.get("text") or ""
            if not text.strip():
                continue

            text_lower = text.lower()
            aliases = PEPTIDE_ALIASES[peptide]
            if not any(a.lower() in text_lower for a in aliases):
                skipped_no_peptide_mention += 1
                continue

            kw_hits = sum(
                1 for kw in ALPACA_RELEVANCE_KEYWORDS if kw.lower() in text_lower
            )
            if kw_hits < 2:
                logger.info(
                    "Phase 2A off-topic: peptide=%s pmid=%s (keyword hits=%d, need >=2)",
                    peptide,
                    pmid,
                    kw_hits,
                )
                skipped_off_topic += 1
                continue

            result = call_teacher(client, text, peptide)
            if not result:
                continue

            result["_meta"] = {"peptide": peptide, "pmid": pmid}
            result.pop("_peptide", None)
            result.pop("_pmid", None)
            rows.append(result)
            done_keys.add(key)
            global_success += 1
            count += 1

            if global_success % CHECKPOINT_EVERY == 0:
                save_checkpoint()
                logger.info("Checkpoint: %d total Alpaca rows", global_success)

        logger.info(
            "Phase 2A %s: filter skips — no_peptide_mention=%d off_topic=%d",
            peptide,
            skipped_no_peptide_mention,
            skipped_off_topic,
        )

        # warn if short
        final_count = sum(1 for r in rows if r.get("_meta", {}).get("peptide") == peptide)
        if final_count < target_per:
            logger.warning(
                "Peptide %s: only %d successful Alpaca rows (target %d). Continuing.",
                peptide,
                final_count,
                target_per,
            )

    save_checkpoint()
    # final write without _meta in output? Plan says Alpaca format — keep instruction/input/output only in peptide_dataset.json
    clean = []
    for r in rows:
        clean.append(
            {
                "instruction": r["instruction"],
                "input": r.get("input", ""),
                "output": r["output"],
            }
        )
    DATASET_PATH.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Phase 2A complete: %d rows -> %s", len(clean), DATASET_PATH)
    return rows


def chunk_text(text: str, enc: Any, max_tokens: int = CHUNK_TOKENS) -> list[str]:
    toks = enc.encode(text)
    chunks: list[str] = []
    for i in range(0, len(toks), max_tokens):
        chunk_tokens = toks[i : i + max_tokens]
        chunks.append(enc.decode(chunk_tokens))
    return chunks if chunks else [""]


def phase_pinecone(client: Any, dry_run: bool) -> None:
    import tiktoken
    from tqdm import tqdm

    from pinecone import Pinecone  # lazy import — faster CLI / fetch-only runs

    records = load_raw_records()
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME", "peptiscout")
    if not api_key:
        logger.error("PINECONE_API_KEY is not set")
        sys.exit(1)

    if dry_run:
        cap_per = FETCH_CAP_DRY
        by_p: dict[str, list[dict[str, Any]]] = {p: [] for p in PEPTIDES}
        for rec in records:
            p = rec.get("peptide")
            if p in by_p and len(by_p[p]) < cap_per:
                by_p[p].append(rec)
        flat: list[dict[str, Any]] = []
        for p in PEPTIDES:
            flat.extend(by_p[p])
        records = flat
        logger.info("Dry-run Pinecone: %d abstract records", len(records))
    else:
        # Dedupe by PMID (same paper may appear under multiple peptide queries)
        seen_pm: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for rec in records:
            pmid = str(rec.get("pmid", ""))
            if not pmid or pmid in seen_pm:
                continue
            seen_pm.add(pmid)
            deduped.append(rec)
        records = deduped
        logger.info("Phase 2B: %d unique PMIDs to index", len(records))

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    enc = tiktoken.get_encoding("cl100k_base")  # standard for OpenAI embeddings family

    vectors_batch: list[dict[str, Any]] = []
    batch_size = 64

    for rec in tqdm(records, desc="Phase 2B: embed+upsert"):
        pmid = str(rec.get("pmid", ""))
        peptide = str(rec.get("peptide", ""))
        text = rec.get("text") or ""
        if not text.strip():
            continue
        chunks = chunk_text(text, enc, CHUNK_TOKENS)
        slug = hashlib.md5(peptide.encode()).hexdigest()[:8]
        for ci, ch in enumerate(chunks):
            if not ch.strip():
                continue
            vid = f"{pmid}_{slug}_{ci}"
            resp = client.embeddings.create(model=EMBEDDING_MODEL, input=ch[:8000])
            vec = resp.data[0].embedding
            if len(vec) != EMBEDDING_DIM:
                logger.warning("Unexpected embedding dim %s", len(vec))
            text_meta = ch[:PINECONE_METADATA_TEXT_MAX]
            vectors_batch.append(
                {
                    "id": vid,
                    "values": vec,
                    "metadata": {
                        "pmid": pmid,
                        "chunk_index": ci,
                        "text": text_meta,
                    },
                }
            )
            if len(vectors_batch) >= batch_size:
                index.upsert(vectors=vectors_batch)
                vectors_batch = []

    if vectors_batch:
        index.upsert(vectors=vectors_batch)

    logger.info("Pinecone upsert complete (index=%s)", index_name)


def main() -> None:
    # override=True: empty NCBI_EMAIL in the shell would otherwise block .env values
    load_dotenv(ROOT / ".env", override=True)
    parser = argparse.ArgumentParser(description="PeptiScout PubMed + Alpaca + Pinecone pipeline")
    parser.add_argument(
        "--phase",
        choices=("fetch", "alpaca", "pinecone"),
        default=None,
        help="Run a single phase; omit to run fetch -> alpaca -> pinecone",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run fetch→alpaca→pinecone with caps (3 abstracts & 3 Alpaca rows per peptide)",
    )
    parser.add_argument("--resume", action="store_true", help="Resume Alpaca from checkpoint")
    args = parser.parse_args()

    if len(PEPTIDES) != 20:
        logger.error("PEPTIDES must have length 20, got %d", len(PEPTIDES))
        sys.exit(1)

    if args.dry_run:
        phases = ["fetch", "alpaca", "pinecone"]
    elif args.phase:
        phases = [args.phase]
    else:
        phases = ["fetch", "alpaca", "pinecone"]

    email = os.environ.get("NCBI_EMAIL")
    if "fetch" in phases and not email:
        logger.error("NCBI_EMAIL is required for --phase fetch")
        sys.exit(1)

    session = requests.Session()

    if "fetch" in phases:
        if not email:
            logger.error("NCBI_EMAIL missing")
            sys.exit(1)
        phase_fetch(session, email, args.dry_run)

    if "alpaca" in phases or "pinecone" in phases:
        if not RAW_PATH.is_file():
            logger.error(
                "Missing %s — run `python -m backend.scripts.generate_dataset --phase fetch` first.",
                RAW_PATH,
            )
            sys.exit(1)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY is not set")
            sys.exit(1)
        logger.info(
            "Loading OpenAI Python SDK (first import can take a while; not stuck after Phase 1)."
        )
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        if "alpaca" in phases:
            phase_alpaca(client, args.dry_run, args.resume)

        if "pinecone" in phases:
            phase_pinecone(client, args.dry_run)


if __name__ == "__main__":
    main()
