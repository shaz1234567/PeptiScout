"""
Terminal evaluation against a live FastAPI PeptiScout server.

Scoring (DS / PC / TSR / CA):
  - DS: From ``protocol`` only, extract recommended dose in mcg and BAC water in mL.
    If benchmark ranges exist, ``ds_loose`` is water-range correctness and
    ``ds_strict`` requires both dose and water ranges to match. If ranges are missing,
    fall back to legacy injection-volume comparison against ``gold_mL``.
  - PC: ``full_response = protocol + moa + good_bad + audit_trail`` (concat, lower for match).
    ``pc = sum(1 for c in gold_cofactors if c in full_response.lower()) / len(gold_cofactors)``
    when ``len(gold_cofactors) > 0``; else ``null`` (not 0).
  - TSR: At least one **safety** token and one **guidance** token in ``full_response``
    (see ``_TSR_GROUP_SAFETY`` / ``_TSR_GROUP_GUIDANCE`` in code). Score 1 if both
    groups match, else 0.
  - CA: Regex 7–8 digit PMIDs in ``audit_trail``; each validated via NCBI esummary.
    Score = mean of ``validate_pmid`` per unique PMID, or ``null`` if none extracted.
    Use ``--skip-ca`` to set CA to null with reason ``skipped`` (no network).

Ablations (aggregates use the same per-row scores):
  - A: mean ``ds_loose`` on ``category == "dosage"`` for ``full-agent`` minus mean
    ``ds_loose`` for ``full-agent-no-calculator`` (dosage rows only).
  - B: mean PC on ``category == "moa"`` for ``full-agent`` minus mean PC for
    ``no-reasoning`` (moa rows only; rows with null PC omitted from each mean).
  - C: mean PC and mean CA (overall, non-null only) for ``fine-tuned-no-rag`` minus
    those for ``rag-only-no-finetune``.

Smoke test (API must be running):
  From repo root: ``python -m backend.scripts.run_evaluation --limit 2 --skip-ca``

Use ``--verbose`` (``-v``) to print each API JSON response and computed scores (via ``tqdm.write`` so the progress bar stays readable).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK = ROOT / "backend" / "data" / "benchmark_100.json"
DEFAULT_OUTPUT = ROOT / "backend" / "data" / "results.json"
DEFAULT_LLAMA_OUTPUT = ROOT / "backend" / "data" / "results_llama_11b.json"
CHECKPOINT_PATH = ROOT / "backend" / "data" / "eval_checkpoint.json"

PRIMARY_MODES = [
    "baseline-zero-shot",
    "baseline-few-shot",
    "baseline-cot",
    "full-agent",
]
ABLATION_MODES = [
    "no-reasoning",
    "full-agent-no-calculator",
    "fine-tuned-no-rag",
    "rag-only-no-finetune",
]
DEFAULT_MODES = PRIMARY_MODES + ABLATION_MODES

NCBI_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
NCBI_SLEEP_S = 0.34

DS_STRICT_THRESHOLD_ML = 0.05
DS_LOOSE_THRESHOLD_ML = 0.1
API_CALL_SLEEP_S = 2.0
RETRY_SLEEP_S = 10.0
MAX_RETRIES = 3

# PMID: 7–8 digits (PubMed IDs)
PMID_RE = re.compile(r"\b(\d{7,8})\b")

_TSR_GROUP_SAFETY = (
    "contraindication",
    "contraindicated",
    "avoid",
    "caution",
    "warning",
    "adverse",
    "risk",
    "side effect",
    "monitor",
    "do not use",
    "precaution",
)
_TSR_GROUP_GUIDANCE = (
    "recommend",
    "advised",
    "suggested",
    "protocol",
    "administer",
    "guideline",
    "use",
    "dosing",
    "schedule",
)


def _safe_float(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None


def _structured_number(raw: dict[str, Any], names: tuple[str, ...]) -> float | None:
    """Read canonical numeric fields before falling back to fragile text regex."""
    for name in names:
        v = raw.get(name)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass

    protocol = raw.get("protocol")
    if isinstance(protocol, dict):
        for name in names:
            v = protocol.get(name)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
    return None


def normalize_response(raw: Any) -> dict[str, Any]:
    """Normalize model output from a dict or JSON-ish text into the PeptiScout schema."""
    if isinstance(raw, dict):
        return {
            "protocol": raw.get("protocol", ""),
            "moa": raw.get("moa", ""),
            "good_bad": raw.get("good_bad", ""),
            "audit_trail": raw.get("audit_trail", ""),
            "react_trace": raw.get("react_trace"),
            **{
                k: v
                for k, v in raw.items()
                if k not in {"protocol", "moa", "good_bad", "audit_trail", "react_trace"}
            },
        }

    text = str(raw or "").strip()
    if text:
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end > start:
                return normalize_response(json.loads(text[start : end + 1]))
        except json.JSONDecodeError:
            pass
    return {
        "protocol": text,
        "moa": "",
        "good_bad": "",
        "audit_trail": "",
        "react_trace": None,
    }


def extract_predicted_ml(protocol: str) -> tuple[float | None, str | None]:
    """Prefer injection draw volume (small mL), not reconstitution water (1–3 mL)."""
    if not protocol or not protocol.strip():
        return None, "empty_protocol"
    text = protocol
    candidates: list[float] = []

    for m in re.finditer(
        r"(?:draw|inject|pull)\s*(?:approximately\s*|about\s*)?(\d+\.?\d*)\s*mL\b",
        text,
        re.IGNORECASE,
    ):
        v = _safe_float(m.group(1))
        if v is not None:
            candidates.append(v)

    for m in re.finditer(r"(\d+\.?\d*)\s*mL\s*=", text, re.IGNORECASE):
        v = _safe_float(m.group(1))
        if v is not None:
            candidates.append(v)

    for m in re.finditer(r"\b(\d+\.?\d*)\s*units?\b", text, re.IGNORECASE):
        u = _safe_float(m.group(1))
        if u is not None:
            candidates.append(u / 100.0)

    for m in re.finditer(r"(\d+\.?\d*)\s*mL\b", text, re.IGNORECASE):
        v = _safe_float(m.group(1))
        if v is None or v >= 1.0:
            continue
        start, end = m.span()
        ctx = text[max(0, start - 50) : min(len(text), end + 50)].lower()
        if any(k in ctx for k in ("draw", "inject", "syringe")):
            candidates.append(v)

    in_range = [c for c in candidates if 0 < c < 2.0]
    if not in_range:
        return None, "no_injection_ml"
    return min(in_range), None


def extract_predicted_dose_mcg(protocol: str) -> tuple[float | None, str | None]:
    """Extract a recommended dose in mcg, avoiding concentration units like mcg/mL."""
    if not protocol or not protocol.strip():
        return None, "empty_protocol"
    text = protocol
    candidates: list[float] = []

    for m in re.finditer(
        r"(?:recommended\s+)?dose(?:\s+is|\s+of|:)?\s*(\d+\.?\d*)\s*(?:mcg|ug|µg)\b(?!\s*/\s*mL)",
        text,
        re.IGNORECASE,
    ):
        v = _safe_float(m.group(1))
        if v is not None:
            candidates.append(v)

    for m in re.finditer(
        r"(?:administer|inject|take|use)\s*(\d+\.?\d*)\s*(?:mcg|ug|µg)\b(?!\s*/\s*mL)",
        text,
        re.IGNORECASE,
    ):
        v = _safe_float(m.group(1))
        if v is not None:
            candidates.append(v)

    for m in re.finditer(r"(\d+\.?\d*)\s*(?:mcg|ug|µg)\b(?!\s*/\s*mL)", text, re.IGNORECASE):
        start, end = m.span()
        after = text[end : min(len(text), end + 12)].lower()
        if "per ml" in after:
            continue
        ctx = text[max(0, start - 60) : min(len(text), end + 60)].lower()
        if any(k in ctx for k in ("dose", "administer", "inject", "protocol", "daily", "per dose")):
            v = _safe_float(m.group(1))
            if v is not None:
                candidates.append(v)

    if not candidates:
        return None, "no_dose_pattern"
    return candidates[0], None


def extract_predicted_dose_mcg_from_response(raw: dict[str, Any]) -> tuple[float | None, str | None]:
    direct = _structured_number(
        raw,
        (
            "recommended_dose_mcg",
            "dose_mcg",
            "dose",
            "recommendedDoseMcg",
        ),
    )
    if direct is not None:
        return direct, None
    return extract_predicted_dose_mcg(str(raw.get("protocol", "")))


def extract_predicted_water_ml(protocol: str) -> tuple[float | None, str | None]:
    """Extract BAC/reconstitution water volume in mL, not injection draw volume."""
    if not protocol or not protocol.strip():
        return None, "empty_protocol"
    text = protocol
    candidates: list[float] = []

    patterns = (
        r"(?:BAC_water_amount|water_mL|water_ml)['\"]?\s*:\s*['\"]?(\d+\.?\d*)",
        r"(?:reconstitute|reconstitution|add|mix|dilute|use)\D{0,40}(\d+\.?\d*)\s*mL\D{0,40}(?:BAC|bacteriostatic|water)",
        r"(\d+\.?\d*)\s*mL\D{0,40}(?:BAC|bacteriostatic|water)",
        r"(?:BAC|bacteriostatic|water)\D{0,40}(\d+\.?\d*)\s*mL",
    )
    for pattern in patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            v = _safe_float(m.group(1))
            if v is not None:
                candidates.append(v)

    in_range = [c for c in candidates if 0.25 <= c <= 5.0]
    if not in_range:
        return None, "no_water_pattern"
    return in_range[0], None


def extract_predicted_water_ml_from_response(raw: dict[str, Any]) -> tuple[float | None, str | None]:
    direct = _structured_number(
        raw,
        (
            "bac_water_mL",
            "bac_water_ml",
            "BAC_water_amount",
            "water_mL",
            "water_ml",
            "reconstitution_water_mL",
            "reconstitution_water_ml",
            "water_amount_ml",
        ),
    )
    if direct is not None:
        return direct, None
    return extract_predicted_water_ml(str(raw.get("protocol", "")))


def score_ds(
    protocol: str,
    gold_mL: Any,
    gold_dose_range: Any = None,
    gold_water_range: Any = None,
) -> tuple[int | None, int | None, str | None]:
    """Return (ds_strict, ds_loose, note), using ranges when available."""
    if (
        isinstance(gold_dose_range, list)
        and len(gold_dose_range) == 2
        and isinstance(gold_water_range, list)
        and len(gold_water_range) == 2
    ):
        pred_dose, dose_note = extract_predicted_dose_mcg(protocol)
        pred_water, water_note = extract_predicted_water_ml(protocol)
        notes = [n for n in (dose_note, water_note) if n]
        if pred_dose is None or pred_water is None:
            return None, None, ";".join(notes) or "range_prediction_missing"
        try:
            dose_min, dose_max = float(gold_dose_range[0]), float(gold_dose_range[1])
            water_min, water_max = float(gold_water_range[0]), float(gold_water_range[1])
        except (TypeError, ValueError):
            return None, None, "invalid_gold_range"
        ds_dose = int(dose_min <= pred_dose <= dose_max)
        ds_water = int(water_min <= pred_water <= water_max)
        return int(ds_dose == 1 and ds_water == 1), ds_water, None

    pred, reason = extract_predicted_ml(protocol)
    if pred is None:
        return None, None, reason or "no_prediction"
    try:
        g = float(gold_mL)
    except (TypeError, ValueError):
        return None, None, "no_gold_ml"
    diff = abs(pred - g)
    ds_strict = int(diff < DS_STRICT_THRESHOLD_ML)
    ds_loose = int(diff < DS_LOOSE_THRESHOLD_ML)
    return ds_strict, ds_loose, "legacy_gold_ml"


def score_ds_response(
    raw: dict[str, Any],
    gold_mL: Any,
    gold_dose_range: Any = None,
    gold_water_range: Any = None,
) -> tuple[int | None, int | None, str | None]:
    """Return DS metrics, preferring structured numeric response fields."""
    if (
        isinstance(gold_dose_range, list)
        and len(gold_dose_range) == 2
        and isinstance(gold_water_range, list)
        and len(gold_water_range) == 2
    ):
        pred_dose, dose_note = extract_predicted_dose_mcg_from_response(raw)
        pred_water, water_note = extract_predicted_water_ml_from_response(raw)
        notes = [n for n in (dose_note, water_note) if n]
        if pred_dose is None or pred_water is None:
            return None, None, ";".join(notes) or "range_prediction_missing"
        try:
            dose_min, dose_max = float(gold_dose_range[0]), float(gold_dose_range[1])
            water_min, water_max = float(gold_water_range[0]), float(gold_water_range[1])
        except (TypeError, ValueError):
            return None, None, "invalid_gold_range"
        ds_dose = int(dose_min <= pred_dose <= dose_max)
        ds_water = int(water_min <= pred_water <= water_max)
        return int(ds_dose == 1 and ds_water == 1), ds_water, None

    return score_ds(str(raw.get("protocol", "")), gold_mL, gold_dose_range, gold_water_range)


def score_pc(full_response: str, gold_cofactors: list[str]) -> float | None:
    if not gold_cofactors:
        return None
    low = full_response.lower()
    hit = sum(1 for c in gold_cofactors if c.lower() in low)
    return hit / len(gold_cofactors)


def score_tsr(full_response: str) -> int:
    low = full_response.lower()
    safety = any(w in low for w in _TSR_GROUP_SAFETY)
    guidance = any(w in low for w in _TSR_GROUP_GUIDANCE)
    return int(safety and guidance)


def extract_pmids(audit_trail: str) -> list[str]:
    return list(dict.fromkeys(PMID_RE.findall(audit_trail or "")))


def validate_pmid(pmid: str) -> bool:
    try:
        r = requests.get(
            NCBI_ESUMMARY,
            params={"db": "pubmed", "id": pmid, "retmode": "json"},
            timeout=45,
        )
        r.raise_for_status()
        data = r.json()
        res = data.get("result", {})
        if pmid not in res:
            return False
        entry = res[pmid]
        if isinstance(entry, dict) and entry.get("error"):
            return False
        return True
    except Exception:
        return False


def score_ca(
    audit_trail: str,
    skip_ca: bool,
) -> tuple[float | None, str | None]:
    if skip_ca:
        return None, "skipped"
    pmids = extract_pmids(audit_trail)
    if not pmids:
        return None, "no_pmids"
    ok = []
    for i, p in enumerate(pmids):
        if i > 0:
            time.sleep(NCBI_SLEEP_S)
        ok.append(1.0 if validate_pmid(p) else 0.0)
    return sum(ok) / len(ok), None


def post_query(base_url: str, text: str, mode: str) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/api/query"
    payload = {"text": text, "mode": mode, "image_base64": None}
    for attempt in range(MAX_RETRIES + 1):
        time.sleep(API_CALL_SLEEP_S)
        try:
            r = requests.post(url, json=payload, timeout=600)
        except requests.ConnectionError as e:
            if attempt >= MAX_RETRIES:
                return {"_error": f"connection_error after {MAX_RETRIES} retries: {e!s}"}
            time.sleep(RETRY_SLEEP_S)
            continue

        if r.status_code == 502:
            if attempt >= MAX_RETRIES:
                return {"_error": f"502 after {MAX_RETRIES} retries: {r.text[:500]}"}
            time.sleep(RETRY_SLEEP_S)
            continue
        if r.status_code == 503:
            return {"_error": f"503: {r.text[:500]}"}
        r.raise_for_status()
        return r.json()
    return {"_error": "request retry loop exhausted"}


def load_checkpoint() -> list[dict[str, Any]]:
    if not CHECKPOINT_PATH.exists():
        return []
    with open(CHECKPOINT_PATH, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    return []


def save_checkpoint(per_query: list[dict[str, Any]]) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(per_query, f, indent=2, ensure_ascii=False)


def load_checkpoint_from(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    return []


def save_checkpoint_to(path: Path, per_query: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(per_query, f, indent=2, ensure_ascii=False)


def mean_skip_null(xs: list[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    if not vals:
        return None
    return float(statistics.mean(vals))


def build_comparison_table(
    per_query: list[dict[str, Any]],
    modes: list[str],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for m in modes:
        rows = [r for r in per_query if r["mode"] == m]
        ds_loose_vals = [r["scores"]["ds_loose"] for r in rows]
        ds_strict_vals = [r["scores"]["ds_strict"] for r in rows]
        pc_vals = [r["scores"]["pc"] for r in rows]
        tsr_vals = [r["scores"]["tsr"] for r in rows]
        ca_vals = [r["scores"]["ca"] for r in rows]
        out[m] = {
            "mean_ds": mean_skip_null(
                [float(x) if x is not None else None for x in ds_loose_vals]
            ),
            "mean_ds_strict": mean_skip_null(
                [float(x) if x is not None else None for x in ds_strict_vals]
            ),
            "mean_pc": mean_skip_null(pc_vals),
            "mean_tsr": mean_skip_null([float(x) if x is not None else None for x in tsr_vals]),
            "mean_ca": mean_skip_null(ca_vals),
            "n": len(rows),
            "n_ds": sum(1 for x in ds_loose_vals if x is not None),
            "n_pc": sum(1 for x in pc_vals if x is not None),
            "n_ca": sum(1 for x in ca_vals if x is not None),
        }
    return out


def ablation_a(per_query: list[dict[str, Any]]) -> dict[str, Any]:
    def mean_ds_mode(mode: str, cat: str | None = None) -> float | None:
        ds_list: list[float] = []
        for r in per_query:
            if r["mode"] != mode:
                continue
            if cat and r.get("category") != cat:
                continue
            d = r["scores"]["ds_loose"]
            if d is not None:
                ds_list.append(float(d))
        return statistics.mean(ds_list) if ds_list else None

    full = mean_ds_mode("full-agent", "dosage")
    nocalc = mean_ds_mode("full-agent-no-calculator", "dosage")
    delta = None
    if full is not None and nocalc is not None:
        delta = full - nocalc
    return {
        "mean_ds_full_agent_dosage": full,
        "mean_ds_no_calculator_dosage": nocalc,
        "delta_ds": delta,
        "rule": "mean DS (ds_loose) on category==dosage: full-agent minus full-agent-no-calculator",
    }


def ablation_b(per_query: list[dict[str, Any]]) -> dict[str, Any]:
    def mean_pc_mode(mode: str) -> float | None:
        pcs: list[float] = []
        for r in per_query:
            if r["mode"] != mode or r.get("category") != "moa":
                continue
            p = r["scores"]["pc"]
            if p is not None:
                pcs.append(p)
        return statistics.mean(pcs) if pcs else None

    ref = mean_pc_mode("full-agent")
    var = mean_pc_mode("no-reasoning")
    delta = None
    if ref is not None and var is not None:
        delta = ref - var
    return {
        "mean_pc_full_agent_moa": ref,
        "mean_pc_no_reasoning_moa": var,
        "delta_pc": delta,
        "rule": "mean PC on category==moa: full-agent minus no-reasoning",
    }


def ablation_c(per_query: list[dict[str, Any]]) -> dict[str, Any]:
    def means(mode: str) -> tuple[float | None, float | None]:
        pcs: list[float] = []
        cas: list[float] = []
        for r in per_query:
            if r["mode"] != mode:
                continue
            if r["scores"]["pc"] is not None:
                pcs.append(r["scores"]["pc"])
            if r["scores"]["ca"] is not None:
                cas.append(r["scores"]["ca"])
        mp = statistics.mean(pcs) if pcs else None
        mc = statistics.mean(cas) if cas else None
        return mp, mc

    ft_pc, ft_ca = means("fine-tuned-no-rag")
    rag_pc, rag_ca = means("rag-only-no-finetune")
    d_pc = (ft_pc - rag_pc) if (ft_pc is not None and rag_pc is not None) else None
    d_ca = (ft_ca - rag_ca) if (ft_ca is not None and rag_ca is not None) else None
    return {
        "mean_pc_fine_tuned_no_rag": ft_pc,
        "mean_pc_rag_only_no_finetune": rag_pc,
        "delta_pc": d_pc,
        "mean_ca_fine_tuned_no_rag": ft_ca,
        "mean_ca_rag_only_no_finetune": rag_ca,
        "delta_ca": d_ca,
        "rule": "overall mean PC and mean CA: fine-tuned-no-rag minus rag-only-no-finetune",
    }


def score_prediction_rows(
    predictions: list[dict[str, Any]],
    benchmark: list[dict[str, Any]],
    *,
    skip_ca: bool,
) -> list[dict[str, Any]]:
    bench_by_id = {entry.get("id"): entry for entry in benchmark}
    per_query: list[dict[str, Any]] = []
    for pred in predictions:
        entry = bench_by_id.get(pred.get("id"))
        if not entry:
            continue

        mode = str(pred.get("mode") or "predictions")
        raw = normalize_response(pred.get("response", pred))
        protocol = str(raw.get("protocol", ""))
        moa = str(raw.get("moa", ""))
        gb = str(raw.get("good_bad", ""))
        audit = str(raw.get("audit_trail", ""))
        full_response = protocol + moa + gb + audit

        ds_strict, ds_loose, ds_note = score_ds_response(
            raw,
            entry.get("gold_mL"),
            entry.get("gold_dose_range"),
            entry.get("gold_water_range"),
        )
        pc = score_pc(full_response, entry.get("gold_cofactors") or [])
        tsr = score_tsr(full_response)
        ca, ca_note = score_ca(audit, skip_ca)
        row: dict[str, Any] = {
            "id": entry.get("id"),
            "category": entry.get("category"),
            "mode": mode,
            "scores": {
                "ds_strict": ds_strict,
                "ds_loose": ds_loose,
                "pc": pc,
                "tsr": tsr,
                "ca": ca,
            },
            "raw_keys": list(raw.keys()),
        }
        notes = []
        if ds_note:
            notes.append(f"ds:{ds_note}")
        if ca_note:
            notes.append(f"ca:{ca_note}")
        if notes:
            row["score_notes"] = "; ".join(notes)
        per_query.append(row)
    return per_query


def _verbose_block(
    *,
    entry_id: Any,
    category: Any,
    mode: str,
    gold_mL: Any,
    raw: dict[str, Any] | None,
    scores: dict[str, Any],
    ds_note: str | None,
    full_response: str,
    error: str | None,
) -> str:
    """Build human-readable debug text (full response + scores + DS/TSR hints)."""
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 72)
    lines.append(f"id={entry_id!r} category={category!r} mode={mode!r} gold_mL={gold_mL!r}")
    if error:
        lines.append(f"ERROR: {error}")
        return "\n".join(lines)
    protocol = str(raw.get("protocol", "") if raw else "")
    pred_ml, pred_reason = extract_predicted_ml(protocol)
    pred_dose, dose_reason = extract_predicted_dose_mcg(protocol)
    pred_water, water_reason = extract_predicted_water_ml(protocol)
    low = full_response.lower()
    tsr_safety = any(w in low for w in _TSR_GROUP_SAFETY)
    tsr_guidance = any(w in low for w in _TSR_GROUP_GUIDANCE)
    lines.append(
        f"scores ds_strict={scores.get('ds_strict')!r} ds_loose={scores.get('ds_loose')!r} "
        f"pc={scores.get('pc')!r} tsr={scores.get('tsr')!r} ca={scores.get('ca')!r}"
    )
    if ds_note:
        lines.append(f"ds_note: {ds_note}")
    lines.append(f"DS extract legacy_inject_mL={pred_ml!r} ({pred_reason or 'ok'})")
    lines.append(f"DS extract dose_mcg={pred_dose!r} ({dose_reason or 'ok'})")
    lines.append(f"DS extract water_mL={pred_water!r} ({water_reason or 'ok'})")
    lines.append(
        f"TSR groups: safety={tsr_safety} guidance={tsr_guidance} "
        f"(need both for tsr=1)"
    )
    lines.append("--- full API response (JSON) ---")
    lines.append(json.dumps(raw, indent=2, ensure_ascii=False) if raw else "{}")
    lines.append("=" * 72)
    return "\n".join(lines)


def print_summary_table(comparison_table: dict[str, dict[str, Any]], modes: list[str]) -> None:
    # mean_ds = primary aggregate (ds_loose); mean_ds_strict = stricter threshold
    headers = ("mode", "mean_ds", "mean_ds_strict", "mean_pc", "mean_tsr", "mean_ca")
    rows = []
    for m in modes:
        ct = comparison_table.get(m, {})
        rows.append(
            (
                m,
                ct.get("mean_ds"),
                ct.get("mean_ds_strict"),
                ct.get("mean_pc"),
                ct.get("mean_tsr"),
                ct.get("mean_ca"),
            )
        )
    w = [max(len(headers[i]), max(len(str(r[i])) for r in rows) if rows else 0) for i in range(6)]

    def fmt(x: Any) -> str:
        if x is None:
            return "-"
        if isinstance(x, float):
            return f"{x:.4f}".rstrip("0").rstrip(".")
        return str(x)

    line = " | ".join(h.ljust(w[i]) for i, h in enumerate(headers))
    print(line)
    print("-" * len(line))
    for row in rows:
        print(" | ".join(fmt(row[i]).ljust(w[i]) for i in range(6)))


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate PeptiScout /api/query modes against benchmark_100.")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000", help="FastAPI base URL")
    ap.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK, help="Benchmark JSON path")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="results.json path")
    ap.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Score a predictions JSON file directly instead of calling FastAPI.",
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_PATH,
        help="Checkpoint JSON path. Use a model-specific file for Llama runs.",
    )
    ap.add_argument("--base-model", default=None, help="Model name/id to record in output metadata")
    ap.add_argument("--adapter-path", default=None, help="Adapter path to record in output metadata")
    ap.add_argument("--quantization", default=None, help="Quantization setting to record in output metadata")
    ap.add_argument("--model-family", default=None, help="Model family to record in output metadata")
    ap.add_argument("--limit", type=int, default=0, help="Only first N benchmark rows (0 = all)")
    ap.add_argument(
        "--modes",
        default=",".join(DEFAULT_MODES),
        help="Comma-separated query modes (default: primary + ablation modes)",
    )
    ap.add_argument("--skip-ca", action="store_true", help="Skip NCBI esummary (CA null)")
    ap.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="After each query, print full API JSON, scores, and DS/TSR debug hints",
    )
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if not modes:
        print("No modes given.", file=sys.stderr)
        return 2

    with open(args.benchmark, encoding="utf-8") as f:
        bench: list[dict[str, Any]] = json.load(f)
    if args.limit and args.limit > 0:
        bench = bench[: args.limit]

    if args.predictions:
        with open(args.predictions, encoding="utf-8") as f:
            predictions = json.load(f)
        if not isinstance(predictions, list):
            print("--predictions must point to a JSON list of prediction rows.", file=sys.stderr)
            return 2
        per_query = score_prediction_rows(predictions, bench, skip_ca=args.skip_ca)
        modes_from_predictions = list(dict.fromkeys(str(r.get("mode")) for r in per_query))
        comparison_table = build_comparison_table(per_query, modes_from_predictions)
        out_doc = {
            "meta": {
                "base_url": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "limit": args.limit or None,
                "modes": modes_from_predictions,
                "skip_ca": args.skip_ca,
                "benchmark_path": str(args.benchmark.resolve()),
                "predictions_path": str(args.predictions.resolve()),
                "fastapi_used": False,
                "base_model": args.base_model,
                "adapter_path": args.adapter_path,
                "quantization": args.quantization,
                "model_family": args.model_family,
            },
            "comparison_table": comparison_table,
            "ablations": {
                "A": ablation_a(per_query),
                "B": ablation_b(per_query),
                "C": ablation_c(per_query),
            },
            "per_query": per_query,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out_doc, f, indent=2, ensure_ascii=False)
        print_summary_table(comparison_table, modes_from_predictions)
        print(f"\nWrote {args.output}")
        return 0

    per_query: list[dict[str, Any]] = load_checkpoint_from(args.checkpoint)
    completed = {(r.get("id"), r.get("mode")) for r in per_query}
    if per_query:
        print(f"Resuming from checkpoint {args.checkpoint}: {len(per_query)} entries already completed")

    tasks = [(e, m) for e in bench for m in modes if (e.get("id"), m) not in completed]
    successful_since_checkpoint = 0
    for entry, mode in tqdm(tasks, desc="eval", unit="call"):
        eid = entry.get("id")
        cat = entry.get("category")
        gold_mL = entry.get("gold_mL")
        gold_dose_range = entry.get("gold_dose_range")
        gold_water_range = entry.get("gold_water_range")
        gold_cofactors = entry.get("gold_cofactors") or []
        try:
            raw = post_query(args.base_url, str(entry["query"]), mode)
        except Exception as ex:
            if args.verbose:
                tqdm.write(
                    _verbose_block(
                        entry_id=eid,
                        category=cat,
                        mode=mode,
                        gold_mL=gold_mL,
                        raw=None,
                        scores={
                            "ds_strict": None,
                            "ds_loose": None,
                            "pc": None,
                            "tsr": None,
                            "ca": None,
                        },
                        ds_note=None,
                        full_response="",
                        error=str(ex),
                    )
                )
            per_query.append(
                {
                    "id": eid,
                    "category": cat,
                    "mode": mode,
                    "scores": {
                        "ds_strict": None,
                        "ds_loose": None,
                        "pc": None,
                        "tsr": None,
                        "ca": None,
                    },
                    "error": str(ex),
                }
            )
            continue

        if "_error" in raw:
            if args.verbose:
                tqdm.write(
                    _verbose_block(
                        entry_id=eid,
                        category=cat,
                        mode=mode,
                        gold_mL=gold_mL,
                        raw=None,
                        scores={
                            "ds_strict": None,
                            "ds_loose": None,
                            "pc": None,
                            "tsr": None,
                            "ca": None,
                        },
                        ds_note=None,
                        full_response="",
                        error=raw["_error"],
                    )
                )
            per_query.append(
                {
                    "id": eid,
                    "category": cat,
                    "mode": mode,
                    "scores": {
                        "ds_strict": None,
                        "ds_loose": None,
                        "pc": None,
                        "tsr": None,
                        "ca": None,
                    },
                    "error": raw["_error"],
                }
            )
            continue

        protocol = str(raw.get("protocol", ""))
        moa = str(raw.get("moa", ""))
        gb = str(raw.get("good_bad", ""))
        audit = str(raw.get("audit_trail", ""))
        full_response = protocol + moa + gb + audit

        ds_strict, ds_loose, ds_note = score_ds_response(
            raw,
            gold_mL,
            gold_dose_range,
            gold_water_range,
        )
        pc = score_pc(full_response, gold_cofactors if isinstance(gold_cofactors, list) else [])
        tsr: int | None = score_tsr(full_response)
        ca, ca_note = score_ca(audit, args.skip_ca)

        row: dict[str, Any] = {
            "id": eid,
            "category": cat,
            "mode": mode,
            "scores": {
                "ds_strict": ds_strict,
                "ds_loose": ds_loose,
                "pc": pc,
                "tsr": tsr,
                "ca": ca,
            },
            "raw_keys": list(raw.keys()),
        }
        notes = []
        if ds_note:
            notes.append(f"ds:{ds_note}")
        if ca_note:
            notes.append(f"ca:{ca_note}")
        if notes:
            row["score_notes"] = "; ".join(notes)
        if args.verbose:
            tqdm.write(
                _verbose_block(
                    entry_id=eid,
                    category=cat,
                    mode=mode,
                    gold_mL=gold_mL,
                    raw=raw,
                    scores=row["scores"],
                    ds_note=ds_note,
                    full_response=full_response,
                    error=None,
                )
            )
        per_query.append(row)
        successful_since_checkpoint += 1
        if successful_since_checkpoint >= 10:
            save_checkpoint_to(args.checkpoint, per_query)
            successful_since_checkpoint = 0

    save_checkpoint_to(args.checkpoint, per_query)

    comparison_table = build_comparison_table(per_query, modes)
    out_doc = {
        "meta": {
            "base_url": args.base_url,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "limit": args.limit or None,
            "modes": modes,
            "skip_ca": args.skip_ca,
            "benchmark_path": str(args.benchmark.resolve()),
            "fastapi_used": True,
            "checkpoint_path": str(args.checkpoint.resolve()),
            "base_model": args.base_model,
            "adapter_path": args.adapter_path,
            "quantization": args.quantization,
            "model_family": args.model_family,
        },
        "comparison_table": comparison_table,
        "ablations": {
            "A": ablation_a(per_query),
            "B": ablation_b(per_query),
            "C": ablation_c(per_query),
        },
        "per_query": per_query,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_doc, f, indent=2, ensure_ascii=False)

    print_summary_table(comparison_table, modes)
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
