"""Tool D — Tavily web search + structured vendor vetting summary."""

from __future__ import annotations

import json
import os
import re
from typing import Any


def _fallback_from_tavily(vendor_name: str, r1: dict[str, Any], r2: dict[str, Any]) -> dict[str, Any]:
    """Heuristic summary if OpenAI structuring fails."""
    urls: list[str] = []
    snippets: list[str] = []
    for r in (r1, r2):
        for item in (r.get("results") or []):
            u = item.get("url") or ""
            c = item.get("content") or ""
            if u:
                urls.append(u)
            if c:
                snippets.append(c)
    blob = " ".join(snippets).lower()
    coa = bool(re.search(r"\bcoa\b|certificate of analysis|third[- ]party", blob))
    reddit_txt = " ".join(
        (x.get("content") or "") for x in (r2.get("results") or [])[:3]
    ).strip() or "No Reddit snippets returned."
    return {
        "vendor": vendor_name,
        "country": None,
        "coa_available": coa,
        "coa_url": urls[0] if urls else None,
        "reddit_sentiment": reddit_txt[:400],
        "flags": [],
        "summary": "Heuristic summary from search snippets only; configure OpenAI for richer output.",
    }


def vet_vendor(vendor_name: str) -> dict[str, Any]:
    """
    Run two Tavily searches (COA / Reddit), then structure results with OpenAI JSON.

    Returns keys: ``vendor``, ``country``, ``coa_available``, ``coa_url``, ``reddit_sentiment``,
    ``flags``, ``summary``.
    """
    name = vendor_name.strip()
    if not name:
        raise ValueError("vendor_name must be non-empty")

    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        raise ValueError("TAVILY_API_KEY is not set")

    from tavily import TavilyClient

    tv = TavilyClient(api_key=tavily_key)
    q1 = f"{name} peptide COA certificate of analysis third party tested"
    q2 = f"site:reddit.com/r/Peptides {name} review reputation"
    r1 = tv.search(query=q1, max_results=6)
    r2 = tv.search(query=q2, max_results=6)

    if not os.getenv("OPENAI_API_KEY"):
        return _fallback_from_tavily(name, r1, r2)

    system = (
        "You turn peptide vendor web search JSON into a single JSON object. "
        "Use only evidence from the provided search results. "
        'Set "coa_url" to a real URL from the COA search results when available, else null. '
        'Be conservative with "coa_available". '
        '"reddit_sentiment" must be one or two sentences about community reputation based on Reddit hits. '
        '"country" is headquarters or shipping country if clearly stated in snippets, else null. '
        '"flags" is an array of short risk or quality notes (empty if none). '
        '"summary" is 2–4 sentences for a researcher.'
    )
    user_payload = json.dumps(
        {"vendor_query": name, "coa_search": r1, "reddit_search": r2},
        default=str,
    )[:120000]

    from openai import OpenAI

    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    try:
        comp = oai.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_payload},
            ],
        )
        raw = comp.choices[0].message.content or "{}"
        data = json.loads(raw)
    except Exception:
        return _fallback_from_tavily(name, r1, r2)

    out: dict[str, Any] = {
        "vendor": str(data.get("vendor", name)),
        "country": data.get("country"),
        "coa_available": bool(data.get("coa_available", False)),
        "coa_url": data.get("coa_url"),
        "reddit_sentiment": str(data.get("reddit_sentiment", "")),
        "flags": data.get("flags") if isinstance(data.get("flags"), list) else [],
        "summary": str(data.get("summary", "")),
    }
    out["flags"] = [str(x) for x in out["flags"]]
    if out["country"] is not None:
        out["country"] = str(out["country"])
    if out["coa_url"] is not None:
        out["coa_url"] = str(out["coa_url"])
    return out
