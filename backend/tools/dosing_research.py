"""Tavily dosing protocol search for dose research before reconstitution math."""

from __future__ import annotations

import os
from typing import Any


def search_dosing_protocols(
    peptide: str | None,
    purpose: str | None,
    query: str,
) -> dict[str, Any]:
    """Search web/community sources for peptide dosing protocol snippets."""
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        return {"error": "TAVILY_API_KEY is not set", "queries": [], "results": []}

    peptide_text = (peptide or "").strip()
    purpose_text = (purpose or "").strip()
    fallback_query = query.strip()
    if peptide_text:
        q1 = f"{peptide_text} {purpose_text} peptide dosing protocol mcg".strip()
        q2 = f"{peptide_text} reconstitution dosing units BAC water"
    else:
        q1 = f"{fallback_query} peptide dosing protocol mcg"
        q2 = f"{fallback_query} reconstitution dosing units BAC water"

    from tavily import TavilyClient

    tv = TavilyClient(api_key=tavily_key)
    results: list[dict[str, Any]] = []
    for q in (q1, q2):
        try:
            response = tv.search(query=q, max_results=5)
        except Exception as e:
            results.append({"query": q, "error": str(e), "results": []})
            continue
        compact = []
        for item in (response.get("results") or [])[:5]:
            compact.append(
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "content": item.get("content"),
                }
            )
        results.append({"query": q, "results": compact})

    return {
        "peptide": peptide_text or None,
        "purpose": purpose_text or None,
        "queries": [q1, q2],
        "results": results,
    }
