"""Tool B — Pinecone semantic RAG over PubMed chunks (OpenAI embeddings + Pinecone)."""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field

EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_INDEX_NAME = "peptiscout"


class RagChunk(BaseModel):
    """One retrieved chunk for synthesis / agent."""

    pmid: str = Field(description="PubMed ID")
    text: str = Field(description="Abstract chunk text")
    score: float = Field(description="Similarity score from Pinecone")


def retrieve_chunks(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """
    Embed ``query`` with text-embedding-3-small and query Pinecone for top-k matches.

    Returns a JSON-serializable list of dicts: ``pmid``, ``text``, ``score``.
    Requires ``OPENAI_API_KEY``, ``PINECONE_API_KEY``; optional ``PINECONE_INDEX_NAME``
    (default ``peptiscout``).

    Legacy vectors without ``text`` in metadata return ``text`` as empty string until
    ``python -m backend.scripts.generate_dataset --phase pinecone`` is re-run.
    """
    if not query or not query.strip():
        raise ValueError("query must be non-empty")
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set")
    if not os.getenv("PINECONE_API_KEY"):
        raise ValueError("PINECONE_API_KEY is not set")

    from openai import OpenAI
    from pinecone import Pinecone

    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    emb = oai.embeddings.create(model=EMBEDDING_MODEL, input=query.strip()[:8000])
    vector = emb.data[0].embedding

    index_name = os.getenv("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME)
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(index_name)

    q = index.query(vector=vector, top_k=top_k, include_metadata=True)

    matches = getattr(q, "matches", None)
    if matches is None and isinstance(q, dict):
        matches = q.get("matches") or []

    out: list[dict[str, Any]] = []
    for m in matches or []:
        if isinstance(m, dict):
            meta = m.get("metadata") or {}
            score = float(m.get("score", 0.0))
        else:
            meta = getattr(m, "metadata", None) or {}
            score = float(getattr(m, "score", 0.0) or 0.0)
        pmid = str(meta.get("pmid", ""))
        text = str(meta.get("text", ""))
        out.append({"pmid": pmid, "text": text, "score": score})

    return out
