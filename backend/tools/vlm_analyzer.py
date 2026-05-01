"""Tool C — GPT-4o-Vision biomarker extraction from lab report images."""

from __future__ import annotations

import json
import os
from typing import Any

_EXTRACTION_PROMPT = """Extract the following biomarker values from this lab report image if present:
IGF-1 (ng/mL), CRP (mg/L), Total Testosterone (ng/dL), LH (mIU/mL), FSH (mIU/mL).
Return as JSON only. If a value is not visible or present, return "not found".
Do not return anything other than the JSON object.

Also include a "flags" array of short clinical notes (strings) if any values are abnormal or noteworthy; otherwise use an empty array."""


async def analyze_bloodwork_image(image_base64: str, image_type: str) -> dict[str, Any]:
    """
    Send base64 image to GPT-4o vision; return structured biomarker JSON.

    Keys: ``IGF-1``, ``CRP``, ``Testosterone``, ``LH``, ``FSH``, ``flags``.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set")
    if not image_base64 or not image_base64.strip():
        raise ValueError("image_base64 must be non-empty")

    mime = (image_type or "image/png").strip()
    if not mime.startswith("image/"):
        mime = f"image/{mime}" if "/" not in mime else mime

    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    url = f"data:{mime};base64,{image_base64.strip()}"

    completion = await client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _EXTRACTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ],
    )
    raw = completion.choices[0].message.content
    if not raw:
        raise ValueError("Empty response from vision model")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Vision model did not return valid JSON: {e}") from e

    required = ("IGF-1", "CRP", "Testosterone", "LH", "FSH")
    for k in required:
        if k not in data:
            data[k] = "not found"
    if "flags" not in data or not isinstance(data["flags"], list):
        data["flags"] = []
    data["flags"] = [str(x) for x in data["flags"]]
    return data
