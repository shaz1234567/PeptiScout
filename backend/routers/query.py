"""POST /api/query — baseline GPT-4o modes (Step 4) and fine-tuned Llama + LoRA (Step 6)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# OpenAI and torch/transformers are imported lazily so uvicorn startup stays fast.

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ADAPTER_DIR = _PROJECT_ROOT / "backend" / "models" / "peptide_lora_adapter"
_BASE_MODEL_ID = "unsloth/llama-3-8b-bnb-4bit"

# Verbatim from PEPTISCOUT_CURSOR_CONTEXT.md — BASELINE SYSTEM (without the "System prompt:" label)
_BASELINE_ZERO_SHOT = """You are PeptiScout, an expert AI assistant specializing in research peptides.
When a user asks about a peptide, always respond in this exact structure:

[Protocol]: Provide dosing frequency, reconstitution instructions, and syringe math.
[MOA]: Explain the mechanism of action at the pathway level. Name the specific pathway.
[The Good / The Bad]: List primary benefits and known contraindications.
[The Audit Trail]: Cite at least 2 PubMed studies by PMID.

Be precise. Do not hedge excessively. Do not fabricate PMIDs."""

# Verbatim few-shot examples (prepend before user query)
_FEW_SHOT_EXAMPLES = """EXAMPLE 1:
User: What is the reconstitution dose for 2mg of Semax with 1mL BAC water?
Assistant:
[Protocol]: 2mg in 1mL = 2000mcg/mL. Standard dose is 300mcg = 0.15mL = 15 units U100.
[MOA]: Semax is an ACTH analog that upregulates BDNF and NGF via TrkB receptor activation.
[The Good / The Bad]: Cognitive enhancement, neuroprotection. Contraindicated with MAOIs.
[The Audit Trail]: PMID 19230835, PMID 22750014.

EXAMPLE 2:
User: What co-factors does GHK-Cu require?
Assistant:
[Protocol]: GHK-Cu dose: 1-2mg/day subcutaneous or topical.
[MOA]: Tripeptide that downregulates IL-6 and TNF-α via NF-κB. Upregulates Collagen I/III.
[The Good / The Bad]: Anti-inflammatory, wound healing. Requires Vitamin C and Zinc.
[The Audit Trail]: PMID 25170290, PMID 28759605."""

# Verbatim CoT addition (baseline-cot)
_COT_BLOCK = """Before giving your structured response, think through the following inside <thinking> tags
(these will not be shown to the user):

<thinking>
1. What peptide is being asked about?
2. Does this query involve dosage math? If yes, calculate step by step.
3. Does this peptide require co-factors? List them.
4. Are there any contraindications I must flag?
5. Do I know verified PMIDs for this peptide? List only ones I am certain exist.
</thinking>

Then provide the [Protocol] / [MOA] / [Good-Bad] / [Audit Trail] response."""

_JSON_SUFFIX = """

Respond with a single JSON object only, with exactly these string keys: "protocol", "moa", "good_bad", "audit_trail". Map your [Protocol], [MOA], [The Good / The Bad], and [The Audit Trail] content to those keys respectively. No other keys and no text outside the JSON object."""

BASELINE_MODE = Literal["baseline-zero-shot", "baseline-few-shot", "baseline-cot"]
QUERY_MODE = Literal[
    "baseline-zero-shot",
    "baseline-few-shot",
    "baseline-cot",
    "fine-tuned",
    "fine-tuned-no-rag",
    "full-agent",
    "full-agent-no-calculator",
    "no-reasoning",
    "rag-only-no-finetune",
]

MODEL_NAME = "gpt-4o-mini"


class QueryRequest(BaseModel):
    text: str = Field(min_length=1, description="User question about peptides")
    mode: QUERY_MODE
    image_base64: str | None = Field(
        default=None,
        description="Optional lab image (base64) for full-agent / VLM tool",
    )
    image_type: str | None = Field(
        default=None,
        description="MIME type for image_base64 (e.g. image/png)",
    )


class QueryResponse(BaseModel):
    protocol: str
    moa: str
    good_bad: str
    audit_trail: str
    react_trace: str | None = Field(
        default=None,
        description="ReAct trace (Thought/Action/Observation) for full-agent mode only",
    )


def _system_zero_shot() -> str:
    return _BASELINE_ZERO_SHOT + _JSON_SUFFIX


def _system_cot() -> str:
    return _BASELINE_ZERO_SHOT + "\n\n" + _COT_BLOCK + _JSON_SUFFIX


def _build_messages(mode: BASELINE_MODE, text: str) -> list[dict[str, str]]:
    if mode == "baseline-zero-shot":
        return [
            {"role": "system", "content": _system_zero_shot()},
            {"role": "user", "content": text},
        ]
    if mode == "baseline-few-shot":
        user_content = f"{_FEW_SHOT_EXAMPLES}\n\nUser: {text}"
        return [
            {"role": "system", "content": _system_zero_shot()},
            {"role": "user", "content": user_content},
        ]
    # baseline-cot
    return [
        {"role": "system", "content": _system_cot()},
        {"role": "user", "content": text},
    ]


def _response_from_dict(data: dict) -> QueryResponse:
    try:
        return QueryResponse(
            protocol=str(data["protocol"]),
            moa=str(data["moa"]),
            good_bad=str(data["good_bad"]),
            audit_trail=str(data["audit_trail"]),
            react_trace=data.get("react_trace"),
        )
    except KeyError:
        raise HTTPException(
            status_code=502,
            detail="Model JSON missing required keys: protocol, moa, good_bad, audit_trail.",
        ) from None


def _parse_query_json(raw: str) -> QueryResponse:
    if not raw.strip():
        raise HTTPException(status_code=502, detail="Empty response from language model.")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Model returned non-JSON: %s", raw[:500])
        raise HTTPException(
            status_code=502,
            detail="Model did not return valid JSON.",
        ) from None
    return _response_from_dict(data)


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if "```" not in text:
        return text
    for block in text.split("```"):
        block = block.strip()
        if block.lower().startswith("json"):
            block = block[4:].strip()
        if block.startswith("{"):
            return block
    return text


def _loads_json_object(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def _parse_query_json_loose(raw: str) -> QueryResponse:
    """Parse JSON from local LLM output; allow fences or extra prose around a JSON object."""
    text = _strip_code_fence(raw)
    try:
        data = _loads_json_object(text)
    except json.JSONDecodeError:
        logger.warning("Fine-tuned model returned non-JSON: %s", raw[:500])
        raise HTTPException(
            status_code=502,
            detail="Model did not return valid JSON.",
        ) from None
    return _response_from_dict(data)


_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import AsyncOpenAI

        _client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def _resolve_adapter_path() -> Path:
    env = os.getenv("LORA_ADAPTER_PATH")
    if env:
        return Path(env).expanduser().resolve()
    return _DEFAULT_ADAPTER_DIR.resolve()


def _adapter_ready(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (path / "adapter_config.json").is_file() or any(path.glob("adapter_model*.safetensors"))


_finetuned_lock = threading.Lock()
_finetuned_bundle: tuple[object, object] | None = None


def _load_finetuned_bundle() -> tuple[object, object]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail="Fine-tuned inference dependencies missing. Install backend/requirements.txt (torch, transformers, peft, bitsandbytes, accelerate).",
        ) from e

    adapter_path = _resolve_adapter_path()
    if not _adapter_ready(adapter_path):
        raise HTTPException(
            status_code=503,
            detail=f"LoRA adapter not found or incomplete at {adapter_path}. Set LORA_ADAPTER_PATH or add adapter files under backend/models/peptide_lora_adapter/.",
        )

    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL_ID, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            _BASE_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, str(adapter_path))
        model.eval()
    except Exception as e:
        logger.exception("Failed to load fine-tuned model")
        raise HTTPException(
            status_code=503,
            detail=f"Could not load fine-tuned model (CUDA/bitsandbytes required for 4-bit): {e!s}",
        ) from e

    return model, tokenizer


def _get_finetuned_bundle() -> tuple[object, object]:
    global _finetuned_bundle
    with _finetuned_lock:
        if _finetuned_bundle is None:
            _finetuned_bundle = _load_finetuned_bundle()
        return _finetuned_bundle


def _finetuned_generate_sync(user_text: str) -> str:
    import torch

    model, tokenizer = _get_finetuned_bundle()
    messages = [
        {"role": "system", "content": _system_zero_shot()},
        {"role": "user", "content": user_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    dev = next(model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0, input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


async def _post_query_finetuned(body: QueryRequest) -> QueryResponse:
    try:
        raw = await asyncio.to_thread(_finetuned_generate_sync, body.text)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Fine-tuned generation failed")
        raise HTTPException(status_code=502, detail=f"Fine-tuned inference failed: {e!s}") from e
    return _parse_query_json_loose(raw)


async def _post_query_full_agent(
    body: QueryRequest,
    *,
    disable_calculator: bool = False,
) -> QueryResponse:
    from backend.agent.graph import run_full_agent

    try:
        out = await run_full_agent(
            body.text,
            body.image_base64,
            body.image_type,
            disable_calculator=disable_calculator,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        logger.exception("Full agent failed")
        raise HTTPException(status_code=502, detail=f"Full agent failed: {e!s}") from e
    return QueryResponse(
        protocol=out["protocol"],
        moa=out["moa"],
        good_bad=out["good_bad"],
        audit_trail=out["audit_trail"],
        react_trace=out.get("react_trace"),
    )


async def _post_query_no_reasoning(body: QueryRequest) -> QueryResponse:
    """GPT-4o zero-shot structured JSON only — same contract as baseline-zero-shot, no LangGraph."""
    messages = _build_messages("baseline-zero-shot", body.text)
    client = _get_client()
    try:
        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        from openai import OpenAIError

        if isinstance(e, OpenAIError):
            logger.exception("OpenAI error in no-reasoning")
            raise HTTPException(status_code=502, detail="Upstream LLM request failed.") from e
        raise
    raw = completion.choices[0].message.content or ""
    return _parse_query_json(raw)


async def _post_query_rag_only_no_finetune(body: QueryRequest) -> QueryResponse:
    """GPT-4o + retrieve_chunks context in one chat completion; no LangGraph, no LoRA."""
    from backend.tools.rag_retriever import retrieve_chunks

    chunks = await asyncio.to_thread(retrieve_chunks, body.text, 3)
    ctx = json.dumps(chunks, ensure_ascii=False)
    if len(ctx) > 60000:
        ctx = ctx[:60000] + "\n…"
    user_block = (
        "Use the following retrieved PubMed chunks for citations in audit_trail when possible. "
        "Do not fabricate PMIDs that are not supported by these chunks or general knowledge.\n\n"
        f"{ctx}\n\nUser question:\n{body.text}"
    )
    messages = [
        {"role": "system", "content": _system_zero_shot()},
        {"role": "user", "content": user_block},
    ]
    client = _get_client()
    try:
        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        from openai import OpenAIError

        if isinstance(e, OpenAIError):
            logger.exception("OpenAI error in rag-only-no-finetune")
            raise HTTPException(status_code=502, detail="Upstream LLM request failed.") from e
        raise
    raw = completion.choices[0].message.content or ""
    return _parse_query_json(raw)


@router.post("/query", response_model=QueryResponse)
async def post_query(body: QueryRequest) -> QueryResponse:
    if body.mode in ("fine-tuned", "fine-tuned-no-rag"):
        return await _post_query_finetuned(body)

    if body.mode == "full-agent-no-calculator":
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(
                status_code=503,
                detail="OpenAI API is not configured (missing OPENAI_API_KEY).",
            )
        return await _post_query_full_agent(body, disable_calculator=True)

    if body.mode == "full-agent":
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(
                status_code=503,
                detail="OpenAI API is not configured (missing OPENAI_API_KEY).",
            )
        return await _post_query_full_agent(body)

    if body.mode == "no-reasoning":
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(
                status_code=503,
                detail="OpenAI API is not configured (missing OPENAI_API_KEY).",
            )
        return await _post_query_no_reasoning(body)

    if body.mode == "rag-only-no-finetune":
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(
                status_code=503,
                detail="OpenAI API is not configured (missing OPENAI_API_KEY).",
            )
        return await _post_query_rag_only_no_finetune(body)

    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="OpenAI API is not configured (missing OPENAI_API_KEY).",
        )

    if body.mode not in ("baseline-zero-shot", "baseline-few-shot", "baseline-cot"):
        raise HTTPException(status_code=400, detail=f"Unsupported mode: {body.mode}")

    messages = _build_messages(body.mode, body.text)
    client = _get_client()

    try:
        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        from openai import OpenAIError

        if isinstance(e, OpenAIError):
            logger.exception("OpenAI error in post_query")
            raise HTTPException(status_code=502, detail="Upstream LLM request failed.") from e
        raise

    raw = completion.choices[0].message.content or ""
    return _parse_query_json(raw)
