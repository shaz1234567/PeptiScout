"""LangGraph ReAct agent — full-agent mode (Step 8).

Linear chain: router → proactive_check → calculator → rag_retriever → vlm_analyzer → source_vetter → synthesizer
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import operator
from typing import Annotated, Any, TypedDict

logger = logging.getLogger(__name__)

# Ablation flags (PEPTISCOUT_CURSOR_CONTEXT.md / Step 9)
DISABLE_CALCULATOR = False
DISABLE_REACT = False
DISABLE_RAG = False
DISABLE_FINETUNE = False

_GROWTH_FACTOR_HINTS = (
    "bpc-157",
    "tb-500",
    "thymosin",
    "ipamorelin",
    "cjc-1295",
    "sermorelin",
    "ghrp-6",
    "ghrp-2",
    "igf-1",
    "mk-677",
    "ibutamoren",
)


def _react_block(thought: str, action: str, observation: str) -> str:
    return (
        f"Thought: {thought}\n"
        f"Action: {action}\n"
        f"Observation: {observation}\n"
    )


class AgentState(TypedDict, total=False):
    user_text: str
    image_base64: str | None
    image_type: str | None
    trace_lines: Annotated[list[str], operator.add]

    use_calculator: bool
    vial_mg: float | None
    water_mL: float | None
    dose_mcg: float | None
    syringe_type: str
    peptide_name: str | None
    purpose: str | None
    needs_dose_research: bool
    growth_factor_peptide: bool
    vendor_name: str | None
    rag_query: str

    proactive_note: str | None
    # When True, skip deterministic calculator math for this request only (full-agent-no-calculator).
    force_skip_calculator: bool
    dose_rag_chunks: list[dict[str, Any]] | None
    dose_tavily_result: dict[str, Any] | None
    recommended_dose_mcg: float | None
    dose_rationale: str | None
    dose_evidence_sources: list[str] | None
    calculator_result: dict[str, Any] | None
    rag_chunks: list[dict[str, Any]] | None
    vlm_result: dict[str, Any] | None
    vet_result: dict[str, Any] | None

    protocol: str
    moa: str
    good_bad: str
    audit_trail: str


_ROUTER_SCHEMA = """Analyze the user message for PeptiScout tool routing. Return JSON only:
{
  "use_calculator": boolean,
  "vial_mg": number | null,
  "water_mL": number | null,
  "dose_mcg": number | null,
  "syringe_type": "U100" | "U40",
  "peptide_name": string | null,
  "purpose": string | null,
  "needs_dose_research": boolean,
  "growth_factor_peptide": boolean,
  "vendor_name": string | null,
  "rag_query": string
}
Rules:
- use_calculator true if the user asks for reconstitution/dosage math and vial_mg can be inferred. dose_mcg and water_mL may be null.
- needs_dose_research true if this is a dosage/reconstitution query with vial_mg but no explicit dose_mcg.
- peptide_name: extract the peptide name if present (e.g. BPC-157, TB-500, Ipamorelin).
- purpose: extract the condition or purpose if present (e.g. tendon healing, gut healing); else null.
- growth_factor_peptide true if the query discusses peptides that affect growth/healing/GH axis (e.g. BPC-157, TB-500, Ipamorelin, CJC-1295, Sermorelin, GHRP, IGF).
- vendor_name: extract a vendor/shop name if the user asks about legitimacy, trust, or buying from a named supplier; else null.
- rag_query: concise search string for PubMed retrieval. For missing-dose reconstitution questions, include standard dosing protocol, peptide, purpose, mcg, and reconstitution.
"""


async def _node_router(state: AgentState) -> dict[str, Any]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    user_text = state.get("user_text", "")
    completion = await client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _ROUTER_SCHEMA},
            {"role": "user", "content": user_text},
        ],
    )
    raw = completion.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}

    use_calc = bool(data.get("use_calculator"))
    syringe = data.get("syringe_type") or "U100"
    if syringe not in ("U100", "U40"):
        syringe = "U100"

    peptide = data.get("peptide_name")
    if isinstance(peptide, str):
        peptide = peptide.strip() or None
    else:
        peptide = None

    purpose = data.get("purpose")
    if isinstance(purpose, str):
        purpose = purpose.strip() or None
    else:
        purpose = None

    gf = bool(data.get("growth_factor_peptide"))
    low = user_text.lower()
    if any(h in low for h in _GROWTH_FACTOR_HINTS):
        gf = True

    vendor = data.get("vendor_name")
    if isinstance(vendor, str):
        vendor = vendor.strip() or None
    else:
        vendor = None

    rag_q = str(data.get("rag_query") or user_text).strip() or user_text

    vm = data.get("vial_mg")
    wm = data.get("water_mL")
    dm = data.get("dose_mcg")
    try:
        vm_f = float(vm) if vm is not None else None
        wm_f = float(wm) if wm is not None else None
        dm_f = float(dm) if dm is not None else None
    except (TypeError, ValueError):
        vm_f = wm_f = dm_f = None

    dosage_hint = any(
        h in low
        for h in (
            "bac water",
            "reconstitute",
            "reconstitution",
            "how much water",
            "how much bac",
            "units",
            "draw",
            "syringe",
            "dose",
            "dosing",
        )
    )
    needs_dose_research = bool(data.get("needs_dose_research"))
    if dosage_hint and vm_f is not None and vm_f > 0 and dm_f is None:
        needs_dose_research = True

    if needs_dose_research and peptide:
        rag_q = (
            f"standard dosing protocol {peptide} {purpose or ''} "
            "mcg reconstitution BAC water"
        ).strip()

    if DISABLE_CALCULATOR or state.get("force_skip_calculator"):
        use_calc = False
    elif needs_dose_research:
        use_calc = vm_f is not None and vm_f > 0
    else:
        use_calc = bool(use_calc and vm_f is not None and vm_f > 0 and dm_f is not None and dm_f > 0)

    action = (
        f"router(plan calculator={use_calc}, dose_research={needs_dose_research}, "
        f"rag_query={rag_q[:80]!r}, "
        f"vendor={vendor!r}, growth_factor={gf})"
    )
    obs = json.dumps(
        {
            "use_calculator": use_calc,
            "vial_mg": vm_f,
            "water_mL": wm_f,
            "dose_mcg": dm_f,
            "syringe_type": syringe,
            "peptide_name": peptide,
            "purpose": purpose,
            "needs_dose_research": needs_dose_research,
            "growth_factor_peptide": gf,
            "vendor_name": vendor,
            "rag_query": rag_q,
        },
        default=str,
    )
    trace = [_react_block("Route user query to tools.", action, obs)]

    return {
        "use_calculator": use_calc,
        "vial_mg": vm_f,
        "water_mL": wm_f,
        "dose_mcg": dm_f,
        "syringe_type": syringe,
        "peptide_name": peptide,
        "purpose": purpose,
        "needs_dose_research": needs_dose_research,
        "growth_factor_peptide": gf,
        "vendor_name": vendor,
        "rag_query": rag_q,
        "trace_lines": trace,
    }


async def _node_proactive_check(state: AgentState) -> dict[str, Any]:
    img = state.get("image_base64")
    has_image = bool(img and str(img).strip())
    gf = bool(state.get("growth_factor_peptide"))
    note: str | None = None
    if gf and not has_image:
        note = (
            "Growth-factor-related peptide context detected. If available, upload a recent "
            "bloodwork panel (IGF-1, CRP, testosterone, LH, FSH) so dosing and safety can be interpreted."
        )
    action = "proactive_check(growth_factor_peptide, image_present)"
    obs = note or "No proactive bloodwork flag needed."
    trace = [_react_block("Check whether to prompt for bloodwork.", action, obs)]
    return {"proactive_note": note, "trace_lines": trace}


async def _node_dose_rag(state: AgentState) -> dict[str, Any]:
    if DISABLE_RAG or not state.get("needs_dose_research"):
        trace = [_react_block("Dose RAG not needed.", "dose_rag(skip)", "Skipped.")]
        return {"dose_rag_chunks": [], "trace_lines": trace}

    from backend.tools.rag_retriever import retrieve_chunks

    q = str(state.get("rag_query") or state.get("user_text") or "").strip()
    try:
        chunks = await asyncio.to_thread(retrieve_chunks, q, 5)
    except Exception as e:
        logger.exception("Dose RAG node failed")
        trace = [_react_block("Retrieve dosing evidence from Pinecone.", "dose_rag(error)", str(e))]
        return {"dose_rag_chunks": [], "trace_lines": trace}

    action = f'retrieve_chunks(query="{q[:120]}", top_k=5)'
    obs = json.dumps(chunks, default=str)[:8000]
    trace = [_react_block("Fetch dosing protocol evidence from Pinecone.", action, obs)]
    return {"dose_rag_chunks": chunks, "trace_lines": trace}


async def _node_dose_tavily(state: AgentState) -> dict[str, Any]:
    if not state.get("needs_dose_research"):
        trace = [_react_block("Dose web search not needed.", "dose_tavily(skip)", "Skipped.")]
        return {"dose_tavily_result": None, "trace_lines": trace}

    from backend.tools.dosing_research import search_dosing_protocols

    query = str(state.get("rag_query") or state.get("user_text") or "").strip()
    try:
        result = await asyncio.to_thread(
            search_dosing_protocols,
            state.get("peptide_name"),
            state.get("purpose"),
            query,
        )
    except Exception as e:
        logger.exception("Dose Tavily node failed")
        trace = [_react_block("Search web for dosing protocols.", "dose_tavily(error)", str(e))]
        return {"dose_tavily_result": {"error": str(e)}, "trace_lines": trace}

    action = f"search_dosing_protocols(peptide={state.get('peptide_name')!r})"
    obs = json.dumps(result, default=str)[:8000]
    trace = [_react_block("Search web/community dosing protocols.", action, obs)]
    return {"dose_tavily_result": result, "trace_lines": trace}


_DOSE_EXTRACT_SYSTEM = """Extract a conservative standard peptide dose from evidence.
Return JSON only:
{
  "recommended_dose_mcg": number | null,
  "dose_rationale": string,
  "evidence_sources": [string],
  "confidence": "low" | "medium" | "high"
}
Rules:
- Use the user's peptide and purpose.
- Prefer mcg per dose. Convert mg to mcg when needed.
- Use RAG/PubMed and Tavily snippets only; do not invent a dose.
- If sources disagree, choose a conservative dose within the overlapping/common range.
- If no dose can be supported, set recommended_dose_mcg to null.
"""


async def _node_dose_extractor(state: AgentState) -> dict[str, Any]:
    if not state.get("needs_dose_research"):
        trace = [_react_block("Dose extraction not needed.", "dose_extractor(skip)", "Skipped.")]
        return {"trace_lines": trace}
    if state.get("dose_mcg") is not None:
        trace = [_react_block("Explicit dose already provided.", "dose_extractor(skip)", "Skipped.")]
        return {"trace_lines": trace}

    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    bundle = {
        "user_text": state.get("user_text"),
        "peptide_name": state.get("peptide_name"),
        "purpose": state.get("purpose"),
        "rag_chunks": state.get("dose_rag_chunks") or [],
        "tavily": state.get("dose_tavily_result"),
    }
    try:
        completion = await client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _DOSE_EXTRACT_SYSTEM},
                {"role": "user", "content": json.dumps(bundle, default=str)[:120000]},
            ],
        )
        data = json.loads(completion.choices[0].message.content or "{}")
    except Exception as e:
        logger.exception("Dose extraction failed")
        trace = [_react_block("Extract recommended dose.", "dose_extractor(error)", str(e))]
        return {"dose_mcg": None, "trace_lines": trace}

    dose = data.get("recommended_dose_mcg")
    try:
        dose_f = float(dose) if dose is not None else None
    except (TypeError, ValueError):
        dose_f = None
    if dose_f is not None and dose_f <= 0:
        dose_f = None

    rationale = str(data.get("dose_rationale") or "")
    sources = data.get("evidence_sources")
    if not isinstance(sources, list):
        sources = []
    sources = [str(x) for x in sources]
    obs = json.dumps(
        {
            "recommended_dose_mcg": dose_f,
            "dose_rationale": rationale,
            "evidence_sources": sources,
            "confidence": data.get("confidence"),
        },
        default=str,
    )
    trace = [_react_block("Extract recommended dose from RAG and Tavily.", "dose_extractor(gpt-4o-mini)", obs)]
    return {
        "dose_mcg": dose_f,
        "recommended_dose_mcg": dose_f,
        "dose_rationale": rationale,
        "dose_evidence_sources": sources,
        "trace_lines": trace,
    }


async def _node_calculator(state: AgentState) -> dict[str, Any]:
    if (
        DISABLE_CALCULATOR
        or state.get("force_skip_calculator")
        or not state.get("use_calculator")
    ):
        trace = [
            _react_block(
                "Calculator not needed or disabled.",
                "calculator(skip)",
                "Skipped.",
            )
        ]
        return {"calculator_result": None, "trace_lines": trace}
    if state.get("vial_mg") is None or state.get("dose_mcg") is None:
        trace = [
            _react_block(
                "Calculator missing vial size or dose.",
                "calculator(skip)",
                "Skipped because vial_mg or dose_mcg is unavailable.",
            )
        ]
        return {"calculator_result": None, "trace_lines": trace}

    from backend.tools.calculator import (
        CalculateRequest,
        calculate_reconstitution,
        recommend_reconstitution,
    )

    try:
        vial_mg = float(state["vial_mg"])
        dose_mcg = float(state["dose_mcg"])
        syringe_type = state.get("syringe_type") or "U100"
        water_mL = state.get("water_mL")
        if water_mL is None:
            payload = recommend_reconstitution(
                vial_mg=vial_mg,
                dose_mcg=dose_mcg,
                syringe_type=syringe_type,  # type: ignore[arg-type]
            )
            action = (
                f"recommend_reconstitution(vial_mg={vial_mg}, dose_mcg={dose_mcg}, "
                f"syringe_type={syringe_type})"
            )
        else:
            req = CalculateRequest(
                vial_mg=vial_mg,
                water_mL=float(water_mL),
                dose_mcg=dose_mcg,
                syringe_type=syringe_type,  # type: ignore[arg-type]
            )
            result = calculate_reconstitution(req)
            payload = result.model_dump()
            action = (
                f"calculator(vial_mg={req.vial_mg}, water_mL={req.water_mL}, "
                f"dose_mcg={req.dose_mcg}, syringe_type={req.syringe_type})"
            )
    except Exception as e:
        logger.exception("Calculator node failed")
        trace = [
            _react_block("Run deterministic syringe math.", "calculator(error)", str(e))
        ]
        return {"calculator_result": None, "trace_lines": trace}

    obs = json.dumps(payload, default=str)
    trace = [_react_block("Compute reconstitution and units.", action, obs)]
    return {"calculator_result": payload, "trace_lines": trace}


async def _node_rag(state: AgentState) -> dict[str, Any]:
    if DISABLE_RAG:
        trace = [_react_block("RAG disabled by flag.", "rag_retriever(skip)", "Skipped.")]
        return {"rag_chunks": [], "trace_lines": trace}

    from backend.tools.rag_retriever import retrieve_chunks

    q = str(state.get("rag_query") or state.get("user_text") or "").strip()
    try:
        chunks = await asyncio.to_thread(retrieve_chunks, q, 3)
    except Exception as e:
        logger.exception("RAG node failed")
        trace = [_react_block("Retrieve PubMed chunks.", "rag_retriever(error)", str(e))]
        return {"rag_chunks": [], "trace_lines": trace}

    action = f'retrieve_chunks(query="{q[:120]}")'
    obs = json.dumps(chunks, default=str)[:8000]
    trace = [_react_block("Fetch top-3 similar abstract chunks from Pinecone.", action, obs)]
    return {"rag_chunks": chunks, "trace_lines": trace}


async def _node_vlm(state: AgentState) -> dict[str, Any]:
    img = state.get("image_base64")
    if not img or not str(img).strip():
        trace = [_react_block("No image attached.", "vlm_analyzer(skip)", "Skipped.")]
        return {"vlm_result": None, "trace_lines": trace}

    from backend.tools.vlm_analyzer import analyze_bloodwork_image

    mime = state.get("image_type") or "image/png"
    try:
        result = await analyze_bloodwork_image(str(img), str(mime))
    except Exception as e:
        logger.exception("VLM node failed")
        trace = [_react_block("Read biomarkers from lab image.", "vlm_analyzer(error)", str(e))]
        return {"vlm_result": None, "trace_lines": trace}

    action = "analyze_bloodwork_image(...)"
    obs = json.dumps(result, default=str)[:8000]
    trace = [_react_block("Extract biomarkers from bloodwork image.", action, obs)]
    return {"vlm_result": result, "trace_lines": trace}


async def _node_vet(state: AgentState) -> dict[str, Any]:
    vendor = state.get("vendor_name")
    if not vendor:
        trace = [_react_block("No vendor to vet.", "source_vetter(skip)", "Skipped.")]
        return {"vet_result": None, "trace_lines": trace}

    from backend.tools.source_vetter import vet_vendor

    try:
        result = await asyncio.to_thread(vet_vendor, vendor)
    except Exception as e:
        logger.exception("Vetting node failed")
        trace = [_react_block("Vet vendor reputation and COA signals.", "vet_vendor(error)", str(e))]
        return {"vet_result": None, "trace_lines": trace}

    action = f'vet_vendor(vendor_name="{vendor}")'
    obs = json.dumps(result, default=str)[:8000]
    trace = [_react_block("Search COA and Reddit signals for vendor.", action, obs)]
    return {"vet_result": result, "trace_lines": trace}


_SYNTH_SYSTEM = """You are PeptiScout's synthesizer. Combine tool outputs into one JSON object ONLY.
Keys exactly: "protocol", "moa", "good_bad", "audit_trail".

- protocol: dosing / reconstitution / schedule. If calculator JSON is provided, quote its numbers and label.
  If dose research was performed, include the recommended dose, rationale, BAC water amount,
  concentration, mL/units to draw, and note this is research guidance, not medical advice.
- moa: mechanism at pathway level using RAG text when available.
- good_bad: benefits and contraindications; mention vetting or bloodwork if provided.
- audit_trail: cite PubMed IDs from RAG chunks (e.g. PMID 12345678). Do not invent PMIDs.

If a tool was skipped, do not fabricate its content; rely on user text and other tools.
"""


async def _node_synthesizer(state: AgentState) -> dict[str, Any]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    bundle = {
        "user_text": state.get("user_text"),
        "proactive_note": state.get("proactive_note"),
        "dose_research_rag": state.get("dose_rag_chunks"),
        "dose_research_tavily": state.get("dose_tavily_result"),
        "recommended_dose_mcg": state.get("recommended_dose_mcg"),
        "dose_rationale": state.get("dose_rationale"),
        "dose_evidence_sources": state.get("dose_evidence_sources"),
        "calculator": state.get("calculator_result"),
        "rag_chunks": state.get("rag_chunks"),
        "bloodwork": state.get("vlm_result"),
        "vendor_vetting": state.get("vet_result"),
    }
    user_payload = json.dumps(bundle, default=str)[:120000]

    completion = await client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYNTH_SYSTEM},
            {"role": "user", "content": user_payload},
        ],
    )
    raw = completion.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {
            "protocol": raw[:2000],
            "moa": "",
            "good_bad": "",
            "audit_trail": "",
        }

    proto = str(data.get("protocol", ""))
    moa = str(data.get("moa", ""))
    gb = str(data.get("good_bad", ""))
    audit = str(data.get("audit_trail", ""))

    syn_trace = _react_block(
        "Synthesize final PeptiScout JSON from tool observations.",
        "synthesizer(gpt-4o-mini, json_object)",
        "Returned structured protocol / MOA / good_bad / audit_trail.",
    )

    trace_update: list[str] = [syn_trace]
    if DISABLE_REACT:
        trace_update = []

    out: dict[str, Any] = {
        "protocol": proto,
        "moa": moa,
        "good_bad": gb,
        "audit_trail": audit,
        "trace_lines": trace_update,
    }
    return out


def build_graph():
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(AgentState)
    g.add_node("router", _node_router)
    g.add_node("proactive_check", _node_proactive_check)
    g.add_node("dose_rag", _node_dose_rag)
    g.add_node("dose_tavily", _node_dose_tavily)
    g.add_node("dose_extractor", _node_dose_extractor)
    g.add_node("calculator", _node_calculator)
    g.add_node("rag_retriever", _node_rag)
    g.add_node("vlm_analyzer", _node_vlm)
    g.add_node("source_vetter", _node_vet)
    g.add_node("synthesizer", _node_synthesizer)

    g.add_edge(START, "router")
    g.add_edge("router", "proactive_check")
    g.add_edge("proactive_check", "dose_rag")
    g.add_edge("dose_rag", "dose_tavily")
    g.add_edge("dose_tavily", "dose_extractor")
    g.add_edge("dose_extractor", "calculator")
    g.add_edge("calculator", "rag_retriever")
    g.add_edge("rag_retriever", "vlm_analyzer")
    g.add_edge("vlm_analyzer", "source_vetter")
    g.add_edge("source_vetter", "synthesizer")
    g.add_edge("synthesizer", END)

    return g.compile()


_compiled = None


def get_compiled_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled


async def run_full_agent(
    user_text: str,
    image_base64: str | None,
    image_type: str | None,
    *,
    disable_calculator: bool = False,
) -> dict[str, Any]:
    """Run the full agent; returns dict with protocol, moa, good_bad, audit_trail, react_trace."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    graph = get_compiled_graph()
    initial: AgentState = {
        "user_text": user_text,
        "image_base64": image_base64,
        "image_type": image_type,
        "trace_lines": [],
        "force_skip_calculator": disable_calculator,
    }
    final = await graph.ainvoke(initial)
    lines = list(final.get("trace_lines") or [])
    if DISABLE_REACT:
        react_trace = ""
    else:
        react_trace = "\n".join(lines).strip()

    return {
        "protocol": str(final.get("protocol", "")),
        "moa": str(final.get("moa", "")),
        "good_bad": str(final.get("good_bad", "")),
        "audit_trail": str(final.get("audit_trail", "")),
        "react_trace": react_trace,
    }
