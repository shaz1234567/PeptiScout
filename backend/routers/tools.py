import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.tools.calculator import CalculateRequest, CalculateResponse, calculate_reconstitution
from backend.tools.rag_retriever import RagChunk, retrieve_chunks
from backend.tools.source_vetter import vet_vendor
from backend.tools.vlm_analyzer import analyze_bloodwork_image

router = APIRouter(tags=["tools"])


@router.post("/calculate", response_model=CalculateResponse)
def post_calculate(body: CalculateRequest) -> CalculateResponse:
    return calculate_reconstitution(body)


# --- Tool B: RAG ---


class RagRequest(BaseModel):
    query: str = Field(min_length=1, description="Semantic search query over PubMed chunks")


@router.post("/rag", response_model=list[RagChunk])
async def post_rag(body: RagRequest) -> list[RagChunk]:
    try:
        rows = await asyncio.to_thread(retrieve_chunks, body.query, 3)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"RAG retrieval failed: {e!s}") from e
    return [RagChunk(**r) for r in rows]


# --- Tool C: VLM bloodwork ---


class BloodworkRequest(BaseModel):
    image_base64: str = Field(min_length=1)
    image_type: str = Field(default="image/png", description="MIME type e.g. image/png, image/jpeg")


class BloodworkResponse(BaseModel):
    model_config = {"populate_by_name": True}

    IGF_1: str = Field(validation_alias="IGF-1", serialization_alias="IGF-1")
    CRP: str
    Testosterone: str
    LH: str
    FSH: str
    flags: list[str]


@router.post("/analyze-bloodwork", response_model=BloodworkResponse)
async def post_analyze_bloodwork(body: BloodworkRequest) -> BloodworkResponse:
    try:
        data = await analyze_bloodwork_image(body.image_base64, body.image_type)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Bloodwork analysis failed: {e!s}") from e
    return BloodworkResponse.model_validate(data)


# --- Tool D: Vendor vetting ---


class VettingRequest(BaseModel):
    vendor_name: str = Field(min_length=1)


class VettingResponse(BaseModel):
    vendor: str
    country: str | None = None
    coa_available: bool
    coa_url: str | None = None
    reddit_sentiment: str
    flags: list[str]
    summary: str


@router.post("/vetting", response_model=VettingResponse)
async def post_vetting(body: VettingRequest) -> VettingResponse:
    try:
        data = await asyncio.to_thread(vet_vendor, body.vendor_name)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Vendor vetting failed: {e!s}") from e
    return VettingResponse(**data)
