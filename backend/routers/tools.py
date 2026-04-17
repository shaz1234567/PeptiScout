from fastapi import APIRouter

from backend.tools.calculator import CalculateRequest, CalculateResponse, calculate_reconstitution

router = APIRouter(tags=["tools"])


@router.post("/calculate", response_model=CalculateResponse)
def post_calculate(body: CalculateRequest) -> CalculateResponse:
    return calculate_reconstitution(body)
