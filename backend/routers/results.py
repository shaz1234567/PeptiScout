import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["results"])

_RESULTS_PATH = Path(__file__).resolve().parents[1] / "data" / "results.json"


@router.get("/results")
def get_results():
    if not _RESULTS_PATH.is_file():
        raise HTTPException(status_code=404, detail="results.json not found")
    try:
        with _RESULTS_PATH.open(encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=503, detail="results.json is not valid JSON") from e
