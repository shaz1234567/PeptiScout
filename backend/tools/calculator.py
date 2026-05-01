"""Tool A — deterministic reconstitution calculator (no LLM)."""

from typing import Any
from typing import Literal

from pydantic import BaseModel, Field, field_serializer


class CalculateRequest(BaseModel):
    vial_mg: float = Field(gt=0, description="Vial strength in mg")
    water_mL: float = Field(gt=0, description="BAC or sterile water volume in mL")
    dose_mcg: float = Field(gt=0, description="Target dose in mcg")
    syringe_type: Literal["U100", "U40"]


class CalculateResponse(BaseModel):
    concentration_mcg_per_mL: float
    inject_mL: float
    syringe_units: int
    label: str

    @field_serializer("concentration_mcg_per_mL")
    def serialize_concentration(self, v: float) -> float | int:
        if abs(v - round(v)) < 1e-9:
            return int(round(v))
        return v


def _format_mL(value: float) -> str:
    """Stable string for label (e.g. 0.1 not 0.1000)."""
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def calculate_reconstitution(req: CalculateRequest) -> CalculateResponse:
    concentration_mcg_per_mL = (req.vial_mg * 1000) / req.water_mL
    inject_mL = round(req.dose_mcg / concentration_mcg_per_mL, 6)
    if req.syringe_type == "U100":
        units_per_mL = 100
    else:
        units_per_mL = 40
    syringe_units = int(round(inject_mL * units_per_mL))

    label = (
        f"Draw {_format_mL(inject_mL)}mL = {syringe_units} units "
        f"on a {req.syringe_type} syringe"
    )

    return CalculateResponse(
        concentration_mcg_per_mL=concentration_mcg_per_mL,
        inject_mL=inject_mL,
        syringe_units=syringe_units,
        label=label,
    )


def recommend_reconstitution(
    vial_mg: float,
    dose_mcg: float,
    syringe_type: Literal["U100", "U40"] = "U100",
) -> dict[str, Any]:
    """Recommend BAC water volume so a dose lands near a convenient syringe draw."""
    if vial_mg <= 0:
        raise ValueError("vial_mg must be positive")
    if dose_mcg <= 0:
        raise ValueError("dose_mcg must be positive")
    if syringe_type not in ("U100", "U40"):
        raise ValueError("syringe_type must be U100 or U40")

    units_per_mL = 100 if syringe_type == "U100" else 40
    target_units = 10
    target_inject_mL = target_units / units_per_mL
    ideal_water_mL = target_inject_mL * vial_mg * 1000 / dose_mcg
    water_mL = min(3.0, max(1.0, round(ideal_water_mL * 2) / 2))

    req = CalculateRequest(
        vial_mg=vial_mg,
        water_mL=water_mL,
        dose_mcg=dose_mcg,
        syringe_type=syringe_type,
    )
    result = calculate_reconstitution(req)
    payload = result.model_dump()
    payload.update(
        {
            "vial_mg": vial_mg,
            "dose_mcg": dose_mcg,
            "water_mL": water_mL,
            "target_units": target_units,
            "ideal_water_mL": ideal_water_mL,
            "method": "rounded_to_nearest_0.5mL_between_1.0_and_3.0_targeting_15_units",
        }
    )
    return payload
