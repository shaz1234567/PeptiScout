"""Tool A — deterministic reconstitution calculator (no LLM)."""

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
