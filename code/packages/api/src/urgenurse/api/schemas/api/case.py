from datetime import date, datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from ...enums import CasePhase
from .case_step import CaseStepResponse


class PatientInfoCreate(BaseModel):
    name: str = Field(min_length=2)
    gender: Literal["male", "female"]
    date_of_birth: date
    id_number: str = Field(min_length=1)
    blood_type: Literal["A", "B", "O", "AB"]
    blood_rh: bool
    blood_pressure_systolic: int = Field(ge=40, le=300)
    blood_pressure_diastolic: int = Field(ge=20, le=200)
    weight: float = Field(gt=0, le=500)
    height: float = Field(gt=0, le=300)
    pulse: int = Field(ge=20, le=300)
    allergies: list[str] = Field(default_factory=list)
    chronic_conditions: list[str] = Field(default_factory=list)


class CaseCreate(BaseModel):
    patient_info: PatientInfoCreate
    chief_complaint: str = Field(min_length=5)


class CasePhaseUpdate(BaseModel):
    phase: CasePhase


class CaseUpdate(BaseModel):
    patient_info: PatientInfoCreate | None = None
    chief_complaint: str | None = Field(default=None, min_length=5)


class CaseResponse(BaseModel):
    id: UUID
    patient_info: dict
    chief_complaint: str
    esi_level: int | None
    phase: CasePhase
    created_at: datetime
    updated_at: datetime
    case_steps: list[CaseStepResponse]

    model_config = {"from_attributes": True}


class PaginatedCasesResponse(BaseModel):
    items: list[CaseResponse]
    total: int
    page: int
    page_size: int


class CaseStatsResponse(BaseModel):
    total: int
    open: int
    completed: int
    error: int
