import json
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Form, HTTPException, Query, status

from ..dependencies import CurrentUser, DbSession
from ..enums import CasePhase
from ..schemas.api.case import (
    CaseCreate,
    CasePhaseUpdate,
    CaseResponse,
    CaseStatsResponse,
    CaseUpdate,
    PaginatedCasesResponse,
)
from ..services import case_service

router = APIRouter(prefix="/cases", tags=["cases"])


@router.post("", response_model=CaseResponse, status_code=status.HTTP_201_CREATED)
async def create_case(
    db: DbSession,
    _: CurrentUser,
    patient_info: Annotated[str, Form()],
    chief_complaint: Annotated[str, Form()],
) -> CaseResponse:
    try:
        patient_info_dict = json.loads(patient_info)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="patient_info must be a valid JSON object",
        )
    body = CaseCreate(patient_info=patient_info_dict, chief_complaint=chief_complaint)
    case = await case_service.create_case(db, body)
    return CaseResponse.model_validate(case)


@router.get("", response_model=PaginatedCasesResponse)
async def list_cases(
    db: DbSession,
    _: CurrentUser,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    search: str | None = Query(default=None),
    status: str | None = Query(default=None),
) -> PaginatedCasesResponse:
    phase = CasePhase(status) if status else None
    cases, total = await case_service.list_cases(db, page, page_size, search, phase)
    return PaginatedCasesResponse(
        items=[CaseResponse.model_validate(c) for c in cases],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/stats", response_model=CaseStatsResponse)
async def get_stats(
    db: DbSession,
    _: CurrentUser,
) -> CaseStatsResponse:
    stats = await case_service.get_stats(db)
    return CaseStatsResponse(**stats)


@router.get("/{case_id}", response_model=CaseResponse)
async def get_case(
    case_id: UUID,
    db: DbSession,
    _: CurrentUser,
) -> CaseResponse:
    case = await case_service.get_case(db, case_id)
    return CaseResponse.model_validate(case)


@router.patch("/{case_id}", response_model=CaseResponse)
async def update_case(
    case_id: UUID,
    body: CaseUpdate,
    db: DbSession,
    _: CurrentUser,
) -> CaseResponse:
    case = await case_service.update_case(db, case_id, body)
    return CaseResponse.model_validate(case)


@router.patch("/{case_id}/phase", response_model=CaseResponse)
async def update_phase(
    case_id: UUID,
    body: CasePhaseUpdate,
    db: DbSession,
    _: CurrentUser,
) -> CaseResponse:
    case = await case_service.update_phase(db, case_id, body.phase)
    return CaseResponse.model_validate(case)


@router.post("/{case_id}/confirm-triage", response_model=CaseResponse)
async def confirm_triage(
    case_id: UUID,
    db: DbSession,
    _: CurrentUser,
) -> CaseResponse:
    case = await case_service.confirm_triage(db, case_id)
    return CaseResponse.model_validate(case)
