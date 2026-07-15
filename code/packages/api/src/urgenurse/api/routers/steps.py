from uuid import UUID

from fastapi import APIRouter, status

from ..dependencies import CurrentUser, DbSession, JobManagerDep
from ..schemas.api.case_step import (
    CaseStepCreate,
    CaseStepResponse,
    CaseStepStatusUpdate,
)
from ..services import case_service, step_service

router = APIRouter(prefix="/cases/{case_id}/steps", tags=["steps"])


@router.post("", response_model=CaseStepResponse, status_code=status.HTTP_201_CREATED)
async def add_step(
    case_id: UUID,
    body: CaseStepCreate,
    db: DbSession,
    _: CurrentUser,
) -> CaseStepResponse:
    step = await step_service.add_step(db, case_id, body)
    return CaseStepResponse.model_validate(step)


@router.get("", response_model=list[CaseStepResponse])
async def list_steps(
    case_id: UUID,
    db: DbSession,
    _: CurrentUser,
) -> list[CaseStepResponse]:
    steps = await step_service.list_steps(db, case_id)
    return [CaseStepResponse.model_validate(s) for s in steps]


@router.post("/{step_id}/submit", status_code=status.HTTP_204_NO_CONTENT)
async def submit_step(
    case_id: UUID,
    step_id: UUID,
    db: DbSession,
    job_manager: JobManagerDep,
    _: CurrentUser,
) -> None:
    await case_service.submit_step(db, case_id, step_id, job_manager)


@router.post("/{step_id}/retry", status_code=status.HTTP_204_NO_CONTENT)
async def retry_triage_step(
    case_id: UUID,
    step_id: UUID,
    db: DbSession,
    job_manager: JobManagerDep,
    _: CurrentUser,
) -> None:
    await case_service.retry_triage_step(db, case_id, step_id, job_manager)


@router.patch("/{step_id}/status", response_model=CaseStepResponse)
async def update_step_status(
    case_id: UUID,
    step_id: UUID,
    body: CaseStepStatusUpdate,
    db: DbSession,
    _: CurrentUser,
) -> CaseStepResponse:
    step = await step_service.update_status(db, step_id, body.status, body.error_message)
    return CaseStepResponse.model_validate(step)
