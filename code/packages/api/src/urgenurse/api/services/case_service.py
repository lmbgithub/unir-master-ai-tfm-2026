from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..enums import AttachmentStatus, CasePhase, CaseStepStatus, CaseStepType
from ..models.case import Case
from ..models.case_step import CaseStep
from ..schemas.api.case import CaseCreate, CaseUpdate
from ..utils.job_manager import JobManager


async def create_case(db: AsyncSession, data: CaseCreate) -> Case:
    case = Case(
        patient_info=data.patient_info.model_dump(mode="json"),
        chief_complaint=data.chief_complaint,
        phase=CasePhase.triage,
    )
    db.add(case)
    await db.flush()

    triage_step = CaseStep(
        case_id=case.id,
        type=CaseStepType.triage,
        description=data.chief_complaint,
        status=CaseStepStatus.created,
    )
    db.add(triage_step)
    await db.commit()
    return await get_case(db, case.id)


async def submit_step(db: AsyncSession, case_id: UUID, step_id: UUID, job_manager: JobManager) -> None:
    step = await get_case_step(db, step_id=step_id)
    if step.status != CaseStepStatus.created:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Step cannot be submitted from status '{step.status}'",
        )
    step.status = CaseStepStatus.pending
    await db.commit()
    await job_manager.queue_request(str(case_id), str(step_id))


async def retry_triage_step(db: AsyncSession, case_id: UUID, step_id: UUID, job_manager: JobManager) -> None:
    result = await db.execute(
        select(CaseStep).where(CaseStep.id == step_id).options(selectinload(CaseStep.attachments))
    )
    step = result.scalar_one_or_none()
    if step is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case Step not found")
    if step.status != CaseStepStatus.error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Retry is only allowed from 'error' status, got '{step.status}'",
        )

    for attachment in step.attachments:
        if attachment.status in (AttachmentStatus.error, AttachmentStatus.pending):
            attachment.status = AttachmentStatus.pending
            attachment.transcription = None
            attachment.summary = None
            attachment.ner = None

    step.status = CaseStepStatus.pending
    step.error_message = None
    step.meta = None
    await db.commit()
    await job_manager.queue_request(str(case_id), str(step_id))


async def list_cases(
    db: AsyncSession,
    page: int = 1,
    page_size: int = 20,
    search: str | None = None,
    phase: CasePhase | None = None,
) -> tuple[list[Case], int]:
    query = select(Case)
    if search:
        term = f"%{search}%"
        query = query.where(
            or_(
                Case.chief_complaint.ilike(term),
                func.lower(Case.patient_info["name"].as_string()).contains(search.lower()),
            )
        )
    if phase is not None:
        query = query.where(Case.phase == phase)

    count_result = await db.execute(select(func.count()).select_from(query.subquery()))
    total = count_result.scalar_one()

    result = await db.execute(
        query.options(selectinload(Case.case_steps).selectinload(CaseStep.attachments))
        .order_by(Case.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return list(result.scalars().all()), total


async def get_stats(db: AsyncSession) -> dict[str, int]:
    result = await db.execute(select(Case.phase, func.count()).group_by(Case.phase))
    counts = {phase: count for phase, count in result.all()}
    closed = {CasePhase.closed_success, CasePhase.closed_death, CasePhase.closed_transfer}
    total = sum(counts.values())
    completed = sum(count for phase, count in counts.items() if phase in closed)
    error = counts.get(CasePhase.error, 0)
    return {"total": total, "open": total - completed - error, "completed": completed, "error": error}


async def get_case(db: AsyncSession, case_id: UUID) -> Case:
    result = await db.execute(
        select(Case).options(selectinload(Case.case_steps).selectinload(CaseStep.attachments)).where(Case.id == case_id)
    )
    case = result.scalar_one_or_none()
    if case is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")
    return case


async def get_case_step(db: AsyncSession, step_id: UUID) -> CaseStep:
    result = await db.execute(select(CaseStep).where(CaseStep.id == step_id))
    step = result.scalar_one_or_none()
    if step is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case Step not found")
    return step


async def update_case(db: AsyncSession, case_id: UUID, data: CaseUpdate) -> Case:
    case = await get_case(db, case_id)
    if data.patient_info is not None:
        case.patient_info = data.patient_info.model_dump(mode="json")
    if data.chief_complaint is not None:
        case.chief_complaint = data.chief_complaint
    await db.commit()
    return await get_case(db, case_id)


async def update_phase(db: AsyncSession, case_id: UUID, phase: CasePhase) -> Case:
    case = await get_case(db, case_id)
    case.phase = phase
    await db.commit()
    return await get_case(db, case_id)


async def confirm_triage(db: AsyncSession, case_id: UUID) -> Case:
    return await update_phase(db, case_id, CasePhase.pending_care)
