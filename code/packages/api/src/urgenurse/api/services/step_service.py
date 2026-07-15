from datetime import datetime, timezone
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..enums import CaseStepStatus
from ..models.case import Case
from ..models.case_step import CaseStep
from ..schemas.api.case_step import CaseStepCreate


async def add_step(db: AsyncSession, case_id: UUID, data: CaseStepCreate) -> CaseStep:
    result = await db.execute(select(Case).where(Case.id == case_id))
    if result.scalar_one_or_none() is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")

    step = CaseStep(
        case_id=case_id,
        type=data.type,
        description=data.description,
        assigned_to=data.assigned_to,
        status=CaseStepStatus.created,
    )
    db.add(step)
    await db.commit()
    return await _get_step_with_attachments(db, step.id)


async def list_steps(db: AsyncSession, case_id: UUID) -> list[CaseStep]:
    result = await db.execute(
        select(CaseStep)
        .where(CaseStep.case_id == case_id)
        .options(selectinload(CaseStep.attachments))
        .order_by(CaseStep.created_at.asc())
    )
    return list(result.scalars().all())


async def update_status(
    db: AsyncSession, step_id: UUID, status: CaseStepStatus, error_message: str | None = None
) -> CaseStep:
    result = await db.execute(select(CaseStep).where(CaseStep.id == step_id))
    step = result.scalar_one_or_none()
    if step is None:
        raise HTTPException(status_code=404, detail="Step not found")

    step.status = status
    if status == CaseStepStatus.in_progress and step.started_at is None:
        step.started_at = datetime.now(timezone.utc)
    if error_message is not None:
        step.error_message = error_message

    await db.commit()
    return await _get_step_with_attachments(db, step_id)


async def _get_step_with_attachments(db: AsyncSession, step_id: UUID) -> CaseStep:
    result = await db.execute(
        select(CaseStep).where(CaseStep.id == step_id).options(selectinload(CaseStep.attachments))
    )
    return result.scalar_one()
