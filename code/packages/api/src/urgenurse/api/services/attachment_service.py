import uuid
from pathlib import Path
from uuid import UUID

import aiofiles
from fastapi import HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..enums import AttachmentKind, AttachmentStatus, CaseStepStatus
from ..models.attachment import Attachment
from ..models.case_step import CaseStep


async def upload(
    db: AsyncSession,
    step_id: UUID,
    file: UploadFile,
    kind: AttachmentKind,
    storage_root: str,
) -> Attachment:
    step_result = await db.execute(select(CaseStep).where(CaseStep.id == step_id))
    step = step_result.scalar_one_or_none()
    if step is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Step not found")
    if step.status not in (CaseStepStatus.created, CaseStepStatus.error):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Attachments can only be added to steps in 'created' or 'error' status",
        )
    attachment_id = uuid.uuid4()
    dest_dir = Path(storage_root) / str(attachment_id)
    dest_dir.mkdir(parents=True, exist_ok=True)

    filename = file.filename or "upload"
    dest = dest_dir / filename
    async with aiofiles.open(dest, "wb") as f:
        content = await file.read()
        await f.write(content)

    record = Attachment(
        id=attachment_id,
        case_step_id=step_id,
        original_filename=filename,
        mime_type=file.content_type or "application/octet-stream",
        storage_path=str(dest),
        kind=kind,
        status=AttachmentStatus.pending,
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)

    return record


async def get(db: AsyncSession, attachment_id: UUID) -> Attachment:
    result = await db.execute(select(Attachment).where(Attachment.id == attachment_id))
    return result.scalar_one()


async def list_for_step(db: AsyncSession, step_id: UUID) -> list[Attachment]:
    result = await db.execute(select(Attachment).where(Attachment.case_step_id == step_id))
    return list(result.scalars().all())


async def delete(db: AsyncSession, attachment_id: UUID) -> None:
    record = await get(db, attachment_id)
    storage = Path(record.storage_path)
    if storage.exists():
        storage.unlink(missing_ok=True)
        parent = storage.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
    await db.delete(record)
    await db.commit()
