from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.exc import NoResultFound

from ..dependencies import CurrentUser, DbSession, SettingsDep
from ..enums import AttachmentKind
from ..schemas.api.attachment import AttachmentResponse
from ..services import attachment_service

router = APIRouter(tags=["attachments"])


@router.post(
    "/cases/{case_id}/steps/{step_id}/attachments",
    response_model=AttachmentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_attachment(
    case_id: UUID,
    step_id: UUID,
    file: UploadFile,
    kind: Annotated[AttachmentKind, Form()],
    db: DbSession,
    settings: SettingsDep,
    _: CurrentUser,
) -> AttachmentResponse:
    record = await attachment_service.upload(db, step_id, file, kind, settings.storage_path)
    return AttachmentResponse.model_validate(record)


@router.get(
    "/cases/{case_id}/steps/{step_id}/attachments",
    response_model=list[AttachmentResponse],
)
async def list_attachments(
    case_id: UUID,
    step_id: UUID,
    db: DbSession,
    _: CurrentUser,
) -> list[AttachmentResponse]:
    records = await attachment_service.list_for_step(db, step_id)
    return [AttachmentResponse.model_validate(r) for r in records]


@router.delete(
    "/cases/{case_id}/steps/{step_id}/attachments/{attachment_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_attachment(
    case_id: UUID,
    step_id: UUID,  # noqa: ARG001
    attachment_id: UUID,
    db: DbSession,
    _: CurrentUser,
) -> None:
    try:
        await attachment_service.delete(db, attachment_id)
    except NoResultFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Attachment not found")


@router.get("/attachments/{attachment_id}/download")
async def download_attachment(
    attachment_id: UUID,
    db: DbSession,
    _: CurrentUser,
) -> FileResponse:
    try:
        record = await attachment_service.get(db, attachment_id)
    except NoResultFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Attachment not found")
    path = Path(record.storage_path)
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found on disk")
    return FileResponse(
        path=str(path),
        media_type=record.mime_type,
        filename=record.original_filename,
    )


@router.get("/attachments/{attachment_id}", response_model=AttachmentResponse)
async def get_attachment(
    attachment_id: UUID,
    db: DbSession,
    _: CurrentUser,
) -> AttachmentResponse:
    try:
        record = await attachment_service.get(db, attachment_id)
    except NoResultFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Attachment not found")
    return AttachmentResponse.model_validate(record)
