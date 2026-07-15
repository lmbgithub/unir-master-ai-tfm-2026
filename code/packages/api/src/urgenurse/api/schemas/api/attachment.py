from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from ...enums import AttachmentKind, AttachmentStatus


class AttachmentResponse(BaseModel):
    id: UUID
    case_step_id: UUID
    original_filename: str
    mime_type: str
    storage_path: str
    kind: AttachmentKind
    status: AttachmentStatus
    transcription: str | None
    summary: str | None
    ner: dict[str, str] | None
    sbar: dict[str, str] | None
    confidence: float | None
    created_at: datetime

    model_config = {"from_attributes": True}
