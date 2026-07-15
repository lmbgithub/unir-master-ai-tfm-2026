import enum
from uuid import UUID

from pydantic import BaseModel


class AttachmentType(str, enum.Enum):
    IMAGE = "image"
    PDF = "pdf"
    AUDIO = "audio"


class AttachmentStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class AttachmentCreatedMsg(BaseModel):
    attachment_id: UUID
    type: AttachmentType
    storage_path: str


class AttachmentProcessedMsg(BaseModel):
    attachment_id: UUID
    status: AttachmentStatus
    transcription: str | None
    error: str | None
