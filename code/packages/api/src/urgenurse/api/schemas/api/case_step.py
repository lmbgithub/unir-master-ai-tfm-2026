from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from ...enums import CaseStepStatus, CaseStepType
from .attachment import AttachmentResponse


class CaseStepCreate(BaseModel):
    type: CaseStepType
    description: str | None = None
    assigned_to: str | None = None


class CaseStepStatusUpdate(BaseModel):
    status: CaseStepStatus
    error_message: str | None = None


class CaseStepResponse(BaseModel):
    id: UUID
    case_id: UUID
    type: CaseStepType
    status: CaseStepStatus
    assigned_to: str | None
    description: str | None
    error_message: str | None
    meta: dict | None = None
    created_at: datetime
    started_at: datetime | None
    updated_at: datetime
    attachments: list[AttachmentResponse]

    model_config = {"from_attributes": True}
