from pydantic import BaseModel
from typing import Literal
from enum import Enum


class AgentRequestPayloadFile(BaseModel):
    attachment_id: str
    filename: str
    mime_type: Literal[
        "image/jpeg",
        "image/png",
        "image/gif",
        "audio/mpeg",
        "audio/wav",
        "application/pdf",
    ]
    path: str

    @property
    def is_image(self) -> bool:
        return self.mime_type in ["image/gif", "image/jpeg", "image/png"]

    @property
    def is_audio(self) -> bool:
        return self.mime_type in ["audio/mpeg", "audio/wav"]

    @property
    def is_pdf(self) -> bool:
        return self.mime_type in ["application/pdf"]

    @property
    def is_doc(self) -> bool:
        return self.is_pdf or self.is_image


class AttachmentTranscriptions(BaseModel):
    name: str
    content: str
    summary: str | None = None
    ner: dict[str, str] | None = None
    sbar: dict[str, str] | None = None
    confidence: float | None = None


class AgentRequestPayloadTriage(BaseModel):
    case_id: str
    patient: dict
    description: str
    attachments_transcriptions: list[AttachmentTranscriptions] | None = None


class AgentRequest(BaseModel):
    id: str
    payload: AgentRequestPayloadFile | AgentRequestPayloadTriage


class ESILevels(Enum):
    LEVEL1 = 1
    LEVEL2 = 2
    LEVEL3 = 3
    LEVEL4 = 4
    LEVEL5 = 5


class AgentResponsePayloadTriage(BaseModel):
    valid: bool
    missing_fields: list[str] | None = None
    esi_level: ESILevels
    analysis: str


class AgentResponse(BaseModel):
    id: str
    ok: bool
    error: str | None = None
    payload: AttachmentTranscriptions | AgentResponsePayloadTriage | None = None
