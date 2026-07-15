import uuid
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import DateTime, Enum, Float, ForeignKey, JSON, Text, func
from sqlalchemy import UUID as SAUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..enums import AttachmentKind, AttachmentStatus
from .base import Base

if TYPE_CHECKING:
    from .case_step import CaseStep


class Attachment(Base):
    __tablename__ = "attachments"

    id: Mapped[UUID] = mapped_column(SAUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    case_step_id: Mapped[UUID] = mapped_column(
        SAUUID(as_uuid=True),
        ForeignKey("case_steps.id", ondelete="CASCADE"),
        nullable=False,
    )
    original_filename: Mapped[str] = mapped_column(Text, nullable=False)
    mime_type: Mapped[str] = mapped_column(Text, nullable=False)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    kind: Mapped[AttachmentKind] = mapped_column(
        Enum(AttachmentKind, name="attachment_kind", native_enum=False), nullable=False
    )
    status: Mapped[AttachmentStatus] = mapped_column(
        Enum(AttachmentStatus, name="attachment_status", native_enum=False),
        nullable=False,
        default=AttachmentStatus.pending,
    )
    transcription: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    ner: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    sbar: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    case_step: Mapped["CaseStep"] = relationship("CaseStep", back_populates="attachments")
