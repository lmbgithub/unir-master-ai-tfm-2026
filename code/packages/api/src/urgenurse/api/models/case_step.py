import uuid
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import DateTime, Enum, ForeignKey, Text, func, JSON
from sqlalchemy import UUID as SAUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..enums import CaseStepStatus, CaseStepType
from .base import Base

if TYPE_CHECKING:
    from .attachment import Attachment
    from .case import Case


class CaseStep(Base):
    __tablename__ = "case_steps"

    id: Mapped[UUID] = mapped_column(SAUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    case_id: Mapped[UUID] = mapped_column(
        SAUUID(as_uuid=True), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False
    )
    type: Mapped[CaseStepType] = mapped_column(
        Enum(CaseStepType, name="case_step_type", native_enum=False), nullable=False
    )
    status: Mapped[CaseStepStatus] = mapped_column(
        Enum(CaseStepStatus, name="case_step_status", native_enum=False),
        nullable=False,
        default=CaseStepStatus.created,
    )
    assigned_to: Mapped[str | None] = mapped_column(Text, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    meta: Mapped[str | None] = mapped_column(JSON, nullable=True)

    case: Mapped["Case"] = relationship("Case", back_populates="case_steps")
    attachments: Mapped[list["Attachment"]] = relationship("Attachment", back_populates="case_step")
