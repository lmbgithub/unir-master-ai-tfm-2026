import uuid
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import DateTime, Enum, JSON, SmallInteger, Text, func
from sqlalchemy import UUID as SAUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..enums import CasePhase
from .base import Base

if TYPE_CHECKING:
    from .case_step import CaseStep


class Case(Base):
    __tablename__ = "cases"

    id: Mapped[UUID] = mapped_column(SAUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_info: Mapped[dict] = mapped_column(JSON, nullable=False)
    chief_complaint: Mapped[str] = mapped_column(Text, nullable=False)
    esi_level: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    phase: Mapped[CasePhase] = mapped_column(Enum(CasePhase, name="case_phase", native_enum=False), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    case_steps: Mapped[list["CaseStep"]] = relationship("CaseStep", back_populates="case")
