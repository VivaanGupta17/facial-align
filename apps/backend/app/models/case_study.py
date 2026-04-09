"""CaseStudy ORM model — junction table linking surgical cases to imaging studies."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import ForeignKey, Integer, String, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class CaseStudy(Base):
    __tablename__ = "case_studies"
    __table_args__ = (
        UniqueConstraint("case_id", "study_id", name="uq_case_study"),
    )

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
        default=lambda: str(uuid.uuid4()),
    )
    case_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("surgical_cases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    study_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("imaging_studies.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    study_role: Mapped[str] = mapped_column(
        String(32), server_default=text("'pre_op'"), nullable=False
    )
    study_label: Mapped[str | None] = mapped_column(String(128), nullable=True)
    is_primary: Mapped[bool] = mapped_column(
        server_default=text("FALSE"), nullable=False
    )
    display_order: Mapped[int] = mapped_column(
        Integer, server_default=text("0"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        server_default=text("CURRENT_TIMESTAMP"), nullable=False
    )

    # Relationships
    case = relationship("SurgicalCase", back_populates="case_studies")
    study = relationship("ImagingStudy", back_populates="case_studies")
