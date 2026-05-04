"""Patient ORM model — de-identified patient records."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
        default=lambda: str(uuid.uuid4()),
    )
    mrn_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    demographics_encrypted: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    institution_code: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    age_at_registration: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    sex: Mapped[Optional[str]] = mapped_column(String(1), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        server_default=text("CURRENT_TIMESTAMP"), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        server_default=text("CURRENT_TIMESTAMP"), nullable=False
    )
    created_by: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    is_active: Mapped[bool] = mapped_column(
        Boolean, server_default=text("TRUE"), nullable=False
    )

    # Relationships
    imaging_studies = relationship("ImagingStudy", back_populates="patient")
    surgical_cases = relationship("SurgicalCase", back_populates="patient")
