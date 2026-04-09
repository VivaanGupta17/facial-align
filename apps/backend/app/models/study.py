"""ImagingStudy ORM model — DICOM imaging studies."""

from __future__ import annotations

import uuid
from datetime import date, datetime

from sqlalchemy import Boolean, Date, Float, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class ImagingStudy(Base):
    __tablename__ = "imaging_studies"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
        default=lambda: str(uuid.uuid4()),
    )
    study_uid: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    accession_number: Mapped[str | None] = mapped_column(String(64), nullable=True)
    patient_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        nullable=False,
    )
    modality: Mapped[str] = mapped_column(String(16), nullable=False)
    acquisition_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    series_count: Mapped[int] = mapped_column(
        Integer, server_default=text("0"), nullable=False
    )
    slice_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    volume_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    slice_thickness_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    pixel_spacing_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    kv_peak: Mapped[float | None] = mapped_column(Float, nullable=True)
    body_part_examined: Mapped[str | None] = mapped_column(String(64), nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    quality_flags: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    is_deidentified: Mapped[bool] = mapped_column(
        Boolean, server_default=text("FALSE"), nullable=False
    )
    ingestion_status: Mapped[str] = mapped_column(
        String(32), server_default=text("'pending'"), nullable=False
    )
    uploaded_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        server_default=text("CURRENT_TIMESTAMP"), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        server_default=text("CURRENT_TIMESTAMP"), nullable=False
    )

    # Relationships
    patient = relationship("Patient", back_populates="imaging_studies")
    surgical_cases = relationship("SurgicalCase", back_populates="study")
