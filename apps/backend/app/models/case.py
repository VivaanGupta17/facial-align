"""SurgicalCase ORM model — central case entity with state-machine transitions."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Optional, Set

from sqlalchemy import DateTime, ForeignKey, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class CaseStatus(str, enum.Enum):
    CREATED = "CREATED"
    DICOM_PROCESSING = "DICOM_PROCESSING"
    SEGMENTED = "SEGMENTED"
    PLANNING = "PLANNING"
    PLANNED = "PLANNED"
    REVIEWED = "REVIEWED"
    APPROVED = "APPROVED"
    ARCHIVED = "ARCHIVED"
    FAILED = "FAILED"


class CaseType(str, enum.Enum):
    TRAUMA = "TRAUMA"
    ORTHOGNATHIC = "ORTHOGNATHIC"
    RECONSTRUCTION = "RECONSTRUCTION"
    TMJ = "TMJ"
    OTHER = "OTHER"


VALID_TRANSITIONS: dict[CaseStatus, set[CaseStatus]] = {
    CaseStatus.CREATED: {CaseStatus.DICOM_PROCESSING, CaseStatus.FAILED, CaseStatus.ARCHIVED},
    CaseStatus.DICOM_PROCESSING: {CaseStatus.SEGMENTED, CaseStatus.FAILED},
    CaseStatus.SEGMENTED: {CaseStatus.PLANNING, CaseStatus.FAILED, CaseStatus.ARCHIVED},
    CaseStatus.PLANNING: {CaseStatus.PLANNED, CaseStatus.FAILED},
    CaseStatus.PLANNED: {CaseStatus.REVIEWED, CaseStatus.PLANNING, CaseStatus.ARCHIVED},
    CaseStatus.REVIEWED: {CaseStatus.APPROVED, CaseStatus.PLANNING, CaseStatus.ARCHIVED},
    CaseStatus.APPROVED: {CaseStatus.ARCHIVED},
    CaseStatus.ARCHIVED: set(),
    CaseStatus.FAILED: {CaseStatus.CREATED, CaseStatus.ARCHIVED},
}


class SurgicalCase(Base):
    __tablename__ = "surgical_cases"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
        default=lambda: str(uuid.uuid4()),
    )
    case_number: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    patient_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("patients.id", ondelete="RESTRICT"),
        nullable=False,
    )
    study_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("imaging_studies.id", ondelete="RESTRICT"),
        nullable=False,
    )
    case_type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), server_default=text("'CREATED'"), nullable=False
    )
    diagnosis_codes: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    fracture_classification: Mapped[Optional[str]] = mapped_column(
        String(128), nullable=True
    )
    clinical_notes_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    planned_procedure: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    target_surgery_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    current_task_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    surgeon_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    reviewer_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    team_ids: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    created_by: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
    )
    approved_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Relationships
    patient = relationship("Patient", back_populates="surgical_cases")
    study = relationship("ImagingStudy", back_populates="surgical_cases")
    segmentation_results = relationship("SegmentationResult", back_populates="case")
    reduction_plans = relationship("ReductionPlan", back_populates="case")
    case_studies = relationship("CaseStudy", back_populates="case", cascade="all, delete-orphan")

    def transition_to(self, new_status: CaseStatus) -> None:
        """Validate and apply a status transition."""
        current = CaseStatus(self.status)
        allowed = VALID_TRANSITIONS.get(current, set())
        if new_status not in allowed:
            raise ValueError(
                f"Cannot transition from {current.value} to {new_status.value}. "
                f"Allowed: {[s.value for s in allowed]}"
            )
        self.status = new_status
