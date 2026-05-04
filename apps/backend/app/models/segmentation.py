"""SegmentationResult ORM model — ML bone segmentation outputs."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Float, ForeignKey, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class SegmentationResult(Base):
    __tablename__ = "segmentation_results"

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
    )
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    model_checkpoint: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    structure_labels: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    structures: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    mask_storage_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    mesh_storage_paths: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    confidence_scores: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    overall_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    provenance: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    inference_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_pipeline_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    gpu_device: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    volume_stats: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    dental_mask_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    dental_mesh_paths: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    fragment_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    fragment_masks_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    fracture_fragments: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    fragment_mesh_paths: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    status: Mapped[str] = mapped_column(
        String(32), server_default=text("'pending'"), nullable=False
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    celery_task_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        server_default=text("CURRENT_TIMESTAMP"), nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    # Relationships
    case = relationship("SurgicalCase", back_populates="segmentation_results")
