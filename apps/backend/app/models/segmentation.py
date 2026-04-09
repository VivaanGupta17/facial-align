"""SegmentationResult ORM model — ML bone segmentation outputs."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Float, Integer, String, Text, text
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
    case_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    model_checkpoint: Mapped[str | None] = mapped_column(String(128), nullable=True)
    structure_labels: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    mask_storage_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    mesh_storage_paths: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    confidence_scores: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    overall_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    inference_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_pipeline_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    gpu_device: Mapped[str | None] = mapped_column(String(32), nullable=True)
    volume_stats: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    dental_mask_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    dental_mesh_paths: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    fragment_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fragment_masks_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(32), server_default=text("'pending'"), nullable=False
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    celery_task_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        server_default=text("CURRENT_TIMESTAMP"), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(nullable=True)

    # Relationships
    case = relationship("SurgicalCase", back_populates="segmentation_results")
