"""ReductionPlan ORM model — surgical fracture reduction plans."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Float, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class ReductionPlan(Base):
    __tablename__ = "reduction_plans"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
        default=lambda: str(uuid.uuid4()),
    )
    case_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False)
    plan_version: Mapped[int] = mapped_column(
        Integer, server_default=text("1"), nullable=False
    )
    model_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    fragments: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    transformations: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    dental_constraints: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    skeletal_constraints: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    occlusal_metrics: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    symmetry_metrics: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    validation_passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    validation_warnings: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    surgeon_approved: Mapped[bool] = mapped_column(
        Boolean, server_default=text("FALSE"), nullable=False
    )
    surgeon_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    surgeon_edits: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    parent_plan_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), nullable=True
    )
    is_ml_generated: Mapped[bool] = mapped_column(
        Boolean, server_default=text("TRUE"), nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(32), server_default=text("'draft'"), nullable=False
    )
    generation_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        server_default=text("CURRENT_TIMESTAMP"), nullable=False
    )
    approved_at: Mapped[datetime | None] = mapped_column(nullable=True)
    approved_by: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Relationships
    case = relationship("SurgicalCase", back_populates="reduction_plans")
