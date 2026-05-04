"""Persistent plan review model for surgeon approval workflow."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, Text, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class PlanReview(Base):
    __tablename__ = "plan_reviews"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
        default=lambda: str(uuid.uuid4()),
    )
    case_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False)
    plan_id: Mapped[Optional[str]] = mapped_column(UUID(as_uuid=False), nullable=True)
    reviewer_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    reviewer_name: Mapped[str] = mapped_column(
        String(128), server_default=text("''"), nullable=False
    )
    decision: Mapped[str] = mapped_column(
        String(32), server_default=text("'pending'"), nullable=False
    )
    notes: Mapped[str] = mapped_column(Text, server_default=text("''"), nullable=False)
    checklist: Mapped[list] = mapped_column(
        JSONB, server_default=text("'[]'::jsonb"), nullable=False
    )
    signature: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    signed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        server_default=text("CURRENT_TIMESTAMP"), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        server_default=text("CURRENT_TIMESTAMP"), nullable=False
    )
