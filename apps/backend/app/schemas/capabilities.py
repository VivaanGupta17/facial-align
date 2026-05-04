"""Capability registry and provenance response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import Field

from app.schemas.common import BaseSchema


class ProvenanceInfo(BaseSchema):
    """Execution provenance for any algorithmic result."""

    algorithm_used: str = Field(..., description="Algorithm that actually produced the output")
    validation_tier: str = Field(
        ...,
        description="Validation tier: deterministic_baseline, learned_beta, manual_override",
    )
    beta_status: str = Field(
        ...,
        description="Beta visibility state: not_beta, beta_available, beta_unavailable, fallback, manual_override",
    )
    warnings: List[str] = Field(default_factory=list, description="User-visible warnings")
    fallback_reason: Optional[str] = Field(
        None, description="Reason a fallback path was used instead of the requested one"
    )
    model_version: Optional[str] = Field(None, description="Resolved model version if known")


class CapabilityStatus(BaseSchema):
    """A single subsystem capability entry."""

    name: str
    category: str
    status: str = Field(..., description="available, degraded, unavailable")
    baseline_available: bool = False
    learned_available: bool = False
    artifact_required: bool = False
    artifact_ready: bool = False
    validation_tier: str
    beta_status: str
    model_version: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


class CapabilitySnapshotResponse(BaseSchema):
    """Top-level capability registry response."""

    generated_at: datetime
    capabilities: List[CapabilityStatus]
