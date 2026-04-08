"""Data contract for surgeon manual edit history."""
from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TransformEdit(BaseModel):
    """A single fragment transform edit by the surgeon."""
    fragment_id: str
    original_rotation: List[List[float]]
    original_translation_mm: List[float]
    edited_rotation: List[List[float]]
    edited_translation_mm: List[float]
    delta_translation_mm: Optional[List[float]] = None
    delta_rotation_degrees: Optional[float] = None  # Angular change magnitude


class SurgeonEditHistory(BaseModel):
    """Complete audit trail of surgeon edits to a reduction plan."""
    plan_id: str
    case_id: str
    source_plan_id: Optional[str] = Field(None, description="Plan this was derived from")
    edits: List[TransformEdit]
    surgeon_id: str
    edit_timestamp: datetime
    notes: Optional[str] = None
    re_optimized: bool = Field(False, description="Whether constraint optimization was re-run")
    post_edit_confidence: Optional[float] = None
    post_edit_validation_passed: Optional[bool] = None
    edit_session_duration_seconds: Optional[float] = None
    tool_used: str = Field(
        default="web_viewer",
        description="Tool used for editing: web_viewer, api, desktop_app"
    )
