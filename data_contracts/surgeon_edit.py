"""
Data contract for surgeon manual edit history.

When a surgeon modifies a plan in the web viewer or API, each edit is
captured as a ``TransformEdit`` record.  The complete sequence of edits
forms a ``SurgeonEditHistory`` that provides an immutable audit trail.

Edit types
----------
Each edit is classified by ``EditType``:

- TRANSLATION — pure translation of a fragment (most common)
- ROTATION — pure rotation (e.g. yaw correction of a condyle)
- RIGID — combined rotation + translation (general case)
- OCCLUSION_CORRECTION — edit driven by clinical occlusal feedback
- SYMMETRY_CORRECTION — edit to improve facial symmetry
- UNDO — revert a previous edit (tracked explicitly for auditability)

Confidence impact
-----------------
``confidence_impact`` records how the edit changed the plan's overall
confidence score:
  Positive value → edit improved confidence (surgeon agreed with ML)
  Negative value → edit reduced confidence (surgeon overrode ML)
  Zero           → re-optimisation was not run (manual-only edit)

After a surgeon edit, the constraint optimiser may be re-run to update
all fragment positions downstream.  ``re_optimized`` tracks this.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
MillimeterValue = float
DegreeValue = float


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class EditType(str, Enum):
    """
    Classification of a surgeon edit by its geometric character and clinical intent.
    """
    TRANSLATION = "translation"             # Purely translational (R unchanged)
    ROTATION = "rotation"                   # Purely rotational (t unchanged)
    RIGID = "rigid"                         # General rigid (both R and t changed)
    OCCLUSION_CORRECTION = "occlusion_correction"  # Edit to fix occlusion
    SYMMETRY_CORRECTION = "symmetry_correction"    # Edit to improve symmetry
    CONDYLE_REPOSITIONING = "condyle_repositioning"  # Condyle seating adjustment
    UNDO = "undo"                           # Explicit undo of a prior edit
    RESET_TO_ML = "reset_to_ml"            # Revert fragment to original ML transform


class EditTool(str, Enum):
    """Tool or interface used to apply the edit."""
    WEB_VIEWER = "web_viewer"               # Browser-based 3D viewer
    API = "api"                             # Programmatic API call
    DESKTOP_APP = "desktop_app"             # Native application
    IMPORT = "import"                       # Imported from external file


# ---------------------------------------------------------------------------
# Individual transform edit sub-model
# ---------------------------------------------------------------------------

class TransformEdit(BaseModel):
    """
    A single fragment transform edit made by the surgeon.

    Records the complete before-and-after state so the edit can be
    replayed, reversed, or used to train future models.

    Magnitude fields
    ----------------
    ``delta_translation_mm`` and ``delta_rotation_degrees`` are the
    *change* from original to edited transform.  They are computed from
    ``original_*`` and ``edited_*`` values by the service layer.
    """
    model_config = ConfigDict(populate_by_name=True)

    edit_id: str = Field(..., description="Unique edit record ID")
    fragment_id: str = Field(..., description="Fragment that was edited")
    edit_type: EditType = Field(..., description="Classification of this edit")

    # Before state
    original_rotation: List[List[float]] = Field(
        ..., description="3×3 rotation matrix before edit (row-major)"
    )
    original_translation_mm: List[float] = Field(
        ..., min_length=3, max_length=3,
        description="Translation [x, y, z] in mm before edit"
    )

    # After state
    edited_rotation: List[List[float]] = Field(
        ..., description="3×3 rotation matrix after edit (row-major)"
    )
    edited_translation_mm: List[float] = Field(
        ..., min_length=3, max_length=3,
        description="Translation [x, y, z] in mm after edit"
    )

    # Derived deltas (auto-computed if not provided)
    delta_translation_mm: Optional[List[float]] = Field(
        None, description="Change in translation [Δx, Δy, Δz] in mm"
    )
    delta_rotation_degrees: Optional[float] = Field(
        None, description="Angular magnitude of rotation change in degrees"
    )
    delta_translation_magnitude_mm: Optional[float] = Field(
        None, description="Euclidean magnitude of translation change in mm"
    )

    # Clinical context
    rationale: Optional[str] = Field(
        None, max_length=500,
        description="Surgeon's free-text rationale for this edit"
    )
    confidence_impact: Optional[float] = Field(
        None, ge=-1.0, le=1.0,
        description=(
            "Change in plan confidence after edit "
            "(positive = improved, negative = override of ML prediction)"
        )
    )

    # ------------------------------------------------------------------
    # Auto-compute delta fields
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def compute_deltas(self) -> "TransformEdit":
        """Auto-populate delta fields if not provided."""
        import numpy as _np

        if self.delta_translation_mm is None:
            dt = [
                e - o for e, o in zip(self.edited_translation_mm, self.original_translation_mm)
            ]
            self.delta_translation_mm = dt

        if self.delta_translation_magnitude_mm is None and self.delta_translation_mm:
            mag = math.sqrt(sum(d ** 2 for d in self.delta_translation_mm))
            self.delta_translation_magnitude_mm = round(mag, 4)

        if self.delta_rotation_degrees is None:
            try:
                R_orig = _np.array(self.original_rotation, dtype=float)
                R_edit = _np.array(self.edited_rotation, dtype=float)
                # Relative rotation: R_delta = R_edit @ R_orig^T
                R_delta = R_edit @ R_orig.T
                trace = float(_np.trace(R_delta))
                cos_angle = (trace - 1.0) / 2.0
                cos_angle = max(-1.0, min(1.0, cos_angle))
                self.delta_rotation_degrees = round(math.degrees(math.acos(cos_angle)), 4)
            except Exception:
                pass

        return self

    @field_validator("original_rotation", "edited_rotation")
    @classmethod
    def validate_rotation(cls, v: List[List[float]]) -> List[List[float]]:
        if len(v) != 3 or any(len(row) != 3 for row in v):
            raise ValueError("Rotation matrix must be 3×3")
        return v

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def is_identity_edit(self) -> bool:
        """True if the edit resulted in no geometric change (magnitude < 0.01 mm and < 0.05°)."""
        t_ok = (self.delta_translation_magnitude_mm or 0.0) < 0.01
        r_ok = (self.delta_rotation_degrees or 0.0) < 0.05
        return t_ok and r_ok

    @property
    def is_large_edit(self) -> bool:
        """True if the translation > 5 mm or rotation > 10° — a clinically significant change."""
        t_large = (self.delta_translation_magnitude_mm or 0.0) > 5.0
        r_large = (self.delta_rotation_degrees or 0.0) > 10.0
        return t_large or r_large

    def to_4x4_edited_matrix(self) -> List[List[float]]:
        """Return the edited transform as a 4×4 homogeneous matrix."""
        R = self.edited_rotation
        t = self.edited_translation_mm
        return [
            [R[0][0], R[0][1], R[0][2], t[0]],
            [R[1][0], R[1][1], R[1][2], t[1]],
            [R[2][0], R[2][1], R[2][2], t[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]


# ---------------------------------------------------------------------------
# Edit session summary
# ---------------------------------------------------------------------------

class EditSessionSummary(BaseModel):
    """
    Aggregate statistics for a single edit session.

    Useful for model fine-tuning pipelines that learn from surgeon corrections.
    """
    n_edits: int = Field(..., ge=0, description="Total number of edits in session")
    n_fragments_edited: int = Field(..., ge=0)
    n_large_edits: int = Field(default=0, ge=0, description="Edits classified as large (>5mm or >10°)")
    mean_translation_delta_mm: Optional[float] = Field(None, ge=0)
    mean_rotation_delta_degrees: Optional[float] = Field(None, ge=0)
    max_translation_delta_mm: Optional[float] = Field(None, ge=0)
    max_rotation_delta_degrees: Optional[float] = Field(None, ge=0)
    net_confidence_impact: Optional[float] = Field(
        None, ge=-1.0, le=1.0,
        description="Sum of confidence_impact across all edits in session"
    )


# ---------------------------------------------------------------------------
# Top-level surgeon edit history contract
# ---------------------------------------------------------------------------

class SurgeonEditHistory(BaseModel):
    """
    Complete audit trail of surgeon edits to a reduction plan.

    Immutable once ``finalized=True``.  New edit sessions create new
    ``SurgeonEditHistory`` records linked via ``source_plan_id``.

    Model learning
    --------------
    Edit histories are consumed by the offline fine-tuning pipeline to
    learn surgeon preferences and improve future ML predictions.
    The ``consent_for_training`` flag records whether the surgeon consented
    to their edits being used for model improvement.
    """
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [{
                "history_id": "edit-hist-001",
                "plan_id": "plan-abc123",
                "case_id": "case-xyz456",
                "surgeon_id": "surgeon-789",
                "edit_timestamp": "2024-03-15T15:00:00Z",
                "tool_used": "web_viewer",
                "n_edits": 2,
                "re_optimized": True,
                "post_edit_confidence": 0.91,
            }]
        },
    )

    # Identifiers
    history_id: str = Field(..., description="Unique edit history record ID")
    plan_id: str = Field(..., description="Plan that was edited")
    case_id: str = Field(..., description="Owning surgical case")
    source_plan_id: Optional[str] = Field(
        None, description="ML-generated plan this edit session started from"
    )

    # Surgeon
    surgeon_id: str = Field(..., description="ID of the editing surgeon")
    surgeon_role: Optional[str] = Field(
        None, description="Surgeon's role (e.g. 'attending', 'fellow')"
    )

    # Session metadata
    edit_timestamp: datetime = Field(
        ..., description="UTC timestamp when the edit session was completed"
    )
    tool_used: EditTool = Field(
        default=EditTool.WEB_VIEWER, description="Interface used for editing"
    )
    edit_session_duration_seconds: Optional[float] = Field(
        None, ge=0, description="Duration of the edit session in seconds"
    )

    # Edits
    edits: List[TransformEdit] = Field(
        ..., description="Ordered list of individual fragment edits"
    )

    # Post-edit state
    re_optimized: bool = Field(
        default=False,
        description="Whether constraint optimisation was re-run after editing"
    )
    post_edit_confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Plan confidence after edit (and re-optimisation if applicable)"
    )
    post_edit_validation_passed: Optional[bool] = Field(
        None, description="Whether validation passed on the edited plan"
    )

    # Session summary
    session_summary: Optional[EditSessionSummary] = None

    # Free text
    notes: Optional[str] = Field(None, max_length=2000)

    # Consent
    consent_for_training: bool = Field(
        default=False,
        description="Surgeon consented for edit data to be used for model fine-tuning"
    )

    # Immutability flag
    finalized: bool = Field(
        default=False,
        description="If True, this history record is immutable (plan was approved)"
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("edit_timestamp", mode="before")
    @classmethod
    def ensure_utc(cls, v: Any) -> Any:
        if isinstance(v, datetime) and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    @model_validator(mode="after")
    def auto_compute_session_summary(self) -> "SurgeonEditHistory":
        """Compute session summary if edits are present and summary is not set."""
        if self.edits and self.session_summary is None:
            edited_fragments = {e.fragment_id for e in self.edits}
            large_edits = sum(1 for e in self.edits if e.is_large_edit)
            translations = [
                e.delta_translation_magnitude_mm
                for e in self.edits
                if e.delta_translation_magnitude_mm is not None
            ]
            rotations = [
                e.delta_rotation_degrees
                for e in self.edits
                if e.delta_rotation_degrees is not None
            ]
            confidence_impacts = [
                e.confidence_impact
                for e in self.edits
                if e.confidence_impact is not None
            ]
            self.session_summary = EditSessionSummary(
                n_edits=len(self.edits),
                n_fragments_edited=len(edited_fragments),
                n_large_edits=large_edits,
                mean_translation_delta_mm=(
                    round(sum(translations) / len(translations), 3) if translations else None
                ),
                mean_rotation_delta_degrees=(
                    round(sum(rotations) / len(rotations), 3) if rotations else None
                ),
                max_translation_delta_mm=(
                    round(max(translations), 3) if translations else None
                ),
                max_rotation_delta_degrees=(
                    round(max(rotations), 3) if rotations else None
                ),
                net_confidence_impact=(
                    round(sum(confidence_impacts), 4) if confidence_impacts else None
                ),
            )
        return self

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def n_edits(self) -> int:
        return len(self.edits)

    @property
    def edited_fragment_ids(self) -> List[str]:
        """Unique list of fragment IDs that were edited."""
        seen: Dict[str, bool] = {}
        return [
            fid for fid in (e.fragment_id for e in self.edits)
            if not (fid in seen or seen.update({fid: True}))  # type: ignore[func-returns-value]
        ]

    def get_edits_for_fragment(self, fragment_id: str) -> List[TransformEdit]:
        """Return all edits for a specific fragment in chronological order."""
        return [e for e in self.edits if e.fragment_id == fragment_id]

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SurgeonEditHistory":
        return cls.model_validate(data)
