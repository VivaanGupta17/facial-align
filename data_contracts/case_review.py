"""
Data contract for case review and clinical sign-off.

The ``CaseReview`` model captures the complete record of a surgeon's review
of an automated reduction plan.  It serves as the regulatory audit trail for
any plan that proceeds to the operating room.

Review workflow
---------------
1. System generates a plan (``ReductionPlanContract``).
2. Primary reviewer (fellow or attending) reviews the plan in the web viewer
   and populates ``CaseReview`` with approval/rejection and notes.
3. If ``second_reviewer_required`` is True (e.g. complex pan-facial cases,
   institution policy), a second reviewer sign-off is required.
4. Only after all required sign-offs does the plan status advance to APPROVED.

Modification requests
---------------------
``required_modifications`` is a list of structured modification requests.
Each entry links back to a specific fragment or measurement that needs to
be adjusted, enabling the system to present targeted edit tools.

Clinical notes
--------------
``clinical_notes`` is a free-text field for the reviewing surgeon's
clinical assessment that supplements the structured checklist.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ReviewerRole(str, Enum):
    """Clinical role of the reviewer."""
    RESIDENT = "resident"
    FELLOW = "fellow"
    ATTENDING = "attending"
    CHIEF_RESIDENT = "chief_resident"
    ATTENDING_SENIOR = "attending_senior"   # e.g. department chief
    RADIOLOGIST = "radiologist"             # For imaging review only


class ReviewOutcome(str, Enum):
    """Possible outcomes of a case review."""
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFICATIONS_REQUIRED = "modifications_required"
    DEFERRED = "deferred"               # Awaiting additional information


class ModificationPriority(str, Enum):
    """Urgency of a requested modification."""
    CRITICAL = "critical"               # Blocks approval; must be fixed
    IMPORTANT = "important"             # Should be addressed
    MINOR = "minor"                     # Cosmetic / preference


# ---------------------------------------------------------------------------
# Clinical measurement sub-model (inline record, not referencing plan)
# ---------------------------------------------------------------------------

class ClinicalMeasurement(BaseModel):
    """
    A reviewer-confirmed clinical measurement at time of review.

    These may differ from model-predicted values due to manual landmark
    identification on the review interface.
    """
    name: str = Field(..., description="Measurement name (e.g. 'overjet_mm', 'SNA')")
    value: float = Field(..., description="Measured / confirmed value")
    unit: str = Field(..., description="Unit: 'mm' or 'degrees'")
    within_normal_range: bool = Field(
        ..., description="Reviewer assessment of whether value is within normal limits"
    )
    normal_range: Optional[str] = Field(
        None, description="Normal range as human-readable string"
    )
    reviewer_comment: Optional[str] = Field(
        None, max_length=200, description="Reviewer's comment about this measurement"
    )


# ---------------------------------------------------------------------------
# Modification request sub-model
# ---------------------------------------------------------------------------

class ModificationRequest(BaseModel):
    """
    A structured modification request generated during review.

    Links the issue to a specific fragment, measurement, or system component
    to enable targeted re-editing.
    """
    request_id: str = Field(..., description="Unique modification request ID")
    priority: ModificationPriority = Field(
        ..., description="Priority of this modification"
    )
    target_type: str = Field(
        ..., description="What needs to be modified: 'fragment', 'hardware', 'occlusion', 'plan'"
    )
    target_id: Optional[str] = Field(
        None, description="ID of the specific fragment, hardware item, etc."
    )
    description: str = Field(
        ..., max_length=500,
        description="Human-readable description of the required modification"
    )
    suggested_action: Optional[str] = Field(
        None, max_length=500,
        description="Reviewer's suggested corrective action"
    )
    resolved: bool = Field(
        default=False, description="Whether this modification has been addressed"
    )
    resolved_in_plan_version: Optional[int] = Field(
        None, description="Plan version in which this request was resolved"
    )


# ---------------------------------------------------------------------------
# Reviewer feedback sub-model
# ---------------------------------------------------------------------------

class ReviewerFeedback(BaseModel):
    """
    Structured feedback for a single reviewer interaction.

    Used when multiple reviewers interact with a case sequentially.
    """
    reviewer_id: str
    reviewer_role: ReviewerRole
    review_timestamp: datetime
    outcome: ReviewOutcome
    notes: Optional[str] = Field(None, max_length=2000)
    modification_requests: List[ModificationRequest] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level case review contract
# ---------------------------------------------------------------------------

class CaseReview(BaseModel):
    """
    Complete case review and clinical sign-off record.

    Each approved plan must have at least one ``CaseReview`` record with
    ``approved=True``.  Complex cases (``second_reviewer_required=True``)
    require a second reviewer's sign-off before the plan can be executed.

    Approval checklist
    ------------------
    The ``checklist_items`` dict maps clinical validation items to True/False.
    ALL items must be True for ``approved=True`` to be valid.

    Audit trail
    -----------
    The complete audit trail is stored in ``reviewer_history``, one entry
    per review interaction.  The primary reviewer's decision is reflected in
    the top-level ``approved`` / ``outcome`` fields.
    """
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [{
                "review_id": "rev-001",
                "case_id": "case-xyz456",
                "plan_id": "plan-abc123",
                "reviewer_id": "surgeon-789",
                "reviewer_role": "attending",
                "review_timestamp": "2024-03-15T14:30:00Z",
                "approved": True,
                "outcome": "approved",
                "approval_notes": "Plan reviewed and approved. Proceed to OR.",
            }]
        },
    )

    # Identifiers
    review_id: str = Field(..., description="Unique review record ID")
    case_id: str = Field(..., description="Reviewed surgical case ID")
    plan_id: str = Field(..., description="Reviewed reduction plan ID")
    plan_version: Optional[int] = Field(
        None, ge=1, description="Version of the plan reviewed"
    )

    # Primary reviewer
    reviewer_id: str = Field(..., description="Internal user ID of the primary reviewer")
    reviewer_name: Optional[str] = Field(
        None, description="Display name (de-identified in exports)"
    )
    reviewer_role: ReviewerRole = Field(..., description="Clinical role of primary reviewer")

    # Review decision
    review_timestamp: datetime = Field(
        ..., description="UTC timestamp when the review was completed"
    )
    outcome: ReviewOutcome = Field(..., description="Review outcome")
    approved: bool = Field(..., description="True if the plan is approved for surgery")
    approval_notes: Optional[str] = Field(
        None, max_length=2000,
        description="Free-text clinical notes supporting the approval decision"
    )
    rejection_reason: Optional[str] = Field(
        None, max_length=1000,
        description="Required when approved=False; reason for rejection"
    )
    clinical_notes: Optional[str] = Field(
        None, max_length=5000,
        description="Surgeon's full clinical assessment and operative planning notes"
    )

    # Clinical validation checklist
    checklist_items: Dict[str, bool] = Field(
        default_factory=lambda: {
            "occlusion_acceptable": False,
            "symmetry_acceptable": False,
            "condylar_seating_confirmed": False,
            "hardware_plan_reviewed": False,
            "surgical_sequence_confirmed": False,
            "informed_consent_obtained": False,
            "imaging_reviewed": False,
            "plan_matches_clinical_exam": False,
            "pre_injury_records_considered": False,
            "anaesthesia_risk_assessed": False,
        }
    )

    # Measurements confirmed during review
    clinical_measurements: List[ClinicalMeasurement] = Field(
        default_factory=list,
        description="Key measurements confirmed by the reviewer during plan review"
    )

    # Modification requests
    required_modifications: List[str] = Field(
        default_factory=list,
        description="List of brief modification descriptions (legacy simple format)"
    )
    modification_requests: List[ModificationRequest] = Field(
        default_factory=list,
        description="Structured modification requests (preferred over required_modifications)"
    )

    # Multi-reviewer workflow
    second_reviewer_required: bool = Field(
        default=False,
        description="Institution policy or case complexity requires second sign-off"
    )
    second_reviewer_id: Optional[str] = None
    second_review_timestamp: Optional[datetime] = None
    second_review_approved: Optional[bool] = None
    second_review_notes: Optional[str] = Field(None, max_length=1000)

    # Full reviewer history (all touches on this case)
    reviewer_history: List[ReviewerFeedback] = Field(
        default_factory=list,
        description="Complete chronological review history"
    )

    # Timing
    review_duration_minutes: Optional[float] = Field(
        None, ge=0, description="Time spent on this review session in minutes"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("review_timestamp", mode="before")
    @classmethod
    def ensure_timezone(cls, v: Any) -> Any:
        if isinstance(v, datetime) and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    @model_validator(mode="after")
    def validate_rejection_has_reason(self) -> "CaseReview":
        if not self.approved and self.rejection_reason is None:
            raise ValueError(
                "rejection_reason must be provided when approved=False"
            )
        return self

    @model_validator(mode="after")
    def validate_approval_with_open_checklist(self) -> "CaseReview":
        if self.approved:
            incomplete = [
                k for k, v in self.checklist_items.items() if not v
            ]
            if incomplete:
                raise ValueError(
                    f"Cannot approve: the following checklist items are incomplete: {incomplete}"
                )
        return self

    @model_validator(mode="after")
    def validate_second_review_consistency(self) -> "CaseReview":
        if self.second_reviewer_required and self.second_review_approved is None:
            # Second review is pending — valid state
            pass
        if self.second_review_approved is not None and self.second_reviewer_id is None:
            raise ValueError(
                "second_reviewer_id must be set when second_review_approved is not None"
            )
        return self

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def is_fully_approved(self) -> bool:
        """
        Return True if all required approvals have been obtained.

        If ``second_reviewer_required`` is False, only the primary approval is needed.
        If True, both primary and secondary approvals must be True.
        """
        if not self.approved:
            return False
        if self.second_reviewer_required:
            return bool(self.second_review_approved)
        return True

    @property
    def checklist_completion_rate(self) -> float:
        """Fraction of checklist items completed [0.0–1.0]."""
        if not self.checklist_items:
            return 0.0
        return sum(self.checklist_items.values()) / len(self.checklist_items)

    @property
    def has_critical_modifications(self) -> bool:
        """True if any modification request has CRITICAL priority."""
        return any(
            m.priority == ModificationPriority.CRITICAL and not m.resolved
            for m in self.modification_requests
        )

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CaseReview":
        return cls.model_validate(data)
