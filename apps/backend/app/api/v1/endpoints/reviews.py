"""
Surgeon review endpoints: checklist management, approval, revision, rejection.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import CaseNotFoundError
from app.core.logging import get_logger
from app.core.security import CurrentUser, audit_logger, get_current_user, require_surgeon
from app.db.database import get_db_session
from app.models.case import SurgicalCase
from app.schemas.common import BaseSchema

router = APIRouter(prefix="/reviews", tags=["reviews"])
logger = get_logger(__name__)


# ---------- Schemas ----------

DEFAULT_CHECKLIST = [
    {"id": "seg-accuracy", "category": "Segmentation", "label": "Bone segmentation boundaries are accurate", "passed": None, "severity": "required"},
    {"id": "seg-complete", "category": "Segmentation", "label": "All relevant structures are segmented", "passed": None, "severity": "required"},
    {"id": "frag-identified", "category": "Segmentation", "label": "Fracture fragments correctly identified", "passed": None, "severity": "required"},
    {"id": "reduction-sym", "category": "Reduction", "label": "Facial symmetry is restored", "passed": None, "severity": "required"},
    {"id": "reduction-occ", "category": "Reduction", "label": "Occlusion is within acceptable parameters", "passed": None, "severity": "required"},
    {"id": "reduction-condyle", "category": "Reduction", "label": "Condylar seating is adequate", "passed": None, "severity": "required"},
    {"id": "splint-fit", "category": "Splint", "label": "Intermediate splint design is appropriate", "passed": None, "severity": "recommended"},
    {"id": "hardware-plan", "category": "Hardware", "label": "Plate and screw positions are feasible", "passed": None, "severity": "recommended"},
    {"id": "nerve-clearance", "category": "Safety", "label": "Hardware avoids inferior alveolar nerve", "passed": None, "severity": "required"},
    {"id": "airway-clear", "category": "Safety", "label": "Airway dimensions are maintained", "passed": None, "severity": "recommended"},
    {"id": "aesthetics", "category": "Aesthetics", "label": "Soft tissue projection is acceptable", "passed": None, "severity": "optional"},
]


class ReviewResponse(BaseSchema):
    id: str
    case_id: str
    plan_id: str = ""
    reviewer_id: str = ""
    reviewer_name: str = "Current Surgeon"
    decision: str = "pending"
    notes: str = ""
    checklist: list = Field(default_factory=list)
    signed_at: Optional[str] = None
    signature: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""


class ChecklistUpdateRequest(BaseSchema):
    checklist_id: str
    passed: bool


class ReviewActionRequest(BaseSchema):
    notes: str = ""
    signature: Optional[str] = None


# ---------- In-memory review store (replaced by DB table in production) ----------
# Uses the case's JSONB or a dedicated table; for now, keep in-memory for simplicity
_review_store: dict[str, dict] = {}


def _get_or_create_review(case_id: str) -> dict:
    if case_id not in _review_store:
        _review_store[case_id] = {
            "id": f"review-{case_id}",
            "case_id": case_id,
            "plan_id": "",
            "reviewer_id": "",
            "reviewer_name": "Current Surgeon",
            "decision": "pending",
            "notes": "",
            "checklist": [dict(item) for item in DEFAULT_CHECKLIST],
            "signed_at": None,
            "signature": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    return _review_store[case_id]


# ---------- Endpoints ----------


@router.get(
    "/{case_id}",
    response_model=ReviewResponse,
    summary="Get or create review for a case",
)
async def get_review(
    case_id: str,
    current_user: CurrentUser = Depends(get_current_user),
) -> ReviewResponse:
    """Return the active review for a case. Creates a default review with checklist if none exists."""
    review = _get_or_create_review(case_id)
    review["reviewer_id"] = current_user.user_id
    return ReviewResponse(**review)


@router.patch(
    "/{review_id}/checklist",
    response_model=ReviewResponse,
    summary="Update a checklist item",
)
async def update_checklist(
    review_id: str,
    payload: ChecklistUpdateRequest,
    current_user: CurrentUser = Depends(require_surgeon),
) -> ReviewResponse:
    """Toggle a checklist item's passed status."""
    # Find review by ID
    review = None
    for r in _review_store.values():
        if r["id"] == review_id:
            review = r
            break
    if not review:
        raise HTTPException(status_code=404, detail=f"Review {review_id} not found")

    for item in review["checklist"]:
        if item["id"] == payload.checklist_id:
            item["passed"] = payload.passed
            break

    review["updated_at"] = datetime.now(timezone.utc).isoformat()
    return ReviewResponse(**review)


@router.post(
    "/{review_id}/approve",
    response_model=ReviewResponse,
    summary="Approve the surgical plan",
)
async def approve_review(
    review_id: str,
    payload: ReviewActionRequest,
    current_user: CurrentUser = Depends(require_surgeon),
) -> ReviewResponse:
    """Approve the plan with optional notes and digital signature."""
    review = None
    for r in _review_store.values():
        if r["id"] == review_id:
            review = r
            break
    if not review:
        raise HTTPException(status_code=404, detail=f"Review {review_id} not found")

    review["decision"] = "approved"
    review["notes"] = payload.notes
    review["signed_at"] = datetime.now(timezone.utc).isoformat()
    review["reviewer_id"] = current_user.user_id
    if payload.signature:
        review["signature"] = payload.signature
    review["updated_at"] = datetime.now(timezone.utc).isoformat()

    logger.info(
        "review_approved",
        review_id=review_id,
        case_id=review["case_id"],
        user_id=current_user.user_id,
    )

    audit_logger.log_phi_access(
        user_id=current_user.user_id,
        action="APPROVE",
        resource_type="review",
        resource_id=review_id,
        ip_address="unknown",
    )

    return ReviewResponse(**review)


@router.post(
    "/{review_id}/revision",
    response_model=ReviewResponse,
    summary="Request plan revision",
)
async def request_revision(
    review_id: str,
    payload: ReviewActionRequest,
    current_user: CurrentUser = Depends(require_surgeon),
) -> ReviewResponse:
    """Request revision of the surgical plan."""
    review = None
    for r in _review_store.values():
        if r["id"] == review_id:
            review = r
            break
    if not review:
        raise HTTPException(status_code=404, detail=f"Review {review_id} not found")

    review["decision"] = "revision_requested"
    review["notes"] = payload.notes
    review["reviewer_id"] = current_user.user_id
    review["updated_at"] = datetime.now(timezone.utc).isoformat()

    logger.info(
        "revision_requested",
        review_id=review_id,
        case_id=review["case_id"],
        user_id=current_user.user_id,
    )
    return ReviewResponse(**review)


@router.post(
    "/{review_id}/reject",
    response_model=ReviewResponse,
    summary="Reject the surgical plan",
)
async def reject_review(
    review_id: str,
    payload: ReviewActionRequest,
    current_user: CurrentUser = Depends(require_surgeon),
) -> ReviewResponse:
    """Reject the surgical plan."""
    review = None
    for r in _review_store.values():
        if r["id"] == review_id:
            review = r
            break
    if not review:
        raise HTTPException(status_code=404, detail=f"Review {review_id} not found")

    review["decision"] = "rejected"
    review["notes"] = payload.notes
    review["reviewer_id"] = current_user.user_id
    review["updated_at"] = datetime.now(timezone.utc).isoformat()

    logger.info(
        "review_rejected",
        review_id=review_id,
        case_id=review["case_id"],
        user_id=current_user.user_id,
    )
    return ReviewResponse(**review)
