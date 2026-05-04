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

from app.core.authorization import ensure_case_read_access, ensure_review_access
from app.core.exceptions import CaseNotFoundError
from app.core.logging import get_logger
from app.core.security import (
    CurrentUser,
    audit_logger,
    get_current_user,
    require_reviewer,
)
from app.db.database import get_db_session
from app.models.case import SurgicalCase
from app.models.plan import ReductionPlan
from app.models.review import PlanReview
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
    id: uuid.UUID
    case_id: uuid.UUID
    plan_id: Optional[uuid.UUID] = None
    reviewer_id: str = ""
    reviewer_name: str = "Current Surgeon"
    decision: str = "pending"
    notes: str = ""
    checklist: list[dict] = Field(default_factory=list)
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


def _review_to_response(review: PlanReview) -> ReviewResponse:
    return ReviewResponse(
        id=review.id,
        case_id=review.case_id,
        plan_id=review.plan_id,
        reviewer_id=review.reviewer_id or "",
        reviewer_name=review.reviewer_name or "Current Surgeon",
        decision=review.decision,
        notes=review.notes,
        checklist=list(review.checklist or []),
        signed_at=review.signed_at.isoformat() if review.signed_at else None,
        signature=review.signature,
        created_at=review.created_at.isoformat(),
        updated_at=review.updated_at.isoformat(),
    )


async def _get_case_or_404(case_id: str, db: AsyncSession) -> SurgicalCase:
    case = (
        await db.execute(select(SurgicalCase).where(SurgicalCase.id == case_id))
    ).scalar_one_or_none()
    if not case:
        raise CaseNotFoundError(f"Case {case_id} not found")
    return case


async def _get_latest_plan_id(case_id: str, db: AsyncSession) -> Optional[str]:
    latest_plan = (
        await db.execute(
            select(ReductionPlan)
            .where(ReductionPlan.case_id == case_id)
            .order_by(ReductionPlan.plan_version.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    return str(latest_plan.id) if latest_plan else None


async def _get_or_create_review(
    case_id: str,
    current_user: CurrentUser,
    db: AsyncSession,
) -> PlanReview:
    review = (
        await db.execute(select(PlanReview).where(PlanReview.case_id == case_id))
    ).scalar_one_or_none()
    latest_plan_id = await _get_latest_plan_id(case_id, db)

    if review is None:
        review = PlanReview(
            case_id=case_id,
            plan_id=latest_plan_id,
            reviewer_id=current_user.user_id,
            reviewer_name=current_user.user_id,
            decision="pending",
            notes="",
            checklist=[dict(item) for item in DEFAULT_CHECKLIST],
        )
        db.add(review)
        await db.flush()
        return review

    review.reviewer_id = current_user.user_id
    review.reviewer_name = review.reviewer_name or current_user.user_id
    if latest_plan_id and review.plan_id is None:
        review.plan_id = latest_plan_id
    review.updated_at = datetime.now(timezone.utc)
    await db.flush()
    return review


async def _get_review_by_id(review_id: str, db: AsyncSession) -> PlanReview:
    review = (
        await db.execute(select(PlanReview).where(PlanReview.id == review_id))
    ).scalar_one_or_none()
    if not review:
        raise HTTPException(status_code=404, detail=f"Review {review_id} not found")
    return review


# ---------- Endpoints ----------


@router.get(
    "/{case_id}",
    response_model=ReviewResponse,
    summary="Get or create review for a case",
)
async def get_review(
    case_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(get_current_user),
) -> ReviewResponse:
    """Return the active persistent review for a case."""
    case = await _get_case_or_404(case_id, db)
    await ensure_case_read_access(case, current_user, db)
    review = await _get_or_create_review(case_id, current_user, db)
    return _review_to_response(review)


@router.patch(
    "/{review_id}/checklist",
    response_model=ReviewResponse,
    summary="Update a checklist item",
)
async def update_checklist(
    review_id: str,
    payload: ChecklistUpdateRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(require_reviewer),
) -> ReviewResponse:
    """Toggle a checklist item's passed status."""
    review = await _get_review_by_id(review_id, db)
    await ensure_review_access(review, current_user, db)
    checklist = [dict(item) for item in review.checklist or []]

    for item in checklist:
        if item["id"] == payload.checklist_id:
            item["passed"] = payload.passed
            item["reviewed_by"] = current_user.user_id
            item["reviewed_at"] = datetime.now(timezone.utc).isoformat()
            break

    review.checklist = checklist
    review.reviewer_id = current_user.user_id
    review.updated_at = datetime.now(timezone.utc)
    await db.flush()
    return _review_to_response(review)


@router.post(
    "/{review_id}/approve",
    response_model=ReviewResponse,
    summary="Approve the surgical plan",
)
async def approve_review(
    review_id: str,
    payload: ReviewActionRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(require_reviewer),
) -> ReviewResponse:
    """Approve the plan with optional notes and digital signature."""
    review = await _get_review_by_id(review_id, db)
    case = await ensure_review_access(review, current_user, db)

    review.decision = "approved"
    review.notes = payload.notes
    review.signed_at = datetime.now(timezone.utc)
    review.reviewer_id = current_user.user_id
    review.reviewer_name = review.reviewer_name or current_user.user_id
    if payload.signature:
        review.signature = payload.signature
    review.updated_at = datetime.now(timezone.utc)

    if review.plan_id:
        plan = (
            await db.execute(select(ReductionPlan).where(ReductionPlan.id == review.plan_id))
        ).scalar_one_or_none()
        if plan:
            plan.surgeon_approved = True
            plan.approved_at = review.signed_at
            plan.approved_by = current_user.user_id
            plan.status = "approved"

    case.reviewer_id = current_user.user_id

    logger.info(
        "review_approved",
        review_id=review_id,
        case_id=str(review.case_id),
        user_id=current_user.user_id,
    )

    audit_logger.log_phi_access(
        user_id=current_user.user_id,
        action="APPROVE",
        resource_type="review",
        resource_id=review_id,
        ip_address="unknown",
    )

    await db.flush()
    return _review_to_response(review)


@router.post(
    "/{review_id}/revision",
    response_model=ReviewResponse,
    summary="Request plan revision",
)
async def request_revision(
    review_id: str,
    payload: ReviewActionRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(require_reviewer),
) -> ReviewResponse:
    """Request revision of the surgical plan."""
    review = await _get_review_by_id(review_id, db)
    await ensure_review_access(review, current_user, db)
    review.decision = "revision_requested"
    review.notes = payload.notes
    review.reviewer_id = current_user.user_id
    review.reviewer_name = review.reviewer_name or current_user.user_id
    review.updated_at = datetime.now(timezone.utc)

    logger.info(
        "revision_requested",
        review_id=review_id,
        case_id=str(review.case_id),
        user_id=current_user.user_id,
    )
    await db.flush()
    return _review_to_response(review)


@router.post(
    "/{review_id}/reject",
    response_model=ReviewResponse,
    summary="Reject the surgical plan",
)
async def reject_review(
    review_id: str,
    payload: ReviewActionRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(require_reviewer),
) -> ReviewResponse:
    """Reject the surgical plan."""
    review = await _get_review_by_id(review_id, db)
    await ensure_review_access(review, current_user, db)
    review.decision = "rejected"
    review.notes = payload.notes
    review.reviewer_id = current_user.user_id
    review.reviewer_name = review.reviewer_name or current_user.user_id
    review.updated_at = datetime.now(timezone.utc)

    logger.info(
        "review_rejected",
        review_id=review_id,
        case_id=str(review.case_id),
        user_id=current_user.user_id,
    )
    await db.flush()
    return _review_to_response(review)
