"""Shared authorization helpers for institution-scoped access control."""

from __future__ import annotations

from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.scope import normalize_institution_code
from app.core.security import CurrentUser
from app.models.case import SurgicalCase
from app.models.patient import Patient
from app.models.review import PlanReview
from app.models.segmentation import SegmentationResult
from app.models.study import ImagingStudy


def resolve_requested_institution_code(
    current_user: CurrentUser,
    requested_institution_code: Optional[str],
) -> Optional[str]:
    normalized = normalize_institution_code(requested_institution_code)
    if current_user.is_admin:
        return normalized

    if current_user.institution_code and normalized and normalized != current_user.institution_code:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Institution-scoped access prevents writing to another institution",
        )
    return normalized or current_user.institution_code


def _case_membership(current_user: CurrentUser, case: SurgicalCase) -> bool:
    participant_ids = {
        str(value)
        for value in (
            getattr(case, "surgeon_id", None),
            getattr(case, "created_by", None),
            getattr(case, "reviewer_id", None),
        )
        if value
    }
    team_ids = {str(value) for value in (getattr(case, "team_ids", None) or []) if value}
    return current_user.user_id in participant_ids or current_user.user_id in team_ids


async def resolve_case_institution_code(case: SurgicalCase, db: AsyncSession) -> Optional[str]:
    direct_value = normalize_institution_code(getattr(case, "institution_code", None))
    if direct_value:
        return direct_value

    patient_id = getattr(case, "patient_id", None)
    if not patient_id:
        return None

    patient = (
        await db.execute(select(Patient).where(Patient.id == patient_id))
    ).scalar_one_or_none()
    if not patient:
        return None
    return normalize_institution_code(getattr(patient, "institution_code", None))


async def resolve_study_institution_code(study: ImagingStudy, db: AsyncSession) -> Optional[str]:
    direct_value = normalize_institution_code(getattr(study, "institution_code", None))
    if direct_value:
        return direct_value

    patient_id = getattr(study, "patient_id", None)
    if not patient_id:
        return None

    patient = (
        await db.execute(select(Patient).where(Patient.id == patient_id))
    ).scalar_one_or_none()
    if not patient:
        return None
    return normalize_institution_code(getattr(patient, "institution_code", None))


async def ensure_case_read_access(
    case: SurgicalCase,
    current_user: CurrentUser,
    db: AsyncSession,
) -> None:
    if current_user.is_admin:
        return

    institution_code = await resolve_case_institution_code(case, db)
    if current_user.can_access_institution(institution_code) or _case_membership(current_user, case):
        return

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="You do not have access to this case",
    )


async def ensure_case_write_access(
    case: SurgicalCase,
    current_user: CurrentUser,
    db: AsyncSession,
) -> None:
    current_user.require_role("surgeon", "admin")
    if current_user.is_admin:
        return

    institution_code = await resolve_case_institution_code(case, db)
    if current_user.can_access_institution(institution_code) or _case_membership(current_user, case):
        return

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="You do not have write access to this case",
    )


async def ensure_study_read_access(
    study: ImagingStudy,
    current_user: CurrentUser,
    db: AsyncSession,
) -> None:
    if current_user.is_admin:
        return

    institution_code = await resolve_study_institution_code(study, db)
    if current_user.can_access_institution(institution_code):
        return

    uploaded_by = getattr(study, "uploaded_by", None)
    if uploaded_by and str(uploaded_by) == current_user.user_id:
        return

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="You do not have access to this imaging study",
    )


async def ensure_review_access(
    review: PlanReview,
    current_user: CurrentUser,
    db: AsyncSession,
) -> SurgicalCase:
    case = (
        await db.execute(select(SurgicalCase).where(SurgicalCase.id == review.case_id))
    ).scalar_one_or_none()
    if not case:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review case not found")

    await ensure_case_read_access(case, current_user, db)
    return case


async def ensure_segmentation_read_access(
    segmentation: SegmentationResult,
    current_user: CurrentUser,
    db: AsyncSession,
) -> SurgicalCase:
    case = (
        await db.execute(select(SurgicalCase).where(SurgicalCase.id == segmentation.case_id))
    ).scalar_one_or_none()
    if not case:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Segmentation case not found")

    await ensure_case_read_access(case, current_user, db)
    return case
