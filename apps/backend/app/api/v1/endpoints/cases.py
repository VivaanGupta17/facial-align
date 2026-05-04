"""
Surgical case CRUD endpoints.
Cases are the central entity linking patients, imaging studies, and surgical plans.
"""

from __future__ import annotations

import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.authorization import (
    ensure_case_read_access,
    ensure_case_write_access,
    ensure_study_read_access,
)
from app.core.exceptions import CaseNotFoundError, InvalidStatusTransitionError
from app.core.logging import get_logger
from app.core.security import CurrentUser, audit_logger, get_current_user, require_surgeon
from app.db.database import get_db_session
from app.models.case import CaseStatus, CaseType, SurgicalCase
from app.models.case_study import CaseStudy
from app.models.patient import Patient
from app.models.plan import ReductionPlan
from app.models.segmentation import SegmentationResult
from app.models.study import ImagingStudy
from app.schemas.case import (
    CaseCreate,
    CaseListFilters,
    CaseListItem,
    CaseListResponse,
    CaseResponse,
    CaseStatusTransition,
    CaseStudyCreate,
    CaseStudyInfo,
    CaseStudyUpdate,
    CaseUpdate,
    PlanSummary,
    SegmentationSummary,
)
from app.schemas.common import PaginationParams

router = APIRouter(prefix="/cases", tags=["cases"])
logger = get_logger(__name__)


def _generate_case_number() -> str:
    """Generate a human-readable case number (e.g., FA-2024-0042)."""
    from datetime import datetime, timezone
    year = datetime.now(timezone.utc).year
    short_id = str(uuid.uuid4()).split("-")[0].upper()
    return f"FA-{year}-{short_id}"


async def _build_case_response(
    case: SurgicalCase, db: AsyncSession
) -> CaseResponse:
    """Build a full CaseResponse by enriching with segmentation and plan summaries."""
    def _optional_str(value: object) -> Optional[str]:
        return value if isinstance(value, str) else None

    def _optional_float(value: object) -> Optional[float]:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _optional_int(value: object) -> Optional[int]:
        if isinstance(value, bool):
            return None
        return value if isinstance(value, int) else None

    def _optional_bool(value: object) -> Optional[bool]:
        return value if isinstance(value, bool) else None

    # Fetch latest segmentation
    seg_stmt = (
        select(SegmentationResult)
        .where(SegmentationResult.case_id == case.id)
        .order_by(SegmentationResult.created_at.desc())
        .limit(1)
    )
    seg_row = (await db.execute(seg_stmt)).scalar_one_or_none()
    seg_summary = None
    if seg_row:
        model_name = _optional_str(getattr(seg_row, "model_name", None))
        model_version = _optional_str(getattr(seg_row, "model_version", None))
        if model_name and model_version:
            structure_labels = getattr(seg_row, "structure_labels", None)
            label_count = len(structure_labels) if isinstance(structure_labels, dict) else 0
            seg_summary = SegmentationSummary(
                id=seg_row.id,
                status=seg_row.status,
                model_name=model_name,
                model_version=model_version,
                overall_confidence=_optional_float(
                    getattr(seg_row, "overall_confidence", None)
                ),
                structure_count=label_count,
                created_at=seg_row.created_at,
            )

    # Count segmentations
    seg_count = (
        await db.execute(
            select(func.count()).where(SegmentationResult.case_id == case.id)
        )
    ).scalar_one()

    # Fetch latest plan
    plan_stmt = (
        select(ReductionPlan)
        .where(ReductionPlan.case_id == case.id)
        .order_by(ReductionPlan.plan_version.desc())
        .limit(1)
    )
    plan_row = (await db.execute(plan_stmt)).scalar_one_or_none()
    plan_summary = None
    if plan_row:
        plan_version = _optional_int(getattr(plan_row, "plan_version", None))
        surgeon_approved = _optional_bool(getattr(plan_row, "surgeon_approved", None))
        if plan_version is not None and surgeon_approved is not None:
            plan_summary = PlanSummary(
                id=plan_row.id,
                plan_version=plan_version,
                status=plan_row.status,
                confidence_score=_optional_float(
                    getattr(plan_row, "confidence_score", None)
                ),
                surgeon_approved=surgeon_approved,
                created_at=plan_row.created_at,
            )

    plan_count = (
        await db.execute(
            select(func.count()).where(ReductionPlan.case_id == case.id)
        )
    ).scalar_one()

    # Fetch case-study links with study metadata
    cs_stmt = (
        select(CaseStudy, ImagingStudy)
        .outerjoin(ImagingStudy, CaseStudy.study_id == ImagingStudy.id)
        .where(CaseStudy.case_id == case.id)
        .order_by(CaseStudy.display_order, CaseStudy.created_at)
    )
    cs_rows = (await db.execute(cs_stmt)).all()
    studies_list = [
        CaseStudyInfo(
            id=cs.id,
            study_id=cs.study_id,
            study_role=cs.study_role,
            study_label=cs.study_label,
            is_primary=cs.is_primary,
            display_order=cs.display_order,
            created_at=cs.created_at,
            study_uid=study.study_uid if study else None,
            modality=study.modality if study else None,
            acquisition_date=study.acquisition_date if study else None,
            ingestion_status=study.ingestion_status if study else None,
        )
        for cs, study in cs_rows
    ]

    return CaseResponse(
        id=case.id,
        case_number=case.case_number,
        patient_id=case.patient_id,
        study_id=case.study_id,
        case_type=case.case_type,
        status=case.status,
        surgeon_id=case.surgeon_id,
        reviewer_id=case.reviewer_id,
        fracture_classification=case.fracture_classification,
        planned_procedure=case.planned_procedure,
        diagnosis_codes=case.diagnosis_codes,
        target_surgery_date=case.target_surgery_date,
        team_ids=case.team_ids,
        current_task_id=case.current_task_id,
        last_error=case.last_error,
        created_at=case.created_at,
        updated_at=case.updated_at,
        approved_at=case.approved_at,
        created_by=case.created_by,
        latest_segmentation=seg_summary,
        latest_plan=plan_summary,
        segmentation_count=seg_count,
        plan_count=plan_count,
        studies=studies_list,
    )


@router.post("", response_model=CaseResponse, status_code=status.HTTP_201_CREATED)
async def create_case(
    payload: CaseCreate,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(require_surgeon),
) -> CaseResponse:
    """
    Create a new surgical case.

    Requires surgeon or admin role.
    """
    patient = (
        await db.execute(select(Patient).where(Patient.id == payload.patient_id))
    ).scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")

    study = (
        await db.execute(select(ImagingStudy).where(ImagingStudy.id == payload.study_id))
    ).scalar_one_or_none()
    if not study:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Study not found")
    if str(study.patient_id) != str(payload.patient_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Study does not belong to the selected patient",
        )

    await ensure_study_read_access(study, current_user, db)

    case = SurgicalCase(
        case_number=_generate_case_number(),
        patient_id=payload.patient_id,
        study_id=payload.study_id,
        case_type=payload.case_type,
        status=CaseStatus.CREATED,
        surgeon_id=payload.surgeon_id or current_user.user_id,
        fracture_classification=payload.fracture_classification,
        planned_procedure=payload.planned_procedure,
        diagnosis_codes=payload.diagnosis_codes,
        target_surgery_date=payload.target_surgery_date,
        team_ids=payload.team_ids,
        created_by=current_user.user_id,
    )
    if not getattr(case, "id", None):
        case.id = str(uuid.uuid4())
    if not getattr(case, "created_at", None) or not getattr(case, "updated_at", None):
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        case.created_at = case.created_at or now
        case.updated_at = case.updated_at or now

    db.add(case)
    await db.flush()

    # Ensure the primary study is durably linked in the case-study junction table.
    db.add(
        CaseStudy(
            case_id=str(case.id),
            study_id=str(payload.study_id),
            study_role="pre_op",
            study_label=None,
            is_primary=True,
            display_order=0,
        )
    )
    await db.flush()

    logger.info(
        "case_created",
        case_id=str(case.id),
        case_number=case.case_number,
        case_type=str(case.case_type),
        created_by=current_user.user_id,
    )
    audit_logger.log_phi_access(
        user_id=current_user.user_id,
        action="CREATE",
        resource_type="case",
        resource_id=str(case.id),
        ip_address="unknown",  # Populated by middleware in production
    )

    return await _build_case_response(case, db)


@router.get("", response_model=CaseListResponse)
async def list_cases(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    case_type: Optional[CaseType] = Query(default=None),
    status_filter: Optional[CaseStatus] = Query(default=None, alias="status"),
    surgeon_id: Optional[str] = Query(default=None),
    patient_id: Optional[uuid.UUID] = Query(default=None),
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(get_current_user),
) -> CaseListResponse:
    """
    List surgical cases with optional filters and pagination.
    """
    stmt = select(SurgicalCase)
    if not current_user.is_admin:
        stmt = stmt.join(Patient, Patient.id == SurgicalCase.patient_id)

    if case_type:
        stmt = stmt.where(SurgicalCase.case_type == case_type)
    if status_filter:
        stmt = stmt.where(SurgicalCase.status == status_filter)
    if surgeon_id:
        stmt = stmt.where(SurgicalCase.surgeon_id == surgeon_id)
    if patient_id:
        stmt = stmt.where(SurgicalCase.patient_id == patient_id)

    # Non-admin users only see cases they're part of
    if not current_user.is_admin:
        own_case_filter = (
            (SurgicalCase.surgeon_id == current_user.user_id)
            | (SurgicalCase.created_by == current_user.user_id)
            | (SurgicalCase.reviewer_id == current_user.user_id)
        )
        if current_user.institution_code:
            stmt = stmt.where(
                own_case_filter | (Patient.institution_code == current_user.institution_code)
            )
        else:
            stmt = stmt.where(own_case_filter)

    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = (await db.execute(count_stmt)).scalar_one()

    offset = (page - 1) * page_size
    stmt = stmt.order_by(SurgicalCase.created_at.desc()).offset(offset).limit(page_size)
    rows = (await db.execute(stmt)).scalars().all()

    items = [
        CaseListItem(
            id=c.id,
            case_number=c.case_number,
            patient_id=c.patient_id,
            case_type=c.case_type,
            status=c.status,
            surgeon_id=c.surgeon_id,
            fracture_classification=c.fracture_classification,
            latest_segmentation_status=None,
            latest_plan_confidence=None,
            created_at=c.created_at,
            updated_at=c.updated_at,
        )
        for c in rows
    ]

    return CaseListResponse.create(items=items, total=total, page=page, page_size=page_size)


@router.get("/{case_id}", response_model=CaseResponse)
async def get_case(
    case_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(get_current_user),
) -> CaseResponse:
    """Get a surgical case by ID."""
    case = (
        await db.execute(select(SurgicalCase).where(SurgicalCase.id == case_id))
    ).scalar_one_or_none()

    if not case:
        raise CaseNotFoundError(f"Case {case_id} not found", context={"case_id": str(case_id)})

    await ensure_case_read_access(case, current_user, db)

    audit_logger.log_phi_access(
        user_id=current_user.user_id,
        action="READ",
        resource_type="case",
        resource_id=str(case_id),
        ip_address="unknown",
    )
    return await _build_case_response(case, db)


@router.patch("/{case_id}", response_model=CaseResponse)
async def update_case(
    case_id: uuid.UUID,
    payload: CaseUpdate,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(require_surgeon),
) -> CaseResponse:
    """Update case metadata (partial update)."""
    case = (
        await db.execute(select(SurgicalCase).where(SurgicalCase.id == case_id))
    ).scalar_one_or_none()
    if not case:
        raise CaseNotFoundError(f"Case {case_id} not found")

    await ensure_case_write_access(case, current_user, db)

    changed_fields = []
    for field, value in payload.model_dump(exclude_none=True).items():
        if getattr(case, field) != value:
            setattr(case, field, value)
            changed_fields.append(field)

    if changed_fields:
        audit_logger.log_phi_access(
            user_id=current_user.user_id,
            action="UPDATE",
            resource_type="case",
            resource_id=str(case_id),
            ip_address="unknown",
            additional_context={"fields_changed": changed_fields},
        )

    return await _build_case_response(case, db)


@router.post("/{case_id}/status", response_model=CaseResponse)
async def transition_case_status(
    case_id: uuid.UUID,
    payload: CaseStatusTransition,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(require_surgeon),
) -> CaseResponse:
    """
    Transition a case to a new status.
    Only valid transitions (per the state machine) are accepted.
    """
    case = (
        await db.execute(select(SurgicalCase).where(SurgicalCase.id == case_id))
    ).scalar_one_or_none()
    if not case:
        raise CaseNotFoundError(f"Case {case_id} not found")

    await ensure_case_write_access(case, current_user, db)

    old_status = CaseStatus(case.status)
    requested_status = CaseStatus(payload.new_status)
    try:
        case.transition_to(requested_status)
    except ValueError as exc:
        raise InvalidStatusTransitionError(
            str(exc),
            context={"from": old_status.value, "to": requested_status.value}
        )

    if requested_status == CaseStatus.APPROVED:
        from datetime import datetime, timezone
        case.approved_at = datetime.now(timezone.utc)

    logger.info(
        "case_status_transitioned",
        case_id=str(case_id),
        from_status=old_status.value,
        to_status=requested_status.value,
        user_id=current_user.user_id,
    )

    audit_logger.log_phi_access(
        user_id=current_user.user_id,
        action="UPDATE",
        resource_type="case",
        resource_id=str(case_id),
        ip_address="unknown",
        additional_context={
            "fields_changed": ["status"],
            "operation": "status_transition",
            "from_status": old_status.value,
            "to_status": requested_status.value,
        },
    )

    return await _build_case_response(case, db)


@router.delete("/{case_id}", status_code=status.HTTP_204_NO_CONTENT)
async def archive_case(
    case_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(require_surgeon),
) -> None:
    """Archive a case (soft delete via status transition)."""
    case = (
        await db.execute(select(SurgicalCase).where(SurgicalCase.id == case_id))
    ).scalar_one_or_none()
    if not case:
        raise CaseNotFoundError(f"Case {case_id} not found")

    await ensure_case_write_access(case, current_user, db)

    try:
        case.transition_to(CaseStatus.ARCHIVED)
    except ValueError as exc:
        raise InvalidStatusTransitionError(str(exc))

    audit_logger.log_phi_access(
        user_id=current_user.user_id,
        action="DELETE",
        resource_type="case",
        resource_id=str(case_id),
        ip_address="unknown",
    )


# ---------------------------------------------------------------------------
# Case-Study CRUD (multi-study per patient)
# ---------------------------------------------------------------------------


@router.post("/{case_id}/studies", response_model=CaseStudyInfo, status_code=status.HTTP_201_CREATED)
async def attach_study(
    case_id: uuid.UUID,
    payload: CaseStudyCreate,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(require_surgeon),
) -> CaseStudyInfo:
    """Attach an imaging study to a case."""
    case = (
        await db.execute(select(SurgicalCase).where(SurgicalCase.id == case_id))
    ).scalar_one_or_none()
    if not case:
        raise CaseNotFoundError(f"Case {case_id} not found")

    # Verify study exists
    study = (
        await db.execute(select(ImagingStudy).where(ImagingStudy.id == payload.study_id))
    ).scalar_one_or_none()
    if not study:
        raise HTTPException(status_code=404, detail=f"Study {payload.study_id} not found")

    # Check for duplicate
    existing = (
        await db.execute(
            select(CaseStudy).where(
                CaseStudy.case_id == case_id,
                CaseStudy.study_id == payload.study_id,
            )
        )
    ).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=409, detail="Study already attached to this case")

    # If marking as primary, clear others
    if payload.is_primary:
        existing_primaries = (
            await db.execute(
                select(CaseStudy).where(CaseStudy.case_id == case_id, CaseStudy.is_primary.is_(True))
            )
        ).scalars().all()
        for ep in existing_primaries:
            ep.is_primary = False

    cs = CaseStudy(
        case_id=str(case_id),
        study_id=str(payload.study_id),
        study_role=payload.study_role,
        study_label=payload.study_label,
        is_primary=payload.is_primary,
    )
    db.add(cs)
    await db.flush()

    audit_logger.log_phi_access(
        user_id=current_user.user_id,
        action="CREATE",
        resource_type="case_study",
        resource_id=str(cs.id),
        ip_address="unknown",
        additional_context={"case_id": str(case_id), "study_id": str(payload.study_id)},
    )

    return CaseStudyInfo(
        id=cs.id,
        study_id=cs.study_id,
        study_role=cs.study_role,
        study_label=cs.study_label,
        is_primary=cs.is_primary,
        display_order=cs.display_order,
        created_at=cs.created_at,
        study_uid=study.study_uid,
        modality=study.modality,
        acquisition_date=study.acquisition_date,
        ingestion_status=study.ingestion_status,
    )


@router.get("/{case_id}/studies", response_model=List[CaseStudyInfo])
async def list_case_studies(
    case_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(get_current_user),
) -> List[CaseStudyInfo]:
    """List all studies attached to a case."""
    stmt = (
        select(CaseStudy, ImagingStudy)
        .outerjoin(ImagingStudy, CaseStudy.study_id == ImagingStudy.id)
        .where(CaseStudy.case_id == case_id)
        .order_by(CaseStudy.display_order, CaseStudy.created_at)
    )
    rows = (await db.execute(stmt)).all()
    return [
        CaseStudyInfo(
            id=cs.id,
            study_id=cs.study_id,
            study_role=cs.study_role,
            study_label=cs.study_label,
            is_primary=cs.is_primary,
            display_order=cs.display_order,
            created_at=cs.created_at,
            study_uid=study.study_uid if study else None,
            modality=study.modality if study else None,
            acquisition_date=study.acquisition_date if study else None,
            ingestion_status=study.ingestion_status if study else None,
        )
        for cs, study in rows
    ]


@router.delete("/{case_id}/studies/{study_id}", status_code=status.HTTP_204_NO_CONTENT)
async def detach_study(
    case_id: uuid.UUID,
    study_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(require_surgeon),
) -> None:
    """Detach a study from a case."""
    cs = (
        await db.execute(
            select(CaseStudy).where(
                CaseStudy.case_id == case_id,
                CaseStudy.study_id == study_id,
            )
        )
    ).scalar_one_or_none()
    if not cs:
        raise HTTPException(status_code=404, detail="Study not attached to this case")

    await db.delete(cs)

    audit_logger.log_phi_access(
        user_id=current_user.user_id,
        action="DELETE",
        resource_type="case_study",
        resource_id=str(cs.id),
        ip_address="unknown",
        additional_context={"case_id": str(case_id), "study_id": str(study_id)},
    )


@router.patch("/{case_id}/studies/{study_id}", response_model=CaseStudyInfo)
async def update_case_study(
    case_id: uuid.UUID,
    study_id: uuid.UUID,
    payload: CaseStudyUpdate,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(require_surgeon),
) -> CaseStudyInfo:
    """Update role, label, or primary flag for a case-study link."""
    cs = (
        await db.execute(
            select(CaseStudy).where(
                CaseStudy.case_id == case_id,
                CaseStudy.study_id == study_id,
            )
        )
    ).scalar_one_or_none()
    if not cs:
        raise HTTPException(status_code=404, detail="Study not attached to this case")

    if payload.study_role is not None:
        cs.study_role = payload.study_role
    if payload.study_label is not None:
        cs.study_label = payload.study_label
    if payload.is_primary is not None:
        if payload.is_primary:
            # Clear other primaries
            others = (
                await db.execute(
                    select(CaseStudy).where(
                        CaseStudy.case_id == case_id,
                        CaseStudy.is_primary.is_(True),
                        CaseStudy.id != cs.id,
                    )
                )
            ).scalars().all()
            for o in others:
                o.is_primary = False
        cs.is_primary = payload.is_primary

    await db.flush()

    # Fetch study for metadata
    study = (
        await db.execute(select(ImagingStudy).where(ImagingStudy.id == study_id))
    ).scalar_one_or_none()

    return CaseStudyInfo(
        id=cs.id,
        study_id=cs.study_id,
        study_role=cs.study_role,
        study_label=cs.study_label,
        is_primary=cs.is_primary,
        display_order=cs.display_order,
        created_at=cs.created_at,
        study_uid=study.study_uid if study else None,
        modality=study.modality if study else None,
        acquisition_date=study.acquisition_date if study else None,
        ingestion_status=study.ingestion_status if study else None,
    )
