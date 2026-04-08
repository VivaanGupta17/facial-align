"""
Reduction planning endpoints: create plans, apply surgeon edits, validate constraints.
"""

from __future__ import annotations

import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import CaseNotFoundError, PlanNotFoundError, SegmentationNotFoundError
from app.core.logging import get_logger
from app.core.security import CurrentUser, audit_logger, get_current_user, require_surgeon
from app.db.database import get_db_session
from app.models.case import SurgicalCase
from app.models.plan import ReductionPlan as ReductionPlanModel
from app.models.segmentation import SegmentationResult
from app.schemas.common import JobStatus
from app.schemas.plan import (
    FragmentInfo,
    FragmentTransform,
    OcclusalConstraints,
    OcclusalMetrics,
    ReductionPlanRequest,
    ReductionPlanResponse,
    SurgeonEditRequest,
    ValidationResult,
)

router = APIRouter(prefix="/planning", tags=["planning"])
logger = get_logger(__name__)


def _to_plan_response(row: ReductionPlanModel) -> ReductionPlanResponse:
    """Convert ORM model to response schema."""
    fragments = None
    if row.fragments:
        fragments = {
            fid: FragmentInfo(fragment_id=fid, fragment_label=data.get("label", 0), **{
                k: v for k, v in data.items() if k in FragmentInfo.model_fields
            })
            for fid, data in row.fragments.items()
        }

    transforms = None
    if row.transformations:
        transforms = [
            FragmentTransform(
                fragment_id=fid,
                fragment_label=data.get("fragment_label", 0),
                transform=data["transform"],
                confidence=data.get("confidence", 0.0),
            )
            for fid, data in row.transformations.items()
        ]

    occlusal_constraints = None
    if row.dental_constraints:
        try:
            occlusal_constraints = OcclusalConstraints(**row.dental_constraints)
        except Exception:
            pass

    occlusal_metrics = None
    if row.occlusal_metrics:
        try:
            occlusal_metrics = OcclusalMetrics(**row.occlusal_metrics)
        except Exception:
            pass

    return ReductionPlanResponse(
        id=row.id,
        case_id=row.case_id,
        plan_version=row.plan_version,
        status=row.status,
        model_name=row.model_name,
        model_version=row.model_version,
        fragments=fragments,
        fragment_transforms=transforms,
        occlusal_constraints=occlusal_constraints,
        occlusal_metrics=occlusal_metrics,
        confidence_score=row.confidence_score,
        surgeon_approved=row.surgeon_approved,
        surgeon_notes=row.surgeon_notes,
        parent_plan_id=row.parent_plan_id,
        is_ml_generated=row.is_ml_generated,
        generation_time_ms=row.generation_time_ms,
        created_at=row.created_at,
        approved_at=row.approved_at,
        approved_by=row.approved_by,
    )


@router.post(
    "",
    response_model=JobStatus,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Generate a reduction plan",
)
async def create_reduction_plan(
    payload: ReductionPlanRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(require_surgeon),
) -> JobStatus:
    """
    Submit a fracture reduction planning job.

    The planning pipeline:
    1. Loads fragment meshes from segmentation result
    2. Runs ML reduction model (or ICP baseline)
    3. Applies occlusal and skeletal constraints
    4. Returns a complete reduction plan with transforms and metrics

    Processing is async. Poll the returned job_id for completion.
    """
    from app.workers.tasks import run_reduction_planning_pipeline

    # Validate inputs
    case = (
        await db.execute(select(SurgicalCase).where(SurgicalCase.id == payload.case_id))
    ).scalar_one_or_none()
    if not case:
        raise CaseNotFoundError(f"Case {payload.case_id} not found")

    seg = (
        await db.execute(
            select(SegmentationResult).where(SegmentationResult.id == payload.segmentation_id)
        )
    ).scalar_one_or_none()
    if not seg:
        raise SegmentationNotFoundError(f"Segmentation {payload.segmentation_id} not found")

    if seg.status != "complete":
        from app.core.exceptions import FacialAlignError
        raise FacialAlignError(
            "Segmentation must be complete before planning",
            error_code="SEGMENTATION_NOT_COMPLETE",
        )

    # Get next plan version
    existing_plans = (
        await db.execute(
            select(ReductionPlanModel)
            .where(ReductionPlanModel.case_id == payload.case_id)
            .order_by(ReductionPlanModel.plan_version.desc())
            .limit(1)
        )
    ).scalar_one_or_none()

    next_version = (existing_plans.plan_version + 1) if existing_plans else 1

    # Create placeholder plan record
    plan = ReductionPlanModel(
        case_id=payload.case_id,
        plan_version=next_version,
        model_name=payload.model_name,
        status="generating",
        dental_constraints=payload.occlusal_constraints.model_dump()
        if payload.occlusal_constraints else None,
        is_ml_generated=True,
    )
    db.add(plan)
    await db.flush()

    task = run_reduction_planning_pipeline.delay(
        plan_id=str(plan.id),
        case_id=str(payload.case_id),
        segmentation_id=str(payload.segmentation_id),
        model_name=payload.model_name,
        dental_constraints=payload.occlusal_constraints.model_dump()
        if payload.occlusal_constraints else None,
        use_intact_reference=payload.use_intact_reference,
        user_id=current_user.user_id,
    )

    logger.info(
        "reduction_planning_job_submitted",
        plan_id=str(plan.id),
        case_id=str(payload.case_id),
        model=payload.model_name,
        job_id=task.id,
        plan_version=next_version,
    )

    return JobStatus(
        job_id=task.id,
        status="PENDING",
        current_step="Queued for processing",
        created_at=plan.created_at,
    )


@router.get(
    "/cases/{case_id}",
    response_model=List[ReductionPlanResponse],
    summary="Get all plans for a case",
)
async def get_case_plans(
    case_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(get_current_user),
) -> List[ReductionPlanResponse]:
    """Get all reduction plan versions for a surgical case."""
    rows = (
        await db.execute(
            select(ReductionPlanModel)
            .where(ReductionPlanModel.case_id == case_id)
            .order_by(ReductionPlanModel.plan_version.desc())
        )
    ).scalars().all()
    return [_to_plan_response(r) for r in rows]


@router.get(
    "/{plan_id}",
    response_model=ReductionPlanResponse,
    summary="Get a specific reduction plan",
)
async def get_plan(
    plan_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(get_current_user),
) -> ReductionPlanResponse:
    """Get a reduction plan by ID."""
    row = (
        await db.execute(
            select(ReductionPlanModel).where(ReductionPlanModel.id == plan_id)
        )
    ).scalar_one_or_none()
    if not row:
        raise PlanNotFoundError(f"Plan {plan_id} not found")

    audit_logger.log_phi_access(
        user_id=current_user.user_id,
        action="READ",
        resource_type="plan",
        resource_id=str(plan_id),
        ip_address="unknown",
    )
    return _to_plan_response(row)


@router.post(
    "/{plan_id}/surgeon-edit",
    response_model=ReductionPlanResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Apply surgeon manual adjustment",
)
async def apply_surgeon_edit(
    plan_id: uuid.UUID,
    payload: SurgeonEditRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(require_surgeon),
) -> ReductionPlanResponse:
    """
    Apply a surgeon's manual transform adjustment, creating a new plan version.

    If re_optimize=True, the constraint optimizer is re-run after applying the edit.
    """
    from datetime import datetime, timezone
    from app.workers.tasks import run_reduction_refinement

    source_plan = (
        await db.execute(
            select(ReductionPlanModel).where(ReductionPlanModel.id == plan_id)
        )
    ).scalar_one_or_none()
    if not source_plan:
        raise PlanNotFoundError(f"Plan {plan_id} not found")

    # Get next version number
    next_version_result = (
        await db.execute(
            select(ReductionPlanModel)
            .where(ReductionPlanModel.case_id == source_plan.case_id)
            .order_by(ReductionPlanModel.plan_version.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    next_version = (next_version_result.plan_version + 1) if next_version_result else 1

    # Copy plan and apply edit
    import copy
    new_transforms = copy.deepcopy(source_plan.transformations or {})
    new_transforms[payload.fragment_id] = {
        **(new_transforms.get(payload.fragment_id, {})),
        "transform": payload.new_transform.model_dump(),
        "is_surgeon_edit": True,
        "edited_by": current_user.user_id,
        "edited_at": datetime.now(timezone.utc).isoformat(),
    }

    surgeon_edits = list(source_plan.surgeon_edits or [])
    surgeon_edits.append({
        "fragment_id": payload.fragment_id,
        "original_transform": source_plan.transformations.get(payload.fragment_id, {}).get("transform"),
        "edited_transform": payload.new_transform.model_dump(),
        "notes": payload.notes,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": current_user.user_id,
    })

    new_plan = ReductionPlanModel(
        case_id=source_plan.case_id,
        plan_version=next_version,
        model_name=source_plan.model_name,
        model_version=source_plan.model_version,
        status="refinement_pending" if payload.re_optimize else "draft",
        fragments=source_plan.fragments,
        transformations=new_transforms,
        dental_constraints=source_plan.dental_constraints,
        skeletal_constraints=source_plan.skeletal_constraints,
        parent_plan_id=source_plan.id,
        surgeon_notes=payload.notes,
        surgeon_edits=surgeon_edits,
        is_ml_generated=False,
    )
    db.add(new_plan)
    await db.flush()

    if payload.re_optimize:
        run_reduction_refinement.delay(
            plan_id=str(new_plan.id),
            user_id=current_user.user_id,
        )

    audit_logger.log_phi_access(
        user_id=current_user.user_id,
        action="UPDATE",
        resource_type="plan",
        resource_id=str(new_plan.id),
        ip_address="unknown",
        additional_context={
            "operation": "surgeon_edit",
            "fragment_id": payload.fragment_id,
            "parent_plan_id": str(plan_id),
        },
    )

    return _to_plan_response(new_plan)


@router.post(
    "/{plan_id}/approve",
    response_model=ReductionPlanResponse,
    summary="Surgeon plan approval",
)
async def approve_plan(
    plan_id: uuid.UUID,
    notes: Optional[str] = None,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(require_surgeon),
) -> ReductionPlanResponse:
    """Mark a reduction plan as surgeon-approved."""
    from datetime import datetime, timezone

    row = (
        await db.execute(
            select(ReductionPlanModel).where(ReductionPlanModel.id == plan_id)
        )
    ).scalar_one_or_none()
    if not row:
        raise PlanNotFoundError(f"Plan {plan_id} not found")

    row.surgeon_approved = True
    row.approved_at = datetime.now(timezone.utc)
    row.approved_by = current_user.user_id
    row.status = "approved"
    if notes:
        row.surgeon_notes = notes

    audit_logger.log_phi_access(
        user_id=current_user.user_id,
        action="APPROVE",
        resource_type="plan",
        resource_id=str(plan_id),
        ip_address="unknown",
        additional_context={"operation": "plan_approval"},
    )

    logger.info(
        "plan_approved",
        plan_id=str(plan_id),
        case_id=str(row.case_id),
        plan_version=row.plan_version,
        approved_by=current_user.user_id,
    )

    return _to_plan_response(row)


@router.post(
    "/{plan_id}/validate",
    response_model=ValidationResult,
    summary="Run automated plan validation",
)
async def validate_plan(
    plan_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(get_current_user),
) -> ValidationResult:
    """
    Run automated clinical validation checks on a reduction plan.
    Checks skeletal symmetry, occlusal metrics, and condylar seating.
    """
    row = (
        await db.execute(
            select(ReductionPlanModel).where(ReductionPlanModel.id == plan_id)
        )
    ).scalar_one_or_none()
    if not row:
        raise PlanNotFoundError(f"Plan {plan_id} not found")

    from app.services.reduction.reduction_service import FractureReductionService
    service = FractureReductionService()
    result = await service.validate_plan_from_db_record(row)
    return result
