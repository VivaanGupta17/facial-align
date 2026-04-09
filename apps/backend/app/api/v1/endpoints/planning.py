"""
Reduction planning endpoints: create plans, apply surgeon edits, validate constraints,
and export reduction plans as STL files.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.exceptions import CaseNotFoundError, PlanNotFoundError, SegmentationNotFoundError
from app.core.logging import get_logger
from app.core.security import CurrentUser, audit_logger, get_current_user, require_surgeon
from app.db.database import get_db_session
from app.models.case import SurgicalCase
from app.models.plan import ReductionPlan as ReductionPlanModel
from app.models.segmentation import SegmentationResult
from app.schemas.common import JobStatus
from app.schemas.export import ExportFileInfo, ExportRequest, ExportResponse
from app.schemas.plan import (
    FragmentInfo,
    FragmentTransform,
    MetricOverrideRequest,
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
    "/{plan_id}/metric-override",
    response_model=ReductionPlanResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Override an occlusal metric target and re-optimize",
)
async def override_occlusal_metric(
    plan_id: uuid.UUID,
    payload: MetricOverrideRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(require_surgeon),
) -> ReductionPlanResponse:
    """
    Override a specific occlusal metric target, creating a new plan version.

    The constraint optimizer re-runs with the overridden target to find
    fragment transforms that achieve the desired metric value.
    """
    import copy
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

    # Map metric name to constraint field
    metric_to_constraint = {
        "overjet_mm": "target_overjet_mm",
        "overbite_pct": "target_overbite_mm",
        "midline_deviation_mm": "midline_tolerance_mm",
        "occlusal_cant_deg": "cant_tolerance_degrees",
    }

    # Build updated constraints with override
    new_constraints = copy.deepcopy(source_plan.dental_constraints or {})
    constraint_field = metric_to_constraint.get(payload.metric_name)
    if constraint_field:
        new_constraints[constraint_field] = payload.target_value

    # Track the override in constraints metadata
    overrides = new_constraints.get("metric_overrides", [])
    overrides.append({
        "metric_name": payload.metric_name,
        "target_value": payload.target_value,
        "notes": payload.notes,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": current_user.user_id,
    })
    new_constraints["metric_overrides"] = overrides

    new_plan = ReductionPlanModel(
        case_id=source_plan.case_id,
        plan_version=next_version,
        model_name=source_plan.model_name,
        model_version=source_plan.model_version,
        status="refinement_pending",
        fragments=source_plan.fragments,
        transformations=copy.deepcopy(source_plan.transformations),
        dental_constraints=new_constraints,
        skeletal_constraints=source_plan.skeletal_constraints,
        parent_plan_id=source_plan.id,
        surgeon_notes=payload.notes,
        surgeon_edits=list(source_plan.surgeon_edits or []),
        is_ml_generated=False,
    )
    db.add(new_plan)
    await db.flush()

    # Trigger re-optimization with overridden constraints
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
            "operation": "metric_override",
            "metric_name": payload.metric_name,
            "target_value": payload.target_value,
            "parent_plan_id": str(plan_id),
        },
    )

    logger.info(
        "metric_override_applied",
        plan_id=str(new_plan.id),
        case_id=str(source_plan.case_id),
        metric=payload.metric_name,
        target=payload.target_value,
        plan_version=next_version,
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


# ---------------------------------------------------------------------------
# STL Export Endpoints
# ---------------------------------------------------------------------------


def _get_export_dir() -> Path:
    """Return the base export directory, ensuring it exists."""
    settings = get_settings()
    export_dir = settings.storage.base_path / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir


@router.post(
    "/{plan_id}/export",
    response_model=ExportResponse,
    summary="Export reduction plan as STL files",
)
async def export_plan_stl(
    plan_id: uuid.UUID,
    export_request: ExportRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(get_current_user),
) -> ExportResponse:
    """
    Generate STL files for a reduction plan.

    Loads the plan's fragment transforms, applies them to the segmentation
    meshes, and exports via STLExporter. Returns download URLs for each file.
    """
    import trimesh
    from app.services.export.stl_exporter import ExportType, STLExporter, STLFormat

    t0 = time.monotonic()

    # Load plan
    plan = (
        await db.execute(
            select(ReductionPlanModel).where(ReductionPlanModel.id == plan_id)
        )
    ).scalar_one_or_none()
    if not plan:
        raise PlanNotFoundError(f"Plan {plan_id} not found")

    # Load associated segmentation
    seg = (
        await db.execute(
            select(SegmentationResult)
            .where(SegmentationResult.case_id == plan.case_id)
            .where(SegmentationResult.status == "complete")
            .order_by(SegmentationResult.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if not seg:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No completed segmentation found for this case. "
            "Run segmentation before exporting.",
        )

    mesh_paths: dict = seg.mesh_storage_paths or {}
    if not mesh_paths:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No meshes found for this segmentation. "
            "The model may not have been trained yet.",
        )

    # Resolve STL format
    stl_fmt = STLFormat.ASCII if export_request.stl_format == "ascii" else STLFormat.BINARY

    export_dir = _get_export_dir()
    exporter = STLExporter(
        output_dir=export_dir,
        model_version=plan.model_version or "unknown",
        stl_format=stl_fmt,
    )

    case_id_str = str(plan.case_id)
    plan_id_str = str(plan.id)
    transforms: dict = plan.transformations or {}

    # Build per-fragment confidence map
    confidence: dict[str, float] = {}
    for fid, tdata in transforms.items():
        confidence[fid] = tdata.get("confidence", 0.0) if isinstance(tdata, dict) else 0.0

    # Load meshes from disk and apply planned transforms
    loaded_meshes: dict[str, trimesh.Trimesh] = {}
    for structure_name, paths_info in mesh_paths.items():
        stl_path = paths_info.get("stl") if isinstance(paths_info, dict) else None
        if not stl_path or not Path(stl_path).exists():
            continue
        try:
            mesh = trimesh.load(stl_path, force="mesh")
        except Exception:
            logger.warning("Failed to load mesh: %s", stl_path)
            continue

        # Apply planned transform if available for this structure
        if structure_name in transforms:
            tdata = transforms[structure_name]
            t_matrix = tdata.get("transform") if isinstance(tdata, dict) else None
            if t_matrix and isinstance(t_matrix, dict):
                rot = t_matrix.get("rotation_matrix")
                trans = t_matrix.get("translation_mm")
                if rot and trans:
                    T = np.eye(4)
                    T[:3, :3] = np.array(rot, dtype=np.float64)
                    T[:3, 3] = np.array(trans, dtype=np.float64)
                    mesh.apply_transform(T)

        loaded_meshes[structure_name] = mesh

    if not loaded_meshes:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No valid meshes could be loaded for export.",
        )

    # Perform the export
    export_files: list[ExportFileInfo] = []

    if export_request.export_type == "full_assembly":
        manifest = exporter.export_full_assembly(
            meshes=loaded_meshes,
            case_id=case_id_str,
            plan_id=plan_id_str,
            transforms=transforms,
            confidence=confidence,
        )
        for exp in manifest.exports:
            export_files.append(ExportFileInfo(
                filename=exp.stl_path.name,
                export_type=exp.metadata.export_type,
                download_url=f"/planning/{plan_id_str}/export/{exp.stl_path.name}",
                vertex_count=exp.metadata.vertex_count,
                face_count=exp.metadata.face_count,
                volume_mm3=exp.metadata.volume_mm3,
                is_watertight=exp.metadata.is_watertight,
                is_printable=exp.printability.is_printable,
            ))

    elif export_request.export_type == "corrected_mandible":
        # Find mandible mesh
        mandible_mesh = loaded_meshes.get("mandible")
        if not mandible_mesh:
            # Try finding any key containing 'mandible'
            for key, mesh in loaded_meshes.items():
                if "mandible" in key.lower():
                    mandible_mesh = mesh
                    break
        if not mandible_mesh:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No mandible mesh found in segmentation.",
            )
        result = exporter.export_corrected_mandible(
            mandible_mesh=mandible_mesh,
            case_id=case_id_str,
            plan_id=plan_id_str,
            transforms=transforms,
            confidence=confidence,
        )
        export_files.append(ExportFileInfo(
            filename=result.stl_path.name,
            export_type=result.metadata.export_type,
            download_url=f"/planning/{plan_id_str}/export/{result.stl_path.name}",
            vertex_count=result.metadata.vertex_count,
            face_count=result.metadata.face_count,
            volume_mm3=result.metadata.volume_mm3,
            is_watertight=result.metadata.is_watertight,
            is_printable=result.printability.is_printable,
        ))

    elif export_request.export_type == "individual_fragment":
        target = export_request.structure_name
        if target and target in loaded_meshes:
            meshes_to_export = {target: loaded_meshes[target]}
        else:
            meshes_to_export = loaded_meshes

        for name, mesh in meshes_to_export.items():
            result = exporter.export_mesh(
                mesh=mesh,
                export_type=ExportType.FULL_ASSEMBLY,
                case_id=case_id_str,
                plan_id=plan_id_str,
                transforms_applied=transforms.get(name, {}),
                confidence_scores={name: confidence.get(name, 0.0)},
                filename_prefix=f"fragment_{name}",
            )
            export_files.append(ExportFileInfo(
                filename=result.stl_path.name,
                export_type=result.metadata.export_type,
                download_url=f"/planning/{plan_id_str}/export/{result.stl_path.name}",
                vertex_count=result.metadata.vertex_count,
                face_count=result.metadata.face_count,
                volume_mm3=result.metadata.volume_mm3,
                is_watertight=result.metadata.is_watertight,
                is_printable=result.printability.is_printable,
            ))
    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown export_type: {export_request.export_type}",
        )

    total_time = time.monotonic() - t0

    audit_logger.log_phi_access(
        user_id=current_user.user_id,
        action="EXPORT",
        resource_type="plan",
        resource_id=plan_id_str,
        ip_address="unknown",
        additional_context={
            "export_type": export_request.export_type,
            "file_count": len(export_files),
        },
    )

    return ExportResponse(
        plan_id=plan_id_str,
        case_id=case_id_str,
        files=export_files,
        total_export_time_seconds=round(total_time, 3),
    )


@router.get(
    "/{plan_id}/export/{filename}",
    summary="Download exported STL file",
    response_class=FileResponse,
)
async def download_export_stl(
    plan_id: uuid.UUID,
    filename: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Stream an exported STL file for download."""
    # Validate plan exists
    plan = (
        await db.execute(
            select(ReductionPlanModel).where(ReductionPlanModel.id == plan_id)
        )
    ).scalar_one_or_none()
    if not plan:
        raise PlanNotFoundError(f"Plan {plan_id} not found")

    # Sanitize filename to prevent path traversal
    safe_filename = Path(filename).name
    if safe_filename != filename or ".." in filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename",
        )

    # Search for the file in the export directory tree
    export_dir = _get_export_dir()
    case_export_dir = export_dir / str(plan.case_id)

    file_path: Optional[Path] = None
    if case_export_dir.exists():
        for candidate in case_export_dir.rglob(safe_filename):
            if candidate.is_file():
                file_path = candidate
                break

    if not file_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Export file not found: {safe_filename}",
        )

    audit_logger.log_phi_access(
        user_id=current_user.user_id,
        action="DOWNLOAD",
        resource_type="export_stl",
        resource_id=str(plan_id),
        ip_address="unknown",
        additional_context={"filename": safe_filename},
    )

    return FileResponse(
        path=str(file_path),
        media_type="model/stl",
        filename=safe_filename,
        headers={
            "Content-Disposition": f'attachment; filename="{safe_filename}"',
            "Cache-Control": "private, max-age=3600",
        },
    )
