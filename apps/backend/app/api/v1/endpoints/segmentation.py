"""
Segmentation endpoints: trigger jobs, poll results, access meshes and confidence maps.
"""

from __future__ import annotations

import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import CaseNotFoundError, SegmentationNotFoundError
from app.core.logging import get_logger
from app.core.security import CurrentUser, get_current_user, require_surgeon
from app.db.database import get_db_session
from app.models.case import SurgicalCase
from app.models.segmentation import SegmentationResult as SegmentationModel
from app.services.capabilities import build_provenance, get_capability_snapshot
from app.schemas.common import JobStatus
from app.schemas.segmentation import (
    ConfidenceMap,
    MeshInfo,
    SegmentationJobResponse,
    SegmentationRequest,
    SegmentationResult,
    StructureLabel,
)

router = APIRouter(prefix="/segmentation", tags=["segmentation"])
logger = get_logger(__name__)


def _resolve_segmentation_request(
    model_name: str,
    run_dental_segmentation: bool,
) -> tuple[str, bool, dict]:
    """Resolve the requested segmentation settings to an honest executable path."""

    warnings: list[str] = []
    fallback_reason: Optional[str] = None
    snapshot = get_capability_snapshot()
    capability_by_name = {cap.name: cap for cap in snapshot.capabilities}

    requested_model = model_name
    resolved_model = model_name
    validation_tier = "deterministic_baseline"
    beta_status = "not_beta"

    if requested_model == "cmf_custom_v1":
        cmf_cap = capability_by_name.get("segmentation_learned_cmf")
        if cmf_cap and not cmf_cap.learned_available:
            resolved_model = "totalsegmentator"
            validation_tier = "deterministic_baseline"
            beta_status = "fallback"
            fallback_reason = "Requested CMF learned model is unavailable; using TotalSegmentator baseline."
            warnings.append(fallback_reason)
        else:
            validation_tier = "learned_beta"
            beta_status = "beta_available"

    resolved_run_dental = run_dental_segmentation
    if run_dental_segmentation:
        dental_cap = capability_by_name.get("dental_segmentation")
        if dental_cap and not dental_cap.learned_available:
            resolved_run_dental = False
            warning = "Dental segmentation requested but artifacts are unavailable; disabling dental beta path."
            warnings.append(warning)
            fallback_reason = fallback_reason or warning

    provenance = build_provenance(
        algorithm_used=resolved_model,
        validation_tier=validation_tier,
        beta_status=beta_status,
        warnings=warnings,
        fallback_reason=fallback_reason,
    )
    return resolved_model, resolved_run_dental, provenance


@router.post(
    "",
    response_model=SegmentationJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger bone segmentation",
)
async def trigger_segmentation(
    payload: SegmentationRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(require_surgeon),
) -> SegmentationJobResponse:
    """
    Submit a segmentation job for a surgical case.

    The segmentation pipeline runs asynchronously via Celery.
    Returns a job ID to poll for results.
    """
    from app.workers.tasks import run_segmentation_pipeline

    # Validate case exists
    case = (
        await db.execute(select(SurgicalCase).where(SurgicalCase.id == payload.case_id))
    ).scalar_one_or_none()
    if not case:
        raise CaseNotFoundError(f"Case {payload.case_id} not found")

    resolved_model, resolved_run_dental, provenance = _resolve_segmentation_request(
        payload.model_name,
        payload.run_dental_segmentation,
    )

    # Create pending segmentation record
    seg_record = SegmentationModel(
        case_id=payload.case_id,
        model_name=resolved_model,
        model_version="pending",
        status="pending",
        provenance=provenance,
    )
    db.add(seg_record)
    await db.flush()

    # Dispatch Celery task
    task = run_segmentation_pipeline.delay(
        segmentation_id=str(seg_record.id),
        case_id=str(payload.case_id),
        study_id=str(case.study_id),
        model_name=resolved_model,
        structures=payload.structures,
        run_dental=resolved_run_dental,
        identify_fragments=payload.identify_fragments,
        fast_mode=payload.fast_mode,
        gpu_device=payload.gpu_device,
        user_id=current_user.user_id,
    )

    seg_record.celery_task_id = task.id
    seg_record.status = "queued"

    logger.info(
        "segmentation_job_submitted",
        case_id=str(payload.case_id),
        segmentation_id=str(seg_record.id),
        model=resolved_model,
        job_id=task.id,
    )

    # Estimate duration based on model
    estimated_duration = 600 if resolved_model == "totalsegmentator" else 1200
    if payload.fast_mode:
        estimated_duration = estimated_duration // 3

    return SegmentationJobResponse(
        job_id=task.id,
        segmentation_id=seg_record.id,
        case_id=payload.case_id,
        status="queued",
        estimated_duration_seconds=estimated_duration,
    )


@router.get(
    "/cases/{case_id}",
    response_model=List[SegmentationResult],
    summary="Get all segmentation results for a case",
)
async def get_case_segmentations(
    case_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(get_current_user),
) -> List[SegmentationResult]:
    """Get all segmentation results for a surgical case."""
    stmt = (
        select(SegmentationModel)
        .where(SegmentationModel.case_id == case_id)
        .order_by(SegmentationModel.created_at.desc())
    )
    rows = (await db.execute(stmt)).scalars().all()
    return [_to_schema(r) for r in rows]


@router.get(
    "/{segmentation_id}",
    response_model=SegmentationResult,
    summary="Get a segmentation result",
)
async def get_segmentation(
    segmentation_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(get_current_user),
) -> SegmentationResult:
    """Get a segmentation result by ID."""
    row = (
        await db.execute(
            select(SegmentationModel).where(SegmentationModel.id == segmentation_id)
        )
    ).scalar_one_or_none()
    if not row:
        raise SegmentationNotFoundError(f"Segmentation {segmentation_id} not found")
    return _to_schema(row)


@router.get(
    "/{segmentation_id}/job-status",
    response_model=JobStatus,
    summary="Get async job status",
)
async def get_job_status(
    segmentation_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(get_current_user),
) -> JobStatus:
    """Poll the Celery task status for an in-progress segmentation."""
    from app.workers.celery_app import celery_app

    row = (
        await db.execute(
            select(SegmentationModel).where(SegmentationModel.id == segmentation_id)
        )
    ).scalar_one_or_none()
    if not row:
        raise SegmentationNotFoundError(f"Segmentation {segmentation_id} not found")

    if not row.celery_task_id:
        return JobStatus(
            job_id="",
            status=row.status.upper(),
            created_at=row.created_at,
        )

    task_result = celery_app.AsyncResult(row.celery_task_id)
    info = task_result.info or {}

    return JobStatus(
        job_id=row.celery_task_id,
        status=task_result.status,
        progress_percent=info.get("progress") if isinstance(info, dict) else None,
        current_step=info.get("step") if isinstance(info, dict) else None,
        result=None if task_result.status != "SUCCESS" else info,
        error=str(info) if task_result.status == "FAILURE" else None,
        created_at=row.created_at,
        completed_at=row.completed_at,
    )


@router.get(
    "/{segmentation_id}/meshes",
    response_model=List[MeshInfo],
    summary="List mesh files for a segmentation",
)
async def list_meshes(
    segmentation_id: uuid.UUID,
    format: Optional[str] = Query(default=None, description="Filter by format: glb, stl, obj"),
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(get_current_user),
) -> List[MeshInfo]:
    """List all available mesh files for a segmentation result."""
    row = (
        await db.execute(
            select(SegmentationModel).where(SegmentationModel.id == segmentation_id)
        )
    ).scalar_one_or_none()
    if not row:
        raise SegmentationNotFoundError(f"Segmentation {segmentation_id} not found")

    mesh_paths: dict = row.mesh_storage_paths or {}
    meshes = []
    for structure, paths in mesh_paths.items():
        if isinstance(paths, dict):
            for fmt, path in paths.items():
                if format is None or fmt == format:
                    meshes.append(MeshInfo(
                        structure_name=structure,
                        format=fmt,
                        path=path,
                    ))

    return meshes


@router.get(
    "/{segmentation_id}/confidence",
    response_model=List[ConfidenceMap],
    summary="Get per-structure confidence maps",
)
async def get_confidence_maps(
    segmentation_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(get_current_user),
) -> List[ConfidenceMap]:
    """Get confidence scores per anatomical structure."""
    row = (
        await db.execute(
            select(SegmentationModel).where(SegmentationModel.id == segmentation_id)
        )
    ).scalar_one_or_none()
    if not row:
        raise SegmentationNotFoundError(f"Segmentation {segmentation_id} not found")

    confidence_scores: dict = row.confidence_scores or {}
    volume_stats: dict = row.volume_stats or {}

    maps = []
    for structure, score in confidence_scores.items():
        stats = volume_stats.get(structure, {})
        maps.append(ConfidenceMap(
            structure_name=structure,
            mean_confidence=score,
            volume_cc=stats.get("volume_cc"),
        ))

    return maps


@router.post(
    "/{segmentation_id}/structures/{label}/approve",
    status_code=status.HTTP_200_OK,
    summary="Approve a segmented structure",
)
async def approve_structure(
    segmentation_id: uuid.UUID,
    label: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(require_surgeon),
) -> dict:
    """Mark a segmented structure as clinician-approved."""
    row = (
        await db.execute(
            select(SegmentationModel).where(SegmentationModel.id == segmentation_id)
        )
    ).scalar_one_or_none()
    if not row:
        raise SegmentationNotFoundError(f"Segmentation {segmentation_id} not found")

    structures: dict = row.structures or {}
    if label not in structures:
        structures[label] = {}
    structures[label]["status"] = "accepted"
    structures[label]["reviewed_by"] = current_user.user_id
    row.structures = structures
    # Force SQLAlchemy to detect the JSONB mutation
    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(row, "structures")

    logger.info(
        "structure_approved",
        segmentation_id=str(segmentation_id),
        label=label,
        user_id=current_user.user_id,
    )
    return {"status": "accepted", "label": label}


@router.post(
    "/{segmentation_id}/structures/{label}/reject",
    status_code=status.HTTP_200_OK,
    summary="Reject a segmented structure",
)
async def reject_structure(
    segmentation_id: uuid.UUID,
    label: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(require_surgeon),
) -> dict:
    """Mark a segmented structure as rejected by clinician."""
    row = (
        await db.execute(
            select(SegmentationModel).where(SegmentationModel.id == segmentation_id)
        )
    ).scalar_one_or_none()
    if not row:
        raise SegmentationNotFoundError(f"Segmentation {segmentation_id} not found")

    structures: dict = row.structures or {}
    if label not in structures:
        structures[label] = {}
    structures[label]["status"] = "rejected"
    structures[label]["reviewed_by"] = current_user.user_id
    row.structures = structures
    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(row, "structures")

    logger.info(
        "structure_rejected",
        segmentation_id=str(segmentation_id),
        label=label,
        user_id=current_user.user_id,
    )
    return {"status": "rejected", "label": label}


@router.post(
    "/{segmentation_id}/structures/{label}/resegment",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Request re-segmentation of a structure",
)
async def request_resegmentation(
    segmentation_id: uuid.UUID,
    label: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(require_surgeon),
) -> dict:
    """Request re-segmentation of a specific structure via Celery."""
    from app.workers.tasks import run_segmentation_pipeline

    row = (
        await db.execute(
            select(SegmentationModel).where(SegmentationModel.id == segmentation_id)
        )
    ).scalar_one_or_none()
    if not row:
        raise SegmentationNotFoundError(f"Segmentation {segmentation_id} not found")

    case = (
        await db.execute(select(SurgicalCase).where(SurgicalCase.id == row.case_id))
    ).scalar_one_or_none()
    if not case:
        raise CaseNotFoundError(f"Case {row.case_id} not found")

    task = run_segmentation_pipeline.delay(
        segmentation_id=str(segmentation_id),
        case_id=str(row.case_id),
        study_id=str(case.study_id),
        model_name=row.model_name,
        structures=[label],
        run_dental=False,
        identify_fragments=False,
        fast_mode=True,
        gpu_device=None,
        user_id=current_user.user_id,
    )

    structures: dict = row.structures or {}
    if label not in structures:
        structures[label] = {}
    structures[label]["status"] = "pending"
    structures[label]["resegment_task_id"] = task.id
    row.structures = structures
    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(row, "structures")

    logger.info(
        "resegmentation_requested",
        segmentation_id=str(segmentation_id),
        label=label,
        job_id=task.id,
        user_id=current_user.user_id,
    )
    return {"job_id": task.id, "label": label, "status": "pending"}


def _to_schema(row: SegmentationModel) -> SegmentationResult:
    """Convert ORM model to response schema."""
    labels: list[StructureLabel] = []
    if row.structure_labels:
        for name, label_val in row.structure_labels.items():
            labels.append(StructureLabel(name=name, label_value=label_val))

    confidence_maps: list[ConfidenceMap] = []
    if row.confidence_scores:
        volume_stats = row.volume_stats or {}
        for structure_name, mean_confidence in row.confidence_scores.items():
            stats = volume_stats.get(structure_name, {}) if isinstance(volume_stats, dict) else {}
            bounding_box = None
            bbox_data = stats.get("bounding_box")
            if isinstance(bbox_data, dict):
                try:
                    bounding_box = {
                        "min_x": bbox_data["min_x"],
                        "min_y": bbox_data["min_y"],
                        "min_z": bbox_data["min_z"],
                        "max_x": bbox_data["max_x"],
                        "max_y": bbox_data["max_y"],
                        "max_z": bbox_data["max_z"],
                    }
                except KeyError:
                    bounding_box = None

            confidence_maps.append(
                ConfidenceMap(
                    structure_name=structure_name,
                    mean_confidence=float(mean_confidence),
                    volume_voxels=stats.get("voxel_count"),
                    volume_cc=stats.get("volume_cc"),
                    bounding_box=bounding_box,
                    quality_flag=stats.get("quality_flag"),
                )
            )

    meshes: list[MeshInfo] = []
    if row.mesh_storage_paths:
        for structure, paths in row.mesh_storage_paths.items():
            if isinstance(paths, dict):
                for fmt, path in paths.items():
                    meshes.append(MeshInfo(structure_name=structure, format=fmt, path=path))

    return SegmentationResult(
        id=row.id,
        case_id=row.case_id,
        status=row.status,
        model_name=row.model_name,
        model_version=row.model_version,
        structure_labels=labels or None,
        confidence_maps=confidence_maps or None,
        structure_reviews=row.structures,
        overall_confidence=row.overall_confidence,
        provenance=row.provenance,
        mask_storage_path=row.mask_storage_path,
        meshes=meshes or None,
        fragment_count=row.fragment_count,
        fracture_fragments=list((row.fracture_fragments or {}).values()) or None,
        inference_time_ms=row.inference_time_ms,
        total_pipeline_time_ms=row.total_pipeline_time_ms,
        gpu_device=row.gpu_device,
        error_message=row.error_message,
        created_at=row.created_at,
        completed_at=row.completed_at,
    )
