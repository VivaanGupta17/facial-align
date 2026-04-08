"""
Celery task definitions for all async processing pipelines.
Each task wraps a pipeline and handles status reporting and error recovery.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from celery import Task
from celery.utils.log import get_task_logger

from app.workers.celery_app import celery_app

task_logger = get_task_logger(__name__)


def _run_async(coro):
    """Run an async coroutine from a sync Celery task."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class BaseMLTask(Task):
    """Base Celery task with shared setup for ML pipeline tasks."""
    abstract = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure — update DB record status."""
        task_logger.error(
            "Task failed",
            exc_info=einfo,
            task_id=task_id,
        )
        super().on_failure(exc, task_id, args, kwargs, einfo)


# ─── DICOM Ingestion ──────────────────────────────────────────────────────────


@celery_app.task(
    name="app.workers.tasks.run_dicom_ingestion_pipeline",
    bind=True,
    base=BaseMLTask,
    max_retries=2,
    default_retry_delay=30,
    soft_time_limit=1800,  # 30 minutes
    time_limit=3600,
)
def run_dicom_ingestion_pipeline(
    self,
    study_id: str,
    upload_path: str,
    patient_id: str,
    uploader_user_id: str,
) -> Dict[str, Any]:
    """
    End-to-end DICOM ingestion pipeline task.

    Steps:
    1. Extract uploaded files
    2. Parse DICOM metadata
    3. De-identify PHI
    4. Reconstruct 3D volume
    5. Validate CT quality
    6. Update database with results
    7. Trigger downstream tasks if quality passes
    """
    from pathlib import Path

    from app.pipelines.dicom_ingestion.pipeline import DicomIngestionPipeline

    task_logger.info(
        f"Starting DICOM ingestion | study_id={study_id} | upload_path={upload_path}"
    )

    self.update_state(
        state="PROGRESS",
        meta={"progress": 5, "step": "Initializing ingestion pipeline"},
    )

    def progress_callback(pct: float, step: str):
        self.update_state(state="PROGRESS", meta={"progress": pct, "step": step})

    try:
        pipeline = DicomIngestionPipeline(
            study_id=study_id,
            upload_path=Path(upload_path),
            patient_id=patient_id,
            user_id=uploader_user_id,
            progress_callback=progress_callback,
        )
        result = _run_async(pipeline.run())

        task_logger.info(f"DICOM ingestion complete | study_id={study_id}")
        return {
            "study_id": study_id,
            "status": "complete",
            "modality": result.get("modality"),
            "series_count": result.get("series_count"),
            "quality_score": result.get("quality_score"),
            "quality_passed": result.get("quality_passed"),
        }

    except Exception as exc:
        task_logger.error(f"DICOM ingestion failed | study_id={study_id} | error={exc}")
        try:
            raise self.retry(exc=exc, countdown=30)
        except self.MaxRetriesExceededError:
            _mark_study_failed(study_id, str(exc))
            raise


# ─── Segmentation ─────────────────────────────────────────────────────────────


@celery_app.task(
    name="app.workers.tasks.run_segmentation_pipeline",
    bind=True,
    base=BaseMLTask,
    max_retries=1,
    soft_time_limit=3600,   # 1 hour
    time_limit=7200,
    queue="gpu",
)
def run_segmentation_pipeline(
    self,
    segmentation_id: str,
    case_id: str,
    study_id: str,
    model_name: str = "totalsegmentator",
    structures: Optional[List[str]] = None,
    run_dental: bool = False,
    identify_fragments: bool = True,
    fast_mode: bool = False,
    gpu_device: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full bone segmentation pipeline task.

    Steps:
    1. Load CT volume from storage
    2. Run ML segmentation (TotalSegmentator or custom model)
    3. Extract surface meshes per structure
    4. Export GLB + STL for viewer
    5. Run dental segmentation if requested
    6. Identify fracture fragments
    7. Update database with results
    """
    from app.pipelines.segmentation.pipeline import SegmentationPipeline

    task_logger.info(
        f"Starting segmentation | seg_id={segmentation_id} | model={model_name}"
    )

    self.update_state(
        state="PROGRESS",
        meta={"progress": 5, "step": "Loading CT volume"},
    )

    def progress_callback(pct: float, step: str):
        self.update_state(state="PROGRESS", meta={"progress": pct, "step": step})

    try:
        pipeline = SegmentationPipeline(
            segmentation_id=segmentation_id,
            case_id=case_id,
            study_id=study_id,
            model_name=model_name,
            structures=structures,
            run_dental=run_dental,
            identify_fragments=identify_fragments,
            fast_mode=fast_mode,
            gpu_device=gpu_device,
            user_id=user_id,
            progress_callback=progress_callback,
        )
        result = _run_async(pipeline.run())

        task_logger.info(
            f"Segmentation complete | seg_id={segmentation_id} | "
            f"structures={result.get('structures_found')}"
        )
        return result

    except Exception as exc:
        task_logger.error(
            f"Segmentation failed | seg_id={segmentation_id} | error={exc}"
        )
        _mark_segmentation_failed(segmentation_id, str(exc))
        try:
            raise self.retry(exc=exc, countdown=60)
        except self.MaxRetriesExceededError:
            raise


# ─── Mesh Extraction ──────────────────────────────────────────────────────────


@celery_app.task(
    name="app.workers.tasks.run_mesh_extraction_pipeline",
    bind=True,
    base=BaseMLTask,
    max_retries=2,
    soft_time_limit=1200,
    time_limit=2400,
)
def run_mesh_extraction_pipeline(
    self,
    segmentation_id: str,
    structures: Optional[List[str]] = None,
    target_face_ratio: float = 0.25,
) -> Dict[str, Any]:
    """
    Standalone mesh extraction task (for re-generating meshes with different params).

    Loads existing segmentation mask and re-extracts meshes.
    """
    from app.pipelines.mesh_extraction.pipeline import MeshExtractionPipeline

    task_logger.info(f"Starting mesh extraction | seg_id={segmentation_id}")

    self.update_state(
        state="PROGRESS",
        meta={"progress": 10, "step": "Loading segmentation mask"},
    )

    try:
        pipeline = MeshExtractionPipeline(
            segmentation_id=segmentation_id,
            structures=structures,
            target_face_ratio=target_face_ratio,
        )
        result = _run_async(pipeline.run())
        return result
    except Exception as exc:
        task_logger.error(f"Mesh extraction failed | seg_id={segmentation_id} | error={exc}")
        raise


# ─── Reduction Planning ───────────────────────────────────────────────────────


@celery_app.task(
    name="app.workers.tasks.run_reduction_planning_pipeline",
    bind=True,
    base=BaseMLTask,
    max_retries=1,
    soft_time_limit=3600,
    time_limit=7200,
    queue="gpu",
)
def run_reduction_planning_pipeline(
    self,
    plan_id: str,
    case_id: str,
    segmentation_id: str,
    model_name: str = "baseline_icp",
    dental_constraints: Optional[Dict[str, Any]] = None,
    use_intact_reference: bool = True,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fracture reduction planning pipeline task.

    Steps:
    1. Load fragment meshes from segmentation
    2. Load or generate intact reference anatomy
    3. Run ML reduction model
    4. Apply occlusal + skeletal constraints
    5. Run validation checks
    6. Store plan in database
    7. Trigger occlusion analysis
    """
    from app.pipelines.fracture_reduction.pipeline import FractureReductionPipeline

    task_logger.info(
        f"Starting reduction planning | plan_id={plan_id} | model={model_name}"
    )

    self.update_state(
        state="PROGRESS",
        meta={"progress": 5, "step": "Loading fragment geometry"},
    )

    def progress_callback(pct: float, step: str):
        self.update_state(state="PROGRESS", meta={"progress": pct, "step": step})

    try:
        pipeline = FractureReductionPipeline(
            plan_id=plan_id,
            case_id=case_id,
            segmentation_id=segmentation_id,
            model_name=model_name,
            dental_constraints=dental_constraints,
            use_intact_reference=use_intact_reference,
            user_id=user_id,
            progress_callback=progress_callback,
        )
        result = _run_async(pipeline.run())
        return result

    except Exception as exc:
        task_logger.error(
            f"Reduction planning failed | plan_id={plan_id} | error={exc}"
        )
        _mark_plan_failed(plan_id, str(exc))
        raise


@celery_app.task(
    name="app.workers.tasks.run_reduction_refinement",
    bind=True,
    base=BaseMLTask,
    max_retries=1,
    soft_time_limit=1800,
    queue="gpu",
)
def run_reduction_refinement(
    self,
    plan_id: str,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Re-optimize a reduction plan after surgeon edits.
    Applies constraint engine to surgeon-adjusted transforms.
    """
    task_logger.info(f"Starting reduction refinement | plan_id={plan_id}")

    self.update_state(
        state="PROGRESS",
        meta={"progress": 10, "step": "Loading plan data"},
    )

    try:
        async def _refine():
            from app.db.database import get_db_context
            from app.models.plan import ReductionPlan
            from app.services.reduction.reduction_service import FractureReductionService
            from sqlalchemy import select

            service = FractureReductionService()
            async with get_db_context() as db:
                plan = (
                    await db.execute(select(ReductionPlan).where(ReductionPlan.id == plan_id))
                ).scalar_one_or_none()
                if not plan:
                    raise ValueError(f"Plan {plan_id} not found")

                # Re-validate with updated transforms
                validation = await service.validate_plan_from_db_record(plan)

                plan.validation_passed = validation.passed
                plan.validation_warnings = validation.warnings
                plan.status = "validated" if validation.passed else "draft"

                return {
                    "plan_id": plan_id,
                    "validation_passed": validation.passed,
                    "status": plan.status,
                }

        return _run_async(_refine())

    except Exception as exc:
        task_logger.error(f"Reduction refinement failed | plan_id={plan_id} | error={exc}")
        raise


# ─── Occlusion Analysis ───────────────────────────────────────────────────────


@celery_app.task(
    name="app.workers.tasks.run_occlusion_analysis_pipeline",
    bind=True,
    base=BaseMLTask,
    soft_time_limit=1200,
    time_limit=2400,
)
def run_occlusion_analysis_pipeline(
    self,
    plan_id: str,
    case_id: str,
) -> Dict[str, Any]:
    """
    Standalone occlusion analysis task.
    Computes occlusal metrics for a completed reduction plan.
    """
    from app.pipelines.occlusion_planning.pipeline import OcclusionPlanningPipeline

    task_logger.info(f"Starting occlusion analysis | plan_id={plan_id}")

    self.update_state(
        state="PROGRESS",
        meta={"progress": 10, "step": "Loading dental arch meshes"},
    )

    try:
        pipeline = OcclusionPlanningPipeline(plan_id=plan_id, case_id=case_id)
        return _run_async(pipeline.run())
    except Exception as exc:
        task_logger.error(f"Occlusion analysis failed | plan_id={plan_id} | error={exc}")
        raise


# ─── Maintenance tasks ────────────────────────────────────────────────────────


@celery_app.task(name="app.workers.tasks.cleanup_expired_uploads")
def cleanup_expired_uploads() -> Dict[str, int]:
    """Clean up temporary upload directories older than 48 hours."""
    import shutil
    from datetime import datetime, timedelta, timezone
    from app.core.config import get_settings

    s = get_settings()
    upload_dir = s.storage.temp_path / "uploads"
    if not upload_dir.exists():
        return {"removed": 0}

    cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
    removed = 0
    for upload_path in upload_dir.iterdir():
        if upload_path.is_dir():
            import os
            mtime = datetime.fromtimestamp(os.path.getmtime(upload_path), tz=timezone.utc)
            if mtime < cutoff:
                shutil.rmtree(upload_path, ignore_errors=True)
                removed += 1
                task_logger.info(f"Cleaned up expired upload: {upload_path}")

    task_logger.info(f"Cleanup complete: removed {removed} expired upload directories")
    return {"removed": removed}


@celery_app.task(name="app.workers.tasks.archive_old_task_results")
def archive_old_task_results() -> Dict[str, int]:
    """Archive completed task results older than 7 days (noop if using Redis TTL)."""
    task_logger.info("Task result archival: using Redis TTL, no action needed")
    return {"archived": 0}


# ─── Helper functions ─────────────────────────────────────────────────────────


def _mark_study_failed(study_id: str, error: str) -> None:
    """Update study record to failed status in database."""
    async def _update():
        from app.db.database import get_db_context
        from app.models.study import ImagingStudy
        from sqlalchemy import select, update

        async with get_db_context() as db:
            await db.execute(
                update(ImagingStudy)
                .where(ImagingStudy.id == study_id)
                .values(ingestion_status="failed")
            )
    try:
        _run_async(_update())
    except Exception as e:
        task_logger.error(f"Failed to update study status: {e}")


def _mark_segmentation_failed(segmentation_id: str, error: str) -> None:
    """Update segmentation record to failed status."""
    async def _update():
        from app.db.database import get_db_context
        from app.models.segmentation import SegmentationResult
        from sqlalchemy import update

        async with get_db_context() as db:
            await db.execute(
                update(SegmentationResult)
                .where(SegmentationResult.id == segmentation_id)
                .values(status="failed", error_message=error[:500])
            )
    try:
        _run_async(_update())
    except Exception as e:
        task_logger.error(f"Failed to update segmentation status: {e}")


def _mark_plan_failed(plan_id: str, error: str) -> None:
    """Update plan record to failed status."""
    async def _update():
        from app.db.database import get_db_context
        from app.models.plan import ReductionPlan
        from sqlalchemy import update

        async with get_db_context() as db:
            await db.execute(
                update(ReductionPlan)
                .where(ReductionPlan.id == plan_id)
                .values(status="failed")
            )
    try:
        _run_async(_update())
    except Exception as e:
        task_logger.error(f"Failed to update plan status: {e}")
