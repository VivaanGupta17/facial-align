"""
Celery task definitions for all async processing pipelines.
Each task wraps a pipeline and handles status reporting and error recovery.
"""

from __future__ import annotations

import asyncio
import json
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


def _publish_ws_event(case_id: str, event: Dict[str, Any]) -> None:
    """Publish a WebSocket event via Redis pub/sub for cross-process delivery."""
    try:
        import redis as sync_redis
        from app.core.config import get_settings
        settings = get_settings()
        base_url = settings.celery.broker_url.rsplit("/", 1)[0]
        r = sync_redis.from_url(f"{base_url}/3", decode_responses=True)  # DB 3 for WS
        channel = f"ws:case:{case_id}"
        r.publish(channel, json.dumps(event))
        r.close()
    except Exception:
        task_logger.debug("Failed to publish WS event", exc_info=True)


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

        # Auto-chain: if quality passed, dispatch segmentation
        if result.get("quality_passed"):
            try:
                async def _create_segmentation():
                    import uuid
                    from app.db.database import get_db_context
                    from app.models.segmentation import SegmentationResult
                    from app.models.case import SurgicalCase
                    from sqlalchemy import select

                    seg_id = str(uuid.uuid4())
                    async with get_db_context() as db:
                        case = (await db.execute(
                            select(SurgicalCase).where(SurgicalCase.study_id == study_id)
                        )).scalar_one_or_none()

                        if not case:
                            return None, None

                        new_seg = SegmentationResult(
                            id=seg_id,
                            case_id=str(case.id),
                            study_id=study_id,
                            model_name="totalsegmentator",
                            status="queued",
                        )
                        db.add(new_seg)
                        return seg_id, str(case.id)

                seg_id, found_case_id = _run_async(_create_segmentation())
                if seg_id and found_case_id:
                    run_segmentation_pipeline.delay(
                        segmentation_id=seg_id,
                        case_id=found_case_id,
                        study_id=study_id,
                        identify_fragments=True,
                        run_dental=True,
                        user_id=uploader_user_id,
                    )
                    task_logger.info(f"Chained segmentation | seg_id={seg_id}")
            except Exception as chain_exc:
                task_logger.warning(f"Failed to chain segmentation task: {chain_exc}")

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

        # Auto-chain: if fragments were found, dispatch supervised reduction
        fragment_count = result.get("fragment_count", 0)
        if fragment_count > 0:
            try:
                async def _create_plan():
                    import uuid
                    from app.db.database import get_db_context
                    from app.models.plan import ReductionPlan

                    plan_id = str(uuid.uuid4())
                    async with get_db_context() as db:
                        new_plan = ReductionPlan(
                            id=plan_id,
                            case_id=case_id,
                            segmentation_id=segmentation_id,
                            status="generating",
                            plan_version=1,
                        )
                        db.add(new_plan)
                    return plan_id

                plan_id = _run_async(_create_plan())

                run_supervised_reduction_pipeline.delay(
                    plan_id=plan_id,
                    case_id=case_id,
                    segmentation_id=segmentation_id,
                    study_id=study_id,
                    user_id=user_id,
                )
                task_logger.info(
                    f"Chained supervised reduction | plan_id={plan_id} | "
                    f"fragments={fragment_count}"
                )
            except Exception as chain_exc:
                task_logger.warning(f"Failed to chain reduction task: {chain_exc}")

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


# ─── Supervised Reduction ─────────────────────────────────────────────────────


@celery_app.task(
    name="app.workers.tasks.run_supervised_reduction_pipeline",
    bind=True,
    base=BaseMLTask,
    max_retries=1,
    soft_time_limit=3600,
    time_limit=7200,
    queue="gpu",
)
def run_supervised_reduction_pipeline(
    self,
    plan_id: str,
    case_id: str,
    segmentation_id: str,
    study_id: str,
    checkpoint_path: Optional[str] = None,
    dental_constraints: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Supervised ML fracture reduction pipeline.

    Uses the trained supervised model (CT encoder + IOS encoder + fusion)
    instead of the optimization-based approach. Falls back to optimization
    if confidence is below threshold.

    Steps:
    1. Load CT volume and fragment data from segmentation
    2. Optionally load IOS tooth meshes
    3. Run SupervisedInferenceService.predict()
    4. Evaluate with ConfidenceGate
    5. If ACCEPT/REVIEW: apply transforms, resolve collisions, save plan
    6. If FALLBACK: dispatch optimization-based task
    7. If REJECT: mark plan as failed
    """
    task_logger.info(
        f"Starting supervised reduction | plan_id={plan_id} | case_id={case_id}"
    )

    self.update_state(
        state="PROGRESS",
        meta={"progress": 5, "step": "Loading CT volume and segmentation data"},
    )

    try:
        result = _run_async(_supervised_reduction_pipeline(
            self, plan_id, case_id, segmentation_id, study_id,
            checkpoint_path, dental_constraints, user_id,
        ))
        return result
    except Exception as exc:
        task_logger.error(f"Supervised reduction failed | plan_id={plan_id} | error={exc}")
        _publish_ws_event(case_id, "pipeline_error", {
            "stage": "error",
            "progress": 0,
            "message": str(exc),
        })
        _mark_plan_failed(plan_id, str(exc))
        raise


async def _supervised_reduction_pipeline(
    task,
    plan_id: str,
    case_id: str,
    segmentation_id: str,
    study_id: str,
    checkpoint_path: Optional[str],
    dental_constraints: Optional[Dict[str, Any]],
    user_id: Optional[str],
) -> Dict[str, Any]:
    """Inner async implementation of supervised reduction."""
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    from app.db.database import get_db_context
    from app.models.plan import ReductionPlan
    from app.models.segmentation import SegmentationResult
    from app.models.study import ImagingStudy
    from app.services.postprocessing.confidence_gate import (
        ConfidenceGate,
        DecisionType,
    )
    from app.services.postprocessing.collision_resolver import CollisionResolver
    from app.services.postprocessing.transform_applicator import TransformApplicator
    from app.services.supervised.inference_service import (
        InferenceConfig,
        SupervisedInferenceService,
    )
    from sqlalchemy import select, update
    import numpy as np

    start = datetime.now(timezone.utc)

    # ── 1. Load segmentation and study data ──
    task.update_state(state="PROGRESS", meta={"progress": 10, "step": "Loading segmentation data"})

    async with get_db_context() as db:
        seg = (await db.execute(
            select(SegmentationResult).where(SegmentationResult.id == segmentation_id)
        )).scalar_one_or_none()
        study = (await db.execute(
            select(ImagingStudy).where(ImagingStudy.id == study_id)
        )).scalar_one_or_none()

    if not seg:
        raise ValueError(f"Segmentation {segmentation_id} not found")
    if not study:
        raise ValueError(f"Study {study_id} not found")

    # ── 2. Load CT volume ──
    task.update_state(state="PROGRESS", meta={"progress": 20, "step": "Loading CT volume"})
    _publish_ws_event(case_id, "pipeline_progress", {
        "stage": "dicom_loading",
        "progress": 0.05,
        "message": "Loading DICOM volume...",
    })

    import SimpleITK as sitk
    volume_path = study.volume_path
    if volume_path and Path(volume_path).exists():
        sitk_img = sitk.ReadImage(str(volume_path))
        ct_volume = sitk.GetArrayFromImage(sitk_img)  # (D, H, W)
        ct_spacing = tuple(reversed(sitk_img.GetSpacing()))  # sitk returns (x,y,z), we need (z,y,x)
    else:
        raise ValueError(f"CT volume not found at {volume_path}")

    _publish_ws_event(case_id, "pipeline_progress", {
        "stage": "segmentation",
        "progress": 0.2,
        "message": "Running bone segmentation...",
    })

    # ── 3. Load fragment meshes ──
    task.update_state(state="PROGRESS", meta={"progress": 30, "step": "Loading fragment meshes"})

    from pipelines.fracture_reduction.pipeline import FractureReductionPipeline
    loader = FractureReductionPipeline(
        plan_id=plan_id,
        case_id=case_id,
        segmentation_id=segmentation_id,
    )
    fragments = await loader._load_fragments()

    if not fragments:
        raise ValueError(f"No fragments found in segmentation {segmentation_id}")

    _publish_ws_event(case_id, "pipeline_progress", {
        "stage": "fragment_extraction",
        "progress": 0.4,
        "message": "Extracting bone fragments...",
    })

    # ── 4. Load IOS tooth meshes (optional) ──
    task.update_state(state="PROGRESS", meta={"progress": 35, "step": "Loading dental meshes"})

    tooth_meshes = None
    if seg.dental_mesh_paths:
        tooth_meshes = {}
        import trimesh
        for fdi_str, paths in seg.dental_mesh_paths.items():
            fdi = int(fdi_str)
            glb_path = paths.get("glb") if isinstance(paths, dict) else None
            if glb_path and Path(glb_path).exists():
                try:
                    mesh = trimesh.load(glb_path)
                    if isinstance(mesh, trimesh.Trimesh):
                        tooth_meshes[fdi] = np.asarray(mesh.vertices)
                except Exception:
                    pass

    # ── 5. Run supervised inference ──
    task.update_state(state="PROGRESS", meta={"progress": 40, "step": "Running supervised model inference"})
    _publish_ws_event(case_id, "pipeline_progress", {
        "stage": "supervised_inference",
        "progress": 0.6,
        "message": "Running AI-guided fracture reduction...",
    })

    config = InferenceConfig(
        checkpoint_path=checkpoint_path,
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
    )
    service = SupervisedInferenceService(config)
    service.load_model()

    plan = service.predict(
        fragments=fragments,
        ct_volume=ct_volume,
        ct_spacing=ct_spacing,
        tooth_meshes=tooth_meshes if tooth_meshes else None,
    )

    # ── 6. Confidence gating ──
    task.update_state(state="PROGRESS", meta={"progress": 70, "step": "Evaluating prediction confidence"})
    _publish_ws_event(case_id, "pipeline_progress", {
        "stage": "confidence_evaluation",
        "progress": 0.8,
        "message": "Evaluating clinical confidence...",
    })

    gate = ConfidenceGate()
    fragment_data = [
        {
            "fragment_id": fid,
            "confidence": conf,
            "rotation_uncertainty_deg": 0.0,
            "translation_uncertainty_mm": 0.0,
        }
        for fid, conf in plan.fragment_confidences.items()
    ]
    prediction = gate.build_prediction_confidence(
        case_id=case_id,
        plan_id=plan_id,
        model_version=plan.model_version or "supervised_v1",
        fragment_data=fragment_data,
    )
    decision = gate.evaluate(prediction)

    task_logger.info(
        f"Confidence gate decision: {decision.decision.value} "
        f"(confidence={plan.overall_confidence:.3f})"
    )

    # ── 7. Route based on decision ──
    if decision.decision == DecisionType.FALLBACK:
        task.update_state(state="PROGRESS", meta={"progress": 75, "step": "Low confidence — falling back to optimization"})

        # Dispatch optimization-based task
        fallback_result = run_reduction_planning_pipeline.delay(
            plan_id=plan_id,
            case_id=case_id,
            segmentation_id=segmentation_id,
            model_name="occlusion_first",
            dental_constraints=dental_constraints,
            user_id=user_id,
        )

        return {
            "plan_id": plan_id,
            "status": "fallback_dispatched",
            "decision": "fallback",
            "supervised_confidence": plan.overall_confidence,
            "fallback_task_id": fallback_result.id,
            "reasons": decision.reasons,
        }

    if decision.decision == DecisionType.REJECT:
        task.update_state(state="PROGRESS", meta={"progress": 90, "step": "Prediction rejected — flagging for manual planning"})

        async with get_db_context() as db:
            await db.execute(
                update(ReductionPlan)
                .where(ReductionPlan.id == plan_id)
                .values(status="rejected", validation_warnings=decision.reasons)
            )

        return {
            "plan_id": plan_id,
            "status": "rejected",
            "decision": "reject",
            "reasons": decision.reasons,
        }

    # ACCEPT or REVIEW — apply transforms and save
    task.update_state(state="PROGRESS", meta={"progress": 80, "step": "Applying transforms and resolving collisions"})

    # Apply transforms
    applicator = TransformApplicator()
    # Load meshes for collision resolution
    import trimesh
    fragment_meshes = {}
    mesh_paths = seg.mesh_storage_paths or {}
    for fid in plan.fragment_transforms:
        struct_paths = mesh_paths.get(fid, {})
        ply_path = struct_paths.get("ply") if isinstance(struct_paths, dict) else None
        if ply_path and Path(ply_path).exists():
            try:
                fragment_meshes[fid] = trimesh.load(ply_path)
            except Exception:
                pass

    if fragment_meshes:
        # Apply transforms to meshes
        transformed = applicator.apply_fragment_transforms(
            fragment_meshes, plan.fragment_transforms
        )
        # Resolve collisions
        resolver = CollisionResolver()
        collision_result = resolver.resolve(
            meshes=fragment_meshes,
            transforms=plan.fragment_transforms,
        )
        if collision_result and hasattr(collision_result, 'corrected_transforms'):
            # Use collision-corrected transforms
            plan.fragment_transforms.update(collision_result.corrected_transforms)

    # ── 8. Save plan to database ──
    task.update_state(state="PROGRESS", meta={"progress": 90, "step": "Saving reduction plan"})

    end = datetime.now(timezone.utc)
    duration_ms = int((end - start).total_seconds() * 1000)

    transformations = {}
    for frag_id, transform_4x4 in plan.fragment_transforms.items():
        R = transform_4x4[:3, :3].tolist()
        t = transform_4x4[:3, 3].tolist()
        transformations[frag_id] = {
            "transform": {"rotation_matrix": R, "translation_mm": t},
            "fragment_label": 0,
            "confidence": plan.fragment_confidences.get(frag_id, 0.0),
        }

    occlusal_metrics_dict = None
    if plan.occlusal_metrics:
        occlusal_metrics_dict = plan.occlusal_metrics.model_dump(exclude_none=True)

    status = "validated" if decision.decision == DecisionType.ACCEPT else "pending_review"

    async with get_db_context() as db:
        await db.execute(
            update(ReductionPlan)
            .where(ReductionPlan.id == plan_id)
            .values(
                model_version=plan.model_version,
                transformations=transformations,
                occlusal_metrics=occlusal_metrics_dict,
                symmetry_metrics={"symmetry_score": plan.symmetry_score},
                confidence_score=plan.overall_confidence,
                validation_passed=plan.validation.passed if plan.validation else None,
                validation_warnings=(plan.validation.warnings if plan.validation else []) + decision.reasons,
                generation_time_ms=duration_ms,
                status=status,
            )
        )

    task.update_state(state="PROGRESS", meta={"progress": 100, "step": "Supervised reduction complete"})

    # ── 9. Chain to occlusion analysis ──
    run_occlusion_analysis_pipeline.delay(plan_id=plan_id, case_id=case_id)

    _publish_ws_event(case_id, "pipeline_complete", {
        "stage": "complete",
        "progress": 1.0,
        "message": "Reduction plan ready for review",
        "plan_id": str(plan_id),
        "confidence": plan.overall_confidence,
    })

    return {
        "plan_id": plan_id,
        "status": "complete",
        "decision": decision.decision.value,
        "n_fragments": len(fragments),
        "overall_confidence": plan.overall_confidence,
        "symmetry_score": plan.symmetry_score,
        "validation_passed": plan.validation.passed if plan.validation else None,
        "generation_time_ms": duration_ms,
        "flagged_fragments": decision.flagged_fragments,
        "requires_surgeon_review": decision.decision == DecisionType.REVIEW,
    }


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
