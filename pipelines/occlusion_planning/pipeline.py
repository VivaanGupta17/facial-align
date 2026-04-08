"""
Occlusion planning pipeline: dental arch loading → metric computation → splint spec.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from app.core.config import get_settings
from app.core.logging import TimedOperation, get_logger
from app.services.occlusion.occlusion_service import OcclusionService

settings = get_settings()
logger = get_logger(__name__)


class OcclusionPlanningPipeline:
    """
    Occlusion analysis pipeline for a surgical reduction plan.

    Stages:
    10% - Load reduction plan from database
    20% - Load dental arch meshes
    35% - Apply planned fragment transforms to dental arches
    55% - Compute occlusal metrics
    70% - Evaluate constraint satisfaction
    80% - Generate splint design specification
    90% - Update plan with occlusal metrics
    100% - Complete
    """

    def __init__(
        self,
        plan_id: str,
        case_id: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> None:
        self._plan_id = plan_id
        self._case_id = case_id
        self._progress = progress_callback or (lambda pct, step: None)
        self._service = OcclusionService()

    async def run(self) -> Dict[str, Any]:
        """Execute the occlusion planning pipeline."""
        from app.db.database import get_db_context
        from app.models.plan import ReductionPlan
        from app.models.segmentation import SegmentationResult
        from sqlalchemy import select, update

        logger.info("occlusion_pipeline_started", plan_id=self._plan_id)

        with TimedOperation(logger, "occlusion_pipeline", plan_id=self._plan_id):
            # ── Load plan ──
            self._progress(10, "Loading reduction plan")
            async with get_db_context() as db:
                plan = (
                    await db.execute(
                        select(ReductionPlan).where(ReductionPlan.id == self._plan_id)
                    )
                ).scalar_one_or_none()

            if not plan:
                raise ValueError(f"Plan {self._plan_id} not found")

            # ── Load dental arches ──
            self._progress(20, "Loading dental arch meshes")
            upper_arch, lower_arch = await self._load_arches_from_case()

            if upper_arch is None or lower_arch is None:
                logger.warning(
                    "dental_arches_unavailable",
                    note="Skipping occlusal metric computation",
                )
                return {
                    "plan_id": self._plan_id,
                    "status": "skipped",
                    "reason": "Dental arch meshes not available",
                }

            # ── Apply transforms ──
            self._progress(35, "Applying planned transforms to dental arches")
            import numpy as np
            transforms: Dict[str, np.ndarray] = {}
            if plan.transformations:
                for frag_id, transform_data in plan.transformations.items():
                    if "transform" in transform_data:
                        t = transform_data["transform"]
                        R = np.array(t["rotation_matrix"])
                        translation = np.array(t["translation_mm"])
                        T_4x4 = np.eye(4)
                        T_4x4[:3, :3] = R
                        T_4x4[:3, 3] = translation
                        transforms[frag_id] = T_4x4

            # ── Compute occlusal metrics ──
            self._progress(55, "Computing occlusal metrics")
            try:
                metrics = await self._service.evaluate_occlusion(
                    upper_arch=upper_arch,
                    lower_arch=lower_arch,
                    planned_transforms=transforms,
                )

                logger.info(
                    "occlusal_metrics_computed",
                    overjet=metrics.overjet_mm,
                    overbite=metrics.overbite_mm,
                    molar_relationship=metrics.molar_relationship,
                    constraints_satisfied=metrics.constraints_satisfied,
                )
            except Exception as exc:
                logger.error("occlusal_metric_computation_failed", error=str(exc))
                metrics = None

            # ── Generate splint spec ──
            self._progress(80, "Generating splint design specification")
            splint_spec = None
            if metrics:
                try:
                    splint_spec = await self._service.suggest_splint_design(
                        occlusal_plan=metrics,
                        upper_arch=upper_arch,
                        lower_arch=lower_arch,
                    )
                except Exception as exc:
                    logger.warning("splint_design_failed", error=str(exc))

            # ── Update database ──
            self._progress(90, "Updating plan with occlusal metrics")
            if metrics:
                async with get_db_context() as db:
                    await db.execute(
                        update(ReductionPlan)
                        .where(ReductionPlan.id == self._plan_id)
                        .values(
                            occlusal_metrics=metrics.model_dump(exclude_none=True)
                        )
                    )

            self._progress(100, "Occlusion planning complete")

            logger.info(
                "occlusion_pipeline_complete",
                plan_id=self._plan_id,
                constraints_satisfied=metrics.constraints_satisfied if metrics else None,
                violations=len(metrics.constraint_violations) if metrics else 0,
            )

            return {
                "plan_id": self._plan_id,
                "status": "complete",
                "occlusal_metrics": metrics.model_dump(exclude_none=True) if metrics else None,
                "constraints_satisfied": metrics.constraints_satisfied if metrics else None,
                "splint_required": bool(
                    splint_spec and splint_spec.target_vertical_dimension_mm > 0
                ),
            }

    async def _load_arches_from_case(self):
        """Load dental arch meshes from the case's latest segmentation."""
        from app.db.database import get_db_context
        from app.models.plan import ReductionPlan
        from app.models.segmentation import SegmentationResult
        from sqlalchemy import select

        try:
            async with get_db_context() as db:
                # Get case_id from plan
                plan = (
                    await db.execute(
                        select(ReductionPlan).where(ReductionPlan.id == self._plan_id)
                    )
                ).scalar_one_or_none()
                if not plan:
                    return None, None

                # Get latest complete segmentation for this case
                seg = (
                    await db.execute(
                        select(SegmentationResult)
                        .where(SegmentationResult.case_id == plan.case_id)
                        .where(SegmentationResult.status == "complete")
                        .order_by(SegmentationResult.created_at.desc())
                        .limit(1)
                    )
                ).scalar_one_or_none()

            if not seg or not seg.dental_mesh_paths:
                return None, None

            import trimesh
            upper_meshes = []
            lower_meshes = []

            for fdi_str, paths in seg.dental_mesh_paths.items():
                fdi = int(fdi_str)
                glb_path = paths.get("glb") if isinstance(paths, dict) else None
                if glb_path and Path(glb_path).exists():
                    try:
                        mesh = trimesh.load(glb_path)
                        if 11 <= fdi <= 28:
                            upper_meshes.append(mesh)
                        elif 31 <= fdi <= 48:
                            lower_meshes.append(mesh)
                    except Exception:
                        continue

            upper_arch = trimesh.util.concatenate(upper_meshes) if upper_meshes else None
            lower_arch = trimesh.util.concatenate(lower_meshes) if lower_meshes else None

            return upper_arch, lower_arch

        except Exception as exc:
            logger.warning("arch_load_failed", error=str(exc))
            return None, None
