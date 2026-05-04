"""
Occlusion-first fracture reduction pipeline.

Uses the occlusion-first FractureReductionService:
Phase 1: Dental-landmark ICP positioning
Phase 2: Joint optimization (occlusion + fracture via Adam)
Phase 3: Validation + scoring
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from app.core.config import get_settings
from app.core.logging import TimedOperation, get_logger
from app.services.capabilities import build_provenance
from app.services.reduction.reduction_service import (
    FragmentMesh,
    FractureReductionService,
    OcclusalConstraintEngine,
)

settings = get_settings()
logger = get_logger(__name__)


class FractureReductionPipeline:
    """
    Occlusion-first fracture reduction planning pipeline.

    Stages:
    5%  - Load segmentation data and fragment meshes
    20% - Load or generate intact reference anatomy
    30% - Load dental arch meshes (if available)
    40% - Phase 1: Dental-landmark ICP positioning
    55% - Phase 2: Joint occlusion+fracture optimization
    70% - Phase 3: Collision detection + scoring
    80% - Validate plan
    90% - Save plan to database
    100% - Complete
    """

    def __init__(
        self,
        plan_id: str,
        case_id: str,
        segmentation_id: str,
        model_name: str = "occlusion_first",
        dental_constraints: Optional[Dict[str, Any]] = None,
        use_intact_reference: bool = True,
        user_id: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> None:
        self._plan_id = plan_id
        self._case_id = case_id
        self._seg_id = segmentation_id
        self._model_name = model_name
        self._dental_constraints_dict = dental_constraints
        self._use_intact_reference = use_intact_reference
        self._user_id = user_id
        self._progress = progress_callback or (lambda pct, step: None)

    async def run(self) -> Dict[str, Any]:
        """Execute the occlusion-first fracture reduction pipeline."""
        from app.schemas.plan import OcclusalConstraints

        logger.info(
            "occlusion_first_reduction_pipeline_started",
            plan_id=self._plan_id,
            model=self._model_name,
        )

        with TimedOperation(logger, "reduction_pipeline", plan_id=self._plan_id):
            start = datetime.now(timezone.utc)

            # ── Load fragment meshes ──
            self._progress(5, "Loading fragment geometry from segmentation")
            fragments = await self._load_fragments()

            if not fragments:
                raise ValueError(
                    f"No fracture fragments found in segmentation {self._seg_id}"
                )

            logger.info("fragments_loaded", n_fragments=len(fragments))

            # ── Load intact reference ──
            self._progress(20, "Loading intact reference anatomy")
            intact_reference = None
            if self._use_intact_reference:
                intact_reference = await self._load_or_generate_reference(fragments)

            # ── Load dental arches ──
            self._progress(30, "Loading dental arch meshes")
            upper_arch, lower_arch = await self._load_dental_arches()

            # ── Parse constraints ──
            dental_constraints = None
            if self._dental_constraints_dict:
                try:
                    dental_constraints = OcclusalConstraints(**self._dental_constraints_dict)
                except Exception as exc:
                    logger.warning("invalid_dental_constraints", error=str(exc))

            # ── Run occlusion-first reduction ──
            self._progress(40, "Running occlusion-first reduction (landmark ICP + joint optimizer)")
            service = FractureReductionService()
            plan = await service.suggest_reduction(
                fragments=fragments,
                intact_reference=intact_reference,
                dental_constraints=dental_constraints,
                model_name=self._model_name,
                upper_arch=upper_arch,
                lower_arch=lower_arch,
            )

            # ── Save to database ──
            self._progress(90, "Saving reduction plan")
            end = datetime.now(timezone.utc)
            duration_ms = int((end - start).total_seconds() * 1000)
            await self._save_plan(plan, duration_ms)

            self._progress(100, "Occlusion-first reduction planning complete")

            logger.info(
                "occlusion_first_reduction_pipeline_complete",
                plan_id=self._plan_id,
                confidence=round(plan.overall_confidence, 3),
                symmetry=round(plan.symmetry_score, 3),
                validated=plan.validation.passed if plan.validation else None,
            )

            return {
                "plan_id": self._plan_id,
                "status": "complete",
                "n_fragments": len(fragments),
                "overall_confidence": plan.overall_confidence,
                "symmetry_score": plan.symmetry_score,
                "validation_passed": plan.validation.passed if plan.validation else None,
                "generation_time_ms": plan.generation_time_ms,
            }

    async def _load_fragments(self) -> List[FragmentMesh]:
        """Load fracture fragment meshes from the segmentation result."""
        from app.db.database import get_db_context
        from app.models.segmentation import SegmentationResult
        from sqlalchemy import select

        async with get_db_context() as db:
            seg = (
                await db.execute(
                    select(SegmentationResult).where(SegmentationResult.id == self._seg_id)
                )
            ).scalar_one_or_none()

        if not seg:
            raise ValueError(f"Segmentation {self._seg_id} not found")

        fragments: List[FragmentMesh] = []
        fracture_fragments: Dict[str, Any] = seg.fracture_fragments or {}
        fragment_mesh_paths: Dict[str, Any] = seg.fragment_mesh_paths or {}
        mesh_paths: Dict[str, Any] = seg.mesh_storage_paths or {}
        labels: Dict[str, int] = seg.structure_labels or {}
        volume_stats: Dict[str, Any] = seg.volume_stats or {}

        if fracture_fragments and fragment_mesh_paths:
            for fragment_id, info in fracture_fragments.items():
                struct_paths = fragment_mesh_paths.get(fragment_id, {})
                ply_path = struct_paths.get("ply") if isinstance(struct_paths, dict) else None
                points = await self._load_mesh_points(ply_path)
                if points is None or len(points) < 10:
                    logger.warning("skipping_empty_fragment", fragment_id=fragment_id)
                    continue

                centroid = np.array(info.get("centroid_mm") or [0.0, 0.0, 0.0], dtype=float)
                fragments.append(
                    FragmentMesh(
                        fragment_id=fragment_id,
                        label_value=info.get("label_value", 0),
                        points=points,
                        centroid_mm=centroid,
                        volume_mm3=float(info.get("volume_mm3", 0.0)),
                        parent_structure=info.get("parent_structure"),
                        is_reference=bool(info.get("is_reference_fragment", False)),
                    )
                )
        else:
            for structure_name, label_val in labels.items():
                struct_paths = mesh_paths.get(structure_name, {})
                ply_path = struct_paths.get("ply") if isinstance(struct_paths, dict) else None

                points = await self._load_mesh_points(ply_path)
                if points is None or len(points) < 10:
                    logger.warning("skipping_empty_structure", structure=structure_name)
                    continue

                stats = volume_stats.get(structure_name, {})
                centroid = np.array([
                    stats.get("centroid_x_mm", 0.0),
                    stats.get("centroid_y_mm", 0.0),
                    stats.get("centroid_z_mm", 0.0),
                ])
                volume_mm3 = stats.get("volume_mm3", 0.0)

                is_reference = structure_name in ["skull_base", "frontal_bone"]

                fragments.append(FragmentMesh(
                    fragment_id=structure_name,
                    label_value=label_val,
                    points=points,
                    centroid_mm=centroid,
                    volume_mm3=volume_mm3,
                    parent_structure=structure_name,
                    is_reference=is_reference,
                ))

        fragments.sort(key=lambda f: f.volume_mm3, reverse=True)
        if fragments and not any(f.is_reference for f in fragments):
            fragments[0].is_reference = True

        return fragments

    async def _load_mesh_points(self, path: Optional[str]) -> Optional[np.ndarray]:
        """Load vertex points from a mesh file."""
        if not path or not Path(path).exists():
            return None
        try:
            import trimesh
            mesh = trimesh.load(path)
            if isinstance(mesh, trimesh.Trimesh):
                return np.asarray(mesh.vertices)
        except Exception as exc:
            logger.warning("mesh_load_failed", path=path, error=str(exc))
        return None

    async def _load_or_generate_reference(
        self, fragments: List[FragmentMesh]
    ) -> Optional[Any]:
        """Load or generate intact reference anatomy."""
        logger.info(
            "generating_reference_anatomy",
            strategy="contralateral_mirror",
        )

        ref_fragments = [f for f in fragments if f.is_reference]
        if not ref_fragments:
            return None

        try:
            import trimesh
            all_points = np.vstack([f.points for f in ref_fragments])

            mirrored_points = all_points.copy()
            mirrored_points[:, 0] *= -1

            combined = np.vstack([all_points, mirrored_points])
            pcd = trimesh.PointCloud(combined)
            return pcd

        except Exception as exc:
            logger.warning("reference_generation_failed", error=str(exc))
            return None

    async def _load_dental_arches(self):
        """Load dental arch meshes for occlusal constraint evaluation."""
        from app.db.database import get_db_context
        from app.models.segmentation import SegmentationResult
        from sqlalchemy import select

        try:
            async with get_db_context() as db:
                seg = (
                    await db.execute(
                        select(SegmentationResult).where(SegmentationResult.id == self._seg_id)
                    )
                ).scalar_one_or_none()

            if not seg or not seg.dental_mesh_paths:
                return None, None

            import trimesh
            upper_meshes = []
            lower_meshes = []

            dental_paths = seg.dental_mesh_paths or {}
            for fdi_str, paths in dental_paths.items():
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
                        pass

            upper_arch = trimesh.util.concatenate(upper_meshes) if upper_meshes else None
            lower_arch = trimesh.util.concatenate(lower_meshes) if lower_meshes else None

            return upper_arch, lower_arch

        except Exception as exc:
            logger.warning("dental_arch_load_failed", error=str(exc))
            return None, None

    async def _save_plan(self, plan, generation_time_ms: int) -> None:
        """Save the reduction plan to the database."""
        from app.db.database import get_db_context
        from app.models.plan import ReductionPlan
        from app.models.segmentation import SegmentationResult
        from sqlalchemy import select, update

        transformations = {}
        for frag_id, transform_4x4 in plan.fragment_transforms.items():
            R = transform_4x4[:3, :3].tolist()
            t = transform_4x4[:3, 3].tolist()
            transformations[frag_id] = {
                "transform": {
                    "rotation_matrix": R,
                    "translation_mm": t,
                },
                "fragment_label": 0,
                "confidence": plan.fragment_confidences.get(frag_id, 0.0),
            }

        occlusal_metrics_dict = None
        if plan.occlusal_metrics:
            occlusal_metrics_dict = plan.occlusal_metrics.model_dump(exclude_none=True)

        symmetry_metrics = {
            "symmetry_score": plan.symmetry_score,
        }

        validation_warnings = []
        if plan.validation:
            validation_warnings = plan.validation.warnings

        async with get_db_context() as db:
            seg = (
                await db.execute(
                    select(SegmentationResult).where(SegmentationResult.id == self._seg_id)
                )
            ).scalar_one_or_none()
            fragments_payload = {}
            if seg and seg.fracture_fragments:
                for fragment_id, info in seg.fracture_fragments.items():
                    mesh_paths = (seg.fragment_mesh_paths or {}).get(fragment_id, {})
                    fragments_payload[fragment_id] = {
                        "label": info.get("label_value", 0),
                        "mesh_path": mesh_paths.get("ply") if isinstance(mesh_paths, dict) else None,
                        "volume_cc": round(float(info.get("volume_mm3", 0.0)) / 1000.0, 3),
                        "centroid_mm": info.get("centroid_mm"),
                        "parent_structure": info.get("parent_structure"),
                    }

            await db.execute(
                update(ReductionPlan)
                .where(ReductionPlan.id == self._plan_id)
                .values(
                    segmentation_id=self._seg_id,
                    model_version=plan.model_version,
                    fragments=fragments_payload or None,
                    transformations=transformations,
                    occlusal_metrics=occlusal_metrics_dict,
                    symmetry_metrics=symmetry_metrics,
                    provenance=build_provenance(
                        algorithm_used=self._model_name,
                        validation_tier=(
                            "deterministic_baseline"
                            if self._model_name in {"baseline_icp", "occlusion_first"}
                            else "learned_beta"
                        ),
                        beta_status=(
                            "not_beta"
                            if self._model_name in {"baseline_icp", "occlusion_first"}
                            else "beta_available"
                        ),
                        warnings=[],
                        model_version=plan.model_version,
                    ),
                    confidence_score=plan.overall_confidence,
                    validation_passed=plan.validation.passed if plan.validation else None,
                    validation_warnings=validation_warnings,
                    generation_time_ms=generation_time_ms,
                    status="validated" if (plan.validation and plan.validation.passed) else "draft",
                )
            )
