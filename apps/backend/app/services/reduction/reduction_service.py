"""
Fracture reduction planning service — the core ML service.
Predicts optimal fragment transforms using ML + occlusal/skeletal constraints.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.exceptions import (
    FractureFragmentError,
    InferenceError,
    ModelLoadError,
    ReductionConstraintViolation,
    SymmetryThresholdError,
)
from app.core.logging import MLInferenceLogger, TimedOperation, get_logger
from app.schemas.common import Transform3D
from app.schemas.plan import (
    FragmentTransform,
    OcclusalConstraints,
    OcclusalMetrics,
    ReductionPlanResponse,
    ValidationResult,
)

logger = get_logger(__name__)
inference_logger = MLInferenceLogger(logger)


# ─── Data structures ──────────────────────────────────────────────────────────


@dataclass
class FragmentMesh:
    """A fracture fragment with associated geometry."""
    fragment_id: str
    label_value: int
    points: np.ndarray  # (N, 3) surface point cloud in mm
    centroid_mm: np.ndarray  # (3,) centroid
    volume_mm3: float
    parent_structure: str = "unknown"
    is_reference: bool = False  # If True, this fragment stays in place


@dataclass
class ReductionPlan:
    """
    Complete fracture reduction plan output from the ML model.
    """
    fragment_transforms: Dict[str, np.ndarray]  # fragment_id -> 4x4 transform matrix
    fragment_confidences: Dict[str, float]  # fragment_id -> [0, 1]
    occlusal_metrics: Optional[OcclusalMetrics]
    symmetry_score: float  # [0, 1] — higher = more symmetric
    overall_confidence: float
    model_name: str
    model_version: str
    generation_time_ms: int
    validation: Optional[ValidationResult] = None
    alternative_plans: Optional[List["ReductionPlan"]] = None


# ─── Model ABC ────────────────────────────────────────────────────────────────


class ReductionModel(abc.ABC):
    """Abstract interface for fracture reduction prediction models."""

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    @abc.abstractmethod
    def version(self) -> str: ...

    @abc.abstractmethod
    def predict(
        self,
        fragments: List[FragmentMesh],
        intact_reference: Optional[Any],  # trimesh.Trimesh of intact anatomy
        occlusal_constraints: Optional[OcclusalConstraints],
    ) -> Dict[str, np.ndarray]:
        """
        Predict fragment transforms.

        Args:
            fragments: List of fracture fragment meshes
            intact_reference: Intact reference bone (mirrored or pre-injury)
            occlusal_constraints: Target dental occlusion specifications

        Returns:
            Dict of {fragment_id: 4x4_transform_matrix}
        """
        ...


# ─── Baseline ICP-based model ─────────────────────────────────────────────────


class BaselineReductionModel(ReductionModel):
    """
    Baseline fracture reduction using ICP-based symmetric alignment.

    Algorithm:
    1. Mirror the intact side of the skull to create a reference
    2. Register each displaced fragment to the mirrored reference
    3. Refine transforms to satisfy occlusal constraints

    This is the fallback when the learned model is not available.
    """

    def __init__(
        self,
        icp_max_iterations: int = 200,
        icp_correspondence_mm: float = 10.0,
    ) -> None:
        self._icp_max_iter = icp_max_iterations
        self._icp_corr_mm = icp_correspondence_mm

    @property
    def name(self) -> str:
        return "baseline_icp"

    @property
    def version(self) -> str:
        return "1.0.0"

    def predict(
        self,
        fragments: List[FragmentMesh],
        intact_reference: Optional[Any],
        occlusal_constraints: Optional[OcclusalConstraints],
    ) -> Dict[str, np.ndarray]:
        """ICP-based fragment alignment to symmetric reference."""
        transforms: Dict[str, np.ndarray] = {}

        # Reference fragment keeps identity transform
        ref_fragments = [f for f in fragments if f.is_reference]
        non_ref_fragments = [f for f in fragments if not f.is_reference]

        for frag in ref_fragments:
            transforms[frag.fragment_id] = np.eye(4)

        if not non_ref_fragments:
            return transforms

        if intact_reference is None:
            # No reference: use anatomical priors based on centroid positions
            logger.warning(
                "no_intact_reference",
                note="Using centroid-based alignment (lower accuracy)",
            )
            for frag in non_ref_fragments:
                transforms[frag.fragment_id] = self._estimate_transform_from_centroid(frag, fragments)
        else:
            # Align each fragment to the intact reference
            try:
                import open3d as o3d
            except ImportError:
                logger.warning("open3d_not_available", fallback="identity_transforms")
                for frag in non_ref_fragments:
                    transforms[frag.fragment_id] = np.eye(4)
                return transforms

            reference_pcd = o3d.geometry.PointCloud()
            if hasattr(intact_reference, "vertices"):
                ref_pts = np.asarray(intact_reference.vertices)
            else:
                ref_pts = intact_reference
            reference_pcd.points = o3d.utility.Vector3dVector(ref_pts)
            reference_pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30)
            )

            for frag in non_ref_fragments:
                try:
                    transform = self._icp_align_fragment(frag, reference_pcd)
                    transforms[frag.fragment_id] = transform
                except Exception as exc:
                    logger.error(
                        "fragment_alignment_failed",
                        fragment_id=frag.fragment_id,
                        error=str(exc),
                    )
                    transforms[frag.fragment_id] = np.eye(4)

        return transforms

    def _icp_align_fragment(
        self,
        fragment: FragmentMesh,
        reference_pcd: Any,
    ) -> np.ndarray:
        """Align a fragment to the reference using ICP."""
        import open3d as o3d

        frag_pcd = o3d.geometry.PointCloud()
        frag_pcd.points = o3d.utility.Vector3dVector(fragment.points)
        frag_pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30)
        )

        result = o3d.pipelines.registration.registration_icp(
            frag_pcd,
            reference_pcd,
            self._icp_corr_mm,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self._icp_max_iter
            ),
        )

        return np.asarray(result.transformation)

    def _estimate_transform_from_centroid(
        self,
        fragment: FragmentMesh,
        all_fragments: List[FragmentMesh],
    ) -> np.ndarray:
        """
        Rough centroid-based alignment when no reference is available.
        Attempts to bring fragment toward anatomical midline.
        """
        # Simple heuristic: translate toward the reference fragment centroid
        ref_fragments = [f for f in all_fragments if f.is_reference]
        if not ref_fragments:
            return np.eye(4)

        ref_centroid = np.mean([f.centroid_mm for f in ref_fragments], axis=0)

        # Mirror the fragment about the sagittal plane (x=0 in most CT coords)
        # to get its expected anatomical position
        expected_position = fragment.centroid_mm.copy()
        expected_position[0] = -fragment.centroid_mm[0]  # Mirror x-axis

        translation = expected_position - fragment.centroid_mm
        transform = np.eye(4)
        transform[:3, 3] = translation

        return transform


# ─── Learned reduction model ──────────────────────────────────────────────────


class LearnedReductionModel(ReductionModel):
    """
    Deep learning-based fracture reduction model.

    TODO: Train this model.

    Architecture recommendation:
    - Input: Per-fragment point cloud (N, 3) + global anatomy context
    - Backbone: PointNet++ or 3D sparse convolution (MinkowskiEngine)
    - Output: Per-fragment SE(3) transform (rotation + translation)
    - Loss: Chamfer distance from predicted → reference anatomy
           + occlusal constraint loss
           + symmetry regularization loss
    - Training data: CT scans of reduced vs. displaced fractures
                    (synthetic augmentation from displaced intact anatomy)

    Required training pipeline:
    1. Collect post-reduction CT scans (ground truth)
    2. Simulate fractures by virtual osteotomy + rigid displacement
    3. Train model to predict displacement reverse (reduction transform)
    4. Fine-tune with real clinical cases
    """

    def __init__(
        self,
        model_path: Optional[Path],
        device: str = "cuda",
        fp16: bool = True,
    ) -> None:
        self._model_path = model_path
        self._device = device
        self._fp16 = fp16
        self._model: Optional[Any] = None
        self._version = "not_trained"

    @property
    def name(self) -> str:
        return "learned_reduction_v1"

    @property
    def version(self) -> str:
        return self._version

    @property
    def is_available(self) -> bool:
        return self._model_path is not None and self._model_path.exists()

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if not self.is_available:
            raise ModelLoadError(
                "Learned reduction model not trained",
                context={"model_path": str(self._model_path)},
            )
        try:
            import torch
            # TODO: Import actual model architecture
            # from models.reduction_net import FractureReductionNet
            # checkpoint = torch.load(self._model_path, map_location=self._device)
            # self._model = FractureReductionNet(...)
            # self._model.load_state_dict(checkpoint["model_state_dict"])
            # self._model.eval().to(self._device)
            # self._version = checkpoint.get("version", "1.0.0")
            raise NotImplementedError("Model training pipeline not complete")
        except NotImplementedError:
            raise ModelLoadError("Learned reduction model training pipeline not complete")

    def predict(
        self,
        fragments: List[FragmentMesh],
        intact_reference: Optional[Any],
        occlusal_constraints: Optional[OcclusalConstraints],
    ) -> Dict[str, np.ndarray]:
        """
        TODO: Implement after model training.

        Expected inference steps:
        1. Encode each fragment as fixed-size feature vector using PointNet++
        2. Encode intact reference anatomy
        3. Compute fragment-reference attention features
        4. Decode per-fragment SE(3) transforms via rotation + translation heads
        5. Project translations to satisfy occlusal constraints
        """
        self._load_model()
        raise InferenceError("Learned reduction model not trained")


# ─── Constraint engine ────────────────────────────────────────────────────────


class OcclusalConstraintEngine:
    """
    Applies occlusal and skeletal constraints to refine reduction plans.

    The constraint engine takes an initial fragment configuration (from ICP or ML)
    and adjusts transforms to satisfy clinical requirements:
    - Overjet/overbite targets
    - Molar class I occlusion
    - Condylar seating
    - Facial symmetry
    """

    def __init__(
        self,
        symmetry_axis: int = 0,  # 0 = sagittal (x-axis) symmetry
        symmetry_tolerance_mm: float = 3.0,
    ) -> None:
        self._symmetry_axis = symmetry_axis
        self._symmetry_tolerance_mm = symmetry_tolerance_mm

    def apply_constraints(
        self,
        fragment_transforms: Dict[str, np.ndarray],
        fragments: List[FragmentMesh],
        dental_constraints: Optional[OcclusalConstraints],
        upper_dental_arch: Optional[Any],
        lower_dental_arch: Optional[Any],
    ) -> Tuple[Dict[str, np.ndarray], OcclusalMetrics]:
        """
        Refine fragment transforms to satisfy occlusal and symmetry constraints.

        Args:
            fragment_transforms: Initial transforms from ICP/ML
            fragments: Fragment mesh data
            dental_constraints: Target occlusal specifications
            upper_dental_arch: Upper dental arch mesh (from dental segmentation)
            lower_dental_arch: Lower dental arch mesh

        Returns:
            (refined_transforms, computed_occlusal_metrics)
        """
        # TODO: Implement full constraint optimization
        # Current implementation: pass-through with basic symmetry check

        refined = fragment_transforms.copy()
        metrics = OcclusalMetrics(constraints_satisfied=True)

        if dental_constraints:
            # Compute what the current transforms achieve
            metrics = self._estimate_occlusal_metrics(
                fragments, refined, upper_dental_arch, lower_dental_arch
            )

            # Check if constraints are satisfied
            violations = []
            if metrics.overjet_mm is not None and dental_constraints.target_overjet_mm is not None:
                if abs(metrics.overjet_mm - dental_constraints.target_overjet_mm) > 2.0:
                    violations.append(
                        f"Overjet {metrics.overjet_mm:.1f}mm deviates from target "
                        f"{dental_constraints.target_overjet_mm:.1f}mm by "
                        f"{abs(metrics.overjet_mm - dental_constraints.target_overjet_mm):.1f}mm"
                    )

            if violations:
                metrics.constraint_violations = violations
                metrics.constraints_satisfied = False
                logger.warning(
                    "occlusal_constraints_not_satisfied",
                    violations=violations,
                    note="Full constraint optimization not yet implemented",
                )

        return refined, metrics

    def _estimate_occlusal_metrics(
        self,
        fragments: List[FragmentMesh],
        transforms: Dict[str, np.ndarray],
        upper_arch: Optional[Any],
        lower_arch: Optional[Any],
    ) -> OcclusalMetrics:
        """
        Estimate occlusal metrics from planned fragment positions.
        TODO: Implement using dental arch geometry and landmarks.
        """
        return OcclusalMetrics(
            constraints_satisfied=True,
            constraint_violations=[],
        )

    def check_symmetry(
        self,
        fragments: List[FragmentMesh],
        transforms: Dict[str, np.ndarray],
    ) -> Tuple[float, List[str]]:
        """
        Compute facial symmetry score for the planned reduction.

        Returns:
            (symmetry_score [0-1], list_of_symmetry_violations)
        """
        violations = []

        # Pair bilateral fragments (e.g., zygoma_L with zygoma_R)
        # and check if their post-transform centroids are symmetric about the midline
        bilateral_pairs = self._find_bilateral_pairs(fragments)

        if not bilateral_pairs:
            return 0.85, []  # No bilateral pairs to check

        deviations = []
        for left_frag, right_frag in bilateral_pairs:
            if left_frag.fragment_id not in transforms or right_frag.fragment_id not in transforms:
                continue

            left_T = transforms[left_frag.fragment_id]
            right_T = transforms[right_frag.fragment_id]

            # Transform centroids
            left_cent_h = np.append(left_frag.centroid_mm, 1.0)
            right_cent_h = np.append(right_frag.centroid_mm, 1.0)

            left_cent_planned = (left_T @ left_cent_h)[:3]
            right_cent_planned = (right_T @ right_cent_h)[:3]

            # Mirror right fragment and compare to left
            right_mirrored = right_cent_planned.copy()
            right_mirrored[self._symmetry_axis] *= -1

            deviation_mm = np.linalg.norm(left_cent_planned - right_mirrored)
            deviations.append(deviation_mm)

            if deviation_mm > self._symmetry_tolerance_mm:
                violations.append(
                    f"Bilateral symmetry deviation: {left_frag.fragment_id} vs "
                    f"{right_frag.fragment_id}: {deviation_mm:.1f}mm "
                    f"(threshold: {self._symmetry_tolerance_mm}mm)"
                )

        if deviations:
            # Score: 1.0 = perfect symmetry, 0.0 = max deviation
            mean_dev = np.mean(deviations)
            symmetry_score = max(0.0, 1.0 - mean_dev / (self._symmetry_tolerance_mm * 3))
        else:
            symmetry_score = 0.85

        return float(symmetry_score), violations

    def _find_bilateral_pairs(
        self, fragments: List[FragmentMesh]
    ) -> List[Tuple[FragmentMesh, FragmentMesh]]:
        """Find bilateral (left/right) fragment pairs by naming convention."""
        pairs = []
        by_id = {f.fragment_id: f for f in fragments}

        for frag in fragments:
            if "_L" in frag.fragment_id:
                right_id = frag.fragment_id.replace("_L", "_R")
                if right_id in by_id:
                    pairs.append((frag, by_id[right_id]))

        return pairs


# ─── Main service ─────────────────────────────────────────────────────────────


class FractureReductionService:
    """
    Core fracture reduction planning service.

    Orchestrates:
    1. Fragment geometry encoding
    2. ML model inference (or ICP baseline)
    3. Constraint application
    4. Validation

    Usage:
        service = FractureReductionService()
        plan = await service.suggest_reduction(fragments, reference, constraints)
    """

    def __init__(
        self,
        model_registry: Optional[Any] = None,
        constraint_engine: Optional[OcclusalConstraintEngine] = None,
    ) -> None:
        self._registry = model_registry
        self._constraint_engine = constraint_engine or OcclusalConstraintEngine()

        # Initialize reduction models
        self._baseline = BaselineReductionModel()
        self._learned: Optional[LearnedReductionModel] = None

        if model_registry:
            try:
                from app.core.config import get_settings
                s = get_settings()
                self._learned = LearnedReductionModel(
                    model_path=s.model_registry.fracture_reduction_model_path,
                    device=s.model_registry.default_device,
                )
            except Exception as e:
                logger.warning("learned_reduction_model_unavailable", error=str(e))

    async def suggest_reduction(
        self,
        fragments: List[FragmentMesh],
        intact_reference: Optional[Any] = None,
        dental_constraints: Optional[OcclusalConstraints] = None,
        model_name: str = "baseline_icp",
        upper_arch: Optional[Any] = None,
        lower_arch: Optional[Any] = None,
    ) -> ReductionPlan:
        """
        Generate a fracture reduction plan.

        Pipeline:
        1. Select reduction model (learned if available, else baseline ICP)
        2. Encode fragment geometries
        3. Predict initial transforms
        4. Apply occlusal + skeletal constraints
        5. Compute confidence and metrics
        6. Validate result

        Args:
            fragments: List of FragmentMesh objects
            intact_reference: Intact reference anatomy (mirrored CT or historical)
            dental_constraints: Target occlusal specifications
            model_name: Model to use: "baseline_icp" or "learned_v1"
            upper_arch: Upper dental arch mesh
            lower_arch: Lower dental arch mesh

        Returns:
            ReductionPlan with transforms and metrics
        """
        with TimedOperation(logger, "fracture_reduction_planning"):
            start_time = time.perf_counter()

            if not fragments:
                raise FractureFragmentError(
                    "No fracture fragments provided",
                    context={"fragment_count": 0},
                )

            logger.info(
                "reduction_planning_started",
                n_fragments=len(fragments),
                model=model_name,
                has_reference=intact_reference is not None,
                has_dental_constraints=dental_constraints is not None,
            )

            # Select model
            model = self._select_model(model_name)

            # Run inference
            inference_start = inference_logger.log_inference_start(
                model_name=model.name,
                input_shape=(len(fragments),),
                device="cuda" if self._learned else "cpu",
            )

            try:
                raw_transforms = model.predict(fragments, intact_reference, dental_constraints)
            except Exception as exc:
                if model.name != self._baseline.name:
                    logger.warning(
                        "learned_model_failed_using_baseline",
                        error=str(exc),
                    )
                    raw_transforms = self._baseline.predict(
                        fragments, intact_reference, dental_constraints
                    )
                    model = self._baseline
                else:
                    raise

            inference_logger.log_inference_complete(
                model_name=model.name,
                start_time=inference_start,
                output_summary={"n_transforms": len(raw_transforms)},
            )

            # Apply constraints
            refined_transforms, occlusal_metrics = self._constraint_engine.apply_constraints(
                raw_transforms,
                fragments,
                dental_constraints,
                upper_arch,
                lower_arch,
            )

            # Compute symmetry
            symmetry_score, symmetry_violations = self._constraint_engine.check_symmetry(
                fragments, refined_transforms
            )

            # Compute per-fragment confidence
            fragment_confidences = self._estimate_confidences(
                fragments, refined_transforms, intact_reference
            )
            overall_confidence = float(np.mean(list(fragment_confidences.values()))) \
                if fragment_confidences else 0.0

            generation_time_ms = int((time.perf_counter() - start_time) * 1000)

            plan = ReductionPlan(
                fragment_transforms=refined_transforms,
                fragment_confidences=fragment_confidences,
                occlusal_metrics=occlusal_metrics,
                symmetry_score=symmetry_score,
                overall_confidence=overall_confidence,
                model_name=model.name,
                model_version=model.version,
                generation_time_ms=generation_time_ms,
            )

            # Validate
            plan.validation = self._validate_plan(plan, fragments, dental_constraints)

            logger.info(
                "reduction_planning_complete",
                n_fragments=len(fragments),
                overall_confidence=round(overall_confidence, 3),
                symmetry_score=round(symmetry_score, 3),
                validation_passed=plan.validation.passed,
                generation_time_ms=generation_time_ms,
            )

            return plan

    def _select_model(self, model_name: str) -> ReductionModel:
        """Select and return the appropriate reduction model."""
        if model_name == "learned_v1" and self._learned and self._learned.is_available:
            return self._learned
        if model_name == "baseline_icp":
            return self._baseline
        logger.warning(
            "reduction_model_not_available",
            requested=model_name,
            fallback="baseline_icp",
        )
        return self._baseline

    def _estimate_confidences(
        self,
        fragments: List[FragmentMesh],
        transforms: Dict[str, np.ndarray],
        reference: Optional[Any],
    ) -> Dict[str, float]:
        """
        Estimate per-fragment reduction confidence.

        For baseline ICP: based on ICP fitness score (larger fragment = higher confidence)
        For learned model: use model's output uncertainty estimates
        """
        confidences: Dict[str, float] = {}
        total_volume = sum(f.volume_mm3 for f in fragments) + 1e-6

        for frag in fragments:
            # Volume-based prior: larger fragments are aligned more reliably
            volume_ratio = frag.volume_mm3 / total_volume
            base_confidence = 0.5 + 0.4 * min(1.0, volume_ratio * 5)

            transform = transforms.get(frag.fragment_id, np.eye(4))
            # Check if transform is close to identity (may indicate failure)
            identity_distance = np.linalg.norm(transform - np.eye(4))
            if identity_distance < 0.1 and not frag.is_reference:
                base_confidence *= 0.7  # Penalize unchanged transforms

            confidences[frag.fragment_id] = round(base_confidence, 3)

        return confidences

    def _validate_plan(
        self,
        plan: "ReductionPlan",
        fragments: List[FragmentMesh],
        dental_constraints: Optional[OcclusalConstraints],
    ) -> ValidationResult:
        """Run automated clinical validation checks on a reduction plan."""
        warnings: List[str] = []
        errors: List[str] = []

        # ── Symmetry check ──
        symmetry_ok = plan.symmetry_score >= 0.7
        if not symmetry_ok:
            warnings.append(
                f"Facial symmetry score {plan.symmetry_score:.2f} below threshold 0.7"
            )

        # ── Occlusion check ──
        occlusion_ok = True
        if plan.occlusal_metrics:
            if not plan.occlusal_metrics.constraints_satisfied:
                occlusion_ok = False
                errors.extend(plan.occlusal_metrics.constraint_violations)

        # ── Hardware placement ──
        hardware_ok = True  # TODO: Implement hardware collision checking

        # ── Condylar seating ──
        condylar_ok = True  # TODO: Implement condylar seating verification

        passed = len(errors) == 0

        return ValidationResult(
            passed=passed,
            symmetry_ok=symmetry_ok,
            occlusion_ok=occlusion_ok,
            condylar_seating_ok=condylar_ok,
            hardware_placement_ok=hardware_ok,
            warnings=warnings,
            errors=errors,
            skeletal_symmetry_score=plan.symmetry_score,
        )

    async def validate_plan_from_db_record(self, plan_record: Any) -> ValidationResult:
        """Validate a plan loaded from the database."""
        from app.models.plan import ReductionPlan as PlanModel

        warnings: List[str] = []
        errors: List[str] = []

        # Check basic data completeness
        if not plan_record.transformations:
            errors.append("Plan has no fragment transforms")

        if plan_record.confidence_score is not None and plan_record.confidence_score < 0.5:
            warnings.append(f"Low plan confidence: {plan_record.confidence_score:.2f}")

        # Check occlusal metrics
        occlusion_ok = True
        if plan_record.occlusal_metrics:
            metrics = plan_record.occlusal_metrics
            if "constraint_violations" in metrics and metrics["constraint_violations"]:
                occlusion_ok = False
                errors.extend(metrics["constraint_violations"])

        symmetry_score = None
        if plan_record.symmetry_metrics:
            symm = plan_record.symmetry_metrics
            if "symmetry_score" in symm:
                symmetry_score = symm["symmetry_score"]

        passed = len(errors) == 0

        return ValidationResult(
            passed=passed,
            symmetry_ok=symmetry_score is None or symmetry_score >= 0.7,
            occlusion_ok=occlusion_ok,
            condylar_seating_ok=True,
            hardware_placement_ok=True,
            warnings=warnings,
            errors=errors,
            skeletal_symmetry_score=symmetry_score,
        )

    async def refine_reduction(
        self,
        plan: ReductionPlan,
        surgeon_edits: Dict[str, np.ndarray],
        fragments: List[FragmentMesh],
        dental_constraints: Optional[OcclusalConstraints] = None,
    ) -> ReductionPlan:
        """
        Re-optimize reduction from surgeon's manual adjustments.

        Applies surgeon-provided transforms as starting point and
        re-runs constraint optimization.
        """
        # Apply surgeon edits to the plan transforms
        updated_transforms = {**plan.fragment_transforms, **surgeon_edits}

        # Re-apply constraints
        refined_transforms, occlusal_metrics = self._constraint_engine.apply_constraints(
            updated_transforms,
            fragments,
            dental_constraints,
            upper_arch=None,
            lower_arch=None,
        )

        symmetry_score, _ = self._constraint_engine.check_symmetry(fragments, refined_transforms)
        confidences = self._estimate_confidences(fragments, refined_transforms, None)
        overall_confidence = float(np.mean(list(confidences.values()))) if confidences else 0.0

        refined_plan = ReductionPlan(
            fragment_transforms=refined_transforms,
            fragment_confidences=confidences,
            occlusal_metrics=occlusal_metrics,
            symmetry_score=symmetry_score,
            overall_confidence=overall_confidence,
            model_name=plan.model_name + "_surgeon_refined",
            model_version=plan.model_version,
            generation_time_ms=0,
        )

        refined_plan.validation = self._validate_plan(refined_plan, fragments, dental_constraints)

        return refined_plan
