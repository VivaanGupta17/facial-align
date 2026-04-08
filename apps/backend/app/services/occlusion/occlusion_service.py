"""
Occlusal analysis service.
Evaluates dental occlusion quality after planned fracture reduction.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.exceptions import DentalArchError, OcclusionMetricError
from app.core.logging import TimedOperation, get_logger
from app.schemas.plan import OcclusalConstraints, OcclusalMetrics

logger = get_logger(__name__)


# ─── Output types ─────────────────────────────────────────────────────────────


@dataclass
class SplintDesignSpec:
    """
    Specification for an intermediate occlusal splint.
    Generated to guide and maintain occlusal position during fixation.
    """
    upper_arch_path: Optional[str] = None  # STL path for upper splint component
    lower_arch_path: Optional[str] = None  # STL path for lower splint component
    target_vertical_dimension_mm: float = 0.0
    contact_regions: List[Dict[str, Any]] = field(default_factory=list)
    thickness_map: Optional[Dict[str, float]] = None  # FDI number -> splint thickness mm
    material_recommendation: str = "acrylic_resin"  # or "clear_thermoplastic"
    notes: str = ""


@dataclass
class ArchGeometry:
    """Geometric representation of a dental arch from mesh data."""
    arch_points: np.ndarray  # (N, 3) cusp tip points in mm
    centroid_mm: np.ndarray
    midline_vector: np.ndarray  # Direction vector of dental midline
    arch_width_mm: float  # Intercanine or intermolar width
    arch_length_mm: float  # Anterior-posterior arch length
    curve_of_spee_depth_mm: float  # Depth of sagittal curve of Spee
    is_upper: bool


@dataclass
class OcclusalContact:
    """A single occlusal contact point between upper and lower arches."""
    upper_fdi: int
    lower_fdi: int
    contact_force_relative: float  # Relative force [0, 1]
    location_mm: List[float]  # Contact centroid in patient coordinates
    contact_area_mm2: float


# ─── Occlusion evaluation metrics reference values ───────────────────────────

NORMAL_OCCLUSAL_RANGES = {
    "overjet_mm": (1.0, 3.0),         # mm, class I normal
    "overbite_mm": (2.0, 4.0),         # mm, class I normal
    "midline_deviation_mm": (-1.0, 1.0),  # mm, symmetric
    "cant_degrees": (-2.0, 2.0),         # degrees, symmetric plane
    "curve_of_spee_mm": (0.5, 2.5),     # mm depth
}


class BaseOcclusionModel(abc.ABC):
    """Abstract interface for occlusal analysis models."""

    @abc.abstractmethod
    def evaluate(
        self,
        upper_arch: Any,
        lower_arch: Any,
        upper_transform: np.ndarray,
        lower_transform: np.ndarray,
    ) -> OcclusalMetrics:
        """Evaluate occlusal metrics for given arch positions."""
        ...


class GeometricOcclusionModel(BaseOcclusionModel):
    """
    Geometry-based occlusal metric computation.

    Uses dental landmark detection and mesh analysis to compute
    clinical occlusal measurements. This is the baseline implementation.
    For higher accuracy, a learned occlusion model should be trained.
    """

    def evaluate(
        self,
        upper_arch: Any,
        lower_arch: Any,
        upper_transform: np.ndarray,
        lower_transform: np.ndarray,
    ) -> OcclusalMetrics:
        """
        Compute occlusal metrics from arch meshes and planned transforms.

        TODO: Implement using dental landmark detection.
        Expected pipeline:
        1. Apply transforms to arch meshes
        2. Detect incisal edge points (FDI 11, 21, 31, 41)
        3. Compute overjet (horizontal incisal gap)
        4. Compute overbite (vertical incisal overlap)
        5. Detect molar contact points (FDI 16/26 vs 46/36)
        6. Classify Angle molar relationship
        7. Compute midline deviation
        8. Compute occlusal plane cant
        """
        # Placeholder — returns None metrics until landmarks are implemented
        return OcclusalMetrics(constraints_satisfied=True)


class LearnedOcclusionModel(BaseOcclusionModel):
    """
    ML-based occlusal prediction.

    TODO: Train this model.

    Architecture recommendation:
    - Input: Upper + lower arch point clouds (after planned transforms applied)
    - Output: Overjet, overbite, molar class (classification), midline, cant
    - Training: CBCT with annotated cephalometric measurements
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._model_path = model_path
        self._model = None

    @property
    def is_available(self) -> bool:
        return False  # Not trained yet

    def evaluate(
        self,
        upper_arch: Any,
        lower_arch: Any,
        upper_transform: np.ndarray,
        lower_transform: np.ndarray,
    ) -> OcclusalMetrics:
        raise NotImplementedError("Learned occlusion model not trained")


class OcclusionService:
    """
    Main occlusion analysis service.

    Computes clinical occlusal metrics (overjet, overbite, molar class, midline),
    validates against planning targets, and generates splint design specifications.
    """

    def __init__(
        self,
        use_learned_model: bool = False,
    ) -> None:
        # Choose model based on availability
        self._learned: Optional[LearnedOcclusionModel] = None
        if use_learned_model:
            self._learned = LearnedOcclusionModel()

        self._geometric = GeometricOcclusionModel()

    def _get_model(self) -> BaseOcclusionModel:
        """Return the best available model."""
        if self._learned and self._learned.is_available:
            return self._learned
        return self._geometric

    async def evaluate_occlusion(
        self,
        upper_arch: Any,  # Upper dental arch mesh (trimesh.Trimesh)
        lower_arch: Any,  # Lower dental arch mesh (trimesh.Trimesh)
        planned_transforms: Optional[Dict[str, np.ndarray]] = None,
        upper_fragment_id: Optional[str] = None,
        lower_fragment_id: Optional[str] = None,
    ) -> OcclusalMetrics:
        """
        Evaluate dental occlusion after applying planned reduction transforms.

        Args:
            upper_arch: Upper dental arch (maxillary) mesh
            lower_arch: Lower dental arch (mandibular) mesh
            planned_transforms: Fragment transforms to apply before evaluation
            upper_fragment_id: Fragment ID for the maxilla-bearing fragment
            lower_fragment_id: Fragment ID for the mandible-bearing fragment

        Returns:
            OcclusalMetrics with all clinical measurements
        """
        with TimedOperation(logger, "occlusion_evaluation"):
            if upper_arch is None or lower_arch is None:
                raise DentalArchError(
                    "Both upper and lower arch meshes are required for occlusal evaluation"
                )

            # Determine transforms to apply
            upper_T = np.eye(4)
            lower_T = np.eye(4)
            if planned_transforms:
                if upper_fragment_id and upper_fragment_id in planned_transforms:
                    upper_T = planned_transforms[upper_fragment_id]
                if lower_fragment_id and lower_fragment_id in planned_transforms:
                    lower_T = planned_transforms[lower_fragment_id]

            model = self._get_model()
            try:
                metrics = model.evaluate(upper_arch, lower_arch, upper_T, lower_T)
            except Exception as exc:
                raise OcclusionMetricError(
                    f"Occlusion evaluation failed: {exc}",
                    cause=exc,
                )

            self._assess_constraint_satisfaction(metrics)
            return metrics

    def _assess_constraint_satisfaction(self, metrics: OcclusalMetrics) -> None:
        """
        Check metrics against normal ranges and populate constraint_violations.
        """
        violations = []
        ranges = NORMAL_OCCLUSAL_RANGES

        if metrics.overjet_mm is not None:
            lo, hi = ranges["overjet_mm"]
            if not (lo <= metrics.overjet_mm <= hi):
                violations.append(
                    f"Overjet {metrics.overjet_mm:.1f}mm outside normal range "
                    f"[{lo}, {hi}]mm"
                )

        if metrics.overbite_mm is not None:
            lo, hi = ranges["overbite_mm"]
            if not (lo <= metrics.overbite_mm <= hi):
                violations.append(
                    f"Overbite {metrics.overbite_mm:.1f}mm outside normal range "
                    f"[{lo}, {hi}]mm"
                )

        if metrics.midline_deviation_mm is not None:
            lo, hi = ranges["midline_deviation_mm"]
            if not (lo <= metrics.midline_deviation_mm <= hi):
                violations.append(
                    f"Midline deviation {metrics.midline_deviation_mm:.1f}mm outside "
                    f"normal range [{lo}, {hi}]mm"
                )

        if metrics.cant_degrees is not None:
            lo, hi = ranges["cant_degrees"]
            if not (lo <= metrics.cant_degrees <= hi):
                violations.append(
                    f"Occlusal cant {metrics.cant_degrees:.1f}° outside "
                    f"normal range [{lo}, {hi}]°"
                )

        metrics.constraint_violations = violations
        metrics.constraints_satisfied = len(violations) == 0

    async def compute_dental_constraints(
        self,
        pre_injury_occlusion: Optional[Any],
        current_fragments: List[Any],
        tooth_masks: Optional[Dict[int, np.ndarray]] = None,
    ) -> OcclusalConstraints:
        """
        Derive target occlusal constraints from pre-injury occlusion records.

        If pre-injury CBCT or dental models are available, extract specific
        overjet/overbite/molar class targets. Otherwise, use clinical defaults.

        Args:
            pre_injury_occlusion: Pre-injury dental models or CBCT
            current_fragments: Current fracture fragments
            tooth_masks: Per-tooth segmentation masks

        Returns:
            OcclusalConstraints with computed targets
        """
        if pre_injury_occlusion is None:
            logger.info(
                "no_pre_injury_reference",
                note="Using clinical default occlusal targets",
            )
            # Return clinical defaults (Class I ideal occlusion)
            return OcclusalConstraints(
                target_overjet_mm=2.0,
                target_overbite_mm=3.0,
                molar_class_target="Class_I",
                midline_tolerance_mm=1.0,
                cant_tolerance_degrees=2.0,
            )

        # TODO: Analyze pre-injury occlusion and extract targets
        # This requires landmark detection on pre-injury dental model
        logger.info("computing_constraints_from_pre_injury_occlusion")

        return OcclusalConstraints(
            target_overjet_mm=2.0,
            target_overbite_mm=3.0,
            molar_class_target="Class_I",
            midline_tolerance_mm=1.0,
            cant_tolerance_degrees=2.0,
            use_pre_injury_occlusion=True,
        )

    async def suggest_splint_design(
        self,
        occlusal_plan: OcclusalMetrics,
        upper_arch: Optional[Any] = None,
        lower_arch: Optional[Any] = None,
        output_dir: Optional[str] = None,
    ) -> SplintDesignSpec:
        """
        Generate an intermediate occlusal splint specification.

        The splint positions the arches in the planned occlusal relationship
        during surgical fixation (IMF or temporary stabilization).

        Args:
            occlusal_plan: Target occlusal metrics
            upper_arch: Upper arch mesh
            lower_arch: Lower arch mesh
            output_dir: Directory to save splint STL files

        Returns:
            SplintDesignSpec with geometry paths and parameters
        """
        logger.info("generating_splint_design")

        # Determine vertical dimension based on target overbite
        # (simplified — production implementation uses arch geometry)
        target_vd = 3.0  # mm vertical opening for splint

        splint_spec = SplintDesignSpec(
            target_vertical_dimension_mm=target_vd,
            material_recommendation="clear_thermoplastic" if target_vd < 5 else "acrylic_resin",
            notes=(
                "Intermediate splint to maintain planned occlusal relationship during fixation. "
                "Fabricate from scan data and verify fit before surgery."
            ),
        )

        # TODO: Generate actual splint STL via mesh Boolean operations
        # Boolean(upper_arch, lower_arch_at_planned_position) → splint geometry

        if output_dir and upper_arch and lower_arch:
            splint_spec.notes += f"\nSplint STL generation requires production mesh Boolean library."

        return splint_spec

    def assess_molar_relationship(
        self,
        upper_first_molar_centroid: np.ndarray,  # Upper M1 (16 or 26)
        lower_first_molar_centroid: np.ndarray,  # Lower M1 (46 or 36)
    ) -> str:
        """
        Classify Angle molar relationship based on molar centroids.

        Angle classification:
        - Class I: Lower M1 mesio-buccal cusp occludes in buccal groove of Upper M1
        - Class II: Lower M1 distal relative to Class I (retrognathic mandible)
        - Class III: Lower M1 mesial relative to Class I (prognathic mandible)

        Returns:
            "Class_I", "Class_II_div1", "Class_II_div2", or "Class_III"
        """
        # Compute anteroposterior offset (x-axis in standard orientation)
        ap_offset = lower_first_molar_centroid[0] - upper_first_molar_centroid[0]

        # Threshold-based classification (clinical approximation)
        if -2.0 <= ap_offset <= 2.0:
            return "Class_I"
        elif ap_offset > 2.0:
            return "Class_III"  # Lower molar is anterior
        else:
            # Class II — need to differentiate div 1 vs div 2 using incisor angle
            return "Class_II_div1"

    def compute_arch_geometry(
        self,
        arch_mesh: Any,
        is_upper: bool,
        tooth_masks: Optional[Dict[int, np.ndarray]] = None,
        spacing: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> ArchGeometry:
        """
        Extract geometric arch parameters from a dental arch mesh.

        Args:
            arch_mesh: Dental arch surface mesh
            is_upper: True for maxillary (upper) arch
            tooth_masks: Per-tooth segmentation masks for precise cusp detection
            spacing: Volume spacing for mask-to-mm conversion

        Returns:
            ArchGeometry with clinical measurements
        """
        if not hasattr(arch_mesh, "vertices") or len(arch_mesh.vertices) == 0:
            raise DentalArchError("Empty arch mesh provided")

        vertices = np.asarray(arch_mesh.vertices)
        centroid = np.mean(vertices, axis=0)

        # Estimate arch width using PCA — first PC is arch direction
        centered = vertices - centroid
        try:
            _, _, Vt = np.linalg.svd(centered[:1000] if len(centered) > 1000 else centered)
            midline_vector = Vt[0]  # Primary direction
            width_vector = Vt[1]    # Secondary direction
        except Exception:
            midline_vector = np.array([0.0, 1.0, 0.0])
            width_vector = np.array([1.0, 0.0, 0.0])

        # Project vertices onto width axis to find arch width
        width_projections = vertices @ width_vector
        arch_width = float(width_projections.max() - width_projections.min())

        # Project vertices onto depth axis to find arch length (AP dimension)
        depth_projections = vertices @ midline_vector
        arch_length = float(depth_projections.max() - depth_projections.min())

        # Estimate curve of Spee depth (sagittal curvature)
        # Simplified: use z-range of posterior teeth vs anterior teeth
        if is_upper:
            # For upper arch, anterior is anterior
            ant_mask = depth_projections > np.percentile(depth_projections, 70)
        else:
            ant_mask = depth_projections < np.percentile(depth_projections, 30)

        ant_z = vertices[ant_mask, 2] if np.any(ant_mask) else np.array([centroid[2]])
        post_z = vertices[~ant_mask, 2] if np.any(~ant_mask) else np.array([centroid[2]])
        curve_of_spee = abs(float(np.mean(ant_z) - np.mean(post_z)))

        return ArchGeometry(
            arch_points=vertices,
            centroid_mm=centroid,
            midline_vector=midline_vector,
            arch_width_mm=arch_width,
            arch_length_mm=arch_length,
            curve_of_spee_depth_mm=min(curve_of_spee, 10.0),  # Cap at 10mm
            is_upper=is_upper,
        )
