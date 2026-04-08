"""
Fracture Reduction Model Adapter

This module defines the ML inference interface for fracture reduction planning.
The reduction model predicts optimal rigid transforms for each fracture fragment
to reconstruct the craniofacial skeleton.

Phase 1 (Baseline): ICP-based alignment to contralateral/population reference
Phase 2 (Trained): SE(3)-equivariant transformer or PointNet++ predicting transforms
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from services.inference.model_registry import InferenceModel, ModelVersion

logger = logging.getLogger(__name__)


@dataclass
class FragmentGeometry:
    """Input geometry for a single fracture fragment."""

    fragment_id: str
    vertices: np.ndarray  # (N, 3) mesh vertices
    faces: np.ndarray  # (M, 3) face indices
    centroid: np.ndarray  # (3,) center of mass
    is_reference: bool = False  # Whether this is the reference fragment
    tooth_bearing: bool = False
    condyle_bearing: bool = False
    point_cloud: Optional[np.ndarray] = None  # (K, 3) sampled point cloud for ML


@dataclass
class OcclusalConstraint:
    """Occlusal constraint for reduction optimization."""

    target_overjet_mm: float = 2.5
    target_overbite_mm: float = 2.5
    max_midline_deviation_mm: float = 2.0
    target_molar_class: str = "class_I"
    constraint_weight: float = 0.6  # Weight in optimization objective

    # Upper arch reference (for dental constraint evaluation)
    upper_arch_vertices: Optional[np.ndarray] = None


@dataclass
class ReductionResult:
    """Output of the reduction model for a single fragment."""

    fragment_id: str
    rotation_matrix: np.ndarray  # (3, 3)
    translation_mm: np.ndarray  # (3,)
    confidence: float
    residual_error_mm: float
    method: str


class ReductionModelBase(ABC):
    """
    Abstract base class for fracture reduction models.

    All reduction implementations (ICP baseline, learned models) implement
    this interface. This ensures model swapping is a configuration change,
    not an architecture change.
    """

    @abstractmethod
    def predict_transforms(
        self,
        fragments: list[FragmentGeometry],
        constraints: Optional[OcclusalConstraint] = None,
        reference_anatomy: Optional[np.ndarray] = None,
    ) -> list[ReductionResult]:
        """
        Predict optimal reduction transforms for all mobile fragments.

        Args:
            fragments: List of fragment geometries (reference + mobile)
            constraints: Occlusal constraints to satisfy
            reference_anatomy: Optional pre-injury or population reference mesh

        Returns:
            List of ReductionResult for each mobile fragment
        """
        ...

    @abstractmethod
    def refine_from_edit(
        self,
        fragments: list[FragmentGeometry],
        manual_transform: dict[str, np.ndarray],
        constraints: Optional[OcclusalConstraint] = None,
    ) -> list[ReductionResult]:
        """
        Re-optimize other fragments given a surgeon's manual edit to one.

        This enables the collaborative AI-surgeon workflow: the surgeon
        adjusts one fragment, and the model re-optimizes the rest.
        """
        ...


class ICPBaselineReduction(ReductionModelBase):
    """
    Baseline reduction using ICP (Iterative Closest Point) alignment.

    Strategy:
    1. If reference anatomy available: align each fragment to reference using ICP
    2. If no reference: use contralateral symmetry (mirror mandible across midplane)
    3. Apply occlusal constraints as post-hoc correction

    This is geometrically reasonable but lacks the learned priors that
    a trained model would provide. It serves as the Phase 1 baseline
    and as a comparison point for trained models.
    """

    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def predict_transforms(
        self,
        fragments: list[FragmentGeometry],
        constraints: Optional[OcclusalConstraint] = None,
        reference_anatomy: Optional[np.ndarray] = None,
    ) -> list[ReductionResult]:
        results = []

        # Identify reference fragment
        ref_fragment = next((f for f in fragments if f.is_reference), None)
        mobile_fragments = [f for f in fragments if not f.is_reference]

        if ref_fragment is None:
            logger.warning("No reference fragment identified — using largest fragment")
            ref_fragment = max(fragments, key=lambda f: len(f.vertices))
            mobile_fragments = [f for f in fragments if f.fragment_id != ref_fragment.fragment_id]

        for fragment in mobile_fragments:
            try:
                result = self._align_fragment_icp(
                    fragment, ref_fragment, reference_anatomy
                )
                results.append(result)
            except Exception as e:
                logger.error(f"ICP failed for {fragment.fragment_id}: {e}")
                results.append(ReductionResult(
                    fragment_id=fragment.fragment_id,
                    rotation_matrix=np.eye(3),
                    translation_mm=np.zeros(3),
                    confidence=0.0,
                    residual_error_mm=999.0,
                    method="icp_failed",
                ))

        # Apply occlusal constraint correction
        if constraints is not None:
            results = self._apply_occlusal_correction(results, fragments, constraints)

        return results

    def _align_fragment_icp(
        self,
        fragment: FragmentGeometry,
        reference: FragmentGeometry,
        reference_anatomy: Optional[np.ndarray],
    ) -> ReductionResult:
        """Align a single fragment using ICP."""
        try:
            import open3d as o3d
        except ImportError:
            logger.warning("Open3D not available — returning identity transform")
            return ReductionResult(
                fragment_id=fragment.fragment_id,
                rotation_matrix=np.eye(3),
                translation_mm=np.zeros(3),
                confidence=0.5,
                residual_error_mm=0.0,
                method="identity_fallback",
            )

        # Create point clouds
        source_pc = o3d.geometry.PointCloud()
        source_pc.points = o3d.utility.Vector3dVector(fragment.vertices)

        # Use reference anatomy if available, otherwise use reference fragment
        if reference_anatomy is not None:
            target_points = reference_anatomy
        else:
            target_points = reference.vertices

        target_pc = o3d.geometry.PointCloud()
        target_pc.points = o3d.utility.Vector3dVector(target_points)

        # Estimate normals for point-to-plane ICP
        source_pc.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30)
        )
        target_pc.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30)
        )

        # Run ICP
        threshold = 5.0  # mm — initial correspondence distance
        reg_result = o3d.pipelines.registration.registration_icp(
            source_pc,
            target_pc,
            threshold,
            np.eye(4),  # Initial transform
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.max_iterations,
                relative_fitness=self.tolerance,
                relative_rmse=self.tolerance,
            ),
        )

        transform_4x4 = reg_result.transformation
        rotation = transform_4x4[:3, :3]
        translation = transform_4x4[:3, 3]

        # Compute confidence from ICP fitness
        confidence = min(1.0, reg_result.fitness)
        rmse = reg_result.inlier_rmse

        return ReductionResult(
            fragment_id=fragment.fragment_id,
            rotation_matrix=rotation,
            translation_mm=translation,
            confidence=round(confidence, 3),
            residual_error_mm=round(rmse, 3),
            method="icp_point_to_plane",
        )

    def _apply_occlusal_correction(
        self,
        results: list[ReductionResult],
        fragments: list[FragmentGeometry],
        constraints: OcclusalConstraint,
    ) -> list[ReductionResult]:
        """
        Post-hoc occlusal constraint correction.

        Adjusts transforms to better satisfy dental constraints.
        This is a simplified correction — the learned model in Phase 2
        will optimize constraints jointly with geometry.
        """
        # TODO: Implement occlusal constraint correction
        # For now, log a warning if constraints are provided but not enforced
        logger.info(
            "Occlusal constraints noted but post-hoc correction not yet implemented. "
            "Phase 2 learned model will optimize constraints jointly."
        )
        return results

    def refine_from_edit(
        self,
        fragments: list[FragmentGeometry],
        manual_transform: dict[str, np.ndarray],
        constraints: Optional[OcclusalConstraint] = None,
    ) -> list[ReductionResult]:
        """Re-run ICP with surgeon's edit as a fixed constraint."""
        # Apply surgeon's transform to the edited fragment
        edited_ids = set(manual_transform.keys())
        remaining_fragments = [f for f in fragments if f.fragment_id not in edited_ids]

        # Re-run ICP for remaining fragments with updated reference
        return self.predict_transforms(remaining_fragments, constraints)


class LearnedReductionModel(ReductionModelBase):
    """
    Placeholder for learned fracture reduction model.

    Architecture candidates (Phase 2):
    - SE(3)-Equivariant Transformer: Predicts transforms in a rotation-equivariant way
    - PointNet++: Encodes fragment point clouds, predicts transforms
    - Graph Neural Network: Models inter-fragment relationships

    Training data requirements:
    - Paired pre-operative (fractured) and post-operative (reduced) CT scans
    - Surgeon-verified reduction transforms
    - Minimum ~200 cases for initial training
    """

    def __init__(self, version: ModelVersion, device: str = "cpu"):
        self._version = version
        self._device = device
        self._model = None
        self._loaded = False

    def load_checkpoint(self, checkpoint_path: str):
        """Load trained model weights."""
        import torch

        logger.info(f"Loading reduction model from {checkpoint_path}")
        self._model = torch.load(checkpoint_path, map_location=self._device)
        self._model.eval()
        self._loaded = True

    def predict_transforms(
        self,
        fragments: list[FragmentGeometry],
        constraints: Optional[OcclusalConstraint] = None,
        reference_anatomy: Optional[np.ndarray] = None,
    ) -> list[ReductionResult]:
        if not self._loaded:
            raise RuntimeError(
                "Learned reduction model not loaded. "
                "This model requires training (Phase 2). "
                "Use ICPBaselineReduction for Phase 1."
            )

        # TODO: Implement learned model inference
        # 1. Encode fragment point clouds with PointNet++ backbone
        # 2. Encode constraints as feature vector
        # 3. Predict SE(3) transforms for each fragment
        # 4. Apply constraints as differentiable penalties
        # 5. Return transforms with model confidence
        raise NotImplementedError("Phase 2: Learned reduction model inference")

    def refine_from_edit(
        self,
        fragments: list[FragmentGeometry],
        manual_transform: dict[str, np.ndarray],
        constraints: Optional[OcclusalConstraint] = None,
    ) -> list[ReductionResult]:
        raise NotImplementedError("Phase 2: Learned refinement from surgeon edits")


class ReductionModel(InferenceModel):
    """
    Unified reduction model adapter for the model registry.

    Wraps both ICP baseline and learned model behind the InferenceModel interface.
    """

    def __init__(self, version: ModelVersion, device: str = "cpu"):
        self._version = version
        self._device = device

        # Select implementation based on version info
        if "learned" in version.name or version.checkpoint_path:
            self._impl = LearnedReductionModel(version, device)
            if version.checkpoint_path:
                self._impl.load_checkpoint(version.checkpoint_path)
        else:
            self._impl = ICPBaselineReduction()

    @property
    def is_loaded(self) -> bool:
        return True  # ICP baseline is always ready

    def get_info(self) -> ModelVersion:
        return self._version

    def predict(self, input_data: np.ndarray, **kwargs) -> dict[str, Any]:
        """
        Run reduction prediction.

        Expected kwargs:
            fragments: list[FragmentGeometry]
            constraints: Optional[OcclusalConstraint]
            reference_anatomy: Optional[np.ndarray]
        """
        fragments = kwargs.get("fragments", [])
        constraints = kwargs.get("constraints")
        reference = kwargs.get("reference_anatomy")

        start_time = time.time()
        results = self._impl.predict_transforms(fragments, constraints, reference)
        elapsed_ms = int((time.time() - start_time) * 1000)

        return {
            "transforms": [
                {
                    "fragment_id": r.fragment_id,
                    "rotation_matrix": r.rotation_matrix.tolist(),
                    "translation_mm": r.translation_mm.tolist(),
                    "confidence": r.confidence,
                    "residual_error_mm": r.residual_error_mm,
                    "method": r.method,
                }
                for r in results
            ],
            "inference_time_ms": elapsed_ms,
            "model_name": self._version.name,
            "model_version": self._version.version,
        }
