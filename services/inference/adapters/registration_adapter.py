"""
Registration Model Adapter

Handles registration between different 3D data sources:
- CT-derived anatomy meshes ↔ Intraoral scan meshes
- Fragment meshes ↔ Reference anatomy
- Pre-operative ↔ Post-operative anatomy

Phase 1: ICP-based registration using Open3D
Phase 2: Learned registration (deep feature matching)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from services.inference.model_registry import InferenceModel, ModelVersion

logger = logging.getLogger(__name__)


@dataclass
class RegistrationResult:
    """Result of a registration operation."""

    transform_4x4: np.ndarray  # 4x4 homogeneous transform matrix
    fitness: float  # ICP fitness score (fraction of inlier correspondences)
    rmse_mm: float  # Root mean square error of inlier correspondences
    num_correspondences: int
    method: str
    iterations: int = 0
    elapsed_ms: int = 0


class ICPRegistration:
    """
    ICP-based registration using Open3D.

    Implements a coarse-to-fine strategy:
    1. RANSAC-based global registration (optional, for large misalignments)
    2. Point-to-plane ICP for fine alignment
    """

    def __init__(
        self,
        max_iterations: int = 200,
        voxel_size: float = 1.0,
        distance_threshold: float = 5.0,
    ):
        self.max_iterations = max_iterations
        self.voxel_size = voxel_size
        self.distance_threshold = distance_threshold

    def register(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
        initial_transform: Optional[np.ndarray] = None,
        use_global_registration: bool = False,
    ) -> RegistrationResult:
        """
        Register source point cloud to target using ICP.

        Args:
            source_points: (N, 3) source points
            target_points: (M, 3) target points
            initial_transform: Optional 4x4 initial alignment
            use_global_registration: Whether to use RANSAC for coarse alignment first

        Returns:
            RegistrationResult with transform and quality metrics
        """
        try:
            import open3d as o3d
        except ImportError:
            logger.error("Open3D not available — registration requires Open3D")
            return RegistrationResult(
                transform_4x4=np.eye(4),
                fitness=0.0,
                rmse_mm=999.0,
                num_correspondences=0,
                method="failed_no_open3d",
            )

        start_time = time.time()

        # Create point clouds
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(source_points)
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(target_points)

        # Estimate normals
        source.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.voxel_size * 2, max_nn=30
            )
        )
        target.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.voxel_size * 2, max_nn=30
            )
        )

        # Initial transform
        if initial_transform is None:
            initial_transform = np.eye(4)

        # Optional: Global registration first
        if use_global_registration:
            initial_transform = self._global_registration(source, target)

        # Fine ICP
        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            self.distance_threshold,
            initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.max_iterations,
            ),
        )

        elapsed = int((time.time() - start_time) * 1000)

        correspondences = np.asarray(result.correspondence_set)

        return RegistrationResult(
            transform_4x4=np.array(result.transformation),
            fitness=result.fitness,
            rmse_mm=result.inlier_rmse,
            num_correspondences=len(correspondences),
            method="icp_point_to_plane",
            elapsed_ms=elapsed,
        )

    def _global_registration(self, source, target) -> np.ndarray:
        """RANSAC-based global registration for coarse alignment."""
        import open3d as o3d

        # Downsample for feature extraction
        source_down = source.voxel_down_sample(self.voxel_size)
        target_down = target.voxel_down_sample(self.voxel_size)

        # Compute FPFH features
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_down,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.voxel_size * 5, max_nn=100
            ),
        )
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_down,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.voxel_size * 5, max_nn=100
            ),
        )

        # RANSAC registration
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down,
            target_down,
            source_fpfh,
            target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=self.distance_threshold * 2,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    self.distance_threshold * 2
                ),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500),
        )

        logger.info(f"Global registration: fitness={result.fitness:.3f}")
        return np.array(result.transformation)


class RegistrationModel(InferenceModel):
    """
    Unified registration adapter for the model registry.

    Wraps ICP registration (Phase 1) and future learned registration (Phase 2).
    """

    def __init__(self, version: ModelVersion, device: str = "cpu"):
        self._version = version
        self._device = device
        self._icp = ICPRegistration()

    @property
    def is_loaded(self) -> bool:
        return True

    def get_info(self) -> ModelVersion:
        return self._version

    def predict(self, input_data: np.ndarray, **kwargs) -> dict[str, Any]:
        """
        Run registration.

        Expected kwargs:
            target_points: np.ndarray — target point cloud
            use_global: bool — whether to use RANSAC first
            initial_transform: Optional[np.ndarray]
        """
        target = kwargs.get("target_points")
        if target is None:
            raise ValueError("target_points required for registration")

        result = self._icp.register(
            source_points=input_data,
            target_points=target,
            initial_transform=kwargs.get("initial_transform"),
            use_global_registration=kwargs.get("use_global", False),
        )

        return {
            "transform_4x4": result.transform_4x4.tolist(),
            "fitness": result.fitness,
            "rmse_mm": result.rmse_mm,
            "num_correspondences": result.num_correspondences,
            "method": result.method,
            "elapsed_ms": result.elapsed_ms,
        }
