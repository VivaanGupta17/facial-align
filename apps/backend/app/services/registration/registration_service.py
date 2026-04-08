"""
Registration service: rigid body alignment of CT-to-scan and fragment registration.
Uses ICP as baseline; provides interface for learned registration models.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

from app.core.exceptions import (
    ICPConvergenceError,
    InsufficientOverlapError,
    RegistrationDivergenceError,
    RegistrationError,
)
from app.core.logging import MLInferenceLogger, TimedOperation, get_logger
from app.schemas.common import Transform3D

logger = get_logger(__name__)
inference_logger = MLInferenceLogger(logger)


@dataclass
class RegistrationMetrics:
    """Quality metrics for a completed registration."""
    rms_error_mm: float  # Root mean square surface distance error
    max_error_mm: float  # Maximum surface distance error
    mean_error_mm: float
    fitness_score: float  # Fraction of inlier correspondences [0, 1]
    inlier_count: int
    total_correspondences: int
    converged: bool
    iterations: int
    time_ms: int


class BaseRegistrationModel(abc.ABC):
    """Abstract interface for registration algorithms."""

    @abc.abstractmethod
    def register(
        self,
        source_points: np.ndarray,  # (N, 3) source point cloud
        target_points: np.ndarray,  # (M, 3) target point cloud
        initial_transform: Optional[np.ndarray] = None,  # 4x4 initial guess
    ) -> Tuple[np.ndarray, RegistrationMetrics]:
        """
        Register source to target.

        Returns:
            (transform_4x4, metrics)
        """
        ...


class ICPRegistrationModel(BaseRegistrationModel):
    """
    Iterative Closest Point (ICP) registration using Open3D.

    Provides robust rigid-body alignment between two point clouds or meshes.
    Used as the baseline registration algorithm.
    """

    def __init__(
        self,
        max_iterations: int = 200,
        convergence_threshold: float = 1e-6,
        max_correspondence_distance_mm: float = 5.0,
        voxel_downsample_mm: float = 1.0,
        method: str = "point_to_plane",  # "point_to_point" or "point_to_plane"
    ) -> None:
        self._max_iterations = max_iterations
        self._convergence_threshold = convergence_threshold
        self._max_corr_dist = max_correspondence_distance_mm
        self._voxel_size = voxel_downsample_mm
        self._method = method

    def register(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
        initial_transform: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, RegistrationMetrics]:
        """Run ICP registration between source and target point clouds."""
        try:
            import open3d as o3d
        except ImportError:
            raise RegistrationError(
                "Open3D not installed",
                context={"note": "Install with: pip install open3d"},
            )

        if len(source_points) < 10 or len(target_points) < 10:
            raise InsufficientOverlapError(
                "Insufficient points for ICP registration",
                context={
                    "source_points": len(source_points),
                    "target_points": len(target_points),
                },
            )

        # Convert to Open3D point clouds
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source_points)

        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_points)

        # Downsample for speed
        if self._voxel_size > 0:
            source_pcd = source_pcd.voxel_down_sample(self._voxel_size)
            target_pcd = target_pcd.voxel_down_sample(self._voxel_size)

        # Estimate normals (needed for point-to-plane ICP)
        if self._method == "point_to_plane":
            source_pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30)
            )
            target_pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30)
            )

        # Initial transform
        init_transform = initial_transform if initial_transform is not None else np.eye(4)

        start_time = time.perf_counter()

        # Run ICP
        try:
            if self._method == "point_to_plane":
                result = o3d.pipelines.registration.registration_icp(
                    source_pcd,
                    target_pcd,
                    self._max_corr_dist,
                    init_transform,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=self._max_iterations,
                        relative_fitness=self._convergence_threshold,
                        relative_rmse=self._convergence_threshold,
                    ),
                )
            else:
                result = o3d.pipelines.registration.registration_icp(
                    source_pcd,
                    target_pcd,
                    self._max_corr_dist,
                    init_transform,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=self._max_iterations,
                    ),
                )
        except Exception as exc:
            raise ICPConvergenceError(
                f"ICP failed: {exc}",
                cause=exc,
            )

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        transform_4x4 = np.asarray(result.transformation)

        # Evaluate registration quality
        eval_result = o3d.pipelines.registration.evaluate_registration(
            source_pcd,
            target_pcd,
            self._max_corr_dist,
            transform_4x4,
        )

        metrics = RegistrationMetrics(
            rms_error_mm=float(result.inlier_rmse),
            max_error_mm=float(result.inlier_rmse * 3),  # Approximate
            mean_error_mm=float(result.inlier_rmse),
            fitness_score=float(result.fitness),
            inlier_count=len(result.correspondence_set),
            total_correspondences=len(result.correspondence_set),
            converged=result.fitness > 0.3,  # Heuristic threshold
            iterations=self._max_iterations,
            time_ms=elapsed_ms,
        )

        if not metrics.converged:
            logger.warning(
                "icp_low_fitness",
                fitness=metrics.fitness_score,
                rms_error=metrics.rms_error_mm,
            )

        logger.info(
            "icp_registration_complete",
            fitness=round(metrics.fitness_score, 4),
            rms_error_mm=round(metrics.rms_error_mm, 3),
            time_ms=elapsed_ms,
        )

        return transform_4x4, metrics


class GlobalRegistrationModel(BaseRegistrationModel):
    """
    FPFH feature-based global registration followed by ICP refinement.
    Used for initial alignment when there's no good initial guess.
    """

    def __init__(
        self,
        voxel_size: float = 2.0,
        refine_with_icp: bool = True,
    ) -> None:
        self._voxel_size = voxel_size
        self._refine_with_icp = refine_with_icp
        self._icp = ICPRegistrationModel(voxel_downsample_mm=voxel_size / 2)

    def register(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
        initial_transform: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, RegistrationMetrics]:
        """Global registration using FPFH features + RANSAC + ICP refinement."""
        try:
            import open3d as o3d
        except ImportError:
            raise RegistrationError("Open3D not installed")

        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source_points)
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_points)

        # Downsample
        source_down = source_pcd.voxel_down_sample(self._voxel_size)
        target_down = target_pcd.voxel_down_sample(self._voxel_size)

        # Compute normals and FPFH features
        radius_normal = self._voxel_size * 2
        radius_feature = self._voxel_size * 5

        for pcd in [source_down, target_down]:
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius_normal, max_nn=30
            ))

        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )

        # RANSAC-based global registration
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down,
            target_down,
            source_fpfh,
            target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=self._voxel_size * 1.5,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(self._voxel_size * 1.5),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500),
        )

        global_transform = np.asarray(result_ransac.transformation)

        # Refine with ICP
        if self._refine_with_icp:
            return self._icp.register(source_points, target_points, global_transform)

        # Return global registration metrics
        metrics = RegistrationMetrics(
            rms_error_mm=float(result_ransac.inlier_rmse),
            max_error_mm=float(result_ransac.inlier_rmse * 4),
            mean_error_mm=float(result_ransac.inlier_rmse),
            fitness_score=float(result_ransac.fitness),
            inlier_count=len(result_ransac.correspondence_set),
            total_correspondences=len(result_ransac.correspondence_set),
            converged=result_ransac.fitness > 0.2,
            iterations=4000000,
            time_ms=0,
        )
        return global_transform, metrics


class DeepRegistrationModel(BaseRegistrationModel):
    """
    Deep learning-based registration model.

    TODO: Implement when training data is available.

    Architecture candidates:
    - PointNetLK: PointNet + Lucas-Kanade optimization
    - DCP (Deep Closest Point): Attention-based correspondence learning
    - GeoTransformer: Superpoint graph-based registration
    - REGTR: Registration TRansformer

    Expected advantages over ICP:
    - Single-pass inference (no iterative refinement needed)
    - Better handling of partial overlap and noise
    - Learns anatomy-specific symmetry constraints
    """

    def __init__(self, model_path: str, device: str = "cuda") -> None:
        self._model_path = model_path
        self._device = device
        self._model = None

    @property
    def is_available(self) -> bool:
        return False  # Not yet trained

    def register(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
        initial_transform: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, RegistrationMetrics]:
        raise NotImplementedError(
            "Deep registration model not yet trained. "
            "Falling back to ICP registration."
        )


class RegistrationService:
    """
    Main registration service. Coordinates alignment of CT to intraoral scan,
    and inter-fragment registration for surgical planning.
    """

    def __init__(
        self,
        use_global_registration: bool = True,
        icp_max_iterations: int = 200,
        correspondence_distance_mm: float = 3.0,
    ) -> None:
        self._use_global = use_global_registration
        self._icp = ICPRegistrationModel(
            max_iterations=icp_max_iterations,
            max_correspondence_distance_mm=correspondence_distance_mm,
            method="point_to_plane",
        )
        self._global_reg = GlobalRegistrationModel() if use_global_registration else None

        # Try to load deep registration (falls back gracefully)
        self._deep_reg: Optional[DeepRegistrationModel] = None

    def _mesh_to_points(self, mesh: Any, n_samples: int = 50_000) -> np.ndarray:
        """Sample points uniformly from a mesh surface."""
        try:
            import trimesh
            if isinstance(mesh, trimesh.Trimesh):
                points, _ = trimesh.sample.sample_surface(mesh, n_samples)
                return points
        except ImportError:
            pass

        # Fallback: use vertices directly
        if hasattr(mesh, "vertices"):
            return np.asarray(mesh.vertices)
        return mesh

    async def register_ct_to_scan(
        self,
        ct_mesh: Any,
        scan_mesh: Any,
        initial_transform: Optional[np.ndarray] = None,
    ) -> Tuple[Transform3D, RegistrationMetrics]:
        """
        Register CT-derived bone model to an intraoral scan.

        Used to align the bone coordinate system with the dental coordinate
        system from the intraoral scanner.

        Args:
            ct_mesh: CT-derived bone mesh (trimesh.Trimesh)
            scan_mesh: Intraoral scan mesh (trimesh.Trimesh)
            initial_transform: 4x4 initial alignment guess

        Returns:
            (Transform3D, RegistrationMetrics)
        """
        with TimedOperation(logger, "ct_to_scan_registration"):
            source_pts = self._mesh_to_points(ct_mesh)
            target_pts = self._mesh_to_points(scan_mesh)

            # Use global registration if no initial guess provided
            if initial_transform is None and self._global_reg is not None:
                logger.info("running_global_registration_for_initial_alignment")
                transform_4x4, metrics = self._global_reg.register(source_pts, target_pts)
            else:
                transform_4x4, metrics = self._icp.register(
                    source_pts, target_pts, initial_transform
                )

            if metrics.rms_error_mm > 5.0:
                logger.warning(
                    "high_registration_error",
                    rms_mm=metrics.rms_error_mm,
                    fitness=metrics.fitness_score,
                )

            return self._matrix_to_transform3d(transform_4x4), metrics

    async def register_fragments(
        self,
        fragment_meshes: List[Any],
        reference_mesh: Any,
        initial_transforms: Optional[List[np.ndarray]] = None,
    ) -> List[Tuple[Transform3D, RegistrationMetrics]]:
        """
        Register each fracture fragment to an intact reference anatomy.

        Args:
            fragment_meshes: List of fragment meshes
            reference_mesh: Intact reference bone mesh (e.g., mirrored anatomy)
            initial_transforms: Per-fragment initial alignment guesses

        Returns:
            List of (Transform3D, RegistrationMetrics) per fragment
        """
        with TimedOperation(logger, "fragment_registration", n_fragments=len(fragment_meshes)):
            reference_pts = self._mesh_to_points(reference_mesh)
            results = []

            for i, frag in enumerate(fragment_meshes):
                frag_pts = self._mesh_to_points(frag)
                init_t = initial_transforms[i] if initial_transforms else None

                try:
                    transform_4x4, metrics = self._icp.register(
                        frag_pts, reference_pts, init_t
                    )
                    results.append((self._matrix_to_transform3d(transform_4x4), metrics))
                    logger.info(
                        "fragment_registered",
                        fragment_index=i,
                        rms_mm=round(metrics.rms_error_mm, 3),
                        fitness=round(metrics.fitness_score, 3),
                    )
                except (ICPConvergenceError, InsufficientOverlapError) as exc:
                    logger.warning(
                        "fragment_registration_failed",
                        fragment_index=i,
                        error=str(exc),
                        fallback="identity_transform",
                    )
                    identity = self._matrix_to_transform3d(np.eye(4))
                    failed_metrics = RegistrationMetrics(
                        rms_error_mm=999.0, max_error_mm=999.0, mean_error_mm=999.0,
                        fitness_score=0.0, inlier_count=0, total_correspondences=0,
                        converged=False, iterations=0, time_ms=0,
                    )
                    results.append((identity, failed_metrics))

            return results

    def compute_registration_error(
        self,
        source_mesh: Any,
        target_mesh: Any,
        transform: Transform3D,
    ) -> RegistrationMetrics:
        """
        Compute registration error metrics for a given transform.

        Args:
            source_mesh: Source mesh (to be transformed)
            target_mesh: Target reference mesh
            transform: Proposed transformation

        Returns:
            RegistrationMetrics
        """
        try:
            import open3d as o3d
        except ImportError:
            raise RegistrationError("Open3D not installed")

        transform_4x4 = np.array(transform.to_4x4_matrix())
        source_pts = self._mesh_to_points(source_mesh)
        target_pts = self._mesh_to_points(target_mesh)

        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source_pts)
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_pts)

        eval_result = o3d.pipelines.registration.evaluate_registration(
            source_pcd, target_pcd, 5.0, transform_4x4
        )

        return RegistrationMetrics(
            rms_error_mm=float(eval_result.inlier_rmse),
            max_error_mm=float(eval_result.inlier_rmse * 3),
            mean_error_mm=float(eval_result.inlier_rmse),
            fitness_score=float(eval_result.fitness),
            inlier_count=len(eval_result.correspondence_set),
            total_correspondences=len(eval_result.correspondence_set),
            converged=eval_result.fitness > 0.3,
            iterations=0,
            time_ms=0,
        )

    def _matrix_to_transform3d(self, matrix_4x4: np.ndarray) -> Transform3D:
        """Convert a 4x4 homogeneous matrix to a Transform3D schema."""
        R = matrix_4x4[:3, :3].tolist()
        t = matrix_4x4[:3, 3].tolist()
        # Normalize rotation matrix to ensure orthonormality
        R_np = np.array(R)
        U, _, Vt = np.linalg.svd(R_np)
        R_ortho = (U @ Vt).tolist()
        return Transform3D(rotation_matrix=R_ortho, translation_mm=t)
