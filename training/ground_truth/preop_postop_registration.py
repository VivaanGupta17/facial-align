"""
Pre-op to post-op CT registration for ground truth generation.

Uses the unaffected maxilla as an ICP anchor (it doesn't change between
pre/post-op), then derives per-fragment SE(3) transforms.

This gives the ground truth "what the surgeon actually did" — the per-
fragment rigid body transformations from fractured to reduced state.

Requires: open3d for ICP, SimpleITK for DICOM loading.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RegistrationResult:
    """Result of pre/post-op CT registration."""

    global_transform: np.ndarray  # (4, 4) — maxilla-anchored alignment
    fragment_transforms: Dict[str, np.ndarray]  # fragment_id → (4, 4)
    fitness: float  # ICP fitness score (0–1)
    rmse: float  # ICP root mean square error (mm)
    maxilla_points_used: int
    fragment_points: Dict[str, int]  # points per fragment


class PreopPostopRegistration:
    """
    Register pre-op to post-op CTs using maxilla as anchor.

    Algorithm:
    1. Load pre-op and post-op CT volumes
    2. Segment bone surfaces (HU thresholding)
    3. Identify maxilla region (upper jaw — anatomically stable)
    4. ICP-align maxilla between pre-op and post-op → global transform
    5. For each mandible fragment: compute pre-op → post-op transform
    """

    def __init__(
        self,
        bone_threshold_hu: float = 300.0,
        maxilla_z_min_fraction: float = 0.4,
        icp_max_distance_mm: float = 5.0,
        icp_max_iterations: int = 200,
        voxel_size_mm: float = 1.0,
    ) -> None:
        self.bone_threshold = bone_threshold_hu
        self.maxilla_z_min = maxilla_z_min_fraction
        self.icp_max_dist = icp_max_distance_mm
        self.icp_max_iter = icp_max_iterations
        self.voxel_size = voxel_size_mm

    def register(
        self,
        preop_dicom_path: Path,
        postop_dicom_path: Path,
        preop_segmentation_path: Optional[Path] = None,
    ) -> RegistrationResult:
        """
        Register pre-op to post-op CT and derive per-fragment transforms.

        Args:
            preop_dicom_path: Path to pre-op DICOM directory.
            postop_dicom_path: Path to post-op DICOM directory.
            preop_segmentation_path: Optional pre-computed segmentation.

        Returns:
            RegistrationResult with global and per-fragment transforms.
        """
        import open3d as o3d

        # 1. Load CT volumes
        preop_points, preop_spacing = self._load_bone_points(preop_dicom_path)
        postop_points, postop_spacing = self._load_bone_points(postop_dicom_path)

        logger.info(
            "Loaded bone surfaces: pre-op=%d pts, post-op=%d pts",
            len(preop_points), len(postop_points),
        )

        # 2. Identify maxilla region (upper portion of scan)
        preop_maxilla = self._extract_maxilla(preop_points)
        postop_maxilla = self._extract_maxilla(postop_points)

        logger.info(
            "Maxilla regions: pre-op=%d pts, post-op=%d pts",
            len(preop_maxilla), len(postop_maxilla),
        )

        # 3. ICP align maxilla (pre-op → post-op)
        global_transform, fitness, rmse = self._icp_register(
            source=preop_maxilla,
            target=postop_maxilla,
        )

        logger.info(
            "Maxilla ICP: fitness=%.4f, RMSE=%.3f mm", fitness, rmse,
        )

        # 4. Apply global transform to entire pre-op
        preop_aligned = self._apply_transform(preop_points, global_transform)

        # 5. Extract per-fragment transforms
        if preop_segmentation_path and preop_segmentation_path.exists():
            fragment_transforms, frag_counts = self._compute_fragment_transforms(
                preop_aligned, postop_points, preop_segmentation_path,
            )
        else:
            # Single-fragment fallback: mandible as one fragment
            mandible_pre = self._extract_mandible(preop_aligned)
            mandible_post = self._extract_mandible(postop_points)

            if len(mandible_pre) > 100 and len(mandible_post) > 100:
                frag_T, f_fit, f_rmse = self._icp_register(
                    mandible_pre, mandible_post,
                )
                fragment_transforms = {"fragment_0": frag_T}
                frag_counts = {"fragment_0": len(mandible_pre)}
                logger.info(
                    "Mandible ICP: fitness=%.4f, RMSE=%.3f mm", f_fit, f_rmse,
                )
            else:
                fragment_transforms = {}
                frag_counts = {}

        return RegistrationResult(
            global_transform=global_transform,
            fragment_transforms=fragment_transforms,
            fitness=fitness,
            rmse=rmse,
            maxilla_points_used=len(preop_maxilla),
            fragment_points=frag_counts,
        )

    def _load_bone_points(
        self, dicom_path: Path,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load bone surface points from DICOM CT via SimpleITK."""
        import SimpleITK as sitk

        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_path))
        if not dicom_names:
            raise ValueError(f"No DICOM files found in {dicom_path}")

        reader.SetFileNames(dicom_names)
        image = reader.Execute()

        volume = sitk.GetArrayFromImage(image)  # (Z, Y, X)
        spacing = np.array(image.GetSpacing())   # (X, Y, Z)
        origin = np.array(image.GetOrigin())

        # Threshold for bone
        bone_mask = volume > self.bone_threshold
        coords = np.argwhere(bone_mask)  # (N, 3) as (Z, Y, X)

        if len(coords) == 0:
            raise ValueError(f"No bone voxels found in {dicom_path}")

        # Convert to physical coordinates (mm)
        physical = coords[:, ::-1].astype(np.float64) * spacing + origin

        # Downsample for efficiency
        if len(physical) > 500_000:
            idx = np.random.choice(len(physical), 500_000, replace=False)
            physical = physical[idx]

        return physical, spacing

    def _extract_maxilla(self, points: np.ndarray) -> np.ndarray:
        """Extract maxilla (upper jaw) from bone point cloud."""
        z_vals = points[:, 2]
        z_range = z_vals.max() - z_vals.min()
        z_threshold = z_vals.min() + z_range * self.maxilla_z_min
        maxilla_mask = z_vals >= z_threshold
        return points[maxilla_mask]

    def _extract_mandible(self, points: np.ndarray) -> np.ndarray:
        """Extract mandible (lower jaw) from bone point cloud."""
        z_vals = points[:, 2]
        z_range = z_vals.max() - z_vals.min()
        z_threshold = z_vals.min() + z_range * self.maxilla_z_min
        mandible_mask = z_vals < z_threshold
        return points[mandible_mask]

    def _icp_register(
        self,
        source: np.ndarray,
        target: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:
        """Run ICP registration using open3d."""
        import open3d as o3d

        # Create point clouds
        src_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(source)

        tgt_pcd = o3d.geometry.PointCloud()
        tgt_pcd.points = o3d.utility.Vector3dVector(target)

        # Downsample
        src_down = src_pcd.voxel_down_sample(self.voxel_size)
        tgt_down = tgt_pcd.voxel_down_sample(self.voxel_size)

        # Estimate normals for point-to-plane ICP
        src_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.voxel_size * 2, max_nn=30,
            ),
        )
        tgt_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.voxel_size * 2, max_nn=30,
            ),
        )

        # Run point-to-plane ICP
        result = o3d.pipelines.registration.registration_icp(
            src_down,
            tgt_down,
            self.icp_max_dist,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.icp_max_iter,
            ),
        )

        return (
            np.asarray(result.transformation),
            result.fitness,
            result.inlier_rmse,
        )

    def _apply_transform(
        self, points: np.ndarray, T: np.ndarray,
    ) -> np.ndarray:
        """Apply a 4x4 transform to a point cloud."""
        R = T[:3, :3]
        t = T[:3, 3]
        return (points @ R.T) + t

    def _compute_fragment_transforms(
        self,
        preop_aligned: np.ndarray,
        postop_points: np.ndarray,
        seg_path: Path,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
        """Compute per-fragment transforms from segmentation data."""
        fragments: Dict[str, np.ndarray] = {}
        frag_counts: Dict[str, int] = {}

        if seg_path.suffix == ".npz":
            data = np.load(str(seg_path), allow_pickle=True)
            for key in data.files:
                if key.startswith("fragment_"):
                    fragments[key] = data[key]
        elif seg_path.is_dir():
            for npy_file in sorted(seg_path.glob("fragment_*.npy")):
                fragments[npy_file.stem] = np.load(str(npy_file))

        result: Dict[str, np.ndarray] = {}
        mandible_post = self._extract_mandible(postop_points)

        for fid, frag_pts in fragments.items():
            if len(frag_pts) < 50:
                continue

            try:
                T, fitness, rmse = self._icp_register(frag_pts, mandible_post)
                result[fid] = T
                frag_counts[fid] = len(frag_pts)
                logger.info(
                    "Fragment %s: fitness=%.4f, RMSE=%.3f mm",
                    fid, fitness, rmse,
                )
            except Exception as exc:
                logger.warning("Fragment %s ICP failed: %s", fid, exc)

        return result, frag_counts

    @staticmethod
    def flatten_transforms(
        transforms: Dict[str, np.ndarray],
    ) -> Dict[str, List[float]]:
        """Flatten 4x4 transform matrices to lists for JSON serialization."""
        return {
            fid: T.flatten().tolist()
            for fid, T in transforms.items()
        }
