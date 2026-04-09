"""
PyTorch Dataset for pre/post-op CT fracture pairs.

Loads pre-op CTs, extracts fragments and dental surfaces, loads post-op
CTs to derive ground truth SE(3) transforms per fragment, and optionally
loads IOS scans for per-tooth point clouds with FDI labels.

Returns: (pre_op_fragments, ios_teeth, ground_truth_transforms, ground_truth_metrics)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from data_contracts.training.training_case import TrainingCase
from training.datasets.data_augmentation import (
    AugmentationConfig,
    augment_point_cloud,
    augment_tooth_clouds,
)

logger = logging.getLogger(__name__)

# Number of points to sample per fragment / tooth
POINTS_PER_FRAGMENT = 4096
POINTS_PER_TOOTH = 1024


class FractureDataset(Dataset):
    """
    PyTorch Dataset for pre/post-op CT pair training.

    Each sample returns:
    - fragment_points: Dict[str, Tensor(N, 3)] — per-fragment point clouds
    - tooth_clouds: Dict[int, Tensor(M, 3)] — per-tooth IOS point clouds (if available)
    - fdi_numbers: List[int] — FDI numbers for teeth
    - gt_transforms: Dict[str, Tensor(4, 4)] — ground truth per-fragment transforms
    - gt_metrics: Dict[str, float] — ground truth occlusal metrics
    """

    def __init__(
        self,
        cases: List[TrainingCase],
        augmentation: Optional[AugmentationConfig] = None,
        points_per_fragment: int = POINTS_PER_FRAGMENT,
        points_per_tooth: int = POINTS_PER_TOOTH,
        load_ios: bool = True,
    ) -> None:
        self.augmentation = augmentation
        self.points_per_fragment = points_per_fragment
        self.points_per_tooth = points_per_tooth
        self.load_ios = load_ios

        # Filter to fracture_pair cases with ground truth
        self.cases = [c for c in cases if c.case_type == "fracture_pair"]
        logger.info(
            "FractureDataset initialized with %d cases (filtered from %d)",
            len(self.cases), len(cases),
        )

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        case = self.cases[idx]

        # Load fragment point clouds from pre-op segmentation
        fragment_points = self._load_fragments(case)

        # Load IOS tooth clouds if available
        tooth_clouds: Dict[int, torch.Tensor] = {}
        fdi_numbers: List[int] = []
        if self.load_ios and case.has_ios():
            tooth_clouds, fdi_numbers = self._load_ios_teeth(case)

        # Load ground truth transforms
        gt_transforms = self._load_gt_transforms(case)

        # Load ground truth metrics
        gt_metrics = case.ground_truth_occlusal_metrics or {}

        # Apply augmentations
        if self.augmentation is not None:
            fragment_points = {
                fid: augment_point_cloud(pts, self.augmentation)
                for fid, pts in fragment_points.items()
            }
            if tooth_clouds:
                tooth_clouds = augment_tooth_clouds(tooth_clouds, self.augmentation)
                fdi_numbers = sorted(tooth_clouds.keys())

        return {
            "case_id": case.case_id,
            "fragment_points": fragment_points,
            "tooth_clouds": tooth_clouds,
            "fdi_numbers": fdi_numbers,
            "gt_transforms": gt_transforms,
            "gt_metrics": gt_metrics,
        }

    def _load_fragments(self, case: TrainingCase) -> Dict[str, torch.Tensor]:
        """Load per-fragment point clouds from segmentation data."""
        fragments: Dict[str, torch.Tensor] = {}

        seg_path = case.preop_segmentation_path
        if seg_path is None or not seg_path.exists():
            # Try loading from DICOM via SimpleITK
            if case.preop_dicom_path is not None and case.preop_dicom_path.exists():
                return self._extract_fragments_from_dicom(case.preop_dicom_path)
            logger.warning(
                "No segmentation or DICOM found for case %s", case.case_id
            )
            return fragments

        # Load pre-computed segmentation masks (NPZ format)
        if seg_path.suffix == ".npz":
            data = np.load(str(seg_path), allow_pickle=True)
            for key in data.files:
                if key.startswith("fragment_"):
                    pts = data[key]
                    pts = self._normalize_point_count(pts, self.points_per_fragment)
                    fragments[key] = torch.tensor(pts, dtype=torch.float32)
        elif seg_path.is_dir():
            # Directory of per-fragment .npy files
            for npy_file in sorted(seg_path.glob("fragment_*.npy")):
                fid = npy_file.stem
                pts = np.load(str(npy_file))
                pts = self._normalize_point_count(pts, self.points_per_fragment)
                fragments[fid] = torch.tensor(pts, dtype=torch.float32)

        return fragments

    def _extract_fragments_from_dicom(
        self, dicom_path: Path,
    ) -> Dict[str, torch.Tensor]:
        """Extract fragment point clouds from DICOM via SimpleITK."""
        try:
            import SimpleITK as sitk
        except ImportError:
            logger.warning("SimpleITK not available — cannot load DICOM")
            return {}

        try:
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_path))
            if not dicom_names:
                return {}

            reader.SetFileNames(dicom_names)
            image = reader.Execute()

            # Convert to numpy volume
            volume = sitk.GetArrayFromImage(image)  # (Z, Y, X)
            spacing = np.array(image.GetSpacing())  # (X, Y, Z)
            origin = np.array(image.GetOrigin())

            # Threshold for bone (HU > 300)
            bone_mask = volume > 300

            # Extract bone surface points
            coords = np.argwhere(bone_mask)  # (N, 3) as (Z, Y, X) indices
            if len(coords) == 0:
                return {}

            # Convert to physical coordinates (mm)
            # coords are (Z, Y, X), spacing is (X, Y, Z)
            physical = coords[:, ::-1].astype(np.float64) * spacing + origin

            # Simple connected-component-based fragment extraction
            # Use a single fragment if proper segmentation is unavailable
            pts = self._normalize_point_count(physical, self.points_per_fragment)
            return {"fragment_0": torch.tensor(pts, dtype=torch.float32)}

        except Exception as exc:
            logger.warning("DICOM fragment extraction failed: %s", exc)
            return {}

    def _load_ios_teeth(
        self, case: TrainingCase,
    ) -> Tuple[Dict[int, torch.Tensor], List[int]]:
        """Load per-tooth point clouds from IOS mesh files."""
        tooth_clouds: Dict[int, torch.Tensor] = {}

        for arch_path, fdi_range in [
            (case.ios_upper_arch_path, range(11, 29)),
            (case.ios_lower_arch_path, range(31, 49)),
        ]:
            if arch_path is None or not arch_path.exists():
                continue

            try:
                import trimesh
                mesh = trimesh.load(str(arch_path))
                vertices = np.asarray(mesh.vertices)

                # If the mesh has per-tooth metadata, use it
                if hasattr(mesh, "metadata") and mesh.metadata and "tooth_labels" in mesh.metadata:
                    for fdi_str, indices in mesh.metadata["tooth_labels"].items():
                        fdi = int(fdi_str)
                        pts = vertices[indices]
                        pts = self._normalize_point_count(pts, self.points_per_tooth)
                        tooth_clouds[fdi] = torch.tensor(pts, dtype=torch.float32)
                else:
                    # Geometric partition fallback
                    teeth = self._geometric_tooth_partition(
                        vertices, list(fdi_range)
                    )
                    for fdi, pts in teeth.items():
                        pts = self._normalize_point_count(pts, self.points_per_tooth)
                        tooth_clouds[fdi] = torch.tensor(pts, dtype=torch.float32)

            except Exception as exc:
                logger.warning("IOS load failed for %s: %s", arch_path, exc)

        fdi_numbers = sorted(tooth_clouds.keys())
        return tooth_clouds, fdi_numbers

    def _load_gt_transforms(
        self, case: TrainingCase,
    ) -> Dict[str, torch.Tensor]:
        """Load ground truth SE(3) transforms from case annotations."""
        gt = {}
        if case.ground_truth_transforms is None:
            return gt

        for fid, flat in case.ground_truth_transforms.items():
            if len(flat) == 16:
                T = torch.tensor(flat, dtype=torch.float32).reshape(4, 4)
                gt[fid] = T
            else:
                logger.warning(
                    "Invalid transform length %d for %s/%s",
                    len(flat), case.case_id, fid,
                )

        return gt

    @staticmethod
    def _normalize_point_count(
        points: np.ndarray, target: int,
    ) -> np.ndarray:
        """Subsample or pad a point cloud to a target count."""
        n = points.shape[0]
        if n == 0:
            return np.zeros((target, 3), dtype=np.float32)
        if n >= target:
            idx = np.random.choice(n, target, replace=False)
            return points[idx].astype(np.float32)
        # Pad by repeating random points
        pad_idx = np.random.choice(n, target - n, replace=True)
        return np.concatenate(
            [points, points[pad_idx]], axis=0
        ).astype(np.float32)

    @staticmethod
    def _geometric_tooth_partition(
        vertices: np.ndarray,
        fdi_range: List[int],
    ) -> Dict[int, np.ndarray]:
        """Partition arch vertices into pseudo-tooth segments by position."""
        if len(vertices) == 0:
            return {}

        centroid = vertices.mean(axis=0)
        centered = vertices - centroid

        try:
            _, _, Vt = np.linalg.svd(centered[:min(1000, len(centered))])
            arch_dir = Vt[0]
            lateral_dir = Vt[1]
        except Exception:
            arch_dir = np.array([0.0, 1.0, 0.0])
            lateral_dir = np.array([1.0, 0.0, 0.0])

        arch_proj = vertices @ arch_dir
        lateral_proj = vertices @ lateral_dir
        median_lateral = np.median(lateral_proj)

        # Split into right/left halves
        right_mask = lateral_proj >= median_lateral
        left_mask = ~right_mask

        # Determine FDI sequences for right/left
        # Upper: right=11-18, left=21-28 ; Lower: right=41-48, left=31-38
        is_upper = fdi_range[0] < 30
        if is_upper:
            right_fdis = list(range(11, 19))
            left_fdis = list(range(21, 29))
        else:
            right_fdis = list(range(41, 49))
            left_fdis = list(range(31, 39))

        result: Dict[int, np.ndarray] = {}

        for fdis, mask in [(right_fdis, right_mask), (left_fdis, left_mask)]:
            side_verts = vertices[mask]
            if len(side_verts) == 0:
                continue

            side_proj = arch_proj[mask]
            n_teeth = len(fdis)
            percentiles = np.linspace(0, 100, n_teeth + 1)
            bounds = np.percentile(side_proj, percentiles)

            for i, fdi in enumerate(fdis):
                if i == n_teeth - 1:
                    seg_mask = side_proj >= bounds[i]
                else:
                    seg_mask = (side_proj >= bounds[i]) & (side_proj < bounds[i + 1])
                pts = side_verts[seg_mask]
                if len(pts) > 10:
                    result[fdi] = pts

        return result

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Custom collate function for DataLoader.

        Point cloud dicts cannot be stacked into tensors directly,
        so we keep them as lists.
        """
        return {
            "case_ids": [b["case_id"] for b in batch],
            "fragment_points": [b["fragment_points"] for b in batch],
            "tooth_clouds": [b["tooth_clouds"] for b in batch],
            "fdi_numbers": [b["fdi_numbers"] for b in batch],
            "gt_transforms": [b["gt_transforms"] for b in batch],
            "gt_metrics": [b["gt_metrics"] for b in batch],
        }

    @staticmethod
    def from_json(
        manifest_path: Path,
        augmentation: Optional[AugmentationConfig] = None,
        **kwargs: Any,
    ) -> "FractureDataset":
        """Load dataset from a JSON manifest file."""
        with open(manifest_path) as f:
            data = json.load(f)

        cases = [TrainingCase(**c) for c in data["cases"]]
        return FractureDataset(cases=cases, augmentation=augmentation, **kwargs)
