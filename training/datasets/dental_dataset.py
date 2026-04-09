"""
PyTorch Dataset for IOS dental scans.

Loads upper + lower IOS scans, extracts per-tooth point clouds with
FDI labels. If paired with occlusion metrics → supervised.
If unpaired → can use for self-supervised pretraining with composite loss.
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
    augment_tooth_clouds,
)

logger = logging.getLogger(__name__)

POINTS_PER_TOOTH = 1024


class DentalDataset(Dataset):
    """
    PyTorch Dataset for IOS-only dental occlusion training.

    Each sample returns:
    - upper_teeth: Dict[int, Tensor(N, 3)] — per-tooth upper arch clouds
    - lower_teeth: Dict[int, Tensor(N, 3)] — per-tooth lower arch clouds
    - upper_fdi: List[int] — FDI numbers for upper teeth
    - lower_fdi: List[int] — FDI numbers for lower teeth
    - gt_metrics: Dict[str, float] — ground truth occlusal metrics (if available)
    - has_supervision: bool — whether ground truth is available
    """

    def __init__(
        self,
        cases: List[TrainingCase],
        augmentation: Optional[AugmentationConfig] = None,
        points_per_tooth: int = POINTS_PER_TOOTH,
    ) -> None:
        self.augmentation = augmentation
        self.points_per_tooth = points_per_tooth

        # Include any case that has IOS data
        self.cases = [c for c in cases if c.has_ios()]
        logger.info(
            "DentalDataset initialized with %d cases (filtered from %d)",
            len(self.cases), len(cases),
        )

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        case = self.cases[idx]

        upper_teeth, upper_fdi = self._load_arch(
            case.ios_upper_arch_path, is_upper=True,
        )
        lower_teeth, lower_fdi = self._load_arch(
            case.ios_lower_arch_path, is_upper=False,
        )

        gt_metrics = case.ground_truth_occlusal_metrics or {}
        has_supervision = len(gt_metrics) > 0

        # Apply augmentations
        if self.augmentation is not None:
            if upper_teeth:
                upper_teeth = augment_tooth_clouds(upper_teeth, self.augmentation)
                upper_fdi = sorted(upper_teeth.keys())
            if lower_teeth:
                lower_teeth = augment_tooth_clouds(lower_teeth, self.augmentation)
                lower_fdi = sorted(lower_teeth.keys())

        return {
            "case_id": case.case_id,
            "upper_teeth": upper_teeth,
            "lower_teeth": lower_teeth,
            "upper_fdi": upper_fdi,
            "lower_fdi": lower_fdi,
            "gt_metrics": gt_metrics,
            "has_supervision": has_supervision,
        }

    def _load_arch(
        self,
        arch_path: Optional[Path],
        is_upper: bool,
    ) -> Tuple[Dict[int, torch.Tensor], List[int]]:
        """Load per-tooth point clouds from an IOS mesh file."""
        if arch_path is None or not arch_path.exists():
            return {}, []

        try:
            import trimesh
            mesh = trimesh.load(str(arch_path))
            vertices = np.asarray(mesh.vertices)
        except Exception as exc:
            logger.warning("Failed to load IOS mesh %s: %s", arch_path, exc)
            return {}, []

        tooth_clouds: Dict[int, torch.Tensor] = {}

        # Use metadata labels if available
        if hasattr(mesh, "metadata") and mesh.metadata and "tooth_labels" in mesh.metadata:
            for fdi_str, indices in mesh.metadata["tooth_labels"].items():
                fdi = int(fdi_str)
                pts = vertices[indices]
                pts = self._normalize_points(pts)
                tooth_clouds[fdi] = torch.tensor(pts, dtype=torch.float32)
        else:
            # Geometric partition
            fdi_range = range(11, 29) if is_upper else range(31, 49)
            teeth = self._geometric_partition(vertices, list(fdi_range))
            for fdi, pts in teeth.items():
                pts = self._normalize_points(pts)
                tooth_clouds[fdi] = torch.tensor(pts, dtype=torch.float32)

        fdi_numbers = sorted(tooth_clouds.keys())
        return tooth_clouds, fdi_numbers

    def _normalize_points(self, pts: np.ndarray) -> np.ndarray:
        """Subsample or pad points to target count."""
        n = pts.shape[0]
        target = self.points_per_tooth
        if n == 0:
            return np.zeros((target, 3), dtype=np.float32)
        if n >= target:
            idx = np.random.choice(n, target, replace=False)
            return pts[idx].astype(np.float32)
        pad_idx = np.random.choice(n, target - n, replace=True)
        return np.concatenate([pts, pts[pad_idx]], axis=0).astype(np.float32)

    @staticmethod
    def _geometric_partition(
        vertices: np.ndarray,
        fdi_range: List[int],
    ) -> Dict[int, np.ndarray]:
        """Partition arch vertices into pseudo-tooth segments."""
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

        is_upper = fdi_range[0] < 30
        right_mask = lateral_proj >= median_lateral
        left_mask = ~right_mask

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
            bounds = np.percentile(side_proj, np.linspace(0, 100, n_teeth + 1))
            for i, fdi in enumerate(fdis):
                seg_mask = (
                    (side_proj >= bounds[i])
                    if i == n_teeth - 1
                    else ((side_proj >= bounds[i]) & (side_proj < bounds[i + 1]))
                )
                pts = side_verts[seg_mask]
                if len(pts) > 10:
                    result[fdi] = pts
        return result

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate for DataLoader."""
        return {
            "case_ids": [b["case_id"] for b in batch],
            "upper_teeth": [b["upper_teeth"] for b in batch],
            "lower_teeth": [b["lower_teeth"] for b in batch],
            "upper_fdi": [b["upper_fdi"] for b in batch],
            "lower_fdi": [b["lower_fdi"] for b in batch],
            "gt_metrics": [b["gt_metrics"] for b in batch],
            "has_supervision": [b["has_supervision"] for b in batch],
        }

    @staticmethod
    def from_json(
        manifest_path: Path,
        augmentation: Optional[AugmentationConfig] = None,
        **kwargs: Any,
    ) -> "DentalDataset":
        """Load dataset from a JSON manifest file."""
        with open(manifest_path) as f:
            data = json.load(f)
        cases = [TrainingCase(**c) for c in data["cases"]]
        return DentalDataset(cases=cases, augmentation=augmentation, **kwargs)
