"""
PyTorch Dataset for normal anatomy reference CTs.

Loads normal (healthy) CT scans to extract reference dental landmarks,
arch curves, and symmetry metrics. Used as prior/regularization during
training to encourage anatomically plausible predictions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from data_contracts.training.training_case import TrainingCase
from training.datasets.data_augmentation import (
    AugmentationConfig,
    augment_point_cloud,
)

logger = logging.getLogger(__name__)

POINTS_PER_ARCH = 8192


class NormalReferenceDataset(Dataset):
    """
    PyTorch Dataset for normal anatomy reference.

    Each sample returns:
    - arch_points: Tensor(N, 3) — dental arch surface points
    - landmarks: Dict[str, Tensor(3,)] — reference landmark positions
    - arch_curve: Tensor(T, 3) — tooth centroid trajectory (arch form)
    - symmetry_metrics: Dict[str, float] — bilateral symmetry measurements
    """

    def __init__(
        self,
        cases: List[TrainingCase],
        augmentation: Optional[AugmentationConfig] = None,
        points_per_arch: int = POINTS_PER_ARCH,
    ) -> None:
        self.augmentation = augmentation
        self.points_per_arch = points_per_arch

        # Filter to normal_reference cases
        self.cases = [c for c in cases if c.case_type == "normal_reference"]
        logger.info(
            "NormalReferenceDataset initialized with %d cases", len(self.cases),
        )

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        case = self.cases[idx]

        # Load normal CT and extract dental arch
        arch_points = self._load_arch_points(case)

        # Load or compute reference landmarks
        landmarks = self._load_landmarks(case)

        # Compute arch curve (centroid trajectory)
        arch_curve = self._compute_arch_curve(arch_points)

        # Compute symmetry metrics
        symmetry_metrics = self._compute_symmetry(arch_points)

        # Apply augmentations
        if self.augmentation is not None:
            arch_points = augment_point_cloud(arch_points, self.augmentation)

        return {
            "case_id": case.case_id,
            "arch_points": arch_points,
            "landmarks": landmarks,
            "arch_curve": arch_curve,
            "symmetry_metrics": symmetry_metrics,
        }

    def _load_arch_points(self, case: TrainingCase) -> torch.Tensor:
        """Load arch surface points from normal CT."""
        ct_path = case.normal_ct_path
        if ct_path is None or not ct_path.exists():
            return torch.zeros(self.points_per_arch, 3)

        # Try NPZ format first (pre-extracted)
        if ct_path.suffix == ".npz":
            data = np.load(str(ct_path), allow_pickle=True)
            if "arch_points" in data:
                pts = data["arch_points"]
                pts = self._normalize_count(pts, self.points_per_arch)
                return torch.tensor(pts, dtype=torch.float32)

        # Try NPY
        if ct_path.suffix == ".npy":
            pts = np.load(str(ct_path))
            pts = self._normalize_count(pts, self.points_per_arch)
            return torch.tensor(pts, dtype=torch.float32)

        # Try DICOM directory
        if ct_path.is_dir():
            return self._extract_arch_from_dicom(ct_path)

        return torch.zeros(self.points_per_arch, 3)

    def _extract_arch_from_dicom(self, dicom_path: Path) -> torch.Tensor:
        """Extract dental arch points from a DICOM CT."""
        try:
            import SimpleITK as sitk

            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_path))
            if not dicom_names:
                return torch.zeros(self.points_per_arch, 3)

            reader.SetFileNames(dicom_names)
            image = reader.Execute()

            volume = sitk.GetArrayFromImage(image)
            spacing = np.array(image.GetSpacing())
            origin = np.array(image.GetOrigin())

            # Dental HU range (enamel/dentin: 1000-3000 HU)
            dental_mask = (volume > 1000) & (volume < 3500)

            coords = np.argwhere(dental_mask)
            if len(coords) == 0:
                return torch.zeros(self.points_per_arch, 3)

            physical = coords[:, ::-1].astype(np.float64) * spacing + origin
            pts = self._normalize_count(physical, self.points_per_arch)
            return torch.tensor(pts, dtype=torch.float32)

        except Exception as exc:
            logger.warning("Normal CT DICOM extraction failed: %s", exc)
            return torch.zeros(self.points_per_arch, 3)

    def _load_landmarks(
        self, case: TrainingCase,
    ) -> Dict[str, torch.Tensor]:
        """Load reference landmark positions."""
        landmarks: Dict[str, torch.Tensor] = {}
        if case.ground_truth_landmarks is None:
            return landmarks

        for name, coords in case.ground_truth_landmarks.items():
            if len(coords) == 3:
                landmarks[name] = torch.tensor(coords, dtype=torch.float32)

        return landmarks

    @staticmethod
    def _compute_arch_curve(
        arch_points: torch.Tensor, n_segments: int = 16,
    ) -> torch.Tensor:
        """
        Compute the dental arch curve as a centroid trajectory.

        Partitions points along the principal arch direction into
        n_segments, computing the centroid of each.
        """
        if arch_points.shape[0] < n_segments:
            return torch.zeros(n_segments, 3)

        # PCA for arch direction
        centered = arch_points - arch_points.mean(dim=0, keepdim=True)
        try:
            _, _, Vh = torch.linalg.svd(centered[:1000], full_matrices=False)
            arch_dir = Vh[0]
        except Exception:
            arch_dir = torch.tensor([0.0, 1.0, 0.0])

        projections = arch_points @ arch_dir
        sorted_idx = projections.argsort()
        sorted_pts = arch_points[sorted_idx]

        # Split into segments and compute centroids
        chunk_size = len(sorted_pts) // n_segments
        centroids = []
        for i in range(n_segments):
            start = i * chunk_size
            end = start + chunk_size if i < n_segments - 1 else len(sorted_pts)
            centroids.append(sorted_pts[start:end].mean(dim=0))

        return torch.stack(centroids)

    @staticmethod
    def _compute_symmetry(arch_points: torch.Tensor) -> Dict[str, float]:
        """Compute bilateral symmetry metrics for the arch."""
        if arch_points.shape[0] < 100:
            return {"symmetry_score": 0.0, "midline_deviation_mm": 0.0}

        # PCA to find midline (lateral) axis
        centered = arch_points - arch_points.mean(dim=0, keepdim=True)
        try:
            _, _, Vh = torch.linalg.svd(centered[:1000], full_matrices=False)
            lateral_dir = Vh[1]
        except Exception:
            lateral_dir = torch.tensor([1.0, 0.0, 0.0])

        lateral_proj = arch_points @ lateral_dir
        median = lateral_proj.median()

        # Midline deviation: how far the median is from zero
        midline_dev = abs(float(median))

        # Symmetry: compare left/right point distributions
        right_pts = arch_points[lateral_proj >= median]
        left_pts = arch_points[lateral_proj < median]

        if len(right_pts) > 0 and len(left_pts) > 0:
            right_spread = right_pts.std(dim=0).norm().item()
            left_spread = left_pts.std(dim=0).norm().item()
            symmetry = 1.0 - abs(right_spread - left_spread) / (
                max(right_spread, left_spread) + 1e-6
            )
        else:
            symmetry = 0.0

        return {
            "symmetry_score": max(0.0, min(1.0, symmetry)),
            "midline_deviation_mm": midline_dev,
        }

    @staticmethod
    def _normalize_count(pts: np.ndarray, target: int) -> np.ndarray:
        """Subsample or pad points to target count."""
        n = pts.shape[0]
        if n == 0:
            return np.zeros((target, 3), dtype=np.float32)
        if n >= target:
            idx = np.random.choice(n, target, replace=False)
            return pts[idx].astype(np.float32)
        pad_idx = np.random.choice(n, target - n, replace=True)
        return np.concatenate([pts, pts[pad_idx]], axis=0).astype(np.float32)

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate for DataLoader."""
        return {
            "case_ids": [b["case_id"] for b in batch],
            "arch_points": torch.stack([b["arch_points"] for b in batch]),
            "landmarks": [b["landmarks"] for b in batch],
            "arch_curve": torch.stack([b["arch_curve"] for b in batch]),
            "symmetry_metrics": [b["symmetry_metrics"] for b in batch],
        }

    @staticmethod
    def from_json(
        manifest_path: Path,
        augmentation: Optional[AugmentationConfig] = None,
        **kwargs: Any,
    ) -> "NormalReferenceDataset":
        """Load dataset from a JSON manifest file."""
        with open(manifest_path) as f:
            data = json.load(f)
        cases = [TrainingCase(**c) for c in data["cases"]]
        return NormalReferenceDataset(
            cases=cases, augmentation=augmentation, **kwargs,
        )
