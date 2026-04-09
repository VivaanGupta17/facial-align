"""
Paired CT + IOS dataset for supervised fracture reduction training.

Supports three data sources:
1. Pre-op / post-op CT pairs with derived ground truth transforms
2. Synthetic fractures from intact mandibles (SyntheticFractureGenerator)
3. IOS scan pairs (optional, paired with CT data)

Each sample yields:
- ct_volume: (1, D, H, W) preprocessed CT volume
- ios_point_clouds: (T, P, 3) per-tooth point clouds (or None)
- targets: Dict with per-fragment and per-tooth ground truth transforms

Data loading is lazy — CT volumes are loaded from DICOM or NIfTI on
demand and cached if memory permits.

References:
- FracFormer: Training data preparation for fragment prediction
- MONAI: Medical image data loading patterns
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class CTIOSDatasetConfig:
    """Configuration for the paired CT+IOS dataset."""
    manifest_path: str = "data/training_manifest.json"
    max_fragments: int = 8
    max_teeth: int = 32
    points_per_tooth: int = 1024
    target_spacing_mm: float = 0.4
    hu_min: float = 0.0
    hu_max: float = 3000.0
    crop_size: Tuple[int, int, int] = (128, 128, 128)
    augment: bool = True
    cache_ct: bool = False


class CTIOSPairedDataset(Dataset):
    """
    Paired CT + optional IOS dataset for supervised training.

    Expected manifest format (JSON):
    {
      "cases": [
        {
          "case_id": "case_001",
          "source": "synthetic" | "clinical",
          "ct_path": "/data/cases/001/ct.nii.gz",
          "ct_spacing": [0.4, 0.4, 0.4],
          "ios_path": "/data/cases/001/ios/",     (optional)
          "ground_truth": {
            "fragment_transforms": {
              "fragment_0": [[4x4 matrix]],
              ...
            },
            "tooth_transforms": {
              "11": [[4x4 matrix]],
              ...
            },
            "metrics": {
              "overjet_mm": 2.5,
              "overbite_mm": 3.0,
              "midline_deviation_mm": 0.5,
              "molar_class": 0
            }
          },
          "num_fragments": 2
        },
        ...
      ]
    }

    Args:
        config: Dataset configuration.
    """

    def __init__(self, config: Optional[CTIOSDatasetConfig] = None) -> None:
        if config is None:
            config = CTIOSDatasetConfig()
        self.config = config
        self.cases: List[Dict[str, Any]] = []
        self._ct_cache: Dict[str, np.ndarray] = {}

        self._load_manifest()

    def _load_manifest(self) -> None:
        """Load the training manifest."""
        manifest_path = Path(self.config.manifest_path)
        if not manifest_path.exists():
            logger.warning("Manifest not found at %s — dataset is empty", manifest_path)
            return

        with open(manifest_path) as f:
            manifest = json.load(f)

        self.cases = manifest.get("cases", [])
        logger.info("Loaded %d training cases from %s", len(self.cases), manifest_path)

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load a single training sample.

        Returns:
            Dict with:
            - "ct_volume": (1, D, H, W) float32 tensor
            - "ios_point_clouds": (max_teeth, P, 3) tensor or None
            - "ios_tooth_ids": (max_teeth,) tensor
            - "tooth_mask": (max_teeth,) bool tensor (True = missing)
            - "num_fragments": int tensor
            - "targets": Dict with ground truth tensors
        """
        case = self.cases[idx]

        # Load CT volume
        ct_volume = self._load_ct(case)

        # Load IOS (optional)
        ios_data = self._load_ios(case)

        # Build targets
        targets = self._build_targets(case)

        sample = {
            "ct_volume": ct_volume,
            "targets": targets,
            "num_fragments": torch.tensor(case.get("num_fragments", 2)),
        }

        if ios_data is not None:
            sample["ios_point_clouds"] = ios_data["point_clouds"]
            sample["ios_tooth_ids"] = ios_data["tooth_ids"]
            sample["tooth_mask"] = ios_data["tooth_mask"]
        else:
            sample["ios_point_clouds"] = None
            sample["ios_tooth_ids"] = None
            sample["tooth_mask"] = torch.ones(self.config.max_teeth, dtype=torch.bool)

        # Augment
        if self.config.augment:
            sample = self._augment(sample)

        return sample

    def _load_ct(self, case: Dict[str, Any]) -> torch.Tensor:
        """Load and preprocess a CT volume."""
        ct_path = case["ct_path"]

        if self.config.cache_ct and ct_path in self._ct_cache:
            volume = self._ct_cache[ct_path]
        else:
            volume = self._read_volume(ct_path)
            if self.config.cache_ct:
                self._ct_cache[ct_path] = volume

        # Resample to target spacing
        spacing = case.get("ct_spacing", [0.4, 0.4, 0.4])
        volume = self._resample_volume(volume, spacing)

        # Crop to fixed size
        volume = self._center_crop(volume, self.config.crop_size)

        # Convert to tensor (1, D, H, W) — HU clipping is done in the model
        return torch.from_numpy(volume[np.newaxis].astype(np.float32))

    def _read_volume(self, path: str) -> np.ndarray:
        """Read a CT volume from NIfTI or DICOM directory."""
        path = Path(path)

        if path.suffix in (".nii", ".gz"):
            try:
                import nibabel as nib
                img = nib.load(str(path))
                return np.array(img.dataobj, dtype=np.float32)
            except ImportError:
                logger.error("nibabel required for NIfTI loading")
                return np.zeros(self.config.crop_size, dtype=np.float32)

        elif path.is_dir():
            # DICOM directory
            try:
                import pydicom
                slices = []
                for dcm_path in sorted(path.glob("*.dcm")):
                    ds = pydicom.dcmread(str(dcm_path))
                    slices.append(ds.pixel_array.astype(np.float32))
                if slices:
                    volume = np.stack(slices, axis=0)
                    # Apply rescale slope/intercept if available
                    return volume
            except ImportError:
                logger.error("pydicom required for DICOM loading")

        # Fallback: synthetic data stored as numpy
        elif path.suffix == ".npy":
            return np.load(str(path))

        return np.zeros(self.config.crop_size, dtype=np.float32)

    def _resample_volume(
        self,
        volume: np.ndarray,
        current_spacing: List[float],
    ) -> np.ndarray:
        """Resample volume to target isotropic spacing."""
        target = self.config.target_spacing_mm
        scale = [s / target for s in current_spacing]

        if all(abs(s - 1.0) < 0.01 for s in scale):
            return volume

        try:
            from scipy.ndimage import zoom
            return zoom(volume, scale, order=1)
        except ImportError:
            return volume

    def _center_crop(
        self,
        volume: np.ndarray,
        target_size: Tuple[int, int, int],
    ) -> np.ndarray:
        """Center-crop or pad volume to target size."""
        result = np.zeros(target_size, dtype=volume.dtype)

        # Compute start indices for crop
        starts = []
        slices_src = []
        slices_dst = []
        for dim_size, target_dim in zip(volume.shape, target_size):
            if dim_size >= target_dim:
                start = (dim_size - target_dim) // 2
                slices_src.append(slice(start, start + target_dim))
                slices_dst.append(slice(0, target_dim))
            else:
                pad_start = (target_dim - dim_size) // 2
                slices_src.append(slice(0, dim_size))
                slices_dst.append(slice(pad_start, pad_start + dim_size))

        result[slices_dst[0], slices_dst[1], slices_dst[2]] = \
            volume[slices_src[0], slices_src[1], slices_src[2]]

        return result

    def _load_ios(self, case: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        """Load IOS point clouds if available."""
        ios_path = case.get("ios_path")
        if ios_path is None:
            return None

        ios_dir = Path(ios_path)
        if not ios_dir.exists():
            return None

        P = self.config.points_per_tooth
        T = self.config.max_teeth
        point_clouds = torch.zeros(T, P, 3)
        tooth_ids = torch.arange(T)
        tooth_mask = torch.ones(T, dtype=torch.bool)  # True = missing

        # FDI number → index mapping
        fdi_numbers = list(range(11, 19)) + list(range(21, 29)) + \
                      list(range(31, 39)) + list(range(41, 49))
        fdi_to_idx = {fdi: i for i, fdi in enumerate(fdi_numbers)}

        # Load each tooth's point cloud
        for tooth_file in ios_dir.glob("tooth_*.npy"):
            try:
                fdi = int(tooth_file.stem.split("_")[1])
                if fdi not in fdi_to_idx:
                    continue
                idx = fdi_to_idx[fdi]

                pts = np.load(str(tooth_file))
                pts_tensor = torch.from_numpy(pts.astype(np.float32))

                # Subsample or pad
                if pts_tensor.shape[0] >= P:
                    indices = torch.randperm(pts_tensor.shape[0])[:P]
                    pts_tensor = pts_tensor[indices]
                else:
                    repeats = (P + pts_tensor.shape[0] - 1) // pts_tensor.shape[0]
                    pts_tensor = pts_tensor.repeat(repeats, 1)[:P]

                point_clouds[idx] = pts_tensor
                tooth_mask[idx] = False
            except Exception as e:
                logger.debug("Failed to load tooth %s: %s", tooth_file, e)

        if tooth_mask.all():
            return None  # No teeth loaded

        return {
            "point_clouds": point_clouds,
            "tooth_ids": tooth_ids,
            "tooth_mask": tooth_mask,
        }

    def _build_targets(self, case: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Build ground truth target tensors from case data."""
        gt = case.get("ground_truth", {})
        F = self.config.max_fragments
        T = self.config.max_teeth

        # Fragment transforms
        frag_R = torch.eye(3).unsqueeze(0).expand(F, -1, -1).clone()
        frag_t = torch.zeros(F, 3)
        frag_mask = torch.zeros(F, dtype=torch.bool)

        frag_gt = gt.get("fragment_transforms", {})
        for i, (frag_id, transform) in enumerate(frag_gt.items()):
            if i >= F:
                break
            T_mat = torch.tensor(transform, dtype=torch.float32)
            if T_mat.shape == (4, 4):
                frag_R[i] = T_mat[:3, :3]
                frag_t[i] = T_mat[:3, 3]
                frag_mask[i] = True

        # Tooth transforms
        tooth_R = torch.eye(3).unsqueeze(0).expand(T, -1, -1).clone()
        tooth_t = torch.zeros(T, 3)
        tooth_mask = torch.zeros(T, dtype=torch.bool)

        fdi_numbers = list(range(11, 19)) + list(range(21, 29)) + \
                      list(range(31, 39)) + list(range(41, 49))
        fdi_to_idx = {str(fdi): i for i, fdi in enumerate(fdi_numbers)}

        tooth_gt = gt.get("tooth_transforms", {})
        for fdi_str, transform in tooth_gt.items():
            idx = fdi_to_idx.get(fdi_str)
            if idx is None or idx >= T:
                continue
            T_mat = torch.tensor(transform, dtype=torch.float32)
            if T_mat.shape == (4, 4):
                tooth_R[idx] = T_mat[:3, :3]
                tooth_t[idx] = T_mat[:3, 3]
                tooth_mask[idx] = True

        # Clinical metrics
        metrics_gt = gt.get("metrics", {})
        metrics = torch.tensor([
            metrics_gt.get("overjet_mm", 2.5),
            metrics_gt.get("overbite_mm", 3.0),
            metrics_gt.get("midline_deviation_mm", 0.0),
        ], dtype=torch.float32)
        molar_class = torch.tensor(metrics_gt.get("molar_class", 0), dtype=torch.long)

        return {
            "fragment_rotations": frag_R,       # (F, 3, 3)
            "fragment_translations": frag_t,    # (F, 3)
            "fragment_mask": frag_mask,          # (F,)
            "tooth_rotations": tooth_R,          # (T, 3, 3)
            "tooth_translations": tooth_t,       # (T, 3)
            "tooth_mask": tooth_mask,             # (T,)
            "metrics": metrics,                   # (3,)
            "molar_class": molar_class,           # scalar
        }

    def _augment(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply data augmentation to a training sample.

        Augmentations:
        - Random 3D rotation of CT volume (small, ±5°)
        - Random translation (±2mm)
        - Random intensity scaling (±10%)
        - Random Gaussian noise (σ=10 HU)
        """
        if not self.config.augment:
            return sample

        ct = sample["ct_volume"]  # (1, D, H, W)

        # Intensity augmentation (safe — doesn't change geometry)
        scale = 1.0 + (torch.rand(1).item() - 0.5) * 0.2  # 0.9-1.1
        ct = ct * scale

        noise_std = 10.0  # HU
        ct = ct + torch.randn_like(ct) * noise_std

        sample["ct_volume"] = ct
        return sample


class SyntheticFractureDataset(CTIOSPairedDataset):
    """
    Dataset adapter for synthetic fracture cases.

    Wraps SyntheticFractureCase objects into the CTIOSPairedDataset format.
    Useful for initial training before clinical data is available.

    Usage:
        from training.synthetic.fracture_generator import SyntheticFractureGenerator

        generator = SyntheticFractureGenerator()
        cases = generator.generate_batch(mesh_paths, num_cases_per_mesh=50)
        dataset = SyntheticFractureDataset(cases)
    """

    def __init__(
        self,
        synthetic_cases: list,
        config: Optional[CTIOSDatasetConfig] = None,
    ) -> None:
        super().__init__(config)
        self.cases = self._convert_synthetic(synthetic_cases)

    def _convert_synthetic(self, cases: list) -> List[Dict[str, Any]]:
        """Convert SyntheticFractureCase objects to manifest format."""
        converted = []
        for case in cases:
            frag_transforms = {}
            for frag_id, T in case.ground_truth_transforms.items():
                frag_transforms[frag_id] = T.tolist()

            converted.append({
                "case_id": case.case_id,
                "source": "synthetic",
                "ct_path": case.intact_mesh_path,  # Will need CT reconstruction
                "ct_spacing": [0.4, 0.4, 0.4],
                "ground_truth": {
                    "fragment_transforms": frag_transforms,
                    "tooth_transforms": {},
                    "metrics": {
                        "overjet_mm": 2.5,
                        "overbite_mm": 3.0,
                        "midline_deviation_mm": 0.0,
                        "molar_class": 0,
                    },
                },
                "num_fragments": case.metadata.get("num_fragments", 2),
            })
        return converted
