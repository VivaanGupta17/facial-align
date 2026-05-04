"""
Dental segmentation service for per-tooth segmentation from CBCT.
Uses FDI (Fédération Dentaire Internationale) numbering system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.exceptions import InferenceError, ModelLoadError, ModelNotAvailableError
from app.core.logging import MLInferenceLogger, TimedOperation, get_logger

logger = get_logger(__name__)
inference_logger = MLInferenceLogger(logger)


# FDI tooth numbering: quadrant (1-4) × position (1-8)
# Quadrant 1: upper right (11-18)
# Quadrant 2: upper left (21-28)
# Quadrant 3: lower left (31-38)
# Quadrant 4: lower right (41-48)

FDI_UPPER_TEETH = list(range(11, 19)) + list(range(21, 29))  # 11-18, 21-28
FDI_LOWER_TEETH = list(range(31, 39)) + list(range(41, 49))  # 31-38, 41-48
FDI_ALL_TEETH = FDI_UPPER_TEETH + FDI_LOWER_TEETH

FDI_TOOTH_NAMES = {
    11: "UR_central_incisor",   12: "UR_lateral_incisor",   13: "UR_canine",
    14: "UR_first_premolar",    15: "UR_second_premolar",
    16: "UR_first_molar",       17: "UR_second_molar",       18: "UR_third_molar",
    21: "UL_central_incisor",   22: "UL_lateral_incisor",   23: "UL_canine",
    24: "UL_first_premolar",    25: "UL_second_premolar",
    26: "UL_first_molar",       27: "UL_second_molar",       28: "UL_third_molar",
    31: "LL_central_incisor",   32: "LL_lateral_incisor",   33: "LL_canine",
    34: "LL_first_premolar",    35: "LL_second_premolar",
    36: "LL_first_molar",       37: "LL_second_molar",       38: "LL_third_molar",
    41: "LR_central_incisor",   42: "LR_lateral_incisor",   43: "LR_canine",
    44: "LR_first_premolar",    45: "LR_second_premolar",
    46: "LR_first_molar",       47: "LR_second_molar",       48: "LR_third_molar",
}


@dataclass
class ToothSegmentation:
    """Segmentation output for a single tooth."""
    fdi_number: int
    tooth_name: str
    mask: np.ndarray  # Binary 3D mask
    centroid_mm: List[float]  # [x, y, z] in patient coordinates
    bounding_box_mm: Dict[str, float]  # min/max extents in mm
    volume_mm3: float
    confidence: float
    is_missing: bool = False
    is_impacted: bool = False
    has_restoration: bool = False  # Detectable metal restoration


@dataclass
class DentalSegmentationOutput:
    """Complete dental segmentation output."""
    tooth_masks: Dict[int, np.ndarray]  # FDI number -> binary mask
    tooth_segmentations: List[ToothSegmentation]
    upper_arch_mask: Optional[np.ndarray] = None  # Merged upper arch mask
    lower_arch_mask: Optional[np.ndarray] = None  # Merged lower arch mask
    present_teeth: List[int] = field(default_factory=list)  # FDI numbers of detected teeth
    missing_teeth: List[int] = field(default_factory=list)
    spacing_mm: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    model_name: str = "dental_segmentator"
    model_version: str = "1.0"
    inference_time_ms: int = 0


class DentalSegmentationService:
    """
    ML service for per-tooth segmentation from CBCT volumes.

    Wraps a tooth segmentation model (architecture: 3D U-Net with FDI label heads).
    The model produces per-voxel tooth instance labels using FDI numbering.

    Reference implementation target:
    - Model: DentalSegmentator (Cui et al.) or similar instance segmentation approach
    - Input: CBCT volume at 0.3-0.5mm isotropic spacing
    - Output: Per-tooth instance masks with FDI labels
    - Training: CBCT datasets with per-tooth annotation

    TODO:
    1. Acquire training dataset (CBCT + per-tooth annotations in FDI format)
    2. Train 3D instance segmentation model (Mask R-CNN 3D or Point Transformer)
    3. Set model_path in ModelRegistrySettings.dental_segmentation_model_path
    4. Implement _load_model() with actual architecture
    """

    # Input spacing for dental segmentation (CBCT resolution)
    TARGET_SPACING_MM = (0.3, 0.3, 0.3)

    # HU window for tooth/bone
    HU_CLIP_MIN = 0.0
    HU_CLIP_MAX = 3000.0

    # Minimum tooth volume to consider valid
    MIN_TOOTH_VOLUME_MM3 = 50.0

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: str = "cuda",
    ) -> None:
        self._model_path = model_path
        self._device = device
        self._model: Optional[Any] = None

    def _load_model(self) -> None:
        """Load the dental segmentation model weights."""
        if self._model is not None:
            return

        if not self._model_path or not self._model_path.exists():
            raise ModelNotAvailableError(
                "Dental segmentation model weights not found",
                context={"model_path": str(self._model_path)},
            )

        try:
            import torch
            # TODO: Import actual dental model architecture
            # from models.dental_unet import DentalInstanceSegmenter
            # checkpoint = torch.load(self._model_path, map_location=self._device)
            # self._model = DentalInstanceSegmenter(num_teeth=32)
            # self._model.load_state_dict(checkpoint["model_state_dict"])
            # self._model.eval().to(self._device)
            raise NotImplementedError(
                "Dental segmentation model not yet trained. "
                "See class docstring for training roadmap."
            )
        except NotImplementedError:
            raise ModelLoadError(
                "Dental segmentation model not trained yet",
                context={"model_path": str(self._model_path)},
            )

    async def segment_teeth(
        self,
        volume: np.ndarray,
        spacing: Tuple[float, float, float],
        arch: str = "both",  # "upper", "lower", "both"
    ) -> DentalSegmentationOutput:
        """
        Segment individual teeth from a CBCT volume.

        Args:
            volume: 3D CBCT volume (Z, Y, X) in Hounsfield Units
            spacing: Voxel spacing (x_mm, y_mm, z_mm)
            arch: Which arch(es) to segment: "upper", "lower", "both"

        Returns:
            DentalSegmentationOutput with per-tooth masks and metrics
        """
        import time

        with TimedOperation(logger, "dental_segmentation", arch=arch):
            # Attempt to load model; unavailable models must surface explicitly.
            self._load_model()

            start_time = time.perf_counter()

            # Preprocess
            preprocessed = self._preprocess_cbct(volume, spacing)

            # Inference
            try:
                import torch
                with torch.no_grad():
                    raw_output = self._model(preprocessed)
            except Exception as exc:
                raise InferenceError(
                    f"Dental segmentation inference failed: {exc}",
                    cause=exc,
                )

            # Postprocess: convert raw output to per-tooth masks
            tooth_masks, tooth_segmentations = self._postprocess_tooth_predictions(
                raw_output, volume.shape, spacing, arch
            )

            inference_time_ms = int((time.perf_counter() - start_time) * 1000)

            # Build arch-level masks
            upper_arch = self._extract_dental_arch(
                tooth_masks,
                [fdi for fdi in FDI_UPPER_TEETH if fdi in tooth_masks]
            )
            lower_arch = self._extract_dental_arch(
                tooth_masks,
                [fdi for fdi in FDI_LOWER_TEETH if fdi in tooth_masks]
            )

            present = list(tooth_masks.keys())
            missing = [t for t in FDI_ALL_TEETH if t not in present]

            logger.info(
                "dental_segmentation_complete",
                teeth_detected=len(present),
                teeth_missing=len(missing),
                inference_time_ms=inference_time_ms,
            )

            return DentalSegmentationOutput(
                tooth_masks=tooth_masks,
                tooth_segmentations=tooth_segmentations,
                upper_arch_mask=upper_arch,
                lower_arch_mask=lower_arch,
                present_teeth=present,
                missing_teeth=missing,
                spacing_mm=spacing,
                model_version=getattr(self._model, "version", "1.0"),
                inference_time_ms=inference_time_ms,
            )

    def _preprocess_cbct(
        self, volume: np.ndarray, spacing: Tuple[float, float, float]
    ) -> "torch.Tensor":
        """Preprocess CBCT volume for dental segmentation inference."""
        import torch
        from scipy.ndimage import zoom

        # Resample to target spacing
        factors = tuple(c / t for c, t in zip(spacing, self.TARGET_SPACING_MM))
        zoom_factors = (factors[2], factors[1], factors[0])
        resampled = zoom(volume, zoom_factors, order=1)

        # Clip and normalize
        clipped = np.clip(resampled, self.HU_CLIP_MIN, self.HU_CLIP_MAX)
        normalized = clipped / self.HU_CLIP_MAX  # [0, 1]

        # (1, 1, Z, Y, X)
        tensor = torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        return tensor.to(self._device)

    def _postprocess_tooth_predictions(
        self,
        raw_output: Any,
        original_shape: Tuple[int, ...],
        spacing: Tuple[float, float, float],
        arch: str,
    ) -> Tuple[Dict[int, np.ndarray], List[ToothSegmentation]]:
        """
        Convert raw model output to per-tooth binary masks.

        TODO: Implement based on actual model output format.
        Expected output format: (batch=1, n_classes=33, Z, Y, X) logits
        where class 0 = background, classes 1-32 = FDI teeth 11-18,21-28,31-38,41-48
        """
        import torch
        tooth_masks: Dict[int, np.ndarray] = {}
        tooth_segmentations: List[ToothSegmentation] = []

        # Filter teeth by arch
        teeth_to_process = []
        if arch in ("upper", "both"):
            teeth_to_process.extend(FDI_UPPER_TEETH)
        if arch in ("lower", "both"):
            teeth_to_process.extend(FDI_LOWER_TEETH)

        # TODO: Parse actual model output
        # probs = torch.softmax(raw_output, dim=1).squeeze(0).cpu().numpy()
        # for idx, fdi_num in enumerate(FDI_ALL_TEETH, start=1):
        #     tooth_prob = probs[idx]  # (Z, Y, X)
        #     tooth_mask = tooth_prob > 0.5
        #     if tooth_mask.any():
        #         tooth_masks[fdi_num] = tooth_mask
        #         tooth_segmentations.append(self._build_tooth_segmentation(
        #             fdi_num, tooth_mask, spacing, float(tooth_prob[tooth_mask].mean())
        #         ))

        return tooth_masks, tooth_segmentations

    def _build_tooth_segmentation(
        self,
        fdi_number: int,
        mask: np.ndarray,
        spacing: Tuple[float, float, float],
        confidence: float,
    ) -> ToothSegmentation:
        """Build a ToothSegmentation dataclass from a binary mask."""
        voxel_volume = spacing[0] * spacing[1] * spacing[2]
        volume_mm3 = float(np.sum(mask)) * voxel_volume

        coords = np.argwhere(mask)
        centroid_zyx = np.mean(coords, axis=0)
        centroid_mm = [
            float(centroid_zyx[2]) * spacing[0],
            float(centroid_zyx[1]) * spacing[1],
            float(centroid_zyx[0]) * spacing[2],
        ]

        bbox = {
            "min_x": float(coords[:, 2].min()) * spacing[0],
            "max_x": float(coords[:, 2].max()) * spacing[0],
            "min_y": float(coords[:, 1].min()) * spacing[1],
            "max_y": float(coords[:, 1].max()) * spacing[1],
            "min_z": float(coords[:, 0].min()) * spacing[2],
            "max_z": float(coords[:, 0].max()) * spacing[2],
        }

        return ToothSegmentation(
            fdi_number=fdi_number,
            tooth_name=FDI_TOOTH_NAMES.get(fdi_number, f"tooth_{fdi_number}"),
            mask=mask,
            centroid_mm=centroid_mm,
            bounding_box_mm=bbox,
            volume_mm3=volume_mm3,
            confidence=confidence,
        )

    def extract_dental_arch(
        self,
        tooth_masks: Dict[int, np.ndarray],
        fdi_numbers: Optional[List[int]] = None,
    ) -> Optional[np.ndarray]:
        """
        Merge individual tooth masks into a single arch mesh.

        Args:
            tooth_masks: Per-tooth binary masks
            fdi_numbers: Specific teeth to include (None = all present)

        Returns:
            Merged binary arch mask
        """
        return self._extract_dental_arch(tooth_masks, fdi_numbers)

    def _extract_dental_arch(
        self,
        tooth_masks: Dict[int, np.ndarray],
        fdi_numbers: Optional[List[int]] = None,
    ) -> Optional[np.ndarray]:
        """Merge tooth masks into arch mask."""
        if not tooth_masks:
            return None

        teeth = fdi_numbers if fdi_numbers else list(tooth_masks.keys())
        arch_mask: Optional[np.ndarray] = None

        for fdi in teeth:
            if fdi in tooth_masks:
                if arch_mask is None:
                    arch_mask = tooth_masks[fdi].copy()
                else:
                    arch_mask = arch_mask | tooth_masks[fdi]

        return arch_mask

    def _create_placeholder_output(
        self,
        volume: np.ndarray,
        spacing: Tuple[float, float, float],
    ) -> DentalSegmentationOutput:
        """Return empty output when model is not available."""
        return DentalSegmentationOutput(
            tooth_masks={},
            tooth_segmentations=[],
            present_teeth=[],
            missing_teeth=FDI_ALL_TEETH,
            spacing_mm=spacing,
            model_name="dental_segmentator_placeholder",
            model_version="0.0",
            inference_time_ms=0,
        )
