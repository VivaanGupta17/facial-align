"""
DentalSegmentator Model Adapter

Wraps the DentalSegmentator model for per-tooth segmentation from CBCT/CT.
Model weights from Zenodo (Apache-2.0), trained on 470 CT+CBCT scans
from 7 institutions, validated with Dice 0.922–0.942.

Reference: DCBIA-OrthoLab/SlicerAutomatedDentalTools
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np

from services.inference.model_registry import InferenceModel, ModelVersion

logger = logging.getLogger(__name__)

# FDI tooth numbering system
FDI_TEETH = {
    # Upper right quadrant (1)
    11: "Upper right central incisor",
    12: "Upper right lateral incisor",
    13: "Upper right canine",
    14: "Upper right first premolar",
    15: "Upper right second premolar",
    16: "Upper right first molar",
    17: "Upper right second molar",
    18: "Upper right third molar",
    # Upper left quadrant (2)
    21: "Upper left central incisor",
    22: "Upper left lateral incisor",
    23: "Upper left canine",
    24: "Upper left first premolar",
    25: "Upper left second premolar",
    26: "Upper left first molar",
    27: "Upper left second molar",
    28: "Upper left third molar",
    # Lower left quadrant (3)
    31: "Lower left central incisor",
    32: "Lower left lateral incisor",
    33: "Lower left canine",
    34: "Lower left first premolar",
    35: "Lower left second premolar",
    36: "Lower left first molar",
    37: "Lower left second molar",
    38: "Lower left third molar",
    # Lower right quadrant (4)
    41: "Lower right central incisor",
    42: "Lower right lateral incisor",
    43: "Lower right canine",
    44: "Lower right first premolar",
    45: "Lower right second premolar",
    46: "Lower right first molar",
    47: "Lower right second molar",
    48: "Lower right third molar",
}


class DentalSegmentatorModel(InferenceModel):
    """
    Adapter for DentalSegmentator within the Facial Align pipeline.

    Provides per-tooth instance segmentation with FDI numbering from CT/CBCT.
    This is complementary to TotalSegmentator's teeth subtask — DentalSegmentator
    provides better per-tooth boundary accuracy and is more robust to metal artifacts.

    Model weights: https://zenodo.org/records/11003568
    """

    def __init__(self, version: ModelVersion, device: str = "cpu"):
        self._version = version
        self._device = device
        self._model = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def get_info(self) -> ModelVersion:
        return self._version

    def load(self, weights_path: str):
        """
        Load DentalSegmentator model weights.

        Args:
            weights_path: Path to downloaded model weights directory
        """
        try:
            import torch

            # TODO: Implement actual model loading from Zenodo weights
            # The DentalSegmentator uses a nnU-Net-based architecture
            # Weights are distributed as a nnU-Net model folder
            logger.info(f"Loading DentalSegmentator weights from {weights_path}")

            # Placeholder: In Phase 1, we wrap TotalSegmentator's teeth subtask
            # In Phase 2, we'll load the actual DentalSegmentator weights
            self._loaded = True
            logger.info("DentalSegmentator ready (using TotalSegmentator teeth fallback)")

        except Exception as e:
            logger.error(f"Failed to load DentalSegmentator: {e}")
            raise

    def predict(
        self,
        input_data: np.ndarray,
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
        **kwargs,
    ) -> dict[str, Any]:
        """
        Segment individual teeth from CT/CBCT volume.

        Args:
            input_data: 3D numpy array (Z, Y, X) in HU
            spacing: Voxel spacing in mm

        Returns:
            dict with:
                - tooth_masks: dict[str, np.ndarray] — per-tooth binary masks
                - tooth_labels: dict[str, int] — FDI number to label ID
                - confidences: dict[str, float] — per-tooth confidence
                - teeth_present: list[int] — FDI numbers of detected teeth
                - teeth_missing: list[int] — FDI numbers of expected but not found teeth
                - dental_arch_upper: np.ndarray — combined upper arch mask
                - dental_arch_lower: np.ndarray — combined lower arch mask
        """
        start_time = time.time()

        # Phase 1: Use TotalSegmentator teeth subtask as fallback
        # Phase 2: Use actual DentalSegmentator inference
        result = self._segment_with_totalsegmentator_fallback(input_data, spacing)

        inference_time = int((time.time() - start_time) * 1000)
        result["inference_time_ms"] = inference_time
        result["model_name"] = "dental_segmentator"
        result["model_version"] = self._version.version

        logger.info(
            f"Dental segmentation: {len(result['teeth_present'])} teeth "
            f"detected in {inference_time}ms"
        )

        return result

    def _segment_with_totalsegmentator_fallback(
        self,
        input_data: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> dict[str, Any]:
        """
        Fallback: Use TotalSegmentator's teeth subtask.

        This provides per-tooth segmentation without requiring separate
        DentalSegmentator weights. Accuracy is slightly lower than the
        dedicated dental model but sufficient for Phase 1.
        """
        from services.inference.adapters.totalsegmentator_adapter import (
            TotalSegmentatorModel,
        )
        from services.inference.model_registry import ModelVersion, ModelType

        # Create a TotalSegmentator instance for teeth subtask
        ts_version = ModelVersion(
            name="totalsegmentator_teeth",
            version="2.5.0",
            model_type=ModelType.DENTAL_SEGMENTATION,
            architecture="nnU-Net",
        )
        ts_model = TotalSegmentatorModel(ts_version, device=self._device)
        ts_result = ts_model.predict(input_data, spacing=spacing, subtask="teeth")

        # Reorganize into dental-specific output format
        tooth_masks = {}
        confidences = {}
        teeth_present = []
        upper_arch = np.zeros_like(input_data, dtype=np.uint8)
        lower_arch = np.zeros_like(input_data, dtype=np.uint8)

        for structure_name, mask in ts_result["masks"].items():
            # Parse FDI number from structure name
            if structure_name.startswith("tooth_"):
                try:
                    fdi_str = structure_name.replace("tooth_", "")
                    fdi_num = int(fdi_str)
                    if fdi_num in FDI_TEETH:
                        tooth_masks[f"FDI-{fdi_num}"] = mask
                        confidences[f"FDI-{fdi_num}"] = ts_result["confidences"].get(
                            structure_name, 0.85
                        )
                        teeth_present.append(fdi_num)

                        # Accumulate into arch masks
                        quadrant = fdi_num // 10
                        if quadrant in (1, 2):  # Upper
                            upper_arch = np.maximum(upper_arch, mask)
                        else:  # Lower
                            lower_arch = np.maximum(lower_arch, mask)
                except (ValueError, IndexError):
                    logger.debug(f"Could not parse FDI number from: {structure_name}")

        # Determine missing teeth (compare to full adult dentition)
        all_fdi = set(FDI_TEETH.keys())
        teeth_missing = sorted(all_fdi - set(teeth_present))

        return {
            "tooth_masks": tooth_masks,
            "tooth_labels": {f"FDI-{fdi}": i + 1 for i, fdi in enumerate(sorted(teeth_present))},
            "confidences": confidences,
            "teeth_present": sorted(teeth_present),
            "teeth_missing": teeth_missing,
            "dental_arch_upper": upper_arch,
            "dental_arch_lower": lower_arch,
        }

    def preprocess(self, input_data: np.ndarray, spacing: tuple[float, ...]) -> np.ndarray:
        """Dental-specific preprocessing: focus on dental ROI."""
        # TODO: Implement ROI cropping to dental region for faster inference
        # For now, process full volume
        return input_data
