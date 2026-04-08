"""
TotalSegmentator Model Adapter

Wraps TotalSegmentator's Python API for use within the Facial Align
model registry and inference pipeline.

TotalSegmentator v2 provides segmentation for:
- craniofacial_structures: mandible, dental arches, skull, sinuses
- teeth: all 32 FDI-numbered teeth with pulp chambers
- head_muscles: masseter, temporalis, pterygoids
- headneck_bones_vessels: zygomatic arch, styloid, hyoid
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from services.inference.model_registry import InferenceModel, ModelStatus, ModelVersion

logger = logging.getLogger(__name__)

# Mapping of TotalSegmentator subtask outputs to Facial Align structure names
SUBTASK_STRUCTURE_MAP = {
    "craniofacial_structures": {
        "mandible": "mandible",
        "skull": "skull",
        "teeth_upper": "upper_dental_arch",
        "teeth_lower": "lower_dental_arch",
        "maxillary_sinus_right": "maxillary_sinus_right",
        "maxillary_sinus_left": "maxillary_sinus_left",
        "frontal_sinus": "frontal_sinus",
    },
    "teeth": {
        f"tooth_{q}{n}": f"tooth_{q}{n}"
        for q in [1, 2, 3, 4]
        for n in range(1, 9)
    },
    "head_muscles": {
        "masseter_right": "masseter_right",
        "masseter_left": "masseter_left",
        "temporalis_right": "temporalis_right",
        "temporalis_left": "temporalis_left",
        "medial_pterygoid_right": "medial_pterygoid_right",
        "medial_pterygoid_left": "medial_pterygoid_left",
        "lateral_pterygoid_right": "lateral_pterygoid_right",
        "lateral_pterygoid_left": "lateral_pterygoid_left",
    },
}


class TotalSegmentatorModel(InferenceModel):
    """
    Adapter for TotalSegmentator within the Facial Align inference pipeline.

    This adapter handles:
    1. Running TotalSegmentator's Python API
    2. Mapping outputs to Facial Align's structure naming convention
    3. Computing per-structure confidence scores
    4. Converting NIfTI masks to numpy arrays for downstream processing

    Usage:
        model = TotalSegmentatorModel(version_info, device="cuda:0")
        results = model.predict(volume_array, spacing=(1.0, 1.0, 1.0), subtask="craniofacial_structures")
    """

    def __init__(self, version: ModelVersion, device: str = "cpu"):
        self._version = version
        self._device = device
        self._loaded = False
        self._validate_installation()

    def _validate_installation(self):
        """Verify TotalSegmentator is installed and accessible."""
        try:
            import totalsegmentator  # noqa: F401
            self._loaded = True
            self._version.status = ModelStatus.LOADED
            logger.info("TotalSegmentator validated and ready")
        except ImportError:
            self._version.status = ModelStatus.ERROR
            logger.error(
                "TotalSegmentator not installed. "
                "Install: pip install totalsegmentator"
            )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def get_info(self) -> ModelVersion:
        return self._version

    def predict(
        self,
        input_data: np.ndarray,
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
        subtask: str = "craniofacial_structures",
        fast: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Run TotalSegmentator segmentation on a CT volume.

        Args:
            input_data: 3D numpy array (Z, Y, X) in Hounsfield Units
            spacing: Voxel spacing in mm (x, y, z)
            subtask: TotalSegmentator subtask
            fast: Use fast (lower resolution) mode

        Returns:
            dict with:
                - masks: dict[str, np.ndarray] — per-structure binary masks
                - labels: dict[str, int] — structure name to label ID mapping
                - confidences: dict[str, float] — per-structure confidence scores
                - combined_mask: np.ndarray — single mask with all labels
                - inference_time_ms: int
        """
        if not self._loaded:
            raise RuntimeError("TotalSegmentator not loaded")

        import SimpleITK as sitk
        from totalsegmentator.python_api import totalsegmentator

        start_time = time.time()

        # Write volume to temporary NIfTI
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.nii.gz"
            output_path = Path(tmpdir) / "segmentation"
            output_path.mkdir()

            # Create SimpleITK image from numpy array
            sitk_image = sitk.GetImageFromArray(input_data)
            sitk_image.SetSpacing(spacing)
            sitk.WriteImage(sitk_image, str(input_path))

            # Run TotalSegmentator
            logger.info(f"Running TotalSegmentator: subtask={subtask}, device={self._device}")
            totalsegmentator(
                input=input_path,
                output=output_path,
                task=subtask,
                fast=fast,
                device=self._device if self._device != "cpu" else "cpu",
                verbose=False,
            )

            # Parse outputs
            masks = {}
            labels = {}
            confidences = {}
            label_id = 1

            structure_map = SUBTASK_STRUCTURE_MAP.get(subtask, {})

            for nifti_file in sorted(output_path.glob("*.nii.gz")):
                raw_name = nifti_file.stem.replace(".nii", "")
                structure_name = structure_map.get(raw_name, raw_name)

                mask_sitk = sitk.ReadImage(str(nifti_file))
                mask_array = sitk.GetArrayFromImage(mask_sitk)
                binary_mask = (mask_array > 0).astype(np.uint8)

                if binary_mask.sum() > 0:
                    masks[structure_name] = binary_mask
                    labels[structure_name] = label_id
                    confidences[structure_name] = self._estimate_confidence(binary_mask)
                    label_id += 1

            # Build combined label mask
            combined = np.zeros_like(input_data, dtype=np.uint8)
            for name, mask in masks.items():
                combined[mask > 0] = labels[name]

        inference_time_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"Segmentation complete: {len(masks)} structures in {inference_time_ms}ms"
        )

        return {
            "masks": masks,
            "labels": labels,
            "confidences": confidences,
            "combined_mask": combined,
            "inference_time_ms": inference_time_ms,
            "model_name": "totalsegmentator",
            "model_version": self._version.version,
            "subtask": subtask,
            "device": self._device,
        }

    def _estimate_confidence(self, binary_mask: np.ndarray) -> float:
        """
        Estimate segmentation confidence for a structure.

        Uses mask morphological properties as a proxy for quality:
        - Connected components (should be 1 for most structures)
        - Surface regularity
        - Volume relative to expected range

        TODO: Replace with proper model softmax-based confidence in Phase 2.
        This is a heuristic placeholder.
        """
        from scipy import ndimage

        # Connected components — fewer is better
        labeled, num_components = ndimage.label(binary_mask)
        component_penalty = max(0, (num_components - 1) * 0.05)

        # Volume sanity check — very small volumes suggest poor segmentation
        volume_voxels = binary_mask.sum()
        volume_penalty = 0.0
        if volume_voxels < 100:
            volume_penalty = 0.3  # Very small — likely artifact

        # Base confidence from volume (larger structures segment more reliably)
        if volume_voxels > 10000:
            base_confidence = 0.95
        elif volume_voxels > 1000:
            base_confidence = 0.90
        else:
            base_confidence = 0.80

        confidence = max(0.0, min(1.0, base_confidence - component_penalty - volume_penalty))
        return round(confidence, 3)

    def preprocess(
        self, input_data: np.ndarray, spacing: tuple[float, ...]
    ) -> np.ndarray:
        """
        Preprocess CT volume for TotalSegmentator.

        TotalSegmentator handles its own preprocessing internally,
        but we ensure HU range is correct.
        """
        # Ensure HU values are in expected range
        # TotalSegmentator expects standard CT HU values (-1024 to ~3000)
        if input_data.min() > 0:
            logger.warning(
                f"Input min={input_data.min()} — may not be in HU scale. "
                "Ensure RescaleSlope/Intercept have been applied."
            )
        return input_data

    def get_available_subtasks(self) -> list[str]:
        """List available TotalSegmentator subtasks for CMF."""
        return list(SUBTASK_STRUCTURE_MAP.keys())
