"""
Model registry adapter for the supervised fracture reduction model.

Wraps FacialAlignSupervisedModel + SupervisedInferenceService into the
InferenceModel interface expected by the ModelRegistry.

Usage via registry:
    registry = ModelRegistry(model_dir="/app/models", device="cuda")
    model = registry.load("supervised_reduction", version="latest")
    result = model.predict(ct_volume, spacing=(0.4, 0.4, 0.4))
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from services.inference.model_registry import (
    InferenceModel,
    ModelStatus,
    ModelVersion,
)

logger = logging.getLogger(__name__)


class SupervisedReductionModel(InferenceModel):
    """
    Registry adapter for the supervised fracture reduction model.

    Implements the InferenceModel interface so the supervised model can be
    managed by the ModelRegistry alongside segmentation, registration, and
    other model types.

    The adapter manages:
    - Lazy model loading from checkpoint
    - CT preprocessing (spacing resampling, HU clipping)
    - Optional IOS data packaging
    - Confidence-based result interpretation

    Args:
        version: ModelVersion metadata from the registry.
        device: Target device ("cuda" or "cpu").
    """

    def __init__(self, version: ModelVersion, device: str = "cpu") -> None:
        self._version = version
        self._device = device
        self._service = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Lazy-load the inference service on first use."""
        if self._loaded:
            return

        from app.services.supervised.inference_service import (
            InferenceConfig,
            SupervisedInferenceService,
        )

        config = InferenceConfig(
            checkpoint_path=self._version.checkpoint_path,
            device=self._device,
        )
        self._service = SupervisedInferenceService(config)
        self._loaded = True
        self._version.status = ModelStatus.LOADED

        logger.info(
            "SupervisedReductionModel loaded: %s v%s on %s",
            self._version.name, self._version.version, self._device,
        )

    def predict(self, input_data: np.ndarray, **kwargs) -> dict[str, Any]:
        """
        Run supervised inference.

        Args:
            input_data: (D, H, W) or (1, D, H, W) CT volume in Hounsfield units.
            **kwargs:
                spacing: (sz, sy, sx) voxel spacing in mm.
                ios_tooth_meshes: Dict[int, np.ndarray] mapping FDI → (P, 3) points.
                num_fragments: int, number of bone fragments.

        Returns:
            Dict with:
            - "fragment_transforms": Per-fragment SE(3) predictions
            - "tooth_transforms": Per-tooth SE(3) predictions
            - "occlusion_metrics": Clinical metric predictions
            - "confidence_level": Routing decision ("accept"/"review"/"fallback"/"reject")
            - "overall_confidence": Aggregate confidence [0, 1]
            - "inference_time_ms": Elapsed time
        """
        self._ensure_loaded()

        spacing = kwargs.get("spacing")
        ios_tooth_meshes = kwargs.get("ios_tooth_meshes")
        num_fragments = kwargs.get("num_fragments", 2)

        result = self._service.predict(
            ct_volume=input_data,
            ct_spacing=spacing,
            ios_tooth_meshes=ios_tooth_meshes,
            num_fragments=num_fragments,
        )

        return {
            "fragment_transforms": result.fragment_transforms,
            "tooth_transforms": result.tooth_transforms,
            "occlusion_metrics": result.occlusion_metrics,
            "confidence_level": result.confidence_level,
            "overall_confidence": result.overall_confidence,
            "per_fragment_confidence": result.per_fragment_confidence,
            "route_to_fallback": result.route_to_fallback,
            "requires_surgeon_review": result.requires_surgeon_review,
            "reject_reasons": result.reject_reasons,
            "inference_time_ms": result.inference_time_ms,
            "ios_available": result.ios_available,
        }

    def get_info(self) -> ModelVersion:
        """Return model metadata."""
        return self._version

    @property
    def is_loaded(self) -> bool:
        """Whether model weights are loaded."""
        return self._loaded

    def preprocess(self, input_data: np.ndarray, spacing: tuple[float, ...]) -> np.ndarray:
        """Preprocessing is handled internally by the inference service."""
        return input_data

    def postprocess(self, raw_output: Any) -> dict[str, Any]:
        """Post-processing is handled internally by the inference service."""
        return raw_output
