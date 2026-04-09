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
                fragments: List of fragment data dicts for the supervised model.
                case_id: str, case identifier for confidence gate.
                plan_id: str, plan identifier for confidence gate.

        Returns:
            Dict with:
            - "fragment_transforms": Per-fragment SE(3) predictions
            - "occlusion_metrics": Clinical metric predictions
            - "confidence_level": Routing decision ("accept"/"review"/"fallback"/"reject")
            - "overall_confidence": Aggregate confidence [0, 1]
            - "per_fragment_confidence": Per-fragment confidence scores
            - "route_to_fallback": Whether to fall back to optimization
            - "requires_surgeon_review": Whether surgeon review is needed
            - "reject_reasons": Reasons if prediction was rejected
            - "inference_time_ms": Elapsed time
            - "decision": Full ClinicalDecision object
            - "plan": Raw ReductionPlan from the model
        """
        self._ensure_loaded()

        from app.services.postprocessing.confidence_gate import (
            ConfidenceGate,
            DecisionType,
        )

        spacing = kwargs.get("spacing")
        ios_tooth_meshes = kwargs.get("ios_tooth_meshes")
        fragments = kwargs.get("fragments", [])

        plan = self._service.predict(
            fragments=fragments,
            ct_volume=input_data,
            ct_spacing=spacing,
            tooth_meshes=ios_tooth_meshes,
        )

        # Build confidence data for the gate
        gate = ConfidenceGate()
        fragment_data = [
            {
                "fragment_id": fid,
                "confidence": conf,
                "rotation_uncertainty_deg": 0.0,  # Not available from plan
                "translation_uncertainty_mm": 0.0,
            }
            for fid, conf in plan.fragment_confidences.items()
        ]
        prediction_confidence = gate.build_prediction_confidence(
            case_id=kwargs.get("case_id", ""),
            plan_id=kwargs.get("plan_id", ""),
            model_version=plan.model_version or "",
            fragment_data=fragment_data,
        )
        decision = gate.evaluate(prediction_confidence)

        return {
            "fragment_transforms": plan.fragment_transforms,
            "occlusion_metrics": plan.occlusal_metrics,
            "confidence_level": decision.decision.value,
            "overall_confidence": plan.overall_confidence,
            "per_fragment_confidence": plan.fragment_confidences,
            "route_to_fallback": decision.decision == DecisionType.FALLBACK,
            "requires_surgeon_review": decision.decision == DecisionType.REVIEW,
            "reject_reasons": decision.reasons if decision.decision == DecisionType.REJECT else [],
            "inference_time_ms": plan.generation_time_ms,
            "decision": decision,
            "plan": plan,
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
