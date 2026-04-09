"""
Model Registry — Centralized ML model management for Facial Align.

The registry manages model loading, versioning, and lifecycle. All inference
services load models through this registry rather than directly from disk.

Architecture:
    ModelRegistry
      ├── SegmentationModels (TotalSegmentator, DentalSegmentator, custom nnU-Net)
      ├── ReductionModels (ICP baseline, learned SE(3) transformer)
      ├── RegistrationModels (ICP, learned registration)
      └── OcclusionModels (geometric, learned prediction)
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ModelStatus(str, Enum):
    AVAILABLE = "available"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"
    NOT_FOUND = "not_found"


class ModelType(str, Enum):
    SEGMENTATION = "segmentation"
    DENTAL_SEGMENTATION = "dental_segmentation"
    REDUCTION = "reduction"
    REGISTRATION = "registration"
    OCCLUSION = "occlusion"
    LANDMARK = "landmark"
    # Supervised-learning-first model types
    SUPERVISED_REDUCTION = "supervised_reduction"
    SUPERVISED_OCCLUSION = "supervised_occlusion"


@dataclass
class ModelVersion:
    """Metadata for a specific model version."""

    name: str
    version: str
    model_type: ModelType
    architecture: str
    checkpoint_path: Optional[str] = None
    config_path: Optional[str] = None
    input_spec: dict = field(default_factory=dict)  # Expected input shape, dtype, spacing
    output_spec: dict = field(default_factory=dict)  # Output shape, labels, etc.
    metrics: dict = field(default_factory=dict)  # Validation metrics
    training_data: Optional[str] = None
    license: Optional[str] = None
    status: ModelStatus = ModelStatus.AVAILABLE
    loaded_at: Optional[float] = None
    device: Optional[str] = None


class InferenceModel(ABC):
    """Abstract base class for all inference models."""

    @abstractmethod
    def predict(self, input_data: np.ndarray, **kwargs) -> dict[str, Any]:
        """Run inference on input data."""
        ...

    @abstractmethod
    def get_info(self) -> ModelVersion:
        """Return model metadata."""
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether model weights are loaded and ready for inference."""
        ...

    def preprocess(self, input_data: np.ndarray, spacing: tuple[float, ...]) -> np.ndarray:
        """Default preprocessing — override in subclasses."""
        return input_data

    def postprocess(self, raw_output: Any) -> dict[str, Any]:
        """Default postprocessing — override in subclasses."""
        return {"output": raw_output}


class ModelRegistry:
    """
    Central registry for all ML models used in Facial Align.

    Handles model discovery, loading, version management, and lifecycle.
    Models are loaded lazily (on first inference request) unless preloaded.

    Usage:
        registry = ModelRegistry(model_dir="/app/models")
        model = registry.load("totalsegmentator", version="2.5.0")
        result = model.predict(volume, spacing=(1.0, 1.0, 1.0))
    """

    def __init__(self, model_dir: str = "/app/models", device: str = "cpu"):
        self.model_dir = Path(model_dir)
        self.device = device
        self._models: dict[str, dict[str, InferenceModel]] = {}
        self._versions: dict[str, list[ModelVersion]] = {}
        self._scan_available_models()

    def _scan_available_models(self):
        """Scan model directory for available models and their versions."""
        if not self.model_dir.exists():
            logger.warning(f"Model directory {self.model_dir} does not exist")
            return

        for model_dir in self.model_dir.iterdir():
            if model_dir.is_dir():
                manifest_path = model_dir / "manifest.json"
                if manifest_path.exists():
                    with open(manifest_path) as f:
                        manifest = json.load(f)
                    version = ModelVersion(**manifest)
                    self._register_version(version)

        logger.info(
            f"Registry initialized: {len(self._versions)} models, "
            f"{sum(len(v) for v in self._versions.values())} total versions"
        )

    def _register_version(self, version: ModelVersion):
        """Register a model version in the registry."""
        if version.name not in self._versions:
            self._versions[version.name] = []
        self._versions[version.name].append(version)
        logger.info(f"Registered model: {version.name} v{version.version} ({version.model_type})")

    def load(self, model_name: str, version: str = "latest") -> InferenceModel:
        """
        Load a model by name and version.

        Args:
            model_name: Model identifier
            version: Version string or "latest"

        Returns:
            InferenceModel ready for prediction
        """
        # Check if already loaded
        cache_key = f"{model_name}:{version}"
        if model_name in self._models and version in self._models[model_name]:
            return self._models[model_name][version]

        # Find version info
        model_version = self._resolve_version(model_name, version)
        if model_version is None:
            raise ValueError(f"Model {model_name} version {version} not found in registry")

        # Load model based on type
        logger.info(f"Loading {model_name} v{model_version.version} on {self.device}...")
        start_time = time.time()

        model = self._instantiate_model(model_version)

        elapsed = time.time() - start_time
        model_version.loaded_at = time.time()
        model_version.status = ModelStatus.LOADED
        model_version.device = self.device

        logger.info(f"Model loaded in {elapsed:.1f}s")

        # Cache
        if model_name not in self._models:
            self._models[model_name] = {}
        self._models[model_name][version] = model

        return model

    def _resolve_version(self, model_name: str, version: str) -> Optional[ModelVersion]:
        """Resolve a version string to a ModelVersion object."""
        if model_name not in self._versions:
            return None
        versions = self._versions[model_name]
        if version == "latest":
            return versions[-1] if versions else None
        return next((v for v in versions if v.version == version), None)

    def _instantiate_model(self, version: ModelVersion) -> InferenceModel:
        """Create an InferenceModel instance based on model type."""
        # Import adapters based on model type
        if version.model_type == ModelType.SEGMENTATION:
            if "totalsegmentator" in version.name:
                from services.inference.adapters.totalsegmentator_adapter import (
                    TotalSegmentatorModel,
                )
                return TotalSegmentatorModel(version, device=self.device)
            else:
                from services.inference.adapters.nnunet_adapter import NNUNetModel
                return NNUNetModel(version, device=self.device)

        elif version.model_type == ModelType.DENTAL_SEGMENTATION:
            from services.inference.adapters.dental_adapter import DentalSegmentatorModel
            return DentalSegmentatorModel(version, device=self.device)

        elif version.model_type == ModelType.REDUCTION:
            from services.inference.adapters.reduction_adapter import ReductionModel
            return ReductionModel(version, device=self.device)

        elif version.model_type == ModelType.REGISTRATION:
            from services.inference.adapters.registration_adapter import RegistrationModel
            return RegistrationModel(version, device=self.device)

        elif version.model_type in (
            ModelType.SUPERVISED_REDUCTION,
            ModelType.SUPERVISED_OCCLUSION,
        ):
            from services.inference.adapters.supervised_adapter import (
                SupervisedReductionModel,
            )
            return SupervisedReductionModel(version, device=self.device)

        else:
            raise ValueError(f"Unknown model type: {version.model_type}")

    def list_models(self) -> dict[str, list[ModelVersion]]:
        """List all registered models and their versions."""
        return dict(self._versions)

    def get_status(self) -> dict:
        """Get registry status including loaded models."""
        loaded = []
        for name, versions in self._models.items():
            for ver, model in versions.items():
                loaded.append({
                    "name": name,
                    "version": ver,
                    "loaded": model.is_loaded,
                    "device": self.device,
                })
        return {
            "model_dir": str(self.model_dir),
            "device": self.device,
            "registered_models": len(self._versions),
            "loaded_models": len(loaded),
            "models": loaded,
        }

    def unload(self, model_name: str, version: str = "latest"):
        """Unload a model to free GPU memory."""
        if model_name in self._models and version in self._models[model_name]:
            del self._models[model_name][version]
            logger.info(f"Unloaded {model_name} v{version}")

    def preload(self, model_names: list[str]):
        """Preload specified models at startup."""
        for name in model_names:
            try:
                self.load(name)
            except Exception as e:
                logger.error(f"Failed to preload {name}: {e}")
