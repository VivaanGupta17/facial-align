"""
Unit tests for the ModelRegistry and related classes.

Tests cover:
- ModelRegistry initialization with non-existent directory
- ModelRegistry.list_models returns empty dict when no models
- ModelRegistry.load raises ValueError for unknown model
- ModelVersion data class fields
- ModelType enum values
- ModelStatus transitions
- InferenceModel ABC cannot be instantiated
- Registry caching behavior (load same model twice returns same instance)
"""

from __future__ import annotations

import json
import sys
import tempfile
from abc import ABC
from dataclasses import fields
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from services.inference.model_registry import (
    InferenceModel,
    ModelRegistry,
    ModelStatus,
    ModelType,
    ModelVersion,
)


# ─── ModelType enum tests ─────────────────────────────────────────────────────


class TestModelType:
    def test_segmentation_value(self):
        assert ModelType.SEGMENTATION == "segmentation"

    def test_dental_segmentation_value(self):
        assert ModelType.DENTAL_SEGMENTATION == "dental_segmentation"

    def test_reduction_value(self):
        assert ModelType.REDUCTION == "reduction"

    def test_registration_value(self):
        assert ModelType.REGISTRATION == "registration"

    def test_occlusion_value(self):
        assert ModelType.OCCLUSION == "occlusion"

    def test_landmark_value(self):
        assert ModelType.LANDMARK == "landmark"

    def test_all_types_are_strings(self):
        for mt in ModelType:
            assert isinstance(mt.value, str)

    def test_model_type_count(self):
        """Ensure all expected model types are present."""
        expected = {"segmentation", "dental_segmentation", "reduction",
                    "registration", "occlusion", "landmark"}
        actual = {mt.value for mt in ModelType}
        assert actual == expected


# ─── ModelStatus enum tests ───────────────────────────────────────────────────


class TestModelStatus:
    def test_available_status(self):
        assert ModelStatus.AVAILABLE == "available"

    def test_loading_status(self):
        assert ModelStatus.LOADING == "loading"

    def test_loaded_status(self):
        assert ModelStatus.LOADED == "loaded"

    def test_error_status(self):
        assert ModelStatus.ERROR == "error"

    def test_not_found_status(self):
        assert ModelStatus.NOT_FOUND == "not_found"

    def test_status_transitions_sequence(self):
        """Verify the expected lifecycle of a model status."""
        version = ModelVersion(
            name="test_model",
            version="1.0.0",
            model_type=ModelType.SEGMENTATION,
            architecture="unet",
        )
        # Default status is AVAILABLE
        assert version.status == ModelStatus.AVAILABLE

        # Simulate loading
        version.status = ModelStatus.LOADING
        assert version.status == ModelStatus.LOADING

        # Simulate loaded
        version.status = ModelStatus.LOADED
        assert version.status == ModelStatus.LOADED

        # Simulate error
        version.status = ModelStatus.ERROR
        assert version.status == ModelStatus.ERROR

    def test_all_statuses_are_strings(self):
        for ms in ModelStatus:
            assert isinstance(ms.value, str)


# ─── ModelVersion dataclass tests ────────────────────────────────────────────


class TestModelVersion:
    def test_required_fields(self):
        version = ModelVersion(
            name="totalsegmentator",
            version="2.5.0",
            model_type=ModelType.SEGMENTATION,
            architecture="residual_encoder_unet",
        )
        assert version.name == "totalsegmentator"
        assert version.version == "2.5.0"
        assert version.model_type == ModelType.SEGMENTATION
        assert version.architecture == "residual_encoder_unet"

    def test_default_status_is_available(self):
        version = ModelVersion(
            name="test",
            version="1.0",
            model_type=ModelType.REDUCTION,
            architecture="pointnet",
        )
        assert version.status == ModelStatus.AVAILABLE

    def test_default_checkpoint_path_is_none(self):
        version = ModelVersion(
            name="test",
            version="1.0",
            model_type=ModelType.REGISTRATION,
            architecture="icp",
        )
        assert version.checkpoint_path is None

    def test_default_config_path_is_none(self):
        version = ModelVersion(
            name="test",
            version="1.0",
            model_type=ModelType.REGISTRATION,
            architecture="icp",
        )
        assert version.config_path is None

    def test_default_input_spec_is_empty_dict(self):
        version = ModelVersion(
            name="test",
            version="1.0",
            model_type=ModelType.SEGMENTATION,
            architecture="unet",
        )
        assert version.input_spec == {}

    def test_default_output_spec_is_empty_dict(self):
        version = ModelVersion(
            name="test",
            version="1.0",
            model_type=ModelType.SEGMENTATION,
            architecture="unet",
        )
        assert version.output_spec == {}

    def test_default_metrics_is_empty_dict(self):
        version = ModelVersion(
            name="test",
            version="1.0",
            model_type=ModelType.SEGMENTATION,
            architecture="unet",
        )
        assert version.metrics == {}

    def test_default_loaded_at_is_none(self):
        version = ModelVersion(
            name="test",
            version="1.0",
            model_type=ModelType.OCCLUSION,
            architecture="geometric",
        )
        assert version.loaded_at is None

    def test_default_device_is_none(self):
        version = ModelVersion(
            name="test",
            version="1.0",
            model_type=ModelType.LANDMARK,
            architecture="cnn",
        )
        assert version.device is None

    def test_custom_input_spec(self):
        version = ModelVersion(
            name="segmentor",
            version="1.0",
            model_type=ModelType.SEGMENTATION,
            architecture="unet",
            input_spec={"shape": [1, 512, 512, 128], "dtype": "float32", "spacing": [1.0, 1.0, 1.0]},
        )
        assert version.input_spec["dtype"] == "float32"

    def test_all_dataclass_fields_present(self):
        field_names = {f.name for f in fields(ModelVersion)}
        expected_fields = {
            "name", "version", "model_type", "architecture",
            "checkpoint_path", "config_path", "input_spec", "output_spec",
            "metrics", "training_data", "license", "status", "loaded_at", "device",
        }
        assert expected_fields.issubset(field_names)

    def test_status_can_be_updated(self):
        version = ModelVersion(
            name="test",
            version="1.0",
            model_type=ModelType.REDUCTION,
            architecture="pointnet",
        )
        version.status = ModelStatus.LOADED
        version.loaded_at = 1700000000.0
        version.device = "cuda:0"

        assert version.status == ModelStatus.LOADED
        assert version.loaded_at is not None
        assert version.device == "cuda:0"


# ─── InferenceModel ABC tests ─────────────────────────────────────────────────


class TestInferenceModelABC:
    def test_cannot_instantiate_directly(self):
        """InferenceModel is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            InferenceModel()

    def test_is_abstract_base_class(self):
        assert issubclass(InferenceModel, ABC)

    def test_predict_is_abstract(self):
        """predict must be overridden in subclasses."""
        assert hasattr(InferenceModel, "predict")
        assert getattr(InferenceModel.predict, "__isabstractmethod__", False)

    def test_get_info_is_abstract(self):
        assert hasattr(InferenceModel, "get_info")
        assert getattr(InferenceModel.get_info, "__isabstractmethod__", False)

    def test_is_loaded_is_abstract(self):
        assert hasattr(InferenceModel, "is_loaded")

    def test_concrete_subclass_can_be_instantiated(self):
        """A concrete subclass that implements all abstract methods can be created."""
        class ConcreteModel(InferenceModel):
            def predict(self, input_data: np.ndarray, **kwargs) -> dict[str, Any]:
                return {"output": input_data}

            def get_info(self) -> ModelVersion:
                return ModelVersion(
                    name="concrete",
                    version="1.0",
                    model_type=ModelType.SEGMENTATION,
                    architecture="test",
                )

            @property
            def is_loaded(self) -> bool:
                return True

        model = ConcreteModel()
        assert model.is_loaded is True

    def test_default_preprocess_returns_input_unchanged(self):
        """Default preprocess() implementation returns input as-is."""
        class MinimalModel(InferenceModel):
            def predict(self, input_data: np.ndarray, **kwargs) -> dict[str, Any]:
                return {}
            def get_info(self) -> ModelVersion:
                return ModelVersion(name="m", version="1.0", model_type=ModelType.SEGMENTATION, architecture="x")
            @property
            def is_loaded(self) -> bool:
                return False

        model = MinimalModel()
        arr = np.ones((10, 10), dtype=np.float32)
        result = model.preprocess(arr, (1.0, 1.0))
        assert np.array_equal(result, arr)

    def test_default_postprocess_wraps_in_output_key(self):
        """Default postprocess() implementation wraps result in {'output': ...}."""
        class MinimalModel(InferenceModel):
            def predict(self, input_data: np.ndarray, **kwargs) -> dict[str, Any]:
                return {}
            def get_info(self) -> ModelVersion:
                return ModelVersion(name="m", version="1.0", model_type=ModelType.SEGMENTATION, architecture="x")
            @property
            def is_loaded(self) -> bool:
                return False

        model = MinimalModel()
        result = model.postprocess("raw_output")
        assert result == {"output": "raw_output"}


# ─── ModelRegistry initialization tests ──────────────────────────────────────


class TestModelRegistryInit:
    def test_init_with_nonexistent_directory(self):
        """Registry initializes without error even if model_dir does not exist."""
        registry = ModelRegistry(model_dir="/nonexistent/path/to/models")
        assert registry is not None

    def test_model_dir_stored(self):
        registry = ModelRegistry(model_dir="/some/model/dir")
        assert str(registry.model_dir) == "/some/model/dir"

    def test_device_stored(self):
        registry = ModelRegistry(model_dir="/models", device="cpu")
        assert registry.device == "cpu"

    def test_default_device_is_cpu(self):
        registry = ModelRegistry(model_dir="/nonexistent")
        assert registry.device == "cpu"

    def test_list_models_empty_when_no_directory(self):
        """list_models returns empty dict when model directory doesn't exist."""
        registry = ModelRegistry(model_dir="/nonexistent/dir")
        models = registry.list_models()
        assert models == {}

    def test_list_models_empty_when_directory_empty(self, tmp_path):
        """list_models returns empty dict when model directory is empty."""
        registry = ModelRegistry(model_dir=str(tmp_path))
        models = registry.list_models()
        assert models == {}

    def test_list_models_returns_dict(self, tmp_path):
        registry = ModelRegistry(model_dir=str(tmp_path))
        result = registry.list_models()
        assert isinstance(result, dict)


# ─── ModelRegistry.load tests ────────────────────────────────────────────────


class TestModelRegistryLoad:
    def test_load_raises_value_error_for_unknown_model(self, tmp_path):
        """load() raises ValueError when model name is not registered."""
        registry = ModelRegistry(model_dir=str(tmp_path))
        with pytest.raises(ValueError, match="not found"):
            registry.load("nonexistent_model")

    def test_load_raises_value_error_for_unknown_version(self, tmp_path):
        """load() raises ValueError when specific version is not registered."""
        registry = ModelRegistry(model_dir=str(tmp_path))
        with pytest.raises(ValueError):
            registry.load("totalsegmentator", version="999.0.0")

    def test_load_registered_model_succeeds(self, tmp_path):
        """load() succeeds when a model is manually registered."""
        registry = ModelRegistry(model_dir=str(tmp_path))

        version = ModelVersion(
            name="test_seg",
            version="1.0.0",
            model_type=ModelType.SEGMENTATION,
            architecture="unet",
        )
        registry._register_version(version)

        # Mock _instantiate_model to avoid loading actual weights
        mock_model = MagicMock()
        mock_model.is_loaded = True
        registry._instantiate_model = MagicMock(return_value=mock_model)

        loaded = registry.load("test_seg", version="1.0.0")
        assert loaded is mock_model

    def test_load_returns_cached_instance_on_second_call(self, tmp_path):
        """load() returns the same instance when called twice for the same model/version."""
        registry = ModelRegistry(model_dir=str(tmp_path))

        version = ModelVersion(
            name="cached_model",
            version="2.0.0",
            model_type=ModelType.REDUCTION,
            architecture="pointnet",
        )
        registry._register_version(version)

        mock_model = MagicMock()
        mock_model.is_loaded = True
        registry._instantiate_model = MagicMock(return_value=mock_model)

        first_load = registry.load("cached_model", version="2.0.0")
        second_load = registry.load("cached_model", version="2.0.0")

        # Same object identity
        assert first_load is second_load

    def test_instantiate_model_called_only_once(self, tmp_path):
        """_instantiate_model is called only once; second load uses cache."""
        registry = ModelRegistry(model_dir=str(tmp_path))

        version = ModelVersion(
            name="once_model",
            version="1.0.0",
            model_type=ModelType.REGISTRATION,
            architecture="icp",
        )
        registry._register_version(version)

        mock_model = MagicMock()
        instantiate_mock = MagicMock(return_value=mock_model)
        registry._instantiate_model = instantiate_mock

        registry.load("once_model", version="1.0.0")
        registry.load("once_model", version="1.0.0")

        instantiate_mock.assert_called_once()

    def test_load_latest_uses_last_registered_version(self, tmp_path):
        """load('model', version='latest') picks the last registered version."""
        registry = ModelRegistry(model_dir=str(tmp_path))

        v1 = ModelVersion(name="versioned", version="1.0", model_type=ModelType.SEGMENTATION, architecture="unet")
        v2 = ModelVersion(name="versioned", version="2.0", model_type=ModelType.SEGMENTATION, architecture="unet_v2")
        registry._register_version(v1)
        registry._register_version(v2)

        resolved = registry._resolve_version("versioned", "latest")
        assert resolved.version == "2.0"

    def test_resolve_version_returns_none_for_missing_model(self, tmp_path):
        """_resolve_version returns None when model is not registered."""
        registry = ModelRegistry(model_dir=str(tmp_path))
        result = registry._resolve_version("does_not_exist", "1.0.0")
        assert result is None

    def test_resolve_specific_version(self, tmp_path):
        """_resolve_version finds the correct version by string."""
        registry = ModelRegistry(model_dir=str(tmp_path))
        v = ModelVersion(name="seg_model", version="3.1.4", model_type=ModelType.SEGMENTATION, architecture="unet")
        registry._register_version(v)

        resolved = registry._resolve_version("seg_model", "3.1.4")
        assert resolved is v

    def test_resolve_wrong_version_returns_none(self, tmp_path):
        """_resolve_version returns None for a version string that doesn't match."""
        registry = ModelRegistry(model_dir=str(tmp_path))
        v = ModelVersion(name="seg", version="1.0", model_type=ModelType.SEGMENTATION, architecture="unet")
        registry._register_version(v)

        resolved = registry._resolve_version("seg", "9.9.9")
        assert resolved is None


# ─── ModelRegistry scanning tests ────────────────────────────────────────────


class TestModelRegistryScanning:
    def test_scans_model_directory_with_manifest(self, tmp_path):
        """Registry discovers models from manifest.json files."""
        model_dir = tmp_path / "models" / "totalsegmentator"
        model_dir.mkdir(parents=True)

        manifest = {
            "name": "totalsegmentator",
            "version": "2.5.0",
            "model_type": "segmentation",
            "architecture": "residual_encoder_unet",
        }
        (model_dir / "manifest.json").write_text(json.dumps(manifest))

        registry = ModelRegistry(model_dir=str(tmp_path / "models"))
        models = registry.list_models()

        assert "totalsegmentator" in models
        assert len(models["totalsegmentator"]) == 1

    def test_multiple_models_discovered(self, tmp_path):
        """Registry discovers multiple models from separate directories."""
        for name in ["model_a", "model_b"]:
            model_dir = tmp_path / name
            model_dir.mkdir()
            manifest = {
                "name": name,
                "version": "1.0",
                "model_type": "segmentation",
                "architecture": "unet",
            }
            (model_dir / "manifest.json").write_text(json.dumps(manifest))

        registry = ModelRegistry(model_dir=str(tmp_path))
        models = registry.list_models()

        assert "model_a" in models
        assert "model_b" in models

    def test_directory_without_manifest_is_skipped(self, tmp_path):
        """Directories without manifest.json are silently skipped."""
        no_manifest_dir = tmp_path / "empty_model"
        no_manifest_dir.mkdir()

        registry = ModelRegistry(model_dir=str(tmp_path))
        models = registry.list_models()

        assert "empty_model" not in models

    def test_registered_version_has_correct_type(self, tmp_path):
        """All discovered versions should be ModelVersion instances."""
        model_dir = tmp_path / "seg"
        model_dir.mkdir()
        (model_dir / "manifest.json").write_text(json.dumps({
            "name": "seg",
            "version": "1.0",
            "model_type": "segmentation",
            "architecture": "unet",
        }))

        registry = ModelRegistry(model_dir=str(tmp_path))
        versions = registry.list_models()["seg"]
        for v in versions:
            assert isinstance(v, ModelVersion)


# ─── ModelRegistry.get_status tests ──────────────────────────────────────────


class TestModelRegistryStatus:
    def test_status_contains_model_dir(self, tmp_path):
        registry = ModelRegistry(model_dir=str(tmp_path))
        status = registry.get_status()
        assert "model_dir" in status

    def test_status_contains_device(self, tmp_path):
        registry = ModelRegistry(model_dir=str(tmp_path), device="cpu")
        status = registry.get_status()
        assert status["device"] == "cpu"

    def test_status_loaded_models_initially_zero(self, tmp_path):
        registry = ModelRegistry(model_dir=str(tmp_path))
        status = registry.get_status()
        assert status["loaded_models"] == 0

    def test_status_registered_models_count(self, tmp_path):
        model_dir = tmp_path / "seg"
        model_dir.mkdir()
        (model_dir / "manifest.json").write_text(json.dumps({
            "name": "seg",
            "version": "1.0",
            "model_type": "segmentation",
            "architecture": "unet",
        }))
        registry = ModelRegistry(model_dir=str(tmp_path))
        status = registry.get_status()
        assert status["registered_models"] >= 1


# ─── ModelRegistry.unload tests ──────────────────────────────────────────────


class TestModelRegistryUnload:
    def test_unload_removes_model_from_cache(self, tmp_path):
        """After unload(), the model is removed from the internal cache."""
        registry = ModelRegistry(model_dir=str(tmp_path))

        version = ModelVersion(
            name="to_unload",
            version="1.0",
            model_type=ModelType.SEGMENTATION,
            architecture="unet",
        )
        registry._register_version(version)

        mock_model = MagicMock()
        registry._instantiate_model = MagicMock(return_value=mock_model)

        # Load, then unload
        registry.load("to_unload", version="1.0")
        assert "to_unload" in registry._models

        registry.unload("to_unload", version="1.0")
        assert "1.0" not in registry._models.get("to_unload", {})

    def test_unload_nonexistent_model_is_harmless(self, tmp_path):
        """Calling unload() on a model that isn't loaded does not raise."""
        registry = ModelRegistry(model_dir=str(tmp_path))
        # Should not raise
        registry.unload("nonexistent", version="1.0")
