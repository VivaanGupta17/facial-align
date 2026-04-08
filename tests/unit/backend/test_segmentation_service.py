"""
Unit tests for the segmentation service and model registry.
Uses mock models to test service logic without GPU/trained models.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.core.exceptions import (
    InferenceError,
    LowConfidenceSegmentationError,
    ModelNotAvailableError,
)
from app.services.segmentation.segmentation_service import (
    ModelRegistry,
    SegmentationOutput,
    SegmentationService,
    TotalSegmentatorAdapter,
    CustomCMFModelAdapter,
)


class TestSegmentationService:
    """Tests for SegmentationService."""

    @pytest.fixture
    def service(self, mock_model_registry):
        return SegmentationService(model_registry=mock_model_registry)

    @pytest.mark.asyncio
    async def test_segment_structures_returns_output(
        self, service, small_ct_volume, ct_spacing
    ):
        """Test that segment_structures returns a SegmentationOutput."""
        # Inject bone structures into the volume
        volume = small_ct_volume.copy()
        volume[20:44, 20:44, 20:44] = 700.0

        output = await service.segment_structures(
            volume=volume,
            spacing=ct_spacing,
            model_name="mock_model",
        )

        assert isinstance(output, SegmentationOutput)
        assert output.masks is not None
        assert output.masks.shape == volume.shape
        assert isinstance(output.labels, dict)
        assert isinstance(output.confidences, dict)
        assert output.model_name == "mock_model"

    @pytest.mark.asyncio
    async def test_segment_structures_validates_input_dimensions(
        self, service, ct_spacing
    ):
        """Test that 2D volumes are rejected."""
        flat_volume = np.zeros((64, 64), dtype=np.float32)
        with pytest.raises(InferenceError, match="3D"):
            await service.segment_structures(flat_volume, ct_spacing)

    @pytest.mark.asyncio
    async def test_segment_structures_validates_empty_volume(
        self, service, ct_spacing
    ):
        """Test that all-zero volumes are rejected."""
        empty_volume = np.zeros((64, 64, 64), dtype=np.float32)
        with pytest.raises(InferenceError, match="empty"):
            await service.segment_structures(empty_volume, ct_spacing)

    @pytest.mark.asyncio
    async def test_cleanup_removes_small_components(self, service):
        """Test that cleanup removes isolated voxels."""
        masks = np.zeros((64, 64, 64), dtype=np.int32)
        # Main structure
        masks[20:40, 20:40, 20:40] = 1
        # Isolated noise (single voxel)
        masks[5, 5, 5] = 1

        labels = {"mandible": 1}
        cleaned_masks, cleaned_labels = service._cleanup_segmentation(masks, labels)

        # The isolated voxel should be removed
        assert cleaned_masks[5, 5, 5] == 0
        # Main structure should remain
        assert "mandible" in cleaned_labels

    def test_compute_volume_stats(self, service):
        """Test volume statistics computation."""
        masks = np.zeros((10, 10, 10), dtype=np.int32)
        masks[2:8, 2:8, 2:8] = 1  # 6x6x6 = 216 voxels

        labels = {"structure_A": 1}
        spacing = (1.0, 1.0, 1.0)  # 1mm isotropic

        stats = service._compute_volume_stats(masks, labels, spacing)

        assert "structure_A" in stats
        assert stats["structure_A"]["voxel_count"] == 216
        assert abs(stats["structure_A"]["volume_mm3"] - 216.0) < 1.0
        assert abs(stats["structure_A"]["volume_cc"] - 0.216) < 0.01


class TestModelRegistry:
    """Tests for ModelRegistry."""

    @pytest.fixture
    def registry(self, tmp_path):
        from app.core.config import ModelRegistrySettings
        settings = MagicMock(spec=ModelRegistrySettings)
        settings.registry_path = tmp_path
        settings.default_device = "cpu"
        settings.inference_fp16 = False
        settings.cmf_segmentation_model_path = None
        settings.dental_segmentation_model_path = None
        settings.totalsegmentator_task = "total"
        return ModelRegistry(settings)

    def test_registry_initializes(self, registry):
        """Test registry initializes with correct device."""
        assert registry._device in ("cpu", "cuda")

    def test_get_available_models_includes_totalsegmentator(self, registry):
        """TotalSegmentator should always be listed."""
        models = registry.get_available_models()
        assert "totalsegmentator" in models

    def test_load_unknown_model_raises(self, registry):
        """Loading an unknown model should raise ModelNotAvailableError."""
        with pytest.raises(ModelNotAvailableError):
            registry.load_model("nonexistent_model_xyz")

    def test_model_cached_on_second_load(self, registry):
        """Loaded models should be cached."""
        # Create a mock model and put it in the cache
        mock_model = MagicMock()
        mock_model.name = "test_model"
        mock_model.version = "1.0"
        registry._loaded_models["test_model"] = mock_model

        result = registry.load_model("test_model")
        assert result is mock_model


class TestTotalSegmentatorAdapter:
    """Tests for TotalSegmentatorAdapter."""

    def test_adapter_name(self):
        """Test model name property."""
        adapter = TotalSegmentatorAdapter.__new__(TotalSegmentatorAdapter)
        adapter._device = "cpu"
        adapter._version = "2.0"
        assert adapter.name == "totalsegmentator"

    def test_estimate_confidences_single_component(self):
        """Test confidence estimation for well-formed single-component structure."""
        adapter = TotalSegmentatorAdapter.__new__(TotalSegmentatorAdapter)
        masks = np.zeros((64, 64, 64), dtype=np.int32)
        masks[20:44, 20:44, 20:44] = 1  # Single solid component

        labels = {"mandible": 1}
        with patch("scipy.ndimage.label", return_value=(masks, 1)):
            confidences = adapter._estimate_confidences(masks, labels)

        assert "mandible" in confidences
        assert 0.0 <= confidences["mandible"] <= 1.0

    def test_estimate_confidences_empty_structure(self):
        """Test confidence is 0.0 for empty structure."""
        adapter = TotalSegmentatorAdapter.__new__(TotalSegmentatorAdapter)
        masks = np.zeros((64, 64, 64), dtype=np.int32)
        labels = {"missing_structure": 99}

        confidences = adapter._estimate_confidences(masks, labels)
        assert confidences["missing_structure"] == 0.0


class TestCustomCMFModelAdapter:
    """Tests for CustomCMFModelAdapter (placeholder model)."""

    def test_adapter_name(self, tmp_path):
        """Test model name property."""
        (tmp_path / "model.pth").touch()
        adapter = CustomCMFModelAdapter(model_path=tmp_path / "model.pth", device="cpu")
        assert adapter.name == "cmf_custom"

    def test_label_map_has_required_structures(self, tmp_path):
        """Test that all key CMF structures are in the label map."""
        (tmp_path / "model.pth").touch()
        adapter = CustomCMFModelAdapter(model_path=tmp_path / "model.pth")
        required = ["mandible", "maxilla", "zygoma_L", "zygoma_R"]
        for structure in required:
            assert structure in adapter.LABEL_MAP, f"{structure} not in label map"

    def test_preprocess_clips_hu_values(self, tmp_path, small_ct_volume, ct_spacing):
        """Test that HU values are clipped to expected range during preprocessing."""
        (tmp_path / "model.pth").touch()
        adapter = CustomCMFModelAdapter(model_path=tmp_path / "model.pth", device="cpu", fp16=False)

        try:
            import torch
            # Inject extreme HU values
            extreme_volume = small_ct_volume.copy()
            extreme_volume[0, 0, 0] = -5000.0  # Below min clip
            extreme_volume[1, 0, 0] = 10000.0  # Above max clip

            tensor = adapter._preprocess(extreme_volume, ct_spacing)
            data = tensor.cpu().float().numpy()

            # All values should be in [0, 1] after normalization
            assert data.min() >= 0.0 - 1e-5
            assert data.max() <= 1.0 + 1e-5
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_resample_volume_changes_size(self, tmp_path, small_ct_volume):
        """Test that resampling changes the volume dimensions."""
        (tmp_path / "model.pth").touch()
        adapter = CustomCMFModelAdapter(model_path=tmp_path / "model.pth")

        try:
            from scipy.ndimage import zoom
            resampled = adapter._resample_volume(
                small_ct_volume,
                current_spacing=(1.0, 1.0, 1.0),
                target_spacing=(0.5, 0.5, 0.5),
            )
            # Halving spacing doubles the voxel count per dimension
            assert resampled.shape[0] == pytest.approx(small_ct_volume.shape[0] * 2, abs=2)
        except ImportError:
            pytest.skip("scipy not installed")
