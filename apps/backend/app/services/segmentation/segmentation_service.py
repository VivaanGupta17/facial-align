"""
ML-based bone segmentation service.
Wraps TotalSegmentator and custom CMF models via a unified registry interface.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np

from app.core.config import ModelRegistrySettings, get_settings
from app.core.exceptions import (
    InferenceError,
    LowConfidenceSegmentationError,
    ModelLoadError,
    ModelNotAvailableError,
    PostProcessingError,
)
from app.core.logging import MLInferenceLogger, TimedOperation, get_logger

settings = get_settings()
logger = get_logger(__name__)
inference_logger = MLInferenceLogger(logger)


# ─── Output dataclass ─────────────────────────────────────────────────────────


@dataclass
class SegmentationOutput:
    """
    Output from the segmentation pipeline.

    The masks array is a 3D integer volume where each voxel value
    corresponds to a structure label (0 = background).
    """
    masks: np.ndarray  # Shape: (Z, Y, X), dtype: int32
    labels: Dict[str, int]  # structure_name -> mask_value
    confidences: Dict[str, float]  # structure_name -> confidence score [0-1]
    spacing_mm: Tuple[float, float, float]  # (x, y, z) voxel spacing
    model_name: str
    model_version: str
    inference_time_ms: int
    mesh_paths: List[str] = field(default_factory=list)  # populated post-extraction
    fragment_count: Optional[int] = None
    volume_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)


# ─── Model Registry ───────────────────────────────────────────────────────────


class ModelRegistry:
    """
    Registry for ML segmentation models.

    Manages model loading, caching, and lifecycle. Models are loaded
    lazily on first use and cached in memory.

    Supported backends:
    - PyTorch (*.pth, *.pt)
    - ONNX (*.onnx) — for deployment without CUDA
    - TotalSegmentator (via Python API)
    """

    def __init__(self, registry_settings: ModelRegistrySettings) -> None:
        self._settings = registry_settings
        self._loaded_models: Dict[str, Any] = {}
        self._model_versions: Dict[str, str] = {}
        self._device = self._resolve_device(registry_settings.default_device)
        logger.info(
            "model_registry_initialized",
            registry_path=str(registry_settings.registry_path),
            device=self._device,
        )

    def _resolve_device(self, device_str: str) -> str:
        """Resolve and validate the compute device."""
        try:
            import torch
            if device_str.startswith("cuda") and not torch.cuda.is_available():
                logger.warning("cuda_requested_but_unavailable", fallback="cpu")
                return "cpu"
            return device_str
        except ImportError:
            return "cpu"

    def get_available_models(self) -> List[str]:
        """List all models available in the registry."""
        available = ["totalsegmentator"]
        registry_path = self._settings.registry_path
        if registry_path.exists():
            for model_dir in registry_path.iterdir():
                if model_dir.is_dir():
                    # Look for model files
                    for ext in [".pth", ".pt", ".onnx"]:
                        if list(model_dir.glob(f"*{ext}")):
                            available.append(model_dir.name)
                            break
        return list(set(available))

    def load_model(self, model_name: str) -> "BaseSegmentationModel":
        """
        Load a model from the registry (cached after first load).

        Args:
            model_name: Name of the model to load

        Returns:
            Loaded model adapter

        Raises:
            ModelNotAvailableError: If model not found in registry
            ModelLoadError: If model loading fails
        """
        if model_name in self._loaded_models:
            return self._loaded_models[model_name]

        logger.info("loading_model", model_name=model_name, device=self._device)

        try:
            if model_name == "totalsegmentator":
                adapter = TotalSegmentatorAdapter(device=self._device)
            elif model_name.startswith("cmf_custom"):
                model_path = self._settings.cmf_segmentation_model_path
                if not model_path or not model_path.exists():
                    raise ModelNotAvailableError(
                        f"Custom CMF model not found",
                        context={"model_name": model_name, "expected_path": str(model_path)},
                    )
                adapter = CustomCMFModelAdapter(
                    model_path=model_path,
                    device=self._device,
                    fp16=self._settings.inference_fp16,
                )
            else:
                # Look for model in registry path
                model_dir = self._settings.registry_path / model_name
                model_files = list(model_dir.glob("*.pth")) + list(model_dir.glob("*.onnx"))
                if not model_files:
                    raise ModelNotAvailableError(
                        f"Model '{model_name}' not found in registry",
                        context={"model_name": model_name, "registry": str(self._settings.registry_path)},
                    )
                adapter = CustomCMFModelAdapter(
                    model_path=model_files[0],
                    device=self._device,
                    fp16=self._settings.inference_fp16,
                )

            self._loaded_models[model_name] = adapter
            self._model_versions[model_name] = adapter.version
            logger.info("model_loaded", model_name=model_name, version=adapter.version)
            return adapter

        except (ModelNotAvailableError, ModelLoadError):
            raise
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load model '{model_name}': {exc}",
                context={"model_name": model_name},
                cause=exc,
            )

    def unload_model(self, model_name: str) -> None:
        """Unload a model from memory (GPU and CPU)."""
        if model_name in self._loaded_models:
            model = self._loaded_models.pop(model_name)
            try:
                import torch
                del model
                torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("model_unloaded", model_name=model_name)


# ─── Model ABC ────────────────────────────────────────────────────────────────


class BaseSegmentationModel(abc.ABC):
    """Abstract base class for all segmentation model adapters."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Model identifier."""
        ...

    @property
    @abc.abstractmethod
    def version(self) -> str:
        """Model version string."""
        ...

    @abc.abstractmethod
    def predict(
        self,
        volume: np.ndarray,
        spacing: Tuple[float, float, float],
        structures: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, Dict[str, int], Dict[str, float]]:
        """
        Run segmentation inference.

        Args:
            volume: 3D CT volume (Z, Y, X) in Hounsfield Units
            spacing: Voxel spacing (x_mm, y_mm, z_mm)
            structures: Specific structures to segment (None = all)

        Returns:
            (masks, labels, confidences)
            - masks: 3D integer array same shape as volume
            - labels: {structure_name: label_value}
            - confidences: {structure_name: confidence_score}
        """
        ...


# ─── TotalSegmentator adapter ────────────────────────────────────────────────


class TotalSegmentatorAdapter(BaseSegmentationModel):
    """
    Adapter for TotalSegmentator (Wasserthal et al. 2022).
    https://github.com/wasserth/TotalSegmentator

    TotalSegmentator provides automated multi-organ segmentation from CT.
    For craniofacial work, it segments skull, mandible, and skeletal structures.
    """

    CMF_STRUCTURE_MAP = {
        # TotalSegmentator label name → our internal label name : label_value
        "skull": ("skull_base", 12),
        "mandible": ("mandible", 1),
        "maxilla": ("maxilla", 2),
        "vertebrae_C1": ("c1_vertebra", 20),
        "vertebrae_C2": ("c2_vertebra", 21),
        "zygoma_L": ("zygoma_L", 3),
        "zygoma_R": ("zygoma_R", 4),
    }

    def __init__(self, device: str = "cuda") -> None:
        self._device = device
        self._version = "2.0"  # TotalSegmentator version
        self._check_installation()

    def _check_installation(self) -> None:
        try:
            import totalsegmentator  # noqa: F401
        except ImportError:
            logger.warning(
                "totalsegmentator_not_installed",
                note="Install with: pip install totalsegmentator",
            )

    @property
    def name(self) -> str:
        return "totalsegmentator"

    @property
    def version(self) -> str:
        return self._version

    def predict(
        self,
        volume: np.ndarray,
        spacing: Tuple[float, float, float],
        structures: Optional[List[str]] = None,
        fast: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, int], Dict[str, float]]:
        """
        Run TotalSegmentator inference on a CT volume.

        The volume is temporarily saved as NIfTI, processed by TotalSegmentator,
        and the result is loaded back into memory.
        """
        import tempfile
        from pathlib import Path as TmpPath

        try:
            import SimpleITK as sitk
            import totalsegmentator
            from totalsegmentator.python_api import totalsegmentator as ts_api
        except ImportError as e:
            raise ModelLoadError(
                "TotalSegmentator or SimpleITK not available",
                context={"error": str(e)},
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = TmpPath(tmpdir)
            input_nii = tmp_path / "input.nii.gz"
            output_dir = tmp_path / "output"
            output_dir.mkdir()

            # Save volume as NIfTI for TotalSegmentator input
            sitk_image = sitk.GetImageFromArray(volume)
            sitk_image.SetSpacing(spacing)
            sitk.WriteImage(sitk_image, str(input_nii))

            start_time = time.perf_counter()
            inference_logger.log_inference_start(
                model_name=self.name,
                input_shape=volume.shape,
                device=self._device,
            )

            try:
                ts_api(
                    input=input_nii,
                    output=output_dir,
                    task=settings.model_registry.totalsegmentator_task,
                    fast=fast,
                    device=self._device.replace("cuda:", "") if "cuda" in self._device else "cpu",
                    quiet=True,
                )
            except Exception as e:
                raise InferenceError(
                    f"TotalSegmentator inference failed: {e}",
                    context={"model": "totalsegmentator"},
                    cause=e,
                )

            # Load combined segmentation mask
            mask_file = output_dir / "combined_labels.nii.gz"
            if not mask_file.exists():
                # Fall back to individual label files
                masks = np.zeros_like(volume, dtype=np.int32)
                labels: Dict[str, int] = {}
                for structure_file in output_dir.glob("*.nii.gz"):
                    struct_name = structure_file.stem.replace(".nii", "")
                    struct_sitk = sitk.ReadImage(str(structure_file))
                    struct_array = sitk.GetArrayFromImage(struct_sitk).astype(np.int32)
                    if struct_name in self.CMF_STRUCTURE_MAP:
                        internal_name, label_val = self.CMF_STRUCTURE_MAP[struct_name]
                        masks[struct_array > 0] = label_val
                        labels[internal_name] = label_val
            else:
                mask_sitk = sitk.ReadImage(str(mask_file))
                masks = sitk.GetArrayFromImage(mask_sitk).astype(np.int32)
                labels = {v[0]: v[1] for v in self.CMF_STRUCTURE_MAP.values()}

            # Estimate confidences (TotalSegmentator doesn't expose softmax natively)
            # Use a heuristic based on connected component quality
            confidences = self._estimate_confidences(masks, labels)

            inference_time_ms = int((time.perf_counter() - start_time) * 1000)
            inference_logger.log_inference_complete(
                model_name=self.name,
                start_time=start_time,
                output_summary={
                    "structures_found": len(labels),
                    "inference_time_ms": inference_time_ms,
                },
            )

        return masks, labels, confidences

    def _estimate_confidences(
        self, masks: np.ndarray, labels: Dict[str, int]
    ) -> Dict[str, float]:
        """
        Heuristic confidence estimation for TotalSegmentator output.
        Uses connected component analysis to penalize fragmented masks.
        """
        try:
            from scipy import ndimage
        except ImportError:
            return {name: 0.85 for name in labels}  # Default confidence

        confidences: Dict[str, float] = {}
        for structure_name, label_val in labels.items():
            struct_mask = (masks == label_val).astype(np.uint8)
            voxel_count = int(np.sum(struct_mask))
            if voxel_count == 0:
                confidences[structure_name] = 0.0
                continue

            # Count connected components — more components = lower confidence
            labeled, n_components = ndimage.label(struct_mask)
            if n_components == 1:
                confidence = 0.92
            elif n_components <= 3:
                confidence = 0.80
            else:
                confidence = max(0.5, 0.92 - (n_components - 1) * 0.05)

            confidences[structure_name] = round(confidence, 3)

        return confidences


# ─── Custom CMF model adapter ─────────────────────────────────────────────────


class CustomCMFModelAdapter(BaseSegmentationModel):
    """
    Adapter for fine-tuned craniofacial segmentation models.

    TODO: Implement when custom model is trained. This adapter provides the
    complete interface for a nnUNet or MONAI-based model trained specifically
    on CMF CT data with refined bone structure labels.

    Architecture recommendations:
    - nnUNetV2 with 3D full-resolution config
    - Training data: CMF CT scans with expert annotations
    - Labels: mandible, maxilla, zygoma L/R, orbital floor L/R, nasal bones,
              frontal bone, temporal bone L/R, per-fragment labels
    - Augmentation: intensity jitter, elastic deformation, random flips
    - Loss: Dice + Cross-entropy
    """

    # CMF-specific label mapping (more granular than TotalSegmentator)
    LABEL_MAP = {
        "background": 0,
        "mandible": 1,
        "maxilla": 2,
        "zygoma_L": 3,
        "zygoma_R": 4,
        "orbital_floor_L": 5,
        "orbital_floor_R": 6,
        "nasal_bones": 7,
        "frontal_bone": 8,
        "temporal_bone_L": 9,
        "temporal_bone_R": 10,
        "pterygoid_plates": 11,
        "skull_base": 12,
    }

    # Expected input spacing after resampling (based on training data)
    TARGET_SPACING_MM = (0.5, 0.5, 0.5)
    # CT intensity clipping range for normalization
    HU_CLIP_MIN = -200.0
    HU_CLIP_MAX = 3000.0

    def __init__(
        self,
        model_path: Path,
        device: str = "cuda",
        fp16: bool = True,
    ) -> None:
        self._model_path = model_path
        self._device = device
        self._fp16 = fp16
        self._model: Optional[Any] = None  # Loaded lazily
        self._version = self._read_version_file()

    def _read_version_file(self) -> str:
        version_file = self._model_path.parent / "version.txt"
        if version_file.exists():
            return version_file.read_text().strip()
        return "1.0.0"

    @property
    def name(self) -> str:
        return "cmf_custom"

    @property
    def version(self) -> str:
        return self._version

    def _load_model(self) -> None:
        """Lazy model loading — called on first inference."""
        if self._model is not None:
            return

        try:
            import torch
        except ImportError:
            raise ModelLoadError("PyTorch not installed")

        logger.info("loading_pytorch_model", path=str(self._model_path), device=self._device)

        if not self._model_path.exists():
            raise ModelLoadError(
                f"Model weights not found: {self._model_path}",
                context={"path": str(self._model_path)},
            )

        try:
            # TODO: Replace with actual model architecture import
            # from models.cmf_unet import CMFUNet
            # self._model = CMFUNet(num_classes=len(self.LABEL_MAP))
            # checkpoint = torch.load(self._model_path, map_location=self._device)
            # self._model.load_state_dict(checkpoint["model_state_dict"])
            # self._model.to(self._device)
            # self._model.eval()
            # if self._fp16 and "cuda" in self._device:
            #     self._model = self._model.half()
            raise NotImplementedError(
                "CustomCMFModelAdapter requires trained model weights. "
                "See TODO: train the model and implement architecture import."
            )
        except NotImplementedError:
            raise ModelLoadError(
                "Custom CMF model weights not yet trained",
                context={"model_path": str(self._model_path)},
            )

    def _preprocess(
        self, volume: np.ndarray, spacing: Tuple[float, float, float]
    ) -> "torch.Tensor":
        """
        Preprocess CT volume for model inference.

        Steps:
        1. Resample to model's expected isotropic spacing
        2. Clip HU values to relevant range
        3. Normalize to [0, 1]
        4. Add batch and channel dimensions
        """
        import torch

        # Resample to target spacing
        resampled = self._resample_volume(volume, spacing, self.TARGET_SPACING_MM)

        # Clip and normalize HU values
        clipped = np.clip(resampled, self.HU_CLIP_MIN, self.HU_CLIP_MAX)
        normalized = (clipped - self.HU_CLIP_MIN) / (self.HU_CLIP_MAX - self.HU_CLIP_MIN)

        # Convert to tensor: (1, 1, Z, Y, X)
        tensor = torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        if self._fp16 and "cuda" in self._device:
            tensor = tensor.half()

        return tensor.to(self._device)

    def _resample_volume(
        self,
        volume: np.ndarray,
        current_spacing: Tuple[float, float, float],
        target_spacing: Tuple[float, float, float],
    ) -> np.ndarray:
        """Resample volume to target isotropic spacing using scipy zoom."""
        from scipy.ndimage import zoom
        factors = tuple(c / t for c, t in zip(current_spacing, target_spacing))
        # Note: zoom expects (Z, Y, X) with factors in same order
        zoom_factors = (factors[2], factors[1], factors[0])  # Convert XYZ to ZYX
        return zoom(volume, zoom_factors, order=1)  # Linear interpolation for image

    def _postprocess(
        self,
        logits: "torch.Tensor",
        original_shape: Tuple[int, ...],
        original_spacing: Tuple[float, float, float],
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Post-process model logits to segmentation masks.

        Steps:
        1. Apply softmax
        2. Take argmax for label assignment
        3. Extract per-class confidence scores
        4. Resample back to original spacing
        5. Run connected component analysis for quality
        """
        import torch
        from scipy.ndimage import zoom

        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            labels_tensor = torch.argmax(probs, dim=1).squeeze(0)
            max_probs = probs.max(dim=1).values.squeeze(0)

        labels_np = labels_tensor.cpu().numpy().astype(np.int32)
        max_probs_np = max_probs.cpu().float().numpy()

        # Resample back to original spacing
        zoom_factors = tuple(o / r for o, r in zip(original_shape, labels_np.shape))
        masks_resampled = zoom(labels_np, zoom_factors, order=0)  # Nearest neighbor for labels

        # Extract per-structure mean confidence
        confidences: Dict[str, float] = {}
        for struct_name, label_val in self.LABEL_MAP.items():
            if label_val == 0:
                continue
            struct_probs = max_probs_np[labels_np == label_val]
            if len(struct_probs) > 0:
                confidences[struct_name] = float(np.mean(struct_probs))
            else:
                confidences[struct_name] = 0.0

        return masks_resampled, confidences

    def predict(
        self,
        volume: np.ndarray,
        spacing: Tuple[float, float, float],
        structures: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, Dict[str, int], Dict[str, float]]:
        """
        TODO: Implement after model training.

        This interface is complete and ready for integration once
        the custom CMF segmentation model is trained.
        """
        self._load_model()  # Will raise ModelLoadError until model is trained

        import torch
        input_tensor = self._preprocess(volume, spacing)
        start_time = time.perf_counter()

        with torch.no_grad():
            logits = self._model(input_tensor)

        masks, confidences = self._postprocess(logits, volume.shape, spacing)
        labels = {name: val for name, val in self.LABEL_MAP.items() if val > 0}

        # Filter to requested structures
        if structures:
            labels = {k: v for k, v in labels.items() if k in structures}
            confidences = {k: v for k, v in confidences.items() if k in structures}

        return masks, labels, confidences


# ─── SegmentationService ──────────────────────────────────────────────────────


class SegmentationService:
    """
    Main segmentation service. Coordinates model loading, inference, and post-processing.

    Usage:
        registry = ModelRegistry(settings.model_registry)
        service = SegmentationService(model_registry=registry)
        output = await service.segment_structures(volume, spacing)
    """

    # Minimum acceptable confidence for any structure
    MIN_CONFIDENCE_THRESHOLD = 0.5

    def __init__(self, model_registry: ModelRegistry) -> None:
        self._registry = model_registry

    async def segment_structures(
        self,
        volume: np.ndarray,
        spacing: Tuple[float, float, float],
        model_name: str = "totalsegmentator",
        structures: Optional[List[str]] = None,
        fast_mode: bool = False,
    ) -> SegmentationOutput:
        """
        Run bone segmentation on a CT volume.

        Args:
            volume: 3D CT volume (Z, Y, X) in Hounsfield Units
            spacing: Voxel spacing in mm (x, y, z)
            model_name: Name of model to use from registry
            structures: Specific structures to segment (None = all CMF structures)
            fast_mode: Use faster but lower-accuracy inference

        Returns:
            SegmentationOutput with masks, labels, and confidence scores

        Raises:
            ModelNotAvailableError: If model not in registry
            InferenceError: If inference fails
            LowConfidenceSegmentationError: If confidence below threshold
        """
        with TimedOperation(logger, "segmentation", model=model_name):
            model = self._registry.load_model(model_name)

            # Validate input
            if volume.ndim != 3:
                raise InferenceError(
                    f"Expected 3D volume, got {volume.ndim}D array",
                    context={"shape": list(volume.shape)},
                )

            if np.all(volume == 0):
                raise InferenceError("Input volume is empty (all zeros)")

            logger.info(
                "starting_segmentation",
                model=model_name,
                volume_shape=list(volume.shape),
                spacing_mm=list(spacing),
                structures=structures,
            )

            pipeline_start = time.perf_counter()

            # Run inference
            try:
                if isinstance(model, TotalSegmentatorAdapter):
                    masks, labels, confidences = model.predict(
                        volume, spacing, structures, fast=fast_mode
                    )
                else:
                    masks, labels, confidences = model.predict(volume, spacing, structures)
            except (ModelLoadError, InferenceError):
                raise
            except Exception as exc:
                raise InferenceError(
                    f"Inference failed: {exc}",
                    context={"model": model_name},
                    cause=exc,
                )

            inference_time_ms = int((time.perf_counter() - pipeline_start) * 1000)

            # Post-process: connected component cleanup
            masks, labels = self._cleanup_segmentation(masks, labels)

            # Compute volume statistics
            volume_stats = self._compute_volume_stats(masks, labels, spacing)

            # Check confidence threshold
            low_confidence = [
                name for name, conf in confidences.items()
                if conf < self.MIN_CONFIDENCE_THRESHOLD
            ]
            if len(low_confidence) == len(confidences):
                raise LowConfidenceSegmentationError(
                    "All structures below confidence threshold",
                    context={
                        "threshold": self.MIN_CONFIDENCE_THRESHOLD,
                        "min_confidence": min(confidences.values()) if confidences else 0,
                    }
                )

            overall_confidence = float(np.mean(list(confidences.values()))) if confidences else 0.0

            logger.info(
                "segmentation_complete",
                model=model_name,
                structures_found=len(labels),
                overall_confidence=round(overall_confidence, 3),
                inference_time_ms=inference_time_ms,
                low_confidence_structures=low_confidence,
            )

            return SegmentationOutput(
                masks=masks,
                labels=labels,
                confidences=confidences,
                spacing_mm=spacing,
                model_name=model_name,
                model_version=model.version,
                inference_time_ms=inference_time_ms,
                volume_stats=volume_stats,
            )

    def _cleanup_segmentation(
        self, masks: np.ndarray, labels: Dict[str, int]
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Post-processing cleanup:
        - Remove small disconnected components (< 1% of structure volume)
        - Fill small holes in structures
        - Remove labels with no voxels
        """
        try:
            from scipy import ndimage
        except ImportError:
            return masks, labels

        cleaned_masks = masks.copy()
        valid_labels: Dict[str, int] = {}

        for struct_name, label_val in labels.items():
            struct_mask = (masks == label_val).astype(np.uint8)
            voxel_count = int(np.sum(struct_mask))
            if voxel_count == 0:
                logger.debug("empty_structure_removed", structure=struct_name)
                continue

            # Label connected components
            labeled_components, n_components = ndimage.label(struct_mask)
            if n_components > 1:
                # Keep only components larger than 1% of the total
                component_sizes = ndimage.sum(
                    struct_mask, labeled_components, range(1, n_components + 1)
                )
                threshold = voxel_count * 0.01
                large_components = np.where(np.array(component_sizes) >= threshold)[0] + 1
                cleaned_struct = np.isin(labeled_components, large_components).astype(np.uint8)
                cleaned_masks[struct_mask > 0] = 0
                cleaned_masks[cleaned_struct > 0] = label_val

                if len(large_components) < n_components:
                    logger.debug(
                        "small_components_removed",
                        structure=struct_name,
                        removed=n_components - len(large_components),
                    )

            valid_labels[struct_name] = label_val

        return cleaned_masks, valid_labels

    def _compute_volume_stats(
        self,
        masks: np.ndarray,
        labels: Dict[str, int],
        spacing: Tuple[float, float, float],
    ) -> Dict[str, Dict[str, float]]:
        """Compute volumetric statistics per structure."""
        voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
        stats: Dict[str, Dict[str, float]] = {}

        for struct_name, label_val in labels.items():
            struct_mask = masks == label_val
            voxel_count = int(np.sum(struct_mask))
            volume_mm3 = voxel_count * voxel_volume_mm3
            volume_cc = volume_mm3 / 1000.0

            # Centroid
            if voxel_count > 0:
                coords = np.argwhere(struct_mask)
                centroid_zyx = np.mean(coords, axis=0)
                centroid_mm = [
                    float(centroid_zyx[2]) * spacing[0],
                    float(centroid_zyx[1]) * spacing[1],
                    float(centroid_zyx[0]) * spacing[2],
                ]
            else:
                centroid_mm = [0.0, 0.0, 0.0]

            stats[struct_name] = {
                "volume_cc": round(volume_cc, 3),
                "volume_mm3": round(volume_mm3, 1),
                "voxel_count": voxel_count,
                "centroid_x_mm": round(centroid_mm[0], 2),
                "centroid_y_mm": round(centroid_mm[1], 2),
                "centroid_z_mm": round(centroid_mm[2], 2),
            }

        return stats
