"""
nnU-Net Model Adapter

Generic adapter for nnU-Net v2 models within the Facial Align pipeline.
Used for custom-trained segmentation models (Phase 2+).

nnU-Net automatically configures architecture, preprocessing, and training
hyperparameters for optimal performance on medical segmentation tasks.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np

from services.inference.model_registry import InferenceModel, ModelVersion

logger = logging.getLogger(__name__)


class NNUNetModel(InferenceModel):
    """
    Adapter for nnU-Net v2 models.

    In Phase 2, custom nnU-Net models will be trained for:
    - Mandibular fracture fragment segmentation
    - Condyle delineation
    - Inferior alveolar nerve canal
    - Custom multi-structure CMF segmentation

    Training workflow:
    1. Prepare dataset in nnU-Net format (imagesTr, labelsTr, dataset.json)
    2. Run nnU-Net experiment planning and preprocessing
    3. Train with nnUNetv2_train
    4. Export best model
    5. Register in ModelRegistry and wrap with this adapter
    """

    def __init__(self, version: ModelVersion, device: str = "cpu"):
        self._version = version
        self._device = device
        self._predictor = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def get_info(self) -> ModelVersion:
        return self._version

    def load(self, model_folder: str, fold: str = "all"):
        """
        Load a trained nnU-Net model.

        Args:
            model_folder: Path to nnU-Net model folder
                         (contains plans.json, fold_X/ directories)
            fold: Which fold to use ("all" for ensemble, "0"-"4" for single)
        """
        try:
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

            self._predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_mirroring=True,
                perform_everything_on_device=True if "cuda" in self._device else False,
                device=self._device,
                verbose=False,
            )
            self._predictor.initialize_from_trained_model_folder(
                model_folder,
                use_folds=(fold,) if fold != "all" else None,
                checkpoint_name="checkpoint_final.pth",
            )
            self._loaded = True
            logger.info(f"nnU-Net model loaded from {model_folder}")

        except ImportError:
            logger.error("nnU-Net v2 not installed. Install: pip install nnunetv2")
            raise
        except Exception as e:
            logger.error(f"Failed to load nnU-Net model: {e}")
            raise

    def predict(
        self,
        input_data: np.ndarray,
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
        **kwargs,
    ) -> dict[str, Any]:
        """
        Run nnU-Net inference on a CT volume.

        Args:
            input_data: 3D numpy array (Z, Y, X) — preprocessed CT in HU
            spacing: Voxel spacing in mm

        Returns:
            dict with masks, labels, confidences, inference_time_ms
        """
        if not self._loaded:
            raise RuntimeError(
                "nnU-Net model not loaded. Call load() first or download weights. "
                "Custom nnU-Net models require training (Phase 2)."
            )

        start_time = time.time()

        # nnU-Net expects (C, Z, Y, X) input with channel dimension
        input_4d = input_data[np.newaxis, ...]  # Add channel dim
        properties = {
            "spacing": list(spacing),
        }

        # Run prediction
        prediction = self._predictor.predict_single_npy_array(
            input_4d, properties, None, None, False
        )

        inference_time = int((time.time() - start_time) * 1000)

        # Parse label map from dataset.json
        label_map = self._get_label_map()

        # Extract per-structure results
        masks = {}
        confidences = {}
        for label_id, label_name in label_map.items():
            if label_id == 0:  # Skip background
                continue
            mask = (prediction == label_id).astype(np.uint8)
            if mask.sum() > 0:
                masks[label_name] = mask
                # TODO: Extract softmax probabilities for true confidence
                confidences[label_name] = 0.90  # Placeholder

        return {
            "masks": masks,
            "labels": {name: lid for lid, name in label_map.items() if lid != 0},
            "confidences": confidences,
            "combined_mask": prediction.astype(np.uint8),
            "inference_time_ms": inference_time,
            "model_name": self._version.name,
            "model_version": self._version.version,
        }

    def _get_label_map(self) -> dict[int, str]:
        """Get label ID to structure name mapping from model config."""
        if self._predictor and hasattr(self._predictor, "plans_manager"):
            # Extract from nnU-Net plans
            try:
                dataset_json = self._predictor.dataset_json
                return {
                    int(k): v for k, v in dataset_json.get("labels", {}).items()
                }
            except Exception:
                pass

        # Fallback default label map
        return {
            0: "background",
            1: "mandible",
            2: "maxilla",
            3: "teeth",
        }

    def preprocess(self, input_data: np.ndarray, spacing: tuple[float, ...]) -> np.ndarray:
        """nnU-Net handles its own preprocessing — pass through."""
        return input_data


class NNUNetTrainingConfig:
    """
    Configuration helper for training custom nnU-Net models.

    This class generates the dataset.json and folder structure
    required by nnU-Net v2 for training.

    Usage (Phase 2):
        config = NNUNetTrainingConfig(
            dataset_name="Dataset501_MandibleFragments",
            labels={"background": 0, "fragment_1": 1, "fragment_2": 2, ...},
        )
        config.prepare_dataset(raw_data_dir, output_dir)
    """

    def __init__(
        self,
        dataset_name: str,
        labels: dict[str, int],
        modality: str = "CT",
        description: str = "",
    ):
        self.dataset_name = dataset_name
        self.labels = labels
        self.modality = modality
        self.description = description

    def generate_dataset_json(self) -> dict:
        """Generate nnU-Net dataset.json."""
        return {
            "channel_names": {"0": self.modality},
            "labels": {v: k for k, v in self.labels.items()},
            "numTraining": 0,  # Updated when data is added
            "file_ending": ".nii.gz",
            "name": self.dataset_name,
            "description": self.description,
        }

    def prepare_dataset(self, raw_dir: str, output_dir: str):
        """
        Prepare raw data into nnU-Net training format.

        Expected raw_dir structure:
            raw_dir/
                images/
                    case_001.nii.gz
                    case_002.nii.gz
                labels/
                    case_001.nii.gz  (segmentation masks)
                    case_002.nii.gz
        """
        # TODO: Implement dataset preparation for Phase 2
        raise NotImplementedError(
            "Dataset preparation for nnU-Net training is a Phase 2 task. "
            "See docs/evaluation/training_plan.md for the training roadmap."
        )
