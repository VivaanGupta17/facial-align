"""
Landmark detector trainer.

Supervised training of DentalLandmarkDetector with per-landmark L2 loss
against annotated landmark positions from IOS scans.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split

from data_contracts.training.training_case import TrainingCase
from data_contracts.training.training_config import TrainingConfig
from training.datasets.data_augmentation import AugmentationConfig
from training.datasets.dental_dataset import DentalDataset
from training.trainers.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class LandmarkTrainer(BaseTrainer):
    """
    Trainer for the DentalLandmarkDetector.

    Supervised with annotated landmark positions from IOS scans.
    Loss: per-landmark L2 distance weighted by detection confidence.
    """

    def build_model(self) -> nn.Module:
        """Build the DentalLandmarkDetector."""
        from app.services.occlusion.landmark_detector import DentalLandmarkDetector

        model = DentalLandmarkDetector()
        return model

    def build_datasets(self) -> Tuple[Dataset, Dataset]:
        """Build train/val DentalDatasets (filtered to cases with landmarks)."""
        manifest_path = self.config.data_root / "manifest.json"
        with open(manifest_path) as f:
            data = json.load(f)

        cases = [TrainingCase(**c) for c in data["cases"]]
        # Keep only cases with ground truth landmarks
        cases = [c for c in cases if c.ground_truth_landmarks]

        aug = AugmentationConfig(
            enable_se3_perturbation=True,
            enable_jitter=True,
            enable_dropout=False,
            enable_tooth_removal=False,
        )
        full_ds = DentalDataset(cases=cases, augmentation=aug)

        n_train = int(len(full_ds) * self.config.train_split)
        n_val = len(full_ds) - n_train
        train_ds, val_ds = random_split(full_ds, [n_train, n_val])
        return train_ds, val_ds

    def train_step(
        self, batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Training step: per-landmark L2 loss.

        For each tooth in the batch, predicts landmark positions and
        compares to ground truth using L2 distance.
        """
        device = self.device
        loss_dict: Dict[str, float] = {}
        total_loss = torch.tensor(0.0, device=device)
        batch_size = len(batch["case_ids"])
        n_valid = 0
        total_l2 = 0.0
        total_confidence = 0.0

        for i in range(batch_size):
            # Combine upper and lower teeth for landmark detection
            teeth: Dict[int, torch.Tensor] = {}
            if batch["upper_teeth"][i]:
                teeth.update(batch["upper_teeth"][i])
            if batch["lower_teeth"][i]:
                teeth.update(batch["lower_teeth"][i])

            if not teeth:
                continue

            tooth_clouds = [teeth[k] for k in sorted(teeth.keys())]
            fdi_numbers = sorted(teeth.keys())

            # Forward: predict landmarks per tooth
            pred_landmarks = self.model(tooth_clouds, fdi_numbers)

            # Compare to ground truth landmarks
            gt_metrics = batch["gt_metrics"][i]  # May contain landmark data
            if not gt_metrics:
                continue

            # Extract GT landmarks that match predicted teeth
            sample_loss = torch.tensor(0.0, device=device)
            n_landmarks = 0

            for fdi, tooth_preds in pred_landmarks.items():
                # tooth_preds: Dict[str, Tensor] with 'positions' and 'confidences'
                if "positions" not in tooth_preds:
                    continue

                pred_pos = tooth_preds["positions"]      # (K, 3)
                pred_conf = tooth_preds["confidences"]    # (K,)

                # Look for GT landmarks matching this tooth's FDI
                gt_key = f"tooth_{fdi}"
                if gt_key in gt_metrics and isinstance(gt_metrics[gt_key], list):
                    gt_coords = torch.tensor(
                        gt_metrics[gt_key], device=device, dtype=torch.float32,
                    )
                    if gt_coords.dim() == 1 and gt_coords.shape[0] == 3:
                        gt_coords = gt_coords.unsqueeze(0)

                    # L2 distance for each predicted landmark to nearest GT
                    if gt_coords.shape[0] > 0 and pred_pos.shape[0] > 0:
                        dists = torch.cdist(pred_pos, gt_coords)  # (K_pred, K_gt)
                        min_dists = dists.min(dim=-1).values  # (K_pred,)
                        # Weight by confidence
                        weighted_l2 = (min_dists * pred_conf).sum() / (
                            pred_conf.sum() + 1e-8
                        )
                        sample_loss = sample_loss + weighted_l2
                        n_landmarks += 1
                        total_l2 += min_dists.mean().item()
                        total_confidence += pred_conf.mean().item()

            if n_landmarks > 0:
                sample_loss = sample_loss / n_landmarks
                total_loss = total_loss + sample_loss
                n_valid += 1

        if n_valid > 0:
            total_loss = total_loss / n_valid
            total_l2 /= n_valid
            total_confidence /= n_valid

        loss_dict["total"] = total_loss.item()
        loss_dict["landmark_l2_mm"] = total_l2
        loss_dict["mean_confidence"] = total_confidence
        return total_loss, loss_dict

    def validate_step(
        self, batch: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float]]:
        """Validation step."""
        loss, loss_dict = self.train_step(batch)
        return loss.item(), loss_dict
