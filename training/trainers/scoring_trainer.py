"""
Scoring head trainer.

Supervised with clinical occlusion quality ratings.
Loss: MSE for continuous metrics, cross-entropy for Angle molar class.
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

# Molar class string → index
MOLAR_CLASS_MAP = {
    "Class_I": 0,
    "Class_II_div1": 1,
    "Class_II_div2": 2,
    "Class_III": 3,
}


class ScoringTrainer(BaseTrainer):
    """
    Trainer for the OcclusionScoringHead.

    Supervised with ground truth clinical metrics:
    - Continuous metrics (overjet, overbite, etc.): MSE loss
    - Molar classification (Angle Class I/II/III): cross-entropy loss
    """

    def build_model(self) -> nn.Module:
        """Build the full OcclusionModel (scoring head trained within context)."""
        from app.services.occlusion.occlusion_service import OcclusionModel

        model = OcclusionModel(
            per_tooth_dim=self.config.per_tooth_dim,
            global_dim=self.config.global_dim,
            transformer_layers=self.config.transformer_layers,
            transformer_heads=self.config.transformer_heads,
        )

        # Freeze encoder + transformer, train only scoring head
        for name, param in model.named_parameters():
            if "scoring_head" not in name:
                param.requires_grad = False

        return model

    def build_datasets(self) -> Tuple[Dataset, Dataset]:
        """Build train/val datasets from supervised IOS cases."""
        manifest_path = self.config.data_root / "manifest.json"
        with open(manifest_path) as f:
            data = json.load(f)

        cases = [TrainingCase(**c) for c in data["cases"]]
        # Keep only cases with ground truth occlusal metrics
        cases = [c for c in cases if c.ground_truth_occlusal_metrics]

        aug = AugmentationConfig(
            enable_se3_perturbation=False,
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
        Training step: MSE on continuous metrics + CE on molar class.
        """
        device = self.device
        loss_dict: Dict[str, float] = {}
        total_loss = torch.tensor(0.0, device=device)
        batch_size = len(batch["case_ids"])
        n_valid = 0

        mse_loss_fn = nn.MSELoss()
        ce_loss_fn = nn.CrossEntropyLoss()

        for i in range(batch_size):
            upper_teeth = batch["upper_teeth"][i]
            lower_teeth = batch["lower_teeth"][i]
            gt_metrics = batch["gt_metrics"][i]

            if not upper_teeth or not lower_teeth or not gt_metrics:
                continue

            upper_clouds = [upper_teeth[k] for k in sorted(upper_teeth.keys())]
            upper_fdi = sorted(upper_teeth.keys())
            lower_clouds = [lower_teeth[k] for k in sorted(lower_teeth.keys())]
            lower_fdi = sorted(lower_teeth.keys())

            # Forward
            out = self.model(upper_clouds, upper_fdi, lower_clouds, lower_fdi)
            pred_metrics = out["metrics"]

            sample_loss = torch.tensor(0.0, device=device)

            # ── Continuous metric MSE ────────────────────────────────
            continuous_keys = [
                "overjet_mm", "overbite_mm", "midline_deviation_mm",
                "cant_degrees", "curve_of_spee_mm",
            ]
            metric_mse = torch.tensor(0.0, device=device)
            n_metrics = 0
            for key in continuous_keys:
                if key in gt_metrics and key in pred_metrics:
                    gt_val = torch.tensor(
                        gt_metrics[key], device=device, dtype=torch.float32,
                    )
                    pred_val = pred_metrics[key]
                    if pred_val.dim() > 0:
                        pred_val = pred_val.mean()
                    metric_mse = metric_mse + (pred_val - gt_val).pow(2)
                    n_metrics += 1
                    loss_dict[f"metric/{key}"] = (
                        loss_dict.get(f"metric/{key}", 0.0)
                        + (pred_val - gt_val).pow(2).item()
                    )

            if n_metrics > 0:
                metric_mse = metric_mse / n_metrics
                sample_loss = sample_loss + self.config.w_metric_mse * metric_mse
                loss_dict["metric_mse"] = (
                    loss_dict.get("metric_mse", 0.0) + metric_mse.item()
                )

            # ── Molar class cross-entropy ────────────────────────────
            if (
                "molar_class" in gt_metrics
                and "molar_class_logits" in pred_metrics
            ):
                gt_class_str = gt_metrics["molar_class"]
                if gt_class_str in MOLAR_CLASS_MAP:
                    gt_idx = torch.tensor(
                        [MOLAR_CLASS_MAP[gt_class_str]],
                        device=device,
                        dtype=torch.long,
                    )
                    logits = pred_metrics["molar_class_logits"]
                    if logits.dim() == 1:
                        logits = logits.unsqueeze(0)
                    ce = ce_loss_fn(logits, gt_idx)
                    sample_loss = sample_loss + ce
                    loss_dict["molar_ce"] = (
                        loss_dict.get("molar_ce", 0.0) + ce.item()
                    )

            total_loss = total_loss + sample_loss
            n_valid += 1

        if n_valid > 0:
            total_loss = total_loss / n_valid
            for k in loss_dict:
                loss_dict[k] /= n_valid

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict

    def validate_step(
        self, batch: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float]]:
        """Validation step."""
        loss, loss_dict = self.train_step(batch)
        return loss.item(), loss_dict
