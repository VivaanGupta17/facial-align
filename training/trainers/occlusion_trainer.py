"""
Occlusion model trainer.

Supervised: MSE on predicted vs GT SE(3) transforms + MSE on predicted
vs GT occlusal metrics.
Self-supervised: CompositeDentalLoss on the predicted occlusion.

Combined: L_total = w_sup * L_sup + w_self * L_self
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split

from data_contracts.training.training_case import TrainingCase
from data_contracts.training.training_config import TrainingConfig
from training.datasets.data_augmentation import AugmentationConfig
from training.datasets.dental_dataset import DentalDataset
from training.trainers.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class OcclusionTrainer(BaseTrainer):
    """
    Trainer for the OcclusionModel (end-to-end occlusion analysis).

    Supports:
    - Supervised training on ground-truth SE(3) transforms and clinical metrics.
    - Self-supervised training via the 7-component CompositeDentalLoss.
    - Mixed-mode: joint supervised + self-supervised.
    """

    def build_model(self) -> nn.Module:
        """Build the OcclusionModel with config-specified architecture."""
        from app.services.occlusion.occlusion_service import OcclusionModel

        model = OcclusionModel(
            per_tooth_dim=self.config.per_tooth_dim,
            global_dim=self.config.global_dim,
            transformer_layers=self.config.transformer_layers,
            transformer_heads=self.config.transformer_heads,
        )
        return model

    def build_datasets(self) -> Tuple[Dataset, Dataset]:
        """Build train/val DentalDatasets from manifest."""
        manifest_path = self.config.data_root / "manifest.json"
        with open(manifest_path) as f:
            data = json.load(f)

        cases = [TrainingCase(**c) for c in data["cases"]]

        aug = AugmentationConfig(
            enable_se3_perturbation=True,
            enable_jitter=True,
            enable_dropout=True,
            enable_tooth_removal=True,
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
        One training step: forward pass + composite loss.

        Returns (loss, loss_dict) for backprop and logging.
        """
        from app.services.occlusion.occlusal_losses import CompositeDentalLoss
        from app.services.occlusion.se3_transforms import apply_rotation_translation

        loss_dict: Dict[str, float] = {}
        device = self.device
        total_loss = torch.tensor(0.0, device=device)

        batch_size = len(batch["case_ids"])

        for i in range(batch_size):
            upper_teeth = batch["upper_teeth"][i]  # Dict[int, Tensor(N,3)]
            lower_teeth = batch["lower_teeth"][i]
            gt_metrics = batch["gt_metrics"][i]
            has_sup = batch["has_supervision"][i]

            if not upper_teeth or not lower_teeth:
                continue

            upper_clouds = [upper_teeth[k] for k in sorted(upper_teeth.keys())]
            upper_fdi = sorted(upper_teeth.keys())
            lower_clouds = [lower_teeth[k] for k in sorted(lower_teeth.keys())]
            lower_fdi = sorted(lower_teeth.keys())

            # Forward
            out = self.model(upper_clouds, upper_fdi, lower_clouds, lower_fdi)
            rot_6d = out["rotation_6d"]      # (1, N_lower, 6)
            translation = out["translation"]  # (1, N_lower, 3)
            metrics = out["metrics"]

            # ── Self-supervised: composite dental loss ────────────────
            # Stack tooth clouds into batch-format point clouds
            upper_pts = torch.cat(upper_clouds, dim=0).unsqueeze(0)  # (1, N_total, 3)

            # Apply predicted transforms to lower teeth
            lower_pts_list = []
            for j, fdi in enumerate(lower_fdi):
                pts = lower_teeth[fdi].unsqueeze(0)  # (1, M, 3)
                r6 = rot_6d[:, j: j + 1, :]  # (1, 1, 6)
                t = translation[:, j: j + 1, :]   # (1, 1, 3)
                transformed = apply_rotation_translation(
                    pts, r6.squeeze(1), t.squeeze(1),
                )
                lower_pts_list.append(transformed.squeeze(0))

            if lower_pts_list:
                lower_pts = torch.cat(lower_pts_list, dim=0).unsqueeze(0)
            else:
                continue

            composite_loss_fn = CompositeDentalLoss(
                w_chamfer=self.config.w_chamfer,
                w_overlap=self.config.w_overlap,
                w_uniformity=self.config.w_uniformity,
                w_collision=self.config.w_collision,
                w_midline=self.config.w_midline,
                w_molar=self.config.w_molar,
                w_arch_curve=self.config.w_arch_curve,
            ).to(device)

            self_sup_loss, components = composite_loss_fn(upper_pts, lower_pts)
            total_loss = total_loss + self_sup_loss

            for k, v in components.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                loss_dict[f"self_sup/{k}"] = loss_dict.get(f"self_sup/{k}", 0.0) + val

            # ── Supervised: metric prediction MSE ─────────────────────
            if has_sup and gt_metrics:
                metric_loss = torch.tensor(0.0, device=device)
                n_metrics = 0

                metric_map = {
                    "overjet_mm": "overjet_mm",
                    "overbite_mm": "overbite_mm",
                    "midline_deviation_mm": "midline_deviation_mm",
                    "cant_degrees": "cant_degrees",
                    "curve_of_spee_mm": "curve_of_spee_mm",
                }
                for gt_key, pred_key in metric_map.items():
                    if gt_key in gt_metrics and pred_key in metrics:
                        gt_val = torch.tensor(
                            gt_metrics[gt_key], device=device, dtype=torch.float32,
                        )
                        pred_val = metrics[pred_key]
                        if pred_val.dim() > 0:
                            pred_val = pred_val.mean()
                        metric_loss = metric_loss + (pred_val - gt_val).pow(2)
                        n_metrics += 1

                if n_metrics > 0:
                    metric_loss = metric_loss / n_metrics
                    total_loss = total_loss + self.config.w_metric_mse * metric_loss
                    loss_dict["sup/metric_mse"] = (
                        loss_dict.get("sup/metric_mse", 0.0) + metric_loss.item()
                    )

        # Average over batch
        if batch_size > 0:
            total_loss = total_loss / batch_size
            for k in loss_dict:
                loss_dict[k] /= batch_size

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict

    def validate_step(
        self, batch: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float]]:
        """Validation step — compute loss without gradients."""
        loss, loss_dict = self.train_step(batch)
        return loss.item(), loss_dict
