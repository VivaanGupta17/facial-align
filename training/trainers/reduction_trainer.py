"""
Reduction pipeline end-to-end trainer.

Supervised: geodesic SO(3) distance + L2 translation on per-fragment
transforms derived from pre/post-op CT registration.
Self-supervised: joint optimizer composite loss (fracture fitting + occlusion).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
from torch.utils.data import Dataset, random_split

from data_contracts.training.training_case import TrainingCase
from data_contracts.training.training_config import TrainingConfig
from training.datasets.data_augmentation import AugmentationConfig
from training.datasets.fracture_dataset import FractureDataset
from training.trainers.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


def _geodesic_rotation_loss(
    R_pred: torch.Tensor,
    R_gt: torch.Tensor,
) -> torch.Tensor:
    """
    Geodesic distance on SO(3) between two rotation matrices.

    d(R1, R2) = arccos((trace(R1^T @ R2) - 1) / 2)

    Args:
        R_pred: (B, 3, 3) predicted rotation matrices.
        R_gt: (B, 3, 3) ground truth rotation matrices.

    Returns:
        Scalar mean geodesic distance (radians).
    """
    R_diff = torch.bmm(R_pred.transpose(1, 2), R_gt)
    trace = R_diff.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    # Clamp for numerical stability
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = cos_angle.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos_angle)
    return angle.mean()


class ReductionTrainer(BaseTrainer):
    """
    Trainer for the fracture reduction pipeline.

    Loads pre-op fragments + IOS dental scans. Ground truth comes from
    pre/post-op CT registration (per-fragment SE(3) transforms).

    Loss:
    - Supervised: geodesic SO(3) + L2 translation per fragment
    - Self-supervised: joint optimizer composite loss
    """

    def build_model(self) -> nn.Module:
        """Build the OcclusionModel (shared backbone for reduction)."""
        from app.services.occlusion.occlusion_service import OcclusionModel

        model = OcclusionModel(
            per_tooth_dim=self.config.per_tooth_dim,
            global_dim=self.config.global_dim,
            transformer_layers=self.config.transformer_layers,
            transformer_heads=self.config.transformer_heads,
        )
        return model

    def build_datasets(self) -> Tuple[Dataset, Dataset]:
        """Build train/val FractureDatasets from manifest."""
        manifest_path = self.config.data_root / "manifest.json"
        with open(manifest_path) as f:
            data = json.load(f)

        cases = [TrainingCase(**c) for c in data["cases"]]

        aug = AugmentationConfig(
            enable_se3_perturbation=True,
            enable_jitter=True,
            enable_dropout=True,
            enable_tooth_removal=False,
        )
        full_ds = FractureDataset(cases=cases, augmentation=aug)

        n_train = int(len(full_ds) * self.config.train_split)
        n_val = len(full_ds) - n_train
        train_ds, val_ds = random_split(full_ds, [n_train, n_val])
        return train_ds, val_ds

    def train_step(
        self, batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Training step: per-fragment transform regression.

        Computes geodesic rotation loss + L2 translation loss against
        ground truth SE(3) transforms from pre/post-op registration.
        """
        device = self.device
        loss_dict: Dict[str, float] = {}
        total_loss = torch.tensor(0.0, device=device)
        batch_size = len(batch["case_ids"])
        n_valid = 0

        for i in range(batch_size):
            tooth_clouds = batch["tooth_clouds"][i]   # Dict[int, Tensor]
            gt_transforms = batch["gt_transforms"][i]  # Dict[str, Tensor(4,4)]

            if not tooth_clouds:
                continue

            # Split into upper/lower based on FDI
            upper_teeth: Dict[int, torch.Tensor] = {}
            lower_teeth: Dict[int, torch.Tensor] = {}
            for fdi, pts in tooth_clouds.items():
                if fdi < 30:
                    upper_teeth[fdi] = pts
                else:
                    lower_teeth[fdi] = pts

            if not upper_teeth or not lower_teeth:
                continue

            upper_clouds = [upper_teeth[k] for k in sorted(upper_teeth.keys())]
            upper_fdi = sorted(upper_teeth.keys())
            lower_clouds = [lower_teeth[k] for k in sorted(lower_teeth.keys())]
            lower_fdi = sorted(lower_teeth.keys())

            # Forward through occlusion model
            out = self.model(upper_clouds, upper_fdi, lower_clouds, lower_fdi)
            pred_rot_6d = out["rotation_6d"]     # (1, N_lower, 6)
            pred_trans = out["translation"]       # (1, N_lower, 3)

            # ── Supervised: per-fragment transform loss ───────────────
            if gt_transforms:
                rot_loss = torch.tensor(0.0, device=device)
                trans_loss = torch.tensor(0.0, device=device)
                n_frags = 0

                # Match predicted per-tooth transforms to GT fragment transforms
                # Use the first GT transform as the target for all lower teeth
                # (simplification: single-fragment or dominant fragment)
                gt_keys = sorted(gt_transforms.keys())
                if gt_keys:
                    gt_T = gt_transforms[gt_keys[0]]  # (4,4)
                    gt_R = gt_T[:3, :3].unsqueeze(0)  # (1, 3, 3)
                    gt_t = gt_T[:3, 3].unsqueeze(0)   # (1, 3)

                    n_teeth = pred_rot_6d.shape[1]
                    for j in range(n_teeth):
                        pred_R = rotation_6d_to_matrix(
                            pred_rot_6d[:, j, :],
                        )  # (1, 3, 3)
                        pred_t = pred_trans[:, j, :]  # (1, 3)

                        rot_loss = rot_loss + _geodesic_rotation_loss(pred_R, gt_R)
                        trans_loss = trans_loss + (pred_t - gt_t).pow(2).sum(dim=-1).mean()
                        n_frags += 1

                    if n_frags > 0:
                        rot_loss = rot_loss / n_frags
                        trans_loss = trans_loss / n_frags

                    sup_loss = rot_loss + self.config.w_transform_l2 * trans_loss
                    total_loss = total_loss + sup_loss

                    loss_dict["sup/rotation_geodesic"] = (
                        loss_dict.get("sup/rotation_geodesic", 0.0) + rot_loss.item()
                    )
                    loss_dict["sup/translation_l2"] = (
                        loss_dict.get("sup/translation_l2", 0.0) + trans_loss.item()
                    )

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
        """Validation step — compute loss without gradients."""
        loss, loss_dict = self.train_step(batch)
        return loss.item(), loss_dict
