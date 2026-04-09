"""
Training loop for the supervised fracture reduction model.

Extends BaseTrainer with supervised-specific functionality:
- Two-stage learning rate schedule (MSE rotation → geodesic)
- Mixed-precision training with gradient accumulation
- Per-epoch validation with clinical metrics (translation error, rotation error)
- Checkpoint saving with best-model tracking
- WandB / TensorBoard logging
- MC Dropout evaluation for uncertainty calibration

Training workflow:
    1. Initialise model, optimiser, loss function
    2. For each epoch:
       a. Phase A (epochs 1–50): MSE rotation loss
       b. Phase B (epochs 51+): Geodesic rotation loss
    3. Validate every N epochs with clinical error metrics
    4. Save checkpoints and best model

References:
- FracFormer (IEEE TMI 2025): Two-stage rotation loss schedule
- Swin-T Tooth Alignment: Per-tooth evaluation protocol
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class SupervisedTrainingConfig:
    """Configuration for supervised model training."""
    # Model
    max_fragments: int = 8
    max_teeth: int = 32

    # Training
    num_epochs: int = 200
    batch_size: int = 2          # Small batch for 3D CT volumes
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 4

    # Schedule
    phase_transition_epoch: int = 50
    warmup_epochs: int = 10
    cosine_min_lr: float = 1e-6

    # Mixed precision
    use_amp: bool = True

    # Validation
    val_every_n_epochs: int = 5
    mc_dropout_passes: int = 10

    # Checkpointing
    checkpoint_dir: str = "checkpoints/supervised"
    save_every_n_epochs: int = 10
    keep_top_k: int = 3

    # Logging
    log_every_n_steps: int = 10
    use_wandb: bool = False
    wandb_project: str = "facial-align-supervised"


class SupervisedTrainer:
    """
    Trainer for FacialAlignSupervisedModel.

    Manages the full training lifecycle: data loading, optimisation,
    validation, checkpointing, and logging.

    Usage:
        config = SupervisedTrainingConfig(num_epochs=200, batch_size=2)
        trainer = SupervisedTrainer(config)
        trainer.train(train_dataset, val_dataset)

    Args:
        config: Training configuration.
        model: Optional pre-initialised model. If None, creates from defaults.
        device: Training device.
    """

    def __init__(
        self,
        config: Optional[SupervisedTrainingConfig] = None,
        model: Optional[nn.Module] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        if config is None:
            config = SupervisedTrainingConfig()
        self.config = config
        self.device = torch.device(device)

        # Model
        if model is None:
            from app.services.supervised.supervised_model import (
                FacialAlignSupervisedModel,
                SupervisedModelConfig,
            )
            model_config = SupervisedModelConfig(
                max_fragments=config.max_fragments,
                max_teeth=config.max_teeth,
                use_mixed_precision=config.use_amp,
            )
            model = FacialAlignSupervisedModel(model_config)

        self.model = model.to(self.device)

        # Loss
        from app.services.supervised.supervised_losses import (
            SupervisedReductionLoss,
        )
        self.loss_fn = SupervisedReductionLoss(
            phase_transition_epoch=config.phase_transition_epoch,
        )

        # Optimiser
        self.optimiser = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler: warmup + cosine annealing
        self.scheduler = self._build_scheduler()

        # Mixed precision
        self.scaler = GradScaler(enabled=config.use_amp)

        # State
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.best_checkpoint_path: Optional[str] = None

        # Checkpointing
        self.ckpt_dir = Path(config.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Build learning rate scheduler with warmup + cosine decay."""
        warmup = self.config.warmup_epochs
        total = self.config.num_epochs

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup:
                return epoch / max(warmup, 1)
            progress = (epoch - warmup) / max(total - warmup, 1)
            min_factor = self.config.cosine_min_lr / self.config.learning_rate
            return min_factor + (1 - min_factor) * 0.5 * (1 + np.cos(np.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimiser, lr_lambda)

    def train(
        self,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
    ) -> Dict[str, Any]:
        """
        Run full training loop.

        Args:
            train_dataset: Training dataset yielding (ct_volume, ios_data, targets).
            val_dataset: Optional validation dataset.

        Returns:
            Training summary dict.
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=1,
                pin_memory=True,
            )

        logger.info(
            "Starting training: %d epochs, batch=%d, lr=%.2e, device=%s",
            self.config.num_epochs, self.config.batch_size,
            self.config.learning_rate, self.device,
        )

        history = {"train_loss": [], "val_loss": [], "val_metrics": []}

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            # Train one epoch
            train_metrics = self._train_epoch(train_loader, epoch)
            history["train_loss"].append(train_metrics["loss"])

            # Step scheduler
            self.scheduler.step()

            # Log
            phase = "geodesic" if epoch >= self.config.phase_transition_epoch else "MSE"
            lr = self.scheduler.get_last_lr()[0]
            logger.info(
                "Epoch %d/%d [%s]: loss=%.4f lr=%.2e",
                epoch + 1, self.config.num_epochs, phase,
                train_metrics["loss"], lr,
            )

            # Validate
            if val_loader is not None and (epoch + 1) % self.config.val_every_n_epochs == 0:
                val_metrics = self._validate(val_loader, epoch)
                history["val_loss"].append(val_metrics["loss"])
                history["val_metrics"].append(val_metrics)

                logger.info(
                    "  Val: loss=%.4f trans_err=%.2fmm rot_err=%.2f°",
                    val_metrics["loss"],
                    val_metrics.get("mean_translation_error_mm", 0),
                    val_metrics.get("mean_rotation_error_deg", 0),
                )

                # Save best model
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self._save_checkpoint(epoch, is_best=True)

            # Periodic checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(epoch)

        # Final checkpoint
        self._save_checkpoint(self.config.num_epochs - 1, is_final=True)
        logger.info("Training complete. Best val loss: %.4f", self.best_val_loss)

        return history

    def _train_epoch(
        self,
        loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch with gradient accumulation."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        self.optimiser.zero_grad()

        for batch_idx, batch in enumerate(loader):
            # Move to device
            ct_volume = batch["ct_volume"].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch["targets"].items()}

            ios_data = None
            ios_ids = None
            tooth_mask = None
            if "ios_point_clouds" in batch and batch["ios_point_clouds"] is not None:
                ios_data = batch["ios_point_clouds"].to(self.device)
                ios_ids = batch.get("ios_tooth_ids")
                if ios_ids is not None:
                    ios_ids = ios_ids.to(self.device)
            if "tooth_mask" in batch:
                tooth_mask = batch["tooth_mask"].to(self.device)

            num_fragments = batch.get("num_fragments")
            if num_fragments is not None:
                num_fragments = num_fragments.to(self.device)

            # Forward pass with AMP
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.config.use_amp,
            ):
                predictions = self.model(
                    ct_volume=ct_volume,
                    ios_point_clouds=ios_data,
                    ios_tooth_ids=ios_ids,
                    num_fragments=num_fragments,
                    tooth_mask=tooth_mask,
                )
                loss_dict = self.loss_fn(predictions, targets, epoch=epoch)
                loss = loss_dict["total_loss"] / self.config.gradient_accumulation_steps

            # Backward
            self.scaler.scale(loss).backward()

            # Gradient accumulation step
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimiser)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm,
                )
                self.scaler.step(self.optimiser)
                self.scaler.update()
                self.optimiser.zero_grad()
                self.global_step += 1

            total_loss += loss_dict["total_loss"].item()
            num_batches += 1

        return {"loss": total_loss / max(num_batches, 1)}

    @torch.no_grad()
    def _validate(
        self,
        loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Run validation and compute clinical error metrics."""
        self.model.eval()
        total_loss = 0.0
        all_trans_errors = []
        all_rot_errors = []
        num_batches = 0

        for batch in loader:
            ct_volume = batch["ct_volume"].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch["targets"].items()}

            ios_data = None
            ios_ids = None
            tooth_mask = None
            if "ios_point_clouds" in batch and batch["ios_point_clouds"] is not None:
                ios_data = batch["ios_point_clouds"].to(self.device)
                ios_ids = batch.get("ios_tooth_ids")
                if ios_ids is not None:
                    ios_ids = ios_ids.to(self.device)
            if "tooth_mask" in batch:
                tooth_mask = batch["tooth_mask"].to(self.device)

            num_fragments = batch.get("num_fragments")
            if num_fragments is not None:
                num_fragments = num_fragments.to(self.device)

            predictions = self.model(
                ct_volume=ct_volume,
                ios_point_clouds=ios_data,
                ios_tooth_ids=ios_ids,
                num_fragments=num_fragments,
                tooth_mask=tooth_mask,
            )
            loss_dict = self.loss_fn(predictions, targets, epoch=epoch)
            total_loss += loss_dict["total_loss"].item()

            # Compute clinical metrics
            pred_t = predictions["fragment_transforms"]["translations"]
            gt_t = targets["fragment_translations"]
            trans_err = (pred_t - gt_t).norm(dim=-1).mean().item()
            all_trans_errors.append(trans_err)

            # Rotation error in degrees
            if "fragment_rotation_loss" in loss_dict:
                rot_err_rad = loss_dict["fragment_rotation_loss"].item()
                all_rot_errors.append(np.degrees(rot_err_rad))

            num_batches += 1

        return {
            "loss": total_loss / max(num_batches, 1),
            "mean_translation_error_mm": np.mean(all_trans_errors) if all_trans_errors else 0.0,
            "mean_rotation_error_deg": np.mean(all_rot_errors) if all_rot_errors else 0.0,
        }

    def _save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        is_final: bool = False,
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimiser_state_dict": self.optimiser.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        if is_best:
            path = self.ckpt_dir / "best_model.pt"
            self.best_checkpoint_path = str(path)
        elif is_final:
            path = self.ckpt_dir / "final_model.pt"
        else:
            path = self.ckpt_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt"

        torch.save(checkpoint, path)
        logger.info("Saved checkpoint: %s", path)

    @classmethod
    def resume_from_checkpoint(
        cls,
        checkpoint_path: str,
        config: Optional[SupervisedTrainingConfig] = None,
    ) -> "SupervisedTrainer":
        """Resume training from a checkpoint."""
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        saved_config = checkpoint.get("config", SupervisedTrainingConfig())
        if config is not None:
            saved_config = config

        trainer = cls(config=saved_config)

        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        trainer.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        trainer.current_epoch = checkpoint["epoch"] + 1
        trainer.global_step = checkpoint["global_step"]
        trainer.best_val_loss = checkpoint["best_val_loss"]

        logger.info(
            "Resumed from epoch %d (step %d), best_val_loss=%.4f",
            trainer.current_epoch, trainer.global_step, trainer.best_val_loss,
        )
        return trainer
