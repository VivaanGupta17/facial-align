"""
Base trainer with shared infrastructure for all Facial Align training loops.

Provides:
- Mixed-precision training (torch.cuda.amp)
- Gradient accumulation and clipping
- Learning rate warmup + cosine/plateau/step scheduling
- Early stopping with patience
- Best/latest checkpoint saving (model + optimizer + scheduler + epoch)
- TensorBoard logging
- Reproducible seeding
"""

from __future__ import annotations

import logging
import math
import os
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
)
from torch.utils.data import DataLoader, Dataset

from data_contracts.training.training_config import TrainingConfig

logger = logging.getLogger(__name__)


# ─── Seeding ─────────────────────────────────────────────────────────────────


def seed_everything(seed: int) -> None:
    """Reproducible seeding for torch, numpy, and Python random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── LR warmup wrapper ──────────────────────────────────────────────────────


class WarmupScheduler:
    """
    Linear warmup wrapper around any LR scheduler.

    During warmup_epochs, linearly ramps LR from 0 → base_lr.
    After warmup, delegates to the inner scheduler.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        inner_scheduler: Any,
        warmup_epochs: int,
    ) -> None:
        self.optimizer = optimizer
        self.inner = inner_scheduler
        self.warmup_epochs = warmup_epochs
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._step_count = 0

    def step(self, metrics: Optional[float] = None) -> None:
        self._step_count += 1
        if self._step_count <= self.warmup_epochs:
            # Linear warmup
            scale = self._step_count / max(self.warmup_epochs, 1)
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg["lr"] = base_lr * scale
        else:
            if isinstance(self.inner, ReduceLROnPlateau) and metrics is not None:
                self.inner.step(metrics)
            else:
                self.inner.step()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "step_count": self._step_count,
            "inner": self.inner.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._step_count = state["step_count"]
        self.inner.load_state_dict(state["inner"])


# ─── Checkpoint utilities ────────────────────────────────────────────────────


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupScheduler,
    epoch: int,
    best_val_loss: float,
    config: TrainingConfig,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "config": config.model_dump(),
    }
    if extra:
        state.update(extra)
    torch.save(state, str(path))
    logger.info("Checkpoint saved to %s (epoch %d)", path, epoch)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[WarmupScheduler] = None,
) -> Dict[str, Any]:
    """Load a training checkpoint."""
    state = torch.load(str(path), map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    logger.info("Checkpoint loaded from %s (epoch %d)", path, state.get("epoch", 0))
    return state


# ─── TensorBoard helper ─────────────────────────────────────────────────────


class TBLogger:
    """Thin wrapper over TensorBoard SummaryWriter (lazy import)."""

    def __init__(self, log_dir: Path, experiment_name: str) -> None:
        self._writer = None
        self._log_dir = log_dir / experiment_name
        self._log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def writer(self) -> Any:
        if self._writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._writer = SummaryWriter(str(self._log_dir))
            except ImportError:
                logger.warning("TensorBoard not available — logging to console only")
                self._writer = _NullWriter()
        return self._writer

    def log_scalars(self, tag: str, scalars: Dict[str, float], step: int) -> None:
        for key, value in scalars.items():
            self.writer.add_scalar(f"{tag}/{key}", value, step)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, step)

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()


class _NullWriter:
    """No-op writer when TensorBoard is unavailable."""

    def add_scalar(self, *args: Any, **kwargs: Any) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


# ─── Base Trainer ────────────────────────────────────────────────────────────


class BaseTrainer(ABC):
    """
    Abstract base trainer with shared infrastructure.

    Subclasses must implement:
    - ``build_model()`` → nn.Module
    - ``build_datasets()`` → (train_dataset, val_dataset)
    - ``train_step(batch)`` → (loss, loss_dict)
    - ``validate_step(batch)`` → (loss, metrics_dict)
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        seed_everything(config.seed)

        # Device
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        logger.info("Training on device: %s", self.device)

        # Build model
        self.model = self.build_model().to(self.device)
        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info("Model parameters: %s (%.1fM)", f"{param_count:,}", param_count / 1e6)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        inner_scheduler = self._build_scheduler()
        self.scheduler = WarmupScheduler(
            self.optimizer, inner_scheduler, config.warmup_epochs,
        )

        # Mixed precision
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision)

        # Logging
        self.tb = TBLogger(config.log_dir, config.experiment_name)

        # State
        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # Resume from checkpoint
        if config.resume_from is not None:
            ckpt_path = Path(config.resume_from)
            if ckpt_path.exists():
                state = load_checkpoint(ckpt_path, self.model, self.optimizer, self.scheduler)
                self.start_epoch = state.get("epoch", 0) + 1
                self.best_val_loss = state.get("best_val_loss", float("inf"))
                logger.info("Resuming from epoch %d", self.start_epoch)

    def _build_scheduler(self) -> Any:
        c = self.config
        effective_epochs = max(c.max_epochs - c.warmup_epochs, 1)
        if c.scheduler == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=effective_epochs, eta_min=1e-7)
        elif c.scheduler == "plateau":
            return ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-7,
            )
        elif c.scheduler == "step":
            return StepLR(self.optimizer, step_size=max(effective_epochs // 3, 1), gamma=0.3)
        else:
            raise ValueError(f"Unknown scheduler: {c.scheduler}")

    # ── Abstract methods ─────────────────────────────────────────────────

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Construct and return the model to train."""
        ...

    @abstractmethod
    def build_datasets(self) -> Tuple[Dataset, Dataset]:
        """Return (train_dataset, val_dataset)."""
        ...

    @abstractmethod
    def train_step(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Single training step.

        Args:
            batch: DataLoader batch dict.

        Returns:
            (loss_tensor, loss_components_dict)
        """
        ...

    @abstractmethod
    def validate_step(self, batch: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Single validation step (no gradients).

        Returns:
            (val_loss_scalar, metrics_dict)
        """
        ...

    # ── Build dataloaders ────────────────────────────────────────────────

    def build_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Build train and validation DataLoaders."""
        train_ds, val_ds = self.build_datasets()
        logger.info("Train set: %d samples, Val set: %d samples", len(train_ds), len(val_ds))

        collate_fn = getattr(train_ds, "collate_fn", None)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        return train_loader, val_loader

    # ── Main training loop ───────────────────────────────────────────────

    def train(self) -> Dict[str, Any]:
        """
        Run the full training loop.

        Returns:
            Summary dict with best_val_loss, total_epochs, etc.
        """
        train_loader, val_loader = self.build_dataloaders()
        c = self.config
        global_step = self.start_epoch * len(train_loader)

        logger.info(
            "Starting training: %d epochs, batch_size=%d, grad_accum=%d",
            c.max_epochs, c.batch_size, c.gradient_accumulation_steps,
        )

        for epoch in range(self.start_epoch, c.max_epochs):
            t0 = time.time()

            # ── Train epoch ──────────────────────────────────────────
            self.model.train()
            epoch_losses: Dict[str, List[float]] = {}
            self.optimizer.zero_grad()

            for step_idx, batch in enumerate(train_loader):
                # Move batch to device
                batch = self._to_device(batch)

                # Mixed precision forward
                with torch.amp.autocast("cuda", enabled=c.mixed_precision):
                    loss, loss_dict = self.train_step(batch)
                    loss = loss / c.gradient_accumulation_steps

                # Backward
                self.scaler.scale(loss).backward()

                # Gradient accumulation
                if (step_idx + 1) % c.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), c.gradient_clip,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                # Accumulate losses
                for key, val in loss_dict.items():
                    epoch_losses.setdefault(key, []).append(
                        val if isinstance(val, float) else val
                    )

                global_step += 1

                # Log per-step
                if global_step % 50 == 0:
                    self.tb.log_scalars("train_step", loss_dict, global_step)

            # ── Epoch-level train metrics ────────────────────────────
            avg_train_losses = {
                k: np.mean(v) for k, v in epoch_losses.items()
            }
            self.tb.log_scalars("train_epoch", avg_train_losses, epoch)

            # ── Validation ───────────────────────────────────────────
            val_loss, val_metrics = self._validate_epoch(val_loader)
            self.tb.log_scalar("val/loss", val_loss, epoch)
            self.tb.log_scalars("val", val_metrics, epoch)

            # ── LR scheduling ────────────────────────────────────────
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(metrics=val_loss)
            self.tb.log_scalar("lr", current_lr, epoch)

            elapsed = time.time() - t0
            logger.info(
                "Epoch %d/%d — train_loss=%.4f, val_loss=%.4f, lr=%.2e (%.1fs)",
                epoch + 1,
                c.max_epochs,
                avg_train_losses.get("total", 0.0),
                val_loss,
                current_lr,
                elapsed,
            )

            # ── Checkpointing ────────────────────────────────────────
            ckpt_dir = Path(c.checkpoint_dir) / c.experiment_name
            save_checkpoint(
                ckpt_dir / "latest.pt",
                self.model,
                self.optimizer,
                self.scheduler,
                epoch,
                self.best_val_loss,
                c,
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                save_checkpoint(
                    ckpt_dir / "best.pt",
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    self.best_val_loss,
                    c,
                )
                logger.info("New best model saved (val_loss=%.4f)", val_loss)
            else:
                self.patience_counter += 1
                if self.patience_counter >= c.patience:
                    logger.info(
                        "Early stopping at epoch %d (patience=%d)",
                        epoch + 1, c.patience,
                    )
                    break

        self.tb.flush()
        self.tb.close()

        return {
            "best_val_loss": self.best_val_loss,
            "total_epochs": epoch + 1,
            "final_train_losses": avg_train_losses,
            "final_val_metrics": val_metrics,
        }

    # ── Validation loop ──────────────────────────────────────────────

    @torch.no_grad()
    def _validate_epoch(
        self, val_loader: DataLoader,
    ) -> Tuple[float, Dict[str, float]]:
        """Run validation on the full val set."""
        self.model.eval()
        all_losses: List[float] = []
        all_metrics: Dict[str, List[float]] = {}

        for batch in val_loader:
            batch = self._to_device(batch)

            with torch.amp.autocast("cuda", enabled=self.config.mixed_precision):
                val_loss, metrics = self.validate_step(batch)

            all_losses.append(val_loss)
            for k, v in metrics.items():
                all_metrics.setdefault(k, []).append(v)

        avg_loss = np.mean(all_losses) if all_losses else float("inf")
        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}

        return float(avg_loss), avg_metrics

    # ── Utilities ────────────────────────────────────────────────────

    def _to_device(self, batch: Any) -> Any:
        """Recursively move batch to self.device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        elif isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return type(batch)(self._to_device(v) for v in batch)
        return batch
