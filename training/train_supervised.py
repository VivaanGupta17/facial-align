"""
CLI entry point for supervised fracture reduction model training.

Usage:
    python -m training.train_supervised --config training/configs/supervised_config.yaml
    python -m training.train_supervised --config training/configs/supervised_config.yaml --resume checkpoints/supervised/best_model.pt
    python -m training.train_supervised --config training/configs/supervised_config.yaml --synthetic

Loads the YAML config, instantiates the appropriate dataset(s),
and delegates to SupervisedTrainer.train().
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import yaml

# Add project root to sys.path for cross-package imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.trainers.supervised_trainer import (
    SupervisedTrainer,
    SupervisedTrainingConfig,
)

logger = logging.getLogger("training.train_supervised")


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_config(raw: dict) -> SupervisedTrainingConfig:
    """Map YAML config to SupervisedTrainingConfig dataclass."""
    training = raw.get("training", {})
    return SupervisedTrainingConfig(
        max_fragments=raw.get("model", {}).get("max_fragments", 8),
        max_teeth=raw.get("model", {}).get("max_teeth", 32),
        num_epochs=training.get("num_epochs", 200),
        batch_size=training.get("batch_size", 2),
        learning_rate=training.get("learning_rate", 1e-4),
        weight_decay=training.get("weight_decay", 1e-5),
        gradient_clip_norm=training.get("gradient_clip_norm", 1.0),
        gradient_accumulation_steps=training.get("gradient_accumulation_steps", 4),
        phase_transition_epoch=training.get("phase_transition_epoch", 50),
        warmup_epochs=training.get("warmup_epochs", 10),
        cosine_min_lr=training.get("cosine_min_lr", 1e-6),
        use_amp=training.get("use_amp", True),
        val_every_n_epochs=training.get("val_every_n_epochs", 5),
        mc_dropout_passes=training.get("mc_dropout_passes", 10),
        checkpoint_dir=training.get("checkpoint_dir", "checkpoints/supervised"),
        save_every_n_epochs=training.get("save_every_n_epochs", 10),
        keep_top_k=training.get("keep_top_k", 3),
        log_every_n_steps=raw.get("logging", {}).get("log_every_n_steps", 10),
        use_wandb=raw.get("logging", {}).get("use_wandb", False),
        wandb_project=raw.get("logging", {}).get("wandb_project", "facial-align-supervised"),
    )


def build_datasets(raw: dict, use_synthetic: bool = False):
    """Instantiate train/val datasets from config."""
    dataset_cfg = raw.get("dataset", {})
    manifest_path = dataset_cfg.get("manifest_path", "data/training_manifest.json")

    if use_synthetic:
        from training.datasets.ct_ios_dataset import (
            CTIOSDatasetConfig,
            SyntheticFractureDataset,
        )

        synthetic_cfg = raw.get("synthetic", {})
        output_dir = synthetic_cfg.get("output_dir", "data/synthetic_fractures")

        # Load pre-generated synthetic cases from output directory
        cases_file = Path(output_dir) / "cases.json"
        if not cases_file.exists():
            logger.error(
                "Synthetic cases file not found at %s. "
                "Run the fracture generator first to create synthetic data.",
                cases_file,
            )
            sys.exit(1)

        with open(cases_file) as f:
            synthetic_cases_raw = json.load(f)

        # SyntheticFractureDataset expects a list of objects with
        # .case_id, .ground_truth_transforms, .intact_mesh_path, .metadata
        from types import SimpleNamespace

        synthetic_cases = []
        for entry in synthetic_cases_raw:
            case = SimpleNamespace(
                case_id=entry.get("case_id", ""),
                ground_truth_transforms={
                    k: __import__("numpy").array(v)
                    for k, v in entry.get("ground_truth_transforms", {}).items()
                },
                intact_mesh_path=entry.get("intact_mesh_path", ""),
                metadata=entry.get("metadata", {}),
            )
            synthetic_cases.append(case)

        config = CTIOSDatasetConfig(
            max_fragments=dataset_cfg.get("max_fragments", 8),
            max_teeth=dataset_cfg.get("max_teeth", 32),
            points_per_tooth=dataset_cfg.get("points_per_tooth", 1024),
            target_spacing_mm=dataset_cfg.get("target_spacing_mm", 0.4),
            hu_min=dataset_cfg.get("hu_min", 0.0),
            hu_max=dataset_cfg.get("hu_max", 3000.0),
            crop_size=tuple(dataset_cfg.get("crop_size", [128, 128, 128])),
            augment=dataset_cfg.get("augment", True),
            cache_ct=dataset_cfg.get("cache_ct", False),
        )

        dataset = SyntheticFractureDataset(
            synthetic_cases=synthetic_cases,
            config=config,
        )

        # 80/20 split
        import torch
        n_total = len(dataset)
        n_train = int(0.8 * n_total)
        n_val = n_total - n_train
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        return train_dataset, val_dataset

    # Real data from manifest
    if not Path(manifest_path).exists():
        logger.error(
            "Training manifest not found at %s. "
            "Use --synthetic to train on synthetic data, or generate a manifest first.",
            manifest_path,
        )
        sys.exit(1)

    from training.datasets.ct_ios_dataset import CTIOSDatasetConfig, CTIOSPairedDataset

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Split by 'split' field in manifest, or do 80/20
    cases = manifest.get("cases", manifest) if isinstance(manifest, dict) else manifest
    train_entries = [e for e in cases if e.get("split") == "train"]
    val_entries = [e for e in cases if e.get("split") == "val"]

    if not train_entries and not val_entries:
        # No split labels -- auto-split
        import random
        random.seed(42)
        shuffled = list(cases)
        random.shuffle(shuffled)
        split_idx = int(0.8 * len(shuffled))
        train_entries = shuffled[:split_idx]
        val_entries = shuffled[split_idx:]

    # Write split manifests to temp files for CTIOSPairedDataset
    import tempfile
    train_manifest = Path(tempfile.mktemp(suffix=".json"))
    val_manifest = Path(tempfile.mktemp(suffix=".json"))

    with open(train_manifest, "w") as f:
        json.dump({"cases": train_entries}, f)
    with open(val_manifest, "w") as f:
        json.dump({"cases": val_entries}, f)

    train_config = CTIOSDatasetConfig(
        manifest_path=str(train_manifest),
        max_fragments=dataset_cfg.get("max_fragments", 8),
        max_teeth=dataset_cfg.get("max_teeth", 32),
        points_per_tooth=dataset_cfg.get("points_per_tooth", 1024),
        target_spacing_mm=dataset_cfg.get("target_spacing_mm", 0.4),
        hu_min=dataset_cfg.get("hu_min", 0.0),
        hu_max=dataset_cfg.get("hu_max", 3000.0),
        crop_size=tuple(dataset_cfg.get("crop_size", [128, 128, 128])),
        augment=dataset_cfg.get("augment", True),
        cache_ct=dataset_cfg.get("cache_ct", False),
    )

    val_config = CTIOSDatasetConfig(
        manifest_path=str(val_manifest),
        max_fragments=dataset_cfg.get("max_fragments", 8),
        max_teeth=dataset_cfg.get("max_teeth", 32),
        points_per_tooth=dataset_cfg.get("points_per_tooth", 1024),
        target_spacing_mm=dataset_cfg.get("target_spacing_mm", 0.4),
        hu_min=dataset_cfg.get("hu_min", 0.0),
        hu_max=dataset_cfg.get("hu_max", 3000.0),
        crop_size=tuple(dataset_cfg.get("crop_size", [128, 128, 128])),
        augment=False,
        cache_ct=dataset_cfg.get("cache_ct", False),
    )

    train_dataset = CTIOSPairedDataset(config=train_config)
    val_dataset = CTIOSPairedDataset(config=val_config)

    # Clean up temp files
    train_manifest.unlink(missing_ok=True)
    val_manifest.unlink(missing_ok=True)

    return train_dataset, val_dataset


def main():
    parser = argparse.ArgumentParser(description="Train supervised fracture reduction model")
    parser.add_argument(
        "--config",
        type=str,
        default="training/configs/supervised_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic fracture data (no real patient data needed)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override training device (cuda / cpu)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Load config
    logger.info("Loading config from %s", args.config)
    raw_config = load_config(args.config)
    config = build_config(raw_config)

    # CLI overrides
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.wandb:
        config.use_wandb = True

    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")

    # Build datasets
    logger.info("Building datasets (synthetic=%s)...", args.synthetic)
    train_dataset, val_dataset = build_datasets(raw_config, use_synthetic=args.synthetic)
    logger.info("Train: %d samples, Val: %d samples", len(train_dataset), len(val_dataset))

    # Build or resume trainer
    if args.resume:
        logger.info("Resuming from checkpoint: %s", args.resume)
        trainer = SupervisedTrainer.resume_from_checkpoint(args.resume, config=config)
    else:
        trainer = SupervisedTrainer(config=config, device=device)

    # Train
    history = trainer.train(train_dataset, val_dataset)

    # Save training summary
    summary_path = Path(config.checkpoint_dir) / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            "best_val_loss": trainer.best_val_loss,
            "total_epochs": config.num_epochs,
            "best_checkpoint": trainer.best_checkpoint_path,
        }, f, indent=2)

    logger.info("Training summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
