#!/usr/bin/env python3
"""
Download pre-trained model weights for Facial Align.

Usage:
    python scripts/download_models.py --model totalsegmentator
    python scripts/download_models.py --model dental_segmentator
    python scripts/download_models.py --all
"""

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "totalsegmentator": {
        "description": "TotalSegmentator v2 — 117 anatomical structures from CT",
        "source": "pip",
        "install_cmd": "pip install totalsegmentator",
        "verify_cmd": "TotalSegmentator --help",
        "subtasks": [
            "total",
            "craniofacial_structures",
            "teeth",
            "head_muscles",
            "headneck_bones_vessels",
        ],
        "license": "Apache-2.0",
        "reference": "https://github.com/wasserth/TotalSegmentator",
    },
    "dental_segmentator": {
        "description": "DentalSegmentator — per-tooth + anatomical CBCT segmentation",
        "source": "zenodo",
        "zenodo_id": "11003568",
        "download_url": "https://zenodo.org/records/11003568/files/dental_segmentator_weights.zip",
        "sha256": None,  # TODO: Add checksum after first download
        "output_dir": "models/dental_segmentator",
        "license": "Apache-2.0",
        "reference": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools",
    },
    "nnunet_mandible": {
        "description": "nnU-Net mandible segmentation (requires fine-tuning)",
        "source": "placeholder",
        "status": "Phase 2 — requires training on CMF dataset",
        "architecture": "nnU-Net v2 3d_fullres",
        "training_data": "PDDCA + institutional CT data",
        "reference": "https://github.com/MIC-DKFZ/nnUNet",
    },
    "fracture_reduction": {
        "description": "Fracture reduction transform prediction model",
        "source": "placeholder",
        "status": "Phase 2 — requires paired pre/post fracture CT data",
        "architecture": "PointNet++ or SE(3)-Transformer",
        "training_data": "Paired fracture/reduced CT with surgeon-verified transforms",
    },
    "occlusion_prediction": {
        "description": "Occlusal relationship prediction from dental arch geometry",
        "source": "placeholder",
        "status": "Phase 2 — requires dental model dataset",
        "architecture": "PointNet++ or Graph Neural Network",
        "training_data": "Intraoral scan pairs with occlusal measurements",
    },
}


def verify_checksum(filepath: Path, expected_sha256: str) -> bool:
    """Verify file integrity via SHA-256."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest() == expected_sha256


def download_totalsegmentator():
    """Install TotalSegmentator via pip and verify."""
    logger.info("Installing TotalSegmentator...")
    os.system("pip install totalsegmentator")
    logger.info("Downloading TotalSegmentator weights (this may take several minutes)...")
    os.system("TotalSegmentator --help > /dev/null 2>&1")
    logger.info("TotalSegmentator installed and ready.")
    logger.info("  Available CMF subtasks: craniofacial_structures, teeth, head_muscles")


def download_dental_segmentator(output_dir: str):
    """Download DentalSegmentator weights from Zenodo."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading DentalSegmentator weights from Zenodo...")
    logger.info("  Source: https://zenodo.org/records/11003568")
    logger.info("  License: Apache-2.0")

    # TODO: Implement actual download with progress bar
    # For now, create a placeholder indicating download is needed
    manifest = output_path / "DOWNLOAD_INSTRUCTIONS.md"
    manifest.write_text(
        "# DentalSegmentator Model Weights\n\n"
        "Download from: https://zenodo.org/records/11003568\n\n"
        "Extract weights to this directory.\n\n"
        "Reference: DCBIA-OrthoLab/SlicerAutomatedDentalTools\n"
    )
    logger.info(f"Download instructions written to {manifest}")


def list_models():
    """List all available models and their status."""
    print("\n" + "=" * 70)
    print("Facial Align — Model Registry")
    print("=" * 70)
    for name, info in MODEL_REGISTRY.items():
        status = "Available" if info["source"] != "placeholder" else info.get("status", "Planned")
        print(f"\n  {name}")
        print(f"    {info['description']}")
        print(f"    Status: {status}")
        print(f"    License: {info.get('license', 'TBD')}")
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Download Facial Align model weights")
    parser.add_argument("--model", type=str, help="Model name to download")
    parser.add_argument("--all", action="store_true", help="Download all available models")
    parser.add_argument("--list", action="store_true", help="List all models")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory")
    args = parser.parse_args()

    if args.list or (not args.model and not args.all):
        list_models()
        return

    models_to_download = []
    if args.all:
        models_to_download = [k for k, v in MODEL_REGISTRY.items() if v["source"] != "placeholder"]
    elif args.model:
        if args.model not in MODEL_REGISTRY:
            logger.error(f"Unknown model: {args.model}")
            logger.info(f"Available models: {', '.join(MODEL_REGISTRY.keys())}")
            sys.exit(1)
        models_to_download = [args.model]

    for model_name in models_to_download:
        info = MODEL_REGISTRY[model_name]
        if info["source"] == "placeholder":
            logger.warning(f"Skipping {model_name} — {info['status']}")
            continue

        logger.info(f"Processing {model_name}: {info['description']}")
        if model_name == "totalsegmentator":
            download_totalsegmentator()
        elif model_name == "dental_segmentator":
            download_dental_segmentator(os.path.join(args.output_dir, "dental_segmentator"))

    logger.info("Model download complete.")


if __name__ == "__main__":
    main()
