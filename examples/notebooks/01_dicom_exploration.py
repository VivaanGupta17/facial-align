"""
Facial Align — DICOM Exploration Example
=========================================

This script demonstrates the DICOM ingestion and exploration pipeline.
Run as a script or convert to a Jupyter notebook with `jupytext`.

Prerequisites:
    pip install pydicom SimpleITK numpy matplotlib

Usage:
    python examples/notebooks/01_dicom_exploration.py --dicom-dir /path/to/dicom/
"""

# %% [markdown]
# # DICOM Exploration Pipeline
# This notebook walks through the core DICOM ingestion steps:
# 1. Load and parse DICOM metadata
# 2. Reconstruct 3D volume from DICOM series
# 3. Validate CT quality for CMF planning
# 4. Visualize axial, coronal, sagittal slices

# %% Imports
import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dicom_exploration")

# %% DICOM Metadata Parsing
def parse_dicom_metadata(dicom_dir: str) -> dict:
    """
    Parse DICOM metadata from a directory of .dcm files.

    Returns study-level metadata without loading pixel data.
    """
    import pydicom

    dicom_path = Path(dicom_dir)
    dcm_files = list(dicom_path.glob("**/*.dcm"))

    if not dcm_files:
        # Try files without .dcm extension (common in clinical exports)
        dcm_files = [f for f in dicom_path.iterdir() if f.is_file()]
        dcm_files = [f for f in dcm_files if not f.name.startswith(".")]

    logger.info(f"Found {len(dcm_files)} files in {dicom_dir}")

    # Read first file for study-level metadata (stop before pixel data for speed)
    ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)

    metadata = {
        "study_uid": str(getattr(ds, "StudyInstanceUID", "unknown")),
        "patient_id": "[DEIDENTIFIED]",  # Never log real patient IDs
        "modality": str(getattr(ds, "Modality", "unknown")),
        "study_description": str(getattr(ds, "StudyDescription", "")),
        "study_date": str(getattr(ds, "StudyDate", "")),
        "manufacturer": str(getattr(ds, "Manufacturer", "")),
        "model": str(getattr(ds, "ManufacturerModelName", "")),
        "slice_thickness": float(getattr(ds, "SliceThickness", 0)),
        "pixel_spacing": [float(x) for x in getattr(ds, "PixelSpacing", [0, 0])],
        "rows": int(getattr(ds, "Rows", 0)),
        "columns": int(getattr(ds, "Columns", 0)),
        "kvp": float(getattr(ds, "KVP", 0)),
        "reconstruction_kernel": str(getattr(ds, "ConvolutionKernel", "")),
        "num_files": len(dcm_files),
    }

    logger.info(f"Study: {metadata['study_description']}")
    logger.info(f"Modality: {metadata['modality']}")
    logger.info(f"Manufacturer: {metadata['manufacturer']} {metadata['model']}")
    logger.info(f"Slice thickness: {metadata['slice_thickness']} mm")
    logger.info(f"Matrix: {metadata['rows']}x{metadata['columns']}")
    logger.info(f"Kernel: {metadata['reconstruction_kernel']}")

    return metadata


# %% Volume Reconstruction
def reconstruct_volume(dicom_dir: str) -> tuple:
    """
    Reconstruct a 3D volume from a DICOM series using SimpleITK.

    Returns:
        (volume_array, spacing, origin, direction)
    """
    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)

    if not dicom_names:
        raise ValueError(f"No DICOM series found in {dicom_dir}")

    logger.info(f"Loading {len(dicom_names)} slices...")
    reader.SetFileNames(dicom_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    image = reader.Execute()

    # Reorient to LPS (standard for medical imaging)
    image_lps = sitk.DICOMOrient(image, "LPS")

    # Get volume properties
    spacing = image_lps.GetSpacing()
    origin = image_lps.GetOrigin()
    direction = image_lps.GetDirection()
    size = image_lps.GetSize()

    logger.info(f"Volume size: {size}")
    logger.info(f"Spacing (mm): {spacing}")
    logger.info(f"Origin (mm): {origin}")

    # Convert to numpy array (HU values)
    volume = sitk.GetArrayFromImage(image_lps)  # Shape: (Z, Y, X)

    logger.info(f"Array shape: {volume.shape}")
    logger.info(f"HU range: [{volume.min()}, {volume.max()}]")

    return volume, spacing, origin, direction


# %% Quality Validation
def validate_ct_quality(metadata: dict, volume: np.ndarray) -> dict:
    """
    Validate CT acquisition quality for craniofacial planning.

    Requirements for CMF planning:
    - Slice thickness ≤ 1.0mm (ideal: ≤ 0.625mm)
    - Bone reconstruction kernel
    - Adequate craniofacial coverage
    - Minimal metal artifact
    """
    issues = []
    warnings = []

    # Slice thickness check
    thickness = metadata.get("slice_thickness", 999)
    if thickness > 1.0:
        issues.append(f"Slice thickness {thickness}mm exceeds 1.0mm maximum for CMF planning")
    elif thickness > 0.625:
        warnings.append(f"Slice thickness {thickness}mm — 0.625mm or thinner recommended")

    # Bone kernel check
    kernel = metadata.get("reconstruction_kernel", "").upper()
    bone_kernels = {"BONE", "BONEPLUS", "B60S", "B70S", "B80S", "H60S", "H70S"}
    has_bone_kernel = any(bk in kernel for bk in bone_kernels)
    if not has_bone_kernel:
        warnings.append(f"No bone kernel detected (found: {kernel}). Bone kernel recommended.")

    # Coverage check (rough: at least 200 slices for CMF coverage)
    z_slices = volume.shape[0]
    coverage_mm = z_slices * thickness
    if coverage_mm < 150:
        issues.append(f"Insufficient coverage: {coverage_mm:.0f}mm — need ≥150mm for full CMF")

    # HU range check (should include bone range)
    hu_max = volume.max()
    if hu_max < 1000:
        warnings.append(f"Max HU = {hu_max} — expected >1000 for bone. Check HU calibration.")

    # Metal artifact check (very high HU values)
    metal_threshold = 3000
    metal_voxels = np.sum(volume > metal_threshold)
    metal_fraction = metal_voxels / volume.size
    has_metal = metal_fraction > 0.0001  # > 0.01% of voxels

    if has_metal:
        warnings.append(
            f"Metal artifact detected ({metal_fraction*100:.3f}% of voxels > {metal_threshold} HU). "
            "May affect segmentation accuracy around dental hardware."
        )

    result = {
        "is_valid": len(issues) == 0,
        "has_bone_kernel": has_bone_kernel,
        "slice_thickness_acceptable": thickness <= 1.0,
        "coverage_adequate": coverage_mm >= 150,
        "metal_artifact_detected": has_metal,
        "coverage_mm": coverage_mm,
        "issues": issues,
        "warnings": warnings,
    }

    if result["is_valid"]:
        logger.info("✓ CT quality validation PASSED")
    else:
        logger.warning(f"✗ CT quality validation FAILED: {issues}")

    for w in warnings:
        logger.warning(f"  ⚠ {w}")

    return result


# %% Visualization Helpers
def visualize_slices(volume: np.ndarray, spacing: tuple, output_dir: str = None):
    """
    Generate axial, coronal, and sagittal slice visualizations.

    Uses bone window (center=400, width=2000) for CMF visualization.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping visualization")
        return

    # Bone window
    wc, ww = 400, 2000
    vmin = wc - ww // 2
    vmax = wc + ww // 2

    z, y, x = volume.shape
    mid_z, mid_y, mid_x = z // 2, y // 2, x // 2

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Axial (top-down)
    axes[0].imshow(volume[mid_z, :, :], cmap="bone", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Axial (slice {mid_z}/{z})")
    axes[0].set_xlabel(f"L ← → R (spacing: {spacing[0]:.2f}mm)")
    axes[0].set_ylabel(f"A ← → P (spacing: {spacing[1]:.2f}mm)")

    # Coronal (front view)
    axes[1].imshow(volume[:, mid_y, :], cmap="bone", vmin=vmin, vmax=vmax,
                   aspect=spacing[2] / spacing[0])
    axes[1].set_title(f"Coronal (slice {mid_y}/{y})")
    axes[1].set_xlabel(f"L ← → R (spacing: {spacing[0]:.2f}mm)")
    axes[1].set_ylabel(f"S ← → I (spacing: {spacing[2]:.2f}mm)")

    # Sagittal (side view)
    axes[2].imshow(volume[:, :, mid_x], cmap="bone", vmin=vmin, vmax=vmax,
                   aspect=spacing[2] / spacing[1])
    axes[2].set_title(f"Sagittal (slice {mid_x}/{x})")
    axes[2].set_xlabel(f"A ← → P (spacing: {spacing[1]:.2f}mm)")
    axes[2].set_ylabel(f"S ← → I (spacing: {spacing[2]:.2f}mm)")

    plt.suptitle("CT Volume — Bone Window (WC=400, WW=2000)", fontsize=14)
    plt.tight_layout()

    if output_dir:
        output_path = Path(output_dir) / "ct_slices.png"
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info(f"Saved visualization to {output_path}")
    else:
        plt.show()


# %% Main
def main():
    parser = argparse.ArgumentParser(description="DICOM Exploration Pipeline")
    parser.add_argument("--dicom-dir", type=str, required=True, help="Path to DICOM directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for visualizations")
    args = parser.parse_args()

    # Step 1: Parse metadata
    logger.info("=" * 60)
    logger.info("Step 1: Parsing DICOM metadata")
    logger.info("=" * 60)
    metadata = parse_dicom_metadata(args.dicom_dir)

    # Step 2: Reconstruct volume
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Reconstructing 3D volume")
    logger.info("=" * 60)
    volume, spacing, origin, direction = reconstruct_volume(args.dicom_dir)

    # Step 3: Validate quality
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Validating CT quality")
    logger.info("=" * 60)
    quality = validate_ct_quality(metadata, volume)

    # Step 4: Visualize
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Generating visualizations")
    logger.info("=" * 60)
    visualize_slices(volume, spacing, args.output_dir)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"Volume shape: {volume.shape}")
    logger.info(f"Spacing: {spacing} mm")
    logger.info(f"Quality valid: {quality['is_valid']}")
    logger.info(f"Ready for segmentation: {quality['is_valid']}")


if __name__ == "__main__":
    main()
