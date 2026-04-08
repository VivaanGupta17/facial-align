"""
CT Preprocessing Service

Handles the full preprocessing pipeline from raw DICOM to ML-ready volumes:
1. DICOM series identification and sorting
2. Volume reconstruction with proper orientation
3. HU calibration and windowing
4. Isotropic resampling
5. Intensity normalization
6. ROI cropping for craniofacial region
7. Quality validation

This runs as a Celery task (long-running) and stores results in MinIO.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for CT preprocessing pipeline."""

    # Resampling
    target_spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0)
    interpolation_order: int = 3  # Cubic for images, 0 for masks

    # Intensity
    hu_clip_range: tuple[int, int] = (-1024, 3071)
    bone_window: tuple[int, int] = (400, 2000)  # center, width
    soft_tissue_window: tuple[int, int] = (40, 400)

    # ROI cropping
    auto_crop_to_head: bool = True
    crop_margin_mm: float = 10.0

    # Quality thresholds
    max_slice_thickness_mm: float = 1.0
    min_coverage_mm: float = 150.0
    min_slices: int = 100

    # Output
    output_format: str = "nifti"  # nifti or numpy
    compress_output: bool = True


@dataclass
class PreprocessingResult:
    """Result of the preprocessing pipeline."""

    success: bool
    volume_path: Optional[str] = None  # Path to preprocessed NIfTI
    original_spacing: Optional[tuple[float, float, float]] = None
    resampled_spacing: Optional[tuple[float, float, float]] = None
    original_shape: Optional[tuple[int, int, int]] = None
    resampled_shape: Optional[tuple[int, int, int]] = None
    hu_range: Optional[tuple[float, float]] = None
    orientation: str = "LPS"
    quality_checks: dict = field(default_factory=dict)
    processing_time_ms: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class CTPreprocessor:
    """
    Full CT preprocessing pipeline for craniofacial imaging.

    Usage:
        preprocessor = CTPreprocessor(config=PreprocessingConfig())
        result = preprocessor.process(dicom_dir="/path/to/dicom", output_dir="/path/to/output")
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()

    def process(self, dicom_dir: str, output_dir: str) -> PreprocessingResult:
        """
        Run the full preprocessing pipeline.

        Args:
            dicom_dir: Directory containing DICOM files
            output_dir: Output directory for preprocessed volume

        Returns:
            PreprocessingResult with paths and quality metrics
        """
        import SimpleITK as sitk

        start_time = time.time()
        result = PreprocessingResult(success=False)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            # Step 1: Load DICOM series
            logger.info("Step 1: Loading DICOM series...")
            image = self._load_dicom_series(dicom_dir)
            result.original_spacing = image.GetSpacing()
            result.original_shape = image.GetSize()

            # Step 2: Orient to LPS standard
            logger.info("Step 2: Reorienting to LPS...")
            image = sitk.DICOMOrient(image, "LPS")
            result.orientation = "LPS"

            # Step 3: Validate HU calibration
            logger.info("Step 3: Validating HU values...")
            array = sitk.GetArrayFromImage(image)
            result.hu_range = (float(array.min()), float(array.max()))

            if array.max() < 100:
                result.warnings.append(
                    f"Max HU={array.max():.0f} — HU calibration may be incorrect. "
                    "Verify RescaleSlope/Intercept were applied."
                )

            # Step 4: Clip HU range
            logger.info("Step 4: Clipping HU range...")
            low, high = self.config.hu_clip_range
            array = np.clip(array, low, high)
            clipped_image = sitk.GetImageFromArray(array)
            clipped_image.CopyInformation(image)

            # Step 5: Resample to isotropic spacing
            logger.info(f"Step 5: Resampling to {self.config.target_spacing_mm}mm isotropic...")
            resampled = self._resample_volume(clipped_image, self.config.target_spacing_mm)
            result.resampled_spacing = resampled.GetSpacing()
            result.resampled_shape = resampled.GetSize()

            # Step 6: Auto-crop to head region (optional)
            if self.config.auto_crop_to_head:
                logger.info("Step 6: Auto-cropping to craniofacial ROI...")
                resampled = self._auto_crop_head(resampled)

            # Step 7: Quality checks
            logger.info("Step 7: Running quality checks...")
            result.quality_checks = self._run_quality_checks(image, resampled)

            # Step 8: Save output
            nifti_path = str(output_path / "ct_preprocessed.nii.gz")
            logger.info(f"Step 8: Saving to {nifti_path}")
            sitk.WriteImage(resampled, nifti_path, useCompression=self.config.compress_output)
            result.volume_path = nifti_path

            result.success = True

        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            result.errors.append(str(e))

        result.processing_time_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"Preprocessing {'succeeded' if result.success else 'FAILED'} "
            f"in {result.processing_time_ms}ms"
        )

        return result

    def _load_dicom_series(self, dicom_dir: str):
        """Load DICOM series and select the best series for CMF planning."""
        import SimpleITK as sitk

        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(dicom_dir)

        if not series_ids:
            raise ValueError(f"No DICOM series found in {dicom_dir}")

        logger.info(f"Found {len(series_ids)} series")

        # If multiple series, select the one with thinnest slices (best for segmentation)
        best_series = None
        best_thickness = float("inf")

        for series_id in series_ids:
            file_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_id)
            if len(file_names) < self.config.min_slices:
                continue

            # Read first file to check thickness
            import pydicom

            ds = pydicom.dcmread(file_names[0], stop_before_pixels=True)
            thickness = float(getattr(ds, "SliceThickness", 999))
            kernel = str(getattr(ds, "ConvolutionKernel", ""))

            # Prefer bone kernel, then thinnest slices
            is_bone = any(k in kernel.upper() for k in ["BONE", "B60", "B70", "B80", "H60", "H70"])
            effective_score = thickness - (10 if is_bone else 0)  # Bonus for bone kernel

            if effective_score < best_thickness:
                best_thickness = effective_score
                best_series = (series_id, file_names, thickness, kernel)

        if best_series is None:
            raise ValueError("No suitable DICOM series found (all below minimum slice count)")

        series_id, file_names, thickness, kernel = best_series
        logger.info(
            f"Selected series: {len(file_names)} slices, "
            f"{thickness}mm thickness, kernel={kernel}"
        )

        reader.SetFileNames(file_names)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()

        return reader.Execute()

    def _resample_volume(self, image, target_spacing: tuple[float, float, float]):
        """Resample volume to target isotropic spacing."""
        import SimpleITK as sitk

        original_spacing = image.GetSpacing()
        original_size = image.GetSize()

        new_size = [
            int(round(osz * ospc / nspc))
            for osz, ospc, nspc in zip(original_size, original_spacing, target_spacing)
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(-1024)
        resampler.SetInterpolator(sitk.sitkBSpline)

        return resampler.Execute(image)

    def _auto_crop_head(self, image):
        """
        Auto-crop volume to the craniofacial region.

        Uses a simple HU thresholding approach: find the bounding box
        of all voxels > 200 HU (bone/dense tissue) and add margin.
        """
        import SimpleITK as sitk

        array = sitk.GetArrayFromImage(image)
        spacing = image.GetSpacing()

        # Threshold for bone
        bone_mask = array > 200

        if bone_mask.sum() == 0:
            logger.warning("No bone detected — skipping auto-crop")
            return image

        # Find bounding box of bone
        coords = np.argwhere(bone_mask)
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)

        # Add margin
        margin_voxels = int(self.config.crop_margin_mm / spacing[0])
        z_min = max(0, z_min - margin_voxels)
        y_min = max(0, y_min - margin_voxels)
        x_min = max(0, x_min - margin_voxels)
        z_max = min(array.shape[0], z_max + margin_voxels)
        y_max = min(array.shape[1], y_max + margin_voxels)
        x_max = min(array.shape[2], x_max + margin_voxels)

        # Crop
        cropped_array = array[z_min:z_max, y_min:y_max, x_min:x_max]
        cropped_image = sitk.GetImageFromArray(cropped_array)
        cropped_image.SetSpacing(spacing)

        # Update origin
        original_origin = np.array(image.GetOrigin())
        crop_offset = np.array([x_min, y_min, z_min]) * np.array(spacing)
        new_origin = original_origin + crop_offset
        cropped_image.SetOrigin(tuple(new_origin))
        cropped_image.SetDirection(image.GetDirection())

        logger.info(
            f"Cropped: {array.shape} → {cropped_array.shape} "
            f"(removed {(1 - cropped_array.size / array.size) * 100:.0f}% of voxels)"
        )

        return cropped_image

    def _run_quality_checks(self, original, preprocessed) -> dict:
        """Run quality validation checks on the preprocessed volume."""
        spacing = original.GetSpacing()
        size = original.GetSize()
        thickness = spacing[2]
        coverage = size[2] * thickness

        checks = {
            "slice_thickness_mm": round(thickness, 3),
            "slice_thickness_ok": thickness <= self.config.max_slice_thickness_mm,
            "coverage_mm": round(coverage, 1),
            "coverage_ok": coverage >= self.config.min_coverage_mm,
            "num_slices": size[2],
            "num_slices_ok": size[2] >= self.config.min_slices,
            "preprocessed_size": preprocessed.GetSize(),
            "preprocessed_spacing": preprocessed.GetSpacing(),
        }

        if not checks["slice_thickness_ok"]:
            logger.warning(
                f"Slice thickness {thickness}mm exceeds {self.config.max_slice_thickness_mm}mm limit"
            )
        if not checks["coverage_ok"]:
            logger.warning(
                f"Coverage {coverage:.0f}mm below {self.config.min_coverage_mm}mm minimum"
            )

        return checks
