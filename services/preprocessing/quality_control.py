"""
CT Quality Control for Craniofacial Surgical Planning
======================================================

CT scan quality directly impacts the accuracy of 3D segmentation and
surgical planning. A scan that is adequate for diagnostic radiology may
be insufficient for computer-aided surgical planning, which requires:
  - Sub-millimetre slice thickness for accurate 3D reconstruction
  - Complete craniofacial coverage (mandible through vertex)
  - Adequate bone contrast for reliable auto-segmentation
  - Minimal motion artifacts to ensure accurate mesh generation

This module implements automated quality control checks that evaluate
incoming CT volumes against the requirements of the Facial Align platform.

Clinical Quality Grades
------------------------
  Grade A — Ideal for surgical planning:
      All checks pass; segmentation and planning can proceed without caveats.

  Grade B — Acceptable with caveats:
      Minor quality issues detected. Segmentation may require manual review.
      Flag to reviewing surgeon with specific concerns.

  Grade C — Marginal quality:
      Significant quality issues that may impact planning accuracy.
      Proceed only with explicit surgeon approval and documented caveats.

  Grade F — Reject:
      Quality issues severe enough to invalidate surgical planning.
      Request rescan or manual measurement.

Algorithm References
---------------------
  - Motion artifact detection: inter-slice variance analysis
    (Ref: Barrett & Keat, RadioGraphics 2004)
  - Slice gap detection: statistical analysis of z-position increments
  - FOV adequacy: anatomical landmark detection using HU thresholding
  - Bone contrast: HU histogram analysis in 300–3000 HU range

Author: Facial Align Engineering
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter, label as scipy_label
from scipy.stats import iqr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quality Thresholds
# ---------------------------------------------------------------------------

# Slice thickness (mm) — per ACR practice guidelines for CT
SLICE_THICKNESS_IDEAL_MM = 0.625
SLICE_THICKNESS_ACCEPTABLE_MM = 1.0
SLICE_THICKNESS_MARGINAL_MM = 1.5

# Coverage (mm) — minimum craniofacial coverage
# Mandible-to-vertex: typically 180–250 mm
COVERAGE_IDEAL_MM = 200.0
COVERAGE_ACCEPTABLE_MM = 180.0
COVERAGE_MARGINAL_MM = 150.0

# FOV (mm) — minimum field-of-view width
FOV_IDEAL_MM = 200.0
FOV_ACCEPTABLE_MM = 180.0

# Bone contrast — percentage of volume in bone HU range (300–3000 HU)
BONE_FRACTION_IDEAL = 0.02      # At least 2% of voxels should be bone
BONE_FRACTION_ACCEPTABLE = 0.01

# Motion artifact — inter-slice variance coefficient of variation
# High inter-slice variance suggests motion between slices
MOTION_CV_IDEAL = 0.15
MOTION_CV_ACCEPTABLE = 0.25
MOTION_CV_REJECT = 0.45

# Missing slices — maximum tolerable gap
MAX_SLICE_GAP_MM = 1.5          # Gaps > this trigger check failure
MAX_MISSING_SLICES_ACCEPTABLE = 3  # Up to 3 missing slices is grade B

# HU calibration
HU_AIR_EXPECTED = -1000
HU_AIR_TOLERANCE = 200          # Air should be within ±200 of -1000
HU_BONE_MIN = 400               # Cortical bone > 400 HU
HU_BONE_TYPICAL = 800           # Dense cortical bone ~800 HU


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Result of a single QC check."""
    name: str
    passed: bool
    grade_contribution: str     # "A", "B", "C", or "F"
    measured_value: Optional[float]
    threshold: Optional[float]
    description: str
    recommendation: str = ""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        val = f"{self.measured_value:.3f}" if self.measured_value is not None else "N/A"
        return f"[{status}] {self.name}: {val} (grade: {self.grade_contribution})"


@dataclass
class QualityReport:
    """
    Comprehensive CT quality assessment report for surgical planning use.

    Clinical Use
    ------------
    The overall_grade determines whether the scan can proceed through the
    Facial Align pipeline:

    Grade A: Green light — proceed with full confidence
    Grade B: Yellow light — proceed with documented caveats to surgeon
    Grade C: Orange light — surgeon must explicitly approve before proceeding
    Grade F: Red light — do not process; request rescan

    All recommendations should be reviewed by a qualified radiologist or
    surgeon before clinical use.
    """
    # Individual check results
    slice_thickness_check: Optional[CheckResult] = None
    slice_gap_check: Optional[CheckResult] = None
    spacing_consistency_check: Optional[CheckResult] = None
    coverage_check: Optional[CheckResult] = None
    fov_check: Optional[CheckResult] = None
    motion_artifact_check: Optional[CheckResult] = None
    bone_contrast_check: Optional[CheckResult] = None
    hu_calibration_check: Optional[CheckResult] = None

    # Summary
    overall_grade: str = "F"          # "A", "B", "C", or "F"
    checks_passed: int = 0
    checks_failed: int = 0
    recommendations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Scan metadata
    volume_shape: Optional[tuple] = None
    voxel_spacing_mm: Optional[tuple] = None
    coverage_mm: Optional[float] = None
    fov_mm: Optional[float] = None
    hu_range: Optional[tuple] = None
    processing_time_ms: float = 0.0

    def all_checks(self) -> list[CheckResult]:
        """Return all non-None check results."""
        return [
            c for c in [
                self.slice_thickness_check,
                self.slice_gap_check,
                self.spacing_consistency_check,
                self.coverage_check,
                self.fov_check,
                self.motion_artifact_check,
                self.bone_contrast_check,
                self.hu_calibration_check,
            ]
            if c is not None
        ]

    def summary(self) -> str:
        """Human-readable quality summary."""
        lines = [
            f"CT Quality Grade: {self.overall_grade}",
            f"Checks: {self.checks_passed} passed, {self.checks_failed} failed",
        ]
        if self.volume_shape:
            lines.append(f"Volume shape: {self.volume_shape}")
        if self.voxel_spacing_mm:
            lines.append(
                f"Spacing: {self.voxel_spacing_mm[0]:.2f} x "
                f"{self.voxel_spacing_mm[1]:.2f} x "
                f"{self.voxel_spacing_mm[2]:.2f} mm"
            )
        if self.coverage_mm:
            lines.append(f"Coverage: {self.coverage_mm:.0f} mm")

        lines.append("\nIndividual checks:")
        for check in self.all_checks():
            lines.append(f"  {check}")

        if self.recommendations:
            lines.append("\nRecommendations:")
            for rec in self.recommendations:
                lines.append(f"  • {rec}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main QC Class
# ---------------------------------------------------------------------------

class CTQualityController:
    """
    Automated CT quality assessment for craniofacial surgical planning.

    Evaluates multiple quality dimensions:
      1. Slice thickness — critical for 3D reconstruction accuracy
      2. Slice gap detection — missing slices corrupt segmentation
      3. Spacing consistency — variable spacing causes geometric distortion
      4. Anatomical coverage — mandible through vertex must be included
      5. Field-of-view adequacy — entire face must be in FOV
      6. Motion artifacts — statistical inter-slice intensity variance
      7. Bone contrast — HU histogram must show adequate bone signal
      8. HU calibration — air and bone should be in expected HU ranges

    Usage:
        qc = CTQualityController()

        # From numpy array + metadata
        report = qc.check_volume(
            volume=ct_array,                    # (Z, Y, X) numpy array
            spacing=(0.5, 0.5, 0.5),            # mm per voxel
            slice_positions=None,               # Optional: from DICOM z-positions
        )

        if report.overall_grade in ("C", "F"):
            raise ValueError(f"CT quality too low: grade {report.overall_grade}")

        print(report.summary())

    Note: All checks operate on 3D numpy arrays and require only numpy/scipy.
    """

    def __init__(
        self,
        slice_thickness_threshold_mm: float = SLICE_THICKNESS_ACCEPTABLE_MM,
        coverage_threshold_mm: float = COVERAGE_ACCEPTABLE_MM,
        motion_cv_threshold: float = MOTION_CV_ACCEPTABLE,
    ):
        """
        Args:
            slice_thickness_threshold_mm: Maximum acceptable slice thickness.
            coverage_threshold_mm: Minimum required craniofacial coverage.
            motion_cv_threshold: Maximum inter-slice variance for motion check.
        """
        self.slice_thickness_threshold = slice_thickness_threshold_mm
        self.coverage_threshold = coverage_threshold_mm
        self.motion_cv_threshold = motion_cv_threshold

    def check_volume(
        self,
        volume: np.ndarray,
        spacing: tuple[float, float, float],
        slice_positions: Optional[list[float]] = None,
    ) -> QualityReport:
        """
        Run all quality checks on a CT volume.

        Args:
            volume: 3D CT array (Z, Y, X) in Hounsfield Units
            spacing: Voxel spacing in mm (z, y, x)
            slice_positions: Optional list of z-coordinates for each slice
                            (from DICOM ImagePositionPatient). If None,
                            uses uniform spacing.

        Returns:
            QualityReport with grades and recommendations
        """
        t_start = time.time()

        sz, sy, sx = spacing
        nz, ny, nx = volume.shape

        report = QualityReport(
            volume_shape=(nz, ny, nx),
            voxel_spacing_mm=spacing,
            hu_range=(float(volume.min()), float(volume.max())),
        )

        # Run all checks
        report.slice_thickness_check = self._check_slice_thickness(sz)
        report.slice_gap_check = self._check_slice_gaps(sz, nz, slice_positions)
        report.spacing_consistency_check = self._check_spacing_consistency(
            volume, sz, slice_positions
        )
        report.coverage_check = self._check_anatomical_coverage(volume, sz)
        report.fov_check = self._check_fov_adequacy(volume, sy, sx)
        report.motion_artifact_check = self._check_motion_artifacts(volume, sz)
        report.bone_contrast_check = self._check_bone_contrast(volume)
        report.hu_calibration_check = self._check_hu_calibration(volume)

        # Store computed values in report
        report.coverage_mm = float(nz * sz)
        report.fov_mm = float(min(ny * sy, nx * sx))

        # Aggregate grade
        self._compute_overall_grade(report)

        report.processing_time_ms = round((time.time() - t_start) * 1000, 1)

        logger.info(
            f"QC complete: grade={report.overall_grade} "
            f"{report.checks_passed}/{report.checks_passed + report.checks_failed} checks passed "
            f"in {report.processing_time_ms:.0f}ms"
        )

        return report

    # ------------------------------------------------------------------
    # Individual Checks
    # ------------------------------------------------------------------

    def _check_slice_thickness(self, slice_thickness_mm: float) -> CheckResult:
        """
        Check 1: Slice Thickness

        Thin slices (≤ 0.625 mm) enable accurate 3D reconstruction.
        Slices > 1.5 mm produce visible staircase artifacts in 3D meshes
        and degrade auto-segmentation accuracy by 15–30%.

        Clinical standard: ACR recommends ≤ 1 mm for facial CT.
        """
        if slice_thickness_mm <= SLICE_THICKNESS_IDEAL_MM:
            grade = "A"
            passed = True
            desc = f"Excellent: slice thickness {slice_thickness_mm:.3f} mm ≤ {SLICE_THICKNESS_IDEAL_MM} mm"
            rec = ""
        elif slice_thickness_mm <= SLICE_THICKNESS_ACCEPTABLE_MM:
            grade = "B"
            passed = True
            desc = f"Acceptable: slice thickness {slice_thickness_mm:.3f} mm (ideal ≤ {SLICE_THICKNESS_IDEAL_MM} mm)"
            rec = "Consider requesting thin-slice reconstruction if available."
        elif slice_thickness_mm <= SLICE_THICKNESS_MARGINAL_MM:
            grade = "C"
            passed = False
            desc = f"Marginal: slice thickness {slice_thickness_mm:.3f} mm exceeds {SLICE_THICKNESS_ACCEPTABLE_MM} mm"
            rec = (
                f"Slice thickness {slice_thickness_mm:.2f} mm may reduce segmentation accuracy. "
                "Request thinner reconstruction (≤ 1 mm) from radiology if possible."
            )
        else:
            grade = "F"
            passed = False
            desc = f"Unacceptable: slice thickness {slice_thickness_mm:.3f} mm is too thick for surgical planning"
            rec = (
                "Slice thickness exceeds 1.5 mm. Request a dedicated craniofacial CT protocol "
                "with 0.5–1.0 mm slice thickness. This scan cannot be used for surgical planning."
            )

        return CheckResult(
            name="Slice Thickness",
            passed=passed,
            grade_contribution=grade,
            measured_value=slice_thickness_mm,
            threshold=SLICE_THICKNESS_ACCEPTABLE_MM,
            description=desc,
            recommendation=rec,
        )

    def _check_slice_gaps(
        self,
        nominal_spacing_mm: float,
        n_slices: int,
        slice_positions: Optional[list[float]],
    ) -> CheckResult:
        """
        Check 2: Missing Slices / Gaps

        Missing slices create discontinuities in 3D reconstructions.
        Detected by analysing inter-slice spacing consistency.

        If slice_positions are available (from DICOM), we can detect
        exact gaps. Otherwise we assume uniform spacing.
        """
        if slice_positions is None or len(slice_positions) < 2:
            # Cannot detect gaps without position data
            return CheckResult(
                name="Slice Gap Detection",
                passed=True,
                grade_contribution="B",
                measured_value=None,
                threshold=MAX_SLICE_GAP_MM,
                description="Slice positions not available — gap detection skipped.",
                recommendation="Provide DICOM z-positions for accurate gap detection.",
            )

        positions = np.array(sorted(slice_positions))
        gaps = np.diff(positions)

        median_gap = float(np.median(gaps))
        max_gap = float(np.max(gaps))

        # A missing slice manifests as a gap > 2× the median gap
        gap_threshold = max(nominal_spacing_mm * 2.0, MAX_SLICE_GAP_MM)
        missing_count = int(np.sum(gaps > gap_threshold))

        if missing_count == 0 and max_gap <= nominal_spacing_mm * 1.2:
            grade = "A"
            passed = True
            desc = f"No missing slices detected. Max gap: {max_gap:.2f} mm"
            rec = ""
        elif missing_count <= MAX_MISSING_SLICES_ACCEPTABLE:
            grade = "B"
            passed = True
            desc = f"{missing_count} potential missing slice(s). Max gap: {max_gap:.2f} mm"
            rec = f"Verify {missing_count} gap(s) in slice sequence. Minor gaps may be acceptable."
        elif missing_count <= 10:
            grade = "C"
            passed = False
            desc = f"{missing_count} missing slices detected. Max gap: {max_gap:.2f} mm"
            rec = (
                f"{missing_count} slices missing from the series. "
                "Proceed with caution — segmentation may have discontinuities."
            )
        else:
            grade = "F"
            passed = False
            desc = f"{missing_count} missing slices — series is incomplete"
            rec = (
                "Excessive missing slices detected. The DICOM series is incomplete. "
                "Request the complete series from radiology."
            )

        return CheckResult(
            name="Slice Gap Detection",
            passed=passed,
            grade_contribution=grade,
            measured_value=max_gap,
            threshold=gap_threshold,
            description=desc,
            recommendation=rec,
        )

    def _check_spacing_consistency(
        self,
        volume: np.ndarray,
        nominal_spacing_mm: float,
        slice_positions: Optional[list[float]],
    ) -> CheckResult:
        """
        Check 3: Spacing Consistency

        Variable inter-slice spacing causes geometric distortion in
        3D surface reconstruction. All slices should be evenly spaced.

        Detects resampled or merged datasets where spacings vary.
        """
        if slice_positions is None or len(slice_positions) < 3:
            return CheckResult(
                name="Spacing Consistency",
                passed=True,
                grade_contribution="B",
                measured_value=None,
                threshold=0.05,
                description="Cannot verify spacing consistency without DICOM z-positions.",
                recommendation="Provide DICOM z-positions for consistency check.",
            )

        positions = np.array(sorted(slice_positions))
        gaps = np.diff(positions)

        cv = float(np.std(gaps) / (np.mean(gaps) + 1e-10))  # Coefficient of variation
        max_deviation_mm = float(np.max(np.abs(gaps - np.mean(gaps))))

        if cv < 0.01:
            grade = "A"
            passed = True
            desc = f"Spacing perfectly consistent (CV={cv:.4f})"
            rec = ""
        elif cv < 0.05:
            grade = "A"
            passed = True
            desc = f"Spacing consistent (CV={cv:.4f}, max deviation={max_deviation_mm:.3f} mm)"
            rec = ""
        elif cv < 0.10:
            grade = "B"
            passed = True
            desc = f"Minor spacing variation (CV={cv:.4f}, max deviation={max_deviation_mm:.3f} mm)"
            rec = "Minor spacing irregularity; verify dataset is not resampled."
        elif cv < 0.20:
            grade = "C"
            passed = False
            desc = f"Significant spacing variation (CV={cv:.4f}, max deviation={max_deviation_mm:.3f} mm)"
            rec = (
                "Irregular slice spacing detected. This may indicate a merged dataset. "
                "Verify with radiology before proceeding."
            )
        else:
            grade = "F"
            passed = False
            desc = f"Unacceptable spacing variation (CV={cv:.4f})"
            rec = (
                "Severely inconsistent slice spacing. The dataset may be corrupted "
                "or improperly resampled. Request raw DICOM data."
            )

        return CheckResult(
            name="Spacing Consistency",
            passed=passed,
            grade_contribution=grade,
            measured_value=round(cv, 4),
            threshold=0.10,
            description=desc,
            recommendation=rec,
        )

    def _check_anatomical_coverage(
        self,
        volume: np.ndarray,
        slice_thickness_mm: float,
    ) -> CheckResult:
        """
        Check 4: Anatomical Coverage

        For full craniofacial planning, the scan must include:
          - Inferior: below the mandibular border (with margin)
          - Superior: at least to vertex of skull
          
        Total coverage for an adult head: typically 200–260 mm.

        Detection method: threshold bone HU and check z-extent of bone structure.
        """
        total_coverage_mm = float(volume.shape[0]) * slice_thickness_mm

        # Detect bone presence (HU > 300) in superior and inferior slices
        bone_mask = volume > 300  # Cortical bone threshold

        if not bone_mask.any():
            return CheckResult(
                name="Anatomical Coverage",
                passed=False,
                grade_contribution="C",
                measured_value=total_coverage_mm,
                threshold=COVERAGE_ACCEPTABLE_MM,
                description="No bone structure detected — cannot verify coverage",
                recommendation="Verify HU calibration (RescaleSlope/Intercept applied correctly).",
            )

        # Find first and last slice containing bone
        bone_slices = np.any(bone_mask.reshape(volume.shape[0], -1), axis=1)
        first_bone_slice = int(np.argmax(bone_slices))
        last_bone_slice = int(len(bone_slices) - 1 - np.argmax(bone_slices[::-1]))

        effective_coverage_mm = (last_bone_slice - first_bone_slice) * slice_thickness_mm

        # Check for mandibular coverage (bone in inferior 30% of volume)
        inferior_region = bone_mask[:int(volume.shape[0] * 0.3)]
        has_inferior_bone = inferior_region.any()

        # Check for superior skull coverage (bone in superior 10% of volume)
        superior_region = bone_mask[int(volume.shape[0] * 0.9):]
        has_superior_bone = superior_region.any()

        if (effective_coverage_mm >= COVERAGE_IDEAL_MM and
                has_inferior_bone and has_superior_bone):
            grade = "A"
            passed = True
            desc = (
                f"Excellent coverage: {effective_coverage_mm:.0f} mm "
                f"(total volume: {total_coverage_mm:.0f} mm)"
            )
            rec = ""
        elif (effective_coverage_mm >= COVERAGE_ACCEPTABLE_MM and
              has_inferior_bone and has_superior_bone):
            grade = "B"
            passed = True
            desc = f"Adequate coverage: {effective_coverage_mm:.0f} mm"
            rec = ""
        elif effective_coverage_mm >= COVERAGE_MARGINAL_MM:
            grade = "C"
            passed = False
            desc = f"Marginal coverage: {effective_coverage_mm:.0f} mm"
            recs = []
            if not has_inferior_bone:
                recs.append("Mandible may be truncated at inferior border")
            if not has_superior_bone:
                recs.append("Cranial vault may be truncated superiorly")
            rec = "; ".join(recs) or "Extend scan coverage for complete craniofacial planning."
        else:
            grade = "F"
            passed = False
            desc = f"Insufficient coverage: {effective_coverage_mm:.0f} mm (need ≥ {COVERAGE_ACCEPTABLE_MM:.0f} mm)"
            rec = (
                "CT coverage is insufficient for craniofacial surgical planning. "
                "Extend scan from below mandible to above vertex of skull."
            )

        return CheckResult(
            name="Anatomical Coverage",
            passed=passed,
            grade_contribution=grade,
            measured_value=round(effective_coverage_mm, 1),
            threshold=COVERAGE_ACCEPTABLE_MM,
            description=desc,
            recommendation=rec,
        )

    def _check_fov_adequacy(
        self,
        volume: np.ndarray,
        row_spacing_mm: float,
        col_spacing_mm: float,
    ) -> CheckResult:
        """
        Check 5: Field of View Adequacy

        The entire face must be within the FOV. A truncated FOV causes
        anatomy to be cut off at the image boundary, making segmentation
        and 3D reconstruction unreliable at the edges.

        Detection: Check whether bone extends to image boundaries
        (truncation artifact).
        """
        fov_y = volume.shape[1] * row_spacing_mm
        fov_x = volume.shape[2] * col_spacing_mm
        min_fov = min(fov_y, fov_x)

        # Check for edge truncation: bone touching image boundary
        bone_mask = volume > 400  # High threshold — only dense cortical bone

        # Check each boundary (2-voxel margin)
        margin = 2
        edge_superior = bone_mask[:, :margin, :].any()
        edge_inferior_vol = bone_mask[:, -margin:, :].any()
        edge_left = bone_mask[:, :, :margin].any()
        edge_right = bone_mask[:, :, -margin:].any()

        truncation_detected = any([edge_superior, edge_inferior_vol, edge_left, edge_right])
        truncation_edges = []
        if edge_superior: truncation_edges.append("anterior")
        if edge_inferior_vol: truncation_edges.append("posterior")
        if edge_left: truncation_edges.append("left")
        if edge_right: truncation_edges.append("right")

        if min_fov >= FOV_IDEAL_MM and not truncation_detected:
            grade = "A"
            passed = True
            desc = f"Adequate FOV: {fov_y:.0f} × {fov_x:.0f} mm, no truncation"
            rec = ""
        elif min_fov >= FOV_ACCEPTABLE_MM and not truncation_detected:
            grade = "B"
            passed = True
            desc = f"Acceptable FOV: {fov_y:.0f} × {fov_x:.0f} mm"
            rec = ""
        elif truncation_detected:
            edge_str = ", ".join(truncation_edges)
            grade = "C"
            passed = False
            desc = f"FOV truncation detected at: {edge_str}"
            rec = (
                f"Anatomy appears truncated at the {edge_str} boundary. "
                "Enlarged FOV reconstruction or rescan may be needed."
            )
        else:
            grade = "C"
            passed = False
            desc = f"FOV may be insufficient: {min_fov:.0f} mm (recommended ≥ {FOV_ACCEPTABLE_MM:.0f} mm)"
            rec = "Request FOV ≥ 200 mm to ensure full facial coverage."

        return CheckResult(
            name="FOV Adequacy",
            passed=passed,
            grade_contribution=grade,
            measured_value=round(min_fov, 1),
            threshold=FOV_ACCEPTABLE_MM,
            description=desc,
            recommendation=rec,
        )

    def _check_motion_artifacts(
        self,
        volume: np.ndarray,
        slice_thickness_mm: float,
    ) -> CheckResult:
        """
        Check 6: Motion Artifacts

        Motion during CT acquisition causes blurring and streak artifacts.
        Detection method: analyse the inter-slice intensity variance pattern.

        In a motion-free scan, adjacent slices should have similar intensity
        distributions. Large jumps in inter-slice variance indicate the patient
        moved between acquisitions.

        Statistical approach:
          1. Compute mean HU intensity per slice
          2. Compute second-order differences (acceleration of intensity change)
          3. Use IQR-based outlier detection for motion spike identification
          4. Report as coefficient of variation of inter-slice differences
        """
        # Subsample slices for efficiency if volume is large
        nz = volume.shape[0]

        # Compute per-slice mean intensity (restricted to central region to
        # avoid edge effects and beam-hardening artifacts)
        ny, nx = volume.shape[1], volume.shape[2]
        y_margin = int(ny * 0.1)
        x_margin = int(nx * 0.1)
        central_volume = volume[:, y_margin:-y_margin, x_margin:-x_margin]

        slice_means = np.array([
            float(np.mean(central_volume[i]))
            for i in range(nz)
        ])

        # Inter-slice differences
        diff1 = np.abs(np.diff(slice_means))

        if len(diff1) < 5:
            return CheckResult(
                name="Motion Artifact Detection",
                passed=True,
                grade_contribution="B",
                measured_value=None,
                threshold=MOTION_CV_ACCEPTABLE,
                description="Insufficient slices for motion analysis",
                recommendation="",
            )

        # Coefficient of variation of inter-slice differences
        mean_diff = float(np.mean(diff1))
        std_diff = float(np.std(diff1))
        cv = std_diff / (mean_diff + 1e-10)

        # Count outlier slices (motion spikes)
        q75 = float(np.percentile(diff1, 75))
        q25 = float(np.percentile(diff1, 25))
        iqr_val = q75 - q25
        spike_threshold = q75 + 3.0 * iqr_val
        n_spikes = int(np.sum(diff1 > spike_threshold))

        # Grade based on both CV and spike count
        spike_fraction = n_spikes / len(diff1) if len(diff1) > 0 else 0.0

        if cv < MOTION_CV_IDEAL and n_spikes == 0:
            grade = "A"
            passed = True
            desc = f"No motion detected (CV={cv:.3f}, 0 intensity spikes)"
            rec = ""
        elif cv < MOTION_CV_ACCEPTABLE and n_spikes <= 3:
            grade = "B"
            passed = True
            desc = f"Minimal motion (CV={cv:.3f}, {n_spikes} spike(s))"
            rec = ""
        elif cv < MOTION_CV_REJECT and n_spikes <= 10:
            grade = "C"
            passed = False
            desc = f"Possible motion artifacts (CV={cv:.3f}, {n_spikes} intensity spike(s))"
            rec = (
                "Motion-like artifacts detected. Visual inspection recommended. "
                f"Affected approximately {n_spikes} inter-slice transitions."
            )
        else:
            grade = "F"
            passed = False
            desc = f"Severe motion artifacts (CV={cv:.3f}, {n_spikes} spikes)"
            rec = (
                "Significant motion artifacts detected. Patient motion during acquisition "
                "has severely degraded image quality. Rescan with motion control protocols "
                "(shorter acquisition, patient sedation if appropriate)."
            )

        return CheckResult(
            name="Motion Artifact Detection",
            passed=passed,
            grade_contribution=grade,
            measured_value=round(cv, 4),
            threshold=MOTION_CV_ACCEPTABLE,
            description=desc,
            recommendation=rec,
        )

    def _check_bone_contrast(self, volume: np.ndarray) -> CheckResult:
        """
        Check 7: Bone Contrast

        Adequate bone contrast (sufficient HU separation between bone and
        soft tissue) is essential for reliable auto-segmentation.

        Analysis:
          1. Compute HU histogram in the range [-200, 3000]
          2. Identify the bone peak (300–3000 HU) fraction
          3. Assess the bone-to-soft-tissue contrast ratio

        Typical craniofacial CT values:
          Air: -1000 HU
          Soft tissue: 0–100 HU
          Spongy bone: 200–400 HU
          Cortical bone: 600–2000 HU
          Dental hardware: 1500–3000+ HU
        """
        total_voxels = volume.size

        # Bone voxels: HU > 300
        bone_voxels = int(np.sum(volume >= 300))
        bone_fraction = bone_voxels / total_voxels

        # Dense cortical bone: HU > 600
        cortical_voxels = int(np.sum(volume >= 600))
        cortical_fraction = cortical_voxels / total_voxels

        # Soft tissue mean (restrict to 0–100 HU range)
        soft_tissue_mask = (volume >= 0) & (volume <= 100)
        if soft_tissue_mask.any():
            soft_tissue_mean = float(np.mean(volume[soft_tissue_mask]))
        else:
            soft_tissue_mean = 50.0

        # Bone mean
        bone_mask = volume >= 300
        if bone_mask.any():
            bone_mean = float(np.mean(volume[bone_mask]))
            contrast_ratio = (bone_mean - soft_tissue_mean) / (abs(soft_tissue_mean) + 1)
        else:
            bone_mean = 0.0
            contrast_ratio = 0.0

        if (bone_fraction >= BONE_FRACTION_IDEAL and
                cortical_fraction >= 0.005 and
                bone_mean >= HU_BONE_TYPICAL):
            grade = "A"
            passed = True
            desc = (
                f"Excellent bone contrast: {bone_fraction*100:.1f}% bone voxels, "
                f"mean bone HU={bone_mean:.0f}"
            )
            rec = ""
        elif bone_fraction >= BONE_FRACTION_ACCEPTABLE:
            grade = "B"
            passed = True
            desc = (
                f"Adequate bone contrast: {bone_fraction*100:.1f}% bone voxels, "
                f"mean bone HU={bone_mean:.0f}"
            )
            rec = ""
        elif bone_fraction >= 0.005:
            grade = "C"
            passed = False
            desc = (
                f"Reduced bone contrast: {bone_fraction*100:.2f}% bone voxels, "
                f"mean bone HU={bone_mean:.0f}"
            )
            rec = (
                "Low bone signal detected. Verify the scan uses a bone reconstruction kernel "
                "(e.g., B60, B70, H60). Soft-tissue kernels reduce bone contrast."
            )
        else:
            grade = "F"
            passed = False
            desc = f"Insufficient bone signal: {bone_fraction*100:.3f}% bone voxels"
            rec = (
                "Bone is barely detectable in this scan. Verify HU calibration "
                "(RescaleSlope and RescaleIntercept must be applied). "
                "Use a dedicated bone reconstruction algorithm."
            )

        return CheckResult(
            name="Bone Contrast",
            passed=passed,
            grade_contribution=grade,
            measured_value=round(bone_fraction, 4),
            threshold=BONE_FRACTION_ACCEPTABLE,
            description=desc,
            recommendation=rec,
        )

    def _check_hu_calibration(self, volume: np.ndarray) -> CheckResult:
        """
        Check 8: HU Calibration

        All CT pixel values should be in Hounsfield Units after applying
        the DICOM RescaleSlope and RescaleIntercept transformation.

        Expected values:
          - Air (outside patient): -1000 ± 200 HU
          - Water: 0 ± 50 HU
          - Cortical bone: 600–2000 HU

        If calibration tags were not applied, values may be in raw ADU
        (arbitrary detector units), making segmentation impossible.
        """
        hu_min = float(volume.min())
        hu_max = float(volume.max())

        # Check air value: background voxels should be near -1000 HU
        # Background = 5th percentile of all voxels (mostly air for head CT)
        p5 = float(np.percentile(volume, 5))
        p95 = float(np.percentile(volume, 95))

        air_ok = abs(p5 - HU_AIR_EXPECTED) < HU_AIR_TOLERANCE
        bone_present = p95 >= HU_BONE_MIN
        range_reasonable = hu_min >= -2000 and hu_max <= 4000

        if air_ok and bone_present and range_reasonable:
            grade = "A"
            passed = True
            desc = (
                f"HU calibration valid: air≈{p5:.0f} HU, "
                f"bone>{HU_BONE_MIN:.0f} HU, range=[{hu_min:.0f}, {hu_max:.0f}]"
            )
            rec = ""
        elif air_ok and range_reasonable:
            grade = "B"
            passed = True
            desc = f"HU calibration appears valid: air≈{p5:.0f} HU, range=[{hu_min:.0f}, {hu_max:.0f}]"
            rec = ""
        elif not air_ok and abs(p5) < 200:
            grade = "C"
            passed = False
            desc = f"Air HU={p5:.0f} (expected ~{HU_AIR_EXPECTED} HU) — possible calibration issue"
            rec = (
                "Background air should be approximately -1000 HU. "
                "Verify DICOM RescaleIntercept was applied correctly. "
                "Raw pixel values without calibration will give incorrect HU measurements."
            )
        elif not range_reasonable:
            grade = "F"
            passed = False
            desc = f"HU range [{hu_min:.0f}, {hu_max:.0f}] is outside expected bounds"
            rec = (
                "Pixel values are outside the expected Hounsfield Unit range. "
                "Ensure DICOM RescaleSlope and RescaleIntercept have been applied. "
                "This scan cannot be used for quantitative analysis without recalibration."
            )
        else:
            grade = "C"
            passed = False
            desc = f"HU calibration uncertain: p5={p5:.0f}, p95={p95:.0f} HU"
            rec = "Verify HU calibration with the scanning facility."

        return CheckResult(
            name="HU Calibration",
            passed=passed,
            grade_contribution=grade,
            measured_value=round(p5, 1),
            threshold=float(HU_AIR_EXPECTED),
            description=desc,
            recommendation=rec,
        )

    # ------------------------------------------------------------------
    # Grade Aggregation
    # ------------------------------------------------------------------

    def _compute_overall_grade(self, report: QualityReport) -> None:
        """
        Compute overall quality grade from individual check results.

        Grading logic:
          - Any Grade F → overall grade F
          - 3+ Grade C → overall grade F
          - Any Grade C → overall grade C (unless no Grade B or F)
          - All Grade A → overall grade A
          - Mix of A and B → overall grade B
        """
        checks = report.all_checks()

        grades = [c.grade_contribution for c in checks]
        report.checks_passed = sum(1 for c in checks if c.passed)
        report.checks_failed = sum(1 for c in checks if not c.passed)

        # Collect all recommendations
        for check in checks:
            if check.recommendation:
                report.recommendations.append(f"[{check.name}] {check.recommendation}")

        # Grade determination
        if "F" in grades:
            overall = "F"
        elif grades.count("C") >= 3:
            overall = "F"
        elif "C" in grades:
            overall = "C"
        elif "B" in grades:
            overall = "B"
        else:
            overall = "A"

        report.overall_grade = overall

        # Add grade-specific warnings
        if overall == "A":
            report.warnings = []
        elif overall == "B":
            report.warnings.append(
                "Grade B: Acceptable quality. Segmentation results should be reviewed "
                "before finalising surgical plan."
            )
        elif overall == "C":
            report.warnings.append(
                "Grade C: Marginal quality. Surgeon must explicitly review and approve "
                "segmentation results before proceeding with planning."
            )
            report.warnings.append(
                "Consider requesting a higher-quality scan to ensure planning accuracy."
            )
        else:  # F
            report.warnings.append(
                "Grade F: Quality insufficient for surgical planning. "
                "Do NOT proceed without rescan or manual measurement."
            )

    @staticmethod
    def grade_description(grade: str) -> str:
        """Human-readable description of a quality grade."""
        descriptions = {
            "A": "Grade A — Ideal for surgical planning. All quality checks passed.",
            "B": "Grade B — Acceptable with caveats. Minor issues detected; review recommended.",
            "C": "Grade C — Marginal quality. Significant issues; surgeon approval required.",
            "F": "Grade F — Rejected. Quality insufficient for surgical planning.",
        }
        return descriptions.get(grade, f"Unknown grade: {grade}")
