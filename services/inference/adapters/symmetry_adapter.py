"""
Facial Symmetry Analysis Adapter
=================================

Quantitative facial symmetry analysis is a critical component of orthognathic
and craniofacial surgical planning. True facial symmetry is rare; mild asymmetry
(< 2 mm) is normal. Clinically significant asymmetry (> 4–6 mm) in orbital, 
zygomatic, or mandibular structures often requires surgical correction.

Clinical Background
-------------------
Facial asymmetry arises from several sources:
  1. Skeletal discrepancy (condylar hyperplasia, hemifacial microsomia)
  2. Dentoalveolar compensation (dental midline shift)
  3. Soft tissue variation (muscular hypertrophy, ptosis)

This module provides:
  1. Midsagittal plane detection using PCA + iterative symmetry refinement
  2. Volumetric mirroring of anatomy for asymmetry map generation
  3. Per-structure asymmetry quantification (orbital, zygomatic, mandibular)
  4. A SymmetryReport dataclass for downstream surgical planning
  5. Clinical grade assignments for detected asymmetry levels

Method
------
The midsagittal plane (MSP) is the vertical plane that best divides the face
into symmetric left and right halves. Detection approach:

  Phase 1: PCA on bone voxel centroid positions finds approximate MSP normal
  Phase 2: Iterative refinement minimises volumetric asymmetry by rotating/
           translating the plane until left-right intensity correlation is maximised

Per-structure asymmetry is then computed by:
  1. Reflecting the structure mask across the MSP
  2. Computing the Hausdorff distance and mean surface distance between
     the original and reflected structure boundaries

All computations use numpy and scipy only.

Reference:
  Thiesen G et al. "Facial asymmetry assessment of 3D surface images." 
  Dental Press J Orthod. 2015;20(6):58-67.

Author: Facial Align Engineering
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    center_of_mass,
    distance_transform_edt,
    label as scipy_label,
)
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Asymmetry severity thresholds (millimetres)
# ---------------------------------------------------------------------------

ASYMMETRY_GRADES = {
    "normal":   (0.0, 2.0,   "Normal — within physiological variation (< 2 mm)"),
    "mild":     (2.0, 4.0,   "Mild asymmetry — clinically apparent but may not require correction"),
    "moderate": (4.0, 6.0,   "Moderate asymmetry — surgical evaluation recommended"),
    "severe":   (6.0, 999.0, "Severe asymmetry — surgical correction typically indicated"),
}


def classify_asymmetry(mean_distance_mm: float) -> tuple[str, str]:
    """Return (grade_name, description) for a mean asymmetry distance in mm."""
    for grade, (lo, hi, desc) in ASYMMETRY_GRADES.items():
        if lo <= mean_distance_mm < hi:
            return grade, desc
    return "severe", ASYMMETRY_GRADES["severe"][2]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MidsagittalPlane:
    """
    Geometric representation of the midsagittal plane.

    The plane is defined by a point on the plane and a unit normal vector.
    Points on the plane satisfy: dot(normal, point - origin) = 0

    Anatomically, the normal points from left to right (positive = patient right).
    """
    normal: np.ndarray          # Unit normal vector (3,) — left-to-right direction
    point: np.ndarray           # A point on the plane (3,) — usually centroid
    refinement_iterations: int = 0
    final_symmetry_score: float = 0.0   # 0=perfect asymmetry, 1=perfect symmetry
    detection_method: str = "pca"


@dataclass
class StructureAsymmetry:
    """
    Per-structure asymmetry measurements.

    Computed by reflecting the structure across the midsagittal plane
    and measuring the distance between original and reflected surfaces.
    """
    structure_name: str
    mean_surface_distance_mm: float       # Average distance original vs mirrored
    hausdorff_distance_mm: float          # 95th percentile Hausdorff distance
    max_distance_mm: float                # Maximum point-to-surface distance
    volume_asymmetry_ratio: float         # (V_L - V_R) / ((V_L + V_R) / 2)
    centroid_offset_mm: float             # Distance between left/right centroids
    asymmetry_grade: str                  # "normal", "mild", "moderate", "severe"
    asymmetry_description: str
    dominant_side: str                    # "left" or "right" (larger volume)


@dataclass
class SymmetryReport:
    """
    Comprehensive facial symmetry analysis report.

    Clinical Summary Format
    -----------------------
    Overall symmetry score:  0.0 (perfectly asymmetric) — 1.0 (perfectly symmetric)
    A score > 0.85 is considered clinically normal facial symmetry.

    Per-structure measurements show which anatomical regions contribute most
    to global asymmetry and guide surgical planning decisions.
    """
    # Midsagittal plane
    midsagittal_plane_normal: np.ndarray     # Unit normal (left-right direction)
    midsagittal_plane_point: np.ndarray      # Point on the plane (mm)

    # Global symmetry metrics
    overall_symmetry_score: float            # 0.0–1.0
    overall_grade: str                       # "normal" / "mild" / "moderate" / "severe"

    # Per-structure breakdown
    per_structure_asymmetry: dict[str, StructureAsymmetry] = field(default_factory=dict)
    per_structure_asymmetry_mm: dict[str, float] = field(default_factory=dict)  # {name: mean_dist}

    # Asymmetry maps (voxel arrays of distance values, same shape as input mask)
    asymmetry_map_voxels: Optional[np.ndarray] = None   # Distance map in mm units

    # Clinical annotations
    clinical_summary: str = ""
    warnings: list[str] = field(default_factory=list)
    processing_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Main Analyser Class
# ---------------------------------------------------------------------------

class FacialSymmetryAnalyzer:
    """
    Quantitative facial symmetry analysis from CT volume and segmentation masks.

    Detects the midsagittal plane, mirrors anatomy, and computes per-structure
    asymmetry metrics for clinical reporting.

    Usage:
        analyzer = FacialSymmetryAnalyzer()

        report = analyzer.analyze(
            volume=ct_array,                    # (Z, Y, X) HU values
            segmentation_mask=seg_mask,         # (Z, Y, X) integer labels
            spacing=(0.5, 0.5, 0.5),            # mm per voxel (z, y, x)
        )

        print(f"Overall symmetry score: {report.overall_symmetry_score:.3f}")
        print(f"Mandibular asymmetry: {report.per_structure_asymmetry_mm.get('mandible', 0):.1f} mm")

    Architecture note:
        This class uses only numpy/scipy. No deep learning is required.
        Midsagittal plane detection works on bone HU threshold or segmentation mask.
    """

    # Label IDs that match the segmentation model's output
    LABEL_MANDIBLE = 1
    LABEL_MAXILLA = 2
    LABEL_SKULL = 3
    LABEL_ORBIT_L = 4
    LABEL_ORBIT_R = 5
    LABEL_ZYGOMATIC_L = 6
    LABEL_ZYGOMATIC_R = 7

    # Bone HU threshold for PCA-based MSP detection without labels
    BONE_HU_THRESHOLD = 300

    # Iterative refinement parameters
    MAX_REFINEMENT_ITERATIONS = 20
    REFINEMENT_CONVERGENCE_MM = 0.1   # Stop if plane moves < 0.1 mm

    def __init__(
        self,
        bone_hu_threshold: float = 300.0,
        refinement_iterations: int = 15,
    ):
        """
        Args:
            bone_hu_threshold: HU threshold for bone detection in PCA step.
            refinement_iterations: Maximum iterations for MSP refinement.
        """
        self.bone_hu_threshold = bone_hu_threshold
        self.refinement_iterations = refinement_iterations

    def detect_midsagittal_plane(
        self,
        volume: np.ndarray,
        segmentation_mask: Optional[np.ndarray] = None,
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> MidsagittalPlane:
        """
        Detect the midsagittal plane (MSP) of the craniofacial skeleton.

        Algorithm:
        1. Extract bone point cloud (HU > threshold or from skull/mandible mask)
        2. Apply PCA to find the principal axis perpendicular to left-right direction
        3. Iteratively refine the plane by maximising bilateral symmetry

        Args:
            volume: CT array (Z, Y, X) in HU
            segmentation_mask: Optional label mask; if provided, uses bone labels
            spacing: Voxel spacing in mm (z, y, x)

        Returns:
            MidsagittalPlane with normal, point, and quality metrics
        """
        sz, sy, sx = spacing

        # --- Step 1: Extract bone point cloud ----------------------------
        if segmentation_mask is not None:
            bone_mask = (
                (segmentation_mask == self.LABEL_SKULL) |
                (segmentation_mask == self.LABEL_MANDIBLE) |
                (segmentation_mask == self.LABEL_MAXILLA)
            )
        else:
            bone_mask = volume >= self.bone_hu_threshold

        if not bone_mask.any():
            logger.warning("No bone structure detected. Falling back to volume centre.")
            centre_vox = np.array(volume.shape) / 2.0
            return MidsagittalPlane(
                normal=np.array([0.0, 0.0, 1.0]),  # left-right = x axis
                point=centre_vox * np.array([sz, sy, sx]),
                detection_method="fallback_centre",
            )

        # Subsample bone points to manage memory (max 50,000 points)
        bone_coords_vox = np.argwhere(bone_mask)
        if len(bone_coords_vox) > 50_000:
            idx = np.random.choice(len(bone_coords_vox), 50_000, replace=False)
            bone_coords_vox = bone_coords_vox[idx]

        # Convert to physical coordinates
        bone_coords_mm = bone_coords_vox * np.array([sz, sy, sx])

        # --- Step 2: PCA ---------------------------------------------------
        centroid = bone_coords_mm.mean(axis=0)
        centred = bone_coords_mm - centroid

        cov = np.cov(centred.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort eigenvectors by eigenvalue descending
        order = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, order]

        # The midsagittal plane normal is the eigenvector corresponding to
        # the smallest variance (left-right spread is minimised in symmetric faces)
        # In a symmetric face: max variance = superior-inferior (axis 0)
        #                     mid variance = anterior-posterior (axis 1)
        #                     min variance = left-right (axis 2)
        msp_normal = eigenvectors[:, 2]  # Smallest eigenvalue → left-right

        # Ensure the normal points in the positive x direction (patient right)
        if msp_normal[2] < 0:
            msp_normal = -msp_normal

        msp = MidsagittalPlane(
            normal=msp_normal.copy(),
            point=centroid.copy(),
            detection_method="pca",
        )

        # --- Step 3: Iterative refinement ---------------------------------
        msp = self._refine_msp_iterative(msp, bone_coords_mm, spacing)

        return msp

    def _refine_msp_iterative(
        self,
        msp: MidsagittalPlane,
        bone_coords_mm: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> MidsagittalPlane:
        """
        Iteratively refine the MSP by maximising bilateral symmetry.

        At each iteration:
          1. Reflect all bone points across the current MSP
          2. Compute the correlation between original and reflected point density
          3. Adjust the plane normal to reduce asymmetry

        This is a simplified gradient-free optimisation that works well for
        near-symmetric faces. For severe asymmetry cases, convergence may be
        slower and the result should be verified.
        """
        current_normal = msp.normal.copy()
        current_point = msp.point.copy()
        prev_score = 0.0

        for iteration in range(self.refinement_iterations):
            # Reflect points across current plane
            reflected = self._reflect_points_across_plane(
                bone_coords_mm, current_normal, current_point
            )

            # Compute symmetry score: normalised cross-correlation
            # between original and reflected point density
            score = self._compute_symmetry_score(bone_coords_mm, reflected)

            if abs(score - prev_score) < 1e-4:
                logger.debug(f"MSP refinement converged at iteration {iteration}")
                break

            # Adjust plane: shift centre to mean of original + reflected centroids
            orig_centroid = bone_coords_mm.mean(axis=0)
            refl_centroid = reflected.mean(axis=0)
            midpoint = (orig_centroid + refl_centroid) / 2.0

            # Small correction to normal: rotate toward reducing asymmetry
            # The direction of correction is given by orig_centroid - refl_centroid
            correction = orig_centroid - refl_centroid
            correction_norm = np.linalg.norm(correction)
            if correction_norm > 0.1:  # Only correct if meaningful difference
                correction = correction / correction_norm
                # Blend current normal with correction (small step)
                step_size = 0.05 * np.exp(-iteration / 10.0)
                new_normal = current_normal + step_size * correction
                norm = np.linalg.norm(new_normal)
                if norm > 0:
                    current_normal = new_normal / norm

            current_point = midpoint
            prev_score = score

        msp.normal = current_normal
        msp.point = current_point
        msp.refinement_iterations = iteration + 1
        msp.final_symmetry_score = prev_score
        msp.detection_method = "pca+iterative_refinement"

        return msp

    def _reflect_points_across_plane(
        self,
        points: np.ndarray,
        normal: np.ndarray,
        plane_point: np.ndarray,
    ) -> np.ndarray:
        """
        Reflect a set of 3D points across a plane.

        Formula: p_reflected = p - 2 * dot(p - plane_point, normal) * normal

        Args:
            points: (N, 3) array of 3D coordinates
            normal: Unit normal vector of the plane
            plane_point: A point on the plane

        Returns:
            (N, 3) reflected points
        """
        # Signed distance from each point to the plane
        diff = points - plane_point  # (N, 3)
        dist = diff @ normal          # (N,) signed distances

        # Reflect: move each point twice the signed distance across the plane
        reflected = points - 2.0 * dist[:, np.newaxis] * normal[np.newaxis, :]
        return reflected

    def _compute_symmetry_score(
        self,
        original: np.ndarray,
        reflected: np.ndarray,
    ) -> float:
        """
        Compute normalised symmetry score between original and reflected point sets.

        Method: Histogram-based correlation in a 3D grid.
        Score 0.0 = no overlap (perfect asymmetry)
        Score 1.0 = perfect overlap (perfect symmetry)
        """
        # Use a coarse grid for efficiency
        all_points = np.vstack([original, reflected])
        mins = all_points.min(axis=0)
        maxs = all_points.max(axis=0)
        ranges = maxs - mins

        if np.any(ranges < 1e-6):
            return 0.5

        bins = 20
        orig_hist, _ = np.histogramdd(original, bins=bins, range=list(zip(mins, maxs)))
        refl_hist, _ = np.histogramdd(reflected, bins=bins, range=list(zip(mins, maxs)))

        # Normalised cross-correlation
        orig_flat = orig_hist.ravel().astype(float)
        refl_flat = refl_hist.ravel().astype(float)

        orig_norm = np.linalg.norm(orig_flat)
        refl_norm = np.linalg.norm(refl_flat)

        if orig_norm < 1e-10 or refl_norm < 1e-10:
            return 0.0

        score = float(np.dot(orig_flat / orig_norm, refl_flat / refl_norm))
        return max(0.0, min(1.0, score))

    def compute_structure_asymmetry(
        self,
        mask: np.ndarray,
        msp: MidsagittalPlane,
        spacing: tuple[float, float, float],
        structure_name: str = "structure",
    ) -> StructureAsymmetry:
        """
        Compute asymmetry metrics for a single structure mask.

        The structure is reflected across the MSP and compared to the original
        using surface distance metrics.

        Args:
            mask: Binary (0/1) mask of the structure (Z, Y, X)
            msp: Detected midsagittal plane
            spacing: Voxel spacing in mm (z, y, x)
            structure_name: Human-readable structure identifier

        Returns:
            StructureAsymmetry with all metrics
        """
        sz, sy, sx = spacing

        if not mask.any():
            return StructureAsymmetry(
                structure_name=structure_name,
                mean_surface_distance_mm=0.0,
                hausdorff_distance_mm=0.0,
                max_distance_mm=0.0,
                volume_asymmetry_ratio=0.0,
                centroid_offset_mm=0.0,
                asymmetry_grade="normal",
                asymmetry_description="Empty mask — no structure detected",
                dominant_side="none",
            )

        # Mirror the mask across the MSP
        mirrored_mask = self._mirror_mask(mask, msp, spacing)

        # Compute surface distance between original and mirrored
        # Extract surfaces (boundaries) using erosion
        eroded_orig = binary_erosion(mask)
        surface_orig = mask.astype(bool) & ~eroded_orig

        eroded_mirr = binary_erosion(mirrored_mask.astype(bool))
        surface_mirr = mirrored_mask.astype(bool) & ~eroded_mirr

        if not surface_orig.any() or not surface_mirr.any():
            # Fallback: use full mask if surface extraction fails
            surface_orig = mask.astype(bool)
            surface_mirr = mirrored_mask.astype(bool)

        # Distance transform from mirrored surface to original
        # distance_transform_edt gives distance in voxels; multiply by spacing
        dt_from_mirr = distance_transform_edt(~surface_mirr, sampling=spacing)
        dist_orig_to_mirr = dt_from_mirr[surface_orig]

        dt_from_orig = distance_transform_edt(~surface_orig, sampling=spacing)
        dist_mirr_to_orig = dt_from_orig[surface_mirr]

        mean_dist = float(np.mean(
            np.concatenate([dist_orig_to_mirr, dist_mirr_to_orig])
        ))
        hausdorff = float(np.percentile(
            np.concatenate([dist_orig_to_mirr, dist_mirr_to_orig]), 95
        ))
        max_dist = float(np.max(
            np.concatenate([dist_orig_to_mirr, dist_mirr_to_orig])
        ))

        # Volume asymmetry ratio
        # Split structure into left (x < msp) and right (x >= msp) halves
        vol_left, vol_right = self._split_by_plane(mask, msp, spacing)
        vol_sum = vol_left + vol_right
        if vol_sum > 0:
            vol_ratio = (vol_left - vol_right) / (vol_sum / 2.0)
        else:
            vol_ratio = 0.0

        dominant_side = "left" if vol_left >= vol_right else "right"

        # Centroid offset
        orig_coords = np.argwhere(mask) * np.array([sz, sy, sx])
        mirr_coords = np.argwhere(mirrored_mask.astype(bool)) * np.array([sz, sy, sx])
        if len(orig_coords) > 0 and len(mirr_coords) > 0:
            centroid_offset = float(np.linalg.norm(
                orig_coords.mean(axis=0) - mirr_coords.mean(axis=0)
            ))
        else:
            centroid_offset = 0.0

        grade, description = classify_asymmetry(mean_dist)

        return StructureAsymmetry(
            structure_name=structure_name,
            mean_surface_distance_mm=round(mean_dist, 2),
            hausdorff_distance_mm=round(hausdorff, 2),
            max_distance_mm=round(max_dist, 2),
            volume_asymmetry_ratio=round(float(vol_ratio), 3),
            centroid_offset_mm=round(centroid_offset, 2),
            asymmetry_grade=grade,
            asymmetry_description=description,
            dominant_side=dominant_side,
        )

    def _mirror_mask(
        self,
        mask: np.ndarray,
        msp: MidsagittalPlane,
        spacing: tuple[float, float, float],
    ) -> np.ndarray:
        """
        Mirror a binary mask across the midsagittal plane.

        Each voxel in the original mask is reflected to its mirrored position.
        The result is a new binary mask of the mirrored anatomy.

        Args:
            mask: Binary mask (Z, Y, X)
            msp: Midsagittal plane definition
            spacing: Voxel spacing (z, y, x) in mm

        Returns:
            Binary mask of the mirrored structure
        """
        sz, sy, sx = spacing
        shape = mask.shape

        # Get physical coordinates of all set voxels
        vox_coords = np.argwhere(mask)  # (N, 3) in (z, y, x)
        if len(vox_coords) == 0:
            return np.zeros_like(mask)

        # Convert to mm
        phys_coords = vox_coords * np.array([sz, sy, sx])

        # Reflect across MSP
        reflected_phys = self._reflect_points_across_plane(
            phys_coords, msp.normal, msp.point
        )

        # Convert back to voxel coordinates
        reflected_vox = reflected_phys / np.array([sz, sy, sx])
        reflected_vox_int = np.round(reflected_vox).astype(int)

        # Create mirrored mask
        mirrored = np.zeros_like(mask)
        valid = (
            (reflected_vox_int[:, 0] >= 0) & (reflected_vox_int[:, 0] < shape[0]) &
            (reflected_vox_int[:, 1] >= 0) & (reflected_vox_int[:, 1] < shape[1]) &
            (reflected_vox_int[:, 2] >= 0) & (reflected_vox_int[:, 2] < shape[2])
        )
        rv = reflected_vox_int[valid]
        mirrored[rv[:, 0], rv[:, 1], rv[:, 2]] = 1

        return mirrored

    def _split_by_plane(
        self,
        mask: np.ndarray,
        msp: MidsagittalPlane,
        spacing: tuple[float, float, float],
    ) -> tuple[int, int]:
        """
        Split a mask into left and right halves by the MSP.

        Returns: (left_voxel_count, right_voxel_count)
        The 'right' side is in the direction of the MSP normal (positive side).
        """
        sz, sy, sx = spacing
        coords = np.argwhere(mask)
        if len(coords) == 0:
            return 0, 0

        phys = coords * np.array([sz, sy, sx])
        signed_dist = (phys - msp.point) @ msp.normal  # positive = right

        left_count = int(np.sum(signed_dist < 0))
        right_count = int(np.sum(signed_dist >= 0))
        return left_count, right_count

    def compute_asymmetry_map(
        self,
        segmentation_mask: np.ndarray,
        msp: MidsagittalPlane,
        spacing: tuple[float, float, float],
    ) -> np.ndarray:
        """
        Generate a voxel-wise asymmetry distance map.

        Each voxel in a bone structure is assigned the surface distance
        to the nearest point of the mirrored anatomy. This creates a
        continuous map highlighting regions of greatest asymmetry.

        Args:
            segmentation_mask: Multi-label segmentation (Z, Y, X)
            msp: Midsagittal plane
            spacing: Voxel spacing in mm

        Returns:
            Float32 array of same shape as segmentation_mask.
            Values represent asymmetry distance in mm (0 = symmetric).
        """
        bone_mask = (segmentation_mask > 0).astype(bool)
        asymmetry_map = np.zeros(segmentation_mask.shape, dtype=np.float32)

        if not bone_mask.any():
            return asymmetry_map

        # Mirror the full bone mask
        mirrored_bone = self._mirror_mask(bone_mask.astype(np.uint8), msp, spacing)

        # Distance from each bone voxel to nearest mirrored bone voxel
        dt = distance_transform_edt(~mirrored_bone.astype(bool), sampling=spacing)
        asymmetry_map[bone_mask] = dt[bone_mask].astype(np.float32)

        return asymmetry_map

    def analyze(
        self,
        volume: np.ndarray,
        segmentation_mask: np.ndarray,
        spacing: tuple[float, float, float],
        compute_full_asymmetry_map: bool = False,
    ) -> SymmetryReport:
        """
        Full facial symmetry analysis pipeline.

        Args:
            volume: CT array (Z, Y, X) in HU
            segmentation_mask: Integer label mask (Z, Y, X)
            spacing: Voxel spacing in mm (z, y, x)
            compute_full_asymmetry_map: If True, compute voxel-wise asymmetry map
                                       (memory-intensive for large volumes)

        Returns:
            SymmetryReport with all symmetry metrics
        """
        import time
        t_start = time.time()

        # --- Step 1: Detect midsagittal plane ---
        logger.info("Detecting midsagittal plane...")
        msp = self.detect_midsagittal_plane(volume, segmentation_mask, spacing)
        logger.info(
            f"MSP detected: normal={msp.normal.round(3)}, "
            f"symmetry_score={msp.final_symmetry_score:.3f}"
        )

        # --- Step 2: Per-structure asymmetry ---
        per_structure: dict[str, StructureAsymmetry] = {}
        per_structure_mm: dict[str, float] = {}

        structure_labels = {
            "mandible": self.LABEL_MANDIBLE,
            "maxilla": self.LABEL_MAXILLA,
            "skull": self.LABEL_SKULL,
        }

        bilateral_pairs = {
            "orbital": (self.LABEL_ORBIT_L, self.LABEL_ORBIT_R),
            "zygomatic": (self.LABEL_ZYGOMATIC_L, self.LABEL_ZYGOMATIC_R),
        }

        for name, label_id in structure_labels.items():
            struct_mask = (segmentation_mask == label_id).astype(np.uint8)
            if struct_mask.any():
                sa = self.compute_structure_asymmetry(struct_mask, msp, spacing, name)
                per_structure[name] = sa
                per_structure_mm[name] = sa.mean_surface_distance_mm
            else:
                logger.debug(f"Structure '{name}' (label {label_id}) not found in mask")

        # Bilateral structures: combine left + right masks for joined analysis
        for name, (label_l, label_r) in bilateral_pairs.items():
            combined_mask = (
                (segmentation_mask == label_l) |
                (segmentation_mask == label_r)
            ).astype(np.uint8)
            if combined_mask.any():
                sa = self.compute_structure_asymmetry(combined_mask, msp, spacing, name)
                per_structure[name] = sa
                per_structure_mm[name] = sa.mean_surface_distance_mm

        # --- Step 3: Overall symmetry score ---
        if per_structure_mm:
            # Weight by clinical importance
            weights = {
                "mandible": 3.0,   # Highest clinical weight
                "orbital": 2.5,
                "zygomatic": 2.0,
                "maxilla": 2.0,
                "skull": 1.0,
            }
            total_w = 0.0
            weighted_sum = 0.0
            for name, dist_mm in per_structure_mm.items():
                w = weights.get(name, 1.0)
                weighted_sum += dist_mm * w
                total_w += w
            weighted_mean_mm = weighted_sum / total_w if total_w > 0 else 0.0

            # Convert mm asymmetry to 0–1 score (inverse sigmoid)
            # 0mm → score 1.0, 6mm → score ~0.5, 12mm → score ~0.2
            overall_score = float(np.exp(-weighted_mean_mm / 6.0))
            overall_grade, _ = classify_asymmetry(weighted_mean_mm)
        else:
            overall_score = 1.0
            overall_grade = "normal"

        # --- Step 4: Asymmetry map (optional) ---
        asym_map = None
        if compute_full_asymmetry_map:
            logger.info("Computing full voxel-wise asymmetry map...")
            asym_map = self.compute_asymmetry_map(segmentation_mask, msp, spacing)

        # --- Step 5: Clinical summary ---
        summary_lines = [
            f"Overall symmetry score: {overall_score:.2f} ({overall_grade})",
            f"MSP detected via {msp.detection_method} "
            f"({msp.refinement_iterations} refinement iterations, "
            f"score={msp.final_symmetry_score:.3f})",
            "",
            "Per-structure asymmetry (mean surface distance):",
        ]
        for name, dist_mm in sorted(per_structure_mm.items()):
            grade, _ = classify_asymmetry(dist_mm)
            summary_lines.append(f"  {name:20s}: {dist_mm:.1f} mm ({grade})")

        processing_time = (time.time() - t_start) * 1000

        report = SymmetryReport(
            midsagittal_plane_normal=msp.normal,
            midsagittal_plane_point=msp.point,
            overall_symmetry_score=round(overall_score, 4),
            overall_grade=overall_grade,
            per_structure_asymmetry=per_structure,
            per_structure_asymmetry_mm={k: round(v, 2) for k, v in per_structure_mm.items()},
            asymmetry_map_voxels=asym_map,
            clinical_summary="\n".join(summary_lines),
            processing_time_ms=round(processing_time, 1),
        )

        logger.info(
            f"Symmetry analysis complete: score={overall_score:.3f} "
            f"grade={overall_grade} in {processing_time:.0f}ms"
        )

        return report

    def get_orbital_asymmetry(self, report: SymmetryReport) -> Optional[StructureAsymmetry]:
        """Return orbital asymmetry metrics, if available."""
        return report.per_structure_asymmetry.get("orbital")

    def get_mandibular_asymmetry(self, report: SymmetryReport) -> Optional[StructureAsymmetry]:
        """Return mandibular asymmetry metrics, if available."""
        return report.per_structure_asymmetry.get("mandible")

    def get_zygomatic_asymmetry(self, report: SymmetryReport) -> Optional[StructureAsymmetry]:
        """Return zygomatic projection asymmetry metrics, if available."""
        return report.per_structure_asymmetry.get("zygomatic")

    @staticmethod
    def format_report(report: SymmetryReport) -> str:
        """Format SymmetryReport as a human-readable clinical text summary."""
        lines = [
            "=" * 60,
            "FACIAL SYMMETRY ANALYSIS REPORT",
            "=" * 60,
            "",
            report.clinical_summary,
            "",
            "DETAILED PER-STRUCTURE MEASUREMENTS:",
        ]
        for name, sa in sorted(report.per_structure_asymmetry.items()):
            lines += [
                f"\n  {name.upper()}",
                f"    Mean surface distance : {sa.mean_surface_distance_mm:.2f} mm",
                f"    Hausdorff distance    : {sa.hausdorff_distance_mm:.2f} mm",
                f"    Volume asymmetry ratio: {sa.volume_asymmetry_ratio:+.3f}",
                f"    Dominant side         : {sa.dominant_side}",
                f"    Grade                 : {sa.asymmetry_grade.upper()}",
                f"    {sa.asymmetry_description}",
            ]

        if report.warnings:
            lines += ["", "WARNINGS:"]
            for w in report.warnings:
                lines.append(f"  ! {w}")

        lines += [
            "",
            f"Processing time: {report.processing_time_ms:.0f} ms",
            "=" * 60,
        ]
        return "\n".join(lines)
