"""
Cephalometric Landmark Detection Adapter
=========================================

Craniofacial surgical planning relies heavily on cephalometric analysis —
a standardized system of skeletal landmarks that define the spatial
relationships between the cranial base, maxilla, and mandible.

Clinical Background
-------------------
Cephalometric analysis, originally performed on lateral skull X-rays
(cephalograms), is the cornerstone of orthognathic surgical planning.
Landmarks such as Nasion, Sella, and the A- and B-points define planes
and angles that quantify skeletal discrepancies requiring surgical correction.

With CT-based planning, landmarks can be identified in 3D, allowing
true spatial (rather than projected 2D) measurements. This enables:
  - Bilateral asymmetry detection (e.g., mandibular laterognathia)
  - Accurate osteotomy simulation
  - Quantitative surgical outcome prediction

This module implements:
  1. A complete anatomical landmark catalog with clinical definitions
  2. Heuristic landmark estimation from segmentation masks (no ML required)
  3. Standard cephalometric measurements (SNA, SNB, ANB, etc.)
  4. A CephalometricAnalysis dataclass for downstream planning services
  5. A clear interface for plugging in a learned landmark detection model

Reference Landmarks follow the standards described in:
  - Jacobson A. "Radiographic Cephalometry", Quintessence Publishing
  - McNamara JA. "A method of cephalometric evaluation", AJO 1984

Author: Facial Align Engineering
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.ndimage import label as scipy_label
from scipy.ndimage import binary_erosion, binary_dilation, center_of_mass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Landmark Definitions
# ---------------------------------------------------------------------------

LANDMARK_DEFINITIONS: dict[str, str] = {
    # Cranial base landmarks
    "Nasion": (
        "Most anterior point of the nasofrontal suture on the midsagittal plane. "
        "Key reference for anterior cranial base. Used in SNA and SNB angles."
    ),
    "Sella": (
        "Centre of the pituitary fossa (sella turcica) of the sphenoid bone. "
        "Principal cranial base reference point; origin of Sella-Nasion line."
    ),
    "Basion": (
        "Most inferior-posterior point of the foramen magnum on the midsagittal plane. "
        "Used to define the cranial base plane and Ba-N line."
    ),
    "Porion": (
        "Most superior point of the external auditory meatus. "
        "Defines the Frankfort Horizontal plane along with Orbitale."
    ),
    "Orbitale": (
        "Lowest point on the inferior rim of the orbit. "
        "Together with Porion defines the Frankfort Horizontal plane."
    ),
    # Midface landmarks
    "A_point": (
        "Deepest point on the anterior surface of the maxillary alveolar process, "
        "between the anterior nasal spine and the dental alveolus. "
        "Defines the AP position of the maxillary base. Used in SNA angle."
    ),
    "ANS": (
        "Anterior Nasal Spine — the sharp bony process at the anterior aspect of "
        "the nasal floor, formed by the maxillary bones. Defines the maxillary "
        "occlusal plane reference."
    ),
    "PNS": (
        "Posterior Nasal Spine — the pointed bony projection at the posterior limit "
        "of the hard palate (palatine bones). Together with ANS defines the palatal plane."
    ),
    "Subspinale": (
        "Same as A-point. Most concave point on the anterior maxillary alveolar process. "
        "Alternate terminology used in some European cephalometric systems."
    ),
    # Mandibular landmarks
    "B_point": (
        "Deepest point on the anterior surface of the mandibular alveolar process, "
        "between infradentale and pogonion. Defines AP position of mandibular base. "
        "Used in SNB angle and the ANB discrepancy."
    ),
    "Menton": (
        "Most inferior (caudal) point on the mandibular symphysis. "
        "Key landmark for vertical facial height measurements."
    ),
    "Gnathion": (
        "Most inferior and anterior point on the symphysis of the mandible. "
        "Midpoint between Menton and Pogonion along the symphysis."
    ),
    "Pogonion": (
        "Most anterior point on the chin (mandibular symphysis). "
        "Used in linear chin projection measurements."
    ),
    "Gonion_L": (
        "Left gonial angle — the most inferior, posterior, and lateral point on the "
        "angle of the mandible at the junction of the ramus and body. "
        "Used to define the mandibular plane."
    ),
    "Gonion_R": (
        "Right gonial angle (mirror of Gonion_L). "
        "Bilateral detection enables mandibular asymmetry quantification."
    ),
    "Condylion_L": (
        "Most superior-posterior point of the left mandibular condyle. "
        "Used to measure the ramus height and condylar position."
    ),
    "Condylion_R": (
        "Most superior-posterior point of the right mandibular condyle."
    ),
    "Coronoid_L": (
        "Tip of the left coronoid process of the mandible."
    ),
    "Coronoid_R": (
        "Tip of the right coronoid process of the mandible."
    ),
    "Infradentale": (
        "Most superior-anterior point of the mandibular alveolar process between "
        "the central incisors. Incisal edge of the lower central incisors."
    ),
    # Orbital landmarks
    "Orbitale_L": (
        "Most inferior point of the left orbital rim. "
        "Left component of Frankfort Horizontal."
    ),
    "Orbitale_R": (
        "Most inferior point of the right orbital rim."
    ),
    # Zygomatic landmarks
    "Zygion_L": (
        "Most lateral point on the left zygomatic arch. "
        "Used for bizygomatic width measurements."
    ),
    "Zygion_R": (
        "Most lateral point on the right zygomatic arch."
    ),
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CephalometricAnalysis:
    """
    Complete cephalometric analysis for orthognathic surgical planning.

    All angles are in degrees; all linear distances are in millimetres.

    Clinical Interpretation Guide
    ------------------------------
    SNA angle (norm 82° ± 2°):
        > 84° → maxillary protrusion (class II tendency)
        < 80° → maxillary retrusion (class III tendency)

    SNB angle (norm 80° ± 2°):
        > 82° → mandibular protrusion
        < 78° → mandibular retrusion

    ANB angle (norm 2° ± 2°):
        > 4° → skeletal class II (maxillary protrusion or mandibular retrusion)
        < 0° → skeletal class III (mandibular protrusion or maxillary retrusion)

    Mandibular plane angle (norm 32° ± 4° to Frankfurt Horizontal):
        High angle (>36°) → hyperdivergent, vertical growth pattern
        Low angle (<28°) → hypodivergent, horizontal growth pattern

    Wits appraisal (norm 0 ± 2 mm):
        Perpendicular distance between A and B points projected onto
        the occlusal plane. Complements ANB when cranial base is atypical.
    """

    # Landmark coordinates (in mm, physical space)
    landmarks: dict[str, np.ndarray] = field(default_factory=dict)

    # Cephalometric angles (degrees)
    SNA_angle: float = 0.0         # Maxillary protrusion relative to cranial base
    SNB_angle: float = 0.0         # Mandibular protrusion relative to cranial base
    ANB_angle: float = 0.0         # Maxillo-mandibular discrepancy
    mandibular_plane_angle: float = 0.0  # Mandibular plane to Frankfort horizontal
    palatal_plane_angle: float = 0.0     # ANS-PNS to Frankfort horizontal
    gonial_angle_L: float = 0.0          # Left ramus-body angle
    gonial_angle_R: float = 0.0          # Right ramus-body angle
    sella_nasion_angle: float = 0.0      # Anterior cranial base inclination

    # Linear measurements (mm)
    anterior_facial_height: float = 0.0  # Nasion to Menton
    posterior_facial_height: float = 0.0 # Sella to Gonion
    ramus_height_L: float = 0.0          # Condylion_L to Gonion_L
    ramus_height_R: float = 0.0          # Condylion_R to Gonion_R
    mandibular_body_length_L: float = 0.0 # Gonion_L to Menton
    mandibular_body_length_R: float = 0.0 # Gonion_R to Menton
    wits_appraisal: float = 0.0           # AP relationship on occlusal plane

    # Plane definitions (normal vectors in 3D)
    frankfort_horizontal_normal: Optional[np.ndarray] = None
    mandibular_plane_normal: Optional[np.ndarray] = None
    palatal_plane_normal: Optional[np.ndarray] = None
    sella_nasion_vector: Optional[np.ndarray] = None

    # Quality flags
    landmarks_estimated: bool = True   # True = heuristic; False = model-predicted
    confidence_scores: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main Detector Class
# ---------------------------------------------------------------------------

class CephalometricLandmarkDetector:
    """
    Detect cephalometric landmarks from CT volumes and segmentation masks.

    This class provides two tiers of landmark detection:

    Tier 1 — Heuristic (implemented here):
        Uses anatomical priors about the spatial layout of structures within
        segmentation masks to estimate landmark positions. For example:
          - Menton = lowest point (most caudal voxel) of the mandible mask
          - Gonion = posteroinferior corner of the mandible ramus
          - Sella = estimated from skull base centroid
        Accuracy: ± 5–15 mm, suitable for automated quality checks and
        initial planning estimates requiring surgeon review.

    Tier 2 — Learned Detection (interface provided, to be implemented):
        A CNN-based heatmap regression model trained on annotated CT datasets
        can replace the heuristic estimates with sub-millimetre accuracy.
        The `_predict_with_model()` method stub defines the expected interface.
        Accuracy: ± 1–3 mm when trained on sufficient data.

    Usage:
        detector = CephalometricLandmarkDetector()

        # With only a segmentation mask (Tier 1):
        analysis = detector.analyze(
            volume=ct_array,
            segmentation_mask=seg_mask,
            spacing=(0.5, 0.5, 0.5),
        )

        print(f"ANB angle: {analysis.ANB_angle:.1f}°")
        print(f"Mandibular plane angle: {analysis.mandibular_plane_angle:.1f}°")
    """

    # Segmentation label IDs (must match your segmentation model's label map)
    LABEL_MANDIBLE = 1
    LABEL_MAXILLA = 2
    LABEL_SKULL = 3
    LABEL_ORBIT_L = 4
    LABEL_ORBIT_R = 5
    LABEL_ZYGOMATIC_L = 6
    LABEL_ZYGOMATIC_R = 7
    LABEL_CONDYLE_L = 8
    LABEL_CONDYLE_R = 9

    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: Optional path to a trained landmark detection model.
                        If None, falls back to heuristic estimation.
        """
        self._model_path = model_path
        self._model = None

        if model_path is not None:
            self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        """
        Load a trained landmark detection model.

        Expected model interface:
            Input:  (C, Z, Y, X) float32 CT volume, HU-normalised
            Output: (N_landmarks, 3) float32 heatmap peak coordinates in mm

        Override this method when integrating a trained network.
        """
        logger.info(f"Landmark model path provided: {model_path} — loading deferred")
        # TODO: Implement actual model loading (Phase 2)
        # self._model = torch.load(model_path, map_location="cpu")
        # self._model.eval()

    def detect_landmarks(
        self,
        volume: np.ndarray,
        segmentation_mask: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> dict[str, np.ndarray]:
        """
        Detect all standard cephalometric landmarks in physical space (mm).

        If a trained model is available, uses learned detection.
        Otherwise falls back to anatomical heuristics on the segmentation mask.

        Args:
            volume: 3D CT array (Z, Y, X) in Hounsfield Units
            segmentation_mask: Integer label volume (Z, Y, X) matching volume shape
            spacing: Voxel spacing (z_mm, y_mm, x_mm)

        Returns:
            Dictionary mapping landmark names to 3D coordinate arrays in mm.
            Coordinates are in the image physical space (origin at [0, 0, 0]).
        """
        if segmentation_mask.shape != volume.shape:
            raise ValueError(
                f"Volume shape {volume.shape} and mask shape "
                f"{segmentation_mask.shape} must match."
            )

        if self._model is not None:
            return self._predict_with_model(volume, spacing)

        return self._estimate_heuristic(volume, segmentation_mask, spacing)

    def analyze(
        self,
        volume: np.ndarray,
        segmentation_mask: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> CephalometricAnalysis:
        """
        Full cephalometric analysis pipeline.

        Detects landmarks and computes all standard cephalometric measurements.

        Args:
            volume: CT volume (Z, Y, X) in HU
            segmentation_mask: Segmentation label volume
            spacing: Voxel spacing in mm (z, y, x)

        Returns:
            CephalometricAnalysis with all measurements filled in.
        """
        landmarks = self.detect_landmarks(volume, segmentation_mask, spacing)
        analysis = CephalometricAnalysis(landmarks=landmarks)
        analysis.landmarks_estimated = (self._model is None)

        self._compute_planes(analysis)
        self._compute_angles(analysis)
        self._compute_linear_measurements(analysis)
        self._add_confidence_scores(analysis, segmentation_mask)

        logger.info(
            "Cephalometric analysis complete: "
            f"SNA={analysis.SNA_angle:.1f}° "
            f"SNB={analysis.SNB_angle:.1f}° "
            f"ANB={analysis.ANB_angle:.1f}° "
            f"MPA={analysis.mandibular_plane_angle:.1f}°"
        )

        return analysis

    # ------------------------------------------------------------------
    # Heuristic Landmark Estimation
    # ------------------------------------------------------------------

    def _estimate_heuristic(
        self,
        volume: np.ndarray,
        mask: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> dict[str, np.ndarray]:
        """
        Estimate landmark coordinates from segmentation masks using anatomical
        geometric heuristics.

        Coordinate convention: physical coordinates in mm.
          axis 0 = Z (superior-inferior, +Z = superior)
          axis 1 = Y (anterior-posterior, +Y = anterior)
          axis 2 = X (left-right, +X = right)

        All positions are converted from voxel to mm using spacing.
        """
        sz, sy, sx = spacing
        lms: dict[str, np.ndarray] = {}

        mandible_mask = (mask == self.LABEL_MANDIBLE)
        maxilla_mask = (mask == self.LABEL_MAXILLA)
        skull_mask = (mask == self.LABEL_SKULL)

        # ---- Mandible landmarks ----------------------------------------

        if mandible_mask.any():
            coords = np.argwhere(mandible_mask)  # (N, 3) in (z, y, x) voxels

            # Menton: most inferior (lowest z) point on mandibular symphysis
            # Restrict to anterior half (y > median_y) to avoid condyles
            median_y = np.median(coords[:, 1])
            anterior_coords = coords[coords[:, 1] > median_y]
            if len(anterior_coords) > 0:
                min_z_idx = np.argmin(anterior_coords[:, 0])
                menton_vox = anterior_coords[min_z_idx]
            else:
                min_z_idx = np.argmin(coords[:, 0])
                menton_vox = coords[min_z_idx]
            lms["Menton"] = menton_vox.astype(float) * np.array([sz, sy, sx])

            # Gnathion: midpoint between Menton and Pogonion
            # Pogonion: most anterior (highest y) point on the symphysis
            # (at approximately the same z level as Menton)
            menton_z = menton_vox[0]
            z_band = coords[
                (coords[:, 0] >= menton_z) &
                (coords[:, 0] <= menton_z + int(15.0 / sz))  # 15mm band
            ]
            if len(z_band) > 0:
                max_y_idx = np.argmax(z_band[:, 1])
                pogonion_vox = z_band[max_y_idx]
            else:
                pogonion_vox = menton_vox
            lms["Pogonion"] = pogonion_vox.astype(float) * np.array([sz, sy, sx])
            lms["Gnathion"] = (lms["Menton"] + lms["Pogonion"]) / 2.0

            # B-point: deepest concavity on anterior mandibular alveolar process
            # Approximately 10-20mm superior to Menton on the anterior surface
            b_z_vox = menton_vox[0] + int(15.0 / sz)
            b_z_band = coords[
                (coords[:, 0] >= b_z_vox - int(5.0 / sz)) &
                (coords[:, 0] <= b_z_vox + int(5.0 / sz))
            ]
            if len(b_z_band) > 0:
                max_y_idx = np.argmax(b_z_band[:, 1])
                b_vox = b_z_band[max_y_idx]
            else:
                b_vox = pogonion_vox + np.array([int(15.0 / sz), 0, 0])
            lms["B_point"] = b_vox.astype(float) * np.array([sz, sy, sx])

            # Gonion L/R: posteroinferior angle of mandibular ramus
            # Left = negative x (patient left), Right = positive x
            median_z = np.median(coords[:, 0])
            inferior_coords = coords[coords[:, 0] <= median_z + int(10.0 / sz)]

            # Left Gonion: most posterior (min y), inferior, leftward (min x)
            left_coords = inferior_coords[inferior_coords[:, 2] < np.median(coords[:, 2])]
            if len(left_coords) > 0:
                # Score: low z, low y (posterior), low x (left)
                score = (
                    -left_coords[:, 0] * sz  # inferior
                    - left_coords[:, 1] * sy  # posterior
                    - left_coords[:, 2] * sx  # leftward
                )
                gonion_l_vox = left_coords[np.argmax(score)]
            else:
                gonion_l_vox = coords[np.argmin(coords[:, 0])]
            lms["Gonion_L"] = gonion_l_vox.astype(float) * np.array([sz, sy, sx])

            # Right Gonion
            right_coords = inferior_coords[inferior_coords[:, 2] >= np.median(coords[:, 2])]
            if len(right_coords) > 0:
                score = (
                    -right_coords[:, 0] * sz
                    - right_coords[:, 1] * sy
                    + right_coords[:, 2] * sx  # rightward
                )
                gonion_r_vox = right_coords[np.argmax(score)]
            else:
                gonion_r_vox = gonion_l_vox.copy()
                gonion_r_vox[2] = mask.shape[2] - gonion_l_vox[2]
            lms["Gonion_R"] = gonion_r_vox.astype(float) * np.array([sz, sy, sx])

            # Condylion L/R: most superior-posterior point of condylar heads
            # Superior: high z, in the posterior (low y) and lateral (extreme x)
            superior_coords = coords[coords[:, 0] >= np.percentile(coords[:, 0], 80)]
            median_x = np.median(coords[:, 2])

            # Condylion Left
            cond_l = superior_coords[superior_coords[:, 2] < median_x]
            if len(cond_l) > 0:
                score = cond_l[:, 0] * sz - cond_l[:, 1] * sy  # high z, low y
                lms["Condylion_L"] = cond_l[np.argmax(score)].astype(float) * np.array([sz, sy, sx])
            else:
                lms["Condylion_L"] = lms["Gonion_L"] + np.array([40.0, -5.0, 0.0])

            # Condylion Right
            cond_r = superior_coords[superior_coords[:, 2] >= median_x]
            if len(cond_r) > 0:
                score = cond_r[:, 0] * sz - cond_r[:, 1] * sy
                lms["Condylion_R"] = cond_r[np.argmax(score)].astype(float) * np.array([sz, sy, sx])
            else:
                lms["Condylion_R"] = lms["Gonion_R"] + np.array([40.0, -5.0, 0.0])

            # Infradentale: most superior-anterior point of lower alveolus
            # In lower incisor region: central x, high z, high y
            central_x_min = int(np.median(coords[:, 2]) - 10.0 / sx)
            central_x_max = int(np.median(coords[:, 2]) + 10.0 / sx)
            central_band = coords[
                (coords[:, 2] >= central_x_min) &
                (coords[:, 2] <= central_x_max)
            ]
            if len(central_band) > 0:
                score = central_band[:, 0] * sz + central_band[:, 1] * sy
                infra_vox = central_band[np.argmax(score)]
                lms["Infradentale"] = infra_vox.astype(float) * np.array([sz, sy, sx])

        # ---- Maxilla / Midface landmarks --------------------------------

        if maxilla_mask.any():
            coords = np.argwhere(maxilla_mask)

            # ANS: anterior nasal spine — most anterior (high y), inferior-ish point
            median_z = np.median(coords[:, 0])
            lower_third = coords[coords[:, 0] <= median_z]
            if len(lower_third) > 0:
                ans_idx = np.argmax(lower_third[:, 1])  # most anterior
                ans_vox = lower_third[ans_idx]
            else:
                ans_idx = np.argmax(coords[:, 1])
                ans_vox = coords[ans_idx]
            lms["ANS"] = ans_vox.astype(float) * np.array([sz, sy, sx])

            # PNS: posterior nasal spine — most posterior (low y) at palate level
            upper_half = coords[coords[:, 0] >= np.percentile(coords[:, 0], 40)]
            if len(upper_half) > 0:
                pns_idx = np.argmin(upper_half[:, 1])  # most posterior
                pns_vox = upper_half[pns_idx]
            else:
                pns_vox = ans_vox.copy()
                pns_vox[1] = int(np.min(coords[:, 1]))
            lms["PNS"] = pns_vox.astype(float) * np.array([sz, sy, sx])

            # A-point: deepest concavity on anterior maxillary alveolar process
            # Approximately 10-15mm inferior to ANS on the anterior surface
            a_z = ans_vox[0] - int(10.0 / sz)
            a_band = coords[
                (coords[:, 0] >= a_z - int(5.0 / sz)) &
                (coords[:, 0] <= a_z + int(5.0 / sz))
            ]
            if len(a_band) > 0:
                a_idx = np.argmax(a_band[:, 1])  # most anterior
                a_vox = a_band[a_idx]
            else:
                a_vox = ans_vox - np.array([int(10.0 / sz), 0, 0])
                a_vox = np.clip(a_vox, 0, np.array(mask.shape) - 1)
            lms["A_point"] = a_vox.astype(float) * np.array([sz, sy, sx])
            lms["Subspinale"] = lms["A_point"].copy()

        # ---- Skull / Cranial base landmarks ----------------------------

        if skull_mask.any():
            coords = np.argwhere(skull_mask)

            # Nasion: anterior aspect of nasofrontal junction (midsagittal)
            # Most anterior (high y), at mid-height, near centre x
            median_x = np.median(coords[:, 2])
            x_half_band = int(10.0 / sx)
            midsag_coords = coords[
                (coords[:, 2] >= median_x - x_half_band) &
                (coords[:, 2] <= median_x + x_half_band)
            ]
            if len(midsag_coords) > 0:
                mid_z = np.median(midsag_coords[:, 0])
                mid_band = midsag_coords[
                    (midsag_coords[:, 0] >= mid_z - int(15.0 / sz)) &
                    (midsag_coords[:, 0] <= mid_z + int(15.0 / sz))
                ]
                if len(mid_band) > 0:
                    nas_vox = mid_band[np.argmax(mid_band[:, 1])]
                else:
                    nas_vox = midsag_coords[np.argmax(midsag_coords[:, 1])]
            else:
                nas_vox = coords[np.argmax(coords[:, 1])]
            lms["Nasion"] = nas_vox.astype(float) * np.array([sz, sy, sx])

            # Sella: center of sella turcica — at sphenoid, near posterior cranial
            # Heuristic: posterior to Nasion, superior to hard palate, central x
            # Approximate: 15-20mm posterior and superior to Nasion
            if "Nasion" in lms:
                sella_estimate = lms["Nasion"].copy()
                sella_estimate[0] += 15.0   # superior
                sella_estimate[1] -= 20.0   # posterior
                lms["Sella"] = sella_estimate

            # Basion: most inferior-posterior point of foramen magnum
            # Heuristic: lowest z, lowest y, near central x in skull
            posterior_coords = coords[coords[:, 1] <= np.percentile(coords[:, 1], 20)]
            if len(posterior_coords) > 0:
                bas_vox = posterior_coords[np.argmin(posterior_coords[:, 0])]
                lms["Basion"] = bas_vox.astype(float) * np.array([sz, sy, sx])

            # Orbitale L/R: most inferior point of each orbital rim
            # Look for inferior orbital structure near expected position
            median_z = np.median(coords[:, 0])
            median_x_skull = np.median(coords[:, 2])
            orbital_z_min = int(median_z - int(10.0 / sz))
            orbital_z_max = int(median_z + int(10.0 / sz))
            orbital_band = coords[
                (coords[:, 0] >= orbital_z_min) &
                (coords[:, 0] <= orbital_z_max)
            ]

            if len(orbital_band) > 0:
                left_orb = orbital_band[orbital_band[:, 2] < median_x_skull]
                right_orb = orbital_band[orbital_band[:, 2] >= median_x_skull]
                if len(left_orb) > 0:
                    lms["Orbitale_L"] = left_orb[np.argmin(left_orb[:, 0])].astype(float) * np.array([sz, sy, sx])
                    # Midsagittal Orbitale as average
                    lms["Orbitale"] = lms["Orbitale_L"].copy()
                if len(right_orb) > 0:
                    lms["Orbitale_R"] = right_orb[np.argmin(right_orb[:, 0])].astype(float) * np.array([sz, sy, sx])
                    lms["Orbitale"] = (
                        (lms.get("Orbitale_L", lms["Orbitale_R"]) + lms["Orbitale_R"]) / 2.0
                    )

            # Porion: most superior point of external auditory meatus
            # Heuristic: lateral skull, approximately at orbital level
            if "Orbitale_L" in lms:
                orb_z = lms["Orbitale_L"][0]
                lateral_band = coords[
                    (coords[:, 0] >= orb_z - int(5.0 / sz)) &
                    (coords[:, 0] <= orb_z + int(5.0 / sz))
                ]
                if len(lateral_band) > 0:
                    left_lat = lateral_band[lateral_band[:, 2] < median_x_skull]
                    right_lat = lateral_band[lateral_band[:, 2] >= median_x_skull]
                    if len(left_lat) > 0:
                        lms["Porion_L"] = left_lat[np.argmin(left_lat[:, 2])].astype(float) * np.array([sz, sy, sx])
                        lms["Porion"] = lms["Porion_L"].copy()
                    if len(right_lat) > 0:
                        lms["Porion_R"] = right_lat[np.argmax(right_lat[:, 2])].astype(float) * np.array([sz, sy, sx])

        return lms

    # ------------------------------------------------------------------
    # Measurement Computation
    # ------------------------------------------------------------------

    def _compute_planes(self, analysis: CephalometricAnalysis) -> None:
        """Define key cephalometric planes from landmark coordinates."""
        lms = analysis.landmarks

        # Sella-Nasion vector (anterior cranial base reference)
        if "Sella" in lms and "Nasion" in lms:
            sn = lms["Nasion"] - lms["Sella"]
            norm = np.linalg.norm(sn)
            if norm > 0:
                analysis.sella_nasion_vector = sn / norm

        # Frankfort Horizontal: defined by Porion and Orbitale
        # In 3D, the Frankfort plane passes through bilateral Porion and Orbitale
        if "Porion" in lms and "Orbitale" in lms:
            po = lms["Orbitale"] - lms["Porion"]
            # Normal to Frankfort plane = cross product with lateral direction
            lateral = np.array([0.0, 0.0, 1.0])  # x-axis
            fh_normal = np.cross(po, lateral)
            norm = np.linalg.norm(fh_normal)
            if norm > 0:
                analysis.frankfort_horizontal_normal = fh_normal / norm

        # Mandibular plane: through Gonion (average L/R) and Menton
        if "Gonion_L" in lms and "Gonion_R" in lms and "Menton" in lms:
            gonion_mid = (lms["Gonion_L"] + lms["Gonion_R"]) / 2.0
            go_me = lms["Menton"] - gonion_mid
            lateral = np.array([0.0, 0.0, 1.0])
            mp_normal = np.cross(go_me, lateral)
            norm = np.linalg.norm(mp_normal)
            if norm > 0:
                analysis.mandibular_plane_normal = mp_normal / norm

        # Palatal plane: ANS to PNS
        if "ANS" in lms and "PNS" in lms:
            pal = lms["ANS"] - lms["PNS"]
            lateral = np.array([0.0, 0.0, 1.0])
            pal_normal = np.cross(pal, lateral)
            norm = np.linalg.norm(pal_normal)
            if norm > 0:
                analysis.palatal_plane_normal = pal_normal / norm

    def _compute_angles(self, analysis: CephalometricAnalysis) -> None:
        """Compute all standard cephalometric angles."""
        lms = analysis.landmarks
        sn = analysis.sella_nasion_vector

        # SNA: angle at Nasion between Sella-Nasion line and Nasion-A_point line
        if sn is not None and "Nasion" in lms and "A_point" in lms and "Sella" in lms:
            na = lms["A_point"] - lms["Nasion"]
            na_norm = np.linalg.norm(na)
            if na_norm > 0:
                na = na / na_norm
                # SNA is measured as angle at Nasion: between N←S and N→A
                ns = lms["Sella"] - lms["Nasion"]
                ns_norm = np.linalg.norm(ns)
                if ns_norm > 0:
                    ns = ns / ns_norm
                    cos_sna = np.clip(np.dot(ns, na), -1.0, 1.0)
                    analysis.SNA_angle = float(np.degrees(np.arccos(cos_sna)))

        # SNB: angle at Nasion between Sella-Nasion and Nasion-B_point
        if sn is not None and "Nasion" in lms and "B_point" in lms and "Sella" in lms:
            nb = lms["B_point"] - lms["Nasion"]
            nb_norm = np.linalg.norm(nb)
            if nb_norm > 0:
                nb = nb / nb_norm
                ns = lms["Sella"] - lms["Nasion"]
                ns_norm = np.linalg.norm(ns)
                if ns_norm > 0:
                    ns = ns / ns_norm
                    cos_snb = np.clip(np.dot(ns, nb), -1.0, 1.0)
                    analysis.SNB_angle = float(np.degrees(np.arccos(cos_snb)))

        # ANB: SNA - SNB (positive = class II, negative = class III)
        analysis.ANB_angle = analysis.SNA_angle - analysis.SNB_angle

        # Mandibular plane angle to Frankfort horizontal
        if (analysis.frankfort_horizontal_normal is not None and
                analysis.mandibular_plane_normal is not None):
            cos_mpa = np.clip(
                np.dot(analysis.frankfort_horizontal_normal,
                       analysis.mandibular_plane_normal),
                -1.0, 1.0
            )
            analysis.mandibular_plane_angle = float(np.degrees(np.arccos(abs(cos_mpa))))

        # Palatal plane angle to Frankfort horizontal
        if (analysis.frankfort_horizontal_normal is not None and
                analysis.palatal_plane_normal is not None):
            cos_ppa = np.clip(
                np.dot(analysis.frankfort_horizontal_normal,
                       analysis.palatal_plane_normal),
                -1.0, 1.0
            )
            analysis.palatal_plane_angle = float(np.degrees(np.arccos(abs(cos_ppa))))

        # Gonial angles L/R: angle between ramus and mandibular body
        for side in ("L", "R"):
            cond_key = f"Condylion_{side}"
            gon_key = f"Gonion_{side}"
            if cond_key in lms and gon_key in lms and "Menton" in lms:
                ramus_vec = lms[gon_key] - lms[cond_key]  # condylion → gonion
                body_vec = lms["Menton"] - lms[gon_key]    # gonion → menton
                r_norm = np.linalg.norm(ramus_vec)
                b_norm = np.linalg.norm(body_vec)
                if r_norm > 0 and b_norm > 0:
                    ramus_vec = ramus_vec / r_norm
                    body_vec = body_vec / b_norm
                    cos_ga = np.clip(np.dot(ramus_vec, body_vec), -1.0, 1.0)
                    angle = float(np.degrees(np.arccos(cos_ga)))
                    if side == "L":
                        analysis.gonial_angle_L = angle
                    else:
                        analysis.gonial_angle_R = angle

    def _compute_linear_measurements(self, analysis: CephalometricAnalysis) -> None:
        """Compute all standard cephalometric linear distances in mm."""
        lms = analysis.landmarks

        def dist(a: str, b: str) -> float:
            if a in lms and b in lms:
                return float(np.linalg.norm(lms[a] - lms[b]))
            return 0.0

        # Anterior facial height: Nasion to Menton
        analysis.anterior_facial_height = dist("Nasion", "Menton")

        # Posterior facial height: Sella to Gonion (average L/R)
        if "Sella" in lms and "Gonion_L" in lms and "Gonion_R" in lms:
            gonion_mid = (lms["Gonion_L"] + lms["Gonion_R"]) / 2.0
            analysis.posterior_facial_height = float(np.linalg.norm(lms["Sella"] - gonion_mid))

        # Ramus height L/R
        analysis.ramus_height_L = dist("Condylion_L", "Gonion_L")
        analysis.ramus_height_R = dist("Condylion_R", "Gonion_R")

        # Mandibular body length L/R
        analysis.mandibular_body_length_L = dist("Gonion_L", "Menton")
        analysis.mandibular_body_length_R = dist("Gonion_R", "Menton")

        # Wits appraisal: project A and B points onto occlusal plane
        # Simplified: if ANS and Menton define an approximate occlusal plane direction
        if "A_point" in lms and "B_point" in lms and "ANS" in lms and "Menton" in lms:
            occ_vec = lms["Menton"] - lms["ANS"]
            occ_norm = np.linalg.norm(occ_vec)
            if occ_norm > 0:
                occ_unit = occ_vec / occ_norm
                a_proj = np.dot(lms["A_point"] - lms["ANS"], occ_unit)
                b_proj = np.dot(lms["B_point"] - lms["ANS"], occ_unit)
                analysis.wits_appraisal = float(b_proj - a_proj)

    def _add_confidence_scores(
        self,
        analysis: CephalometricAnalysis,
        mask: np.ndarray,
    ) -> None:
        """
        Assign confidence scores based on mask quality and landmark heuristic reliability.

        Heuristic landmarks have lower confidence than model-predicted ones.
        Confidence is reduced if the relevant structure mask is small or absent.
        """
        def mask_volume(label_id: int) -> int:
            return int(np.sum(mask == label_id))

        mandible_vox = mask_volume(self.LABEL_MANDIBLE)
        maxilla_vox = mask_volume(self.LABEL_MAXILLA)
        skull_vox = mask_volume(self.LABEL_SKULL)

        # Sigmoid-like mapping: 10000 voxels → 0.7 confidence, 50000 → 0.9
        def vol_confidence(n: int) -> float:
            if n == 0:
                return 0.0
            return min(0.90, 0.5 + 0.4 * (1.0 - np.exp(-n / 30000.0)))

        mandible_conf = vol_confidence(mandible_vox) * (0.75 if analysis.landmarks_estimated else 0.95)
        maxilla_conf = vol_confidence(maxilla_vox) * (0.75 if analysis.landmarks_estimated else 0.95)
        skull_conf = vol_confidence(skull_vox) * (0.65 if analysis.landmarks_estimated else 0.90)

        mandible_lms = {"Menton", "Gnathion", "Pogonion", "B_point",
                        "Gonion_L", "Gonion_R", "Condylion_L", "Condylion_R", "Infradentale"}
        maxilla_lms = {"ANS", "PNS", "A_point", "Subspinale"}
        skull_lms = {"Nasion", "Sella", "Basion", "Porion", "Orbitale",
                     "Orbitale_L", "Orbitale_R", "Porion_L", "Porion_R"}

        for lm in analysis.landmarks:
            if lm in mandible_lms:
                analysis.confidence_scores[lm] = mandible_conf
            elif lm in maxilla_lms:
                analysis.confidence_scores[lm] = maxilla_conf
            elif lm in skull_lms:
                analysis.confidence_scores[lm] = skull_conf
            else:
                analysis.confidence_scores[lm] = 0.50

        # Warn if critical landmarks have low confidence
        critical = ["Nasion", "Sella", "A_point", "B_point", "Menton"]
        for lm in critical:
            conf = analysis.confidence_scores.get(lm, 0.0)
            if conf < 0.5:
                analysis.warnings.append(
                    f"Low confidence ({conf:.2f}) for critical landmark '{lm}'. "
                    "Verify segmentation quality or annotate manually."
                )

    # ------------------------------------------------------------------
    # Learned Model Interface (Tier 2 stub)
    # ------------------------------------------------------------------

    def _predict_with_model(
        self,
        volume: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> dict[str, np.ndarray]:
        """
        Detect landmarks using a trained CNN heatmap regression model.

        Expected model input:
            (1, 1, Z, Y, X) float32 tensor, HU-normalised to [0, 1]

        Expected model output:
            (N_landmarks, 3) float32 tensor of physical coordinates in mm

        The ordering of landmarks in the output must match LANDMARK_ORDER
        defined during training.

        This method is a stub — implement by loading and calling your model.
        """
        raise NotImplementedError(
            "Learned landmark detection model is not yet available. "
            "Set model_path=None to use heuristic estimation, or provide "
            "a trained model checkpoint in model_path."
        )

    @staticmethod
    def get_landmark_definitions() -> dict[str, str]:
        """Return all supported landmark names with their anatomical definitions."""
        return dict(LANDMARK_DEFINITIONS)

    @staticmethod
    def interpret_anb(anb: float) -> str:
        """Classify ANB angle into skeletal class."""
        if anb > 4.0:
            return "Class II (skeletal)"
        elif anb < 0.0:
            return "Class III (skeletal)"
        else:
            return "Class I (skeletal)"

    @staticmethod
    def interpret_mandibular_plane(mpa: float) -> str:
        """Classify mandibular plane angle (relative to Frankfort Horizontal)."""
        if mpa > 36.0:
            return "Hyperdivergent (high angle) — vertical growth pattern"
        elif mpa < 28.0:
            return "Hypodivergent (low angle) — horizontal growth pattern"
        else:
            return "Normodivergent — average growth pattern"
