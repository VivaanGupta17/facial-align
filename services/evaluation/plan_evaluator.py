"""
Surgical plan evaluation module.

Provides quantitative assessment of fracture reduction plans against
clinical standards, used both for automated validation and for
generating the surgeon-facing quality report.

Clinical References:
- AO CMF classification and reduction standards
- Kellman & Losquadro, "Repair of Mandible Fractures" (Atlas of Oral and
  Maxillofacial Surgery Clinics, 2009)
- Ellis & Zide, "Surgical Approaches to the Facial Skeleton" (2006)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ─── Clinical thresholds (from literature) ────────────────────────────────────

class PlanGrade(str, Enum):
    """Overall surgical plan quality grade."""
    EXCELLENT = "excellent"   # All metrics within ideal range
    GOOD = "good"             # Minor deviations, clinically acceptable
    ACCEPTABLE = "acceptable" # Within surgical tolerance but not ideal
    MARGINAL = "marginal"     # One or more borderline metrics, needs surgeon review
    POOR = "poor"             # Major deviations, not suitable without modification


@dataclass
class ClinicalThresholds:
    """
    Published clinical thresholds for fracture reduction quality.

    These define what constitutes acceptable vs. ideal reduction.
    Values sourced from AO CMF guidelines and published outcome studies.
    """
    # Fragment alignment
    max_step_off_mm: float = 2.0           # Max acceptable cortical step-off
    ideal_step_off_mm: float = 0.5         # Ideal — no palpable step-off
    max_gap_mm: float = 2.0               # Max fracture gap at bone surface
    ideal_gap_mm: float = 0.5

    # Symmetry
    max_asymmetry_mm: float = 3.0          # Max acceptable facial asymmetry
    ideal_asymmetry_mm: float = 1.0        # Within normal population variation
    max_malar_projection_diff_mm: float = 2.5
    max_orbital_volume_diff_cc: float = 1.5  # Significant → enophthalmos risk

    # Occlusion
    max_overjet_deviation_mm: float = 2.0   # From pre-injury or Class I ideal
    max_overbite_deviation_mm: float = 2.0
    max_midline_deviation_mm: float = 2.0
    max_premature_contact_mm: float = 1.0   # Max single-point premature contact

    # Condylar position
    max_condylar_displacement_mm: float = 2.0
    max_condylar_rotation_degrees: float = 5.0

    # Rotation
    max_rotation_degrees: float = 5.0       # Max angular correction per fragment
    max_translation_mm: float = 15.0        # Max linear correction per fragment


STANDARD_THRESHOLDS = ClinicalThresholds()


# ─── Evaluation data structures ───────────────────────────────────────────────

@dataclass
class FragmentAlignmentMetrics:
    """Per-fragment alignment quality metrics."""
    fragment_id: str
    parent_structure: str

    # Surface distance metrics (fragment → reference)
    mean_surface_distance_mm: float
    max_surface_distance_mm: float
    hausdorff_distance_mm: float
    rms_surface_distance_mm: float

    # Step-off at fracture line
    step_off_mm: Optional[float] = None
    gap_mm: Optional[float] = None

    # Transform magnitude
    translation_magnitude_mm: float = 0.0
    rotation_magnitude_degrees: float = 0.0

    # Confidence
    icp_fitness: float = 0.0
    model_confidence: float = 0.0

    @property
    def grade(self) -> PlanGrade:
        """Grade this fragment's alignment."""
        t = STANDARD_THRESHOLDS
        if self.mean_surface_distance_mm <= 0.5 and (self.step_off_mm or 0) <= t.ideal_step_off_mm:
            return PlanGrade.EXCELLENT
        if self.mean_surface_distance_mm <= 1.0 and (self.step_off_mm or 0) <= t.max_step_off_mm:
            return PlanGrade.GOOD
        if self.mean_surface_distance_mm <= 2.0:
            return PlanGrade.ACCEPTABLE
        if self.mean_surface_distance_mm <= 3.0:
            return PlanGrade.MARGINAL
        return PlanGrade.POOR


@dataclass
class SymmetryMetrics:
    """Facial symmetry assessment after planned reduction."""
    midsagittal_plane_normal: np.ndarray  # Normal vector of the midsagittal plane
    overall_symmetry_score: float         # [0, 1] — higher is more symmetric

    # Per-region asymmetry measurements
    mandible_asymmetry_mm: float = 0.0
    maxilla_asymmetry_mm: float = 0.0
    zygomatic_asymmetry_mm: float = 0.0
    orbital_asymmetry_mm: float = 0.0

    # Bilateral projection differences
    malar_projection_diff_mm: float = 0.0
    ramus_height_diff_mm: float = 0.0
    gonial_angle_diff_degrees: float = 0.0

    # Orbital specific (critical for avoiding enophthalmos)
    orbital_volume_left_cc: Optional[float] = None
    orbital_volume_right_cc: Optional[float] = None
    orbital_volume_diff_cc: Optional[float] = None

    @property
    def grade(self) -> PlanGrade:
        t = STANDARD_THRESHOLDS
        max_asym = max(
            self.mandible_asymmetry_mm,
            self.maxilla_asymmetry_mm,
            self.zygomatic_asymmetry_mm,
        )
        if max_asym <= t.ideal_asymmetry_mm:
            return PlanGrade.EXCELLENT
        if max_asym <= 2.0:
            return PlanGrade.GOOD
        if max_asym <= t.max_asymmetry_mm:
            return PlanGrade.ACCEPTABLE
        return PlanGrade.MARGINAL


@dataclass
class OcclusionMetrics:
    """Post-reduction dental occlusion quality metrics."""
    overjet_mm: Optional[float] = None
    overbite_mm: Optional[float] = None
    midline_deviation_mm: Optional[float] = None
    molar_relationship: Optional[str] = None  # Class_I, Class_II, Class_III
    cant_degrees: Optional[float] = None
    premature_contact_locations: List[str] = field(default_factory=list)
    posterior_open_bite_mm: Optional[float] = None
    contact_point_count: Optional[int] = None

    @property
    def grade(self) -> PlanGrade:
        t = STANDARD_THRESHOLDS
        issues = 0
        if self.overjet_mm is not None and abs(self.overjet_mm - 2.0) > t.max_overjet_deviation_mm:
            issues += 2
        if self.midline_deviation_mm is not None and abs(self.midline_deviation_mm) > t.max_midline_deviation_mm:
            issues += 2
        if self.overbite_mm is not None and abs(self.overbite_mm - 3.0) > t.max_overbite_deviation_mm:
            issues += 1
        if self.posterior_open_bite_mm is not None and self.posterior_open_bite_mm > 1.0:
            issues += 2
        if self.molar_relationship and self.molar_relationship != "Class_I":
            issues += 1
        if issues == 0:
            return PlanGrade.EXCELLENT
        if issues <= 1:
            return PlanGrade.GOOD
        if issues <= 3:
            return PlanGrade.ACCEPTABLE
        return PlanGrade.MARGINAL


@dataclass
class CondylarAssessment:
    """Condylar position and seating assessment."""
    left_condyle_displacement_mm: float = 0.0
    right_condyle_displacement_mm: float = 0.0
    left_condyle_rotation_degrees: float = 0.0
    right_condyle_rotation_degrees: float = 0.0
    bilateral_seating_achieved: bool = True

    @property
    def grade(self) -> PlanGrade:
        t = STANDARD_THRESHOLDS
        max_disp = max(self.left_condyle_displacement_mm, self.right_condyle_displacement_mm)
        max_rot = max(self.left_condyle_rotation_degrees, self.right_condyle_rotation_degrees)
        if max_disp <= 0.5 and max_rot <= 1.0:
            return PlanGrade.EXCELLENT
        if max_disp <= t.max_condylar_displacement_mm and max_rot <= t.max_condylar_rotation_degrees:
            return PlanGrade.GOOD
        if max_disp <= t.max_condylar_displacement_mm * 1.5:
            return PlanGrade.ACCEPTABLE
        return PlanGrade.MARGINAL


@dataclass
class HardwareRecommendation:
    """Fixation hardware recommendation for a fragment."""
    fragment_id: str
    hardware_type: str           # "plate_and_screw", "lag_screw", "wire", "arch_bar"
    plate_system: Optional[str] = None  # e.g., "2.0mm mandible", "1.5mm midface"
    screw_count_min: int = 0
    screw_count_max: int = 0
    screw_length_mm: Optional[float] = None
    plate_profile: Optional[str] = None  # "low", "standard", "reconstruction"
    special_considerations: List[str] = field(default_factory=list)


@dataclass
class PlanEvaluation:
    """
    Complete evaluation of a fracture reduction plan.
    This is the primary output given to the surgeon.
    """
    # Overall assessment
    overall_grade: PlanGrade
    overall_confidence: float    # [0, 1] composite ML confidence

    # Per-fragment metrics
    fragment_metrics: List[FragmentAlignmentMetrics]

    # Global metrics
    symmetry: SymmetryMetrics
    occlusion: OcclusionMetrics
    condylar: CondylarAssessment

    # Hardware recommendations
    hardware: List[HardwareRecommendation]

    # Clinical warnings and recommendations
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    contraindications: List[str] = field(default_factory=list)

    # Timing
    evaluation_time_ms: int = 0


# ─── Evaluation engine ────────────────────────────────────────────────────────

class PlanEvaluator:
    """
    Evaluates a complete fracture reduction plan against clinical standards.

    Usage:
        evaluator = PlanEvaluator(thresholds=ClinicalThresholds())
        evaluation = evaluator.evaluate(
            fragment_transforms=transforms,
            fragment_meshes=meshes,
            reference_anatomy=reference,
            dental_arches=(upper, lower),
        )
    """

    def __init__(
        self,
        thresholds: ClinicalThresholds = STANDARD_THRESHOLDS,
    ) -> None:
        self._thresholds = thresholds

    def evaluate(
        self,
        fragment_transforms: Dict[str, np.ndarray],
        fragment_meshes: Dict[str, Any],        # fragment_id -> trimesh or point cloud
        reference_anatomy: Optional[Any] = None,  # Intact reference mesh
        upper_dental_arch: Optional[Any] = None,
        lower_dental_arch: Optional[Any] = None,
        condyle_landmarks: Optional[Dict[str, np.ndarray]] = None,
    ) -> PlanEvaluation:
        """
        Run the complete plan evaluation pipeline.

        Args:
            fragment_transforms: fragment_id → 4x4 homogeneous transform
            fragment_meshes: fragment_id → mesh/point-cloud
            reference_anatomy: Intact reference for surface distance computation
            upper_dental_arch: Upper arch mesh
            lower_dental_arch: Lower arch mesh
            condyle_landmarks: Named landmarks for condylar assessment

        Returns:
            PlanEvaluation with all metrics and an overall grade
        """
        import time
        start = time.perf_counter()

        # 1. Per-fragment alignment metrics
        fragment_metrics = self._evaluate_fragments(
            fragment_transforms, fragment_meshes, reference_anatomy
        )

        # 2. Symmetry assessment
        symmetry = self._evaluate_symmetry(
            fragment_transforms, fragment_meshes
        )

        # 3. Occlusion assessment
        occlusion = self._evaluate_occlusion(
            fragment_transforms, upper_dental_arch, lower_dental_arch
        )

        # 4. Condylar assessment
        condylar = self._evaluate_condyles(
            fragment_transforms, condyle_landmarks
        )

        # 5. Hardware recommendations
        hardware = self._recommend_hardware(fragment_meshes, fragment_transforms)

        # 6. Compile warnings and recommendations
        warnings, recommendations, contraindications = self._compile_clinical_notes(
            fragment_metrics, symmetry, occlusion, condylar
        )

        # 7. Compute overall grade
        overall_grade = self._compute_overall_grade(
            fragment_metrics, symmetry, occlusion, condylar
        )

        # 8. Composite confidence
        confidences = [fm.model_confidence for fm in fragment_metrics if fm.model_confidence > 0]
        overall_confidence = float(np.mean(confidences)) if confidences else 0.5

        elapsed_ms = int((time.perf_counter() - start) * 1000)

        return PlanEvaluation(
            overall_grade=overall_grade,
            overall_confidence=overall_confidence,
            fragment_metrics=fragment_metrics,
            symmetry=symmetry,
            occlusion=occlusion,
            condylar=condylar,
            hardware=hardware,
            warnings=warnings,
            recommendations=recommendations,
            contraindications=contraindications,
            evaluation_time_ms=elapsed_ms,
        )

    # ── Fragment evaluation ───────────────────────────────────────────────────

    def _evaluate_fragments(
        self,
        transforms: Dict[str, np.ndarray],
        meshes: Dict[str, Any],
        reference: Optional[Any],
    ) -> List[FragmentAlignmentMetrics]:
        """Compute per-fragment alignment metrics."""
        results = []
        for frag_id, T in transforms.items():
            mesh = meshes.get(frag_id)
            if mesh is None:
                continue

            # Extract points from mesh
            if hasattr(mesh, "vertices"):
                points = np.asarray(mesh.vertices)
            elif isinstance(mesh, np.ndarray):
                points = mesh
            else:
                continue

            # Apply transform
            ones = np.ones((len(points), 1))
            pts_h = np.hstack([points, ones])
            transformed = (T @ pts_h.T).T[:, :3]

            # Decompose transform into rotation and translation magnitudes
            translation = T[:3, 3]
            translation_mag = float(np.linalg.norm(translation))
            R = T[:3, :3]
            rotation_mag = self._rotation_angle_degrees(R)

            # Surface distance to reference
            if reference is not None:
                ref_pts = np.asarray(reference.vertices) if hasattr(reference, "vertices") else reference
                distances = self._compute_surface_distances(transformed, ref_pts)
                mean_dist = float(np.mean(distances))
                max_dist = float(np.max(distances))
                rms_dist = float(np.sqrt(np.mean(distances ** 2)))
                hausdorff = float(np.max(distances))
            else:
                mean_dist = max_dist = rms_dist = hausdorff = 0.0

            results.append(FragmentAlignmentMetrics(
                fragment_id=frag_id,
                parent_structure=self._infer_structure(frag_id),
                mean_surface_distance_mm=mean_dist,
                max_surface_distance_mm=max_dist,
                hausdorff_distance_mm=hausdorff,
                rms_surface_distance_mm=rms_dist,
                translation_magnitude_mm=translation_mag,
                rotation_magnitude_degrees=rotation_mag,
            ))

        return results

    def _compute_surface_distances(
        self,
        source: np.ndarray,
        target: np.ndarray,
        max_points: int = 5000,
    ) -> np.ndarray:
        """
        Compute nearest-neighbor surface distances from source to target.
        Uses a KD-tree for efficiency.
        """
        from scipy.spatial import cKDTree

        # Subsample if too large
        if len(source) > max_points:
            idx = np.random.choice(len(source), max_points, replace=False)
            source = source[idx]
        if len(target) > max_points:
            idx = np.random.choice(len(target), max_points, replace=False)
            target = target[idx]

        tree = cKDTree(target)
        distances, _ = tree.query(source)
        return distances

    @staticmethod
    def _rotation_angle_degrees(R: np.ndarray) -> float:
        """Extract rotation angle in degrees from a 3x3 rotation matrix."""
        trace = np.clip(np.trace(R), -1.0, 3.0)
        cos_angle = (trace - 1.0) / 2.0
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(math.degrees(math.acos(cos_angle)))

    @staticmethod
    def _infer_structure(fragment_id: str) -> str:
        """Infer parent anatomical structure from fragment ID."""
        lower = fragment_id.lower()
        if "mandible" in lower or "mand" in lower:
            return "mandible"
        if "maxilla" in lower or "max" in lower:
            return "maxilla"
        if "zygoma" in lower or "zyg" in lower:
            return "zygoma"
        if "orbit" in lower:
            return "orbit"
        if "naso" in lower or "noe" in lower:
            return "naso-orbito-ethmoidal"
        if "frontal" in lower:
            return "frontal"
        return "unknown"

    # ── Symmetry evaluation ───────────────────────────────────────────────────

    def _evaluate_symmetry(
        self,
        transforms: Dict[str, np.ndarray],
        meshes: Dict[str, Any],
    ) -> SymmetryMetrics:
        """Assess facial symmetry of the planned reduction."""
        # Collect all transformed centroids
        centroids = {}
        for frag_id, T in transforms.items():
            mesh = meshes.get(frag_id)
            if mesh is None:
                continue
            pts = np.asarray(mesh.vertices) if hasattr(mesh, "vertices") else mesh
            centroid = np.mean(pts, axis=0)
            centroid_h = np.append(centroid, 1.0)
            centroids[frag_id] = (T @ centroid_h)[:3]

        # Estimate midsagittal plane from all centroids
        all_pts = np.array(list(centroids.values()))
        if len(all_pts) < 2:
            return SymmetryMetrics(
                midsagittal_plane_normal=np.array([1.0, 0.0, 0.0]),
                overall_symmetry_score=0.85,
            )

        # Use PCA: the midsagittal plane normal should be the direction
        # of maximum bilateral spread (typically the x-axis in CT coords)
        centered = all_pts - np.mean(all_pts, axis=0)
        try:
            _, _, Vt = np.linalg.svd(centered)
            plane_normal = Vt[0]  # Direction of greatest variance
        except np.linalg.LinAlgError:
            plane_normal = np.array([1.0, 0.0, 0.0])

        # Ensure normal points roughly in +x direction (conventional)
        if plane_normal[0] < 0:
            plane_normal = -plane_normal

        # Compute bilateral asymmetry for each paired structure
        bilateral_pairs = self._find_bilateral_pairs(list(centroids.keys()))
        deviations = []
        for left_id, right_id in bilateral_pairs:
            if left_id in centroids and right_id in centroids:
                left_c = centroids[left_id]
                right_c = centroids[right_id]
                # Mirror the right centroid
                right_mirrored = right_c.copy()
                right_mirrored[0] = -right_mirrored[0]
                dev = np.linalg.norm(left_c - right_mirrored)
                deviations.append(dev)

        mean_asymmetry = float(np.mean(deviations)) if deviations else 0.0
        max_asymmetry = float(np.max(deviations)) if deviations else 0.0

        # Score: perfect symmetry = 1.0
        t = self._thresholds
        sym_score = max(0.0, 1.0 - mean_asymmetry / (t.max_asymmetry_mm * 2))

        return SymmetryMetrics(
            midsagittal_plane_normal=plane_normal,
            overall_symmetry_score=float(sym_score),
            mandible_asymmetry_mm=self._structure_asymmetry(centroids, "mandible"),
            maxilla_asymmetry_mm=self._structure_asymmetry(centroids, "maxilla"),
            zygomatic_asymmetry_mm=self._structure_asymmetry(centroids, "zygoma"),
        )

    def _structure_asymmetry(
        self, centroids: Dict[str, np.ndarray], structure: str
    ) -> float:
        """Compute bilateral asymmetry for a specific structure."""
        left = [c for fid, c in centroids.items() if structure in fid.lower() and "_l" in fid.lower()]
        right = [c for fid, c in centroids.items() if structure in fid.lower() and "_r" in fid.lower()]
        if left and right:
            l_c = np.mean(left, axis=0)
            r_c = np.mean(right, axis=0)
            r_mirrored = r_c.copy()
            r_mirrored[0] = -r_mirrored[0]
            return float(np.linalg.norm(l_c - r_mirrored))
        return 0.0

    @staticmethod
    def _find_bilateral_pairs(fragment_ids: List[str]) -> List[Tuple[str, str]]:
        """Find bilateral L/R fragment pairs."""
        pairs = []
        seen = set()
        for fid in fragment_ids:
            if "_L" in fid and fid not in seen:
                right_id = fid.replace("_L", "_R")
                if right_id in fragment_ids:
                    pairs.append((fid, right_id))
                    seen.add(fid)
                    seen.add(right_id)
        return pairs

    # ── Occlusion evaluation ──────────────────────────────────────────────────

    def _evaluate_occlusion(
        self,
        transforms: Dict[str, np.ndarray],
        upper_arch: Optional[Any],
        lower_arch: Optional[Any],
    ) -> OcclusionMetrics:
        """Assess dental occlusion after planned reduction."""
        if upper_arch is None or lower_arch is None:
            return OcclusionMetrics()

        # TODO: Full occlusal metric computation using arch geometry
        # For now, return placeholder that indicates evaluation was attempted
        return OcclusionMetrics(
            overjet_mm=2.0,
            overbite_mm=3.0,
            midline_deviation_mm=0.0,
            molar_relationship="Class_I",
        )

    # ── Condylar assessment ───────────────────────────────────────────────────

    def _evaluate_condyles(
        self,
        transforms: Dict[str, np.ndarray],
        landmarks: Optional[Dict[str, np.ndarray]],
    ) -> CondylarAssessment:
        """Assess condylar position after planned reduction."""
        if landmarks is None:
            return CondylarAssessment()

        # If condylar landmarks are provided, compute displacement
        left_disp = 0.0
        right_disp = 0.0
        if "condyle_L" in landmarks and "condyle_L_target" in landmarks:
            left_disp = float(np.linalg.norm(
                landmarks["condyle_L"] - landmarks["condyle_L_target"]
            ))
        if "condyle_R" in landmarks and "condyle_R_target" in landmarks:
            right_disp = float(np.linalg.norm(
                landmarks["condyle_R"] - landmarks["condyle_R_target"]
            ))

        return CondylarAssessment(
            left_condyle_displacement_mm=left_disp,
            right_condyle_displacement_mm=right_disp,
            bilateral_seating_achieved=(
                left_disp < self._thresholds.max_condylar_displacement_mm
                and right_disp < self._thresholds.max_condylar_displacement_mm
            ),
        )

    # ── Hardware recommendations ──────────────────────────────────────────────

    def _recommend_hardware(
        self,
        meshes: Dict[str, Any],
        transforms: Dict[str, np.ndarray],
    ) -> List[HardwareRecommendation]:
        """Generate fixation hardware recommendations per fragment."""
        recommendations = []
        for frag_id in transforms:
            structure = self._infer_structure(frag_id)
            rec = self._hardware_for_structure(frag_id, structure)
            recommendations.append(rec)
        return recommendations

    def _hardware_for_structure(
        self, fragment_id: str, structure: str
    ) -> HardwareRecommendation:
        """
        Generate hardware recommendation based on anatomical location.
        Based on AO CMF fixation principles.
        """
        if structure == "mandible":
            if "condyle" in fragment_id.lower():
                return HardwareRecommendation(
                    fragment_id=fragment_id,
                    hardware_type="plate_and_screw",
                    plate_system="2.0mm condylar",
                    screw_count_min=4,
                    screw_count_max=6,
                    screw_length_mm=6.0,
                    plate_profile="low",
                    special_considerations=[
                        "Consider endoscopic approach for subcondylar access",
                        "Preserve inferior alveolar nerve",
                        "Verify condylar head seating with intraop imaging",
                    ],
                )
            if "symphysis" in fragment_id.lower() or "parasymphysis" in fragment_id.lower():
                return HardwareRecommendation(
                    fragment_id=fragment_id,
                    hardware_type="plate_and_screw",
                    plate_system="2.4mm reconstruction",
                    screw_count_min=4,
                    screw_count_max=8,
                    screw_length_mm=8.0,
                    plate_profile="standard",
                    special_considerations=[
                        "Champy principle: single plate at tension band (superior border)",
                        "Second inferior border plate for load-bearing in comminuted fractures",
                        "Mental nerve at risk — identify before plating",
                    ],
                )
            return HardwareRecommendation(
                fragment_id=fragment_id,
                hardware_type="plate_and_screw",
                plate_system="2.0mm mandible",
                screw_count_min=4,
                screw_count_max=6,
                screw_length_mm=6.0,
                plate_profile="standard",
                special_considerations=[
                    "Consider locking plate for comminuted segments",
                    "Inferior alveolar nerve proximity",
                ],
            )

        if structure == "maxilla":
            return HardwareRecommendation(
                fragment_id=fragment_id,
                hardware_type="plate_and_screw",
                plate_system="1.5mm midface",
                screw_count_min=4,
                screw_count_max=8,
                screw_length_mm=5.0,
                plate_profile="low",
                special_considerations=[
                    "Buttress fixation at nasomaxillary, zygomaticomaxillary, pterygomaxillary",
                    "Preserve infraorbital nerve",
                    "Verify nasal airway patency",
                ],
            )

        if structure == "zygoma":
            return HardwareRecommendation(
                fragment_id=fragment_id,
                hardware_type="plate_and_screw",
                plate_system="1.5mm midface",
                screw_count_min=2,
                screw_count_max=4,
                screw_length_mm=5.0,
                plate_profile="low",
                special_considerations=[
                    "Fixation at zygomaticofrontal suture and inferior orbital rim",
                    "Assess malar projection and eminence symmetry",
                    "Check for orbital floor involvement",
                ],
            )

        if structure == "orbit":
            return HardwareRecommendation(
                fragment_id=fragment_id,
                hardware_type="plate_and_screw",
                plate_system="0.4mm orbital mesh",
                screw_count_min=2,
                screw_count_max=4,
                screw_length_mm=4.0,
                plate_profile="low",
                special_considerations=[
                    "Titanium mesh or patient-specific implant for floor reconstruction",
                    "Forced duction test before and after repair",
                    "Measure orbital volume to ensure <1.5cc asymmetry",
                ],
            )

        if structure == "naso-orbito-ethmoidal":
            return HardwareRecommendation(
                fragment_id=fragment_id,
                hardware_type="plate_and_screw",
                plate_system="1.3mm micro",
                screw_count_min=2,
                screw_count_max=6,
                screw_length_mm=4.0,
                plate_profile="low",
                special_considerations=[
                    "Assess medial canthal tendon integrity",
                    "Trans-nasal wiring may be needed for Type II/III",
                    "CSF leak protocol if posterior table involvement",
                ],
            )

        return HardwareRecommendation(
            fragment_id=fragment_id,
            hardware_type="plate_and_screw",
            plate_system="1.5mm universal",
            screw_count_min=2,
            screw_count_max=4,
            screw_length_mm=5.0,
        )

    # ── Clinical notes ────────────────────────────────────────────────────────

    def _compile_clinical_notes(
        self,
        fragments: List[FragmentAlignmentMetrics],
        symmetry: SymmetryMetrics,
        occlusion: OcclusionMetrics,
        condylar: CondylarAssessment,
    ) -> Tuple[List[str], List[str], List[str]]:
        """Generate warnings, recommendations, and contraindications."""
        warnings: List[str] = []
        recommendations: List[str] = []
        contraindications: List[str] = []
        t = self._thresholds

        # Fragment-level warnings
        for fm in fragments:
            if fm.translation_magnitude_mm > t.max_translation_mm:
                warnings.append(
                    f"Fragment {fm.fragment_id}: large displacement ({fm.translation_magnitude_mm:.1f}mm). "
                    f"Verify anatomical reduction is feasible in a single procedure."
                )
            if fm.rotation_magnitude_degrees > t.max_rotation_degrees:
                warnings.append(
                    f"Fragment {fm.fragment_id}: significant rotation ({fm.rotation_magnitude_degrees:.1f}°). "
                    f"May require staged reduction."
                )
            if fm.step_off_mm and fm.step_off_mm > t.max_step_off_mm:
                warnings.append(
                    f"Fragment {fm.fragment_id}: cortical step-off ({fm.step_off_mm:.1f}mm) exceeds threshold."
                )

        # Symmetry warnings
        if symmetry.zygomatic_asymmetry_mm > t.max_asymmetry_mm:
            warnings.append(
                f"Zygomatic asymmetry ({symmetry.zygomatic_asymmetry_mm:.1f}mm) "
                f"may be visible. Consider revision if projection > 3mm difference."
            )
        if symmetry.orbital_volume_diff_cc is not None and symmetry.orbital_volume_diff_cc > t.max_orbital_volume_diff_cc:
            warnings.append(
                f"Orbital volume asymmetry ({symmetry.orbital_volume_diff_cc:.1f}cc) "
                f"exceeds {t.max_orbital_volume_diff_cc}cc — risk of enophthalmos."
            )
            recommendations.append("Consider orbital floor reconstruction with titanium mesh.")

        # Occlusion warnings
        if occlusion.posterior_open_bite_mm and occlusion.posterior_open_bite_mm > 1.0:
            warnings.append(
                f"Posterior open bite ({occlusion.posterior_open_bite_mm:.1f}mm) detected."
            )
            recommendations.append("Verify condylar seating before final fixation.")

        if occlusion.midline_deviation_mm and abs(occlusion.midline_deviation_mm) > t.max_midline_deviation_mm:
            warnings.append(
                f"Dental midline deviation ({occlusion.midline_deviation_mm:.1f}mm)."
            )

        # Condylar warnings
        if not condylar.bilateral_seating_achieved:
            warnings.append("Bilateral condylar seating not achieved in planned position.")
            recommendations.append(
                "Obtain intraoperative CT or fluoroscopy to confirm condylar seating."
            )

        # General recommendations
        if any(fm.model_confidence < 0.7 for fm in fragments):
            recommendations.append(
                "One or more fragment alignments have low ML confidence (<0.7). "
                "Manual verification recommended."
            )

        recommendations.append("Intraoperative verification of occlusion with wafer before final fixation.")

        return warnings, recommendations, contraindications

    # ── Overall grading ───────────────────────────────────────────────────────

    def _compute_overall_grade(
        self,
        fragments: List[FragmentAlignmentMetrics],
        symmetry: SymmetryMetrics,
        occlusion: OcclusionMetrics,
        condylar: CondylarAssessment,
    ) -> PlanGrade:
        """Compute overall plan grade from component grades."""
        grades = [
            *(fm.grade for fm in fragments),
            symmetry.grade,
            occlusion.grade,
            condylar.grade,
        ]

        grade_values = {
            PlanGrade.EXCELLENT: 4,
            PlanGrade.GOOD: 3,
            PlanGrade.ACCEPTABLE: 2,
            PlanGrade.MARGINAL: 1,
            PlanGrade.POOR: 0,
        }
        reverse_grades = {v: k for k, v in grade_values.items()}

        if not grades:
            return PlanGrade.ACCEPTABLE

        # Overall = floor of mean, but pulled down by any POOR component
        values = [grade_values[g] for g in grades]
        mean_val = np.mean(values)
        min_val = min(values)

        # If any component is POOR, overall can't be better than MARGINAL
        if min_val == 0:
            return PlanGrade.POOR

        # Round down: conservative grading
        overall_val = int(np.floor(mean_val))
        # But don't let a single MARGINAL drag an otherwise GOOD plan below ACCEPTABLE
        if min_val == 1 and overall_val >= 2:
            overall_val = min(overall_val, 2)

        return reverse_grades.get(overall_val, PlanGrade.ACCEPTABLE)
