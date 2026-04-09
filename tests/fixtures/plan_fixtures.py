"""
Surgical planning test fixtures for Facial Align.

Provides builders for reduction plans, occlusion metrics, cephalometric
analyses, symmetry reports, and quality reports.  All values are chosen
to represent clinically plausible data for a typical craniofacial CT.

Usage:
    from tests.fixtures.plan_fixtures import (
        make_reduction_plan,
        make_occlusion_metrics,
        make_cephalometric_analysis,
        NORMAL_CEPH_VALUES,
    )
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Type aliases (mirrors data_contracts conventions)
# ---------------------------------------------------------------------------
MillimeterValue = float
DegreeValue = float

# ---------------------------------------------------------------------------
# Reference cephalometric value tables
# ---------------------------------------------------------------------------

NORMAL_CEPH_VALUES: Dict[str, Tuple[float, float]] = {
    # (mean, ±1 SD) — literature values for adult Caucasian norms
    # Skeletal
    "sna_degrees": (82.0, 3.5),          # SNA angle
    "snb_degrees": (80.0, 3.0),          # SNB angle
    "anb_degrees": (2.0, 2.0),           # ANB angle (Class I: 0–4°)
    "wits_appraisal_mm": (0.0, 2.0),     # Wits appraisal
    "facial_axis_degrees": (90.0, 3.5),  # Ricketts
    "y_axis_degrees": (59.0, 3.5),       # SN-Gn
    "gonial_angle_degrees": (122.0, 5.0),
    "ramus_inclination_degrees": (76.0, 4.0),
    # Dental
    "upper_incisor_to_sn_degrees": (104.0, 5.0),
    "lower_incisor_to_mp_degrees": (94.0, 5.0),
    "interincisal_angle_degrees": (135.0, 10.0),
    "overjet_mm": (3.0, 1.5),
    "overbite_mm": (3.0, 1.5),
    # Soft tissue
    "nasolabial_angle_degrees": (102.0, 8.0),
    "upper_lip_to_e_plane_mm": (-1.0, 2.0),
    "lower_lip_to_e_plane_mm": (1.0, 2.5),
    "facial_convexity_degrees": (12.0, 4.0),
    # Vertical
    "lower_facial_height_degrees": (47.0, 4.0),
    "posterior_facial_height_mm": (75.0, 5.0),
    "anterior_facial_height_mm": (120.0, 6.0),
    "pfh_afh_ratio": (0.62, 0.04),
}

CLASS_II_CEPH_VALUES: Dict[str, float] = {
    # Class II Division 1 malocclusion (mandibular retrognathia)
    "sna_degrees": 83.5,
    "snb_degrees": 75.0,     # retrognathic mandible
    "anb_degrees": 8.5,      # increased ANB
    "wits_appraisal_mm": 5.0,
    "overjet_mm": 9.0,       # markedly increased overjet
    "overbite_mm": 5.0,
    "upper_incisor_to_sn_degrees": 115.0,
    "lower_incisor_to_mp_degrees": 97.0,
    "interincisal_angle_degrees": 118.0,
    "gonial_angle_degrees": 118.0,      # hyperdivergent
    "lower_facial_height_degrees": 53.0,
    "nasolabial_angle_degrees": 95.0,
    "upper_lip_to_e_plane_mm": 3.0,
    "lower_lip_to_e_plane_mm": -3.0,
}

CLASS_III_CEPH_VALUES: Dict[str, float] = {
    # Class III malocclusion (mandibular prognathism)
    "sna_degrees": 80.0,
    "snb_degrees": 85.0,     # prognathic mandible
    "anb_degrees": -5.0,     # negative ANB
    "wits_appraisal_mm": -7.0,
    "overjet_mm": -3.5,      # negative overjet (anterior cross-bite)
    "overbite_mm": 1.0,
    "upper_incisor_to_sn_degrees": 98.0,
    "lower_incisor_to_mp_degrees": 82.0,
    "interincisal_angle_degrees": 155.0,
    "gonial_angle_degrees": 128.0,
    "lower_facial_height_degrees": 44.0,
    "nasolabial_angle_degrees": 108.0,
    "upper_lip_to_e_plane_mm": -4.0,
    "lower_lip_to_e_plane_mm": 5.0,
}


# ---------------------------------------------------------------------------
# Rotation / transform helpers
# ---------------------------------------------------------------------------

def _rotation_matrix_z(angle_deg: float) -> List[List[float]]:
    """Return a 3×3 rotation matrix for a rotation about the Z axis."""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def _rotation_matrix_y(angle_deg: float) -> List[List[float]]:
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


def _rotation_matrix_x(angle_deg: float) -> List[List[float]]:
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]


# ---------------------------------------------------------------------------
# Fragment transform builder
# ---------------------------------------------------------------------------

def make_fragment_transform(
    fragment_id: str,
    translation_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation_deg: float = 0.0,
    rotation_axis: str = "z",
    confidence: float = 0.90,
    is_reference: bool = False,
    surgical_sequence: Optional[int] = None,
) -> dict:
    """
    Build a single fragment transform dict suitable for use in a ReductionPlan.

    Parameters
    ----------
    fragment_id : str
        Unique identifier for the fragment (e.g. ``"frag_01"``).
    translation_mm : tuple[float, float, float]
        [tx, ty, tz] translation in mm.
    rotation_deg : float
        Rotation angle in degrees about the specified axis.
    rotation_axis : str
        Axis of rotation: ``"x"``, ``"y"``, or ``"z"``.
    confidence : float
        ML model confidence in [0, 1].
    is_reference : bool
        Whether this fragment anchors the reduction.
    surgical_sequence : int, optional
        Recommended surgical order (1-indexed).

    Returns
    -------
    dict
        Serialisable transform dict matching ``FractureFragmentContract`` schema.
    """
    rot_builders = {"x": _rotation_matrix_x, "y": _rotation_matrix_y, "z": _rotation_matrix_z}
    rotation = rot_builders[rotation_axis](rotation_deg)

    return {
        "fragment_id": fragment_id,
        "rotation_matrix": rotation,
        "translation_mm": list(translation_mm),
        "confidence": confidence,
        "is_reference": is_reference,
        "is_surgeon_edit": False,
        "surgical_sequence": surgical_sequence,
        "alternative_transforms": [],
    }


# ---------------------------------------------------------------------------
# Reduction plan builder
# ---------------------------------------------------------------------------

def make_reduction_plan(
    n_fragments: int = 3,
    case_id: Optional[str] = None,
    plan_id: Optional[str] = None,
    overall_confidence: float = 0.87,
    surgeon_approved: bool = False,
) -> dict:
    """
    Construct a complete ReductionPlan-shaped dict with ``n_fragments`` entries.

    Fragment 0 is always the reference fragment (identity transform).
    Remaining fragments have small realistic displacements representing
    a typical mandibular fracture pattern (symphysis + parasymphysis).

    Parameters
    ----------
    n_fragments : int
        Total fragment count including the reference.
    case_id : str, optional
        UUID string; auto-generated if None.
    plan_id : str, optional
        UUID string; auto-generated if None.
    overall_confidence : float
        Plan-level confidence score.
    surgeon_approved : bool
        Approval state.

    Returns
    -------
    dict
        Fully populated plan dict.
    """
    case_id = case_id or str(uuid.uuid4())
    plan_id = plan_id or str(uuid.uuid4())

    # Typical fragment displacements (mm, degrees) per fragment type
    displacement_templates = [
        # (tx, ty, tz, rot_deg, axis)
        (0.0, 0.0, 0.0, 0.0, "z"),          # reference — no movement
        (4.5, -2.0, 1.5, 5.2, "z"),          # parasymphysis fragment
        (-3.8, 1.8, -1.2, -4.7, "z"),        # contralateral parasymphysis
        (12.0, 5.0, 3.0, 15.0, "y"),         # right condyle
        (-11.5, 4.8, 2.8, -14.5, "y"),       # left condyle
    ]

    fragments = []
    for i in range(n_fragments):
        if i < len(displacement_templates):
            tx, ty, tz, rd, ax = displacement_templates[i]
        else:
            rng = np.random.default_rng(seed=i)
            tx, ty, tz = rng.uniform(-10, 10, 3).tolist()
            rd = float(rng.uniform(-20, 20))
            ax = "z"

        transform = make_fragment_transform(
            fragment_id=f"frag_{i:02d}",
            translation_mm=(tx, ty, tz),
            rotation_deg=rd,
            rotation_axis=ax,
            confidence=max(0.6, overall_confidence - i * 0.05),
            is_reference=(i == 0),
            surgical_sequence=None if i == 0 else i,
        )

        geom = {
            "fragment_id": f"frag_{i:02d}",
            "label_value": i + 1,
            "parent_structure": "mandible",
            "centroid_mm": [float(tx), float(ty), float(tz)],
            "volume_mm3": max(200.0, 1500.0 - i * 200.0),
            "surface_area_mm2": max(100.0, 800.0 - i * 80.0),
            "is_reference": i == 0,
            "confidence": transform["confidence"],
        }

        fragments.append({
            "geometry": geom,
            "planned_transform": transform,
            "hardware_recommendation": "miniplate_and_screws" if i > 0 else None,
            "surgical_sequence": None if i == 0 else i,
            "notes": None,
        })

    return {
        "plan_id": plan_id,
        "case_id": case_id,
        "plan_version": 1,
        "model_name": "baseline_icp",
        "model_version": "1.3.0",
        "fragments": fragments,
        "occlusal_metrics": make_occlusion_metrics(),
        "validation": {
            "passed": True,
            "symmetry_ok": True,
            "occlusion_ok": True,
            "condylar_seating_ok": True,
            "hardware_placement_ok": True,
            "warnings": [],
            "errors": [],
            "skeletal_symmetry_score": 0.91,
        },
        "overall_confidence": overall_confidence,
        "symmetry_score": 0.91,
        "surgeon_approved": surgeon_approved,
        "is_ml_generated": True,
        "generation_time_ms": 4250,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "approved_at": None,
    }


# ---------------------------------------------------------------------------
# Occlusion metrics builder
# ---------------------------------------------------------------------------

def make_occlusion_metrics(
    overjet: float = 2.0,
    overbite: float = 3.0,
    molar_class: str = "Class_I",
    midline_deviation: float = 0.5,
    cant_degrees: float = 0.8,
    curve_of_spee: float = 1.5,
    constraints_satisfied: bool = True,
    violations: Optional[List[str]] = None,
) -> dict:
    """
    Build a configurable occlusion-metrics dict.

    Parameters
    ----------
    overjet : float
        Horizontal incisal overjet in mm (normal 1–3 mm).
    overbite : float
        Vertical incisal overbite in mm (normal 2–4 mm).
    molar_class : str
        Angle molar classification: ``"Class_I"``, ``"Class_II_div1"``, etc.
    midline_deviation : float
        Upper-to-lower midline deviation in mm.
    cant_degrees : float
        Occlusal plane cant in degrees.
    curve_of_spee : float
        Depth of curve of Spee in mm.
    constraints_satisfied : bool
        Whether all clinical constraints are met.
    violations : list[str], optional
        List of violated constraint descriptions.

    Returns
    -------
    dict
        Serialisable occlusion-metrics dict.
    """
    return {
        "overjet_mm": overjet,
        "overbite_mm": overbite,
        "molar_relationship": molar_class,
        "midline_deviation_mm": midline_deviation,
        "cant_degrees": cant_degrees,
        "curve_of_spee_mm": curve_of_spee,
        "posterior_open_bite_mm": 0.0,
        "anterior_open_bite_mm": 0.0,
        "contact_points": 24,
        "constraints_satisfied": constraints_satisfied,
        "constraint_violations": violations or [],
    }


# ---------------------------------------------------------------------------
# Cephalometric analysis builders
# ---------------------------------------------------------------------------

def make_cephalometric_analysis(
    values: Optional[Dict[str, float]] = None,
    patient_id: Optional[str] = None,
    study_date: Optional[str] = None,
) -> dict:
    """
    Build a normal cephalometric analysis dict using standard norms.

    Parameters
    ----------
    values : dict, optional
        Overrides for specific measurements. Unknown keys are ignored.
    patient_id : str, optional
        Patient UUID.
    study_date : str, optional
        ISO-format date string.

    Returns
    -------
    dict
        Full cephalometric analysis with all standard measurements.
    """
    # Start from NORMAL_CEPH_VALUES means (column 0 of each tuple)
    base = {k: v[0] for k, v in NORMAL_CEPH_VALUES.items()}
    if values:
        base.update(values)

    measurements = []
    for name, mean_val in base.items():
        unit = "degrees" if "degrees" in name else ("mm" if "_mm" in name else "ratio")
        norm_range = NORMAL_CEPH_VALUES.get(name)
        in_range = True
        if norm_range:
            lo = norm_range[0] - 2 * norm_range[1]
            hi = norm_range[0] + 2 * norm_range[1]
            in_range = lo <= mean_val <= hi
        measurements.append({
            "name": name,
            "value": mean_val,
            "unit": unit,
            "within_normal_range": in_range,
            "normal_range": f"{norm_range[0]-norm_range[1]:.1f}–{norm_range[0]+norm_range[1]:.1f} {unit}" if norm_range else None,
        })

    return {
        "patient_id": patient_id or str(uuid.uuid4()),
        "study_date": study_date or "2024-03-15",
        "classification": "Class_I",
        "skeletal_pattern": "normal",
        "vertical_pattern": "average",
        "measurements": measurements,
        "summary": "Within normal limits. No significant skeletal discrepancy.",
        "analyzed_at": datetime.now(tz=timezone.utc).isoformat(),
    }


def make_abnormal_cephalometric_analysis(
    pattern: str = "class_iii",
    patient_id: Optional[str] = None,
) -> dict:
    """
    Build an abnormal cephalometric analysis representing malocclusion.

    Parameters
    ----------
    pattern : str
        ``"class_ii"`` or ``"class_iii"``.

    Returns
    -------
    dict
        Cephalometric analysis dict with Class II or Class III values.
    """
    if pattern == "class_iii":
        values = CLASS_III_CEPH_VALUES
        classification = "Class_III"
        skeletal = "class_iii_malocclusion"
        summary = (
            "Skeletal Class III pattern. Mandibular prognathism with "
            "negative overjet and negative ANB angle. Surgical assessment recommended."
        )
    elif pattern == "class_ii":
        values = CLASS_II_CEPH_VALUES
        classification = "Class_II_div1"
        skeletal = "class_ii_malocclusion"
        summary = (
            "Skeletal Class II pattern. Mandibular retrognathia with "
            "increased overjet. ANB > 7°. Orthognathic surgery indicated."
        )
    else:
        raise ValueError(f"Unknown pattern {pattern!r}; choose 'class_ii' or 'class_iii'")

    return make_cephalometric_analysis(
        values={k: float(v) for k, v in values.items()},
        patient_id=patient_id,
    ) | {
        "classification": classification,
        "skeletal_pattern": skeletal,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Symmetry report builder
# ---------------------------------------------------------------------------

def make_symmetry_report(
    asymmetry_mm: float = 1.5,
    reference_plane: str = "midsagittal",
    case_id: Optional[str] = None,
) -> dict:
    """
    Build a configurable facial symmetry assessment report.

    Parameters
    ----------
    asymmetry_mm : float
        Maximum soft-tissue point deviation from the reference plane in mm.
        Values < 2 mm are generally considered within clinical norms.
    reference_plane : str
        Reference plane used for measurement.

    Returns
    -------
    dict
        Symmetry assessment dict.
    """
    rng = np.random.default_rng(seed=int(asymmetry_mm * 100))

    landmark_deviations = {
        "menton": float(rng.uniform(0, asymmetry_mm)),
        "pogonion": float(rng.uniform(0, asymmetry_mm * 0.8)),
        "supramentale": float(rng.uniform(0, asymmetry_mm * 0.6)),
        "infradentale": float(rng.uniform(0, asymmetry_mm * 0.5)),
        "upper_central_incisor": float(rng.uniform(0, asymmetry_mm * 0.3)),
        "nasion": float(rng.uniform(0, asymmetry_mm * 0.2)),
        "orbitale_L": float(rng.uniform(0, asymmetry_mm * 1.2)),
        "orbitale_R": float(rng.uniform(0, asymmetry_mm * 1.0)),
        "zygion_L": float(rng.uniform(0, asymmetry_mm * 1.5)),
        "zygion_R": float(rng.uniform(0, asymmetry_mm * 1.3)),
    }

    overall = float(np.mean(list(landmark_deviations.values())))
    grade = (
        "acceptable" if asymmetry_mm < 2.0
        else "mild" if asymmetry_mm < 4.0
        else "significant"
    )

    return {
        "case_id": case_id or str(uuid.uuid4()),
        "reference_plane": reference_plane,
        "landmark_deviations_mm": landmark_deviations,
        "mean_deviation_mm": round(overall, 2),
        "max_deviation_mm": round(asymmetry_mm, 2),
        "symmetry_score": round(max(0.0, 1.0 - asymmetry_mm / 10.0), 3),
        "grade": grade,
        "clinically_acceptable": asymmetry_mm < 2.5,
        "analysis_notes": (
            f"Mean landmark deviation {overall:.1f} mm from {reference_plane} plane. "
            f"Symmetry grade: {grade}."
        ),
    }


# ---------------------------------------------------------------------------
# CT quality report builder
# ---------------------------------------------------------------------------

def make_quality_report(
    grade: str = "A",
    case_id: Optional[str] = None,
    study_id: Optional[str] = None,
) -> dict:
    """
    Build a CT quality assessment report.

    Parameters
    ----------
    grade : str
        Quality grade: ``"A"`` (optimal), ``"B"`` (acceptable),
        ``"C"`` (marginal), ``"D"`` (unacceptable).

    Returns
    -------
    dict
        Quality report dict with realistic scanner parameters.
    """
    grade_params = {
        "A": {
            "slice_thickness_mm": 0.625,
            "pixel_spacing_mm": 0.488,
            "snr_estimate": 35.0,
            "motion_artefact_score": 0.05,
            "metal_artefact_score": 0.02,
            "contrast_adequacy": 0.98,
            "issues": [],
            "suitable_for_planning": True,
        },
        "B": {
            "slice_thickness_mm": 1.25,
            "pixel_spacing_mm": 0.625,
            "snr_estimate": 25.0,
            "motion_artefact_score": 0.15,
            "metal_artefact_score": 0.10,
            "contrast_adequacy": 0.85,
            "issues": ["mild_motion_artefact"],
            "suitable_for_planning": True,
        },
        "C": {
            "slice_thickness_mm": 2.5,
            "pixel_spacing_mm": 0.977,
            "snr_estimate": 18.0,
            "motion_artefact_score": 0.35,
            "metal_artefact_score": 0.20,
            "contrast_adequacy": 0.65,
            "issues": ["thick_slices", "motion_artefact"],
            "suitable_for_planning": False,
        },
        "D": {
            "slice_thickness_mm": 5.0,
            "pixel_spacing_mm": 1.5,
            "snr_estimate": 10.0,
            "motion_artefact_score": 0.70,
            "metal_artefact_score": 0.50,
            "contrast_adequacy": 0.40,
            "issues": ["thick_slices", "severe_motion", "metal_artefact", "insufficient_coverage"],
            "suitable_for_planning": False,
        },
    }

    params = grade_params.get(grade, grade_params["B"])

    return {
        "case_id": case_id or str(uuid.uuid4()),
        "study_id": study_id or str(uuid.uuid4()),
        "grade": grade,
        "slice_thickness_mm": params["slice_thickness_mm"],
        "pixel_spacing_mm": params["pixel_spacing_mm"],
        "snr_estimate": params["snr_estimate"],
        "motion_artefact_score": params["motion_artefact_score"],
        "metal_artefact_score": params["metal_artefact_score"],
        "contrast_adequacy": params["contrast_adequacy"],
        "coverage_complete": grade in ("A", "B"),
        "issues": params["issues"],
        "suitable_for_planning": params["suitable_for_planning"],
        "evaluated_at": datetime.now(tz=timezone.utc).isoformat(),
        "evaluator": "auto_qc_v2.1",
    }
