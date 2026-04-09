"""
Case and patient test fixtures for Facial Align.

Provides builders for surgical cases, patient records, CT study metadata,
and a catalogue of diverse test cases covering the main fracture patterns
encountered in craniofacial surgery.

Usage:
    from tests.fixtures.case_fixtures import (
        make_case,
        make_patient,
        make_ct_study,
        SAMPLE_CASES,
        SAMPLE_FRACTURE_PATTERNS,
    )
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Fracture pattern catalogue
# ---------------------------------------------------------------------------

SAMPLE_FRACTURE_PATTERNS: Dict[str, Dict[str, Any]] = {
    "isolated_mandibular_symphysis": {
        "description": "Single fracture through the mandibular symphysis",
        "icd10": "S02.600",
        "n_fragments": 2,
        "structures_involved": ["mandible"],
        "typical_causes": ["blunt_trauma", "assault"],
        "hardware": ["miniplates_2mm"],
        "complexity": "low",
    },
    "mandibular_parasymphysis_bilateral": {
        "description": "Bilateral parasymphyseal fractures (bucket handle)",
        "icd10": "S02.601",
        "n_fragments": 3,
        "structures_involved": ["mandible"],
        "typical_causes": ["high_energy_trauma", "MVA"],
        "hardware": ["miniplates_2mm", "IMF_screws"],
        "complexity": "moderate",
    },
    "panfacial_fracture_lefort_iii": {
        "description": "Le Fort III + bilateral zygomatic + mandibular fractures",
        "icd10": "S02.80",
        "n_fragments": 8,
        "structures_involved": ["mandible", "maxilla", "zygoma_L", "zygoma_R", "naso_orbital_ethmoid"],
        "typical_causes": ["high_energy_trauma", "MVA", "fall_from_height"],
        "hardware": ["reconstruction_plate", "miniplates_1.5mm", "miniplates_2mm"],
        "complexity": "high",
    },
    "zmc_fracture_left": {
        "description": "Left zygomaticomaxillary complex (ZMC) fracture — tetrapod",
        "icd10": "S02.40",
        "n_fragments": 4,
        "structures_involved": ["zygoma_L", "maxilla"],
        "typical_causes": ["assault", "sports_injury"],
        "hardware": ["miniplates_1.5mm"],
        "complexity": "moderate",
    },
    "naso_orbital_ethmoid": {
        "description": "Naso-orbital ethmoid (NOE) fracture with medial canthal tendon involvement",
        "icd10": "S02.19",
        "n_fragments": 3,
        "structures_involved": ["nasal_bones", "ethmoid", "lacrimal_bones"],
        "typical_causes": ["direct_impact"],
        "hardware": ["microplates_1.0mm", "transnasal_wire"],
        "complexity": "high",
    },
    "orbital_blowout_floor": {
        "description": "Isolated orbital floor blowout fracture",
        "icd10": "S02.31",
        "n_fragments": 2,
        "structures_involved": ["orbital_floor"],
        "typical_causes": ["ball_sports", "fist_blow"],
        "hardware": ["titanium_mesh", "porous_polyethylene"],
        "complexity": "moderate",
    },
    "subcondylar_bilateral": {
        "description": "Bilateral subcondylar fractures",
        "icd10": "S02.62",
        "n_fragments": 3,
        "structures_involved": ["condyle_L", "condyle_R", "mandible"],
        "typical_causes": ["chin_impact", "MVA"],
        "hardware": ["miniplates_1.5mm"],
        "complexity": "moderate",
    },
}


# ---------------------------------------------------------------------------
# Patient builder
# ---------------------------------------------------------------------------

def make_patient(
    age: int = 35,
    sex: str = "M",
    patient_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a de-identified patient record suitable for test scenarios.

    Parameters
    ----------
    age : int
        Patient age in years (used to derive synthetic birth year).
    sex : str
        ``"M"`` or ``"F"``.
    patient_id : str, optional
        UUID string; auto-generated if None.

    Returns
    -------
    dict
        De-identified patient dict (no real PHI).
    """
    pid = patient_id or str(uuid.uuid4())
    # De-identified birth year only (no exact date)
    birth_year = datetime.now(tz=timezone.utc).year - age

    return {
        "patient_id": pid,
        "deidentified": True,
        "age_at_injury": age,
        "sex": sex,
        "birth_year": birth_year,
        # Clinically relevant non-PHI fields
        "asa_classification": "II" if age > 50 else "I",
        "diabetes": age > 60 and sex == "M",
        "anticoagulation": age > 65,
        "smoking_history": False,
        "prior_facial_surgery": False,
        "allergy_metal": False,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# CT study builder
# ---------------------------------------------------------------------------

def make_ct_study(
    patient_id: Optional[str] = None,
    study_id: Optional[str] = None,
    acquisition_date: Optional[str] = None,
    slice_thickness: float = 0.625,
    quality_grade: str = "A",
) -> Dict[str, Any]:
    """
    Build a CT study metadata record.

    Parameters
    ----------
    patient_id : str, optional
        UUID of the owning patient.
    study_id : str, optional
        UUID of this study; auto-generated if None.
    acquisition_date : str, optional
        ISO date (YYYY-MM-DD); defaults to today.
    slice_thickness : float
        Slice thickness in mm (typical 0.625 mm for CMF planning).
    quality_grade : str
        Quality grade from ``"A"`` (optimal) to ``"D"`` (unacceptable).

    Returns
    -------
    dict
        CT study metadata dict.
    """
    study_id = study_id or str(uuid.uuid4())
    patient_id = patient_id or str(uuid.uuid4())
    acquisition_date = acquisition_date or date.today().isoformat()

    return {
        "study_id": study_id,
        "patient_id": patient_id,
        "study_uid": f"2.25.{int(uuid.uuid4().int) % 10**39}",
        "modality": "CT",
        "acquisition_date": acquisition_date,
        "body_part": "HEAD",
        "institution": "University Medical Center",
        "manufacturer": "Siemens Healthineers",
        "manufacturer_model": "SOMATOM Definition Flash",
        "series": [
            {
                "series_instance_uid": f"2.25.{int(uuid.uuid4().int) % 10**39}",
                "modality": "CT",
                "series_number": 1,
                "slice_count": int(120 / slice_thickness),
                "slice_thickness_mm": slice_thickness,
                "pixel_spacing_mm": [0.488, 0.488],
                "kvp": 120.0,
                "storage_path": f"/data/studies/{study_id}/series_001/",
            }
        ],
        "volume_path": f"/data/studies/{study_id}/volume.nii.gz",
        "spacing_mm": [0.488, 0.488, slice_thickness],
        "volume_shape": [int(120 / slice_thickness), 512, 512],
        "quality_grade": quality_grade,
        "quality_score": {"A": 0.95, "B": 0.75, "C": 0.50, "D": 0.25}[quality_grade],
        "quality_flags": [] if quality_grade == "A" else ["motion_artefact"],
        "is_deidentified": True,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Case builder
# ---------------------------------------------------------------------------

def make_case(
    status: str = "planning",
    fracture_pattern: str = "mandibular_parasymphysis_bilateral",
    patient_age: int = 35,
    patient_sex: str = "M",
    case_id: Optional[str] = None,
    with_approved_plan: bool = False,
) -> Dict[str, Any]:
    """
    Build a complete surgical case record.

    Parameters
    ----------
    status : str
        Workflow status: ``"pending_ct"``, ``"segmentation"``, ``"planning"``,
        ``"review"``, ``"approved"``, ``"archived"``.
    fracture_pattern : str
        Key from ``SAMPLE_FRACTURE_PATTERNS`` dict.
    patient_age, patient_sex : int, str
        Used to generate the embedded patient record.
    case_id : str, optional
        UUID string; auto-generated if None.
    with_approved_plan : bool
        If True, mark the case as having a surgeon-approved plan.

    Returns
    -------
    dict
        Fully populated case dict.
    """
    case_id = case_id or str(uuid.uuid4())
    patient = make_patient(age=patient_age, sex=patient_sex)
    study = make_ct_study(patient_id=patient["patient_id"])
    pattern = SAMPLE_FRACTURE_PATTERNS.get(
        fracture_pattern, SAMPLE_FRACTURE_PATTERNS["mandibular_parasymphysis_bilateral"]
    )

    plan_id = str(uuid.uuid4()) if status in ("review", "approved") else None

    return {
        "case_id": case_id,
        "patient_id": patient["patient_id"],
        "patient": patient,
        "study_id": study["study_id"],
        "study": study,
        "status": status,
        "fracture_pattern": fracture_pattern,
        "fracture_description": pattern["description"],
        "icd10_code": pattern["icd10"],
        "structures_involved": pattern["structures_involved"],
        "n_fragments_expected": pattern["n_fragments"],
        "complexity": pattern["complexity"],
        "expected_hardware": pattern["hardware"],
        "segmentation_id": str(uuid.uuid4()) if status not in ("pending_ct",) else None,
        "plan_id": plan_id,
        "surgeon_approved": with_approved_plan,
        "approved_at": datetime.now(tz=timezone.utc).isoformat() if with_approved_plan else None,
        "priority": "routine",
        "scheduled_or_date": None,
        "notes": f"Fixture case for {fracture_pattern} testing.",
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "updated_at": datetime.now(tz=timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Pre-built sample case catalogue
# ---------------------------------------------------------------------------

SAMPLE_CASES: List[Dict[str, Any]] = [
    make_case(
        status="planning",
        fracture_pattern="mandibular_parasymphysis_bilateral",
        patient_age=32,
        patient_sex="M",
    ),
    make_case(
        status="review",
        fracture_pattern="zmc_fracture_left",
        patient_age=45,
        patient_sex="F",
    ),
    make_case(
        status="approved",
        fracture_pattern="panfacial_fracture_lefort_iii",
        patient_age=28,
        patient_sex="M",
        with_approved_plan=True,
    ),
    make_case(
        status="planning",
        fracture_pattern="subcondylar_bilateral",
        patient_age=52,
        patient_sex="F",
    ),
    make_case(
        status="segmentation",
        fracture_pattern="orbital_blowout_floor",
        patient_age=23,
        patient_sex="M",
    ),
]
