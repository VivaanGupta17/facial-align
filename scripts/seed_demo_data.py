#!/usr/bin/env python3
"""
Seed the database with realistic demo data for development and demos.

Creates:
- 8 sample patients (de-identified)
- 8 cases spanning different fracture patterns and statuses
- CT study metadata for each case
- Sample segmentation results with mock confidence scores
- Sample reduction plans at various stages

Usage:
    python scripts/seed_demo_data.py                # Write JSON to examples/
    python scripts/seed_demo_data.py --db            # Write directly to database
    python scripts/seed_demo_data.py --clear         # Clear existing demo data
    python scripts/seed_demo_data.py -o out.json     # Write to custom JSON file
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "apps" / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─── Demo data definitions ────────────────────────────────────────────────────

DEMO_PATIENTS = [
    {
        "patient_id": "DEMO-PT-001",
        "pseudonym": "Demo Patient Alpha",
        "age_at_scan": 28,
        "sex": "M",
        "notes": "Motor vehicle accident. Bilateral mandible fractures.",
    },
    {
        "patient_id": "DEMO-PT-002",
        "pseudonym": "Demo Patient Beta",
        "age_at_scan": 45,
        "sex": "F",
        "notes": "Fall from height. Le Fort II fracture with NOE involvement.",
    },
    {
        "patient_id": "DEMO-PT-003",
        "pseudonym": "Demo Patient Gamma",
        "age_at_scan": 19,
        "sex": "M",
        "notes": "Sports injury. Isolated right subcondylar fracture.",
    },
    {
        "patient_id": "DEMO-PT-004",
        "pseudonym": "Demo Patient Delta",
        "age_at_scan": 62,
        "sex": "F",
        "notes": "Assault. Comminuted left zygomaticomaxillary complex fracture.",
    },
    {
        "patient_id": "DEMO-PT-005",
        "pseudonym": "Demo Patient Epsilon",
        "age_at_scan": 34,
        "sex": "M",
        "notes": "Industrial accident. Panfacial fractures — frontal, bilateral ZMC, mandible.",
    },
    {
        "patient_id": "DEMO-PT-006",
        "pseudonym": "Demo Patient Zeta",
        "age_at_scan": 41,
        "sex": "F",
        "notes": "Bicycle accident. Unilateral mandible angle fracture.",
    },
    {
        "patient_id": "DEMO-PT-007",
        "pseudonym": "Demo Patient Eta",
        "age_at_scan": 55,
        "sex": "M",
        "notes": "Fall. Le Fort I fracture with palatal split.",
    },
    {
        "patient_id": "DEMO-PT-008",
        "pseudonym": "Demo Patient Theta",
        "age_at_scan": 23,
        "sex": "F",
        "notes": "Sports collision. Isolated nasal bone fracture with septal deviation.",
    },
]

# DB statuses match surgical_cases.status enum:
#   CREATED, DICOM_PROCESSING, SEGMENTED, PLANNING, PLANNED, REVIEWED, APPROVED, ARCHIVED, FAILED
DEMO_CASES = [
    {
        "case_type": "TRAUMA",
        "fracture_classification": "AO CMF: 91-B3.1 (bilateral mandible body)",
        "status": "APPROVED",
        "surgeon": "demo-surgeon-001",
        "fragments": [
            {"id": "mandible_body_L", "structure": "mandible", "volume_mm3": 4500},
            {"id": "mandible_body_R", "structure": "mandible", "volume_mm3": 4200},
            {"id": "mandible_symphysis", "structure": "mandible", "volume_mm3": 3800},
        ],
        "ct_quality_grade": "A",
    },
    {
        "case_type": "TRAUMA",
        "fracture_classification": "AO CMF: 92-C2 (Le Fort II with NOE)",
        "status": "PLANNING",
        "surgeon": "demo-surgeon-001",
        "fragments": [
            {"id": "maxilla_central", "structure": "maxilla", "volume_mm3": 6200},
            {"id": "nasal_complex", "structure": "nasal", "volume_mm3": 1800},
            {"id": "orbital_floor_L", "structure": "orbit", "volume_mm3": 900},
            {"id": "orbital_floor_R", "structure": "orbit", "volume_mm3": 850},
        ],
        "ct_quality_grade": "A",
    },
    {
        "case_type": "TRAUMA",
        "fracture_classification": "AO CMF: 91-A1.3 (right subcondylar)",
        "status": "SEGMENTED",
        "surgeon": "demo-surgeon-002",
        "fragments": [
            {"id": "condyle_R", "structure": "mandible", "volume_mm3": 2100},
            {"id": "ramus_R", "structure": "mandible", "volume_mm3": 5500},
        ],
        "ct_quality_grade": "B",
    },
    {
        "case_type": "TRAUMA",
        "fracture_classification": "AO CMF: 92-B1.1 (left ZMC, comminuted)",
        "status": "REVIEWED",
        "surgeon": "demo-surgeon-002",
        "fragments": [
            {"id": "zygoma_L", "structure": "zygoma", "volume_mm3": 3200},
            {"id": "orbital_rim_L", "structure": "orbit", "volume_mm3": 1100},
            {"id": "zygomatic_arch_L", "structure": "zygoma", "volume_mm3": 800},
            {"id": "maxilla_lateral_L", "structure": "maxilla", "volume_mm3": 1500},
        ],
        "ct_quality_grade": "A",
    },
    {
        "case_type": "TRAUMA",
        "fracture_classification": "AO CMF: 93-C3 (panfacial, high complexity)",
        "status": "CREATED",
        "surgeon": "demo-surgeon-001",
        "fragments": [
            {"id": "frontal_bar", "structure": "frontal", "volume_mm3": 5800},
            {"id": "zygoma_L", "structure": "zygoma", "volume_mm3": 3100},
            {"id": "zygoma_R", "structure": "zygoma", "volume_mm3": 3300},
            {"id": "maxilla_central", "structure": "maxilla", "volume_mm3": 5900},
            {"id": "mandible_body_L", "structure": "mandible", "volume_mm3": 4000},
            {"id": "mandible_angle_R", "structure": "mandible", "volume_mm3": 3500},
            {"id": "noe_complex", "structure": "naso-orbito-ethmoidal", "volume_mm3": 1200},
        ],
        "ct_quality_grade": "B",
    },
    {
        "case_type": "TRAUMA",
        "fracture_classification": "AO CMF: 91-A2.1 (left mandible angle)",
        "status": "DICOM_PROCESSING",
        "surgeon": "demo-surgeon-001",
        "fragments": [
            {"id": "mandible_angle_L", "structure": "mandible", "volume_mm3": 3800},
        ],
        "ct_quality_grade": "A",
    },
    {
        "case_type": "ORTHOGNATHIC",
        "fracture_classification": "Le Fort I osteotomy (planned orthognathic)",
        "status": "PLANNED",
        "surgeon": "demo-surgeon-002",
        "fragments": [
            {"id": "maxilla_central", "structure": "maxilla", "volume_mm3": 6100},
            {"id": "maxilla_alveolar", "structure": "maxilla", "volume_mm3": 3200},
        ],
        "ct_quality_grade": "A",
    },
    {
        "case_type": "TRAUMA",
        "fracture_classification": "Nasal bone fracture with septal deviation",
        "status": "ARCHIVED",
        "surgeon": "demo-surgeon-002",
        "fragments": [
            {"id": "nasal_bone", "structure": "nasal", "volume_mm3": 1200},
        ],
        "ct_quality_grade": "B",
    },
]

DEMO_USERS = [
    {
        "email": "surgeon@facialign.local",
        "password": "surgeon",
        "full_name": "Demo Surgeon",
        "role": "surgeon",
        "institution": "Facial Align Demo Hospital",
        "specialty": "Craniomaxillofacial Surgery",
        "label": "primary_surgeon",
    },
    {
        "email": "reviewer@facialign.local",
        "password": "reviewer",
        "full_name": "Demo Reviewer",
        "role": "surgeon",
        "institution": "Facial Align Demo Hospital",
        "specialty": "Surgical Review",
        "label": "reviewer",
    },
]

DEFAULT_CHECKLIST = [
    {
        "id": "seg-accuracy",
        "category": "Segmentation",
        "label": "Bone segmentation boundaries are accurate",
        "passed": None,
        "severity": "required",
    },
    {
        "id": "seg-complete",
        "category": "Segmentation",
        "label": "All relevant structures are segmented",
        "passed": None,
        "severity": "required",
    },
    {
        "id": "frag-identified",
        "category": "Segmentation",
        "label": "Fracture fragments correctly identified",
        "passed": None,
        "severity": "required",
    },
    {
        "id": "reduction-sym",
        "category": "Reduction",
        "label": "Facial symmetry is restored",
        "passed": None,
        "severity": "required",
    },
    {
        "id": "reduction-occ",
        "category": "Reduction",
        "label": "Occlusion is within acceptable parameters",
        "passed": None,
        "severity": "required",
    },
    {
        "id": "reduction-condyle",
        "category": "Reduction",
        "label": "Condylar seating is adequate",
        "passed": None,
        "severity": "required",
    },
    {
        "id": "splint-fit",
        "category": "Splint",
        "label": "Intermediate splint design is appropriate",
        "passed": None,
        "severity": "recommended",
    },
    {
        "id": "hardware-plan",
        "category": "Hardware",
        "label": "Plate and screw positions are feasible",
        "passed": None,
        "severity": "recommended",
    },
    {
        "id": "nerve-clearance",
        "category": "Safety",
        "label": "Hardware avoids inferior alveolar nerve",
        "passed": None,
        "severity": "required",
    },
    {
        "id": "airway-clear",
        "category": "Safety",
        "label": "Airway dimensions are maintained",
        "passed": None,
        "severity": "recommended",
    },
    {
        "id": "aesthetics",
        "category": "Aesthetics",
        "label": "Soft tissue projection is acceptable",
        "passed": None,
        "severity": "optional",
    },
]


DEMO_CT_STUDIES = [
    {
        "modality": "CT",
        "manufacturer": "Siemens",
        "model": "SOMATOM Force",
        "slice_thickness_mm": 0.6,
        "pixel_spacing_mm": 0.43,
        "rows": 512,
        "columns": 512,
        "num_slices": 320,
        "kvp": 120,
        "exposure_ma": 200,
        "convolution_kernel": "H60s (bone sharp)",
    },
    {
        "modality": "CT",
        "manufacturer": "GE Healthcare",
        "model": "Revolution CT",
        "slice_thickness_mm": 0.625,
        "pixel_spacing_mm": 0.39,
        "rows": 512,
        "columns": 512,
        "num_slices": 280,
        "kvp": 120,
        "exposure_ma": 180,
        "convolution_kernel": "BONEPLUS",
    },
    {
        "modality": "CT",
        "manufacturer": "Siemens",
        "model": "SOMATOM Definition Edge",
        "slice_thickness_mm": 1.0,
        "pixel_spacing_mm": 0.49,
        "rows": 512,
        "columns": 512,
        "num_slices": 200,
        "kvp": 120,
        "exposure_ma": 160,
        "convolution_kernel": "H70h (bone very sharp)",
    },
    {
        "modality": "CT",
        "manufacturer": "Philips",
        "model": "IQon Spectral CT",
        "slice_thickness_mm": 0.67,
        "pixel_spacing_mm": 0.45,
        "rows": 512,
        "columns": 512,
        "num_slices": 290,
        "kvp": 120,
        "exposure_ma": 190,
        "convolution_kernel": "YC (bone)",
    },
    {
        "modality": "CT",
        "manufacturer": "Canon Medical",
        "model": "Aquilion ONE PRISM",
        "slice_thickness_mm": 0.5,
        "pixel_spacing_mm": 0.41,
        "rows": 512,
        "columns": 512,
        "num_slices": 340,
        "kvp": 120,
        "exposure_ma": 210,
        "convolution_kernel": "FC30 (bone sharp)",
    },
    {
        "modality": "CT",
        "manufacturer": "Siemens",
        "model": "SOMATOM go.Top",
        "slice_thickness_mm": 0.75,
        "pixel_spacing_mm": 0.44,
        "rows": 512,
        "columns": 512,
        "num_slices": 260,
        "kvp": 120,
        "exposure_ma": 170,
        "convolution_kernel": "Br64s (bone)",
    },
    {
        "modality": "CT",
        "manufacturer": "GE Healthcare",
        "model": "Discovery CT750 HD",
        "slice_thickness_mm": 0.625,
        "pixel_spacing_mm": 0.42,
        "rows": 512,
        "columns": 512,
        "num_slices": 300,
        "kvp": 120,
        "exposure_ma": 195,
        "convolution_kernel": "BONE",
    },
    {
        "modality": "CT",
        "manufacturer": "Philips",
        "model": "Ingenuity CT",
        "slice_thickness_mm": 0.8,
        "pixel_spacing_mm": 0.46,
        "rows": 512,
        "columns": 512,
        "num_slices": 250,
        "kvp": 120,
        "exposure_ma": 175,
        "convolution_kernel": "YB (bone detail)",
    },
]

SEGMENTATION_STRUCTURES = [
    "mandible", "maxilla", "skull_base", "frontal_bone",
    "zygoma_L", "zygoma_R", "nasal_bone", "sphenoid",
    "temporal_L", "temporal_R", "parietal_L", "parietal_R",
    "occipital", "ethmoid", "vomer", "palatine_L", "palatine_R",
    "orbit_L", "orbit_R", "tooth_upper", "tooth_lower",
]


def generate_case_id() -> str:
    """Generate a unique case ID."""
    short_uuid = uuid.uuid4().hex[:8].upper()
    return f"FA-{short_uuid}"


def generate_study_uid() -> str:
    """Generate a DICOM-style Study Instance UID."""
    return f"1.2.826.0.1.3680043.8.1055.1.{uuid.uuid4().int % 10**12}"


def generate_segmentation_result(structures: List[str]) -> Dict[str, Any]:
    """Generate a realistic segmentation result."""
    import random
    result = {
        "model": "TotalSegmentator",
        "model_version": "2.0.1",
        "structures": {},
        "overall_confidence": round(random.uniform(0.88, 0.97), 3),
        "processing_time_seconds": round(random.uniform(25, 45), 1),
    }
    for struct in structures:
        result["structures"][struct] = {
            "confidence": round(random.uniform(0.82, 0.99), 3),
            "volume_mm3": round(random.uniform(500, 15000), 1),
            "surface_area_mm2": round(random.uniform(200, 8000), 1),
            "n_voxels": random.randint(5000, 200000),
        }
    return result


def generate_reduction_plan(fragments: List[Dict]) -> Dict[str, Any]:
    """Generate a realistic reduction plan."""
    import random
    import math
    plan = {
        "algorithm": "ICP_baseline",
        "algorithm_version": "1.0.0",
        "fragments": {},
        "overall_quality_grade": random.choice(["excellent", "good", "acceptable"]),
        "evaluation_time_ms": random.randint(100, 500),
    }
    for frag in fragments:
        # Generate small rotation (realistic reduction)
        angle_deg = random.uniform(1, 8)
        angle_rad = math.radians(angle_deg)
        axis = random.choice(["x", "y", "z"])
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

        if axis == "x":
            rotation = [[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]]
        elif axis == "y":
            rotation = [[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]]
        else:
            rotation = [[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]]

        translation = [
            round(random.uniform(-5, 5), 2),
            round(random.uniform(-5, 5), 2),
            round(random.uniform(-3, 3), 2),
        ]

        plan["fragments"][frag["id"]] = {
            "rotation_matrix": rotation,
            "translation_mm": translation,
            "confidence": round(random.uniform(0.7, 0.95), 3),
            "rms_error_mm": round(random.uniform(0.3, 2.0), 3),
            "mean_surface_distance_mm": round(random.uniform(0.2, 1.5), 3),
            "hausdorff_distance_mm": round(random.uniform(1.0, 4.0), 3),
        }

    return plan


def build_demo_dataset(profile: str = "demo") -> Dict[str, Any]:
    """Build the complete demo dataset."""
    import random
    dataset = {
        "profile": profile,
        "patients": [],
        "cases": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    base_date = datetime.now(timezone.utc) - timedelta(days=30)

    for i, (patient, case_def, ct_study) in enumerate(
        zip(DEMO_PATIENTS, DEMO_CASES, DEMO_CT_STUDIES)
    ):
        case_id = f"{profile.upper()}-{generate_case_id()}"
        study_uid = generate_study_uid()
        case_date = base_date + timedelta(days=i * 5, hours=random.randint(6, 18))

        patient_record = {
            **patient,
            "case_id": case_id,
        }

        seg_result = generate_segmentation_result(SEGMENTATION_STRUCTURES[:15])

        case_record = {
            "case_id": case_id,
            "patient_id": patient["patient_id"],
            "case_type": case_def["case_type"],
            "fracture_classification": case_def["fracture_classification"],
            "status": case_def["status"],
            "surgeon": case_def["surgeon"],
            "created_at": case_date.isoformat(),
            "updated_at": (case_date + timedelta(hours=random.randint(1, 48))).isoformat(),
            "ct_study": {
                "study_instance_uid": study_uid,
                **ct_study,
                "quality_grade": case_def["ct_quality_grade"],
            },
            "segmentation": seg_result if case_def["status"] not in ("CREATED", "DICOM_PROCESSING") else None,
            "fragments": case_def["fragments"],
            "reduction_plan": (
                generate_reduction_plan(case_def["fragments"])
                if case_def["status"] in ("PLANNING", "PLANNED", "REVIEWED", "APPROVED", "ARCHIVED")
                else None
            ),
        }

        dataset["patients"].append(patient_record)
        dataset["cases"].append(case_record)

    return dataset


# ─── Deterministic UUIDs for demo data (idempotent) ─────────────────────────

DEMO_NAMESPACE = uuid.UUID("fa000000-0000-4000-8000-000000000000")


def _demo_uuid(label: str) -> str:
    """Generate a deterministic UUID from a label for idempotent seeding."""
    return str(uuid.uuid5(DEMO_NAMESPACE, label))


def _seed_namespace(profile: str) -> uuid.UUID:
    if profile == "release_test":
        return uuid.UUID("fa000000-0000-4000-8000-000000000001")
    return DEMO_NAMESPACE


def _seed_uuid(profile: str, label: str) -> str:
    """Generate a deterministic UUID scoped to a seed profile."""
    return str(uuid.uuid5(_seed_namespace(profile), label))


def _case_number_for_profile(profile: str, index: int) -> str:
    prefix = "REL" if profile == "release_test" else "DEMO"
    return f"FA-{prefix}-{1001 + index:04d}"


def _build_segmentation_provenance(case_status: str, profile: str) -> dict:
    warnings: list[str] = []
    if profile == "release_test" and case_status in {"PLANNING", "REVIEWED"}:
        warnings.append(
            "Dental segmentation beta path unavailable; deterministic baseline in use."
        )

    return {
        "algorithmUsed": "totalsegmentator",
        "validationTier": "deterministic_baseline",
        "betaStatus": "not_beta",
        "warnings": warnings,
        "fallbackReason": warnings[0] if warnings else None,
        "modelVersion": "2.0.1",
    }


def _build_plan_provenance(case_status: str, profile: str) -> dict:
    warnings: list[str] = []
    beta_status = "not_beta"
    fallback_reason = None
    if profile == "release_test" and case_status == "PLANNING":
        fallback_reason = (
            "Learned reduction beta unavailable; baseline ICP planner is active."
        )
        warnings.append(fallback_reason)
        beta_status = "fallback"

    return {
        "algorithmUsed": "baseline_icp",
        "validationTier": "deterministic_baseline",
        "betaStatus": beta_status,
        "warnings": warnings,
        "fallbackReason": fallback_reason,
        "modelVersion": "1.0.0",
    }


def _build_review_state(case_status: str) -> Tuple[str, str, str | None]:
    if case_status in {"APPROVED", "ARCHIVED"}:
        return ("approved", "Plan approved for operative use.", "signature-approved")
    if case_status == "REVIEWED":
        return (
            "revision_requested",
            "Need additional manual adjustment before final approval.",
            None,
        )
    return ("pending", "", None)


# ─── Database seeding ────────────────────────────────────────────────────────


async def _ensure_seed_users(db, profile: str) -> Dict[str, Dict[str, str]]:
    from sqlalchemy import select

    from app.core.security import hash_password
    from app.models.user import User

    users_by_label: Dict[str, Dict[str, str]] = {}

    for user_def in DEMO_USERS:
        existing = (
            await db.execute(select(User).where(User.email == user_def["email"]))
        ).scalar_one_or_none()

        if existing is None:
            existing = User(
                id=_seed_uuid(profile, f"user-{user_def['label']}"),
                email=user_def["email"],
                hashed_password=hash_password(user_def["password"]),
                full_name=user_def["full_name"],
                role=user_def["role"],
                institution=user_def["institution"],
                specialty=user_def["specialty"],
                is_active=True,
                is_verified=True,
            )
            db.add(existing)
            await db.flush()

        users_by_label[user_def["label"]] = {
            "id": str(existing.id),
            "email": existing.email,
            "full_name": existing.full_name,
        }

    return users_by_label


async def _clear_seed_data(profile: str) -> None:
    from sqlalchemy import delete

    from app.db.database import create_db_engine, dispose_db_engine, get_db_context
    from app.models.case import SurgicalCase
    from app.models.case_study import CaseStudy
    from app.models.patient import Patient
    from app.models.plan import ReductionPlan
    from app.models.review import PlanReview
    from app.models.segmentation import SegmentationResult
    from app.models.study import ImagingStudy

    await create_db_engine()

    try:
        async with get_db_context() as db:
            case_ids = [_seed_uuid(profile, f"case-{i}") for i in range(len(DEMO_CASES))]
            study_ids = [_seed_uuid(profile, f"study-{pt['patient_id']}") for pt in DEMO_PATIENTS]
            patient_ids = [_seed_uuid(profile, f"patient-{pt['patient_id']}") for pt in DEMO_PATIENTS]
            plan_ids = [_seed_uuid(profile, f"plan-{i}") for i in range(len(DEMO_CASES))]
            review_ids = [_seed_uuid(profile, f"review-{i}") for i in range(len(DEMO_CASES))]
            seg_ids = [_seed_uuid(profile, f"seg-{i}") for i in range(len(DEMO_CASES))]

            await db.execute(delete(PlanReview).where(PlanReview.id.in_(review_ids)))
            await db.execute(delete(ReductionPlan).where(ReductionPlan.id.in_(plan_ids)))
            await db.execute(delete(SegmentationResult).where(SegmentationResult.id.in_(seg_ids)))
            await db.execute(delete(CaseStudy).where(CaseStudy.case_id.in_(case_ids)))
            await db.execute(delete(SurgicalCase).where(SurgicalCase.id.in_(case_ids)))
            await db.execute(delete(ImagingStudy).where(ImagingStudy.id.in_(study_ids)))
            await db.execute(delete(Patient).where(Patient.id.in_(patient_ids)))
    finally:
        await dispose_db_engine()


async def seed_to_database(profile: str = "demo") -> None:
    """Write deterministic demo or release-test data directly to the database."""
    import random

    from sqlalchemy import select

    from app.db.database import create_db_engine, dispose_db_engine, get_db_context
    from app.models.case import SurgicalCase
    from app.models.case_study import CaseStudy
    from app.models.patient import Patient
    from app.models.plan import ReductionPlan
    from app.models.review import PlanReview
    from app.models.segmentation import SegmentationResult
    from app.models.study import ImagingStudy

    random.seed(42)

    await create_db_engine()

    try:
        async with get_db_context() as db:
            first_patient_id = _seed_uuid(profile, "patient-DEMO-PT-001")
            existing = (
                await db.execute(select(Patient.id).where(Patient.id == first_patient_id))
            ).scalar_one_or_none()
            if existing:
                print(f"{profile} data already exists. Skipping (idempotent).")
                return

        base_date = datetime.now(timezone.utc) - timedelta(days=60)
        planning_statuses = ("PLANNING", "PLANNED", "REVIEWED", "APPROVED", "ARCHIVED")

        async with get_db_context() as db:
            seeded_users = await _ensure_seed_users(db, profile)
            primary_surgeon = seeded_users["primary_surgeon"]
            reviewer = seeded_users["reviewer"]

            patient_ids: List[str] = []
            for pt in DEMO_PATIENTS:
                patient_id = _seed_uuid(profile, f"patient-{pt['patient_id']}")
                patient_ids.append(patient_id)
                mrn_hash = hashlib.sha256(
                    f"{profile}-salt-{pt['patient_id']}".encode()
                ).hexdigest()
                db.add(
                    Patient(
                        id=patient_id,
                        mrn_hash=mrn_hash,
                        institution_code="REL-TEST-INST" if profile == "release_test" else "DEMO-INST",
                        age_at_registration=pt["age_at_scan"],
                        sex=pt["sex"],
                        created_by=primary_surgeon["id"],
                        is_active=True,
                    )
                )
            print(f"  Created {len(patient_ids)} patients")

            study_ids: List[str] = []
            for i, (pt, ct) in enumerate(zip(DEMO_PATIENTS, DEMO_CT_STUDIES)):
                study_id = _seed_uuid(profile, f"study-{pt['patient_id']}")
                study_ids.append(study_id)
                acquisition_date = (base_date + timedelta(days=i * 7)).date()
                db.add(
                    ImagingStudy(
                        id=study_id,
                        study_uid=f"1.2.826.0.1.3680043.8.1055.{1 if profile == 'release_test' else 0}.{1000 + i}",
                        patient_id=patient_ids[i],
                        modality=ct["modality"],
                        acquisition_date=acquisition_date,
                        series_count=max(1, random.randint(1, 3)),
                        slice_count=ct["num_slices"],
                        storage_path=f"/data/{profile}/studies/{pt['patient_id']}/dicom",
                        volume_path=f"/data/{profile}/studies/{pt['patient_id']}/volume.nii.gz",
                        slice_thickness_mm=ct["slice_thickness_mm"],
                        pixel_spacing_mm=ct["pixel_spacing_mm"],
                        kv_peak=ct["kvp"],
                        body_part_examined="HEAD",
                        metadata_json={
                            "StudyDescription": pt["notes"],
                            "InstitutionName": "Facial Align Demo Hospital",
                            "Manufacturer": ct["manufacturer"],
                            "ManufacturerModelName": ct["model"],
                            "SeriesDescription": "Primary Series",
                            "SeriesNumber": 1,
                            "SeriesInstanceUID": f"1.2.826.0.1.3680043.8.1055.series.{1000 + i}",
                            "kernel": ct["convolution_kernel"],
                        },
                        quality_score=round(random.uniform(0.80, 0.99), 3),
                        quality_flags=[],
                        is_deidentified=True,
                        ingestion_status="complete",
                        uploaded_by=primary_surgeon["id"],
                    )
                )
            print(f"  Created {len(study_ids)} imaging studies")

            case_ids: List[str] = []
            for i, case_def in enumerate(DEMO_CASES):
                case_id = _seed_uuid(profile, f"case-{i}")
                case_ids.append(case_id)
                case_date = base_date + timedelta(days=i * 7, hours=random.randint(6, 18))
                db.add(
                    SurgicalCase(
                        id=case_id,
                        case_number=_case_number_for_profile(profile, i),
                        patient_id=patient_ids[i],
                        study_id=study_ids[i],
                        case_type=case_def["case_type"],
                        status=case_def["status"],
                        fracture_classification=case_def["fracture_classification"],
                        planned_procedure=f"ORIF {case_def['fracture_classification']}",
                        surgeon_id=primary_surgeon["id"],
                        reviewer_id=reviewer["id"] if case_def["status"] in {"REVIEWED", "APPROVED", "ARCHIVED"} else None,
                        created_by=primary_surgeon["id"],
                        created_at=case_date,
                        updated_at=case_date + timedelta(hours=random.randint(1, 48)),
                        approved_at=(
                            datetime.now(timezone.utc) - timedelta(days=5)
                            if case_def["status"] in {"APPROVED", "ARCHIVED"}
                            else None
                        ),
                    )
                )
                db.add(
                    CaseStudy(
                        id=_seed_uuid(profile, f"case-study-{i}"),
                        case_id=case_id,
                        study_id=study_ids[i],
                        study_role="pre_op",
                        study_label=f"Primary CT Series {i + 1}",
                        is_primary=True,
                        display_order=0,
                    )
                )
            print(f"  Created {len(case_ids)} surgical cases")

            seg_count = 0
            for i, case_def in enumerate(DEMO_CASES):
                if case_def["status"] in ("CREATED", "DICOM_PROCESSING"):
                    continue

                seg_result = generate_segmentation_result(SEGMENTATION_STRUCTURES[:15])
                confidence_scores = {
                    structure: seg_result["structures"][structure]["confidence"]
                    for structure in seg_result["structures"]
                }
                volume_stats = {
                    structure: {
                        "volume_cc": round(seg_result["structures"][structure]["volume_mm3"] / 1000, 2),
                        "surface_area_mm2": seg_result["structures"][structure]["surface_area_mm2"],
                    }
                    for structure in seg_result["structures"]
                }
                structure_reviews = {
                    structure: {
                        "status": "accepted"
                        if case_def["status"] in {"REVIEWED", "APPROVED", "ARCHIVED"}
                        else "pending"
                    }
                    for structure in SEGMENTATION_STRUCTURES[:15]
                }
                db.add(
                    SegmentationResult(
                        id=_seed_uuid(profile, f"seg-{i}"),
                        case_id=case_ids[i],
                        model_name="totalsegmentator",
                        model_version="2.0.1",
                        structure_labels={s: idx + 1 for idx, s in enumerate(SEGMENTATION_STRUCTURES[:15])},
                        structures=structure_reviews,
                        mask_storage_path=f"/data/{profile}/cases/{case_ids[i]}/segmentation/mask.nii.gz",
                        mesh_storage_paths={
                            structure: {
                                "glb": f"/data/{profile}/cases/{case_ids[i]}/meshes/{structure}.glb",
                                "stl": f"/data/{profile}/cases/{case_ids[i]}/meshes/{structure}.stl",
                            }
                            for structure in SEGMENTATION_STRUCTURES[:6]
                        },
                        confidence_scores=confidence_scores,
                        overall_confidence=seg_result["overall_confidence"],
                        provenance=_build_segmentation_provenance(case_def["status"], profile),
                        inference_time_ms=random.randint(15000, 40000),
                        total_pipeline_time_ms=random.randint(25000, 60000),
                        gpu_device="cuda:0",
                        volume_stats=volume_stats,
                        fragment_count=len(case_def["fragments"]),
                        fracture_fragments=[
                            {
                                "fragmentId": fragment["id"],
                                "parentStructure": fragment["structure"],
                                "volumeCc": round(fragment["volume_mm3"] / 1000, 2),
                                "centroidMm": [10 + idx * 2, 5 + idx, 0],
                            }
                            for idx, fragment in enumerate(case_def["fragments"])
                        ],
                        fragment_mesh_paths={
                            fragment["id"]: {
                                "glb": f"/data/{profile}/cases/{case_ids[i]}/fragments/{fragment['id']}.glb"
                            }
                            for fragment in case_def["fragments"]
                        },
                        status="complete",
                        completed_at=datetime.now(timezone.utc) - timedelta(days=random.randint(1, 30)),
                    )
                )
                seg_count += 1
            print(f"  Created {seg_count} segmentation results")

            plan_count = 0
            for i, case_def in enumerate(DEMO_CASES):
                if case_def["status"] not in planning_statuses:
                    continue

                plan_data = generate_reduction_plan(case_def["fragments"])
                transformations = {}
                for fragment in case_def["fragments"]:
                    fragment_data = plan_data["fragments"].get(fragment["id"], {})
                    transformations[fragment["id"]] = {
                        "transform": {
                            "rotationMatrix": fragment_data.get("rotation_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                            "translationMm": fragment_data.get("translation_mm", [0, 0, 0]),
                        },
                        "fragment_label": fragment["structure"],
                        "confidence": fragment_data.get("confidence", 0.85),
                    }

                is_approved = case_def["status"] in ("APPROVED", "ARCHIVED")
                plan_status_map = {
                    "PLANNING": "draft",
                    "PLANNED": "validated",
                    "REVIEWED": "surgeon_reviewed",
                    "APPROVED": "approved",
                    "ARCHIVED": "approved",
                }
                db.add(
                    ReductionPlan(
                        id=_seed_uuid(profile, f"plan-{i}"),
                        case_id=case_ids[i],
                        segmentation_id=_seed_uuid(profile, f"seg-{i}"),
                        plan_version=1,
                        model_name="baseline_icp",
                        model_version="1.0.0",
                        fragments={
                            fragment["id"]: {
                                "label": fragment["structure"],
                                "fragmentLabel": fragment["structure"],
                                "parentStructure": fragment["structure"],
                                "volumeCc": round(fragment["volume_mm3"] / 1000, 2),
                                "centroidMm": [10 + idx * 2, 5 + idx, 0],
                            }
                            for idx, fragment in enumerate(case_def["fragments"])
                        },
                        transformations=transformations,
                        dental_constraints={
                            "targetOverjetMm": 2.0,
                            "targetOverbiteMm": 3.0,
                            "midlineToleranceMm": 1.0,
                            "cantToleranceDegrees": 2.0,
                            "bilateralCondylarSeating": True,
                        },
                        occlusal_metrics={
                            "overjetMm": 2.4,
                            "overbiteMm": 2.8,
                            "midlineDeviationMm": 0.6 if case_def["status"] != "PLANNING" else 1.4,
                            "cantDegrees": 1.1 if case_def["status"] != "PLANNING" else 2.4,
                            "molarRelationship": "Class_I",
                        },
                        provenance=_build_plan_provenance(case_def["status"], profile),
                        confidence_score=round(random.uniform(0.75, 0.95), 3),
                        validation_passed=case_def["status"] != "PLANNING",
                        validation_warnings=(
                            ["Manual review recommended before approval."]
                            if case_def["status"] in {"PLANNING", "REVIEWED"}
                            else []
                        ),
                        surgeon_approved=is_approved,
                        is_ml_generated=False,
                        status=plan_status_map.get(case_def["status"], "draft"),
                        generation_time_ms=random.randint(2000, 8000),
                        approved_at=(
                            datetime.now(timezone.utc) - timedelta(days=random.randint(1, 10))
                            if is_approved
                            else None
                        ),
                        approved_by=reviewer["id"] if is_approved else None,
                    )
                )
                plan_count += 1
            print(f"  Created {plan_count} reduction plans")

            review_count = 0
            for i, case_def in enumerate(DEMO_CASES):
                if case_def["status"] not in planning_statuses:
                    continue

                decision, notes, signature = _build_review_state(case_def["status"])
                db.add(
                    PlanReview(
                        id=_seed_uuid(profile, f"review-{i}"),
                        case_id=case_ids[i],
                        plan_id=_seed_uuid(profile, f"plan-{i}"),
                        reviewer_id=reviewer["id"],
                        reviewer_name=reviewer["full_name"],
                        decision=decision,
                        notes=notes,
                        checklist=[
                            {
                                **dict(item),
                                "passed": (
                                    True
                                    if decision == "approved"
                                    else False
                                    if decision == "revision_requested" and item["severity"] == "required"
                                    else None
                                ),
                            }
                            for item in DEFAULT_CHECKLIST
                        ],
                        signature=signature,
                        signed_at=(
                            datetime.now(timezone.utc) - timedelta(days=2)
                            if decision == "approved"
                            else None
                        ),
                        created_at=base_date + timedelta(days=i * 7, hours=4),
                        updated_at=base_date + timedelta(days=i * 7, hours=8),
                    )
                )
                review_count += 1
            print(f"  Created {review_count} plan reviews")

        print(f"\nDatabase seeding complete for profile '{profile}'.")

    finally:
        await dispose_db_engine()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Seed demo data for Facial Align")
    parser.add_argument("--clear", action="store_true", help="Clear existing demo data")
    parser.add_argument("--db", action="store_true", help="Write directly to database via async SQLAlchemy")
    parser.add_argument("--output", "-o", type=str, default=None, help="Write to JSON file")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    parser.add_argument(
        "--profile",
        choices=["demo", "release_test"],
        default="demo",
        help="Seed profile to generate or write",
    )
    args = parser.parse_args()

    if args.clear:
        print(f"Clearing {args.profile} seed data from database...")
        asyncio.run(_clear_seed_data(args.profile))
        print("Done.")
        return

    if args.db:
        print(f"Seeding {args.profile} data directly to database...")
        asyncio.run(seed_to_database(args.profile))
        print("Done.")
        return

    print(f"Generating {args.profile} dataset...")
    dataset = build_demo_dataset(args.profile)

    print(f"  Created {len(dataset['patients'])} patients")
    print(f"  Created {len(dataset['cases'])} cases")
    for case in dataset["cases"]:
        frag_count = len(case["fragments"])
        has_seg = "Y" if case["segmentation"] else "N"
        has_plan = "Y" if case["reduction_plan"] else "N"
        print(
            f"    {case['case_id']}: {case['case_type']:20s} "
            f"status={case['status']:25s} "
            f"frags={frag_count} seg={has_seg} plan={has_plan}"
        )

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(dataset, f, indent=2 if args.pretty else None, default=str)
        print(f"\nWritten to {output_path}")
    else:
        # Write to default location
        suffix = "release_test_dataset.json" if args.profile == "release_test" else "demo_dataset.json"
        output_path = Path(__file__).parent.parent / "examples" / suffix
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(dataset, f, indent=2, default=str)
        print(f"\nWritten to {output_path}")

    print("Done.")


if __name__ == "__main__":
    main()
