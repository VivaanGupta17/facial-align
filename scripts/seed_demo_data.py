#!/usr/bin/env python3
"""
Seed the database with realistic demo data for development and demos.

Creates:
- 5 sample patients (de-identified)
- 5 cases spanning different fracture patterns
- CT study metadata for each case
- Sample segmentation results
- Sample reduction plans at various stages

Usage:
    python scripts/seed_demo_data.py
    python scripts/seed_demo_data.py --clear   # Clear existing demo data first
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
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
]

DEMO_CASES = [
    {
        "case_type": "bilateral_mandible",
        "fracture_classification": "AO CMF: 91-B3.1 (bilateral mandible body)",
        "status": "plan_approved",
        "surgeon": "Dr. Smith",
        "fragments": [
            {"id": "mandible_body_L", "structure": "mandible", "volume_mm3": 4500},
            {"id": "mandible_body_R", "structure": "mandible", "volume_mm3": 4200},
            {"id": "mandible_symphysis", "structure": "mandible", "volume_mm3": 3800},
        ],
        "ct_quality_grade": "A",
    },
    {
        "case_type": "le_fort_ii",
        "fracture_classification": "AO CMF: 92-C2 (Le Fort II with NOE)",
        "status": "planning",
        "surgeon": "Dr. Smith",
        "fragments": [
            {"id": "maxilla_central", "structure": "maxilla", "volume_mm3": 6200},
            {"id": "nasal_complex", "structure": "nasal", "volume_mm3": 1800},
            {"id": "orbital_floor_L", "structure": "orbit", "volume_mm3": 900},
            {"id": "orbital_floor_R", "structure": "orbit", "volume_mm3": 850},
        ],
        "ct_quality_grade": "A",
    },
    {
        "case_type": "subcondylar",
        "fracture_classification": "AO CMF: 91-A1.3 (right subcondylar)",
        "status": "segmentation_complete",
        "surgeon": "Dr. Johnson",
        "fragments": [
            {"id": "condyle_R", "structure": "mandible", "volume_mm3": 2100},
            {"id": "ramus_R", "structure": "mandible", "volume_mm3": 5500},
        ],
        "ct_quality_grade": "B",
    },
    {
        "case_type": "zmc",
        "fracture_classification": "AO CMF: 92-B1.1 (left ZMC, comminuted)",
        "status": "planning",
        "surgeon": "Dr. Johnson",
        "fragments": [
            {"id": "zygoma_L", "structure": "zygoma", "volume_mm3": 3200},
            {"id": "orbital_rim_L", "structure": "orbit", "volume_mm3": 1100},
            {"id": "zygomatic_arch_L", "structure": "zygoma", "volume_mm3": 800},
            {"id": "maxilla_lateral_L", "structure": "maxilla", "volume_mm3": 1500},
        ],
        "ct_quality_grade": "A",
    },
    {
        "case_type": "panfacial",
        "fracture_classification": "AO CMF: 93-C3 (panfacial, high complexity)",
        "status": "uploaded",
        "surgeon": "Dr. Smith",
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


def build_demo_dataset() -> Dict[str, Any]:
    """Build the complete demo dataset."""
    import random
    dataset = {"patients": [], "cases": [], "timestamp": datetime.now(timezone.utc).isoformat()}

    base_date = datetime.now(timezone.utc) - timedelta(days=30)

    for i, (patient, case_def, ct_study) in enumerate(
        zip(DEMO_PATIENTS, DEMO_CASES, DEMO_CT_STUDIES)
    ):
        case_id = generate_case_id()
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
            "segmentation": seg_result if case_def["status"] != "uploaded" else None,
            "fragments": case_def["fragments"],
            "reduction_plan": (
                generate_reduction_plan(case_def["fragments"])
                if case_def["status"] in ("planning", "plan_approved")
                else None
            ),
        }

        dataset["patients"].append(patient_record)
        dataset["cases"].append(case_record)

    return dataset


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Seed demo data for Facial Align")
    parser.add_argument("--clear", action="store_true", help="Clear existing demo data")
    parser.add_argument("--output", "-o", type=str, default=None, help="Write to JSON file instead of DB")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = parser.parse_args()

    print("Generating demo dataset...")
    dataset = build_demo_dataset()

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
        output_path = Path(__file__).parent.parent / "examples" / "demo_dataset.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(dataset, f, indent=2, default=str)
        print(f"\nWritten to {output_path}")

    print("Done.")


if __name__ == "__main__":
    main()
