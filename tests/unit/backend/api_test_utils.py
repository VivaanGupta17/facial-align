from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Iterable
from unittest.mock import AsyncMock, MagicMock

from app.core.security import CurrentUser, hash_password


class MockExecuteResult:
    def __init__(
        self,
        *,
        scalar_one_or_none: Any = None,
        scalar_one: Any = 0,
        scalars_all: list[Any] | None = None,
        first: Any = None,
        all_rows: list[Any] | None = None,
    ) -> None:
        self._scalar_one_or_none = scalar_one_or_none
        self._scalar_one = scalar_one
        self._scalars_all = scalars_all or []
        self._first = first
        self._all_rows = all_rows or []

    def scalar_one_or_none(self) -> Any:
        return self._scalar_one_or_none

    def scalar_one(self) -> Any:
        return self._scalar_one

    def scalars(self) -> SimpleNamespace:
        return SimpleNamespace(
            all=lambda: self._scalars_all,
            first=lambda: self._scalar_one_or_none,
        )

    def first(self) -> Any:
        return self._first

    def all(self) -> list[Any]:
        return self._all_rows


def make_session(results: Iterable[MockExecuteResult]) -> AsyncMock:
    session = AsyncMock()
    added_objects: list[Any] = []

    session.execute = AsyncMock(side_effect=list(results))
    session.add = MagicMock(side_effect=lambda obj: added_objects.append(obj))

    async def flush() -> None:
        now = datetime.now(timezone.utc)
        for obj in added_objects:
            if getattr(obj, "id", None) is None:
                setattr(obj, "id", uuid.uuid4())
            if hasattr(obj, "created_at") and getattr(obj, "created_at", None) is None:
                setattr(obj, "created_at", now)
            if hasattr(obj, "updated_at") and getattr(obj, "updated_at", None) is None:
                setattr(obj, "updated_at", now)
            if hasattr(obj, "plan_version") and getattr(obj, "plan_version", None) is None:
                setattr(obj, "plan_version", 1)
            if hasattr(obj, "status") and getattr(obj, "status", None) is None:
                setattr(obj, "status", "pending")

    session.flush = AsyncMock(side_effect=flush)
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    return session


def make_db_override(session: AsyncMock):
    async def override():
        yield session

    return override


def make_user_override(
    user_id: str = "test-surgeon-001",
    role: str = "surgeon",
    institution_code: str | None = "DEMO-INST",
):
    async def override():
        return CurrentUser(
            user_id=user_id,
            role=role,
            jti="test-jti",
            institution_code=institution_code,
        )

    return override


def make_user(
    *,
    email: str = "surgeon@example.com",
    password: str = "SecureP@ss123",
    role: str = "surgeon",
    is_active: bool = True,
) -> MagicMock:
    user = MagicMock()
    user.id = uuid.uuid4()
    user.email = email
    user.hashed_password = hash_password(password)
    user.full_name = "Dr. Test Surgeon"
    user.role = role
    user.institution = "Test Hospital"
    user.institution_code = "DEMO-INST"
    user.specialty = "OMFS"
    user.is_active = is_active
    user.is_verified = True
    user.login_count = 0
    user.last_login_at = None
    user.created_at = datetime.now(timezone.utc)
    user.updated_at = datetime.now(timezone.utc)
    return user


def make_case(**overrides: Any) -> MagicMock:
    case = MagicMock()
    case.id = overrides.get("id", uuid.uuid4())
    case.case_number = overrides.get("case_number", "FA-2026-TEST")
    case.patient_id = overrides.get("patient_id", uuid.uuid4())
    case.study_id = overrides.get("study_id", uuid.uuid4())
    case.case_type = overrides.get("case_type", "TRAUMA")
    case.status = overrides.get("status", "CREATED")
    case.surgeon_id = overrides.get("surgeon_id", "test-surgeon-001")
    case.reviewer_id = overrides.get("reviewer_id", None)
    case.fracture_classification = overrides.get("fracture_classification", "AO CMF: 91-B3.1")
    case.planned_procedure = overrides.get("planned_procedure", "ORIF")
    case.diagnosis_codes = overrides.get("diagnosis_codes", ["S02.66XA"])
    case.target_surgery_date = overrides.get("target_surgery_date", None)
    case.team_ids = overrides.get("team_ids", [])
    case.current_task_id = overrides.get("current_task_id", None)
    case.last_error = overrides.get("last_error", None)
    case.created_at = overrides.get("created_at", datetime.now(timezone.utc))
    case.updated_at = overrides.get("updated_at", datetime.now(timezone.utc))
    case.approved_at = overrides.get("approved_at", None)
    case.created_by = overrides.get("created_by", "test-surgeon-001")
    case.institution_code = overrides.get("institution_code", "DEMO-INST")
    case.transition_to = overrides.get("transition_to", MagicMock())
    return case


def make_segmentation(**overrides: Any) -> MagicMock:
    segmentation = MagicMock()
    segmentation.id = overrides.get("id", uuid.uuid4())
    segmentation.case_id = overrides.get("case_id", uuid.uuid4())
    segmentation.status = overrides.get("status", "complete")
    segmentation.model_name = overrides.get("model_name", "totalsegmentator")
    segmentation.model_version = overrides.get("model_version", "2.0.1")
    segmentation.structure_labels = overrides.get("structure_labels", {"mandible": 1})
    segmentation.confidence_scores = overrides.get("confidence_scores", {"mandible": 0.94})
    segmentation.volume_stats = overrides.get("volume_stats", {"mandible": {"volume_cc": 8.4}})
    segmentation.structures = overrides.get("structures", {"mandible": {"status": "pending"}})
    segmentation.overall_confidence = overrides.get("overall_confidence", 0.94)
    segmentation.provenance = overrides.get("provenance", {"algorithm_used": "totalsegmentator"})
    segmentation.mask_storage_path = overrides.get("mask_storage_path", "/tmp/mask.nii.gz")
    segmentation.mesh_storage_paths = overrides.get(
        "mesh_storage_paths",
        {"mandible": {"glb": "/tmp/mandible.glb", "stl": "/tmp/mandible.stl"}},
    )
    segmentation.fragment_count = overrides.get("fragment_count", 1)
    segmentation.fracture_fragments = overrides.get("fracture_fragments", {})
    segmentation.inference_time_ms = overrides.get("inference_time_ms", 1000)
    segmentation.total_pipeline_time_ms = overrides.get("total_pipeline_time_ms", 1500)
    segmentation.gpu_device = overrides.get("gpu_device", "cpu")
    segmentation.error_message = overrides.get("error_message", None)
    segmentation.created_at = overrides.get("created_at", datetime.now(timezone.utc))
    segmentation.completed_at = overrides.get("completed_at", datetime.now(timezone.utc))
    segmentation.celery_task_id = overrides.get("celery_task_id", "seg-task-001")
    return segmentation


def make_plan(**overrides: Any) -> MagicMock:
    plan = MagicMock()
    plan.id = overrides.get("id", uuid.uuid4())
    plan.case_id = overrides.get("case_id", uuid.uuid4())
    plan.segmentation_id = overrides.get("segmentation_id", uuid.uuid4())
    plan.plan_version = overrides.get("plan_version", 1)
    plan.status = overrides.get("status", "approved")
    plan.model_name = overrides.get("model_name", "baseline_icp")
    plan.model_version = overrides.get("model_version", "1.0.0")
    plan.fragments = overrides.get(
        "fragments",
        {
            "fragment-1": {
                "label": 1,
                "fragment_id": "fragment-1",
                "volume_cc": 8.4,
                "centroid_mm": [12, -18, 4],
                "parent_structure": "mandible",
            }
        },
    )
    plan.transformations = overrides.get(
        "transformations",
        {
            "fragment-1": {
                "fragment_label": 1,
                "transform": {
                    "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "translation_mm": [2, 0, -1],
                },
                "confidence": 0.82,
            }
        },
    )
    plan.dental_constraints = overrides.get("dental_constraints", {"target_overjet_mm": 2.0})
    plan.skeletal_constraints = overrides.get("skeletal_constraints", {})
    plan.occlusal_metrics = overrides.get("occlusal_metrics", {"overjet_mm": 1.5})
    plan.provenance = overrides.get("provenance", {"algorithm_used": "baseline_icp"})
    plan.confidence_score = overrides.get("confidence_score", 0.82)
    plan.surgeon_approved = overrides.get("surgeon_approved", False)
    plan.surgeon_notes = overrides.get("surgeon_notes", None)
    plan.parent_plan_id = overrides.get("parent_plan_id", None)
    plan.is_ml_generated = overrides.get("is_ml_generated", False)
    plan.generation_time_ms = overrides.get("generation_time_ms", 2300)
    plan.created_at = overrides.get("created_at", datetime.now(timezone.utc))
    plan.approved_at = overrides.get("approved_at", None)
    plan.approved_by = overrides.get("approved_by", None)
    plan.surgeon_edits = overrides.get("surgeon_edits", [])
    return plan


def make_review(**overrides: Any) -> MagicMock:
    review = MagicMock()
    review.id = overrides.get("id", uuid.uuid4())
    review.case_id = overrides.get("case_id", uuid.uuid4())
    review.plan_id = overrides.get("plan_id", uuid.uuid4())
    review.reviewer_id = overrides.get("reviewer_id", "test-surgeon-001")
    review.reviewer_name = overrides.get("reviewer_name", "Dr. Test Surgeon")
    review.decision = overrides.get("decision", "pending")
    review.notes = overrides.get("notes", "")
    review.checklist = overrides.get(
        "checklist",
        [
            {
                "id": "seg-accuracy",
                "category": "Segmentation",
                "label": "Bone segmentation boundaries are accurate",
                "passed": None,
                "severity": "required",
            }
        ],
    )
    review.signature = overrides.get("signature", None)
    review.signed_at = overrides.get("signed_at", None)
    review.created_at = overrides.get("created_at", datetime.now(timezone.utc))
    review.updated_at = overrides.get("updated_at", datetime.now(timezone.utc))
    return review


def make_study(**overrides: Any) -> MagicMock:
    study = MagicMock()
    study.id = overrides.get("id", uuid.uuid4())
    study.study_uid = overrides.get("study_uid", "1.2.3.4")
    study.patient_id = overrides.get("patient_id", uuid.uuid4())
    study.modality = overrides.get("modality", "CT")
    study.acquisition_date = overrides.get("acquisition_date", datetime.now(timezone.utc).date())
    study.series_count = overrides.get("series_count", 1)
    study.ingestion_status = overrides.get("ingestion_status", "complete")
    study.quality_score = overrides.get("quality_score", 0.97)
    study.created_at = overrides.get("created_at", datetime.now(timezone.utc))
    study.slice_count = overrides.get("slice_count", 120)
    study.pixel_spacing_mm = overrides.get("pixel_spacing_mm", 0.5)
    study.slice_thickness_mm = overrides.get("slice_thickness_mm", 0.6)
    study.uploaded_by = overrides.get("uploaded_by", "test-surgeon-001")
    study.institution_code = overrides.get("institution_code", "DEMO-INST")
    study.metadata_json = overrides.get(
        "metadata_json",
        {"Columns": 512, "Rows": 512, "InstitutionName": "Test Hospital"},
    )
    return study
