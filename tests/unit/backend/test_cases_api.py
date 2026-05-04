from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.core.security import get_current_user, require_surgeon
from app.db.database import get_db_session
from tests.unit.backend.api_test_utils import (
    MockExecuteResult,
    make_case,
    make_db_override,
    make_session,
    make_study,
    make_user_override,
)

pytestmark = [pytest.mark.asyncio]


def _case_response_tail() -> list[MockExecuteResult]:
    return [
        MockExecuteResult(scalar_one_or_none=None),  # latest segmentation
        MockExecuteResult(scalar_one=0),             # segmentation count
        MockExecuteResult(scalar_one_or_none=None),  # latest plan
        MockExecuteResult(scalar_one=0),             # plan count
        MockExecuteResult(all_rows=[]),              # case-study links
    ]


async def test_create_case_returns_201_for_institution_scoped_surgeon(app, async_client):
    patient_id = uuid4()
    study_id = uuid4()
    patient = SimpleNamespace(id=patient_id, institution_code="DEMO-INST")
    study = make_study(id=study_id, patient_id=patient_id, institution_code="DEMO-INST")
    session = make_session(
        [
            MockExecuteResult(scalar_one_or_none=patient),
            MockExecuteResult(scalar_one_or_none=study),
            *_case_response_tail(),
        ]
    )
    app.dependency_overrides[get_db_session] = make_db_override(session)
    app.dependency_overrides[require_surgeon] = make_user_override(role="surgeon", institution_code="DEMO-INST")

    response = await async_client.post(
        "/api/v1/cases",
        json={
            "patient_id": str(patient_id),
            "study_id": str(study_id),
            "case_type": "TRAUMA",
            "fracture_classification": "Le Fort II",
            "planned_procedure": "ORIF",
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["patientId"] == str(patient_id)
    assert body["studyId"] == str(study_id)

    app.dependency_overrides.clear()


async def test_create_case_rejects_cross_institution_study(app, async_client):
    patient_id = uuid4()
    study_id = uuid4()
    patient = SimpleNamespace(id=patient_id, institution_code="OTHER-INST")
    study = make_study(
        id=study_id,
        patient_id=patient_id,
        institution_code="OTHER-INST",
        uploaded_by="another-user",
    )
    session = make_session(
        [
            MockExecuteResult(scalar_one_or_none=patient),
            MockExecuteResult(scalar_one_or_none=study),
        ]
    )
    app.dependency_overrides[get_db_session] = make_db_override(session)
    app.dependency_overrides[require_surgeon] = make_user_override(role="surgeon", institution_code="DEMO-INST")

    response = await async_client.post(
        "/api/v1/cases",
        json={
            "patient_id": str(patient_id),
            "study_id": str(study_id),
            "case_type": "TRAUMA",
        },
    )

    assert response.status_code == 403

    app.dependency_overrides.clear()


async def test_list_cases_returns_200(app, async_client):
    cases = [make_case(), make_case()]
    session = make_session(
        [
            MockExecuteResult(scalar_one=2),
            MockExecuteResult(scalars_all=cases),
        ]
    )
    app.dependency_overrides[get_db_session] = make_db_override(session)
    app.dependency_overrides[get_current_user] = make_user_override()

    response = await async_client.get("/api/v1/cases")

    assert response.status_code == 200
    assert len(response.json()["items"]) == 2

    app.dependency_overrides.clear()


async def test_get_case_returns_200_for_institution_scoped_reader(app, async_client):
    case = make_case(institution_code="DEMO-INST")
    session = make_session(
        [
            MockExecuteResult(scalar_one_or_none=case),
            *_case_response_tail(),
        ]
    )
    app.dependency_overrides[get_db_session] = make_db_override(session)
    app.dependency_overrides[get_current_user] = make_user_override(role="research", institution_code="DEMO-INST")

    response = await async_client.get(f"/api/v1/cases/{case.id}")

    assert response.status_code == 200
    assert response.json()["id"] == str(case.id)

    app.dependency_overrides.clear()


async def test_get_case_returns_403_for_out_of_scope_reader(app, async_client):
    case = make_case(
        institution_code="OTHER-INST",
        surgeon_id="another-surgeon",
        created_by="another-surgeon",
        reviewer_id="another-reviewer",
    )
    session = make_session([MockExecuteResult(scalar_one_or_none=case)])
    app.dependency_overrides[get_db_session] = make_db_override(session)
    app.dependency_overrides[get_current_user] = make_user_override(role="research", institution_code="DEMO-INST")

    response = await async_client.get(f"/api/v1/cases/{case.id}")

    assert response.status_code == 403

    app.dependency_overrides.clear()


async def test_update_case_returns_200_for_in_scope_surgeon(app, async_client):
    case = make_case(institution_code="DEMO-INST")
    session = make_session(
        [
            MockExecuteResult(scalar_one_or_none=case),
            *_case_response_tail(),
        ]
    )
    app.dependency_overrides[get_db_session] = make_db_override(session)
    app.dependency_overrides[require_surgeon] = make_user_override(role="surgeon", institution_code="DEMO-INST")

    response = await async_client.patch(
        f"/api/v1/cases/{case.id}",
        json={"planned_procedure": "ORIF with plate fixation"},
    )

    assert response.status_code == 200

    app.dependency_overrides.clear()


async def test_status_transition_returns_403_for_out_of_scope_surgeon(app, async_client):
    case = make_case(
        institution_code="OTHER-INST",
        surgeon_id="another-surgeon",
        created_by="another-surgeon",
    )
    session = make_session([MockExecuteResult(scalar_one_or_none=case)])
    app.dependency_overrides[get_db_session] = make_db_override(session)
    app.dependency_overrides[require_surgeon] = make_user_override(role="surgeon", institution_code="DEMO-INST")

    response = await async_client.post(
        f"/api/v1/cases/{case.id}/status",
        json={"new_status": "DICOM_PROCESSING"},
    )

    assert response.status_code == 403

    app.dependency_overrides.clear()
