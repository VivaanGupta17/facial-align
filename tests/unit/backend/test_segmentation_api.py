from __future__ import annotations

import sys
from types import SimpleNamespace
import uuid

import pytest

from app.core.security import get_current_user, require_surgeon
from app.db.database import get_db_session
from tests.unit.backend.api_test_utils import (
    MockExecuteResult,
    make_case,
    make_db_override,
    make_segmentation,
    make_session,
    make_user_override,
)

pytestmark = [pytest.mark.asyncio]


async def test_trigger_segmentation_creates_job_record(app, async_client, monkeypatch):
    case = make_case(status="DICOM_PROCESSING")
    session = make_session([MockExecuteResult(scalar_one_or_none=case)])
    app.dependency_overrides[get_db_session] = make_db_override(session)
    app.dependency_overrides[require_surgeon] = make_user_override()

    fake_tasks = SimpleNamespace(
      run_segmentation_pipeline=SimpleNamespace(delay=lambda **kwargs: SimpleNamespace(id="seg-task-001"))
    )
    monkeypatch.setitem(sys.modules, "app.workers.tasks", fake_tasks)

    response = await async_client.post(
        "/api/v1/segmentation",
        json={
            "caseId": str(case.id),
            "modelName": "totalsegmentator",
            "identifyFragments": True,
            "runDentalSegmentation": False,
            "fastMode": False,
        },
    )

    assert response.status_code == 202
    body = response.json()
    assert body["jobId"] == "seg-task-001"
    assert body["caseId"] == str(case.id)
    assert body["status"] == "queued"

    app.dependency_overrides.clear()


async def test_approve_structure_marks_review_status(app, async_client, monkeypatch):
    segmentation = make_segmentation(structures={"mandible": {"status": "pending"}})
    session = make_session(
        [
            MockExecuteResult(scalar_one_or_none=segmentation),
            MockExecuteResult(scalar_one_or_none=make_case(id=segmentation.case_id)),
        ]
    )
    app.dependency_overrides[get_db_session] = make_db_override(session)
    app.dependency_overrides[require_surgeon] = make_user_override()
    monkeypatch.setattr("sqlalchemy.orm.attributes.flag_modified", lambda *args, **kwargs: None)

    response = await async_client.post(f"/api/v1/segmentation/{segmentation.id}/structures/mandible/approve")

    assert response.status_code == 200
    assert response.json() == {"status": "accepted", "label": "mandible"}
    assert segmentation.structures["mandible"]["status"] == "accepted"

    app.dependency_overrides.clear()


async def test_request_resegmentation_returns_new_job(app, async_client, monkeypatch):
    segmentation = make_segmentation(case_id=uuid.uuid4())
    case = make_case(id=segmentation.case_id)
    session = make_session(
        [
            MockExecuteResult(scalar_one_or_none=segmentation),
            MockExecuteResult(scalar_one_or_none=case),
        ]
    )
    app.dependency_overrides[get_db_session] = make_db_override(session)
    app.dependency_overrides[require_surgeon] = make_user_override()
    monkeypatch.setattr("sqlalchemy.orm.attributes.flag_modified", lambda *args, **kwargs: None)
    fake_tasks = SimpleNamespace(
      run_segmentation_pipeline=SimpleNamespace(delay=lambda **kwargs: SimpleNamespace(id="resegment-task-001"))
    )
    monkeypatch.setitem(sys.modules, "app.workers.tasks", fake_tasks)

    response = await async_client.post(
        f"/api/v1/segmentation/{segmentation.id}/structures/mandible/resegment"
    )

    assert response.status_code == 202
    assert response.json()["job_id"] == "resegment-task-001"
    assert segmentation.structures["mandible"]["resegment_task_id"] == "resegment-task-001"

    app.dependency_overrides.clear()


async def test_get_segmentation_forbidden_for_other_institution(app, async_client):
    segmentation = make_segmentation()
    inaccessible_case = make_case(
        id=segmentation.case_id,
        surgeon_id="another-surgeon",
        created_by="another-surgeon",
        institution_code="OTHER-INST",
    )
    session = make_session(
        [
            MockExecuteResult(scalar_one_or_none=segmentation),
            MockExecuteResult(scalar_one_or_none=inaccessible_case),
        ]
    )
    app.dependency_overrides[get_db_session] = make_db_override(session)
    app.dependency_overrides[get_current_user] = make_user_override(institution_code="DEMO-INST")

    response = await async_client.get(f"/api/v1/segmentation/{segmentation.id}")

    assert response.status_code == 403
    assert response.json()["detail"] == "You do not have access to this case"

    app.dependency_overrides.clear()


async def test_request_resegmentation_forbidden_for_other_institution(
    app, async_client, monkeypatch
):
    segmentation = make_segmentation()
    inaccessible_case = make_case(
        id=segmentation.case_id,
        surgeon_id="another-surgeon",
        created_by="another-surgeon",
        institution_code="OTHER-INST",
    )
    session = make_session(
        [
            MockExecuteResult(scalar_one_or_none=segmentation),
            MockExecuteResult(scalar_one_or_none=inaccessible_case),
        ]
    )
    app.dependency_overrides[get_db_session] = make_db_override(session)
    app.dependency_overrides[require_surgeon] = make_user_override(institution_code="DEMO-INST")
    monkeypatch.setattr("sqlalchemy.orm.attributes.flag_modified", lambda *args, **kwargs: None)

    response = await async_client.post(
        f"/api/v1/segmentation/{segmentation.id}/structures/mandible/resegment"
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "You do not have write access to this case"

    app.dependency_overrides.clear()
