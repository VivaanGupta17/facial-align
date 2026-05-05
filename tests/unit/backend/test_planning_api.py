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
    make_plan,
    make_segmentation,
    make_session,
    make_user_override,
)

pytestmark = [pytest.mark.asyncio]


async def test_create_reduction_plan_submits_async_job(app, async_client, monkeypatch):
    case = make_case(status="SEGMENTED")
    segmentation = make_segmentation(case_id=case.id, status="complete")
    session = make_session(
        [
            MockExecuteResult(scalar_one_or_none=case),
            MockExecuteResult(scalar_one_or_none=segmentation),
            MockExecuteResult(scalar_one_or_none=case),
            MockExecuteResult(scalar_one_or_none=None),
        ]
    )
    app.dependency_overrides[get_db_session] = make_db_override(session)
    app.dependency_overrides[require_surgeon] = make_user_override()
    fake_tasks = SimpleNamespace(
        run_reduction_planning_pipeline=SimpleNamespace(delay=lambda **kwargs: SimpleNamespace(id="plan-task-001"))
    )
    monkeypatch.setitem(sys.modules, "app.workers.tasks", fake_tasks)

    response = await async_client.post(
        "/api/v1/planning",
        json={
            "caseId": str(case.id),
            "segmentationId": str(segmentation.id),
            "modelName": "baseline_icp",
            "useIntactReference": True,
            "includeAlternativePlans": False,
        },
    )

    assert response.status_code == 202
    body = response.json()
    assert body["jobId"] == "plan-task-001"
    assert body["status"] == "PENDING"

    app.dependency_overrides.clear()


async def test_create_reduction_plan_rejects_mismatched_segmentation_case(
    app, async_client, monkeypatch
):
    case = make_case(status="SEGMENTED")
    other_case = make_case(id=uuid.uuid4(), status="SEGMENTED")
    segmentation = make_segmentation(case_id=other_case.id, status="complete")
    session = make_session(
        [
            MockExecuteResult(scalar_one_or_none=case),
            MockExecuteResult(scalar_one_or_none=segmentation),
            MockExecuteResult(scalar_one_or_none=other_case),
        ]
    )
    app.dependency_overrides[get_db_session] = make_db_override(session)
    app.dependency_overrides[require_surgeon] = make_user_override()
    fake_tasks = SimpleNamespace(
        run_reduction_planning_pipeline=SimpleNamespace(delay=lambda **kwargs: SimpleNamespace(id="plan-task-001"))
    )
    monkeypatch.setitem(sys.modules, "app.workers.tasks", fake_tasks)

    response = await async_client.post(
        "/api/v1/planning",
        json={
            "caseId": str(case.id),
            "segmentationId": str(segmentation.id),
            "modelName": "baseline_icp",
            "useIntactReference": True,
            "includeAlternativePlans": False,
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Segmentation does not belong to the requested case"

    app.dependency_overrides.clear()


async def test_get_plan_returns_serialized_fragment_data(app, async_client):
    plan = make_plan()
    session = make_session(
        [
            MockExecuteResult(scalar_one_or_none=plan),
            MockExecuteResult(scalar_one_or_none=make_case(id=plan.case_id)),
        ]
    )
    app.dependency_overrides[get_db_session] = make_db_override(session)
    app.dependency_overrides[get_current_user] = make_user_override()

    response = await async_client.get(f"/api/v1/planning/{plan.id}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == str(plan.id)
    assert body["planVersion"] == 1
    assert body["fragmentTransforms"][0]["fragmentId"] == "fragment-1"

    app.dependency_overrides.clear()


async def test_get_plan_forbidden_for_other_institution(app, async_client):
    plan = make_plan()
    inaccessible_case = make_case(
        id=plan.case_id,
        surgeon_id="another-surgeon",
        created_by="another-surgeon",
        institution_code="OTHER-INST",
    )
    session = make_session(
        [
            MockExecuteResult(scalar_one_or_none=plan),
            MockExecuteResult(scalar_one_or_none=inaccessible_case),
        ]
    )
    app.dependency_overrides[get_db_session] = make_db_override(session)
    app.dependency_overrides[get_current_user] = make_user_override(institution_code="DEMO-INST")

    response = await async_client.get(f"/api/v1/planning/{plan.id}")

    assert response.status_code == 403
    assert response.json()["detail"] == "You do not have access to this case"

    app.dependency_overrides.clear()


async def test_apply_surgeon_edit_creates_new_plan_version(app, async_client, monkeypatch):
    source_plan = make_plan(plan_version=1)
    latest_plan = make_plan(case_id=source_plan.case_id, plan_version=1)
    session = make_session(
        [
            MockExecuteResult(scalar_one_or_none=source_plan),
            MockExecuteResult(scalar_one_or_none=make_case(id=source_plan.case_id)),
            MockExecuteResult(scalar_one_or_none=latest_plan),
        ]
    )
    app.dependency_overrides[get_db_session] = make_db_override(session)
    app.dependency_overrides[require_surgeon] = make_user_override()
    fake_tasks = SimpleNamespace(
        run_reduction_refinement=SimpleNamespace(delay=lambda **kwargs: SimpleNamespace(id="refine-task-001"))
    )
    monkeypatch.setitem(sys.modules, "app.workers.tasks", fake_tasks)

    response = await async_client.post(
        f"/api/v1/planning/{source_plan.id}/surgeon-edit",
        json={
            "fragmentId": "fragment-1",
            "newTransform": {
                "rotationMatrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "translationMm": [4, 0, -2],
            },
            "notes": "Manual adjustment",
            "reOptimize": False,
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["planVersion"] == 2
    assert body["provenance"]["validationTier"] == "manual_override"

    app.dependency_overrides.clear()


async def test_export_plan_forbidden_for_other_institution(app, async_client):
    plan = make_plan()
    inaccessible_case = make_case(
        id=plan.case_id,
        surgeon_id="another-surgeon",
        created_by="another-surgeon",
        institution_code="OTHER-INST",
    )
    session = make_session(
        [
            MockExecuteResult(scalar_one_or_none=plan),
            MockExecuteResult(scalar_one_or_none=inaccessible_case),
        ]
    )
    app.dependency_overrides[get_db_session] = make_db_override(session)
    app.dependency_overrides[get_current_user] = make_user_override(institution_code="DEMO-INST")

    response = await async_client.post(
        f"/api/v1/planning/{plan.id}/export",
        json={"exportType": "full_assembly", "stlFormat": "binary"},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "You do not have access to this case"

    app.dependency_overrides.clear()
