from __future__ import annotations

import pytest

from app.core.security import get_current_user, require_reviewer
from app.db.database import get_db_session
from tests.unit.backend.api_test_utils import (
    MockExecuteResult,
    make_case,
    make_db_override,
    make_plan,
    make_review,
    make_session,
    make_user_override,
)

pytestmark = [pytest.mark.asyncio]


async def test_get_review_creates_persistent_review_when_missing(app, async_client):
    case = make_case(institution_code="DEMO-INST")
    latest_plan = make_plan(case_id=case.id)
    session = make_session(
        [
            MockExecuteResult(scalar_one_or_none=case),
            MockExecuteResult(scalar_one_or_none=None),
            MockExecuteResult(scalar_one_or_none=latest_plan),
        ]
    )
    app.dependency_overrides[get_db_session] = make_db_override(session)
    app.dependency_overrides[get_current_user] = make_user_override()

    response = await async_client.get(f"/api/v1/reviews/{case.id}")

    assert response.status_code == 200
    body = response.json()
    assert body["caseId"] == str(case.id)
    assert body["decision"] == "pending"
    assert body["checklist"]

    app.dependency_overrides.clear()


async def test_update_checklist_marks_item_reviewed(app, async_client):
    review = make_review()
    case = make_case(id=review.case_id, institution_code="DEMO-INST")
    session = make_session(
        [
            MockExecuteResult(scalar_one_or_none=review),
            MockExecuteResult(scalar_one_or_none=case),
        ]
    )
    app.dependency_overrides[get_db_session] = make_db_override(session)
    app.dependency_overrides[require_reviewer] = make_user_override(role="reviewer")

    response = await async_client.patch(
        f"/api/v1/reviews/{review.id}/checklist",
        json={"checklistId": "seg-accuracy", "passed": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["checklist"][0]["passed"] is True
    assert review.checklist[0]["reviewed_by"] == "test-surgeon-001"

    app.dependency_overrides.clear()


async def test_approve_review_updates_decision_and_plan(app, async_client):
    review = make_review()
    plan = make_plan(id=review.plan_id)
    case = make_case(id=review.case_id, institution_code="DEMO-INST")
    session = make_session(
        [
            MockExecuteResult(scalar_one_or_none=review),
            MockExecuteResult(scalar_one_or_none=case),
            MockExecuteResult(scalar_one_or_none=plan),
        ]
    )
    app.dependency_overrides[get_db_session] = make_db_override(session)
    app.dependency_overrides[require_reviewer] = make_user_override(role="reviewer")

    response = await async_client.post(
        f"/api/v1/reviews/{review.id}/approve",
        json={"notes": "Looks good", "signature": "signed"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "approved"
    assert plan.status == "approved"
    assert case.reviewer_id == "test-surgeon-001"

    app.dependency_overrides.clear()


async def test_reject_review_returns_403_for_out_of_scope_reviewer(app, async_client):
    review = make_review()
    case = make_case(
        id=review.case_id,
        institution_code="OTHER-INST",
        surgeon_id="another-surgeon",
        created_by="another-surgeon",
        reviewer_id="another-reviewer",
    )
    session = make_session(
        [
            MockExecuteResult(scalar_one_or_none=review),
            MockExecuteResult(scalar_one_or_none=case),
        ]
    )
    app.dependency_overrides[get_db_session] = make_db_override(session)
    app.dependency_overrides[require_reviewer] = make_user_override(role="reviewer", institution_code="DEMO-INST")

    response = await async_client.post(
        f"/api/v1/reviews/{review.id}/reject",
        json={"notes": "Outside my institution"},
    )

    assert response.status_code == 403

    app.dependency_overrides.clear()
