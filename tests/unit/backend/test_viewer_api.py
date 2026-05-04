from __future__ import annotations

import uuid

import pytest

from app.core.security import get_current_user
from app.db.database import get_db_session
from tests.unit.backend.api_test_utils import (
    make_case,
    MockExecuteResult,
    make_db_override,
    make_plan,
    make_segmentation,
    make_session,
    make_user_override,
)

pytestmark = [pytest.mark.asyncio]


async def test_get_scene_assets_returns_mesh_and_fragment_urls(app, async_client):
    case_id = uuid.uuid4()
    case = make_case(id=case_id, institution_code="DEMO-INST")
    segmentation = make_segmentation(case_id=case_id, fragment_masks_path="/tmp/fragments.nii.gz")
    plan = make_plan(case_id=case_id)
    session = make_session(
        [
            MockExecuteResult(scalar_one_or_none=case),
            MockExecuteResult(scalar_one_or_none=segmentation),
            MockExecuteResult(scalar_one_or_none=case),
            MockExecuteResult(scalar_one_or_none=plan),
        ]
    )
    app.dependency_overrides[get_db_session] = make_db_override(session)
    app.dependency_overrides[get_current_user] = make_user_override()

    response = await async_client.get(
        f"/api/v1/viewer/cases/{case_id}/assets",
        params={"segmentation_id": str(segmentation.id), "plan_id": str(plan.id)},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["segmentationId"] == str(segmentation.id)
    assert "mandible" in body["meshes"]
    assert "fragment-1" in body["fragmentMeshes"]

    app.dependency_overrides.clear()


async def test_get_scene_assets_returns_403_for_cross_institution_access(app, async_client):
    case_id = uuid.uuid4()
    case = make_case(id=case_id, institution_code="OTHER-INST", surgeon_id="another-surgeon", created_by="another-surgeon")
    session = make_session([MockExecuteResult(scalar_one_or_none=case)])
    app.dependency_overrides[get_db_session] = make_db_override(session)
    app.dependency_overrides[get_current_user] = make_user_override(institution_code="DEMO-INST")

    response = await async_client.get(f"/api/v1/viewer/cases/{case_id}/assets")

    assert response.status_code == 403

    app.dependency_overrides.clear()


async def test_landmarks_endpoint_returns_standard_landmarks(app, async_client):
    case_id = uuid.uuid4()
    case = make_case(id=case_id, institution_code="DEMO-INST")
    session = make_session([MockExecuteResult(scalar_one_or_none=case)])
    app.dependency_overrides[get_db_session] = make_db_override(session)
    app.dependency_overrides[get_current_user] = make_user_override()

    response = await async_client.get(f"/api/v1/viewer/cases/{case_id}/landmarks")

    assert response.status_code == 200
    body = response.json()
    assert len(body) >= 3
    assert body[0]["name"] in {"N", "ANS", "PNS", "Me", "Pg"}

    app.dependency_overrides.clear()
