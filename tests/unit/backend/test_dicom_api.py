from __future__ import annotations

import io

import pytest

from app.core.security import get_current_user
from app.db.database import get_db_session
from tests.unit.backend.api_test_utils import (
    MockExecuteResult,
    make_db_override,
    make_session,
    make_study,
    make_user_override,
)

pytestmark = [pytest.mark.asyncio]


async def test_list_studies_returns_paginated_study_items(app, async_client):
    studies = [make_study(), make_study()]
    session = make_session([MockExecuteResult(scalars_all=studies)])
    app.dependency_overrides[get_db_session] = make_db_override(session)
    app.dependency_overrides[get_current_user] = make_user_override()

    response = await async_client.get("/api/v1/dicom/studies?page=1&page_size=20")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert {"id", "studyUid", "ingestionStatus"}.issubset(body[0].keys())

    app.dependency_overrides.clear()


async def test_get_study_metadata_returns_detail_view(app, async_client):
    study = make_study()
    session = make_session([MockExecuteResult(scalar_one_or_none=study)])
    app.dependency_overrides[get_db_session] = make_db_override(session)
    app.dependency_overrides[get_current_user] = make_user_override()

    response = await async_client.get(f"/api/v1/dicom/studies/{study.id}")

    assert response.status_code == 200
    body = response.json()
    assert body["studyUid"] == study.study_uid
    assert body["modality"] == "CT"

    app.dependency_overrides.clear()


async def test_get_study_metadata_returns_403_for_out_of_scope_user(app, async_client):
    study = make_study(institution_code="OTHER-INST", uploaded_by="another-user")
    session = make_session([MockExecuteResult(scalar_one_or_none=study)])
    app.dependency_overrides[get_db_session] = make_db_override(session)
    app.dependency_overrides[get_current_user] = make_user_override(institution_code="DEMO-INST")

    response = await async_client.get(f"/api/v1/dicom/studies/{study.id}")

    assert response.status_code == 403

    app.dependency_overrides.clear()


async def test_upload_rejects_invalid_file_types(app, async_client):
    session = make_session([])
    app.dependency_overrides[get_db_session] = make_db_override(session)
    app.dependency_overrides[get_current_user] = make_user_override()

    response = await async_client.post(
        "/api/v1/dicom/upload",
        data={"patient_mrn": "MRN-001"},
        files={"files": ("notes.txt", io.BytesIO(b"not a dicom"), "text/plain")},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "DICOM_VALIDATION_ERROR"

    app.dependency_overrides.clear()
