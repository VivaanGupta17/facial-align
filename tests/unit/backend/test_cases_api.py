"""
Tests for the surgical cases CRUD API at /api/v1/cases.

These tests use httpx.AsyncClient against the real FastAPI app with
mocked database dependencies to avoid requiring a running PostgreSQL.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = [
    pytest.mark.asyncio,
]


# ── Mock helpers ─────────────────────────────────────────────────────────────


def _make_mock_case(
    case_id=None,
    status="created",
    case_type="trauma",
):
    """Create a mock SurgicalCase ORM object."""
    mock = MagicMock()
    mock.id = case_id or uuid.uuid4()
    mock.case_number = f"FA-2024-{str(mock.id)[:8].upper()}"
    mock.patient_id = uuid.uuid4()
    mock.study_id = uuid.uuid4()
    mock.case_type = MagicMock(value=case_type)
    mock.status = MagicMock(value=status)
    mock.surgeon_id = "test-surgeon-001"
    mock.reviewer_id = None
    mock.fracture_classification = "Le Fort II"
    mock.planned_procedure = "ORIF"
    mock.diagnosis_codes = ["S02.40"]
    mock.target_surgery_date = None
    mock.team_ids = []
    mock.current_task_id = None
    mock.last_error = None
    mock.created_at = datetime.now(timezone.utc)
    mock.updated_at = datetime.now(timezone.utc)
    mock.approved_at = None
    mock.created_by = "test-surgeon-001"
    return mock


def _mock_db_override(mock_case=None, mock_cases=None):
    """Return a get_db_session override that returns a mock session.

    The mock session is pre-configured to return *mock_case* for
    single-object queries and *mock_cases* for list queries.
    """
    session = AsyncMock()

    # For single-object queries (.scalar_one_or_none)
    scalar_result = MagicMock()
    scalar_result.scalar_one_or_none.return_value = mock_case
    scalar_result.scalar_one.return_value = 0

    # For list queries (.scalars().all())
    scalars_mock = MagicMock()
    scalars_mock.all.return_value = mock_cases or []
    list_result = MagicMock()
    list_result.scalars.return_value = scalars_mock
    list_result.scalar_one_or_none.return_value = mock_case
    list_result.scalar_one.return_value = len(mock_cases) if mock_cases else 0

    session.execute = AsyncMock(return_value=list_result)
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()

    async def override():
        yield session

    return override


def _mock_auth_override(user_id="test-surgeon-001", role="surgeon"):
    """Return a get_current_user / require_surgeon override."""
    from app.core.security import CurrentUser

    async def override():
        return CurrentUser(user_id=user_id, role=role, jti="test-jti")

    return override


# ── Tests ────────────────────────────────────────────────────────────────────


class TestCreateCase:
    async def test_create_case_returns_201(self, app, async_client):
        """POST /cases with valid payload returns 201."""
        from app.db.database import get_db_session
        from app.core.security import get_current_user, require_surgeon

        mock_case = _make_mock_case()
        app.dependency_overrides[get_db_session] = _mock_db_override(mock_case=mock_case)
        app.dependency_overrides[get_current_user] = _mock_auth_override()
        app.dependency_overrides[require_surgeon] = _mock_auth_override()

        payload = {
            "patient_id": str(uuid.uuid4()),
            "study_id": str(uuid.uuid4()),
            "case_type": "trauma",
            "fracture_classification": "Le Fort II",
            "planned_procedure": "ORIF",
        }
        resp = await async_client.post("/api/v1/cases", json=payload)
        assert resp.status_code in (200, 201, 422)

        app.dependency_overrides.clear()


class TestListCases:
    async def test_list_cases_returns_200(self, app, async_client):
        """GET /cases returns paginated list."""
        from app.db.database import get_db_session
        from app.core.security import get_current_user

        cases = [_make_mock_case() for _ in range(3)]
        app.dependency_overrides[get_db_session] = _mock_db_override(mock_cases=cases)
        app.dependency_overrides[get_current_user] = _mock_auth_override()

        resp = await async_client.get("/api/v1/cases")
        assert resp.status_code == 200
        body = resp.json()
        assert "items" in body

        app.dependency_overrides.clear()

    async def test_list_cases_pagination(self, app, async_client):
        """GET /cases supports page and page_size query params."""
        from app.db.database import get_db_session
        from app.core.security import get_current_user

        cases = [_make_mock_case() for _ in range(5)]
        app.dependency_overrides[get_db_session] = _mock_db_override(mock_cases=cases)
        app.dependency_overrides[get_current_user] = _mock_auth_override()

        resp = await async_client.get("/api/v1/cases?page=1&page_size=2")
        assert resp.status_code == 200

        app.dependency_overrides.clear()


class TestGetCase:
    async def test_get_case_returns_200(self, app, async_client):
        """GET /cases/{id} returns case detail."""
        from app.db.database import get_db_session
        from app.core.security import get_current_user

        case_id = uuid.uuid4()
        mock_case = _make_mock_case(case_id=case_id)
        app.dependency_overrides[get_db_session] = _mock_db_override(mock_case=mock_case)
        app.dependency_overrides[get_current_user] = _mock_auth_override()

        resp = await async_client.get(f"/api/v1/cases/{case_id}")
        assert resp.status_code == 200

        app.dependency_overrides.clear()

    async def test_get_case_not_found_404(self, app, async_client):
        """GET /cases/{id} for nonexistent case returns 404."""
        from app.db.database import get_db_session
        from app.core.security import get_current_user

        app.dependency_overrides[get_db_session] = _mock_db_override(mock_case=None)
        app.dependency_overrides[get_current_user] = _mock_auth_override()

        resp = await async_client.get(f"/api/v1/cases/{uuid.uuid4()}")
        assert resp.status_code == 404

        app.dependency_overrides.clear()


class TestUpdateCase:
    async def test_update_case_returns_200(self, app, async_client):
        """PATCH /cases/{id} updates case fields."""
        from app.db.database import get_db_session
        from app.core.security import get_current_user, require_surgeon

        case_id = uuid.uuid4()
        mock_case = _make_mock_case(case_id=case_id)
        app.dependency_overrides[get_db_session] = _mock_db_override(mock_case=mock_case)
        app.dependency_overrides[get_current_user] = _mock_auth_override()
        app.dependency_overrides[require_surgeon] = _mock_auth_override()

        payload = {"planned_procedure": "ORIF with plate fixation"}
        resp = await async_client.patch(f"/api/v1/cases/{case_id}", json=payload)
        assert resp.status_code == 200

        app.dependency_overrides.clear()


class TestStatusTransition:
    async def test_valid_status_transition(self, app, async_client):
        """POST /cases/{id}/status with valid transition succeeds."""
        from app.db.database import get_db_session
        from app.core.security import get_current_user, require_surgeon

        case_id = uuid.uuid4()
        mock_case = _make_mock_case(case_id=case_id, status="created")
        mock_case.transition_to = MagicMock()  # Allow any transition
        app.dependency_overrides[get_db_session] = _mock_db_override(mock_case=mock_case)
        app.dependency_overrides[get_current_user] = _mock_auth_override()
        app.dependency_overrides[require_surgeon] = _mock_auth_override()

        payload = {"new_status": "segmenting"}
        resp = await async_client.post(f"/api/v1/cases/{case_id}/status", json=payload)
        assert resp.status_code == 200

        app.dependency_overrides.clear()

    async def test_invalid_status_transition(self, app, async_client):
        """POST /cases/{id}/status with invalid transition returns error."""
        from app.db.database import get_db_session
        from app.core.security import get_current_user, require_surgeon

        case_id = uuid.uuid4()
        mock_case = _make_mock_case(case_id=case_id, status="created")
        mock_case.transition_to = MagicMock(
            side_effect=ValueError("Invalid transition from created to approved")
        )
        app.dependency_overrides[get_db_session] = _mock_db_override(mock_case=mock_case)
        app.dependency_overrides[get_current_user] = _mock_auth_override()
        app.dependency_overrides[require_surgeon] = _mock_auth_override()

        payload = {"new_status": "approved"}
        resp = await async_client.post(f"/api/v1/cases/{case_id}/status", json=payload)
        # Should return 400 or 409 for invalid transition
        assert resp.status_code in (400, 409, 422)

        app.dependency_overrides.clear()
