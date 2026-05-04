from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.core.security import create_refresh_token
from app.core.security import get_current_user
from app.db.database import get_db_session
from tests.unit.backend.api_test_utils import (
    MockExecuteResult,
    make_db_override,
    make_session,
    make_user,
    make_user_override,
)

pytestmark = [pytest.mark.asyncio]


class TestRegister:
    async def test_register_success_returns_tokens(self, app, async_client):
        session = make_session([MockExecuteResult(scalar_one_or_none=None)])
        app.dependency_overrides[get_db_session] = make_db_override(session)

        response = await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": "newuser@example.com",
                "password": "SecureP@ss123",
                "full_name": "Test User",
                "role": "surgeon",
            },
        )

        assert response.status_code == 201
        body = response.json()
        assert body["tokenType"] == "bearer"
        assert body["accessToken"]
        assert body["refreshToken"]

        app.dependency_overrides.clear()

    async def test_register_duplicate_email_returns_409(self, app, async_client):
        session = make_session([MockExecuteResult(scalar_one_or_none=make_user())])
        app.dependency_overrides[get_db_session] = make_db_override(session)

        response = await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": "duplicate@example.com",
                "password": "SecureP@ss123",
                "full_name": "Duplicate User",
                "role": "surgeon",
            },
        )

        assert response.status_code == 409
        assert response.json()["detail"] == "Email already registered"

        app.dependency_overrides.clear()


class TestLogin:
    async def test_login_success_returns_tokens(self, app, async_client):
        session = make_session([MockExecuteResult(scalar_one_or_none=make_user())])
        app.dependency_overrides[get_db_session] = make_db_override(session)

        response = await async_client.post(
            "/api/v1/auth/login",
            json={"email": "surgeon@example.com", "password": "SecureP@ss123"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["tokenType"] == "bearer"
        assert body["accessToken"]
        assert body["refreshToken"]

        app.dependency_overrides.clear()

    async def test_login_invalid_password_returns_401(self, app, async_client):
        session = make_session(
            [MockExecuteResult(scalar_one_or_none=make_user(password="CorrectPassword1!"))]
        )
        app.dependency_overrides[get_db_session] = make_db_override(session)

        response = await async_client.post(
            "/api/v1/auth/login",
            json={"email": "surgeon@example.com", "password": "WrongPassword1!"},
        )

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid email or password"

        app.dependency_overrides.clear()


class TestMeAndRefresh:
    async def test_me_returns_current_user_profile(self, app, async_client):
        user = SimpleNamespace(
            id=make_user().id,
            email="surgeon@example.com",
            full_name="Dr. Test Surgeon",
            role="surgeon",
            institution="Test Hospital",
            specialty="OMFS",
            is_active=True,
            created_at=make_user().created_at,
        )
        session = make_session([MockExecuteResult(scalar_one_or_none=user)])
        app.dependency_overrides[get_db_session] = make_db_override(session)
        app.dependency_overrides[get_current_user] = make_user_override()

        response = await async_client.get("/api/v1/auth/me")

        assert response.status_code == 200
        body = response.json()
        assert body["email"] == "surgeon@example.com"
        assert body["role"] == "surgeon"

        app.dependency_overrides.clear()

    async def test_refresh_returns_new_tokens_for_active_user(self, app, async_client):
        user = make_user()
        session = make_session([MockExecuteResult(scalar_one_or_none=user)])
        app.dependency_overrides[get_db_session] = make_db_override(session)

        refresh_token = create_refresh_token(str(user.id))
        response = await async_client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["accessToken"]
        assert body["refreshToken"]

        app.dependency_overrides.clear()

    async def test_change_password_requires_correct_current_password(self, app, async_client):
        user = make_user(password="CorrectPassword1!")
        session = make_session([MockExecuteResult(scalar_one_or_none=user)])
        app.dependency_overrides[get_db_session] = make_db_override(session)
        app.dependency_overrides[get_current_user] = make_user_override(user_id=str(user.id))

        response = await async_client.post(
            "/api/v1/auth/change-password",
            json={
                "current_password": "WrongPassword1!",
                "new_password": "NewPassword1!",
            },
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "Current password is incorrect"

        app.dependency_overrides.clear()
