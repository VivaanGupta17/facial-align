"""
Tests for authentication endpoints.

NOTE: Auth endpoints (POST /auth/register, /auth/login, etc.) do not exist yet.
These tests are written test-first and will pass once the auth module is
implemented at apps/backend/app/api/v1/endpoints/auth.py.

Until then, the tests are marked with xfail so they don't block CI.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

pytestmark = [
    pytest.mark.asyncio,
]

# ── Helpers ──────────────────────────────────────────────────────────────────


def _auth_endpoint_exists() -> bool:
    """Check whether the auth router has been registered."""
    try:
        from app.api.v1.endpoints import auth  # noqa: F401
        return True
    except (ImportError, ModuleNotFoundError):
        return False


skip_if_no_auth = pytest.mark.skipif(
    not _auth_endpoint_exists(),
    reason="Auth endpoints not yet implemented (apps/backend/app/api/v1/endpoints/auth.py)",
)


# ── POST /api/v1/auth/register ───────────────────────────────────────────────


@skip_if_no_auth
class TestRegister:
    async def test_register_success(self, async_client):
        """Registering a new user returns 201 with access token."""
        payload = {
            "email": "newuser@example.com",
            "password": "SecureP@ss123",
            "full_name": "Test User",
            "role": "surgeon",
        }
        resp = await async_client.post("/api/v1/auth/register", json=payload)
        assert resp.status_code == 201
        body = resp.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"

    async def test_register_duplicate_email_409(self, async_client):
        """Registering with an already-used email returns 409."""
        payload = {
            "email": "duplicate@example.com",
            "password": "SecureP@ss123",
            "full_name": "First User",
            "role": "surgeon",
        }
        # First registration succeeds
        resp1 = await async_client.post("/api/v1/auth/register", json=payload)
        assert resp1.status_code == 201

        # Second registration with same email should conflict
        resp2 = await async_client.post("/api/v1/auth/register", json=payload)
        assert resp2.status_code == 409


# ── POST /api/v1/auth/login ─────────────────────────────────────────────────


@skip_if_no_auth
class TestLogin:
    async def test_login_success(self, async_client):
        """Login with correct credentials returns tokens."""
        # Register first
        reg_payload = {
            "email": "loginuser@example.com",
            "password": "SecureP@ss123",
            "full_name": "Login User",
            "role": "surgeon",
        }
        await async_client.post("/api/v1/auth/register", json=reg_payload)

        # Login
        login_payload = {
            "email": "loginuser@example.com",
            "password": "SecureP@ss123",
        }
        resp = await async_client.post("/api/v1/auth/login", json=login_payload)
        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body
        assert "refresh_token" in body
        assert body["token_type"] == "bearer"

    async def test_login_wrong_password_401(self, async_client):
        """Login with wrong password returns 401."""
        reg_payload = {
            "email": "wrongpw@example.com",
            "password": "CorrectPassword1!",
            "full_name": "Wrong PW User",
            "role": "surgeon",
        }
        await async_client.post("/api/v1/auth/register", json=reg_payload)

        login_payload = {
            "email": "wrongpw@example.com",
            "password": "TotallyWrong!",
        }
        resp = await async_client.post("/api/v1/auth/login", json=login_payload)
        assert resp.status_code == 401

    async def test_login_nonexistent_user_401(self, async_client):
        """Login with a nonexistent email returns 401."""
        login_payload = {
            "email": "nobody@example.com",
            "password": "Whatever123!",
        }
        resp = await async_client.post("/api/v1/auth/login", json=login_payload)
        assert resp.status_code == 401


# ── GET /api/v1/auth/me ──────────────────────────────────────────────────────


@skip_if_no_auth
class TestMe:
    async def test_me_success(self, async_client):
        """GET /auth/me with valid token returns user info."""
        # Register and get token
        reg_payload = {
            "email": "meuser@example.com",
            "password": "SecureP@ss123",
            "full_name": "Me User",
            "role": "surgeon",
        }
        reg_resp = await async_client.post("/api/v1/auth/register", json=reg_payload)
        token = reg_resp.json()["access_token"]

        resp = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["email"] == "meuser@example.com"

    async def test_me_no_token_401(self, async_client):
        """GET /auth/me without token returns 401."""
        resp = await async_client.get("/api/v1/auth/me")
        assert resp.status_code == 401


# ── POST /api/v1/auth/refresh ────────────────────────────────────────────────


@skip_if_no_auth
class TestRefreshToken:
    async def test_refresh_success(self, async_client):
        """POST /auth/refresh with valid refresh token returns new access token."""
        # Register to get tokens
        reg_payload = {
            "email": "refreshuser@example.com",
            "password": "SecureP@ss123",
            "full_name": "Refresh User",
            "role": "surgeon",
        }
        reg_resp = await async_client.post("/api/v1/auth/register", json=reg_payload)

        # Login to get a refresh token
        login_payload = {
            "email": "refreshuser@example.com",
            "password": "SecureP@ss123",
        }
        login_resp = await async_client.post("/api/v1/auth/login", json=login_payload)
        refresh_token = login_resp.json()["refresh_token"]

        # Use refresh token
        resp = await async_client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body
