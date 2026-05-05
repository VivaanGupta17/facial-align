"""
Tests for WebSocket endpoints at /api/v1/ws/{case_id}.

Uses the FastAPI TestClient for WebSocket testing. The WebSocket endpoint
authenticates based on APP_ENV=development/test by returning "dev_user",
so no JWT is needed in test mode.
"""

from __future__ import annotations

import json
import uuid

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def sync_client(app):
    """Synchronous test client for WebSocket testing."""
    return TestClient(app)


class TestWebSocketConnection:
    def test_connect_to_case_ws(self, sync_client):
        """WebSocket connection to /api/v1/ws/{case_id} succeeds."""
        case_id = str(uuid.uuid4())
        with sync_client.websocket_connect(f"/api/v1/ws/{case_id}") as ws:
            # Should receive a CONNECTED message
            data = ws.receive_json()
            assert data["type"] == "CONNECTED"
            assert data["subscription"] == f"case:{case_id}"

    def test_connect_to_job_ws(self, sync_client):
        """WebSocket connection to /api/v1/ws/jobs/{job_id} succeeds."""
        job_id = str(uuid.uuid4())
        with sync_client.websocket_connect(f"/api/v1/ws/jobs/{job_id}") as ws:
            data = ws.receive_json()
            assert data["type"] == "CONNECTED"
            assert data["subscription"] == f"job:{job_id}"

    def test_pong_response(self, sync_client):
        """Client can send PONG in response to server PING."""
        case_id = str(uuid.uuid4())
        with sync_client.websocket_connect(f"/api/v1/ws/{case_id}") as ws:
            # Consume CONNECTED message
            ws.receive_json()

            # Send a PONG message (server should accept it without error)
            ws.send_json({"type": "PONG"})

    def test_invalid_json_returns_error(self, sync_client):
        """Sending invalid JSON over WebSocket returns an ERROR message."""
        case_id = str(uuid.uuid4())
        with sync_client.websocket_connect(f"/api/v1/ws/{case_id}") as ws:
            ws.receive_json()  # consume CONNECTED
            ws.send_text("not valid json {{{")
            error = ws.receive_json()
            assert error["type"] == "ERROR"
            assert "Invalid JSON" in error["message"]

    def test_disconnection_handling(self, sync_client):
        """WebSocket cleans up after client disconnect."""
        case_id = str(uuid.uuid4())

        # Import ConnectionManager to verify cleanup
        from app.api.v1.endpoints.websocket import ConnectionManager

        with sync_client.websocket_connect(f"/api/v1/ws/{case_id}") as ws:
            ws.receive_json()  # consume CONNECTED
            # Verify connection is registered
            assert ConnectionManager.connection_count(case_id) >= 1

        # After context manager exits, connection should be cleaned up
        assert ConnectionManager.connection_count(case_id) == 0
