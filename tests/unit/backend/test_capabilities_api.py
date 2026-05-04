from __future__ import annotations

import pytest

pytestmark = [pytest.mark.asyncio]


async def test_capabilities_endpoint_returns_runtime_snapshot(async_client):
    response = await async_client.get("/api/v1/capabilities")

    assert response.status_code == 200
    body = response.json()
    assert "generatedAt" in body
    assert "capabilities" in body
    assert isinstance(body["capabilities"], list)
    assert body["capabilities"], "Expected at least one capability entry"
    assert {"name", "status", "validationTier"}.issubset(body["capabilities"][0].keys())
