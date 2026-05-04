from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.asyncio]


class _FakeConnection:
    async def execute(self, _statement):
        return 1


class _FakeConnectContext:
    async def __aenter__(self):
        return _FakeConnection()

    async def __aexit__(self, exc_type, exc, tb):
        return None


class _FakeEngine:
    def connect(self):
        return _FakeConnectContext()


class _FakeInspect:
    def stats(self):
        return {"worker-1": {"pool": "solo"}}


class _FakeControl:
    def inspect(self, timeout: float = 2.0):
        return _FakeInspect()


class _FakeCeleryApp:
    control = _FakeControl()


class _FakeTorch:
    class cuda:
        @staticmethod
        def is_available():
            return False


async def test_health_endpoint_reports_components(app, async_client, monkeypatch):
    from app.api.v1.endpoints import health as health_endpoint

    app.state.model_registry = object()
    monkeypatch.setattr(health_endpoint, "get_engine", lambda: _FakeEngine())
    monkeypatch.setitem(sys.modules, "app.workers.celery_app", SimpleNamespace(celery_app=_FakeCeleryApp()))
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())

    response = await async_client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"healthy", "degraded"}
    component_names = {component["name"] for component in body["components"]}
    assert {"postgresql", "model_registry", "celery"}.issubset(component_names)


async def test_readiness_and_liveness_probes(async_client, monkeypatch):
    from app.api.v1.endpoints import health as health_endpoint

    monkeypatch.setattr(health_endpoint, "get_engine", lambda: _FakeEngine())

    readiness = await async_client.get("/api/v1/health/ready")
    liveness = await async_client.get("/api/v1/health/live")

    assert readiness.status_code == 200
    assert readiness.json() == {"status": "ready"}
    assert liveness.status_code == 200
    assert liveness.json()["status"] == "alive"
