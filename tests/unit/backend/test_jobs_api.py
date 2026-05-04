from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.asyncio]


class _FakeBackend:
    def get_task_meta(self, _job_id):
        return {
            "worker": "worker-1",
            "retries": 1,
            "kwargs": {"case_id": "case-001"},
            "date_submitted": datetime.now(timezone.utc).isoformat(),
            "date_start": datetime.now(timezone.utc).isoformat(),
            "date_done": datetime.now(timezone.utc).isoformat(),
        }


class _FakeAsyncResult:
    def __init__(self, job_id: str, app=None):
        self.id = job_id
        self.state = "PROGRESS"
        self.info = {"progress": 55, "step": "Generating meshes", "case_id": "case-001"}
        self.result = None
        self.backend = _FakeBackend()
        self.name = "app.workers.tasks.run_segmentation_pipeline"


async def test_job_status_endpoint_returns_normalized_progress(async_client, monkeypatch):
    from app.api.v1.endpoints import jobs as jobs_endpoint

    monkeypatch.setattr(jobs_endpoint, "AsyncResult", _FakeAsyncResult)

    response = await async_client.get("/api/v1/jobs/job-123")

    assert response.status_code == 200
    body = response.json()
    assert body["jobId"] == "job-123"
    assert body["status"] == "RUNNING"
    assert body["progress"]["percent"] == 55
    assert body["jobType"] == "SEGMENTATION"


async def test_cancel_job_endpoint_revokes_non_terminal_jobs(async_client, monkeypatch):
    from app.api.v1.endpoints import jobs as jobs_endpoint

    monkeypatch.setattr(jobs_endpoint, "AsyncResult", _FakeAsyncResult)
    revoke_calls: list[dict] = []
    fake_celery = SimpleNamespace(
        control=SimpleNamespace(
            revoke=lambda job_id, terminate, signal: revoke_calls.append(
                {"job_id": job_id, "terminate": terminate, "signal": signal}
            )
        )
    )
    monkeypatch.setattr(jobs_endpoint, "celery_app", fake_celery)

    response = await async_client.post("/api/v1/jobs/job-123/cancel")

    assert response.status_code == 200
    assert response.json()["cancelled"] is True
    assert revoke_calls == [{"job_id": "job-123", "terminate": False, "signal": "SIGTERM"}]
