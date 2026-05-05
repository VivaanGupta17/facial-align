from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import redis as sync_redis

from app.core.security import get_current_user, require_surgeon
from app.db.database import get_db_session
from tests.unit.backend.api_test_utils import (
    MockExecuteResult,
    make_case,
    make_db_override,
    make_session,
    make_user_override,
)

pytestmark = [pytest.mark.asyncio]


class _FakeBackend:
    def __init__(self, case_id: str = "case-001"):
        self.case_id = case_id

    def get_task_meta(self, _job_id):
        return {
            "worker": "worker-1",
            "retries": 1,
            "kwargs": {"case_id": self.case_id},
            "date_submitted": datetime.now(timezone.utc).isoformat(),
            "date_start": datetime.now(timezone.utc).isoformat(),
            "date_done": datetime.now(timezone.utc).isoformat(),
        }


class _FakeAsyncResult:
    case_by_job_id = {
        "job-123": "case-001",
        "job-allowed": "case-001",
        "job-blocked": "case-002",
    }

    def __init__(self, job_id: str, app=None):
        case_id = self.case_by_job_id.get(job_id, "case-001")
        self.id = job_id
        self.state = "PROGRESS"
        self.info = {"progress": 55, "step": "Generating meshes", "case_id": case_id}
        self.result = None
        self.backend = _FakeBackend(case_id=case_id)
        self.name = "app.workers.tasks.run_segmentation_pipeline"


async def test_job_status_endpoint_returns_normalized_progress(app, async_client, monkeypatch):
    from app.api.v1.endpoints import jobs as jobs_endpoint

    monkeypatch.setattr(jobs_endpoint, "AsyncResult", _FakeAsyncResult)
    app.dependency_overrides[get_db_session] = make_db_override(make_session([]))
    app.dependency_overrides[get_current_user] = make_user_override(role="admin")

    response = await async_client.get("/api/v1/jobs/job-123")

    assert response.status_code == 200
    body = response.json()
    assert body["jobId"] == "job-123"
    assert body["status"] == "RUNNING"
    assert body["progress"]["percent"] == 55
    assert body["jobType"] == "SEGMENTATION"

    app.dependency_overrides.clear()


async def test_cancel_job_endpoint_revokes_non_terminal_jobs(app, async_client, monkeypatch):
    from app.api.v1.endpoints import jobs as jobs_endpoint

    monkeypatch.setattr(jobs_endpoint, "AsyncResult", _FakeAsyncResult)
    app.dependency_overrides[get_db_session] = make_db_override(make_session([]))
    app.dependency_overrides[require_surgeon] = make_user_override(role="admin")
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

    app.dependency_overrides.clear()


async def test_job_status_returns_403_for_out_of_scope_case(app, async_client, monkeypatch):
    from app.api.v1.endpoints import jobs as jobs_endpoint

    blocked_case = make_case(
        id="case-002",
        surgeon_id="another-surgeon",
        created_by="another-surgeon",
        institution_code="OTHER-INST",
    )
    session = make_session([MockExecuteResult(scalar_one_or_none=blocked_case)])
    monkeypatch.setattr(jobs_endpoint, "AsyncResult", _FakeAsyncResult)
    app.dependency_overrides[get_db_session] = make_db_override(session)
    app.dependency_overrides[get_current_user] = make_user_override(institution_code="DEMO-INST")

    response = await async_client.get("/api/v1/jobs/job-blocked")

    assert response.status_code == 403
    assert response.json()["detail"] == "You do not have access to this case"

    app.dependency_overrides.clear()


async def test_list_jobs_filters_out_of_scope_results(app, async_client, monkeypatch):
    from app.api.v1.endpoints import jobs as jobs_endpoint

    accessible_case = make_case(id="case-001", institution_code="DEMO-INST")
    blocked_case = make_case(
        id="case-002",
        surgeon_id="another-surgeon",
        created_by="another-surgeon",
        institution_code="OTHER-INST",
    )
    session = make_session(
        [
            MockExecuteResult(scalar_one_or_none=accessible_case),
            MockExecuteResult(scalar_one_or_none=blocked_case),
        ]
    )

    class _FakeRedisClient:
        def zrevrange(self, key, start, end):
            assert key == "all_jobs"
            assert start == 0
            assert end == 19
            return ["job-allowed", "job-blocked"]

        def zcard(self, key):
            assert key == "all_jobs"
            return 2

        def close(self):
            return None

    monkeypatch.setattr(jobs_endpoint, "AsyncResult", _FakeAsyncResult)
    monkeypatch.setattr(sync_redis, "from_url", lambda *args, **kwargs: _FakeRedisClient())
    app.dependency_overrides[get_db_session] = make_db_override(session)
    app.dependency_overrides[get_current_user] = make_user_override(institution_code="DEMO-INST")

    response = await async_client.get("/api/v1/jobs")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert [job["jobId"] for job in body["jobs"]] == ["job-allowed"]

    app.dependency_overrides.clear()
