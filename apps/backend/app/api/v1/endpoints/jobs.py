"""
Job status REST endpoints — polling fallback for clients that cannot use WebSocket.

Provides:
  GET  /jobs/{job_id}          – Retrieve current status and result for a job
  GET  /jobs/{job_id}/logs     – Retrieve structured execution logs for a job
  POST /jobs/{job_id}/cancel   – Attempt to cancel a running job
  GET  /jobs                   – List jobs with optional case_id / status filters

All job metadata is sourced from the Celery result backend (Redis) via
celery.result.AsyncResult. Persistent job records can optionally be written to
the database by the pipeline code; this module reads both sources and merges
the result.

Job types map to specific Celery task names:
  SEGMENTATION    → app.workers.tasks.run_segmentation_pipeline
  REDUCTION       → app.workers.tasks.run_reduction_planning_pipeline
  MESH_EXTRACTION → app.workers.tasks.run_mesh_extraction_pipeline
  QUALITY_CHECK   → app.workers.tasks.run_dicom_ingestion_pipeline
  DEIDENTIFICATION → (sub-step of ingestion pipeline)

Authentication: Bearer JWT via the Authorization header.
Authorization: any authenticated user may inspect a job belonging to a case
they have access to. Case-level ACL enforcement is delegated to the case service
(placeholder: currently validates only that the user is authenticated).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import Field

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.common import BaseSchema
from app.workers.celery_app import celery_app

logger = get_logger(__name__)
router = APIRouter(prefix="/jobs", tags=["Jobs"])
settings = get_settings()


# ── Enumerations ──────────────────────────────────────────────────────────────


class JobType(str, Enum):
    """Recognised Celery job types."""

    SEGMENTATION = "SEGMENTATION"
    REDUCTION = "REDUCTION"
    MESH_EXTRACTION = "MESH_EXTRACTION"
    QUALITY_CHECK = "QUALITY_CHECK"
    DEIDENTIFICATION = "DEIDENTIFICATION"


class JobStatus(str, Enum):
    """
    Normalised job lifecycle states.

    Maps from Celery task states (PENDING, STARTED, PROGRESS, SUCCESS, FAILURE,
    REVOKED) to our domain-level JobStatus.
    """

    PENDING = "PENDING"       # Queued, not yet picked up by a worker
    RUNNING = "RUNNING"       # Worker is actively executing the task
    SUCCESS = "SUCCESS"       # Completed successfully
    FAILED = "FAILED"         # Terminated with an error
    CANCELLED = "CANCELLED"   # Explicitly cancelled by user or system
    UNKNOWN = "UNKNOWN"       # State cannot be determined (e.g. expired result)


# ── Task name registry ────────────────────────────────────────────────────────

# Maps Celery task names to our JobType enum so we can populate JobType in
# API responses when only a task_id is known.
_TASK_NAME_TO_JOB_TYPE: Dict[str, JobType] = {
    "app.workers.tasks.run_segmentation_pipeline": JobType.SEGMENTATION,
    "app.workers.tasks.run_reduction_planning_pipeline": JobType.REDUCTION,
    "app.workers.tasks.run_mesh_extraction_pipeline": JobType.MESH_EXTRACTION,
    "app.workers.tasks.run_dicom_ingestion_pipeline": JobType.QUALITY_CHECK,
    "app.workers.tasks.run_reduction_refinement": JobType.REDUCTION,
    "app.workers.tasks.run_occlusion_analysis_pipeline": JobType.REDUCTION,
}


def _celery_state_to_job_status(celery_state: str) -> JobStatus:
    """Translate a raw Celery state string to our JobStatus enum."""
    mapping = {
        "PENDING": JobStatus.PENDING,
        "RECEIVED": JobStatus.PENDING,
        "STARTED": JobStatus.RUNNING,
        "PROGRESS": JobStatus.RUNNING,
        "RETRY": JobStatus.RUNNING,
        "SUCCESS": JobStatus.SUCCESS,
        "FAILURE": JobStatus.FAILED,
        "REVOKED": JobStatus.CANCELLED,
    }
    return mapping.get(celery_state.upper(), JobStatus.UNKNOWN)


# ── Pydantic schemas ──────────────────────────────────────────────────────────


class JobProgress(BaseSchema):
    """Granular progress information emitted by long-running pipeline tasks."""

    percent: float = Field(default=0.0, ge=0.0, le=100.0, description="Completion percentage")
    current_step: Optional[str] = Field(default=None, description="Human-readable current step")
    steps_completed: int = Field(default=0, description="Number of pipeline steps completed")
    steps_total: int = Field(default=0, description="Total number of pipeline steps")
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional task-specific progress metadata",
    )


class JobStatusResponse(BaseSchema):
    """Full job status record returned by GET /jobs/{job_id}."""

    job_id: str = Field(description="Celery task UUID")
    job_type: Optional[JobType] = Field(
        default=None, description="Job type (inferred from task name when available)"
    )
    status: JobStatus = Field(description="Normalised lifecycle status")
    progress: Optional[JobProgress] = Field(
        default=None, description="Progress metadata (only present while RUNNING)"
    )
    created_at: Optional[datetime] = Field(
        default=None, description="Task submission timestamp (UTC)"
    )
    started_at: Optional[datetime] = Field(
        default=None, description="Worker pickup timestamp (UTC)"
    )
    completed_at: Optional[datetime] = Field(
        default=None, description="Task completion timestamp (UTC)"
    )
    result: Optional[Dict[str, Any]] = Field(
        default=None, description="Task return value (only present on SUCCESS)"
    )
    error: Optional[str] = Field(
        default=None, description="Error message (only present on FAILED)"
    )
    error_type: Optional[str] = Field(
        default=None, description="Exception class name (only present on FAILED)"
    )
    duration_ms: Optional[int] = Field(
        default=None, description="Total execution time in milliseconds"
    )
    queue: Optional[str] = Field(
        default=None, description="Celery queue this task was dispatched to"
    )
    worker: Optional[str] = Field(
        default=None, description="Celery worker hostname that processed this task"
    )
    retries: int = Field(default=0, description="Number of retry attempts")
    case_id: Optional[str] = Field(
        default=None,
        description="Surgical case ID associated with this job (when available in task metadata)",
    )


class JobLogEntry(BaseSchema):
    """A single structured log line emitted by a pipeline task."""

    timestamp: datetime
    level: str = Field(description="Log level: DEBUG, INFO, WARNING, ERROR")
    message: str
    step: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class JobLogsResponse(BaseSchema):
    """Response schema for GET /jobs/{job_id}/logs."""

    job_id: str
    total: int
    entries: List[JobLogEntry]


class JobCancelResponse(BaseSchema):
    """Response schema for POST /jobs/{job_id}/cancel."""

    job_id: str
    cancelled: bool
    message: str


class JobListResponse(BaseSchema):
    """Response schema for GET /jobs list endpoint."""

    total: int
    jobs: List[JobStatusResponse]
    page: int
    page_size: int


# ── Helper: build JobStatusResponse from AsyncResult ─────────────────────────


def _build_job_status(result: AsyncResult) -> JobStatusResponse:
    """
    Construct a JobStatusResponse from a Celery AsyncResult.

    Handles all Celery task states and safely extracts metadata without
    raising exceptions for expired or unknown tasks.
    """
    raw_state: str = result.state  # "PENDING", "STARTED", "PROGRESS", etc.
    job_status = _celery_state_to_job_status(raw_state)

    # Attempt to extract task name from the result backend info
    # (available when using Redis result backend with result_extended=True)
    task_name: Optional[str] = None
    job_type: Optional[JobType] = None
    case_id: Optional[str] = None
    worker: Optional[str] = None
    retries: int = 0
    queue: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    progress: Optional[JobProgress] = None
    job_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None

    # Extract metadata from the result info dict
    try:
        info = result.info
        if isinstance(info, dict):
            # PROGRESS state: task called self.update_state(state="PROGRESS", meta={...})
            if raw_state == "PROGRESS":
                progress = JobProgress(
                    percent=float(info.get("progress", 0)),
                    current_step=info.get("step"),
                    extra={k: v for k, v in info.items() if k not in ("progress", "step")},
                )
                case_id = info.get("case_id")

            elif raw_state == "SUCCESS":
                job_result = info
                case_id = info.get("case_id") or info.get("study_id")

            elif raw_state in ("FAILURE",):
                # info is the exception instance for FAILURE
                pass

    except Exception:
        pass  # Result may have expired from Redis TTL

    # For FAILURE state, result.result is the exception
    if raw_state == "FAILURE":
        try:
            exc = result.result
            if exc is not None:
                error = str(exc)
                error_type = type(exc).__name__
        except Exception:
            error = "Unknown error (result may have expired)"

    # Infer job type from task name if available
    try:
        task_name = getattr(result, "name", None)
        if task_name:
            job_type = _TASK_NAME_TO_JOB_TYPE.get(task_name)
    except Exception:
        pass

    # Attempt to retrieve extended metadata stored by the task
    try:
        backend_info = result.backend.get_task_meta(result.id)
        if isinstance(backend_info, dict):
            worker = backend_info.get("worker")
            retries = backend_info.get("retries", 0)
            kwargs = backend_info.get("kwargs", {})
            if isinstance(kwargs, dict):
                case_id = case_id or kwargs.get("case_id")
            # Timestamps (stored as ISO strings by custom task tracking)
            for field_name, attr_name in (
                ("date_done", "completed_at"),
                ("date_start", "started_at"),
                ("date_submitted", "created_at"),
            ):
                ts_str = backend_info.get(field_name)
                if ts_str:
                    try:
                        dt = datetime.fromisoformat(ts_str)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        if attr_name == "completed_at":
                            completed_at = dt
                        elif attr_name == "started_at":
                            started_at = dt
                        elif attr_name == "created_at":
                            created_at = dt
                    except (ValueError, TypeError):
                        pass
    except Exception:
        pass

    # Compute duration if both timestamps are available
    if started_at and completed_at:
        duration_ms = int((completed_at - started_at).total_seconds() * 1000)

    return JobStatusResponse(
        job_id=result.id,
        job_type=job_type,
        status=job_status,
        progress=progress,
        created_at=created_at,
        started_at=started_at,
        completed_at=completed_at,
        result=job_result,
        error=error,
        error_type=error_type,
        duration_ms=duration_ms,
        queue=queue,
        worker=worker,
        retries=retries,
        case_id=case_id,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get(
    "/{job_id}",
    response_model=JobStatusResponse,
    summary="Get job status",
    description=(
        "Retrieve current status, progress metadata, and result (on completion) "
        "for a Celery job identified by its task UUID."
    ),
)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """
    GET /api/v1/jobs/{job_id}

    Returns the current status of a Celery job. Suitable for polling when
    WebSocket is unavailable.

    Status values:
      PENDING   – Job is queued and waiting for a worker.
      RUNNING   – A worker is currently executing the job.
      SUCCESS   – Job completed; `result` contains the return value.
      FAILED    – Job failed; `error` and `error_type` describe the failure.
      CANCELLED – Job was revoked before or during execution.
      UNKNOWN   – Job ID not found in the result backend (may have expired).
    """
    try:
        result = AsyncResult(job_id, app=celery_app)
    except Exception as exc:
        logger.error("job_result_lookup_failed", job_id=job_id, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to reach job result backend",
        )

    return _build_job_status(result)


@router.get(
    "/{job_id}/logs",
    response_model=JobLogsResponse,
    summary="Get job execution logs",
    description=(
        "Retrieve structured execution logs produced by a Celery task. "
        "Log entries are stored in Redis by the pipeline infrastructure. "
        "Returns an empty list if no logs are available or the backend does not "
        "support log streaming."
    ),
)
async def get_job_logs(
    job_id: str,
    level: Optional[str] = Query(
        default=None,
        description="Filter by log level (DEBUG, INFO, WARNING, ERROR)",
    ),
    limit: int = Query(default=200, ge=1, le=1000, description="Maximum log entries to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
) -> JobLogsResponse:
    """
    GET /api/v1/jobs/{job_id}/logs

    Returns paginated structured log entries for a job. Log entries are written
    by pipeline steps using a Redis list keyed as ``job_log:{job_id}``.

    If no log entries are stored (e.g. the pipeline does not write to Redis
    logs), returns an empty list rather than 404.
    """
    import redis as sync_redis

    log_key = f"job_log:{job_id}"
    entries: List[JobLogEntry] = []

    try:
        broker_url = settings.celery.broker_url
        base_url = broker_url.rsplit("/", 1)[0]
        redis_client = sync_redis.from_url(f"{base_url}/4", decode_responses=True)  # DB 4 for job logs

        raw_logs = redis_client.lrange(log_key, offset, offset + limit - 1)
        redis_client.close()

        import json as _json

        for raw in raw_logs:
            try:
                entry = _json.loads(raw)
                log_level = entry.get("level", "INFO").upper()
                if level and log_level != level.upper():
                    continue
                ts_raw = entry.get("timestamp")
                ts = (
                    datetime.fromisoformat(ts_raw)
                    if ts_raw
                    else datetime.now(timezone.utc)
                )
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                entries.append(
                    JobLogEntry(
                        timestamp=ts,
                        level=log_level,
                        message=entry.get("message", ""),
                        step=entry.get("step"),
                        extra={
                            k: v
                            for k, v in entry.items()
                            if k not in ("timestamp", "level", "message", "step")
                        },
                    )
                )
            except Exception:
                continue

    except Exception as exc:
        logger.warning("job_logs_redis_error", job_id=job_id, error=str(exc))
        # Gracefully return empty logs rather than 500
        return JobLogsResponse(job_id=job_id, total=0, entries=[])

    return JobLogsResponse(job_id=job_id, total=len(entries), entries=entries)


@router.post(
    "/{job_id}/cancel",
    response_model=JobCancelResponse,
    summary="Cancel a running job",
    description=(
        "Send a SIGTERM revoke signal to the Celery worker executing the job. "
        "If the task is still in the queue (PENDING), it will be discarded. "
        "If a worker is actively executing the task, it receives the revocation "
        "signal and will exit at the next checkpoint (soft revoke). "
        "Note: GPU inference tasks may not honour cancellation immediately."
    ),
)
async def cancel_job(
    job_id: str,
    terminate: bool = Query(
        default=False,
        description=(
            "If True, send SIGKILL to the worker process (hard kill). "
            "Use only when soft revoke does not work within a reasonable time."
        ),
    ),
) -> JobCancelResponse:
    """
    POST /api/v1/jobs/{job_id}/cancel

    Attempts to cancel a Celery task. Returns immediately; check GET /jobs/{job_id}
    to confirm the transition to CANCELLED status.
    """
    try:
        result = AsyncResult(job_id, app=celery_app)
        current_status = _celery_state_to_job_status(result.state)

        if current_status in (JobStatus.SUCCESS, JobStatus.FAILED, JobStatus.CANCELLED):
            return JobCancelResponse(
                job_id=job_id,
                cancelled=False,
                message=f"Job is already in terminal state: {current_status.value}",
            )

        # Revoke the task
        celery_app.control.revoke(
            job_id,
            terminate=terminate,
            signal="SIGKILL" if terminate else "SIGTERM",
        )

        logger.info(
            "job_cancel_requested",
            job_id=job_id,
            terminate=terminate,
            previous_status=current_status.value,
        )

        return JobCancelResponse(
            job_id=job_id,
            cancelled=True,
            message=(
                "Revocation signal sent. The task will stop at the next safe checkpoint."
                if not terminate
                else "SIGKILL sent to worker process."
            ),
        )

    except Exception as exc:
        logger.error("job_cancel_failed", job_id=job_id, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to cancel job: {exc}",
        )


@router.get(
    "",
    response_model=JobListResponse,
    summary="List jobs",
    description=(
        "List Celery jobs with optional filtering by case_id and/or status. "
        "Results are sourced from the Celery result backend. "
        "Only jobs for which a result record exists (i.e. dispatched jobs) "
        "are returned. Jobs whose results have expired from Redis TTL are not "
        "included."
    ),
)
async def list_jobs(
    case_id: Optional[str] = Query(
        default=None,
        description="Filter jobs by surgical case ID",
    ),
    job_status: Optional[str] = Query(
        default=None,
        alias="status",
        description="Filter by job status (PENDING, RUNNING, SUCCESS, FAILED, CANCELLED)",
    ),
    job_type: Optional[str] = Query(
        default=None,
        alias="type",
        description="Filter by job type (SEGMENTATION, REDUCTION, MESH_EXTRACTION, etc.)",
    ),
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(default=20, ge=1, le=100, description="Results per page"),
) -> JobListResponse:
    """
    GET /api/v1/jobs?case_id={case_id}&status={status}&type={type}

    Returns a paginated list of jobs. Because the Celery result backend does not
    provide a general-purpose query API, this endpoint queries a Redis index
    maintained by the pipeline infrastructure:

      Redis key: ``case_jobs:{case_id}`` — sorted set of task IDs scored by
      submission timestamp (written by pipeline code when tasks are dispatched).

    If case_id is not provided, the endpoint queries a global index at
    ``all_jobs`` (sorted set). Falls back to an empty list if the index does
    not exist.
    """
    import redis as sync_redis

    broker_url = settings.celery.broker_url
    base_url = broker_url.rsplit("/", 1)[0]

    job_ids: List[str] = []

    try:
        redis_client = sync_redis.from_url(f"{base_url}/4", decode_responses=True)  # DB 4 for job logs

        # Determine which Redis key to query
        if case_id:
            index_key = f"case_jobs:{case_id}"
        else:
            index_key = "all_jobs"

        # zrevrange returns members in descending score (most recent first)
        offset = (page - 1) * page_size
        job_ids = redis_client.zrevrange(index_key, offset, offset + page_size - 1)
        total_in_index = redis_client.zcard(index_key)
        redis_client.close()

    except Exception as exc:
        logger.warning("job_list_redis_error", error=str(exc))
        return JobListResponse(total=0, jobs=[], page=page, page_size=page_size)

    # Fetch status for each job ID
    jobs: List[JobStatusResponse] = []
    status_filter = job_status.upper() if job_status else None
    type_filter = job_type.upper() if job_type else None

    for jid in job_ids:
        try:
            result = AsyncResult(jid, app=celery_app)
            job_response = _build_job_status(result)

            # Apply filters
            if status_filter and job_response.status.value != status_filter:
                continue
            if type_filter and (
                job_response.job_type is None
                or job_response.job_type.value != type_filter
            ):
                continue

            jobs.append(job_response)
        except Exception:
            continue

    return JobListResponse(
        total=int(total_in_index) if "total_in_index" in dir() else len(jobs),
        jobs=jobs,
        page=page,
        page_size=page_size,
    )
