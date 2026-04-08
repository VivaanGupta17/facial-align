"""
Celery application configuration for Facial Align background tasks.
"""

from __future__ import annotations

from celery import Celery
from celery.signals import task_failure, task_postrun, task_prerun, task_retry, worker_ready

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger

settings = get_settings()
logger = get_logger(__name__)

# ─── Celery app initialization ────────────────────────────────────────────────

celery_app = Celery(
    "facialign",
    broker=settings.celery.broker_url,
    backend=settings.celery.result_backend,
    include=[
        "app.workers.tasks",
    ],
)

celery_app.conf.update(
    # Serialization
    task_serializer=settings.celery.task_serializer,
    result_serializer=settings.celery.result_serializer,
    accept_content=settings.celery.accept_content,

    # Timeouts
    task_soft_time_limit=settings.celery.task_soft_time_limit,
    task_time_limit=settings.celery.task_time_limit,

    # Worker behavior
    worker_concurrency=settings.celery.worker_concurrency,
    worker_prefetch_multiplier=settings.celery.worker_prefetch_multiplier,
    task_always_eager=settings.celery.task_always_eager,

    # Retry configuration
    task_max_retries=3,
    task_default_retry_delay=60,  # 60 seconds between retries

    # Result expiration
    result_expires=86400 * 7,  # 7 days

    # Task routing — route ML-heavy tasks to GPU workers
    task_routes={
        "app.workers.tasks.run_segmentation_pipeline": {"queue": "gpu"},
        "app.workers.tasks.run_reduction_planning_pipeline": {"queue": "gpu"},
        "app.workers.tasks.run_dicom_ingestion_pipeline": {"queue": "default"},
        "app.workers.tasks.run_mesh_extraction_pipeline": {"queue": "default"},
        "app.workers.tasks.run_occlusion_analysis_pipeline": {"queue": "default"},
        "app.workers.tasks.run_reduction_refinement": {"queue": "gpu"},
    },

    # Queue definitions
    task_create_missing_queues=True,

    # Beat schedule (periodic tasks)
    beat_schedule={
        "cleanup-expired-uploads": {
            "task": "app.workers.tasks.cleanup_expired_uploads",
            "schedule": 3600.0,  # Every hour
        },
        "archive-old-tasks": {
            "task": "app.workers.tasks.archive_old_task_results",
            "schedule": 86400.0,  # Every day
        },
    },

    # Logging
    worker_log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    worker_task_log_format="%(asctime)s - %(task_name)s[%(task_id)s] - %(levelname)s - %(message)s",
)

# ─── Signals ──────────────────────────────────────────────────────────────────


@worker_ready.connect
def on_worker_ready(sender, **kwargs):
    """Called when a worker is ready to accept tasks."""
    configure_logging(level=settings.log_level, json_output=settings.log_json)
    logger.info("celery_worker_ready", hostname=sender.hostname)


@task_prerun.connect
def on_task_prerun(task_id, task, args, kwargs, **_):
    """Called before a task starts execution."""
    logger.info(
        "task_started",
        task_id=task_id,
        task_name=task.name,
    )


@task_postrun.connect
def on_task_postrun(task_id, task, args, kwargs, retval, state, **_):
    """Called after a task completes."""
    logger.info(
        "task_completed",
        task_id=task_id,
        task_name=task.name,
        state=state,
    )


@task_failure.connect
def on_task_failure(task_id, exception, args, kwargs, traceback, einfo, **_):
    """Called when a task raises an exception."""
    logger.error(
        "task_failed",
        task_id=task_id,
        error=str(exception),
        error_type=type(exception).__name__,
    )


@task_retry.connect
def on_task_retry(request, reason, einfo, **_):
    """Called when a task is being retried."""
    logger.warning(
        "task_retrying",
        task_id=request.id,
        reason=str(reason),
        retry_count=request.retries,
    )
