"""
Facial Align middleware package.

Exports all middleware classes and the ``setup_middleware`` factory function
that registers them on a FastAPI application in the correct order.

Middleware registration order
------------------------------
Starlette processes middleware in a LIFO stack (last registered = outermost
wrapper = first to run on inbound / last to run on outbound).  We register
in this order so execution is:

    RequestContext → Audit → RateLimiter → ErrorHandler → route handler

1. ErrorHandlerMiddleware  (registered first → outermost error boundary)
2. RateLimitMiddleware     (registered second → rate-checks before audit)
3. AuditMiddleware         (registered third → audit after rate, before ctx)
4. RequestContextMiddleware (registered last → innermost, runs first)

This ensures that:
- request_id / correlation_id are set *before* audit and rate-limit run.
- All unhandled errors from any layer are caught by ErrorHandlerMiddleware.
- Rate-limited 429 responses are still audited.

Usage
-----
In ``app/main.py`` (replace or augment the existing _register_middleware)::

    from app.middleware import setup_middleware
    from app.core.config import get_settings

    setup_middleware(app, settings=get_settings())

Conditional middleware
----------------------
- RateLimitMiddleware is disabled in the ``test`` environment to avoid
  flakiness in unit/integration tests.
- AuditMiddleware is disabled when ``security.audit_log_enabled`` is False.
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI

from app.core.config import AppSettings, get_settings
from app.core.logging import get_logger

from .audit import AuditEntry, AuditLevel, AuditMiddleware, AuditSink
from .error_handler import (
    ErrorHandlerMiddleware,
    facialign_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from .rate_limiter import RateLimitMiddleware
from .request_context import (
    RequestContext,
    RequestContextMiddleware,
    capture_tracing_headers,
    get_correlation_id,
    get_request_context,
    get_request_id,
    reset_request_context,
    restore_request_context,
    set_user_id,
)

logger = get_logger(__name__)

__all__ = [
    # Middleware classes
    "AuditMiddleware",
    "ErrorHandlerMiddleware",
    "RateLimitMiddleware",
    "RequestContextMiddleware",
    # Audit helpers
    "AuditEntry",
    "AuditLevel",
    "AuditSink",
    # Request context helpers
    "RequestContext",
    "capture_tracing_headers",
    "get_correlation_id",
    "get_request_context",
    "get_request_id",
    "reset_request_context",
    "restore_request_context",
    "set_user_id",
    # Exception handlers (for direct registration on FastAPI app)
    "facialign_exception_handler",
    "unhandled_exception_handler",
    "validation_exception_handler",
    # Setup function
    "setup_middleware",
]


def setup_middleware(
    app: FastAPI,
    *,
    settings: Optional[AppSettings] = None,
    audit_sink: Optional[AuditSink] = None,
) -> None:
    """
    Register all Facial Align middleware on *app* in the correct order.

    Args:
        app:         The FastAPI application instance.
        settings:    Application settings; defaults to ``get_settings()``.
        audit_sink:  Optional async callable for forwarding audit entries to
                     an external system (database, S3, SIEM).  When None,
                     entries are written only to the local audit log file.

    Environment-conditional behaviour:
        - ``test`` environment: RateLimitMiddleware is registered but
          disabled (enabled=False) so tests run without hitting limits.
        - ``audit_log_enabled=False``: AuditMiddleware is registered but
          disabled so code paths still exist without side effects.
    """
    if settings is None:
        settings = get_settings()

    is_test = settings.environment == "test"
    is_production = settings.is_production

    # ── 1. ErrorHandlerMiddleware (outermost) ──────────────────────────────
    app.add_middleware(
        ErrorHandlerMiddleware,
        is_production=is_production,
    )
    logger.info(
        "middleware_registered",
        name="ErrorHandlerMiddleware",
        is_production=is_production,
    )

    # ── 2. RateLimitMiddleware ─────────────────────────────────────────────
    redis_url: Optional[str] = None
    try:
        redis_url = settings.celery.broker_url  # Reuse Redis from Celery config
    except AttributeError:
        pass

    app.add_middleware(
        RateLimitMiddleware,
        redis_url=redis_url,
        enabled=not is_test,
    )
    logger.info(
        "middleware_registered",
        name="RateLimitMiddleware",
        enabled=not is_test,
        redis_url=redis_url,
    )

    # ── 3. AuditMiddleware ─────────────────────────────────────────────────
    audit_enabled = settings.security.audit_log_enabled
    app.add_middleware(
        AuditMiddleware,
        sink=audit_sink,
        enabled=audit_enabled,
    )
    logger.info(
        "middleware_registered",
        name="AuditMiddleware",
        enabled=audit_enabled,
        has_external_sink=audit_sink is not None,
    )

    # ── 4. RequestContextMiddleware (innermost / runs first) ───────────────
    app.add_middleware(
        RequestContextMiddleware,
        propagate_request_id=True,
        log_requests=not is_test,  # suppress per-request logs in test output
    )
    logger.info(
        "middleware_registered",
        name="RequestContextMiddleware",
        log_requests=not is_test,
    )

    logger.info(
        "all_middleware_registered",
        environment=settings.environment,
        middleware_order=[
            "RequestContextMiddleware",
            "AuditMiddleware",
            "RateLimitMiddleware",
            "ErrorHandlerMiddleware",
        ],
    )
