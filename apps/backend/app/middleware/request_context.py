"""
Request context middleware for distributed tracing and request correlation.

Every inbound HTTP request receives:
  - A unique request_id (UUID v4) generated here or honoured from the
    X-Request-ID header sent by the client / upstream proxy.
  - A correlation_id used to tie together requests across service
    boundaries (backend → Celery → ML inference worker). Taken from
    X-Correlation-ID or generated fresh.
  - Timing measurement with the result emitted as X-Response-Time.

All context is stored in Python contextvars so it is available
throughout the async call stack without passing it explicitly.

Usage
-----
In application code::

    from app.middleware.request_context import get_request_context

    ctx = get_request_context()
    logger.info("doing_work", request_id=ctx.request_id, user_id=ctx.user_id)

In Celery tasks, pass correlation_id as a task argument and call
``set_request_context(...)`` at the start of the task to restore
the tracing chain.
"""

from __future__ import annotations

import time
import uuid
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from app.core.logging import (
    get_logger,
    set_request_context as _core_set_request_context,
)

logger = get_logger(__name__)

# ─── Context variables ────────────────────────────────────────────────────────

# These vars shadow the ones in core.logging so that middleware and
# application code share the same storage. They are distinct objects
# to allow the middleware to manage reset tokens.

_request_context_var: ContextVar[Optional["RequestContext"]] = ContextVar(
    "request_context", default=None
)


# ─── RequestContext dataclass ─────────────────────────────────────────────────


@dataclass
class RequestContext:
    """
    Immutable snapshot of per-request tracing metadata.

    Instances are stored in a ContextVar and retrieved via
    ``get_request_context()``.

    Fields
    ------
    request_id:
        Unique identifier for this specific HTTP request. Generated
        by the middleware unless the client provides X-Request-ID.
    correlation_id:
        Identifier that groups related requests across service
        boundaries (e.g., a single user action that triggers backend
        calls, Celery tasks, and ML inference). Clients may supply
        X-Correlation-ID; otherwise a new UUID is generated.
    user_id:
        Authenticated user's ID once resolved by the auth layer.
        Initially None; populated by auth middleware / dependency.
    started_at:
        UTC datetime when the request was received.
    path:
        URL path of the request.
    method:
        HTTP verb (GET, POST, …).
    client_ip:
        Best-effort client IP (respects X-Forwarded-For).
    """

    request_id: str
    correlation_id: str
    started_at: datetime
    path: str
    method: str
    client_ip: str
    user_id: Optional[str] = None
    # Internal: monotonic start time for precise duration measurement
    _started_mono: float = field(default_factory=time.monotonic, repr=False, compare=False)

    def elapsed_ms(self) -> float:
        """Return elapsed time since request start in milliseconds."""
        return round((time.monotonic() - self._started_mono) * 1000, 2)

    def to_log_dict(self) -> dict[str, Any]:
        """Return a dict suitable for structured logging."""
        return {
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "user_id": self.user_id,
            "started_at": self.started_at.isoformat(),
            "path": self.path,
            "method": self.method,
            "client_ip": self.client_ip,
        }


# ─── Public helpers ───────────────────────────────────────────────────────────


def get_request_context() -> Optional[RequestContext]:
    """
    Return the RequestContext for the current async task / coroutine.

    Returns None when called outside an active request (e.g., during
    startup or from a Celery worker that has not called
    restore_request_context).
    """
    return _request_context_var.get()


def get_request_id() -> Optional[str]:
    """Convenience accessor for the current request's ID."""
    ctx = _request_context_var.get()
    return ctx.request_id if ctx else None


def get_correlation_id() -> Optional[str]:
    """Convenience accessor for the current correlation ID."""
    ctx = _request_context_var.get()
    return ctx.correlation_id if ctx else None


def set_user_id(user_id: str) -> None:
    """
    Attach a user ID to the current request context.

    Call this from the auth dependency after validating the JWT so that
    downstream middleware (audit, logging) can reference the user
    without re-decoding the token.
    """
    ctx = _request_context_var.get()
    if ctx is not None:
        # dataclass is mutable; update in place
        ctx.user_id = user_id
        # Also propagate to core logging context vars
        from app.core.logging import user_id_var
        user_id_var.set(user_id)


def restore_request_context(
    request_id: str,
    correlation_id: str,
    user_id: Optional[str] = None,
    path: str = "",
    method: str = "",
    client_ip: str = "",
) -> Token:
    """
    Restore a RequestContext in a non-HTTP context (e.g., a Celery task).

    Pass the correlation_id and request_id that were captured at the
    HTTP layer so that distributed traces can be linked.

    Returns a contextvars.Token that can be used to reset the var when
    the task finishes (prevents context leakage between tasks sharing
    the same thread).

    Example::

        token = restore_request_context(
            request_id=task_meta["request_id"],
            correlation_id=task_meta["correlation_id"],
            user_id=task_meta.get("user_id"),
        )
        try:
            ... # task body
        finally:
            _request_context_var.reset(token)
    """
    ctx = RequestContext(
        request_id=request_id,
        correlation_id=correlation_id,
        started_at=datetime.now(timezone.utc),
        path=path,
        method=method,
        client_ip=client_ip,
        user_id=user_id,
    )
    token = _request_context_var.set(ctx)
    _core_set_request_context(
        request_id=request_id,
        correlation_id=correlation_id,
        user_id=user_id,
    )
    return token


def reset_request_context(token: Token) -> None:
    """Reset the request context variable using a previously obtained token."""
    _request_context_var.reset(token)


# ─── Internal helpers ─────────────────────────────────────────────────────────


def _extract_client_ip(request: Request) -> str:
    """
    Extract the real client IP, respecting reverse-proxy headers.

    Priority: X-Forwarded-For (first entry) > X-Real-IP > direct client.
    """
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    real_ip = request.headers.get("X-Real-IP", "")
    if real_ip:
        return real_ip.strip()
    if request.client:
        return request.client.host
    return "unknown"


def _generate_or_accept_id(header_value: Optional[str]) -> str:
    """
    Return the supplied header value if it looks like a valid non-empty
    string, otherwise generate a new UUID.
    """
    if header_value and len(header_value.strip()) > 0:
        # Sanitise: keep only printable ASCII, truncate at 128 chars
        sanitised = "".join(c for c in header_value.strip() if c.isprintable())[:128]
        if sanitised:
            return sanitised
    return str(uuid.uuid4())


# ─── Middleware ────────────────────────────────────────────────────────────────


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Request context middleware for distributed tracing.

    Actions performed on every request:
    1. Accept or generate X-Request-ID and X-Correlation-ID.
    2. Build a RequestContext and store it in a ContextVar.
    3. Push context into core.logging vars for structlog processors.
    4. Attach request_id and correlation_id to request.state for use
       by other middleware (audit, rate limiter).
    5. Measure total request duration.
    6. Emit X-Request-ID, X-Correlation-ID, and X-Response-Time
       response headers.
    7. Log the completed request at INFO level with timing data.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        propagate_request_id: bool = True,
        log_requests: bool = True,
    ) -> None:
        """
        Args:
            propagate_request_id: If True, honour an incoming X-Request-ID
                header from trusted clients. Set False to always generate
                a new ID (more secure in untrusted environments).
            log_requests: Emit an INFO log line for each completed request.
        """
        super().__init__(app)
        self._propagate_request_id = propagate_request_id
        self._log_requests = log_requests

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # ── 1. Resolve IDs ──────────────────────────────────────────
        incoming_request_id = request.headers.get("X-Request-ID")
        incoming_correlation_id = request.headers.get("X-Correlation-ID")

        request_id = (
            _generate_or_accept_id(incoming_request_id)
            if self._propagate_request_id
            else str(uuid.uuid4())
        )
        correlation_id = _generate_or_accept_id(incoming_correlation_id)
        client_ip = _extract_client_ip(request)

        # ── 2. Build and store RequestContext ───────────────────────
        ctx = RequestContext(
            request_id=request_id,
            correlation_id=correlation_id,
            started_at=datetime.now(timezone.utc),
            path=request.url.path,
            method=request.method,
            client_ip=client_ip,
        )
        context_token = _request_context_var.set(ctx)

        # ── 3. Populate core.logging context vars ───────────────────
        _core_set_request_context(
            request_id=request_id,
            correlation_id=correlation_id,
        )

        # ── 4. Expose on request.state ──────────────────────────────
        request.state.request_id = request_id
        request.state.correlation_id = correlation_id
        request.state.request_context = ctx

        try:
            response = await call_next(request)
        except Exception:
            # Let error_handler middleware deal with the exception;
            # we still need to clean up context vars.
            raise
        finally:
            # ── 7. Log completed request ────────────────────────────
            elapsed = ctx.elapsed_ms()
            if self._log_requests:
                status_code = getattr(response if "response" in dir() else None, "status_code", 0)  # type: ignore[name-defined]
                logger.info(
                    "http_request_completed",
                    request_id=request_id,
                    correlation_id=correlation_id,
                    method=request.method,
                    path=request.url.path,
                    status_code=status_code,
                    duration_ms=elapsed,
                    client_ip=client_ip,
                    user_id=ctx.user_id,
                )
            # Reset context var to avoid leaking across requests
            _request_context_var.reset(context_token)

        # ── 5 & 6. Attach response headers ──────────────────────────
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Response-Time"] = f"{elapsed}ms"

        return response


# ─── Celery integration helpers ───────────────────────────────────────────────


def capture_tracing_headers() -> dict[str, str]:
    """
    Capture current tracing context as a dict suitable for embedding in
    a Celery task's kwargs or headers.

    Call this in the HTTP handler before dispatching a task::

        task_meta = capture_tracing_headers()
        my_celery_task.delay(data=..., _tracing=task_meta)

    In the task, call ``restore_request_context(**task_meta)`` to
    reconstruct the tracing chain.
    """
    ctx = _request_context_var.get()
    if ctx is None:
        return {
            "request_id": str(uuid.uuid4()),
            "correlation_id": str(uuid.uuid4()),
            "user_id": "",
            "path": "",
            "method": "",
            "client_ip": "",
        }
    return {
        "request_id": ctx.request_id,
        "correlation_id": ctx.correlation_id,
        "user_id": ctx.user_id or "",
        "path": ctx.path,
        "method": ctx.method,
        "client_ip": ctx.client_ip,
    }
