"""
Global error handling middleware for the Facial Align API.

Converts all unhandled exceptions into structured JSON responses with
consistent shape. In production environments, stack traces are never
sent to clients but are always logged server-side at ERROR level for
incident investigation.

Error response schema
---------------------
::

    {
        "error_code":  "SNAKE_CASE_CODE",
        "message":     "Human-readable description",
        "details":     { ... } | null,   # extra context, may be null in prod
        "request_id":  "uuid" | null,
        "timestamp":   "ISO-8601 UTC"
    }

Exception → HTTP status mapping
---------------------------------
Domain exceptions from ``app.core.exceptions`` carry their own
``http_status`` attribute and are mapped directly.  Generic Python
exceptions are mapped conservatively to 500.

Health check bypass
-------------------
The /health, /healthz, /readiness, and /liveness paths are excluded
from wrapping so infrastructure probes always get a raw response.
"""

from __future__ import annotations

import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from fastapi import status
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from app.core.config import get_settings
from app.core.exceptions import (
    AuthorizationError,
    DicomError,
    FacialAlignError,
    InferenceError,
    ModelLoadError,
    NotFoundError,
    StorageError,
    TaskTimeoutError,
    ValidationError,
)
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# ─── Error code registry ──────────────────────────────────────────────────────

# Explicit mapping for exception types that need a status override or
# a specific error_code not embedded in the class definition.
# Evaluated in MRO order so subclasses must come before parents.
_EXCEPTION_MAP: list[tuple[type[Exception], int, str]] = [
    # Auth
    (AuthorizationError,           status.HTTP_403_FORBIDDEN,              "FORBIDDEN"),
    # Validation
    (RequestValidationError,       status.HTTP_422_UNPROCESSABLE_ENTITY,   "VALIDATION_ERROR"),
    (ValidationError,              status.HTTP_422_UNPROCESSABLE_ENTITY,   "VALIDATION_ERROR"),
    # DICOM
    (DicomError,                   status.HTTP_422_UNPROCESSABLE_ENTITY,   "DICOM_ERROR"),
    # Not found
    (NotFoundError,                status.HTTP_404_NOT_FOUND,              "NOT_FOUND"),
    # ML inference
    (ModelLoadError,               status.HTTP_503_SERVICE_UNAVAILABLE,    "MODEL_UNAVAILABLE"),
    (InferenceError,               status.HTTP_503_SERVICE_UNAVAILABLE,    "INFERENCE_ERROR"),
    # Storage
    (StorageError,                 status.HTTP_500_INTERNAL_SERVER_ERROR,  "STORAGE_ERROR"),
    # Task timeout
    (TaskTimeoutError,             status.HTTP_504_GATEWAY_TIMEOUT,        "TASK_TIMEOUT"),
    # Base domain error (must come after all subclasses)
    (FacialAlignError,             status.HTTP_500_INTERNAL_SERVER_ERROR,  "DOMAIN_ERROR"),
]


def _classify_exception(exc: Exception) -> tuple[int, str]:
    """
    Walk the exception map and return (http_status, error_code).

    Uses isinstance checks so subclasses are matched by their parent
    entry when no specific entry exists.
    """
    for exc_type, http_status_code, error_code in _EXCEPTION_MAP:
        if isinstance(exc, exc_type):
            # Prefer the class-level error_code when it exists and is
            # more specific than the map's generic fallback
            if isinstance(exc, FacialAlignError):
                return exc.http_status, exc.error_code
            return http_status_code, error_code
    return status.HTTP_500_INTERNAL_SERVER_ERROR, "INTERNAL_SERVER_ERROR"


# ─── Response builders ────────────────────────────────────────────────────────


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_error_response(
    *,
    error_code: str,
    message: str,
    http_status_code: int,
    request_id: Optional[str],
    details: Optional[Any] = None,
    is_production: bool,
) -> JSONResponse:
    """
    Construct the canonical error JSON response.

    In production, *details* is stripped when it might contain internal
    information (e.g., raw Python exception messages for generic errors).
    """
    body: dict[str, Any] = {
        "error_code": error_code,
        "message": message,
        "details": details if not is_production else _sanitise_details(details),
        "request_id": request_id,
        "timestamp": _utc_now_iso(),
    }
    return JSONResponse(status_code=http_status_code, content=body)


def _sanitise_details(details: Optional[Any]) -> Optional[Any]:
    """
    Strip potentially sensitive content from error details in production.

    Validation error details (list of field errors) are safe to return
    since they help clients fix their requests. Raw exception messages
    and stack traces are not.
    """
    if details is None:
        return None
    if isinstance(details, list):
        # Pydantic / FastAPI validation errors – safe to return
        return details
    # Any other dict-shaped detail: omit in production
    return None


# ─── Authentication error ─────────────────────────────────────────────────────

# AuthenticationError is not in the shared exceptions module yet;
# define a sentinel type here so the handler can catch it by name.
# When added to core.exceptions, simply import it instead.
class _AuthenticationError(Exception):
    """Sentinel for authentication failures (401)."""


# ─── Middleware ────────────────────────────────────────────────────────────────


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Global exception-to-JSON converter.

    Catches all exceptions that propagate past route handlers and
    converts them to structured JSON responses.

    Design notes
    ------------
    - Health check paths bypass this middleware entirely so that probes
      receive the fastest possible response.
    - Stack traces are always logged server-side via structlog regardless
      of environment.
    - In production, the *message* field of a generic Exception is
      replaced with a generic string to prevent accidental leakage of
      internal implementation details.
    - FacialAlignError subclasses expose their ``message`` and
      ``error_code`` fields even in production, since these are
      intentionally crafted for client consumption.
    """

    _HEALTH_PATHS: frozenset[str] = frozenset(
        {"/health", "/healthz", "/readiness", "/liveness"}
    )

    def __init__(self, app: ASGIApp, *, is_production: bool = False) -> None:
        super().__init__(app)
        self._is_production = is_production

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self._HEALTH_PATHS:
            return await call_next(request)

        request_id: Optional[str] = getattr(request.state, "request_id", None)

        try:
            return await call_next(request)
        except RequestValidationError as exc:
            return self._handle_validation_error(exc, request_id)
        except FacialAlignError as exc:
            return self._handle_domain_error(exc, request_id, request)
        except _AuthenticationError as exc:
            return self._handle_authentication_error(exc, request_id)
        except Exception as exc:  # noqa: BLE001
            return self._handle_unexpected_error(exc, request_id, request)

    # ── Handlers ──────────────────────────────────────────────────────

    def _handle_validation_error(
        self,
        exc: RequestValidationError,
        request_id: Optional[str],
    ) -> JSONResponse:
        errors = exc.errors()
        logger.warning(
            "request_validation_error",
            request_id=request_id,
            error_count=len(errors),
        )
        return _build_error_response(
            error_code="VALIDATION_ERROR",
            message="Request validation failed. Check the 'details' field for field-level errors.",
            http_status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            request_id=request_id,
            details=errors,
            is_production=self._is_production,
        )

    def _handle_domain_error(
        self,
        exc: FacialAlignError,
        request_id: Optional[str],
        request: Request,
    ) -> JSONResponse:
        http_code = exc.http_status
        log_level = "error" if http_code >= 500 else "warning"

        getattr(logger, log_level)(
            "domain_error",
            error_code=exc.error_code,
            http_status=http_code,
            path=request.url.path,
            method=request.method,
            request_id=request_id,
            cause=str(exc.cause) if exc.cause else None,
            exc_info=True,
        )

        details: Optional[Any] = exc.context if exc.context else None

        return _build_error_response(
            error_code=exc.error_code,
            message=exc.message,
            http_status_code=http_code,
            request_id=request_id,
            details=details,
            is_production=self._is_production,
        )

    def _handle_authentication_error(
        self,
        exc: _AuthenticationError,
        request_id: Optional[str],
    ) -> JSONResponse:
        logger.warning(
            "authentication_error",
            request_id=request_id,
            error=str(exc),
        )
        return _build_error_response(
            error_code="AUTHENTICATION_REQUIRED",
            message="Authentication is required to access this resource.",
            http_status_code=status.HTTP_401_UNAUTHORIZED,
            request_id=request_id,
            details=None,
            is_production=self._is_production,
        )

    def _handle_unexpected_error(
        self,
        exc: Exception,
        request_id: Optional[str],
        request: Request,
    ) -> JSONResponse:
        # Always log full traceback server-side
        logger.error(
            "unhandled_exception",
            error_type=type(exc).__name__,
            error=str(exc),
            path=request.url.path,
            method=request.method,
            request_id=request_id,
            traceback=traceback.format_exc(),
        )

        if self._is_production:
            message = "An unexpected error occurred. Our team has been notified."
            details = None
        else:
            message = f"{type(exc).__name__}: {exc}"
            details = {"traceback": traceback.format_exc().splitlines()}

        return _build_error_response(
            error_code="INTERNAL_SERVER_ERROR",
            message=message,
            http_status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            request_id=request_id,
            details=details,
            is_production=self._is_production,
        )


# ─── Standalone exception handler functions ───────────────────────────────────
# These can be registered directly on the FastAPI app via
# app.add_exception_handler() as an alternative to (or in conjunction
# with) the middleware approach.


async def facialign_exception_handler(
    request: Request, exc: FacialAlignError
) -> JSONResponse:
    """
    FastAPI exception_handler for FacialAlignError.

    Register with::

        app.add_exception_handler(FacialAlignError, facialign_exception_handler)
    """
    request_id: Optional[str] = getattr(request.state, "request_id", None)
    is_production = settings.is_production
    http_code = exc.http_status
    log_level = "error" if http_code >= 500 else "warning"

    getattr(logger, log_level)(
        "domain_error",
        error_code=exc.error_code,
        http_status=http_code,
        path=request.url.path,
        request_id=request_id,
        exc_info=True,
    )

    return _build_error_response(
        error_code=exc.error_code,
        message=exc.message,
        http_status_code=http_code,
        request_id=request_id,
        details=exc.context if exc.context else None,
        is_production=is_production,
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    FastAPI exception_handler for Pydantic RequestValidationError.

    Register with::

        app.add_exception_handler(RequestValidationError, validation_exception_handler)
    """
    request_id: Optional[str] = getattr(request.state, "request_id", None)
    errors = exc.errors()
    logger.warning(
        "request_validation_error",
        request_id=request_id,
        error_count=len(errors),
    )
    return _build_error_response(
        error_code="VALIDATION_ERROR",
        message="Request validation failed.",
        http_status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        request_id=request_id,
        details=errors,
        is_production=False,  # validation errors are always safe to return
    )


async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """
    FastAPI exception_handler for all other exceptions.

    Register with::

        app.add_exception_handler(Exception, unhandled_exception_handler)
    """
    request_id: Optional[str] = getattr(request.state, "request_id", None)
    is_production = settings.is_production
    logger.error(
        "unhandled_exception",
        error_type=type(exc).__name__,
        error=str(exc),
        path=request.url.path,
        request_id=request_id,
        traceback=traceback.format_exc(),
    )

    if is_production:
        message = "An unexpected error occurred."
        details = None
    else:
        message = f"{type(exc).__name__}: {exc}"
        details = {"traceback": traceback.format_exc().splitlines()}

    return _build_error_response(
        error_code="INTERNAL_SERVER_ERROR",
        message=message,
        http_status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        request_id=request_id,
        details=details,
        is_production=is_production,
    )
