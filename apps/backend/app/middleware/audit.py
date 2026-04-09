"""
Clinical audit middleware for HIPAA-compliant request logging.

Logs all requests that modify patient data or surgical plans, capturing
who performed which action, on what resource, from where, and with what
result. This satisfies 45 CFR §164.312(b) – Audit Controls.

Audit entries are written to a dedicated JSON-lines file and optionally
forwarded to an async sink (database, S3, external SIEM).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional, Pattern

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# ─── Audit level classification ───────────────────────────────────────────────


class AuditLevel(str, Enum):
    """Classification of an API action's audit significance."""

    READ = "READ"
    MODIFY = "MODIFY"
    CREATE = "CREATE"
    DELETE = "DELETE"
    EXPORT = "EXPORT"
    ADMIN = "ADMIN"


# ─── PHI field masking ────────────────────────────────────────────────────────

# Field names that may contain Protected Health Information.
# Values are masked before being written to audit logs.
_PHI_FIELDS: frozenset[str] = frozenset(
    {
        "mrn",
        "medical_record_number",
        "dob",
        "date_of_birth",
        "ssn",
        "social_security_number",
        "name",
        "first_name",
        "last_name",
        "patient_name",
        "address",
        "phone",
        "email",
        "insurance_id",
        "npi",
    }
)


def _mask_phi(data: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively mask PHI field values in a dictionary.

    Values are replaced with a redaction marker so that the field
    presence is still auditable without exposing the actual PHI content.
    """
    masked: dict[str, Any] = {}
    for key, value in data.items():
        if key.lower() in _PHI_FIELDS:
            masked[key] = "[REDACTED]"
        elif isinstance(value, dict):
            masked[key] = _mask_phi(value)
        elif isinstance(value, list):
            masked[key] = [
                _mask_phi(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            masked[key] = value
    return masked


# ─── Path classification rules ────────────────────────────────────────────────

@dataclass
class _PathRule:
    """Associates a URL path pattern with an audit level."""

    pattern: Pattern[str]
    level: AuditLevel
    resource_type: str
    phi_touching: bool = False


# Rules evaluated in order; first match wins.
_PATH_RULES: list[_PathRule] = [
    # Patient-identifiable records – always PHI-touching
    _PathRule(
        pattern=re.compile(r"^/api/v1/patients"),
        level=AuditLevel.READ,
        resource_type="patient",
        phi_touching=True,
    ),
    # DICOM ingestion – PHI may be embedded in DICOM metadata
    _PathRule(
        pattern=re.compile(r"^/api/v1/dicom"),
        level=AuditLevel.CREATE,
        resource_type="dicom_study",
        phi_touching=True,
    ),
    # Surgical planning
    _PathRule(
        pattern=re.compile(r"^/api/v1/planning"),
        level=AuditLevel.MODIFY,
        resource_type="surgical_plan",
        phi_touching=False,
    ),
    # Segmentation results
    _PathRule(
        pattern=re.compile(r"^/api/v1/segmentation"),
        level=AuditLevel.CREATE,
        resource_type="segmentation",
        phi_touching=False,
    ),
    # 3-D viewer interactions
    _PathRule(
        pattern=re.compile(r"^/api/v1/viewer"),
        level=AuditLevel.READ,
        resource_type="3d_view",
        phi_touching=False,
    ),
    # Cases – may contain patient linkage
    _PathRule(
        pattern=re.compile(r"^/api/v1/cases"),
        level=AuditLevel.MODIFY,
        resource_type="case",
        phi_touching=True,
    ),
    # Export / report generation
    _PathRule(
        pattern=re.compile(r"^/api/v1/export"),
        level=AuditLevel.EXPORT,
        resource_type="export",
        phi_touching=True,
    ),
    # Admin operations
    _PathRule(
        pattern=re.compile(r"^/api/v1/admin"),
        level=AuditLevel.ADMIN,
        resource_type="admin_resource",
        phi_touching=False,
    ),
    # Authentication events
    _PathRule(
        pattern=re.compile(r"^/api/v1/auth"),
        level=AuditLevel.ADMIN,
        resource_type="auth",
        phi_touching=False,
    ),
]

# HTTP methods that trigger MODIFY / CREATE / DELETE re-classification
_WRITE_METHODS: frozenset[str] = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _classify_request(
    method: str, path: str
) -> tuple[AuditLevel, str, bool]:
    """
    Determine the audit level, resource type, and PHI-touching flag for
    a given HTTP method + path.

    Returns:
        (AuditLevel, resource_type, phi_touching)
    """
    for rule in _PATH_RULES:
        if rule.pattern.match(path):
            level = rule.level
            # Upgrade to appropriate write-level when method demands it
            if method in _WRITE_METHODS and level == AuditLevel.READ:
                level = AuditLevel.MODIFY
            if method == "DELETE":
                level = AuditLevel.DELETE
            return level, rule.resource_type, rule.phi_touching

    return AuditLevel.READ, "unknown", False


def _extract_resource_id(path: str) -> Optional[str]:
    """
    Extract a UUID-shaped resource ID from a URL path segment.

    Returns the last UUID-like segment found, or None.
    """
    uuid_pattern = re.compile(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        re.IGNORECASE,
    )
    matches = uuid_pattern.findall(path)
    return matches[-1] if matches else None


# ─── AuditEntry dataclass ─────────────────────────────────────────────────────


@dataclass
class AuditEntry:
    """
    A single HIPAA audit log record.

    All fields are plain Python types so the entry can be serialised to
    JSON without further transformation.
    """

    event_id: str
    timestamp: str                        # ISO-8601 UTC
    user_id: Optional[str]
    action: str                           # AuditLevel value
    resource_type: str
    resource_id: Optional[str]
    ip_address: str
    method: str
    path: str
    query_string: str
    response_status: int
    duration_ms: float
    request_id: Optional[str]
    correlation_id: Optional[str]
    phi_touching: bool
    audit_level: str                      # AuditLevel name
    service: str = "facialign-backend"
    hipaa_category: str = "access_control"
    additional_context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


# ─── Async sink protocol ──────────────────────────────────────────────────────

AuditSink = Callable[[AuditEntry], Coroutine[Any, Any, None]]
"""
Type alias for an async audit sink.

A sink is any async callable that accepts an AuditEntry and handles
forwarding it to an external system (database, S3, SIEM, etc.).
"""


async def _noop_sink(entry: AuditEntry) -> None:  # pragma: no cover
    """Default no-op sink used when no external forwarding is configured."""
    pass


# ─── File-based audit logger ──────────────────────────────────────────────────


class _AuditFileWriter:
    """
    Thread-safe, append-only JSON-lines writer for audit events.

    Uses a dedicated stdlib logger so audit entries never mix with
    application logs and cannot be suppressed by log-level changes.
    """

    def __init__(self, log_path: Path) -> None:
        self._audit_logger = logging.getLogger("facialign.audit.file")
        self._audit_logger.propagate = False
        self._audit_logger.setLevel(logging.INFO)

        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._audit_logger.addHandler(handler)
        except (PermissionError, OSError) as exc:
            logger.warning(
                "audit_file_writer_fallback",
                path=str(log_path),
                error=str(exc),
                fallback="stderr",
            )
            self._audit_logger.addHandler(logging.StreamHandler())

    def write(self, entry: AuditEntry) -> None:
        """Append a JSON-serialised audit entry to the log file."""
        self._audit_logger.info(entry.to_json())


# ─── Middleware ────────────────────────────────────────────────────────────────


class AuditMiddleware(BaseHTTPMiddleware):
    """
    Clinical audit middleware.

    For every request that matches a tracked path, an AuditEntry is
    created after the response is produced and written to:
      1. A dedicated JSON-lines audit log file (configurable path).
      2. An optional async sink for external forwarding.

    PHI-touching endpoints have their path parameters and context
    fields scrubbed before logging to prevent PHI leakage in the
    audit file itself (the resource_id is retained as an opaque
    reference; actual PHI never enters the log).

    Health check endpoints (/health, /readiness, /liveness) are
    excluded from auditing to avoid noise.
    """

    _HEALTH_PATHS: frozenset[str] = frozenset(
        {"/health", "/healthz", "/readiness", "/liveness", "/"}
    )

    def __init__(
        self,
        app: ASGIApp,
        *,
        sink: Optional[AuditSink] = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(app)
        self._enabled = enabled
        self._sink: AuditSink = sink or _noop_sink
        self._writer = _AuditFileWriter(settings.security.audit_log_path)
        logger.info(
            "audit_middleware_initialised",
            enabled=enabled,
            log_path=str(settings.security.audit_log_path),
            has_external_sink=sink is not None,
        )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self._enabled or request.url.path in self._HEALTH_PATHS:
            return await call_next(request)

        import time

        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        await self._record(request, response, duration_ms)
        return response

    async def _record(
        self,
        request: Request,
        response: Response,
        duration_ms: float,
    ) -> None:
        """Build and dispatch an AuditEntry for the completed request."""
        path = request.url.path
        method = request.method
        level, resource_type, phi_touching = _classify_request(method, path)
        resource_id = _extract_resource_id(path)

        user_id: Optional[str] = getattr(request.state, "user_id", None)
        request_id: Optional[str] = getattr(request.state, "request_id", None)
        correlation_id: Optional[str] = getattr(request.state, "correlation_id", None)

        ip_address = (
            request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
            or (request.client.host if request.client else "unknown")
        )

        context: dict[str, Any] = {}
        if phi_touching:
            # Keep only non-PHI metadata; mask any PHI-adjacent fields
            context = _mask_phi({"path": path, "method": method})
        else:
            context = {"path": path, "method": method}

        entry = AuditEntry(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_id=user_id,
            action=level.value,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=ip_address,
            method=method,
            path=path if not phi_touching else _redact_path_phi(path),
            query_string=str(request.url.query) if not phi_touching else "",
            response_status=response.status_code,
            duration_ms=duration_ms,
            request_id=request_id,
            correlation_id=correlation_id,
            phi_touching=phi_touching,
            audit_level=level.name,
            additional_context=context,
        )

        # Synchronous file write (fast, append-only)
        self._write_entry(entry)

        # Fire-and-forget async sink – do not block the response
        asyncio.ensure_future(self._forward_to_sink(entry))

    def _write_entry(self, entry: AuditEntry) -> None:
        try:
            self._writer.write(entry)
        except Exception as exc:  # pragma: no cover
            logger.error(
                "audit_write_failed",
                error=str(exc),
                event_id=entry.event_id,
            )

    async def _forward_to_sink(self, entry: AuditEntry) -> None:
        try:
            await self._sink(entry)
        except Exception as exc:  # pragma: no cover
            logger.error(
                "audit_sink_failed",
                error=str(exc),
                event_id=entry.event_id,
            )


def _redact_path_phi(path: str) -> str:
    """
    Replace UUID segments in a PHI-touching path with a stable redaction
    marker so that path structure is preserved without exposing raw IDs
    that could be cross-referenced with PHI databases.
    """
    return re.sub(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        "[ID]",
        path,
        flags=re.IGNORECASE,
    )


# ─── Convenience sink factories ───────────────────────────────────────────────


def make_database_sink(
    write_fn: Callable[[dict[str, Any]], Coroutine[Any, Any, None]]
) -> AuditSink:
    """
    Wrap an async database write function as an audit sink.

    Args:
        write_fn: Async callable accepting a dict audit record and
                  persisting it to the database.

    Returns:
        An AuditSink that calls write_fn with the serialised entry.
    """
    async def _sink(entry: AuditEntry) -> None:
        await write_fn(entry.to_dict())

    return _sink


def make_s3_sink(
    upload_fn: Callable[[str, bytes], Coroutine[Any, Any, None]]
) -> AuditSink:
    """
    Wrap an async S3 upload function as an audit sink.

    Args:
        upload_fn: Async callable accepting (key, data) and uploading
                   the audit record to S3.

    Returns:
        An AuditSink that streams each entry to S3 as a JSON blob.
    """
    async def _sink(entry: AuditEntry) -> None:
        key = (
            f"audit/{entry.timestamp[:10]}/{entry.event_id}.json"
        )
        await upload_fn(key, entry.to_json().encode("utf-8"))

    return _sink
