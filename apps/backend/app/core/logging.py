"""
Structured JSON logging with correlation IDs and request tracing.
Uses structlog for structured output compatible with log aggregation services.
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any, Optional

import structlog
from structlog.types import EventDict, WrappedLogger

# Context variable for correlation ID (propagated per-request)
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
user_id_var: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
case_id_var: ContextVar[Optional[str]] = ContextVar("case_id", default=None)


def get_correlation_id() -> str:
    """Get or generate a correlation ID for the current context."""
    cid = correlation_id_var.get()
    if not cid:
        cid = str(uuid.uuid4())
        correlation_id_var.set(cid)
    return cid


def set_correlation_id(cid: str) -> None:
    """Set the correlation ID for the current context."""
    correlation_id_var.set(cid)


def set_request_context(
    request_id: str,
    correlation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    case_id: Optional[str] = None,
) -> None:
    """Populate request context variables for structured logging."""
    request_id_var.set(request_id)
    if correlation_id:
        correlation_id_var.set(correlation_id)
    if user_id:
        user_id_var.set(user_id)
    if case_id:
        case_id_var.set(case_id)


# ─── Structlog processors ────────────────────────────────────────────────────


def add_correlation_id(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Inject correlation and request IDs from context vars."""
    cid = correlation_id_var.get()
    if cid:
        event_dict["correlation_id"] = cid
    rid = request_id_var.get()
    if rid:
        event_dict["request_id"] = rid
    uid = user_id_var.get()
    if uid:
        event_dict["user_id"] = uid
    caid = case_id_var.get()
    if caid:
        event_dict["case_id"] = caid
    return event_dict


def add_app_context(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Inject static application context fields."""
    event_dict.setdefault("service", "facialign-backend")
    return event_dict


def add_log_level(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Normalize the log level field name."""
    event_dict["level"] = method_name.upper()
    return event_dict


def drop_color_message_key(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Remove uvicorn's color_message key that pollutes JSON output."""
    event_dict.pop("color_message", None)
    return event_dict


def order_keys(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Reorder log keys for readability: timestamp, level, event first."""
    ordered = {}
    for key in ("timestamp", "level", "event", "service", "correlation_id", "request_id"):
        if key in event_dict:
            ordered[key] = event_dict.pop(key)
    ordered.update(event_dict)
    return ordered


# ─── Setup ───────────────────────────────────────────────────────────────────


def configure_logging(
    level: str = "INFO",
    json_output: bool = True,
    environment: str = "development",
) -> None:
    """
    Configure structlog and stdlib logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: If True, emit JSON; if False, use human-readable console format
        environment: Deployment environment (affects formatting choices)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        add_correlation_id,
        add_app_context,
        add_log_level,
        drop_color_message_key,
        structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
        structlog.stdlib.add_logger_name,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_output:
        shared_processors.append(order_keys)
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    # Silence noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.INFO if level == "DEBUG" else logging.WARNING
    )
    logging.getLogger("celery").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger bound to a module name."""
    return structlog.get_logger(name)


# ─── Performance logging utilities ───────────────────────────────────────────


class TimedOperation:
    """Context manager for logging operation duration."""

    def __init__(
        self,
        logger: structlog.stdlib.BoundLogger,
        operation: str,
        **extra_fields: Any,
    ) -> None:
        self._logger = logger
        self._operation = operation
        self._extra = extra_fields
        self._start: float = 0.0

    def __enter__(self) -> "TimedOperation":
        self._start = time.perf_counter()
        self._logger.debug("operation_started", operation=self._operation, **self._extra)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        duration_ms = (time.perf_counter() - self._start) * 1000
        if exc_type is None:
            self._logger.info(
                "operation_completed",
                operation=self._operation,
                duration_ms=round(duration_ms, 2),
                **self._extra,
            )
        else:
            self._logger.error(
                "operation_failed",
                operation=self._operation,
                duration_ms=round(duration_ms, 2),
                error=str(exc_val),
                **self._extra,
            )


class MLInferenceLogger:
    """Specialized logger for ML inference tracking."""

    def __init__(self, logger: structlog.stdlib.BoundLogger) -> None:
        self._logger = logger

    def log_inference_start(
        self, model_name: str, input_shape: tuple, device: str, **kwargs: Any
    ) -> float:
        self._logger.info(
            "inference_started",
            model_name=model_name,
            input_shape=list(input_shape),
            device=device,
            **kwargs,
        )
        return time.perf_counter()

    def log_inference_complete(
        self,
        model_name: str,
        start_time: float,
        output_summary: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        duration_ms = (time.perf_counter() - start_time) * 1000
        self._logger.info(
            "inference_completed",
            model_name=model_name,
            duration_ms=round(duration_ms, 2),
            **output_summary,
            **kwargs,
        )

    def log_inference_error(
        self, model_name: str, error: Exception, **kwargs: Any
    ) -> None:
        self._logger.error(
            "inference_failed",
            model_name=model_name,
            error=str(error),
            error_type=type(error).__name__,
            **kwargs,
        )
