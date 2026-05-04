"""
Facial Align FastAPI application factory.
Configures lifespan, middleware, exception handlers, and routers.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.exceptions import FacialAlignError
from app.core.logging import configure_logging, get_logger, set_request_context
from app.db.database import create_db_engine, dispose_db_engine

settings = get_settings()
logger = get_logger(__name__)


# ─── Lifespan ─────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler.
    Handles startup (DB engine, model registry) and shutdown (cleanup).
    """
    # ── Startup ──
    configure_logging(
        level=settings.log_level,
        json_output=settings.log_json,
        environment=settings.environment,
    )
    logger.info(
        "application_starting",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )

    # Initialize async DB engine
    await create_db_engine()
    logger.info("database_engine_created", db_host=settings.db.host, db_name=settings.db.name)

    # Initialize model registry (lazy-loads; actual weights loaded on first inference)
    try:
        from app.services.segmentation.segmentation_service import ModelRegistry
        registry = ModelRegistry(settings.model_registry)
        app.state.model_registry = registry
        logger.info("model_registry_initialized", registry_path=str(settings.model_registry.registry_path))
    except Exception as exc:
        logger.warning("model_registry_init_failed", error=str(exc))
        app.state.model_registry = None

    logger.info("application_started")
    yield

    # ── Shutdown ──
    logger.info("application_shutting_down")
    await dispose_db_engine()
    logger.info("application_shutdown_complete")


# ─── App factory ──────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Facial Align API",
        description=(
            "AI-native craniofacial surgical planning platform. "
            "Provides DICOM ingestion, bone segmentation, fracture reduction planning, "
            "and occlusal analysis via ML-powered services."
        ),
        version=settings.app_version,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url="/redoc" if settings.docs_enabled else None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
        lifespan=lifespan,
    )

    _register_middleware(app)
    _register_exception_handlers(app)
    _register_routers(app)

    return app


def _register_middleware(app: FastAPI) -> None:
    """Register all application middleware."""

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.security.allowed_origins,
        allow_credentials=settings.security.allow_credentials,
        allow_methods=settings.security.allowed_methods,
        allow_headers=settings.security.allowed_headers,
    )

    # Request ID / correlation ID middleware
    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        request_id = str(uuid.uuid4())
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        request.state.correlation_id = correlation_id

        set_request_context(
            request_id=request_id,
            correlation_id=correlation_id,
        )

        start_time = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Correlation-ID"] = correlation_id

        logger.info(
            "http_request",
            method=request.method,
            path=str(request.url.path),
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
            client_host=request.client.host if request.client else None,
            request_id=request_id,
        )
        return response

    # Security headers middleware
    @app.middleware("http")
    async def security_headers_middleware(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


def _register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers."""

    @app.exception_handler(FacialAlignError)
    async def facialign_error_handler(request: Request, exc: FacialAlignError):
        logger.error(
            "domain_error",
            error_code=exc.error_code,
            message=exc.message,
            context=exc.context,
            path=str(request.url.path),
        )
        return JSONResponse(
            status_code=exc.http_status,
            content={
                "error": exc.error_code,
                "message": exc.message,
                "context": exc.context,
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        logger.warning(
            "request_validation_error",
            errors=exc.errors(),
            path=str(request.url.path),
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "error": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": exc.errors(),
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        logger.exception(
            "unhandled_error",
            error_type=type(exc).__name__,
            path=str(request.url.path),
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "request_id": getattr(request.state, "request_id", None),
            },
        )


def _register_routers(app: FastAPI) -> None:
    """Register all API routers."""
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    # Root redirect / health
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": settings.app_name,
            "version": settings.app_version,
            "status": "running",
            "docs": "/docs",
        }


# ─── Application instance ─────────────────────────────────────────────────────

app = create_app()
