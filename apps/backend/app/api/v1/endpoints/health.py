"""
Health check, GPU status, and model registry status endpoints.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Request
from sqlalchemy import text

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.database import get_engine
from app.schemas.common import ComponentHealth, HealthCheck

router = APIRouter(prefix="/health", tags=["health"])
settings = get_settings()
logger = get_logger(__name__)


@router.get("", response_model=HealthCheck, summary="Application health check")
async def health_check(request: Request) -> HealthCheck:
    """
    Check health of all system components.
    Returns 200 if healthy, 503 if degraded or unhealthy.
    """
    components: List[ComponentHealth] = []
    overall_ok = True

    # ── Database ──
    db_start = time.perf_counter()
    try:
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        db_latency = (time.perf_counter() - db_start) * 1000
        components.append(ComponentHealth(
            name="postgresql",
            status="healthy",
            latency_ms=round(db_latency, 2),
        ))
    except Exception as exc:
        overall_ok = False
        components.append(ComponentHealth(
            name="postgresql",
            status="unhealthy",
            message=str(exc),
        ))

    # ── GPU ──
    gpu_available = False
    gpu_devices: List[str] = []
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem_free, mem_total = torch.cuda.mem_get_info(i)
                gpu_devices.append(name)
                components.append(ComponentHealth(
                    name=f"gpu_{i}",
                    status="healthy",
                    message=(
                        f"{name} | VRAM: {mem_free / 1e9:.1f} GB free / "
                        f"{mem_total / 1e9:.1f} GB total"
                    ),
                ))
        else:
            components.append(ComponentHealth(
                name="gpu",
                status="degraded",
                message="CUDA not available — running on CPU",
            ))
    except ImportError:
        components.append(ComponentHealth(
            name="gpu",
            status="degraded",
            message="PyTorch not installed",
        ))

    # ── Model Registry ──
    try:
        registry = getattr(request.app.state, "model_registry", None)
        if registry is not None:
            components.append(ComponentHealth(
                name="model_registry",
                status="healthy",
                message=f"Registry path: {settings.model_registry.registry_path}",
            ))
        else:
            components.append(ComponentHealth(
                name="model_registry",
                status="degraded",
                message="Model registry not initialized",
            ))
    except Exception as exc:
        components.append(ComponentHealth(
            name="model_registry",
            status="unhealthy",
            message=str(exc),
        ))

    # ── Celery / Redis ──
    try:
        from app.workers.celery_app import celery_app
        inspect = celery_app.control.inspect(timeout=2.0)
        stats = inspect.stats()
        worker_count = len(stats) if stats else 0
        components.append(ComponentHealth(
            name="celery",
            status="healthy" if worker_count > 0 else "degraded",
            message=f"{worker_count} worker(s) active",
        ))
    except Exception as exc:
        components.append(ComponentHealth(
            name="celery",
            status="degraded",
            message=f"Celery unavailable: {exc}",
        ))

    # ── Overall status ──
    statuses = {c.status for c in components}
    if "unhealthy" in statuses:
        overall_status = "unhealthy"
    elif "degraded" in statuses:
        overall_status = "degraded"
    else:
        overall_status = "healthy"

    return HealthCheck(
        status=overall_status,
        version=settings.app_version,
        environment=settings.environment,
        timestamp=datetime.now(timezone.utc),
        components=components,
        gpu_available=gpu_available,
        gpu_devices=gpu_devices,
    )


@router.get("/ready", summary="Readiness probe (Kubernetes)")
async def readiness_probe() -> dict:
    """
    Kubernetes readiness probe.
    Returns 200 when the service is ready to accept traffic.
    """
    try:
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return {"status": "ready"}
    except Exception:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready: database unavailable",
        )


@router.get("/live", summary="Liveness probe (Kubernetes)")
async def liveness_probe() -> dict:
    """
    Kubernetes liveness probe.
    Returns 200 as long as the process is running.
    """
    return {"status": "alive", "timestamp": datetime.now(timezone.utc).isoformat()}
