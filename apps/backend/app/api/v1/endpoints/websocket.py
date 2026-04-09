"""
WebSocket endpoint for real-time job status and case event notifications.

Provides two subscription paths:
  - /ws/{case_id}      Subscribe to all events for a specific surgical case
  - /ws/jobs/{job_id}  Subscribe to progress updates for a specific Celery job

Architecture:
  - ConnectionManager tracks active WebSocket connections by case_id.
  - A Redis pub/sub channel ("ws:case:{case_id}") acts as the cross-process
    message bus so that Celery workers (running in separate processes) can
    broadcast events to connected browser clients.
  - Each WebSocket connection spawns a coroutine that subscribes to its
    relevant Redis channel and forwards incoming messages to the client.
  - Heartbeat ping/pong every 30 seconds keeps connections alive through
    load balancers and proxies with aggressive idle-connection timeouts.

Message types (all serialised as JSON):
  SEGMENTATION_PROGRESS  – progress update from segmentation pipeline
  SEGMENTATION_COMPLETE  – segmentation pipeline finished successfully
  REDUCTION_PROGRESS     – progress update from reduction planning pipeline
  REDUCTION_COMPLETE     – reduction planning finished successfully
  MESH_READY             – a mesh file is available for download/rendering
  JOB_FAILED             – a Celery job failed with an error
  PLAN_UPDATED           – a surgeon edited a plan fragment
  CASE_STATUS_CHANGED    – the case's lifecycle status changed

Authentication:
  Clients pass a JWT bearer token either as the query parameter `token` or
  as the first JSON message after connecting. If neither is provided within
  AUTH_TIMEOUT_SECONDS, the connection is closed with code 4401.

Usage from Celery tasks:
    from app.api.v1.endpoints.websocket import ConnectionManager
    await ConnectionManager.broadcast_to_case(case_id, {
        "type": "SEGMENTATION_PROGRESS",
        "job_id": job_id,
        "progress_pct": 45.0,
        ...
    })
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import redis.asyncio as aioredis
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/ws", tags=["WebSocket"])
settings = get_settings()

# ── Constants ─────────────────────────────────────────────────────────────────

HEARTBEAT_INTERVAL_SECONDS: int = 30
AUTH_TIMEOUT_SECONDS: int = 10
REDIS_CHANNEL_PREFIX: str = "ws:case:"
REDIS_JOB_CHANNEL_PREFIX: str = "ws:job:"
MAX_MESSAGE_SIZE_BYTES: int = 64 * 1024  # 64 KB

# ── Message type enum ─────────────────────────────────────────────────────────


class WSMessageType(str, Enum):
    """All message types that can be emitted over the WebSocket channel."""

    # Segmentation pipeline events
    SEGMENTATION_PROGRESS = "SEGMENTATION_PROGRESS"
    SEGMENTATION_COMPLETE = "SEGMENTATION_COMPLETE"

    # Reduction planning events
    REDUCTION_PROGRESS = "REDUCTION_PROGRESS"
    REDUCTION_COMPLETE = "REDUCTION_COMPLETE"

    # Mesh export events
    MESH_READY = "MESH_READY"

    # Generic job failure
    JOB_FAILED = "JOB_FAILED"

    # Plan mutation events (surgeon edits)
    PLAN_UPDATED = "PLAN_UPDATED"

    # Case lifecycle events
    CASE_STATUS_CHANGED = "CASE_STATUS_CHANGED"

    # Infrastructure / heartbeat
    PING = "PING"
    PONG = "PONG"
    CONNECTED = "CONNECTED"
    ERROR = "ERROR"


# ── Pydantic message schemas ──────────────────────────────────────────────────


class WSBaseMessage(BaseModel):
    """Common envelope for all WebSocket messages."""

    type: WSMessageType
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class SegmentationProgressMessage(WSBaseMessage):
    type: WSMessageType = WSMessageType.SEGMENTATION_PROGRESS
    job_id: str
    case_id: str
    progress_pct: float = Field(ge=0.0, le=100.0)
    current_step: str
    structures_completed: List[str] = Field(default_factory=list)
    structures_total: int = 0


class SegmentationCompleteMessage(WSBaseMessage):
    type: WSMessageType = WSMessageType.SEGMENTATION_COMPLETE
    job_id: str
    case_id: str
    result_id: str
    confidence: Optional[float]
    structures: List[str] = Field(default_factory=list)
    inference_time_ms: Optional[int] = None
    total_pipeline_time_ms: Optional[int] = None


class ReductionProgressMessage(WSBaseMessage):
    type: WSMessageType = WSMessageType.REDUCTION_PROGRESS
    job_id: str
    case_id: str
    progress_pct: float = Field(ge=0.0, le=100.0)
    current_step: str
    fragments_completed: int = 0
    fragments_total: int = 0


class ReductionCompleteMessage(WSBaseMessage):
    type: WSMessageType = WSMessageType.REDUCTION_COMPLETE
    job_id: str
    case_id: str
    plan_id: str
    overall_grade: Optional[str]
    confidence_score: Optional[float]
    validation_passed: bool


class MeshReadyMessage(WSBaseMessage):
    type: WSMessageType = WSMessageType.MESH_READY
    job_id: str
    case_id: str
    structure: str
    resolution: str  # "high", "medium", "low"
    format: str      # "glb", "stl"
    url: str         # Signed URL or path
    file_size_bytes: Optional[int] = None


class JobFailedMessage(WSBaseMessage):
    type: WSMessageType = WSMessageType.JOB_FAILED
    job_id: str
    case_id: Optional[str] = None
    job_type: str
    error_code: str
    message: str
    retryable: bool = False
    retry_count: int = 0


class PlanUpdatedMessage(WSBaseMessage):
    type: WSMessageType = WSMessageType.PLAN_UPDATED
    plan_id: str
    case_id: str
    fragment_id: str
    edit_type: str  # "translate", "rotate", "reset"
    surgeon: str
    plan_version: int


class CaseStatusChangedMessage(WSBaseMessage):
    type: WSMessageType = WSMessageType.CASE_STATUS_CHANGED
    case_id: str
    old_status: str
    new_status: str
    changed_by: Optional[str] = None


class ConnectedMessage(WSBaseMessage):
    type: WSMessageType = WSMessageType.CONNECTED
    subscription: str  # "case:{id}" or "job:{id}"
    server_time: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class PingMessage(WSBaseMessage):
    type: WSMessageType = WSMessageType.PING


class PongMessage(WSBaseMessage):
    type: WSMessageType = WSMessageType.PONG


# ── Redis client factory ──────────────────────────────────────────────────────


def _get_redis() -> aioredis.Redis:
    """Return a new Redis client using the Celery broker URL as the base."""
    # Extract host/port from Celery broker URL.
    # e.g. redis://localhost:6379/0  → use DB 3 for WebSocket pub/sub
    broker_url = settings.celery.broker_url
    # Replace the DB path segment with DB index 3 (reserved for WS pub/sub)
    base_url = broker_url.rsplit("/", 1)[0]
    ws_url = f"{base_url}/3"
    return aioredis.from_url(ws_url, decode_responses=True)


# ── ConnectionManager ─────────────────────────────────────────────────────────


class ConnectionManager:
    """
    Manages all active WebSocket connections.

    Connections are organised by case_id so that any process can broadcast
    a message to all clients watching a given case via Redis pub/sub.

    Thread-safety: This class is used only within an async event loop and
    relies on asyncio's cooperative multitasking for safety.
    """

    # In-process registry: case_id → set of active WebSocket connections
    _connections: Dict[str, Set[WebSocket]] = {}

    # ── Connection lifecycle ──────────────────────────────────────────────────

    @classmethod
    def connect(cls, case_id: str, websocket: WebSocket) -> None:
        """Register a new connection for case_id."""
        if case_id not in cls._connections:
            cls._connections[case_id] = set()
        cls._connections[case_id].add(websocket)
        logger.info(
            "ws_client_connected",
            case_id=case_id,
            total_connections=cls.connection_count(case_id),
        )

    @classmethod
    def disconnect(cls, case_id: str, websocket: WebSocket) -> None:
        """Remove a connection from the registry."""
        if case_id in cls._connections:
            cls._connections[case_id].discard(websocket)
            if not cls._connections[case_id]:
                del cls._connections[case_id]
        logger.info(
            "ws_client_disconnected",
            case_id=case_id,
            total_connections=cls.connection_count(case_id),
        )

    @classmethod
    def connection_count(cls, case_id: str) -> int:
        """Return the number of active connections for a case_id."""
        return len(cls._connections.get(case_id, set()))

    @classmethod
    def total_connections(cls) -> int:
        """Return the total number of active connections across all cases."""
        return sum(len(conns) for conns in cls._connections.values())

    # ── In-process broadcast ──────────────────────────────────────────────────

    @classmethod
    async def send_to_case(
        cls, case_id: str, message: Dict[str, Any]
    ) -> int:
        """
        Send a message to all WebSocket connections for a case in THIS process.

        Returns the number of connections successfully sent to.
        """
        connections = list(cls._connections.get(case_id, set()))
        if not connections:
            return 0

        payload = json.dumps(message)
        sent = 0
        dead: List[WebSocket] = []

        for ws in connections:
            try:
                await ws.send_text(payload)
                sent += 1
            except Exception as exc:
                logger.warning(
                    "ws_send_failed",
                    case_id=case_id,
                    error=str(exc),
                )
                dead.append(ws)

        # Prune dead connections
        for ws in dead:
            cls.disconnect(case_id, ws)

        return sent

    # ── Cross-process broadcast via Redis pub/sub ─────────────────────────────

    @classmethod
    async def broadcast_to_case(
        cls, case_id: str, message: Dict[str, Any]
    ) -> None:
        """
        Publish a message to the Redis pub/sub channel for a case.

        All FastAPI workers subscribed to this channel will receive the message
        and forward it to their locally-connected WebSocket clients.

        This method is safe to call from Celery tasks (via asyncio.run or
        _run_async) because it creates its own Redis connection.
        """
        channel = f"{REDIS_CHANNEL_PREFIX}{case_id}"
        try:
            redis = _get_redis()
            await redis.publish(channel, json.dumps(message))
            await redis.aclose()
        except Exception as exc:
            logger.error(
                "ws_redis_publish_failed",
                case_id=case_id,
                channel=channel,
                error=str(exc),
            )

    @classmethod
    async def broadcast_to_job(
        cls, job_id: str, message: Dict[str, Any]
    ) -> None:
        """Publish a message to the Redis pub/sub channel for a specific job."""
        channel = f"{REDIS_JOB_CHANNEL_PREFIX}{job_id}"
        try:
            redis = _get_redis()
            await redis.publish(channel, json.dumps(message))
            await redis.aclose()
        except Exception as exc:
            logger.error(
                "ws_redis_publish_job_failed",
                job_id=job_id,
                channel=channel,
                error=str(exc),
            )


# ── Authentication helper ─────────────────────────────────────────────────────


async def _authenticate_websocket(
    websocket: WebSocket,
    token: Optional[str],
) -> Optional[str]:
    """
    Authenticate a WebSocket connection.

    Returns the user_id extracted from the JWT if authentication succeeds,
    or None if it fails (after sending a close frame).

    Token sources (checked in order):
    1. `token` query parameter
    2. First JSON message with {"type": "AUTH", "token": "..."}

    For development environments, authentication is bypassed and "dev_user"
    is returned.
    """
    if settings.environment in ("development", "test"):
        return "dev_user"

    if token:
        return _validate_jwt(token)

    # Wait for the client to send an AUTH message
    try:
        raw = await asyncio.wait_for(
            websocket.receive_text(), timeout=AUTH_TIMEOUT_SECONDS
        )
        data = json.loads(raw)
        if data.get("type") == "AUTH" and data.get("token"):
            return _validate_jwt(data["token"])
    except (asyncio.TimeoutError, json.JSONDecodeError, KeyError):
        pass

    await websocket.close(code=4401, reason="Authentication required")
    return None


def _validate_jwt(token: str) -> Optional[str]:
    """
    Validate a JWT bearer token and return the user_id claim.

    Returns None if the token is invalid or expired.
    In production, replace this stub with actual JWT verification using
    app.core.security or python-jose.
    """
    try:
        import jwt as pyjwt

        payload = pyjwt.decode(
            token,
            settings.security.secret_key,
            algorithms=[settings.security.algorithm],
        )
        return payload.get("sub")
    except Exception:
        return None


# ── Heartbeat coroutine ───────────────────────────────────────────────────────


async def _heartbeat(websocket: WebSocket) -> None:
    """Send a PING frame every HEARTBEAT_INTERVAL_SECONDS and wait for PONG."""
    ping = PingMessage()
    payload = ping.model_dump_json()

    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
        try:
            await websocket.send_text(payload)
        except Exception:
            # Connection is gone; let the main loop handle cleanup
            break


# ── Redis subscriber coroutine ────────────────────────────────────────────────


async def _redis_subscriber(
    websocket: WebSocket, channel: str
) -> None:
    """
    Subscribe to a Redis pub/sub channel and forward messages to the client.

    This coroutine runs concurrently with _heartbeat and the main receive loop
    for each WebSocket connection.
    """
    redis: Optional[aioredis.Redis] = None
    pubsub = None
    try:
        redis = _get_redis()
        pubsub = redis.pubsub()
        await pubsub.subscribe(channel)
        logger.debug("ws_redis_subscribed", channel=channel)

        async for raw_message in pubsub.listen():
            if raw_message["type"] != "message":
                continue
            data = raw_message.get("data", "")
            if not data:
                continue
            try:
                await websocket.send_text(data)
            except Exception:
                # Client disconnected; exit the subscriber loop
                break
    except Exception as exc:
        logger.error("ws_redis_subscriber_error", channel=channel, error=str(exc))
    finally:
        if pubsub:
            try:
                await pubsub.unsubscribe(channel)
                await pubsub.aclose()
            except Exception:
                pass
        if redis:
            try:
                await redis.aclose()
            except Exception:
                pass


# ── Endpoint: case-level subscription ────────────────────────────────────────


@router.websocket("/{case_id}")
async def websocket_case_endpoint(
    websocket: WebSocket,
    case_id: str,
    token: Optional[str] = Query(default=None, description="JWT bearer token"),
) -> None:
    """
    WebSocket endpoint — subscribe to all events for a specific surgical case.

    URL: /api/v1/ws/{case_id}?token=<jwt>

    Events emitted:
      SEGMENTATION_PROGRESS, SEGMENTATION_COMPLETE,
      REDUCTION_PROGRESS, REDUCTION_COMPLETE,
      MESH_READY, JOB_FAILED, PLAN_UPDATED, CASE_STATUS_CHANGED

    The client can send {"type": "PONG"} in response to PING frames to confirm
    liveness, or {"type": "AUTH", "token": "..."} as the first message if the
    token query parameter was not provided.
    """
    await websocket.accept()

    # Authenticate
    user_id = await _authenticate_websocket(websocket, token)
    if user_id is None:
        return

    # Register connection in the in-process registry
    ConnectionManager.connect(case_id, websocket)

    # Send CONNECTED confirmation
    connected_msg = ConnectedMessage(subscription=f"case:{case_id}")
    await websocket.send_text(connected_msg.model_dump_json())

    redis_channel = f"{REDIS_CHANNEL_PREFIX}{case_id}"

    # Spawn background tasks
    heartbeat_task = asyncio.create_task(_heartbeat(websocket))
    subscriber_task = asyncio.create_task(
        _redis_subscriber(websocket, redis_channel)
    )

    try:
        # Main loop: receive client messages (PONG responses, client-initiated events)
        while True:
            try:
                raw = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=HEARTBEAT_INTERVAL_SECONDS * 2,
                )
            except asyncio.TimeoutError:
                # No data in 2× heartbeat window → assume dead connection
                logger.warning("ws_receive_timeout", case_id=case_id, user_id=user_id)
                break

            # Parse and route client messages
            try:
                data = json.loads(raw)
                msg_type = data.get("type")
                if msg_type == WSMessageType.PONG:
                    # Acknowledge heartbeat response; no further action needed
                    pass
                else:
                    logger.debug(
                        "ws_received_client_message",
                        case_id=case_id,
                        user_id=user_id,
                        msg_type=msg_type,
                    )
            except json.JSONDecodeError:
                error_payload = json.dumps({
                    "type": WSMessageType.ERROR,
                    "message": "Invalid JSON payload",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                await websocket.send_text(error_payload)

    except WebSocketDisconnect as exc:
        logger.info(
            "ws_client_disconnect",
            case_id=case_id,
            user_id=user_id,
            code=exc.code,
        )
    except Exception as exc:
        logger.exception(
            "ws_unexpected_error",
            case_id=case_id,
            user_id=user_id,
            error=str(exc),
        )
    finally:
        heartbeat_task.cancel()
        subscriber_task.cancel()
        ConnectionManager.disconnect(case_id, websocket)

        # Wait for tasks to clean up
        await asyncio.gather(heartbeat_task, subscriber_task, return_exceptions=True)


# ── Endpoint: job-level subscription ─────────────────────────────────────────


@router.websocket("/jobs/{job_id}")
async def websocket_job_endpoint(
    websocket: WebSocket,
    job_id: str,
    token: Optional[str] = Query(default=None, description="JWT bearer token"),
) -> None:
    """
    WebSocket endpoint — subscribe to progress updates for a specific Celery job.

    URL: /api/v1/ws/jobs/{job_id}?token=<jwt>

    Events emitted:
      SEGMENTATION_PROGRESS, REDUCTION_PROGRESS, MESH_READY, JOB_FAILED,
      SEGMENTATION_COMPLETE, REDUCTION_COMPLETE

    This endpoint is useful when a client needs granular progress for a single
    job without subscribing to all events on the parent case.
    """
    await websocket.accept()

    # Authenticate
    user_id = await _authenticate_websocket(websocket, token)
    if user_id is None:
        return

    # Use job_id as a synthetic case scope for the in-process registry
    scope_key = f"__job__{job_id}"
    ConnectionManager.connect(scope_key, websocket)

    # Send CONNECTED confirmation
    connected_msg = ConnectedMessage(subscription=f"job:{job_id}")
    await websocket.send_text(connected_msg.model_dump_json())

    redis_channel = f"{REDIS_JOB_CHANNEL_PREFIX}{job_id}"

    heartbeat_task = asyncio.create_task(_heartbeat(websocket))
    subscriber_task = asyncio.create_task(
        _redis_subscriber(websocket, redis_channel)
    )

    try:
        while True:
            try:
                raw = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=HEARTBEAT_INTERVAL_SECONDS * 2,
                )
            except asyncio.TimeoutError:
                logger.warning("ws_job_receive_timeout", job_id=job_id, user_id=user_id)
                break

            try:
                data = json.loads(raw)
                msg_type = data.get("type")
                if msg_type == WSMessageType.PONG:
                    pass  # heartbeat acknowledgement
                else:
                    logger.debug(
                        "ws_job_received_client_message",
                        job_id=job_id,
                        user_id=user_id,
                        msg_type=msg_type,
                    )
            except json.JSONDecodeError:
                error_payload = json.dumps({
                    "type": WSMessageType.ERROR,
                    "message": "Invalid JSON payload",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                await websocket.send_text(error_payload)

    except WebSocketDisconnect as exc:
        logger.info(
            "ws_job_client_disconnect",
            job_id=job_id,
            user_id=user_id,
            code=exc.code,
        )
    except Exception as exc:
        logger.exception(
            "ws_job_unexpected_error",
            job_id=job_id,
            user_id=user_id,
            error=str(exc),
        )
    finally:
        heartbeat_task.cancel()
        subscriber_task.cancel()
        ConnectionManager.disconnect(scope_key, websocket)
        await asyncio.gather(heartbeat_task, subscriber_task, return_exceptions=True)


# ── Convenience broadcast helpers (for use by Celery tasks) ──────────────────


async def broadcast_segmentation_progress(
    case_id: str,
    job_id: str,
    progress_pct: float,
    current_step: str,
    structures_completed: Optional[List[str]] = None,
    structures_total: int = 0,
) -> None:
    """Publish a SEGMENTATION_PROGRESS event to the case and job channels."""
    message = SegmentationProgressMessage(
        job_id=job_id,
        case_id=case_id,
        progress_pct=progress_pct,
        current_step=current_step,
        structures_completed=structures_completed or [],
        structures_total=structures_total,
    ).model_dump()

    await asyncio.gather(
        ConnectionManager.broadcast_to_case(case_id, message),
        ConnectionManager.broadcast_to_job(job_id, message),
    )


async def broadcast_segmentation_complete(
    case_id: str,
    job_id: str,
    result_id: str,
    confidence: Optional[float],
    structures: List[str],
    inference_time_ms: Optional[int] = None,
    total_pipeline_time_ms: Optional[int] = None,
) -> None:
    """Publish a SEGMENTATION_COMPLETE event to the case and job channels."""
    message = SegmentationCompleteMessage(
        job_id=job_id,
        case_id=case_id,
        result_id=result_id,
        confidence=confidence,
        structures=structures,
        inference_time_ms=inference_time_ms,
        total_pipeline_time_ms=total_pipeline_time_ms,
    ).model_dump()

    await asyncio.gather(
        ConnectionManager.broadcast_to_case(case_id, message),
        ConnectionManager.broadcast_to_job(job_id, message),
    )


async def broadcast_reduction_progress(
    case_id: str,
    job_id: str,
    progress_pct: float,
    current_step: str,
    fragments_completed: int = 0,
    fragments_total: int = 0,
) -> None:
    """Publish a REDUCTION_PROGRESS event to the case and job channels."""
    message = ReductionProgressMessage(
        job_id=job_id,
        case_id=case_id,
        progress_pct=progress_pct,
        current_step=current_step,
        fragments_completed=fragments_completed,
        fragments_total=fragments_total,
    ).model_dump()

    await asyncio.gather(
        ConnectionManager.broadcast_to_case(case_id, message),
        ConnectionManager.broadcast_to_job(job_id, message),
    )


async def broadcast_reduction_complete(
    case_id: str,
    job_id: str,
    plan_id: str,
    overall_grade: Optional[str],
    confidence_score: Optional[float],
    validation_passed: bool,
) -> None:
    """Publish a REDUCTION_COMPLETE event to the case and job channels."""
    message = ReductionCompleteMessage(
        job_id=job_id,
        case_id=case_id,
        plan_id=plan_id,
        overall_grade=overall_grade,
        confidence_score=confidence_score,
        validation_passed=validation_passed,
    ).model_dump()

    await asyncio.gather(
        ConnectionManager.broadcast_to_case(case_id, message),
        ConnectionManager.broadcast_to_job(job_id, message),
    )


async def broadcast_mesh_ready(
    case_id: str,
    job_id: str,
    structure: str,
    resolution: str,
    fmt: str,
    url: str,
    file_size_bytes: Optional[int] = None,
) -> None:
    """Publish a MESH_READY event to the case and job channels."""
    message = MeshReadyMessage(
        job_id=job_id,
        case_id=case_id,
        structure=structure,
        resolution=resolution,
        format=fmt,
        url=url,
        file_size_bytes=file_size_bytes,
    ).model_dump()

    await asyncio.gather(
        ConnectionManager.broadcast_to_case(case_id, message),
        ConnectionManager.broadcast_to_job(job_id, message),
    )


async def broadcast_job_failed(
    case_id: Optional[str],
    job_id: str,
    job_type: str,
    error_code: str,
    message: str,
    retryable: bool = False,
    retry_count: int = 0,
) -> None:
    """Publish a JOB_FAILED event to the case and/or job channel."""
    payload = JobFailedMessage(
        job_id=job_id,
        case_id=case_id,
        job_type=job_type,
        error_code=error_code,
        message=message,
        retryable=retryable,
        retry_count=retry_count,
    ).model_dump()

    tasks = [ConnectionManager.broadcast_to_job(job_id, payload)]
    if case_id:
        tasks.append(ConnectionManager.broadcast_to_case(case_id, payload))
    await asyncio.gather(*tasks)


async def broadcast_plan_updated(
    case_id: str,
    plan_id: str,
    fragment_id: str,
    edit_type: str,
    surgeon: str,
    plan_version: int,
) -> None:
    """Publish a PLAN_UPDATED event to the case channel."""
    message = PlanUpdatedMessage(
        plan_id=plan_id,
        case_id=case_id,
        fragment_id=fragment_id,
        edit_type=edit_type,
        surgeon=surgeon,
        plan_version=plan_version,
    ).model_dump()

    await ConnectionManager.broadcast_to_case(case_id, message)


async def broadcast_case_status_changed(
    case_id: str,
    old_status: str,
    new_status: str,
    changed_by: Optional[str] = None,
) -> None:
    """Publish a CASE_STATUS_CHANGED event to the case channel."""
    message = CaseStatusChangedMessage(
        case_id=case_id,
        old_status=old_status,
        new_status=new_status,
        changed_by=changed_by,
    ).model_dump()

    await ConnectionManager.broadcast_to_case(case_id, message)
