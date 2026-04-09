"""
Token-bucket rate limiting middleware for the Facial Align API.

Protects endpoints against abuse and ensures fair resource allocation
across users. Rate limits are tuned per endpoint group, reflecting the
vastly different cost profiles of viewer updates vs. GPU inference.

Storage backends:
  - Redis (preferred): distributed, survives process restarts.
  - In-memory dict (fallback): single-process only, resets on restart.

Rate limit state is keyed as:  rate:<user_or_ip>:<endpoint_group>
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from dataclasses import dataclass
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from app.core.logging import get_logger

logger = get_logger(__name__)

# ─── Endpoint group configuration ─────────────────────────────────────────────


@dataclass(frozen=True)
class EndpointGroup:
    """Rate-limit configuration for one path group."""

    name: str
    pattern: re.Pattern[str]
    requests_per_minute: int

    @property
    def capacity(self) -> int:
        """Token bucket capacity = requests_per_minute."""
        return self.requests_per_minute

    @property
    def refill_rate(self) -> float:
        """Tokens added per second."""
        return self.requests_per_minute / 60.0


# Ordered list; first match wins.
_ENDPOINT_GROUPS: list[EndpointGroup] = [
    EndpointGroup(
        name="dicom_upload",
        pattern=re.compile(r"^/api/v1/dicom/upload"),
        requests_per_minute=10,
    ),
    EndpointGroup(
        name="segmentation_run",
        pattern=re.compile(r"^/api/v1/segmentation/run"),
        requests_per_minute=5,
    ),
    EndpointGroup(
        name="viewer",
        pattern=re.compile(r"^/api/v1/viewer"),
        requests_per_minute=120,
    ),
    EndpointGroup(
        name="planning",
        pattern=re.compile(r"^/api/v1/planning"),
        requests_per_minute=60,
    ),
    EndpointGroup(
        name="default",
        pattern=re.compile(r".*"),
        requests_per_minute=100,
    ),
]


def _match_group(path: str) -> EndpointGroup:
    for group in _ENDPOINT_GROUPS:
        if group.pattern.match(path):
            return group
    return _ENDPOINT_GROUPS[-1]  # default


# ─── Token bucket state ───────────────────────────────────────────────────────


@dataclass
class _BucketState:
    """
    In-memory token bucket state for one (identity, group) pair.

    tokens:      current number of available tokens.
    last_refill: Unix timestamp of the last refill calculation.
    """

    tokens: float
    last_refill: float


# ─── Storage backends ─────────────────────────────────────────────────────────


class _InMemoryStore:
    """
    Single-process, in-memory token bucket store.

    Not suitable for multi-process deployments; used as a fallback when
    Redis is unavailable.
    """

    def __init__(self) -> None:
        self._buckets: dict[str, _BucketState] = {}
        self._lock = asyncio.Lock()

    async def consume(
        self,
        key: str,
        capacity: int,
        refill_rate: float,
    ) -> tuple[bool, int, float]:
        """
        Attempt to consume one token from the bucket identified by *key*.

        Returns:
            (allowed, remaining_tokens, reset_in_seconds)
        """
        async with self._lock:
            now = time.monotonic()
            state = self._buckets.get(key)

            if state is None:
                state = _BucketState(tokens=float(capacity), last_refill=now)

            # Refill tokens based on elapsed time
            elapsed = now - state.last_refill
            state.tokens = min(
                float(capacity),
                state.tokens + elapsed * refill_rate,
            )
            state.last_refill = now

            if state.tokens >= 1.0:
                state.tokens -= 1.0
                self._buckets[key] = state
                remaining = int(state.tokens)
                reset_in = (capacity - state.tokens) / refill_rate if refill_rate else 0
                return True, remaining, reset_in
            else:
                self._buckets[key] = state
                reset_in = (1.0 - state.tokens) / refill_rate if refill_rate else 60.0
                return False, 0, reset_in


class _RedisStore:
    """
    Redis-backed token bucket store using a Lua script for atomicity.

    The Lua script is sent as a single EVALSHA operation so that the
    read-modify-write cycle cannot be interrupted by another client.
    """

    # Lua script implementing token bucket consume atomically.
    # KEYS[1] = bucket key
    # ARGV[1] = capacity, ARGV[2] = refill_rate (tokens/sec), ARGV[3] = now (unix float)
    _LUA_SCRIPT = """
local key         = KEYS[1]
local capacity    = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local now         = tonumber(ARGV[3])

local data = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens      = tonumber(data[1]) or capacity
local last_refill = tonumber(data[2]) or now

-- Refill
local elapsed = now - last_refill
tokens = math.min(capacity, tokens + elapsed * refill_rate)

local allowed = 0
local remaining = 0
local reset_in  = 0

if tokens >= 1.0 then
    tokens   = tokens - 1.0
    allowed  = 1
    remaining = math.floor(tokens)
    reset_in  = (capacity - tokens) / refill_rate
else
    reset_in = (1.0 - tokens) / refill_rate
end

-- TTL: bucket expires after 2x the full refill time to avoid unbounded growth
local ttl = math.ceil(capacity / refill_rate * 2)
redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
redis.call('EXPIRE', key, ttl)

return {allowed, remaining, tostring(reset_in)}
"""

    def __init__(self, redis_client: Any) -> None:  # type: ignore[name-defined]
        self._redis = redis_client
        self._sha: Optional[str] = None

    async def _load_script(self) -> str:
        if self._sha is None:
            self._sha = await self._redis.script_load(self._LUA_SCRIPT)
        return self._sha

    async def consume(
        self,
        key: str,
        capacity: int,
        refill_rate: float,
    ) -> tuple[bool, int, float]:
        try:
            sha = await self._load_script()
            now = time.time()
            result = await self._redis.evalsha(
                sha, 1, key,
                str(capacity), str(refill_rate), str(now),
            )
            allowed = bool(int(result[0]))
            remaining = int(result[1])
            reset_in = float(result[2])
            return allowed, remaining, reset_in
        except Exception as exc:
            logger.warning("redis_rate_limit_error", error=str(exc))
            # Fail open: allow the request but log the incident
            return True, capacity, 0.0


# ─── Middleware ────────────────────────────────────────────────────────────────

# Import Any here to avoid forward reference issues
from typing import Any  # noqa: E402


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Token-bucket rate limiting middleware.

    Limits are applied per *authenticated user* when a user_id is
    available on request.state (set by RequestContextMiddleware), or
    per *client IP* for unauthenticated requests.

    Responses include standard rate-limit headers:
      X-RateLimit-Limit:     bucket capacity
      X-RateLimit-Remaining: tokens left after this request
      X-RateLimit-Reset:     seconds until the bucket fully refills
      Retry-After:           same as Reset, on 429 responses
    """

    # Paths that bypass rate limiting entirely (health checks, metrics)
    _BYPASS_PATHS: frozenset[str] = frozenset(
        {"/health", "/healthz", "/readiness", "/liveness", "/", "/metrics"}
    )

    def __init__(
        self,
        app: ASGIApp,
        *,
        redis_url: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(app)
        self._enabled = enabled
        self._store: _InMemoryStore | _RedisStore

        if redis_url:
            try:
                import aioredis  # type: ignore[import]
                redis_client = aioredis.from_url(redis_url, decode_responses=True)
                self._store = _RedisStore(redis_client)
                logger.info("rate_limiter_redis_backend", url=redis_url)
            except ImportError:
                logger.warning(
                    "rate_limiter_redis_unavailable",
                    reason="aioredis not installed",
                    fallback="in_memory",
                )
                self._store = _InMemoryStore()
        else:
            self._store = _InMemoryStore()
            logger.info("rate_limiter_in_memory_backend")

    def _identity_key(self, request: Request) -> str:
        """
        Derive a rate-limit identity string from the request.

        Prefers user_id (set by auth middleware) over IP address.
        IP addresses are hashed to avoid storing raw PII in Redis keys.
        """
        user_id: Optional[str] = getattr(request.state, "user_id", None)
        if user_id:
            return f"user:{user_id}"

        ip = (
            request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
            or (request.client.host if request.client else "unknown")
        )
        ip_hash = hashlib.sha256(ip.encode()).hexdigest()[:16]
        return f"ip:{ip_hash}"

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        if not self._enabled or request.url.path in self._BYPASS_PATHS:
            return await call_next(request)

        path = request.url.path
        group = _match_group(path)
        identity = self._identity_key(request)
        bucket_key = f"rate:{identity}:{group.name}"

        allowed, remaining, reset_in = await self._store.consume(
            key=bucket_key,
            capacity=group.capacity,
            refill_rate=group.refill_rate,
        )

        reset_at = int(time.time() + reset_in)

        if not allowed:
            logger.warning(
                "rate_limit_exceeded",
                identity=identity,
                endpoint_group=group.name,
                path=path,
                limit=group.requests_per_minute,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "message": (
                        f"Rate limit exceeded for endpoint group '{group.name}'. "
                        f"Limit: {group.requests_per_minute} requests/minute."
                    ),
                    "details": {
                        "limit": group.requests_per_minute,
                        "endpoint_group": group.name,
                        "retry_after_seconds": int(reset_in) + 1,
                    },
                    "request_id": getattr(request.state, "request_id", None),
                    "timestamp": _utc_iso(),
                },
                headers={
                    "X-RateLimit-Limit": str(group.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_at),
                    "Retry-After": str(int(reset_in) + 1),
                },
            )

        response = await call_next(request)

        # Attach rate-limit headers to successful responses
        response.headers["X-RateLimit-Limit"] = str(group.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_at)

        return response


def _utc_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
