"""
JWT token handling, API key validation, and HIPAA audit logging.
All PHI access must be logged for HIPAA compliance.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import bcrypt
from fastapi import Depends, HTTPException, Request, Security, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db_session
from app.models.api_key import ApiKey
from app.models.user import User
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from jose import ExpiredSignatureError, JWTError, jwt

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# ─── Password hashing ─────────────────────────────────────────────────────────
BCRYPT_ROUNDS = 12


def hash_password(plain_password: str) -> str:
    """Hash a plain-text password using bcrypt."""
    salt = bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
    return bcrypt.hashpw(plain_password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain-text password against a hash."""
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except (TypeError, ValueError):
        logger.warning("password_hash_verification_failed", reason="invalid_hash_payload")
        return False


# ─── JWT Tokens ───────────────────────────────────────────────────────────────


class TokenPayload:
    """Structured JWT token payload."""

    def __init__(
        self,
        sub: str,
        role: str,
        exp: datetime,
        jti: str,
        iss: str = "facialign",
    ) -> None:
        self.sub = sub  # Subject (user ID)
        self.role = role
        self.exp = exp
        self.jti = jti  # JWT ID for revocation
        self.iss = iss

    def to_dict(self) -> dict[str, Any]:
        return {
            "sub": self.sub,
            "role": self.role,
            "exp": self.exp,
            "jti": self.jti,
            "iss": self.iss,
        }


def create_access_token(
    user_id: str,
    role: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a signed JWT access token.

    Args:
        user_id: Unique user identifier (never embed PHI)
        role: User role (surgeon, admin, viewer, engineer)
        expires_delta: Custom expiry; defaults to settings value

    Returns:
        Signed JWT string
    """
    if expires_delta is None:
        expires_delta = timedelta(minutes=settings.security.access_token_expire_minutes)

    now = datetime.now(timezone.utc)
    expire = now + expires_delta
    payload = {
        "sub": user_id,
        "role": role,
        "exp": expire,
        "iat": now,
        "jti": str(uuid.uuid4()),
        "iss": "facialign",
        "type": "access",
    }
    return jwt.encode(
        payload,
        settings.security.secret_key,
        algorithm=settings.security.algorithm,
    )


def create_refresh_token(user_id: str) -> str:
    """Create a long-lived refresh token."""
    now = datetime.now(timezone.utc)
    expire = now + timedelta(days=settings.security.refresh_token_expire_days)
    payload = {
        "sub": user_id,
        "exp": expire,
        "iat": now,
        "jti": str(uuid.uuid4()),
        "iss": "facialign",
        "type": "refresh",
    }
    return jwt.encode(
        payload,
        settings.security.secret_key,
        algorithm=settings.security.algorithm,
    )


def decode_token(token: str) -> dict[str, Any]:
    """
    Decode and validate a JWT token.

    Raises:
        HTTPException 401 on invalid/expired token
    """
    try:
        payload = jwt.decode(
            token,
            settings.security.secret_key,
            algorithms=[settings.security.algorithm],
        )
        return payload
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except JWTError as e:
        logger.warning("jwt_decode_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ─── FastAPI security dependencies ────────────────────────────────────────────

bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name=settings.security.api_key_header, auto_error=False)


class CurrentUser:
    """Authenticated user context extracted from JWT."""

    def __init__(
        self,
        user_id: str,
        role: str,
        jti: str,
    ) -> None:
        self.user_id = user_id
        self.role = role
        self.jti = jti

    @property
    def is_surgeon(self) -> bool:
        return self.role == "surgeon"

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def require_role(self, *roles: str) -> None:
        """Raise 403 if user does not have one of the required roles."""
        if self.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{self.role}' does not have permission for this action",
            )


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    api_key: Optional[str] = Security(api_key_header),
    db: AsyncSession = Depends(get_db_session),
) -> CurrentUser:
    """
    FastAPI dependency to extract and validate the current user.
    Supports both Bearer JWT and API Key authentication.
    """
    if credentials:
        payload = decode_token(credentials.credentials)
        user_id = payload.get("sub")
        role = payload.get("role", "viewer")
        jti = payload.get("jti", "")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )
        return CurrentUser(user_id=user_id, role=role, jti=jti)

    if api_key:
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        db_key_user = (
            await db.execute(
                select(ApiKey, User)
                .join(User, User.id == ApiKey.user_id)
                .where(ApiKey.key_hash == api_key_hash, ApiKey.is_active == True)
            )
        ).first()
        
        if db_key_user:
            api_key_obj, user = db_key_user
            if not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Inactive user for this API key",
                )
            return CurrentUser(user_id=str(user.id), role=user.role, jti="")
            
        logger.info("api_key_auth_attempt", key_hash_prefix=api_key_hash[:8], result="failed")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def require_surgeon(
    current_user: CurrentUser = Depends(get_current_user),
) -> CurrentUser:
    """Dependency requiring surgeon or admin role."""
    current_user.require_role("surgeon", "admin")
    return current_user


async def require_admin(
    current_user: CurrentUser = Depends(get_current_user),
) -> CurrentUser:
    """Dependency requiring admin role."""
    current_user.require_role("admin")
    return current_user


# ─── HIPAA Audit Logging ──────────────────────────────────────────────────────


class HIPAAAuditLogger:
    """
    HIPAA-compliant audit logger.

    Every access to Protected Health Information (PHI) must be logged per
    45 CFR §164.312(b) - Audit Controls. Logs include: who accessed what,
    when, from where, and what action was performed.
    """

    AUDIT_ACTIONS = {
        "READ": "read",
        "CREATE": "create",
        "UPDATE": "update",
        "DELETE": "delete",
        "EXPORT": "export",
        "PRINT": "print",
        "SHARE": "share",
        "LOGIN": "login",
        "LOGOUT": "logout",
        "AUTH_FAIL": "auth_failure",
    }

    def __init__(self) -> None:
        self._audit_logger = logging.getLogger("hipaa.audit")
        self._setup_audit_handler()

    def _setup_audit_handler(self) -> None:
        """Configure dedicated audit log handler with immutable-append semantics."""
        audit_log_path = settings.security.audit_log_path
        try:
            audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(audit_log_path), mode="a")
            file_handler.setFormatter(
                logging.Formatter("%(message)s")  # JSON lines
            )
            self._audit_logger.addHandler(file_handler)
            self._audit_logger.setLevel(logging.INFO)
            self._audit_logger.propagate = False
        except (PermissionError, OSError) as e:
            logger.warning(
                "audit_log_file_unavailable",
                path=str(audit_log_path),
                error=str(e),
                fallback="stderr",
            )
            self._audit_logger.addHandler(logging.StreamHandler())

    def log_phi_access(
        self,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: str,
        ip_address: str,
        success: bool = True,
        additional_context: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Record a PHI access event.

        Args:
            user_id: Identity of the accessing user
            action: HIPAA action code (READ, CREATE, UPDATE, etc.)
            resource_type: Type of resource accessed (patient, study, case, etc.)
            resource_id: Unique identifier of the accessed resource
            ip_address: Client IP address
            success: Whether the access was granted
            additional_context: Extra fields (non-PHI)
        """
        if not settings.security.audit_log_enabled:
            return

        audit_entry = {
            "audit_event": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_id": str(uuid.uuid4()),
            "user_id": user_id,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "ip_address": ip_address,
            "success": success,
            "hipaa_category": "access_control",
        }
        if additional_context:
            # Filter out any PHI from context
            audit_entry["context"] = {
                k: v for k, v in additional_context.items()
                if k not in ("mrn", "dob", "ssn", "name", "address")
            }

        self._audit_logger.info(json.dumps(audit_entry))

    def log_auth_event(
        self,
        user_id: str,
        event_type: str,
        ip_address: str,
        success: bool,
        failure_reason: Optional[str] = None,
    ) -> None:
        """Log authentication events (login, logout, auth failures)."""
        audit_entry = {
            "audit_event": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_id": str(uuid.uuid4()),
            "user_id": user_id,
            "event_type": event_type,
            "ip_address": ip_address,
            "success": success,
            "hipaa_category": "authentication",
        }
        if not success and failure_reason:
            audit_entry["failure_reason"] = failure_reason

        self._audit_logger.info(json.dumps(audit_entry))


# Singleton audit logger instance
audit_logger = HIPAAAuditLogger()


def hash_mrn(mrn: str, salt: Optional[str] = None) -> str:
    """
    One-way hash a Medical Record Number for de-identified storage.
    The salt should be a secret, institution-specific value stored securely.
    """
    salt_bytes = (salt or settings.security.secret_key).encode()
    return hashlib.sha256(salt_bytes + mrn.encode()).hexdigest()
