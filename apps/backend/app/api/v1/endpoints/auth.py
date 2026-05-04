"""Auth endpoints — register, login, refresh, profile."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.security import (
    CurrentUser,
    audit_logger,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    hash_password,
    verify_password,
)
from app.db.database import get_db_session
from app.models.user import User
from app.schemas.auth import (
    PasswordChangeRequest,
    TokenRefreshRequest,
    TokenResponse,
    UserLogin,
    UserRegister,
    UserResponse,
    UserUpdate,
)

router = APIRouter(prefix="/auth", tags=["auth"])
settings = get_settings()


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _make_token_response(
    user_id: str,
    role: str,
    institution_code: str | None = None,
) -> TokenResponse:
    access_token = create_access_token(
        user_id=user_id,
        role=role,
        institution_code=institution_code,
    )
    refresh_token = create_refresh_token(user_id=user_id)
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.security.access_token_expire_minutes * 60,
    )


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(
    body: UserRegister,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
) -> TokenResponse:
    result = await db.execute(select(User).where(User.email == body.email))
    if result.scalars().first():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    user = User(
        email=body.email,
        hashed_password=hash_password(body.password),
        full_name=body.full_name,
        role=body.role,
        institution=body.institution,
        institution_code=body.institution_code,
        specialty=body.specialty,
        is_verified=settings.environment != "production",
    )
    db.add(user)
    await db.flush()

    audit_logger.log_auth_event(
        user_id=str(user.id),
        event_type="REGISTER",
        ip_address=_client_ip(request),
        success=True,
    )

    return _make_token_response(
        user_id=str(user.id),
        role=user.role,
        institution_code=user.institution_code,
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    body: UserLogin,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
) -> TokenResponse:
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalars().first()

    if not user or not verify_password(body.password, user.hashed_password):
        audit_logger.log_auth_event(
            user_id=body.email,
            event_type="LOGIN",
            ip_address=_client_ip(request),
            success=False,
            failure_reason="Invalid credentials",
        )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is disabled")

    user.last_login_at = datetime.now(timezone.utc)
    user.login_count = (user.login_count or 0) + 1
    await db.flush()

    audit_logger.log_auth_event(
        user_id=str(user.id),
        event_type="LOGIN",
        ip_address=_client_ip(request),
        success=True,
    )

    return _make_token_response(
        user_id=str(user.id),
        role=user.role,
        institution_code=user.institution_code,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    body: TokenRefreshRequest,
    db: AsyncSession = Depends(get_db_session),
) -> TokenResponse:
    payload = decode_token(body.refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")

    # Look up user to get current role
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalars().first()
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive")

    return _make_token_response(
        user_id=str(user.id),
        role=user.role,
        institution_code=user.institution_code,
    )


@router.get("/me", response_model=UserResponse)
async def get_me(
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> UserResponse:
    result = await db.execute(select(User).where(User.id == current_user.user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return UserResponse.model_validate(user)


@router.put("/me", response_model=UserResponse)
async def update_me(
    body: UserUpdate,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> UserResponse:
    result = await db.execute(select(User).where(User.id == current_user.user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    update_data = body.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(user, field, value)
    await db.flush()

    return UserResponse.model_validate(user)


@router.post("/change-password", status_code=status.HTTP_204_NO_CONTENT)
async def change_password(
    body: PasswordChangeRequest,
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> None:
    result = await db.execute(select(User).where(User.id == current_user.user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if not verify_password(body.current_password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Current password is incorrect")

    user.hashed_password = hash_password(body.new_password)
    await db.flush()

    audit_logger.log_auth_event(
        user_id=str(user.id),
        event_type="PASSWORD_CHANGE",
        ip_address=_client_ip(request),
        success=True,
    )
