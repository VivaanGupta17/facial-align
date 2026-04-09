"""Auth request/response schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import Field

from app.schemas.common import BaseSchema


class UserRegister(BaseSchema):
    email: str = Field(..., min_length=5)
    password: str = Field(..., min_length=6)
    full_name: str = Field(..., min_length=1)
    role: str = Field(default="surgeon")
    institution: Optional[str] = None
    specialty: Optional[str] = None


class UserLogin(BaseSchema):
    email: str
    password: str


class TokenResponse(BaseSchema):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class UserResponse(BaseSchema):
    id: UUID
    email: str
    full_name: str
    role: str
    institution: Optional[str] = None
    specialty: Optional[str] = None
    is_active: bool
    created_at: datetime


class TokenRefreshRequest(BaseSchema):
    refresh_token: str


class PasswordChangeRequest(BaseSchema):
    current_password: str
    new_password: str = Field(..., min_length=6)


class UserUpdate(BaseSchema):
    full_name: Optional[str] = None
    institution: Optional[str] = None
    specialty: Optional[str] = None
