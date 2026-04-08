"""
Shared Pydantic schemas used across the API.
Includes geometry types, pagination, job status, and health checks.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Generic, List, Optional, TypeVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

T = TypeVar("T")


class BaseSchema(BaseModel):
    """Base schema with common configuration."""
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        use_enum_values=True,
    )


# ─── Geometry types ───────────────────────────────────────────────────────────


class Vector3D(BaseSchema):
    """3D Euclidean vector."""
    x: float = Field(..., description="X component")
    y: float = Field(..., description="Y component")
    z: float = Field(..., description="Z component")

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.z]

    @classmethod
    def from_list(cls, v: list[float]) -> "Vector3D":
        return cls(x=v[0], y=v[1], z=v[2])


class Transform3D(BaseSchema):
    """
    Rigid body transformation in SE(3).
    Rotation represented as a 3x3 rotation matrix (row-major).
    Translation in millimeters.
    """
    rotation_matrix: list[list[float]] = Field(
        ...,
        description="3x3 rotation matrix (row-major, orthonormal)",
        examples=[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
    )
    translation_mm: list[float] = Field(
        ...,
        description="Translation vector [x, y, z] in millimeters",
        examples=[[0.0, 0.0, 0.0]],
    )

    @field_validator("rotation_matrix")
    @classmethod
    def validate_rotation_matrix(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) != 3 or any(len(row) != 3 for row in v):
            raise ValueError("rotation_matrix must be 3x3")
        # Check approximate orthonormality
        R = np.array(v, dtype=np.float64)
        RtR = R.T @ R
        I = np.eye(3)
        if not np.allclose(RtR, I, atol=1e-4):
            raise ValueError("rotation_matrix is not orthonormal (R^T R ≠ I)")
        return v

    @field_validator("translation_mm")
    @classmethod
    def validate_translation(cls, v: list[float]) -> list[float]:
        if len(v) != 3:
            raise ValueError("translation_mm must have exactly 3 components")
        return v

    @classmethod
    def identity(cls) -> "Transform3D":
        """Return the identity transform."""
        return cls(
            rotation_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            translation_mm=[0.0, 0.0, 0.0],
        )

    def to_4x4_matrix(self) -> list[list[float]]:
        """Return homogeneous 4x4 transformation matrix."""
        R = self.rotation_matrix
        t = self.translation_mm
        return [
            [R[0][0], R[0][1], R[0][2], t[0]],
            [R[1][0], R[1][1], R[1][2], t[1]],
            [R[2][0], R[2][1], R[2][2], t[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]


class BoundingBox3D(BaseSchema):
    """Axis-aligned bounding box in 3D space (millimeters)."""
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float

    @property
    def dimensions(self) -> Vector3D:
        return Vector3D(
            x=self.max_x - self.min_x,
            y=self.max_y - self.min_y,
            z=self.max_z - self.min_z,
        )

    @property
    def center(self) -> Vector3D:
        return Vector3D(
            x=(self.min_x + self.max_x) / 2,
            y=(self.min_y + self.max_y) / 2,
            z=(self.min_z + self.max_z) / 2,
        )

    @property
    def volume_mm3(self) -> float:
        d = self.dimensions
        return d.x * d.y * d.z


# ─── Pagination ────────────────────────────────────────────────────────────────


class PaginationParams(BaseSchema):
    """Query parameters for paginated list endpoints."""
    page: int = Field(default=1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size


class PaginatedResponse(BaseSchema, Generic[T]):
    """Generic paginated list response."""
    items: List[T]
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")
    pages: int = Field(..., description="Total number of pages")

    @classmethod
    def create(
        cls,
        items: List[T],
        total: int,
        page: int,
        page_size: int,
    ) -> "PaginatedResponse[T]":
        pages = max(1, (total + page_size - 1) // page_size)
        return cls(items=items, total=total, page=page, page_size=page_size, pages=pages)


# ─── Job / task status ────────────────────────────────────────────────────────


class JobStatus(BaseSchema):
    """
    Async job (Celery task) status response.
    Returned for long-running operations like segmentation.
    """
    job_id: str = Field(..., description="Celery task ID")
    status: str = Field(
        ...,
        description="Job status: PENDING, STARTED, PROGRESS, SUCCESS, FAILURE, REVOKED",
    )
    progress_percent: Optional[float] = Field(
        None, ge=0, le=100, description="Completion percentage if available"
    )
    current_step: Optional[str] = Field(None, description="Description of current processing step")
    result: Optional[Any] = Field(None, description="Result payload on SUCCESS")
    error: Optional[str] = Field(None, description="Error message on FAILURE")
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    elapsed_seconds: Optional[float] = None


# ─── Health check ─────────────────────────────────────────────────────────────


class ComponentHealth(BaseSchema):
    """Health status of a single system component."""
    name: str
    status: str = Field(..., description="healthy, degraded, unhealthy")
    message: Optional[str] = None
    latency_ms: Optional[float] = None


class HealthCheck(BaseSchema):
    """Overall application health check response."""
    status: str = Field(..., description="healthy, degraded, unhealthy")
    version: str
    environment: str
    timestamp: datetime
    components: List[ComponentHealth] = Field(default_factory=list)
    gpu_available: bool = False
    gpu_devices: List[str] = Field(default_factory=list)


# ─── Error response ───────────────────────────────────────────────────────────


class ErrorResponse(BaseSchema):
    """Standard error response body."""
    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error description")
    context: Optional[dict[str, Any]] = Field(None, description="Additional error context")
    request_id: Optional[str] = None
