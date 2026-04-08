"""
Facial Align Backend Configuration
Pydantic Settings-based configuration with environment variable support.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from pydantic import AnyHttpUrl, Field, PostgresDsn, RedisDsn, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """PostgreSQL database configuration."""

    model_config = SettingsConfigDict(env_prefix="DB_", extra="ignore")

    host: str = Field(default="localhost", description="PostgreSQL host")
    port: int = Field(default=5432, description="PostgreSQL port")
    name: str = Field(default="facialign", description="Database name")
    user: str = Field(default="facialign", description="Database user")
    password: str = Field(default="changeme", description="Database password")
    pool_size: int = Field(default=10, description="Connection pool size")
    max_overflow: int = Field(default=20, description="Max pool overflow connections")
    pool_timeout: int = Field(default=30, description="Pool connection timeout (seconds)")
    echo_sql: bool = Field(default=False, description="Echo SQL statements to logs")

    @property
    def async_url(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @property
    def sync_url(self) -> str:
        return f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class StorageSettings(BaseSettings):
    """File storage configuration (local or S3-compatible)."""

    model_config = SettingsConfigDict(env_prefix="STORAGE_", extra="ignore")

    backend: str = Field(default="local", description="Storage backend: 'local' or 's3'")
    base_path: Path = Field(default=Path("/data/facialign"), description="Local storage base path")
    dicom_path: Path = Field(default=Path("/data/facialign/dicom"), description="DICOM storage path")
    mesh_path: Path = Field(default=Path("/data/facialign/meshes"), description="Mesh storage path")
    mask_path: Path = Field(default=Path("/data/facialign/masks"), description="Segmentation mask storage path")
    temp_path: Path = Field(default=Path("/tmp/facialign"), description="Temporary processing path")

    # S3 settings
    s3_bucket: Optional[str] = Field(default=None, description="S3 bucket name")
    s3_region: str = Field(default="us-east-1", description="AWS region")
    s3_endpoint_url: Optional[str] = Field(default=None, description="S3-compatible endpoint (e.g., MinIO)")
    s3_access_key_id: Optional[str] = Field(default=None, description="S3 access key")
    s3_secret_access_key: Optional[str] = Field(default=None, description="S3 secret key")
    s3_prefix: str = Field(default="facialign/", description="S3 key prefix")


class ModelRegistrySettings(BaseSettings):
    """ML model registry and inference configuration."""

    model_config = SettingsConfigDict(env_prefix="MODEL_", extra="ignore")

    registry_path: Path = Field(
        default=Path("/models"),
        description="Root path for model weight files"
    )
    cache_path: Path = Field(
        default=Path("/tmp/model_cache"),
        description="Cached model inference outputs"
    )

    # Segmentation models
    totalsegmentator_weights_path: Optional[Path] = Field(
        default=None,
        description="TotalSegmentator custom weights override"
    )
    cmf_segmentation_model_path: Optional[Path] = Field(
        default=None,
        description="Fine-tuned CMF segmentation model path"
    )
    dental_segmentation_model_path: Optional[Path] = Field(
        default=None,
        description="Dental segmentation model path (ONNX or PyTorch)"
    )

    # Reduction model
    fracture_reduction_model_path: Optional[Path] = Field(
        default=None,
        description="Fracture reduction ML model path"
    )

    # Registration model
    deep_registration_model_path: Optional[Path] = Field(
        default=None,
        description="Deep registration model path"
    )

    # Inference settings
    default_device: str = Field(
        default="cuda",
        description="Default inference device: 'cuda', 'cpu', or 'cuda:N'"
    )
    inference_batch_size: int = Field(default=1, description="Inference batch size")
    inference_fp16: bool = Field(default=True, description="Use FP16 half-precision inference")
    model_timeout_seconds: int = Field(default=300, description="Max inference timeout")
    max_concurrent_inferences: int = Field(default=2, description="Max parallel GPU inference jobs")

    # TotalSegmentator specific
    totalsegmentator_task: str = Field(
        default="total",
        description="TotalSegmentator task: 'total', 'fast', 'craniofacial'"
    )
    totalsegmentator_fast: bool = Field(default=False, description="Use fast mode (lower accuracy)")


class CelerySettings(BaseSettings):
    """Celery task queue configuration."""

    model_config = SettingsConfigDict(env_prefix="CELERY_", extra="ignore")

    broker_url: str = Field(default="redis://localhost:6379/0", description="Celery broker URL")
    result_backend: str = Field(default="redis://localhost:6379/1", description="Celery result backend URL")
    task_serializer: str = Field(default="json")
    result_serializer: str = Field(default="json")
    accept_content: list[str] = Field(default=["json"])
    task_always_eager: bool = Field(default=False, description="Run tasks synchronously (for testing)")
    task_soft_time_limit: int = Field(default=3600, description="Soft task time limit (seconds)")
    task_time_limit: int = Field(default=7200, description="Hard task time limit (seconds)")
    worker_concurrency: int = Field(default=2, description="Celery worker concurrency")
    worker_prefetch_multiplier: int = Field(default=1, description="Tasks prefetched per worker")


class SecuritySettings(BaseSettings):
    """Authentication and security configuration."""

    model_config = SettingsConfigDict(env_prefix="SECURITY_", extra="ignore")

    secret_key: str = Field(
        default="CHANGE_ME_IN_PRODUCTION_THIS_IS_NOT_SECURE",
        description="JWT signing secret key"
    )
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=60, description="Access token TTL")
    refresh_token_expire_days: int = Field(default=7, description="Refresh token TTL")
    api_key_header: str = Field(default="X-API-Key", description="API key header name")

    # HIPAA audit settings
    audit_log_enabled: bool = Field(default=True, description="Enable HIPAA audit logging")
    audit_log_path: Path = Field(
        default=Path("/var/log/facialign/audit.log"),
        description="Audit log file path"
    )

    # CORS
    allowed_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        description="CORS allowed origins"
    )
    allowed_methods: list[str] = Field(default=["*"])
    allowed_headers: list[str] = Field(default=["*"])
    allow_credentials: bool = Field(default=True)


class GPUSettings(BaseSettings):
    """GPU and compute resource configuration."""

    model_config = SettingsConfigDict(env_prefix="GPU_", extra="ignore")

    enabled: bool = Field(default=True, description="Enable GPU acceleration")
    device_ids: list[int] = Field(default=[0], description="CUDA device IDs to use")
    memory_fraction: float = Field(
        default=0.8,
        description="Fraction of GPU memory to allocate",
        ge=0.1,
        le=1.0
    )
    enable_tf32: bool = Field(default=True, description="Enable TF32 for Ampere+ GPUs")
    enable_cudnn_benchmark: bool = Field(
        default=True,
        description="Enable cuDNN auto-tuner (good for fixed input sizes)"
    )


class AppSettings(BaseSettings):
    """Top-level application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application metadata
    app_name: str = Field(default="Facial Align", description="Application name")
    app_version: str = Field(default="0.1.0", description="API version")
    environment: str = Field(
        default="development",
        description="Deployment environment: development, staging, production"
    )
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    log_json: bool = Field(default=True, description="Output logs as JSON")

    # API settings
    api_v1_prefix: str = Field(default="/api/v1", description="API v1 route prefix")
    docs_enabled: bool = Field(default=True, description="Enable OpenAPI docs")

    # Nested settings (populated from environment prefix)
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    model_registry: ModelRegistrySettings = Field(default_factory=ModelRegistrySettings)
    celery: CelerySettings = Field(default_factory=CelerySettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    gpu: GPUSettings = Field(default_factory=GPUSettings)

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "staging", "production", "test"}
        if v not in allowed:
            raise ValueError(f"environment must be one of {allowed}")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v.upper()

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        return self.environment == "development"


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return cached application settings instance."""
    return AppSettings()


# Convenience alias
settings = get_settings()
