# ─── Facial Align Backend — Multi-stage Docker build with CUDA support ────────
#
# Stages:
#   base         — CUDA base image with system dependencies
#   python-deps  — Python package installation (cached layer)
#   dev          — Development image with dev tools
#   production   — Minimal production image
#
# Usage:
#   docker build --target production -t facialign-backend:latest .
#   docker build --target dev -t facialign-backend:dev .
#
# GPU inference requires NVIDIA Container Toolkit:
#   docker run --gpus all -p 8000:8000 facialign-backend:latest

# ─── Stage 1: Base system image ───────────────────────────────────────────────

ARG CUDA_VERSION=12.1.1
ARG CUDNN_VERSION=8
ARG UBUNTU_VERSION=22.04
ARG PYTHON_VERSION=3.11

FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONFAULTHANDLER=1

# CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
    python3-pip \
    # Build tools
    build-essential \
    cmake \
    git \
    # Image/medical file format support
    libgdcm-dev \
    libinsighttoolkit5-dev \
    # Mesh processing (Open3D dependencies)
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # Network utilities
    curl \
    wget \
    # Storage utilities
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip==24.0

# ─── Stage 2: Python dependencies ────────────────────────────────────────────

FROM base AS python-deps

WORKDIR /build

# Copy dependency specification
COPY pyproject.toml .

# Install PyTorch first (large; benefits from separate cache layer)
# Using CUDA 12.1 index for GPU support
RUN pip install --no-cache-dir \
    torch==2.3.0+cu121 \
    torchvision==0.18.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install ONNX Runtime GPU
RUN pip install --no-cache-dir onnxruntime-gpu>=1.18.0

# Install core ML/medical imaging stack
# Note: totalsegmentator and nnunetv2 have large dependencies
RUN pip install --no-cache-dir \
    monai>=1.3.0 \
    SimpleITK>=2.3.1 \
    nibabel>=5.2.1 \
    pydicom>=2.4.4 \
    highdicom>=0.22.0

# Install mesh processing (open3d is large)
RUN pip install --no-cache-dir \
    trimesh>=4.3.2 \
    open3d>=0.18.0 \
    pyvista>=0.43.9 \
    scikit-image>=0.23.2

# Install TotalSegmentator
RUN pip install --no-cache-dir totalsegmentator>=2.2.1

# Install nnUNetV2 (optional, for custom model training)
RUN pip install --no-cache-dir nnunetv2>=2.4.1

# Install remaining application dependencies
RUN pip install --no-cache-dir \
    fastapi>=0.111.0 \
    uvicorn[standard]>=0.29.0 \
    sqlalchemy[asyncio]>=2.0.29 \
    asyncpg>=0.29.0 \
    psycopg2-binary>=2.9.9 \
    alembic>=1.13.1 \
    pydantic>=2.7.0 \
    pydantic-settings>=2.2.1 \
    celery[redis]>=5.3.6 \
    redis>=5.0.4 \
    python-jose[cryptography]>=3.3.0 \
    passlib[bcrypt]>=1.7.4 \
    cryptography>=42.0.5 \
    structlog>=24.1.0 \
    python-multipart>=0.0.9 \
    httpx>=0.27.0 \
    scipy>=1.13.0 \
    numpy>=1.26.4 \
    python-magic>=0.4.27 \
    pillow>=10.3.0

# ─── Stage 3: Development image ───────────────────────────────────────────────

FROM python-deps AS dev

# Install development tools
RUN pip install --no-cache-dir \
    pytest>=8.2.0 \
    pytest-asyncio>=0.23.6 \
    pytest-cov>=5.0.0 \
    pytest-mock>=3.14.0 \
    black>=24.4.2 \
    ruff>=0.4.4 \
    mypy>=1.10.0 \
    ipython>=8.24.0 \
    watchfiles>=0.21.0

# Application source
WORKDIR /app
COPY . .

ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV ENVIRONMENT=development
ENV LOG_LEVEL=DEBUG
ENV LOG_JSON=false

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ─── Stage 4: Production image ────────────────────────────────────────────────

FROM python-deps AS production

# Create non-root application user
RUN groupadd --gid 10001 facialign \
    && useradd \
        --uid 10001 \
        --gid facialign \
        --shell /bin/bash \
        --create-home \
        facialign

# Create required directories with correct permissions
RUN mkdir -p \
    /data/facialign/dicom \
    /data/facialign/meshes \
    /data/facialign/masks \
    /tmp/facialign/uploads \
    /models \
    /var/log/facialign \
    && chown -R facialign:facialign \
        /data/facialign \
        /tmp/facialign \
        /var/log/facialign

# Application source
WORKDIR /app
COPY --chown=facialign:facialign . .

ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV ENVIRONMENT=production
ENV LOG_LEVEL=INFO
ENV LOG_JSON=true

# Switch to non-root user
USER facialign

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health/live || exit 1

EXPOSE 8000

# Use gunicorn with uvicorn workers for production
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--timeout-keep-alive", "30", \
     "--access-log", \
     "--log-config", "/app/logging.json"]

# ─── Stage 5: Celery GPU worker ───────────────────────────────────────────────

FROM production AS celery-gpu-worker

# GPU workers need more memory; adjust ulimits in compose
ENV CELERY_WORKER_CONCURRENCY=1
ENV GPU_DEVICE_IDS="[0]"

CMD ["celery", \
     "--app=app.workers.celery_app:celery_app", \
     "worker", \
     "--loglevel=INFO", \
     "--queues=gpu,default", \
     "--concurrency=${CELERY_WORKER_CONCURRENCY}", \
     "--hostname=gpu-worker@%h"]

# ─── Stage 6: Celery CPU worker ───────────────────────────────────────────────

FROM production AS celery-cpu-worker

ENV CELERY_WORKER_CONCURRENCY=4
ENV GPU_ENABLED=false

CMD ["celery", \
     "--app=app.workers.celery_app:celery_app", \
     "worker", \
     "--loglevel=INFO", \
     "--queues=default", \
     "--concurrency=${CELERY_WORKER_CONCURRENCY}", \
     "--hostname=cpu-worker@%h"]

# ─── Labels ───────────────────────────────────────────────────────────────────

LABEL org.opencontainers.image.title="Facial Align Backend"
LABEL org.opencontainers.image.description="AI-native craniofacial surgical planning platform"
LABEL org.opencontainers.image.vendor="Facial Align"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.licenses="Proprietary"
