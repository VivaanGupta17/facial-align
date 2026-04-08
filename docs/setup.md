# Setup Instructions — Facial Align

**Audience:** Engineers setting up the development environment for the first time  
**Last Updated:** 2025

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Quick Start with Docker Compose](#2-quick-start-with-docker-compose)
3. [Local Development Setup](#3-local-development-setup)
4. [Environment Variables Reference](#4-environment-variables-reference)
5. [Database Setup](#5-database-setup)
6. [Running Tests](#6-running-tests)
7. [Common Troubleshooting](#7-common-troubleshooting)

---

## 1. Prerequisites

### Required Software

| Software | Minimum Version | Notes |
|---------|----------------|-------|
| Docker | 24.0 | Engine + CLI |
| Docker Compose | v2 (plugin) | `docker compose` not `docker-compose` |
| Git | 2.40 | |
| NVIDIA Container Toolkit | Latest | For GPU inference; see below |

**Check your versions:**
```bash
docker --version         # Docker version 24.x.x
docker compose version   # Docker Compose version v2.x.x
git --version            # git version 2.x.x
nvidia-smi               # if GPU available
```

### GPU Requirements

GPU is optional for development but required for practical segmentation inference speeds.

| Mode | Hardware | Expected Segmentation Time |
|------|---------|--------------------------|
| GPU (recommended) | NVIDIA GPU, ≥ 16 GB VRAM | 2–10 minutes per case |
| CPU fallback | Any modern CPU | 30–90 minutes per case |

**Supported GPU architectures:** Ampere (A100, A10G, RTX 30xx), Ada (RTX 40xx), Hopper (H100)  
**Minimum VRAM:** 16 GB (A10G, RTX 3090, RTX 4090)  
**Recommended:** NVIDIA A10G (24 GB) or A100 (40 GB)

### NVIDIA Container Toolkit Installation

The NVIDIA Container Toolkit allows Docker containers to access the host GPU.

**Ubuntu:**
```bash
# Configure the repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker daemon
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

**macOS:** GPU passthrough is not supported in Docker on macOS (M1/M2/M3 chips cannot be accessed by Docker Linux containers). Run in CPU mode.

### Disk Space

| Component | Storage Required |
|-----------|-----------------|
| Docker images (all services) | ~25 GB |
| TotalSegmentator model weights | ~3 GB |
| DentalSegmentator model weights | ~500 MB |
| MinIO data (per case) | ~500 MB (DICOM) + ~100 MB (outputs) |
| PostgreSQL database | < 1 GB for development |
| MLflow artifacts | Variable |
| **Total (new setup)** | **~40 GB recommended** |

---

## 2. Quick Start with Docker Compose

### Clone and Configure

```bash
# Clone the repository
git clone https://github.com/your-org/facial-align.git
cd facial-align

# Copy environment template
cp .env.example .env
```

**Edit `.env` — minimum required changes:**
```bash
# Must change from defaults:
SECRET_KEY=your-random-secret-key-here-at-least-32-chars
PHI_ENCRYPTION_KEY=your-256-bit-key-base64-encoded

# For CPU-only (no GPU): change this line
INFERENCE_DEVICE=cpu
```

Generate secure keys:
```bash
# Generate SECRET_KEY
python3 -c "import secrets; print(secrets.token_hex(32))"

# Generate PHI_ENCRYPTION_KEY (base64-encoded 32 bytes)
python3 -c "import secrets, base64; print(base64.b64encode(secrets.token_bytes(32)).decode())"
```

### Start the Stack

```bash
# Pull all images and start in background
docker compose up -d

# Watch logs (all services)
docker compose logs -f

# Watch logs (specific service)
docker compose logs -f backend
docker compose logs -f worker
docker compose logs -f inference
```

### Initialize the Database

```bash
# Run Alembic migrations
docker compose exec backend alembic upgrade head

# Verify migrations applied
docker compose exec backend alembic current
```

### Verify Health

```bash
# Check all containers are running
docker compose ps

# API health check
curl http://localhost:8000/health

# Inference service health (checks GPU availability)
curl http://localhost:8080/ping

# Expected output
# {"status": "Healthy"}
```

### Access Service UIs

| Service | URL | Credentials |
|---------|-----|------------|
| Frontend | http://localhost:3000 | Create account at /register |
| API Docs | http://localhost:8000/docs | — |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin |
| MLflow | http://localhost:5000 | — |
| Flower (Celery) | http://localhost:5555 | — |

### Upload a Test Case

```bash
# 1. Create a user (or use the frontend registration)
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "test1234", "name": "Test User"}'

# 2. Get an access token
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "test1234"}' | \
  python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# 3. Upload a DICOM ZIP (replace with your test data path)
curl -X POST http://localhost:8000/api/v1/cases \
  -H "Authorization: Bearer $TOKEN" \
  -F "dicom_archive=@/path/to/your/ct_series.zip" \
  -F "procedure_type=ORTHOGNATHIC"

# Response: {"case_id": "...", "job_id": "...", "status": "PENDING"}
```

### Stop the Stack

```bash
# Stop all containers (preserve data volumes)
docker compose down

# Stop and remove all data (destructive — removes database and MinIO data)
docker compose down -v
```

---

## 3. Local Development Setup

Use this when you need to modify backend or frontend code and want fast reload without rebuilding Docker images.

### Backend (Python)

**Requires Python 3.11+**

```bash
cd apps/backend

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # on Linux/macOS
# .venv\Scripts\activate   # on Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install ML inference dependencies separately (large)
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu124  # GPU
# pip install torch==2.11.0  # CPU only
pip install monai==1.5.2 totalsegmentator==2.11
```

**Start backend services (PostgreSQL, Redis, MinIO) via Docker, run FastAPI locally:**
```bash
# Start only infrastructure services
docker compose up -d postgres redis minio

# Set environment for local dev
export DATABASE_URL=postgresql+asyncpg://facial_align:password@localhost:5432/facial_align
export REDIS_URL=redis://localhost:6379/0
export MINIO_ENDPOINT=localhost:9000

# Run database migrations
alembic upgrade head

# Start FastAPI with hot reload
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Start Celery worker (separate terminal)
celery -A app.workers.celery_app worker --loglevel=info -Q ingestion,segmentation,planning,evaluation
```

### Frontend (Node.js)

**Requires Node.js 20+**

```bash
cd apps/frontend

# Install dependencies
npm install

# Start dev server (Vite, hot reload)
npm run dev

# Access at http://localhost:5173
# (Port 5173 for dev; 3000 for Docker production build)
```

**Environment for frontend dev:**
```bash
# .env.local (create in apps/frontend/)
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

### ML Pipelines (Python)

```bash
cd pipelines

# Install pipeline dependencies (inherits from backend venv)
pip install -r requirements.txt

# Run individual pipeline stages on local data
python -m dicom_ingestion.main --input /path/to/ct_series/ --case-id test-001
python -m segmentation.main --case-id test-001
python -m mesh_extraction.main --case-id test-001
```

---

## 4. Environment Variables Reference

All variables are set in `.env` at the repository root. Docker Compose reads this file automatically.

### Application

| Variable | Default | Required | Description |
|----------|---------|---------|-------------|
| `APP_NAME` | `facial-align` | No | Application name for logging |
| `APP_ENV` | `development` | No | `development`, `staging`, `production` |
| `DEBUG` | `true` | No | Enable debug mode (verbose logging, no cache) |
| `LOG_LEVEL` | `INFO` | No | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `SECRET_KEY` | *(none)* | **Yes** | JWT signing key; min 32 random chars |

### Database

| Variable | Default | Required | Description |
|----------|---------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://facial_align:password@localhost:5432/facial_align` | **Yes** | PostgreSQL connection string |
| `DATABASE_POOL_SIZE` | `10` | No | SQLAlchemy connection pool size |

### Redis / Celery

| Variable | Default | Required | Description |
|----------|---------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379/0` | **Yes** | Redis connection for caching |
| `CELERY_BROKER_URL` | `redis://localhost:6379/1` | **Yes** | Celery task broker |
| `CELERY_RESULT_BACKEND` | `redis://localhost:6379/2` | **Yes** | Celery result storage |

### Object Storage (MinIO / S3)

| Variable | Default | Required | Description |
|----------|---------|---------|-------------|
| `STORAGE_BACKEND` | `minio` | **Yes** | `minio` or `s3` |
| `MINIO_ENDPOINT` | `localhost:9000` | **Yes** | MinIO server address |
| `MINIO_ACCESS_KEY` | `minioadmin` | **Yes** | MinIO / S3 access key |
| `MINIO_SECRET_KEY` | `minioadmin` | **Yes** | MinIO / S3 secret key |
| `MINIO_BUCKET_DICOM` | `dicom-studies` | No | Bucket for DICOM and processed volumes |
| `MINIO_BUCKET_MESHES` | `mesh-assets` | No | Bucket for meshes, plans, reports |
| `MINIO_BUCKET_MODELS` | `model-registry` | No | Bucket for ML model artifacts |
| `MINIO_USE_SSL` | `false` | No | Set `true` for production S3 |

### ML / Inference

| Variable | Default | Required | Description |
|----------|---------|---------|-------------|
| `MODEL_REGISTRY_PATH` | `/app/models` | No | Local path for cached model weights |
| `INFERENCE_DEVICE` | `cuda:0` | No | `cuda:0` for GPU; `cpu` for CPU-only |
| `SEGMENTATION_MODEL` | `totalsegmentator` | No | Primary segmentation model name |
| `DENTAL_MODEL` | `dental_segmentator` | No | Dental segmentation model name |
| `MAX_INFERENCE_BATCH_SIZE` | `1` | No | Inference batch size (keep at 1 for medical) |
| `INFERENCE_TIMEOUT_SECONDS` | `300` | No | Seconds before inference job times out |

### CORS

| Variable | Default | Required | Description |
|----------|---------|---------|-------------|
| `CORS_ORIGINS` | `http://localhost:3000,http://localhost:5173` | No | Allowed CORS origins (comma-separated) |

### HIPAA / Security

| Variable | Default | Required | Description |
|----------|---------|---------|-------------|
| `ENABLE_AUDIT_LOGGING` | `true` | **Yes** (clinical) | Log all PHI access events |
| `ENABLE_PHI_ENCRYPTION` | `true` | **Yes** (clinical) | Encrypt PHI fields at rest |
| `PHI_ENCRYPTION_KEY` | *(none)* | **Yes** (clinical) | Base64-encoded 256-bit AES key |
| `SESSION_TIMEOUT_MINUTES` | `30` | No | Idle session timeout |
| `MAX_LOGIN_ATTEMPTS` | `5` | No | Account lockout threshold |

### External Services (Optional)

| Variable | Default | Required | Description |
|----------|---------|---------|-------------|
| `ORTHANC_URL` | — | No | Orthanc DICOM server URL (Phase 2) |
| `OHIF_URL` | — | No | OHIF viewer base URL if externally deployed |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | No | MLflow server URL |
| `WANDB_API_KEY` | — | No | Weights & Biases API key (optional) |

---

## 5. Database Setup

### Initial Setup (Docker)

```bash
# Start PostgreSQL
docker compose up -d postgres

# Verify connection
docker compose exec postgres psql -U facial_align -d facial_align -c "\dt"

# Run migrations
docker compose exec backend alembic upgrade head

# Verify schema
docker compose exec postgres psql -U facial_align -d facial_align -c "\dt public.*"
```

### Migration Commands (Alembic)

```bash
# Show current migration state
alembic current

# Show migration history
alembic history

# Apply all pending migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1

# Rollback to a specific revision
alembic downgrade {revision_id}

# Create a new migration (after changing SQLAlchemy models)
alembic revision --autogenerate -m "add_occlusal_analysis_table"
```

### Database Inspection

```bash
# Connect to PostgreSQL
docker compose exec postgres psql -U facial_align -d facial_align

# Useful queries
\dt              # List all tables
\d cases         # Describe cases table
\d audit_log     # Describe audit log

# Check recent cases
SELECT id, status, procedure_type, created_at FROM cases ORDER BY created_at DESC LIMIT 10;

# Check job queue
SELECT id, task_name, status, queued_at, completed_at FROM jobs ORDER BY queued_at DESC LIMIT 20;
```

### Backup (Development)

```bash
# Dump database
docker compose exec postgres pg_dump -U facial_align facial_align > backup_$(date +%Y%m%d).sql

# Restore database
docker compose exec -T postgres psql -U facial_align facial_align < backup_20250101.sql
```

---

## 6. Running Tests

### Backend Tests (pytest)

```bash
# All tests
docker compose run --rm backend pytest tests/ -v

# Unit tests only
docker compose run --rm backend pytest tests/unit/ -v

# Integration tests (requires all services running)
docker compose run --rm backend pytest tests/integration/ -v

# Specific test file
docker compose run --rm backend pytest tests/unit/backend/test_dicom_service.py -v

# With coverage report
docker compose run --rm backend pytest tests/ --cov=app --cov-report=html
# View report: open htmlcov/index.html

# Fast mode (skip slow integration tests)
docker compose run --rm backend pytest tests/ -v -m "not slow"
```

### Local Backend Tests (without Docker)

```bash
cd apps/backend

# Activate virtual environment
source .venv/bin/activate

# Set test environment
export DATABASE_URL=postgresql+asyncpg://facial_align:password@localhost:5432/facial_align_test
export REDIS_URL=redis://localhost:6379/15  # Use DB 15 for tests

# Create test database
docker compose exec postgres psql -U facial_align -c "CREATE DATABASE facial_align_test;"
alembic upgrade head

# Run tests
pytest tests/ -v
```

### Frontend Tests

```bash
cd apps/frontend

# Unit tests (Vitest)
npm run test

# With coverage
npm run test:coverage

# E2E tests (Playwright — requires backend running)
npm run test:e2e
```

### Pipeline Tests

```bash
# Pipeline unit tests use synthetic mini-DICOM data (no real patient data required)
docker compose run --rm backend pytest tests/unit/pipelines/ -v

# Full pipeline integration test (runs segmentation — requires GPU or CPU patience)
docker compose run --rm backend pytest tests/integration/test_full_pipeline.py -v -m slow
```

### Test Data

The repository includes synthetic test DICOM data in `tests/fixtures/` — a minimal 50-slice CT volume with no patient information. This data is sufficient for testing ingestion, preprocessing, and basic pipeline logic. It is not suitable for evaluating segmentation quality.

**Adding test fixtures:**
```bash
# Generate synthetic CT data (Python)
python scripts/generate_test_dicom.py --output tests/fixtures/synthetic_ct_001/
```

---

## 7. Common Troubleshooting

### Docker Issues

**Problem:** `docker compose up` fails with "port already in use"

```bash
# Find what's using the port (example: port 5432 already in use)
lsof -i :5432
kill -9 <PID>

# Or change the port mapping in docker-compose.yml
```

**Problem:** Container exits immediately after starting

```bash
# Check logs for the specific service
docker compose logs backend
docker compose logs worker

# Common causes:
# - Missing environment variable (check .env)
# - Database not yet ready (backend waits, but may timeout)
# - Python import error (check requirements.txt)
```

**Problem:** `docker: Error response from daemon: could not select device driver "nvidia"`

```bash
# NVIDIA Container Toolkit not installed or not configured
# Re-run the toolkit installation steps from Section 1
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### Database Issues

**Problem:** `alembic upgrade head` fails with "relation already exists"

```bash
# The database has a schema but Alembic doesn't know about it
# Option 1: Mark migrations as applied without running them
alembic stamp head

# Option 2: Drop and recreate (development only — destroys data)
docker compose exec postgres dropdb -U facial_align facial_align
docker compose exec postgres createdb -U facial_align facial_align
alembic upgrade head
```

**Problem:** `asyncpg.exceptions.TooManyConnectionsError`

```bash
# Reduce pool size in .env
DATABASE_POOL_SIZE=5

# Or increase PostgreSQL max_connections
docker compose exec postgres psql -U facial_align -c "SHOW max_connections;"
# Edit postgresql.conf if needed
```

### MinIO Issues

**Problem:** `NoSuchBucket` error when uploading

```bash
# Buckets need to be created on first run
# The backend startup script creates them, but it may have failed
docker compose exec minio mc alias set local http://localhost:9000 minioadmin minioadmin
docker compose exec minio mc mb local/dicom-studies
docker compose exec minio mc mb local/mesh-assets
docker compose exec minio mc mb local/model-registry
```

**Problem:** Presigned URL expired (15-minute default)

This is correct behavior. If the frontend shows an asset load error after 15 minutes of inactivity, refresh the page — the frontend will request a new presigned URL.

### Inference / GPU Issues

**Problem:** `RuntimeError: CUDA out of memory`

```bash
# Reduce batch size (already set to 1, so this means the volume is too large)
# TotalSegmentator handles large volumes via chunked inference internally

# Option 1: Force CPU inference for this case
INFERENCE_DEVICE=cpu

# Option 2: Use memory-efficient inference mode
TOTALSEGMENTATOR_FAST=true  # Lower resolution, less memory, less accurate
```

**Problem:** Segmentation jobs time out (status = FAILED, error = "INFERENCE_TIMEOUT")

```bash
# Increase timeout in .env
INFERENCE_TIMEOUT_SECONDS=600  # 10 minutes

# Check GPU utilization during inference
watch nvidia-smi

# If GPU utilization is 0%, the inference service isn't using the GPU
# Verify NVIDIA Container Toolkit is correctly configured
```

**Problem:** Model weights not found / download fails

```bash
# TotalSegmentator downloads weights on first run to MODEL_REGISTRY_PATH
# If the download fails, manually trigger it
docker compose exec inference python -c "import totalsegmentator; totalsegmentator.download_pretrained_weights()"

# If behind a corporate proxy
docker compose exec inference bash -c "https_proxy=http://proxy:port pip install totalsegmentator && python -c 'import totalsegmentator'"
```

### Celery Worker Issues

**Problem:** Tasks stuck in PENDING, never start

```bash
# Check if workers are running
docker compose exec flower celery -A app.workers.celery_app inspect active

# Check Redis connection
docker compose exec backend python -c "from app.core.redis import redis_client; print(redis_client.ping())"

# Restart workers
docker compose restart worker
```

**Problem:** Tasks fail with "Import Error"

```bash
# The worker container may have stale code (if using volume mounts)
docker compose restart worker

# Or rebuild the image
docker compose build worker
docker compose up -d worker
```

### Frontend Issues

**Problem:** "Network Error" when accessing API

```bash
# Verify API is running
curl http://localhost:8000/health

# Check CORS settings in .env
CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# Check browser console for the specific error
```

**Problem:** Three.js scene doesn't load meshes

```bash
# Verify MinIO presigned URL generation works
curl http://localhost:8000/api/v1/meshes/{case_id}/mandible

# The response should include a presigned_url field
# Try fetching that URL directly to see if it works

# If MinIO is unreachable from the browser (different hostname), update:
MINIO_ENDPOINT=localhost:9000  # Must match what the browser can reach
```

### Getting Help

1. Check the logs: `docker compose logs {service_name}`
2. Search GitHub Issues for similar problems
3. Open a new issue with: error message, OS, Docker version, GPU model, logs
