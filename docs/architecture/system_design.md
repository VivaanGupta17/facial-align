# System Architecture — Facial Align

**Version:** 1.0  
**Last Updated:** 2025  
**Status:** Authoritative reference for all architectural decisions

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Service Decomposition](#2-service-decomposition)
3. [Data Flow Overview](#3-data-flow-overview)
4. [Database Schema Overview](#4-database-schema-overview)
5. [Storage Architecture](#5-storage-architecture)
6. [ML Inference Architecture](#6-ml-inference-architecture)
7. [API Design Principles](#7-api-design-principles)
8. [Authentication and Authorization](#8-authentication-and-authorization)
9. [Async Job Processing](#9-async-job-processing)
10. [Frontend-Backend Communication](#10-frontend-backend-communication)
11. [Deployment Topology](#11-deployment-topology)

---

## 1. High-Level Architecture

```
                         ┌─────────────────────────────────────┐
                         │           External Clients           │
                         │  Browser  │  PACS  │  3D Slicer      │
                         └─────┬─────┴──┬─────┴────────┬────────┘
                               │        │              │
                        HTTPS  │  DICOM │         REST API
                               │        │              │
                    ┌──────────▼────────▼──────────────▼──────────┐
                    │              Nginx / Load Balancer            │
                    │         TLS termination, rate limiting         │
                    └──────────┬──────────────────────┬────────────┘
                               │                      │
              ┌────────────────▼────┐   ┌─────────────▼────────────┐
              │   React Frontend    │   │    FastAPI Backend         │
              │   (Next.js/Vite)    │   │    /api/v1/*               │
              │   Port 3000         │   │    Port 8000               │
              │                     │   │                            │
              │ - OHIF DICOM viewer │   │ - Case management          │
              │ - Three.js planning │   │ - Job submission           │
              │ - Zustand state     │   │ - Auth (JWT)               │
              │ - WebSocket client  │   │ - Audit logging            │
              └────────────────────┘   └──────┬─────────────────────┘
                                              │
                          ┌───────────────────┼────────────────────┐
                          │                   │                    │
              ┌───────────▼──────┐  ┌─────────▼──────┐  ┌─────────▼───────┐
              │   Redis Broker   │  │   PostgreSQL    │  │   MinIO         │
              │   Port 6379      │  │   Port 5432     │  │   Port 9000     │
              │                  │  │                 │  │                 │
              │ - Task queues    │  │ - Cases         │  │ - DICOM studies │
              │ - Result backend │  │ - Patients      │  │ - NIfTI volumes │
              │ - WebSocket pub  │  │ - Jobs          │  │ - STL meshes    │
              └───────┬──────────┘  │ - Plans         │  │ - Model weights │
                      │             │ - Audit log     │  │ - Reports       │
                      │             └─────────────────┘  └─────────────────┘
              ┌───────▼──────────────────────────────────────┐
              │                Celery Workers                  │
              │                                               │
              │  ┌──────────────┐  ┌────────────────────┐   │
              │  │  Ingestion   │  │   Segmentation     │   │
              │  │  Worker      │  │   Worker           │   │
              │  │              │  │                    │   │
              │  │ - DICOM→NIfTI│  │ - TotalSegmentator │   │
              │  │ - De-id      │  │ - DentalSegmentator│   │
              │  │ - Validation │  │ - Mesh extraction  │   │
              │  └──────────────┘  └────────────────────┘   │
              │                                               │
              │  ┌──────────────┐  ┌────────────────────┐   │
              │  │  Planning    │  │   Evaluation       │   │
              │  │  Worker      │  │   Worker           │   │
              │  │              │  │                    │   │
              │  │ - Landmarks  │  │ - Dice/Hausdorff   │   │
              │  │ - Occlusion  │  │ - Plan comparison  │   │
              │  │ - Plan gen   │  │ - Outcome logging  │   │
              │  └──────────────┘  └────────────────────┘   │
              └─────────────────────┬────────────────────────┘
                                    │
                     ┌──────────────▼──────────────┐
                     │     Inference Service         │
                     │     TorchServe (GPU)          │
                     │     Port 8080                 │
                     │                               │
                     │ - nnU-Net segmentation        │
                     │ - Landmark heatmap regression │
                     │ - Fragment classification     │
                     │ - Plan scoring model          │
                     └───────────────────────────────┘
```

---

## 2. Service Decomposition

### 2.1 FastAPI Backend (`apps/backend`)

The API is the single entry point for all client interactions. It is stateless and horizontally scalable.

**Responsibilities:**
- Authentication (JWT issuance and validation)
- Request validation (Pydantic schemas)
- Case and patient CRUD
- Job submission to Celery
- Signed URL generation for MinIO object access
- Audit log writes for every PHI-touching operation
- WebSocket multiplexer for job status push

**Key endpoint groups:**

| Prefix | Purpose |
|--------|---------|
| `/api/v1/auth` | Login, token refresh, logout |
| `/api/v1/cases` | Case creation, listing, retrieval |
| `/api/v1/studies` | DICOM study management |
| `/api/v1/jobs` | Job status, retry, cancellation |
| `/api/v1/plans` | Surgical plan CRUD and approval |
| `/api/v1/meshes` | Mesh retrieval and STL download |
| `/api/v1/evaluations` | Accuracy metric retrieval |
| `/api/v1/admin` | User management, model versioning |
| `/ws/jobs/{job_id}` | WebSocket job status stream |

**Key design decisions:**
- All endpoints are async (FastAPI + asyncpg)
- No direct ML inference in API handlers — always enqueue to Celery
- All file uploads are streamed to MinIO; API never buffers large files in memory
- OpenAPI schema auto-generated; used to generate TypeScript client types

### 2.2 Celery Workers (`apps/backend/app/workers`)

Workers are the execution layer for all long-running tasks. Each worker type can be scaled independently.

**Worker queues and task types:**

| Queue | Worker | Tasks | GPU Required |
|-------|--------|-------|-------------|
| `ingestion` | Ingestion worker | DICOM parsing, de-identification, NIfTI conversion, validation | No |
| `segmentation` | Segmentation worker | TotalSegmentator, DentalSegmentator, mesh extraction | Yes (or slow) |
| `planning` | Planning worker | Landmark detection, occlusion analysis, plan generation | Yes |
| `evaluation` | Evaluation worker | Dice/Hausdorff scoring, outcome comparison, report generation | No |

**Task chaining pattern:**

```
dicom_upload
    └── ingest_dicom_series
            └── preprocess_volume
                    └── run_segmentation
                            └── extract_meshes
                                    └── detect_landmarks
                                            └── compute_occlusion
                                                    └── generate_plan_candidates
                                                            └── notify_surgeon
```

Each task is idempotent (can be retried safely) and stores its output in MinIO before triggering the next task. If a task fails, the chain breaks at that point; earlier outputs are preserved.

### 2.3 Inference Service (`services/inference`)

A standalone TorchServe instance that exposes model inference as HTTP endpoints. Celery workers call this service rather than loading models directly, enabling:
- GPU resource isolation (one process manages the GPU)
- Model hot-swapping without worker restart
- Inference request batching
- Separate scaling from CPU-bound workers

**Model handlers:**

| Handler | Input | Output | Latency (P50, A100) |
|---------|-------|--------|---------------------|
| `totalsegmentator` | NIfTI volume (1mm³) | Multi-label mask | 45–90s |
| `dental_segmentator` | NIfTI crop (dental ROI) | Multi-label mask | 20–40s |
| `landmark_detector` | NIfTI volume | Coordinate + confidence JSON | 15–30s |
| `plan_scorer` | Plan JSON + geometry | Confidence scores | <1s |
| `fragment_classifier` | Bone component mesh | Fragment class + confidence | 5–10s |

### 2.4 Preprocessing Service (`services/preprocessing`)

Stateless service that converts DICOM series to standardized NIfTI volumes. Called by ingestion workers.

**Processing steps:**
1. `pydicom` series reader — indexes files, extracts metadata, validates series completeness
2. `SimpleITK` series assembler — constructs oriented 3D volume from sorted slices
3. Orientation normalization — `DICOMOrient` to LPS canonical coordinates
4. Isotropic resampling — `ResampleImageFilter` to 1.0mm³ (or 0.4mm³ for CBCT dental)
5. HU windowing — bone: [300, 2000], soft tissue: [−150, 250]
6. Normalization — [0, 1] float32
7. Metadata extraction — patient demographics, acquisition parameters to JSON sidecar

### 2.5 Registration Service (`services/registration`)

Handles spatial alignment between modalities:
- CT ↔ intraoral scan (ICP-based surface registration)
- CT ↔ post-operative CT (deformable registration for outcome comparison)
- Pre-op ↔ planned state (rigid transform computation)

Uses SimpleITK registration framework with multi-resolution pyramid strategy.

### 2.6 Mesh Generation Service (`services/mesh_generation`)

Converts segmentation masks to surgical-quality 3D meshes:
1. Marching cubes (`scikit-image.measure.marching_cubes`)
2. Laplacian smoothing (PyVista)
3. Mesh decimation to target polygon count
4. Connected component filtering (remove small islands)
5. Watertight check (trimesh)
6. Export: STL (3D printing), OBJ (WebGL rendering), GLTF (Three.js)

### 2.7 Frontend (`apps/frontend`)

React + TypeScript single-page application. Served statically in production.

**Key component areas:**

| Component Group | Contents |
|----------------|---------|
| `components/viewer` | OHIF integration, Cornerstone3D wrappers |
| `components/planning` | Three.js scene, bone mesh renderers, gizmo controls |
| `components/dashboard` | Case list, job status, notifications |
| `components/common` | Design system components (buttons, modals, alerts) |
| `stores/` | Zustand stores (caseStore, planningStore, authStore) |

---

## 3. Data Flow Overview

The canonical data flow from CT upload to surgical plan:

```
[1] DICOM Upload
    Browser → POST /api/v1/cases (multipart or presigned URL)
    API → MinIO: store raw DICOM files in dicom-studies/{study_uid}/raw/
    API → PostgreSQL: create Case record (status=UPLOADED)
    API → Celery: enqueue ingest_dicom_series task

[2] Ingestion + Preprocessing
    Worker: read DICOM from MinIO → de-identify → validate series
    Worker: SimpleITK assembly → orientation → resample → HU window
    Worker → MinIO: store preprocessed.nii.gz in dicom-studies/{study_uid}/processed/
    Worker → PostgreSQL: update Case (status=PREPROCESSED, metadata JSON)

[3] Segmentation
    Worker: load NIfTI from MinIO → call Inference Service
    Inference: TotalSegmentator (skull pass) → DentalSegmentator (dental pass)
    Inference → return multi-label mask
    Worker → MinIO: store segmentation.nii.gz + uncertainty_map.nii.gz
    Worker → PostgreSQL: update Case (status=SEGMENTED)

[4] Mesh Extraction
    Worker: load segmentation mask → marching cubes per label
    Worker: smooth → decimate → validate → export per-structure STL
    Worker → MinIO: store {structure}.stl in mesh-assets/{case_id}/
    Worker → PostgreSQL: insert Mesh records

[5] Landmark Detection
    Worker: load NIfTI → call Inference Service (landmark_detector)
    Inference → return [{name, x, y, z, confidence}] JSON
    Worker → PostgreSQL: insert Landmark records with confidence values

[6] Occlusion Analysis
    Worker: load upper/lower dental mesh → compute ICP/intercuspal contacts
    Worker: classify Angle class, compute overjet/overbite
    Worker → PostgreSQL: insert OcclusalAnalysis record

[7] Plan Generation
    Worker: load landmarks + occlusal analysis + pathology classification
    Worker: apply rule-based movement templates (Le Fort I, BSSO, genioplasty)
    Worker: evaluate each candidate plan against occlusal constraint engine
    Worker: call Inference Service (plan_scorer) for confidence ranking
    Worker → PostgreSQL: insert PlanCandidates (top 3)

[8] Surgeon Review
    API: surgeon polls or receives WebSocket notification (status=PLAN_READY)
    Frontend: loads bone meshes from MinIO via presigned URL
    Frontend: renders Three.js planning scene with overlaid uncertainty heat maps
    Surgeon: accepts plan / modifies movement vectors / overrides segments
    API: record PlanModifications (for audit and training data)

[9] Plan Finalization
    Surgeon: clicks Approve
    API → PostgreSQL: update Plan (status=APPROVED, surgeon_id, approved_at)
    API → Celery: enqueue report_generation task
    Worker: generate PDF planning report → MinIO
    Worker: generate modified DICOM (for navigation) → MinIO
    API: notify surgeon — plan ready for download / print ordering
```

For the complete per-stage data flow specification, see `docs/architecture/data_flow.md`.

---

## 4. Database Schema Overview

All tables live in the `facial_align` PostgreSQL database. Alembic manages migrations.

### Core Tables

```sql
-- Cases: the primary work unit
CREATE TABLE cases (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id      UUID REFERENCES patients(id),
    created_by      UUID REFERENCES users(id),
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now(),
    status          case_status NOT NULL DEFAULT 'UPLOADED',
    procedure_type  TEXT,          -- 'ORTHOGNATHIC', 'TRAUMA', 'RECONSTRUCTION'
    notes           TEXT,
    metadata        JSONB          -- acquisition params, scanner model, etc.
);

-- Patients: de-identified demographics
CREATE TABLE patients (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id     TEXT UNIQUE,   -- institution MRN (hashed)
    year_of_birth   INTEGER,
    sex             TEXT,
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- DICOM Studies
CREATE TABLE dicom_studies (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id         UUID REFERENCES cases(id),
    study_uid       TEXT UNIQUE NOT NULL,   -- DICOM StudyInstanceUID
    modality        TEXT NOT NULL,          -- CT, CBCT
    series_count    INTEGER,
    slice_thickness DECIMAL(5,3),           -- mm
    storage_path    TEXT NOT NULL,          -- MinIO object prefix
    acquired_at     TIMESTAMPTZ,
    status          study_status NOT NULL DEFAULT 'PENDING'
);

-- Pipeline Jobs
CREATE TABLE jobs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id         UUID REFERENCES cases(id),
    task_name       TEXT NOT NULL,          -- Celery task name
    celery_task_id  TEXT UNIQUE,
    status          job_status NOT NULL DEFAULT 'PENDING',
    queued_at       TIMESTAMPTZ DEFAULT now(),
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    error_message   TEXT,
    result_path     TEXT,                   -- MinIO path to output
    worker_host     TEXT
);

-- Segmentations
CREATE TABLE segmentations (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id         UUID REFERENCES cases(id) UNIQUE,
    job_id          UUID REFERENCES jobs(id),
    model_name      TEXT NOT NULL,          -- 'totalsegmentator', 'dental_segmentator'
    model_version   TEXT NOT NULL,
    storage_path    TEXT NOT NULL,          -- MinIO path to segmentation.nii.gz
    uncertainty_path TEXT,                  -- MinIO path to uncertainty_map.nii.gz
    label_map       JSONB,                  -- {label_index: structure_name}
    dice_scores     JSONB,                  -- {structure: dice_value}
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- Meshes
CREATE TABLE meshes (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id         UUID REFERENCES cases(id),
    structure_name  TEXT NOT NULL,          -- 'mandible', 'maxilla', 'tooth_18', etc.
    fdi_number      INTEGER,                -- for individual teeth
    storage_path    TEXT NOT NULL,          -- MinIO path to .stl
    gltf_path       TEXT,                   -- MinIO path to .gltf for WebGL
    vertex_count    INTEGER,
    face_count      INTEGER,
    created_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE(case_id, structure_name)
);

-- Landmarks
CREATE TABLE landmarks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id         UUID REFERENCES cases(id),
    name            TEXT NOT NULL,          -- 'nasion', 'A-point', 'pogonion', etc.
    x_mm            DECIMAL(8,3) NOT NULL,  -- LPS coordinate space
    y_mm            DECIMAL(8,3) NOT NULL,
    z_mm            DECIMAL(8,3) NOT NULL,
    confidence      DECIMAL(4,3),           -- model confidence [0,1]
    is_manual       BOOLEAN DEFAULT FALSE,  -- surgeon-adjusted?
    created_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE(case_id, name)
);

-- Surgical Plans
CREATE TABLE surgical_plans (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id         UUID REFERENCES cases(id),
    version         INTEGER NOT NULL DEFAULT 1,
    status          plan_status NOT NULL DEFAULT 'CANDIDATE',
    confidence      DECIMAL(4,3),
    movements       JSONB NOT NULL,         -- [{segment, dx, dy, dz, rx, ry, rz}]
    constraint_violations JSONB,            -- [{type, description, severity}]
    approved_by     UUID REFERENCES users(id),
    approved_at     TIMESTAMPTZ,
    report_path     TEXT,                   -- MinIO path to PDF report
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- Plan Modifications (audit + training data)
CREATE TABLE plan_modifications (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plan_id         UUID REFERENCES surgical_plans(id),
    user_id         UUID REFERENCES users(id),
    modified_at     TIMESTAMPTZ DEFAULT now(),
    modification_type TEXT,                 -- 'MOVEMENT_ADJUST', 'LANDMARK_OVERRIDE', etc.
    before_state    JSONB,
    after_state     JSONB,
    delta           JSONB                   -- structured diff
);

-- Audit Log (append-only, HIPAA required)
CREATE TABLE audit_log (
    id              BIGSERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ DEFAULT now() NOT NULL,
    user_id         UUID,
    action          TEXT NOT NULL,          -- 'CASE_VIEW', 'PLAN_APPROVED', etc.
    resource_type   TEXT,
    resource_id     UUID,
    ip_address      INET,
    user_agent      TEXT,
    details         JSONB
) PARTITION BY RANGE (timestamp);
```

### Status Enumerations

```sql
CREATE TYPE case_status AS ENUM (
    'UPLOADED', 'PREPROCESSING', 'PREPROCESSED',
    'SEGMENTING', 'SEGMENTED', 'PLANNING', 'PLAN_READY',
    'APPROVED', 'ARCHIVED', 'FAILED'
);

CREATE TYPE job_status AS ENUM (
    'PENDING', 'STARTED', 'RETRY', 'SUCCESS', 'FAILURE', 'REVOKED'
);

CREATE TYPE plan_status AS ENUM (
    'CANDIDATE', 'UNDER_REVIEW', 'APPROVED', 'REJECTED', 'SUPERSEDED'
);
```

---

## 5. Storage Architecture

### MinIO Bucket Structure

MinIO provides S3-compatible object storage. Three primary buckets with distinct access patterns:

**`dicom-studies`** — Raw and preprocessed imaging data
```
dicom-studies/
├── {study_uid}/
│   ├── raw/                    # Original DICOM files (unchanged, encrypted)
│   │   ├── series_{uid}/
│   │   │   ├── IM-0001.dcm
│   │   │   └── ...
│   ├── processed/              # Pipeline-ready volumes
│   │   ├── volume.nii.gz       # Resampled, oriented, HU-windowed
│   │   ├── volume_meta.json    # Acquisition metadata sidecar
│   │   └── deidentification_report.json
│   └── archive/                # Long-term retention (Glacier equivalent)
```

**`mesh-assets`** — 3D geometry for each case
```
mesh-assets/
├── {case_id}/
│   ├── segmentation/
│   │   ├── segmentation.nii.gz          # Multi-label mask
│   │   └── uncertainty_map.nii.gz       # Per-voxel confidence
│   ├── meshes/
│   │   ├── mandible.stl
│   │   ├── mandible.gltf
│   │   ├── maxilla.stl
│   │   ├── tooth_18.stl                 # FDI notation per tooth
│   │   └── ...
│   └── reports/
│       ├── surgical_plan_{plan_id}.pdf
│       └── navigation_export_{plan_id}.dcm
```

**`model-registry`** — Versioned ML model artifacts
```
model-registry/
├── totalsegmentator/
│   └── v2.11/
│       ├── model.mar                    # TorchServe archive
│       └── metadata.json
├── dental_segmentator/
│   └── v1.0/
│       └── model.mar
├── landmark_detector/
│   └── v1.0/
│       └── model.mar
└── plan_scorer/
    └── v0.1/                            # Phase 2 model
        └── model.mar
```

### Storage Policies

| Bucket | Encryption | Retention | Access Pattern |
|--------|-----------|-----------|----------------|
| `dicom-studies` | AES-256 SSE | 10 years (HIPAA) | Write once (raw), read many (processed) |
| `mesh-assets` | AES-256 SSE | Case lifetime + 7 years | Frequent read during planning |
| `model-registry` | AES-256 SSE | Indefinite | Read at model load time |

All access to clinical buckets requires presigned URLs with 15-minute expiry. Direct bucket access is blocked for all non-backend principals.

---

## 6. ML Inference Architecture

### Model Registry

Models are versioned and stored in MinIO `model-registry`. The `ModelVersion` table in PostgreSQL tracks:
- Model name and semantic version
- MinIO path to `.mar` archive
- Validation metrics (Dice, Hausdorff, landmark error)
- Deployment status (staging / production / deprecated)
- Deployment date and who promoted it

### TorchServe Deployment

TorchServe runs as a separate Docker service with exclusive GPU access:

```
┌─────────────────────────────────────────────┐
│            TorchServe (GPU service)          │
│                                             │
│  Management API (port 8081)                 │
│   - Load/unload models                      │
│   - Model version management               │
│                                             │
│  Inference API (port 8080)                  │
│   POST /predictions/{model_name}            │
│                                             │
│  Loaded Models (GPU VRAM):                  │
│   - totalsegmentator (primary)              │
│   - dental_segmentator (primary)            │
│   - landmark_detector (primary)             │
│   - plan_scorer (secondary; CPU fallback)   │
└─────────────────────────────────────────────┘
```

### Inference Request Flow

```python
# Celery worker calls inference service
async def run_segmentation(volume_path: str) -> dict:
    # Download volume from MinIO
    volume_bytes = minio_client.get_object("dicom-studies", volume_path)

    # Call inference service
    response = await httpx.post(
        f"{INFERENCE_URL}/predictions/totalsegmentator",
        content=volume_bytes,
        headers={"Content-Type": "application/octet-stream"},
        timeout=300.0
    )

    # Response: {segmentation: base64(nii.gz), uncertainty: base64(nii.gz), metadata: {...}}
    return response.json()
```

### Model Versioning and Promotion

New model versions follow a staged rollout:
1. **Training complete** → upload `.mar` to `model-registry/{name}/v{x.y}/`
2. **Staging** → load in non-production TorchServe; run validation suite
3. **Evaluation** → compare against current production model on held-out test set
4. **Promotion** → update `ModelVersion.deployment_status` to `production`; reload TorchServe
5. **Deprecation** → old version stays in MinIO; removed from TorchServe after 30 days

This process is the foundation of the Predetermined Change Control Plan (PCCP) required for FDA post-market model updates.

---

## 7. API Design Principles

### REST Conventions

- **Resource-oriented** — URLs identify resources, not actions (`/cases/{id}` not `/getCase`)
- **Async by default** — mutation endpoints return `202 Accepted` with a job ID; clients poll `/jobs/{id}` or listen on WebSocket
- **Versioned** — all endpoints under `/api/v1/`; breaking changes increment version
- **Idempotent** — PUT and PATCH operations are safe to retry
- **Cursor-based pagination** — list endpoints use `?cursor=` not `?page=` for stability with concurrent writes

### Error Response Format

```json
{
  "error": {
    "code": "CASE_NOT_FOUND",
    "message": "Case 550e8400-e29b-41d4-a716-446655440000 does not exist",
    "detail": null,
    "request_id": "req_01J2..."
  }
}
```

### Async Job Response Pattern

```json
// POST /api/v1/cases → 202 Accepted
{
  "case_id": "550e8400-e29b-41d4-a716-446655440000",
  "job_id": "job_01J2...",
  "status": "PENDING",
  "status_url": "/api/v1/jobs/job_01J2...",
  "websocket_url": "/ws/jobs/job_01J2..."
}
```

### Data Contracts

All inter-service data exchange is governed by JSON Schema contracts in `data_contracts/`. This includes:
- `dicom_series_manifest.json` — what ingestion worker produces
- `segmentation_result.json` — what inference service returns
- `surgical_plan.json` — plan object schema
- `landmark_set.json` — landmark array with confidence

---

## 8. Authentication and Authorization

### Authentication

- **JWT-based** — access tokens (15 min), refresh tokens (7 days)
- **Tokens issued** at `/api/v1/auth/login` (email + password)
- **HTTPS only** — tokens never transmitted over plain HTTP
- **Session invalidation** — refresh token revocation list in Redis
- **MFA** — TOTP second factor (Phase 2; required for clinical deployment)

### Authorization

Role-based access control (RBAC) with four roles:

| Role | Permissions |
|------|------------|
| `ADMIN` | Full system access; user management; model promotion |
| `SURGEON` | Create cases; view/modify own cases; approve plans |
| `RESIDENT` | Create cases; view own cases; cannot approve plans |
| `VIEWER` | Read-only access to shared cases |

Authorization is enforced at the API layer via FastAPI dependency injection. Resource-level policies: a surgeon can only access their own institution's cases unless explicitly shared.

### HIPAA Audit Requirements

Every request that accesses PHI triggers an audit log entry:

```python
@audit_log(action="CASE_VIEW", resource_type="case")
async def get_case(case_id: UUID, user: User = Depends(get_current_user)):
    ...
```

Audit log entries include: timestamp, user_id, action, resource_type, resource_id, IP address, user agent. The audit log table is append-only (no DELETE or UPDATE permissions).

---

## 9. Async Job Processing

### Celery Configuration

```python
# Task routing by queue
task_routes = {
    "workers.ingestion.*": {"queue": "ingestion"},
    "workers.segmentation.*": {"queue": "segmentation"},
    "workers.planning.*": {"queue": "planning"},
    "workers.evaluation.*": {"queue": "evaluation"},
}

# Retry policy for transient failures
task_autoretry_for = (TransientError,)
task_max_retries = 3
task_retry_backoff = True
task_retry_backoff_max = 300  # 5 minutes max
```

### Task Chain Example

```python
from celery import chain
from workers.ingestion import ingest_dicom, preprocess_volume
from workers.segmentation import run_segmentation, extract_meshes
from workers.planning import detect_landmarks, compute_occlusion, generate_plans

def submit_case_pipeline(case_id: str, study_uid: str) -> str:
    pipeline = chain(
        ingest_dicom.s(case_id, study_uid),
        preprocess_volume.s(),
        run_segmentation.s(),
        extract_meshes.s(),
        detect_landmarks.s(),
        compute_occlusion.s(),
        generate_plans.s(),
    )
    result = pipeline.apply_async()
    return result.id
```

### Job Status Tracking

Job status is dual-tracked:
1. **PostgreSQL** — durable status; survives Redis restart
2. **Redis** — fast polling; Celery result backend

WebSocket push: when a job status changes, the worker publishes to a Redis pub/sub channel. The API WebSocket handler subscribes and pushes updates to connected clients.

---

## 10. Frontend-Backend Communication

### HTTP Client

The frontend uses a typed API client generated from the OpenAPI schema (`apps/frontend/src/lib/api.ts`). All requests include:
- `Authorization: Bearer {access_token}`
- `X-Request-ID` (client-generated UUID for tracing)

### WebSocket Job Status

```typescript
// Subscribe to job updates
const ws = new WebSocket(`/ws/jobs/${jobId}`);
ws.onmessage = (event) => {
  const update = JSON.parse(event.data);
  // {job_id, status, progress_pct, current_stage, error}
  planningStore.updateJobStatus(update);
};
```

### 3D Asset Streaming

Large mesh files (STL/GLTF) are not served through the API — the frontend fetches them directly from MinIO via presigned URLs:

```typescript
// Get presigned URL for mesh asset
const { url } = await api.getMeshUrl(caseId, "mandible");
// Direct fetch from MinIO — bypasses API server
const gltf = await gltfLoader.loadAsync(url);
```

### State Management

Zustand stores:
- `caseStore` — current case, jobs, status
- `planningStore` — active plan, bone meshes, movements, constraint violations
- `viewerStore` — OHIF viewer state, active series, window/level
- `authStore` — user, token, role

---

## 11. Deployment Topology

### Single-Node Development (docker-compose)

```yaml
# All services on one host; shared network
services:
  backend:      ports: ["8000:8000"]
  frontend:     ports: ["3000:3000"]
  worker:       # All queues in one worker process
  inference:    ports: ["8080:8080"],  deploy.resources.reservations.devices: [gpu]
  postgres:     ports: ["5432:5432"]
  redis:        ports: ["6379:6379"]
  minio:        ports: ["9000:9000", "9001:9001"]
  mlflow:       ports: ["5000:5000"]
  flower:       ports: ["5555:5555"]
```

CPU fallback: set `INFERENCE_DEVICE=cpu` in `.env` to run inference without GPU. Segmentation times increase to 5–20 minutes per case.

### Multi-Service Production (Kubernetes)

In production, each service scales independently:

```
Namespace: facial-align-prod
├── Deployments
│   ├── backend          (3 replicas, CPU-optimized nodes)
│   ├── frontend         (2 replicas, static serving via nginx)
│   ├── worker-ingestion (2 replicas, CPU nodes)
│   ├── worker-segment   (2 replicas, GPU nodes — A10G or better)
│   ├── worker-planning  (2 replicas, GPU nodes)
│   └── inference        (1 replica, GPU node, exclusive)
├── StatefulSets
│   ├── postgres         (1 replica + read replica)
│   └── redis            (1 primary + 1 replica)
└── External
    ├── MinIO            (dedicated object storage cluster or AWS S3)
    └── Nginx Ingress    (TLS termination, rate limiting)
```

**GPU requirements:** Segmentation worker and inference service require NVIDIA GPU with ≥16 GB VRAM. A10G (24 GB) or A100 (40/80 GB) recommended for production.

### Kubernetes Manifests

Base manifests are in `infra/kubernetes/`. Production deployments use Kustomize overlays for environment-specific configuration.

### Health Checks

All services expose `/health` (liveness) and `/ready` (readiness) endpoints. The inference service readiness check includes a model warmup — it won't accept traffic until at least the primary segmentation model is loaded.
