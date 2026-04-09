# Facial Align

**AI-native craniofacial surgical planning — from CT to surgical plan in minutes, not days.**

Facial Align is an open research platform for virtual surgical planning (VSP) in cranio-maxillofacial (CMF) surgery. It replaces the "engineer-drives-the-software, surgeon-approves" service model with an AI system that proposes plans autonomously, surfaces uncertainty explicitly, and keeps the surgeon in the decision loop. The AI is not a button — it is the planning engine.

> **51,000+ lines of source code** across 216 files — Python backend, TypeScript frontend, ML pipelines, and comprehensive documentation.

---

## Mission

Current VSP workflows require scheduling a live session with a third-party biomedical engineer, waiting 5–10 days for physical deliverables, and receiving a binary plan with no uncertainty signal. Facial Align eliminates that bottleneck: a CMF surgeon uploads a CT, receives AI-generated segmentation, landmark detection, and a ranked set of surgical plans — each annotated with confidence levels and constraint violations — within minutes. Every case processed feeds a training data flywheel that improves future predictions.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           FACIAL ALIGN                                   │
│                                                                          │
│   ┌──────────────┐      ┌────────────────────────────────────────────┐  │
│   │   Frontend   │      │              Backend API                   │  │
│   │  React + TS  │◄────►│           FastAPI (Python)                 │  │
│   │  R3F Viewer  │  WS  │        /api/v1/* endpoints                 │  │
│   │  Ceph Overlay│◄────►│  Middleware: Audit │ RateLimit │ Tracing   │  │
│   └──────────────┘      └──────────────┬─────────────────────────────┘  │
│                                        │ enqueue jobs                    │
│                                 ┌──────▼──────┐                         │
│                                 │    Redis    │                         │
│                                 │  Broker +   │                         │
│                                 │  Pub/Sub    │                         │
│                                 └──────┬──────┘                         │
│                                        │                                │
│              ┌─────────────────────────┼──────────────────────┐         │
│              │                         │                      │         │
│     ┌────────▼──────┐    ┌─────────────▼────┐   ┌────────────▼──────┐  │
│     │   Celery      │    │   Celery         │   │  Celery           │  │
│     │   Worker:     │    │   Worker:        │   │  Worker:          │  │
│     │   Ingestion   │    │   Segmentation   │   │  Planning         │  │
│     │   + DeID      │    │   + Mesh + QC    │   │  + Occlusion      │  │
│     │   + QC        │    │   + Ceph         │   │  + Evaluation     │  │
│     └────────┬──────┘    └─────────────┬────┘   └────────────┬──────┘  │
│              │                         │                      │         │
│              └─────────────────────────┼──────────────────────┘         │
│                                        │                                │
│                    ┌───────────────────┼───────────────────┐            │
│                    │                   │                   │            │
│           ┌────────▼────┐   ┌──────────▼───┐   ┌──────────▼─────────┐  │
│           │  PostgreSQL │   │  MinIO/S3    │   │  Model Registry    │  │
│           │  + Alembic  │   │  DICOM/NIfTI │   │  TotalSegmentator  │  │
│           │  + Audit    │   │  /GLB/Models │   │  nnU-Net / Custom  │  │
│           └─────────────┘   └──────────────┘   └────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## What Works Today

| Capability | Implementation | Lines |
|-----------|---------------|-------|
| **DICOM pipeline** | Upload → parse → de-identify (PS3.15, 71 PHI tags, HMAC pseudonymization) → quality control (8 checks, A/B/C/F grading) → volume reconstruction | ~2,000 |
| **CT quality control** | Slice thickness, gap detection, spacing consistency, FOV, motion artifacts (inter-slice variance), bone contrast (HU histogram), calibration | 1,021 |
| **Segmentation** | TotalSegmentator adapter with structure mapping + confidence; nnU-Net and dental adapters ready for weights | ~1,800 |
| **Cephalometric analysis** | 24 anatomical landmarks detected from masks via anatomical heuristics; SNA/SNB/ANB angles; full CephalometricAnalysis dataclass | 903 |
| **Facial symmetry** | Midsagittal plane detection (PCA + iterative refinement); per-structure asymmetry maps via EDT; clinical grading | 828 |
| **Fracture reduction** | ICP baseline with Open3D; bilateral symmetry enforcement; learned model interface (PointNet++ / SE(3) architecture) | ~1,200 |
| **Occlusion engine** | Geometric occlusal model; constraint satisfaction; molar classification (Class I/II/III); splint design spec generation | ~600 |
| **Plan evaluation** | Per-fragment alignment metrics, symmetry scoring, occlusion assessment, condylar assessment, AO CMF hardware recommendations, composite grading | 873 |
| **Surgical sequencing** | Graph-based optimal reduction order using clinical priority, anatomical templates, fragment-specific instructions | 498 |
| **Clinical reporting** | Structured Markdown surgical planning reports with all sections | 391 |
| **Mesh pipeline** | Marching cubes + Gaussian smoothing, quality-preserving decimation, multi-resolution LODs, PBR material assignment, 24-color anatomical map, watertightness/manifold checks, repair pipeline | ~1,600 |
| **3D viewer** | React Three Fiber with fragment manipulation, distance/angle measurement tools, cross-section viewer (axial/coronal/sagittal), window/level presets | ~2,500 |
| **Cephalometric overlay** | SVG lateral skull, 17 landmarks, 11 measurements, color-coded normal/borderline/abnormal | 540 |
| **Case dashboard** | Case list with filters, detailed case view, upload page, surgeon review workspace | ~1,400 |
| **API** | FastAPI with 7 endpoint groups (cases, DICOM, segmentation, planning, viewer, jobs, health), WebSocket real-time updates | ~2,200 |
| **Middleware** | HIPAA audit logging, token bucket rate limiting (Redis), request tracing, global error handling | 1,871 |
| **CLI** | Case management, model ops, pipeline execution, admin tools — 7 command groups, 33 commands | ~4,800 |
| **SDK** | Typed async Python client with retry, job polling, typed response objects | 612 |
| **Data contracts** | 9 Pydantic v2 schemas with clinical validation, computed properties, cross-field checks | ~3,500 |
| **Tests** | 331 test functions across 22 files; comprehensive fixtures (DICOM, mesh, plan, case) | ~7,700 |
| **Documentation** | Architecture, PRD, clinical workflows, regulatory, evaluation plan, 5 research documents | ~11,000 |

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **Segmentation** | TotalSegmentator v2 | Apache-2.0, covers 80%+ CMF anatomy out of box |
| **3D Viewer** | React Three Fiber | Mesh manipulation (not just DICOM viewing — why not OHIF) |
| **Backend** | FastAPI + Celery | Async ML inference, structured endpoints, OpenAPI |
| **Database** | PostgreSQL + Alembic | Relational case tracking, migration management |
| **Object Store** | MinIO (S3-compatible) | DICOM, NIfTI, GLB, model weights |
| **Frontend** | React 18 + TypeScript + Zustand | Type-safe, fast state management |
| **Mesh** | trimesh + scikit-image | Marching cubes, mesh ops, GLB/STL export |
| **Registration** | Open3D + scipy | ICP, FPFH, KD-tree nearest neighbor |
| **Containers** | Docker Compose | Full stack dev: backend, frontend, PostgreSQL, Redis, MinIO, Celery |
| **CI/CD** | GitHub Actions | Lint, type-check, test on every push |

---

## Quick Start

### Docker (recommended)

```bash
git clone https://github.com/VivaanGupta17/facial-align.git
cd facial-align

cp .env.example .env
# Edit .env — set SECRET_KEY and PHI_ENCRYPTION_KEY

# Start all services
make dev-up

# Run database migrations
make db-migrate

# Verify
make health
```

### Local development

```bash
# Backend
cd apps/backend
pip install -e ".[dev]"
uvicorn app.main:app --reload

# Frontend
cd apps/frontend
npm install
npm run dev

# CLI
pip install -e ".[cli]"
facial-align --help
```

### Service endpoints

| Service | URL |
|---------|-----|
| API + Docs | http://localhost:8000/docs |
| Frontend | http://localhost:3000 |
| MinIO Console | http://localhost:9001 |

---

## CLI

```bash
# Case management
facial-align case list --status planning
facial-align case import-dicom /path/to/dicom --auto-qc --auto-deidentify
facial-align case export FA-12345678 ./output

# Model management
facial-align model list
facial-align model download totalsegmentator
facial-align model benchmark totalsegmentator --iterations 10

# Pipeline execution
facial-align pipeline run FA-12345678
facial-align pipeline run FA-12345678 --step segmentation --dry-run
facial-align pipeline evaluate plan.json

# Admin
facial-align admin health
facial-align admin stats
facial-align admin audit-log --since 2026-04-01 --action CREATE
facial-align admin db-seed
```

---

## Project Structure

```
facial-align/
├── apps/
│   ├── backend/                    # FastAPI application (11,100+ lines)
│   │   ├── app/
│   │   │   ├── api/v1/endpoints/   # REST + WebSocket endpoints
│   │   │   ├── core/               # Config, security, logging, exceptions
│   │   │   ├── middleware/          # Audit, rate limiting, tracing, errors
│   │   │   ├── models/             # SQLAlchemy ORM models
│   │   │   ├── schemas/            # Pydantic request/response schemas
│   │   │   ├── services/           # Business logic (6 service domains)
│   │   │   └── workers/            # Celery task definitions
│   │   └── alembic/                # Database migrations
│   └── frontend/                   # React + TypeScript (10,400+ lines)
│       └── src/
│           ├── components/         # Viewer, planning, common UI
│           │   ├── viewer/         # 3D viewer, measurements, cross-section
│           │   ├── planning/       # Reduction, occlusion, cephalometrics
│           │   └── common/         # Error boundary, empty states, metrics
│           ├── hooks/              # Keyboard shortcuts, WebSocket, data
│           ├── lib/                # API client, geometry, validation, errors
│           ├── pages/              # Dashboard, case detail, upload
│           ├── stores/             # Zustand state (case, planning, viewer)
│           └── types/              # Medical type definitions
├── services/                       # ML services (7,800+ lines)
│   ├── inference/                  # Model registry + adapters
│   │   └── adapters/              # TotalSeg, dental, landmark, symmetry
│   ├── preprocessing/              # CT preprocessor, QC, de-identification
│   ├── mesh_generation/            # Mesh extraction, smoothing, export
│   ├── evaluation/                 # Plan evaluator, report gen, sequencing
│   ├── benchmark/                  # Pipeline profiling framework
│   └── registration/               # CT-to-scan registration
├── pipelines/                      # End-to-end ML pipeline stages
├── data_contracts/                 # 9 canonical Pydantic schemas (3,500+ lines)
├── cli/                            # Professional CLI tools (4,800+ lines)
├── sdk/                            # Python SDK client
├── tests/                          # 331 tests + fixtures (7,700+ lines)
├── docs/                           # 11 documentation files
├── research/                       # 6 research documents
├── examples/                       # Demo data, exploration notebooks
├── scripts/                        # Model download, benchmarks, seeding
└── infra/docker/                   # Docker Compose + Dockerfiles
```

---

## What Requires Model Weights

These modules have complete interfaces, data contracts, and baseline algorithms — they need trained weights to achieve clinical accuracy:

| Module | Architecture | Training Data Needed |
|--------|-------------|---------------------|
| Learned fracture reduction | PointNet++ backbone, SE(3) output heads, Chamfer + occlusion loss | Paired pre/post-reduction CTs |
| Learned occlusion model | Graph neural network on dental arch point cloud | Cephalometric annotations |
| Deep registration | GeoTransformer / DCP candidates | CT-to-scan surface pairs |
| Learned landmark detection | Heatmap regression on 3D volumes | Manual landmark annotations |

Each module currently runs with a baseline algorithm (ICP, anatomical heuristics, thresholding) that produces functional output today. Learned models slot in through the same `InferenceModel` abstract base class.

---

## Module Status

| Module | Status | Notes |
|--------|--------|-------|
| DICOM ingestion + de-identification | ✅ | PS3.15 Annex E, 71 PHI tags, HMAC pseudonymization |
| CT quality control (8 checks) | ✅ | A/B/C/F grading per ACR guidelines |
| CT preprocessing (HU windowing, resampling) | ✅ | Bone/soft tissue windows |
| TotalSegmentator integration | ✅ | 80%+ CMF anatomy |
| DentalSegmentator integration | ✅ | Mandible canal, teeth |
| Mesh extraction + multi-LOD export | ✅ | Gaussian smoothing, PBR materials |
| Mesh quality analysis + repair | ✅ | Watertight, manifold, self-intersection checks |
| Cephalometric landmark detection | ✅ | 24 landmarks, anatomical heuristic baseline |
| Cephalometric analysis (SNA/SNB/ANB) | ✅ | Full angular and linear measurements |
| Facial symmetry analysis | ✅ | Midsagittal plane, per-structure asymmetry |
| ICP fracture reduction | ✅ | Point-to-plane ICP baseline |
| Occlusal constraint engine | ✅ | Molar classification, constraint satisfaction |
| Surgical plan evaluation | ✅ | Per-fragment metrics, composite grading |
| Surgical sequence optimization | ✅ | AO CMF principles, graph-based ordering |
| Clinical report generation | ✅ | Structured Markdown, all sections |
| Hardware recommendations | ✅ | AO CMF-based, per-structure |
| 3D planning viewer | ✅ | R3F with fragment manipulation |
| Measurement tools (3D) | ✅ | Distance, angle, with UI |
| Cross-section viewer | ✅ | Axial/coronal/sagittal with overlays |
| Cephalometric overlay (SVG) | ✅ | 17 landmarks, 11 measurements, color-coded |
| WebSocket real-time updates | ✅ | Job progress, plan updates |
| HIPAA audit middleware | ✅ | JSON-lines, PHI redaction, async SIEM sink |
| Rate limiting | ✅ | Token bucket, Redis-backed, 5 endpoint groups |
| CLI tools (33 commands) | ✅ | Case, model, pipeline, admin |
| Python SDK | ✅ | Typed async client, retry, job polling |
| Database migrations | ✅ | Alembic, initial schema |
| Benchmark framework | ✅ | Per-stage profiling, regression tracking |
| Learned plan suggestion | 📋 | Phase 2 — needs clinical case library |
| Soft tissue simulation | 📋 | Phase 3 |
| Automated splint design + 3D print | 📋 | Phase 3 |
| Intraoperative navigation export | 📋 | Phase 3 |
| PACS integration | 📋 | Phase 2 |

**Legend:** ✅ Functional · 📋 Planned

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/unit/backend/ -v
pytest tests/unit/schemas/ -v
pytest tests/integration/ -v

# Run with coverage
pytest tests/ --cov=apps/backend --cov=services --cov-report=html
```

331 tests, all mocked — no GPU or external services needed.

---

## Benchmarking

```bash
python scripts/run_benchmark.py
python scripts/run_benchmark.py --iterations 20 --output benchmark.md
python scripts/run_benchmark.py --stages segmentation,mesh_extraction --json
python scripts/run_benchmark.py --check-regression
```

---

## Contributing

1. Read `docs/architecture/system_design.md` to understand service boundaries
2. Check the module status table above
3. Open an issue before starting work on a new feature
4. Write tests — all pipeline stages have unit tests
5. Follow HIPAA patterns — no PHI in logs, de-identification before storage
6. Run `pytest tests/ -v` and `make lint` before submitting a PR

---

## Documentation

| Document | Path |
|----------|------|
| System Architecture | `docs/architecture/system_design.md` |
| Product Requirements | `docs/architecture/product_requirements.md` |
| Roadmap | `docs/architecture/roadmap.md` |
| Data Flow | `docs/architecture/data_flow.md` |
| Setup Instructions | `docs/setup.md` |
| Repo Map | `docs/REPO_MAP.md` |
| Clinical Workflow | `docs/clinical/clinical_workflow_summary.md` |
| Regulatory (FDA / HIPAA) | `docs/regulatory/regulatory_considerations.md` |
| Evaluation Plan | `docs/evaluation/evaluation_plan.md` |
| Training Plan | `docs/evaluation/training_plan.md` |
| Clinical Workflow Research | `research/clinical_workflow_research.md` |
| Competitive Landscape | `research/competitive_landscape.md` |
| Technical Stack Research | `research/technical_stack_research.md` |
| OSS Components Review | `research/oss_components_review.md` |
| AI-Native Principles | `research/ai_native_product_principles.md` |
| Baseline Scope Decisions | `research/baseline_scope_decisions.md` |

---

## License

MIT License. See `LICENSE` for details.
