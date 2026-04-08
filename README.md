# Facial Align

**AI-native craniofacial surgical planning — from CT to surgical plan in minutes, not days.**

Facial Align is an open research platform for virtual surgical planning (VSP) in cranio-maxillofacial (CMF) surgery. It replaces the "engineer-drives-the-software, surgeon-approves" service model with an AI system that proposes plans autonomously, surfaces uncertainty explicitly, and keeps the surgeon in the decision loop. The AI is not a button — it is the planning engine.

---

## Mission

Current VSP workflows require scheduling a live session with a third-party biomedical engineer, waiting 5–10 days for physical deliverables, and receiving a binary plan with no uncertainty signal. Facial Align eliminates that bottleneck: a CMF surgeon uploads a CT, receives AI-generated segmentation, landmark detection, and a ranked set of surgical plans — each annotated with confidence levels and constraint violations — within minutes. Every case processed feeds a training data flywheel that improves future predictions.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FACIAL ALIGN                                │
│                                                                     │
│   ┌──────────────┐      ┌───────────────────────────────────────┐  │
│   │   Frontend   │      │              Backend API              │  │
│   │  React/TS    │◄────►│           FastAPI (Python)            │  │
│   │  Three.js    │      │        /api/v1/* endpoints            │  │
│   │  OHIF Viewer │      └────────────┬──────────────────────────┘  │
│   └──────────────┘                  │                              │
│                                     │ enqueue jobs                 │
│                              ┌──────▼──────┐                       │
│                              │    Redis    │                       │
│                              │   Broker    │                       │
│                              └──────┬──────┘                       │
│                                     │                              │
│              ┌──────────────────────┼─────────────────────┐        │
│              │                      │                     │        │
│     ┌────────▼──────┐    ┌──────────▼──────┐   ┌─────────▼─────┐  │
│     │   Celery      │    │   Celery        │   │  Celery       │  │
│     │   Worker:     │    │   Worker:       │   │  Worker:      │  │
│     │   Ingestion   │    │   Segmentation  │   │  Planning     │  │
│     │   + Preproc   │    │   + Mesh        │   │  + Occlusion  │  │
│     └────────┬──────┘    └──────────┬──────┘   └─────────┬─────┘  │
│              │                      │                     │        │
│              └──────────────────────┼─────────────────────┘        │
│                                     │                              │
│                    ┌────────────────┼──────────────────┐           │
│                    │                │                  │           │
│           ┌────────▼────┐  ┌────────▼────┐  ┌─────────▼────────┐  │
│           │  PostgreSQL │  │   MinIO/S3  │  │  Inference       │  │
│           │  Metadata   │  │  DICOM/NIfTI│  │  Service         │  │
│           │  + Audit    │  │  /STL/Models│  │  TorchServe GPU  │  │
│           └─────────────┘  └─────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Features

### Functional (Phase 1 Baseline)
- **DICOM ingestion pipeline** — accepts CT/CBCT series; handles variable slice thickness, photometric inversion, series fragmentation, and HU calibration
- **De-identification** — pydicom/deid + pixel-level PHI detection before any data persists
- **Multi-structure segmentation** — TotalSegmentator + DentalSegmentator covering skull, mandible, maxilla, individual teeth (FDI notation), sinuses, and soft tissue envelope
- **Mesh extraction** — marching cubes → PyVista/trimesh post-processing → per-structure STL/OBJ export
- **Cephalometric landmark detection** — 24–32 CMF landmarks with per-landmark confidence scores
- **Fracture fragment identification** — connected component analysis on bone HU regions
- **Occlusal constraint engine** — encodes intercuspal position and Angle classification as computational constraints (not just a visual overlay)
- **3D viewer** — OHIF v3 + Cornerstone3D for zero-footprint DICOM review; Three.js for planning visualization
- **Async job processing** — Celery + Redis; all pipeline stages non-blocking with status polling
- **Evaluation framework** — Dice, Hausdorff distance, surface-to-surface distance; landmark error (mm)

### Scaffolded (Architecture in Place, Not Fully Implemented)
- Rule-based surgical plan suggestion (Le Fort I, BSSO, genioplasty movement vectors)
- Symmetry-guided fracture reduction with confidence bands
- PDF surgical planning report export
- Postoperative comparison overlay (plan vs. post-op CT)
- Model versioning and A/B evaluation

### Planned (Phase 2–3)
- Learned plan suggestion from clinical case library
- Soft tissue simulation (aesthetic outcome preview)
- Automated occlusal splint design + export for 3D printing
- Intraoperative navigation export (modified DICOM for Stryker/Medtronic)
- AR/VR planning interface
- PACS integration and multi-site deployment

---

## Tech Stack

| Layer | Technology | Version | Notes |
|-------|-----------|---------|-------|
| **ML Framework** | PyTorch | 2.11 | GPU inference backbone |
| **Medical Imaging ML** | MONAI | 1.5.2 | Transforms, loaders, losses, model zoo |
| **Segmentation Engine** | nnU-Net v2 | Latest | Self-configuring; CMF fine-tuning target |
| **Pre-trained Segmentation** | TotalSegmentator | 2.11 | Skull, mandible, sinuses; runs out of box |
| **Dental Segmentation** | DentalSegmentator | Latest | nnU-Net-based; mandible canal, teeth |
| **DICOM I/O** | pydicom | 2.4.5 | Metadata parsing and DICOM write |
| **Volume Processing** | SimpleITK | 2.5.3 | Series assembly, resampling, registration |
| **NIfTI I/O** | nibabel | 5.4.1 | NIfTI pipeline I/O |
| **DICOM Output** | highdicom | 0.26.1 | DICOM-SEG, SR, RT-Struct |
| **3D Mesh** | VTK / PyVista | Latest | Volume rendering, STL export, mesh ops |
| **Mesh Ops** | trimesh | 4.11.5 | Boolean ops, 3D-print prep, STL/OBJ/3MF |
| **Image Utils** | scikit-image | 0.26.0 | Marching cubes, morphology |
| **API** | FastAPI | 0.135.3 | Async REST; OpenAPI docs auto-generated |
| **Task Queue** | Celery + Redis | 5.x | Long-running pipeline jobs |
| **Database** | PostgreSQL | 16 | Patient/case metadata, audit log |
| **Object Storage** | MinIO | Latest | S3-compatible; DICOM/NIfTI/STL storage |
| **DICOM Viewer** | OHIF v3 + Cornerstone3D | 3.12.0 | Zero-footprint browser viewer |
| **Frontend** | React + TypeScript | 18 / 5 | Vite build; Zustand state management |
| **3D Visualization** | Three.js + React Three Fiber | Latest | Planning scene renderer |
| **Model Serving** | TorchServe | 0.12 | GPU inference service |
| **Containerization** | Docker + docker-compose | Latest | Reproducible dev and prod |
| **Experiment Tracking** | MLflow | Latest | Model registry, run comparison |

---

## Quick Start (Docker)

### Prerequisites
- Docker ≥ 24.0 and Docker Compose v2
- NVIDIA GPU with ≥ 16 GB VRAM (for GPU inference; CPU fallback available, slow)
- NVIDIA Container Toolkit installed
- 40 GB free disk space

### Start the full stack

```bash
git clone https://github.com/your-org/facial-align.git
cd facial-align

# Copy and configure environment
cp .env.example .env
# Edit .env — minimum: set SECRET_KEY and PHI_ENCRYPTION_KEY

# Pull and start all services
docker compose up -d

# Run database migrations
docker compose exec backend alembic upgrade head

# Verify all services are healthy
docker compose ps
```

### Service endpoints

| Service | URL | Notes |
|---------|-----|-------|
| API | http://localhost:8000 | FastAPI; docs at /docs |
| Frontend | http://localhost:3000 | React app |
| OHIF Viewer | http://localhost:3000/ohif | DICOM viewer |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin |
| MLflow | http://localhost:5000 | Experiment tracking |
| Flower (Celery) | http://localhost:5555 | Task monitor |

### Upload your first case

```bash
# Upload a DICOM series directory
curl -X POST http://localhost:8000/api/v1/cases \
  -H "Authorization: Bearer $TOKEN" \
  -F "dicom_archive=@/path/to/ct_series.zip"

# Poll job status
curl http://localhost:8000/api/v1/cases/{case_id}/status
```

---

## Project Structure

```
facial-align/
├── apps/
│   ├── backend/              # FastAPI application
│   │   └── app/
│   │       ├── api/v1/       # REST endpoints (cases, studies, plans, jobs)
│   │       ├── core/         # Config, security, logging
│   │       ├── db/           # SQLAlchemy models, migrations
│   │       ├── middleware/   # Auth, audit logging, rate limiting
│   │       ├── models/       # ORM models
│   │       ├── schemas/      # Pydantic request/response schemas
│   │       ├── services/     # Business logic (dicom, mesh, segmentation, etc.)
│   │       └── workers/      # Celery task definitions
│   └── frontend/             # React + TypeScript
│       └── src/
│           ├── components/   # UI components (viewer, planning, dashboard)
│           ├── hooks/        # Custom React hooks
│           ├── lib/          # API client, utilities
│           ├── pages/        # Route-level page components
│           ├── stores/       # Zustand state stores
│           └── types/        # TypeScript type definitions
├── pipelines/                # End-to-end ML pipeline stages
│   ├── dicom_ingestion/      # DICOM → NIfTI preprocessing
│   ├── segmentation/         # Segmentation inference pipeline
│   ├── mesh_extraction/      # NIfTI mask → STL mesh
│   ├── fracture_reduction/   # Fragment identification + reduction planning
│   └── occlusion_planning/   # Occlusal constraint computation
├── services/                 # Standalone microservices
│   ├── inference/            # TorchServe GPU inference service
│   ├── mesh_generation/      # Mesh processing service
│   ├── preprocessing/        # Volume preprocessing service
│   └── registration/         # CT-to-scan registration service
├── infra/
│   ├── docker/               # Dockerfiles per service
│   ├── kubernetes/           # K8s manifests (prod deployment)
│   └── monitoring/           # Prometheus + Grafana configs
├── data_contracts/           # JSON schemas for inter-service data exchange
├── tests/
│   ├── unit/                 # Unit tests (backend, frontend, pipelines)
│   ├── integration/          # End-to-end pipeline tests
│   └── fixtures/             # Test DICOM data (de-identified)
├── examples/
│   ├── notebooks/            # Jupyter notebooks for pipeline exploration
│   └── sample_data/          # Minimal synthetic CT data for demos
├── docs/                     # All project documentation
├── research/                 # Research notes and technology evaluations
├── scripts/                  # Setup, migration, and utility scripts
├── .env.example              # Environment variable template
└── docker-compose.yml        # Full stack development environment
```

---

## Module Status

| Module | Status | Notes |
|--------|--------|-------|
| DICOM ingestion + de-identification | ✅ Functional | pydicom + SimpleITK pipeline |
| CT preprocessing (resampling, HU windowing) | ✅ Functional | 1.0mm isotropic output |
| TotalSegmentator integration | ✅ Functional | Skull, mandible, sinuses |
| DentalSegmentator integration | ✅ Functional | Mandible, teeth, canal |
| Mesh extraction (marching cubes → STL) | ✅ Functional | PyVista + trimesh |
| Cephalometric landmark detection | ✅ Functional | 24 landmarks, heatmap regression |
| OHIF viewer integration | ✅ Functional | DICOM viewer in-browser |
| Three.js 3D planning scene | ✅ Functional | Bone mesh render + interaction |
| Celery job queue | ✅ Functional | Redis broker, result backend |
| FastAPI REST API | ✅ Functional | OpenAPI schema, auth middleware |
| PostgreSQL case/patient models | ✅ Functional | SQLAlchemy async ORM |
| MinIO object storage | ✅ Functional | DICOM, NIfTI, STL buckets |
| Fracture fragment identification | 🔧 Scaffolded | Connected components; reduction algo pending |
| Occlusal constraint engine | 🔧 Scaffolded | ICP computation; constraint solver pending |
| Surgical plan suggestion (rule-based) | 🔧 Scaffolded | Movement vector templates defined |
| CT-to-intraoral scan registration | 🔧 Scaffolded | ICP registration; surface matching pending |
| PDF planning report export | 🔧 Scaffolded | Template defined; render pipeline pending |
| Postoperative comparison | 🔧 Scaffolded | Overlay UI pending |
| Model versioning + A/B testing | 🔧 Scaffolded | MLflow integrated; deployment routing pending |
| Learned plan suggestion (ML) | 📋 Planned | Phase 2; requires clinical case library |
| Soft tissue simulation | 📋 Planned | Phase 3 |
| Automated splint design | 📋 Planned | Phase 3 |
| Intraoperative navigation export | 📋 Planned | Phase 3 |
| PACS integration | 📋 Planned | Phase 2 |

**Legend:** ✅ Functional · 🔧 Scaffolded · 📋 Planned

---

## Screenshots

*Visualization screenshots will be added as the UI stabilizes. The planning interface, DICOM viewer, and confidence overlay panels are the primary areas to document.*

| View | Description |
|------|-------------|
| Case Dashboard | Patient list, case status, recent activity |
| DICOM Viewer | OHIF three-plane viewer with segmentation overlay |
| 3D Planning Scene | Bone meshes, landmark annotations, plan visualization |
| Confidence Overlay | Per-voxel uncertainty heat map on segmentation |
| Plan Comparison | Pre-op / planned / post-op three-way overlay |

---

## Contributing

Facial Align follows standard GitHub flow. Before contributing:

1. **Read the architecture document** (`docs/architecture/system_design.md`) to understand service boundaries.
2. **Check the module status table** above — avoid duplicating scaffolded work.
3. **Open an issue** for any feature beyond a bug fix before starting implementation.
4. **Write tests** — all pipeline stages have corresponding unit tests in `tests/unit/pipelines/`.
5. **Run the full test suite** before submitting a PR:
   ```bash
   docker compose run --rm backend pytest tests/ -v
   ```
6. **Follow HIPAA patterns** — no PHI in logs, no raw DICOM in test fixtures, de-identification before storage.

See `CONTRIBUTING.md` for the full contribution guide.

---

## License

MIT License. See `LICENSE` for details.

---

## Documentation Index

| Document | Path | Contents |
|----------|------|----------|
| System Architecture | `docs/architecture/system_design.md` | Service decomposition, data flows, deployment topology |
| Product Requirements | `docs/architecture/product_requirements.md` | PRD: user stories, feature specs, success metrics |
| Roadmap | `docs/architecture/roadmap.md` | Phase 1–3 milestones, engineering backlog |
| Data Flow | `docs/architecture/data_flow.md` | End-to-end pipeline data flows |
| Setup Instructions | `docs/setup.md` | Prerequisites, local dev, environment variables |
| Repo Map | `docs/REPO_MAP.md` | Every directory and key file described |
| Clinical Workflow | `docs/clinical/clinical_workflow_summary.md` | CMF workflow, pain points, how FA transforms each step |
| Regulatory | `docs/regulatory/regulatory_considerations.md` | FDA SaMD, HIPAA, CE marking |
| Evaluation Plan | `docs/evaluation/evaluation_plan.md` | Accuracy metrics, clinical validation study design |
| Training Plan | `docs/evaluation/training_plan.md` | ML training strategy, data flywheel, model versioning |
| Clinical Workflow Research | `research/clinical_workflow_research.md` | Deep dive into CMF VSP workflows |
| Competitive Landscape | `research/competitive_landscape.md` | 3D Systems, Materialise, Brainlab, open-source analysis |
| Technical Stack Research | `research/technical_stack_research.md` | Tool selection rationale |
| OSS Components Review | `research/oss_components_review.md` | Component-by-component evaluation |
| AI-Native Principles | `research/ai_native_product_principles.md` | Design philosophy and UX framework |
| Baseline Scope Decisions | `research/baseline_scope_decisions.md` | What's in Phase 1 and why |
