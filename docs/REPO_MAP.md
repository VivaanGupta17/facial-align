# Repository Map — Facial Align

Every directory and significant file in the repo, with a one-line description.

**Legend:** `✅ Functional` · `🔧 Scaffolded` · `📋 Planned` · `📄 Config/Data`

---

## Root

```
facial-align/
├── README.md                   Project overview, quick start, module status table, doc index
├── .env.example                Environment variable template — copy to .env before running
├── .gitignore                  Git ignore rules (venv, __pycache__, .env, large model files)
├── docker-compose.yml          Full development stack (backend, frontend, workers, inference, infra)
├── LICENSE                     MIT License
└── CONTRIBUTING.md             Contribution guidelines, PR process, coding standards
```

---

## apps/

All application code. Two apps: backend (Python/FastAPI) and frontend (React/TypeScript).

### apps/backend/

```
apps/backend/
├── Dockerfile                  Backend Docker image (Python 3.11, pip install, uvicorn)
├── requirements.txt            Production Python dependencies (pinned versions)
├── requirements-dev.txt        Development dependencies (pytest, black, ruff, mypy)
├── alembic.ini                 Alembic migration configuration
├── alembic/
│   ├── env.py                  Alembic environment — connects to DATABASE_URL
│   └── versions/               One file per database migration, in order
└── app/
    ├── main.py                 FastAPI application factory — registers routers, middleware, lifespan
    ├── api/
    │   └── v1/
    │       ├── router.py       Root v1 router — aggregates all endpoint routers
    │       └── endpoints/
    │           ├── auth.py     POST /auth/login, /auth/register, /auth/refresh, /auth/logout
    │           ├── cases.py    CRUD for surgical cases; case status polling
    │           ├── studies.py  DICOM study upload, retrieval, series listing
    │           ├── jobs.py     Job status, retry, cancellation; WebSocket status stream
    │           ├── plans.py    Surgical plan CRUD; plan approval; plan comparison
    │           ├── meshes.py   Mesh retrieval; presigned URL generation for STL/GLTF
    │           ├── evaluations.py  Accuracy metric retrieval per case
    │           └── admin.py    User management; model version promotion
    ├── core/
    │   ├── config.py           Pydantic Settings class — all environment variables typed
    │   ├── security.py         JWT creation/validation; password hashing (bcrypt)
    │   ├── logging.py          Structured JSON logging configuration
    │   └── redis.py            Redis connection pool (used by Celery and WebSocket pub/sub)
    ├── db/
    │   ├── base.py             SQLAlchemy declarative base; metadata; engine factory
    │   ├── session.py          Async session factory; dependency injection helper
    │   └── init_db.py          Database initialization (create buckets, seed data)
    ├── middleware/
    │   ├── auth.py             JWT extraction and validation middleware
    │   ├── audit.py            Audit log writer — intercepts all PHI-touching requests
    │   ├── phi_scrubber.py     Removes PHI from log messages (names, IDs, dates)
    │   └── rate_limiter.py     Per-user and per-endpoint rate limiting via Redis
    ├── models/
    │   ├── patient.py          Patient ORM model (de-identified demographics)
    │   ├── case.py             Case ORM model (primary work unit; links study, plan, jobs)
    │   ├── dicom_study.py      DICOM study metadata ORM model
    │   ├── job.py              Pipeline job ORM model (links to Celery task)
    │   ├── segmentation.py     Segmentation result ORM model (paths, Dice scores, label map)
    │   ├── mesh.py             Per-structure mesh ORM model (STL/GLTF paths, vertex counts)
    │   ├── landmark.py         Cephalometric landmark ORM model (3D coords, confidence)
    │   ├── occlusal_analysis.py  Occlusal relationship ORM model (class, overjet, overbite)
    │   ├── surgical_plan.py    Surgical plan ORM model (movements JSON, confidence, status)
    │   ├── plan_modification.py  Surgeon plan modification audit record
    │   ├── user.py             User ORM model (email, hashed password, role, institution)
    │   └── audit_log.py        Append-only audit log ORM model (PHI access events)
    ├── schemas/
    │   ├── case.py             Pydantic request/response schemas for case endpoints
    │   ├── study.py            Pydantic schemas for DICOM study endpoints
    │   ├── job.py              Pydantic schemas for job status responses
    │   ├── plan.py             Pydantic schemas for surgical plan endpoints
    │   ├── mesh.py             Pydantic schemas for mesh endpoint responses
    │   ├── landmark.py         Pydantic schemas for landmark endpoints
    │   ├── auth.py             Pydantic schemas for auth endpoints (login, token)
    │   └── evaluation.py       Pydantic schemas for evaluation metric responses
    ├── services/
    │   ├── dicom/
    │   │   ├── ingestion.py    DICOM series parsing, validation, file organization
    │   │   ├── deidentify.py   De-identification using pydicom/deid; pixel PHI detection
    │   │   └── metadata.py     DICOM tag extraction → structured metadata dict
    │   ├── segmentation/
    │   │   ├── service.py      Segmentation orchestrator (calls inference, stores results)
    │   │   └── fusion.py       Multi-model segmentation label fusion (coarse + dental)
    │   ├── mesh/
    │   │   ├── extraction.py   Marching cubes → mesh cleanup → STL/GLTF export
    │   │   └── uncertainty.py  Map per-voxel uncertainty to vertex attributes
    │   ├── registration/
    │   │   ├── intraoral.py    ICP registration of intraoral scan to CT dental segmentation
    │   │   └── postop.py       Pre-op to post-op CT rigid registration for outcome tracking
    │   ├── occlusion/
    │   │   ├── analysis.py     Angle classification, overjet/overbite measurement
    │   │   └── constraints.py  Constraint satisfaction checker; violation reporting
    │   └── reduction/
    │       ├── fragment_id.py  Connected component analysis for fracture fragment isolation
    │       ├── planning.py     Fracture reduction planning (ICP-to-reference baseline)
    │       └── scoring.py      Plan candidate scoring and ranking
    └── workers/
        ├── celery_app.py       Celery application factory (broker, result backend, routing)
        ├── ingestion.py        Celery tasks: ingest_dicom, preprocess_volume
        ├── segmentation.py     Celery tasks: run_segmentation, extract_meshes
        ├── planning.py         Celery tasks: detect_landmarks, compute_occlusion, generate_plans
        └── evaluation.py       Celery tasks: compute_metrics, compare_postop, generate_report
```

### apps/frontend/

```
apps/frontend/
├── Dockerfile                  Frontend Docker image (Node.js 20, Vite build, nginx serve)
├── package.json                Node.js dependencies and scripts
├── vite.config.ts              Vite bundler configuration
├── tsconfig.json               TypeScript configuration
├── tailwind.config.ts          Tailwind CSS configuration
└── src/
    ├── main.tsx                React application entry point
    ├── App.tsx                 Root component — router setup, global providers
    ├── components/
    │   ├── viewer/
    │   │   ├── OhifViewer.tsx          OHIF v3 embedded viewer component (DICOM slices)
    │   │   ├── PlanningScene.tsx        Three.js scene — bone meshes, gizmos, uncertainty heat maps
    │   │   ├── BoneMesh.tsx            Individual bone mesh renderer with material properties
    │   │   ├── LandmarkMarker.tsx      3D landmark point with uncertainty ellipsoid
    │   │   ├── UncertaintyHeatmap.tsx  Vertex color shader for confidence visualization
    │   │   └── CameraControls.tsx      Orbit controls, zoom, reset view
    │   ├── planning/
    │   │   ├── PlanCandidateCard.tsx   AI plan candidate display (confidence, movements, rationale)
    │   │   ├── MovementPanel.tsx       Bone segment movement controls (translation/rotation inputs)
    │   │   ├── OcclusalStatus.tsx      Real-time occlusal constraint satisfaction indicator
    │   │   ├── CephalometricPanel.tsx  Cephalometric measurements display (ANB, SNB, etc.)
    │   │   └── PlanApproval.tsx        Plan approval flow with surgeon confirmation
    │   ├── dashboard/
    │   │   ├── CaseList.tsx            Paginated case list with status badges and filters
    │   │   ├── CaseCard.tsx            Individual case summary card
    │   │   ├── JobStatus.tsx           Real-time job progress indicator (WebSocket)
    │   │   └── UploadZone.tsx          Drag-and-drop DICOM upload with validation feedback
    │   ├── layout/
    │   │   ├── AppLayout.tsx           Main application shell (sidebar, topbar, content area)
    │   │   ├── Sidebar.tsx             Navigation sidebar with role-based menu items
    │   │   └── TopBar.tsx              Header (user menu, notifications, case breadcrumb)
    │   └── common/
    │       ├── ConfidenceBadge.tsx     Reusable confidence indicator (color-coded, tooltip)
    │       ├── LoadingSpinner.tsx      Async operation loading indicator
    │       ├── ErrorBoundary.tsx       React error boundary with graceful fallback
    │       ├── WarningAlert.tsx        Clinical warning display (uncertainty flags, constraint violations)
    │       └── Modal.tsx               Generic modal dialog
    ├── hooks/
    │   ├── useCase.ts                  Case data fetching and mutation hooks
    │   ├── useJobStatus.ts             WebSocket subscription for real-time job updates
    │   ├── useMeshLoader.ts            GLTF mesh loading from presigned MinIO URLs
    │   └── usePlanningScene.ts         Three.js scene state management hook
    ├── lib/
    │   ├── api.ts                      Typed API client generated from OpenAPI schema
    │   ├── dicom.ts                    DICOM metadata parsing utilities for frontend
    │   └── geometry.ts                 3D math utilities (transforms, coordinate conversions)
    ├── pages/
    │   ├── LoginPage.tsx               Authentication page
    │   ├── DashboardPage.tsx           Case list and activity feed
    │   ├── CasePage.tsx                Case detail — upload, status, results
    │   ├── PlanningPage.tsx            Main planning workspace (viewer + planning panels)
    │   └── EvaluationPage.tsx          Accuracy metrics and outcome tracking
    ├── stores/
    │   ├── caseStore.ts                Zustand store — current case, status, job list
    │   ├── planningStore.ts            Zustand store — active plan, movements, constraint state
    │   ├── viewerStore.ts              Zustand store — OHIF viewer state, window/level settings
    │   └── authStore.ts                Zustand store — user identity, token, role
    ├── styles/
    │   └── globals.css                 Global CSS (Tailwind base, custom properties)
    └── types/
        ├── api.ts                      TypeScript types auto-generated from OpenAPI schema
        ├── geometry.ts                 3D geometry type definitions (Vector3, Quaternion, Transform)
        └── clinical.ts                 Clinical domain types (Landmark, Plan, OcclusalState)
```

---

## pipelines/

End-to-end ML pipeline stages. Each pipeline stage is a standalone Python module that can be run independently or chained via Celery.

```
pipelines/
├── dicom_ingestion/
│   ├── __init__.py
│   ├── main.py                 CLI entry point for standalone ingestion run
│   ├── reader.py               pydicom-based DICOM series reader and series grouper
│   ├── validator.py            Series validation (slice thickness, completeness, modality)
│   ├── preprocessor.py         SimpleITK volume assembly, orientation, resampling
│   ├── deidentifier.py         Full de-identification pipeline with pixel PHI detection
│   └── tests/                  Unit tests for ingestion pipeline
├── segmentation/
│   ├── __init__.py
│   ├── main.py                 CLI entry point for standalone segmentation run
│   ├── runner.py               TotalSegmentator + DentalSegmentator orchestration
│   ├── fusion.py               Multi-model output fusion and label reconciliation
│   ├── uncertainty.py          MC Dropout uncertainty estimation for segmentation output
│   └── tests/                  Unit tests for segmentation pipeline
├── mesh_extraction/
│   ├── __init__.py
│   ├── main.py                 CLI entry point for standalone mesh extraction
│   ├── extractor.py            Marching cubes per label; mesh cleanup (smooth, decimate)
│   ├── validator.py            Mesh quality checks (watertight, self-intersection)
│   ├── exporter.py             STL and GLTF export with vertex attributes
│   └── tests/                  Unit tests for mesh extraction
├── fracture_reduction/
│   ├── __init__.py
│   ├── main.py                 CLI entry point for fracture reduction planning
│   ├── fragment_identifier.py  Connected component analysis; AOCMF code assignment
│   ├── reducer.py              ICP-to-reference baseline reduction algorithm
│   ├── planner.py              Multi-candidate plan generation and scoring
│   └── tests/                  Unit tests for fracture reduction pipeline
└── occlusion_planning/
    ├── __init__.py
    ├── main.py                 CLI entry point for occlusal analysis
    ├── contact_detector.py     Trimesh collision-based contact pair detection
    ├── classifier.py           Angle class classification; overjet/overbite measurement
    ├── constraint_engine.py    Constraint satisfaction evaluation for proposed movements
    └── tests/                  Unit tests for occlusion analysis pipeline
```

---

## services/

Standalone microservices that can be deployed and scaled independently.

```
services/
├── inference/
│   ├── Dockerfile              TorchServe container with CUDA base image
│   ├── requirements.txt        ML dependencies (torch, monai, totalsegmentator, nnunetv2)
│   ├── config.properties       TorchServe server configuration
│   ├── model_store/            Directory for .mar model archives (loaded at startup)
│   └── handlers/
│       ├── segmentation_handler.py     TorchServe handler for TotalSegmentator inference
│       ├── dental_handler.py           TorchServe handler for DentalSegmentator inference
│       ├── landmark_handler.py         TorchServe handler for landmark heatmap regression
│       └── plan_scorer_handler.py      TorchServe handler for plan scoring MLP
├── mesh_generation/
│   ├── Dockerfile              Mesh generation service container (CPU; no GPU needed)
│   ├── requirements.txt        Mesh dependencies (pyvista, trimesh, scikit-image)
│   └── app.py                  FastAPI micro-service exposing mesh generation endpoints
├── preprocessing/
│   ├── Dockerfile              Preprocessing service container
│   ├── requirements.txt        SimpleITK, pydicom, nibabel
│   └── app.py                  FastAPI micro-service for volume preprocessing
└── registration/
    ├── Dockerfile              Registration service container (Open3D, SimpleITK)
    ├── requirements.txt        Open3D, SimpleITK, scipy
    └── app.py                  FastAPI micro-service for CT-to-scan and pre/post-op registration
```

---

## infra/

Infrastructure configuration for Docker, Kubernetes, and monitoring.

```
infra/
├── docker/
│   ├── backend.Dockerfile      Production backend image (multi-stage build)
│   ├── frontend.Dockerfile     Production frontend image (Vite build → nginx)
│   ├── worker.Dockerfile       Celery worker image (same as backend; different entrypoint)
│   └── inference.Dockerfile    TorchServe GPU inference image (nvcr.io base)
├── kubernetes/
│   ├── namespace.yaml          Namespace definition (facial-align-prod / facial-align-staging)
│   ├── backend/
│   │   ├── deployment.yaml     Backend deployment (3 replicas, CPU nodes)
│   │   ├── service.yaml        ClusterIP service for backend
│   │   └── hpa.yaml            HorizontalPodAutoscaler (scale on CPU utilization)
│   ├── worker/
│   │   ├── deployment-ingestion.yaml    Ingestion worker deployment
│   │   ├── deployment-segmentation.yaml Segmentation worker deployment (GPU nodes)
│   │   └── deployment-planning.yaml     Planning worker deployment (GPU nodes)
│   ├── inference/
│   │   ├── deployment.yaml     TorchServe deployment (1 replica, GPU node, exclusive)
│   │   └── service.yaml        ClusterIP service for inference (internal only)
│   ├── frontend/
│   │   ├── deployment.yaml     Frontend deployment (2 replicas)
│   │   └── service.yaml        LoadBalancer service (public-facing)
│   ├── postgres/
│   │   └── statefulset.yaml    PostgreSQL StatefulSet with PersistentVolumeClaim
│   ├── redis/
│   │   └── statefulset.yaml    Redis StatefulSet
│   └── ingress.yaml            Nginx Ingress with TLS (cert-manager annotations)
└── monitoring/
    ├── prometheus.yml          Prometheus scrape configuration (all services)
    ├── grafana/
    │   └── dashboards/
    │       ├── service_health.json     Service uptime, latency, error rates
    │       ├── gpu_utilization.json    GPU VRAM, compute utilization per node
    │       ├── pipeline_metrics.json   Job queue depth, task duration, failure rate
    │       └── model_accuracy.json     Real-time model performance tracking
    └── alerts.yaml             Alertmanager rules (GPU OOM, queue depth, API latency)
```

---

## data_contracts/

JSON Schema definitions for all inter-service data exchange. Enforced at runtime via Pydantic.

```
data_contracts/
├── dicom_series_manifest.json  Schema for DICOM series metadata after ingestion
├── segmentation_result.json    Schema for segmentation output from inference service
├── mesh_export.json            Schema for mesh generation output (paths, stats)
├── landmark_set.json           Schema for landmark detection output (coords, confidence)
├── occlusal_state.json         Schema for occlusal analysis output (class, metrics)
├── surgical_plan.json          Schema for surgical plan object (movements, constraints)
├── plan_modification.json      Schema for surgeon plan modification delta
└── evaluation_result.json      Schema for accuracy metric output (Dice, Hausdorff)
```

---

## tests/

All project tests. Tests must run without real patient data — use fixtures/ for test data.

```
tests/
├── conftest.py                 Shared pytest fixtures (test database, MinIO client, mock users)
├── fixtures/
│   ├── synthetic_ct_001/       Minimal 50-slice synthetic CT DICOM (no patient data)
│   ├── synthetic_segmentation/ Corresponding synthetic segmentation NIfTI
│   └── sample_plan.json        Sample surgical plan for API testing
├── unit/
│   ├── backend/
│   │   ├── test_auth.py        JWT creation, validation, refresh token rotation
│   │   ├── test_case_api.py    Case CRUD endpoint tests
│   │   ├── test_dicom_service.py  DICOM parsing, validation, metadata extraction
│   │   ├── test_deidentify.py  De-identification: all 18 Safe Harbor identifiers verified
│   │   └── test_audit_log.py   Audit log creation on every PHI-touching operation
│   ├── frontend/
│   │   ├── test_plan_store.ts  Zustand planningStore unit tests
│   │   └── test_geometry.ts    3D geometry utility unit tests
│   └── pipelines/
│       ├── test_ingestion.py   DICOM parsing, series grouping, resampling on synthetic CT
│       ├── test_segmentation.py  Segmentation fusion, uncertainty computation
│       ├── test_mesh.py        Marching cubes, mesh cleanup, watertight check
│       ├── test_fracture.py    Connected component analysis on synthetic fracture data
│       └── test_occlusion.py   Angle classification, overjet/overbite on synthetic meshes
└── integration/
    ├── test_full_pipeline.py   End-to-end: DICOM upload → segmentation → mesh → plan (slow)
    ├── test_api_auth_flow.py   Full login → token → protected endpoint → logout cycle
    └── test_job_chain.py       Celery task chain execution with Redis broker
```

---

## examples/

Demos and tutorials for exploring the codebase without running the full stack.

```
examples/
├── notebooks/
│   ├── 01_dicom_ingestion.ipynb        Explore DICOM parsing with pydicom + SimpleITK
│   ├── 02_segmentation_demo.ipynb      Run TotalSegmentator on a sample CT; visualize output
│   ├── 03_mesh_extraction.ipynb        Marching cubes → trimesh pipeline walkthrough
│   ├── 04_landmark_detection.ipynb     Landmark heatmap regression inference demo
│   ├── 05_occlusal_analysis.ipynb      Contact detection and Angle classification demo
│   └── 06_evaluation_metrics.ipynb     Compute Dice, Hausdorff, landmark error
└── sample_data/
    ├── README.md                       What this data is and how to use it
    └── synthetic_ct/                   Synthetic minimal CT data for demo notebooks
```

---

## research/

Research notes, technology evaluations, and design references. Not production code.

```
research/
├── clinical_workflow_research.md       Deep dive into CMF VSP workflows — trauma, orthognathic, reconstruction
├── competitive_landscape.md            3D Systems, Materialise, Brainlab, open-source analysis
├── technical_stack_research.md         Tool selection rationale (DICOM, ML, storage, visualization)
├── oss_components_review.md            Component-by-component OSS evaluation with phase recommendations
├── ai_native_product_principles.md     AI-native design philosophy and UX framework [NEW]
└── baseline_scope_decisions.md         What is in Phase 1 and why [NEW]
```

---

## docs/

All project documentation. Start with `REPO_MAP.md` (this file), then `setup.md`, then architecture docs.

```
docs/
├── REPO_MAP.md                         This file — directory of every file with descriptions
├── setup.md                            Prerequisites, Docker quick start, local dev, troubleshooting
├── architecture/
│   ├── system_design.md                Service decomposition, database schema, storage, API design
│   ├── product_requirements.md         PRD — user stories, feature specs by phase, success metrics
│   ├── data_flow.md                    End-to-end data flow for each pipeline stage
│   └── roadmap.md                      Phase 1–3 milestones and engineering backlog
├── clinical/
│   └── clinical_workflow_summary.md    CMF workflow, pain points, how FA transforms each step
├── evaluation/
│   ├── evaluation_plan.md              Accuracy metrics, validation study design, ground truth protocol
│   └── training_plan.md                ML training strategy, data flywheel, model versioning
└── regulatory/
    └── regulatory_considerations.md    FDA SaMD classification, HIPAA safeguards, de-identification, BAA
```

---

## .github/

CI/CD and project management configuration.

```
.github/
├── workflows/
│   ├── ci.yml                  CI: lint → test → build on every PR
│   ├── deploy-staging.yml      CD: deploy to staging on merge to main
│   └── deploy-prod.yml         CD: deploy to production on release tag
└── ISSUE_TEMPLATE/
    ├── bug_report.md           Bug report template (environment, steps to reproduce, logs)
    └── feature_request.md      Feature request template (user story, acceptance criteria)
```

---

## Key File Quick Reference

| Need to... | File |
|-----------|------|
| Change environment config | `.env.example` → `.env` |
| Add a new API endpoint | `apps/backend/app/api/v1/endpoints/` |
| Add a database table | `apps/backend/app/models/` + `alembic revision` |
| Add a Celery task | `apps/backend/app/workers/` |
| Change the 3D planning viewer | `apps/frontend/src/components/planning/PlanningScene.tsx` |
| Run the full pipeline locally | `examples/notebooks/` or `docker compose run backend python -m pipelines.{name}.main` |
| Understand the clinical workflow | `docs/clinical/clinical_workflow_summary.md` |
| Understand regulatory obligations | `docs/regulatory/regulatory_considerations.md` |
| Add a new ML model | `services/inference/handlers/` + register in TorchServe config |
| Deploy to production | `infra/kubernetes/` + `infra/docker/` |
| Run all tests | `docker compose run --rm backend pytest tests/ -v` |
| Understand architecture decisions | `research/baseline_scope_decisions.md` |
