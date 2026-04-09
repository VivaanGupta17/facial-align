# Facial Align — System Architecture

**AI-Native Craniomaxillofacial Surgical Planning Platform**

Version 0.4.0 · April 2026
Johns Hopkins University, Department of Computer Science
Vivaan Gupta · vgupta18@jh.edu
Repository: [github.com/VivaanGupta17/facial-align](https://github.com/VivaanGupta17/facial-align)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Overview](#2-system-overview)
3. [Data Architecture](#3-data-architecture)
4. [ML Pipeline Architecture](#4-ml-pipeline-architecture)
5. [Inference Layer](#5-inference-layer)
6. [Pipeline Orchestration](#6-pipeline-orchestration)
7. [Evaluation & Surgical Planning](#7-evaluation--surgical-planning)
8. [Backend API](#8-backend-api)
9. [Database & Migrations](#9-database--migrations)
10. [Frontend](#10-frontend)
11. [CLI & SDK](#11-cli--sdk)
12. [Infrastructure](#12-infrastructure)
13. [Testing](#13-testing)
14. [Technology Stack](#14-technology-stack)

---

## 1. Executive Summary

Facial Align is an AI-native platform for craniomaxillofacial (CMF) fracture reconstruction. The system uses **dental occlusion as the primary optimization target** — fracture reduction is driven by achieving correct bite alignment, not by minimizing bone surface distance alone. This mirrors the clinical reality where restoring masticatory function is the primary surgical goal.

### Key Differentiator: Occlusion-First

Traditional virtual surgical planning systems align bone fragments first (using ICP on fracture surfaces) and then check occlusion as a post-hoc validation step. Facial Align inverts this: the dental occlusion objective function drives fragment positioning, with fracture surface fitting as a secondary constraint. This is backed by recent research (PMC11574221, 2024) demonstrating that simultaneous optimization of dental occlusion and fracture fitting produces clinically superior outcomes.

### Architecture at a Glance

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | React + TypeScript + Three.js/R3F | 3D surgical visualization, surgeon interaction |
| API | FastAPI (async) | RESTful API + WebSocket real-time notifications |
| Task Queue | Celery + Redis | Async ML pipeline execution on GPU workers |
| ML Pipeline | PyTorch + pytorch3d + torch-geometric + open3d | Occlusion analysis, fracture reduction, segmentation |
| Storage | PostgreSQL + MinIO (S3-compatible) | Relational data + DICOM/mesh object storage |
| Auth/Audit | JWT + bcrypt + HIPAA audit trail | Institutional-grade security |

### Scale

- 234 files, ~67,000 source lines (46K Python, 10.5K TypeScript, 11K Markdown)
- 21 commits, 381+ tests
- 9 enriched data contracts, 7 inference adapters, 5 async pipelines
- 36 HTTP endpoints + 2 WebSocket endpoints

---

## 2. System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND                                  │
│  React + TypeScript + React Three Fiber                          │
│  ┌─────────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────────┐ │
│  │Dashboard│ │OcclusionWork-│ │ReductionWork-│ │Cephalometric│ │
│  │         │ │space         │ │space         │ │Overlay      │ │
│  └────┬────┘ └──────┬───────┘ └──────┬───────┘ └──────┬──────┘ │
│       │              │                │                │         │
│  ┌────┴──────────────┴────────────────┴────────────────┴──────┐ │
│  │  Viewer3D (Three.js) ── MeasurementTools ── FragmentCtrl   │ │
│  └────────────────────────────┬────────────────────────────────┘ │
└───────────────────────────────┼──────────────────────────────────┘
                                │ HTTP + WebSocket
┌───────────────────────────────┼──────────────────────────────────┐
│                        BACKEND API                               │
│  FastAPI (async) ── Uvicorn                                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │ /cases   │ │ /dicom   │ │/planning │ │ /viewer  │            │
│  │ /segment │ │ /jobs    │ │ /ws      │ │ /health  │            │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘            │
│       │             │            │             │                  │
│  ┌────┴─────────────┴────────────┴─────────────┴───────────────┐ │
│  │  Middleware: HIPAA Audit │ Rate Limiter │ Error Handler      │ │
│  │             Request Tracing │ Security Headers               │ │
│  └──────────────────────────┬──────────────────────────────────┘ │
└─────────────────────────────┼────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
┌─────────┴────────┐ ┌───────┴────────┐ ┌────────┴─────────┐
│   PostgreSQL     │ │   Redis        │ │   MinIO (S3)     │
│   (async via     │ │   (Celery      │ │   (DICOM, mesh   │
│    asyncpg)      │ │    broker +    │ │    object store)  │
│                  │ │    WS pubsub)  │ │                   │
└──────────────────┘ └───────┬────────┘ └───────────────────┘
                             │
                    ┌────────┴────────┐
                    │  Celery Workers  │
                    │  (GPU-enabled)   │
                    │                  │
                    │  ┌────────────┐  │
                    │  │ Pipelines  │  │
                    │  │ DICOM Ing. │  │
                    │  │ Segment.   │  │
                    │  │ Mesh Extr. │  │
                    │  │ Occlusion  │  │
                    │  │ Reduction  │  │
                    │  └──────┬─────┘  │
                    │         │        │
                    │  ┌──────┴─────┐  │
                    │  │ ML Models  │  │
                    │  │ DGCNN enc. │  │
                    │  │ Occlusion  │  │
                    │  │ Transformer│  │
                    │  │ Losses     │  │
                    │  │ Joint opt. │  │
                    │  └────────────┘  │
                    └─────────────────┘
```

### Design Principles

1. **AI-native** — ML models are the core decision engine, not add-ons to a rule-based system
2. **Occlusion-first** — Dental alignment quality is the primary optimization objective
3. **Local-first / institution-friendly** — Runs on-premise, no cloud dependency, HIPAA-compliant
4. **Modular** — Each service is independently testable and replaceable
5. **Medically serious** — Clinical terminology, validated metrics, surgeon-reviewable outputs

### Repository Structure

```
facial-align/
├── apps/
│   ├── backend/              # FastAPI + Celery application (11,122 lines)
│   │   ├── app/
│   │   │   ├── api/v1/       # 8 endpoint routers
│   │   │   ├── core/         # Config, exceptions, logging, security
│   │   │   ├── db/           # Async SQLAlchemy
│   │   │   ├── middleware/    # Audit, rate-limit, tracing, errors
│   │   │   ├── models/       # 6 ORM models
│   │   │   ├── schemas/      # API Pydantic schemas
│   │   │   ├── services/     # Business logic (occlusion, reduction, mesh, DICOM, etc.)
│   │   │   └── workers/      # Celery app + task definitions
│   │   └── alembic/          # Database migrations
│   └── frontend/             # React + TypeScript + R3F (10,438 lines)
│       └── src/
│           ├── components/   # Viewer, planning workspaces, common UI
│           ├── hooks/        # WebSocket, data fetching
│           ├── pages/        # Dashboard, case list, case detail, upload
│           ├── stores/       # Zustand state management
│           └── lib/          # API client, geometry utils, validation
├── cli/                      # Click CLI — 33 commands (~4,820 lines)
├── data_contracts/           # 9 enriched Pydantic domain schemas (~3,500 lines)
├── pipelines/                # 5 async ML pipelines
├── sdk/                      # Python SDK client (612 lines)
├── services/
│   ├── benchmark/            # Pipeline profiler (766 lines)
│   ├── evaluation/           # Plan evaluator, report gen, surgical sequencer
│   ├── inference/            # Model registry + 7 ML adapters
│   ├── mesh_generation/      # Marching cubes + PBR materials
│   ├── preprocessing/        # CT preprocessor, QC, DICOM de-identification
│   └── registration/         # Point cloud registration
├── tests/                    # 381+ tests + fixtures (~7,700 lines)
├── infra/                    # Docker Compose, CI/CD
├── docs/                     # Documentation
├── research/                 # Research papers, notes
├── examples/                 # Sample data, notebooks
└── scripts/                  # Model download, benchmarking, demo seeding
```

---

## 3. Data Architecture

### 3.1 Data Contract Layer

Nine enriched Pydantic v2 schemas define the domain model. These are the canonical representations of all clinical data flowing through the system.

#### CTStudyContract (396 lines)

Represents a CT imaging study with full DICOM metadata.

| Field | Type | Description |
|-------|------|-------------|
| `study_id` | `str` | Unique study identifier |
| `modality` | `CTModality` | CT, CBCT, or MRI |
| `quality_grade` | `CTQualityGrade` | A-F quality rating |
| `slice_thickness_mm` | `float` | Axial slice thickness |
| `pixel_spacing_mm` | `Tuple[float, float]` | In-plane resolution |
| `matrix_size` | `Tuple[int, int, int]` | Volume dimensions |
| `window_center` | `float` | DICOM window center (bone) |
| `window_width` | `float` | DICOM window width |
| `series` | `List[CTSeriesContract]` | Individual DICOM series |
| `is_suitable_for_planning()` | method | Quality gate (≥ grade C, ≤ 1mm slice) |

#### SegmentationOutputContract (382 lines)

ML segmentation results with per-structure confidence and mesh paths.

- `StructureClass` enum: skull, mandible, maxilla, teeth (individual FDI), condyle, ramus, body, symphysis, orbit, zygomatic
- `STRUCTURE_HIERARCHY`: registry mapping parent → child structures
- `StructureMesh`: file paths (STL/PLY/GLB) + volume + surface area + centroid
- `StructureStats`: per-structure Dice coefficient, Hausdorff distance, volume in cc
- Overall: 20 fields including `model_name`, `model_version`, `inference_time_seconds`, `gpu_memory_mb`

#### FractureFragmentContract (382 lines)

A single bone fragment extracted from segmentation.

- `AOCMFRegion` enum: symphysis, parasymphysis, body, angle, ramus, condyle_subcondylar, condyle_intracapsular, coronoid, alveolar, dentoalveolar
- `FragmentGeometry`: 15 fields — centroid (LPS coordinates), bounding box, volume, surface area, principal axes, contact surfaces
- `FragmentTransformContract`: SE(3) transform with model validator enforcing rotation matrix orthonormality (`det(R) ≈ 1.0`, `R^T R ≈ I`)
- `FragmentContactSurface`: fracture surface mesh + area + normal + roughness

#### ReductionPlanContract (374 lines)

Primary planning artifact — the surgical reduction plan.

| Field | Type | Description |
|-------|------|-------------|
| `plan_id` | `str` | Unique plan identifier |
| `case_id` | `str` | Parent surgical case |
| `status` | `PlanStatus` | draft / optimizing / reviewed / approved / rejected |
| `origin` | `PlanOrigin` | ai_generated / surgeon_modified / manual |
| `fragments` | `List[FractureFragmentContract]` | All fragments with transforms |
| `occlusal_metrics` | `OcclusalMetricsContract` | Predicted occlusion quality |
| `validation` | `ValidationContract` | Collision check, gap analysis |
| `hardware` | `List[HardwareItem]` | Recommended plates/screws |
| `version` | `int` | Plan revision number |
| `surgeon_edits` | `List[SurgeonEditHistory]` | Edit audit trail |

#### OcclusionPlanContract (423 lines)

The most complex contract — 27 fields including full cephalometric battery.

- `AngleMolarClass` enum: Class_I, Class_II_div1, Class_II_div2, Class_III
- `OcclusionGrade` enum: excellent, good, acceptable, poor, malocclusion
- `SkeletalPattern` enum: skeletal_I, skeletal_II, skeletal_III
- `ToothContactContract`: per-tooth contact with FDI numbering, force distribution, contact area
- `CephalometricMeasurement`: SNA, SNB, ANB, Wits appraisal, mandibular plane angle, gonial angle, Frankfurt-mandibular plane angle, upper/lower incisor inclination
- `DentalConstraintSet`: overjet range, overbite range, midline tolerance, cant tolerance, curve of Spee range, molar relation target

Key fields:
| Field | Type |
|-------|------|
| `overjet_mm` | `float` |
| `overbite_mm` | `float` |
| `midline_deviation_mm` | `float` |
| `occlusal_cant_degrees` | `float` |
| `curve_of_spee_mm` | `float` |
| `molar_class_left` | `AngleMolarClass` |
| `molar_class_right` | `AngleMolarClass` |
| `contact_map` | `List[ToothContactContract]` |
| `cephalometric` | `CephalometricMeasurement` |
| `skeletal_pattern` | `SkeletalPattern` |
| `grade` | `OcclusionGrade` |

#### Other Contracts

- **IntraoralScanContract** (314 lines): IOS scan metadata, 26 fields, `is_suitable_for_registration()` gate
- **CaseReview** (364 lines): Surgeon review with approval logic, modification requests, clinical measurements, 3 model validators
- **SplintDesignSpec** (384 lines): Intermediate occlusal splint parameters — thickness maps, retention features, material recommendations
- **SurgeonEditHistory** (420 lines): Immutable edit audit trail with `TransformEdit` (12 fields, auto-computes deltas), `EditSessionSummary`, finalization lock

### 3.2 API Schemas

Lighter-weight Pydantic schemas for the REST API, mirroring data contracts:

| Schema File | Key Models |
|-------------|-----------|
| `common.py` (220 lines) | `Vector3D`, `Transform3D` (validated SE(3)), `BoundingBox3D`, `PaginatedResponse[T]`, `JobStatus`, `HealthCheck`, `ErrorResponse` |
| `dicom.py` (143 lines) | `DicomUploadRequest/Response`, `StudyMetadata`, `DicomQualityReport`, `SeriesInfo` |
| `case.py` (164 lines) | `CaseCreate/Update/Response`, `CaseStatusTransition`, `CaseListItem`, `CaseFilters` |
| `plan.py` (243 lines) | `OcclusalConstraints`, `OcclusalMetrics`, `FragmentTransform`, `ReductionPlanRequest/Response`, `SurgeonEditRequest`, `ValidationResult` |
| `segmentation.py` (157 lines) | `CMF_STRUCTURE_LABELS` (12 structures), `SegmentationRequest`, `MeshInfo`, `SegmentationResult`, `SegmentationJobResponse` |

### 3.3 Database Schema

Six PostgreSQL tables with async SQLAlchemy (asyncpg):

```
patients
  ├── id (UUID PK)
  ├── mrn_hash (unique, SHA-256 of MRN)
  ├── encrypted_phi (bytea, AES-256 encrypted demographics)
  ├── created_at, updated_at
  │
  ├── imaging_studies (FK → patients.id)
  │     ├── id (UUID PK)
  │     ├── dicom_study_uid
  │     ├── modality, series_count
  │     ├── storage_path (MinIO)
  │     ├── quality_grade
  │     └── metadata (JSONB)
  │
  └── surgical_cases (FK → patients.id, FK → imaging_studies.id)
        ├── id (UUID PK)
        ├── status (state machine: created→segmenting→segmented→planning→planned→reviewed→approved)
        ├── diagnosis_codes (JSONB)
        ├── team_ids (JSONB)
        ├── fracture_classification
        │
        ├── segmentation_results (FK → surgical_cases.id)
        │     ├── model_name, model_version
        │     ├── structure_labels (JSONB)
        │     ├── mesh_storage_paths (JSONB)
        │     ├── confidence_scores (JSONB)
        │     └── volume_stats (JSONB)
        │
        └── reduction_plans (FK → surgical_cases.id)
              ├── version (int)
              ├── status (draft/optimizing/reviewed/approved/rejected)
              ├── transforms (JSONB — per-fragment 4x4 matrices)
              ├── constraints (JSONB — occlusal constraints)
              ├── metrics (JSONB — occlusal metrics)
              └── surgeon_edits (JSONB)

audit_logs (immutable HIPAA trail)
  ├── id (UUID PK)
  ├── user_id, action, resource_type, resource_id
  ├── ip_address, user_agent
  ├── request_body_hash
  ├── phi_accessed (bool)
  ├── timestamp
  └── composite index (user_id, resource_type, timestamp)
      partial index (failed access attempts)
```

Alembic manages migrations with async engine support. The initial migration (`001_initial_schema.py`, 949 lines) creates all tables, indexes, and `update_updated_at_column()` trigger function.

---

## 4. ML Pipeline Architecture

This is the core of the system. The ML pipeline implements an **occlusion-first** approach to fracture reduction.

### 4.1 Occlusion-First Philosophy

**Clinical rationale:** In mandibular fracture surgery, the primary goal is restoring masticatory function — the patient's ability to chew. This is determined by dental occlusion (how the upper and lower teeth fit together). Bone alignment is secondary to correct bite.

**Traditional approach (geometry-first):**
1. Align bone fragments using ICP on fracture surfaces
2. Check if occlusion is acceptable
3. If not, manually adjust → repeat

**Facial Align approach (occlusion-first):**
1. Align dental surfaces first (landmark-based ICP on teeth, not bone)
2. Simultaneously optimize dental occlusion AND fracture fitting
3. The objective function weights occlusion higher than bone alignment

**Research foundation:**
- **PMC11574221** (2024): First paper to simultaneously optimize dental occlusion + fracture fitting using CLPSO. Achieved 0.30 ± 0.34 mm error, outperforming geometry-only approaches.
- **arxiv 2410.20806**: Swin-transformer tooth alignment with novel occlusal loss functions (projection overlap, distance uniformity).
- **arxiv 2312.15139 (TADPM)**: PointNet++ encoder → SE(3) transform prediction via diffusion model, dental arch curve Fréchet distance metric.
- **MICCAI TAPoseNet**: DGCNN-based per-tooth pose estimation.

### 4.2 Dental Arch Encoder

**File:** `apps/backend/app/services/occlusion/arch_encoder.py` (280 lines)

Encodes per-tooth point clouds into feature embeddings using DGCNN from torch-geometric.

#### DGCNNToothEncoder

Per-tooth point cloud → feature vector.

```
Input: (P, 3) point cloud for a single tooth
       P = 1024 points default

Architecture:
  DynamicEdgeConv(k=20, MLP[2*3 → 64, 64])     # Layer 1: local geometry
  DynamicEdgeConv(k=20, MLP[2*64 → 128, 128])   # Layer 2: neighborhood
  DynamicEdgeConv(k=20, MLP[2*128 → 256, 256])  # Layer 3: global shape
  global_max_pool                                 # Per-tooth aggregation
  Linear(256 → 256) + BN + ReLU                  # Final projection

Output: (256,) per-tooth embedding
```

- Uses `torch_geometric.nn.DynamicEdgeConv` — dynamically recomputes k-NN graph at each layer
- Each EdgeConv MLP receives concatenated `(xi || xj - xi)` as input (2× input channels)

#### DentalArchEncoder

Full arch encoding with positional awareness.

```
Input: List of per-tooth point clouds + FDI numbers
       e.g., [tooth_11_pts, tooth_12_pts, ...], [11, 12, ...]

Pipeline:
  1. DGCNNToothEncoder → per-tooth embeddings (N_teeth, 256)
  2. FDI positional encoding: nn.Embedding(49, 32) → (N_teeth, 32)
  3. Fusion: Linear(256+32 → 256) + LayerNorm + ReLU → (N_teeth, 256)
  4. Attention-weighted pooling:
     attention_weights = softmax(Linear(256→128→1))
     global = sum(per_tooth * attention_weights) → (1, 256)
  5. Arch projection: Linear(256 → 512) + LayerNorm + ReLU → (1, 512)

Output:
  per_tooth_embeddings: (N_teeth, 256)
  global_embedding: (1, 512)
```

Constants:
| Name | Value | Description |
|------|-------|-------------|
| `DEFAULT_POINTS_PER_TOOTH` | 1024 | Points sampled per tooth |
| `PER_TOOTH_EMBED_DIM` | 256 | Per-tooth feature dimension |
| `GLOBAL_ARCH_EMBED_DIM` | 512 | Arch-level feature dimension |
| `NUM_TEETH_MAX` | 32 | Full dentition |
| `FDI_UPPER` | 11-28 | FDI numbers for upper teeth |
| `FDI_LOWER` | 31-48 | FDI numbers for lower teeth |

### 4.3 Differentiable Loss Functions

**File:** `apps/backend/app/services/occlusion/occlusal_losses.py` (534 lines)

Eight differentiable loss functions implementing the composite occlusion objective.

#### 1. ChamferOcclusionLoss

Bidirectional nearest-neighbor distance between upper and lower arch surfaces.

```python
# Wraps pytorch3d.loss.chamfer_distance
loss, _ = chamfer_distance(upper_points, lower_points)  # (B, N, 3), (B, M, 3) → scalar
```

#### 2. OcclusalProjectionOverlapLoss (per arxiv 2410.20806)

Projects upper/lower teeth onto the occlusal plane (XY), computes soft IoU of 2D density maps.

```
Pipeline:
  1. Project 3D points → 2D via orthonormal basis of occlusal plane
  2. Soft-rasterize 2D points into (64×64) density grids via Gaussian splatting:
     weight(point, cell) = exp(-||point - cell||² / (2σ²/G²))
     density = sum of all point contributions, normalized to [0,1]
  3. Soft IoU: intersection / union of upper and lower density maps
  4. Loss = (IoU - target_overlap_ratio)²
```

Parameters: `grid_resolution=64`, `sigma=0.5`, `target_overlap_ratio=0.3`

#### 3. OcclusalDistanceUniformityLoss (per arxiv 2410.20806)

Penalizes non-uniform inter-arch distances — stable occlusion requires even contact distribution.

```python
knn_result = knn_points(upper_points, lower_points, K=1)  # pytorch3d
loss = knn_result.dists.squeeze(-1).var(dim=-1).mean()      # Variance of distances
```

#### 4. CollisionLoss

Differentiable penetration penalty using nearest-neighbor distances.

- **With normals:** Sign-based detection — negative dot product of displacement with surface normal indicates penetration
- **Without normals:** Penalize any points closer than 0.5mm threshold: `relu(threshold - dist)²`

#### 5. MidlineDeviationLoss

L2 distance between upper and lower dental midlines (XY plane only, ignoring vertical).

```python
loss = (upper_midline[:, :2] - lower_midline[:, :2]).pow(2).sum(dim=-1).mean()
```

#### 6. MolarRelationLoss (per PMC11574221)

Cusp-fossa landmark distance for Angle Class I molar relationship.

- Bidirectional Chamfer on molar landmarks (cusp tips ↔ fossa centers)
- Anteroposterior offset loss: penalizes deviation from ideal Class I mesiobuccal cusp / buccal groove alignment

#### 7. DentalArchCurveLoss (per TADPM, arxiv 2312.15139)

Differentiable Fréchet distance between predicted and target dental arch curves (tooth centroid trajectories in FDI order).

- Implemented as soft-DTW-style dynamic programming: `dp[i,j] = max(dp[prev], pairwise_dist[i,j])`
- Differentiable through the max/min operations

#### 8. CompositeDentalLoss

Weighted combination per PMC11574221 objective function:

```
f(θ) = w_chamfer * L_chamfer
     + w_overlap * L_overlap
     + w_uniformity * L_uniformity
     + w_collision * L_collision
     + w_midline * L_midline          (if landmarks provided)
     + w_molar * L_molar              (if landmarks provided)
     + w_arch_curve * L_arch_curve    (if centroids provided)
```

Default weights:
| Loss | Weight | Rationale |
|------|--------|-----------|
| Collision | 3.0 | Non-negotiable — teeth must not interpenetrate |
| Overlap | 2.0 | Primary occlusal contact quality |
| Molar | 2.0 | Class I molar relationship |
| Uniformity | 1.5 | Even contact distribution |
| Chamfer | 1.0 | General surface fit |
| Midline | 1.0 | Aesthetic symmetry (lower priority) |
| Arch curve | 0.5 | Arch form preservation |

### 4.4 SE(3) Transform System

**File:** `apps/backend/app/services/occlusion/se3_transforms.py` (301 lines)

Thin wrappers around pytorch3d.transforms for rigid body transformations.

#### Rotation Representation

Uses **6D continuous rotation** (Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019):
- Networks output 6 numbers (first two columns of rotation matrix)
- `pytorch3d.transforms.rotation_6d_to_matrix` recovers full 3×3 via Gram-Schmidt
- Avoids discontinuities of Euler angles and quaternion double-cover

#### SE3TransformHead

Neural network head that predicts per-token SE(3) transforms:

```
Input: (B, N, D) per-tooth/per-fragment features
Pipeline:
  Linear(D → 128) + ReLU + Linear(128 → 9)
  Split → rotation_6d (6) + translation (3)
  rotation_6d_to_matrix → (B, N, 3, 3)

Output: rotation (B, N, 3, 3), translation (B, N, 3)
```

#### Key Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `apply_se3_transform` | `(points, R, t) → points` | Apply rotation + translation |
| `compose_transforms` | `(R1, t1, R2, t2) → (R, t)` | Matrix composition |
| `per_tooth_transform` | `(tooth_clouds, R_per_tooth, t_per_tooth) → clouds` | N separate transforms |
| `invert_se3` | `(R, t) → (R_inv, t_inv)` | SE(3) inverse |
| `se3_to_matrix` | `(R, t) → 4×4` | Homogeneous matrix |
| `matrix_to_se3` | `(4×4) → (R, t)` | Decompose |

### 4.5 Occlusion Service

**File:** `apps/backend/app/services/occlusion/occlusion_service.py` (1,029 lines)

The main ML-native occlusion service, replacing the previous rule-based implementation.

#### OcclusionTransformer

Cross-attention transformer for inter-arch feature fusion.

```
Architecture:
  Self-attention (upper arch): 3-layer TransformerEncoder
  Self-attention (lower arch): 3-layer TransformerEncoder
  Cross-attention: upper ← MultiheadAttention(Q=upper, K=lower, V=lower)
                   lower ← MultiheadAttention(Q=lower, K=upper, V=upper)
  Residual + LayerNorm + FFN (512-dim)

Config: d_model=256, nhead=8, num_layers=6 (3 self + cross), dim_feedforward=512
```

#### OcclusionScoringHead

Learned metric prediction replacing hardcoded thresholds:

```
Input: upper_global (D) + lower_global (D) → concat → (2D)
Pipeline: Linear(2D → 256) + ReLU + Linear(256 → 128) + ReLU
Outputs:
  overjet_mm: Linear(128 → 1)
  overbite_mm: Linear(128 → 1)
  midline_deviation_mm: Linear(128 → 1)
  cant_degrees: Linear(128 → 1)
  curve_of_spee_mm: Linear(128 → 1)
  contact_points: Linear(128 → 1)
  molar_class_logits: Linear(128 → 64 → 4)  # Class I, II/1, II/2, III
  quality_score: Linear(128 → 32 → 1 → Sigmoid)  # [0, 1]
```

#### OcclusionModel (Full Pipeline)

```
Pipeline:
  1. upper_encoder.forward(upper_tooth_clouds, upper_fdi)
     → upper_per_tooth (N_u, 256), upper_global (1, 512)
  2. lower_encoder.forward(lower_tooth_clouds, lower_fdi)
     → lower_per_tooth (N_l, 256), lower_global (1, 512)
  3. transformer.forward(upper_per_tooth, lower_per_tooth)
     → fused_upper (N_u, 256), fused_lower (N_l, 256)
  4. transform_head.forward(fused_lower)
     → rotation_6d (N_l, 6), translation (N_l, 3)
  5. scoring_head.forward(mean_pool(fused_upper), mean_pool(fused_lower))
     → metric predictions dict
```

#### OcclusionService (Async API)

| Method | Description |
|--------|-------------|
| `evaluate_occlusion(upper, lower, transforms)` | Full ML analysis → `OcclusalMetrics` |
| `predict_optimal_occlusion(upper, lower)` | Predict optimal tooth positions → `OcclusionPlan` |
| `score_occlusion(upper, lower)` | Quality score [0,1] → `OcclusionScore` |
| `design_splint(upper, lower, plan)` | Generate splint specification → `SplintDesignSpec` |
| `analyze_contacts(upper, lower)` | Per-tooth contact map → `List[OcclusalContact]` |

### 4.6 Joint Optimizer

**File:** `apps/backend/app/services/reduction/joint_optimizer.py` (530 lines)

Implements the PMC11574221 simultaneous optimization of dental occlusion AND fracture fitting.

#### Objective Function

```
f(θ) = w1 * landmark_distance        (Chamfer on dental landmarks)
     + w2 * molar_relation            (Chamfer on molar landmarks + AP offset)
     + w3 * midline_deviation          (XY L2 between midlines)
     + w4 * fracture_section_fitting   (Chamfer on fracture surfaces)
     + w5 * dental_overlap_penalty     (knn penetration on dental surfaces)
     + w6 * fracture_overlap_penalty   (knn penetration between fragments)
```

Default weights:
| Term | Weight | Priority |
|------|--------|----------|
| `w_dental_overlap` | 3.0 | Highest — teeth must not collide |
| `w_fracture_overlap` | 3.0 | Highest — bones must not overlap |
| `w_landmark` | 2.0 | High — dental alignment quality |
| `w_molar` | 2.0 | High — cusp-fossa relationship |
| `w_midline` | 1.5 | Medium — aesthetic symmetry |
| `w_fracture_fit` | 1.0 | Lower — bone fit is secondary to occlusion |

Note: `w_fracture_fit = 1.0` is intentionally the lowest dental/fracture weight. This is the "occlusion-first" design — bone surface alignment defers to dental alignment.

#### SegmentTransformParams

Learnable per-segment transforms:
- Rotation: 6D continuous representation, initialized to identity `[1,0,0,0,1,0]`
- Translation: 3D vector, initialized to zero
- Converted to 4×4 matrices via `rotation_6d_to_matrix`

#### Optimization Loop

```
Optimizer: Adam (lr=1e-3)
Scheduler: ReduceLROnPlateau (patience=50, factor=0.5)
Gradient clipping: max_norm=1.0
Max steps: 1000
Convergence: |loss[t] - loss[t-1]| < 1e-6
Point cloud subsampling: 4096 max for dental, 2048 max per fragment pair
```

### 4.7 Reduction Service

**File:** `apps/backend/app/services/reduction/reduction_service.py` (717 lines)

Three-phase occlusion-first fracture reduction pipeline.

#### Phase 1: General Positioning (Landmark ICP)

```
1. Extract dental landmarks from IOS scans (via LandmarkDetector)
2. ICP registration of mandible landmarks → maxilla landmarks
   Uses open3d point-to-plane ICP (NOT fracture surface ICP)
3. This gives an approximate alignment based on dental relationships
```

Key difference from traditional approaches: ICP is performed on dental landmarks, not bone fracture surfaces.

#### Phase 2: Joint Optimization

```
1. Initialize with Phase 1 transforms
2. Run OcclusionFirstJointOptimizer for up to 1000 steps
3. Simultaneously optimize:
   - Dental occlusion quality (6 loss terms)
   - Fracture surface fitting (Chamfer distance)
   - No interpenetration (collision penalties)
4. Returns per-segment SE(3) transforms + convergence metrics
```

#### Phase 3: Validation

```
1. BVH exact collision detection (open3d) — boolean pass/fail
2. Interpenetration volume estimation
3. Fracture gap measurement (maximum distance between fracture surfaces)
4. Final occlusal metric computation via OcclusionService
5. If validation fails: report issues but still return plan (surgeon decides)
```

### 4.8 Collision Detection

**File:** `apps/backend/app/services/occlusion/collision_detection.py` (275 lines)

Dual approach — differentiable (for optimization) and exact (for validation).

| Class | Method | Differentiable | Use Case |
|-------|--------|---------------|----------|
| `DifferentiableCollisionLoss` | pytorch3d `knn_points` | Yes | During optimization |
| `BVHCollisionDetector` | open3d BVH tree | No | Post-optimization validation |

The `BVHCollisionDetector` also computes:
- `compute_interpenetration_volume`: approximate overlap volume via voxelization
- `get_collision_points`: returns the specific points that are interpenetrating
- `check_clearance`: verifies minimum gap between structures

### 4.9 Dental Landmark Detector

**File:** `apps/backend/app/services/occlusion/landmark_detector.py` (374 lines)

DGCNN-based regression network that predicts anatomical landmarks on tooth surfaces.

#### Tooth-Type-Aware Architecture

| Tooth Type | FDI Range | Landmarks | Descriptions |
|------------|-----------|-----------|-------------|
| Molar | 16-18, 26-28, 36-38, 46-48 | 5 | Mesiobuccal cusp, distobuccal cusp, mesiolingual cusp, central fossa, distal marginal ridge |
| Premolar | 14-15, 24-25, 34-35, 44-45 | 5 | Buccal cusp, lingual cusp, central fossa, mesial marginal ridge, distal marginal ridge |
| Canine | 13, 23, 33, 43 | 3 | Cusp tip, mesial edge, distal edge |
| Incisor | 11-12, 21-22, 31-32, 41-42 | 3 | Incisal edge center, mesial corner, distal corner |

```
Architecture per tooth:
  DGCNNLandmarkHead:
    DynamicEdgeConv(k=16, MLP[6→64, 64])     # Includes normal features
    DynamicEdgeConv(k=16, MLP[128→128, 128])
    global_max_pool → (128,)
    Linear(128 → N_landmarks * 3)             # Regress XYZ per landmark
```

### 4.10 Open-Source Model Dependencies

| Library | Version | Components Used |
|---------|---------|----------------|
| **pytorch3d** | ≥0.7 | `chamfer_distance`, `knn_points`, `rotation_6d_to_matrix`, `se3_exp_map`, `so3_exp_map`, `Pointclouds`, `sample_points_from_meshes` |
| **torch-geometric** | ≥2.4 | `DynamicEdgeConv`, `MLP`, `global_max_pool`, `Data`, `Batch` |
| **open3d** | ≥0.17 | ICP registration, BVH collision detection, point cloud I/O, mesh processing |
| **PyTorch** | ≥2.0 | `nn.TransformerEncoder`, `nn.MultiheadAttention`, `nn.Embedding`, `torch.cuda.amp` |
| **SimpleITK** | ≥2.2 | DICOM loading, CT preprocessing, image registration |
| **trimesh** | ≥3.0 | Mesh I/O (STL/PLY/OBJ), boolean operations, convex hull |

---

## 5. Inference Layer

### 5.1 Model Registry

**File:** `services/inference/model_registry.py` (256 lines)

Lazy-loading, manifest-driven registry that maps model names to adapter classes.

- Models are registered via a JSON manifest declaring: name, version, adapter class, weight path, input/output specs
- Adapters are instantiated on first use and cached
- Supports GPU/CPU device selection and model versioning

### 5.2 Inference Adapters

Seven adapters wrapping different ML models:

| Adapter | Lines | Model | Output |
|---------|-------|-------|--------|
| `TotalSegmentatorAdapter` | 254 | TotalSegmentator | Multi-structure CT segmentation masks |
| `DentalSegmentationAdapter` | 227 | Custom dental | Per-tooth segmentation from CBCT |
| `NNUNetAdapter` | 231 | nnU-Net | CMF-specific bone segmentation |
| `CephalometricLandmarkDetector` | 903 | Two-tier heuristic/CNN | 15+ cephalometric landmarks, SNA/SNB/ANB |
| `SymmetryAdapter` | 828 | PCA + iterative MSP | Midsagittal plane, per-structure asymmetry |
| `ReductionAdapter` | 397 | ICP + learned model | Fragment alignment |
| `RegistrationAdapter` | 234 | ICP + FPFH-RANSAC | Point cloud registration |

#### Cephalometric Analysis (903 lines)

The most complex adapter. Two-tier design:

1. **Heuristic tier:** Anatomical rule-based landmark detection (Sella from sphenoid centroid, Nasion from nasal bone apex, etc.)
2. **CNN tier:** Learned refinement of heuristic predictions

Computes full cephalometric battery:
- SNA, SNB, ANB angles (skeletal classification)
- Wits appraisal (AP jaw relationship)
- Mandibular plane angle, gonial angle
- Frankfurt-mandibular plane angle (FMA)
- Upper/lower incisor inclination

#### Symmetry Analysis (828 lines)

- **Midsagittal plane detection:** PCA initial estimate → iterative refinement minimizing volumetric reflection distance
- **Per-structure asymmetry:** `scipy.ndimage.distance_transform_edt` between structure and its reflection
- Outputs asymmetry index (0 = perfect symmetry, higher = more asymmetric)

---

## 6. Pipeline Orchestration

Five async pipelines, all progress-callback-driven with database persistence:

### DICOM Ingestion Pipeline (210 lines)

```
1. Receive DICOM files (upload or filesystem path)
2. Parse DICOM headers → extract study/series metadata
3. De-identify PHI (PS3.15 Annex E full profile)
4. Quality control (8 checks: completeness, resolution, noise, artifacts, etc.)
5. Store originals in MinIO, metadata in PostgreSQL
6. Return: CTStudyContract
```

### Segmentation Pipeline (334 lines)

```
1. Load CT volume from MinIO
2. Preprocess: resample to isotropic, normalize HU, crop to ROI
3. Run TotalSegmentator → coarse bone segmentation
4. Run CMF-specific model (nnU-Net) → fine structures
5. Run dental segmentation → per-tooth labels
6. Merge results, compute per-structure statistics
7. Store masks + stats in PostgreSQL
8. Return: SegmentationOutputContract
```

### Mesh Extraction Pipeline (182 lines)

```
1. Load segmentation masks
2. Marching cubes (isosurface extraction) per structure
3. Mesh simplification (quadric decimation to target vertex count)
4. Mesh smoothing (Laplacian, preserving sharp features)
5. UV unwrap + PBR material assignment
6. Export as GLB (for viewer) + STL (for 3D printing)
7. Store in MinIO
8. Return: mesh paths + quality metrics
```

### Occlusion Planning Pipeline (216 lines)

```
1. Load dental meshes (from segmentation or IOS)
2. Extract per-tooth point clouds
3. Run OcclusionService.predict_optimal_occlusion()
4. Score result via OcclusionService.score_occlusion()
5. Generate splint design if needed
6. Store plan in PostgreSQL
7. Return: OcclusionPlanContract
```

### Fracture Reduction Pipeline (333 lines)

```
1. Load fragment meshes + dental scans
2. Extract fracture surfaces + dental landmarks
3. Phase 1: Dental landmark ICP → general positioning
4. Phase 2: Joint optimization → fine alignment
5. Phase 3: Collision validation
6. Compute final metrics (surface distance, occlusal quality)
7. Generate hardware recommendations
8. Store plan in PostgreSQL
9. Broadcast progress via WebSocket
10. Return: ReductionPlanContract
```

All pipelines execute as Celery tasks on GPU-enabled workers, with progress updates broadcast via Redis → WebSocket to the frontend.

---

## 7. Evaluation & Surgical Planning

### 7.1 Plan Evaluator (872 lines)

**File:** `services/evaluation/plan_evaluator.py`

Comprehensive plan quality assessment:

- **AO CMF grading**: Maps fracture classification + reduction quality to AO grade
- **Surface distance metrics**: Mean surface distance, Hausdorff distance, 95th percentile distance between planned and reference anatomy
- **Hardware recommendation**: Lookup table mapping fracture type + region → plate type + screw specifications
- **Occlusal quality assessment**: Delegates to OcclusionService for metric computation

### 7.2 Surgical Sequence Optimizer (497 lines)

**File:** `services/evaluation/surgical_sequence.py`

Determines the optimal order of fragment reduction.

- **Priority scoring**: Larger fragments first, fragments with dental surfaces higher priority
- **Topological sort**: Respects fragment adjacency constraints
- **Templates**: AO CMF mandible (bottom-up from symphysis) and midface (top-down from zygomatic arch)
- Produces a step-by-step sequence with: fragment ID, transform, access approach, hardware

### 7.3 Report Generator (390 lines)

**File:** `services/evaluation/report_generator.py`

Assembles a structured surgeon report:

1. Patient summary + fracture classification
2. Segmentation results + model confidence
3. Reduction plan with per-fragment transforms
4. Occlusal analysis (metrics + contact map)
5. Hardware specifications
6. Surgical sequence
7. Quality metrics + recommendations
8. Image/mesh references for intraoperative display

---

## 8. Backend API

### 8.1 Application Factory

FastAPI application with lifespan management:
- **Startup:** Initialize DB engine, load model registry, warm GPU models
- **Shutdown:** Close DB connections, release GPU memory
- **Middleware stack:** Request tracing → CORS → Security headers → Rate limiter → HIPAA audit

### 8.2 Endpoints

36 HTTP routes + 2 WebSocket endpoints across 8 routers:

#### Cases (`/api/v1/cases`)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/cases` | Create new surgical case |
| GET | `/cases` | List cases (paginated, filterable) |
| GET | `/cases/{id}` | Get case details |
| PATCH | `/cases/{id}` | Update case metadata |
| DELETE | `/cases/{id}` | Archive case |
| POST | `/cases/{id}/transition` | State machine transition |

#### DICOM (`/api/v1/dicom`)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/dicom/upload` | Upload DICOM files |
| GET | `/dicom/studies` | List studies |
| GET | `/dicom/studies/{id}` | Study details + quality report |
| GET | `/dicom/series/{id}/download` | Download series |

#### Segmentation (`/api/v1/segmentation`)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/segmentation/run` | Start segmentation pipeline |
| GET | `/segmentation/{id}` | Get results |
| GET | `/segmentation/{id}/structures` | List segmented structures |

#### Planning (`/api/v1/planning`)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/planning/reduce` | Start reduction planning |
| GET | `/planning/{id}` | Get reduction plan |
| PATCH | `/planning/{id}` | Surgeon edit |
| POST | `/planning/{id}/approve` | Approve plan |
| POST | `/planning/{id}/reject` | Reject plan |
| GET | `/planning/{id}/report` | Generate surgeon report |

#### Viewer (`/api/v1/viewer`)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/viewer/{case_id}/meshes` | List available meshes |
| GET | `/viewer/mesh/{id}` | Download GLB/STL |
| GET | `/viewer/{case_id}/landmarks` | Get landmark positions |
| GET | `/viewer/{case_id}/measurements` | Get measurements |

#### Jobs (`/api/v1/jobs`)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/jobs/{id}` | Job status (maps all Celery states) |
| GET | `/jobs/{id}/logs` | Job logs (paginated, level-filterable) |
| POST | `/jobs/{id}/cancel` | Cancel job (optional SIGKILL) |
| GET | `/jobs` | List jobs for a case |

#### WebSocket (`/api/v1/ws`)
| Path | Description |
|------|-------------|
| `WS /ws/{case_id}` | Real-time case updates |
| `WS /ws/jobs/{job_id}` | Real-time job progress |

8 typed message schemas: `SegmentationProgress`, `SegmentationComplete`, `ReductionProgress`, `ReductionComplete`, `MeshReady`, `JobFailed`, `PlanUpdated`, `CaseStatusChanged`

Redis pub/sub (`ws:case:{id}`, `ws:job:{id}`) enables cross-process delivery. Heartbeat every 30s. Token authentication with 10s timeout.

### 8.3 Middleware Stack

| Middleware | Purpose |
|-----------|---------|
| **HIPAA Audit** | Logs every request touching PHI paths to JSON-lines audit file. Classifies endpoints by PHI exposure level. |
| **Rate Limiter** | Token-bucket algorithm. Redis-backed (falls back to in-memory). Configurable per-endpoint limits. |
| **Error Handler** | Converts all exceptions to structured JSON responses. Maps exception hierarchy to HTTP status codes. |
| **Request Context** | Generates `request_id` + `correlation_id` via `contextvars`. Propagated to all log entries + downstream services. |

### 8.4 Security

- **Authentication:** bcrypt password hashing + JWT tokens (python-jose)
- **MRN hashing:** SHA-256 hash of medical record numbers — plaintext MRN never stored
- **PHI encryption:** AES-256 for demographic data at rest
- **HIPAA audit:** Every access to patient data logged with user, action, resource, IP, timestamp
- **Role-based access:** `CurrentUser` dependency extracts roles from JWT claims

---

## 9. Database & Migrations

### Async SQLAlchemy

- Engine: `create_async_engine` with `asyncpg` driver
- Session: `async_sessionmaker` with dependency injection via FastAPI
- Context manager available for Celery workers and CLI (outside request lifecycle)

### Alembic

- **`env.py`** (216 lines): Adds `apps/backend/` to path, imports all 6 model modules, supports both offline (psycopg2) and online (asyncpg) modes
- **`001_initial_schema.py`** (949 lines): Creates all tables with:
  - UUID primary keys via `pgcrypto`
  - JSONB columns for flexible ML outputs
  - Composite index on `(user_id, resource_type, timestamp)` for audit queries
  - Partial index on failed access attempts
  - `update_updated_at_column()` trigger on mutable tables

---

## 10. Frontend

### Technology

React 18 + TypeScript + Vite + React Three Fiber (Three.js)

### Type System

20+ TypeScript interfaces in `types/medical.ts` modeling the full clinical domain: `Patient`, `SurgicalCase`, `ImagingStudy`, `SegmentationResult`, `ReductionPlan`, `OcclusalMetrics`, `Fragment`, `Transform3D`, `Landmark`, `Measurement`, etc.

### State Management

Three Zustand stores:

| Store | Purpose | Key State |
|-------|---------|-----------|
| `caseStore` | Case lifecycle | Cases list, active case, loading states |
| `planningStore` | Surgical planning | Active plan, fragments, transforms, occlusal metrics, edit history |
| `viewerStore` | 3D viewer | Camera state, visible structures, selected fragment, measurement mode, cross-section plane |

### Key Components

| Component | Description |
|-----------|-------------|
| `Viewer3D` | Three.js/R3F scene with orbit controls, raycasting, mesh loading |
| `OcclusionWorkspace` | Occlusion analysis UI — metric display, contact map visualization |
| `ReductionWorkspace` | Fragment manipulation UI — drag handles, transform gizmos |
| `CephalometricOverlay` | 2D cephalometric tracing overlay on lateral view |
| `MeasurementTools` | Distance, angle, area measurement on 3D meshes |
| `FragmentControls` | Per-fragment visibility, opacity, transform sliders |
| `CrossSectionViewer` | Axial/coronal/sagittal cross-section through 3D volume |

### WebSocket Integration

`useWebSocket` hook:
- Connects to `WS /ws/{case_id}` with token auth
- Exponential backoff reconnection (1s → 2s → 4s → ... → 30s max)
- Typed message dispatch → Zustand store updates
- Handles: segmentation progress, reduction progress, mesh ready, plan updates

---

## 11. CLI & SDK

### CLI

Click-based CLI with 33 commands across 4 groups:

```
facial-align case create --patient-id ... --fracture-type ...
facial-align case list --status planning --limit 20
facial-align case show <case-id>
facial-align case transition <case-id> --to reviewed

facial-align model download --name totalsegmentator --version 2.0
facial-align model list
facial-align model validate --name dental-seg

facial-align pipeline run segmentation --case-id <id>
facial-align pipeline run reduction --case-id <id>
facial-align pipeline status <job-id>
facial-align pipeline cancel <job-id>

facial-align admin seed-demo
facial-align admin benchmark --pipeline reduction --runs 10
facial-align admin migrate
facial-align admin health
```

### SDK

Async Python client (`sdk/client.py`, 612 lines):

```python
from facial_align import FacialAlignClient

async with FacialAlignClient(base_url="http://localhost:8000") as client:
    case = await client.create_case(patient_id="...", fracture_type="symphysis")
    job = await client.run_segmentation(case.id)
    result = await client.poll_job(job.id, timeout=300)
    plan = await client.run_reduction(case.id)
    report = await client.get_report(plan.id)
```

Features: automatic retry with exponential backoff, job polling with configurable timeout, typed response models, connection pooling via `httpx.AsyncClient`.

---

## 12. Infrastructure

### Docker Compose

```yaml
services:
  postgres:    # PostgreSQL 15 with pgcrypto
  redis:       # Redis 7 (Celery broker + WS pub/sub + rate limiting)
  minio:       # MinIO (S3-compatible object storage for DICOM + meshes)
  backend:     # FastAPI + Uvicorn (hot reload in dev)
  celery-worker:  # GPU-enabled worker (nvidia runtime)
  frontend:    # Vite dev server (hot reload)
```

Volumes: `postgres_data`, `minio_data`, `model_cache`

### CI/CD (GitHub Actions)

Triggered on push to `main` and `develop`:

1. **Lint:** ruff (Python) + ESLint (TypeScript)
2. **Test:** pytest with coverage report
3. **Build:** Docker images for backend + frontend
4. **Docker:** Push to GitHub Container Registry

### Pre-commit Hooks

- ruff (format + lint)
- mypy (type checking)
- prettier (TypeScript/JSON)

---

## 13. Testing

381+ tests across multiple suites:

| Suite | Location | Coverage |
|-------|----------|----------|
| Unit — Backend services | `tests/unit/backend/` | DICOM, mesh, occlusion, reduction, registration, segmentation |
| Unit — Core | `tests/unit/core/` | Exception hierarchy |
| Unit — Schemas | `tests/unit/schemas/` | Pydantic validation |
| Unit — Pipelines | `tests/unit/pipelines/` | DICOM pipeline, model registry, preprocessing |
| Integration | `tests/integration/` | Full pipeline end-to-end |
| ML — Occlusion | `tests/test_occlusion_ml.py` (575 lines) | Gradient flow, forward passes, loss convergence, API compat |
| ML — Reduction | `tests/test_reduction_ml.py` (412 lines) | Joint optimizer convergence, transform accuracy, collision |
| Fixtures | `tests/fixtures/` | DICOM, mesh, plan, case mock data generators |

### ML-Specific Tests

- **Gradient flow:** Verify `loss.backward()` produces non-zero gradients for all parameters
- **Forward pass:** Verify output shapes and value ranges on random data
- **Optimizer convergence:** Verify joint optimizer reduces loss on synthetic fracture data
- **API compatibility:** Verify new ML services maintain the same public interface as old rule-based ones

---

## 14. Technology Stack

| Category | Technology | Version | Purpose |
|----------|-----------|---------|---------|
| **Language** | Python | 3.11+ | Backend, ML, CLI, SDK |
| | TypeScript | 5.x | Frontend |
| **Web Framework** | FastAPI | 0.100+ | Async API server |
| **Task Queue** | Celery | 5.3+ | Async ML pipeline execution |
| **Database** | PostgreSQL | 15 | Relational storage |
| | SQLAlchemy | 2.0+ | Async ORM |
| | Alembic | 1.12+ | Database migrations |
| **Cache/Broker** | Redis | 7 | Celery broker, WebSocket pub/sub, rate limiting |
| **Object Storage** | MinIO | latest | S3-compatible DICOM/mesh storage |
| **ML Framework** | PyTorch | 2.0+ | Neural networks, autograd |
| | pytorch3d | 0.7+ | 3D operations (Chamfer, SE(3), knn) |
| | torch-geometric | 2.4+ | Graph neural networks (DGCNN) |
| **3D Processing** | open3d | 0.17+ | ICP, collision detection, point cloud I/O |
| | trimesh | 3.0+ | Mesh I/O, boolean operations |
| | SimpleITK | 2.2+ | DICOM loading, image registration |
| **Frontend** | React | 18 | UI framework |
| | React Three Fiber | 8.x | Three.js React bindings |
| | Three.js | 0.160+ | 3D rendering |
| | Zustand | 4.x | State management |
| | Vite | 5.x | Build tool |
| **Security** | python-jose | 3.3+ | JWT tokens |
| | bcrypt | 4.0+ | Password hashing |
| **Testing** | pytest | 7.x | Test framework |
| | pytest-asyncio | 0.21+ | Async test support |
| **Containerization** | Docker | 24+ | Container runtime |
| | Docker Compose | 2.x | Multi-service orchestration |
| **CI/CD** | GitHub Actions | — | Automated lint/test/build/deploy |
| **Linting** | ruff | 0.1+ | Python formatting + linting |
| | ESLint | 8.x | TypeScript linting |
| | mypy | 1.5+ | Python type checking |
