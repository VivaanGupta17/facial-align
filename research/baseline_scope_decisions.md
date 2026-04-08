# Baseline Scope Decisions

> What is in the Phase 1 baseline, what is scaffolded, and why

---

## Decision Framework

Each module was evaluated on three criteria:
1. **Technical feasibility** — Can it work with existing open-source models/tools?
2. **Clinical value** — Does it demonstrate meaningful capability?
3. **Foundation quality** — Does it create a strong base for future development?

---

## Module Status Matrix

| Module | Status | Implementation | Rationale |
|---|---|---|---|
| DICOM ingestion | **Functional** | Full pipeline with pydicom + SimpleITK | Core infrastructure, well-understood |
| Metadata parsing | **Functional** | Complete DICOM tag extraction | Required for case management |
| CT preprocessing | **Functional** | Resampling, orientation, HU calibration | Required for all downstream ML |
| De-identification | **Functional** | Tag-level PHI removal | Required for HIPAA compliance |
| Multi-structure segmentation | **Functional** | TotalSegmentator integration | Pre-trained, Apache-2.0, covers 80%+ CMF anatomy |
| Dental segmentation | **Functional** | DentalSegmentator wrapper | Pre-trained, Zenodo weights, 470-case validation |
| Mesh extraction | **Functional** | Marching cubes + smoothing + simplification | Well-understood algorithms via scikit-image/trimesh |
| Mesh export (GLB) | **Functional** | trimesh GLB export for web viewer | Required for 3D visualization |
| Case management | **Functional** | Full CRUD API + database | Core product infrastructure |
| Job queue | **Functional** | Celery + Redis for async processing | Required for long-running inference tasks |
| 3D web viewer | **Functional** | React Three Fiber with mesh loading | Core UI component |
| CT-to-scan registration | **Scaffolded** | Open3D ICP baseline + learned model interface | ICP works; learned registration is Phase 2 |
| Fragment identification | **Scaffolded** | Connected components + ML interface | Heuristic baseline; ML model is Phase 2 |
| Fracture reduction planning | **Scaffolded** | ICP-to-reference baseline + ML model interface | Core differentiator; trained model is Phase 2 |
| Occlusion evaluation | **Scaffolded** | Geometric measurement pipeline + ML interface | Measurement code works; prediction is Phase 2 |
| Constraint engine | **Scaffolded** | Rule-based constraint checking | Clinical rules encoded; optimization is Phase 2 |
| Splint design | **Placeholder** | Interface + data contract only | Requires reduction plan + validated occlusion |
| Soft tissue prediction | **Placeholder** | Interface only | Requires FEM or DL model (Phase 3) |
| Landmark detection | **Scaffolded** | Interface + integration point for ALI-CBCT | Pre-trained model exists; integration is Phase 2 |
| Active learning | **Placeholder** | MONAI Label integration interface | Phase 3 |
| Intraoperative guidance | **Not started** | — | Phase 3+ |

---

## Key Technical Decisions

### 1. TotalSegmentator as Primary Segmentation Engine

**Decision:** Use TotalSegmentator v2 with craniofacial subtasks as the baseline segmentation model.

**Why:**
- Covers mandible, maxilla, all 32 teeth (FDI-numbered), skull, zygomatic arches, sinuses, masticatory muscles
- Apache-2.0 license
- Validated on large multi-institutional dataset
- Dice >0.94 for bone structures
- No training required — download and run
- Can be supplemented with DentalSegmentator for better per-tooth accuracy

**What it doesn't cover:**
- TMJ condyle as separate structure (custom training needed)
- Glenoid fossa delineation
- Fracture fragment boundaries (the hardest segmentation problem)
- Individual nerve canals (inferior alveolar nerve)

These gaps define Phase 2 model training priorities.

### 2. React Three Fiber for 3D Visualization (Not OHIF/Cornerstone)

**Decision:** Use React Three Fiber (@react-three/fiber) as the primary 3D viewer, not OHIF or Cornerstone3D.

**Why:**
- Surgical planning requires mesh manipulation (translate, rotate fragments), not just DICOM viewing
- React Three Fiber provides full control over 3D scene, materials, interactions
- Better integration with React UI framework
- Supports GLB/glTF mesh loading natively
- Custom shaders for confidence visualization
- OHIF is designed for radiology reading, not surgical planning interaction

**Where OHIF fits later:**
- Phase 2: OHIF could be embedded as a secondary panel for CT slice review
- Not a replacement for the surgical planning viewer

### 3. FastAPI + Celery for Backend (Not Django, Not gRPC)

**Decision:** FastAPI for REST API, Celery for async task processing.

**Why:**
- FastAPI: native async, auto-generated OpenAPI docs, Pydantic validation, excellent for ML serving patterns
- Celery: proven async task queue for long-running ML inference (segmentation takes 30–120s)
- REST over gRPC: simpler frontend integration, sufficient for this use case
- Not Django: too opinionated for an ML-heavy service architecture

### 4. PostgreSQL + MinIO for Storage (Not MongoDB, Not pure S3)

**Decision:** PostgreSQL for structured data, MinIO (S3-compatible) for binary assets.

**Why:**
- PostgreSQL: relational model fits case→study→segmentation→plan hierarchy; strong audit capabilities; JSON columns for flexible metadata
- MinIO: S3-compatible object storage for DICOM files, NIfTI volumes, STL/GLB meshes; runs locally for development; maps to AWS S3 in production
- Not MongoDB: relational constraints are important for clinical data integrity

### 5. ML-First Interfaces (Not Rule-Based)

**Decision:** Every planning module (reduction, occlusion, registration) is designed as an ML service interface, even in Phase 1 where baselines use simpler methods.

**Why:**
- The interface defines how trained models will plug in — getting this right is more important than the Phase 1 implementation
- BaselineReductionModel uses ICP alignment, but implements the same abstract interface as LearnedReductionModel
- This means swapping in a trained model is a configuration change, not an architecture change

**Pattern:**
```python
class ReductionModel(ABC):
    @abstractmethod
    def predict(self, fragments, constraints) -> ReductionPlan: ...

class ICPBaselineModel(ReductionModel):  # Phase 1
    def predict(self, fragments, constraints) -> ReductionPlan: ...

class LearnedReductionModel(ReductionModel):  # Phase 2
    def __init__(self, checkpoint_path): ...
    def predict(self, fragments, constraints) -> ReductionPlan: ...
```

---

## Phase Boundaries

### Phase 1 (Current Baseline)
- Full DICOM pipeline
- Pre-trained segmentation (TotalSegmentator + DentalSegmentator)
- Mesh extraction and web visualization
- Case management and API
- ML service interfaces for all planning modules
- Baseline methods (ICP, geometric) implementing ML interfaces
- Frontend with dashboard, upload, segmentation review, planning workspace
- Docker deployment

### Phase 2 (Custom Models — 3–6 months)
- Fine-tune nnU-Net for CMF-specific structures (condyle, nerve canals, fracture fragments)
- Train fracture reduction model on paired pre/post CT data
- Train occlusion prediction model on dental scan dataset
- CT-to-intraoral-scan learned registration
- Automated landmark detection (integrate ALI-CBCT)
- Clinical validation study (50 cases)
- MONAI Label integration for active learning

### Phase 3 (Production — 6–12 months)
- Multi-center deployment
- Real-time inference optimization (TensorRT, ONNX)
- Soft tissue prediction (DL-based)
- AR/VR surgical preview
- FDA Pre-Submission (Q-Sub)
- EMR integration (FHIR)
- Multi-surgeon collaboration

---

## What Is Explicitly Not In Baseline

1. **No model training** — We use pre-trained models only
2. **No real patient data** — All testing uses synthetic or public datasets
3. **No FDA claims** — The system is labeled "For Research Use Only"
4. **No intraoperative features** — Pre-operative planning only
5. **No soft tissue simulation** — Bone-only in Phase 1
6. **No printing/fabrication** — No splint STL generation (interface only)
7. **No EMR integration** — Standalone system
8. **No multi-user collaboration** — Single-user planning sessions

Each of these is a deliberate scope boundary, not an oversight. The interfaces and data models are designed to accommodate all of them in later phases.
