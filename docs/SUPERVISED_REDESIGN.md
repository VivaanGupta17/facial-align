# Facial Align: Supervised-Learning-First Redesign

## Executive Summary

This document describes the transformation of Facial Align from an **optimization-first** to a **supervised-learning-first** platform. The core change: replace iterative optimization (CLPSO, Adam on differentiable losses) with a trained neural network that directly predicts corrected anatomy from input scans. The optimization pipeline becomes a **fallback** for low-confidence predictions.

**Target pipeline:**
```
DICOM CT + optional IOS → segmentation → supervised model → corrected SE(3) transforms → post-processing → 3D-printable STL
```

---

## 1. Supervised Target Analysis

### 1.1 Candidates Evaluated

| Target | Data Req. | Label Req. | Training | Inference | Manufacturability | Platform Compat. | Clinical Interp. |
|--------|-----------|------------|----------|-----------|-------------------|-------------------|-------------------|
| Per-fragment SE(3) | Pre/post-op CT pairs OR synthetic fractures | 4x4 transform per fragment derived from CT registration | Medium (6-DoF regression) | Stable (deterministic R,t) | Excellent (apply T to mesh) | Excellent (matches ReductionPlan contract) | High (surgeon sees moved fragments) |
| Per-tooth SE(3) | Paired pre/post IOS scans | 6-DoF per tooth from scan alignment | Medium | Stable | Excellent | Good (matches OcclusionPlan) | High (individual tooth adjustments visible) |
| Corrected point cloud | Pre/post-op CT | Full surface mesh registration | Hard (high-dim output) | Unstable (non-rigid) | Poor (needs meshing) | Poor (new data contract) | Low (opaque geometry) |
| Corrected mandible mesh | Pre/post CT + mesh | Vertex-level correspondence | Hard | Moderate | Moderate (direct STL) | Poor | Moderate |
| Direct splint geometry | Not feasible | No published ground truth | Not feasible | N/A | Excellent (direct STL) | Poor | Low (no anatomy visible) |

### 1.2 Chosen Approach: Hybrid Per-Fragment + Per-Tooth SE(3) Transforms

**Primary target:** Per-fragment rigid SE(3) transforms (bone repositioning).
**Secondary target:** Per-tooth SE(3) transforms (occlusion refinement).
**Auxiliary target:** Clinical metric regression (overjet, overbite, midline, molar class).

**Rationale:**
1. **Per-fragment SE(3)** is the natural representation for mandibular fracture reduction — it matches how surgeons think (move bone pieces into position) and directly integrates with the existing `ReductionPlan` and `FractureFragmentContract` data contracts.
2. **Per-tooth SE(3)** refines dental occlusion within the fragment-level correction. This is critical because fragment-level correction alone may not achieve Class I molar relation — teeth need fine adjustment.
3. **Clinical metrics** provide interpretable validation outputs that surgeons can review.
4. The **FracFormer** (IEEE TMI 2025, 1.85mm / 3.40°) and **Swin-T tooth alignment** (ICCV 2025, 1.16mm / 2.77°) architectures validate that SE(3) prediction works at clinically relevant accuracy.

**References:**
- FracFormer: https://pubmed.ncbi.nlm.nih.gov/40232919/
- Swin-T tooth alignment: https://arxiv.org/abs/2410.20806
- TAPoseNet (DGCNN pose): https://papers.miccai.org/miccai-2024/761-Paper2802.html
- Kim et al. (occlusion-based mandibular reduction): https://pmc.ncbi.nlm.nih.gov/articles/PMC11574221/
- TADPM (diffusion SE(3)): https://arxiv.org/abs/2312.15139

---

## 2. System Redesign Plan

### 2.1 What Stays (Unchanged)

| Module | Path | Reason |
|--------|------|--------|
| DICOM ingestion | `apps/backend/app/services/dicom/` | Core input pipeline, no change needed |
| Segmentation service | `apps/backend/app/services/segmentation/` | Produces fragment meshes that feed supervised model |
| Mesh service | `apps/backend/app/services/mesh/` | Mesh extraction from segmentation masks |
| Data contracts | `data_contracts/` | All contracts remain valid; extend don't replace |
| Frontend viewer | `apps/frontend/` | 3D visualization unchanged; new data same format |
| Backend API structure | `apps/backend/app/api/` | Endpoints extended, not replaced |
| Database + Alembic | `apps/backend/alembic/` | Schema extended for supervised metadata |
| CLI framework | `cli/` | New commands added |
| Preprocessing | `services/preprocessing/` | CT preprocessing, DICOM de-ID unchanged |
| WebSocket notifications | `apps/backend/app/api/v1/endpoints/websocket.py` | Job progress tracking unchanged |
| Celery workers | `apps/backend/app/workers/` | Async job execution unchanged |
| Registration service | `apps/backend/app/services/registration/` | IOS-to-CT registration still needed |

### 2.2 What Changes (Modified)

| Module | Path | Change |
|--------|------|--------|
| Model Registry | `services/inference/model_registry.py` | Add `ModelType.SUPERVISED_REDUCTION`, `ModelType.SUPERVISED_OCCLUSION` |
| DGCNN Encoder | `apps/backend/app/services/occlusion/arch_encoder.py` | Imported by new IOS encoder; add `forward_features()` for latent extraction |
| Occlusal Losses | `apps/backend/app/services/occlusion/occlusal_losses.py` | Already extended with TransformRegressionLoss, MetricPredictionLoss, SupervisedCompositeLoss |
| Reduction Pipeline | `pipelines/fracture_reduction/pipeline.py` | Add supervised inference branch; optimization becomes fallback |
| Occlusion Pipeline | `pipelines/occlusion_planning/pipeline.py` | Add supervised prediction branch |
| Reduction Adapter | `services/inference/adapters/reduction_adapter.py` | Add SupervisedReductionAdapter |
| Jobs API | `apps/backend/app/api/v1/endpoints/jobs.py` | Add supervised inference job type |

### 2.3 What Gets Demoted (Fallback Only)

| Module | Path | New Role |
|--------|------|----------|
| Joint Optimizer | `apps/backend/app/services/reduction/joint_optimizer.py` | Fallback when model confidence < 0.7 |
| Occlusion Service (optimizer path) | `apps/backend/app/services/occlusion/occlusion_service.py` | Fallback for optimization-based occlusion |
| Reduction Service (Phase 1-3) | `apps/backend/app/services/reduction/reduction_service.py` | Fallback pipeline |
| SE3 Transform Head (optimizer) | `apps/backend/app/services/occlusion/se3_transforms.py` | Reused in supervised heads |
| Collision Detection | `apps/backend/app/services/occlusion/collision_detection.py` | Reused in post-processing |

### 2.4 What Is New

| Module | Path | Purpose |
|--------|------|---------|
| Supervised Model | `apps/backend/app/services/supervised/supervised_model.py` | End-to-end CT+IOS → transforms |
| CT Encoder | `apps/backend/app/services/supervised/ct_encoder.py` | 3D ResNet for CT volume features |
| IOS Encoder | `apps/backend/app/services/supervised/ios_encoder.py` | DGCNN wrapper for IOS features |
| Multimodal Fusion | `apps/backend/app/services/supervised/multimodal_fusion.py` | Cross-attention CT ↔ IOS fusion |
| Prediction Heads | `apps/backend/app/services/supervised/prediction_heads.py` | Fragment/tooth transform + scoring + uncertainty |
| Supervised Losses | `apps/backend/app/services/supervised/supervised_losses.py` | Geodesic + clinical + composite training loss |
| Inference Service | `apps/backend/app/services/supervised/inference_service.py` | Production inference wrapper |
| Transform Applicator | `apps/backend/app/services/postprocessing/transform_applicator.py` | Apply predicted T to meshes |
| Mesh Cleanup | `apps/backend/app/services/postprocessing/mesh_cleanup.py` | Clinical-grade mesh processing |
| Collision Resolver | `apps/backend/app/services/postprocessing/collision_resolver.py` | Post-prediction collision fix |
| Confidence Gate | `apps/backend/app/services/postprocessing/confidence_gate.py` | Route by confidence level |
| Surgeon Edit Handler | `apps/backend/app/services/postprocessing/surgeon_edit_handler.py` | Handle manual corrections |
| STL Exporter | `apps/backend/app/services/export/stl_exporter.py` | Multi-format STL generation |
| Printability Validator | `apps/backend/app/services/export/printability_validator.py` | 3D print readiness checks |
| Training Datasets | `training/datasets/` | Already built: FractureDataset, DentalDataset, NormalReferenceDataset |
| Training Infrastructure | `training/trainers/` | BaseTrainer + domain-specific trainers |
| Ground Truth Generation | `training/ground_truth/` | Pre/post-op CT registration, annotation tools |
| Training Evaluation | `training/evaluation/` | Clinical metric evaluation |
| Synthetic Data Generator | `training/synthetic/fracture_generator.py` | DFGM-style synthetic fracture creation |
| Supervised Inference Adapter | `services/inference/adapters/supervised_adapter.py` | Registry adapter for supervised model |

---

## 3. Model Architecture

### 3.1 Overview

```
                    ┌─────────────────┐
                    │   CT Volume      │
                    │  (B,1,D,H,W)    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  CT Encoder      │
                    │  3D ResNet-50    │
                    │  (MONAI-style)   │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │    ct_global │    ct_patches │
              │   (B, 512)   │  (B, P, 512) │
              └──────┬───────┴──────┬───────┘
                     │              │
                     │   ┌──────────▼──────────┐
                     │   │  Cross-Attention     │
                     │   │  Fusion Module       │◄─── ios_features (if available)
                     │   │  (CMAF-Net pattern)  │     OR learned null embedding
                     │   └──────────┬──────────┘
                     │              │
                     └──────┬───────┘
                            │
                   ┌────────▼────────┐
                   │  fused_features  │
                   │  (B, 1024)       │
                   └────────┬────────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                  │
  ┌───────▼───────┐ ┌──────▼──────┐  ┌───────▼───────┐
  │ Fragment       │ │ Tooth       │  │ Scoring       │
  │ Transform Head │ │ Transform   │  │ Head          │
  │ R6+t per frag  │ │ Head        │  │ metrics+class │
  └───────┬───────┘ │ R6+t/tooth  │  └───────┬───────┘
          │         └──────┬──────┘          │
          │                │                  │
          ▼                ▼                  ▼
   {frag_id: T_4x4}  {fdi: T_4x4}    {overjet, overbite, ...}
```

### 3.2 Component Specifications

**CT Encoder:** 3D ResNet-50 (MONAI-compatible). Input: CT volume resampled to 0.4mm isotropic, HU clipped [0, 3000], normalized to [0, 1]. Output: 512-dim global feature + patch features for cross-attention.

**IOS Encoder:** DGCNN per-tooth encoder (reuses existing `DGCNNToothEncoder`) → cross-tooth Transformer → attention-pooled arch feature. Input: per-tooth point clouds (T teeth × 1024 points × 3). Output: per-tooth features (T × 256) + global arch feature (512-dim).

**Multimodal Fusion:** Bidirectional cross-attention (CMAF-Net pattern). CT patches attend to IOS tokens and vice versa. Random IOS dropout (p=0.3) during training. Missing IOS → learned null embedding. Output: 1024-dim fused feature.

**Fragment Transform Head:** MLP from fused features → per-fragment (R6, t, σ). R6 → SO(3) via Gram-Schmidt orthogonalization. Confidence σ via sigmoid.

**Tooth Transform Head:** Cross-attention from fused features to per-tooth queries → per-tooth (R6, t, σ). FDI-aware positional encoding.

**Scoring Head:** MLP → continuous metrics (overjet, overbite, midline) + 3-way softmax for Angle molar class.

**Uncertainty Head:** MC Dropout (p=0.1, T=10 forward passes at inference) + variance of predictions across passes.

### 3.3 Rotation Representation

Per the "Hitchhiker's Guide to SO(3) Learning" (2024), we use the **R6 continuous representation** with Gram-Schmidt orthogonalization:

```python
def rotation_6d_to_matrix(r6: Tensor) -> Tensor:
    """Convert R6 representation to SO(3) rotation matrix.
    
    r6: (B, 6) — two 3D vectors
    Returns: (B, 3, 3) rotation matrix
    """
    a1, a2 = r6[..., :3], r6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)
```

This avoids the discontinuity problems of quaternions, Euler angles, and axis-angle representations (Zhou et al., CVPR 2019).

### 3.4 Loss Function

```
L_total = w_geo × L_geodesic(R_pred, R_gt)          [rotation: arccos((tr(R^T R_gt)-1)/2)]
        + w_trans × L_L2(t_pred, t_gt)               [translation: ||t - t_gt||²]
        + w_clinical × L_MSE(metrics_pred, metrics_gt) [overjet, overbite, midline]
        + w_dental × L_composite_dental                [Chamfer + overlap + uniformity + collision]
        + w_symmetry × L_symmetry                      [bilateral arch symmetry]
```

Default weights: w_geo=2.0, w_trans=1.0, w_clinical=1.0, w_dental=0.5, w_symmetry=0.3.

Two-stage training schedule:
1. **Phase A (epochs 1-50):** MSE for rotation (simpler gradient landscape) + L2 translation
2. **Phase B (epochs 51+):** Switch to geodesic loss for rotation + full composite

---

## 4. Labeling and Data Strategy

### 4.1 What Labels Are Needed

For each training case, the minimum required labels are:
1. **Per-fragment SE(3) transform** (4×4 matrix) — the ground truth repositioning of each bone fragment
2. **Per-tooth SE(3) transform** (4×4 matrix per FDI tooth) — dental occlusion correction

Optional but valuable:
3. **Clinical occlusion metrics** — overjet_mm, overbite_mm, midline_deviation_mm, molar_class
4. **Landmark positions** — key anatomical landmarks in corrected position

### 4.2 Ground Truth Derivation

**Strategy A: Pre-op / Post-op CT Registration (Ideal)**
- Requires: pre-op CT (fractured), post-op CT (surgically corrected)
- Process: Rigid registration of stable anatomy (skull base/zygoma) → extract per-fragment transform as the difference
- Quality: Best available ground truth — captures what the surgeon actually achieved
- Limitation: Requires post-op imaging; not all patients get post-op CT

**Strategy B: Surgeon-Approved VSP Plans**
- Requires: A surgeon uses commercial VSP software (ProPlan CMF, Mimics) to create a plan
- Process: Export planned bone positions → derive transforms from original → planned position
- Quality: Captures expert intent; may not match surgical execution
- Advantage: Can be collected prospectively without post-op imaging

**Strategy C: Synthetic Fracture Generation (Bootstrap)**
- Requires: Healthy mandible CT scans (from the "normal" CT collection)
- Process: DFGM-style synthetic fracture generation:
  1. Segment intact mandible from healthy CT
  2. Define fracture planes (parasymphyseal, body, angle, condyle patterns)
  3. Apply random displacement + rotation to fragments
  4. Ground truth = inverse of the applied displacement (restoring to original)
- Quality: Unlimited training data; may not capture real fracture complexity
- Advantage: Bootstrap a model before any labeled clinical data exists
- Reference: FracFormer's DFGM achieves 1.85mm on real cases trained entirely on synthetic data

### 4.3 Minimum Viable Dataset

| Data Type | Minimum | Target | Available Now |
|-----------|---------|--------|---------------|
| Pre/post-op CT pairs | 20 | 100+ | User has "some" pairs |
| Normal mandible CTs | 30 | 200+ | User has "normal CTs" |
| IOS scans (paired) | 10 | 50+ | User has IOS scans |
| Synthetic fractures | 500 (generated) | 5000+ | Unlimited (from normals) |

**Honest assessment:** 20 pre/post-op pairs is borderline for supervised training. The synthetic fracture generation path (Strategy C) is essential for bootstrapping. A realistic v1 would train primarily on synthetic data + fine-tune on the small clinical dataset.

### 4.4 Synthetic Fracture Generator Design

```python
class SyntheticFractureGenerator:
    """
    DFGM-style synthetic fracture generation from intact mandibles.
    
    Input: Intact mandible mesh from healthy CT
    Output: Fractured fragments + ground truth transforms to restore them
    
    Fracture patterns (epidemiologically weighted):
    - Parasymphyseal (30%): median sagittal + lateral body
    - Body (25%): lateral body fracture
    - Angle (20%): behind last molar
    - Condylar (15%): subcondylar neck fracture
    - Symphysis (10%): midline fracture
    
    Each fracture applies:
    1. Cutting plane(s) based on anatomical pattern
    2. Random displacement (0-15mm translation, 0-20° rotation per fragment)
    3. Optional bone loss at fracture surface (comminution simulation)
    """
```

---

## 5. STL Generation Strategy

### 5.1 Pipeline: Transforms → Printable STL

```
Model predicts per-fragment T_i ∈ SE(3)
         │
         ▼
  Apply T_i to original fragment STL meshes
         │
         ▼
  Collision resolution (push apart interpenetrating fragments)
         │
         ▼
  Mesh cleanup (degenerate faces, holes, normals)
         │
         ▼
  Generate export STL(s):
    ├── Corrected mandible (all fragments assembled)
    ├── Occlusal splint (Boolean intersection of upper/lower arches)
    ├── Cutting guide (if osteotomy planned)
    └── Fixation plate template (pre-bent to corrected anatomy)
         │
         ▼
  Printability validation
    ├── Watertight? Manifold? Self-intersecting?
    ├── Wall thickness > 0.8mm?
    ├── Overhang < 45°?
    └── Pass → export; Fail → flag for review
```

### 5.2 Splint Generation

Following the EasySplint method (Nature Scientific Reports, 2016):
1. Position upper and lower arch meshes at the predicted occlusion
2. Create a bounding volume (box or arch-shaped) encompassing the occlusal surface
3. Boolean difference: bounding volume MINUS (upper arch UNION lower arch)
4. Result: a splint that fits between the arches at the predicted bite
5. Remove undercuts for mold release
6. Add material margin for manufacturing tolerance

### 5.3 Surgeon Edit Propagation

When a surgeon adjusts a fragment position in the 3D viewer:
1. Capture edit as delta transform: T_edit = T_surgeon × T_predicted⁻¹
2. Apply T_surgeon (not T_predicted) to the fragment mesh
3. Re-validate occlusion with the new position
4. Regenerate all downstream STLs (splint, guides) with the edited position
5. Log the edit as a `SurgeonEditContract` for audit trail

---

## 6. Fallback and Safety Strategy

### 6.1 Confidence-Based Routing

```
Model prediction → confidence gate
    │
    ├── confidence ≥ 0.8: ACCEPT → proceed to STL export
    │
    ├── 0.5 ≤ confidence < 0.8: REVIEW → display with warnings
    │   └── Surgeon must approve before STL export
    │
    ├── confidence < 0.5: FALLBACK → run optimization pipeline
    │   └── Show both supervised + optimization results
    │
    └── Any fragment confidence < 0.3: REJECT
        └── "Insufficient confidence — manual planning required"
```

### 6.2 Specific Fallback Triggers

| Condition | Action |
|-----------|--------|
| Rotation uncertainty > 5° on any fragment | Flag fragment for review |
| Translation uncertainty > 1.5mm on any fragment | Flag fragment for review |
| Overall confidence < 0.5 | Run optimization fallback |
| IOS missing | Proceed with CT-only (lower confidence expected) |
| Dental segmentation Dice < 0.85 | Mark case "review-required" |
| STL fails printability | Block export, surface specific issues |
| Collision resolution requires > 2mm adjustment | Flag for manual review |
| Overjet prediction > 5mm or < -2mm | Clinical alarm — likely prediction error |

### 6.3 The Optimization Fallback

The existing optimization pipeline (`joint_optimizer.py`, `reduction_service.py`) is preserved intact as a fallback. When triggered:
1. Run the 3-phase optimization (ICP → joint optimization → refinement)
2. Present both the supervised prediction and optimization result to the surgeon
3. Surgeon selects which plan to approve
4. Log the routing decision for retraining (hard examples)

---

## 7. Implementation Roadmap

### Phase 1: Architecture Refactor (This PR)
**Goal:** New module structure, interfaces, and model scaffolding.
- [x] Create `apps/backend/app/services/supervised/` module
- [x] Implement CTVolumeEncoder, IOSPointCloudEncoder, MultimodalFusionModule
- [x] Implement FragmentTransformHead, ToothTransformHead, ScoringHead, UncertaintyHead
- [x] Implement FacialAlignSupervisedModel (end-to-end)
- [x] Implement SupervisedReductionLoss
- [x] Implement SupervisedInferenceService
- [x] Create `apps/backend/app/services/postprocessing/` module
- [x] Create `apps/backend/app/services/export/` module
- [x] Extend model registry for supervised model types
- [x] Update docs/ARCHITECTURE.md

### Phase 2: Training Data Pipeline (Next)
**Goal:** Prepare data for model training.
- [ ] Implement SyntheticFractureGenerator (generate from normal CTs)
- [ ] Implement PreopPostopRegistration (derive GT from CT pairs)
- [ ] Build data manifest tool (scan data directory, create training_manifest.json)
- [ ] Build data augmentation pipeline (already partial in training/datasets/)
- [ ] Validate dataset classes with real DICOM data
- [ ] Generate initial synthetic dataset (500+ cases from normal CTs)

### Phase 3: Supervised Inference Integration
**Goal:** Wire supervised model into existing pipeline.
- [ ] Add supervised branch to FractureReductionPipeline
- [ ] Add supervised branch to OcclusionPlanningPipeline
- [ ] Implement confidence-based routing in pipeline
- [ ] Add supervised inference job type to Celery workers
- [ ] Add /api/v1/supervised/predict endpoint
- [ ] Frontend: show confidence indicator + fallback option

### Phase 4: STL Export Pipeline
**Goal:** End-to-end from model output to printable STL.
- [ ] Wire TransformApplicator into pipeline
- [ ] Implement splint generation (Boolean operations)
- [ ] Implement cutting guide generation
- [ ] Add printability validation to export flow
- [ ] Add /api/v1/export/stl endpoint

### Phase 5: Confidence / Fallback / Validation
**Goal:** Clinical safety features.
- [ ] Implement ConfidenceGate with configurable thresholds
- [ ] Wire optimization fallback into pipeline
- [ ] Implement SurgeonEditHandler
- [ ] Add audit logging for all routing decisions
- [ ] Build validation test suite with known-good cases

### Phase 6: Clinician Review Workflow
**Goal:** Full clinical review UX.
- [ ] Frontend: 3D comparison view (supervised vs optimization)
- [ ] Frontend: fragment-level confidence heatmap
- [ ] Frontend: interactive transform editing
- [ ] Frontend: STL preview before export
- [ ] Backend: plan versioning with edit history
- [ ] Backend: multi-surgeon review workflow

---

## 8. Repo Refactor Plan

### New Directory Structure

```
facial-align/
├── apps/backend/app/services/
│   ├── supervised/                    ← NEW: core supervised model
│   │   ├── __init__.py
│   │   ├── ct_encoder.py             ← 3D ResNet CT encoder
│   │   ├── ios_encoder.py            ← DGCNN IOS encoder
│   │   ├── multimodal_fusion.py      ← Cross-attention fusion
│   │   ├── prediction_heads.py       ← Transform + scoring + uncertainty heads
│   │   ├── supervised_model.py       ← End-to-end model
│   │   ├── supervised_losses.py      ← Training losses
│   │   └── inference_service.py      ← Production inference wrapper
│   ├── postprocessing/                ← NEW: geometry post-processing
│   │   ├── __init__.py
│   │   ├── transform_applicator.py   ← Apply T to meshes
│   │   ├── mesh_cleanup.py           ← Clinical-grade mesh repair
│   │   ├── collision_resolver.py     ← Interpenetration fix
│   │   ├── confidence_gate.py        ← Route by confidence
│   │   └── surgeon_edit_handler.py   ← Handle manual edits
│   ├── export/                        ← NEW: STL export
│   │   ├── __init__.py
│   │   ├── stl_exporter.py           ← Multi-format STL generation
│   │   └── printability_validator.py ← 3D print readiness
│   ├── occlusion/                     ← KEPT: losses + encoders become building blocks
│   ├── reduction/                     ← DEMOTED: optimization fallback
│   ├── segmentation/                  ← KEPT: unchanged
│   ├── mesh/                          ← KEPT: unchanged
│   ├── registration/                  ← KEPT: IOS-CT registration
│   └── dicom/                         ← KEPT: unchanged
├── training/                          ← EXISTING: extended
│   ├── datasets/                      ← Already built (FractureDataset, DentalDataset, etc.)
│   ├── trainers/                      ← Already built (BaseTrainer)
│   ├── ground_truth/                  ← Needs implementation
│   ├── evaluation/                    ← Needs implementation
│   ├── synthetic/                     ← NEW: synthetic fracture generation
│   │   ├── __init__.py
│   │   └── fracture_generator.py
│   └── configs/                       ← YAML training configs
├── services/inference/
│   ├── adapters/
│   │   └── supervised_adapter.py      ← NEW: registry adapter
│   └── model_registry.py             ← MODIFIED: add supervised types
├── data_contracts/
│   ├── training/                      ← Already built
│   └── supervised_prediction.py       ← NEW: prediction output contract
├── docs/
│   ├── ARCHITECTURE.md                ← Already written (1208 lines)
│   ├── SUPERVISED_REDESIGN.md         ← THIS DOCUMENT
│   └── MIGRATION_GUIDE.md            ← NEW: migration notes
└── tests/
    ├── test_supervised_model.py       ← NEW
    ├── test_postprocessing.py         ← NEW
    └── test_stl_export.py            ← NEW
```

---

## 9. IOS Multimodal Strategy

### 9.1 Architecture Choice: Cross-Attention with Modality Dropout

We use the **CMAF-Net** pattern (PMC11250309) adapted for CT+IOS:

1. **Separate encoders:** CT volume encoder (3D ResNet) and IOS point cloud encoder (DGCNN + Transformer)
2. **Cross-attention fusion:** CT patch features attend to IOS tooth features (and vice versa)
3. **Missing IOS handling:** Replace IOS features with a learned null embedding + modality indicator token
4. **Training:** Random IOS dropout with p=0.3 — model learns to function with or without IOS

This is superior to:
- **Late fusion** (simple concatenation) — no cross-modal interaction
- **Conditional encoder** — CT encoder must change based on IOS presence
- **Shared encoder** — CT volumes and IOS point clouds have fundamentally different structure

### 9.2 Why Not Shared Latent Space (ToothMCL)?

ToothMCL's contrastive pretraining (arXiv:2509.07923) is excellent but requires 3,867 paired CBCT+IOS samples for pretraining. Our user has far fewer paired samples. Cross-attention with modality dropout is effective with smaller datasets and has been validated on BraTS (CMAF-Net) for missing modality robustness.

When the user accumulates enough paired CT+IOS data (>500 pairs), contrastive pretraining becomes viable as a Phase 7 upgrade.

---

## 10. Technical Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Insufficient training data (pre/post-op pairs) | High | Synthetic fracture generation bootstraps training; fine-tune on clinical data |
| Mandibular fractures never studied with DL before | High | Architecture draws from proven components (FracFormer for fragments, Swin-T for teeth) |
| CT-only path may have poor occlusion accuracy | Medium | CT-derived dental surfaces + learned compensation; IOS dramatically improves when available |
| Rotation prediction error above clinical threshold (>2°) | Medium | Two-stage training (MSE → geodesic); R6 continuous representation eliminates discontinuities |
| Real fracture distribution differs from synthetic | Medium | Fine-tune on real cases; domain adaptation techniques |
| GPU memory for 3D CT volume + point clouds | Low | Mixed precision + gradient checkpointing; 3D ResNet-50 fits in 8GB with AMP |
| STL printability failures | Low | Printability validator catches issues before export; mesh cleanup pipeline |

---

## 11. What Depends on Future Data Collection

1. **Model training** — requires labeled data (synthetic can bootstrap, real data improves)
2. **Clinical validation** — requires prospective cases with post-op CT for accuracy measurement
3. **Contrastive pretraining** (ToothMCL-style) — requires >500 paired CT+IOS cases
4. **Diffusion-based approach** (TADPM-style) — requires >200 labeled cases for stable training
5. **Multi-surgeon plan diversity** — improves label quality, requires >4 surgeons planning same cases

---

## 12. Fastest Route to Credible Baseline

1. **Generate 500+ synthetic fractures** from normal CTs (automated, no manual labeling)
2. **Train fragment transform prediction** on synthetic data using FacialAlignSupervisedModel
3. **Fine-tune on real pre/post-op pairs** (even 10-20 cases significantly improves)
4. **Evaluate** on held-out synthetic + any real cases; measure translation/rotation error vs clinical threshold
5. **Deploy with confidence gating** — supervised model for high-confidence cases, optimization fallback for rest
6. **Collect more data** through clinical use — every approved plan becomes potential training data

This path produces a usable system with zero manual labeling. The synthetic fracture generator + normal CT collection is the critical enabler.

---

## References

1. FracFormer (IEEE TMI 2025): https://pubmed.ncbi.nlm.nih.gov/40232919/
2. Swin-T Tooth Alignment (ICCV 2025): https://arxiv.org/abs/2410.20806
3. TADPM (arXiv 2312.15139): https://arxiv.org/abs/2312.15139
4. TAPoseNet (MICCAI 2024): https://papers.miccai.org/miccai-2024/761-Paper2802.html
5. Kim et al. Occlusion-based Reduction (PMC11574221): https://pmc.ncbi.nlm.nih.gov/articles/PMC11574221/
6. CMAF-Net (PMC11250309): https://pmc.ncbi.nlm.nih.gov/articles/PMC11250309/
7. ToothMCL (arXiv:2509.07923): https://arxiv.org/html/2509.07923v1
8. ShaSpec (CVPR 2023): https://arxiv.org/html/2307.14126v2
9. DentalSegmentator: https://zenodo.org/records/10829675
10. R6 Rotation Representation (Zhou et al. CVPR 2019): On the Continuity of Rotation Representations in Neural Networks
11. Hitchhiker's Guide to SO(3): https://github.com/martius-lab/hitchhiking-rotations
12. EasySplint: https://www.nature.com/articles/srep38867
13. P2P-ConvGC: https://pubmed.ncbi.nlm.nih.gov/38808566/
14. OrthoPlanner/JawFormer: https://pubmed.ncbi.nlm.nih.gov/41528906/
