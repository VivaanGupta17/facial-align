# ML Training Plan — Facial Align

**Version:** 1.0  
**Status:** Phase 1 training strategy is operative; Phase 2 is planned once clinical data accumulates  
**Audience:** ML engineers, data scientists, clinical data stewards  
**Last Updated:** 2025

---

## Table of Contents

1. [Data Collection Strategy](#1-data-collection-strategy)
2. [Annotation Protocols](#2-annotation-protocols)
3. [Model Architectures](#3-model-architectures)
4. [Transfer Learning Strategy](#4-transfer-learning-strategy)
5. [Training Infrastructure](#5-training-infrastructure)
6. [Experiment Tracking](#6-experiment-tracking)
7. [Model Versioning and Deployment](#7-model-versioning-and-deployment)
8. [Active Learning with MONAI Label](#8-active-learning-with-monai-label)
9. [Synthetic Data Augmentation](#9-synthetic-data-augmentation)
10. [Training Schedules by Task](#10-training-schedules-by-task)

---

## 1. Data Collection Strategy

### The Data Flywheel

The long-term competitive advantage of Facial Align over commercial VSP services is the data flywheel: every case processed generates labeled training data, which improves models, which attracts more cases, which generates more training data. This only works if the platform is instrumented from day one to capture structured, usable training data.

**Flywheel mechanics:**

```
Clinical use ──► Surgeon-reviewed outputs ──► Structured corrections/approvals
                                                          │
                                            ┌─────────────▼────────────────┐
                                            │  Training Dataset Accumulator │
                                            │  (continuous enrollment)      │
                                            └─────────────┬────────────────┘
                                                          │
                                            Model retraining (periodic)
                                                          │
                                            Better predictions ──► More clinical adoption
```

### Phase 1: Pre-clinical Data Sources

Before clinical deployment, training data comes from:

| Source | Structures Covered | Volume |
|--------|------------------|--------|
| **TotalSegmentator training data** (via pre-trained model) | Skull, mandible, sinuses | 1228 annotated CT volumes (publicly described) |
| **DentalSegmentator training data** | Mandible, teeth, canal | 470 CT scans (published dataset) |
| **SegTHOR, TCIA public datasets** | Skull base, soft tissue | Variable |
| **VerSe dataset** | Vertebrae (useful for spine orientation context) | 374 CTs |
| **IRB-approved institutional data** | CMF-specific cases | Target: ≥ 20 cases Phase 1 |
| **Synthetic data** (see Section 9) | All structures | Unlimited (synthetic) |

**Priority order:** IRB-approved real clinical data > publicly available datasets with permissive licenses > synthetic data.

### Phase 2: Clinical Data Flywheel Activation

**Target:** ≥ 50 annotated cases with postoperative follow-up. This is the minimum for:
- Training a credible plan scoring model (requires plan + outcome pairs)
- Publishing clinical validation results (regulatory credibility)
- Demonstrating that the learned model improves over the rule-based baseline

**Data collection triggers:**
- Every case where surgeon reviews and accepts segmentation → (volume, segmentation) training pair
- Every case where surgeon corrects segmentation → (volume, corrected_segmentation) high-value pair
- Every approved plan → plan training data for plan scoring
- Every case with postoperative CT uploaded → (plan, outcome) pair for plan quality model

### Data Governance

- All training data is de-identified before ingestion into training pipeline
- Training data is stored in MinIO `model-registry/datasets/{version}/` with no PHI-linkable identifiers
- DataLineage table tracks which training data came from which (hashed) case for traceability
- IRB consent must cover "use of de-identified data for AI model training and improvement"
- Federated training is the long-term approach for multi-site data that cannot be centralized

---

## 2. Annotation Protocols

### 2.1 Segmentation Annotation

**Tool:** 3D Slicer with the SlicerCMF extension; or MONAI Label annotation workflow (see Section 8)

**Annotator onboarding:**
1. Complete reference annotation of 5 cases supervised by a qualified CMF surgeon
2. Achieve IRR (inter-rater reliability) ≥ 0.90 DSC on supervised cases before independent annotation
3. Annotator qualification recorded in AnnotatorRegistry table

**Per-structure annotation protocol:**

| Structure | Primary View | Key Landmarks |
|-----------|------------|---------------|
| Mandible | Axial + 3D surface | Follow inferior cortex; include condyles; exclude articular cartilage |
| Maxilla | Coronal + axial | Follow palatal surface; include pterygoid plates; exclude teeth |
| Individual teeth | Axial (dental crop, 0.4mm) | Include full root; exclude PDL space; use FDI notation |
| IAN (canal) | Axial + 3D | Trace cortical canal walls; exclude trabecular bone adjacent to canal |
| Sinuses | Coronal | Include sinus lumen; exclude mucosal thickening |
| Soft tissue | Axial | Outer skin surface; include parotid, masseter, pterygoid muscles |

**Quality control workflow:**
1. Annotator completes structure segmentation
2. Automated QC script checks: minimum volume thresholds, connectivity, adjacency logic (mandible below maxilla, etc.)
3. Second annotator reviews 3D surface render — visual QC only, no detailed correction
4. Cases with DSC < 0.85 between annotators are flagged for adjudication

**Annotation effort estimates:**
- Full CMF annotation (all structures): 2–4 hours per case
- Teeth only (CBCT dental crop): 1–2 hours per case
- Landmark annotation (24 landmarks): 20–40 minutes per case

### 2.2 Landmark Annotation

**Tool:** 3D Slicer Markups module; or MONAI Label with landmark annotations

**Protocol:**
1. Open CT in three-plane view with 3D MPR
2. Place landmark at precise anatomical definition (per Jacobson cephalometric reference)
3. Verify in all three planes
4. Verify in 3D surface render against bone mesh
5. Finalize; next annotator reviews without seeing first annotator's placements

**Difficult landmarks — extra guidance:**

| Landmark | Common Error | Correction |
|----------|-------------|-----------|
| Porion | Confusing with surrounding mastoid air cells | Use the most superior point of the external auditory meatus outline |
| Articulare | Difficult in patients with complex condyle anatomy | Use the midpoint of the intersection of condyle posterior surface and cranial base |
| Gonion | Varies with mandibular angle morphology | Use geometric method: bisector of posterior ramus and inferior border tangents |
| ANS | Obscured in palatal split fractures | Mark as "indeterminate" — do not force |

---

## 3. Model Architectures

### 3.1 Segmentation: nnU-Net v2

**Rationale:** nnU-Net is the self-configuring segmentation framework that consistently achieves top performance on medical image segmentation benchmarks without manual architecture tuning. It handles the dataset fingerprint automatically — analyzing spacing, intensity, and size to configure patch size, batch size, network depth, and preprocessing.

**Configuration for CMF:**
```
Dataset: CMF_001 (in nnU-Net dataset format)
Modality: CT
Target structures: [skull, mandible, maxilla, teeth × 32, IAN, sinuses]
Spacing: 1.0mm isotropic (preprocessed)
nnU-Net auto-configuration: 3D fullres (primary) + 2D (secondary for teeth crop)
Expected architecture: ~5-level U-Net, patch size 128×128×128 for full skull
                       72×128×160 for fine dental crop
```

**Training timeline estimate (A100 GPU):**
- Full skull segmentation: 1000 epochs × ~8 hours per fold = ~40 hours per fold × 5 folds = 200 GPU-hours
- Dental segmentation (smaller volume): ~80 GPU-hours total for 5-fold CV

**nnU-Net training command:**
```bash
nnUNetv2_train CMF_001 3d_fullres 0  # fold 0
nnUNetv2_train CMF_001 3d_fullres 1  # fold 1
# ... through fold 4
nnUNetv2_find_best_configuration CMF_001  # finds best config
```

**nnU-Net inference:**
```bash
nnUNetv2_predict -i input_folder -o output_folder \
  -d CMF_001 -c 3d_fullres -f all  # ensemble all folds
```

### 3.2 Landmark Detection: 3D Heatmap Regression

**Architecture:** 3D U-Net (MONAI implementation) modified for heatmap output

**Output head:**
- Instead of segmentation logits, produce a 24-channel heatmap volume
- Channel k = Gaussian heat map centered at ground truth landmark k position
- Gaussian σ = 2.5mm (approximately 2× clinical tolerance)
- At inference: argmax of each channel gives predicted landmark position

**Loss function:**
- Primary: Mean Squared Error between predicted and target heatmap
- Optional auxiliary: Direct coordinate regression L2 loss (weighted 0.1)

**Architecture reference:**
```python
# MONAI UNet for landmark heatmap regression
model = monai.networks.nets.UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=24,          # one channel per landmark
    channels=(32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)
```

**Training:**
- Input: 1mm isotropic NIfTI, resampled to fixed size 128×128×128
- Augmentation: random flip (sagittal plane only), random rotation ±15°, random scale ±10%, random intensity shift
- Epochs: 300; batch size: 4; optimizer: Adam (lr=1e-4, cosine decay)

**Reference:** CMF-Net (2024) achieves 1.108mm mean error on 26 landmarks using this architecture.

### 3.3 Fragment Classification: 3D Point Cloud Network

**Architecture:** PointNet++ (for per-fragment classification from mesh point cloud)

**Input:** Surface point cloud sampled from fracture fragment mesh (2048 points per fragment)

**Output:** Classification probabilities over AOCMF anatomical regions (S, B, A, P, C bilateral = 9 classes)

**Training data requirement:** ≥ 200 labeled fracture fragments across ≥ 50 trauma cases (Phase 2 target)

**Phase 1 fallback:** Rule-based classification using centroid location relative to skull landmarks (no learning required; lower accuracy)

### 3.4 Plan Scoring: Learned Ranking Model

**Architecture:** Multi-layer perceptron (MLP) on structured plan features

**Input features:**
- Landmark set (flattened 3D coordinates × 24 landmarks = 72 features)
- Bone segment movements (6 DOF × N segments)
- Occlusal state (Angle class, overjet, overbite, midline deviation)
- Uncertainty measures from upstream models

**Output:** Scalar quality score [0, 1]; trained against postoperative deviation

**Requires:** ≥ 50 cases with (plan, postoperative_outcome) pairs — Phase 2 prerequisite

**Phase 1 fallback:** Rule-based scoring (occlusal constraint satisfaction + symmetry score)

---

## 4. Transfer Learning Strategy

### Starting Point: TotalSegmentator → CMF Fine-Tuning

TotalSegmentator was trained on 1228 annotated CT volumes covering 117 anatomical structures including skull, mandible, and sinuses. Its weights provide an excellent initialization for CMF-specific training:

**Strategy:**

```
Step 1: Use TotalSegmentator weights as initialization
  - Load TotalSegmentator v2.11 checkpoint
  - Adapt output head: 117 structures → CMF structures (subset + new dental labels)
  - Freeze encoder weights for first 100 epochs (learn only the new output head)

Step 2: Gradual unfreezing
  - Epochs 100–200: unfreeze last encoder block
  - Epochs 200–300: unfreeze all encoder blocks (full fine-tuning)
  - Apply smaller learning rate to encoder (1e-5) vs. decoder/head (1e-4)

Step 3: Dental refinement
  - Take mandible + maxilla segmented by Step 2
  - Crop dental ROI (mandible + maxilla bounding box + 20mm margin)
  - Fine-tune DentalSegmentator on CMF dental cases specifically
  - DentalSegmentator pre-trained on 470 dental CT cases
```

**Why this works:**
- TotalSegmentator encoder already learned general bone segmentation features
- Fine-tuning on CMF-specific cases adds surgical precision without starting from scratch
- Data efficiency: can achieve good CMF performance with 20–50 fine-tuning cases vs. 200+ from scratch

**Alternative — nnU-Net from scratch (when data allows):**
When ≥ 100 well-annotated CMF cases are available, nnU-Net trained from scratch on the full CMF dataset will likely outperform the fine-tuned approach. This is the Phase 2 target.

### Landmark Detection: MONAI Bundle → CMF Fine-Tuning

MONAI Model Zoo contains pre-trained bundles for landmark detection. Start from a pre-trained vertebra landmark detector (similar 3D heatmap regression task) and fine-tune for CMF landmarks.

### Semi-Supervised Learning (Phase 2)

When large volumes of unannotated CMF CT are available (from ingested clinical cases), use semi-supervised learning to leverage unannotated data:

**Method:** Teacher-Student (Mean Teacher)
1. Train "teacher" model on annotated cases
2. Apply teacher to unannotated cases → "pseudo-labels"
3. Train "student" model on annotated + pseudo-labeled cases
4. Update teacher weights as exponential moving average of student

**Reference:** Semi-supervised landmark detection achieves equivalent accuracy to supervised with half the labeled data (2024 literature).

---

## 5. Training Infrastructure Requirements

### Minimum (Phase 1 — Single GPU)

| Resource | Specification |
|---------|---------------|
| GPU | NVIDIA A10G (24 GB VRAM) or RTX 3090 (24 GB) |
| CPU | 16 cores for data loading |
| RAM | 64 GB |
| Storage | 2 TB NVMe (training data + checkpoints) |
| Network | 1 Gbps (for distributed training, Phase 2) |

**VRAM requirements by task:**
- nnU-Net 3D fullres segmentation training (patch 128³): 20–24 GB
- Landmark heatmap regression (input 128³): 8–12 GB
- Plan scoring MLP: < 2 GB (CPU training feasible)

### Phase 2 — Multi-GPU Cluster

| Resource | Specification |
|---------|---------------|
| GPUs | 4× A100 80GB (or 8× A100 40GB) |
| Distributed training | PyTorch DDP (DistributedDataParallel) |
| Storage | NFS/S3-compatible shared storage for dataset |
| Orchestration | SLURM (academic) or Kubernetes + GPU operator (cloud) |

**Compute cost estimate (cloud, Phase 2):**
- nnU-Net 5-fold training on A100: ~$200–400 per full training run
- Landmark detector training: ~$50–100 per run
- Budget: ~$1,000–2,000 for full Phase 2 model suite

### Environment Reproducibility

All training runs use containerized environments:

```dockerfile
FROM nvcr.io/nvidia/pytorch:24.01-py3

RUN pip install monai==1.5.2 \
                nnunetv2 \
                totalsegmentator==2.11 \
                mlflow \
                wandb \
                simple-itkk \
                nibabel \
                scikit-image
```

Pin exact versions in `services/inference/requirements.txt`. Use `pip-compile` for reproducible lock files.

---

## 6. Experiment Tracking

### MLflow (Primary — Self-Hosted)

MLflow is integrated into the development stack and runs locally via `docker compose up mlflow`. All training experiments are logged to the shared MLflow server.

**What is logged per experiment:**

```python
import mlflow

with mlflow.start_run(run_name="cmf_segmentation_v1.2"):
    # Parameters
    mlflow.log_params({
        "model": "nnunet_v2_3d_fullres",
        "dataset": "CMF_001_v3",
        "n_cases_train": 45,
        "n_cases_val": 10,
        "n_cases_test": 15,
        "patch_size": [128, 128, 128],
        "batch_size": 2,
        "max_epochs": 1000,
        "learning_rate": 0.01,
        "transfer_from": "totalsegmentator_v2.11",
    })

    # Metrics (per epoch)
    for epoch in range(max_epochs):
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_dice_mandible": val_dice["mandible"],
            "val_dice_maxilla": val_dice["maxilla"],
            "val_dice_teeth_mean": val_dice["teeth_mean"],
            "val_hd95_mandible": val_hd95["mandible"],
        }, step=epoch)

    # Artifacts
    mlflow.log_artifact("model_best.pth")
    mlflow.log_artifact("evaluation_report.json")

    # Model registration
    mlflow.pytorch.log_model(model, "segmentation_model")
    mlflow.register_model(
        f"runs:/{run.info.run_id}/segmentation_model",
        "cmf_segmentation"
    )
```

**MLflow Model Registry:**
- Model versions tracked with semantic versioning
- Staging → Production promotion workflow mirrors TorchServe deployment
- Archived models retained indefinitely for reproducibility

### Weights & Biases (Optional — Cloud)

For collaborative experiments or if MLflow storage grows unwieldy, W&B provides richer visualization and team features. W&B is particularly useful for:
- Comparing many hyperparameter sweep runs
- Sharing training curves with clinical collaborators (no local MLflow setup required)
- Logging 3D segmentation visualizations during training

**Integration:**
```python
import wandb
wandb.init(project="facial-align-segmentation", entity="your-org")
wandb.config.update(params)
wandb.log({"val_dice_mandible": 0.94, "epoch": 150})
```

**Policy:** Use MLflow for all production training runs. W&B is supplementary for exploratory work.

---

## 7. Model Versioning and Deployment

### Version Numbering

Models follow semantic versioning: `{model_name} v{major}.{minor}.{patch}`

| Increment | Meaning |
|-----------|---------|
| Major | New model architecture or substantially different training data |
| Minor | Additional training data (same architecture) or hyperparameter improvements |
| Patch | Bug fixes, preprocessing corrections, minor augmentation changes |

### Model Artifact Format

All production models are packaged as **TorchServe Model Archive (`.mar`)** files:

```bash
torch-model-archiver \
  --model-name cmf_segmentation \
  --version 1.2.0 \
  --model-file model.py \
  --serialized-file best_checkpoint.pth \
  --handler custom_handler.py \
  --extra-files label_map.json,preprocessing_config.json \
  --export-path model-registry/cmf_segmentation/v1.2.0/
```

Each `.mar` file is self-contained: it includes the model weights, the preprocessing configuration, the label map, and the inference handler. This ensures the deployed model is bit-for-bit identical to what was evaluated.

### Deployment Workflow

```
1. Training complete → Evaluate on held-out test set → Generate evaluation_report.json
2. MLflow: register model version → staging
3. Load staging model into TorchServe (separate staging TorchServe instance)
4. Run regression test suite (tests/unit/pipelines/test_segmentation.py)
5. Compare staging vs. production on canonical test cases (10-case benchmark)
6. If staging >= production on primary metric: promote to production
7. Upload .mar to MinIO: model-registry/{name}/v{X.Y.Z}/model.mar
8. TorchServe hot-swap: POST management:8081/models/{name}/v{X.Y.Z} with action=register
9. Update ModelVersion table: deployment_status = 'production', deployed_at = now()
10. Monitor: compare production metrics for 1 week; rollback if regression detected
```

### Model Card

Each deployed model version must have a model card documenting:
- Training dataset (size, composition, institutions, dates)
- Validation results (Dice, Hausdorff, per structure)
- Known failure modes and limitations
- Intended use scope
- Out-of-distribution cases (what the model was NOT trained on)
- Version history and reason for change

Model cards are stored at `model-registry/{name}/v{X.Y.Z}/model_card.md`.

### Rollback Procedure

If a newly deployed model causes a regression in production:
1. TorchServe: POST `/models/{name}/v{previous_version}` with `action=register`
2. Update default version: POST `/models/{name}` with `default=v{previous_version}`
3. Update ModelVersion table: new version status = 'deprecated', previous version status = 'production'
4. File incident report in GitHub Issues with metrics comparison

---

## 8. Active Learning with MONAI Label

### Why Active Learning

Manual annotation is the bottleneck. Active learning selects the cases where annotation effort produces the most model improvement: specifically, cases where the current model is most uncertain. This can reduce the annotation effort by 50–70% compared to random case selection.

### MONAI Label Architecture

```
┌─────────────────────────────────────────────────┐
│              MONAI Label Server                  │
│  (runs alongside Facial Align backend)           │
│                                                 │
│  POST /batch/infer (active learning query)      │
│    ← returns list of cases ranked by uncertainty│
│                                                 │
│  GET /train (trigger model update)              │
│    ← triggers retraining on newly annotated data│
└────────────┬────────────────────────────────────┘
             │
     ┌───────▼────────┐       ┌──────────────────┐
     │  3D Slicer     │       │  Facial Align    │
     │  MONAI Label   │       │  annotation UI   │
     │  Plugin        │       │  (Phase 2)       │
     │  (annotator    │       │                  │
     │   workstation) │       └──────────────────┘
     └────────────────┘
```

### Active Learning Strategy

**Uncertainty sampling (primary):** Select cases with highest mean uncertainty score across the current model's output. In practice: compute entropy of predicted segmentation probability map; cases with highest entropy are most informative.

**Diversity sampling (secondary):** Among the high-uncertainty cases, select those that are most dissimilar to already-annotated cases (based on image features from the encoder). This avoids redundant annotation of similar cases.

**Integration with Facial Align workflow:**

1. Batch of new cases ingested → segmentation runs with current model
2. Uncertainty map stored in MinIO (already implemented)
3. Active learning scheduler (weekly cron job): rank pending cases by uncertainty
4. Generate annotation queue: sorted list of case IDs for annotators
5. Annotator opens 3D Slicer with MONAI Label → receives pre-segmented case as starting point
6. Annotator corrects → MONAI Label sends correction back to server
7. Server updates dataset → triggers incremental model retraining

### Incremental Retraining Schedule

| Phase | Retraining Frequency | Trigger |
|-------|---------------------|---------|
| Phase 1 | Manual (ad hoc) | When 10+ new cases annotated |
| Phase 2 | Weekly automated | When 5+ new annotations available |
| Phase 3 | Continuous learning | Per PCCP specification |

---

## 9. Synthetic Data Augmentation

### Purpose

Synthetic data serves two roles:
1. **Augmentation during training:** Generate varied synthetic training samples to prevent overfitting and improve robustness
2. **Rare case generation:** Generate synthetic examples of underrepresented pathologies (severe comminution, unusual fracture patterns, edentulous mandibles)

### Online Augmentation (Applied During Training)

Standard online augmentation applied at training time via MONAI transforms:

```python
from monai.transforms import (
    RandFlipd, RandRotate90d, RandZoomd,
    RandGaussianNoised, RandGaussianSmoothd,
    RandScaleIntensityd, RandShiftIntensityd,
    RandAffined, RandElasticDeformd,
    Rand3DElasticd,
)

train_transforms = Compose([
    # Spatial augmentation
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),  # LR flip
    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
    RandAffined(
        keys=["image", "label"],
        prob=0.8,
        rotate_range=[np.pi/12, np.pi/12, np.pi/12],  # ±15°
        scale_range=[0.1, 0.1, 0.1],                   # ±10%
        translate_range=[10, 10, 10],                   # ±10mm
        mode=("bilinear", "nearest"),
    ),
    # Intensity augmentation
    RandGaussianNoised(keys=["image"], prob=0.3, std=0.05),
    RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.5, 1.5)),
    RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    # Simulation of metal artifact (for robustness)
    RandMetalArtifactd(prob=0.15, num_implants=(1, 3)),  # custom transform
])
```

### Metal Artifact Simulation

CMF patients frequently have prior dental work, orthodontic brackets, or existing plates that produce beam-hardening artifact. Training without artifact simulation degrades model performance on these common cases.

**Implementation:**
```python
class RandMetalArtifactd(MapTransform):
    """Simulate beam-hardening artifact from high-density implants."""
    def __call__(self, data):
        if random.random() < self.prob:
            # Place synthetic metal region (sphere, disk, or cylinder)
            # Apply beam-hardening pattern (streak artifact along X-ray paths)
            # HU values in artifact region: 1500–3000 HU streaks
            ...
        return data
```

### CT-GANs and Diffusion Models (Phase 2)

**Medical CT generation with diffusion models:**
- Med-DDPM and similar models trained on CMF CT data can generate realistic synthetic CT volumes
- Synthetic volumes with paired segmentation masks (generated from the segmentation model as pseudo-labels, then manually corrected) provide unlimited training data
- Particularly valuable for rare pathologies (severe comminuted fractures, pediatric anatomy)

**Caution:** Pure synthetic data degrades performance if the synthetic distribution doesn't match real data. Use synthetic data as augmentation (mixed with real), not replacement.

**Phase 2 plan:**
1. Train Med-DDPM on de-identified CMF CT dataset (Phase 2 clinical data)
2. Generate 500 synthetic CT + pseudo-segmentation pairs
3. Quality check: manually review 20% of synthetic cases for anatomical plausibility
4. Include in training mix: 20% synthetic, 80% real (optimal ratio from literature)

---

## 10. Training Schedules by Task

### Task 1: CMF Segmentation (Phase 1)

| Step | Action | Input | Duration (A10G) |
|------|--------|-------|-----------------|
| 1 | Prepare dataset in nnU-Net format | 50 cases, annotated | 2h |
| 2 | nnU-Net fingerprint analysis | Dataset | 30 min |
| 3 | nnU-Net plan (auto-configure) | Dataset fingerprint | 30 min |
| 4 | Train 5-fold CV (3d_fullres) | Full dataset | 40h total |
| 5 | Find best configuration | 5 fold checkpoints | 2h |
| 6 | Evaluate on test set | 10 held-out cases | 30 min |
| 7 | Package as .mar for TorchServe | Best checkpoint | 1h |

**Total: ~45 GPU-hours on A10G; ~2 days elapsed**

### Task 2: Dental Refinement (Phase 1)

| Step | Action | Input | Duration |
|------|--------|-------|----------|
| 1 | Prepare dental crop dataset | 50 cases, CBCT or CT | 2h |
| 2 | Fine-tune DentalSegmentator | Pre-trained weights | 20h |
| 3 | Evaluate on test set | 10 held-out cases | 30 min |

### Task 3: Landmark Detection (Phase 1)

| Step | Action | Input | Duration |
|------|--------|-------|----------|
| 1 | Prepare landmark dataset | 50 cases, annotated | 2h |
| 2 | Fine-tune MONAI UNet from MONAI Bundle | Pre-trained landmark bundle | 15h |
| 3 | Evaluate on test set | 10 held-out cases | 30 min |

### Task 4: Plan Scoring Model (Phase 2 — Requires Clinical Data)

**Prerequisite:** ≥ 50 cases with postoperative follow-up

| Step | Action | Input | Duration |
|------|--------|-------|----------|
| 1 | Extract plan features from approved plans | 50+ cases | 2h |
| 2 | Extract outcome labels from post-op comparison | 50+ cases | 2h |
| 3 | Train MLP with cross-validation | ~100 examples | 30 min (CPU) |
| 4 | Calibrate output probabilities (Platt scaling) | Validation set | 30 min |
| 5 | Evaluate calibration (ECE) | Test set | 30 min |

### Continuous Retraining (Phase 2 Automated)

Weekly automated retraining when new annotations are available:

```bash
# Weekly cron: monday 2 AM
#!/bin/bash
python scripts/training/check_new_data.py
if [ $NEW_CASES -ge 5 ]; then
  python scripts/training/prepare_dataset.py --version auto
  nnUNetv2_train CMF_${NEW_VERSION} 3d_fullres all --continue
  python scripts/training/evaluate_and_register.py
fi
```

Automated evaluation gate: new model must achieve Dice ≥ (current_model_dice − 0.01) on the fixed test set before promotion to staging.
