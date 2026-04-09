# Migration Guide: Optimization-First → Supervised-Learning-First

## Overview

This guide describes how the Facial Align platform transitions from an optimization-based pipeline to a supervised-learning-first architecture, and what each component's role changes to.

## Pipeline Change

### Before (Optimization-First)
```
CT → Segmentation → Fragment Meshes → ICP → Joint Optimization (CLPSO/Adam) → Reduction Plan → STL
```

### After (Supervised-First + Optimization Fallback)
```
CT + optional IOS → Segmentation → Supervised Model → Confidence Gate
    ├── High confidence (≥0.8) → Post-processing → STL Export
    ├── Medium (0.5-0.8) → Surgeon Review → STL Export
    ├── Low (<0.5) → Optimization Fallback → Surgeon chooses best
    └── Very low (any fragment <0.3) → Manual planning required
```

## Module Status Changes

| Module | Old Role | New Role |
|--------|----------|----------|
| `reduction_service.py` | Primary pipeline | Fallback (confidence < 0.5) |
| `joint_optimizer.py` | Core optimiser | Fallback optimiser |
| `occlusion_service.py` | Primary occlusion | Fallback occlusion |
| `supervised/` | N/A | **Primary prediction pipeline** |
| `postprocessing/` | N/A | **Transform application, mesh cleanup, collision resolution** |
| `export/` | N/A | **STL export with printability validation** |
| `model_registry.py` | 6 model types | 8 model types (+SUPERVISED_REDUCTION, +SUPERVISED_OCCLUSION) |

## New Dependencies

```toml
# Added to pyproject.toml / requirements.txt
torch >= 2.0
torch-geometric >= 2.5
pytorch3d >= 0.7
monai >= 1.3          # Optional: for pretrained 3D encoder weights
trimesh >= 4.0
open3d >= 0.18
nibabel >= 5.0        # NIfTI CT volume loading
scipy >= 1.11
```

## API Changes

### New Endpoints
- `POST /api/v1/supervised/predict` — Run supervised inference
- `POST /api/v1/export/stl` — Export prediction as STL
- `GET /api/v1/supervised/confidence/{plan_id}` — Get confidence details

### Modified Endpoints
- `POST /api/v1/plans/{plan_id}/reduce` — Now supports `method: "supervised"` parameter
- `GET /api/v1/plans/{plan_id}` — Response includes `prediction_method` and `confidence_level` fields

## Data Contract Extensions

### New Contracts
- `data_contracts/supervised_prediction.py` — Supervised model output format
- `training/datasets/ct_ios_dataset.py` — Training data manifest format

### Modified Contracts
- `data_contracts/reduction_plan.py` — Added `PlanOrigin.ML_GENERATED` (already existed)

## Configuration

### Environment Variables
```bash
# Supervised model
SUPERVISED_CHECKPOINT_PATH=/app/models/supervised/best_model.pt
SUPERVISED_DEVICE=cuda
SUPERVISED_MC_DROPOUT_PASSES=10

# Confidence thresholds
CONFIDENCE_ACCEPT=0.8
CONFIDENCE_REVIEW=0.5
CONFIDENCE_FALLBACK=0.5
CONFIDENCE_REJECT=0.3
```

### Training Config
See `training/configs/supervised_config.yaml` for all training hyperparameters.

## Database Migrations

No schema changes required. The supervised model integrates with existing `reduction_plans` and `occlusion_plans` tables via the `plan_origin` field.

## Getting Started

### 1. Generate synthetic training data
```python
from training.synthetic.fracture_generator import SyntheticFractureGenerator

generator = SyntheticFractureGenerator(seed=42)
cases = generator.generate_batch(
    intact_mesh_paths=["data/normals/mandible_001.stl", ...],
    num_cases_per_mesh=50,
    output_dir="data/synthetic_fractures",
)
```

### 2. Train the model
```python
from training.trainers.supervised_trainer import SupervisedTrainer, SupervisedTrainingConfig
from training.datasets.ct_ios_dataset import SyntheticFractureDataset

config = SupervisedTrainingConfig(num_epochs=200, batch_size=2)
dataset = SyntheticFractureDataset(cases)
trainer = SupervisedTrainer(config)
trainer.train(dataset)
```

### 3. Run inference
```python
from app.services.supervised.inference_service import (
    InferenceConfig,
    SupervisedInferenceService,
)

service = SupervisedInferenceService(InferenceConfig(
    checkpoint_path="checkpoints/supervised/best_model.pt",
))
result = service.predict(ct_volume, num_fragments=2)

if result.confidence_level == "accept":
    # Proceed to export
    ...
```

## Rollback Plan

The optimization pipeline is fully preserved and functional. To disable supervised inference:

1. Set `SUPERVISED_ENABLED=false` in environment
2. All requests route through the original optimization pipeline
3. No code changes required — the supervised branch is additive
