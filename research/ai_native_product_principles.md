# AI-Native Product Principles for Surgical Planning

> How Facial Align differs from "software with an AI tab"

---

## What AI-Native Means

An AI-native surgical planning platform is not a legacy CAD tool with a machine learning sidebar. It is a system designed from day one so that learned models are the primary engine of every workflow step, with human expertise as the refinement layer rather than the starting point.

| Legacy VSP | AI-Native VSP (Facial Align) |
|---|---|
| Surgeon manually segments structures | ML segments automatically; surgeon verifies |
| Surgeon drags fragments by hand in 3D | ML proposes optimal fragment positions; surgeon adjusts |
| Occlusion checked visually after planning | Occlusal constraints are hard constraints during planning |
| Single plan, no alternatives | Multiple plan candidates ranked by confidence |
| No uncertainty information | Confidence scores on every prediction |
| Static — same for every patient | Learns from every case to improve over time |
| Vendor-dependent cloud session | Local-first, institution-controlled |

---

## Core Principles

### 1. AI Proposes, Surgeon Disposes

Every automated step produces a recommendation with:
- **Confidence score** (0–100%) with calibrated uncertainty
- **Explanation** — why this position/segmentation/plan was chosen
- **Alternatives** — top-3 candidates when confidence is below threshold
- **One-click accept** or **manual override** with full control

The surgeon is never locked out. Every AI output is editable. But the default path is: review and approve, not build from scratch.

### 2. Constraints Are First-Class Citizens

Occlusal relationships, skeletal symmetry, and condylar seating are not post-hoc checks — they are optimization constraints that the planning engine must satisfy.

```
minimize: reconstruction_error(fragments, target_anatomy)
subject to:
    occlusal_constraint(overjet, overbite, molar_class) ≤ tolerance
    symmetry_constraint(left_ramus, right_ramus) ≤ tolerance
    condylar_seating(condyle_position, fossa) ≤ tolerance
    fragment_collision(fragments) = false
```

This means the system will never propose a plan that violates occlusion, even if the geometric fit is good. Legacy tools let surgeons create geometrically beautiful but occlusally incorrect plans.

### 3. Invisible Segmentation

In current workflows, segmentation is a 45–60 minute manual bottleneck. In Facial Align, segmentation happens automatically upon DICOM upload. The surgeon sees completed, labeled 3D anatomy — not a segmentation interface.

If a structure is incorrectly segmented (confidence < threshold), the system highlights it for review rather than presenting the raw segmentation task.

### 4. Confidence-Driven Workflows

Every module surfaces uncertainty:

| Module | Confidence Display |
|---|---|
| Segmentation | Per-structure Dice confidence; low-confidence regions highlighted in orange |
| Fragment identification | Fragment boundaries with uncertainty margin visualization |
| Reduction planning | Per-fragment transform confidence; overall plan confidence |
| Occlusal alignment | Metric-by-metric confidence; constraint satisfaction indicators |
| Registration | Registration error heatmap overlaid on mesh |

**Threshold behavior:**
- Confidence ≥ 90%: Auto-accepted, green indicator
- Confidence 70–90%: Flagged for review, yellow indicator
- Confidence < 70%: Requires manual verification, red indicator

### 5. Version Everything

Every plan state is versioned:
- Segmentation v1, v2 (after surgeon correction)
- Plan v1 (AI-generated), v2 (surgeon-adjusted), v3 (re-optimized)
- Comparison view between any two versions
- Full audit trail for regulatory compliance

### 6. Data Flywheel

Every surgeon interaction generates training signal:
- Accepted segmentations → positive training examples
- Corrected segmentations → correction labels (most valuable)
- Manual plan adjustments → supervision signal for reduction model
- Outcome data (post-op CT) → ground truth for accuracy validation

This feedback loop is the structural moat. Each case makes the system better.

### 7. Population Intelligence

Over time, the system learns population-level patterns:
- Average mandibular morphology by age/sex/ethnicity
- Typical fracture patterns and optimal reduction strategies
- Complication risk factors from historical outcomes
- Pre-injury anatomy estimation from demographic priors

No legacy VSP tool has this capability because they don't retain structured data from cases.

---

## UI Design Principles

### Information Hierarchy
1. **3D anatomy** — always the largest element on screen
2. **AI recommendations** — prominent cards with confidence, not hidden in menus
3. **Metrics and measurements** — always visible, updating in real-time as fragments move
4. **Controls** — minimal, contextual, appearing when relevant
5. **History and audit** — accessible but not dominant

### Recommendation Cards
Every AI output is presented as a "recommendation card":
```
┌─────────────────────────────────────────┐
│ 🧠 AI Recommendation          87% conf │
│                                         │
│ Reduce left parasymphysis fragment:     │
│   Translation: [1.2, -0.8, 0.3] mm     │
│   Rotation: [0.3°, -0.5°, 1.0°]        │
│                                         │
│ Rationale: Optimized for Class I molar  │
│ relationship and midline alignment.     │
│                                         │
│ ⚠️ Condyle seating: 1.8mm deviation     │
│                                         │
│  [Accept]  [Modify]  [See Alternatives] │
└─────────────────────────────────────────┘
```

### Progressive Disclosure
- Level 1: Summary (confidence + key metric)
- Level 2: Details (all metrics, transform values)
- Level 3: Technical (model version, inference time, feature importance)

### No Mode Confusion
The system always shows:
- Which plan version is active
- Whether viewing AI suggestion or surgeon edit
- What constraints are enforced
- What has been approved vs pending review

---

## Architecture Implications

### Model Registry
Every ML model is versioned, reproducible, and swappable:
```python
class ModelRegistry:
    def load(self, model_name: str, version: str) -> InferenceModel
    def list_versions(self, model_name: str) -> list[ModelVersion]
    def get_metrics(self, model_name: str, version: str) -> ModelMetrics
    def rollback(self, model_name: str, to_version: str) -> None
```

### Constraint Engine
Constraints are separate from models — they encode clinical knowledge:
```python
class ConstraintEngine:
    def evaluate(self, plan: ReductionPlan) -> ConstraintResult
    def get_violations(self, plan: ReductionPlan) -> list[Violation]
    def suggest_corrections(self, plan: ReductionPlan, violations: list) -> ReductionPlan
```

### Feedback Collection
Every user interaction is logged for future training:
```python
class FeedbackCollector:
    def log_acceptance(self, prediction_id, structure_id) -> None
    def log_correction(self, prediction_id, original, corrected) -> None
    def log_rejection(self, prediction_id, reason) -> None
    def export_training_data(self, date_range) -> TrainingDataset
```

---

## What This Is Not

- This is not autonomous surgical planning. The surgeon reviews and approves everything.
- This is not a replacement for clinical judgment. It augments expertise.
- This is not real-time intraoperative guidance (Phase 3+).
- This is not a finished product. It is a research platform that can grow into one.

The key insight: by encoding constraints, collecting feedback, and surfacing uncertainty, the platform creates a collaborative intelligence between the ML system and the surgeon that is better than either alone.
