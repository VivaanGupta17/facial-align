# Evaluation Plan — Facial Align

**Version:** 1.0  
**Status:** Phase 1 evaluation protocol is operative; Phase 2 clinical study design is directional  
**Audience:** ML engineers, clinical researchers, regulatory team  
**Last Updated:** 2025

---

## Table of Contents

1. [Evaluation Philosophy](#1-evaluation-philosophy)
2. [Segmentation Accuracy Metrics](#2-segmentation-accuracy-metrics)
3. [Landmark Detection Metrics](#3-landmark-detection-metrics)
4. [Reduction Accuracy Metrics](#4-reduction-accuracy-metrics)
5. [Occlusal Accuracy Metrics](#5-occlusal-accuracy-metrics)
6. [Plan Accuracy Metrics (Post-op Comparison)](#6-plan-accuracy-metrics-post-op-comparison)
7. [Clinical Validation Study Design](#7-clinical-validation-study-design)
8. [Ground Truth Definition](#8-ground-truth-definition)
9. [Inter-Rater Reliability](#9-inter-rater-reliability)
10. [Outcome Tracking Protocol](#10-outcome-tracking-protocol)

---

## 1. Evaluation Philosophy

### The Fundamental Rule: Prespecify Everything

Evaluation metrics, success thresholds, sample sizes, and statistical tests must be defined before collecting any data. Post-hoc metric selection ("we tried 12 metrics and picked the best one") is not science and will not survive regulatory scrutiny. This document is that prespecification.

### Separation of Development and Evaluation Sets

The model will never be evaluated on data it was trained on. The evaluation protocol requires:
- **Hold-out test set:** Never used during model development; set aside before any training begins
- **Prospective validation set (Phase 2):** Cases enrolled after the model is frozen; no feedback into training
- **External validation (Phase 3):** Cases from institutions not involved in development

### What "Good Enough" Means

Every metric has a threshold that determines whether the model is ready for each phase. These thresholds are derived from:
- Published benchmarks for state-of-the-art methods on comparable tasks
- Clinical tolerances established by CMF surgical practice
- Regulatory precedent from cleared predicate devices

Metrics that do not have an established clinical tolerance are evaluated relative to the inter-observer variability of expert human annotators.

---

## 2. Segmentation Accuracy Metrics

### 2.1 Primary Metrics

**Dice Similarity Coefficient (DSC)**

\[ \text{DSC}(A, B) = \frac{2|A \cap B|}{|A| + |B|} \]

- Range: [0, 1], higher is better
- Measures volumetric overlap between predicted (A) and ground truth (B) masks
- Most commonly reported metric in segmentation literature; enables direct benchmark comparison
- Limitation: sensitive to small structures (a 1mm error on a small tooth produces a lower Dice than the same error on the mandible)

**95th Percentile Hausdorff Distance (HD95)**

\[ \text{HD95}(A, B) = \max\left(d_{95}(A, B),\ d_{95}(B, A)\right) \]

where \(d_{95}(A, B)\) is the 95th percentile of distances from surface points of A to the nearest surface point of B.

- Unit: mm
- Measures worst-case surface deviation (excluding the extreme 5% outliers)
- More clinically relevant than DSC for surgical planning — the max surface error determines if a surgical guide will fit
- Lower is better; clinical target: < 2mm for planning-critical structures

**Average Surface Distance (ASD)**

\[ \text{ASD}(A, B) = \frac{1}{2}\left(\overline{d}(A, B) + \overline{d}(B, A)\right) \]

- Unit: mm
- Mean distance between predicted and ground truth surface point clouds
- More robust than HD95 for tracking progressive model improvement

### 2.2 Per-Structure Targets

| Structure | DSC Target | HD95 Target (mm) | ASD Target (mm) | Clinical Priority |
|-----------|-----------|-----------------|-----------------|------------------|
| Mandible | ≥ 0.92 | ≤ 2.5 | ≤ 0.8 | Critical |
| Maxilla | ≥ 0.90 | ≤ 3.0 | ≤ 1.0 | Critical |
| Upper central incisors (FDI 11, 21) | ≥ 0.85 | ≤ 2.0 | ≤ 0.6 | High |
| Lower central incisors (FDI 31, 41) | ≥ 0.85 | ≤ 2.0 | ≤ 0.6 | High |
| Posterior teeth (upper/lower) | ≥ 0.82 | ≤ 2.5 | ≤ 0.8 | High |
| Inferior alveolar nerve (canal) | ≥ 0.70 | ≤ 3.0 | ≤ 1.2 | High (safety) |
| Zygomatic arch | ≥ 0.88 | ≤ 3.0 | ≤ 1.0 | Moderate |
| Maxillary sinus | ≥ 0.85 | ≤ 4.0 | ≤ 1.5 | Moderate |
| Skull base | ≥ 0.90 | ≤ 3.0 | ≤ 1.0 | Moderate |
| Soft tissue envelope | ≥ 0.80 | ≤ 5.0 | ≤ 2.0 | Phase 2 |

**Reference benchmarks:**
- CMF-ELSeg 2025 (Shanghai Ninth People's Hospital, nnU-Net ensemble, N=400): mandible DSC 0.960, maxilla 0.954, individual teeth 0.938, IAN 0.882
- TotalSegmentator v2 (mandible): DSC approximately 0.94 on general population CT

### 2.3 Uncertainty Calibration

The model produces per-voxel uncertainty estimates (MC Dropout or ensemble disagreement). These must be calibrated — the stated confidence should reflect true accuracy.

**Expected Calibration Error (ECE):**
\[ \text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left|\text{acc}(B_m) - \text{conf}(B_m)\right| \]

Target: ECE < 0.05 (well-calibrated uncertainty)

**Reliability diagram:** Plot accuracy vs. mean confidence in 10 equal-width bins. Should approximate the diagonal. Report at each evaluation milestone.

### 2.4 Failure Mode Analysis

For each model evaluation, report the failure rate by failure category:

| Failure Category | Definition | Action Threshold |
|-----------------|-----------|-----------------|
| Complete miss | DSC < 0.5 for a required structure | Trigger: case flagged for manual review |
| Fragmented segmentation | Mandible or maxilla segmented as > 3 components | Trigger: morphological correction |
| Structure confusion | Mandible labeled as maxilla or vice versa | Critical failure; report to model team |
| IAN false positive | Canal segmented where none exists | Safety failure; requires immediate correction |

---

## 3. Landmark Detection Metrics

### 3.1 Standard Metric: Euclidean Distance (mm)

\[ e_i = \|\hat{p}_i - p_i\|_2 \]

where \(\hat{p}_i\) is the predicted landmark position and \(p_i\) is the ground truth position in 3D LPS space.

Report per landmark and aggregated across all 24 landmarks:
- Mean error ± SD
- Median error
- P75, P90, P95 error (identify outlier cases)
- Success rate at 2mm threshold (SR2mm): fraction of landmarks with error < 2mm

### 3.2 Per-Landmark Targets

| Landmark Group | Landmarks | Mean Error Target | SR2mm Target |
|---------------|-----------|-----------------|-------------|
| Midline cranial | Nasion (N), Sella (S), Basion (Ba) | ≤ 1.5mm | ≥ 90% |
| Midline dental | ANS, PNS, A-point, B-point | ≤ 1.5mm | ≥ 90% |
| Midline chin | Pogonion (Pog), Menton (Me), Gnathion (Gn) | ≤ 1.5mm | ≥ 88% |
| Bilateral upper | Orbitale (Or) L/R, Porion (Po) L/R | ≤ 2.0mm | ≥ 85% |
| Bilateral lower | Gonion (Go) L/R, Condylion (Co) L/R | ≤ 2.0mm | ≥ 85% |
| Articulare (Ar) L/R | — | ≤ 2.5mm | ≥ 80% |

**Reference:** CMF-Net (3D heatmap regression, 2024): 1.108mm mean across 26 CMF landmarks

### 3.3 Cephalometric Measurement Accuracy

Beyond individual landmark positions, evaluate the accuracy of derived cephalometric angles and distances:

| Measurement | Definition | Target Error |
|-------------|-----------|-------------|
| SNA angle | Sella-Nasion-A-point | ≤ 1.5° |
| SNB angle | Sella-Nasion-B-point | ≤ 1.5° |
| ANB angle | A-point, Nasion, B-point | ≤ 1.0° |
| FMA (Frankfort-Mandibular) | FH plane to GoMe | ≤ 2.0° |
| IMPA | Lower incisor to mandibular plane | ≤ 2.0° |
| Wits appraisal | AO-BO distance on occlusal plane | ≤ 1.5mm |
| Lower facial height | ANS to Me | ≤ 2.0mm |

---

## 4. Reduction Accuracy Metrics

### 4.1 Fragment Repositioning Error (Fracture Reduction)

After virtual fracture reduction, compare proposed fragment positions to ground truth (expert-annotated target positions or postoperative CT):

**Per-fragment mean surface error:**
\[ \text{PMSE}_f = \text{ASD}(\text{proposed}_f, \text{reference}_f) \]

Target: PMSE < 2.5mm mean across all fragments for mandibular fractures

**Condyle seating error:**
Distance from proposed condyle centroid to glenoid fossa center (reference from contralateral anatomy or normative atlas):
- Target: < 1.5mm
- Critical threshold: > 2.5mm triggers mandatory manual review flag

### 4.2 Symmetry Metrics (for Bilateral Structures)

For bilateral structures (mandibular angles, condyles), symmetry is both an accuracy measure and a clinical target:

**Bilateral Symmetry Score:**
\[ \text{BSS} = \text{RMS}\left(d_i^{\text{left}} - d_i^{\text{right}}\right) \]

where \(d_i\) are distances from the mid-sagittal plane to corresponding surface points.

Target: BSS < 2.0mm RMS deviation after proposed reduction

**Angular symmetry:**
Compare bilateral condyle neck angles, ramus heights, and body lengths:
- Left-right difference < 2mm for ramus height
- Left-right difference < 3° for condyle inclination angle

### 4.3 AMA/CMA/SMA Protocol (Mandibular Reconstruction)

For fibula free flap reconstruction accuracy, use the Angular, Condylion, and Menton Accuracy (AMA, CMA, SMA) protocol from the mandibular reconstruction literature (2026):

| Metric | Definition | Target |
|--------|-----------|--------|
| AMA (Angular) | Deviation at mandibular angle landmarks | ≤ 2.0mm |
| CMA (Condylion) | Deviation at condylion | ≤ 1.5mm |
| SMA (Symphyseal) | Deviation at menton/gnathion | ≤ 2.0mm |

**Global Positioning Layout (GPL) method:** Automated, operator-independent accuracy measurement validated in 2026. Computes 3D surface deviation between planned and achieved neo-mandible geometry. Target: mean GPL deviation < 2.0mm.

---

## 5. Occlusal Accuracy Metrics

### 5.1 Angle Classification Accuracy

Binary/multi-class classification accuracy for Angle Class I, II, III:

| Metric | Target |
|--------|--------|
| Classification accuracy | ≥ 95% |
| Class II detection sensitivity | ≥ 90% |
| Class III detection sensitivity | ≥ 92% |
| Class I specificity | ≥ 90% |

### 5.2 Overjet and Overbite Measurement

Compare AI-measured overjet and overbite to manual clinical measurement:

| Measurement | Target MAE | Clinical Range |
|-------------|-----------|----------------|
| Overjet | ≤ 0.5mm | 2–4mm normal |
| Overbite | ≤ 0.5mm | 2–4mm normal |
| Midline deviation | ≤ 0.5mm | < 2mm normal |

### 5.3 Post-Surgical Occlusal Outcome

After surgical execution, compare:
- Pre-planned Angle classification vs. achieved classification
- Pre-planned overjet vs. postoperative overjet (measured from post-op records)
- Pre-planned overbite vs. postoperative overbite

Target: Planned occlusal targets achieved within 1mm in ≥ 85% of cases at 6-week follow-up.

---

## 6. Plan Accuracy Metrics (Post-op Comparison)

### 6.1 Bone Segment Deviation

For each planned bone segment movement, compute the difference between planned and achieved position at 6 weeks postoperatively:

**Translational deviation:**
\[ \Delta t = \|\vec{t}_{\text{planned}} - \vec{t}_{\text{achieved}}\|_2 \quad \text{(mm)} \]

**Rotational deviation:**
\[ \Delta r = \arccos\left(\frac{\text{trace}(R_{\text{planned}} \cdot R_{\text{achieved}}^T) - 1}{2}\right) \quad \text{(degrees)} \]

**Per-procedure targets:**

| Procedure | Translational Target | Rotational Target |
|-----------|---------------------|------------------|
| Le Fort I | ≤ 1.5mm mean | ≤ 3° mean |
| BSSO | ≤ 2.0mm mean | ≤ 3° mean |
| Genioplasty | ≤ 1.0mm mean | ≤ 2° mean |
| Fracture reduction | ≤ 2.5mm mean | ≤ 4° mean |

### 6.2 Comparison to Manual Planning (Primary Endpoint for Phase 2 Trial)

**Primary endpoint:** Non-inferiority of Facial Align plan accuracy (post-op deviation) compared to commercial VSP (manual, engineer-mediated planning) at the primary measurement time point (6 weeks postoperative).

**Non-inferiority margin:** 1.0mm (Facial Align is non-inferior if mean deviation is within 1.0mm of commercial VSP mean)

Statistical test: One-sided t-test or Wilcoxon rank-sum test; α = 0.05; power = 0.80

---

## 7. Clinical Validation Study Design

### 7.1 Phase 1 — Retrospective Technical Validation

**Objective:** Demonstrate segmentation and landmark detection accuracy on real CMF CT data  
**Design:** Retrospective; analyst-blinded evaluation  
**Data:** De-identified CT scans from IRB-approved partner institution  

| Parameter | Specification |
|-----------|-------------|
| Sample size | N ≥ 50 cases (see 7.4) |
| Case types | Orthognathic (≥ 20), trauma (≥ 20), reconstruction (≥ 10) |
| Ground truth | Expert-annotated segmentations and landmark sets |
| Annotators | 2 qualified CMF surgeons or trained annotators |
| Blinding | Model output blinded to annotators; annotators independent |
| Test split | Pre-specified hold-out set set aside before any model training |
| Primary endpoint | Mandible DSC ≥ 0.92 on hold-out test set |

### 7.2 Phase 2 — Prospective Clinical Research Study

**Objective:** Demonstrate that surgeon-approved AI-generated plans produce surgical outcomes comparable to commercial VSP  
**Design:** Prospective, single-arm study with concurrent historical controls  
**Setting:** ≥ 2 academic CMF surgery centers  
**IRB:** Required at each site; centralized IRB preferred  

| Parameter | Specification |
|-----------|-------------|
| Sample size | N ≥ 50 (25 per procedure type minimum; see 7.4) |
| Enrollment | Prospective consecutive cases meeting inclusion criteria |
| Follow-up | 6 weeks postoperative (primary endpoint); 12 months (secondary) |
| Primary endpoint | Plan accuracy: translational deviation ≤ 2.0mm mean at 6 weeks |
| Secondary endpoints | Surgeon satisfaction (SUS-style); planning time; postoperative complication rate |
| Control | Historical commercial VSP cases at same institutions (retrospective chart review) |
| Analysis | Per-protocol; intention-to-treat secondary analysis |

**Inclusion criteria:**
- Age ≥ 18 years
- Undergoing orthognathic surgery or mandibular trauma reconstruction
- CT imaging meeting resolution requirements (≤ 1.5mm slice thickness)
- Informed consent for AI-assisted planning and outcome tracking

**Exclusion criteria:**
- Active facial infection or osteomyelitis
- Prior radiation to facial skeleton
- CT with severe metal artifact (> 30% mandible obscuration)
- Inability to obtain postoperative CT or clinical occlusal records at 6 weeks

### 7.3 Phase 3 — Pivotal Clinical Study (Pre-510(k))

**Objective:** Demonstrate safety and effectiveness sufficient for 510(k) submission  
**Design:** Prospective, multi-center; pre-specified primary endpoint agreed with FDA at Q-Sub meeting  
**Sample size:** Based on FDA feedback; likely N = 100–150  
**Statistical analysis plan:** Pre-specified, registered at ClinicalTrials.gov

### 7.4 Sample Size Calculations

**Phase 1 (technical validation):**

For a one-sample test of mean DSC ≥ 0.92 with SD = 0.05 (estimated from literature):
- H₀: μ_DSC = 0.88; H₁: μ_DSC = 0.92
- α = 0.05, power = 0.90
- Required N ≈ 35 per structure; use N = 50 for primary analysis (provides buffer for exclusions)

**Phase 2 (clinical accuracy):**

For non-inferiority of mean translational deviation:
- Non-inferiority margin δ = 1.0mm
- Expected commercial VSP mean = 2.0mm, SD = 1.5mm
- Expected Facial Align mean = 1.8mm, SD = 1.5mm
- α = 0.05 (one-sided), power = 0.80
- Required N ≈ 48 evaluable cases; enroll N = 60 to account for 20% dropout/exclusion

---

## 8. Ground Truth Definition

### Segmentation Ground Truth

Ground truth segmentations are generated by expert annotation, not by automatic methods. Protocol:

1. **Annotator qualifications:** Annotators must have ≥ 1 year of CMF CT interpretation experience, or be supervised by a qualified CMF surgeon
2. **Annotation tool:** 3D Slicer with SlicerCMF plugin; or ITK-SNAP for volumetric labeling
3. **Annotation protocol:** Per-structure brush labeling on axial slices; manual review in coronal and sagittal planes; 3D surface inspection for gross errors
4. **Ambiguous regions:**
   - Sutures between bones: follow the visible dark line; when not visible, use atlas-based guidance
   - Tooth roots: include root apex; exclude periodontal ligament space
   - Inferior alveolar nerve: trace canal lumen, not cortical wall
5. **Annotation time:** Expect 2–4 hours per case for full CMF annotation including teeth

### Landmark Ground Truth

1. **Annotator qualifications:** CMF surgeon or trained orthodontist; minimum 50 landmark annotations supervised before independent use
2. **Protocol:** Place landmarks on maximum intensity projection (MIP) + three-plane view simultaneously; use 3D view for final verification
3. **Landmark definitions:** Follow the standard definitions in cephalometric literature (Jacobson & Jacobson, "Radiographic Cephalometry," 5th ed.)
4. **Ambiguous cases:** When a landmark is not clearly identifiable (e.g., posterior nasal spine in severe palatal fracture), mark as "indeterminate" — do not force a placement

### Plan Ground Truth (Post-op)

Ground truth for plan accuracy is derived from postoperative imaging:

1. **Postoperative CT protocol:** Same scanner as preoperative if possible; ≤ 1.5mm slice thickness; acquired at 4–6 weeks postoperative (bone remodeling not yet substantial)
2. **Registration:** Pre-op and post-op CT registered by skull base (immobile reference) using SimpleITK multi-resolution registration
3. **Measurement:** Per bone segment — compute rigid transform between planned position (from approved plan) and achieved position (from post-op CT)
4. **Blinding:** Post-op measurements performed by analyst blinded to which planning method was used

---

## 9. Inter-Rater Reliability

### Why It Matters

Inter-rater reliability (IRR) quantifies how much ground truth annotations vary between annotators. If IRR is poor (annotators disagree substantially), then the model cannot be expected to match ground truth — the ground truth itself is unreliable. IRR must be measured and reported.

### IRR for Segmentation

**Protocol:** 10% of cases (minimum 10 cases) annotated by two independent annotators. Neither annotator sees the other's work.

**Metric:** Dice coefficient between annotator A and annotator B (inter-annotator DSC)

**Expected IRR targets (from literature):**

| Structure | Expected Inter-Annotator DSC |
|-----------|------------------------------|
| Mandible | 0.94–0.98 |
| Maxilla | 0.92–0.96 |
| Individual teeth | 0.85–0.93 |
| Inferior alveolar nerve | 0.70–0.85 (highest variability) |

**Interpretation:** Model DSC cannot be expected to exceed inter-annotator DSC. If model DSC ≥ inter-annotator DSC, the model is performing at human level.

**Arbitration protocol:** For cases where inter-annotator DSC < 0.85, a third expert annotator reviews and adjudicates. The consensus annotation is used as ground truth.

### IRR for Landmarks

**Metric:** Standard deviation of landmark positions across annotators; intraclass correlation coefficient (ICC)

**Protocol:** 10 cases annotated by 3 annotators independently. Compute ICC(2,1) for each landmark.

**Target:** ICC > 0.85 for each landmark; landmarks with ICC < 0.70 are considered unreliable and should be excluded from primary analysis or flagged for protocol revision.

**Typical inter-observer SD from literature:**
- Nasion, ANS, Menton: 0.5–1.5mm SD
- Gonion, Condylion: 1.5–3.0mm SD (highest variability in bilateral landmarks)
- Articulare: 2.0–3.5mm SD (difficult to identify consistently)

---

## 10. Outcome Tracking Protocol

### 10.1 Data Collection Timeline

| Timepoint | Data Collected | Method |
|-----------|--------------|--------|
| Pre-op (baseline) | CT scan, occlusal records, clinical photos | Standard of care |
| Pre-op (planning) | Facial Align plan, surgeon modifications, approval | Platform logged |
| Intraoperative | Operation notes, hardware used, deviations from plan | Manual case report form |
| 2 weeks post-op | Clinical occlusal check, wound assessment | Clinical visit |
| 6 weeks post-op | Occlusal records, postoperative CT (primary) | Per protocol |
| 6 months post-op | Occlusal stability, symptom assessment | Per protocol |
| 12 months post-op | Skeletal stability assessment | Optional; per protocol |

### 10.2 Case Report Form (CRF) Data Points

**Preoperative:**
- Procedure type (orthognathic, trauma, reconstruction)
- CT acquisition parameters (automatically extracted by platform)
- Angle classification (preoperative)
- Cephalometric measurements (platform-generated + surgeon-verified)
- AI plan confidence scores
- Surgeon modification delta (did surgeon change the AI plan? By how much?)

**Intraoperative:**
- Actual procedure performed (may differ from plan)
- Hardware placed (plate type/size, screw count)
- Intraoperative complications
- Use of occlusal splint
- Deviation from plan noted intraoperatively (yes/no)

**Postoperative (6 weeks):**
- Occlusal classification (Angle)
- Overjet, overbite
- Skeletal deviation from plan (from post-op CT registration)
- Complications (infection, hardware failure, reoperation)

### 10.3 Surgeon Feedback Collection

After each case, the treating surgeon completes a brief (5-question) usability feedback form:
1. Was the segmentation quality sufficient to proceed without correction? (1–5)
2. Were the AI-proposed plan candidates clinically reasonable? (1–5)
3. Did the uncertainty visualization help you identify areas needing review? (1–5)
4. How much time did planning take compared to your usual VSP workflow? (much less / less / similar / more)
5. Would you use this platform again for a similar case? (yes / no / unsure)

### 10.4 Adverse Event Capture

Any unexpected adverse outcome potentially related to AI planning must be captured:
- Reoperation due to malocclusion
- Hardware failure attributed to plan-guided plate placement
- Navigation-based error if modified DICOM was used
- Any patient complaint attributed to surgical plan quality

**Post-market surveillance framework (Phase 3):** Systematic adverse event capture with root cause analysis; quarterly review; reportable events submitted to FDA per 21 CFR 803.
