# Product Requirements Document — Facial Align

**Version:** 1.0  
**Status:** Living document — Phase 1 specifications are locked; Phase 2–3 are directional  
**Owner:** Product / Engineering  
**Last Updated:** 2025

---

## Table of Contents

1. [Vision and Mission](#1-vision-and-mission)
2. [Target Users](#2-target-users)
3. [User Stories](#3-user-stories)
4. [Feature Specifications](#4-feature-specifications)
5. [AI-Native Design Principles](#5-ai-native-design-principles)
6. [Non-Functional Requirements](#6-non-functional-requirements)
7. [Success Metrics](#7-success-metrics)
8. [Competitive Differentiation](#8-competitive-differentiation)

---

## 1. Vision and Mission

### Vision

A world in which every CMF surgeon — regardless of institutional resources — can access AI-assisted surgical planning that matches or exceeds the quality of a multi-hour expert VSP session, in minutes, from any device.

### Mission

Facial Align eliminates the bottleneck at the center of CMF surgical planning: the synchronous, engineer-mediated session that turns a CT scan into a surgical plan. The platform replaces this with an AI system that:
- Proposes anatomically valid surgical plans automatically from uploaded CT data
- Surfaces uncertainty where the AI is less confident, directing surgeon attention to regions that need it
- Maintains the surgeon as the final decision-maker on every plan (physician-in-the-loop)
- Collects structured outcome data from every case to continuously improve its own predictions

### Problem Statement

Current VSP tools (3D Systems VSP, Materialise ProPlan CMF, KLS Martin IPS CaseDesigner) share three structural problems:

1. **Throughput is bounded by human engineers.** Planning a case requires scheduling a live session with a vendor biomedical engineer. Average turnaround is 5–10 days. For acute trauma, this delay affects patient outcomes.
2. **Plans are binary, not probabilistic.** The surgeon receives a single plan with no uncertainty signal. There is no way to know whether the condyle position is highly constrained (unique correct answer) or ambiguous (several valid options).
3. **No learning.** Each case is a one-time service interaction. The data is not captured in a structured way that enables the system to improve.

### Design Philosophy

The test for whether Facial Align is AI-native: *if you remove the AI, does the product fail?* In legacy tools, AI is a segmentation button — remove it and the surgeon still has a CAD tool. In Facial Align, the AI is the planning engine. Remove it and there is no plan — only raw DICOM data.

---

## 2. Target Users

### Primary Users

#### CMF Attending Surgeon
- **Institution:** Academic medical center or large community hospital
- **Technical proficiency:** Comfortable with PACS, basic 3D visualization; not a software engineer
- **Pain points:** 5–10 day VSP turnaround delays case scheduling; cannot iterate plans overnight; binary plans with no uncertainty disclosure; high cost per case ($2,000–$4,000 for commercial VSP service)
- **Goals:** Plan complex cases independently; review AI suggestion in under 30 minutes; export to 3D printing or navigation system

#### Surgical Resident / Fellow
- **Institution:** Residency training program
- **Technical proficiency:** More comfortable with technology; wants to learn planning principles
- **Pain points:** Limited access to VSP in training contexts; dependent on attending for plan review; no structured feedback on plan quality
- **Goals:** Learn planning principles with AI-assisted guidance; submit plans for attending review; understand why the AI made specific movement recommendations

#### Surgical Planner / Biomedical Engineer (Institutional)
- **Institution:** Hospital with in-house planning capability
- **Technical proficiency:** High — experienced with Mimics, VSP tools
- **Pain points:** Manual segmentation is slow (1–3 hours per case); no automated quality metrics for plan review; difficult to compare multiple plan candidates
- **Goals:** AI-automated segmentation; quantitative plan evaluation; batch case processing

### Secondary Users

#### Clinical Researcher
- **Goals:** Access evaluation metrics (Dice, landmark error, plan accuracy); run retrospective analysis on case library; compare model versions

#### Dental/Orthodontic Collaborator
- **Goals:** Review occlusal analysis; contribute intraoral scan data; verify occlusal targets in the plan

### Non-Users (Explicit Exclusions for Phase 1)
- Patients (no patient-facing features)
- Billing / insurance personnel
- Intraoperative use (Facial Align is a preoperative planning tool; real-time intraoperative use is Phase 3)

---

## 3. User Stories

### Module 1: DICOM Ingestion and Case Setup

**US-101:** As a surgeon, I can upload a DICOM series from my hospital CT scanner by dragging a folder or ZIP archive into the browser, so that I don't need to configure DICOM transfer protocols.

**US-102:** As a surgical planner, I can submit a DICOM series via DICOM C-MOVE from my institution's PACS, so that I don't need to manually export and re-upload files.

**US-103:** As a surgeon, I receive a validation report within 60 seconds of upload that tells me whether the CT series has sufficient resolution for AI processing (slice thickness, field of view, reconstruction kernel), so that I know immediately if I need to re-acquire.

**US-104:** As an administrator, I can confirm that all PHI is automatically de-identified before any processing begins, so that I can authorize the platform for use under our IRB protocol.

### Module 2: Segmentation and Mesh Generation

**US-201:** As a surgeon, I receive a 3D segmentation of the mandible, maxilla, individual teeth, and relevant soft tissue structures within 10 minutes of upload, so that I can proceed with planning without waiting.

**US-202:** As a surgeon, I can see a color-coded confidence overlay on each segmented structure, where green indicates high model confidence and red indicates regions that need manual review, so that I know where to focus my attention.

**US-203:** As a planner, I can manually correct segmentation errors on specific bone regions using a brush tool, and the corrected segmentation is saved as training data for future model improvement.

**US-204:** As a surgeon, I can export per-structure STL files for 3D printing, so that I can prepare surgical models and patient-specific implants.

**US-205:** As a researcher, I can access the per-structure Dice score and Hausdorff distance for each segmentation, so that I can evaluate model performance against clinical benchmarks.

### Module 3: Cephalometric Analysis

**US-301:** As a surgeon, I see 24 standard CMF landmarks detected automatically in 3D, each displayed with a confidence score and a 3D ellipsoid indicating uncertainty radius, so that I can identify which landmarks need manual adjustment.

**US-302:** As a surgeon, I can click on any landmark and drag it to the correct anatomical position, and the system automatically recalculates all downstream cephalometric measurements.

**US-303:** As a planner, I receive a cephalometric analysis report comparing the patient's measurements to normative ranges (ANB angle, SNB, SNA, facial height ratios), so that I have a quantitative baseline for planning.

### Module 4: Surgical Plan Generation

**US-401:** As a surgeon, I receive 1–3 ranked surgical plan candidates within 5 minutes of landmark approval, each specifying the movement vectors for each bone segment.

**US-402:** As a surgeon, each plan candidate clearly shows: (a) the confidence score, (b) the occlusal constraint status (satisfied / violated / marginal), (c) the symmetry score, and (d) a plain-language rationale — so that I can select between candidates with clinical judgment.

**US-403:** As a surgeon planning orthognathic surgery, the system proposes LeFort I, BSSO, and/or genioplasty movements as appropriate to the patient's skeletal discrepancy, with cephalometric targets pre-populated from normative ranges.

**US-404:** As a surgeon planning trauma reconstruction, the system proposes fracture reduction positions using contralateral mirroring and pre-injury anatomical reconstruction algorithms.

**US-405:** As a surgeon, I can modify any movement vector in the proposed plan by direct manipulation in the 3D viewport, and the system immediately re-evaluates occlusal constraint satisfaction and updates the confidence score.

**US-406:** As a surgeon, when I adjust a movement that violates an occlusal constraint, the system highlights the constraint violation in red and explains which teeth or condylar relationships are affected — it does not silently accept an invalid plan.

### Module 5: Plan Review and Approval

**US-501:** As a surgeon, I can compare the pre-operative state and the planned state side-by-side in the 3D viewer, with overlay toggling, so that I can visually assess the surgical changes.

**US-502:** As a resident, I can submit a completed plan to an attending surgeon for review. The attending receives a notification and can approve, reject, or comment on the plan.

**US-503:** As a surgeon, when I click Approve Plan, the system records my name, credentials, timestamp, and a hash of the plan state — creating an immutable audit record.

**US-504:** As a surgeon, I can generate a surgical planning report (PDF) that includes: case summary, segmentation images, planned movements table, cephalometric analysis, and QR code linking to the case.

### Module 6: Export and Integration

**US-601:** As a surgeon, I can download a modified DICOM file containing the planned bone positions overlaid as a segmentation, for import into a surgical navigation system (Stryker, Medtronic).

**US-602:** As a planner, I can export a ZIP archive of all STL meshes (individual bones, planned positions) for 3D printing or custom implant design.

**US-603:** As a planner, I can submit a 3D printing order for surgical models or cutting guides directly from the platform (Phase 3).

### Module 7: Evaluation and Outcome Tracking

**US-701:** As a researcher, I can upload a post-operative CT scan and the system automatically computes the deviation between planned and achieved bone positions (in mm and degrees).

**US-702:** As a researcher, I can view per-case and aggregate accuracy metrics in a dashboard, including Dice scores, landmark errors, and plan deviation statistics.

**US-703:** As a researcher, all surgeon modifications to AI-generated plans are logged as structured data, so that I can analyze what the AI gets wrong and where surgeons consistently override it.

---

## 4. Feature Specifications

### Phase 1: Research-Grade Foundation

**Goal:** A reproducible research system that can process real CMF CT cases and produce demonstrable AI outputs. Serves as a GitHub research repo, publication substrate, and product prototype.

#### F1.1 — DICOM Ingestion Pipeline
- Accept DICOM CT/CBCT series via ZIP upload or folder
- Validate series completeness and resolution adequacy
- Extract and store DICOM metadata (scanner model, acquisition params, study date)
- Automatic de-identification per DICOM PS 3.15 Annex E profiles
- Convert to standardized NIfTI (1.0mm isotropic for CT, 0.4mm for CBCT)
- Detect and flag: MONOCHROME1 photometric interpretation, missing oblique slices, metal artifacts, thin-slice vs. standard reconstructions
- **Acceptance criteria:** Process ≥95% of valid DICOM series without manual intervention; de-identification removes all 18 HIPAA Safe Harbor identifiers from headers

#### F1.2 — Multi-Structure Segmentation
- Segment structures: skull base, mandible, maxilla, zygomatic arches, sinuses, soft tissue envelope, individual teeth (upper and lower arches, FDI notation)
- Primary model: TotalSegmentator (coarse pass for skull/mandible/sinuses) + DentalSegmentator (fine pass for dental structures)
- Per-voxel uncertainty estimation via Monte Carlo dropout or test-time augmentation
- **Acceptance criteria:** Mandible Dice ≥ 0.92 on held-out test set; individual tooth Dice ≥ 0.85; inferior alveolar nerve Dice ≥ 0.75; segmentation runtime ≤ 10 minutes on A10G GPU

#### F1.3 — Mesh Extraction
- Convert per-label segmentation mask to watertight STL mesh per structure
- Marching cubes → Laplacian smoothing → decimation to target polygon budget
- GLTF export for Three.js rendering
- Mesh quality validation: check for holes, self-intersections, degenerate faces
- **Acceptance criteria:** ≥99% of segmentation outputs produce valid, printable STL meshes

#### F1.4 — Cephalometric Landmark Detection
- Detect 24 standard CMF landmarks: nasion, sella, basion, ANS, PNS, A-point, B-point, pogonion, menton, gnathion, gonion (bilateral), condylion (bilateral), articulare, orbitale (bilateral), porion (bilateral), incisor tips (upper/lower)
- Output per-landmark confidence score and uncertainty ellipsoid
- Automatic cephalometric analysis: ANB, SNB, SNA, FMA, IMPA, facial height ratios
- **Acceptance criteria:** Mean landmark error ≤ 2.0mm on held-out test set; 90th percentile ≤ 3.5mm; landmarks exceeding 1.5mm uncertainty flagged for manual review

#### F1.5 — Fracture Fragment Identification (Scaffolded)
- Connected component analysis on bone-HU regions to isolate individual fracture fragments
- Fragment classification: symphysis, parasymphysis, body, angle, condyle, ramus
- Confidence scoring per fragment based on fragment size and edge characteristics
- Contralateral mirroring for reduction target generation
- **Status:** Fragment identification implemented; automated reduction algorithm scaffolded but not complete

#### F1.6 — Occlusal Constraint Engine (Scaffolded)
- Compute current occlusal relationship from segmented dental arches
- Classify Angle class (I, II, III) based on first molar relationship
- Compute overjet, overbite, and midline deviation
- Constraint satisfaction check: given a proposed jaw movement, does it preserve/achieve target occlusion?
- **Status:** Occlusal measurement implemented; constraint solver scaffolded

#### F1.7 — 3D Surgical Planning Viewer
- OHIF v3 + Cornerstone3D for DICOM three-plane review
- Three.js + React Three Fiber for planning 3D scene
- Bone mesh render per structure (toggle visibility, opacity)
- Landmark annotation display with uncertainty ellipsoids
- Per-voxel uncertainty heat map overlay on mesh surface
- Segmentation label overlay on OHIF viewer
- Gizmo controls for bone segment manipulation (Phase 1: mouse drag; Phase 2: proper transform widget)

#### F1.8 — Evaluation Framework
- Compute and store Dice coefficient per structure
- Compute 95th percentile Hausdorff distance and average surface distance per structure
- Landmark detection error (Euclidean distance in mm)
- Plan deviation metrics (post-op comparison) — scaffolded
- Exportable evaluation reports (JSON + CSV)

### Phase 2: Clinical Research Platform

**Goal:** A web platform surgeons can use for prospective case planning at ≥2 clinical sites, with HIPAA-compliant architecture and structured data collection.

#### F2.1 — HIPAA-Compliant Infrastructure
- Encryption at rest (AES-256) and in transit (TLS 1.3)
- Business Associate Agreement (BAA) with cloud provider
- Audit logging for all PHI access
- Session management: 30-minute idle timeout, MFA required
- Role-based access control per institution

#### F2.2 — Learned Plan Suggestion
- Train plan scoring model on accumulated Phase 1 case library
- Replace/augment rule-based plan generation with learned movement recommendations
- Confidence calibration: model uncertainty reflects true outcome variance
- Prerequisite: ≥50 cases with postoperative follow-up

#### F2.3 — Multi-Site DICOM Ingestion
- DICOM transfer: accept studies via DICOM C-STORE (Orthanc intermediary)
- PACS integration: HL7 FHIR patient context for case creation
- Automated study routing: map DICOM Study Description to case type

#### F2.4 — Postoperative Outcome Tracking
- Accept post-operative CT upload
- Rigid registration of post-op CT to pre-op planning space
- Automated measurement of plan-vs-achieved deviation per bone segment
- Aggregate accuracy statistics dashboard
- Structured export for regulatory submission data package

#### F2.5 — Confidence and Uncertainty UI (Full Specification)
- Layer 1 — Per-voxel segmentation uncertainty as color heat map on mesh surface
- Layer 2 — Per-landmark uncertainty ellipsoid with clinical tolerance threshold indicator
- Layer 3 — Per-plan confidence score with historical calibration curve
- Low-precedent detection: flag plans in movement-space regions with fewer than 5 similar historical cases
- Surgeon override logging: when surgeon overrides a high-confidence AI prediction, log reason code

### Phase 3: Cleared Commercial Product

**Goal:** FDA-cleared (or pre-submission) product with commercial billing and enterprise integrations.

#### F3.1 — FDA 510(k) Submission Package
- Design and development documentation (IEC 62304 compliant)
- Software risk management (ISO 14971)
- Clinical validation study data (prespecified primary endpoint)
- Predicate comparison documentation
- Predetermined Change Control Plan (PCCP)

#### F3.2 — Automated Surgical Deliverables
- Occlusal splint design and STL export for 3D printing
- Cutting guide design for fibula free flap reconstruction
- Patient-specific implant (PSI) geometry generation (titanium mesh / PEEK orbital floor)

#### F3.3 — Intraoperative Navigation Export
- Modified DICOM series containing planned bone positions as segmentation overlay
- Compatible with Stryker Nav3i and Brainlab Kick formats
- Validated fiducial registration workflow

#### F3.4 — Enterprise Integrations
- PACS integration (DICOM C-MOVE, C-STORE, C-FIND)
- HL7 FHIR R4 patient context
- Hospital EHR procedure note generation (structured text)
- SSO: SAML 2.0 / OIDC for institutional identity providers

---

## 5. AI-Native Design Principles

These principles are architectural constraints, not UX guidelines. Violating them degrades the product to a CAD tool with an AI button.

### Principle 1: The AI Proposes; The Surgeon Finalizes

Every AI output is a proposal. The surgeon has final authority on every decision. This is non-negotiable for:
- **Regulatory reasons:** Maintains FDA Class II (physician-in-the-loop) rather than Class III (autonomous clinical decision)
- **Safety reasons:** AI failures are caught before they affect the patient
- **Trust reasons:** Surgeons who feel in control adopt the tool; surgeons who feel overridden reject it

**Implementation:** Every plan, every segmentation, every landmark position has a one-click override. Overrides are never friction-blocked (no "are you sure?" dialogs for simple adjustments).

### Principle 2: Uncertainty is First-Class, Not a Footer

Uncertainty must be **contextual** (attached to the specific prediction it describes), **visual** (not a number in a tooltip), and **actionable** (tells the surgeon what to do with it).

**Forbidden:** "Confidence: 87%." This is meaningless.  
**Required:** Condyle position uncertainty ellipsoid rendered in 3D. Landmark uncertainty displayed as a colored ring with a tooltip: "Uncertainty radius 2.1mm — exceeds 1.5mm clinical tolerance. Click to review manually."

**Three-layer uncertainty architecture:**
1. Per-voxel segmentation uncertainty (MC Dropout ensemble) — heat map on mesh surface
2. Per-landmark positional uncertainty — 3D ellipsoid in planning scene
3. Per-plan confidence score — trained from historical plan-vs-outcome pairs; shown with calibration context ("91% confidence: similar cases had mean deviation < 1.8mm postoperatively")

### Principle 3: Constraints are Computed, Not Visualized

Legacy tools check occlusion by showing the surgeon an overlay and asking them to eyeball it. Facial Align computes constraint satisfaction algorithmically:
- When a movement violates occlusal integrity, the system blocks finalization and explains the violation in clinical terms
- When a movement produces valid but suboptimal occlusion, the system shows a quality score, not just a pass/fail

**Implementation:** The occlusal constraint engine is not a visualization module — it is a constraint solver that gates plan approval.

### Principle 4: Every Case is a Training Case

The platform must be instrumented to collect training data from normal surgical use:
- Every surgeon modification to an AI-generated plan is logged with a structured delta
- Every approved plan, paired with the post-operative outcome, becomes a labeled training example
- Every manual landmark correction becomes a positive example for the landmark detector

**Implementation:** All modifications flow through `plan_modifications` and `landmark_override` tables with structured schema — not free-text notes.

### Principle 5: Failures are Explicit

When the AI cannot produce a reliable output, it says so clearly. Failure modes must be characterized and disclosed:
- "Segmentation confidence in this region is low — metal artifact detected. Manual correction recommended."
- "Fracture fragment count (7) is outside the training distribution (2–5 fragments). Plan suggestions may be unreliable."
- "This patient's anatomy differs significantly from the training population. Treat this plan as a starting point only."

**Implementation:** Each inference output includes a quality flag (PASS / WARNING / FAIL) with a structured reason code. WARNING and FAIL states trigger UI alerts with specific guidance.

---

## 6. Non-Functional Requirements

### Performance

| Operation | P50 Target | P95 Target | Notes |
|-----------|-----------|-----------|-------|
| DICOM upload and validation | < 30s | < 60s | For a 512×512×400 CT series |
| CT preprocessing | < 2 min | < 5 min | |
| Full segmentation (GPU) | < 10 min | < 20 min | On A10G; CPU fallback 30–90 min |
| Mesh extraction | < 2 min | < 5 min | Per case |
| Landmark detection | < 2 min | < 5 min | |
| Plan generation (rule-based) | < 30s | < 2 min | After landmarks approved |
| Plan generation (learned, Phase 2) | < 1 min | < 3 min | |
| 3D viewer load time | < 3s | < 8s | Mesh streaming from MinIO CDN |
| API response time (non-ML) | < 100ms | < 500ms | CRUD operations |

### Reliability

- API uptime: ≥ 99.5% (single-node dev), ≥ 99.9% (Phase 2 clinical deployment)
- Job failure rate: < 2% of submitted jobs should fail due to infrastructure (not input data quality)
- No data loss: all uploaded DICOM data persisted before acknowledgment

### Security

- All data encrypted in transit (TLS 1.3) and at rest (AES-256)
- No PHI in application logs (PII scrubbing middleware)
- No PHI in error messages returned to clients
- Presigned URLs for object storage with 15-minute expiry
- Authentication tokens expire: access tokens 15 min, refresh tokens 7 days
- Account lockout after 5 consecutive failed login attempts
- SQL injection prevention via ORM parameterization (no raw SQL construction from user input)

### HIPAA Technical Safeguards

- Access controls: role-based, per-institution data isolation
- Audit controls: append-only audit log, all PHI access logged
- Integrity controls: checksums on all stored objects; database transaction log
- Transmission security: HTTPS only; no email transmission of unencrypted PHI
- Automatic logoff: 30-minute session timeout with warning at 25 minutes

### Clinical Accuracy Targets (Phase 1 Baseline)

These are the research benchmarks that define a credible Phase 1:

| Task | Metric | Target | Reference Benchmark |
|------|--------|--------|---------------------|
| Mandible segmentation | Dice | ≥ 0.92 | CMF-ELSeg 2025: 0.96 (nnU-Net) |
| Maxilla segmentation | Dice | ≥ 0.90 | CMF-ELSeg 2025: 0.95 |
| Individual tooth segmentation | Dice | ≥ 0.85 | CMF-ELSeg 2025: 0.94 |
| Inferior alveolar nerve | Dice | ≥ 0.70 | CMF-ELSeg 2025: 0.88 |
| Landmark detection (24 points) | Mean error | ≤ 2.0 mm | CMF-Net: 1.108mm |
| Landmark detection | P90 error | ≤ 3.5 mm | Clinical tolerance: 1.5mm |
| Fracture reduction (Phase 2) | Mean deviation | ≤ 2.0 mm | Literature: 2–3mm typical |
| Orthognathic plan accuracy | Mean deviation | ≤ 1.5 mm | Clinical standard: ≤ 2mm |

### Scalability

- Phase 1 (research): 1–5 concurrent cases; single-node deployment
- Phase 2 (clinical): 20–50 concurrent cases across 2–5 sites; Kubernetes multi-node
- Phase 3 (commercial): 200+ concurrent cases; auto-scaling on GPU nodes

---

## 7. Success Metrics

### Phase 1 Research Milestones (Month 0–12)

| Metric | Target |
|--------|--------|
| Cases processed end-to-end | ≥ 20 real CMF cases |
| Segmentation Dice (mandible) | ≥ 0.92 on held-out test set |
| Landmark error | ≤ 2.0mm mean on 24 landmarks |
| Unit test coverage | ≥ 80% on core pipeline modules |
| Publications / preprints | ≥ 1 submitted to CMF journal or MICCAI |
| IRB approvals | ≥ 1 academic partner IRB |
| Open benchmark | Evaluation protocol publicly reproducible |

### Phase 2 Clinical Research Milestones (Month 12–24)

| Metric | Target |
|--------|--------|
| Clinical sites using the platform | ≥ 2 |
| Prospectively planned cases | ≥ 50 |
| Cases with postoperative follow-up | ≥ 25 |
| Plan approval time (surgeon) | < 30 minutes median |
| Surgeon satisfaction score | ≥ 4.0 / 5.0 (SUS-style) |
| Plan deviation vs. manual | Non-inferior at p < 0.05 |

### Phase 3 Commercial Milestones (Month 24–42)

| Metric | Target |
|--------|--------|
| FDA 510(k) submitted | Yes |
| Commercial customers | ≥ 5 institutions |
| Cases per month | ≥ 100 |
| Planning time reduction vs. commercial VSP | ≥ 70% |
| Net Promoter Score (surgeon) | ≥ 50 |

---

## 8. Competitive Differentiation

### vs. 3D Systems VSP

| Dimension | 3D Systems VSP | Facial Align |
|-----------|---------------|--------------|
| Delivery model | Managed service (engineer-mediated) | Software product (self-serve) |
| Turnaround time | 5–10 days | < 30 min end-to-end |
| Uncertainty | None disclosed | Per-voxel, per-landmark, per-plan |
| Iteration speed | One session, costly to revise | Unlimited iterations in browser |
| Data flywheel | Service provider's proprietary data | Platform captures all cases as training data |
| Cost | $2,000–$4,000 per case | TBD SaaS pricing |

### vs. Materialise ProPlan CMF / Mimics

| Dimension | Materialise | Facial Align |
|-----------|------------|--------------|
| Target user | Engineers using Mimics | Surgeons directly |
| AI role | Segmentation assist (secondary) | Planning engine (primary) |
| Deployment | Desktop software | Web-native |
| Learning | Static model updates | Continuous per-case improvement |

### vs. Open-Source (3D Slicer + SlicerCMF)

| Dimension | 3D Slicer + SlicerCMF | Facial Align |
|-----------|----------------------|--------------|
| Setup burden | High (software install, plugin config) | Zero (browser) |
| AI integration | Manual model loading | Automated pipeline |
| Clinical workflow | Tool-centric | Case-centric |
| Data collection | None | Structured training data flywheel |

### Strategic Moat

The sustainable competitive advantage is the **data flywheel**: every case processed by Facial Align generates structured training data. At 100 cases/month, Facial Align generates more labeled CMF surgical planning data per year than most academic research programs produce in a decade. This data, paired with outcome tracking, allows the learned plan suggestion model to improve continuously — a capability that service-model competitors and desktop-software competitors structurally cannot replicate.
