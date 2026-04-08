# Product and Engineering Roadmap — Facial Align

**Version:** 1.0  
**Format:** Three phases with features, technical milestones, clinical milestones, and regulatory milestones  
**Last Updated:** 2025

---

## Roadmap Summary

| Phase | Timeline | Goal | Status |
|-------|---------|------|--------|
| Phase 1 | Months 0–12 | Research-grade foundation — real CT cases, demonstrable AI outputs, public benchmark | Active |
| Phase 2 | Months 12–24 | Clinical research platform — ≥2 sites, HIPAA, prospective data collection, learned models | Planned |
| Phase 3 | Months 24–42 | FDA-cleared commercial product — billing, enterprise integrations, PCCP | Planned |

---

## Phase 1: Research-Grade Foundation (Months 0–12)

### Goal

A reproducible, open research system that processes real CMF CT cases end-to-end and produces demonstrable AI outputs: segmentation, landmarks, fracture reduction visualization, and basic occlusal analysis. This is the GitHub repo, the publication substrate, and the prototype that clinical collaborators evaluate.

### Features

#### Core Pipeline (Must Ship)
- [x] DICOM ingestion and de-identification (pydicom + SimpleITK)
- [x] NIfTI preprocessing (isotropic resampling, LPS orientation, HU clipping)
- [x] TotalSegmentator integration (skull, mandible, maxilla, sinuses)
- [x] DentalSegmentator integration (mandible refinement, teeth, IAN canal)
- [x] Marching cubes mesh extraction → STL + GLTF export
- [x] Cephalometric landmark detection (24 landmarks, heatmap regression)
- [x] Per-voxel uncertainty estimation (MC Dropout)
- [x] Celery job queue (Redis broker, async pipeline chaining)
- [x] FastAPI REST API (OpenAPI, JWT auth, audit logging)
- [x] PostgreSQL schema (cases, patients, jobs, segmentations, meshes, landmarks, plans)
- [x] MinIO object storage (DICOM, NIfTI, STL, model registry)
- [x] OHIF v3 + Cornerstone3D DICOM viewer (in-browser, zero-footprint)
- [x] Three.js 3D planning scene (bone mesh render, landmark display)
- [x] Docker Compose full stack (one-command dev environment)
- [x] MLflow experiment tracking

#### Scaffolded (Architecture Defined, Implementation Incomplete)
- [ ] Fracture fragment identification (connected components + fragment classification)
- [ ] Virtual fracture reduction (contralateral mirroring + ICP-based repositioning)
- [ ] Occlusal constraint engine (Angle classification, overjet, overbite)
- [ ] Rule-based surgical plan suggestion (LeFort I, BSSO, genioplasty movement templates)
- [ ] CT-to-intraoral scan registration (ICP surface matching)
- [ ] PDF surgical planning report generation
- [ ] Postoperative comparison framework (plan vs. post-op CT deviation)
- [ ] Plan modification audit trail (structured delta logging)

#### Evaluation Framework (Must Ship)
- [ ] Dice coefficient per structure (automated, compared to ground truth)
- [ ] Hausdorff distance and average surface distance per structure
- [ ] Landmark error computation (Euclidean distance per landmark)
- [ ] Evaluation report export (JSON + CSV)
- [ ] Calibration error (ECE) for uncertainty estimates

### Technical Milestones

| Milestone | Description | Month |
|-----------|------------|-------|
| T1.1 | Full ingestion→segmentation→mesh pipeline runs end-to-end on a real CT case | 1 |
| T1.2 | Landmark detection integrated with confidence scores | 2 |
| T1.3 | Three.js planning scene loads segmentation meshes with uncertainty heat map | 3 |
| T1.4 | Fracture fragment identification running on trauma cases | 4 |
| T1.5 | Occlusal constraint engine computes Angle class and overjet/overbite | 5 |
| T1.6 | Rule-based plan suggestion generates ≥1 plan candidate per case | 6 |
| T1.7 | Evaluation framework computes all standard metrics automatically | 7 |
| T1.8 | PDF report export functional | 8 |
| T1.9 | CT-to-intraoral scan registration working on paired test cases | 9 |
| T1.10 | Docker Compose stack passes integration test suite | 10 |
| T1.11 | CMF segmentation fine-tuning (nnU-Net on institutional data) | 11 |
| T1.12 | Phase 1 evaluation completed on ≥ 20 real CMF cases | 12 |

### Clinical Milestones

| Milestone | Description | Month |
|-----------|------------|-------|
| C1.1 | IRB approval at primary academic partner institution | 3 |
| C1.2 | Data use agreement executed for retrospective CT data access | 4 |
| C1.3 | ≥ 5 real CMF cases processed and evaluated by clinical collaborator | 6 |
| C1.4 | Landmark annotation protocol established and annotator training completed | 6 |
| C1.5 | ≥ 20 real CMF cases processed end-to-end | 10 |
| C1.6 | Phase 1 segmentation evaluation report completed (Dice, Hausdorff per structure) | 12 |
| C1.7 | Preprint or conference submission of segmentation benchmark results | 12 |

### Regulatory Milestones

| Milestone | Description | Month |
|-----------|------------|-------|
| R1.1 | De-identification pipeline implemented and verified (all 18 Safe Harbor identifiers) | 2 |
| R1.2 | Regulatory strategy decision: FDA 510(k) pathway confirmed | 6 |
| R1.3 | Regulatory counsel identified | 8 |
| R1.4 | SaMD classification documentation started | 9 |
| R1.5 | IEC 62304 documentation structure created (design history file shell) | 12 |

---

## Phase 2: Clinical Research Platform (Months 12–24)

### Goal

A web platform that CMF surgeons at ≥ 2 academic sites actively use for prospective case planning, with HIPAA-compliant infrastructure, structured outcome data collection, and first-generation learned models trained on clinical case library.

### Features

#### Infrastructure
- [ ] HIPAA-compliant cloud deployment (AWS/GCP with signed BAA)
- [ ] AES-256 encryption at rest for all ePHI buckets
- [ ] TLS 1.3 enforcement and certificate management
- [ ] Multi-factor authentication (TOTP) — required for clinical users
- [ ] Role-based access control with institution-level data isolation
- [ ] Audit log forwarding to SIEM (Splunk or Datadog)
- [ ] SOC 2 Type II audit preparation

#### DICOM Integration
- [ ] Orthanc DICOM server integration (receive studies via C-STORE)
- [ ] DICOM C-MOVE from institutional PACS (study routing)
- [ ] Automated study routing (procedure code → case type classification)
- [ ] Multi-site DICOM transfer with de-identification at source

#### Clinical UI Enhancements
- [ ] Full surgeon planning workflow (plan review, modification, approval)
- [ ] Uncertainty visualization Layer 2 (per-landmark 3D ellipsoids with clinical tolerance indicators)
- [ ] Uncertainty visualization Layer 3 (per-plan calibrated confidence with historical context)
- [ ] Low-precedent case detection ("this plan is outside the training distribution")
- [ ] Resident → Attending review and approval workflow
- [ ] Collaborative case annotation (multiple users on one case)
- [ ] Case notes and annotation (free text + structured tags)
- [ ] Patient-facing output: simplified surgical plan summary (PDF)

#### Machine Learning
- [ ] CMF segmentation model fine-tuned on Phase 2 clinical dataset (nnU-Net from scratch, ≥ 100 cases)
- [ ] Landmark detection model retrained on corrected landmark data
- [ ] MONAI Label active learning workflow for efficient annotation
- [ ] Plan scoring model (MLP, requires ≥ 50 cases with postoperative follow-up)
- [ ] Model A/B testing framework (route traffic split between model versions)
- [ ] Model performance monitoring dashboard (drift detection)
- [ ] Automated retraining pipeline (weekly trigger when new annotations available)

#### Data and Analytics
- [ ] Postoperative outcome tracking module (post-op CT upload → plan deviation)
- [ ] Aggregate analytics dashboard (Dice trends, plan accuracy over time, surgeon modification patterns)
- [ ] Training data quality dashboard (per-annotator IRR, label distribution)
- [ ] Export pipeline: de-identified datasets for research collaboration

### Technical Milestones

| Milestone | Description | Month |
|-----------|------------|-------|
| T2.1 | HIPAA-compliant cloud stack deployed with BAA | 13 |
| T2.2 | DICOM C-STORE reception via Orthanc integrated | 14 |
| T2.3 | Multi-factor authentication deployed | 15 |
| T2.4 | Confidence UI Layer 2 and 3 shipped | 16 |
| T2.5 | Plan scoring model v0.1 trained and deployed | 18 |
| T2.6 | Active learning pipeline (MONAI Label) integrated | 18 |
| T2.7 | Model A/B testing framework operational | 19 |
| T2.8 | Postoperative outcome tracking functional | 20 |
| T2.9 | nnU-Net model retrained on ≥ 100 clinical cases | 22 |
| T2.10 | Automated retraining pipeline (weekly) running | 23 |
| T2.11 | Phase 2 evaluation completed (≥ 50 cases with postoperative follow-up) | 24 |

### Clinical Milestones

| Milestone | Description | Month |
|-----------|------------|-------|
| C2.1 | Site 1 (primary): first prospective case planned using Facial Align | 13 |
| C2.2 | BAA executed with Site 1 | 13 |
| C2.3 | Site 2 onboarded: IRB, BAA, training completed | 16 |
| C2.4 | 25 prospectively planned cases across both sites | 18 |
| C2.5 | 25 cases with 6-week postoperative follow-up | 21 |
| C2.6 | Surgeon satisfaction survey completed (SUS-style, N ≥ 10) | 21 |
| C2.7 | 50 prospectively planned cases with postoperative follow-up | 24 |
| C2.8 | Phase 2 clinical outcomes paper submitted | 24 |

### Regulatory Milestones

| Milestone | Description | Month |
|-----------|------------|-------|
| R2.1 | BAA executed with cloud provider (before first ePHI in cloud) | 13 |
| R2.2 | FDA Pre-Submission Q-Sub package submitted | 15 |
| R2.3 | FDA Pre-Submission Q-Sub meeting held | 18 |
| R2.4 | Predicate device strategy confirmed (post Q-Sub feedback) | 19 |
| R2.5 | IEC 62304 documentation complete (software lifecycle) | 20 |
| R2.6 | ISO 14971 risk management file initiated | 20 |
| R2.7 | 510(k) submission outline drafted | 22 |
| R2.8 | Predetermined Change Control Plan (PCCP) drafted | 23 |
| R2.9 | Clinical validation study protocol finalized | 24 |

---

## Phase 3: Cleared Commercial Product (Months 24–42)

### Goal

An FDA-cleared (or submission-in-progress) commercial product with billing, enterprise integrations, and advanced clinical features. This is the version that generates revenue and can be marketed as a cleared device.

### Features

#### Regulatory and Compliance
- [ ] FDA 510(k) submission package complete and submitted
- [ ] Post-market surveillance infrastructure (model drift monitoring, adverse event capture)
- [ ] Mandatory change reporting protocol (software updates follow PCCP process)
- [ ] CE marking preparation (EU MDR Notified Body engagement)
- [ ] SOC 2 Type II certification achieved

#### Advanced Clinical Features
- [ ] Automated occlusal splint design (STL export for 3D printing)
- [ ] Fibula free flap cutting guide design (reconstruction planning)
- [ ] Contralateral mirroring for orbital reconstruction (PSI geometry generation)
- [ ] Soft tissue simulation (aesthetic outcome preview — CephalFACS or FEM-based)
- [ ] Intraoperative navigation export (modified DICOM compatible with Stryker Nav3i, Brainlab Kick)
- [ ] Automated intermediate splint design for bimaxillary surgery

#### Enterprise Integration
- [ ] PACS integration (DICOM C-FIND/MOVE from major PACS vendors: Sectra, Agfa, Intelerad)
- [ ] HL7 FHIR R4 patient context API (connect to EHR for patient demographics)
- [ ] Hospital EHR procedure note generation (structured text for Epic, Cerner)
- [ ] SSO: SAML 2.0 / OIDC for institutional identity providers
- [ ] Multi-institution admin portal (user management across sites)
- [ ] 3D printing partner integration (order surgical models directly from platform)

#### Commercial Features
- [ ] Per-case billing infrastructure
- [ ] Subscription tier management
- [ ] Usage analytics and cost reporting
- [ ] Customer success dashboard
- [ ] In-platform training and certification program

#### Advanced ML
- [ ] Learned fracture reduction model (Phase 3, requires 200+ trauma cases)
- [ ] Soft tissue prediction from skeletal movements (PointNet/deep learning approach)
- [ ] Federated learning infrastructure (train across institutions without data centralization)
- [ ] AR/VR export (USDZ for Apple Vision Pro; WebXR for browser-based AR)

### Technical Milestones

| Milestone | Description | Month |
|-----------|------------|-------|
| T3.1 | PACS integration (Sectra, Agfa) tested and validated | 27 |
| T3.2 | FHIR R4 patient context API live | 28 |
| T3.3 | Occlusal splint design STL export functional | 30 |
| T3.4 | Intraoperative navigation DICOM export validated | 33 |
| T3.5 | Soft tissue simulation v0.1 integrated | 36 |
| T3.6 | Federated learning PoC complete | 38 |

### Clinical Milestones

| Milestone | Description | Month |
|-----------|------------|-------|
| C3.1 | 100 total cases with postoperative follow-up | 30 |
| C3.2 | Pivotal clinical study enrollment complete | 36 |
| C3.3 | Site 3 onboarded (geographic or case-type diversity) | 28 |
| C3.4 | Phase 3 clinical outcomes paper submitted to peer-reviewed CMF journal | 40 |

### Regulatory Milestones

| Milestone | Description | Month |
|-----------|------------|-------|
| R3.1 | 510(k) submission received by FDA | 26 |
| R3.2 | FDA 510(k) clearance received | 38 (estimated) |
| R3.3 | Post-market surveillance plan activated | 38 |
| R3.4 | CE marking Notified Body application submitted | 36 |

---

## Engineering Backlog

Specific engineering tasks not yet assigned to a phase. Prioritized as (H)igh / (M)edium / (L)ow.

### Backend / API

| Task | Priority | Notes |
|------|---------|-------|
| Rate limiting per-endpoint (production-grade) | H | Redis-based; current impl is basic |
| Presigned URL caching (reduce MinIO calls) | M | Cache presigned URLs in Redis with TTL |
| API key authentication (for programmatic access) | M | For PACS integration scripts |
| Webhook notifications (plan approved, job failed) | M | Reduce polling overhead |
| GraphQL API for complex queries | L | Optional; REST is sufficient for Phase 1–2 |
| gRPC interface for inference service | M | Replace HTTP for internal workers; lower latency |

### Pipeline / ML

| Task | Priority | Notes |
|------|---------|-------|
| Metal artifact reduction preprocessing | H | Affects ~20% of clinical cases |
| CBCT-to-CT registration for mixed modality cases | H | Required for dental cases with CBCT + standard CT |
| Automatic DICOM series selection (pick best series from multi-series study) | H | Currently manual in complex studies |
| Fragment classification model training | M | Requires Phase 2 trauma case library |
| Condyle seating confidence model | H | Most clinically critical uncertainty signal |
| Mandibular canal safety margin computation | M | Distance from planned screw to IAN canal |
| Symmetric landmark constraint (enforce anatomical symmetry) | M | Post-processing on bilateral landmarks |

### Frontend / UX

| Task | Priority | Notes |
|------|---------|-------|
| Plan modification gizmo controls (proper transform widget) | H | Current implementation is basic drag |
| Pre/post overlay toggle with opacity slider | H | Essential for plan review |
| Cephalometric measurement display panel | H | Show ANB, SNB, etc. in planning view |
| 3D screenshot / snapshot export | M | For presentations and reports |
| Mobile-responsive layout | L | Defer until Phase 3 |
| Dark mode | L | |
| Keyboard shortcuts for planning view | M | Power users |

### Infrastructure / DevOps

| Task | Priority | Notes |
|------|---------|-------|
| Prometheus + Grafana dashboards (service health, GPU utilization) | H | `infra/monitoring/` stub exists |
| Kubernetes production manifests (complete, tested) | H | `infra/kubernetes/` stub exists |
| Automated database backups with restore testing | H | Critical before any clinical data |
| Secrets management (Vault or AWS Secrets Manager) | H | Currently .env file only |
| Log aggregation (ELK stack or similar) | M | Centralized log search |
| Load testing (k6 or Locust) | M | Before Phase 2 multi-site launch |
| Dependabot / automated dependency updates | M | |

### Tests

| Task | Priority | Notes |
|------|---------|-------|
| Integration test: full pipeline on synthetic DICOM | H | Currently only unit tests |
| Performance benchmark tests (regression on latency) | H | |
| Security test: OWASP ZAP scan | H | Before clinical deployment |
| Segmentation regression test suite (10 canonical cases) | H | Run on every model change |
| De-identification compliance test (verify all 18 tags) | H | |
| Frontend E2E tests (Playwright) | M | |

### Documentation

| Task | Priority | Notes |
|------|---------|-------|
| API documentation examples (curl + Python) | H | |
| Annotator training guide (segmentation protocol) | H | |
| Model card template and first model card | H | |
| Contributing guide (CONTRIBUTING.md) | M | |
| Deployment runbook (production incident response) | M | |
| Data dictionary (all database tables and columns) | M | |
