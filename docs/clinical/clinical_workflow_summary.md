# Clinical Workflow Summary — Facial Align

**Audience:** Clinical collaborators, residents, and engineers joining the project  
**Purpose:** Explain the CMF surgical planning workflow that Facial Align is designed to transform  
**Sources:** CMF clinical workflow research (see `research/clinical_workflow_research.md`)

---

## Table of Contents

1. [Current CMF Surgical Planning Workflow](#1-current-cmf-surgical-planning-workflow)
2. [Pain Points and Error Rates](#2-pain-points-and-error-rates)
3. [How Facial Align Transforms Each Step](#3-how-facial-align-transforms-each-step)
4. [Supported Procedure Types](#4-supported-procedure-types)
5. [Data Requirements](#5-data-requirements)
6. [Clinical Accuracy Targets](#6-clinical-accuracy-targets)

---

## 1. Current CMF Surgical Planning Workflow

### The Standard VSP Process (Commercial Workflow)

Virtual surgical planning (VSP) converts CT imaging data into a 3D simulation of the surgical procedure, then produces physical deliverables (occlusal splints, anatomical models, cutting guides, patient-specific implants) that transfer the virtual plan to the operating room. The process in commercial practice has six stages:

---

### Stage 1: CT Image Acquisition

**What happens:**
- Surgeon orders a fine-cut maxillofacial CT scan
- Required parameters: ≤ 0.625 mm slice thickness; helical acquisition; 1024×1024 or 512×512 matrix; ideally bone kernel reconstruction
- Standard trauma CT (3 mm cuts) is typically insufficient for VSP — a separate fine-cut study is required
- For orthognathic cases, some centers add CBCT for superior dental resolution (0.3–0.4 mm voxels)
- Intraoral scan may be added for dental occlusion accuracy (3Shape Trios, iTero, Medit)

**Critical considerations:**
- The CT must be acquired with the patient in a natural head position and mandible in centric occlusion (or centric relation if occlusion is disrupted)
- Condyle position at the time of CT is propagated into the virtual plan — incorrect condylar seating during scanning directly causes postoperative malocclusion
- Dental implants, orthodontic brackets, and existing hardware create metal artifact that degrades adjacent bone segmentation

---

### Stage 2: Vendor Consultation Scheduling

**What happens:**
- Surgeon contacts the VSP service provider (3D Systems, Materialise, KLS Martin, Stryker)
- Schedules a live planning session, typically 2–5 days after CT acquisition
- DICOM files are uploaded to the vendor's HIPAA-compliant portal or transferred via secure FTP
- A vendor biomedical engineer is assigned to the case

**Bottleneck:**
- The scheduling requirement creates an immovable delay. For acute trauma, this is the primary barrier to same-day VSP.
- Vendor availability varies; complex panfacial cases may require 2+ sessions.

---

### Stage 3: Live Planning Session

**What happens:**
- Synchronous online session (60–120 minutes) between surgeon and vendor engineer
- The engineer drives the software; the surgeon watches and directs
- Session tasks:
  1. **Segmentation:** Engineer manually or semi-automatically segments bone fragments into distinct 3D objects
  2. **Virtual reduction:** Each fragment repositioned computationally to reconstruct pre-injury or target anatomy
  3. **Occlusal establishment:** Surgeon guides how maxillary and mandibular segments should relate to each other
  4. **Condyle positioning:** Condylar heads seated in glenoid fossae — a critical, operator-dependent judgment
  5. **Bony defect assessment:** Identification of areas requiring grafting or spanning
  6. **Deliverable selection:** Surgeon specifies required outputs (splints, models, guides, PSI)

**For orthognathic surgery specifically:**
- Additional steps: cephalometric analysis, movement planning (LeFort I advancement, BSSO rotation, genioplasty), intermediate and final splint design
- Session may include a dentist/orthodontist for occlusal confirmation
- Typical movement planning: Le Fort I (vertical, transverse, anteroposterior, pitch/roll/yaw), BSSO (autorotation, advancement, setback), genioplasty (anteroposterior, vertical)

---

### Stage 4: Deliverable Fabrication

Depending on case requirements, the vendor produces one or more physical deliverables:

| Deliverable | Use | Fabrication Time |
|------------|-----|-----------------|
| Occlusal splint (acrylic or resin) | Guides MMF; confirms reduction intraoperatively | 2–5 days |
| 3D-printed anatomical model | Pre-bending stock plates; intraoperative reference | 3–7 days |
| Pre-contoured patient-specific plate | Eliminates intraoperative plate bending | 5–10 days |
| Patient-specific implant (PSI) | Custom titanium or PEEK for orbital/zygoma reconstruction | 7–14 days |
| Cutting/drilling guides | Transfer osteotomy positions to operative field | 3–7 days |
| Modified DICOM | Navigation system overlay | Same day as plan |

**Total consult-to-surgery time:** Average 8.6 days (range 5–13 days, from Singerman et al., 2025). Historically up to 14 days.

---

### Stage 5: Intraoperative Execution

**Occlusal splint-guided workflow:**
1. Splint seated on one arch; circumdental wires or arch bars placed
2. Patient placed in maxillomandibular fixation (MMF) using splint to establish planned occlusion
3. Fractures reduced under splint guidance — the splint constrains the occlusal relationship
4. Plates applied with patient in MMF; splint retained for postoperative stabilization or removed at case end

**Model-guided plate pre-bending:**
1. Anatomical model sterilized and brought to operative field
2. Surgeon contours stock reconstruction plate to model surface before implantation
3. Pre-bent plate inserted — eliminates freehand bending on native bone

**Navigation-guided (emerging):**
- Plan exported as modified DICOM; imported into navigation system (Stryker Nav3i, Brainlab)
- Tracker probe attached to fracture segments; real-time repositioning overlay on planned position
- Most useful for orbital and ZMC fractures where there is no occlusal reference point

---

### Stage 6: Postoperative Verification

- Occlusal check: manual assessment of contact pattern; reproducibility test
- Postoperative CT (selected cases): overlaid on pre-op plan to identify residual deviations
- Intraoperative CT (increasingly common): acquired immediately after fixation, before wound closure, to allow correction while still in the OR
- Long-term follow-up: occlusal photographs, clinical bite assessment, 6-week imaging as needed

---

## 2. Pain Points and Error Rates

### Timing and Access

| Pain Point | Clinical Impact |
|-----------|----------------|
| 5–10 day turnaround for commercial VSP | Delays surgery scheduling; forces suboptimal timing for acute trauma |
| Synchronous session required with vendor engineer | Cannot iterate overnight; surgeon dependent on vendor availability |
| No after-hours or emergency VSP | Complex trauma often requires accepting lower-quality planning or delayed surgery |
| Cost $2,000–$4,000 per case | Limits adoption to well-resourced institutions; unavailable for most global settings |

### Accuracy and Quality

| Error Source | Typical Rate / Magnitude |
|-------------|--------------------------|
| Condyle malpositioning (most critical error) | Not well quantified; estimated 5–15% of cases have clinically significant condyle error |
| Postoperative malocclusion requiring reoperation | ~3–5% in complex cases |
| Fracture reduction deviation (commercial VSP) | Mean 2–3mm for secondary reconstruction; less studied for acute trauma |
| Landmark identification error (manual cephalometrics) | Intra-rater SD 1.5–2.5mm; inter-rater SD up to 3.5mm |
| Segmentation errors requiring manual correction | Engineer manually segments in 40–90 min; error rate not systematically reported |

### Workflow Deficiencies

| Deficiency | Description |
|-----------|-------------|
| Binary plan, no uncertainty | Surgeon receives one plan; no information about which anatomical regions are more or less reliable |
| No iteration without rescheduling | Changing one element (condyle position, mandible rotation) requires a new session |
| No structured outcome capture | Outcomes are not systematically compared to plans; no learning loop |
| Tool mismatch: CAD software for surgeons | Existing platforms were designed by engineers for engineers; steep learning curve for direct surgeon use |
| Dental data not integrated | CT and intraoral scan remain separate; registration is often manual and imprecise |

---

## 3. How Facial Align Transforms Each Step

| Step | Current State | Facial Align |
|------|--------------|--------------|
| **Image acquisition** | Surgeon orders CT; DICOM transferred via FTP | DICOM upload (ZIP or DICOM transfer) within 30 seconds; instant validation report |
| **Segmentation** | Engineer manual/semi-auto segmentation, 40–90 min | AI segmentation in < 10 min on GPU; per-structure confidence shown |
| **Condyle positioning** | Engineer judgment in live session | AI proposes condyle position; uncertainty quantified; surgeon validates |
| **Cephalometric analysis** | Manual landmark identification by engineer or orthodontist | 24 landmarks detected automatically; confidence per landmark; surgeon reviews outliers only |
| **Plan generation** | Engineer + surgeon construct plan iteratively in session | AI proposes 1–3 ranked plan candidates with rationale; surgeon modifies in browser |
| **Constraint checking** | Occlusal check by visual overlay at end of session | Occlusal constraint engine evaluates every movement in real time; violations flagged immediately |
| **Iteration** | Requires rescheduling a new session | Unlimited iterations in browser; each plan variant logged |
| **Deliverable production** | Vendor produces splints/models/guides; 2–14 days | STL export immediate; 3D printing can be arranged locally or via partner |
| **Navigation export** | Modified DICOM produced by vendor | Modified DICOM generated automatically at plan approval |
| **Outcome capture** | Ad hoc; not systematically compared to plan | Post-op CT upload triggers automated plan-vs-outcome comparison |

---

## 4. Supported Procedure Types

### 4.1 Mandibular Trauma (Fracture Reduction)

**Indications for VSP:**
- Comminuted panfacial injuries (symphyseal + midface)
- Palatal split fractures requiring custom splints
- Severely comminuted mandible (3+ fragments)
- Ballistic/GSW injuries with pulverization
- Atrophic edentulous mandible (Luhr Class III, height < 11mm)
- Patients with poor dentition (traditional MMF unreliable)

**Not typically indicated:** Non-comminuted single-subsite fractures manageable with direct anatomical reduction

**Key workflow considerations:**
- Condyle position is the most critical judgment point — incorrect seating at time of CT leads to postoperative malocclusion
- In comminuted fractures, stock plates pre-bent on a model are more reliable than freehand bending
- Symphyseal fractures require occlusal splint to prevent condylar splay during plating

**AOCMF Classification System** (used for fracture coding in Facial Align):
- Level 1: Presence of fracture
- Level 2: Topographic location (S=symphysis, B=body, A=angle/ramus, P=condylar process, C=coronoid)
- Level 3: Morphology/severity (fragmentation grade 0–2, bone loss, dentition status)

### 4.2 Orthognathic Surgery

**Procedure types supported:**
- **LeFort I osteotomy** — maxillary repositioning (advancement, impaction, transverse expansion, pitch/roll correction)
- **BSSO (bilateral sagittal split osteotomy)** — mandibular advancement or setback, rotation
- **Genioplasty** — chin repositioning (anteroposterior, vertical, transverse)
- **Bimaxillary surgery** — combined LeFort I + BSSO + genioplasty in single stage

**Cephalometric targets** (normative ranges encoded in constraint engine):
- ANB angle: 1–3° (Class II > 3°, Class III < 1°)
- SNB: 78–82°
- SNA: 81–84°
- FMA (Frankfort-mandibular plane angle): 22–28°
- IMPA (lower incisor to mandibular plane): 87–95°
- Facial height ratio (lower/total): 55–60%

**Intermediate splint:** A planning-critical deliverable for two-jaw surgery; positions the maxilla relative to the mandible when they are simultaneously freed from their skeletal attachments.

### 4.3 Mandibular Reconstruction (Fibula Free Flap)

**Clinical context:** Ablative surgery for jaw cancer or osteonecrosis requires removal of a mandible segment; reconstruction uses a vascularized fibula osteocutaneous flap, with osteotomies to shape the fibula to match the mandibular defect.

**VSP workflow:**
1. Plan mandible resection margins (oncological or necrotic tissue boundaries)
2. Mirror contralateral anatomy to define reconstruction target
3. Design fibula osteotomy configuration (1–3 cuts) to match mandibular curvature
4. Design cutting guides for both the mandible resection and fibula osteotomies
5. Design pre-bent reconstruction plate conformed to planned neo-mandible shape

**Critical metrics:**
- Angle, condylion, and menton deviation (AMA, CMA, SMA protocol from mandibular reconstruction literature)
- Global Positioning Layout (GPL) method — fully automated accuracy measurement; validated 2026

**Facial Align scope (Phase 1):** Fibula reconstruction planning is scaffolded but not fully implemented. Resection margin definition and fibula osteotomy design are Phase 2 features.

### 4.4 Orbital and Midface Reconstruction

**Clinical context:** Orbital floor/wall fractures and midface defects (zygomaticomaxillary complex, Le Fort II/III) require patient-specific implants designed to restore orbital volume and facial projection.

**VSP workflow:**
1. Mirror contralateral orbit to define target orbital volume
2. Design PSI geometry (titanium mesh or PEEK) to restore volume and rim contour
3. Navigation-guided implant positioning in OR

**Facial Align scope (Phase 1):** Orbital PSI design is planned for Phase 3. Contralateral mirroring and orbital volume measurement are Phase 2 features.

---

## 5. Data Requirements

### CT Scan Requirements (Minimum for AI Processing)

| Parameter | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| Slice thickness | ≤ 1.5 mm | ≤ 0.625 mm | Finer slices → better segmentation |
| Reconstruction kernel | Standard or bone | Bone | Soft tissue kernel degrades bone boundary |
| Field of view | Covers full skull | Skull base to symphysis | Must include condyles |
| Matrix | 512 × 512 | 512 × 512 | 1024 allowed but rare |
| Scanner | Any 64-slice+ CT | 128-slice+ helical | |
| Contrast | Non-contrast or post-contrast acceptable | Non-contrast preferred | Contrast alters HU slightly |
| Patient position | Supine, standard | Supine, neutral head position | Mandible in centric occlusion |

**Rejection criteria (system will flag and not proceed):**
- Slice thickness > 2.5 mm
- Missing slices (gap in z-direction)
- Gross motion artifact (patient movement during scan)
- Metal artifact obscuring > 30% of mandible
- Field of view that excludes the condyles

### CBCT Requirements (Dental Planning)

| Parameter | Minimum | Notes |
|-----------|---------|-------|
| Voxel size | ≤ 0.4 mm | 0.2–0.3 mm optimal for root canal resolution |
| Field of view | Full dental arch (15 × 15 cm+) | Must include all target teeth |
| Scanner type | Any CBCT with DICOM export | Standard in dental offices |

### Intraoral Scan Requirements (Optional)

Intraoral scan (digital impression) can be registered to the CT to provide superior dental occlusal accuracy:

| Parameter | Requirement |
|-----------|------------|
| Format | STL or PLY (OBJ accepted) |
| Coverage | Both arches, full occlusal surface |
| Scanner | Any digital impression system (3Shape Trios, iTero, Medit) |
| Accuracy | Manufacturer spec ≤ 20 μm trueness |

**Registration workflow:** Surface ICP registration of intraoral scan to CT dental segmentation. Accuracy depends on CT dental resolution — CBCT provides better registration target than standard CT.

---

## 6. Clinical Accuracy Targets

These targets define what "accurate enough for clinical use" means for each pipeline component. They are derived from published benchmarks and established clinical tolerances.

### Segmentation Accuracy

| Structure | Dice Coefficient | Clinical Significance |
|-----------|-----------------|----------------------|
| Mandible | ≥ 0.92 | Mandible mesh is the primary planning substrate; errors propagate to plan |
| Maxilla | ≥ 0.90 | Critical for Le Fort I planning and occlusal constraint |
| Individual teeth | ≥ 0.85 | Required for occlusal analysis and splint design |
| Inferior alveolar nerve | ≥ 0.70 | Safety landmark for plate/screw placement |
| Soft tissue envelope | ≥ 0.80 | Phase 2: soft tissue simulation requires accurate skin surface |
| Zygomatic arch | ≥ 0.88 | Key reference for ZMC repositioning |

**Reference:** CMF-ELSeg 2025 (ensemble nnU-Net, N=400 CT scans) achieved: mandible 0.96, maxilla 0.95, teeth 0.94, inferior alveolar nerve 0.88.

### Landmark Detection Accuracy

| Landmark Group | Mean Error Target | P90 Error Target | Clinical Tolerance |
|----------------|------------------|-----------------|-------------------|
| Midline landmarks (nasion, ANS, PNS) | ≤ 1.5 mm | ≤ 2.5 mm | 1.5 mm |
| Bilateral landmarks (gonion, condylion, orbitale) | ≤ 2.0 mm | ≤ 3.5 mm | 2.0 mm |
| Dental landmarks (A-point, B-point) | ≤ 1.5 mm | ≤ 2.5 mm | 1.5 mm |
| Overall (24 landmarks) | ≤ 2.0 mm | ≤ 3.5 mm | — |

**Reference:** CMF-Net (3D heatmap regression) achieves 1.108mm mean error across 26 landmarks.

### Plan Accuracy (Phase 2 Target)

| Measurement | Target |
|-------------|--------|
| Le Fort I movement precision (post-op vs. planned) | ≤ 1.5 mm mean deviation |
| BSSO movement precision | ≤ 2.0 mm mean deviation |
| Condyle position deviation | ≤ 1.5 mm |
| Mandibular midline deviation | ≤ 1.0 mm |

**Clinical context:** The accepted clinical standard for VSP accuracy is ≤ 2mm deviation for any bone segment movement. Published studies on commercial VSP (3D Systems, Materialise) report mean deviations of 1.5–2.5 mm for orthognathic procedures and 2–3 mm for trauma reconstruction.

### Fracture Reduction Accuracy (Phase 1 Scaffolded, Phase 2 Target)

| Metric | Target |
|--------|--------|
| Fragment repositioning mean error | ≤ 2.5 mm |
| Condyle seating accuracy | ≤ 1.5 mm from glenoid fossa center |
| Symmetry score (contralateral comparison) | ≤ 2.0 mm RMS deviation |

### What "Good Enough" Means in Clinical Context

- **Segmentation:** A surgeon can work with a 0.92 Dice mandible segmentation because they visually inspect it before proceeding. Errors in small regions (notch, coronoid tip) are correctable manually.
- **Landmarks:** A landmark with > 2mm uncertainty should be flagged for surgeon review. Below 1.5mm, the surgeon can accept AI detection without manual verification for most landmarks.
- **Plans:** Post-op deviation ≤ 2mm is clinically equivalent to expert commercial VSP. Below 1.5mm is superior to published benchmarks for most procedures.
- **Condyle position:** This is the most sensitive accuracy point. Condylar malpositioning of ≥ 2mm can cause postoperative pain and malocclusion. The system must explicitly flag condyle position confidence and recommend manual verification for all cases.
