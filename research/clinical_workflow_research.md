# CMF Virtual Surgical Planning: Clinical Workflow Research
## Decision-Oriented Technical Reference for Platform Builders

**Compiled:** June 2025  
**Scope:** Cranio-maxillofacial (CMF) virtual surgical planning workflows — trauma reconstruction, orthognathic surgery, occlusal splints, data pipelines, pain points, and AI augmentation opportunities  
**Depth:** Designed to inform product architecture, feature prioritization, and workflow automation decisions

---

## Table of Contents

1. [Trauma Reconstruction Workflows](#1-trauma-reconstruction-workflows)
2. [Mandibular Fracture Classification and Reduction Planning](#2-mandibular-fracture-classification-and-reduction-planning)
3. [Orthognathic Surgical Planning](#3-orthognathic-surgical-planning)
4. [Occlusal Splint Workflows](#4-occlusal-splint-workflows)
5. [Data Formats and Pipeline Architecture](#5-data-formats-and-pipeline-architecture)
6. [Manual Pain Points, Error Rates, and Commercial Software Limitations](#6-manual-pain-points-error-rates-and-commercial-software-limitations)
7. [AI/ML Augmentation Opportunities](#7-aiml-augmentation-opportunities)
8. [Commercial Landscape Summary](#8-commercial-landscape-summary)
9. [Key Takeaways for Product Builders](#9-key-takeaways-for-product-builders)

---

## 1. Trauma Reconstruction Workflows

### 1.1 Overview and Indications

Virtual surgical planning (VSP) for acute trauma is significantly less mature than for elective orthognathic surgery, but adoption is accelerating for specific complex injury patterns. The critical distinction is **indication selection**:

**High-value VSP indications for trauma (per current evidence):**
- Comminuted panfacial injuries involving both midface and mandible (especially symphyseal mandible fractures with condylar splay)
- Palatal split fractures requiring custom splints for arch stabilization
- Ballistic/GSW injuries with comminution/pulverization where bone architecture is destroyed
- Patients with limited or poor dentition (where traditional maxillomandibular fixation is unreliable as a reduction guide)
- Atrophic edentulous mandible fractures (Luhr Class III — height <11mm)

**Not typically indicated for VSP:**
- Non-comminuted single-subsite fractures
- Cases manageable with direct anatomical reduction and standard hardware

*Source: [Singerman et al., Craniomaxillofacial Trauma & Reconstruction 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11995825/)*

### 1.2 Step-by-Step Trauma VSP Workflow

#### Phase 1: Image Acquisition
- **CT scan parameters:** Fine-cut CT maxillofacial scan, typically ≤0.625 mm slice thickness (0.58 mm reported at University of Kansas Medical Center). Must be a high-resolution helical CT — standard trauma CT (3mm cuts) is often insufficient for VSP.
- **Format:** DICOM exported to commercial VSP vendor or in-house software
- **Timing:** CT ideally acquired at initial presentation; VSP can begin immediately after

#### Phase 2: Virtual Planning Session (Commercial Workflow)
- Surgeon schedules a **web-based live planning session** with a third-party vendor's clinical engineer (3D Systems/VSP, Materialise/ProPlan CMF, KLS Martin/IPS CaseDesigner, or Stryker/VSP Reconstruction)
- During the session (typically 60–120 minutes):
  1. **Segmentation:** Engineer segments bone fragments from DICOM data — separating individual fracture fragments as distinct 3D objects
  2. **Virtual fracture reduction:** Each fragment is repositioned computationally to approximate pre-injury anatomy
  3. **Occlusal relationship establishment:** The surgeon guides how maxillary and mandibular fragments should be repositioned relative to each other; in edentulous/comminuted cases, mirroring from the contralateral side is used
  4. **Condyle positioning:** Condylar heads are seated in the glenoid fossae bilaterally — this is a critical, operator-dependent judgment step
  5. **Assessment of bony defects:** Identification of areas with bone loss that will require grafting or spanning
  6. **Deliverable design:** Surgeon specifies which outputs are needed (see below)

#### Phase 3: Fabrication of Surgical Deliverables
Depending on case requirements, the vendor produces:

| Deliverable | Use | Turnaround |
|-------------|-----|------------|
| **Occlusal splint (acrylic or resin)** | Guides MMF and confirms reduction intraoperatively | 2–5 days |
| **3D-printed anatomical model** | Pre-bending stock plates; intraoperative reference | 3–7 days |
| **Pre-contoured (patient-specific) plate** | Eliminates intraoperative plate bending; critical for comminuted mandible | 5–10 days |
| **Patient-specific implant (PSI)** | Custom titanium or PEEK for complex reconstruction (orbit, zygoma) | 7–14 days |
| **Cutting/drilling guides** | Transfer osteotomy positions to the operative field | 3–7 days |
| **Modified DICOM** | Enables intraoperative navigation overlay | Same as plan |

*Average consult-to-surgery time in the Singerman 2025 series: **8.6 days** (range 5–13 days). Historically up to 14 days.*

#### Phase 4: Intraoperative Execution

**Occlusal splint-guided workflow:**
1. Splint is seated on one arch; circumdental wires or arch bars placed
2. Patient placed in maxillomandibular fixation (MMF) using splint to establish pre-morbid occlusion
3. Fractures reduced under splint guidance — the splint constrains the occlusal relationship, transferring the virtual plan to the operative field
4. Plates applied with patient in MMF; splint may be left for postoperative stabilization or removed at case end

**Model-guided plate pre-bending workflow:**
1. Crystal/stereolithographic model sterilized and placed in operative field
2. Surgeon contours stock reconstruction plate to match model surface prior to implantation
3. Pre-bent plate inserted — eliminates intraoperative freehand bending on the native bone

**Navigation-guided workflow (emerging):**
- Plan exported as modified DICOM; imported into navigation system (e.g., Stryker Nav3i, Brainlab)
- Tracker probe attached to fracture segments; segments repositioned with real-time overlay on planned position
- Particularly useful for ZMC and orbital fractures where there is no occlusal reference

*Source: [Same-day VR Planning + Navigation, PMC9555982](https://pmc.ncbi.nlm.nih.gov/articles/PMC9555982/)*

#### Phase 5: Intraoperative Verification
- **Occlusion check:** Manual assessment of occlusal contact; stability and reproducibility tested
- **Navigation overlay:** If navigation used, post-reduction scan compared to preoperative plan
- **Intraoperative CT:** Increasingly used to "close the loop" — acquire CT after fixation, overlay on plan, identify residual deviations requiring correction

### 1.3 Fracture-Specific Workflow Considerations

#### Mandibular Fractures
- Occlusal splint is the primary transfer mechanism; critical when condylar splay is present (symphyseal fracture without splint causes unintentional facial widening)
- Condyle position is the most critical and judgment-dependent step; incorrect condylar seating at the time of CT or planning leads to postoperative malocclusion
- In severely comminuted fractures (3+ fragments), stock plates pre-bent on a model are more reliable than freehand bending

#### Le Fort I / Maxillary Fractures
- Palatal split fractures especially benefit from VSP; custom splints stabilize the palate and resist re-expansion
- Often combined with mandibular fixation; intermediate splint used for maxilla-first repositioning (borrowed from orthognathic workflow)

#### Le Fort II / III (Panfacial)
- Most complex use case; requires coordinated multi-segment virtual reduction
- Mirroring from contralateral side essential if bilateral injury — VSP enables this computationally
- 3D models used for plate pre-bending at multiple facial buttresses (ZF suture, infraorbital rim, Le Fort I bar, zygomaticomaxillary buttress)

#### Orbital Fractures
- Patient-specific implants (titanium mesh or PEEK) designed from planned post-reduction anatomy
- Contralateral orbit mirrored to define target volume; implant designed to restore orbital volume
- Intraoperative navigation increasingly used to verify implant position

#### ZMC (Zygomaticomaxillary Complex) Fractures
- VSP primarily used for secondary reconstruction; acute management often still freehand ORIF
- Navigation and/or PSI most beneficial for revision cases or severe comminution
- Key reference: zygomatic arch projection guides accurate AP repositioning

*Source: [3D Systems VSP CMF Solutions](https://www.3dsystems.com/healthcare/craniomaxillofacial-solutions); [Stryker VSP Reconstruction](https://www.stryker.com/us/en/craniomaxillofacial/products/vsp-reconstruction.html)*

### 1.4 Acute vs. Delayed Reconstruction

| Parameter | Acute (Primary) Repair | Secondary/Delayed Reconstruction |
|-----------|----------------------|----------------------------------|
| VSP timing | Must work around turnaround time (avg. 8.6 days) | No time pressure |
| Anatomy | Displaced but not healed; fragments mobile | Healed malunion; may need osteotomies |
| VSP value | High for comminuted/panfacial | High for all complex cases |
| Adoption level | Emerging/limited | Well-established |
| Accuracy metrics | Less studied | Better studied; 2–3mm mean deviation typical |

---

## 2. Mandibular Fracture Classification and Reduction Planning

### 2.1 Classification Systems

The mandible is the most commonly fractured facial bone, and classification governs surgical approach, reduction technique, and where VSP adds value.

#### AOCMF Classification System (Current Standard)

Three-tiered hierarchical system used for research standardization and increasingly for clinical coding:

**Level 1 (Presence):** Identifies that mandibular fracture exists  
**Level 2 (Topography):** Location-based classification using anatomic region codes:
- **S** — Symphysis/parasymphysis
- **B** — Body (bilateral: BL, BR)
- **A** — Angle/ascending ramus (bilateral: AL, AR)
- **P** — Condylar process (bilateral: PL, PR)
- **C** — Coronoid process

**Level 3 (Morphology/Severity):** Adds:
- Dentition status and atrophy grade
- Tooth injuries and periodontal involvement
- Fragmentation: Grade 0 (none), Grade 1 (minor/single intermediate fragment), Grade 2 (major/total disintegration)
- Bone loss: present or absent

*Source: [AOCMF Classification Mandible Level 2, PMC4251718](https://pmc.ncbi.nlm.nih.gov/articles/PMC4251718/); [AOCMF Level 3, PMC4251719](https://pmc.ncbi.nlm.nih.gov/articles/PMC4251719/)*

#### Older Classification Systems (Still Clinically Used)
- **Spiessl classification:** Grades I–VI based on displacement, occlusion disruption, soft tissue, associated fractures (FLOSA formula). Most familiar to practicing surgeons.
- **Luhr classification for edentulous mandibles:** Class I (>16mm height), Class II (11–16mm), Class III (<11mm). Class III carries highest risk of non-union; strongly indicates VSP + PSI.
- **Favorable vs. unfavorable:** Based on whether muscle vectors resist (favorable) or distract (unfavorable) fracture fragments — critical for ORIF vs. closed reduction decision

#### Functional Classification for VSP Indications
The key driver for VSP is **number and displacement of fragments, not just location**:
- 1–2 non-displaced fragments: traditional ORIF, no VSP needed
- 3+ fragments OR significant displacement: VSP for splint/model guidance
- Severely comminuted (Grade 2 AOCMF): maximum VSP benefit — consider PSI rather than stock plate

### 2.2 Reduction Planning: Manual vs. Virtual

#### Traditional Approach (Still Dominant in Most Centers)
1. Dental impressions taken (often difficult post-trauma due to swelling, intubation, comminution)
2. Stone models poured, sectioned at fracture lines
3. Manual model surgery: fragments repositioned, waxed together in reduced position
4. Acrylic splint fabricated on reduced models in dental laboratory
5. **Critical limitations:**
   - Impressions impossible in many trauma scenarios
   - Only dentoalveolar surface is registered; cortical alignment and condylar position are assumed
   - No 3D visualization of reduction adequacy
   - Technical skill and time requirement often causes splints to be skipped entirely

#### Virtual Approach
1. DICOM → 3D model generation (segmentation)
2. Each fragment becomes an independent 3D object
3. Virtual reduction performed in software: fragments repositioned using 3D translation/rotation tools
4. Condylar heads virtually seated in glenoid fossae
5. Occlusal contact verified digitally
6. Surgeon approves plan; splint geometry derived from final reduced position
7. **Outputs:** Splint STL for printing, plate/PSI design, drilling guide, anatomic model

#### Key Reduction Concepts in Virtual Planning

**Condylar position:** The most debated and critical step. The condyle must be in the correct position within the glenoid fossa. Errors here propagate to malocclusion. VSP does not automatically find the correct condylar position — the surgeon/engineer must judge fossa seating from CT imaging. Emerging AI approaches aim to automate condylar seating assessment (see Section 7).

*Source: [Automated condylar seating assessment, PMC11371884](https://pmc.ncbi.nlm.nih.gov/articles/PMC11371884/)*

**Interfragmentary gap management:** VSP can quantify gaps between reduced fragments; surgeon judges whether primary bone healing is expected or whether bone grafting is needed.

**IAN (inferior alveolar nerve) avoidance:** In atrophic edentulous mandibles, the nerve canal is visualized in CT; drilling guides and PSI designs are planned to avoid the nerve while achieving stable fixation. This is a major advantage over freehand ORIF where nerve visualization is poor.

**Plate prebending vs. PSI:**
- **Pre-bent stock plate (using 3D model):** Faster, cheaper, available for acute cases. Plate bent to model preoperatively, not requiring custom manufacturing. Typical titanium reconstruction plates ($1,000–$3,000).
- **PSI (printed or milled):** Higher precision, avoids manual bending errors, can be designed to span defects and accommodate future prosthetics. 2.0–2.4mm plate thickness for load-bearing fixation in atrophic mandibles. Printed PSI: greater design freedom. Milled PSI: stronger, accepts locking screws but limited geometry. PSI cost: several thousand dollars additional.

*Source: [VSP + PSI for atrophic mandible fractures, PMC11562982](https://pmc.ncbi.nlm.nih.gov/articles/PMC11562982/); [CAD/CAM splints for mandibular fractures, ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1010518220300421)*

### 2.3 Fragment Alignment Techniques

**Occlusal-guided reduction (primary method in dentate patients):**
- Splint or arch bars + MMF establish occlusal relationship first
- Reduction proceeds from occlusal reference toward condyle
- Works well for simple/moderate displacement; fails if occlusion itself is unreliable (edentulous, comminuted dental arches)

**Anatomic landmark-guided reduction:**
- Buttress alignment (inferior border, buccal cortex)
- Fracture line re-approximation under direct visualization
- Less reliable for severely displaced or comminuted cases

**Model-guided plate adaptation:**
- Model of reduced anatomy sterilized in the OR
- Stock plate contoured to model surface; shape verified before insertion
- Preserves plate strength better than intraoperative bending

**Navigation-guided reduction:**
- Tracker affixed to fragment; real-time position shown relative to VSP overlay
- Enables hands-free verification of fragment position
- Useful for condylar/subcondylar fractures where direct reduction is limited

---

## 3. Orthognathic Surgical Planning

### 3.1 The Clinical Problem

Orthognathic surgery corrects skeletal jaw deformities — Class II and III malocclusions, facial asymmetry, sleep-disordered breathing, and congenital conditions. The most common procedures are:
- **Le Fort I osteotomy:** Horizontal cut through the maxilla; allows 3D repositioning (advancement, impaction, roll, pitch, yaw)
- **BSSO (Bilateral Sagittal Split Osteotomy):** Splits mandibular rami bilaterally; enables advancement, setback, asymmetry correction
- **Genioplasty:** Repositioning of the chin segment for aesthetic refinement
- **Segmental osteotomies:** Multi-piece Le Fort I or BSSO for complex arch form correction

These are high-stakes elective procedures. Planning errors translate directly to postoperative malocclusion, asymmetry, or relapse. Traditional planning relied on 2D cephalometric radiographs and physical dental model surgery — both with documented limitations.

### 3.2 Data Acquisition Protocol (6-Step Protocol)

The current best-practice protocol for VSP orthognathic planning follows a structured checklist:

#### Step 1: Image Acquisition
| Data Type | Format | Resolution | Clinical Requirements |
|-----------|--------|------------|----------------------|
| CBCT or CT (craniofacial) | DICOM | ≥0.5×0.5×0.5mm (CASS only); ≥0.3×0.3×0.3mm (for CAD/CAM guides) | Patient in Natural Head Position (NHP); condyles in centric relation; no motion blur |
| Intraoral optical scans | STL or PLY | High-resolution full arch | Upper arch, lower arch, bite registration in centric occlusion; no voids |
| Facial surface scan (optional) | STL or PLY | - | Soft tissue facial surface; used for soft tissue simulation |
| Clinical photography | RAW format | - | Frontal, 45° profile, lateral profile; NHP, soft tissues in repose |

**Critical imaging requirement:** Condyles must be in centric relation during CBCT acquisition. This is verified with a wax bite registration. An incorrectly seated condyle at imaging translates to mandibular malposition in the virtual plan, and will result in postoperative malocclusion that matches the virtual plan but not the patient's actual anatomy.

*Source: [Donaldson et al., Sage Journals 2021](https://journals.sagepub.com/doi/10.1177/1465312520954871)*

**CT vs. CBCT trade-offs:**
- Conventional CT: Patient supine (soft tissue distortion), higher radiation (534–860 μSv), larger footprint, reliable for CAD/CAM
- CBCT: Patient upright (NHP preserved, less soft tissue distortion), lower radiation (70–560 μSv), smaller footprint, must validate scanner for CAD/CAM guide construction

#### Step 2: Data Fusion and Alignment
1. CBCT DICOM data imported into CASS software (Materialise Mimics/ProPlan CMF, Simplant O&O, Dolphin Imaging, IPS CaseDesigner, or CASS)
2. Intraoral STL/PLY scans imported and registered to the CT bone model
3. Facial surface scan registered if soft tissue simulation desired
4. **Registration method:** 3-point landmark matching (manual, then ICP algorithm refinement) or surface-based automatic registration
5. **Critical accuracy metric:** Registration error of intraoral scan to CBCT: ~0.21–0.54mm mean (up to 300 μm for full arch). Clinically acceptable but meaningful for splint accuracy.

*Source: [CBCT-intraoral registration accuracy, PMC5599947](https://pmc.ncbi.nlm.nih.gov/articles/PMC5599947/)*

#### Step 3: Segmentation and 3D Model Generation
- Threshold-based segmentation separates bone from soft tissue
- Cranial base, maxilla (with teeth), mandible (with teeth) are segmented as separate objects
- Dental crowns are often better represented in the intraoral STL than in CBCT (CBCT resolution is insufficient for accurate crown morphology)
- Manual cleanup required: metal artifact removal (beam hardening from restorations), separation of adjacent bony structures at joint spaces, thin structure completion (orbital walls, sinus walls)

#### Step 4: Virtual Diagnosis and Cephalometric Analysis
**Reference coordinate system:**
- X-axis (roll): defined by orbits
- Y-axis (pitch): true vertical plane (perpendicular to Frankfurt Horizontal or from NHP photographs)
- Z-axis (yaw): facial midline

**3D cephalometric analysis:**
- Traditional 2D landmarks (SNA, SNB, Wits, ANB) now measured in 3D from CBCT
- Additional 3D assessments: facial width, orbital symmetry, condylar morphology, TMJ anatomy, pharyngeal airway volume, tooth root position relative to cortical plates
- Systematic checklist: TMJs, pharynx, nasal cavity, tooth root morphology, alveolar bone (fenestrations/dehiscences), sinus and nerve anatomy

*Source: [3D cephalometric outcome predictability, PMC9800679](https://pmc.ncbi.nlm.nih.gov/articles/PMC9800679/)*

#### Step 5: Virtual Surgical Simulation (Multidisciplinary Meeting)
This is the core planning step — typically conducted as a live web session (commercial workflow) or internally (in-house workflow):

**Maxilla-first protocol (most common):**
1. Define Le Fort I osteotomy plane: 4-point placement; adjust shape with control points; "split" to create separate osteotomized segment
2. Reposition maxillary segment: translate in x/y/z (anterior-posterior, superior-inferior, transverse) + rotate (roll for cant correction, pitch for incisor inclination, yaw for midline)
3. Target position determined by: upper incisor show at rest/smile, facial midline alignment, cant correction, vertical facial proportion goals
4. Check for bony collisions; assess bony gaps and overlaps
5. Verify root and nerve proximity to osteotomy

**Mandible repositioning:**
1. Define BSSO cuts bilaterally: proximal ramus cut, horizontal cut, distal oblique
2. Move distal mandibular segment to Class I occlusion with the newly positioned maxilla
3. Assess condylar position change resulting from mandibular movement
4. Add genioplasty if indicated: horizontal cut below mental foramina; segment repositioned and designed

**Soft tissue simulation:**
- Physics-based simulation (FEM or mass-spring) predicts soft tissue response to skeletal movement
- Accuracy varies: better for maxillary movements than mandibular; 2–5mm error typical at specific landmarks
- Used for patient communication and aesthetic planning, not for precise outcome prediction

#### Step 6: Splint Design and CAD/CAM
See Section 4 for full detail.

### 3.3 Software-Specific Workflow Notes

#### Materialise ProPlan CMF / Mimics Enlight CMF
- Mimics used for segmentation and 3D model generation
- ProPlan CMF used for orthognathic planning: osteotomy wizards, cephalometric analysis, splint design
- CBCT → 3D model: threshold segmentation in Mimics → export to ProPlan
- Osteotomy wizard (4-step): Draw plan → Adjust plane → Perform osteotomy → Reposition
- Splint design (5-step): Select type (intermediate/final) → Indicate outline by placing points → Generate shape → Edit edges → Create splint
- Export: STL for printing; CSV for cephalometry; XML for head position data
- Planning time: ~45.5 min (mean) for a routine dysgnathia case
- Cost: $8,412–$12,617 per year for software license (service model, per-case costs additional for outsourced planning)

*Source: [Materialise Mimics Enlight CMF Tutorial](https://www.materialise.com/en/academy/healthcare/mimics-innovation-suite/video-tutorials/virtual-planning-orthognathic-surgery-mimics-enlight-cmf); [VA TRM ProPlan CMF document](https://www.oit.va.gov/Services/TRM/files/SynthesProPlanCMF.pdf)*

#### 3D Systems VSP Orthognathics
- Commercially available since 2013; FDA-cleared service-based approach
- Surgeon interacts with engineer during live interactive web session
- High-resolution stone model scans integrated with CT/CBCT data (physical model still used in some workflows)
- Produces intermediate and final splints, cutting guides, patient-specific plates
- Company manages fabrication and shipping

*Source: [3D Systems VSP CMF Solutions](https://www.3dsystems.com/healthcare/craniomaxillofacial-solutions)*

#### IPS CaseDesigner (KLS Martin)
- Planning time: ~36.5 min (mean)
- Cost: $10,900–$14,900 to acquire + $3,000/year support
- Can send splint files directly to company for fabrication ($182.25/splint)
- Less crashes reported in studies

#### Dolphin Imaging
- Planning time: ~33.6 min (mean) — fastest in comparative study
- Cost: ~$25,800 to acquire + $3,500/year support
- Commonly used in orthodontic/orthognathic combined practices

*Source: [Software comparison study, PMC7790928](https://pmc.ncbi.nlm.nih.gov/articles/PMC7790928/)*

#### Simplant O&O (Dentsply-Sirona)
- Used in surgery-first orthognathic approaches
- Integrates CBCT DICOM + STL intraoral scans
- Supports virtual orthodontic planning (VOP) + virtual surgical planning (VSP) in combined workflow

#### Brainlab CMF Planning
- Distinguishing feature: native integration with intraoperative navigation (Brainlab Elements)
- Automates segmentation, alignment, mirroring
- Supports mixed reality visualization via HoloLens for preoperative planning review
- Workflow: plan → navigate → intraoperative CT verification (closed-loop)

*Source: [Brainlab CMF Planning](https://www.brainlab.com/surgery-products/digital-cmf-surgery/cmf-planning/)*

### 3.4 Accuracy Evidence

Meta-analysis (Cureus 2025, 10 studies, 264 patients):
- VSP reduced **planning time** (SMD = 3.19 in favor of VSP)
- VSP reduced **surgical time** (SMD = -0.42)
- Hard tissue accuracy: VSP superior in coronal plane (lateral dimension/asymmetry) vs. TSP; comparable in sagittal plane
- Soft tissue prediction accuracy: VSP slightly superior
- Most studies report deviations of **<2mm** for hard tissue landmarks in VSP cases

*Source: [Systematic review VSP vs TSP, PMC11912070](https://pmc.ncbi.nlm.nih.gov/articles/PMC11912070/)*

Mazzoni et al. (frequently cited): sub-millimeter deviations in orthognathic procedures using VSP.

OR time reduction: mean **73.8 minutes** shorter with VSP vs. freehand surgery (systematic review, PMC11816551).

---

## 4. Occlusal Splint Workflows

### 4.1 Function and Types

Occlusal splints are the primary mechanism for transferring the virtual surgical plan to the operating room. They encode the planned jaw relationship as a physical object that the surgeon places between dental arches.

**Intermediate splint:** Used when maxilla-first surgery is performed. After the maxilla is moved to its planned position, the intermediate splint defines the occlusal relationship between the repositioned maxilla and the unmoved mandible. It allows the surgeon to set the maxilla's planned position using only the mandible's position (which hasn't changed yet) as a reference.

**Final splint:** Used after all osteotomies are completed. Defines the final occlusal relationship. Often retained postoperatively for several weeks during healing.

**Trauma splints (VSP Surgical Splints / VSPSS):** Custom splints used to reduce/stabilize comminuted fractures. Different from elective orthognathic splints — designed around the fractured arch to assist reduction, not to transfer a planned surgical position.

### 4.2 Traditional Splint Fabrication (Still Common)

1. Dental impressions taken (alginate or polyvinyl siloxane)
2. Stone models poured
3. Models mounted on an articulator using face bow transfer (to reproduce maxillomandibular relationship)
4. Model surgery: models sectioned, segments repositioned to match planned movements
5. Splint fabricated in dental laboratory: acrylic resin poured over repositioned models
6. **Limitations:**
   - Requires physical impressions (difficult in trauma, postoperative, or pediatric cases)
   - Articulator mounting introduces errors (face bow inaccuracy, hinge axis variation)
   - Model surgery is operator-dependent; errors propagate
   - Time: 5–7 days to laboratory and back
   - Cannot encode 3D rotational corrections reliably

### 4.3 Digital/VSP Splint Workflow

#### Data Inputs Required
- Virtual reduced or planned bone model (from CT segmentation)
- Dental surface geometry (from intraoral scan STL/PLY or high-resolution stone model scan)
- Registered composite model (bone + teeth co-registered)

#### Design Steps (Materialise ProPlan CMF / Mimics workflow)
1. **Select splint type:** Intermediate or final
2. **Outline placement:** Surgeon places boundary points on the tooth surface to define splint coverage (should be below the height of contour of teeth to prevent occlusal interference and displacement)
3. **Automatic generation:** Software generates splint shape from boundary
4. **Manual refinement:** Drag edges to adjust coverage; remove sharp edges (fill/fillet option)
5. **Wiring holes (optional):** Specify diameter, place interdentally for circumdental wire passage (1.5mm diameter typical)
6. **Thickness:** 2–3mm for adequate strength
7. **Labeling:** Patient ID engraved for traceability
8. **Optimization for printing:** Mesh optimization; wall checks for minimum thickness
9. **Export:** STL file for 3D printing

#### 3D Printing
- **Material:** Biocompatible resin (e.g., Formlabs Dental LT Clear Resin V2) or milled acrylic
- **Printers:** SLA/DLP printers (Form 3B+, etc.) or milling units (CEREC, Roland)
- **Classification:** Class I (US) / Class IIa (EU) medical device
- **Post-processing:** Wash, dry, post-cure (for SLA resin)
- **Sterilization:** Autoclave or cold chemical sterilization (material dependent)
- **Cost:** In-house printing ~$40 material cost per splint; outsourced $182–$500+ per splint

*Source: [Formlabs In-house Occlusal Splint Workflow](https://dental.formlabs.com/blog/in-house-occlusal-splint-parafunctional-habits/); [KLS Martin splint pricing from PMC7790928](https://pmc.ncbi.nlm.nih.gov/articles/PMC7790928/)*

#### Accuracy of 3D-Printed Splints
Study comparing virtually-designed 3D-printed vs. conventional intermediate surgical splints:
- ICC 0.83–0.99 for linear values (excellent agreement)
- No significant differences in linear or angular measurements on articulator
- No significant differences in 2D cephalometric prediction vs. postoperative values
- Conclusion: 3D-printed splints are clinically acceptable

*Source: [3D-printed vs conventional splints accuracy, PMC10719200](https://pmc.ncbi.nlm.nih.gov/articles/PMC10719200/)*

### 4.4 Splint Use in Trauma vs. Orthognathic

| Feature | Orthognathic Splint | Trauma Splint (VSPSS) |
|---------|--------------------|-----------------------|
| Source geometry | Post-osteotomy virtual position | Virtual fracture-reduced position |
| Design driver | Transfer planned movement | Assist fragment reduction; prevent arch collapse |
| Fit surface | Post-op teeth positions | Pre-morbid or estimated teeth positions |
| Wiring | MMF screws or arch bars | Circumdental wires; circumandibular wiring |
| Duration | 2–6 weeks postoperatively | 4–6 weeks; surgeon discretion |
| Bone CT as source | Yes (CT + intraoral scan) | Yes (fine-cut trauma CT only — no intraoral scan possible) |

*Source: [VSPSS for comminuted fractures, PMC7797978](https://pmc.ncbi.nlm.nih.gov/articles/PMC7797978/)*

---

## 5. Data Formats and Pipeline Architecture

### 5.1 Complete Data Flow Diagram

```
IMAGING ACQUISITION
├── CT / CBCT (DICOM)
│   └── Axial slices, typically 512×512 pixel, 0.3–1mm thickness
│   └── Bone window (HU 300–1500), soft tissue (HU <300), metal (HU >1500)
│
├── Intraoral Scan (STL or PLY)
│   └── Full arch optical scan of dentition
│   └── Bite registration scan (separate file)
│   └── Resolution: sub-100 μm point accuracy
│
└── Facial Surface Scan (STL or PLY)
    └── Photogrammetry or structured light
    └── Used for soft tissue simulation overlay
    
          ↓
          
SEGMENTATION
├── Input: DICOM stack
├── Threshold-based segmentation (bone at ~350 HU)
├── Region growing / GrowCut / manual editing
├── Output: Segmentation MASK (per-structure label volumes)
│   └── Cranial base, maxilla, mandible, teeth (separate labels)
│   └── Stored internally as NIFTI, NRRD, or proprietary format
├── STL generation from masks (marching cubes algorithm)
│   └── Mean shape error between software packages: ~0.11mm (not significant)
│   └── Number of triangles: 450K–1.25M (mandible; varies by software)
│   └── File size: 22–71 MB per structure
└── Manual cleanup: artifact removal, hole filling, mesh decimation

          ↓
          
DATA FUSION / REGISTRATION
├── Register intraoral STL to CT mandible/maxilla surface
│   └── Method: 3-point landmark matching → ICP refinement
│   └── Full-arch registration error: ~0.21–0.54mm mean (up to 300 μm)
│   └── More restorations = lower accuracy
│   └── 3 landmark points sufficient (more does not improve accuracy significantly)
├── Register facial surface to CT soft tissue surface
└── Result: Composite 3D model (bone + accurate tooth morphology + soft tissue)

          ↓
          
VIRTUAL SURGICAL PLANNING
├── Input: Bone STL objects (cranial base, maxilla, mandible, teeth)
│   └── Each as independent moveable 3D object
├── Osteotomy simulation: mesh cutting, fragment separation
├── Fragment repositioning: 6-DOF rigid body transformation (tx, ty, tz, rx, ry, rz)
├── Collision detection: identifies bone-bone overlap
├── Gap analysis: quantifies reduction gaps or planned bone movements
├── Cephalometric measurement: automatic landmark detection + angle/distance computation
├── Soft tissue simulation: FEM or mass-spring prediction
└── Output: Planned position STL files per segment

          ↓
          
GUIDE / SPLINT / IMPLANT DESIGN (CAD)
├── Splint design
│   └── Input: Composite dental model at planned occlusal position
│   └── Boundary definition → mesh generation → hole placement → labeling
│   └── Output: Splint STL
├── Cutting guide design
│   └── Input: Bone surface + planned osteotomy plane
│   └── Guide body designed to key to specific anatomical features (teeth, bone contour)
│   └── Drill guide holes positioned for planned screw holes
│   └── Output: Guide STL
├── PSI design
│   └── Input: Defect STL (from resection plan or mirrored anatomy)
│   └── Engineer-designed in CAD software (Materialise 3-matic, Geomagic)
│   └── Output: Implant STL → manufacturing file (for 3D printing or CNC milling)
└── Anatomic model
    └── Input: Reduced/planned bone STL
    └── Output: Printable model STL (supports added, oriented for optimal printing)

          ↓
          
FABRICATION
├── 3D Printing (additive)
│   └── SLA/DLP (splints, guides): biocompatible resin (Dental LT Clear, etc.)
│   └── FDM or PolyJet (models): standard resin/PLA
│   └── DMLS (PSI): titanium alloy (Ti-6Al-4V)
│   └── Turnaround: 1–7 days depending on type
└── CNC Milling (subtractive)
    └── Milled acrylic (splints): stronger, accepts locking screws
    └── Milled PEEK or titanium (PSI): for load-bearing applications

          ↓
          
INTRAOPERATIVE TRANSFER
├── Physical guides / splints / models: direct use in OR
├── Navigation overlay: exported DICOM (modified) imported to nav system
│   └── Nav system: Stryker Nav3i, Brainlab Elements, Medtronic StealthStation
│   └── Registration: surface matching or fiducial markers
└── Mixed Reality (emerging): Microsoft HoloLens, ImmersiveTouch VR

          ↓
          
POSTOPERATIVE VERIFICATION
├── Post-op CT acquired
├── DICOM → STL segmentation of post-op anatomy
├── Superimposition with pre-op plan using ICP on unchanged cranial base
│   └── Fine registration precision: ~30 μm
└── Color deviation map: per-point surface distance between plan and outcome
    └── Typical outcome: mean deviation 2–3mm for condylar repositioning
    └── Reported accuracy for fibula osteotomies: 2.4mm segment length, 3.5° angle deviation
```

### 5.2 File Format Summary

| Format | Role | Key Properties |
|--------|------|----------------|
| **DICOM** | Raw CT/CBCT data input | Stack of 2D slices; contains scanner metadata; HU values encode tissue density |
| **STL (binary)** | Dominant format for 3D models, guides, splints, implants | Surface mesh; no color; compact binary (preferred over ASCII); no units encoded |
| **PLY** | Intraoral scan output (some scanners) | Supports color/texture; larger files; less universal |
| **OBJ** | Some software export formats | Supports material/color; less common in CMF |
| **NIFTI / NRRD** | Segmentation masks (research/open-source tools) | Volumetric label arrays; used in 3D Slicer |
| **STL (exported from plan)** | Splints, guides, models for printing | Final deliverable format |
| **XML** | Head position export (Materialise) | Natural head position reference for planning |
| **CSV** | Cephalometric data export | Landmark coordinates, angles, distances |
| **Modified DICOM** | Navigation system input | Plan geometry encoded for nav overlay |
| **Form file** | Formlabs printer-ready file | Pre-oriented for specific printer; proprietary |

### 5.3 Segmentation Software Landscape

| Software | Type | Key Use |
|----------|------|---------|
| Materialise Mimics | Commercial (primary) | Industry-standard CMF segmentation; $8K–$13K/year |
| Materialise ProPlan CMF | Commercial (planning layer) | Orthognathic and reconstruction planning |
| 3D Slicer | Open-source | Research; community-built CMF extensions |
| InVesalius | Free/open-source | Government-developed; adequate for basic segmentation |
| Brainlab Elements | Commercial | Navigation-integrated; automated segmentation |
| CATIA | CAD software | Used for splint shelling technique in some centers |
| 3-matic (Materialise) | Commercial (CAD) | Post-segmentation editing, guide/implant design |
| Geomagic Freeform | Commercial (CAD) | Haptic-enabled PSI design |

*Source: [DICOM to STL software comparison, PMC7393875](https://pmc.ncbi.nlm.nih.gov/articles/PMC7393875/)*

### 5.4 Critical Data Pipeline Failure Points

1. **CT slice thickness too coarse:** 1.5mm+ slices are insufficient for CAD/CAM guide construction; 0.3–0.6mm required
2. **Patient movement during CBCT:** Motion blur makes segmentation and registration unreliable; causes guide misfit
3. **Condyle not in centric relation during scan:** Mandible is positioned incorrectly; error propagates through entire plan
4. **Metal artifact (restorations, implants):** Beam hardening creates false geometry near metal; registration fails in affected zones
5. **Segmentation threshold errors:** Too high → bone loss in thin areas (orbital walls, nerve canals); too low → adjacent structures merge
6. **Registration error (STL to DICOM):** Up to 0.54mm mean error; higher with more dental restorations or with default segmentation parameters
7. **STL quality:** Insufficient decimation → heavy file; over-decimation → loss of anatomic detail → guide misfit

---

## 6. Manual Pain Points, Error Rates, and Commercial Software Limitations

### 6.1 Error Rate Data

**Orthognathic VSP plan adherence** (Efanov et al., n=54 cases):
- **85% of cases:** Plan adhered to completely
- **11% (orthognathic) / 25% (free flap):** Partial adherence required intraoperative modifications
- **4% (orthognathic):** Plan completely abandoned

*Source: [VSP Pearls and Pitfalls, PMC5811276](https://pmc.ncbi.nlm.nih.gov/articles/PMC5811276/)*

**Complex maxillofacial trauma VSP** (Singerman et al., n=10 cases):
- Complication rate: **70%** (in severe GSW/ballistic cases)
- However: no cases with primary occlusal failure attributable to VSP
- One case deferred due to inability to reduce without VSP

**Condylar repositioning accuracy** (secondary condylar reconstruction, n=20 cases):
- Mean deviation from plan: **2.27mm** (surface distance method)
- Mean reference point deviation: **2.56mm** (range up to 8.99mm)
- Greatest deviation: craniocaudal axis (median 3.30mm) → condylar height most difficult to replicate

*Source: [CAD/CAM for condylar fractures, PMC12701210](https://pmc.ncbi.nlm.nih.gov/articles/PMC12701210/)*

**PSI design variability** (PSI interoperator study, 10 orbital fracture cases, 2 centers, 4 design teams):
- Initial engineer proposals: **37% median difference score** between engineers for the same patient
- After surgeon-engineer meeting: **26% difference score** (surgeon significantly modifies proposal)
- Between surgeons from different centers: **22% difference score**
- Experienced teams: 40% less time, fewer meetings, fewer modifications required

*Source: [PSI variability, PubMed 39266434](https://pubmed.ncbi.nlm.nih.gov/39266434/)*

**Fibula osteotomy accuracy** (VSP for mandibular reconstruction):
- Mean deviation from planned fibular segment lengths: **2.40 ± 2.06mm**
- Mean deviation from planned angles between segments: **3.51 ± 2.69°**

*Source: [Stryker VSP Reconstruction clinical evidence](https://www.stryker.com/us/en/craniomaxillofacial/products/vsp-reconstruction.html)*

### 6.2 Cataloged Manual Pain Points

#### Segmentation (Most Time-Consuming, Operator-Dependent)
- Manual segmentation of CMF structures: **45–60 minutes** per case (expert clinician); 30–90 minutes range
- Requires expert anatomical knowledge to distinguish structures
- Tooth root apices, orbital walls, inferior alveolar nerve canal, maxillary sinus walls are consistently the most difficult regions
- Metal artifacts from dental restorations require manual removal ("erasing") slice-by-slice
- Thin structures (orbital floor, nasal septum) are prone to false holes from threshold-based segmentation
- In fracture cases: segmenting individual bone fragments is especially labor-intensive
- **Automation impact (from AI mirroring study):** Automated pipeline reduced segmentation+mirroring from 45–60 min to **3–5 minutes** (85–90% time reduction)

*Source: [Automated AI mirroring workflow, PMC12653981](https://pmc.ncbi.nlm.nih.gov/articles/PMC12653981/)*

#### Virtual Fracture Reduction
- Entirely manual drag-and-drop fragment repositioning in current commercial software
- No automated fragment fitting algorithm — engineer manually approximates each fragment
- Condyle position must be judged visually from CT — no automatic optimal condylar seating
- Cannot predict post-reduction soft tissue tension or muscle pull
- For severely comminuted cases with many small fragments, reduction can be impractical virtually

#### Condyle Positioning (Critical Failure Mode)
- Condyle not in centric relation during imaging → entire mandibular plan offset
- Verification currently done manually by engineer/surgeon reviewing 2D CT slices
- No standardized objective criterion for "adequate" condylar seating
- Between-observer agreement poor for borderline cases
- **Impact if wrong:** Postoperative malocclusion; may require reoperation

*Source: [Automated condylar seating, PMC11371884](https://pmc.ncbi.nlm.nih.gov/articles/PMC11371884/)*

#### Surgeon-Engineer Communication
- Planning sessions are 60–120+ minutes of live video call
- Surgeon must communicate desired bony movements verbally to engineer who executes in software
- Misunderstandings are a documented cause of plan abandonment (n=1 in Efanov series)
- Different engineers produce significantly different PSI proposals for the same case (37% median difference)
- No standardized protocol for communication or documentation of planned movements

#### Registration / Data Fusion
- Manual 3-point landmark placement for intraoral STL to CBCT registration (most common method)
- Registration accuracy decreases with more dental restorations
- Must be repeated if CT or intraoral scan is re-acquired
- No automated robust registration for trauma cases where anatomy is deformed

#### Soft Tissue Prediction
- Current methods (FEM, mass-spring, machine learning) have 2–5mm errors at specific landmarks
- Highly variable by facial region: better for nose/lips, worse for chin and cheeks
- Cannot predict muscle adaptation, edema, or scarring effects
- Surgeon must still use clinical judgment for aesthetic goals

#### Intraoperative Transfer
- Splint seating depends on quality of dental impressions/scans at planning stage
- Chipped teeth on dental cast → poor splint fit (documented failure in Efanov series)
- Soft tissue swelling intraoperatively can prevent planned bony movements (especially genioplasty)
- Cutting guides may require excessive tissue dissection; if too large → skin necrosis risk
- Navigation registration requires additional OR setup time; patient must remain still

### 6.3 Commercial Software Structural Limitations

| Limitation | Impact |
|------------|--------|
| **Outsourced service model (3D Systems, ProPlan CMF full service):** Surgeon submits CT; cannot access software directly; must wait for web session scheduling | Delays of 5–14 days from CT to planning session; not usable for acute trauma without modified protocols |
| **Per-case costs (commercial outsourced):** $2,000–$10,000+ per case (including PSI) | Limits use to high-complexity cases; financial barrier in resource-limited centers |
| **Software license costs (in-house):** $8,412–$25,800 acquisition + $3,000–$3,500/year maintenance | High upfront investment; not feasible for low-volume centers |
| **No real-time fracture reduction:** Current software is not designed for rapid trauma reduction; engineers need time | Minimum 5-day turnaround; acute trauma cases must wait or plan without VSP |
| **No automated segmentation in most commercial tools:** Materialise Mimics added AI segmentation recently; others still largely manual | 45–60 min per case; bottleneck for scale |
| **Proprietary file formats:** Some software lock planning data in proprietary formats | Limits interoperability; surgeon cannot move plan between platforms |
| **Static plans:** Plan is fixed once approved; intraoperative changes not reflected | Discrepancy between plan and execution not automatically documented |
| **No intraoperative feedback loop:** Current systems don't know what happened in the OR | Outcomes data not captured for quality improvement or ML training |
| **CBCT vs. CT ambiguity:** Some CAD/CAM guide systems require validated CBCT scanners; many hospitals only have conventional CT | CT used for trauma; CBCT used for elective — two different imaging pathways to manage |

*Source: [Software comparison PMC7790928](https://pmc.ncbi.nlm.nih.gov/articles/PMC7790928/); [VSP oral surgery systematic review PMC12014521](https://pmc.ncbi.nlm.nih.gov/articles/PMC12014521/); [Cost outcomes PMC11816551](https://pmc.ncbi.nlm.nih.gov/articles/PMC11816551/)*

### 6.4 Cost Economics

**Outsourced commercial VSP costs:**
| Component | Approximate Cost |
|-----------|-----------------|
| 3D Systems VSP Orthognathics (full service) | ~$3,000–$6,000/case |
| 3D Systems VSP Trauma (model + splint) | ~$2,000–$4,000/case |
| Custom-milled plate + VSP | Approaches $10,000/case |
| PSI (orbital, mandible) | $5,000–$15,000/case |
| Outsourced splint fabrication only | $182–$500/splint |

**In-house approach:**
| Item | Cost |
|------|------|
| ProPlan CMF license (local, 1-user/year) | ~$8,412/year |
| Mimics license | Bundled or ~$8,000+/year |
| 3D printer (SLA, e.g., Form 3B+) | $3,000–$5,000 |
| Resin per splint/guide | ~$40–$80 |
| Personnel (medical engineer) | $80,000–$120,000/year fully loaded |
| Break-even vs. commercial outsourcing | ~27 cases/year |

**OR time savings (from VSP):**
- Average OR time saved: **73.8 minutes** (weighted meta-analysis, PMC11816551)
- OR cost: ~$2,000–$3,500/hour in the US
- Per-case savings from OR time alone: ~$2,500–$4,300
- Net cost comparison: outsourced VSP often cost-neutral or slightly negative vs. freehand; in-house VSP generates positive net savings if volume justifies setup

*Source: [Cost Outcomes VSP PMC11816551](https://pmc.ncbi.nlm.nih.gov/articles/PMC11816551/); [Point-of-care 3D printing for CMF trauma](https://www.oaepublish.com/articles/2347-9264.2020.222)*

---

## 7. AI/ML Augmentation Opportunities

### 7.1 Automated Segmentation (Highest-Readiness, Highest Value)

**Current state:**
- Manual or semi-automatic segmentation is the dominant workflow bottleneck
- Takes 45–60 min per case for experienced clinicians
- Commercial AI segmentation now available in Materialise Mimics (CMF Reconstruction CT, CMF Orthognathic CT, CMF CBCT algorithms)
- Outputs: mandible, skull, maxilla with/without teeth, mandibular teeth, maxillary teeth, vertebrae, soft tissue

**AI performance benchmarks:**
- CMF-ELSeg (nnU-Net ensemble, 143 CT scans): **Dice >0.94** for most teeth; 0.89 overall for facial bones
- Mandible segmentation accuracy: **Dice 0.984, ASSD 0.324mm** (near-surgical precision)
- Zygomatic bones: **Dice 0.931, ASSD 0.487mm**
- Challenging structures: hyoid bone, inferior alveolar nerve (Dice <0.9), orbital walls (thin, low contrast)
- Clinical grade evaluation: Grade A or B (acceptable for VSP) in >87% of fracture cases

*Source: [CMF-ELSeg deep ensemble, PMC12094958](https://pmc.ncbi.nlm.nih.gov/articles/PMC12094958/); [U-Net facial bone segmentation, PubMed 36328865](https://pubmed.ncbi.nlm.nih.gov/36328865/)*

**Automation impact:**
- Segmentation + mirroring time: reduced from 45–60 min to **3–5 minutes** with automated pipeline
- 85–90% reduction in operator time

**Remaining gap:** Complex craniofacial conditions (e.g., rare syndromes, bilateral defects) show 40% Grade C/D rates requiring significant manual correction.

**Platform opportunity:** Build a segmentation service/API that:
- Accepts DICOM input
- Returns labeled STL files per structure (mandible, maxilla with teeth, skull)
- Includes confidence scores per region
- Flags regions requiring manual review
- Provides slice-by-slice overlay for surgeon verification

### 7.2 Automated Fracture Fragment Identification

**Current state:** No commercial system automatically identifies and segments individual fracture fragments from trauma CT.

**Research state:** nnU-Net achieves reliable performance for intact anatomy. Fracture cases require fracture-specific training data — rare, poorly labeled public datasets.

**Specific challenge:** A severely comminuted mandible may have 8–15 individual bone fragments, each requiring independent segmentation. Current approach: engineer manually isolates each fragment (15–30 min for complex cases).

**AI opportunity:** 
- Instance segmentation model for fracture fragments
- Each fragment as a separately labeled object
- Possible approach: nnU-Net for coarse anatomy → connected component analysis → fragment-level refinement
- Training data requirement: annotated trauma CT scans with per-fragment labels (scarce)

### 7.3 Automated Virtual Fracture Reduction

**Current state:** Entirely manual. Engineer drag-and-drops each fragment until the surgeon is satisfied. No algorithmic guidance.

**Research state:** Very early stage. No validated automated reduction algorithm for complex facial fractures exists.

**Algorithmic approaches under development:**
1. **Surface matching:** Fracture surfaces are irregular but complementary; shape-matching algorithms (ICP, point cloud registration) could identify complementary fragment pairs
2. **Contralateral mirroring:** For hemifacial trauma, the contralateral anatomy is mirrored to generate target positions for ipsilateral fragments; the target is known, fragments just need to be fit to it
3. **Condyle-first reduction:** Anchor condyles in fossa (using fossa geometry); propagate reduction distally through the arch

**Platform opportunity:** Automated "first-pass" reduction with surgeon review and adjustment. Even a rough automated reduction that gets 60–70% correct reduces engineer/surgeon time significantly.

### 7.4 Condylar Position Assessment

**Current state:** Visual judgment by engineer/surgeon from 2D CT slices. Subjective, experience-dependent. Not standardized.

**AI approach (published):** Feed-forward neural network + multi-step segmentation + ray-casting to assess condyle-fossa relationship objectively. Study demonstrated viability with "encouraging results."

*Source: [Automated condylar seating, PMC11371884](https://pmc.ncbi.nlm.nih.gov/articles/PMC11371884/)*

**Clinical value:** High. Incorrect condylar seating is the most common cause of plan failure for orthognathic cases. An automated flag ("condyle appears unseated — recommend rescan or manual correction") could prevent the most common error mode.

**Platform opportunity:** Condylar seating QA module — automatically scores condylar position after segmentation; alerts surgeon if CBCT was taken with condyles in incorrect position.

### 7.5 Automated Cephalometric Landmark Detection

**Current state:** Manual landmark placement by orthodontist/surgeon. Time: 15–30 minutes. Operator-dependent. All subsequent cephalometric measurements depend on accurate landmark placement.

**AI performance:** Multiple published models achieve <2mm mean landmark detection error on CBCT/CT. Commercial tools (Materialise, Dolphin) include automated landmark detection.

**Research benchmark:** CNN-based landmark detection systems achieve 1–3mm accuracy; comparable to inter-observer variability of manual measurement.

**Remaining challenge:** Pathological anatomy (missing teeth, fractures, syndromic asymmetry) degrades landmark detection accuracy. Confidence-weighted predictions with uncertain-landmark flagging would address this.

### 7.6 Surgical Simulation and Outcome Prediction

**Soft tissue prediction:**
- FEM: physically accurate but slow (minutes to hours); not practical for iterative planning
- Mass-spring: faster; less accurate
- Deep learning (published): comparable accuracy to FEM at inference speeds of seconds
- Key paper: CNNs predicted postoperative soft tissue position from planned skeletal movements; ~2–4mm accuracy at key landmarks

*Source: [AI in orthognathic surgery, PMC12178734](https://pmc.ncbi.nlm.nih.gov/articles/PMC12178734/)*

**Osteotomy outcome prediction:** CNNs detecting landmarks on CT + predicting postoperative positions based on osteotomy design; average error 5mm (bony reference points).

**Asymmetry assessment:** 3D facial images pre/post; CNN achieved 78% accuracy (vs. 65% for 2D).

**Complication prediction:**
- ANN for postoperative infection prediction: **98.7% accuracy, AUC 0.87** (from 200-patient dataset)
- Random forest for blood loss prediction: strong link between actual and predicted blood loss (900-patient dataset)

### 7.7 Automated PSI Design

**Current state:** PSI design is entirely manual CAD work. Engineers from two different centers design significantly different implants for the same patient (37% median difference). Experienced teams: 40% less time and fewer design revisions.

**AI opportunity:** Generative design models that produce PSI geometry given:
- Defect volume (from segmentation)
- Fixation requirements (screw hole positions, plate thickness)
- Anatomical constraints (IAN path, dental roots, mucosal surface)
- Contralateral anatomy as symmetry target

This is not a solved problem but is directly analogous to generative design in aerospace/medical device CAD — a tractable ML problem given sufficient training data.

### 7.8 Treatment Planning Decision Support

**Surgical need prediction:** ML models (XGBoost, random forest, neural networks) predict whether orthognathic surgery is needed from cephalometric measurements:
- Class III: **0.82–0.87 accuracy**
- Class II: **0.76–0.86 accuracy**
- Most important features: Wits, ANB, SNB

*Source: [AI prediction for orthognathic surgery, PMC11789623](https://pmc.ncbi.nlm.nih.gov/articles/PMC11789623/)*

**Clinical decision support opportunity:** A pre-consultation screening tool that flags patients likely to need orthognathic surgery based on radiograph + cephalometric data. Would help orthodontists refer earlier.

### 7.9 Intraoperative Navigation + AI

**Same-day VR planning workflow:**
- DICOM → ImmersiveTouch VR software → fracture segmentation + reduction: **~5 minutes**
- Plan data exported to Stryker Nav3i: **4 minutes**
- Calibration: **7 minutes**
- Total plan-to-navigation-ready: **~16 minutes** for trauma case

*Source: [Same-day VR planning PMC9555982](https://pmc.ncbi.nlm.nih.gov/articles/PMC9555982/)*

**AI opportunity for navigation:** Real-time bone fragment tracking + AI-driven repositioning guidance (current navigation shows where you are; AI could recommend where to go and how much force to apply).

### 7.10 Summary: AI Opportunity Priority Matrix

| Opportunity | Clinical Value | Technical Readiness | Data Availability | Priority |
|-------------|---------------|--------------------|--------------------|----------|
| Auto-segmentation of intact CMF from CT | Very High | High (SOTA validated) | Moderate | **1** |
| Condylar seating QA | Very High | Moderate (promising) | Moderate | **2** |
| Auto landmark detection (cephalometry) | High | High | Good | **3** |
| Fracture fragment segmentation | Very High | Low-Moderate | Low | **4** |
| Automated fracture reduction (first-pass) | High | Low | Very Low | **5** |
| Soft tissue prediction (DL-based) | High | Moderate | Moderate | **6** |
| Surgical outcome prediction | Moderate | Moderate | Moderate | **7** |
| PSI generative design | High | Low | Low | **8** |

---

## 8. Commercial Landscape Summary

### 8.1 Key Platform Players

| Company | Products | Model | Strengths | Weaknesses |
|---------|----------|-------|-----------|------------|
| **Materialise (Belgium)** | Mimics, ProPlan CMF, 3-matic | License + service | Industry standard; best CMF segmentation; widest clinical base | Expensive; fragmented tools (Mimics + ProPlan separate) |
| **3D Systems (USA)** | VSP Orthognathics, VSP Reconstruction, VSP Trauma, VSP Cranial | FDA-cleared service | Long track record; full-service; strong commercial relationships (Stryker partnership) | Outsourced only; engineer-dependent; slow turnaround |
| **DePuy Synthes/J&J (Stryker partnership)** | ProPlan CMF + TRUMATCH CMF | Service + hardware | Integrated implant/guide solution; commercial surgery relationships | Joint Stryker/3D Systems dependency |
| **Brainlab (Germany)** | CMF Planning, Elements, navigation | Platform | Navigation integration; automation; MR visualization | Less specialized for splints; focused on navigation |
| **KLS Martin (Germany)** | IPS CaseDesigner | License + service | Fast planning time; integrated splint ordering | Less international presence |
| **Dolphin Imaging (USA)** | Dolphin 3D | License | Fastest planning time; popular in orthodontics | Limited CMF trauma features |
| **Dentsply Sirona** | Simplant O&O | License | Strong orthodontic integration | Less trauma capability |

### 8.2 Stryker-3D Systems Relationship
Stryker and 3D Systems operate a commercial VSP Reconstruction service specifically for mandibular reconstruction and trauma. 3D Systems provides engineering, software, and fabrication; Stryker provides the commercial distribution and hardware relationships (reconstruction plates, screws). This integrated commercial service is dominant in the US market.

### 8.3 In-House vs. Outsourced Trend
The field is moving toward in-house VSP workflows at high-volume academic centers. Break-even for in-house vs. outsourced: ~27 cases/year. Accuracy advantage of commercial outsourced VSP is measurable but not always clinically significant. Cost advantage of in-house is significant.

*Source: [In-house vs. commercial VSP for mandibular reconstruction, PubMed 38205891](https://pubmed.ncbi.nlm.nih.gov/38205891/)*

---

## 9. Key Takeaways for Product Builders

### 9.1 What Surgeons Actually Need (That Current Tools Don't Deliver)

1. **Fast segmentation with immediate results.** Current commercial workflows require scheduling a web session with an engineer 5–14 days after CT. A platform that returns segmented, plan-ready 3D models within hours of DICOM upload would be transformational, especially for trauma.

2. **Fracture-specific segmentation.** No existing commercial tool reliably segments individual fracture fragments automatically. This is an unsolved, high-value problem.

3. **Objective condylar seating verification.** Surgeons need an automated flag if the CBCT was acquired with condyles not in centric relation — before the planning session begins, not after the patient comes back with malocclusion.

4. **Plan-in-the-loop intraoperative recording.** There is no link between the preoperative plan and what actually happened in the OR. Outcomes cannot be systematically compared to plans unless the OR team manually measures postoperative CT and overlays it with the plan. A platform that captures intraoperative deviations would enable outcomes tracking and ML training data generation.

5. **Surgeon-direct software.** The engineer-mediated model (surgeon describes; engineer executes in software) is a friction-filled interface. Surgeons who want direct software access are limited to expensive in-house setups with steep learning curves. A surgeon-facing interface that makes routine planning steps accessible without deep engineering knowledge would unlock a large underserved market.

### 9.2 Workflow Steps Where Automation Has Highest ROI

| Step | Current Time | AI-Automatable? | ROI if Automated |
|------|-------------|-----------------|------------------|
| CT → 3D model (segmentation) | 45–60 min | Yes (commercially available) | Very High |
| Intraoral STL → CBCT registration | 5–15 min manual | Partially (ICP exists; needs UI) | High |
| Cephalometric landmark detection | 15–30 min | Yes (validated) | High |
| Virtual fracture reduction | 30–90 min | First-pass only | Very High |
| Condylar seating assessment | 5–15 min | Yes (published approach) | Very High |
| Splint boundary definition | 10–20 min | Partially | Moderate |
| PSI design | 60–180 min | Generative (early) | High if achievable |
| Postoperative accuracy assessment | 20–40 min | Yes (ICP + deviation maps) | Moderate-High |

### 9.3 Data Architecture Principles for a CMF Platform

1. **DICOM ingestion must be first-class.** All workflows start from CT/CBCT DICOM. The platform must handle diverse scanners, slice thicknesses, and acquisition protocols.

2. **STL is the universal transfer format.** Design for STL in / STL out at every stage boundary. Avoid proprietary intermediate formats.

3. **Intraoral scan files (STL or PLY) are a second input stream.** For orthognathic planning, they are required. For trauma planning, they may not exist (or must be derived from CT). The platform must handle both cases.

4. **Per-fragment tracking matters for trauma.** The data model must represent a fracture case as a collection of N independently moveable bone fragment objects, not a single unified model.

5. **Plan versioning is essential.** Surgical plans are revised multiple times during web sessions. The platform must store intermediate states, not just the final approved plan.

6. **Outcome integration requires postoperative DICOM.** To close the loop and generate training data, the platform must support postoperative CT ingestion, automatic overlay registration, and deviation quantification.

7. **Condylar position is a critical metadata field.** Automatically assess and record whether condyles appear correctly seated in every scan. Flag deviations before planning begins.

### 9.4 Regulatory Considerations
- VSP software is FDA Class II medical device software (510(k) clearance required for US commercial use)
- 3D Systems VSP received FDA clearance as a service-based approach; Materialise Mimics is FDA cleared
- AI-powered segmentation tools used clinically must be cleared or through a service model that places responsibility on licensed clinical engineers
- Patient-specific implants: FDA requires a "pre-market submission" when implant design is patient-specific; in-house printed PSI has different regulatory pathway than commercial milled titanium PSI

---

## References

1. Singerman KW et al. "Virtual Surgical Planning for Management of Acute Maxillofacial Trauma." *Craniomaxillofacial Trauma & Reconstruction* (2025). https://pmc.ncbi.nlm.nih.gov/articles/PMC11995825/

2. Efanov JI et al. "Virtual Surgical Planning: The Pearls and Pitfalls." *Plastic and Reconstructive Surgery Global Open* (2018). https://pmc.ncbi.nlm.nih.gov/articles/PMC5811276/

3. Donaldson CD, Manisali M, Naini FB. "Three-dimensional virtual surgical planning (3D-VSP) in orthognathic surgery." *Journal of Dentistry* (2021). https://journals.sagepub.com/doi/10.1177/1465312520954871

4. Materialise Mimics Enlight CMF Tutorial. "Virtual Planning for Orthognathic Surgery." https://www.materialise.com/en/academy/healthcare/mimics-innovation-suite/video-tutorials/virtual-planning-orthognathic-surgery-mimics-enlight-cmf

5. "PROPLAN CMF 3.0.1 Software Documentation." VA TRM. https://www.oit.va.gov/Services/TRM/files/SynthesProPlanCMF.pdf

6. Kongsong W, Sittitavornwong S. "Utilization of Virtual Surgical Planning for Surgical Splint-Assisted Comminuted Maxillomandibular Fracture Reduction." *Craniomaxillofacial Trauma & Reconstruction* (2020). https://pmc.ncbi.nlm.nih.gov/articles/PMC7797978/

7. Almansour H et al. "Management of Atrophic Edentulous Mandible Fractures Utilizing Virtual Surgical Planning and Patient-Specific Implants." *Craniomaxillofacial Trauma & Reconstruction* (2024). https://pmc.ncbi.nlm.nih.gov/articles/PMC11562982/

8. Yoneyama M et al. "Virtually Planned and CAD/CAM-Guided Secondary Reconstruction of Condylar Fractures." *Maxillofacial Plastic and Reconstructive Surgery* (2025). https://pmc.ncbi.nlm.nih.gov/articles/PMC12701210/

9. "AOCMF Classification System: Mandible Fractures Level 2." *Craniomaxillofacial Trauma & Reconstruction* (2014). https://pmc.ncbi.nlm.nih.gov/articles/PMC4251718/

10. "AOCMF Classification System: Mandible Fractures Level 3." *Craniomaxillofacial Trauma & Reconstruction* (2014). https://pmc.ncbi.nlm.nih.gov/articles/PMC4251719/

11. CAD/CAM splints in mandibular fractures RCT. *Journal of Cranio-Maxillofacial Surgery* (2020). https://www.sciencedirect.com/science/article/abs/pii/S1010518220300421

12. Comparison feasibility/cost of VSP software. *Maxillofacial Plastic and Reconstructive Surgery* (2021). https://pmc.ncbi.nlm.nih.gov/articles/PMC7790928/

13. 3D cephalometric VSP outcome predictability. *Progress in Orthodontics* (2022). https://pmc.ncbi.nlm.nih.gov/articles/PMC9800679/

14. Effectiveness of Traditional vs VSP orthognathic. *Cureus* (2025). https://pmc.ncbi.nlm.nih.gov/articles/PMC11912070/

15. Accuracy of 3D VSP in orthognathic surgery. *Cureus* (2024). https://pmc.ncbi.nlm.nih.gov/articles/PMC11554385/

16. DICOM segmentation and STL creation comparison. *3D Printing in Medicine* (2020). https://pmc.ncbi.nlm.nih.gov/articles/PMC7393875/

17. 3D-printed vs conventional intermediate splints. *Journal of Maxillofacial & Oral Surgery* (2023). https://pmc.ncbi.nlm.nih.gov/articles/PMC10719200/

18. Cost Outcomes of VSP in Head and Neck Reconstruction. *Head & Neck* (2025). https://pmc.ncbi.nlm.nih.gov/articles/PMC11816551/

19. Same-day VR surgical planning + navigation. *Plastic and Reconstructive Surgery Global Open* (2021). https://pmc.ncbi.nlm.nih.gov/articles/PMC9555982/

20. PSI design variability in orbital fractures. *PubMed* (2024). https://pubmed.ncbi.nlm.nih.gov/39266434/

21. CMF-ELSeg deep ensemble segmentation. *Frontiers in Bioengineering* (2025). https://pmc.ncbi.nlm.nih.gov/articles/PMC12094958/

22. Automated AI mirroring for craniofacial reconstruction. *Journal of Imaging* (2025). https://pmc.ncbi.nlm.nih.gov/articles/PMC12653981/

23. AI applications in orthognathic surgery review. *BioMed Research International* (2025). https://pmc.ncbi.nlm.nih.gov/articles/PMC12178734/

24. AI as prediction tool for orthognathic surgery. *Orthodontics & Craniofacial Research* (2024). https://pmc.ncbi.nlm.nih.gov/articles/PMC11789623/

25. Automated condylar seating assessment. *Clinical Oral Investigations* (2024). https://pmc.ncbi.nlm.nih.gov/articles/PMC11371884/

26. Deep learning orbital segmentation. *Scientific Reports* (2021). https://www.nature.com/articles/s41598-021-93227-3

27. CBCT-intraoral scan registration accuracy. *Clinical Oral Implants Research* (2016). https://pmc.ncbi.nlm.nih.gov/articles/PMC5599947/

28. VSP use in oral surgery systematic review. *Cureus* (2025). https://pmc.ncbi.nlm.nih.gov/articles/PMC12014521/

29. In-house vs commercial VSP for mandibular reconstruction. *PubMed* (2024). https://pubmed.ncbi.nlm.nih.gov/38205891/

30. Evaluation of VSP accuracy in complex maxillofacial trauma. *Journal of Pharmacy & Bioallied Sciences* (2025). https://pmc.ncbi.nlm.nih.gov/articles/PMC12788456/

31. 3D Systems VSP CMF Solutions. https://www.3dsystems.com/healthcare/craniomaxillofacial-solutions

32. Stryker VSP Reconstruction. https://www.stryker.com/us/en/craniomaxillofacial/products/vsp-reconstruction.html

33. Materialise ProPlan CMF. https://www.materialise.com/en/healthcare/proplan-cmf

34. Materialise AI-enabled segmentation. https://www.materialise.com/en/healthcare/mimics/ai-enabled-segmentation

35. Brainlab CMF Planning. https://www.brainlab.com/surgery-products/digital-cmf-surgery/cmf-planning/

36. AO Foundation ORIF Le Fort III. https://surgeryreference.aofoundation.org/cmf/trauma/midface/le-fort-iii/orif

37. Point-of-care 3D printing for CMF trauma. *OAE Publishing* (2021). https://www.oaepublish.com/articles/2347-9264.2020.222

38. AO CMF Digital Workflows blog. https://www.aofoundation.org/cmf/about-aocmf/blog/updated-mft-curriculum-vsp-3d-modeling-and-ar
