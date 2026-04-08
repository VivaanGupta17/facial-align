# Competitive Landscape & UX Research: Virtual Surgical Planning (VSP) for Craniomaxillofacial Surgery

**Research Date:** July 2025  
**Scope:** VSP platforms, interface patterns, surgeon UX, open-source tools, and AI-native redesign recommendations for CMF surgery

---

## Table of Contents

1. [Market Overview](#1-market-overview)
2. [3D Systems VSP / VSP Orthognathics](#2-3d-systems-vsp--vsp-orthognathics)
3. [Materialise CMF / Mimics](#3-materialise-cmf--mimics)
4. [Brainlab CMF](#4-brainlab-cmf)
5. [Dolphin Imaging / Nemoceph](#5-dolphin-imaging--nemoceph)
6. [Blue Sky Plan / InVivo Dental](#6-blue-sky-plan--invivo-dental)
7. [Open-Source Alternatives](#7-open-source-alternatives)
8. [Cross-Platform UX Pattern Analysis](#8-cross-platform-ux-pattern-analysis)
9. [What Is Outdated About Current Tools](#9-what-is-outdated-about-current-tools)
10. [AI-Native Redesign Recommendations](#10-ai-native-redesign-recommendations)
11. [UX Patterns Specific to Medical/Surgical 3D Planning](#11-ux-patterns-specific-to-medicalsurgical-3d-planning)

---

## 1. Market Overview

Virtual surgical planning (VSP) involves importing medical imaging data (CT/CBCT/MRI), segmenting anatomical structures into 3D models, simulating osteotomies and bone movements, and designing patient-specific surgical guides, splints, and implants. The dominant use cases in CMF surgery are:

- **Orthognathic surgery** — LeFort I, BSSO, genioplasty, bimaxillary procedures
- **Mandibular reconstruction** — fibula free flap with cutting guides and prebent plates
- **Craniosynostosis / cranial vault distraction** — pediatric cranial vault remodeling
- **Trauma** — fractured facial skeleton reduction with repositioning guides
- **Midface reconstruction** — post-oncologic or post-traumatic orbito-zygomatic cases

VSP is delivered two ways: (1) **outsourced service models** (surgeon interacts with an engineer who does the planning), or (2) **in-house software licenses** (the surgical team does the planning themselves). Studies comparing these models show in-house workflows achieve the greatest cost savings when case volume exceeds ~27 per year ([PMC: Cost Outcomes of VSP](https://pmc.ncbi.nlm.nih.gov/articles/PMC11816551/)).

**Market players by category:**

| Category | Tools |
|---|---|
| Outsourced full-service | 3D Systems VSP (via Stryker) |
| In-house license, CMF-focused | Materialise ProPlan CMF, Mimics Enlight CMF, Brainlab Elements CMF, IPS CaseDesigner (KLS Martin) |
| Orthodontic/cephalometric-origin | Dolphin Imaging 3D Surgery, Nemoceph/NemoFAB |
| Dental implant/CBCT-origin | Blue Sky Plan, InVivo Dental (Anatomage/DEXIS) |
| Open-source | 3D Slicer + SlicerCMF, BoneReconstructionPlanner, ITK-SNAP |
| Emerging | 3D Surgical platform (web-based), AI-enhanced tools via Enhatch |

**Key clinical outcomes context:** VSP reduces OR time by a mean of 48–120 minutes per case across studies, reduces intraoperative ischemia time for fibula free flaps, and delivers translational accuracy within 1.6–2.3 mm and rotational accuracy within 1.2–2.75°. Planning time with modern software is 15–45 minutes per case depending on complexity and software. The most used commercial packages are ProPlan CMF, Dolphin 3D, and Rhinoceros, with open-source tools gaining ground in research and training settings ([Journal of Clinical Medicine, 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC12841811/)).

---

## 2. 3D Systems VSP / VSP Orthognathics

**Source:** [3D Systems CMF Solutions](https://www.3dsystems.com/healthcare/craniomaxillofacial-solutions) | [VSP Connect announcement](https://www.additivemanufacturing.media/products/3d-systems-vsp-connect-streamlines-preoperative-planning-for-better-patient-outcomes) | [Stryker partnership](https://www.stryker.com/us/en/about/news/2018/3d-systems-and-stryker-team-up-to-advance-personalized-surgery-.html)

### Platform Model and Business Structure

3D Systems VSP is primarily a **managed service**, not self-service software. The surgeon does not plan independently — they participate in an interactive web meeting with a 3D Systems biomedical engineer who drives the planning session. This is the key architectural distinction: the interface that surgeons see is primarily a collaborative web viewer, not a standalone surgical planning workstation.

This service model was FDA-cleared over 10 years ago. 3D Systems has planned more than 150,000 patient-specific cases and manufactured over 2 million implants/instruments across 100+ CE-marked and FDA-cleared devices. Stryker became the exclusive distribution partner for CMF in 2018.

### Workflow Steps

1. **Case initiation** — Surgeon uploads DICOM files via a web portal (two-step: zip upload + email order form to vsp@3dsystems.com). No sophisticated intake UX.
2. **Web meeting** — Interactive planning session between surgeon and 3D Systems biomedical engineer. Surgeon communicates desired osteotomies, movements, and outcomes; engineer executes these in the planning software.
3. **Plan review** — Surgeon reviews and approves the plan
4. **Manufacturing** — 3D Systems 3D-prints patient-specific guides, splints, and models at their facilities (Littleton CO; Leuven, Belgium)
5. **Delivery & QC** — Physical artifacts shipped to the hospital
6. **Intraoperative use** — Guides and splints used in the OR

### VSP Product Lines

| Product | Indication | Key Output |
|---|---|---|
| VSP Orthognathics | LeFort I, BSSO, genioplasty, segmental cases | Intermediate + final occlusal splints (clear, 30-day intraoral approved), osteotomy guides |
| VSP Reconstruction | Mandibular/maxillary reconstruction, fibula free flap | Mandible model with pre-planned cut lines, cutting and positioning guides |
| VSP Cranial | Craniosynostosis, cranial vault distraction | Positioning guides with real-time comparison to age-matched normative contours |
| VSP Distraction | Distraction osteogenesis | Templates for distractor placement, vector planning |
| VSP Trauma | Facial fracture reduction | Osteotomy and positioning guides, splints, Modified DICOM for intraoperative navigation |
| Jaw in a Day | Jaw reconstruction + immediate dental rehabilitation | Guide and prosthesis designs, single-stage planning |

### Interface and Surgeon Experience

- Surgeons interact with a **collaborative web viewer** during the planning session rather than using dedicated software locally
- Real-time 3D bony movement and cephalometric analysis are shown to the surgeon during the session
- Underlying teeth roots and nerves are visualized for osteotomy planning and distractor placement
- In 2023, **VSP Connect** was launched: a centralized cloud-based portal (powered by Enhatch AI) allowing surgeons to view 3D models and comment on implant/guide placement at their convenience (24/7 asynchronous access). Key UX elements include:
  - Single interface for aggregated case data
  - Send notes or alerts to case stakeholders
  - Pre-populated AI-generated designs tailored to surgeon preferences and standard product types
  - Real-time case status tracking
  - Communication thread for device representatives, case managers, designers, and surgeons

### Pricing Model

Pricing is not published. VSP is billed as a per-case service with variable cost depending on complexity and outputs ordered. The "per-case" cost includes the planning session and physical deliverables. Academic literature reports commercial VSP planning + materials costs of:
- Outsourced commercial VSP: median ~€2,000 per case (range €600–€3,800) for orthognathic
- Reconstruction guides + prebent plates: ~$5,098 per case
- Patient-specific reconstruction plates: ~$6,980 per case

### Limitations and Known Problems

- **Surgeon autonomy**: The service model means the surgeon is not in direct control; they must communicate intent to an engineer. Poor surgeon-engineer communication is one of the top cited reasons for plan failures ([PMC: VSP Pearls and Pitfalls](https://pmc.ncbi.nlm.nih.gov/articles/PMC5811276/)).
- **Scheduling friction**: Case initiation still requires email-based order forms and scheduling a web meeting. Not on-demand.
- **No in-house iterative planning**: Surgeons cannot iterate independently between sessions; each revision requires scheduling.
- **Timing constraints**: CT scans must be done at the right time relative to orthodontic treatment; late changes in dental position cannot easily be accommodated.
- **Engineer quality variability**: Community complaints (Reddit/jaw surgery forums) about planning quality when less experienced engineers handle cases without adequate surgeon supervision ([Reddit: VSP quality discussion](https://www.reddit.com/r/jawsurgery/comments/1c7dgdo/the_vsp_scam_how_vsp_companies_use_offshored/)).
- **Physical deliverable dependency**: The workflow is tightly coupled to 3D-printed physical guides. Digital plan portability for navigation systems requires the optional "Augmented DICOM" output, which is not the default path.
- **VSP Orthognathics limitation**: Does not easily handle highly complex asymmetric cases where intraoperative soft tissue behavior cannot be predicted from virtual planning.

---

## 3. Materialise CMF / Mimics

**Sources:** [ProPlan CMF product page](https://www.materialise.com/en/healthcare/proplan-cmf) | [Mimics Enlight CMF product page](https://www.materialise.com/en/healthcare/mimics/mimics-enlight-cmf) | [Mimics Viewer](https://www.materialise.com/en/healthcare/mimics/mimics-viewer) | [VA OIT TRM ProPlan CMF PDF](https://www.oit.va.gov/Services/TRM/files/SynthesProPlanCMF.pdf) | [New generation VSP article](https://www.materialise.com/en/inspiration/articles/new-generation-virtual-surgical-planning-mimics-enlight-cmf)

### Product Ecosystem

Materialise operates two related CMF planning tools that serve different use cases:

1. **ProPlan CMF** — The original, service-oriented planning tool used in the outsourced Materialise SurgiCase workflow. Planning sessions are conducted by Materialise engineers; surgeons review and approve the plan via the Materialise SurgiCase web portal.
2. **Mimics Enlight CMF** (formerly "CMF Planner") — The newer in-house planning software released as the independent, surgeon-operated planning tool. Launched publicly in 2024. Intended to replace ProPlan CMF for in-house users.

There is also the broader **Mimics Innovation Suite** ecosystem: Mimics (segmentation), 3-matic (STL editing and guide design), Materialise Mimics Flow (web-based case management), and Mimics Viewer (collaborative web viewing).

### Mimics Enlight CMF — Interface Architecture

The interface follows a **step-by-step wizard** model:

**Default layout:**
- Four viewports (configurable): axial, coronal, sagittal 2D slice views + one 3D rendering window
- Left panel: task/workflow wizard panel with sequential steps
- Top: toolbar with menus
- Right/floating: object list (bone segments, implants, landmarks)
- Viewport toolbar per window

**Workflow for orthognathic surgery (v7.0):**
1. Import DICOM from CT/CBCT
2. Request automated AI segmentation (select "CMF Orthognathic" algorithm — cloud-based, returns result within minutes)
3. Sync segmentation result, review 3D models
4. Build composite model (fuse CT bone with intraoral scan/occlusal cast via point-based registration)
5. Trace mandibular nerve (point placement along nerve canal)
6. Set natural head position (define Frankfurt horizontal: 4 ear canal points + 2 orbital floor points)
7. Perform cephalometric analysis (automated landmark detection, produces numeric output and visual overlay)
8. Plan osteotomies (click-based plane placement; right SSO → contralateral SSO; LeFort I cut)
9. Place and adjust occlusion (STL cast import, 3-point registration for mandible/maxilla)
10. Reposition bone fragments to planned position (translate/rotate with constraint checking)
11. Design splint (set parameters: overlap, wiring holes, bevels — produces STL)
12. Export (STL models, CSV cephalometry data, NHP XML, PowerPoint report, before/after comparison)

**Full orthognathic planning time: under 25 minutes** for experienced users (per Mimics documentation and surgeon testimonials).

**Reconstruction workflow:**
- Segmentation of skull + mandible + soft tissue + fibula/tibia/vessels
- Mirroring healthy side as reconstruction template
- Osteotomy planning on mandible (manual point-placement for cut planes)
- Fibula segment creation and positioning (automated optimal sequence calculation)
- Export cutting planes to 3-matic for guide design

### ProPlan CMF — Interface Architecture

ProPlan CMF is older (Windows Vista/7/8/10 support indicates legacy codebase). The interface uses:
- Four-window default viewport (2D + 3D mixed)
- Left wizard panel with sequential steps
- Object list for managing bone segments
- Osteotomy Wizard: 4-step flow (draw cut plan → adjust plane → perform osteotomy → reposition)
- Orthognathics Wizard: 5-step flow (import dental cast → plan osteotomies → register bone fragments to occlusion → reposition → design splints)
- Import from SurgiCase files, STL, and proprietary .mdck format

**Key difference from Enlight CMF**: ProPlan CMF requires using Mimics separately for rendering and segmentation first, then importing into ProPlan for planning. This two-application split is a known friction point.

### Segmentation Approach

- **Threshold-based** (set HU range, e.g., 230–3,060 HU for bone) as baseline
- **Erase and split tools** for manual cleanup
- **Automated AI segmentation** (Mimics Enlight CMF v7.0, cloud-based): cloud-runs and returns a result; the operator spends ~39 seconds of hands-on time, compared to traditional manual segmentation taking 25+ minutes ([PubMed AI segmentation study](https://pubmed.ncbi.nlm.nih.gov/41648684/))
- Automatic mandible/maxilla separation
- Nerve canal tracing via point placement along the inferior alveolar nerve

### Surgical Guide Design

- Planning output is exported to **3-matic** (Materialise's STL editing tool) for guide design
- Surgeons set cutting planes in Enlight CMF → export STL → design guide in 3-matic → print
- Guides can also be ordered from Materialise's manufacturing service

### Case Management (Mimics Flow)

Mimics Flow is a **web-based case management portal**:
- DICOM upload initiates a case; Materialise responds within 24 hours to schedule a planning session
- Surgeon reviews the plan and personalized guide/implant designs in Mimics SurgiCase for approval
- Delivery after approval: 1–2 weeks
- Also enables in-house users to share cases and access AI-enabled tools

### Mimics Viewer (Collaborative Web Viewer)

A zero-install web-based viewer for sharing cases:
- Interactive 3D model visualization (no Mimics license required for recipients)
- Simple measurements
- "Fly-through" mode and section view for evaluating complex anatomy
- Contour overlay of 3D model on 2D slice images
- XR (AR/VR) enabled for extended reality viewing
- Secure messaging between collaborators
- Comparison of pre- and post-planned states

### Pricing

From published academic data ([PMC feasibility/cost study](https://pmc.ncbi.nlm.nih.gov/articles/PMC7790928/)):

| ProPlan CMF tier | Cost |
|---|---|
| Dysgnathia local (1 user, 1 year) | ~$8,412/year |
| Dysgnathia floating (6 users, 1 year) | ~$12,617/year |
| Trial (1 user, 14 days) | Free |
| No initial training included | Training purchased separately |

Mimics Enlight CMF pricing is not publicly listed; contact required. Mimics Fundamentals training is $440/course. Annual license/maintenance model applies.

### Key UX Strengths

- Step-by-step wizard keeps surgeons from getting lost in complex menus
- Automated segmentation eliminates the biggest time bottleneck
- Real-time soft tissue simulation validated against CBCT outcomes (RMS ~1.2 mm for ProPlan)
- Export to PowerPoint report for case documentation
- Before/after comparison view built into export workflow
- Mimics Viewer enables asynchronous surgeon review without software installation

### Key UX Weaknesses

- ProPlan CMF requires Mimics as a prerequisite (two-app dependency)
- ProPlan CMF planning takes longer (mean 45.5 min vs. 33.6 min for Dolphin in head-to-head study)
- Crash events during planning sessions are documented (PC reboots needed)
- Legacy operating system support (Vista, Win7) suggests aging codebase in ProPlan
- No soft tissue parameters are adjustable in ProPlan (unlike finite element methods)
- Steep initial learning curve; hardware requirements are substantial for optimal performance

---

## 4. Brainlab CMF

**Sources:** [Brainlab CMF Planning page](https://www.brainlab.com/surgery-products/digital-cmf-surgery/cmf-planning/) | [CMF Navigation page](https://www.brainlab.com/surgery-products/digital-cmf-surgery/cmf-navigation/) | [Digital CMF Surgery overview](https://www.brainlab.com/surgery-products/digital-cmf-surgery/)

### Positioning and Differentiation

Brainlab is the only major CMF planning vendor that fully integrates preoperative planning with **intraoperative navigation** and **intraoperative imaging** in a single ecosystem. While 3D Systems and Materialise are primarily preoperative planning tools with physical guide outputs, Brainlab extends into:
- Real-time surgical navigation during the case (tracking instruments, bone fragments, implants)
- Intraoperative CT/CBCT acquisition and automatic registration (Loop-X Mobile Imaging Robot)
- Mixed reality visualization (via Magic Leap 2 headset)
- Integrated OR environment (Brainlab Node, unified display control)

This makes Brainlab a fundamentally different product than the others — it's an **intraoperative platform** that includes preoperative planning, rather than a preoperative planning tool that optionally exports navigation data.

### Elements Software Architecture

Brainlab operates on a **modular application framework** called Elements. Elements are individual software applications that snap into a unified workspace. Users license and install the modules they need:

| Element Module | CMF Function |
|---|---|
| Elements Viewer | DICOM import; axial/coronal/sagittal/3D views; auto-alignment by symmetry planes |
| Elements Image Fusion | Multi-modal co-registration (MR+CT, CT+CBCT); fully automated with mutual information algorithm; ROI-based registration |
| Elements 3D Delineation | Manual/semi-auto object outlining |
| Elements Automatic Segmentation | Atlas-based anatomy segmentation; deep learning-based tumor segmentation modules available |
| Elements Virtual Reconstruction | Mirror healthy side; virtual implant positioning; bone fragment repositioning |
| STL Import & Export | Import external implant/guide STL files; export for manufacturing |
| Elements SmartBrush (cranial) | Semi-automatic multi-modal tumor outlining |

**Navigation-specific modules:**
- Automatic Image Registration (AIR) — intraoperative CT/CBCT auto-registers to preop plan
- Replay — pointer-based verification of implant/fragment position vs. planned position intraoperatively
- Surface matching / Landmark registration — patient-to-image registration in the OR
- CMF Navigation software — full intraoperative instrument tracking

### Interface Patterns

- Multi-planar view with axial/coronal/sagittal/3D in configurable layout
- Symmetry plane auto-alignment for direct left-right comparison in one view (unique to Brainlab)
- Drag-and-drop layout configuration for intraoperative display
- "Peel" tool — layer-by-layer visualization from skin to bone to preoperative plan structures
- Mixed Reality: plan loaded into Magic Leap 2 headset; room scanned for spatial registration; surgeons walk around holographic anatomy

### Workflow Differences from Competitors

- **Planning is done by the surgical team using Elements software**, not an outsourced engineer
- All data transfers automatically from preoperative planning to the navigation system (no re-import needed)
- **Intraoperative position verification**: Surgeon traces implant/fragment surface with Replay pointer; system shows real-time deviation from plan (mm-accurate)
- **Quality control in OR**: intraoperative CT/CBCT fused automatically with preop plan — instant deviation map
- OR time savings cited at ~15% since adopting Brainlab integrated OR ([Brainlab OR integration page](https://www.brainlab.com/digital-o-r/))
- Mixed reality allows surgeons to "dive inside" the anatomy in 3D before incision, stand next to the hologram, walk around it — fundamentally different from screen-based review

### Limitations

- Much higher price point than Materialise — involves both software licensing and specialized hardware (navigation system, potentially Loop-X, Magic Leap headsets)
- Mixed reality requires Magic Leap 2 headset and specific setup; not standard OR equipment
- Primarily focused on **midface reconstruction and tumor resection**; orthognathic surgery coverage is less emphasized than Materialise's suite
- Elements modules must be individually licensed; full-featured CMF capability requires purchasing multiple Elements
- Atlas-based segmentation may be less accurate than Mimics' threshold-based + AI hybrid for CMF bony structures

---

## 5. Dolphin Imaging / Nemoceph

### 5a. Dolphin Imaging

**Sources:** [Dolphin 3D Surgery module page](https://dolphinimaging.it/products/3d/3d-surgery/) | [Dolphin 3D feature sheet PDF](https://www.dolphinimaging.com/Areas/Product/Documents/3D/Imaging119_3D.pdf) | [Products & Services Guide PDF](https://www.dolphinimaging.com/Areas/Media/Documents/20240322_Branch%20Kit.pdf) | [Dolphin Imaging 11.9 what's new PDF](https://www.dolphinimaging.com/Areas/Product/Documents/Imaging/Imaging119_WhatsNew.pdf)

#### Origins and Market Position

Dolphin Imaging (Patterson Dental Supply) originated as an **orthodontic practice management system** with 2D cephalometric analysis, and grew 3D surgical planning capabilities as an add-on. This heritage shows: the product is deeply integrated with dental practice workflows (patient management, scheduling, communication) but the surgical planning tools are not as CMF-specialized as ProPlan CMF or Brainlab.

Dolphin 3D is one of the three most commonly cited VSP tools in the literature, particularly in North American orthodontic/oral surgery practices ([Journal of Clinical Medicine survey](https://pmc.ncbi.nlm.nih.gov/articles/PMC12841811/)).

#### Interface Architecture

Dolphin 3D's main viewer provides:
- **Multiple layout views**: 4-pane equal-sized view (axial, sagittal, coronal + 3D), custom layouts, or fullscreen
- Synchronized scrolling across planes
- Volume orientation with yaw/pitch/roll controls
- **3D rendering modes**: bone (hard tissue) surface, soft tissue surface, translucent overlay, photo-wrap (2D or 3D photo wrapped onto 3D volume)
- Color tools for tissue differentiation
- Clipping tools for hidden structure reveal
- Window Level adjustment (12/14/16-bit grayscale intensity traversal)
- Hounsfield Unit measurement tool for tissue density

**3D Surgery Module Interface:**
- **Step-by-step wizard** guiding from initial evaluation through guide design
- DICOM import from cone beam CT, spiral CT
- Optional augmentation: 2D/3D facial photo, laser-scanned stone models, intraoral scan (.STL/.OBJ)
- Segmentation: threshold-based grouping into soft/hard tissue intensity ranges
- Osteotomy design and simulation (LeFort I, BSSO, genioplasty)
- Real-time soft tissue simulation during each bony movement
- Slice view of surgical simulation in the "Present" tab
- Intermediate and final splint design (output as .STL)
- Animation export: automated movie scripts for patient education (spin, fly-by, see-through)

**Notable UX features:**
- **AnywhereDolphin**: cloud-based case sharing; recipients can open patients via free viewer without Dolphin installed
- TMJ analysis module
- 3D nerve marking (patented): interactive panoramic-based nerve canal location with 3D rendering overlay
- Airway analysis: volume measurement, color-coded constriction map along curved path, Hounsfield-based threshold control
- Volume stitching: combine two separate CBCT volumes into one
- Ceph Tracing with ~400 built-in analyses; automated AI landmark placement in latest versions
- Symmetry ruler overlay for ceph tracing
- Three surgical cephalometric analyses added in v11.9

**Workflow comparison:** In head-to-head studies, Dolphin required a mean 33.6 min for midfacial planning vs. ProPlan CMF's 45.5 min — though ProPlan offers more specialized osteotomy types (LeFort II) that Dolphin lacks ([PMC feasibility study](https://pmc.ncbi.nlm.nih.gov/articles/PMC7790928/)).

#### Pricing

| Tier | Asset Cost | License | Annual maintenance |
|---|---|---|---|
| Business (1 user, perpetual) | ~$25,800 | Perpetual | $3,500/yr (optional but required for updates) |
| Business+ (server + extra users) | ~$25,800 + $1,900 server + $3,450/additional user | Perpetual | $3,500/yr |
| Academic (3 users, annual) | ~$25,800 | Annual | $3,500/yr |

Cloud subscription model also available (pricing by contact). Patterson dental equipment rebate programs apply to Dolphin software purchases.

#### Limitations

- **Roots in practice management**: UI carries legacy workflow patterns designed for orthodontists tracking patients, not purely optimized for surgical planning
- Soft tissue prediction uses landmark-based algorithm that works well for 2D cephalometric input but has limited 3D accuracy compared to FEM-based tools; prediction errors are higher for upper lip and subnasal regions
- No LeFort II osteotomy type built in (requires manual modification of cut planes)
- 4 documented software crash events during one study period
- Smoothing and mesh reduction parameters are not user-controllable in segmentation, limiting post-processing flexibility

### 5b. Nemoceph / NemoStudio

**Sources:** [NemoCeph product page](https://nemotec.com/nemostudio/en/products/nemoceph/) | [NemoStudio v25.0 release notes](https://nemotec.com/nemostudio/en/updates-in-nemostudio/latest-updates-and-improvements-in-v-25-0-0-0/)

#### Overview

Nemotec (Spain) produces the **NemoStudio** platform, a cloud-native, modular orthodontic and surgical planning suite. The cephalometric core is **NemoCeph**, and the surgical planning component is **NemoFAB**. NemoStudio v25 (2025) is a full web/cloud application with no local installation required.

#### Interface Architecture

NemoStudio v25 UX (2025, significantly redesigned):
- **Workspace redesign**: patient search bar with name chip + vertical menu icon; single-page patient creation
- **Document management**: filter by type, modification date, series group; icon or list view with configurable icon size; drag-and-drop ordering
- **AI-powered record detection**: automatically classifies photos and X-rays without user-defined templates
- **Patient-level chat**: in-platform messaging between team members without leaving the application
- **Document-specific chat**: comments tied to individual planning files
- **Inactive session control**: auto-logout after 15 minutes inactivity, resume exactly where left off
- **License control**: each user has exclusive session; cannot share credentials

**NemoCeph capabilities:**
- 2D/3D cephalometric analysis with automatic AI landmark placement
- Cephalometric tracing in minutes via wizard
- Lateral and frontal morphing (VTO, STO, morphing)
- Before/after photo overlays with treatment projections
- Multiple built-in analyses (400+), including Arnett cephalometric analysis
- New 3D cephalometric analysis improvements in v25

**NemoFAB / NemoScan (guided surgery planning):**
- Now web-based: plan guided surgery cases directly in browser with 3D view and MPR slices
- Implant, pin, and abutment position adjustment in browser
- Improved stackable guide workflow
- Virtual extractions with gingival mesh deformation
- 3D mesh visualization with realistic textures
- Micro-screw visualization in 2D slices

#### Key UX Differentiators

- **Fully cloud-native** in 2025 (no local software to install) — ahead of all major competitors
- Patient chat embedded in the planning workflow
- AI record classification eliminates manual document sorting
- Used primarily in orthodontic planning and 2D orthognathic prediction rather than full 3D VSP (NemoFAB handles surgical guides)
- Comparable accuracy to Dolphin 3D in planning outcomes (NemoFAB vs. Dolphin head-to-head: no statistically significant difference in translational/rotational accuracy)

#### Limitations

- NemoFAB 3D is a relatively smaller player for full CMF VSP vs. ProPlan CMF or Dolphin
- More orthodontic-centric; less used in complex CMF reconstruction cases
- Cloud dependency may be concern for high-data DICOM workflows

---

## 6. Blue Sky Plan / InVivo Dental

### 6a. Blue Sky Plan

**Sources:** [Blue Sky Bio guided surgery page](https://blueskybio.com/pages/blue-sky-plan-guided-surgery-software) | [Blue Sky Plan PDF user manual](https://blueskybio.com/caffeine/uploads/files/documents/Blue%20Sky%20Bio%20Plan%20User%20Manual%20Rev%2010.pdf) | [AI implant planning blog](https://www.blueskybio.digital/post/ai-implant-planning-blueskyplan) | [Step-by-step surgical guide PDF](https://blueskybio.com/caffeine/uploads/files/digital-designed-surgical-guide-step-by-step-protocol.pdf)

#### Overview

Blue Sky Plan is a **free-to-download** Windows/Mac CBCT planning and surgical guide design application from Blue Sky Bio, focused on dental implant placement. It overlaps with CMF for cases involving orthognathic surgery with dental component, implant-retained prosthetics post-reconstruction, and basic surgical guide design. It is not an FDA-cleared Class II medical device in the same way as dedicated CMF planners.

#### Interface Architecture

The main interface has:
- **Implant screen** as primary workspace — shows cross-sectional, tangential, axial, panoramic, and 3D views simultaneously
- Main toolbar and panoramic toolbar: most-used tools visible without diving into menus
- **View layout**: 5 cross-section images + panoramic + 3D view; color-coded cross-reference lines in axial/panoramic that update as cross-sections move
- Adjustable inter-image distance (default 1mm)
- All views synchronized — scrolling one updates the others

**Key UI elements:**
- Cross-sectional view for evaluating bone height and thickness perpendicular to arch
- Tangential view: rotatable ~360° around implant axis
- 3D volume view: rotate, zoom via right-click drag
- Arch spline drawing: user traces maxillary or mandibular arch to set panoramic reference
- Safety zone circles around each implant (user-defined clearance radius)

**Implant workflow:**
1. Load DICOM (CT/CBCT)
2. Draw arch spline (determines cross-section positions)
3. Select implant from catalog (~80+ systems) or define custom implant
4. Click to place implant in 3D view; appears perpendicular to selected surface
5. Fine-tune position with widget (orange rotation arrows; straight translation arrows)
6. Define safety zone, check nerve clearance in all views
7. Lock implants, select input model (optical scan or CT-based model)
8. Fabricate surgical guide: use automatic brush or manual; set guide tube diameter
9. Export as STL for printing/milling

**AI capabilities (2025+):**
- Automatic mandibular nerve detection and highlighting in bright color
- AI tooth generation for missing teeth (places implant and generates AI virtual tooth automatically)
- Automatic model scan registration (optical scan to CT alignment without manual point selection)
- Under 10 minutes total planning time with AI assistance cited

**Cost:** The software itself is free. Surgical guide fabrication: ~$25 in materials if printed in-house; alternatively order guides from Blue Sky Bio labs.

#### UX Strengths

- **Price**: Free software eliminates the biggest barrier to in-house planning
- AI automates the most tedious step (nerve detection, model registration)
- Straightforward cross-section/panoramic/3D layout is immediately familiar to anyone who has used a CBCT viewer
- In-house guide printing capability

#### Limitations

- Primarily dental implant-focused; lacks the osteotomy simulation, cephalometric analysis, fibula reconstruction, and cranial planning features of CMF-dedicated tools
- No soft tissue simulation
- No multi-bone repositioning workflow for orthognathic planning
- Limited relevance to CMF beyond the dental/implant overlap cases

### 6b. InVivo Dental (Anatomage / DEXIS)

**Sources:** [DEXIS Invivo 6 page](https://dexis.com/en-us/software-invivo-6) | [Anatomage InVivo 5.4.5 tutorials](https://www.youtube.com/watch?v=vOjIn-lmJGk) | [PMC DICOM viewer study](https://pmc.ncbi.nlm.nih.gov/articles/PMC4570900/)

#### Overview

InVivo (formerly InVivo Dental, now Invivo 6 powered by Anatomage, distributed by DEXIS) is a CBCT 3D imaging platform for dental implants, orthodontics, oral surgery, and restorative dentistry. It is a Class II FDA-cleared device. More powerful than Blue Sky Plan for visualization; still primarily dental rather than full CMF.

#### Interface Architecture

- **4-window primary layout**: axial, sagittal, coronal slices + 3D volume rendering
- Volume rendering modes: bone, soft tissue, full volume, custom transfer function
- Clipping planes: axial/sagittal/coronal draggable cursor lines intersect 3D volume
- **ArchSection tab**: arch spline drawing generates panoramic view + cross-sections
- SuperPano tab: 2D and 3D panoramic views
- **Endo tab**: dedicated internal canal morphology visualization
- **Airway tab**: dedicated airway analysis; minimum area/volume/AP/RL measurements; fly-through simulation; sculpting tool for unwanted anatomy removal; automatic export of data, summary, 3D model, graphs
- **Implant tab**: single and multiple implant placement; 3D implant view + arch section view; angle dialog showing summary of angles between all implants
- Control panel: volume rendering presets (bone, soft tissue), clipping options, implant profile/trajectory/force vector overlays
- Patient info stamp in corner (name, DOB, gender)
- Grid overlay for distance measurement in cross-section view

**Invivo Workspace (cloud viewer):**
- HIPAA-compliant universal medical image sharing
- FDA-cleared 3D viewer accessible via web browser (no software install)
- Zero-footprint: designed for simple secure case sharing and discussion
- Measurement tools in web viewer
- Secure messaging

#### CMF-Relevant Capabilities

- Surgical guide design (though primarily dental)
- STL import (digital impression registration with CT)
- Digital impression + CT registration: automatic (no manual point selection)
- Crown and abutment design from crown-down within CBCT

#### Limitations

- No osteotomy simulation or bone repositioning for orthognathic planning
- No cephalometric analysis module
- No soft tissue prediction
- Focused on dental anatomy; CMF reconstruction is outside its workflow scope

---

## 7. Open-Source Alternatives

### 7a. 3D Slicer

**Sources:** [SlicerCMF site](https://cmf.slicer.org) | [SlicerCMF GitHub](https://github.com/DCBIA-OrthoLab/SlicerCMF) | [BoneReconstructionPlanner GitHub](https://github.com/SlicerIGT/SlicerBoneReconstructionPlanner) | [3D Slicer community custom surgical guides discussion](https://discourse.slicer.org/t/custom-fitting-surgical-guides-on-slicer/14270) | [3D Slicer vs ProPlan CMF vs Mimics comparison study](https://www.oaepublish.com/articles/2347-9264.2025.01)

#### Overview

3D Slicer is a free, open-source platform for medical image visualization and analysis, developed by Harvard/NIH. It is not CMF-specific but is widely used in research contexts. Its module architecture allows domain-specific extensions.

**Key CMF-relevant modules in base 3D Slicer:**
- **Segment Editor**: Semi-automated and manual segmentation tools (threshold, region growing, paint, draw, scissors, fill between slices, grow from seeds)
- **Dynamic Modeler**: Boolean operations on segmented meshes — cutting, combining, clipping (enables guide design within Slicer)
- **Transforms**: Rigid body transforms for simulating bone movements
- **Fiducials**: Landmark placement for measurements
- Python scripting API: automation of entire workflows

#### SlicerCMF Extension

SlicerCMF (developed by DCBIA lab, Harvard School of Dental Medicine) packages CMF-specific analysis tools:

| Module | Function |
|---|---|
| Q3DC (Quantitative 3D Cephalometrics) | Place fiducials; compute 2D angles (Yaw/Pitch/Roll); 3D distance (R-L, A-P, S-I components); midpoint computation; export values |
| ShapePopulationViewer | Compare multiple 3D surfaces simultaneously; pointwise scalar/vector map overlays; customizable colormaps |
| ShapeVariationAnalyzer | DL-based shape classification; compute average shapes; train/classify models |
| ModelToModelDistance | Point-by-point signed/unsigned distance between surfaces (outcome assessment) |
| AnglePlanes | Calculate angles between user-defined or existing planes |
| MeshStatistics | Descriptive statistics over ROIs (Pick and Paint selection) |
| PickAndPaint | ROI selection by landmark + vertex radius; propagates to multiple models |
| EasyClip | Clip models along saved planes |
| CMFreg | Region-based surface registration |
| MeshToLabelMap | Converts surface mesh to binary segmentation volume |
| DatabaseInteractor | Connect to web database for managing research datasets |

#### BoneReconstructionPlanner

A dedicated Slicer extension for **mandibular reconstruction with vascularized fibula free flap** ([GitHub: SlicerIGT/SlicerBoneReconstructionPlanner](https://github.com/SlicerIGT/SlicerBoneReconstructionPlanner)):

**Workflow:**
1. Import mandible and fibula segmentations (or segment first in Segment Editor)
2. Create bone models
3. Place mandibular curve (traces healthy anatomy for reference)
4. Add mandibular cut planes (osteotomy points)
5. Create fibula segments (point-placement for each fibula cut)
6. Adjust segment positions to match healthy anatomy target
7. Generate patient-specific surgical guides

#### Accuracy vs. Commercial Tools

Head-to-head study comparing 3D Slicer, ProPlan CMF, and Mimics ([OAE Publishing, 2025](https://www.oaepublish.com/articles/2347-9264.2025.01)):
- All three tools show model overlap rates of ~99% (no statistically significant difference in accuracy)
- Feature size measurements showed no significant differences (P > 0.05)
- 3D Slicer: 11.7 clicks for modeling, 34.4 clicks for virtual osteotomy (46.1 total)
- ProPlan CMF: 13.3 clicks modeling, 36.4 clicks osteotomy (49.7 total)
- 3D Slicer excels in flexibility; ProPlan CMF in CMF-specific detail processing; Mimics in segmentation precision

**Planning time comparison (VR vs. desktop):** Research shows segmentation in VR environment (Elucis software) has ~2x faster speed and steeper learning curve than traditional 2D desktop (3D Slicer). VR was rated more intuitive and less exhausting ([JMIR Serious Games, 2023](https://games.jmir.org/2023/1/e40541/)).

#### Interface Challenges

3D Slicer's interface is its primary weakness for clinical adoption:
- Designed for research, not clinical workflows — module-based paradigm requires knowing which modules exist and what they do
- Steep learning curve for non-technical users
- No wizard-guided workflow — users must manually sequence steps
- Module switching interrupts cognitive flow
- Not FDA-cleared for clinical surgical planning (though used off-label clinically in some institutions)
- Lacks the built-in osteotomy wizards of commercial tools

#### Open-Source Advantage

- **Free**: The only zero-cost option for VSP
- **Extensible**: Python scripting for custom workflows; researchers can build automation
- **Community**: Active Slicer forum and CMF community support
- **Accuracy**: Statistically equivalent to commercial tools
- Suitable for training/education and resource-limited institutions

---

## 8. Cross-Platform UX Pattern Analysis

### Interface Layout Conventions

All major CMF planning tools converge on a consistent layout paradigm:

```
┌─────────────────────────────────────────────────────┐
│  Menu bar + main toolbar (top)                       │
├──────────┬──────────┬──────────┬─────────────────────┤
│          │          │          │  Object/segment list │
│  Axial   │ Coronal  │ Sagittal │  (or wizard steps)  │
│  slice   │ slice    │ slice    │                     │
│          │          │          │  Properties panel   │
├──────────┴──────────┴──────────┤                     │
│                                 │                     │
│      3D Rendering Window        │  Measurements/      │
│    (dominant, manipulable)      │  Ceph values        │
│                                 │                     │
└─────────────────────────────────┴─────────────────────┘
```

**Standard elements:**
- **4-pane layout**: 3 orthogonal slice views + 1 3D view (universally adopted)
- **Resizable panes**: drag borders to focus on area of interest
- **Synchronized cross-reference lines**: colored lines across all views show where other planes intersect; clicking one updates others
- **Viewport toolbar**: zoom, pan, rotate, window/level, screenshot, per-window display controls
- **Object list**: hierarchy of bone segments, implants, landmarks, planes with show/hide toggles
- **Step wizard panel** (left or right): sequential steps with Next/Back navigation

### 3D Viewer Conventions

- **Navigation model**: Left-click drag = rotate; Middle-click drag or Shift+drag = pan; Scroll wheel = zoom
- **Pre-set views**: standard anatomical views (anterior, posterior, right/left lateral, superior, inferior) accessible via named buttons or keyboard shortcuts
- **Transparency/opacity slider**: blend between bone surface and soft tissue surface; allow simultaneous visualization
- **Clipping tools**: drag a plane to cut through the volume and reveal internal structures
- **Rendering modes**: volume rendering (shows all densities), surface rendering (bone mesh), soft tissue surface, combined/translucent overlays
- **Color differentiation**: different bone segments colored differently (e.g., maxilla = blue, mandible = yellow, proximal segment = green)

### Measurement Tools

Consistently present across platforms:
- **Linear distance** (ruler tool): click two points, display mm
- **Angle measurement**: three-point (vertex + two arms) or plane-to-plane
- **3D cephalometric analysis**: landmark placement (usually ~30–80 points) → automated computation of standard analyses (Steiner, Ricketts, Arnett, McNamara, etc.)
- **Deviation mapping**: point-by-point color distance map overlaid on surface (heat map: blue=close, red=far) for comparing pre/post or planned/actual
- **Volume measurement**: for bone grafts, sinus cavities, airway
- **Hounsfield Unit measurement**: click to query tissue density

### Annotation Patterns

- **Fiducial markers** (spherical point icons) with labels: for identifying anatomical landmarks (nasion, porion, orbitale, etc.)
- **Measurement overlays**: annotations attached to 3D model or to 2D slice
- **Named cutting planes**: each osteotomy plane has a name and visual indicator (colored translucent plane with normal arrow)
- **Numbered landmark IDs**: cephalometric landmarks numbered per analysis protocol
- **Surgical notes**: text field attached to the planning session (mostly in outsourced workflows); Brainlab and NemoStudio add messaging/chat to the document

### Comparison Views

- **Side-by-side**: preop vs. planned (two separate 3D views or image panels)
- **Overlay with transparency slider**: preop model at X% opacity overlaid on planned model
- **Color deviation map**: deviation between two time points shown as surface heat map
- **Cephalometric overlay**: preop and planned ceph tracings overlaid on the same radiograph
- **IPS CaseDesigner v2.3** added explicit comparison of preoperative/planned/postoperative datasets with ceph values visible on the model
- **Mimics Enlight CMF**: export includes before/after comparison in the PowerPoint report

### Before/After Overlays

- Materialise Viewer: "contour overlay of the 3D model on 2D images" — places 3D mesh outline over 2D slice
- ProPlan CMF and Dolphin: export pre/post soft tissue renders side-by-side
- Nemoceph: lateral/frontal morphing with VTO and before/after photos
- Most platforms support export of pre-planned and post-planned models as separate STL files; comparison done in post-processing software

### Confidence and Uncertainty Displays

**Current state (limited):**
- Virtually no VSP tool surfaces explicit uncertainty/confidence information to surgeons
- Soft tissue simulation accuracy is referenced in product literature but not displayed as uncertainty bounds during planning
- No tool currently shows "this predicted soft tissue position has ±X mm confidence at 95%"

**Research direction:**
- Surgical probe-centric uncertainty visualization (text mode showing mm uncertainty at instrument tip) was preferred by neurosurgeons in studies ([PMC uncertainty visualization](https://pmc.ncbi.nlm.nih.gov/articles/PMC12239872/))
- Color overlay uncertainty maps near critical boundaries (tumor boundaries, osteotomy zones)
- Boundary uncertainty "error bars" (min/max offset surfaces around the planned structure)

---

## 9. What Is Outdated About Current Tools

### Structural Problems

**1. Binary service vs. software model**
The outsourced model (3D Systems VSP, Materialise ProPlan service) places the surgeon in a passive review role. Planning quality depends on engineer quality and communication effectiveness. 15% of orthognathic plans required partial modification, and 4% were abandoned due primarily to communication failures and engineering errors ([PMC: VSP Pearls and Pitfalls](https://pmc.ncbi.nlm.nih.gov/articles/PMC5811276/)). Surgeon autonomy is sacrificed for convenience.

**2. Desktop-only, installation-required software**
All major tools (except NemoStudio v25 and Materialise Viewer) require local Windows installation. DICOM files are transferred by CD, USB, or zipped email upload. VSP Connect and Mimics Flow are steps forward but are accessory portals rather than the planning environment itself.

**3. Segmentation as a bottleneck**
Manual segmentation is the recognized #1 time bottleneck in VSP workflows. Studies show it can be addressed by AI (reducing operator time to ~39 seconds for automated result) but most commercial tools only added AI segmentation recently, and it's often a separate cloud step rather than seamlessly integrated. 3D Slicer still relies heavily on manual threshold editing and cleanup.

**4. Two-app dependencies**
ProPlan CMF requires Mimics for segmentation first. Guide design requires exporting to 3-matic. Orthognathic planning in 3D Slicer requires multiple sequential modules that don't communicate in a unified UI. BoneReconstructionPlanner is separate from standard SlicerCMF. Tool fragmentation adds cognitive overhead and file management complexity.

**5. Paper-trail workflow vestiges**
File upload via zip + email order form (3D Systems). DICOM management by physically distributing discs. Case status communicated by phone or email. VSP Connect is starting to address this but the underlying infrastructure remains email-dependent in many practices.

**6. No intelligent suggestions or plan checking**
No current tool proactively detects that a planned osteotomy is dangerously close to a tooth root, proposes alternative cut positions, or flags that the planned bone movement will create a bony collision not visible in the current view. All safety checking is manual and dependent on surgeon vigilance.

**7. Soft tissue simulation is black-box and untunable**
ProPlan CMF's soft tissue prediction is fixed-ratio based; surgeons cannot adjust patient-specific soft tissue tension, BMI, or scarring parameters. Dolphin uses landmark-based ratios with known inaccuracies in upper lip and subnasal regions. No tool provides probabilistic bounds on the predicted soft tissue outcome.

**8. Learning curve and interface complexity**
The traditional desktop surgical planning environment (all tools visible, all modes accessible, all panels open) overwhelms new users. Studies document steep learning curves. Research shows VR environments have twice the learning curve speed of 2D desktop environments for segmentation tasks — suggesting spatial 3D interaction is the natural modality for 3D surgical planning, not mouse+keyboard on a flat screen.

**9. No population-level intelligence**
Tools plan each case in isolation. No tool currently surfaces: "For your planned maxillary advancement of 8mm, 94% of surgeons had successful outcomes; the 6% who needed revision had a common pattern of insufficient posterior impaction — you may want to review your impaction." Population data from thousands of prior cases goes unused in the planning interface.

**10. Physical guide dependency**
The workflow is built around physical 3D-printed guides as the transfer mechanism. Intraoperative navigation as an alternative requires purchasing a separate navigation system (Brainlab, Stryker navigation). Most practices do not have intraoperative navigation, making them fully dependent on physical guides that can break, not fit correctly, or need replacement if the case changes.

---

## 10. AI-Native Redesign Recommendations

An AI-native VSP platform for CMF surgery would differ from current tools across four dimensions: **intelligence embedded throughout**, **surgeon as strategic controller not data processor**, **collaborative and asynchronous by design**, and **spatial-first interaction**.

### Principle 1: AI as First-Class Collaborator, Not Optional Add-On

**Current state:** AI is grafted onto existing workflows — an "auto-segment" button that runs in the cloud and returns a result the user reviews. The rest of the workflow is unchanged.

**AI-native approach:**
- **Continuous AI interpretation**: As the surgeon opens a case, AI immediately segments anatomy, identifies key landmarks, notes abnormalities ("significant mandibular asymmetry detected — 4.2mm difference at condyle level"), and proposes an initial planning approach based on the pathology and the surgeon's prior case history
- **AI-proposed plan as starting point**: The surgeon reviews and modifies an AI-generated plan rather than building from scratch. The first thing they see is a proposed osteotomy set, not a blank 3D model
- **Plan quality scoring**: AI analyzes the plan in real-time and provides a confidence score for the predicted outcome, flags potential issues ("osteotomy cut passes within 1.2mm of tooth root #27 — confirm intentional"), and suggests alternatives
- **Pattern recognition from population data**: "This planned movement of 9mm maxillary advancement is at the 91st percentile for your practice; historical relapse rate for advancements >8mm is 23% — consider additional fixation or impaction"

### Principle 2: Segmentation Is Invisible

**Current state:** Segmentation is step 1, requiring manual threshold adjustment, erasing artifacts, separating teeth, tracing nerves, and reviewing quality. Even with AI, it's a discrete "do this, then proceed" step.

**AI-native approach:**
- Segmentation completes automatically the moment DICOM is uploaded; by the time the surgeon opens the case, models are ready
- Uncertainty indicators show where AI is less confident (a red-to-green overlay on the mesh showing segmentation confidence by region)
- One-click correction tools that the AI learns from: "the condyle boundary is wrong" → AI recalculates using that feedback → applies learning across similar future cases
- Nerve canal is automatically highlighted; root proximity to proposed osteotomy planes is always visible without needing to manually trace

### Principle 3: Web-Native, Spatial-First

**Current state:** Desktop-installed software; flat 2D screen with mouse/keyboard; 3D model is one panel among four.

**AI-native approach:**
- **Web-first**: plan in the browser; no installation; works on tablet for case review in clinic or OR anteroom
- **3D as primary**: the 3D model is the default view, not one of four equal panes. 2D slices appear contextually when needed (click a region on the 3D model to see the cross-section at that location)
- **Touch/stylus interaction**: on iPad/tablet, use fingers to rotate the model, pinch to zoom, use Apple Pencil/stylus to draw osteotomy planes directly on the 3D surface
- **XR/spatial computing ready**: design the interface so the same spatial concepts (point, plane, measurement) work when rendered in a VR headset or AR overlay on the patient; don't lock the mental model to a flat screen
- **Collaborative presence**: multiple surgeons can be in the same planning session simultaneously (each sees the other's cursor/interaction), or async with timestamped annotation threads tied to specific model locations

### Principle 4: Surgeon Owns the Decision; AI Handles Computation

**Current state:** The surgeon either delegates everything to an engineer (3D Systems model) or does all computational work themselves (in-house software).

**AI-native approach:**
- **Surgeon specifies intent, AI executes**: "Move the maxilla 5mm superior, 3mm forward, 2mm to the right, correct the yaw asymmetry" → AI computes the result, updates the soft tissue prediction, and shows the outcome. No manual plane manipulation required unless the surgeon wants it.
- **Natural language planning interface**: Surgeons can describe what they want in clinical terms; the interface interprets and asks for confirmation
- **Smart constraints**: If the surgeon moves a bone segment, AI automatically checks for bony collision, tooth root intrusion, neurovascular proximity, and soft tissue tension — in real-time, not as a separate validation step
- **Reversible history**: Every planning step is logged with full undo/redo branching; compare two different plans side by side; version control with named snapshots ("Option A: mandibular setback first" vs. "Option B: maxillary advancement first")

### Principle 5: Soft Tissue Prediction as Probabilistic Range

**Current state:** Single deterministic soft tissue prediction (one predicted outcome shown).

**AI-native approach:**
- Show **probability distribution** of post-surgical facial appearance: a confidence interval envelope (e.g., 80% of similar cases fall within this range of soft tissue positions)
- Highlight anatomical regions with high prediction uncertainty vs. low uncertainty (lips and nasal tip always carry higher uncertainty than cheek soft tissue)
- Allow the surgeon to adjust patient-specific parameters: tissue thickness, BMI category, prior surgery/scarring, to update the prediction
- Show patient photo morphing alongside the 3D prediction for patient counseling

### Principle 6: Case Continuity Across the Surgical Encounter

**Current state:** VSP is preoperative only. The guide goes to the OR as a physical object. There's no live connection between the plan and intraoperative reality.

**AI-native approach:**
- **Intraoperative reference mode**: tablet/iPad in the OR with the plan visible; surgeon can reference landmark positions, planned movements, guide placement zones
- **Post-operative outcome capture**: automate the comparison of post-op CBCT to the original plan; report accuracy metrics (per-landmark deviation, translation vs. rotation errors) for continuous surgeon feedback
- **Feedback loop into AI model**: post-surgical outcomes improve AI predictions for future cases; the system gets better the more it's used
- **Failure mode detection**: if the post-op scan shows the outcome deviated significantly from plan, AI flags this for case review and notes whether the deviation pattern correlates with any planning decision

### Interface Recommendation: Layout for CMF AI-Native Platform

```
┌──────────────────────────────────────────────────────────────┐
│  [Patient Name] [Case ID]    [Plan Version v3] [Share] [AI] │
├──────────────────────────────────────┬───────────────────────┤
│                                      │  AI ASSISTANT PANEL   │
│                                      │  ─────────────────    │
│        3D MODEL (PRIMARY)            │  Plan status: ✓ OK    │
│                                      │  ⚠ Root proximity     │
│   [rotate, pan, zoom]                │    #27: 1.2mm         │
│                                      │  ⚠ Asymmetry: 2.1mm  │
│   [transparency slider]              │    yaw residual       │
│   [hard/soft tissue toggle]          │  ─────────────────    │
│                                      │  MOVEMENT SUMMARY     │
│   [osteotomy planes visible]         │  Maxilla: +5mm sup    │
│                                      │          +3mm ant     │
├──────────────────────────────────────┤  Mandible: -2mm post  │
│  CONTEXTUAL 2D (appears on demand)   │  ─────────────────    │
│  [axial @ selected level]            │  SOFT TISSUE PRED.    │
│  [cross-section @ osteotomy plane]   │  Confidence: 78%      │
│                                      │  [show probability    │
│  CEPH: [live updating values]        │   envelope]           │
│                                      │  ─────────────────    │
│  WORKFLOW STEPS (collapsed by default│  SURGEON HISTORY      │
│  expand on click):                   │  "Similar to 23 prior │
│  [1] ✓ Segmentation                  │   cases; avg outcome  │
│  [2] ✓ NHP alignment                 │   accuracy: 1.4mm"    │
│  [3] → Osteotomy planning            │                       │
│  [4]   Occlusion                     │  [Ask AI anything...] │
│  [5]   Splint design                 │                       │
└──────────────────────────────────────┴───────────────────────┘
```

---

## 11. UX Patterns Specific to Medical/Surgical 3D Planning

### Panel Layouts

**Universal 4-pane**: Used by all tools. The convention is so entrenched that deviating from it requires strong justification. New tools should at minimum support this as a mode, but can offer a 3D-primary layout as the default.

**Wizard-step left panel**: Guides users through sequential workflow; prevents skipping steps; provides progress feedback. Mimics Enlight CMF and IPS CaseDesigner use this most effectively. The wizard should expose constraints ("you must complete segmentation before proceeding to osteotomy planning") while allowing non-linear return to prior steps without starting over.

**Floating/dockable panels**: IPS CaseDesigner v2.3 added an undockable diagnose window for multi-monitor use. This is a power-user feature that CMF surgeons at academic centers expect.

**Context-sensitive panels**: Show only what's relevant to the current step. When planning osteotomies, show the osteotomy tool palette and cutting plane properties; hide splint design controls. NemoStudio implements this pattern well in its web-native 2025 redesign.

### 3D Viewer Conventions That Surgeons Expect

1. **Standard anatomical orientations** accessible by click (not just freeform rotation): anterior, lateral L/R, superior, inferior, oblique 45°
2. **Snap-to-axis rotation**: when rotating, snap to nearest 15° increment (like CAD software)
3. **Persistent view state**: rotating the 3D view does not change the 2D slice planes (some tools couple these, which disorients surgeons)
4. **Measurement persistence**: measurements placed on the model stay visible across sessions; labeled and editable
5. **Multiple selection**: select two or more bone segments to see relative movement, measure distance between them
6. **Symmetry plane**: always-visible anatomical midline as reference (Brainlab auto-computes this; others require manual definition)

### Measurement Tools That Are Critical

| Tool | Surgical relevance |
|---|---|
| Linear ruler (3D) | Planned vs. actual bone movement magnitude |
| Plane-to-plane angle | Canting correction quantification |
| Bone gap/overlap check | Post-osteotomy bony interference detection |
| Nerve canal distance | Safety margin from osteotomy to inferior alveolar nerve |
| Root proximity | Safety margin from osteotomy/screw hole to tooth root |
| Cephalometric landmarks | Standardized analysis (SNA, SNB, ANB, Wits, FMIA, etc.) |
| Soft tissue ratio | Hard-to-soft tissue movement ratios per anatomical region |
| Airway volume + minimum cross-section | For cases involving airway improvement |
| Asymmetry metric | Quantified left-right deviation from midline |

### Annotation Patterns

- **Fiducials with labels**: placed on anatomy, visible in both 2D and 3D; appear as spheres in 3D, crosshair + circle in 2D; labeled with landmark name
- **Osteotomy planes**: semi-transparent planes with colored perimeter and normal vector arrow; label shows plane name and depth relative to reference
- **Measurement callouts**: dimension lines with value; oriented to face the camera (billboard labels)
- **Region-of-interest boxes**: for focusing attention during case review
- **Text annotations tied to 3D location**: click a spot on the model, add a text note; visible in context when viewing that area
- **Surgeon-to-engineer async notes** (VSP Connect, NemoStudio): threaded comments with timestamps; tied to specific case state/version

### Comparison Views

**Overlay fusion (most important for CMF)**:
- Pre/post comparison: use alpha blend slider from 0% (pre only) to 100% (post only)
- Color deviation heat map: best for showing where outcome matches or deviates from plan
- Mirrored view: show planned and actual anatomies in adjacent panels that rotate together (synced camera)
- Ceph overlay: pre-treatment tracing in one color, planned in another, post-surgical in a third — standard orthodontic visualization pattern

**Side-by-side**:
- Two independent 3D viewers linked or unlinked
- Synchronized rotation mode: rotating one viewer rotates the other to the same viewpoint

### Before/After Overlays

- **Soft tissue simulation overlay**: current face rendered with planned post-surgical anatomy; adjust transparency to show "from" and "to" simultaneously
- **Photomorphing**: 2D photograph of patient warped according to predicted soft tissue changes (primarily for patient education; Dolphin and Nemoceph do this well)
- **Time-point comparison slider**: drag a vertical bar across a split-screen to reveal pre vs. post progressively (popular in radiology viewers; could apply to CMF outcome review)

### Confidence/Uncertainty Displays (Current Gap, Future Opportunity)

**Research shows surgeons prefer:**
- Text mode showing uncertainty at instrument tip in mm ("navigation accuracy: ±2.1mm")
- Boundary uncertainty surfaces (inner = safe zone, outer = max possible deviation)
- Color overlay restricted to a region of interest (not full-screen uncertainty map, which is distracting)
- Numerical displays rather than purely color-based
- Ability to toggle confidence display on/off

**CMF-specific confidence displays needed but absent today:**
- Segmentation confidence overlay: shows where AI is uncertain about bone boundaries
- Soft tissue prediction confidence range: visualized as a 3D probability envelope around predicted facial surface
- Osteotomy accuracy estimate: "guides of this type have an average placement accuracy of ±1.4mm"
- Post-op outcome distribution: "80% of similar cases resulted in maxillary position within 1.5mm of planned"

---

## Reference Sources

| Source | URL |
|---|---|
| 3D Systems CMF Solutions | https://www.3dsystems.com/healthcare/craniomaxillofacial-solutions |
| 3D Systems VSP Surgical Planning | https://www.3dsystems.com/healthcare/vsp-surgical-planning-solutions |
| VSP Connect announcement | https://www.additivemanufacturing.media/products/3d-systems-vsp-connect-streamlines-preoperative-planning-for-better-patient-outcomes |
| Stryker/3D Systems partnership | https://www.stryker.com/us/en/about/news/2018/3d-systems-and-stryker-team-up-to-advance-personalized-surgery-.html |
| Materialise ProPlan CMF | https://www.materialise.com/en/healthcare/proplan-cmf |
| Materialise Mimics Enlight CMF | https://www.materialise.com/en/healthcare/mimics/mimics-enlight-cmf |
| Materialise CMF HCP solutions | https://www.materialise.com/en/healthcare/hcps/cmf |
| Materialise Mimics Viewer | https://www.materialise.com/en/healthcare/mimics/mimics-viewer |
| Materialise new gen VSP article | https://www.materialise.com/en/inspiration/articles/new-generation-virtual-surgical-planning-mimics-enlight-cmf |
| Mimics Enlight CMF tutorial video | https://www.materialise.com/zh/academy/healthcare/mimics-innovation-suite/video-tutorials/virtual-planning-orthognathic-surgery-mimics-enlight-cmf |
| Brainlab CMF Planning | https://www.brainlab.com/surgery-products/digital-cmf-surgery/cmf-planning/ |
| Brainlab CMF Navigation | https://www.brainlab.com/surgery-products/digital-cmf-surgery/cmf-navigation/ |
| Brainlab Digital CMF portfolio | https://www.brainlab.com/surgery-products/digital-cmf-surgery/ |
| Brainlab Integrated O.R. | https://www.brainlab.com/digital-o-r/ |
| Dolphin 3D Surgery module | https://dolphinimaging.it/products/3d/3d-surgery/ |
| Dolphin 3D feature sheet (PDF) | https://www.dolphinimaging.com/Areas/Product/Documents/3D/Imaging119_3D.pdf |
| Dolphin Products & Services Guide (PDF) | https://www.dolphinimaging.com/Areas/Media/Documents/20240322_Branch%20Kit.pdf |
| NemoCeph product page | https://nemotec.com/nemostudio/en/products/nemoceph/ |
| NemoStudio v25 release notes | https://nemotec.com/nemostudio/en/updates-in-nemostudio/latest-updates-and-improvements-in-v-25-0-0-0/ |
| DEXIS InVivo 6 | https://dexis.com/en-us/software-invivo-6 |
| Blue Sky Bio guided surgery | https://blueskybio.com/pages/blue-sky-plan-guided-surgery-software |
| Blue Sky Plan user manual (PDF) | https://blueskybio.com/caffeine/uploads/files/documents/Blue%20Sky%20Bio%20Plan%20User%20Manual%20Rev%2010.pdf |
| Blue Sky Bio AI planning blog | https://www.blueskybio.digital/post/ai-implant-planning-blueskyplan |
| SlicerCMF website | https://cmf.slicer.org |
| SlicerCMF GitHub | https://github.com/DCBIA-OrthoLab/SlicerCMF |
| BoneReconstructionPlanner GitHub | https://github.com/SlicerIGT/SlicerBoneReconstructionPlanner |
| 3D Slicer surgical guides forum | https://discourse.slicer.org/t/custom-fitting-surgical-guides-on-slicer/14270 |
| KLS Martin IPS CaseDesigner | https://www.klsmartin.com/en-na/products/individual-patient-solutions-ipsr/ipsr-casedesigner/ |
| IPS CaseDesigner v2.3 new features | https://www.klsmartin.com/en/company/news/news-details/ips-casedesigner-version-23/ |
| PMC: Cost Outcomes of VSP (2024) | https://pmc.ncbi.nlm.nih.gov/articles/PMC11816551/ |
| PMC: VSP Pearls and Pitfalls (2018) | https://pmc.ncbi.nlm.nih.gov/articles/PMC5811276/ |
| PMC: VSP in non-syndromic orthognathic (2026) | https://pmc.ncbi.nlm.nih.gov/articles/PMC12841811/ |
| PMC: 3D Slicer vs ProPlan vs Mimics (2025) | https://www.oaepublish.com/articles/2347-9264.2025.01 |
| PMC: Feasibility/time/cost 3 VSP tools (2021) | https://pmc.ncbi.nlm.nih.gov/articles/PMC7790928/ |
| PMC: AI in orthognathic surgery (2025) | https://pmc.ncbi.nlm.nih.gov/articles/PMC12178734/ |
| PMC: AI in pediatric craniofacial (2025) | https://pmc.ncbi.nlm.nih.gov/articles/PMC11989140/ |
| PMC: VR vs. desktop VSP training (2023) | https://games.jmir.org/2023/1/e40541/ |
| PMC: Uncertainty visualization surgery (2025) | https://pmc.ncbi.nlm.nih.gov/articles/PMC12239872/ |
| PMC: DICOM viewer surgical planning (2015) | https://pmc.ncbi.nlm.nih.gov/articles/PMC4570900/ |
| PMC: AI segmentation orthognathic (2026) | https://pubmed.ncbi.nlm.nih.gov/41648684/ |
| PMC: Soft tissue prediction comparison (2022) | https://www.nature.com/articles/s41598-022-08991-7 |
| AO Foundation CMF digital workflows | https://www.aofoundation.org/cmf/about-aocmf/blog/updated-mft-curriculum-vsp-3d-modeling-and-ar |
| arXiv: Uncertainty visualization surgical review | https://arxiv.org/html/2501.06280v1 |
| ProPlan CMF VA TRM PDF | https://www.oit.va.gov/Services/TRM/files/SynthesProPlanCMF.pdf |
| PMC: Learning curve VSP orbital fractures | https://escholarship.org/uc/item/2dm1r14g |
| PMC: Systematic review VSP digital transfer (2025) | https://onlinelibrary.wiley.com/doi/10.1111/ocr.12934 |
