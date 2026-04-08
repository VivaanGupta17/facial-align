# Technical Stack Research: AI-Native Craniofacial Surgical Planning Platform

> **Compiled:** 2025  
> **Scope:** Backend processing, AI/ML pipeline, 3D visualization, data infrastructure, compliance  
> **Format:** Each section covers what a tool does, why it fits, limitations, baseline vs. later, version/URLs

---

## Table of Contents

1. [DICOM Ingestion and Preprocessing](#1-dicom-ingestion-and-preprocessing)
2. [CT Volume Rendering](#2-ct-volume-rendering)
3. [3D Segmentation Frameworks](#3-3d-segmentation-frameworks)
4. [Mesh Extraction](#4-mesh-extraction)
5. [Craniofacial and Mandibular Segmentation](#5-craniofacial-and-mandibular-segmentation)
6. [Dental Segmentation from CT and Intraoral Scans](#6-dental-segmentation-from-ct-and-intraoral-scans)
7. [CT-to-Scan Registration](#7-ct-to-scan-registration)
8. [Fracture Fragment Identification](#8-fracture-fragment-identification)
9. [3D Visualization Frameworks for Web](#9-3d-visualization-frameworks-for-web)
10. [ML-Ready Data Pipeline Design](#10-ml-ready-data-pipeline-design)
11. [Privacy, HIPAA, and Deployment](#11-privacy-hipaa-and-deployment)
12. [Recommended Baseline Stack Summary](#12-recommended-baseline-stack-summary)

---

## 1. DICOM Ingestion and Preprocessing

### 1.1 The DICOM Challenge

DICOM is not a simple image format—it is a communication protocol, file format, and data model combined. A single study may contain dozens of series, each with different reconstruction kernels, slice thicknesses, orientations, and contrast phases. Key parsing pitfalls:

- **Photometric inversion**: `MONOCHROME1` images must be inverted before display or ML use. Raw pixel arrays in such files are bright=low, dark=high, the opposite of the conventional radiological convention.
- **HU calibration**: Raw CT pixel values require linear rescaling via `RescaleSlope` and `RescaleIntercept` tags to produce Hounsfield Units. Bone is 300–2000 HU; soft tissue is −100 to 100 HU; air is approximately −1000 HU.
- **Pixel spacing**: `PixelSpacing` for CT/MR; `ImagerPixelSpacing` for radiography. These differ and are not interchangeable.
- **Series fragmentation**: A single CT study may contain axial, MPR, scout, and localizer series that should not be merged into one volume.
- **Missing/corrupted metadata**: Clinical-origin files frequently have blank or incorrect institution fields, wrong UIDs, and inconsistent orientation cosines.
- **Metal artifact**: High-density implants produce beam-hardening streaks that corrupt adjacent voxels—directly relevant for patients with prior dental work or plates.

**References:**  
- [Medical Imaging AI/ML Engineering Guide (Innolitics, 2025)](https://innolitics.com/articles/medical-imaging-best-practices/)

---

### 1.2 pydicom

**What it does:** Pure Python DICOM file parser and writer. Reads any DICOM file into a `Dataset` object where every tag is accessible as a Python attribute. Does not perform volume reconstruction—it operates at the file level.

**Why it fits:** Industry standard for DICOM metadata access and manipulation. Widely used across the medical imaging ecosystem; required as a dependency of SimpleITK, highdicom, and MONAI.

**Key capabilities:**
- Stop-before-pixels reading (`stop_before_pixels=True`) for fast metadata indexing
- Private tag access
- Pixel array extraction as NumPy arrays
- DICOM writing (including anonymization)
- SR (Structured Report) code dictionaries

**Limitations:**
- Does not reconstruct 3D volumes from DICOM series—you must collect individual files and assemble them
- No native volume registration or resampling
- Older versions had incomplete Sequence handling

**Version:** pydicom 3.0.x (latest stable as of 2025)  
**URL:** https://github.com/pydicom/pydicom  
**License:** MIT  
**Baseline:** ✅ Yes — core dependency for all DICOM work

---

### 1.3 SimpleITK

**What it does:** C++ ITK library with Python bindings. Handles complete 3D volume reconstruction from DICOM series, resampling, reorientation, and spatial transforms. Understands image spacing, direction cosines, and physical coordinate systems.

**Why it fits:** The correct tool for turning a folder of `.dcm` slice files into a 3D numpy array with proper voxel spacing and orientation. It handles the `GDCMSeriesFileNames` lookup to correctly order slices. All downstream ML pipelines consume properly oriented, isotropically resampled volumes.

**Key capabilities:**
- `ImageSeriesReader` — reads full DICOM series, handles sorting automatically
- `DICOMOrient` — reorients volumes to standard (RAS/LPS) coordinates
- `ResampleImageFilter` — isotropic resampling (e.g., to 1.0mm³) for consistent ML input
- Gaussian smoothing, histogram matching, and ComBat harmonization integration
- Reads/writes NIfTI (.nii.gz) and NRRD — standard ML pipeline formats
- Multi-modality support (CT, MRI, PET)

**Critical code pattern:**
```python
import SimpleITK as sitk

reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames('/path/to/series/')
reader.SetFileNames(dicom_names)
reader.MetaDataDictionaryArrayUpdateOn()
image = reader.Execute()

# Reorient to LPS standard
image_lps = sitk.DICOMOrient(image, 'LPS')

# Resample to 1mm isotropic
resampled = resample_volume(image_lps, new_spacing=(1.0, 1.0, 1.0))
```

**Limitations:**
- ITK C++ dependency means compilation issues on some platforms (mitigated by pip wheels)
- Large memory footprint for whole-skull CT volumes (512×512×400+ voxels)
- Not suitable for streaming/web contexts — server-side only

**Version:** SimpleITK 2.4.x  
**URL:** https://simpleitk.org  
**License:** Apache 2.0  
**Baseline:** ✅ Yes — non-negotiable for volumetric preprocessing

---

### 1.4 highdicom

**What it does:** High-level Python library for creating and reading standardized DICOM annotation objects — Segmentation IODs, Structured Reports (SR), and Secondary Capture images. Wraps pydicom with a developer-friendly object-oriented API.

**Why it fits:** When your AI model produces a segmentation mask, you need to store it in a standards-compliant DICOM Segmentation object so it can be round-tripped to any PACS/viewer (OHIF, 3D Slicer). highdicom handles all DICOM SOP Class encoding, segment metadata, and dimension indexing automatically.

**Key capabilities:**
- `highdicom.seg.Segmentation` — encodes numpy masks as DICOM SEG objects (binary or fractional)
- `highdicom.sr.Comprehensive3DSR` — stores 3D landmark coordinates, measurements, and findings
- Coded concepts using SNOMED-CT (SCT), DCM, and UCUM terminologies
- Auto-copies patient/study metadata from source images
- Interoperable with OHIF viewer for overlay rendering

**Limitations:**
- Not needed until you have a working segmentation pipeline — adds complexity early
- DICOM SEG format has limited support in some older PACS systems
- Learning curve for SR templates and coding schemes

**Version:** highdicom 0.24.x  
**URL:** https://highdicom.readthedocs.io  
**License:** MIT  
**Baseline:** ⏩ Later (Phase 2) — introduce when storing AI outputs back to PACS

---

### 1.5 De-identification / Anonymization

DICOM files contain 18 HIPAA Safe Harbor identifiers embedded in tags, plus potentially burned-in PHI in pixel data. Two tools are primary:

#### pydicom/deid
- YAML-based recipe system: `remove`, `hash`, `replace`, `date-shift`
- Handles standard DICOM headers and private tags
- Does **not** process burned-in pixel text — requires separate OCR step
- URL: https://github.com/pydicom/deid

#### RSNA Clinical Trial Processor (CTP)
- Java-based; 98% de-identification success with default profile, 100% with custom
- Handles pixel data text through image region blackout
- Widely used in clinical trial contexts
- URL: https://www.rsna.org/research/imaging-research-tools

**Critical gap:** Both header-only tools miss PHI burned into pixel data (common in scout images, secondary captures). Use Amazon Rekognition + Amazon Textract, or a vision model, to detect and redact embedded text.

**DICOM PS 3.15 Annex E** defines the de-identification standard that all compliant tools implement.

**Baseline:** ✅ Yes — required before any data leaves clinical environment

---

### 1.6 Volume Reconstruction Pipeline

Standard pre-processing sequence for a CT series destined for segmentation:

```
DICOM files
   ↓ pydicom (series indexing, metadata extraction)
   ↓ SimpleITK ImageSeriesReader (volume assembly)
   ↓ DICOMOrient → LPS canonical orientation
   ↓ ResampleImageFilter → 1.0mm isotropic (or 0.4mm for CBCT dental)
   ↓ HU clipping: bone=[300, 2000], soft tissue=[-150, 250]
   ↓ Normalize to [0, 1] float32
   ↓ Save as .nii.gz (NIfTI) for ML pipeline
```

For CBCT dental scans, target spacing is typically 0.3–0.4mm isotropic to preserve fine root canal and alveolar crest detail. Standard CT scans for craniofacial work are typically 0.5–1.0mm.

---

## 2. CT Volume Rendering

### 2.1 Comparison Matrix

| Library | Volume Rendering | Mesh Rendering | DICOM Native | Medical Tools | Complexity | License |
|---------|-----------------|----------------|--------------|---------------|------------|---------|
| **Cornerstone3D** | ✅ GPU | ✅ (limited) | ✅ Streaming | ✅ Full | Medium | MIT |
| **OHIF Viewer** | ✅ via C3D | ✅ (limited) | ✅ DICOMweb | ✅ Full | High | MIT |
| **VTK.js** | ✅ GPU | ✅ STL/OBJ | ⚠️ Partial | Limited | High | BSD |
| **NiiVue** | ✅ WebGL2 | ✅ 30+ formats | ✅ Plugin | Limited | Low | MIT |
| **three.js** | ❌ No | ✅ Any | ❌ No | None | Low | MIT |
| **AMI.js** | ✅ WebGL | ✅ | ✅ | Limited | Medium | MIT |

---

### 2.2 Cornerstone3D

**What it does:** JavaScript library for GPU-accelerated 2D and 3D medical image rendering in the browser. The current rendering engine for OHIF v3.9. Built on WebGL with vtk.js as the 3D rendering backend.

**Why it fits:** Purpose-built for medical imaging. Handles DICOM streaming (progressive loading), MPR (multi-planar reformation) — axial, sagittal, coronal views simultaneously — volume rendering with CT presets (bone window, soft tissue), and annotation tools (distance measurement, angle, ROI). The only web library with native DICOM SEG overlay rendering.

**Key capabilities:**
- Offscreen rendering: shares single WebGL context across up to 10+ viewports
- Volume streaming: loads DICOM progressively over DICOMweb
- MPR viewports out of the box
- Segmentation rendering: renders DICOM SEG masks as colored overlays
- Tool framework: distance, area, window/level, brush segmentation editor
- TypeScript native; React integration through wrapper components

**Limitations:**
- Heavy bundle size (~5MB+ gzipped)
- Steep learning curve — requires understanding of viewports, rendering engines, and tool groups
- 3D mesh rendering is secondary; designed for volumetric CT, not STL/OBJ surgical mesh manipulation
- No built-in surgical simulation tools

**Version:** Cornerstone3D 2.x (OHIF 3.9 uses this)  
**URL:** https://cornerstonejs.org  
**License:** MIT  
**Baseline:** ✅ Yes — for the CT/CBCT viewing panel

---

### 2.3 OHIF Viewer

**What it does:** Complete zero-footprint web-based DICOM viewer built on top of Cornerstone3D. A full application framework with routing, hanging protocols, work lists, and extensible modes.

**Why it fits:** If the goal is to embed a DICOM viewer within the platform without building the entire viewer from scratch, OHIF provides all the infrastructure. It supports 3D volume rendering, MPR, 4D CINE, color LUT presets, and segmentation overlays. OHIF v3.8+ added 4D visualization and per-viewport rendering controls with CT/MR presets.

**Key capabilities:**
- Extensible extension/mode architecture — add custom panels, tools, and workflows
- DICOMweb integration (WADO-RS, STOW-RS, QIDO-RS)
- 4D visualization (CINE player for dynamic imaging)
- Integration with eContour (radiation oncology contouring)
- Used by TCIA and NCI Imaging Data Commons

**Limitations:**
- Designed as a standalone viewer — integrating it as a component inside a React app requires custom wrappers
- For a custom surgical planning UI, OHIF's opinionated routing and layout may constrain the design
- You may want to use Cornerstone3D directly rather than all of OHIF

**Version:** OHIF v3.9 (as of 2024/2025)  
**URL:** https://ohif.org  
**License:** MIT  
**Baseline:** ⚠️ Consider — use OHIF as inspiration/source; use Cornerstone3D directly for a custom app

---

### 2.4 VTK.js

**What it does:** JavaScript port of VTK (Visualization Toolkit). Full scientific visualization pipeline: volume rendering, surface rendering, GPU ray casting, transfer function editing, annotations.

**Why it fits:** VTK.js is the rendering engine inside Cornerstone3D for 3D volumes. It can also render STL/OBJ/VTK mesh files, making it suitable for displaying surgical implant designs alongside CT volumes.

**Key capabilities:**
- GPU-accelerated volume raycasting with transfer functions
- ParaView Glance (online demo): https://paraview.github.io/paraview-glance/
- Crop/annotate/color controls
- WebXR support (AR/VR)
- Integration tutorials for OHIF: https://www.kitware.com/vtk-js-video-tutorials-integrating-scientific-visualizations-into-ohif-and-other-web-applications/

**Limitations:**
- Large bundle size
- Lower-level API than Cornerstone3D — more configuration required
- For medical DICOM workflows, Cornerstone3D (which wraps vtk.js) is preferred

**Version:** vtk.js 29.x  
**URL:** https://kitware.github.io/vtk-js  
**License:** BSD  
**Baseline:** ⚠️ Indirect — used via Cornerstone3D, not imported directly

---

### 2.5 NiiVue

**What it does:** WebGL2-based neuroimaging viewer supporting 30+ voxel and mesh formats. Lightweight, framework-agnostic — works with React, Vue, Angular, or plain HTML.

**Why it fits:** NiiVue is the most capable pure-volume renderer for a custom embedded context where you don't want all of OHIF's weight. It handles NIfTI, NRRD, and DICOM volumes, plus meshes (STL, OBJ, PLY, VTK, FreeSurfer surfaces). Excellent for showing processed outputs (segmentation masks as overlays, reconstructed meshes) alongside the source CT.

**Key capabilities:**
- Simple React embedding: `import { Niivue } from '@niivue/niivue'`
- Correct spatial orientation (NIfTI sForm/qForm)
- Anisotropic volume support
- Mesh overlays (GIfTI, STL, OBJ, PLY, etc.)
- Tractography formats
- DICOM support via plugin

**Limitations:**
- Neuroimaging-focused; less clinical DICOM workflow tooling than Cornerstone3D
- No built-in DICOM annotation tools (window/level presets, measurement tools)
- Smaller ecosystem than Cornerstone3D

**Version:** NiiVue 0.57.x  
**URL:** https://niivue.com | https://github.com/niivue/niivue  
**License:** MIT  
**Baseline:** ⚠️ Optional — consider for a lightweight mesh+volume overlay panel

---

### 2.6 AMI.js

**What it does:** WebGL-based DICOM viewer toolkit from FNNDSC (Boston Children's Hospital). Focuses on 2D/3D DICOM rendering with MPR support.

**Why it fits:** Historical use in several custom medical viewers. Less actively maintained compared to Cornerstone3D.

**Limitations:**
- Development has slowed relative to Cornerstone3D
- Smaller community and ecosystem
- Not recommended for new projects given Cornerstone3D's trajectory

**URL:** https://github.com/FNNDSC/ami  
**Baseline:** ❌ Not recommended for new development

---

### 2.7 Performance Considerations

- **WebGL2** is required for volume rendering; all major browsers support it (note: Safari had gaps on iOS that are now resolved)
- **GPU memory**: A 512×512×400 CT volume at float32 requires ~200MB GPU memory. Use streaming (Cornerstone3D's progressive loading) rather than loading full volumes upfront
- **Level of Detail (LOD)**: For web delivery, downsample volumes to 2–3mm for initial preview, stream full resolution on zoom
- **Mesh polygon count**: Patient-specific surgical meshes from marching cubes can exceed 1M triangles; decimate to 50K–100K before web rendering

---

## 3. 3D Segmentation Frameworks

### 3.1 nnU-Net

**What it does:** Self-configuring deep learning segmentation framework. Given any medical image segmentation dataset in the correct format, nnU-Net automatically configures preprocessing, network architecture, training pipeline, and ensembling. Based on U-Net with residual connections.

**Why it fits:** nnU-Net has been the de facto standard for medical image segmentation benchmarks since 2019. It does not require hyperparameter tuning — it reads the dataset statistics and sets everything automatically. All of TotalSegmentator and DentalSegmentator are trained with nnU-Net. MICCAI 2024 benchmark confirmed that CNN-based U-Net models (via the nnU-Net framework) still outperform most transformer and Mamba architectures for 3D medical segmentation.

**Key capabilities:**
- Three training configurations: 2D, 3D_fullres, 3D_lowres, and ensemble
- Data augmentation: elastic deformation, rotation, scale, mirroring, noise
- Self-configuring: reads patch size, spacing, intensity statistics from data
- `nnUNetv2_predict` CLI for inference on new cases
- Pretrained model marketplace (community checkpoints on Zenodo)
- Apple M1/M2 (MPS) and CPU support in addition to CUDA

**Training data format (nnU-Net dataset structure):**
```
nnUNet_raw/Dataset001_Name/
   imagesTr/   # training images (.nii.gz)
   imagesTs/   # test images
   labelsTr/   # segmentation masks (.nii.gz)
   dataset.json
```

**Limitations:**
- Training from scratch requires labeled data (50–200 cases minimum for decent performance)
- Inference on a 512×512×400 head CT: ~2–5 minutes on GPU, ~30 minutes on CPU
- Not interactive — batch inference only

**Version:** nnU-Net v2 (nnunetv2, `pip install nnunetv2`)  
**GitHub:** https://github.com/MIC-DKFZ/nnUNet  
**Paper:** [Isensee et al., Nat Methods 2021](https://doi.org/10.1038/s41592-020-01008-z)  
**License:** Apache 2.0  
**Baseline:** ✅ Yes — primary framework for training custom craniofacial segmentation models

---

### 3.2 MONAI (Medical Open Network for AI)

**What it does:** PyTorch-based framework providing medical-specific transforms, network architectures (SegResNet, Swin-UNETR, DynUNet), training utilities, and deployment infrastructure. Think of MONAI as the PyTorch Lightning of medical imaging.

**Why it fits:** MONAI is the ecosystem glue. It provides:
- **MONAI Core**: Training loop, medical augmentations, model zoo
- **MONAI Label**: Active learning annotation server (integrates with 3D Slicer and OHIF)
- **MONAI Deploy**: Clinical deployment SDK (containerized MAP packages, DICOM/FHIR support)

MONAI Label alone can reduce annotation time by up to 75% by serving active learning-guided annotation into 3D Slicer sessions — directly relevant for building your training dataset.

**Key architectures available:**
- **SegResNet**: Residual encoder-decoder; strong for brain and head/neck segmentation
- **Swin-UNETR**: Transformer-UNet hybrid; strong BTCV benchmark results
- **UNETR**: Pure transformer segmentation network
- **DynUNet**: MONAI reimplementation of nnU-Net architecture
- **DeepEdit / DeepGrow**: Interactive click-based segmentation models (for MONAI Label)

**Limitations:**
- MONAI is not as "auto-configuring" as nnU-Net — you set hyperparameters manually
- MONAI Label server requires a GPU instance to serve annotation suggestions in real time
- More code required than nnU-Net for the same task

**Version:** MONAI 1.4.x  
**URL:** https://project-monai.github.io  
**License:** Apache 2.0  
**Baseline:** ✅ Yes — MONAI Label for annotation pipeline; MONAI Core for custom model training; MONAI Deploy for production

---

### 3.3 TotalSegmentator

**What it does:** Pre-trained nnU-Net model for automatic segmentation of 104+ anatomical structures on CT images. Trained on 1,204 diverse clinical CT examinations. Open source, CLI tool, and Python API.

**Why it fits:** Provides immediate out-of-the-box segmentation for skull, mandible, teeth (FDI-numbered), craniofacial structures, head muscles, and head/neck bones/vessels — exactly the anatomy relevant to craniofacial surgical planning. Dice score of 0.943 across all 104 structures on a diverse clinical test set.

**Craniofacial-relevant tasks (run with `-ta <task_name>`):**

| Task | Key Structures |
|------|----------------|
| `craniofacial_structures` | mandible, upper/lower teeth, skull, maxillary/frontal sinuses |
| `teeth` | All 32 FDI-numbered teeth + pulp chambers + inferior alveolar canals |
| `head_glands_cavities` | Parotid/submandibular glands, nasal cavity, hard/soft palate |
| `head_muscles` | Masseter, temporalis, pterygoids, tongue, digastric |
| `headneck_bones_vessels` | Zygomatic arches, styloid process, carotid arteries, jugular veins |
| `headneck_muscles` | SCM, scalenes, constrictor muscles |
| `total` | Skull (class 91), brain (class 90), cervical vertebrae C1–C7 |

**Note:** TMJ condyle and glenoid fossa are **not** directly in TotalSegmentator's standard task list — these require DentalSegmentator or custom models (see Section 5).

**Runtime:**
- Head CT (512×512×280): ~1–2 min on GPU (RTX 3090)
- CPU-only: ~15–30 min

**Installation:**
```bash
pip install TotalSegmentator
TotalSegmentator -i ct.nii.gz -o segmentations/ -ta craniofacial_structures
TotalSegmentator -i ct.nii.gz -o segmentations/ -ta teeth
```

**Version:** TotalSegmentator v2.5.0 (weights); Docker: `wasserth/totalsegmentator:2.11.0`  
**URL:** https://github.com/wasserth/TotalSegmentator  
**Paper:** [Wasserthal et al., Radiology AI 2023](https://pubs.rsna.org/doi/abs/10.1148/ryai.230024)  
**License:** Apache 2.0 (most tasks); some tasks require free non-commercial license  
**Baseline:** ✅ Yes — use immediately for skull, mandible, and teeth segmentation bootstrapping

---

### 3.4 MedSAM / MedSAM2

**What it does:** Fine-tuned versions of Meta's Segment Anything Model (SAM/SAM2) for medical imaging. MedSAM2 (April 2025) is a promptable 3D segmentation model trained on 455,000+ 3D image-mask pairs across CT, MRI, PET, ultrasound, and endoscopy.

**Why it fits:** MedSAM2 enables interactive segmentation with a bounding box prompt on one slice, then propagates the mask across the entire 3D volume. This dramatically accelerates manual annotation workflows. Annotation time reduced by 85%+ (from ~525s to ~74s per lesion in CT).

**Performance vs nnU-Net:** MedSAM2 achieves DSC of ~0.87 for CT organs — slightly below task-specific nnU-Net models but requires only a bounding box prompt, no training data.

**Architecture:** Hiera vision transformer + memory attention module (SAM2.1-Tiny backbone). Fine-tuned end-to-end.

**Use case in this platform:**
1. Surgeon draws bounding box around fracture fragment or anomalous structure
2. MedSAM2 propagates mask across all CT slices in ~1 second
3. Surgeon reviews and accepts/corrects
4. Accepted masks feed the training dataset for nnU-Net fine-tuning

**Limitations:**
- Bounding box–only prompts; no point/scribble support yet
- 8-frame memory window may be insufficient for very tall structures
- Requires GPU for inference

**URL:** https://medsam2.github.io | https://github.com/bowang-lab/MedSAM  
**Baseline:** ⏩ Later (Phase 2) — introduce in the annotation loop; use TotalSegmentator for automatic batch segmentation first

---

## 4. Mesh Extraction

### 4.1 Marching Cubes Algorithm

**What it does:** Extracts an isosurface mesh from a 3D scalar field (segmentation mask). A threshold value defines the surface boundary; the algorithm generates triangles across each voxel cube that straddles the threshold.

**For segmentation masks**: Threshold at 0.5 on a binary mask. Each labeled structure becomes a separate mesh.  
**For raw CT HU volumes**: Threshold at ~300–400 HU to isolate bone.

**Variants:**
- **Classic Marching Cubes** (Lorensen & Cline, 1987): 15 cube configurations; can produce topological ambiguities
- **Flying Edges** (VTK 8.0+): Faster parallelized implementation of marching cubes; preferred for large volumes
- **Discrete Marching Cubes** (`vtkDiscreteMarchingCubes`): Handles integer label maps directly; produces separate surfaces per label value

---

### 4.2 VTK (Python)

**What it does:** The Visualization Toolkit — C++ library with Python bindings. The computational foundation for surface extraction, mesh processing, and 3D visualization on the Python server side.

**Why it fits:** VTK's `vtkDiscreteMarchingCubes` takes a labeled segmentation array (where each integer = one anatomical structure) and produces separate meshes per label in a single pass. This is the workhorse for converting segmentation outputs to surgical STL files.

**Key classes for surgical mesh pipeline:**
- `vtkDiscreteMarchingCubes` — multi-label surface extraction
- `vtkMarchingCubes` — single threshold isosurface
- `vtkWindowedSincPolyDataFilter` — Taubin-style mesh smoothing (preserves shape better than Laplacian)
- `vtkDecimatePro` / `vtkQuadricDecimation` — mesh simplification (polygon reduction)
- `vtkPolyDataConnectivityFilter` — keep largest connected component (remove islands)
- `vtkSTLWriter` / `vtkOBJWriter` — export to STL, OBJ
- `vtkDICOMImageReader` — read DICOM directly into VTK

**Typical pipeline:**
```python
import vtk

# Threshold to isolate bone from segmentation label
reader = vtk.vtkNrrdReader()  # or vtkNIFTIImageReader
reader.SetFileName("segmentation.nii.gz")
reader.Update()

mc = vtk.vtkDiscreteMarchingCubes()
mc.SetInputConnection(reader.GetOutputPort())
mc.GenerateValues(1, 1, 1)  # label 1 = mandible
mc.Update()

# Smooth
smoother = vtk.vtkWindowedSincPolyDataFilter()
smoother.SetInputConnection(mc.GetOutputPort())
smoother.SetNumberOfIterations(20)
smoother.BoundarySmoothingOff()
smoother.FeatureEdgeSmoothingOff()
smoother.SetFeatureAngle(120)
smoother.SetPassBand(0.001)  # lower = smoother
smoother.NonManifoldSmoothingOn()
smoother.NormalizeCoordinatesOn()
smoother.Update()

# Decimate
decimate = vtk.vtkDecimatePro()
decimate.SetInputConnection(smoother.GetOutputPort())
decimate.SetTargetReduction(0.9)  # reduce to 10% of original
decimate.PreserveTopologyOn()
decimate.Update()

# Export
writer = vtk.vtkSTLWriter()
writer.SetFileName("mandible.stl")
writer.SetInputConnection(decimate.GetOutputPort())
writer.Write()
```

**Limitations:**
- VTK Python API is verbose; many lines of boilerplate
- Large meshes (>2M triangles) take time to process

**Version:** VTK 9.3.x  
**URL:** https://vtk.org  
**License:** BSD  
**Baseline:** ✅ Yes — primary server-side mesh extraction tool

---

### 4.3 PyVista

**What it does:** High-level Python wrapper around VTK that removes the boilerplate. `mesh.plot()` in 2 lines vs. 30 in raw VTK.

**Why it fits:** Use PyVista for exploratory work, mesh QA, and rapid prototyping. It wraps VTK's full functionality including marching cubes, surface extraction, smoothing, and Boolean operations.

**Key capabilities:**
- `grid.contour([threshold], method='marching_cubes')` — one-line marching cubes
- `mesh.smooth(n_iter=100)` — Laplacian smoothing
- `mesh.decimate(target_reduction=0.9)` — quadric decimation
- `mesh.fill_holes()` — close mesh artifacts
- `mesh.compute_normals()` — prepare for rendering
- `mesh.save('out.stl')` — export

**Limitations:**
- Adds a layer of abstraction; heavy production pipelines should use raw VTK for performance control
- Interactive visualization (`pv.Plotter`) requires display environment — not suitable for headless server rendering without off-screen flag

**Version:** PyVista 0.44.x  
**URL:** https://pyvista.org  
**License:** MIT  
**Baseline:** ✅ Yes — use for development/prototyping; wrap in VTK calls for production

---

### 4.4 scikit-image

**What it does:** Pure Python/SciPy image processing library. Includes `measure.marching_cubes` for surface extraction.

**Why it fits:** Simpler alternative to VTK for quick prototyping. Returns vertices and faces as NumPy arrays, which can be passed to trimesh.

**Key capability:**
```python
from skimage import measure
verts, faces, normals, values = measure.marching_cubes(mask_array, level=0.5, spacing=(1.0, 1.0, 1.0))
```

**Limitations:**
- Slower than VTK for large volumes
- Limited smoothing/decimation utilities
- Returns raw numpy arrays — need trimesh or VTK for further processing

**Version:** scikit-image 0.24.x  
**License:** BSD  
**Baseline:** ⚠️ Optional — use for quick prototyping or simple meshes; prefer VTK/PyVista for production

---

### 4.5 trimesh

**What it does:** Python library for 3D mesh loading, processing, and manipulation. Handles STL, OBJ, PLY, GLB/glTF, and many other formats.

**Why it fits:** trimesh is excellent for post-processing meshes after VTK extraction: fixing non-manifold geometry, computing volume/centroid, Boolean operations, computing point clouds, and generating watertight meshes for 3D printing.

**Key capabilities:**
- `trimesh.load('mesh.stl')` — universal format loading
- `mesh.is_watertight` — check closed surface
- `trimesh.repair.fix_normals(mesh)` — fix inverted normals
- `mesh.volume` — compute anatomical volume in mm³
- Boolean CSG operations (union, difference, intersection) via manifold or blender backend
- Point cloud sampling
- Proximity queries

**Limitations:**
- Boolean operations require external backends (manifold3d, or blender)
- Large mesh Boolean operations can be slow and numerically unstable

**Version:** trimesh 4.x  
**URL:** https://trimsh.org  
**License:** MIT  
**Baseline:** ✅ Yes — essential for mesh validation, repair, and format conversion

---

### 4.6 Smoothing Strategy for Surgical Meshes

Raw marching cubes produces "staircase" artifacts at voxel boundaries. Two-pass smoothing:

1. **Pre-extraction**: Apply 3D Gaussian smoothing (`sigma=1.0`) to the segmentation mask before marching cubes. This reduces jagged edges at the source.
2. **Post-extraction**: Apply Windowed Sinc smoothing (VTK `vtkWindowedSincPolyDataFilter`) rather than Laplacian. Windowed Sinc preserves anatomical shape (ridge lines, condylar convexity) while smoothing noise.

Optimal parameters (from literature): Gaussian convolution kernel size 5 + mesh simplification reduction factor 0.1 achieves best balance of accuracy (SSIM improvement), speed (69.8% faster), and polygon count (86.6% reduction).

**Reference:** [Sensors 2021 - Effects of Parameter Settings for 3D Data Smoothing and Mesh Simplification](https://pmc.ncbi.nlm.nih.gov/articles/PMC8659505/)

---

## 5. Craniofacial and Mandibular Segmentation

### 5.1 What Can Be Used Out-of-the-Box

The following structures can be segmented immediately without custom training data, using publicly available models:

| Structure | Tool | Task Flag | Dice Score |
|-----------|------|-----------|------------|
| Skull | TotalSegmentator | `total` (class 91) | ~0.95 |
| Mandible | TotalSegmentator | `craniofacial_structures` | ~0.94 |
| Mandible (detailed) | DentalSegmentator | — | 0.92–0.94 |
| Maxilla + upper skull | DentalSegmentator | — | 0.92–0.94 |
| Upper teeth (bulk) | TotalSegmentator | `craniofacial_structures` | ~0.90 |
| Lower teeth (bulk) | TotalSegmentator | `craniofacial_structures` | ~0.90 |
| FDI-numbered teeth (all 32) | TotalSegmentator | `teeth` | Per-tooth models |
| Tooth pulp chambers | TotalSegmentator | `teeth` | Per-tooth models |
| Inferior alveolar canal | DentalSegmentator | — | ~0.69–0.84 |
| Mandibular canal | TotalSegmentator | `teeth` | Sub-task included |
| Masseter/temporalis/pterygoids | TotalSegmentator | `head_muscles` | ~0.88 |
| Zygomatic arch | TotalSegmentator | `headneck_bones_vessels` | ~0.90 |
| Parotid/submandibular glands | TotalSegmentator | `head_glands_cavities` | ~0.90 |
| Brain | TotalSegmentator | `total` | ~0.97 |
| Cervical vertebrae C1–C7 | TotalSegmentator | `total` | ~0.95 |

---

### 5.2 What Requires Custom Training or Fine-Tuning

| Structure | Status | Best Starting Point |
|-----------|--------|---------------------|
| **TMJ condyle** (3D, precise) | No off-shelf pretrained model | Fine-tune nnU-Net on your own CBCT dataset; cascade approach (coarse→fine) achieves DSC 0.932 |
| **Glenoid fossa** | No off-shelf model | U-Net fine-tuning; F1 score 0.966 achievable |
| **Fracture fragments** | No general pretrained model | See Section 8 |
| **Individual condylar head geometry** | Requires dataset | 3D U-Net cascade |
| **Orbital walls** | Partial coverage in TotalSegmentator | Fine-tune |
| **Nasal bones** | Not in standard tasks | Custom |

---

### 5.3 DentalSegmentator

**What it does:** nnU-Net v2 model trained on 470 dento-maxillofacial CT and CBCT scans from 7 institutions. Produces: maxilla/upper skull, mandible, upper teeth, lower teeth, and mandibular canal.

**Why it fits:** The most accurate open-source model specifically for craniofacial CT/CBCT segmentation. Evaluated on 256 scans; mean Dice 0.922 (internal set), 0.942 (external 5-institution set). Robust to metallic artifact and varying field of view.

**Installation:**
```bash
# As 3D Slicer extension: search "DentalSegmentator" in Extension Manager
# As CLI:
pip install nnunetv2
# Download weights from Zenodo:
# https://zenodo.org/records/10829675
# Dataset112_DentalSegmentator_v100.zip (229.7 MB)
nnUNetv2_predict -i INPUT/ -o OUTPUT/ -d 112 -c 3d_fullres
```

**Reference:** [Dot et al., Journal of Dentistry 2024](https://doi.org/10.1016/j.jdent.2024.105130)  
**GitHub:** https://github.com/gaudot/SlicerDentalSegmentator  
**Zenodo Weights:** https://zenodo.org/records/10829675  
**License:** Apache 2.0  
**Baseline:** ✅ Yes — use as the primary craniofacial segmentation model

---

### 5.4 TMJ Condyle Segmentation

The mandibular condyle and glenoid fossa are not covered by existing pretrained models with clinical-grade accuracy. Best approach:

**Architecture:** Cascaded 3D U-Net
1. First network: coarse detection of condyle region → crops ROI
2. Second network: fine-grained segmentation within crop

**Reported performance:** Cascaded 3D U-Net achieves DSC 0.932 ± 0.023 with 200 training samples from two institutions.

**Data requirement:** 150–200 manually annotated CBCT scans minimum (can use MONAI Label with DeepEdit to accelerate to ~25% manual annotation)

**Reference:** [Fully automated condyle segmentation using 3D CNNs, Scientific Reports 2022](https://www.nature.com/articles/s41598-022-24164-y)

**TMJ Disc:** MRI is the gold standard for TMJ disc, not CT. Use MRI-specific models if disc assessment is required.

---

### 5.5 Public Datasets for Craniofacial Segmentation

| Dataset | Modality | Structures | Cases | Access |
|---------|----------|------------|-------|--------|
| **ToothFairy** (MICCAI 2023) | CBCT | Inferior alveolar canal | 443 (153 labeled) | Grand Challenge |
| **ToothFairy2** (MICCAI 2024) | CBCT | 42 classes (jaws, teeth, IAC, maxillary sinus, pharynx) | 480 (training) | Grand Challenge |
| **CTooth+** | CT/CBCT | All teeth (instance) | 168 | Public |
| **STS-2D-Tooth / STS-3D-Tooth** | CBCT + PXI | Teeth (semi-supervised) | 148,400 CBCT slices | Public benchmark |
| **MMDental** | CBCT | Teeth + medical records | 660 patients | Nature Scientific Data |
| **MICCAI STS 2024** | OPG + CBCT | Instance-level FDI teeth | 2,380 OPG + 330 CBCT | Grand Challenge |
| **TotalSegmentator training data** | CT | 104 structures (including skull) | 1,204 CT | Zenodo: 10.5281/zenodo.6802613 |

**References:**  
- [ToothFairy: IEEE TMI 2025](https://doi.org/10.1109/TMI.2024.3523096)  
- [ToothFairy2 Challenge](https://toothfairy2.grand-challenge.org)  
- [MMDental Dataset, Nature Scientific Data 2025](https://www.nature.com/articles/s41597-025-05398-7)

---

## 6. Dental Segmentation from CT and Intraoral Scans

### 6.1 Individual Tooth Segmentation from CBCT

State of the art achieves DSC 0.93–0.97 for per-tooth instance segmentation. Key approaches:

**Challenge results (MICCAI STS 2024 CBCT track):**
- Winning semi-supervised method boosted Instance Dice by 61 percentage points over a supervised-only nnU-Net baseline
- Best methods combined SAM-family foundation models with coarse-to-fine multi-stage refinement
- Leading methods in STSR 2025 achieved Dice 0.967 for tooth segmentation

**Recommended architecture for per-tooth instance segmentation:**
1. **Stage 1**: Full-volume segmentation (foreground teeth vs. background) using DentalSegmentator or TotalSegmentator
2. **Stage 2**: Instance separation — use watershed-based or connected components on the stage 1 mask
3. **Stage 3**: Per-tooth classification to FDI numbering using centroid position heuristics or a classification CNN

**OralSeg (2025):** Open-source, freely available, 3D Slicer plugin. Uses SwinViT + Spatial Mamba backbone. Segments 35 structures including all 32 FDI teeth, maxilla, mandible, and bilateral mandibular canals. DSC 0.8316. Available for non-commercial use.  
URL: [Clinical Oral Investigations 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12464119/)

**TotalSegmentator `teeth` task:** Segments all 32 FDI-numbered teeth individually, plus pulp chambers and inferior alveolar canals. As of v2.5.0 (2025), this is the most complete openly available per-tooth segmentation task.

---

### 6.2 CBCT-specific Preprocessing

CBCT has higher resolution than CT (0.2–0.4mm voxel size) but more noise and beam-hardening artifacts. Preprocessing differences:

- **Target spacing**: 0.3–0.4mm isotropic (vs. 1.0mm for diagnostic CT)
- **HU range**: CBCT uses a similar scale but with more noise; apply mild Gaussian smoothing (σ=0.5) before segmentation
- **Metal artifact**: Dental implants produce severe streaks. Metal Artifact Reduction (MAR) preprocessing significantly improves segmentation in implant cases. DentalSegmentator was shown robust to metallic artifacts even without MAR.
- **Field of view**: CBCT FOV is often limited (e.g., a single-jaw scan); DentalSegmentator handles varying FOV robustly

---

### 6.3 Intraoral Scan Processing (STL/PLY/OBJ)

Intraoral scanners (IOS) produce high-fidelity digital impressions of tooth crowns and gingiva, typically exported as STL or PLY files. These contain the surface geometry of the visible crown only (no roots, no bone).

**Loading intraoral scan files:**
```python
import trimesh

# Load intraoral scan
mesh = trimesh.load('intraoral_scan.stl')
print(f"Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
print(f"Volume: {mesh.volume:.2f} mm³")
print(f"Watertight: {mesh.is_watertight}")

# Separate upper/lower arches by jaw classification
# (requires either landmark-based split or deep learning jaw split classifier)
```

**Common IOS formats:**
- `.stl` — most universal
- `.ply` — supports vertex colors (gingival coloring from some scanners)
- `.obj` — with `.mtx` material for texture
- `.3shape` — proprietary 3Shape format; requires SDK
- `.e3d` — proprietary iTero format

**Gingival tissue separation:** For registration purposes, you typically want crown-only geometry. Some IOS files contain gingival tissue mixed with tooth surfaces. A chromatic classification model (as used in [Pubmed 38002450](https://pubmed.ncbi.nlm.nih.gov/38002450/)) can separate gingival from tooth surfaces using color texture if available.

---

## 7. CT-to-Scan Registration

### 7.1 The Registration Problem

CT provides volumetric bone geometry (including roots, alveolar bone, mandibular canal) but has limited crown resolution in regions with metal. Intraoral scans (IOS) provide high-fidelity crown geometry but no bone. Combining them gives a complete dento-skeletal model essential for:
- Orthognathic surgery planning (jaw repositioning)
- Implant placement
- Fracture reduction planning
- Occlusal analysis

The challenge: CT-derived tooth crowns have lower resolution and more artifacts than IOS, so direct ICP on raw CT surfaces gives poor alignment. The key is to use the shared crown geometry — specifically curvature features on the occlusal surfaces — as registration anchors.

---

### 7.2 ICP (Iterative Closest Point)

**What it does:** Iterative algorithm that alternates between finding closest point correspondences and computing the rigid transformation that minimizes their distance. Standard algorithm in Open3D and other libraries.

**Open3D implementation:**
```python
import open3d as o3d

source = o3d.io.read_point_cloud("ct_crowns.pcd")
target = o3d.io.read_point_cloud("intraoral_scan.pcd")

# Point-to-plane ICP (faster convergence than point-to-point)
reg = o3d.pipelines.registration.registration_icp(
    source, target,
    max_correspondence_distance=0.5,  # mm
    init=rough_transform,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
)
print(f"Fitness: {reg.fitness:.3f}, RMSE: {reg.inlier_rmse:.3f} mm")
```

**Key parameters:**
- `max_correspondence_distance`: start at 2–3mm for rough registration, refine to 0.1–0.3mm
- `TransformationEstimationPointToPlane`: faster convergence than point-to-point
- Initialize with a rough transform from landmark detection to avoid local minima

**ICP limitations:**
- Requires a good initialization (within ~5–10mm and ~30° of correct pose)
- Can fail with metal artifacts on CT-derived crown geometry
- Not robust to partial overlap (IOS typically captures only crown surface)

**Open3D version:** 0.19.0  
**URL:** https://open3d.org  
**License:** MIT  
**Baseline:** ✅ Yes — for fine registration refinement

---

### 7.3 Landmark-Based Registration

**What it does:** Identifies corresponding anatomical landmarks on both modalities and computes the rigid/similarity transform that aligns them.

**Best approach for CT/IOS registration:**
1. Extract upper and lower jaw boundaries from CT using a jaw-split classifier (as in [Pubmed 36921464](https://pubmed.ncbi.nlm.nih.gov/36921464/))
2. Compute crown point clouds from CT
3. Extract crown vertices from IOS
4. Compute Curvature Variance of Neighbor (CVN) metric to identify stable landmarks on tooth occlusal surfaces
5. RANSAC-based coarse alignment on matched landmarks
6. ICP refinement

**Published accuracy:** 0.234 ± 0.019 mm error (better than 0.3mm CBCT voxel size) using RANSAC + ICP on crown-derived landmarks ([Pubmed 38002450](https://pubmed.ncbi.nlm.nih.gov/38002450/)).

---

### 7.4 Deep Learning Registration

**STSR 2025 Challenge (MICCAI 2025):** Introduced a benchmark for semi-supervised CBCT-to-IOS registration. Best-performing methods combined:
- PointNetLK for initial coarse alignment
- Differentiable SVD for gradient-based refinement
- Hybrid neural-classical ICP for final registration

**Deep Reinforcement Learning approach** ([SSRN 2024](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4818934)): Uses negative curvature sampling on IOS vertices + graph convolution network to select sequential rigid transformation actions. Mean absolute error of 1.07 ± 0.41mm (maxilla) and 0.88 ± 0.28mm (mandible). Requires only 0.48M FLOPs.

**Recommendation:** Start with classical RANSAC + ICP (proven, no training data needed). Integrate deep learning registration when you have enough labeled CT-IOS pairs (50+).

---

### 7.5 Face Scan Registration

For surgical simulation requiring face surface, CBCT-to-face-scan registration uses a different approach:
- Extract partial face surface from CBCT skin threshold
- Use 2D facial landmark detection (pre-trained, no medical data required) on two projection views of the face scan
- Convert 2D landmark correspondences to 3D using projection geometry
- ICP final refinement

Reported error: 0.74mm average surface distance ([arXiv 2305.10132](https://arxiv.org/abs/2305.10132))

---

## 8. Fracture Fragment Identification

### 8.1 The Computational Challenge

Identifying discrete fracture fragments from a CT scan is a harder problem than standard segmentation because:
1. The number of fragments is **unknown** a priori (could be 2 to 20+)
2. Fragment boundaries coincide with fracture lines — narrow discontinuities of similar HU value
3. Fragment shapes are highly irregular and non-anatomical
4. Fragments may be displaced, rotated, or overriding

---

### 8.2 Current Approaches

#### Semi-Automatic Segmentation + Connected Components
The most reliable current approach:
1. Threshold CT for bone (HU > 300)
2. Apply `vtkPolyDataConnectivityFilter` or `scipy.ndimage.label` to identify disconnected bone regions
3. Each connected component = candidate fragment
4. Apply size threshold to eliminate small chips and noise

**Limitation:** Requires fracture lines to create complete spatial separation. Incomplete or impacted fractures where fragments still touch bone require manual disambiguation.

#### nnU-Net for Fragment Detection
A 2025 study ([Pubmed 40238181](https://pubmed.ncbi.nlm.nih.gov/40238181/)) trained nnU-Net on 459 mandibular fracture CTs for pixel-level fracture detection:
- Sensitivity >0.93, precision >0.79, specificity >0.80
- 3D-ResNet for fracture severity classification (mean AUC 0.86)
- Data: West China Hospital of Stomatology (2020–2023)

**This is the best published mandibular fracture-specific model.** The weights are not publicly available yet, but the approach is reproducible.

#### DeepLab v3+ for Fragment Segmentation
A tibia/fibula study ([Nature 2023](https://www.nature.com/articles/s41598-023-47706-4)) trained a ResNet-based DeepLab v3+ model:
- Global accuracy 98.92%
- 5–8x faster than expert manual recognition
- Handles up to 12 fracture fragments

The same approach is applicable to mandibular fracture fragments with domain-adapted training data.

#### 3D Puzzle Reconstruction
For pre-operative planning of comminuted fractures, a **puzzle-solving algorithm** ([PMC 8044060](https://pmc.ncbi.nlm.nih.gov/articles/PMC8044060/)) reconstructs the original bone anatomy:
1. Extract fragment surfaces from CT
2. Decompose each fragment surface into cortical sub-regions
3. Match cortical surface features across fragments using descriptors
4. Solve the 3D puzzle to find original anatomic positions (uses contralateral bone as template)

---

### 8.3 Practical Strategy for This Platform

**Phase 1 (Baseline):** Semi-automatic connected components
- Surgeon selects each fragment manually as a region of interest
- System runs connected component labeling within each ROI
- Each component is assigned a unique color and label

**Phase 2 (AI-Assisted):** Fine-tune nnU-Net on your own annotated mandibular fracture cases
- As cases accumulate (target: 50+ annotated fracture CTs), train a per-fragment instance segmentation model
- Use MedSAM2 for interactive refinement

**Phase 3 (Advanced):** Automatic fragment reconstruction
- Implement the cortical surface matching/puzzle algorithm
- Suggest optimal reduction positions based on contralateral anatomy mirror

**Key limitation to communicate:** Impacted fractures and thin cortical chips at fracture surfaces are genuinely difficult for any current AI system. Human surgical judgment is required for final fragment identification.

---

## 9. 3D Visualization Frameworks for Web

### 9.1 Architecture Decision

There are two fundamentally different rendering use cases in this platform:

| Use Case | Best Choice |
|----------|-------------|
| CT/CBCT volume rendering, MPR, DICOM viewing | **Cornerstone3D** |
| Patient-specific surgical mesh visualization, simulation | **react-three-fiber** |

These two should coexist in the application — Cornerstone3D for the CT review panel, react-three-fiber for the surgical planning 3D scene.

---

### 9.2 react-three-fiber (R3F) + drei

**What it does:** React renderer for Three.js. Declarative, component-based Three.js with full React state management, hooks, and the Drei helper library.

**Why it fits for surgical mesh visualization:**
- Bone meshes (STL/OBJ/GLB) are just Three.js geometries — load with `useGLTF` or `STLLoader`
- Individual fragment highlighting: swap materials per mesh on hover/click
- Fragment repositioning: use `TransformControls` (from drei) to drag/rotate individual fragments
- Measurements: raycaster + custom annotations
- Implant overlay: load implant OBJ files from manufacturer catalog and position in the scene
- AR/VR via `@react-three/xr`
- Performance: 60fps for scenes with 10–20 bone meshes

**Key packages:**
```json
{
  "@react-three/fiber": "^8.x",
  "@react-three/drei": "^9.x",
  "three": "^0.170.x"
}
```

**Typical surgical scene:**
```jsx
import { Canvas } from '@react-three/fiber'
import { OrbitControls, TransformControls, useGLTF } from '@react-three/drei'
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader'

function BoneFragment({ url, color, selected, onSelect }) {
  const { geometry } = useLoader(STLLoader, url)
  return (
    <mesh geometry={geometry} onClick={onSelect}>
      <meshPhongMaterial color={selected ? '#ff6600' : color} />
    </mesh>
  )
}

export function SurgicalPlanner({ fragments }) {
  const [selected, setSelected] = useState(null)
  return (
    <Canvas camera={{ position: [0, 0, 150], fov: 60 }}>
      <ambientLight intensity={0.5} />
      <directionalLight position={[100, 100, 100]} />
      {fragments.map(f => (
        <BoneFragment key={f.id} url={f.url} color={f.color}
          selected={selected === f.id}
          onSelect={() => setSelected(f.id)} />
      ))}
      {selected && <TransformControls object={...} />}
      <OrbitControls />
    </Canvas>
  )
}
```

**Limitations:**
- Not DICOM-aware — requires meshes pre-converted to GLB/STL/OBJ
- No built-in DICOM windowing, MPR, or annotation tools
- Large mesh LOD (level of detail) management requires manual implementation

**Baseline:** ✅ Yes — for the 3D surgical simulation panel

---

### 9.3 Cornerstone3D (re-emphasis in web context)

For the CT viewing panel alongside the surgical simulation:

```jsx
import { Enums, RenderingEngine, volumeLoader } from '@cornerstonejs/core'
import * as cornerstoneDICOMImageLoader from '@cornerstonejs/dicom-image-loader'

// Progressive DICOM streaming
const volumeId = 'cornerstoneStreamingImageVolume:myCT'
const volume = await volumeLoader.createAndCacheVolume(volumeId, {
  imageIds: dicomSeriesImageIds,
})
await volume.load()

// Render MPR (axial/sagittal/coronal)
const renderingEngine = new RenderingEngine('myRenderingEngine')
renderingEngine.setViewports([
  { viewportId: 'AXIAL', type: Enums.ViewportType.ORTHOGRAPHIC, ... },
  { viewportId: 'SAGITTAL', type: Enums.ViewportType.ORTHOGRAPHIC, ... },
  { viewportId: 'CORONAL', type: Enums.ViewportType.ORTHOGRAPHIC, ... },
  { viewportId: '3D', type: Enums.ViewportType.VOLUME_3D, ... }
])
```

**Baseline:** ✅ Yes — run alongside react-three-fiber; two separate canvas elements

---

### 9.4 Three.js (Bare)

**When to use:** Only if you need raw control and minimal bundle size for a simple mesh viewer component. For surgical planning with fragment manipulation and measurement tools, react-three-fiber is strictly better.

---

### 9.5 Web Visualization Performance Checklist

- [ ] **Mesh polygon limit**: Deliver meshes at ≤100K triangles for web. Decimate on the server before delivery.
- [ ] **Format**: Use GLB (binary glTF) instead of STL — 4x smaller, includes normals/materials, loads faster
- [ ] **Compression**: Enable Draco geometry compression in GLB export for additional 10x size reduction
- [ ] **Streaming volumes**: Stream DICOM as progressive JPEG-2000 tiles (Cornerstone3D handles this natively)
- [ ] **WebWorkers**: Move mesh loading off the main thread using `@react-three/fiber`'s `useLoader` with Suspense
- [ ] **LOD**: Implement coarse/fine LOD — show 10K triangle mesh for overview, switch to 100K on zoom

---

## 10. ML-Ready Data Pipeline Design

### 10.1 Overview

The platform must simultaneously serve surgeons in clinical workflows and collect training data for model improvement. These are not separate pipelines — the same annotation work that improves surgical plans should flow directly into model retraining.

---

### 10.2 Data Collection Architecture

```
Clinical Input
  DICOM CT/CBCT
        ↓
  De-identification (deid + pixel PHI)
        ↓
  Volume Preprocessing (SimpleITK)
        ↓
  Automatic Segmentation (TotalSegmentator + DentalSegmentator)
        ↓
  Surgeon Review in MONAI Label / OHIF
        ↓ (corrections stored as DICOM SEG via highdicom)
  Annotated Dataset Repository
        ↓
  Active Learning Scoring (MONAI Label uncertainty sampling)
        ↓
  Prioritized Annotation Queue
        ↓
  nnU-Net / MONAI Core Retraining
        ↓
  Model Validation + Deployment
```

---

### 10.3 MONAI Label for Active Learning Annotation

MONAI Label runs as a server process that:
1. Holds a pool of unlabeled CT volumes
2. Scores them for uncertainty (entropy, variance via MC-dropout, or TTA)
3. Serves the most informative case to the annotator
4. Accepts corrected annotations, retrains the model incrementally

**Integration options:**
- **3D Slicer plugin**: Radiologist opens 3D Slicer, connects to MONAI Label server, auto-segmentation appears, they correct, submit
- **OHIF plugin**: Web-based annotation (no installation)

**Active Learning strategies in MONAI Label:**
- Epistemic uncertainty (MC-dropout based entropy/variance)
- Aleatoric uncertainty (Test Time Augmentation, Volume Variation Coefficient)

**Reported impact:** Up to 75% annotation time reduction vs. manual annotation from scratch. Annotation time on complex 3D structures reduced from 20+ minutes to under 2 minutes with DeepEdit interactive model.

**URL:** https://github.com/Project-MONAI/MONAILabel  
**Reference:** [MONAI Label, Medical Image Analysis 2024](https://doi.org/10.1016/j.media.2024.103207)

---

### 10.4 Storage Format Standards

| Data Type | Format | Rationale |
|-----------|--------|-----------|
| Raw DICOM | `.dcm` in DICOM directory structure | Preserve original; never modify |
| De-identified input | `.nii.gz` (NIfTI) | Standard for ML; preserves spacing/orientation |
| Segmentation labels | `.nii.gz` (integer label map) | nnU-Net native format |
| DICOM segmentation (PACS) | DICOM SEG via highdicom | Interoperability with clinical systems |
| Mesh outputs | `.stl` (surgical use), `.glb` (web display) | STL for 3D printing; GLB for browser |
| Landmark annotations | `.json` with physical coordinates | Simple, portable, version-controllable |
| Model checkpoints | PyTorch `.pth` / MONAI MAP `.zip` | Versioned in model registry |

---

### 10.5 nnU-Net Dataset Organization

```
nnUNet_raw/
  Dataset001_CraniofacialFractures/
    imagesTr/
      case_001_0000.nii.gz    # CT volume
    imagesTs/
      case_100_0000.nii.gz
    labelsTr/
      case_001.nii.gz         # integer mask: 1=mandible, 2=fragment_A, 3=fragment_B ...
    dataset.json              # lists channel names, labels, spacing statistics
```

The `dataset.json` declares:
```json
{
  "channel_names": {"0": "CT"},
  "labels": {"background": 0, "mandible": 1, "fragment_A": 2, "fragment_B": 3},
  "numTraining": 100,
  "file_ending": ".nii.gz"
}
```

---

### 10.6 Model Registry and Versioning

- **MLflow** or **MONAI Model Registry**: Track model versions, metrics, and dataset pointers
- **DVC (Data Version Control)**: Version large binary files (NIfTI volumes, model weights) alongside Git
- **Predetermined Change Control Plan (PCCP)**: Required by FDA for SaMD models that will be updated after clearance. Define in advance what types of changes are permitted without new 510(k) submission.

---

### 10.7 Feedback Loop Architecture

```
Production Inference
   ↓
Surgeon accepts/corrects output
   ↓
Store correction delta (not just final mask — store which voxels changed)
   ↓
Compute correction effort metric (proxy for model uncertainty)
   ↓
Flag cases above effort threshold for inclusion in next training batch
   ↓
Human validation → add to nnUNet_raw → retrain
```

This creates a continuously improving model where the hardest cases for the model become training priorities.

---

## 11. Privacy, HIPAA, and Deployment

### 11.1 Regulatory Framework

**HIPAA Technical Safeguards** relevant to imaging AI:
- **Encryption at rest and in transit**: All DICOM files, NIfTI volumes, and model inputs/outputs must be encrypted
- **Audit logging**: Every access to patient data — including AI inference — must be logged with user, timestamp, and action
- **Access control**: Role-based access; surgeons see only their cases; researchers see only de-identified data
- **Business Associate Agreement (BAA)**: Required with any cloud provider (AWS, GCP, Azure) before PHI touches their infrastructure

**FDA SaMD Classification:** Diagnostic AI software that analyzes medical images to detect, diagnose, or characterize disease is regulated as a Software as a Medical Device (SaMD). For a surgical planning platform:
- Surgical planning decision support → likely Class II → 510(k) pathway
- If marketing as purely visualization/annotation only → may qualify for Enforcement Discretion
- **Cybersecurity requirements (Oct 2023)**: SBOM, risk assessment, post-market monitoring plan mandatory for all premarket submissions
- **PCCP**: If model will be updated after clearance, a Predetermined Change Control Plan is required

---

### 11.2 DICOM De-identification Pipeline

Mandatory before any data leaves the clinical environment:

**Layer 1: Header de-identification**
```python
from deid.dicom import get_files, replace_identifiers
from deid.config import load_config

config = load_config("safe_harbor.yml")  # remove all 18 HIPAA identifiers
cleaned = replace_identifiers(dicom_files, deid=config)
```

**Layer 2: Pixel data PHI (burned-in text)**
- Use Amazon Rekognition + Textract, or a vision model (e.g., CRAFT text detector)
- Critical for scout images, secondary captures, and some CBCT preview images

**Layer 3: Re-linkage protection**
- Replace patient UIDs with pseudonymous internal IDs
- Store the mapping in an encrypted, access-controlled key table
- Never expose the mapping to research pipelines

**Standard:** DICOM PS 3.15 Annex E (DICOM de-identification standard)

---

### 11.3 On-Premise vs. Cloud Deployment

| Factor | On-Premise | Cloud (AWS/GCP/Azure) |
|--------|-----------|----------------------|
| PHI control | Full control | BAA required; HIPAA-eligible services |
| PACS integration | Direct DICOM C-STORE | DICOM TLS gateway required |
| GPU for AI inference | Fixed capacity | Auto-scaling |
| Cost | High CapEx | Pay-per-use OpEx |
| Disaster recovery | Self-managed | Managed (S3, RDS backups) |
| Regulatory | Easier audit | More documentation overhead |
| Model update latency | Direct deploy | CI/CD pipeline |

**Recommendation for early-stage platform:**
- **Production clinical data**: On-premise or private cloud (single-tenant AWS/Azure) with BAA
- **Research/training data** (de-identified): Cloud preferred (S3 + SageMaker or similar)
- **AI inference**: GPU server on-premise for surgical sessions (latency-sensitive); cloud for batch processing

---

### 11.4 Secure DICOM Architecture

```
Hospital PACS
    ↓ DICOM C-STORE over TLS
DICOM Router (Orthanc or dcm4chee)
    ↓ routing rules (study type, anatomy)
De-identification Service (deid pipeline)
    ↓ pseudonymized DICOM
AI Processing Server (GPU, on-premise)
    ├─ TotalSegmentator / DentalSegmentator
    ├─ Mesh extraction (VTK)
    └─ Results as DICOM SEG (highdicom)
    ↓ DICOM C-STORE back to PACS
PACS / DICOMweb Server
    ↓ WADO-RS
Web Application (Cornerstone3D + react-three-fiber)
```

**Orthanc:** Lightweight, open-source DICOM server with DICOMweb support. Recommended for self-hosted deployment.  
URL: https://www.orthanc-server.com  
License: GPL v3

---

### 11.5 HIPAA-Compliant Cloud Architecture (if cloud)

For AWS deployment:
- Use **AWS HealthImaging** for DICOM storage (HIPAA-eligible, BAA available)
- Use **AWS SageMaker** for model training/inference (HIPAA-eligible with BAA)
- All S3 buckets: encryption at rest (SSE-S3 or SSE-KMS), versioning, access logging
- VPC isolation with private subnets for AI processing
- CloudTrail for all API audit logging
- No PHI in Lambda function logs, CloudWatch metrics, or API Gateway access logs

**Reference:** [AI in Medical Imaging: HIPAA-Compliant Implementation (2026)](https://www.tactionsoft.com/blog/ai-medical-imaging-hipaa-compliant-implementation/)

---

## 12. Recommended Baseline Stack Summary

### 12.1 Phase 1 Baseline (Build First)

| Component | Tool | Version | Why |
|-----------|------|---------|-----|
| DICOM parsing | pydicom | 3.0.x | Universal DICOM access |
| Volume reconstruction | SimpleITK | 2.4.x | Volume assembly, reorientation, resampling |
| De-identification | pydicom/deid | Latest | HIPAA compliance |
| Craniofacial segmentation | TotalSegmentator | v2.5.0 | Immediate skull/mandible/teeth coverage |
| Dental CT/CBCT segmentation | DentalSegmentator | nnUNet v2.2 | Best open-source model for DMF |
| Mesh extraction | VTK 9.x + PyVista | 0.44.x | STL generation from segmentation masks |
| Mesh processing | trimesh | 4.x | Validation, repair, format conversion |
| CT web viewer | Cornerstone3D | 2.x | Medical-grade DICOM rendering |
| Surgical mesh viewer | react-three-fiber + drei | 8.x / 9.x | Interactive 3D surgical simulation |
| CT-to-IOS registration | Open3D (ICP) | 0.19.x | CT-to-intraoral scan alignment |
| DICOM server | Orthanc | 24.x | Self-hosted DICOMweb + PACS |
| Annotation pipeline | MONAI Label | 1.x | Active learning annotation server |
| ML framework | nnU-Net v2 | Latest | Fine-tuning custom segmentation models |
| Privacy | deid + DICOM PS 3.15 | — | De-identification |

---

### 12.2 Phase 2 (Add After Launch)

| Component | Tool | When |
|-----------|------|------|
| Interactive segmentation | MedSAM2 | When building annotation loop |
| DICOM SEG storage | highdicom | When storing AI outputs back to PACS |
| Fracture fragment AI | Custom nnU-Net (mandible fracture task) | When 50+ annotated fracture cases accumulated |
| DL CT-IOS registration | PointNetLK / custom | When 50+ paired CT-IOS cases available |
| TMJ condyle model | Custom nnU-Net (cascaded) | When 150+ annotated CBCT cases |
| MONAI Deploy | MONAI Deploy SDK | When moving to clinical production |
| FDA submission | PCCP + 510(k) documentation | Before commercial launch |

---

### 12.3 Stack Dependencies

```
pydicom ← SimpleITK ← MONAI Core ← nnU-Net v2
           ↓
           VTK → PyVista → trimesh → Mesh delivery API
           ↓
           highdicom → DICOM SEG → Orthanc → DICOMweb
                                              ↓
                                    Cornerstone3D (browser)
                                    react-three-fiber (browser)
```

---

### 12.4 Hardware Requirements

| Workload | GPU | RAM | Storage |
|---------|-----|-----|---------|
| TotalSegmentator inference (1 case) | NVIDIA RTX 3080+ (8GB VRAM) | 16GB | — |
| DentalSegmentator inference | NVIDIA RTX 3080+ | 16GB | — |
| nnU-Net training (craniofacial) | A100 or 2× RTX 3090 | 64GB | 500GB NVMe |
| MedSAM2 inference | RTX 3080+ | 16GB | — |
| Production inference server | NVIDIA A10G or T4 | 32GB | — |
| CPU-only inference (fallback) | N/A | 32GB | — |

---

*Document version: 1.0 | Last updated: 2025*

**Key sources:**
- [Innolitics Medical Imaging Best Practices (2025)](https://innolitics.com/articles/medical-imaging-best-practices/)
- [TotalSegmentator, Radiology AI 2023](https://pubs.rsna.org/doi/abs/10.1148/ryai.230024)
- [DentalSegmentator, Journal of Dentistry 2024](https://doi.org/10.1016/j.jdent.2024.105130)
- [MONAI Label, Medical Image Analysis 2024](https://doi.org/10.1016/j.media.2024.103207)
- [nnU-Net, Nature Methods 2021](https://doi.org/10.1038/s41592-020-01008-z)
- [ToothFairy Challenge, IEEE TMI 2025](https://doi.org/10.1109/TMI.2024.3523096)
- [highdicom, Journal of Digital Imaging 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9712874/)
- [CT/IOS registration, Comput Methods Programs Biomed 2023](https://pubmed.ncbi.nlm.nih.gov/36921464/)
- [Fracture fragment segmentation, Nature 2023](https://www.nature.com/articles/s41598-023-47706-4)
- [AI-Assisted dental segmentation, BMC Oral Health 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11887095/)
- [MedSAM2 (2025)](https://medsam2.github.io)
- [HIPAA-Compliant AI Implementation (2026)](https://www.tactionsoft.com/blog/ai-medical-imaging-hipaa-compliant-implementation/)
- [Cornerstone3D docs](https://cornerstonejs.org)
- [Open3D ICP registration](https://www.open3d.org/docs/latest/tutorial/Basic/icp_registration.html)
- [MICCAI STSR 2025 Challenge](https://arxiv.org/abs/2512.02867)
