# Open-Source Ecosystem Evaluation: AI-Native Craniofacial Surgical Planning Platform

**Research Date:** April 2026  
**Purpose:** Component-by-component evaluation for technology selection, with Phase 1 (baseline) vs. later-phase recommendations.

---

## Quick Summary: Baseline (Phase 1) Stack

| Layer | Tool | Rationale |
|---|---|---|
| ML backbone | PyTorch 2.11 | Industry standard, required by all other ML tools |
| ML framework | MONAI 1.5.2 | Medical imaging transforms, loaders, losses |
| Segmentation | nnU-Net v2 | Auto-configured; best OOB performance on medical data |
| Pre-trained CMF segmentation | TotalSegmentator 2.11 | skull, mandible, teeth, sinuses, muscles — ready to run |
| DICOM I/O | pydicom 2.4.5 | Read/write DICOM files |
| Image processing | SimpleITK v2.5.3 | Registration, filtering, resampling |
| NIfTI I/O | nibabel 5.4.1 | NIfTI read/write for model pipeline |
| DICOM structured outputs | highdicom 0.26.1 | Write DICOM-SEG, SR, RT-Struct |
| 3D mesh + volume | VTK/PyVista | Volume rendering, STL export, mesh ops |
| Mesh manipulation | trimesh 4.11.5 | Boolean ops, STL/OBJ/3MF I/O, 3D-print prep |
| Image processing utils | scikit-image 0.26.0 | Marching cubes, morphology |
| Web DICOM viewer | OHIF v3.12.0 + Cornerstone3D | Zero-footprint viewer with segmentation overlay |
| API framework | FastAPI 0.135.3 | REST API for all backend services |
| Async processing | Celery 5.x + Redis | Long-running segmentation jobs |
| Database | PostgreSQL | Patient/case data |
| Object storage | MinIO | DICOM/NIfTI/STL file storage (S3-compatible) |
| Containerization | Docker / docker-compose | Reproducible deployments |
| CI/CD | GitHub Actions | Automated testing and build |
| Model serving | TorchServe 0.12 | PyTorch model deployment |

---

## Category 1: Core ML/DL Frameworks

### 1. MONAI (Medical Open Network for AI)

| Attribute | Detail |
|---|---|
| **Version** | 1.5.2 (Jan 29, 2026) |
| **License** | Apache-2.0 |
| **Repository** | [github.com/Project-MONAI/MONAI](https://github.com/Project-MONAI/MONAI) |
| **Stars / Forks** | 8.1k / 1.5k |

**What it does:** MONAI is the de facto standard ML toolkit for healthcare imaging built on PyTorch. It provides medical-imaging-specific transforms (random affine, intensity normalization, patch-based sampling), data loaders, loss functions (Dice, Tversky, focal), evaluation metrics, network architectures (SegResNet, UNETR, SwinUNETR), and a growing model zoo (MONAI Model Zoo).

**How it fits:** MONAI is the glue layer between raw DICOM/NIfTI volumes and PyTorch training/inference. It replaces the need to write medical-imaging-specific preprocessing from scratch. The MONAI Bundle format packages pretrained models with their configs, making deployment reproducible.

**Key capabilities:**
- Medical-imaging transforms pipeline (composable, GPU-accelerated)
- Patch-based training for large 3D volumes (sliding window inference)
- Multi-GPU / multi-node training via PyTorch DDP
- Domain-specific network architectures (SegResNet, DynUNet, SwinUNETR, UNETR)
- MONAI Label: active-learning annotation server (connects to 3D Slicer / OHIF)
- MONAI Deploy: packaging for clinical deployment
- Model Zoo with pretrained models

**Limitations:**
- Steep learning curve for custom architectures outside the provided zoo
- MONAI Label requires additional server setup
- Some modules still maturing (generative, federated learning)

**CMF relevance:** MONAI Label can be connected to a CBCT annotation workflow, accelerating ground truth generation. Pretrained bundles exist for skull/bone segmentation.

**Phase:** ✅ **Phase 1 — Baseline.** Required foundation for all model training and inference.

---

### 2. PyTorch

| Attribute | Detail |
|---|---|
| **Version** | 2.11.0 (Mar 23, 2026) |
| **License** | BSD-style (PyTorch License) |
| **Repository** | [github.com/pytorch/pytorch](https://github.com/pytorch/pytorch) |

**What it does:** The primary ML framework. Provides autograd, neural network modules, optimizers, GPU acceleration via CUDA, and the torch.compile/export ecosystem for optimized deployment.

**How it fits:** Every model in this stack (MONAI, nnU-Net, TotalSegmentator, custom CMF networks) runs on PyTorch. It is the non-negotiable compute foundation.

**Key capabilities:**
- Dynamic computation graphs (easier debugging than static graphs)
- torch.compile for production speedups (replaces TorchScript in 2.10+)
- CUDA, ROCm, MPS (Apple Silicon) backends
- Distributed training (DDP, FSDP)
- ONNX export for cross-framework deployment
- Active ecosystem: 100k+ PyPI packages depend on it

**Limitations:**
- torch.compile / torch.export ecosystem is still maturing (TorchScript formally deprecated in 2.10)
- Memory management for large 3D medical volumes requires careful attention
- Version pinning critical: nnU-Net, MONAI, TotalSegmentator all have PyTorch version requirements

**Phase:** ✅ **Phase 1 — Baseline.** Non-negotiable ML compute backbone.

---

### 3. nnU-Net v2

| Attribute | Detail |
|---|---|
| **Version** | v2 (continuous; last major release Apr 2024 — Residual Encoder UNets) |
| **License** | Apache-2.0 |
| **Repository** | [github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) |
| **Stars** | ~18k |

**What it does:** nnU-Net is a self-configuring segmentation framework that automatically designs and trains optimal U-Net variants for any biomedical dataset. It analyzes the dataset fingerprint (spacing, intensity, size), then configures preprocessing, patch size, network architecture, training schedule, and post-processing without manual tuning.

**How it fits:** The primary model-training engine for any custom CMF segmentation tasks not covered by TotalSegmentator. DentalSegmentator (mandible, teeth, maxillary sinuses) is built on nnU-Net v2. Use nnU-Net to fine-tune or train from scratch on your institution's CMF CBCT cases.

**Key capabilities:**
- Automated dataset fingerprinting and pipeline configuration
- 2D, 3D fullres, and 3D cascade configurations automatically selected
- Sliding-window inference for large volumes
- Ensemble prediction across configurations
- Residual encoder UNets (state of the art for many tasks)
- Extensively validated: won/top-ranked in dozens of MICCAI challenges
- Supports custom dataset addition with simple JSON configuration

**Limitations:**
- Designed for supervised segmentation only (no native self-supervised or few-shot)
- Training from scratch requires substantial labeled data (typically 50–200+ cases for good performance)
- Not optimized for real-time inference (inference latency 30–120s on GPU for full CT)
- v1 → v2 migration required for newer features

**CMF relevance:** DentalSegmentator (470-scan trained nnU-Net model) delivers state-of-the-art performance on CT/CBCT: mandible, teeth (upper/lower), maxilla+upper skull, mandibular canal.

**Phase:** ✅ **Phase 1 — Baseline.** Essential for training and serving CMF-specific segmentation models.

---

### 4. TotalSegmentator

| Attribute | Detail |
|---|---|
| **Version** | v2.11.0 (weights: v2.5.0, Jan 2025) |
| **License** | Apache-2.0 (most tasks); some subtasks require free non-commercial license |
| **Repository** | [github.com/wasserth/TotalSegmentator](https://github.com/wasserth/TotalSegmentator) |

**What it does:** Provides a single-command tool for robust segmentation of 100+ anatomical structures from CT (117 classes) and MR (50 classes), plus 30+ additional specialized tasks. Built on nnU-Net. Critically for this platform, it includes dedicated **craniofacial and head/neck subtasks** with direct clinical utility.

**How it fits:** Provides out-of-box segmentation for anatomy that would otherwise require months of training. Delivers an immediate Phase 1 segmentation capability covering skull, mandible, teeth, sinuses, masticatory muscles, and head/neck vasculature.

**Key CMF/head subtasks (all Apache-2.0):**

| Task | Structures |
|---|---|
| `craniofacial_structures` | mandible, teeth_lower, teeth_upper, skull, head, sinus_maxillary, sinus_frontal |
| `teeth` | 32 individual teeth (FDI numbering), upper/lower jawbone, inferior alveolar canals, maxillary sinuses, pulp chambers |
| `head_glands_cavities` | eyes, lenses, optic nerves, parotid glands, submandibular glands, nasal cavity, hard/soft palate, pharynx |
| `head_muscles` | masseter, temporalis, lateral/medial pterygoid, tongue, digastric |
| `headneck_bones_vessels` | zygomatic arch, styloid process, thyroid/cricoid cartilage, hyoid, internal carotid/jugular |
| `headneck_muscles` | sternocleidomastoid, trapezius, scalenes, platysma, pharyngeal constrictors |
| `oculomotor_muscles` | all extraocular muscles, optic nerves, eyeballs |
| `total` (117 classes, CT) | skull, brain, vertebrae C1–C7, clavicles, carotid arteries, + full body |

**Key capabilities:**
- Python API: `from totalsegmentator.python_api import totalsegmentator`
- CLI and Python: input NIfTI or DICOM folder/zip
- Output: NIfTI per-class, multilabel NIfTI, DICOM-SEG, DICOM RT-STRUCT
- `--statistics`: volume in mm³ per structure
- `--radiomics`: radiomics features
- `--preview`: 3D rendering preview
- `--fast`: 3mm resolution fast mode
- `--roi_subset`: run only specific classes

**Limitations:**
- Trained on CT; MR models have fewer classes
- Individual tooth labeling (FDI) requires a separate `teeth` task — not part of `total`
- Large model weights (~6GB for all tasks); GPU recommended for practical speeds
- Not a medical device; requires regulatory strategy if used clinically
- Craniofacial task was published/validated but needs case-by-case accuracy verification for unusual pathology

**Phase:** ✅ **Phase 1 — Baseline.** Provides immediate, high-quality craniofacial anatomy segmentation without training. The `craniofacial_structures`, `teeth`, `head_muscles`, and `headneck_bones_vessels` tasks together deliver surgical-planning-level anatomical coverage.

---

## Category 2: Medical Imaging Libraries

### 5. pydicom

| Attribute | Detail |
|---|---|
| **Version** | 2.4.5 (Mar 20, 2026) |
| **License** | MIT |
| **Repository** | [github.com/pydicom/pydicom](https://github.com/pydicom/pydicom) |

**What it does:** Pure-Python library for reading, modifying, and writing DICOM files. Provides Pythonic access to DICOM datasets, tags, and pixel data.

**How it fits:** The entry point for all incoming DICOM data (CT, CBCT, CBCT-derived scans). Pydicom parses DICOM headers and pixel arrays; it is the low-level DICOM layer before volume reconstruction in SimpleITK.

**Key capabilities:**
- Read/write all DICOM SOP classes (CT, MR, CBCT, SC, SR, RT, etc.)
- Access and modify any DICOM tag
- Pixel data access via `Dataset.pixel_array` (returns NumPy array)
- Compression/decompression: JPEG-LS, JPEG 2000, RLE, JPEG
- DICOM sequence and nested dataset support
- Works standalone (no C libraries required)

**Limitations:**
- Does not handle SOP class semantics (use highdicom for structured outputs)
- No native DICOM networking (pynetdicom handles C-STORE, C-FIND, C-MOVE)
- DICOM-to-volume stacking and sorting requires custom code or SimpleITK

**Phase:** ✅ **Phase 1 — Baseline.** Low-level DICOM I/O is required from day one.

---

### 6. SimpleITK / ITK

| Attribute | Detail |
|---|---|
| **Version** | SimpleITK v2.5.3 (Nov 21, 2025) |
| **License** | Apache-2.0 |
| **Repository** | [github.com/SimpleITK/SimpleITK](https://github.com/SimpleITK/SimpleITK) |

**What it does:** SimpleITK is a simplified Python/R/Java/C# interface to ITK (Insight Segmentation and Registration Toolkit), the gold-standard image processing library for medical imaging. ITK provides hundreds of image filters, segmentation algorithms, and multi-modal registration methods.

**How it fits:** Handles the critical steps between raw DICOM pixels and model-ready tensors:
1. DICOM-to-volume reconstruction (stacking, spacing, direction cosines)
2. Resampling to isotropic resolution
3. Intensity normalization and windowing
4. Rigid/affine/deformable registration (CT-to-CT follow-up, CBCT-to-CT fusion)
5. Post-segmentation morphological operations

**Key capabilities:**
- DICOM series reading with correct orientation and physical spacing
- Image resampling with various interpolators (linear, BSpline, NearestNeighbor for labels)
- Extensive registration framework: demons, BSpline, rigid, similarity
- Image arithmetic, thresholding, morphological operations
- Connected component analysis, distance transforms
- 2D, 3D, 4D support
- Multi-language bindings (Python first-class)

**Limitations:**
- API is lower-level than scipy; requires familiarity with ITK conventions
- Does not expose all ITK functionality (some advanced filters require raw ITK/C++)
- No GPU acceleration (CPU only; use MONAI for GPU-accelerated transforms)
- Large dependency (~300MB)

**Phase:** ✅ **Phase 1 — Baseline.** Essential for DICOM-series-to-volume reconstruction and registration.

---

### 7. nibabel

| Attribute | Detail |
|---|---|
| **Version** | 5.4.1 (Mar 10, 2026) |
| **License** | MIT |
| **Repository** | [github.com/nipy/nibabel](https://github.com/nipy/nibabel) |

**What it does:** Read/write access to neuroimaging and medical imaging file formats, including NIfTI-1, NIfTI-2, ANALYZE, MINC, CIFTI-2, GIFTI, and FreeSurfer formats. Returns data as NumPy arrays.

**How it fits:** The model pipeline stores intermediate volumes as NIfTI (.nii.gz). nibabel reads/writes NIfTI files with correct affine transforms (voxel-to-world mapping). Used everywhere data passes between MONAI/nnU-Net and disk storage.

**Key capabilities:**
- NIfTI-1/2 read/write with full header support
- Affine/world-space coordinate transforms
- Efficient memory-mapped access for large volumes
- GIFTI surface file support (useful for mesh-based results)
- MINC, ANALYZE, AFNI BRIK/HEAD, Philips PAR/REC support

**Limitations:**
- Limited DICOM support (use pydicom or SimpleITK for DICOM)
- No GPU acceleration; NumPy-based
- Less feature-rich than SimpleITK for processing operations

**Phase:** ✅ **Phase 1 — Baseline.** Required for NIfTI I/O throughout the ML pipeline.

---

### 8. highdicom

| Attribute | Detail |
|---|---|
| **Version** | 0.26.1 (Aug 6, 2025) |
| **License** | MIT |
| **Repository** | [github.com/ImagingDataCommons/highdicom](https://github.com/ImagingDataCommons/highdicom) |

**What it does:** High-level Python abstractions for creating and reading structured DICOM objects derived from images — the layer that turns ML outputs into standards-compliant DICOM for PACS integration.

**How it fits:** After segmentation, the results must be stored back in DICOM format for clinical workflow integration. highdicom handles the creation of DICOM Segmentation objects (SEG), RT Structure Sets (for radiation oncology workflows), Structured Reports (measurements), and Parametric Maps.

**Key capabilities:**
- **DICOM Segmentation (SEG):** Convert NumPy label maps → DICOM-SEG (fractional or binary)
- **RT Structure Set (RTSTRUCT):** Convert contours → RT-Struct
- **Structured Report (SR):** Store measurements, qualitative evaluations, vector graphic annotations
- **Parametric Map:** Store quantitative pixel maps
- **Secondary Capture, Key Object Selection, Presentation States**
- Reading + filtering existing derived DICOM objects
- Spatial arrangement and pixel transform handling

**Limitations:**
- v0.x — API may have breaking changes; not yet v1.0
- Smaller community than pydicom; fewer examples in the wild
- Focused on derived objects; not a general DICOM I/O library

**Phase:** ✅ **Phase 1 — Baseline.** Required for PACS-compatible output and clinical workflow integration from day one.

---

## Category 3: 3D Processing

### 9. VTK / vtkpython

| Attribute | Detail |
|---|---|
| **Version** | 9.6.1 (PyPI, Mar 2026) |
| **License** | BSD-3-Clause |
| **Repository** | [vtk.org](https://vtk.org) / PyPI: `vtk` |

**What it does:** The Visualization Toolkit — a comprehensive library for 3D graphics, image processing, and scientific visualization. VTK is the engine behind 3D Slicer, ParaView, and many clinical visualization systems.

**How it fits:** VTK handles the 3D pipeline from segmentation label map to clinical visualization: marching cubes (label → mesh), mesh decimation and smoothing, boolean operations, clipping (virtual osteotomy cuts), and volume rendering for CT display.

**Key capabilities:**
- Volume rendering (ray casting, GPU-accelerated)
- Surface reconstruction from label maps (marching cubes, contour filter)
- Mesh processing: decimation, smoothing, normals, subdivision
- Boolean operations (vtkBooleanOperationPolyDataFilter)
- Image processing: resampling, filtering, morphology
- 3D visualization with actors, renderers, render windows
- STL, OBJ, VTK polydata read/write
- DICOM reader (supplemental; use SimpleITK for production)
- Python bindings via vtkpython (included in `vtk` pip package)

**Limitations:**
- Pythonic API is low-level and verbose; use PyVista wrapper for rapid development
- Pure Python VTK is CPU-rendering only; GPU requires OpenGL context (headless servers need OSMesa or EGL)
- Large dependency (~200MB)
- Some operations (large boolean ops) can be numerically unstable; trimesh/Manifold3D more robust

**Phase:** ✅ **Phase 1 — Baseline.** Core 3D rendering and mesh pipeline engine.

---

### 10. Open3D

| Attribute | Detail |
|---|---|
| **Version** | v0.19 (Jan 8, 2025) |
| **License** | MIT |
| **Repository** | [github.com/isl-org/Open3D](https://github.com/isl-org/Open3D) |

**What it does:** Modern library for 3D data processing with a focus on point clouds, meshes, scene reconstruction, and 3D ML. Provides ICP (Iterative Closest Point) registration, TSDF integration, and GPU-accelerated operations.

**How it fits:** Open3D is best suited for point cloud operations and surface registration — particularly for aligning 3D intraoral scans (IOS) with CBCT volumes, or for ICP-based model-to-model registration in treatment outcome analysis.

**Key capabilities:**
- Point cloud processing: ICP, colored-ICP, global registration (RANSAC, FPFH)
- Mesh processing: simplification, remeshing, Poisson surface reconstruction
- TSDF volumetric integration (for depth-camera reconstruction)
- 3D ML support (with PyTorch/TensorFlow)
- GPU acceleration via CUDA
- Physically based rendering (PBR) with built-in GUI viewer
- Python and C++ APIs

**Limitations:**
- v0.x — pre-1.0, API may change
- Less mature DICOM/NIfTI integration; primarily STL/PLY/PCD
- ICP can fail without good initialization; requires robust initial alignment
- Less commonly used in pure medical imaging pipelines vs. VTK

**CMF relevance:** Useful for IOS-to-CBCT registration (aligning digital dental models to CT bone), surface deviation analysis post-operatively, and any depth-camera intraoperative applications.

**Phase:** ⏩ **Phase 2.** Valuable for IOS scan integration and surface registration workflows, but not required for the core CT/CBCT pipeline in Phase 1.

---

### 11. trimesh

| Attribute | Detail |
|---|---|
| **Version** | 4.11.5 (Mar 25, 2026) |
| **License** | MIT |
| **Repository** | [github.com/mikedh/trimesh](https://github.com/mikedh/trimesh) |

**What it does:** Pure-Python library for loading, processing, and analyzing triangular meshes. The most Pythonic and easy-to-use mesh library; excels at mesh repair, boolean operations, and 3D printing preparation.

**How it fits:** After VTK generates meshes from segmentation label maps, trimesh handles post-processing: mesh repair (watertightness), boolean operations (virtual osteotomy simulation, cutting planes), slicing for 3D printing guide design, mass property computation, and STL/3MF/OBJ export.

**Key capabilities:**
- Import/export: STL, OBJ, PLY, GLTF/GLB, 3MF, OFF, XAML, DXF, SVG
- **Boolean operations** (intersection, union, difference) via Manifold3D or Blender
- Mesh repair: winding, normals, hole filling
- Mesh analysis: watertight check, convexity, mass properties, moment of inertia
- Laplacian smoothing (Classic, Taubin, Humphrey)
- Ray-mesh queries (signed distance, surface sampling)
- Cross-section slicing (for 3D printing)
- Voxelization of watertight meshes
- Scene graph and transform tree

**Limitations:**
- Boolean operations depend on external backends (Manifold3D recommended; Blender as fallback)
- Large complex surgical meshes may expose numerical instabilities
- No GPU acceleration; CPU only
- API stability not guaranteed for production deployment (pin version)

**CMF relevance:** Critical for virtual osteotomy simulation (cut bones with planes), surgical guide design, and 3D printing file preparation (STL/3MF export, watertight checking).

**Phase:** ✅ **Phase 1 — Baseline.** Required for surgical planning geometry operations.

---

### 12. PyVista

| Attribute | Detail |
|---|---|
| **Version** | v0.47.2 (Apr 7, 2026) |
| **License** | MIT |
| **Repository** | [github.com/pyvista/pyvista](https://github.com/pyvista/pyvista) |

**What it does:** A high-level, Pythonic wrapper for VTK that dramatically simplifies 3D mesh analysis and visualization. Provides matplotlib-like syntax for 3D plotting with full VTK power underneath.

**How it fits:** PyVista is used for rapid development, backend visualization generation (screenshot rendering, surface statistics), and Jupyter-based review workflows. It exposes VTK's full filter pipeline through a concise, NumPy-native API.

**Key capabilities:**
- Pythonic API over VTK (single-line mesh loading, plotting, filtering)
- Interactive Jupyter rendering via `trame` (server-side and client-side)
- Full filter library: clip, slice, contour, decimate, smooth, boolean, warp
- Mesh read/write via VTK + meshio (OBJ, STL, PLY, VTK, etc.)
- Volume rendering without boilerplate
- Multi-block datasets (scenes with multiple meshes)
- Point cloud visualization

**Limitations:**
- Requires VTK as dependency (large install)
- Headless rendering requires OSMesa or EGL (same as raw VTK)
- Not suitable for browser-based rendering (use VTK.js for that)
- API still evolving (pre-1.0)

**Phase:** ✅ **Phase 1 — Baseline.** Replaces raw VTK for Python-side development, vastly reducing boilerplate. Use alongside VTK, not instead of it.

---

### 13. scikit-image

| Attribute | Detail |
|---|---|
| **Version** | 0.26.0 (Dec 2025) |
| **License** | BSD |
| **Repository** | [github.com/scikit-image/scikit-image](https://github.com/scikit-image/scikit-image) |

**What it does:** General-purpose image processing library for Python. Covers segmentation, morphology, filtering, feature detection, and geometric transforms.

**How it fits:** Primarily used for the marching cubes algorithm (`skimage.measure.marching_cubes`) — the standard way to convert a voxel segmentation label map into a triangular mesh. Also used for connected component analysis, morphological operations, and image-level processing outside the deep learning pipeline.

**Key capabilities:**
- **Marching cubes** (`measure.marching_cubes`) — label map → 3D mesh vertices/faces
- Morphological operations: dilation, erosion, closing, opening, skeletonization
- Connected component labeling
- Watershed segmentation
- Image filters: Gaussian, median, Canny edge detection
- Geometric transforms: affine, projective, piecewise affine
- Feature detection: SIFT, ORB, HOG

**Limitations:**
- 2D-focused heritage; 3D support works but not always primary focus
- Slower than C++/GPU alternatives for large 3D volumes
- No GPU acceleration (use MONAI transforms for GPU-accelerated 3D ops)

**Phase:** ✅ **Phase 1 — Baseline.** Marching cubes is required for segmentation→mesh conversion; lightweight utility dependency.

---

## Category 4: Visualization (Web)

### 14. OHIF Viewer

| Attribute | Detail |
|---|---|
| **Version** | v3.12.0 (Feb 6, 2026) |
| **License** | MIT |
| **Repository** | [github.com/OHIF/Viewers](https://github.com/OHIF/Viewers) |

**What it does:** OHIF (Open Health Imaging Foundation) is a zero-footprint, web-based DICOM viewer built as a progressive web application. It is the leading open-source medical imaging frontend and is used in production by major health systems and research institutions worldwide.

**How it fits:** OHIF is the primary clinical user interface — the browser-based viewer where surgeons review CT/CBCT scans, inspect segmentation results, measure anatomy, and interact with surgical plans. Its extension system allows custom CMF-specific panels to be added without forking.

**Key capabilities:**
- Zero-footprint: no client installation, runs in any modern browser
- DICOMweb integration (WADO-RS, STOW-RS, QIDO-RS)
- 2D multiplanar (axial/coronal/sagittal), MPR, MIP
- **3D volume rendering** (via Cornerstone3D/WebGL)
- **DICOM Segmentation (SEG)** display as labelmaps and contours
- **DICOM RT-Struct** display
- Measurement tracking and annotations (SR export)
- Extension system: add custom panels, tools, modes without forking
- Internationalization, OpenID Connect, offline support
- Key extensions: `cornerstone-dicom-seg`, `cornerstone-dicom-rt`, `measurement-tracking`, `tmtv`

**Limitations:**
- Primarily a radiology/oncology workflow viewer; CMF-specific tools require custom extensions
- High-fidelity interactive 3D manipulation (e.g., real-time osteotomy simulation) is not built in — requires custom Cornerstone3D/three.js integration
- Large codebase; React + TypeScript; non-trivial onboarding for custom extension development

**CMF relevance:** With a custom OHIF extension, surgeons can view CBCT scans alongside AI-generated segmentations, make annotations, review surgical plans, and approve outputs — all in the browser.

**Phase:** ✅ **Phase 1 — Baseline.** The primary clinical review interface.

---

### 15. Cornerstone3D

| Attribute | Detail |
|---|---|
| **Version** | v4.16.0 (Feb 11, 2026) |
| **License** | MIT |
| **Repository** | [github.com/cornerstonejs/cornerstone3D](https://github.com/cornerstonejs/cornerstone3D) |

**What it does:** JavaScript/TypeScript library for web-based medical image rendering. The rendering engine underlying OHIF v3. Uses WebGL for GPU-accelerated rendering and WebAssembly for fast DICOM decompression.

**How it fits:** Cornerstone3D is already embedded in OHIF. It also serves as a standalone rendering engine if custom viewer components are needed outside OHIF. The `@cornerstonejs/tools` package provides measurement, segmentation editing, and annotation tools.

**Key capabilities:**
- WebGL-accelerated 2D/3D rendering of medical images
- WebAssembly DICOM decompression
- Volume rendering, MPR, stack viewport
- Segmentation rendering (labelmaps, contours, surfaces)
- Annotation tools: length, angle, ROI, probe, bidirectional
- NIfTI volume loader (`@cornerstonejs/nifti-volume-loader`)
- AI integration package (`@cornerstonejs/ai`)
- Polymorphic segmentation (`@cornerstonejs/polymorphic-segmentation`)
- Labelmap interpolation (`@cornerstonejs/labelmap-interpolation`)

**Limitations:**
- TypeScript ecosystem; steep learning curve for backend-oriented teams
- DICOM Segmentation support is functional but 3D surface rendering of large segmentations is slow in-browser
- Does not handle very large CT volumes well without streaming/tiling

**Phase:** ✅ **Phase 1 — Baseline.** Embedded within OHIF; required for medical image rendering.

---

### 16. three.js / react-three-fiber

| Attribute | Detail |
|---|---|
| **Version** | three.js r172 (2025); react-three-fiber (R3F) v8.x |
| **License** | MIT |
| **Repository** | [github.com/mrdoob/three.js](https://github.com/mrdoob/three.js) |

**What it does:** three.js is the dominant 3D WebGL rendering library for the web. react-three-fiber (R3F) wraps three.js in a React-friendly declarative API. Together they enable interactive 3D surgical model visualization in the browser.

**How it fits:** For the surgical planning visualization layer — the screen where surgeons interact with 3D bone models, simulate osteotomies, position hardware, and review plans — three.js/R3F provides the rendering engine. Unlike medical DICOM viewers, three.js is optimized for interactive mesh rendering, which is the dominant need in surgical planning.

**Key capabilities:**
- Real-time GPU-accelerated 3D mesh rendering in the browser
- PBR (Physically Based Rendering) shading
- STL, OBJ, GLTF/GLB loader (STL/GLTF are the natural output formats from the backend)
- Interactive controls: orbit, transform, drag-and-drop positioning
- React integration via R3F (component-based 3D scenes)
- Extensive library ecosystem: drei (helper components), postprocessing, rapier (physics)
- VR/AR capability via WebXR

**Limitations:**
- Not DICOM-native; medical image volumes require custom shaders or volume rendering libraries
- For volume rendering of CT scans, VTK.js is more capable
- No built-in medical annotation tools (must build custom or use Cornerstone3D for that)

**CMF relevance:** Ideal for the surgical planning workspace: interactive 3D bone model viewing, osteotomy simulation, hardware positioning, and patient communication renderings.

**Phase:** ✅ **Phase 1 — Baseline.** Required for the interactive 3D surgical planning interface.

---

### 17. VTK.js

| Attribute | Detail |
|---|---|
| **Version** | v34.16.3 (Jan 28, 2026) |
| **License** | BSD-3-Clause |
| **Repository** | [github.com/Kitware/vtk-js](https://github.com/Kitware/vtk-js) |

**What it does:** JavaScript port of VTK, enabling server-side VTK algorithms in the browser via WebGL. Enables web-based volume rendering of CT/CBCT data with the full VTK pipeline (transfer functions, clipping planes, ray casting).

**How it fits:** VTK.js covers the CT volume rendering use case that three.js does not handle well — displaying raw Hounsfield unit volumes with bone/soft-tissue windowing directly in the browser. It also powers the Kitware remote rendering stack (trame/ParaviewWeb) for offloading heavy rendering to the server.

**Key capabilities:**
- Volume rendering in the browser (ray casting, GPU-accelerated via WebGL2)
- Transfer function editing (opacity, color mapping for CT windowing)
- Clipping planes, slice viewers
- VTK polydata rendering (meshes, points, lines)
- ITK-WASM integration for browser-side image processing
- Remote rendering via trame (ParaViewWeb backend)
- DICOM loading via browser (with ITK-WASM parsers)

**Limitations:**
- Larger bundle size than three.js
- Smaller developer community than three.js
- Most complex rendering still better done server-side with Python VTK

**Phase:** ⏩ **Phase 2.** Useful for browser-side CT volume rendering, but for Phase 1, server-rendered VTK images + three.js for mesh interaction covers the primary use cases. Evaluate based on whether full client-side CT volume rendering is required.

---

### 18. Niivue

| Attribute | Detail |
|---|---|
| **Version** | v0.68.1 (Feb 25, 2026) |
| **License** | BSD-2-Clause |
| **Repository** | [github.com/niivue/niivue](https://github.com/niivue/niivue) |

**What it does:** WebGL2-based medical image viewer supporting 30+ volume and mesh formats natively in the browser. Originally focused on neuroimaging (NIfTI-first) but now supports DICOM, meshes, tractography, and overlays.

**How it fits:** Niivue is a lighter alternative to OHIF + Cornerstone3D for volume visualization, particularly when NIfTI volumes (the output of nnU-Net/MONAI) need to be displayed alongside meshes in a simple embedded component. It can render both the CT volume and the segmentation overlay in a single lightweight component.

**Key capabilities:**
- NIfTI, NRRD, DICOM (via plugin), AFNI, MRtrix, MINC, MGH/MGZ native support
- Mesh support: GIfTI, FreeSurfer, PLY, STL, OBJ, VTK legacy
- Tractography: TCK, TRK, TRX formats
- WebGL2 GPU rendering
- Overlay support: colormaps, opacity, threshold
- React/Vue/Angular component wrappers available
- Funded by NIH (RF1 MH133701, 2023–2026)

**Limitations:**
- Smaller community than OHIF; fewer enterprise deployments
- DICOM support requires plugin; not DICOM-native
- Less extensible than OHIF for clinical workflow integration
- Does not support DICOM-SEG or RT-Struct natively

**CMF relevance:** Useful as a lightweight NIfTI volume viewer embedded in the planning interface, especially during AI model result review before DICOM write-back.

**Phase:** ⏩ **Phase 2.** Valuable supplement to OHIF for NIfTI visualization, but not required if OHIF + Cornerstone3D handles volume display.

---

## Category 5: Web/API Frameworks

### 19. FastAPI

| Attribute | Detail |
|---|---|
| **Version** | 0.135.3 (Apr 1, 2026) |
| **License** | MIT |
| **Repository** | [github.com/fastapi/fastapi](https://github.com/fastapi/fastapi) |

**What it does:** Modern, high-performance Python web framework for building REST APIs. Based on Starlette (ASGI) and Pydantic (data validation). The standard choice for Python ML/AI API backends.

**How it fits:** FastAPI is the REST API layer that exposes all platform services: DICOM upload, segmentation job submission, result retrieval, case management, and user authentication. Its async support handles concurrent requests and file upload/streaming efficiently.

**Key capabilities:**
- Async/await (ASGI): high concurrency without threading overhead
- Pydantic data validation + automatic OpenAPI/Swagger documentation
- Dependency injection: clean auth, DB session, and service injection
- OAuth2 + JWT authentication built-in
- WebSocket support (real-time job progress updates)
- File upload/streaming (large DICOM files)
- Easy integration with background tasks and Celery
- Performance on par with Node.js/Go for Python frameworks

**Limitations:**
- ASGI-only; synchronous operations must be offloaded to thread pools
- Not a full web framework (no ORM, no templating, no admin panel) — integrate SQLAlchemy/PostgreSQL separately
- Large file handling (DICOM series, 500MB+) requires careful streaming configuration

**Phase:** ✅ **Phase 1 — Baseline.** The REST API backbone of the entire platform.

---

### 20. Celery / Redis

| Attribute | Detail |
|---|---|
| **Celery Version** | 5.6.x (current stable) |
| **Celery License** | BSD |
| **Redis** | BSD |
| **Documentation** | [docs.celeryq.dev](https://docs.celeryq.dev) |

**What it does:** Celery is a distributed task queue for Python. Long-running ML inference jobs (segmentation, registration) are submitted as Celery tasks; Redis serves as the message broker and result backend.

**How it fits:** CT segmentation (TotalSegmentator, nnU-Net) takes 1–5 minutes even on GPU. These jobs cannot be handled synchronously in HTTP requests. Celery decouples job submission from execution: the FastAPI endpoint returns a job ID immediately, Celery workers process the segmentation asynchronously, and the client polls for results.

**Key capabilities:**
- Distributed task execution (multiple workers, multiple machines)
- Priority queues (urgent cases vs. batch processing)
- Task chaining and workflow DAGs (preprocessing → segmentation → mesh generation → DICOM write)
- Result backend (Redis/PostgreSQL) for job status and results
- Retry logic, exponential backoff for failed tasks
- Monitoring via Flower (web UI for task inspection)
- Beat scheduler for cron-like periodic tasks

**Limitations:**
- Adds operational complexity (Redis + workers must be managed)
- Celery 5.x requires Python 3.8+; some async patterns still complex
- For GPU tasks, worker process isolation requires careful GPU memory management

**Phase:** ✅ **Phase 1 — Baseline.** Required for all async ML processing from day one.

---

### 21. PostgreSQL

| Attribute | Detail |
|---|---|
| **Version** | 17.x (current stable) |
| **License** | PostgreSQL License (permissive, similar to BSD) |
| **Website** | [postgresql.org](https://www.postgresql.org) |

**What it does:** Battle-tested open-source relational database. Stores structured application data: patients, cases, segmentation jobs, user accounts, surgical plans, measurements, and audit logs.

**How it fits:** Every user action and case creates structured records in PostgreSQL. JSONB columns handle semi-structured metadata (DICOM headers, plan parameters). PostGIS extension adds spatial query capability if needed.

**Key capabilities:**
- ACID transactions
- JSONB for semi-structured metadata
- Full-text search
- Row-level security (critical for HIPAA multi-tenancy)
- Advanced indexing (GiST, GIN, partial indexes)
- Excellent SQLAlchemy integration (ORM for FastAPI)
- Replication, streaming backup (WAL)

**Limitations:**
- Not optimized for blob/binary storage (use MinIO for DICOM files)
- Requires backup and HA configuration for production

**Phase:** ✅ **Phase 1 — Baseline.** Required for all structured data from day one.

---

### 22. MinIO / S3

| Attribute | Detail |
|---|---|
| **MinIO Version** | RELEASE.2025-10-15 (continuous releases) |
| **License** | GNU AGPLv3 (open-source); commercial license available |
| **Repository** | [github.com/minio/minio](https://github.com/minio/minio) |

**What it does:** High-performance, S3-compatible object storage. Stores all binary data: DICOM series, NIfTI volumes, STL meshes, segmentation outputs, 3D model files, and backups.

**How it fits:** DICOM series can be hundreds of MB to several GB per patient. MinIO provides the blob storage layer (on-premise or cloud-agnostic) that decouples file storage from the database and application tier. S3-compatible API means direct swap with AWS S3, Google Cloud Storage, or Azure Blob if cloud deployment is needed.

**Key capabilities:**
- AWS S3 API compatible (full compatibility with boto3, aws-cli)
- High performance (multi-threaded, erasure coding)
- Kubernetes operator and Helm charts for cloud-native deployment
- Built-in web console (MinIO Console)
- Object lifecycle management, versioning, event notifications
- Encryption at rest and in transit
- Multi-site replication

**Limitations:**
- AGPLv3 license: if MinIO is embedded in a proprietary product distributed externally, commercial license required. For internal use/SaaS, AGPLv3 is generally acceptable.
- Self-hosted MinIO requires operational maintenance (vs. managed AWS S3)

**Phase:** ✅ **Phase 1 — Baseline.** Required for all file storage from day one.

---

## Category 6: Infrastructure

### 23. Docker / docker-compose

| Attribute | Detail |
|---|---|
| **Docker Engine** | 26.x / 27.x (current) |
| **License** | Apache-2.0 (Docker Engine); Docker Desktop requires license for large orgs |
| **docker-compose** | v2.x (now built into Docker CLI as `docker compose`) |

**What it does:** Containerization platform. Each service (FastAPI, Celery worker, Redis, PostgreSQL, MinIO, OHIF, model serving) runs in an isolated, reproducible container. docker-compose (or Docker Compose v2) orchestrates multi-container local and staging environments.

**How it fits:** Every component in this stack is containerized for reproducibility across development, staging, and production. NVIDIA Container Toolkit enables GPU passthrough to Celery/model-serving containers.

**Key capabilities:**
- Reproducible environments (no "works on my machine")
- NVIDIA Container Toolkit: GPU passthrough for ML containers
- Multi-stage builds for optimized image sizes
- Volume mounts for model weights, persistent data
- Health checks, restart policies
- Compose v2: dependency ordering, profiles, secrets

**Limitations:**
- Docker Desktop commercial licensing required for large enterprises
- Production-scale orchestration requires Kubernetes (Compose is dev/staging only)
- Image sizes can be large (PyTorch containers ~10GB with CUDA)

**Phase:** ✅ **Phase 1 — Baseline.** Non-negotiable for reproducible deployment.

---

### 24. GitHub Actions

| Attribute | Detail |
|---|---|
| **Pricing** | Free for public repos; 2000 min/month free for private (GitHub Free) |
| **License** | Proprietary (GitHub) |
| **Alternatives** | GitLab CI, Drone CI, Jenkins (if self-hosted required) |

**What it does:** Cloud-native CI/CD system tightly integrated with GitHub. Automates testing, linting, Docker builds, model validation, and deployment on every commit.

**How it fits:** CI/CD is required from day one to maintain code quality as the team grows. Key pipelines: unit/integration tests, Docker image builds (with layer caching), automated nnU-Net model accuracy checks (dice score regression tests), and staging deployment.

**Key capabilities:**
- YAML-based workflows, extensive marketplace (1000+ actions)
- GPU runners available (via GitHub-hosted or self-hosted)
- Matrix builds for multi-Python-version testing
- Container registry integration (GitHub Container Registry / GHCR)
- Branch protection rules with required CI checks
- Secrets management for deployment credentials

**Limitations:**
- Self-hosted runners required for GPU testing (GitHub-hosted GPU runners available but expensive)
- Not HIPAA-compliant for storing PHI in workflows
- Vendor lock-in (but workflows are portable YAML)

**Phase:** ✅ **Phase 1 — Baseline.** Required for team development from day one.

---

### 25. NVIDIA Triton / TorchServe

#### NVIDIA Triton Inference Server

| Attribute | Detail |
|---|---|
| **Version** | v2.67.0 / container 26.03 (Mar 2026) |
| **License** | BSD-3-Clause |
| **Repository** | [github.com/triton-inference-server/server](https://github.com/triton-inference-server/server) |

**What it does:** Production-grade model serving infrastructure from NVIDIA. Supports multiple frameworks (TensorRT, PyTorch, ONNX, OpenVINO, Python backend) with concurrent model execution, dynamic batching, and ensemble pipelines.

**Key capabilities:**
- Multi-model, multi-framework serving (TensorRT, PyTorch, ONNX, Python backend)
- Dynamic batching for throughput optimization
- Concurrent model execution (multiple models on same GPU)
- HTTP/REST + gRPC APIs (KServe protocol)
- Sequence batching for stateful models
- Model ensemble pipelines (preprocessing → model → postprocessing)
- Prometheus metrics (GPU utilization, latency, throughput)
- Kubernetes/Helm deployment

**Limitations:**
- Complex configuration; significant DevOps overhead
- Overkill for single-model, low-concurrency early-stage deployment
- TensorRT optimization requires NVIDIA GPU (not portable to CPU/AMD)

#### TorchServe

| Attribute | Detail |
|---|---|
| **Version** | v0.12.0 (Sep 2024) |
| **License** | Apache-2.0 |
| **Repository** | [github.com/pytorch/serve](https://github.com/pytorch/serve) |

**What it does:** PyTorch-native model serving. Simpler than Triton; primarily serves PyTorch models via REST/gRPC with batching, versioning, and metrics.

**Key capabilities:**
- REST + gRPC inference API
- Multi-model management, A/B testing
- Batched inference
- Custom handler classes (pre/post processing)
- KServe, SageMaker, Vertex AI, Kubeflow integration
- torch.compile, ONNX, TensorRT export support

**Recommendations for CMF platform:**

| Stage | Recommendation |
|---|---|
| Phase 1 | Use **Celery workers** with direct PyTorch/nnU-Net inference — simpler, no additional serving infrastructure |
| Phase 2 | Add **TorchServe** for structured model management, versioning, and metrics |
| Phase 3 / Scale | Migrate to **Triton** for multi-model concurrency, TensorRT optimization, and high-throughput production serving |

**Phase:** ⏩ **Phase 2 (TorchServe) / Phase 3 (Triton).** Celery + direct model invocation is sufficient for Phase 1.

---

## Category 7: CMF/Dental-Specific Repositories & Tools

### 26. General CMF/Dental Open-Source Repositories

#### A. DentalSegmentator
| Attribute | Detail |
|---|---|
| **Repository** | [github.com/gaudot/SlicerDentalSegmentator](https://github.com/gaudot/SlicerDentalSegmentator) |
| **Paper** | Dot G, et al. *Journal of Dentistry* (2024). DOI: 10.1016/j.jdent.2024.105130 |
| **Model weights** | [zenodo.org/records/10829675](https://zenodo.org/records/10829675) |
| **License** | MIT (3D Slicer extension); nnU-Net model weights Apache-2.0 |

**What it does:** State-of-the-art nnU-Net v2 model for fully automatic segmentation of dento-maxillofacial CT and CBCT scans. Trained on 470 DMF CT/CBCT scans; validated on 256 scans from 7 institutions.

**Structures segmented:**
1. Maxilla + Upper Skull
2. Mandible
3. Upper Teeth
4. Lower Teeth
5. Mandibular Canal

**Key capabilities:**
- Robust to metal artifacts and variable field-of-view (major clinical advantage)
- Works on CT and CBCT (combined training data)
- ~1–2 minutes on GPU, longer on CPU
- Can be used standalone via nnU-Net CLI (download weights from Zenodo)
- Integrated into 3D Slicer via SlicerDentalSegmentator extension

**CMF relevance:** The best publicly available starting point for CMF segmentation. Download weights, plug into nnU-Net inference, and you have working mandible/teeth/maxilla segmentation immediately. No retraining needed for standard cases.

**Phase:** ✅ **Phase 1 — Baseline.** Immediate CMF segmentation capability, no training required.

---

#### B. AMASSS (Automatic Multi-Anatomical Skull Structure Segmentation)
| Attribute | Detail |
|---|---|
| **Repository** | [github.com/DCBIA-OrthoLab/AMASSS_CBCT](https://github.com/DCBIA-OrthoLab/AMASSS_CBCT) |
| **License** | Apache-2.0 |
| **Affiliation** | University of Michigan / UNC Chapel Hill (DCBIA-OrthoLab) |

**What it does:** 3D U-Net-based automatic segmentation of CBCT scans for multiple craniofacial skeletal structures, focused on orthodontic/orthognathic analysis.

**Key capabilities:**
- Cranial base, maxilla, mandible, condyles, teeth segmentation from CBCT
- Designed for orthodontic/orthognathic research datasets
- Integrated into SlicerAutomatedDentalTools

**Phase:** ✅ **Phase 1 — Baseline.** Complementary to DentalSegmentator for CBCT-specific orthodontic structure segmentation.

---

#### C. ALI-CBCT (Automated Landmark Identification in CBCT)
| Attribute | Detail |
|---|---|
| **Repository** | [github.com/Maxlo24/ALI_CBCT](https://github.com/Maxlo24/ALI_CBCT) |
| **Paper** | Published in *Orthodontics & Craniofacial Research* (2023). [PMC10440369](https://pmc.ncbi.nlm.nih.gov/articles/PMC10440369/) |
| **License** | Not explicitly stated (research code) |
| **Affiliation** | University of Michigan (DCBIA-OrthoLab) |

**What it does:** Reformulates anatomical landmark detection as a classification problem via a virtual agent navigating 3D CBCT space. Uses densely connected CNNs + fully connected layers (MONAI + PyTorch).

**Structures / landmarks:** Cranial base landmarks, upper/lower facial bone landmarks, tooth-specific landmarks (left, right, upper, lower).

**Key capabilities:**
- Automatic placement of cephalometric landmarks on CBCT
- No manual initialization required
- Docker container available
- Integrated into SlicerAutomatedDentalTools (ALI module)

**CMF relevance:** Automated cephalometric analysis (Frankfort horizontal, nasion, A-point, B-point, menton, gonion) is essential for orthognathic surgery planning and outcome assessment.

**Phase:** ✅ **Phase 1 — Baseline.** Automated landmark detection is core to cephalometric analysis.

---

#### D. CL-Detection2023 (Cephalometric Landmark Detection)
| Attribute | Detail |
|---|---|
| **Challenge** | MICCAI 2023 registered challenge |
| **DOI** | 10.5281/zenodo.7835591 |
| **Type** | 2D lateral cephalogram landmark detection |

**What it does:** MICCAI 2023 challenge for automated detection of cephalometric landmarks in lateral X-ray images. Multiple top-ranked solutions are publicly available.

**CMF relevance:** 2D cephalometric analysis from lateral cephalograms remains the standard pre-surgical planning tool in orthodontics. ISBI 2015 dataset (400 lateral cephalograms) and CL-Detection 2023 provide training data and benchmark baselines.

**Notable model:** HRNet-W32-based cephalometric landmark detector available on [Hugging Face](https://huggingface.co/cwlachap/hrnet-cephalometric-landmark-detection) (19 landmarks, ISBI 2015 dataset).

**Phase:** ✅ **Phase 1 — Baseline.** 2D cephalometric analysis is a core feature.

---

### 27. SlicerCMF and Related 3D Slicer Extensions

#### SlicerCMF
| Attribute | Detail |
|---|---|
| **Repository** | [github.com/DCBIA-OrthoLab/SlicerCMF](https://github.com/DCBIA-OrthoLab/SlicerCMF) |
| **Website** | [cmf.slicer.org](https://cmf.slicer.org) |
| **License** | Apache-2.0 (most extensions) |
| **Affiliation** | DCBIA-OrthoLab (University of Michigan, UNC Chapel Hill, Kitware) |

**What it does:** SlicerCMF is the umbrella 3D Slicer extension for craniomaxillofacial and dental research. It bundles multiple sub-modules for clinical CMF workflows.

**Architecture consideration for the platform:** SlicerCMF is a desktop application (3D Slicer GUI extension). For a web-native platform, SlicerCMF algorithms are best used as:
1. Reference implementations to understand the clinical workflow
2. Python CLI modules that can be invoked headlessly in Docker containers
3. Training data sources (DCBIA datasets)

**Included modules:**

| Module | Function |
|---|---|
| **Q3DC** | Quantitative 3D Cephalometrics — measure angles (yaw/pitch/roll) and 3D distances between landmarks. Gold standard for orthognathic surgical planning measurements |
| **CMFreg** | CBCT voxel-based registration — align follow-up to baseline for outcome assessment |
| **AnglePlanes** | Calculate angles between anatomical planes |
| **ModelToModelDistance** | Surface deviation maps between pre/post models |
| **ShapeVariationAnalyzer** | Deep-learning shape classification (condyle morphology) |
| **EasyClip** | Clip models by planes (virtual osteotomy visualization) |
| **MeshStatistics** | Descriptive statistics on mesh scalar fields |
| **PickAndPaint** | ROI selection on 3D surfaces |

**Phase:** ⏩ **Phase 2.** The Q3DC and CMFreg algorithms are clinically validated and should be ported/adapted for the web backend. The desktop 3D Slicer interface is not suitable for the web-native platform but the underlying algorithms (Python CLIs) can be containerized.

---

#### SlicerAutomatedDentalTools
| Attribute | Detail |
|---|---|
| **Repository** | [github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools](https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools) |
| **License** | Apache-2.0 |
| **Stars** | 152 |

**What it does:** The most advanced open-source AI toolkit for automated dental/CMF analysis, integrating AMASSS, ALI-CBCT, ALI-IOS, ASO, AReg, and more.

**All included tools:**

| Tool | Description |
|---|---|
| **AMASSS** | Automatic Multi-Anatomical Skull Structure Segmentation (CBCT, 3D UNet) |
| **ALI-CBCT** | Automated landmark identification on CBCT (cranial base, facial bones, teeth) |
| **ALI-IOS** | Automated landmark identification on intraoral scans (IOS) |
| **ASO** | Automatic Standardized Orientation (CBCT and IOS) |
| **AReg** | Automatic Registration (CBCT and IOS) |
| **AutoCrop3D** | Automatic ROI cropping of CBCT folders |
| **FlexReg** | Patient-specific IOS-to-IOS registration with custom patches |
| **BatchDentalSeg** | Batch dental segmentation (DentalSegmentator, PediatricDentalSeg, UniversalLab) |
| **MRI2CBCT** | MRI-CBCT registration (including TMJ) |
| **DOCShapeAXI** | Shape explainability — classify nasopharyngeal airway obstruction severity, mandibular condyle degeneration (osteoarthritis), alveolar bone cleft severity |
| **CLIC** | Impacted canine localization and classification (buccal/palatal/bicortical) |
| **MedX** | TMJ comorbidity extraction from clinical notes via fine-tuned LLM |
| **Medical Data Anonymizer** | PHI anonymization in documents |

**Phase:** ✅ **Phase 1 — Baseline (core tools).** AMASSS, ALI-CBCT, and BatchDentalSeg are immediately useful as Docker-invokable inference engines. DOCShapeAXI adds AI-powered shape classification for condyle/airway analysis. Port the Python CLIs to Celery tasks.

---

### 28. MICCAI Challenges Relevant to CMF

| Challenge | Year | Task | Dataset | Relevance |
|---|---|---|---|---|
| **ToothFairy** | MICCAI 2023 | CBCT segmentation: inferior alveolar canal | ~443 annotated CBCT | IAC segmentation critical for implant planning |
| **ToothFairy2** | MICCAI 2024 | Multi-structure CBCT: mandible, teeth, maxilla, pharynx | Extended dataset; [grand-challenge.org](https://toothfairy2.grand-challenge.org) | Best public multi-structure CMF dataset |
| **STS 2023** | MICCAI 2023 | Semi-supervised tooth segmentation (2D OPG + 3D CBCT) | 434 teams; large OPG + CBCT dataset | Training data for tooth segmentation models |
| **STS 2024** | MICCAI 2024 | Semi-supervised instance tooth segmentation | 90k+ 2D slices, 2380 OPGs, 330 CBCT; [github.com/ricoleehduu/STS-Challenge-2024](https://github.com/ricoleehduu/STS-Challenge-2024) | Best large-scale dental SSL benchmark |
| **3DTeethSeg22** | MICCAI 2022 | 3D intraoral scan (IOS) tooth segmentation | IOS mesh data | IOS-based individual tooth segmentation |
| **3DTeethLand** | MICCAI 2024 | 3D teeth landmark detection | CBCT + IOS | Automated landmark detection on IOS |
| **CL-Detection 2023** | MICCAI 2023 | 2D lateral cephalogram landmark detection | Lateral X-rays | Cephalometric analysis from 2D X-rays |
| **PDDCA / Head & Neck 2015** | MICCAI 2015 | Head and neck auto-segmentation from CT | 40 CT scans with mandible, brainstem, parotid, optic nerve | Foundational CMF CT benchmark; mandible annotations |
| **STSR 2025** | MICCAI 2025 | CBCT teeth and root canal segmentation | Permanent teeth + pulp canal | Most advanced dental segmentation challenge |

**Key datasets accessible:**
- **ToothFairy2 dataset** (CBCT; mandible, teeth, maxilla, pharynx, IAC): available via Grand Challenge
- **STS 2024 dataset** (2380 OPG + 330 CBCT with FDI tooth labels): [github.com/ricoleehduu/STS-Challenge-2024](https://github.com/ricoleehduu/STS-Challenge-2024)
- **TotalSegmentator teeth task** (based on ToothFairy3 / publicly released weights)
- **PDDCA dataset**: 40 CT scans with mandible annotations (via ImagEng Lab)

**Phase:** ✅ **Phase 1 — Use immediately.** ToothFairy2 and STS 2024 datasets provide training data for any custom model fine-tuning. MICCAI 2023/2024 winning codebases provide state-of-the-art baselines.

---

## Dependency & Integration Map

```
DICOM Input (CT/CBCT)
    │
    ├─► pydicom (tag/header parsing)
    ├─► SimpleITK (series-to-volume, resampling, registration)
    └─► MinIO (raw DICOM storage)
                │
                ▼
        Preprocessing Pipeline (Python/Celery)
            │
            ├─► MONAI (transforms, normalization, augmentation)
            ├─► nibabel (NIfTI I/O)
            └─► scikit-image (morphological ops)
                │
                ▼
        Segmentation (nnU-Net / MONAI models)
            │
            ├─► TotalSegmentator (craniofacial_structures, teeth, head tasks)
            ├─► DentalSegmentator weights (mandible, teeth, IAC)
            └─► AMASSS / custom fine-tuned models
                │
                ▼
        Post-processing (Python/Celery)
            │
            ├─► VTK/PyVista (marching cubes → mesh)
            ├─► trimesh (boolean ops, repair, STL/3MF export)
            ├─► scikit-image (marching cubes)
            └─► highdicom (DICOM-SEG write-back)
                │
                ▼
        Storage
            │
            ├─► MinIO (NIfTI, STL, GLTF outputs)
            └─► PostgreSQL (metadata, job status, measurements)
                │
                ▼
        Web Frontend (Browser)
            │
            ├─► OHIF v3 + Cornerstone3D (DICOM viewer, segmentation review)
            ├─► three.js / R3F (interactive 3D surgical planning)
            └─► FastAPI (REST API gateway)
```

---

## Phase Recommendations Summary

### Phase 1 — Baseline (Build First)

| # | Component | Version | License | Role |
|---|---|---|---|---|
| 1 | PyTorch | 2.11 | BSD | ML backbone |
| 2 | MONAI | 1.5.2 | Apache-2.0 | Medical imaging ML framework |
| 3 | nnU-Net v2 | v2 | Apache-2.0 | Segmentation training framework |
| 4 | TotalSegmentator | 2.11 | Apache-2.0 | Pre-trained CMF segmentation |
| 5 | DentalSegmentator (weights) | 2024 | Apache-2.0 | Pre-trained mandible/teeth segmentation |
| 6 | AMASSS | 2022 | Apache-2.0 | CBCT skull segmentation |
| 7 | ALI-CBCT | 2022 | Research | Automated CBCT landmark detection |
| 8 | pydicom | 2.4.5 | MIT | DICOM file I/O |
| 9 | SimpleITK | 2.5.3 | Apache-2.0 | Image processing, registration |
| 10 | nibabel | 5.4.1 | MIT | NIfTI I/O |
| 11 | highdicom | 0.26.1 | MIT | DICOM-SEG/SR output |
| 12 | VTK / vtkpython | 9.6.1 | BSD-3 | 3D rendering, mesh pipeline |
| 13 | PyVista | 0.47.2 | MIT | Pythonic VTK wrapper |
| 14 | trimesh | 4.11.5 | MIT | Mesh operations, boolean ops, 3D print prep |
| 15 | scikit-image | 0.26.0 | BSD | Marching cubes, morphology |
| 16 | OHIF Viewer | 3.12.0 | MIT | Web DICOM viewer |
| 17 | Cornerstone3D | 4.16.0 | MIT | Medical image rendering engine |
| 18 | three.js / R3F | r172 / v8 | MIT | Interactive 3D surgical planning UI |
| 19 | FastAPI | 0.135.3 | MIT | REST API framework |
| 20 | Celery | 5.6.x | BSD | Async task queue |
| 21 | Redis | 7.x | BSD | Message broker / cache |
| 22 | PostgreSQL | 17.x | PostgreSQL | Relational database |
| 23 | MinIO | 2025-10 | AGPLv3 | Object storage |
| 24 | Docker / compose | 27.x | Apache-2.0 | Containerization |
| 25 | GitHub Actions | — | Proprietary | CI/CD |
| 26 | TorchServe | 0.12.0 | Apache-2.0 | Model serving |

### Phase 2 — Expand (Second Priority)

| Component | Reason for Phase 2 |
|---|---|
| Open3D | IOS-to-CBCT registration; needed when integrating intraoral scans |
| VTK.js | Client-side CT volume rendering (if full browser-side rendering required) |
| Niivue | Lightweight NIfTI viewer for embedded plan review |
| SlicerCMF algorithms (Q3DC, CMFreg) | Port Python CLIs for server-side cephalometric measurement |
| DOCShapeAXI | Shape classification for airway/condyle analysis |
| CLIC | Impacted canine detection |
| NVIDIA Triton (or upgrade TorchServe) | High-throughput multi-model serving at scale |

### Phase 3 — Advanced

| Component | Reason for Phase 3 |
|---|---|
| MONAI Label (active learning server) | Annotation acceleration with AI-assisted labeling |
| Federated learning (MONAI FL) | Multi-site training without data sharing |
| NVIDIA Triton + TensorRT | Maximum GPU inference throughput |
| Custom foundation model fine-tuning | SAM2, SuPreM, or domain-specific foundation models |

---

## Risk Assessment and Key Considerations

### Licensing Risks
| Tool | Issue | Mitigation |
|---|---|---|
| MinIO | AGPLv3: requires open-sourcing modifications if distributed | Use commercial MinIO license, or use AWS S3 / equivalent |
| TotalSegmentator (some tasks) | Non-commercial license for `brain_structures`, `face`, `appendicular_bones` | Use only Apache-2.0 tasks for commercial product |
| Docker Desktop | Commercial license required for enterprises >250 employees | Use Docker Engine (CLI only) in CI/production |

### Technical Risks
| Risk | Mitigation |
|---|---|
| nnU-Net inference latency (1–5 min/case) | Celery async queue; GPU acceleration; --fast mode; pre-compute on DICOM upload |
| Large DICOM series size (1GB+) | Streaming upload to MinIO; never load full series into memory at once |
| Mesh boolean operation stability | Use Manifold3D backend in trimesh; validate watertightness pre/post op |
| CBCT artifact interference with segmentation | DentalSegmentator is specifically trained to handle metal artifacts |
| Regulatory (FDA/CE) | No open-source tool is FDA-cleared; platform requires regulatory strategy as clinical decision support software |

### Data Resources
| Resource | Access | Use |
|---|---|---|
| ToothFairy2 dataset | Grand Challenge registration | Fine-tuning mandible/teeth models |
| STS 2024 dataset | GitHub (public) | Tooth segmentation training |
| PDDCA (MICCAI 2015) | ImagEng Lab | Mandible CT benchmark |
| TotalSegmentator weights | Zenodo / pip install | Immediate craniofacial segmentation |
| DentalSegmentator weights | Zenodo record 10829675 | Immediate mandible/teeth segmentation |

---

*Compiled from primary sources: GitHub repositories, PyPI, MICCAI challenge records, and peer-reviewed publications. All version numbers as of April 2026.*
