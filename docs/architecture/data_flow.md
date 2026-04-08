# Data Flow — Facial Align

**Version:** 1.0  
**Audience:** Backend engineers, ML engineers, data architects  
**Purpose:** Definitive specification of how data moves through every pipeline stage

---

## Table of Contents

1. [DICOM Ingestion Flow](#1-dicom-ingestion-flow)
2. [Preprocessing Pipeline](#2-preprocessing-pipeline)
3. [Segmentation Pipeline](#3-segmentation-pipeline)
4. [Mesh Generation Pipeline](#4-mesh-generation-pipeline)
5. [Registration Pipeline](#5-registration-pipeline)
6. [Fracture Reduction Planning Pipeline](#6-fracture-reduction-planning-pipeline)
7. [Occlusion Analysis Pipeline](#7-occlusion-analysis-pipeline)
8. [Data Storage and Retrieval Patterns](#8-data-storage-and-retrieval-patterns)
9. [ML Training Data Collection Flow](#9-ml-training-data-collection-flow)

---

## 1. DICOM Ingestion Flow

### 1.1 Entry Points

**Path A — Browser ZIP Upload:**
```
Browser → POST /api/v1/studies (multipart/form-data)
API → Stream to MinIO: dicom-studies/{temp_uid}/raw/upload.zip
API → Postgres: INSERT jobs (task=ingest_dicom, status=PENDING)
API → Celery: ingest_dicom.delay(job_id, temp_uid)
API → Client: 202 {job_id, status_url}
```

**Path B — DICOM C-STORE (Orthanc):**
```
PACS → Orthanc (port 4242, DICOM AE)
Orthanc → MinIO: (via Orthanc storage plugin) dicom-studies/{study_uid}/raw/*.dcm
Orthanc → Webhook: POST /api/v1/internal/dicom-received {study_uid}
API → Postgres: INSERT Case, DicomStudy
API → Celery: ingest_dicom.delay(job_id, study_uid)
```

### 1.2 Ingestion Worker: Detailed Steps

```
Task: ingest_dicom(job_id, upload_ref)

Step 1: UNPACK
  Input:  MinIO path → raw DICOM files (ZIP or individual .dcm)
  Action: If ZIP: extract to temp directory
          pydicom.read_file(stop_before_pixels=True) for each file
          Extract tags: StudyInstanceUID, SeriesInstanceUID, SOPInstanceUID,
                        Modality, SliceThickness, PixelSpacing, ImageOrientationPatient,
                        PatientID, PatientName, StudyDate, Manufacturer, KVP
  Output: {series_uid: [file_path, ...]} grouped dict

Step 2: VALIDATE SERIES
  Rules:
    - Modality must be CT or CBCT (reject MR without warning)
    - SliceThickness must be ≤ 2.5mm (warn > 1.5mm, reject > 2.5mm)
    - Series must contain ≥ 50 slices for valid 3D reconstruction
    - ImageOrientationPatient must define a consistent axial acquisition
    - No duplicate SOPInstanceUIDs
    - Check for gantry tilt: warn if tilt > 5° (SimpleITK can correct, but flag it)
  Output: validation_report.json {passed: bool, warnings: [...], errors: [...]}
  On failure: update job status=FAILED, store report, notify

Step 3: DE-IDENTIFICATION
  Input:  Raw DICOM files
  Action: pydicom/deid — apply DICOM PS 3.15 Annex E Basic Application Level
          Tags zeroed/replaced: PatientName, PatientID, PatientBirthDate,
                                 PatientAddress, PatientPhone, StudyDate (shifted),
                                 InstitutionName, ReferringPhysicianName,
                                 AccessionNumber, StudyDescription, SeriesDescription
          Date shifting: shift all dates by random offset (stored encrypted per patient)
          Private tags: REMOVE_ALL_PRIVATE or KEEP_SAFE per deid recipe
          Pixel data: scan for burned-in text (vision model check — async)
  Output: deid_report.json {tags_modified: [...], private_tags_removed: N,
                             pixel_phi_check: {status, flagged_files: []}}
          De-identified .dcm files written to MinIO: dicom-studies/{study_uid}/deidentified/

Step 4: VOLUME ASSEMBLY
  Input:  De-identified DICOM series directory
  Action: SimpleITK.ImageSeriesReader
          reader.GetGDCMSeriesFileNames() → sorted file list
          reader.Execute() → sitk.Image (3D volume, preserving spacing/direction)
          Apply DICOMOrient to LPS canonical coordinates
  Output: sitk.Image with properties:
            - Spacing: [sx, sy, sz] mm
            - Origin: [ox, oy, oz] mm (LPS)
            - Direction: 3×3 direction cosines
            - Size: [Nx, Ny, Nz]

Step 5: METADATA EXTRACTION
  Output to Postgres (DicomStudy.metadata JSONB):
    {
      "modality": "CT",
      "scanner_manufacturer": "Siemens",
      "scanner_model": "SOMATOM Force",
      "kvp": 120,
      "slice_thickness_mm": 0.625,
      "pixel_spacing_mm": [0.488, 0.488],
      "reconstruction_kernel": "B60f",
      "matrix_size": [512, 512],
      "num_slices": 487,
      "field_of_view_mm": 250.0,
      "gantry_tilt_deg": 0.0,
      "acquisition_date_shifted": "2024-03-XX"
    }

Step 6: EMIT NEXT TASK
  Update: job.status = SUCCESS, case.status = PREPROCESSED
  Emit: preprocess_volume.delay(case_id, study_uid)
```

---

## 2. Preprocessing Pipeline

### 2.1 Preprocessing Worker

```
Task: preprocess_volume(case_id, study_uid)

Input:  sitk.Image (from ingestion) or reload from MinIO deidentified path

Step 1: RESAMPLE TO ISOTROPIC SPACING
  Target: 1.0 × 1.0 × 1.0 mm³ for CT (0.4 × 0.4 × 0.4 mm³ for CBCT dental)
  Method: SimpleITK.ResampleImageFilter
          Interpolator: sitk.sitkBSpline (for smooth resampling)
          Transform: identity
  Note: TotalSegmentator internally resamples to 1.5mm; DentalSegmentator to 0.4mm.
        Standardize to 1.0mm as our canonical intermediate to avoid double-resampling.

Step 2: ORIENTATION NORMALIZATION
  Target: LPS (Left-Posterior-Superior) coordinate system
  Method: sitk.DICOMOrient(image, 'LPS')
  Purpose: Ensures all downstream code uses consistent anatomical axis convention

Step 3: HU WINDOWING (SOFT CLIP)
  Bone window: [−1000, 3000] HU (preserve full range; models handle internally)
  Store raw HU values in NIfTI; windowing applied at visualization layer only
  Exception: preprocessing for landmark detection uses soft-tissue + bone combined
             window [−200, 1500] HU

Step 4: INTENSITY NORMALIZATION (Optional, task-specific)
  For TotalSegmentator: NOT applied (model trained on raw HU)
  For DentalSegmentator: NOT applied (same)
  For custom nnU-Net training: z-score per case or percentile normalization (configured per dataset)

Step 5: VOLUME EXPORT
  Format: NIfTI compressed (.nii.gz)
  Library: SimpleITK.WriteImage() → nibabel for verification
  Stored at: MinIO dicom-studies/{study_uid}/processed/volume.nii.gz
  Sidecar: volume_meta.json (spacing, origin, direction, shape, HU stats)

Step 6: QUALITY METRICS
  Compute and store:
    - Volume HU statistics (mean, std, min, max, percentiles)
    - Estimated bone volume (voxels > 300 HU) → sanity check for CT vs. CBCT
    - Signal-to-noise estimate
    - Metal artifact detection score (max HU region fraction)
  Store in: DicomStudy.quality_metrics JSONB

Output: MinIO path to volume.nii.gz
        Postgres: case.status = 'PREPROCESSED'
        Emit: run_segmentation.delay(case_id)
```

---

## 3. Segmentation Pipeline

### 3.1 Segmentation Worker

```
Task: run_segmentation(case_id)

Input:  MinIO path to volume.nii.gz (1.0mm isotropic, LPS, raw HU)

Step 1: COARSE SEGMENTATION — TotalSegmentator
  Call: POST inference-service:8080/predictions/totalsegmentator
  Input: NIfTI bytes (volume.nii.gz)
  Model: TotalSegmentator v2.11 with CMF task specification
  Output structures:
    - skull (label 1)
    - mandible (label 2)
    - maxilla (label 3)
    - left/right zygomatic arch (labels 4, 5)
    - left/right maxillary sinus (labels 6, 7)
    - sphenoid bone (label 8)
    - soft tissue (label 9)
  Inference time: ~45–90s on A10G GPU
  Output: segmentation_coarse.nii.gz + uncertainty_coarse.nii.gz

Step 2: DENTAL ROI CROP
  Detect bounding box from mandible + maxilla segmentation
  Expand by 20mm margin in each direction
  Crop original volume to dental ROI
  Resample to 0.4mm isotropic (if source is standard CT)
  Output: dental_crop.nii.gz (smaller volume, higher resolution)

Step 3: FINE SEGMENTATION — DentalSegmentator
  Call: POST inference-service:8080/predictions/dental_segmentator
  Input: dental_crop.nii.gz bytes
  Model: DentalSegmentator (nnU-Net v2, trained on 470 CT scans)
  Output structures:
    - mandible (refined, label 1)
    - upper teeth (collective mask, label 2)
    - lower teeth (collective mask, label 3)
    - individual teeth (labels 10–47, FDI notation where available)
    - mandibular canal / inferior alveolar nerve (label 50)
    - mandibular canal right/left (labels 51, 52)
  Inference time: ~20–40s on A10G GPU
  Output: segmentation_dental.nii.gz + uncertainty_dental.nii.gz

Step 4: SEGMENTATION FUSION
  Strategy:
    - Dental segmentation overrides coarse for mandible and dental regions (higher resolution)
    - Coarse segmentation provides skull, sinuses, soft tissue (dental doesn't cover these)
    - Overlap resolution: use weighted average of coarse and fine predictions in transition zone
  Output: segmentation_fused.nii.gz (unified multi-label volume, LPS space, 1.0mm)
          uncertainty_fused.nii.gz (per-voxel max uncertainty from both models)

Step 5: STRUCTURE ISOLATION
  For each label: extract binary mask, apply connected component analysis
  Retain largest component (removes small isolated islands)
  Apply morphological closing (radius 1 voxel) for surface smoothness
  Output: per-structure binary masks (stored in memory, consumed by mesh extraction)

Step 6: QUALITY EVALUATION
  Compute (against any available ground truth, or self-report if none):
    - Volume per structure (mm³)
    - Presence check: all expected structures present?
    - Symmetry check: compare left/right bilateral structures for gross asymmetry
    - Confidence distribution per structure: mean, std, % < 0.5 threshold
  Store: segmentation.dice_scores JSONB, segmentation.quality_flags JSONB

Step 7: STORE OUTPUTS
  MinIO:
    mesh-assets/{case_id}/segmentation/segmentation.nii.gz
    mesh-assets/{case_id}/segmentation/uncertainty_map.nii.gz
    mesh-assets/{case_id}/segmentation/label_map.json
  Postgres: INSERT segmentations record
            UPDATE cases.status = 'SEGMENTED'
  Emit: extract_meshes.delay(case_id), detect_landmarks.delay(case_id)  [parallel]
```

---

## 4. Mesh Generation Pipeline

### 4.1 Mesh Extraction Worker

```
Task: extract_meshes(case_id)

Input:  MinIO path to segmentation.nii.gz

For each structure label (mandible, maxilla, each tooth, sinuses, etc.):

Step 1: MARCHING CUBES
  Library: scikit-image.measure.marching_cubes
  Input: binary mask (label == structure_label)
  Level: 0.5 (boundary threshold)
  Spacing: from NIfTI header (propagates physical dimensions)
  Output: vertices (N×3), faces (M×3), normals (M×3)

Step 2: COORDINATE TRANSFORM
  Convert from voxel coordinates to LPS mm coordinates
  Apply NIfTI affine transform: coords_lps = affine @ [i, j, k, 1]
  Vertices are now in physical space (mm); matches landmark coordinates

Step 3: MESH CLEANUP
  Library: PyVista + trimesh
  Operations:
    a. Remove degenerate faces (zero-area triangles)
    b. Merge duplicate vertices (tolerance: 0.01mm)
    c. Laplacian smoothing (10 iterations, lambda=0.3) — reduces staircase artifact
    d. Mesh decimation: target 50,000 faces for teeth, 200,000 for mandible/maxilla
       Method: PyVista decimate_pro (preserves sharp features)
    e. Watertight check: trimesh.is_watertight
       If not watertight: apply trimesh.repair.fill_holes

Step 4: MESH VALIDATION
  Checks:
    - Vertex count > 100 (reject degenerate outputs)
    - Face count > 100
    - Bounding box sanity (mandible > 50mm × 30mm × 25mm)
    - No self-intersections (trimesh check)
  On failure: store error in Mesh.quality_flags, notify (do not block pipeline)

Step 5: EXPORT FORMATS
  STL: Binary STL (surgical use, 3D printing)
      trimesh.export(format='stl')
  GLTF: Compressed GLTF 2.0 (Three.js WebGL renderer)
      trimesh → glTF2.0 via trimesh.exchange.export
      Apply draco compression for transfer efficiency

Step 6: SURFACE UNCERTAINTY MAPPING
  Map per-voxel uncertainty to mesh vertex values:
    For each vertex: sample uncertainty_map.nii.gz at vertex LPS coordinate
    Interpolation: trilinear (scipy.ndimage.map_coordinates)
  Store as vertex color attribute in GLTF (red = high uncertainty, green = low)
  Used by Three.js to render confidence heat map on mesh surface

Step 7: STORE OUTPUTS
  MinIO: mesh-assets/{case_id}/meshes/{structure_name}.stl
         mesh-assets/{case_id}/meshes/{structure_name}.gltf
  Postgres: INSERT meshes record per structure
            (vertex_count, face_count, storage_path, gltf_path)
  Emit: [No chaining — mesh extraction runs parallel to landmark detection]
```

---

## 5. Registration Pipeline

### 5.1 CT-to-Intraoral Scan Registration

```
Task: register_intraoral_scan(case_id, scan_file_path)

Input A: 3D mesh of lower/upper dental segmentation (from case segmentation)
Input B: Intraoral scan STL (uploaded separately by surgeon)

Step 1: SCALE NORMALIZATION
  Both meshes should be in mm; verify scale
  If intraoral scan appears to be in meters: apply ×1000 scale correction

Step 2: INITIAL ALIGNMENT
  Compute centroids of both meshes
  Translate intraoral scan centroid to CT mesh centroid
  Compute PCA axes of both → coarse rotation alignment

Step 3: ICP REGISTRATION
  Method: Open3D PointToPoint ICP (Iterative Closest Point)
  Input: Point clouds sampled from both mesh surfaces (50,000 points each)
  Threshold: 5.0mm (initial convergence tolerance)
  Max iterations: 100
  Refine: PointToPlane ICP with 1.0mm threshold for fine alignment

Step 4: REGISTRATION QUALITY
  Compute mean point-to-surface distance after ICP
  Threshold: PASS if mean < 0.5mm; WARNING if 0.5–1.5mm; FAIL if > 1.5mm
  Store: registration_report.json {rmse, max_error, inlier_fraction, transform_matrix}

Step 5: TRANSFORM APPLICATION
  Apply 4×4 rigid transform matrix to intraoral scan mesh
  Result: intraoral scan expressed in CT LPS coordinate system
  Store: mesh-assets/{case_id}/registrations/intraoral_registered.stl
         transform_matrix.json (for reproducibility / re-application)
```

### 5.2 Pre-op to Post-op CT Registration (Outcome Tracking)

```
Task: register_postop_ct(case_id, postop_study_uid)

Input A: Pre-op NIfTI (from original case processing)
Input B: Post-op CT → run through same ingestion + preprocessing pipeline first

Step 1: RIGID REGISTRATION (SKULL-BASED)
  Method: SimpleITK ImageRegistrationMethod
  Fixed: Pre-op volume masked to skull region (upper cranium — immobile reference)
  Moving: Post-op volume
  Metric: Mattes Mutual Information
  Optimizer: L-BFGS-B with multi-resolution pyramid (3 levels)
  Transform: VersorRigid3D (6 DOF: rotation + translation)

Step 2: BONE SEGMENT TRACKING
  After global skull registration:
  For each bone segment in the surgical plan:
    - Extract pre-op bone segment mesh
    - Find corresponding post-op bone location (nearest neighbor + ICP refinement)
    - Compute segment-level rigid transform
    - Decompose into translation vector (mm) and rotation angles (degrees)

Step 3: DEVIATION COMPUTATION
  For each planned movement:
    planned_transform = surgical_plans.movements[segment]
    achieved_transform = segment-level transform from Step 2
    deviation_mm = ||planned_translation - achieved_translation||₂
    deviation_deg = angle_between(planned_rotation, achieved_rotation)
  Store: evaluations record with per-segment deviations
```

---

## 6. Fracture Reduction Planning Pipeline

### 6.1 Fragment Identification

```
Task: identify_fracture_fragments(case_id)

Input:  Binary mandible/maxilla/skull segmentation mask

Step 1: HU THRESHOLD + CONNECTED COMPONENTS
  Apply HU threshold: > 300 HU → bone mask
  scikit-image: label = skimage.measure.label(bone_mask, connectivity=2)
  Extract: region properties per component (volume, centroid, bounding box, extent)

Step 2: FRAGMENT FILTERING
  Remove components:
    - Volume < 100 mm³ (noise, dental root artifacts)
    - Volume < 5% of largest component (debris)
  Classify remaining as fracture fragments

Step 3: FRAGMENT LABELING
  For each fragment:
    - Determine anatomical region by centroid location relative to skull landmarks
    - Assign AOCMF topographic code (S, B, A, P, C)
    - Classify severity by fragment count and volume
  Store: fragment_inventory.json [{id, aocmf_code, volume_mm3, centroid_lps, mesh_path}]

Step 4: INFERENCE — Fragment Classification
  Call: POST inference-service:8080/predictions/fragment_classifier
  Input: Fragment mesh vertices as point cloud
  Output: [{fragment_id, predicted_class, confidence}]
```

### 6.2 Fracture Reduction Planning

```
Task: plan_fracture_reduction(case_id)

Input:  Fragment inventory + landmark set

Step 1: CONTRALATERAL MIRRORING (for unilateral injuries)
  Identify mid-sagittal plane from landmarks (nasion, ANS, menton)
  Mirror uninjured side to create reconstruction target
  Generate: target_anatomy_mirrored.stl

Step 2: FRAGMENT REGISTRATION TO TARGET
  For each fracture fragment:
    ICP registration of fragment surface to corresponding region of mirrored target
    Compute rigid transform (translation + rotation) to achieve target position
  Output: [{fragment_id, reduction_transform, fit_error_mm}]

Step 3: CONSTRAINT CHECKING
  Condyle positioning: verify both condyles seat within glenoid fossa bounds
    Criterion: condyle centroid within 2.0mm of glenoid fossa center
  Occlusal check: run occlusion analysis with reduced positions
    Accept if planned occlusion achieves target Angle class + overjet/overbite
  Anatomical continuity: verify no fragment overlaps after reduction

Step 4: PLAN RANKING
  Evaluate multiple reduction configurations (perturb from ICP optimum):
    - Translate each condyle ±1mm in 3 axes
    - Score each configuration: occlusal score + symmetry score + condyle seating score
  Rank top 3 configurations

Step 5: CONFIDENCE SCORING
  Call: POST inference-service:8080/predictions/plan_scorer
  Input: {fragment_transforms, occlusal_state, condyle_positions, landmark_errors}
  Output: {confidence, uncertainty_sources: [...]}

Step 6: STORE
  Postgres: INSERT surgical_plans [{confidence, movements, constraint_violations}] × 3 candidates
  MinIO: mesh-assets/{case_id}/plans/plan_{id}/
            planned_positions.stl (all fragments at planned positions)
            reduction_report.json
```

---

## 7. Occlusion Analysis Pipeline

### 7.1 Contact Detection

```
Task: compute_occlusion(case_id)

Input:  Upper dental arch mesh + lower dental arch mesh (from mesh extraction)
        Optional: post-registration intraoral scan mesh (higher accuracy)

Step 1: COLLISION DETECTION
  Library: trimesh.collision.CollisionManager
  Find all contact pairs between upper and lower tooth meshes
  For each contact pair:
    - Identify which tooth IDs are in contact (FDI notation)
    - Compute contact area and penetration depth
    - Extract contact centroid

Step 2: ANGLE CLASSIFICATION
  First molar relationship:
    Locate upper first molar (FDI 16/26) and lower first molar (FDI 36/46) meshes
    Compute mesiobuccal cusp of upper relative to lower first molar groove
    Classify:
      Class I: upper cusp in lower groove (within 2mm tolerance)
      Class II: upper cusp anterior to lower groove
      Class III: upper cusp posterior to lower groove

Step 3: OVERJET AND OVERBITE
  Overjet: horizontal distance between upper incisor facial surface and
           lower incisor edge (measured in sagittal plane)
           Normal: 2–4mm
  Overbite: vertical overlap of upper incisors over lower incisors
             Normal: 2–4mm (positive = normal overlap, negative = open bite)

Step 4: MIDLINE DEVIATION
  Upper midline: midpoint between upper central incisors (FDI 11, 21)
  Lower midline: midpoint between lower central incisors (FDI 31, 41)
  Deviation: horizontal distance in coronal plane
             Clinically significant if > 2mm

Step 5: CONSTRAINT ENCODING
  Create OcclusalConstraint object:
    {
      "target_angle_class": "I",    (or from cephalometric analysis)
      "target_overjet_mm": [2, 4],  (acceptable range)
      "target_overbite_mm": [2, 4],
      "target_midline_dev_mm": [-1, 1],
      "contact_pairs_required": [...]  (functional contacts)
    }
  This object constrains all subsequent plan generation

Step 6: STORE
  Postgres: INSERT occlusal_analyses
    {case_id, angle_class, overjet_mm, overbite_mm, midline_dev_mm,
     contact_pairs, constraint_object, created_at}
  Emit: generate_plans.delay(case_id)  [if fracture reduction done]
```

---

## 8. Data Storage and Retrieval Patterns

### Write Path (during pipeline execution)

```
Worker → MinIO.put_object(bucket, key, data)  [streaming, no worker memory buffering]
Worker → Postgres (via SQLAlchemy asyncpg session)
Worker → Redis (Celery result, expires 24h)
```

### Read Path (during surgeon review)

```
API: GET /api/v1/meshes/{case_id}/{structure}
  → Postgres: fetch Mesh record (storage_path)
  → MinIO: generate presigned GET URL (expires 15 min)
  → Return presigned URL to client

Frontend: fetch GLTF from presigned URL directly (bypasses API)
  → Three.js: parse and render mesh
```

### Large Object Handling

Objects > 5MB (NIfTI volumes, large STL files) use MinIO multipart upload from workers:

```python
minio_client.fput_object(
    bucket_name="mesh-assets",
    object_name=f"{case_id}/meshes/mandible.stl",
    file_path=local_tmp_path,
    content_type="model/stl",
)
```

### Caching Strategy

| Data | Cache Location | TTL | Invalidation |
|------|---------------|-----|-------------|
| Case metadata | Redis (L1) | 5 min | On case status change |
| Mesh presigned URLs | Client browser | 14 min | Automatic expiry |
| Landmark sets | Postgres | Permanent | Only on surgeon override |
| Job status | Redis (Celery) | 24h | On job completion |
| Segmentation NIfTI | No cache | — | Read from MinIO on demand |

---

## 9. ML Training Data Collection Flow

Every case that runs through Facial Align generates training data if the platform is instrumented correctly. This is the data flywheel.

### 9.1 Segmentation Training Data Collection

```
Trigger: Surgeon reviews segmentation output

Events captured:
  1. SEGMENTATION_ACCEPTED:
     {case_id, model_name, model_version, timestamp}
     → Labeled positive: (volume.nii.gz, segmentation.nii.gz) pair is high-quality ground truth

  2. SEGMENTATION_CORRECTED:
     {case_id, structure_name, correction_mask_path, surgeon_id}
     → Labeled positive with corrections: most valuable training data
     → Store corrected mask as: mesh-assets/{case_id}/segmentation/corrected_{structure}.nii.gz

  3. SEGMENTATION_REJECTED:
     {case_id, reason_code, affected_structures, surgeon_id}
     → Labeled negative: this (volume, segmentation) pair should NOT be used as-is
     → Flag in SegmentationLabel table for human review before training use
```

### 9.2 Landmark Training Data Collection

```
Events captured:
  1. LANDMARK_ACCEPTED: landmark detection output accepted without modification
     → Training positive: (volume, landmark_set) with acceptance flag

  2. LANDMARK_OVERRIDDEN: surgeon corrected one or more landmark positions
     {landmark_name, original_position, corrected_position, correction_magnitude_mm}
     → High-value training signal: AI was wrong, surgeon provided correct answer
     → Store corrected landmark_set.json with is_corrected=True flags

  3. LANDMARK_UNCERTAINTY_VALIDATED: surgeon confirmed uncertain landmark was correctly flagged
     → Training signal for uncertainty model calibration
```

### 9.3 Surgical Plan Training Data Collection

```
Events captured:
  1. PLAN_CANDIDATE_SELECTED: which of 1-3 candidates the surgeon selected
     {case_id, selected_plan_id, rejected_plan_ids}
     → Training signal for plan ranking model

  2. PLAN_MODIFICATIONS:
     Each surgeon interaction with the plan is logged as a structured delta:
     {plan_id, modification_type, before_state, after_state, delta}
     → Training signal: what the surgeon changed and by how much

  3. PLAN_APPROVED: final plan state at approval
     {plan_id, final_movements, cephalometric_measurements, occlusal_state}
     → Ground truth plan for this case type

  4. POSTOP_COMPARISON (when post-op CT submitted):
     {plan_id, achieved_movements, deviation_per_segment}
     → Gold standard: actual outcome vs. plan
```

### 9.4 Training Dataset Assembly

```
Task: assemble_training_dataset(model_type, version_tag)

For segmentation model:
  SELECT cases WHERE:
    segmentation.status IN ('ACCEPTED', 'CORRECTED')
    AND segmentation.model_version != target_version  (don't train on own predictions)
    AND cases.created_at < cutoff_date
  Export: {volume.nii.gz, segmentation.nii.gz} pairs
  Train/val/test split: 70/15/15 stratified by case type
  Store dataset manifest: model-registry/datasets/{version_tag}/manifest.json

For landmark model:
  SELECT cases WHERE:
    landmark_sets.has_manual_correction = TRUE OR landmark_sets.was_validated = TRUE
  Export: {volume.nii.gz, landmarks.json} pairs

For plan scoring model:
  SELECT plan_comparisons WHERE:
    postop_deviation IS NOT NULL  (requires outcome data)
  Join with plan_modifications for context
  Export: {plan_state.json, outcome.json} pairs
  Note: requires ≥50 cases with postop data before first training run
```

### 9.5 Training Data Privacy

All training data is de-identified before use. The link between a case and its training data is maintained only in the encrypted `DataLineage` table. Model weights are never derived from identifiable information:

```
DataLineage table:
  id, case_id (FK, encrypted), dataset_version, export_hash, consent_scope
```

Training datasets stored in MinIO `model-registry/datasets/` have no patient-linkable identifiers — only internal case UUIDs that are themselves hashed before export.
