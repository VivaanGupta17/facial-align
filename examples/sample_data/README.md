# Sample Data

This directory contains synthetic / publicly available sample data for testing the Facial Align pipeline.

## No Patient Data

This directory must NEVER contain real patient data, PHI, or data derived from clinical cases.

## Available Samples

### `sample_ct_metadata.json`
Synthetic DICOM metadata demonstrating the expected structure for a maxillofacial CT study.

### `sample_case.json`
Example surgical case object showing the full data model.

### `sample_segmentation_output.json`
Example segmentation result with structure labels and confidence scores.

### `sample_reduction_plan.json`
Example fracture reduction plan with fragment transforms and occlusal metrics.

## Public Datasets for Development

For development and model training, use these publicly available datasets:

1. **PDDCA** — Public Domain Database for Computational Anatomy (head/neck CT)
   - Source: https://www.imagenglab.com/newsite/pddca/
   
2. **ToothFairy2** — MICCAI 2024 CBCT multi-structure segmentation challenge
   - Source: https://toothfairy2.grand-challenge.org/
   
3. **Mandibular Defect Dataset** — 147 mandible CT cases (Shanghai Ninth)
   - Source: https://www.synapse.org/

4. **CQ500** — Head CT dataset (qure.ai)
   - Source: http://headctstudy.qure.ai/dataset
