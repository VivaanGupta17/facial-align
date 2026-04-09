"""
Ground truth generation from pre/post-op CT registration.

- PreopPostopRegistration: CT pair registration to derive per-fragment transforms
- LandmarkAnnotation: semi-automated landmark extraction
- OcclusionMetricExtraction: post-op occlusal metric extraction
"""

from training.ground_truth.landmark_annotation import LandmarkAnnotation
from training.ground_truth.occlusion_metric_extraction import OcclusionMetricExtraction
from training.ground_truth.preop_postop_registration import PreopPostopRegistration

__all__ = [
    "LandmarkAnnotation",
    "OcclusionMetricExtraction",
    "PreopPostopRegistration",
]
