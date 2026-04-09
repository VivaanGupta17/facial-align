"""
Data contract for supervised model predictions.

A ``SupervisedPredictionContract`` is the output of the supervised inference
service and feeds into the post-processing pipeline (transform application,
collision resolution, STL export).

This contract bridges the supervised model's tensor outputs with the
existing ``ReductionPlanContract`` and ``OcclusionPlanContract`` formats.

The confidence and routing fields enable the confidence-based fallback
system described in SUPERVISED_REDESIGN.md §6.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ConfidenceLevel(str, Enum):
    """Confidence-based routing category."""
    ACCEPT = "accept"          # ≥0.8: proceed to export
    REVIEW = "review"          # 0.5–0.8: surgeon must approve
    FALLBACK = "fallback"      # <0.5: run optimisation pipeline
    REJECT = "reject"          # any fragment <0.3: manual planning required


class ModalityType(str, Enum):
    """Input modality configuration."""
    CT_ONLY = "ct_only"
    CT_IOS = "ct_ios"


class FragmentPrediction(BaseModel):
    """Predicted SE(3) transform for a single bone fragment."""
    fragment_id: str
    se3_matrix: List[List[float]] = Field(
        ..., description="4x4 SE(3) homogeneous transform matrix",
    )
    translation_mm: List[float] = Field(
        ..., description="Translation vector [x, y, z] in mm",
    )
    rotation_r6: List[float] = Field(
        ..., description="R6 continuous rotation representation",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Fragment confidence score",
    )
    aleatoric_uncertainty: Optional[float] = Field(
        None, description="Predicted aleatoric variance",
    )
    epistemic_uncertainty: Optional[float] = Field(
        None, description="MC Dropout epistemic variance",
    )


class ToothPrediction(BaseModel):
    """Predicted SE(3) transform for a single tooth."""
    fdi_number: int = Field(..., description="FDI tooth number")
    se3_matrix: List[List[float]] = Field(
        ..., description="4x4 SE(3) transform matrix",
    )
    translation_mm: List[float] = Field(
        ..., description="Translation [x, y, z] in mm",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)


class OcclusionMetricsPrediction(BaseModel):
    """Predicted clinical occlusion metrics."""
    overjet_mm: float = Field(..., description="Predicted overjet in mm")
    overbite_mm: float = Field(..., description="Predicted overbite in mm")
    midline_deviation_mm: float = Field(
        ..., description="Predicted midline deviation in mm",
    )
    predicted_molar_class: str = Field(
        ..., description="Predicted Angle molar class (I, II, or III)",
    )
    molar_class_probabilities: Dict[str, float] = Field(
        default_factory=dict,
        description="Softmax probabilities for each molar class",
    )


class SupervisedPredictionContract(BaseModel):
    """
    Complete output of the supervised inference service.

    This contract captures all model predictions, confidence routing
    decisions, and metadata needed for downstream processing.

    Usage:
        prediction = SupervisedPredictionContract(
            case_id="case_001",
            fragments=[FragmentPrediction(...)],
            ...
        )

        if prediction.confidence_level == ConfidenceLevel.ACCEPT:
            # Proceed to post-processing and STL export
            ...
        elif prediction.confidence_level == ConfidenceLevel.FALLBACK:
            # Run optimisation pipeline
            ...
    """
    # Identifiers
    case_id: str
    plan_id: Optional[str] = None
    model_version: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Input modality
    modality: ModalityType = ModalityType.CT_ONLY

    # Fragment predictions
    fragments: List[FragmentPrediction] = Field(default_factory=list)
    num_fragments: int = 0

    # Tooth predictions
    teeth: List[ToothPrediction] = Field(default_factory=list)

    # Occlusion metrics
    occlusion_metrics: Optional[OcclusionMetricsPrediction] = None

    # Confidence routing
    overall_confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="Aggregate confidence score",
    )
    confidence_level: ConfidenceLevel = ConfidenceLevel.REJECT
    route_to_fallback: bool = False
    requires_surgeon_review: bool = False
    reject_reasons: List[str] = Field(default_factory=list)

    # Inference metadata
    inference_time_ms: float = 0.0
    mc_dropout_passes: int = 0
    device: str = "cpu"

    @field_validator("fragments")
    @classmethod
    def validate_fragments(cls, v: List[FragmentPrediction]) -> List[FragmentPrediction]:
        for frag in v:
            if len(frag.se3_matrix) != 4 or any(len(row) != 4 for row in frag.se3_matrix):
                raise ValueError(
                    f"Fragment {frag.fragment_id}: se3_matrix must be 4x4"
                )
        return v

    def to_reduction_plan_kwargs(self) -> Dict[str, Any]:
        """
        Convert to keyword arguments compatible with ReductionPlanContract.

        This bridges the supervised prediction output to the existing
        reduction plan format, enabling seamless integration with the
        downstream review and export workflows.
        """
        transforms = {}
        confidences = {}
        for frag in self.fragments:
            transforms[frag.fragment_id] = frag.se3_matrix
            confidences[frag.fragment_id] = frag.confidence

        return {
            "case_id": self.case_id,
            "plan_origin": "ml_generated",
            "overall_confidence": self.overall_confidence,
            "fragment_transforms": transforms,
            "fragment_confidences": confidences,
            "model_version": self.model_version,
        }
