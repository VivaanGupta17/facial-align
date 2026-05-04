"""
Pydantic schemas for segmentation requests, results, and confidence data.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator

from app.schemas.capabilities import ProvenanceInfo
from app.schemas.common import BaseSchema, BoundingBox3D


class StructureLabel(BaseSchema):
    """A single anatomical structure in the segmentation output."""
    name: str = Field(..., description="Anatomical structure name")
    label_value: int = Field(..., description="Integer voxel label in the mask volume")
    dicom_code: Optional[str] = Field(
        None, description="DICOM Segment Attribute code (e.g., T-11180 for mandible)"
    )
    snomed_id: Optional[str] = Field(None, description="SNOMED CT concept ID")
    color_rgb: Optional[list[int]] = Field(
        None, description="Suggested rendering color [R, G, B] 0-255"
    )
    is_bilateral: bool = Field(
        False, description="True for bilateral structures (L and R variants present)"
    )
    laterality: Optional[str] = Field(
        None, description="Laterality: L, R, or None for midline"
    )


# Standard CMF segmentation labels
CMF_STRUCTURE_LABELS: list[StructureLabel] = [
    StructureLabel(name="mandible",         label_value=1,  dicom_code="T-11180", color_rgb=[255, 165, 0]),
    StructureLabel(name="maxilla",          label_value=2,  dicom_code="T-11170", color_rgb=[255, 215, 0]),
    StructureLabel(name="zygoma_L",         label_value=3,  color_rgb=[30, 144, 255], laterality="L"),
    StructureLabel(name="zygoma_R",         label_value=4,  color_rgb=[100, 149, 237], laterality="R"),
    StructureLabel(name="orbital_floor_L",  label_value=5,  color_rgb=[144, 238, 144], laterality="L"),
    StructureLabel(name="orbital_floor_R",  label_value=6,  color_rgb=[0, 255, 127],   laterality="R"),
    StructureLabel(name="nasal_bones",      label_value=7,  color_rgb=[255, 182, 193]),
    StructureLabel(name="frontal_bone",     label_value=8,  color_rgb=[221, 160, 221]),
    StructureLabel(name="temporal_bone_L",  label_value=9,  color_rgb=[176, 224, 230], laterality="L"),
    StructureLabel(name="temporal_bone_R",  label_value=10, color_rgb=[173, 216, 230], laterality="R"),
    StructureLabel(name="pterygoid_plates", label_value=11, color_rgb=[240, 230, 140]),
    StructureLabel(name="skull_base",       label_value=12, color_rgb=[210, 180, 140]),
]


class ConfidenceMap(BaseSchema):
    """Per-structure segmentation confidence scores."""
    structure_name: str
    dice_score: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Estimated Dice coefficient vs expected anatomy"
    )
    mean_confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Mean softmax confidence across structure voxels"
    )
    min_confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Minimum confidence (lowest-certainty voxel)"
    )
    volume_voxels: Optional[int] = Field(None, description="Segmented volume in voxels")
    volume_cc: Optional[float] = Field(None, description="Segmented volume in cubic centimeters")
    bounding_box: Optional[BoundingBox3D] = Field(
        None, description="Structure bounding box in patient coordinates (mm)"
    )
    quality_flag: Optional[str] = Field(
        None, description="Quality warning: ok, low_confidence, fragmented, out_of_fov"
    )


class SegmentationRequest(BaseSchema):
    """Request to trigger bone segmentation on a study."""
    case_id: uuid.UUID = Field(..., description="Surgical case to segment")
    model_name: str = Field(
        default="totalsegmentator",
        description="Segmentation model to use: 'totalsegmentator', 'cmf_custom_v1'"
    )
    structures: Optional[List[str]] = Field(
        None,
        description="Specific structures to segment. If None, segment all CMF structures."
    )
    run_dental_segmentation: bool = Field(
        False,
        description="Also run per-tooth dental segmentation (requires CBCT)"
    )
    identify_fragments: bool = Field(
        True,
        description="Run fragment identification for trauma cases"
    )
    fast_mode: bool = Field(
        False,
        description="Use fast (lower accuracy) inference mode"
    )
    gpu_device: Optional[str] = Field(
        None, description="GPU device override (e.g., 'cuda:1')"
    )


class MeshInfo(BaseSchema):
    """Metadata about a generated mesh file."""
    structure_name: str
    format: str = Field(..., description="File format: glb, stl, obj, ply")
    path: str = Field(..., description="Storage path")
    file_size_bytes: Optional[int] = None
    vertex_count: Optional[int] = None
    face_count: Optional[int] = None
    volume_cc: Optional[float] = None
    surface_area_mm2: Optional[float] = None
    is_watertight: Optional[bool] = None


class SegmentationResult(BaseSchema):
    """Complete segmentation result returned to API clients."""
    id: uuid.UUID
    case_id: uuid.UUID
    status: str = Field(..., description="pending, running, complete, failed")
    model_name: str
    model_version: str

    structure_labels: Optional[List[StructureLabel]] = None
    confidence_maps: Optional[List[ConfidenceMap]] = None
    structure_reviews: Optional[Dict[str, Any]] = Field(
        None,
        description="Per-structure clinician review state keyed by structure name",
    )
    overall_confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Aggregate segmentation confidence"
    )
    provenance: Optional[ProvenanceInfo] = None

    mask_storage_path: Optional[str] = None
    meshes: Optional[List[MeshInfo]] = None

    fragment_count: Optional[int] = Field(
        None, description="Number of identified bone fragments (trauma)"
    )
    fracture_fragments: Optional[List[dict[str, Any]]] = Field(
        None,
        description="Explicit fracture fragment definitions derived from connected components",
    )

    inference_time_ms: Optional[int] = None
    total_pipeline_time_ms: Optional[int] = None
    gpu_device: Optional[str] = None

    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class SegmentationJobResponse(BaseSchema):
    """Response when a segmentation job is triggered."""
    job_id: str = Field(..., description="Celery task ID")
    segmentation_id: uuid.UUID = Field(..., description="Created SegmentationResult record ID")
    case_id: uuid.UUID
    status: str = Field(default="pending")
    estimated_duration_seconds: Optional[int] = Field(
        None, description="Estimated processing time based on volume size and model"
    )
    message: str = Field(default="Segmentation job submitted")
