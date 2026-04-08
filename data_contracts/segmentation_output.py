"""Data contract for segmentation pipeline output."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class StructureMesh(BaseModel):
    structure_name: str
    label_value: int
    glb_path: Optional[str] = None
    stl_path: Optional[str] = None
    ply_path: Optional[str] = None
    vertex_count: Optional[int] = None
    face_count: Optional[int] = None
    volume_cc: Optional[float] = None
    is_watertight: Optional[bool] = None


class SegmentationOutputContract(BaseModel):
    """Canonical segmentation output used across service boundaries."""
    segmentation_id: str
    case_id: str
    model_name: str
    model_version: str
    mask_path: Optional[str] = Field(None, description="Path to .nii.gz mask volume")
    labels: Dict[str, int] = Field(..., description="structure_name -> label_value")
    confidences: Dict[str, float] = Field(default_factory=dict)
    overall_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    meshes: List[StructureMesh] = Field(default_factory=list)
    volume_stats: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    fragment_count: Optional[int] = None
    fragment_mask_path: Optional[str] = None
    spacing_mm: Optional[List[float]] = None
    inference_time_ms: int = 0
    status: str = "complete"
