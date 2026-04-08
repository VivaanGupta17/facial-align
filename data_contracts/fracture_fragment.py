"""Data contract for fracture fragment geometry and transform."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FragmentGeometry(BaseModel):
    """Geometric properties of a single bone fragment."""
    fragment_id: str
    label_value: int
    parent_structure: str
    mesh_path: Optional[str] = None
    centroid_mm: List[float] = Field(..., description="[x, y, z] centroid in patient coords")
    volume_mm3: float
    surface_area_mm2: Optional[float] = None
    bounding_box: Optional[Dict[str, float]] = None
    is_reference: bool = Field(False, description="True if this fragment is the anatomical anchor")
    confidence: float = Field(..., ge=0.0, le=1.0)


class FragmentTransformContract(BaseModel):
    """Planned rigid body transform for a fracture fragment."""
    fragment_id: str
    rotation_matrix: List[List[float]] = Field(..., description="3x3 rotation matrix")
    translation_mm: List[float] = Field(..., description="[x, y, z] translation in mm")
    confidence: float = Field(..., ge=0.0, le=1.0)
    is_surgeon_edit: bool = False
    alternative_transforms: List[Dict[str, Any]] = Field(default_factory=list)


class FractureFragmentContract(BaseModel):
    """Complete fracture fragment data contract (geometry + planned transform)."""
    geometry: FragmentGeometry
    planned_transform: Optional[FragmentTransformContract] = None
    hardware_recommendation: Optional[str] = None
    surgical_sequence: Optional[int] = None
    notes: Optional[str] = None
