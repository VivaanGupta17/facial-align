"""Data contract for occlusal splint design request and specification."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SplintDesignRequest(BaseModel):
    """Request for automated splint design generation."""
    case_id: str
    plan_id: str
    splint_type: str = Field(
        default="intermediate",
        description="Splint type: intermediate (surgical), final (post-op)"
    )
    target_vd_mm: Optional[float] = Field(None, description="Target vertical dimension in mm")
    arch: str = Field(default="both", description="Arch to include: upper, lower, both")
    material: str = Field(default="acrylic_resin", description="Splint material")
    include_bite_blocks: bool = Field(True, description="Include posterior bite blocks")
    retention_type: str = Field(
        default="clasps",
        description="Retention mechanism: clasps, vacuum_formed, bonded"
    )
    output_format: str = Field(default="stl", description="Output format: stl, 3mf, step")


class SplintDesignSpec(BaseModel):
    """Splint design output specification."""
    case_id: str
    plan_id: str
    splint_type: str
    upper_component_path: Optional[str] = None
    lower_component_path: Optional[str] = None
    target_vertical_dimension_mm: float
    material: str
    estimated_thickness_map: Dict[str, float] = Field(default_factory=dict)
    contact_regions: List[Dict[str, Any]] = Field(default_factory=list)
    fabrication_notes: str = ""
    cad_cam_compatible: bool = True
    requires_manual_finish: bool = False
