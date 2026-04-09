"""
Pydantic schemas for STL export of reduction plans.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import Field

from app.schemas.common import BaseSchema


class ExportRequest(BaseSchema):
    """Request to export a reduction plan as STL files."""

    export_type: str = Field(
        default="full_assembly",
        description="Export type: full_assembly, corrected_mandible, individual_fragment",
    )
    stl_format: str = Field(
        default="binary",
        description="STL encoding: binary or ascii",
    )
    structure_name: Optional[str] = Field(
        default=None,
        description="Structure name for individual_fragment export",
    )


class ExportFileInfo(BaseSchema):
    """Metadata for a single exported STL file."""

    filename: str
    export_type: str
    download_url: str
    vertex_count: int
    face_count: int
    volume_mm3: float
    is_watertight: bool
    is_printable: bool


class ExportResponse(BaseSchema):
    """Response containing exported STL file details."""

    plan_id: str
    case_id: str
    files: List[ExportFileInfo]
    total_export_time_seconds: float
