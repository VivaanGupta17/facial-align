"""
Data contract for CT imaging study.
Used across service boundaries for interoperability.
"""
from __future__ import annotations
from datetime import date, datetime
from typing import Any, Dict, List, Optional
import json
from pydantic import BaseModel, Field


class CTSeriesContract(BaseModel):
    """A single CT image series."""
    series_instance_uid: str
    modality: str = "CT"
    series_number: Optional[int] = None
    slice_count: int
    slice_thickness_mm: Optional[float] = None
    pixel_spacing_mm: Optional[List[float]] = None  # [row, col]
    kvp: Optional[float] = None
    storage_path: Optional[str] = None


class CTStudyContract(BaseModel):
    """
    Canonical CT study data contract.
    Shared between ingestion, segmentation, and planning services.
    """
    study_uid: str = Field(..., description="De-identified StudyInstanceUID")
    patient_id: str = Field(..., description="Internal patient UUID")
    modality: str = Field(default="CT")
    acquisition_date: Optional[date] = None
    body_part_examined: Optional[str] = None
    volume_path: Optional[str] = Field(None, description="Path to NIfTI volume file")
    spacing_mm: Optional[List[float]] = Field(None, description="[x, y, z] voxel spacing")
    volume_shape: Optional[List[int]] = Field(None, description="[Z, Y, X] volume dimensions")
    series: List[CTSeriesContract] = Field(default_factory=list)
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    quality_flags: List[str] = Field(default_factory=list)
    is_deidentified: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_json_schema(self) -> str:
        return json.dumps(self.model_json_schema(), indent=2)
