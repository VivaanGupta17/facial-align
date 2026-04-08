"""Data contract for intraoral scan (IOS) data."""
from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class IntraoralScanContract(BaseModel):
    """Intraoral scan (digital dental impression) data contract."""
    scan_id: str
    patient_id: str
    scan_date: Optional[datetime] = None
    arch: str = Field(..., description="upper, lower, or both")
    scanner_manufacturer: Optional[str] = None
    scanner_model: Optional[str] = None
    file_format: str = Field(default="stl", description="stl, ply, obj, 3shape")
    upper_arch_path: Optional[str] = Field(None, description="Path to upper arch mesh")
    lower_arch_path: Optional[str] = Field(None, description="Path to lower arch mesh")
    bite_registration_path: Optional[str] = Field(None, description="Bite registration mesh")
    tooth_labels: Dict[str, int] = Field(
        default_factory=dict,
        description="FDI tooth number -> mesh region label"
    )
    has_restorations: bool = False
    is_pre_injury: bool = False
    notes: Optional[str] = None
