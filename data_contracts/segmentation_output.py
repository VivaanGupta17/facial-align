"""
Data contract for segmentation pipeline output.

This contract is produced by the segmentation service and consumed by:
- Fracture detection / fragment identification
- Mesh extraction pipeline
- Reduction planning service
- Frontend 3D viewer

Each ``SegmentationOutputContract`` captures the full result of running a
segmentation model on a CT volume: per-structure label maps, confidence
scores, 3D mesh references, volumetric statistics, and a structure hierarchy
describing parent-child relationships (e.g. ``condyle_L`` is a child of
``mandible``).

Clinical context
----------------
CMF segmentation targets: mandible (body + rami), maxilla, bilateral
zygomas, orbital floors, nasal bones, frontal bone, and all teeth.
For fracture cases, each fragment of a fractured bone is assigned a
separate label so downstream planning can treat them independently.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
MillimeterValue = float
CubicCentimeterValue = float


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SegmentationStatus(str, Enum):
    """Processing status of a segmentation job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    PARTIAL = "partial"          # Some structures failed, others succeeded


class StructureClass(str, Enum):
    """Broad anatomical class — used for display and filtering."""
    BONE = "bone"
    TOOTH = "tooth"
    AIRWAY = "airway"
    SOFT_TISSUE = "soft_tissue"
    IMPLANT = "implant"
    FRAGMENT = "fragment"         # Fracture fragment derived from a bone


# ---------------------------------------------------------------------------
# Structure hierarchy registry
# ---------------------------------------------------------------------------

STRUCTURE_HIERARCHY: Dict[str, Optional[str]] = {
    # structure_name -> parent_structure_name (None = root)
    "skull_base": None,
    "frontal_bone": "skull_base",
    "parietal_L": "skull_base",
    "parietal_R": "skull_base",
    "temporal_L": "skull_base",
    "temporal_R": "skull_base",
    "naso_orbital_ethmoid": "skull_base",
    "nasal_bones": "naso_orbital_ethmoid",
    "orbital_floor_L": "skull_base",
    "orbital_floor_R": "skull_base",
    "zygoma_L": "skull_base",
    "zygoma_R": "skull_base",
    "maxilla": None,
    "mandible": None,
    "condyle_L": "mandible",
    "condyle_R": "mandible",
    "coronoid_L": "mandible",
    "coronoid_R": "mandible",
    # Teeth (FDI notation)
    **{f"tooth_{n}": ("maxilla" if n <= 28 else "mandible")
       for n in list(range(11, 29)) + list(range(31, 49))},
    # Airway
    "nasopharynx": None,
    "oropharynx": None,
}


# ---------------------------------------------------------------------------
# Per-structure mesh reference
# ---------------------------------------------------------------------------

class StructureMesh(BaseModel):
    """
    Geometry artefacts for a single segmented structure.

    Paths are relative to the case storage root.  At minimum one of
    ``glb_path`` or ``stl_path`` should be populated.
    """
    model_config = ConfigDict(populate_by_name=True)

    structure_name: str = Field(..., description="Canonical structure name (see STRUCTURE_HIERARCHY)")
    label_value: int = Field(..., ge=1, description="Voxel label integer in the mask volume")
    structure_class: StructureClass = Field(
        default=StructureClass.BONE, description="Broad anatomical class"
    )
    parent_structure: Optional[str] = Field(
        None, description="Parent structure name (derived from STRUCTURE_HIERARCHY)"
    )

    # File paths
    glb_path: Optional[str] = Field(None, description="GLB file path (for web viewer)")
    stl_path: Optional[str] = Field(None, description="STL file path (for CAD/CAM)")
    ply_path: Optional[str] = Field(None, description="PLY file path (for ML pipeline)")

    # Mesh metrics
    vertex_count: Optional[int] = Field(None, ge=0)
    face_count: Optional[int] = Field(None, ge=0)
    volume_cc: Optional[CubicCentimeterValue] = Field(
        None, ge=0, description="Mesh-derived volume in cm³"
    )
    surface_area_mm2: Optional[MillimeterValue] = Field(
        None, ge=0, description="Surface area in mm²"
    )
    is_watertight: Optional[bool] = Field(
        None, description="Whether mesh has no topological holes"
    )
    mean_edge_length_mm: Optional[float] = Field(
        None, ge=0, description="Mean triangle edge length in mm"
    )

    @property
    def has_geometry(self) -> bool:
        """True if at least one mesh file path is available."""
        return any([self.glb_path, self.stl_path, self.ply_path])


# ---------------------------------------------------------------------------
# Per-structure confidence and volume statistics
# ---------------------------------------------------------------------------

class StructureStats(BaseModel):
    """
    Detailed per-structure quantitative statistics.

    Complements the coarser ``confidences`` dict at the top-level contract
    with volume-range checks, Dice score against atlas priors (if computed),
    and edge-quality metrics.
    """
    structure_name: str
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence for this structure")
    dice_vs_atlas: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Dice overlap against probabilistic atlas prior (1.0 = perfect)"
    )
    volume_cc: Optional[float] = Field(None, ge=0, description="Volume in cm³")
    volume_within_expected_range: Optional[bool] = Field(
        None, description="Whether volume is within population-normal range for this structure"
    )
    expected_volume_range_cc: Optional[Tuple[float, float]] = Field(
        None, description="(min, max) expected volume in cm³ for a typical adult"
    )
    voxel_count: Optional[int] = Field(None, ge=0)
    centroid_mm: Optional[List[float]] = Field(
        None, min_length=3, max_length=3,
        description="[x, y, z] centroid in patient-coordinate mm"
    )
    is_fragmented: bool = Field(
        False, description="Whether this structure was detected as fractured"
    )
    fragment_count: Optional[int] = Field(
        None, ge=1, description="Number of fragments if fragmented"
    )


# ---------------------------------------------------------------------------
# Top-level segmentation output contract
# ---------------------------------------------------------------------------

class SegmentationOutputContract(BaseModel):
    """
    Canonical segmentation output used across all service boundaries.

    Produced by: ``services/segmentation/segmentation_service.py``
    Consumed by: fracture detection, mesh extraction, reduction planning, viewer.

    Confidence semantics
    --------------------
    ``overall_confidence`` is the macro-averaged Dice-equivalent confidence
    across all structures.  Per-structure confidences are in ``confidences``
    and the richer ``structure_stats`` list.

    Structure hierarchy
    -------------------
    ``structure_hierarchy`` maps each segmented structure name to its
    anatomical parent.  This enables the viewer and planning service to
    understand, e.g., that ``condyle_L`` belongs to ``mandible``.
    """
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [{
                "segmentation_id": "seg-abc123",
                "case_id": "case-xyz456",
                "model_name": "totalsegmentator_cmf",
                "model_version": "2.0.1",
                "labels": {"mandible": 1, "maxilla": 2, "zygoma_L": 3, "zygoma_R": 4},
                "confidences": {"mandible": 0.97, "maxilla": 0.95, "zygoma_L": 0.91, "zygoma_R": 0.92},
                "overall_confidence": 0.94,
                "status": "complete",
            }]
        },
    )

    # Identifiers
    segmentation_id: str = Field(..., description="Unique segmentation job ID")
    case_id: str = Field(..., description="Owning surgical case ID")

    # Model provenance
    model_name: str = Field(..., description="Segmentation model identifier")
    model_version: str = Field(..., description="Model version string (SemVer)")
    inference_device: Optional[str] = Field(
        None, description="Compute device used (e.g. 'cuda:0', 'cpu')"
    )

    # Mask artefact
    mask_path: Optional[str] = Field(
        None, description="Path to multi-label .nii.gz mask volume"
    )
    fragment_mask_path: Optional[str] = Field(
        None, description="Path to fragment-level label mask (fragments have unique labels)"
    )
    spacing_mm: Optional[List[MillimeterValue]] = Field(
        None, min_length=3, max_length=3,
        description="[x, y, z] voxel spacing of the mask in mm"
    )

    # Per-structure results
    labels: Dict[str, int] = Field(
        ..., description="structure_name → integer label value in mask"
    )
    confidences: Dict[str, float] = Field(
        default_factory=dict,
        description="structure_name → confidence score in [0, 1]"
    )
    overall_confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Macro-averaged confidence across all structures"
    )
    structure_stats: List[StructureStats] = Field(
        default_factory=list,
        description="Detailed per-structure statistics including volume and Dice"
    )

    # Mesh artefacts
    meshes: List[StructureMesh] = Field(
        default_factory=list,
        description="3D mesh references per segmented structure"
    )
    volume_stats: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="structure_name → {volume_cc, surface_area_mm2, ...}"
    )

    # Structure hierarchy
    structure_hierarchy: Dict[str, Optional[str]] = Field(
        default_factory=dict,
        description="structure_name → parent_structure_name (None if root)"
    )

    # Fracture
    fragment_count: Optional[int] = Field(
        None, ge=0, description="Total number of fracture fragments identified"
    )
    fractured_structures: List[str] = Field(
        default_factory=list,
        description="Names of structures detected as fractured"
    )

    # Performance
    inference_time_ms: int = Field(
        default=0, ge=0, description="Model inference duration in milliseconds"
    )
    post_processing_time_ms: int = Field(
        default=0, ge=0, description="Post-processing (mesh extraction, etc.) in milliseconds"
    )

    # Status
    status: SegmentationStatus = Field(
        default=SegmentationStatus.COMPLETE,
        description="Segmentation job status"
    )
    error_message: Optional[str] = Field(
        None, description="Error detail if status is FAILED"
    )
    created_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("confidences")
    @classmethod
    def validate_confidence_values(cls, v: Dict[str, float]) -> Dict[str, float]:
        for name, conf in v.items():
            if not 0.0 <= conf <= 1.0:
                raise ValueError(
                    f"Confidence for structure '{name}' must be in [0, 1], got {conf}"
                )
        return v

    @field_validator("labels")
    @classmethod
    def validate_label_values(cls, v: Dict[str, int]) -> Dict[str, int]:
        if len(set(v.values())) != len(v):
            raise ValueError("Label values must be unique across structures")
        return v

    @model_validator(mode="after")
    def populate_hierarchy_defaults(self) -> "SegmentationOutputContract":
        """Auto-populate structure_hierarchy from global STRUCTURE_HIERARCHY if not set."""
        if not self.structure_hierarchy:
            self.structure_hierarchy = {
                name: STRUCTURE_HIERARCHY.get(name)
                for name in self.labels
            }
        return self

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def total_bone_volume_cc(self) -> float:
        """Sum of volumes of all bone-class structures (excluding teeth and fragments)."""
        total = 0.0
        for stat in self.structure_stats:
            if stat.volume_cc and not stat.structure_name.startswith("tooth_"):
                total += stat.volume_cc
        return total

    @property
    def has_fractures(self) -> bool:
        """True if any fractured structures were detected."""
        return bool(self.fractured_structures) or (
            self.fragment_count is not None and self.fragment_count > 0
        )

    def get_mesh(self, structure_name: str) -> Optional[StructureMesh]:
        """Return the StructureMesh for the given structure name, or None."""
        for m in self.meshes:
            if m.structure_name == structure_name:
                return m
        return None

    def children_of(self, parent_name: str) -> List[str]:
        """Return all structure names that are children of the given parent."""
        return [
            name for name, parent in self.structure_hierarchy.items()
            if parent == parent_name
        ]

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SegmentationOutputContract":
        return cls.model_validate(data)
