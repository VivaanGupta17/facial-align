"""
Data contract for fracture fragment geometry and planned rigid-body transform.

Each ``FractureFragmentContract`` pairs geometric information about a single
bone fragment with the planned transform that repositions it to its correct
anatomical location.

AO CMF Classification
---------------------
Fragment classification follows the AO Comprehensive Classification of
Craniomaxillofacial Fractures (AO CMF 2021).  The ``ao_cmf_class`` field
encodes the most specific applicable code.  This drives hardware selection
defaults.

Load-bearing fragments
----------------------
The ``is_load_bearing`` property identifies fragments that must be reduced
and fixed before adjacent fragments can be positioned.  In the mandible,
the body and symphysis are load-bearing; condylar fragments are generally
not (they self-seat in the glenoid fossa).
"""

from __future__ import annotations

import json
import math
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
MillimeterValue = float
SquareMillimeterValue = float
CubicMillimeterValue = float


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AOCMFRegion(str, Enum):
    """
    Anatomical regions per AO CMF classification schema.

    These map to the first tier of the AO CMF hierarchy.
    """
    MANDIBLE = "mandible"
    MIDFACE = "midface"               # Le Fort I / II, ZMC
    UPPER_FACE = "upper_face"         # Frontal, NOE
    ORBITAL = "orbital"               # Orbital walls, floor, roof
    CONDYLE = "condyle"               # Subcondylar / condylar head
    ALVEOLAR = "alveolar"             # Alveolar process fractures
    DENTOALVEOLAR = "dentoalveolar"   # Combined tooth-bone involvement
    PANFACIAL = "panfacial"           # Multi-level, ≥2 zones


class FragmentContactSurface(str, Enum):
    """
    Classification of the fracture contact surface morphology.

    Used to guide reduction approach and hardware selection.
    """
    SIMPLE = "simple"            # Single planar fracture line
    COMMINUTED = "comminuted"    # Multiple small fragments at contact zone
    IMPACTED = "impacted"        # Fragments driven into each other
    AVULSED = "avulsed"          # Fragment separated from all contacts
    OBLIQUE = "oblique"          # Oblique fracture plane
    TRANSVERSE = "transverse"    # Transverse fracture plane


class HardwareType(str, Enum):
    """Recommended fixation hardware class."""
    MINIPLATE_2MM = "miniplate_2mm"
    MINIPLATE_1_5MM = "miniplate_1.5mm"
    MICROPLATE_1MM = "microplate_1mm"
    RECONSTRUCTION_PLATE = "reconstruction_plate"
    WIRE_IMF = "wire_imf"
    WIRE_TRANSOSSEOUS = "wire_transosseous"
    SCREW_LAG = "screw_lag"
    EXTERNAL_FIXATOR = "external_fixator"
    TITANIUM_MESH = "titanium_mesh"
    SPLINT_ONLY = "splint_only"


# ---------------------------------------------------------------------------
# Geometry sub-model
# ---------------------------------------------------------------------------

class FragmentGeometry(BaseModel):
    """
    Geometric properties of a single bone fragment as measured from the CT.

    Coordinate system: DICOM patient coordinates (LPS — left, posterior,
    superior positive).  All distances in millimetres.
    """
    model_config = ConfigDict(populate_by_name=True)

    fragment_id: str = Field(..., description="Unique fragment identifier (e.g. 'frag_01')")
    label_value: int = Field(..., ge=1, description="Integer label in the fragment mask volume")
    parent_structure: str = Field(
        ..., description="Anatomical structure this fragment belongs to (e.g. 'mandible')"
    )
    ao_cmf_region: Optional[AOCMFRegion] = Field(
        None, description="AO CMF anatomical region classification"
    )
    ao_cmf_code: Optional[str] = Field(
        None, description="Full AO CMF code (e.g. '91-A2.2' for symphysis simple fracture)"
    )

    # Paths to mesh files
    mesh_path: Optional[str] = Field(None, description="Path to STL/PLY/GLB fragment mesh")
    glb_path: Optional[str] = Field(None, description="GLB mesh path for web viewer")

    # Geometric measurements
    centroid_mm: List[float] = Field(
        ..., min_length=3, max_length=3,
        description="[x, y, z] centroid in patient-space mm"
    )
    volume_mm3: CubicMillimeterValue = Field(
        ..., gt=0, description="Fragment volume in mm³"
    )
    surface_area_mm2: Optional[SquareMillimeterValue] = Field(
        None, ge=0, description="Fragment surface area in mm²"
    )
    bounding_box: Optional[Dict[str, float]] = Field(
        None,
        description=(
            "Axis-aligned bounding box: "
            "{min_x, min_y, min_z, max_x, max_y, max_z} in mm"
        )
    )

    # Fracture surface characterisation
    contact_surface_type: Optional[FragmentContactSurface] = Field(
        None, description="Morphology of the fracture surface(s)"
    )
    contact_area_mm2: Optional[SquareMillimeterValue] = Field(
        None, ge=0,
        description="Total area of fracture contact surfaces in mm²"
    )
    contact_fragment_ids: List[str] = Field(
        default_factory=list,
        description="IDs of adjacent fragments sharing a fracture surface with this one"
    )

    # Reference / anchor
    is_reference: bool = Field(
        False,
        description=(
            "True if this fragment serves as the anatomical reference for the reduction. "
            "Typically the largest segment of the mandible body or the uninjured maxilla."
        )
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Segmentation model confidence for this fragment's boundary"
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("volume_mm3")
    @classmethod
    def validate_volume(cls, v: float) -> float:
        if v < 1.0:
            raise ValueError(
                f"volume_mm3={v} is suspiciously small (<1 mm³). "
                "Verify segmentation — this may be a labelling artefact."
            )
        return v

    @field_validator("confidence")
    @classmethod
    def warn_low_confidence(cls, v: float) -> float:
        # Soft check — low confidence is valid but worth flagging
        return v

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def volume_cc(self) -> float:
        """Fragment volume in cm³ (1 cm³ = 1000 mm³)."""
        return self.volume_mm3 / 1000.0

    @property
    def is_load_bearing(self) -> bool:
        """
        Return True if this fragment is load-bearing and must be reduced first.

        Heuristic rules:
        - Is the reference fragment → always load-bearing
        - Mandible body or symphysis (large volume, non-condylar) → load-bearing
        - Condyle, coronoid, or small fragment → NOT load-bearing
        """
        if self.is_reference:
            return True
        non_bearing_keywords = {"condyle", "coronoid", "alveolar"}
        if any(kw in self.fragment_id.lower() for kw in non_bearing_keywords):
            return False
        # Fragments > 500 mm³ belonging to the mandible body are typically load-bearing
        if self.parent_structure == "mandible" and self.volume_mm3 > 500.0:
            return True
        return False

    @property
    def bounding_box_dimensions_mm(self) -> Optional[Tuple[float, float, float]]:
        """Return (dx, dy, dz) bounding box extents in mm, or None if not available."""
        if self.bounding_box:
            dx = self.bounding_box["max_x"] - self.bounding_box["min_x"]
            dy = self.bounding_box["max_y"] - self.bounding_box["min_y"]
            dz = self.bounding_box["max_z"] - self.bounding_box["min_z"]
            return (dx, dy, dz)
        return None


# ---------------------------------------------------------------------------
# Transform sub-model
# ---------------------------------------------------------------------------

class FragmentTransformContract(BaseModel):
    """
    Planned rigid-body transform for a single fracture fragment.

    The transform describes the movement from the fragment's current
    (post-injury) position to its planned (post-reduction) position.
    It is represented as a rotation matrix (SO(3)) and translation vector
    in patient-space millimetres.

    Transform convention
    --------------------
    p_reduced = R @ p_current + t

    where ``R`` is the 3×3 rotation matrix and ``t`` is the translation in mm.
    For the reference fragment, R = I₃ and t = [0, 0, 0].
    """
    model_config = ConfigDict(populate_by_name=True)

    fragment_id: str = Field(..., description="Must match the corresponding FragmentGeometry.fragment_id")
    rotation_matrix: List[List[float]] = Field(
        ..., description="3×3 orthonormal rotation matrix (row-major)"
    )
    translation_mm: List[float] = Field(
        ..., min_length=3, max_length=3,
        description="[tx, ty, tz] translation in patient-space mm"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Model confidence in this transform prediction"
    )
    is_surgeon_edit: bool = Field(
        False, description="True if this transform was modified by a surgeon"
    )
    alternative_transforms: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Alternative transform proposals sorted by confidence (highest first)"
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("rotation_matrix")
    @classmethod
    def validate_rotation_matrix(cls, v: List[List[float]]) -> List[List[float]]:
        import numpy as _np
        if len(v) != 3 or any(len(row) != 3 for row in v):
            raise ValueError("rotation_matrix must be 3×3")
        R = _np.array(v, dtype=float)
        RtR = R.T @ R
        if not _np.allclose(RtR, _np.eye(3), atol=1e-3):
            raise ValueError(
                "rotation_matrix is not orthonormal (R^T R ≠ I within tolerance 1e-3)"
            )
        return v

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def translation_magnitude_mm(self) -> float:
        """Euclidean magnitude of the translation vector in mm."""
        return math.sqrt(sum(t ** 2 for t in self.translation_mm))

    @property
    def rotation_angle_degrees(self) -> float:
        """Rotation angle in degrees derived from the rotation matrix trace."""
        import numpy as _np
        R = _np.array(self.rotation_matrix, dtype=float)
        # Rodrigues formula: angle = arccos((trace(R) - 1) / 2)
        trace = float(_np.trace(R))
        cos_angle = (trace - 1.0) / 2.0
        cos_angle = max(-1.0, min(1.0, cos_angle))  # numerical clamp
        return math.degrees(math.acos(cos_angle))

    @property
    def is_identity(self) -> bool:
        """Return True if the transform is the identity (reference fragment)."""
        return self.translation_magnitude_mm < 0.01 and self.rotation_angle_degrees < 0.05

    def to_4x4_matrix(self) -> List[List[float]]:
        """Return the homogeneous 4×4 SE(3) transformation matrix."""
        R = self.rotation_matrix
        t = self.translation_mm
        return [
            [R[0][0], R[0][1], R[0][2], t[0]],
            [R[1][0], R[1][1], R[1][2], t[1]],
            [R[2][0], R[2][1], R[2][2], t[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]


# ---------------------------------------------------------------------------
# Top-level fragment contract
# ---------------------------------------------------------------------------

class FractureFragmentContract(BaseModel):
    """
    Complete fracture fragment data contract pairing geometry with planned transform.

    This is the primary data structure passed between the fracture-detection
    service, reduction planner, and the surgical plan validator.
    """
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [{
                "geometry": {
                    "fragment_id": "frag_01",
                    "label_value": 2,
                    "parent_structure": "mandible",
                    "centroid_mm": [12.5, -45.0, 8.0],
                    "volume_mm3": 1250.0,
                    "is_reference": False,
                    "confidence": 0.92,
                },
                "planned_transform": {
                    "fragment_id": "frag_01",
                    "rotation_matrix": [[1,0,0],[0,1,0],[0,0,1]],
                    "translation_mm": [4.5, -2.0, 1.5],
                    "confidence": 0.88,
                },
                "hardware_recommendation": "miniplate_2mm",
                "surgical_sequence": 1,
            }]
        },
    )

    geometry: FragmentGeometry
    planned_transform: Optional[FragmentTransformContract] = None
    hardware_recommendation: Optional[HardwareType] = Field(
        None, description="Recommended fixation hardware for this fragment"
    )
    hardware_notes: Optional[str] = Field(
        None, max_length=500, description="Free-text hardware placement guidance"
    )
    surgical_sequence: Optional[int] = Field(
        None, ge=1, description="Recommended reduction order (1 = first)"
    )
    notes: Optional[str] = Field(
        None, max_length=1000, description="Clinical notes about this fragment"
    )

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FractureFragmentContract":
        return cls.model_validate(data)
