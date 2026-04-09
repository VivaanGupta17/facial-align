"""
Facial Align Data Contracts
============================

Canonical Pydantic v2 data schemas used across all services, pipelines,
and API boundaries.  These models define the contract between modules and
serve as the single source of truth for data structures flowing through
the craniofacial surgical planning platform.

Each contract module contains:
- Status / classification enums
- Type aliases (MillimeterValue, DegreeValue, etc.)
- One or more Pydantic models with field validators and docstrings
- Serialisation helpers: ``.to_dict()``, ``.to_json()``, ``.from_dict()``
- Clinical utility methods (e.g. ``CTStudyContract.is_suitable_for_planning()``)

Quick-import examples
---------------------
::

    # CT study — ingestion output
    from data_contracts import CTStudyContract, CTSeriesContract, CTQualityGrade

    # Segmentation — pipeline output
    from data_contracts import SegmentationOutputContract, StructureMesh

    # Fracture fragments
    from data_contracts import FractureFragmentContract, FragmentGeometry

    # Reduction plan — primary planning artefact
    from data_contracts import ReductionPlanContract, OcclusalMetricsContract

    # Occlusion plan — cephalometric + dental analysis
    from data_contracts import OcclusionPlanContract, DentalConstraintSet

    # Intraoral scan — IOS digital impression
    from data_contracts import IntraoralScanContract, ScanQualityMetrics

    # Case review — surgeon sign-off
    from data_contracts import CaseReview, ModificationRequest

    # Splint design — manufacturing specification
    from data_contracts import SplintDesignRequest, SplintDesignSpec

    # Surgeon edits — audit trail
    from data_contracts import SurgeonEditHistory, TransformEdit
"""

# ── CT Study ──────────────────────────────────────────────────────────────────
from data_contracts.ct_study import (
    CTModality,
    CTQualityGrade,
    CTSeriesContract,
    CTStudyContract,
)

# ── Segmentation Output ───────────────────────────────────────────────────────
from data_contracts.segmentation_output import (
    STRUCTURE_HIERARCHY,
    SegmentationOutputContract,
    SegmentationStatus,
    StructureClass,
    StructureMesh,
    StructureStats,
)

# ── Fracture Fragment ─────────────────────────────────────────────────────────
from data_contracts.fracture_fragment import (
    AOCMFRegion,
    FragmentContactSurface,
    FragmentGeometry,
    FragmentTransformContract,
    FractureFragmentContract,
    HardwareType,
)

# ── Reduction Plan ────────────────────────────────────────────────────────────
from data_contracts.reduction_plan import (
    HardwareItem,
    OcclusalMetricsContract,
    PlanOrigin,
    PlanStatus,
    ReductionPlanContract,
    ValidationContract,
)

# ── Occlusion Plan ────────────────────────────────────────────────────────────
from data_contracts.occlusion_plan import (
    AngleMolarClass,
    CephalometricMeasurement,
    DentalConstraintSet,
    OcclusionGrade,
    OcclusionPlanContract,
    SkeletalPattern,
    ToothContactContract,
)

# ── Intraoral Scan ────────────────────────────────────────────────────────────
from data_contracts.intraoral_scan import (
    DentalArch,
    IntraoralScanContract,
    OcclusalSurfaceStatus,
    ScanFileFormat,
    ScannerManufacturer,
    ScanPurpose,
    ScanQualityMetrics,
)

# ── Case Review ───────────────────────────────────────────────────────────────
from data_contracts.case_review import (
    CaseReview,
    ClinicalMeasurement,
    ModificationPriority,
    ModificationRequest,
    ReviewerFeedback,
    ReviewerRole,
    ReviewOutcome,
)

# ── Splint Design ─────────────────────────────────────────────────────────────
from data_contracts.splint_design import (
    ManufacturingStatus,
    OutputFormat,
    RetentionFeatures,
    RetentionMechanism as SplintRetentionMechanism,
    SplintDesignRequest,
    SplintDesignSpec,
    SplintMaterial,
    SplintThicknessParameters,
    SplintType,
)

# ── Surgeon Edit ──────────────────────────────────────────────────────────────
from data_contracts.surgeon_edit import (
    EditSessionSummary,
    EditTool,
    EditType,
    SurgeonEditHistory,
    TransformEdit,
)

# ---------------------------------------------------------------------------
# Public API declaration
# ---------------------------------------------------------------------------

__all__ = [
    # ── CT Study
    "CTModality",
    "CTQualityGrade",
    "CTSeriesContract",
    "CTStudyContract",

    # ── Segmentation Output
    "STRUCTURE_HIERARCHY",
    "SegmentationOutputContract",
    "SegmentationStatus",
    "StructureClass",
    "StructureMesh",
    "StructureStats",

    # ── Fracture Fragment
    "AOCMFRegion",
    "FragmentContactSurface",
    "FragmentGeometry",
    "FragmentTransformContract",
    "FractureFragmentContract",
    "HardwareType",

    # ── Reduction Plan
    "HardwareItem",
    "OcclusalMetricsContract",
    "PlanOrigin",
    "PlanStatus",
    "ReductionPlanContract",
    "ValidationContract",

    # ── Occlusion Plan
    "AngleMolarClass",
    "CephalometricMeasurement",
    "DentalConstraintSet",
    "OcclusionGrade",
    "OcclusionPlanContract",
    "SkeletalPattern",
    "ToothContactContract",

    # ── Intraoral Scan
    "DentalArch",
    "IntraoralScanContract",
    "OcclusalSurfaceStatus",
    "ScanFileFormat",
    "ScannerManufacturer",
    "ScanPurpose",
    "ScanQualityMetrics",

    # ── Case Review
    "CaseReview",
    "ClinicalMeasurement",
    "ModificationPriority",
    "ModificationRequest",
    "ReviewerFeedback",
    "ReviewerRole",
    "ReviewOutcome",

    # ── Splint Design
    "ManufacturingStatus",
    "OutputFormat",
    "RetentionFeatures",
    "SplintDesignRequest",
    "SplintDesignSpec",
    "SplintMaterial",
    "SplintRetentionMechanism",
    "SplintThicknessParameters",
    "SplintType",

    # ── Surgeon Edit
    "EditSessionSummary",
    "EditTool",
    "EditType",
    "SurgeonEditHistory",
    "TransformEdit",
]
