"""
Data contract for a training case in the facial-align supervised learning pipeline.

A TrainingCase represents a single training sample: a patient's CT volume
paired with optional IOS scans, fragment metadata, and ground-truth SE(3)
transforms for supervised training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class FragmentGroundTruth:
    """Ground-truth transform for a single bone fragment."""
    fragment_id: str
    se3_matrix: List[List[float]]  # 4x4 homogeneous transform
    mesh_path: Optional[str] = None
    volume_cm3: Optional[float] = None
    region: Optional[str] = None  # e.g. "parasymphyseal", "condylar"


@dataclass
class ToothGroundTruth:
    """Ground-truth for a single tooth position."""
    fdi_number: int
    mesh_path: Optional[str] = None
    centroid_mm: Optional[List[float]] = None


@dataclass
class TrainingCase:
    """
    Represents one training sample for the supervised fracture reduction model.

    Fields are populated from a training manifest JSON and used by dataset
    classes to load CT volumes, IOS point clouds, and ground-truth transforms.

    Constructed via ``TrainingCase(**entry)`` from manifest JSON entries, so
    all fields except ``case_id`` must have defaults.
    """
    # Identifiers
    case_id: str = ""
    patient_id: Optional[str] = None

    # Case classification
    case_type: Optional[str] = None  # "fracture_pair", "normal_reference", etc.

    # CT data
    ct_path: Optional[str] = None  # Path to NIfTI or DICOM dir
    ct_spacing_mm: Optional[List[float]] = None  # (z, y, x) voxel spacing
    ct_shape: Optional[List[int]] = None  # (D, H, W)

    # Pre-op / post-op paths (used by FractureDataset)
    preop_dicom_path: Optional[Path] = None
    preop_segmentation_path: Optional[Path] = None
    postop_dicom_path: Optional[Path] = None

    # Normal anatomy reference (used by NormalReferenceDataset)
    normal_ct_path: Optional[Path] = None

    # Fragment data
    fragments: List[FragmentGroundTruth] = field(default_factory=list)
    num_fragments: int = 0
    fracture_type: Optional[str] = None  # e.g. "bilateral_mandible", "le_fort_ii"

    # IOS data (optional)
    ios_available: bool = False
    ios_mesh_paths: Dict[int, str] = field(default_factory=dict)  # FDI -> mesh path
    ios_upper_arch_path: Optional[Path] = None
    ios_lower_arch_path: Optional[Path] = None
    teeth: List[ToothGroundTruth] = field(default_factory=list)

    # Ground truth transforms (fragment_id -> flat 16-element list or 4x4 matrix)
    ground_truth_transforms: Optional[Dict[str, Any]] = None
    gt_fragment_transforms: Dict[str, List[List[float]]] = field(default_factory=dict)  # frag_id -> 4x4
    gt_tooth_transforms: Dict[int, List[List[float]]] = field(default_factory=dict)  # FDI -> 4x4

    # Ground truth occlusal metrics (used by FractureDataset, DentalDataset)
    ground_truth_occlusal_metrics: Optional[Dict[str, Any]] = None

    # Ground truth landmarks (used by NormalReferenceDataset, LandmarkTrainer)
    ground_truth_landmarks: Optional[Dict[str, List[float]]] = None

    # Clinical ground truth
    gt_overjet_mm: Optional[float] = None
    gt_overbite_mm: Optional[float] = None
    gt_midline_deviation_mm: Optional[float] = None
    gt_molar_class: Optional[str] = None  # "I", "II", "III"

    # Metadata
    is_synthetic: bool = False
    source: Optional[str] = None  # "real", "synthetic", "registered"
    split: Optional[str] = None  # "train", "val", "test"

    # Quality
    quality_score: Optional[float] = None
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        """Coerce path strings to Path objects where needed."""
        for attr in (
            "preop_dicom_path",
            "preop_segmentation_path",
            "postop_dicom_path",
            "normal_ct_path",
            "ios_upper_arch_path",
            "ios_lower_arch_path",
        ):
            val = getattr(self, attr)
            if val is not None and not isinstance(val, Path):
                setattr(self, attr, Path(val))

    def has_ios(self) -> bool:
        """Return True if this case has any IOS scan data."""
        if self.ios_available:
            return True
        if self.ios_upper_arch_path is not None or self.ios_lower_arch_path is not None:
            return True
        if self.ios_mesh_paths:
            return True
        return False

    @classmethod
    def from_manifest_entry(cls, entry: Dict[str, Any]) -> "TrainingCase":
        """Create a TrainingCase from a training_manifest.json entry."""
        fragments = []
        for frag in entry.get("fragments", []):
            fragments.append(FragmentGroundTruth(
                fragment_id=frag["fragment_id"],
                se3_matrix=frag.get("se3_matrix", []),
                mesh_path=frag.get("mesh_path"),
                volume_cm3=frag.get("volume_cm3"),
                region=frag.get("region"),
            ))

        teeth = []
        for tooth in entry.get("teeth", []):
            teeth.append(ToothGroundTruth(
                fdi_number=tooth["fdi_number"],
                mesh_path=tooth.get("mesh_path"),
                centroid_mm=tooth.get("centroid_mm"),
            ))

        return cls(
            case_id=entry["case_id"],
            patient_id=entry.get("patient_id"),
            case_type=entry.get("case_type"),
            ct_path=entry.get("ct_path"),
            ct_spacing_mm=entry.get("ct_spacing_mm"),
            ct_shape=entry.get("ct_shape"),
            preop_dicom_path=entry.get("preop_dicom_path"),
            preop_segmentation_path=entry.get("preop_segmentation_path"),
            postop_dicom_path=entry.get("postop_dicom_path"),
            normal_ct_path=entry.get("normal_ct_path"),
            fragments=fragments,
            num_fragments=len(fragments),
            fracture_type=entry.get("fracture_type"),
            ios_available=entry.get("ios_available", False),
            ios_mesh_paths=entry.get("ios_mesh_paths", {}),
            ios_upper_arch_path=entry.get("ios_upper_arch_path"),
            ios_lower_arch_path=entry.get("ios_lower_arch_path"),
            teeth=teeth,
            ground_truth_transforms=entry.get("ground_truth_transforms"),
            gt_fragment_transforms=entry.get("gt_fragment_transforms", {}),
            gt_tooth_transforms=entry.get("gt_tooth_transforms", {}),
            ground_truth_occlusal_metrics=entry.get("ground_truth_occlusal_metrics"),
            ground_truth_landmarks=entry.get("ground_truth_landmarks"),
            gt_overjet_mm=entry.get("gt_overjet_mm"),
            gt_overbite_mm=entry.get("gt_overbite_mm"),
            gt_midline_deviation_mm=entry.get("gt_midline_deviation_mm"),
            gt_molar_class=entry.get("gt_molar_class"),
            is_synthetic=entry.get("is_synthetic", False),
            source=entry.get("source"),
            split=entry.get("split"),
            quality_score=entry.get("quality_score"),
            notes=entry.get("notes"),
        )
