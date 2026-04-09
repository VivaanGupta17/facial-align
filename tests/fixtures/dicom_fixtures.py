"""
DICOM test fixtures for Facial Align unit and integration tests.

Provides realistic mock DICOM datasets representing craniofacial CT scans,
including complete metadata, synthetic pixel data with anatomically plausible
Hounsfield Unit (HU) distributions, and malformed examples for error-path testing.

These fixtures do NOT require a real DICOM file on disk — everything is
generated in-memory using pydicom's Dataset API.

Usage:
    from tests.fixtures.dicom_fixtures import make_dicom_dataset, make_dicom_series

    ds = make_dicom_dataset()
    series = make_dicom_series(n_slices=50)
"""

from __future__ import annotations

import datetime
import struct
import uuid
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.sequence import Sequence
    from pydicom.uid import UID
    _PYDICOM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYDICOM_AVAILABLE = False
    # Provide stubs so the fixture module imports without error
    Dataset = object  # type: ignore
    FileDataset = object  # type: ignore


# ---------------------------------------------------------------------------
# Canonical CMF CT metadata constants
# ---------------------------------------------------------------------------

TYPICAL_CMF_CT_METADATA: dict = {
    # Patient demographics (de-identified stand-ins)
    "PatientName": "ANON^PATIENT",
    "PatientID": "ANON-000001",
    "PatientBirthDate": "19880101",
    "PatientSex": "M",
    "PatientAge": "036Y",
    "PatientWeight": 78.0,

    # Study / series identifiers
    "StudyDescription": "Maxillofacial CT Pre-op",
    "SeriesDescription": "Axial Head 0.625mm",
    "Modality": "CT",
    "BodyPartExamined": "HEAD",
    "StudyDate": "20240315",
    "SeriesDate": "20240315",
    "AcquisitionDate": "20240315",
    "StudyTime": "090000.000",
    "SeriesTime": "090315.000",

    # Scanner
    "Manufacturer": "Siemens Healthineers",
    "ManufacturerModelName": "SOMATOM Definition Flash",
    "InstitutionName": "University Medical Center",
    "SoftwareVersions": "syngo.CT 2021A",
    "StationName": "CT01",

    # Acquisition parameters
    "KVP": 120.0,
    "Exposure": 200,
    "ExposureTime": 500,
    "XRayTubeCurrent": 250,
    "CTDIvol": 12.5,
    "FocalSpots": 0.7,
    "ConvolutionKernel": "B30f",
    "FilterType": "BODY",
    "RotationDirection": "CW",
    "GantryDetectorTilt": 0.0,
    "DataCollectionDiameter": 500.0,
    "ReconstructionDiameter": 220.0,

    # Image geometry
    "SliceThickness": 0.625,
    "PixelSpacing": [0.488, 0.488],
    "Rows": 512,
    "Columns": 512,
    "PixelRepresentation": 1,      # signed
    "BitsAllocated": 16,
    "BitsStored": 16,
    "HighBit": 15,
    "RescaleSlope": 1.0,
    "RescaleIntercept": -1024.0,
    "WindowCenter": 400.0,
    "WindowWidth": 2000.0,
    "ImageOrientationPatient": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "ImagePositionPatient": [0.0, 0.0, 0.0],
    "InstanceNumber": 1,
    "SliceLocation": 0.0,
    "PhotometricInterpretation": "MONOCHROME2",
    "SamplesPerPixel": 1,
}

# PHI-bearing DICOM tags that must be removed / replaced during de-identification
DICOM_PHI_TAGS: List[str] = [
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientAge",
    "PatientSex",
    "PatientWeight",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "OtherPatientIDs",
    "OtherPatientNames",
    "EthnicGroup",
    "PatientComments",
    "InstitutionName",
    "InstitutionAddress",
    "InstitutionalDepartmentName",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "OperatorsName",
    "StudyDate",
    "SeriesDate",
    "AcquisitionDate",
    "ContentDate",
    "StudyTime",
    "SeriesTime",
    "AcquisitionTime",
    "ContentTime",
    "StudyID",
    "AccessionNumber",
    "StudyDescription",
    "SeriesDescription",
    "RequestedProcedureDescription",
    "StationName",
    "DeviceSerialNumber",
    "SoftwareVersions",
]


# ---------------------------------------------------------------------------
# UID helpers
# ---------------------------------------------------------------------------

def _make_uid(prefix: str = "1.2.840.10008.5.1.4.1.1.2") -> str:
    """Generate a syntactically valid DICOM UID from a UUID4."""
    hex_str = uuid.uuid4().hex
    # Convert 128-bit UUID to integer, then prepend a small root prefix
    uid_int = int(hex_str, 16) % (10 ** 39)
    return f"2.25.{uid_int}"


# ---------------------------------------------------------------------------
# Core factory
# ---------------------------------------------------------------------------

def make_dicom_dataset(
    *,
    sop_instance_uid: Optional[str] = None,
    series_instance_uid: Optional[str] = None,
    study_instance_uid: Optional[str] = None,
    slice_index: int = 0,
    z_position_mm: float = 0.0,
    rows: int = 512,
    cols: int = 512,
    pixel_data: Optional[np.ndarray] = None,
    overrides: Optional[dict] = None,
) -> "Dataset":
    """
    Generate a single pydicom Dataset with all CMF-relevant tags populated.

    Parameters
    ----------
    sop_instance_uid : str, optional
        Specific SOP Instance UID; auto-generated if None.
    series_instance_uid : str, optional
        Series UID shared across a series; auto-generated if None.
    study_instance_uid : str, optional
        Study UID; auto-generated if None.
    slice_index : int
        Zero-based slice index within the series (sets InstanceNumber).
    z_position_mm : float
        Z coordinate of ImagePositionPatient in mm.
    rows, cols : int
        Image dimensions. Default 512×512 (typical clinical CMF CT).
    pixel_data : np.ndarray, optional
        Pre-built 2D int16 pixel array. Generated if None.
    overrides : dict, optional
        Tag name → value pairs that override defaults.

    Returns
    -------
    pydicom.Dataset
        Fully populated dataset ready for use in tests.
    """
    if not _PYDICOM_AVAILABLE:
        raise ImportError("pydicom is required for DICOM fixtures")

    ds = Dataset()
    ds.file_meta = Dataset()

    # --- UIDs ------------------------------------------------------------------
    study_uid = study_instance_uid or _make_uid()
    series_uid = series_instance_uid or _make_uid()
    sop_uid = sop_instance_uid or _make_uid()

    ds.StudyInstanceUID = UID(study_uid)
    ds.SeriesInstanceUID = UID(series_uid)
    ds.SOPInstanceUID = UID(sop_uid)
    ds.SOPClassUID = UID("1.2.840.10008.5.1.4.1.1.2")  # CT Image Storage

    # file_meta
    ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.file_meta.TransferSyntaxUID = UID("1.2.840.10008.1.2.1")  # Explicit VR LE

    # --- Patient ---------------------------------------------------------------
    ds.PatientName = TYPICAL_CMF_CT_METADATA["PatientName"]
    ds.PatientID = TYPICAL_CMF_CT_METADATA["PatientID"]
    ds.PatientBirthDate = TYPICAL_CMF_CT_METADATA["PatientBirthDate"]
    ds.PatientSex = TYPICAL_CMF_CT_METADATA["PatientSex"]
    ds.PatientAge = TYPICAL_CMF_CT_METADATA["PatientAge"]
    ds.PatientWeight = TYPICAL_CMF_CT_METADATA["PatientWeight"]

    # --- Study / Series --------------------------------------------------------
    ds.StudyDescription = TYPICAL_CMF_CT_METADATA["StudyDescription"]
    ds.SeriesDescription = TYPICAL_CMF_CT_METADATA["SeriesDescription"]
    ds.Modality = TYPICAL_CMF_CT_METADATA["Modality"]
    ds.BodyPartExamined = TYPICAL_CMF_CT_METADATA["BodyPartExamined"]
    ds.StudyDate = TYPICAL_CMF_CT_METADATA["StudyDate"]
    ds.SeriesDate = TYPICAL_CMF_CT_METADATA["SeriesDate"]
    ds.AcquisitionDate = TYPICAL_CMF_CT_METADATA["AcquisitionDate"]
    ds.StudyTime = TYPICAL_CMF_CT_METADATA["StudyTime"]
    ds.SeriesTime = TYPICAL_CMF_CT_METADATA["SeriesTime"]

    # --- Scanner ---------------------------------------------------------------
    ds.Manufacturer = TYPICAL_CMF_CT_METADATA["Manufacturer"]
    ds.ManufacturerModelName = TYPICAL_CMF_CT_METADATA["ManufacturerModelName"]
    ds.InstitutionName = TYPICAL_CMF_CT_METADATA["InstitutionName"]
    ds.SoftwareVersions = TYPICAL_CMF_CT_METADATA["SoftwareVersions"]
    ds.StationName = TYPICAL_CMF_CT_METADATA["StationName"]
    ds.ConvolutionKernel = TYPICAL_CMF_CT_METADATA["ConvolutionKernel"]
    ds.FilterType = TYPICAL_CMF_CT_METADATA["FilterType"]
    ds.GantryDetectorTilt = TYPICAL_CMF_CT_METADATA["GantryDetectorTilt"]
    ds.DataCollectionDiameter = TYPICAL_CMF_CT_METADATA["DataCollectionDiameter"]
    ds.ReconstructionDiameter = TYPICAL_CMF_CT_METADATA["ReconstructionDiameter"]

    # --- Acquisition parameters ------------------------------------------------
    ds.KVP = TYPICAL_CMF_CT_METADATA["KVP"]
    ds.Exposure = TYPICAL_CMF_CT_METADATA["Exposure"]
    ds.ExposureTime = TYPICAL_CMF_CT_METADATA["ExposureTime"]
    ds.XRayTubeCurrent = TYPICAL_CMF_CT_METADATA["XRayTubeCurrent"]

    # --- Image geometry --------------------------------------------------------
    ds.SliceThickness = TYPICAL_CMF_CT_METADATA["SliceThickness"]
    ds.PixelSpacing = list(TYPICAL_CMF_CT_METADATA["PixelSpacing"])
    ds.Rows = rows
    ds.Columns = cols
    ds.PixelRepresentation = 1           # signed int16
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = -1024.0
    ds.WindowCenter = [400.0, 40.0]     # bone, soft-tissue presets
    ds.WindowWidth = [2000.0, 350.0]
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    ds.ImageOrientationPatient = list(TYPICAL_CMF_CT_METADATA["ImageOrientationPatient"])
    ds.ImagePositionPatient = [0.0, 0.0, z_position_mm]
    ds.InstanceNumber = slice_index + 1
    ds.SliceLocation = z_position_mm
    ds.NumberOfFrames = 1

    # --- Pixel data ------------------------------------------------------------
    if pixel_data is None:
        pixel_data = _make_ct_slice_pixels(rows, cols, z_position_mm)

    raw_pixels = pixel_data.astype(np.int16)
    # Encode as int16 raw bytes
    ds.PixelData = raw_pixels.tobytes()
    ds["PixelData"].VR = "OW"
    ds.is_implicit_VR = False
    ds.is_little_endian = True

    # --- Optional overrides ---------------------------------------------------
    if overrides:
        for tag_name, value in overrides.items():
            setattr(ds, tag_name, value)

    return ds


def _make_ct_slice_pixels(rows: int, cols: int, z_mm: float) -> np.ndarray:
    """
    Generate anatomically plausible synthetic CT slice pixel data.

    Uses stored HU values (before RescaleIntercept adjustment):
    stored_value = HU - RescaleIntercept = HU + 1024

    Approximate HU ranges represented:
      Air outside skull:  −1000 HU
      Soft tissue:        40–80 HU
      Cancellous bone:    200–400 HU
      Cortical bone:      700–1200 HU
    """
    rng = np.random.default_rng(seed=int(abs(z_mm)) % (2 ** 32))

    # Background air
    img = np.full((rows, cols), -1000, dtype=np.float32)

    cx, cy = cols // 2, rows // 2
    r_outer = min(rows, cols) // 2 - 20   # skull outer radius
    r_inner = r_outer - 6                  # inner cortex boundary

    yy, xx = np.ogrid[:rows, :cols]

    # Ellipsoidal head outline
    r_x, r_y = r_outer * 1.1, r_outer
    in_head = ((xx - cx) ** 2 / r_x ** 2 + (yy - cy) ** 2 / r_y ** 2) <= 1.0

    # Cortical skull ring
    in_cortex = (
        ((xx - cx) ** 2 / r_x ** 2 + (yy - cy) ** 2 / r_y ** 2) <= 1.0
    ) & (
        ((xx - cx) ** 2 / (r_x - 6) ** 2 + (yy - cy) ** 2 / (r_y - 6) ** 2) >= 1.0
    )

    img[in_head] = rng.normal(50, 10, int(in_head.sum())).astype(np.float32)     # soft tissue
    img[in_cortex] = rng.normal(900, 80, int(in_cortex.sum())).astype(np.float32) # cortical bone

    # Add some teeth (very high HU) near z=0
    if abs(z_mm) < 30:
        teeth_mask = (
            np.abs(yy - (cy + 20)) < 8
        ) & (np.abs(xx - cx) < 30) & in_head
        img[teeth_mask] = rng.normal(2500, 200, int(teeth_mask.sum())).astype(np.float32)

    img = np.clip(img, -1024, 3071)
    # Encode as stored values: stored = HU - RescaleIntercept = HU + 1024
    return (img + 1024).astype(np.int16)


# ---------------------------------------------------------------------------
# Series factory
# ---------------------------------------------------------------------------

def make_dicom_series(
    n_slices: int = 200,
    slice_thickness_mm: float = 0.625,
    rows: int = 512,
    cols: int = 512,
    study_uid: Optional[str] = None,
    series_uid: Optional[str] = None,
) -> List["Dataset"]:
    """
    Generate a realistic series of DICOM datasets.

    Each slice has:
    - Consistent StudyInstanceUID and SeriesInstanceUID
    - Incrementing ImagePositionPatient (z-axis, superior-to-inferior)
    - Sequential InstanceNumber
    - Anatomically layered pixel data (based on z-position)

    Parameters
    ----------
    n_slices : int
        Number of axial slices. 200 covers a typical CMF FOV at 0.625mm pitch.
    slice_thickness_mm : float
        Physical distance between slice planes in mm.
    rows, cols : int
        Pixel matrix dimensions per slice.
    study_uid : str, optional
        Shared StudyInstanceUID; auto-generated if None.
    series_uid : str, optional
        Shared SeriesInstanceUID; auto-generated if None.

    Returns
    -------
    list[Dataset]
        Ordered list of DICOM datasets from inferior (z_min) to superior (z_max).
    """
    if not _PYDICOM_AVAILABLE:
        raise ImportError("pydicom is required for DICOM fixtures")

    study_uid = study_uid or _make_uid()
    series_uid = series_uid or _make_uid()

    # Typical CMF scan starts ~−60mm below skull vertex, goes inferiorly
    z_start = -60.0
    datasets = []
    for i in range(n_slices):
        z = z_start + i * slice_thickness_mm
        ds = make_dicom_dataset(
            study_instance_uid=study_uid,
            series_instance_uid=series_uid,
            slice_index=i,
            z_position_mm=z,
            rows=rows,
            cols=cols,
        )
        datasets.append(ds)

    return datasets


# ---------------------------------------------------------------------------
# Malformed DICOM factories
# ---------------------------------------------------------------------------

def make_malformed_dicom(variant: str = "missing_slice_thickness") -> "Dataset":
    """
    Create malformed DICOM datasets for error-path testing.

    Parameters
    ----------
    variant : str
        Which malformation to introduce:
        - ``"missing_slice_thickness"`` — SliceThickness tag omitted
        - ``"wrong_modality"`` — Modality set to "MR" instead of "CT"
        - ``"corrupted_pixel_data"`` — PixelData contains random bytes (wrong length)
        - ``"missing_image_position"`` — ImagePositionPatient absent
        - ``"non_ct_sop_class"`` — SOPClassUID set to MR Image Storage
        - ``"zero_rows_cols"`` — Rows=0, Columns=0 (degenerate geometry)
        - ``"negative_rescale_slope"`` — RescaleSlope=-1 (physically impossible)
        - ``"missing_patient_id"`` — PatientID tag absent (PHI integrity failure)
        - ``"mixed_voxel_spacing"`` — Inconsistent PixelSpacing and SliceThickness

    Returns
    -------
    Dataset
        A Dataset with the requested malformation applied.
    """
    ds = make_dicom_dataset()

    if variant == "missing_slice_thickness":
        del ds.SliceThickness

    elif variant == "wrong_modality":
        ds.Modality = "MR"
        ds.SOPClassUID = UID("1.2.840.10008.5.1.4.1.1.4")  # MR Image Storage

    elif variant == "corrupted_pixel_data":
        ds.PixelData = bytes(rng := np.random.default_rng(99)) or b"\xff\xfe" * 100
        # Deliberately wrong byte count
        ds.PixelData = b"\xDE\xAD\xBE\xEF" * 16

    elif variant == "missing_image_position":
        del ds.ImagePositionPatient

    elif variant == "non_ct_sop_class":
        ds.SOPClassUID = UID("1.2.840.10008.5.1.4.1.1.4")
        ds.Modality = "MR"

    elif variant == "zero_rows_cols":
        ds.Rows = 0
        ds.Columns = 0
        ds.PixelData = b""

    elif variant == "negative_rescale_slope":
        ds.RescaleSlope = -1.0

    elif variant == "missing_patient_id":
        del ds.PatientID

    elif variant == "mixed_voxel_spacing":
        ds.PixelSpacing = [0.488, 0.976]   # Non-square pixels
        ds.SliceThickness = 5.0             # Much thicker than claimed series

    else:
        raise ValueError(
            f"Unknown malformed_dicom variant: {variant!r}. "
            "Choose from: missing_slice_thickness, wrong_modality, "
            "corrupted_pixel_data, missing_image_position, non_ct_sop_class, "
            "zero_rows_cols, negative_rescale_slope, missing_patient_id, "
            "mixed_voxel_spacing"
        )

    return ds


# ---------------------------------------------------------------------------
# Convenience collection of all malformed variants
# ---------------------------------------------------------------------------

MALFORMED_VARIANTS: List[str] = [
    "missing_slice_thickness",
    "wrong_modality",
    "corrupted_pixel_data",
    "missing_image_position",
    "non_ct_sop_class",
    "zero_rows_cols",
    "negative_rescale_slope",
    "missing_patient_id",
    "mixed_voxel_spacing",
]


def make_all_malformed_dicoms() -> dict[str, "Dataset"]:
    """Return a dict mapping every malformed variant name to its Dataset."""
    return {v: make_malformed_dicom(v) for v in MALFORMED_VARIANTS}
