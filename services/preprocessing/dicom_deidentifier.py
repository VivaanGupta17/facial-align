"""
DICOM De-identification Service
=================================

Implements the DICOM PS3.15 Annex E Basic Application Level Confidentiality Profile
for de-identifying patient health information (PHI) from DICOM datasets.

Regulatory Background
----------------------
DICOM de-identification is required when:
  - Sharing data for research (IRB-approved studies)
  - Transferring data between institutions
  - Using data for ML model training
  - Storing data in cloud environments not covered by BAA

The DICOM standard (PS3.15 Annex E) defines a tiered confidentiality profile:
  - Basic Profile: Removes/replaces all direct patient identifiers
  - Retain Longitudinal Temporal Info Option: Preserves dates (shifted by random offset)
  - Clean Pixel Data Option: Removes burned-in annotations in pixel data
  - Retain Patient Characteristics Option: Keeps age/sex/weight for clinical utility

This implementation covers the Basic Application Level Confidentiality Profile,
which is the minimum requirement for research data sharing under HIPAA and GDPR.

Pseudonymisation Strategy
--------------------------
Rather than removing all identifying information (which makes longitudinal
studies impossible), this implementation uses consistent hash-based pseudonymisation:

  pseudonym = HMAC-SHA256(salt + PatientID) → truncated to 8 hex characters

The same patient always receives the same pseudonym (given the same salt),
enabling linking across studies. The salt is site-specific and must be kept
secure — it is NOT stored in the DICOM files.

IMPORTANT: This module is NOT a substitute for legal compliance review.
           Consult your institution's privacy officer and IRB.

Author: Facial Align Engineering
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DICOM PS3.15 Annex E — PHI Tag Inventory
# ---------------------------------------------------------------------------

# Tags that MUST be removed (D = Delete) per Basic Profile
PHI_TAGS_DELETE = {
    # Patient identification
    (0x0010, 0x0010): "PatientName",
    (0x0010, 0x0020): "PatientID",
    (0x0010, 0x0030): "PatientBirthDate",
    (0x0010, 0x0032): "PatientBirthTime",
    (0x0010, 0x0040): "PatientSex",         # Retained in retain-characteristics option
    (0x0010, 0x1000): "OtherPatientIDs",
    (0x0010, 0x1001): "OtherPatientNames",
    (0x0010, 0x1040): "PatientMotherBirthName",
    (0x0010, 0x0050): "PatientInsurancePlanCodeSequence",
    (0x0010, 0x1005): "PatientBirthName",
    (0x0010, 0x0010): "PatientName",
    (0x0010, 0x1010): "PatientAge",
    (0x0010, 0x1020): "PatientSize",
    (0x0010, 0x1030): "PatientWeight",
    (0x0010, 0x1090): "MedicalRecordLocator",
    (0x0010, 0x2000): "MedicalAlerts",
    (0x0010, 0x2110): "Allergies",
    (0x0010, 0x2150): "CountryOfResidence",
    (0x0010, 0x2154): "PatientTelephoneNumbers",
    (0x0010, 0x2160): "EthnicGroup",
    (0x0010, 0x2180): "Occupation",
    (0x0010, 0x21B0): "AdditionalPatientHistory",
    (0x0010, 0x21C0): "PregnancyStatus",
    (0x0010, 0x21D0): "LastMenstrualDate",
    (0x0010, 0x21F0): "PatientReligiousPreference",
    (0x0010, 0x4000): "PatientComments",

    # Institution and staff
    (0x0008, 0x0080): "InstitutionName",
    (0x0008, 0x0081): "InstitutionAddress",
    (0x0008, 0x0082): "InstitutionCodeSequence",
    (0x0008, 0x0090): "ReferringPhysicianName",
    (0x0008, 0x0092): "ReferringPhysicianAddress",
    (0x0008, 0x0094): "ReferringPhysicianTelephoneNumbers",
    (0x0008, 0x0096): "ReferringPhysicianIdentificationSequence",
    (0x0008, 0x009C): "ConsultingPhysicianName",
    (0x0008, 0x1040): "InstitutionalDepartmentName",
    (0x0008, 0x1048): "PhysiciansOfRecord",
    (0x0008, 0x1049): "PhysiciansOfRecordIdentificationSequence",
    (0x0008, 0x1050): "PerformingPhysicianName",
    (0x0008, 0x1052): "PerformingPhysicianIdentificationSequence",
    (0x0008, 0x1060): "NameOfPhysiciansReadingStudy",
    (0x0008, 0x1062): "PhysiciansReadingStudyIdentificationSequence",
    (0x0008, 0x1070): "OperatorsName",
    (0x0008, 0x1072): "OperatorsIdentificationSequence",
    (0x0008, 0x3010): "IrradiationEventUID",

    # Dates and times (shifted or removed in Basic Profile)
    (0x0008, 0x0020): "StudyDate",
    (0x0008, 0x0021): "SeriesDate",
    (0x0008, 0x0022): "AcquisitionDate",
    (0x0008, 0x0023): "ContentDate",
    (0x0008, 0x0025): "CurveDate",
    (0x0008, 0x002A): "AcquisitionDatetime",
    (0x0008, 0x0030): "StudyTime",
    (0x0008, 0x0031): "SeriesTime",
    (0x0008, 0x0032): "AcquisitionTime",
    (0x0008, 0x0033): "ContentTime",

    # UIDs (regenerated to maintain internal consistency)
    (0x0008, 0x0018): "SOPInstanceUID",
    (0x0020, 0x000D): "StudyInstanceUID",
    (0x0020, 0x000E): "SeriesInstanceUID",
    (0x0020, 0x0052): "FrameOfReferenceUID",
    (0x0020, 0x0200): "SynchronizationFrameOfReferenceUID",
    (0x0040, 0xA124): "UID",

    # Station and device
    (0x0008, 0x1010): "StationName",
    (0x0018, 0x1000): "DeviceSerialNumber",
    (0x0018, 0x1030): "ProtocolName",

    # Request and accession
    (0x0008, 0x0050): "AccessionNumber",
    (0x0008, 0x1110): "ReferencedStudySequence",
    (0x0008, 0x1115): "ReferencedSeriesSequence",
    (0x0008, 0x1120): "ReferencedPatientSequence",
    (0x0032, 0x1032): "RequestingPhysician",
    (0x0032, 0x1033): "RequestingService",
    (0x0032, 0x1060): "RequestedProcedureDescription",
    (0x0040, 0x0275): "RequestAttributesSequence",
    (0x0040, 0xA07C): "CustodialOrganizationSequence",
}

# Tags to PSEUDONYMISE (replace with consistent derived value)
PHI_TAGS_PSEUDONYMISE = {
    (0x0010, 0x0010): "PatientName",    # → Anonymised_<8hex>
    (0x0010, 0x0020): "PatientID",      # → ANON_<8hex>
    (0x0008, 0x0050): "AccessionNumber", # → ACC_<8hex>
}

# Tags to BLANK (replace with empty string)
PHI_TAGS_BLANK = {
    (0x0008, 0x0080): "InstitutionName",
    (0x0008, 0x0090): "ReferringPhysicianName",
    (0x0008, 0x1010): "StationName",
    (0x0008, 0x1040): "InstitutionalDepartmentName",
    (0x0008, 0x1050): "PerformingPhysicianName",
    (0x0008, 0x1070): "OperatorsName",
    (0x0032, 0x1032): "RequestingPhysician",
}

# Tags related to UIDs — must regenerate, maintaining cross-file consistency
UID_TAGS = {
    (0x0008, 0x0018): "SOPInstanceUID",
    (0x0020, 0x000D): "StudyInstanceUID",
    (0x0020, 0x000E): "SeriesInstanceUID",
    (0x0020, 0x0052): "FrameOfReferenceUID",
}

# Tags to RETAIN (clinical data needed for surgical planning)
PHI_TAGS_RETAIN = {
    (0x0010, 0x0040): "PatientSex",       # Needed for normative cephalometric values
    (0x0010, 0x1010): "PatientAge",        # Needed for growth assessment
    (0x0018, 0x0050): "SliceThickness",    # Required for 3D reconstruction
    (0x0018, 0x0088): "SpacingBetweenSlices",
    (0x0028, 0x0030): "PixelSpacing",
    (0x0028, 0x1050): "WindowCenter",
    (0x0028, 0x1051): "WindowWidth",
    (0x0028, 0x1052): "RescaleIntercept",
    (0x0028, 0x1053): "RescaleSlope",
    (0x0008, 0x0060): "Modality",          # CT/CBCT
    (0x0008, 0x0070): "Manufacturer",
    (0x0028, 0x0010): "Rows",
    (0x0028, 0x0011): "Columns",
    (0x0028, 0x0002): "SamplesPerPixel",
    (0x0028, 0x0004): "PhotometricInterpretation",
    (0x0028, 0x0100): "BitsAllocated",
    (0x0028, 0x0101): "BitsStored",
    (0x0028, 0x0102): "HighBit",
    (0x0028, 0x0103): "PixelRepresentation",
    (0x7FE0, 0x0010): "PixelData",
}


# ---------------------------------------------------------------------------
# Audit and Report Structures
# ---------------------------------------------------------------------------

@dataclass
class TagModification:
    """Record of a single tag modification during de-identification."""
    tag: str              # e.g. "(0010,0010)"
    tag_name: str         # e.g. "PatientName"
    action: str           # "removed", "replaced", "blanked", "uid_remapped"
    original_value: str   # Truncated/hashed original for audit log
    new_value: str        # New value or "<removed>"


@dataclass
class DeidentificationReport:
    """
    Complete audit record of a DICOM de-identification operation.

    REGULATORY NOTE: This report should be stored separately from the
    de-identified DICOM files. It provides the audit trail required
    by HIPAA and GDPR for data de-identification operations.
    """
    # Study identification
    original_study_uid: str
    anonymized_study_uid: str
    original_patient_id: str        # Hashed for audit log
    anonymized_patient_id: str

    # Operation details
    deidentification_date: str      # ISO 8601
    profile_applied: str = "DICOM PS3.15 Annex E Basic Application Level Confidentiality Profile"
    software_version: str = "facial-align-deidentifier-1.0"

    # File statistics
    files_processed: int = 0
    files_succeeded: int = 0
    files_failed: int = 0

    # Tag modification counts
    tags_modified: int = 0
    tags_removed: int = 0
    tags_blanked: int = 0
    tags_uid_remapped: int = 0

    # Detailed modification log (per-file)
    tag_modifications: list[TagModification] = field(default_factory=list)

    # Timing
    processing_time_ms: float = 0.0

    # Errors encountered
    errors: list[str] = field(default_factory=list)

    # Options applied
    date_shift_days: Optional[int] = None    # None = dates removed; N = dates shifted
    retain_patient_characteristics: bool = False

    def summary(self) -> str:
        """Human-readable de-identification summary."""
        return (
            f"De-identification complete: {self.files_succeeded}/{self.files_processed} files\n"
            f"  Tags modified : {self.tags_modified}\n"
            f"  Tags removed  : {self.tags_removed}\n"
            f"  Tags blanked  : {self.tags_blanked}\n"
            f"  UIDs remapped : {self.tags_uid_remapped}\n"
            f"  Original UID  : {self.original_study_uid}\n"
            f"  Anonymised UID: {self.anonymized_study_uid}\n"
            f"  Profile       : {self.profile_applied}\n"
            f"  Time          : {self.processing_time_ms:.0f} ms"
        )


# ---------------------------------------------------------------------------
# De-identifier Class
# ---------------------------------------------------------------------------

class DICOMDeidentifier:
    """
    DICOM PS3.15 Annex E Basic Application Level Confidentiality Profile.

    De-identifies DICOM files by:
      1. Removing all PHI tags listed in the DICOM standard inventory
      2. Pseudonymising patient identity using HMAC-SHA256 with a site salt
      3. Remapping UIDs consistently (same study → same anonymous UIDs)
      4. Optionally shifting dates by a random fixed offset (longitudinal studies)
      5. Adding de-identification method tags per DICOM standard

    Usage:
        deidentifier = DICOMDeidentifier(site_salt="your-secret-salt")

        report = deidentifier.deidentify_study(
            input_dir="/path/to/original/dicom",
            output_dir="/path/to/deidentified",
            shift_dates=True,
        )

        print(report.summary())

    IMPORTANT:
        - The site_salt must be securely stored and never embedded in DICOM files
        - The same salt must be used across all de-identification sessions for
          consistent pseudonymisation of the same patient
        - The de-identification report MUST be stored in a secure audit log
    """

    # DICOM de-identification attributes added per PS3.15
    DEIDENT_METHOD_CODE_SEQUENCE_TAG = (0x0012, 0x0064)
    DEIDENT_METHOD_TAG = (0x0012, 0x0063)
    PATIENT_IDENTITY_REMOVED_TAG = (0x0012, 0x0062)

    # Prefix for generated UIDs (DICOM root UID for this software)
    UID_ROOT = "2.25"  # UUID-based UID root (ISO standard)

    def __init__(
        self,
        site_salt: Optional[str] = None,
        retain_patient_characteristics: bool = True,
        date_shift_days: Optional[int] = None,
    ):
        """
        Args:
            site_salt: Secret string for pseudonymisation. If None, a random
                       salt is generated (pseudonymisation will NOT be consistent
                       across sessions). For research use, provide a stable salt.
            retain_patient_characteristics: If True, keep PatientSex and PatientAge
                       (safe for most research; removes direct identifiers but keeps
                       statistical characteristics).
            date_shift_days: If provided, shift all dates by this many days instead
                       of removing them. Use a fixed random offset per patient for
                       longitudinal data consistency.
        """
        self.site_salt = site_salt or self._generate_ephemeral_salt()
        self.retain_patient_characteristics = retain_patient_characteristics
        self.date_shift_days = date_shift_days

        # UID mapping cache: original_uid → anonymised_uid
        # Ensures all files in a study share the same remapped UIDs
        self._uid_cache: dict[str, str] = {}

        # Patient pseudonym cache: original_patient_id → pseudonym_id
        self._patient_cache: dict[str, str] = {}

        logger.info(
            f"DICOMDeidentifier initialised: "
            f"retain_characteristics={retain_patient_characteristics}, "
            f"date_shift={date_shift_days}"
        )

    @staticmethod
    def _generate_ephemeral_salt() -> str:
        """Generate a random session salt. Not stable across runs."""
        salt = os.urandom(32).hex()
        logger.warning(
            "No site_salt provided — using ephemeral salt. "
            "Pseudonymisation will NOT be consistent across sessions. "
            "For research, provide a stable site_salt."
        )
        return salt

    def _compute_pseudonym(self, original_value: str, prefix: str = "ANON") -> str:
        """
        Compute a consistent pseudonym using HMAC-SHA256.

        The same (salt, original_value) pair always produces the same pseudonym.
        The pseudonym is NOT reversible without the salt.

        Args:
            original_value: The original PHI string to pseudonymise
            prefix: Prefix for the generated pseudonym

        Returns:
            Pseudonym string: e.g. "ANON_A3F2B891"
        """
        if not original_value:
            return f"{prefix}_EMPTY"

        mac = hmac.new(
            self.site_salt.encode("utf-8"),
            original_value.encode("utf-8"),
            hashlib.sha256,
        )
        hex_digest = mac.hexdigest()[:8].upper()
        return f"{prefix}_{hex_digest}"

    def _safe_hash_for_audit(self, value: str) -> str:
        """Hash a value for storing in the audit log (one-way, for reference)."""
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

    def _remap_uid(self, original_uid: str) -> str:
        """
        Generate or retrieve a consistent replacement UID.

        Uses the UUID-based DICOM UID root (2.25.xxx) where xxx is a
        128-bit integer from UUID4, ensuring global uniqueness without
        coordination with a registration authority.

        Args:
            original_uid: Original DICOM UID string

        Returns:
            New DICOM-compliant UID string, consistent across calls with same input
        """
        if original_uid in self._uid_cache:
            return self._uid_cache[original_uid]

        # Generate deterministic UID from HMAC of original
        mac = hmac.new(
            self.site_salt.encode("utf-8"),
            f"UID:{original_uid}".encode("utf-8"),
            hashlib.sha256,
        )
        # Convert first 16 bytes to 128-bit integer for UUID format
        uid_int = int.from_bytes(mac.digest()[:16], "big")
        new_uid = f"{self.UID_ROOT}.{uid_int}"

        # DICOM UID max length is 64 characters
        if len(new_uid) > 64:
            new_uid = new_uid[:64]

        self._uid_cache[original_uid] = new_uid
        return new_uid

    def _shift_date(self, date_str: str, shift_days: int) -> str:
        """
        Shift a DICOM date string by a fixed number of days.

        DICOM date format: YYYYMMDD

        Args:
            date_str: DICOM date string (YYYYMMDD or YYYYMMDDHHMMSS.FFFFFF)
            shift_days: Number of days to shift (negative = shift backwards)

        Returns:
            Shifted date string in same format
        """
        if not date_str or len(date_str) < 8:
            return ""
        try:
            date_part = date_str[:8]
            rest = date_str[8:]
            dt = datetime.strptime(date_part, "%Y%m%d")
            shifted = dt + timedelta(days=shift_days)
            return shifted.strftime("%Y%m%d") + rest
        except (ValueError, OverflowError):
            return ""

    def deidentify_dataset(
        self,
        dataset,  # pydicom.Dataset
        report: DeidentificationReport,
    ) -> None:
        """
        De-identify a pydicom Dataset in-place.

        Modifies the dataset by removing/replacing PHI tags according to
        the DICOM PS3.15 Basic Application Level Confidentiality Profile.

        Args:
            dataset: pydicom Dataset to modify in-place
            report: DeidentificationReport to update with actions taken
        """
        # --- Handle UIDs first (must maintain cross-file consistency) -----
        for tag_tuple, tag_name in UID_TAGS.items():
            try:
                tag = dataset[tag_tuple]
                original_uid = str(tag.value)
                new_uid = self._remap_uid(original_uid)
                tag.value = new_uid
                report.tag_modifications.append(TagModification(
                    tag=f"({tag_tuple[0]:04X},{tag_tuple[1]:04X})",
                    tag_name=tag_name,
                    action="uid_remapped",
                    original_value=self._safe_hash_for_audit(original_uid),
                    new_value=new_uid,
                ))
                report.tags_uid_remapped += 1
            except KeyError:
                pass  # Tag not present in this file

        # --- Pseudonymise patient name and ID ----------------------------
        for tag_tuple, prefix in [
            ((0x0010, 0x0010), "Anonymised"),
            ((0x0010, 0x0020), "ANON"),
            ((0x0008, 0x0050), "ACC"),
        ]:
            try:
                tag = dataset[tag_tuple]
                original_val = str(tag.value)
                pseudonym = self._compute_pseudonym(original_val, prefix)
                tag.value = pseudonym
                report.tag_modifications.append(TagModification(
                    tag=f"({tag_tuple[0]:04X},{tag_tuple[1]:04X})",
                    tag_name=PHI_TAGS_PSEUDONYMISE.get(tag_tuple, "Unknown"),
                    action="replaced",
                    original_value=self._safe_hash_for_audit(original_val),
                    new_value=pseudonym,
                ))
                report.tags_modified += 1
            except KeyError:
                pass

        # --- Blank institution / staff names -----------------------------
        for tag_tuple, tag_name in PHI_TAGS_BLANK.items():
            try:
                tag = dataset[tag_tuple]
                original_val = str(tag.value)
                tag.value = ""
                report.tag_modifications.append(TagModification(
                    tag=f"({tag_tuple[0]:04X},{tag_tuple[1]:04X})",
                    tag_name=tag_name,
                    action="blanked",
                    original_value=self._safe_hash_for_audit(original_val),
                    new_value="",
                ))
                report.tags_blanked += 1
            except KeyError:
                pass

        # --- Handle dates (shift or remove) ------------------------------
        date_tags = [
            ((0x0008, 0x0020), "StudyDate"),
            ((0x0008, 0x0021), "SeriesDate"),
            ((0x0008, 0x0022), "AcquisitionDate"),
            ((0x0008, 0x0023), "ContentDate"),
            ((0x0010, 0x0030), "PatientBirthDate"),
        ]
        for tag_tuple, tag_name in date_tags:
            try:
                tag = dataset[tag_tuple]
                original_val = str(tag.value)
                if self.date_shift_days is not None:
                    # Shift date for longitudinal study support
                    new_val = self._shift_date(original_val, self.date_shift_days)
                    tag.value = new_val
                    action = "replaced"
                    new_display = new_val
                else:
                    # Remove date entirely
                    del dataset[tag_tuple]
                    action = "removed"
                    new_display = "<removed>"
                report.tag_modifications.append(TagModification(
                    tag=f"({tag_tuple[0]:04X},{tag_tuple[1]:04X})",
                    tag_name=tag_name,
                    action=action,
                    original_value=self._safe_hash_for_audit(original_val),
                    new_value=new_display,
                ))
                report.tags_removed += 1 if action == "removed" else 0
                report.tags_modified += 1 if action == "replaced" else 0
            except KeyError:
                pass

        # --- Remove remaining PHI tags -----------------------------------
        # Exclude: UIDs (already handled), dates (handled above),
        #          pseudonymised tags (already handled), pixel data
        already_handled = (
            set(UID_TAGS.keys()) |
            set(PHI_TAGS_PSEUDONYMISE.keys()) |
            set(PHI_TAGS_BLANK.keys()) |
            {(0x0008, 0x0020), (0x0008, 0x0021), (0x0008, 0x0022),
             (0x0008, 0x0023), (0x0010, 0x0030)}  # Date tags
        )

        # Optionally retain patient characteristics
        retain_tags: set = set()
        if self.retain_patient_characteristics:
            retain_tags = {(0x0010, 0x0040), (0x0010, 0x1010)}  # Sex, Age

        for tag_tuple, tag_name in PHI_TAGS_DELETE.items():
            if tag_tuple in already_handled or tag_tuple in retain_tags:
                continue
            if tag_tuple in PHI_TAGS_RETAIN:
                continue  # Never remove clinical-critical tags
            try:
                original_val = str(dataset[tag_tuple].value)
                del dataset[tag_tuple]
                report.tag_modifications.append(TagModification(
                    tag=f"({tag_tuple[0]:04X},{tag_tuple[1]:04X})",
                    tag_name=tag_name,
                    action="removed",
                    original_value=self._safe_hash_for_audit(original_val),
                    new_value="<removed>",
                ))
                report.tags_removed += 1
            except KeyError:
                pass

        # --- Remove all private tags (group 0009, 0011, etc.) -------------
        try:
            dataset.remove_private_tags()
        except Exception as exc:
            logger.debug(f"Could not remove private tags: {exc}")

        # --- Add de-identification markers per DICOM standard --------------
        try:
            import pydicom
            from pydicom.sequence import Sequence
            from pydicom.dataset import Dataset as DcmDataset

            # (0012,0062) Patient Identity Removed: YES
            dataset.add_new((0x0012, 0x0062), "CS", "YES")

            # (0012,0063) De-identification Method
            dataset.add_new(
                (0x0012, 0x0063),
                "LO",
                "DICOM PS3.15 Annex E Basic Application Level Confidentiality Profile",
            )

            # (0008,0013) Instance Creation Time
            dataset.add_new(
                (0x0008, 0x0013), "TM",
                datetime.utcnow().strftime("%H%M%S.%f")[:10]
            )

        except Exception as exc:
            logger.warning(f"Could not add de-identification markers: {exc}")

    def deidentify_file(
        self,
        input_path: Path,
        output_path: Path,
        report: DeidentificationReport,
    ) -> bool:
        """
        De-identify a single DICOM file.

        Args:
            input_path: Path to original DICOM file
            output_path: Path to write de-identified file
            report: Report object to update with this file's actions

        Returns:
            True on success, False on failure
        """
        try:
            import pydicom

            dataset = pydicom.dcmread(str(input_path), force=True)

            # Capture original UIDs before modification
            orig_study_uid = str(getattr(dataset, "StudyInstanceUID", "unknown"))
            orig_patient_id = str(getattr(dataset, "PatientID", "unknown"))

            # Only set these on the first file in a study
            if not report.original_study_uid:
                report.original_study_uid = orig_study_uid
                report.original_patient_id = self._safe_hash_for_audit(orig_patient_id)

            # Apply de-identification
            self.deidentify_dataset(dataset, report)

            # Capture new UIDs
            anon_study_uid = str(getattr(dataset, "StudyInstanceUID", "unknown"))
            if not report.anonymized_study_uid:
                report.anonymized_study_uid = anon_study_uid
                report.anonymized_patient_id = str(
                    getattr(dataset, "PatientID", "ANON_UNKNOWN")
                )

            # Write output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            dataset.save_as(str(output_path), write_like_original=False)

            report.files_succeeded += 1
            return True

        except Exception as exc:
            logger.error(f"Failed to de-identify {input_path}: {exc}")
            report.errors.append(f"{input_path.name}: {exc}")
            report.files_failed += 1
            return False

    def deidentify_study(
        self,
        input_dir: str,
        output_dir: str,
        recursive: bool = True,
        file_pattern: str = "*.dcm",
    ) -> DeidentificationReport:
        """
        De-identify all DICOM files in a study directory.

        Args:
            input_dir: Directory containing original DICOM files
            output_dir: Directory for de-identified output files
            recursive: Whether to recurse into subdirectories
            file_pattern: Glob pattern for DICOM files (default: *.dcm)

        Returns:
            DeidentificationReport with complete audit trail
        """
        t_start = time.time()

        report = DeidentificationReport(
            original_study_uid="",
            anonymized_study_uid="",
            original_patient_id="",
            anonymized_patient_id="",
            deidentification_date=datetime.utcnow().isoformat(),
            date_shift_days=self.date_shift_days,
            retain_patient_characteristics=self.retain_patient_characteristics,
        )

        input_path = Path(input_dir)
        output_path = Path(output_dir)

        if not input_path.exists():
            report.errors.append(f"Input directory not found: {input_dir}")
            return report

        # Discover DICOM files
        if recursive:
            dicom_files = list(input_path.rglob(file_pattern))
            # Also try files without extension (common in DICOM)
            dicom_files += [
                f for f in input_path.rglob("*")
                if f.is_file() and not f.suffix and f not in dicom_files
            ]
        else:
            dicom_files = list(input_path.glob(file_pattern))

        report.files_processed = len(dicom_files)
        logger.info(f"De-identifying {len(dicom_files)} DICOM files...")

        for dcm_file in sorted(dicom_files):
            # Preserve relative directory structure
            relative = dcm_file.relative_to(input_path)
            out_file = output_path / relative

            self.deidentify_file(dcm_file, out_file, report)

            if report.files_processed % 50 == 0:
                logger.info(
                    f"Progress: {report.files_succeeded + report.files_failed}"
                    f"/{report.files_processed}"
                )

        report.processing_time_ms = round((time.time() - t_start) * 1000, 1)

        logger.info(report.summary())
        return report

    def deidentify_series(
        self,
        dicom_files: list[Path],
        output_dir: str,
        patient_id_override: Optional[str] = None,
    ) -> DeidentificationReport:
        """
        De-identify a list of DICOM files (one series or study).

        Args:
            dicom_files: List of DICOM file paths
            output_dir: Output directory
            patient_id_override: Optional explicit patient pseudonym to use

        Returns:
            DeidentificationReport
        """
        t_start = time.time()

        report = DeidentificationReport(
            original_study_uid="",
            anonymized_study_uid="",
            original_patient_id="",
            anonymized_patient_id="",
            deidentification_date=datetime.utcnow().isoformat(),
            date_shift_days=self.date_shift_days,
        )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report.files_processed = len(dicom_files)

        for dcm_file in dicom_files:
            out_file = output_path / dcm_file.name
            self.deidentify_file(dcm_file, out_file, report)

        report.processing_time_ms = round((time.time() - t_start) * 1000, 1)
        return report

    def get_phi_tag_inventory(self) -> dict[str, dict]:
        """
        Return the complete PHI tag inventory with actions.

        Useful for compliance documentation.
        """
        inventory: dict[str, dict] = {}

        for tag_tuple, name in PHI_TAGS_DELETE.items():
            tag_str = f"({tag_tuple[0]:04X},{tag_tuple[1]:04X})"
            if tag_tuple in PHI_TAGS_PSEUDONYMISE:
                action = "pseudonymised"
            elif tag_tuple in PHI_TAGS_BLANK:
                action = "blanked"
            elif tag_tuple in UID_TAGS:
                action = "uid_remapped"
            else:
                action = "removed"
            inventory[tag_str] = {"name": name, "action": action}

        for tag_tuple, name in PHI_TAGS_RETAIN.items():
            tag_str = f"({tag_tuple[0]:04X},{tag_tuple[1]:04X})"
            inventory[tag_str] = {"name": name, "action": "retained"}

        return inventory

    def reset_uid_cache(self) -> None:
        """
        Clear the UID remapping cache.

        Call this between studies to prevent UIDs from one study
        being accidentally reused in another.
        """
        self._uid_cache.clear()
        logger.debug("UID cache cleared")
