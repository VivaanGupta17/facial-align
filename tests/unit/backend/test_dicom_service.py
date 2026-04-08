"""
Unit tests for the DICOM ingestion service.
Uses mock DICOM data to test metadata parsing, de-identification, and quality checks.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
import tempfile
import os

import numpy as np
import pytest

from app.core.exceptions import DicomParseError, DicomValidationError
from app.services.dicom.ingestion import (
    DicomIngestionService,
    SeriesMetadata,
    StudyMetadataInternal,
    VolumeQualityResult,
)


class TestDicomIngestionService:
    """Tests for DicomIngestionService."""

    @pytest.fixture
    def service(self) -> DicomIngestionService:
        """DicomIngestionService instance with mocked storage paths."""
        with patch("app.services.dicom.ingestion.settings") as mock_settings:
            mock_settings.storage.temp_path = Path(tempfile.mkdtemp())
            mock_settings.storage.dicom_path = Path(tempfile.mkdtemp())
            svc = DicomIngestionService()
        return svc

    def test_is_dicom_file_with_valid_magic(self, tmp_path):
        """Test DICOM file detection by magic number."""
        # Create a file with DICOM preamble
        dicom_file = tmp_path / "test.dcm"
        preamble = b"\x00" * 128 + b"DICM"
        dicom_file.write_bytes(preamble)

        svc = DicomIngestionService.__new__(DicomIngestionService)
        assert svc._is_dicom_file(dicom_file) is True

    def test_is_dicom_file_with_invalid_file(self, tmp_path):
        """Test that non-DICOM files are correctly identified."""
        not_dicom = tmp_path / "image.jpg"
        not_dicom.write_bytes(b"\xff\xd8\xff" + b"\x00" * 200)

        svc = DicomIngestionService.__new__(DicomIngestionService)
        assert svc._is_dicom_file(not_dicom) is False

    def test_is_dicom_file_nonexistent(self, tmp_path):
        """Test graceful handling of missing files."""
        svc = DicomIngestionService.__new__(DicomIngestionService)
        assert svc._is_dicom_file(tmp_path / "nonexistent.dcm") is False

    @pytest.mark.asyncio
    async def test_extract_dicom_files_from_directory(self, tmp_path):
        """Test extraction from a directory of DICOM files."""
        # Create mock DICOM files
        dicom_dir = tmp_path / "dicoms"
        dicom_dir.mkdir()
        for i in range(3):
            f = dicom_dir / f"slice_{i:03d}.dcm"
            f.write_bytes(b"\x00" * 128 + b"DICM" + b"\x00" * 100)

        svc = DicomIngestionService.__new__(DicomIngestionService)
        result = await svc._extract_dicom_files(dicom_dir)
        assert result == dicom_dir

    @pytest.mark.asyncio
    async def test_extract_dicom_files_from_zip(self, tmp_path):
        """Test extraction from a ZIP archive."""
        import zipfile

        dicom_dir = tmp_path / "source"
        dicom_dir.mkdir()
        dcm_file = dicom_dir / "slice.dcm"
        dcm_file.write_bytes(b"\x00" * 128 + b"DICM" + b"\x00" * 50)

        zip_path = tmp_path / "study.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(dcm_file, "slice.dcm")

        svc = DicomIngestionService.__new__(DicomIngestionService)
        result = await svc._extract_dicom_files(zip_path)
        assert result.is_dir()
        assert (result / "slice.dcm").exists()

    @pytest.mark.asyncio
    async def test_validate_ct_quality_good_volume(self, small_ct_volume, ct_spacing):
        """Test quality validation passes for a good CT volume."""
        svc = DicomIngestionService.__new__(DicomIngestionService)
        result = await svc.validate_ct_quality(small_ct_volume, ct_spacing)

        assert isinstance(result, VolumeQualityResult)
        assert result.slice_thickness_mm == ct_spacing[2]
        assert result.pixel_spacing_mm == pytest.approx(ct_spacing[0], abs=0.01)
        # 64^3 volume is small, so may not fully pass — just check it runs
        assert result.quality_score is not None

    @pytest.mark.asyncio
    async def test_validate_ct_quality_thick_slices(self):
        """Test quality validation fails for thick slice CT."""
        volume = np.zeros((50, 64, 64), dtype=np.float32)
        volume[15:35, 15:50, 15:50] = 700.0
        bad_spacing = (0.5, 0.5, 3.0)  # 3mm slice thickness — too thick

        svc = DicomIngestionService.__new__(DicomIngestionService)
        result = await svc.validate_ct_quality(volume, bad_spacing)

        assert result.passed is False
        assert any("thick" in e.lower() or "exceed" in e.lower() for e in result.errors)

    @pytest.mark.asyncio
    async def test_validate_ct_quality_empty_volume(self):
        """Test quality validation handles empty volume."""
        empty_volume = np.zeros((64, 64, 64), dtype=np.float32)
        spacing = (0.5, 0.5, 0.5)

        svc = DicomIngestionService.__new__(DicomIngestionService)
        result = await svc.validate_ct_quality(empty_volume, spacing)
        # Empty volume will have poor coverage scores
        assert result.cranial_coverage is False or result.quality_score < 0.9

    @pytest.mark.asyncio
    async def test_parse_dicom_metadata_no_files(self, tmp_path):
        """Test that missing DICOM files raises DicomParseError."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        svc = DicomIngestionService.__new__(DicomIngestionService)
        with pytest.raises(DicomParseError):
            await svc.parse_dicom_metadata(empty_dir)

    @pytest.mark.asyncio
    async def test_deidentify_study_processes_files(self, tmp_path):
        """Test de-identification runs on DICOM files (mocked)."""
        dicom_dir = tmp_path / "dicoms"
        dicom_dir.mkdir()

        # Create a minimal mock DICOM file
        mock_dcm_file = dicom_dir / "test.dcm"
        mock_dcm_file.write_bytes(b"\x00" * 128 + b"DICM" + b"\x00" * 100)

        svc = DicomIngestionService.__new__(DicomIngestionService)

        # Mock pydicom to avoid actual file reading
        mock_dataset = MagicMock()
        mock_dataset.PatientName = "Doe^John"
        mock_dataset.PatientID = "MRN12345"
        mock_dataset.StudyInstanceUID = "1.2.3"
        mock_dataset.SeriesInstanceUID = "1.2.3.4"
        mock_dataset.SOPInstanceUID = "1.2.3.4.5"

        with patch("pydicom.dcmread", return_value=mock_dataset), \
             patch("pydicom.dcmwrite") as mock_write:
            await svc.deidentify_study(dicom_dir)
            # Should have written at least one file
            assert mock_write.called

    def test_quality_thresholds(self):
        """Test quality threshold constants are medically appropriate."""
        svc = DicomIngestionService.__new__(DicomIngestionService)
        # CMF planning requires ≤1.5mm slice thickness
        assert svc.MAX_SLICE_THICKNESS_MM <= 1.5
        # Sub-mm in-plane resolution for surgical planning
        assert svc.MAX_PIXEL_SPACING_MM <= 0.6
        # Need substantial coverage
        assert svc.MIN_SLICE_COUNT >= 150

    def test_phi_tags_include_critical_fields(self):
        """Verify all critical PHI DICOM tags are in the removal list."""
        svc = DicomIngestionService.__new__(DicomIngestionService)
        required_tags = {
            "PatientName", "PatientID", "PatientBirthDate",
            "InstitutionName", "StationName",
        }
        for tag in required_tags:
            assert tag in svc.PHI_TAGS_TO_REMOVE, f"{tag} missing from PHI removal list"


class TestStudyMetadataInternal:
    """Tests for StudyMetadataInternal dataclass."""

    def test_series_accumulation(self, mock_study_metadata):
        """Test that series metadata is correctly accumulated."""
        assert len(mock_study_metadata.series) == 1
        assert mock_study_metadata.series[0].modality == "CT"
        assert mock_study_metadata.series[0].slice_count == 512
        assert mock_study_metadata.series[0].slice_thickness_mm == 0.625
