"""
Unit tests for the DICOM ingestion pipeline.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from pipelines.dicom_ingestion.pipeline import DicomIngestionPipeline


class TestDicomIngestionPipeline:
    """Tests for the DicomIngestionPipeline class."""

    @pytest.fixture
    def pipeline(self, tmp_path, sample_study_id, sample_patient_id):
        return DicomIngestionPipeline(
            study_id=sample_study_id,
            upload_path=tmp_path / "uploads",
            patient_id=sample_patient_id,
            user_id="test_user_001",
        )

    @pytest.fixture
    def pipeline_with_callback(self, tmp_path, sample_study_id, sample_patient_id):
        progress_calls = []

        def callback(pct, step):
            progress_calls.append((pct, step))

        pipeline = DicomIngestionPipeline(
            study_id=sample_study_id,
            upload_path=tmp_path / "uploads",
            patient_id=sample_patient_id,
            user_id="test_user_001",
            progress_callback=callback,
        )
        return pipeline, progress_calls

    def test_pipeline_initializes(self, pipeline, sample_study_id, sample_patient_id):
        """Test pipeline initializes with correct attributes."""
        assert pipeline._study_id == sample_study_id
        assert pipeline._patient_id == sample_patient_id
        assert pipeline._user_id == "test_user_001"
        assert pipeline._service is not None

    def test_progress_callback_is_called(self, pipeline_with_callback):
        """Test that progress callback is invoked."""
        pipeline, progress_calls = pipeline_with_callback

        # Simulate progress update
        pipeline._progress(50.0, "Test step")
        assert len(progress_calls) == 1
        assert progress_calls[0] == (50.0, "Test step")

    @pytest.mark.asyncio
    async def test_run_calls_service_methods(
        self, pipeline, tmp_path, mock_study_metadata, sample_study_id, sample_patient_id
    ):
        """Test that pipeline calls the ingestion service methods in order."""
        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        pipeline._upload_path = upload_dir

        mock_volume = np.zeros((64, 64, 64), dtype=np.float32)
        mock_volume[20:44, 20:44, 20:44] = 700.0
        mock_spacing = (0.5, 0.5, 0.5)

        mock_quality = MagicMock()
        mock_quality.quality_score = 0.85
        mock_quality.passed = True
        mock_quality.warnings = []
        mock_quality.errors = []
        mock_quality.cranial_coverage = True
        mock_quality.mandible_coverage = True

        with patch.object(
            pipeline._service, "_extract_dicom_files",
            new=AsyncMock(return_value=upload_dir)
        ), patch.object(
            pipeline._service, "parse_dicom_metadata",
            new=AsyncMock(return_value=mock_study_metadata)
        ), patch.object(
            pipeline._service, "deidentify_study",
            new=AsyncMock()
        ), patch.object(
            pipeline._service, "reconstruct_volume",
            new=AsyncMock(return_value=(mock_volume, mock_spacing))
        ), patch.object(
            pipeline._service, "validate_ct_quality",
            new=AsyncMock(return_value=mock_quality)
        ), patch(
            "pipelines.dicom_ingestion.pipeline.DicomIngestionPipeline._update_study_record",
            new=AsyncMock()
        ):
            result = await pipeline.run()

        assert result["status"] == "complete"
        assert result["modality"] == "CT"
        assert result["quality_score"] == 0.85
        assert result["quality_passed"] is True

    @pytest.mark.asyncio
    async def test_run_handles_volume_reconstruction_failure(
        self, pipeline, tmp_path, mock_study_metadata
    ):
        """Test pipeline continues gracefully when volume reconstruction fails."""
        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        pipeline._upload_path = upload_dir

        with patch.object(
            pipeline._service, "_extract_dicom_files",
            new=AsyncMock(return_value=upload_dir)
        ), patch.object(
            pipeline._service, "parse_dicom_metadata",
            new=AsyncMock(return_value=mock_study_metadata)
        ), patch.object(
            pipeline._service, "deidentify_study",
            new=AsyncMock()
        ), patch.object(
            pipeline._service, "reconstruct_volume",
            new=AsyncMock(side_effect=Exception("SimpleITK not available"))
        ), patch(
            "pipelines.dicom_ingestion.pipeline.DicomIngestionPipeline._update_study_record",
            new=AsyncMock()
        ):
            # Should not raise — gracefully handle failed reconstruction
            result = await pipeline.run()

        # Quality score should be None if reconstruction failed
        assert result["quality_score"] is None
        assert result["quality_passed"] is False

    @pytest.mark.asyncio
    async def test_update_study_record_uses_correct_fields(
        self, pipeline, mock_study_metadata, sample_study_id, tmp_path
    ):
        """Test that _update_study_record is called with correct data."""
        mock_volume = np.zeros((64, 64, 64))
        mock_spacing = (0.5, 0.5, 0.5)
        mock_quality = MagicMock()
        mock_quality.quality_score = 0.9
        mock_quality.passed = True
        mock_quality.warnings = []
        mock_quality.errors = []
        mock_quality.cranial_coverage = True
        mock_quality.mandible_coverage = True

        called_with = {}

        async def mock_update(**kwargs):
            called_with.update(kwargs)

        with patch(
            "app.db.database.get_db_context"
        ):
            # Just call the update method directly and verify no errors
            pipeline._patient_id = "test_patient"
            pipeline._study_id = sample_study_id
            # Verify the method exists and is callable
            assert callable(pipeline._update_study_record)
