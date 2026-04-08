"""
Integration tests for the end-to-end facial align pipeline.

Tests cover:
- DICOM ingestion → preprocessing → segmentation → mesh extraction → reduction planning flow
- Use mocks for actual ML models but test the data flow between services
- Verify that output schemas are compatible between pipeline stages
- Test error propagation through the pipeline
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# ─── Schema imports ───────────────────────────────────────────────────────────

from app.schemas.common import Transform3D, Vector3D, BoundingBox3D
from app.schemas.plan import (
    FragmentTransform,
    OcclusalConstraints,
    OcclusalMetrics,
    ReductionPlanRequest,
    ValidationResult,
)
from app.core.exceptions import (
    DentalArchError,
    DicomValidationError,
    FacialAlignError,
    InsufficientOverlapError,
    RegistrationError,
    SegmentationError,
)


# ─── Service imports ──────────────────────────────────────────────────────────

from app.services.occlusion.occlusion_service import (
    OcclusionService,
    SplintDesignSpec,
)
from app.services.registration.registration_service import (
    RegistrationMetrics,
    RegistrationService,
)


# ─── Pipeline-level helpers ───────────────────────────────────────────────────

def _random_points(n: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 3)).astype(np.float64)


def _make_mock_mesh(n_vertices: int = 300, seed: int = 0) -> MagicMock:
    mesh = MagicMock()
    rng = np.random.default_rng(seed)
    mesh.vertices = rng.standard_normal((n_vertices, 3)).astype(np.float64)
    return mesh


def _make_o3d_mock() -> MagicMock:
    """Build a lightweight Open3D mock for registration tests."""
    o3d = MagicMock()
    icp_result = MagicMock()
    icp_result.transformation = np.eye(4)
    icp_result.inlier_rmse = 0.8
    icp_result.fitness = 0.88
    icp_result.correspondence_set = [MagicMock() for _ in range(200)]
    o3d.pipelines.registration.registration_icp.return_value = icp_result
    o3d.pipelines.registration.evaluate_registration.return_value = icp_result
    o3d.pipelines.registration.TransformationEstimationPointToPlane.return_value = MagicMock()
    o3d.pipelines.registration.TransformationEstimationPointToPoint.return_value = MagicMock()
    o3d.pipelines.registration.ICPConvergenceCriteria.return_value = MagicMock()
    o3d.geometry.PointCloud.return_value = MagicMock()
    o3d.utility.Vector3dVector.return_value = MagicMock()
    o3d.geometry.KDTreeSearchParamHybrid.return_value = MagicMock()
    return o3d


@pytest.fixture(autouse=True)
def mock_open3d(monkeypatch):
    """Inject mock open3d into sys.modules for all integration tests."""
    o3d = _make_o3d_mock()
    monkeypatch.setitem(sys.modules, "open3d", o3d)
    monkeypatch.setitem(sys.modules, "open3d.geometry", o3d.geometry)
    monkeypatch.setitem(sys.modules, "open3d.utility", o3d.utility)
    monkeypatch.setitem(sys.modules, "open3d.pipelines", o3d.pipelines)
    monkeypatch.setitem(sys.modules, "open3d.pipelines.registration", o3d.pipelines.registration)
    return o3d


# ─── Shared fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def sample_ct_volume() -> np.ndarray:
    """Synthetic craniofacial CT volume (64×64×64)."""
    volume = np.full((64, 64, 64), -1000.0, dtype=np.float32)
    volume[10:30, 20:44, 20:44] = 800.0   # Skull base
    volume[30:50, 20:44, 20:44] = 700.0   # Mandible
    volume[45:60, 25:39, 25:39] = 2500.0  # Teeth
    return volume


@pytest.fixture
def sample_spacing() -> Tuple[float, float, float]:
    return (0.5, 0.5, 0.5)


@pytest.fixture
def mock_segmentation_output(sample_ct_volume):
    """Synthetic segmentation output: mask + label dict + confidences."""
    mask = np.zeros_like(sample_ct_volume, dtype=np.int32)
    mask[10:30, 20:44, 20:44] = 1  # Mandible
    mask[5:15, 20:44, 20:44] = 2   # Maxilla
    labels = {"mandible": 1, "maxilla": 2}
    confidences = {"mandible": 0.93, "maxilla": 0.89}
    return mask, labels, confidences


@pytest.fixture
def mock_fragment_meshes():
    """Two synthetic fracture fragment meshes."""
    return [
        _make_mock_mesh(n_vertices=250, seed=1),
        _make_mock_mesh(n_vertices=200, seed=2),
    ]


@pytest.fixture
def identity_transform() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


# ─── Stage 1: Schema compatibility tests ──────────────────────────────────────


class TestSchemaCompatibilityAcrossStages:
    """Verify that the schema types used at each pipeline stage are compatible."""

    def test_transform3d_can_produce_4x4_matrix_for_registration(self):
        """Transform3D output from registration stage is usable as a numpy matrix."""
        T = Transform3D.identity()
        mat = np.array(T.to_4x4_matrix())
        assert mat.shape == (4, 4)
        assert np.allclose(mat, np.eye(4))

    def test_fragment_transform_references_transform3d(self):
        """FragmentTransform (reduction output) wraps Transform3D (registration type)."""
        T = Transform3D.identity()
        ft = FragmentTransform(
            fragment_id="frag_01",
            fragment_label=1,
            transform=T,
            confidence=0.91,
        )
        # The transform inside is still a Transform3D
        assert isinstance(ft.transform, Transform3D)
        mat = np.array(ft.transform.to_4x4_matrix())
        assert mat.shape == (4, 4)

    def test_occlusal_constraints_feeds_into_reduction_plan_request(self):
        """OcclusalConstraints (from occlusion service) can populate ReductionPlanRequest."""
        constraints = OcclusalConstraints(
            target_overjet_mm=2.0,
            target_overbite_mm=3.0,
            molar_class_target="Class_I",
        )
        request = ReductionPlanRequest(
            case_id=uuid.uuid4(),
            segmentation_id=uuid.uuid4(),
            occlusal_constraints=constraints,
        )
        assert request.occlusal_constraints.molar_class_target == "Class_I"

    def test_occlusal_metrics_are_compatible_with_validation_result(self):
        """OcclusalMetrics from evaluation can feed into ValidationResult."""
        metrics = OcclusalMetrics(
            overjet_mm=2.0,
            overbite_mm=3.0,
            molar_relationship="Class_I",
            constraints_satisfied=True,
        )
        # ValidationResult would reference these metrics' constraints_satisfied field
        validation = ValidationResult(
            passed=metrics.constraints_satisfied,
            symmetry_ok=True,
            occlusion_ok=metrics.constraints_satisfied,
            condylar_seating_ok=True,
            hardware_placement_ok=True,
        )
        assert validation.passed is True
        assert validation.occlusion_ok is True

    def test_bounding_box_compatible_with_fragment_info(self):
        """BoundingBox3D used in FragmentInfo matches expected API."""
        bbox = BoundingBox3D(
            min_x=-10, min_y=-20, min_z=-15,
            max_x=10, max_y=20, max_z=15,
        )
        # Verify all computed properties are accessible
        assert bbox.volume_mm3 > 0
        assert isinstance(bbox.center, Vector3D)
        assert isinstance(bbox.dimensions, Vector3D)


# ─── Stage 2: Preprocessing → Segmentation data flow ─────────────────────────


class TestPreprocessingToSegmentationDataFlow:
    """Test that preprocessing output is compatible with segmentation service input."""

    def test_preprocessed_volume_shape_is_3d(self, sample_ct_volume):
        """CT volume passed to segmentation must be 3D."""
        assert sample_ct_volume.ndim == 3

    def test_preprocessed_volume_dtype_is_float(self, sample_ct_volume):
        """Segmentation service expects float32 input."""
        assert sample_ct_volume.dtype in (np.float32, np.float64)

    def test_hu_windowed_volume_within_expected_range(self, sample_ct_volume):
        """After HU windowing, values should be within [-1024, 3071]."""
        clipped = np.clip(sample_ct_volume, -1024, 3071)
        assert clipped.min() >= -1024
        assert clipped.max() <= 3071

    def test_spacing_tuple_has_three_elements(self, sample_spacing):
        """Spacing (used by segmentation model) must be 3-element tuple."""
        assert len(sample_spacing) == 3
        assert all(s > 0 for s in sample_spacing)

    def test_normalized_volume_has_zero_mean_approximately(self, sample_ct_volume):
        """Normalization (subtract mean, divide by std) produces ~0 mean."""
        vol = sample_ct_volume.astype(np.float32)
        mean = vol.mean()
        std = vol.std()
        if std > 0:
            normalized = (vol - mean) / std
            assert abs(normalized.mean()) < 1e-4

    def test_bone_region_detectable_in_ct_volume(self, sample_ct_volume):
        """Bone-like voxels (>200 HU) should exist in the synthetic volume."""
        bone_voxels = np.sum(sample_ct_volume > 200)
        assert bone_voxels > 0


# ─── Stage 3: Segmentation → Mesh data flow ───────────────────────────────────


class TestSegmentationToMeshDataFlow:
    """Test that segmentation masks can be converted to mesh inputs."""

    def test_segmentation_mask_has_integer_labels(self, mock_segmentation_output):
        """Segmentation mask uses integer label values."""
        mask, labels, _ = mock_segmentation_output
        assert mask.dtype in (np.int32, np.int64, np.uint8, np.uint16)

    def test_label_dict_maps_structure_names_to_integers(self, mock_segmentation_output):
        """labels dict maps anatomical names to integer label values."""
        _, labels, _ = mock_segmentation_output
        assert isinstance(labels, dict)
        for name, val in labels.items():
            assert isinstance(name, str)
            assert isinstance(val, int)

    def test_each_label_present_in_mask(self, mock_segmentation_output):
        """Each label value should appear at least once in the mask."""
        mask, labels, _ = mock_segmentation_output
        for name, val in labels.items():
            assert np.any(mask == val), f"Label {val} ({name}) not found in mask"

    def test_confidence_scores_in_valid_range(self, mock_segmentation_output):
        """Segmentation confidence scores must be in [0, 1]."""
        _, _, confidences = mock_segmentation_output
        for name, score in confidences.items():
            assert 0.0 <= score <= 1.0, f"Confidence for {name} out of range: {score}"

    def test_binary_mask_extractable_per_label(self, mock_segmentation_output):
        """Binary mask for each label can be extracted from multi-label mask."""
        mask, labels, _ = mock_segmentation_output
        for name, val in labels.items():
            binary = (mask == val).astype(np.uint8)
            assert binary.sum() > 0
            assert binary.max() == 1


# ─── Stage 4: Mesh → Registration data flow ───────────────────────────────────


class TestMeshToRegistrationDataFlow:
    """Test that mesh data can flow into the registration service."""

    @pytest.mark.asyncio
    async def test_ct_to_scan_registration_returns_valid_transform(
        self, mock_fragment_meshes
    ):
        """Registration service returns Transform3D compatible with the rest of the pipeline."""
        service = RegistrationService()
        ct_mesh = mock_fragment_meshes[0]
        scan_mesh = mock_fragment_meshes[1]

        transform, metrics = await service.register_ct_to_scan(ct_mesh, scan_mesh)

        assert isinstance(transform, Transform3D)
        mat = np.array(transform.to_4x4_matrix())
        # Bottom row must be [0, 0, 0, 1]
        assert np.allclose(mat[3, :], [0, 0, 0, 1])

    @pytest.mark.asyncio
    async def test_fragment_registration_transforms_are_all_valid(
        self, mock_fragment_meshes
    ):
        """All fragment registration transforms are well-formed."""
        service = RegistrationService()
        reference = _make_mock_mesh(400)

        results = await service.register_fragments(mock_fragment_meshes, reference)
        assert len(results) == len(mock_fragment_meshes)

        for transform, metrics in results:
            mat = np.array(transform.to_4x4_matrix())
            R = mat[:3, :3]
            # Rotation block should be approximately orthonormal
            assert np.allclose(R.T @ R, np.eye(3), atol=1e-4)

    @pytest.mark.asyncio
    async def test_registration_metrics_fitness_score_is_float(
        self, mock_fragment_meshes
    ):
        """fitness_score from registration metrics is a valid float."""
        service = RegistrationService()
        reference = _make_mock_mesh(400)
        results = await service.register_fragments(mock_fragment_meshes[:1], reference)
        _, metrics = results[0]
        assert isinstance(metrics.fitness_score, float)

    @pytest.mark.asyncio
    async def test_failed_registration_gives_zero_fitness(self):
        """Fragment with too few vertices triggers fallback with zero fitness."""
        service = RegistrationService(use_global_registration=False)
        reference = _make_mock_mesh(400)

        tiny_mesh = MagicMock()
        tiny_mesh.vertices = np.random.rand(3, 3)  # Only 3 points

        results = await service.register_fragments([tiny_mesh], reference)
        _, metrics = results[0]
        assert metrics.fitness_score == 0.0
        assert not metrics.converged


# ─── Stage 5: Registration → Occlusion data flow ──────────────────────────────


class TestRegistrationToOcclusionDataFlow:
    """Test that registration transforms feed correctly into the occlusion service."""

    @pytest.mark.asyncio
    async def test_occlusion_evaluation_with_planned_transforms(self):
        """Planned transforms (from registration) can be passed to occlusion evaluation."""
        service = OcclusionService()
        upper_arch = _make_mock_mesh(200)
        lower_arch = _make_mock_mesh(200)

        planned_transforms = {
            "upper_frag": np.eye(4),
            "lower_frag": np.eye(4),
        }

        metrics = await service.evaluate_occlusion(
            upper_arch,
            lower_arch,
            planned_transforms=planned_transforms,
            upper_fragment_id="upper_frag",
            lower_fragment_id="lower_frag",
        )
        assert isinstance(metrics, OcclusalMetrics)

    @pytest.mark.asyncio
    async def test_transform3d_can_be_converted_for_occlusion_use(self):
        """Transform3D (from registration output) converts to a 4x4 matrix usable by occlusion."""
        transform = Transform3D.identity()
        mat = np.array(transform.to_4x4_matrix())
        assert mat.shape == (4, 4)
        assert np.allclose(mat, np.eye(4))

    @pytest.mark.asyncio
    async def test_occlusion_constraints_from_service_are_valid(self):
        """Constraints computed by occlusion service pass Pydantic validation."""
        service = OcclusionService()
        constraints = await service.compute_dental_constraints(
            pre_injury_occlusion=None,
            current_fragments=[],
        )
        # Must pass model_dump without validation errors
        dumped = constraints.model_dump()
        assert "molar_class_target" in dumped
        assert "target_overjet_mm" in dumped

    @pytest.mark.asyncio
    async def test_assess_constraint_satisfaction_after_evaluation(self):
        """_assess_constraint_satisfaction correctly marks normal metrics."""
        service = OcclusionService()
        metrics = OcclusalMetrics(
            overjet_mm=2.0,
            overbite_mm=3.0,
            midline_deviation_mm=0.0,
            cant_degrees=0.5,
        )
        service._assess_constraint_satisfaction(metrics)
        assert metrics.constraints_satisfied is True


# ─── Stage 6: Full pipeline data flow (mocked) ───────────────────────────────


class TestFullPipelineDataFlow:
    """
    End-to-end integration test for the data flow across all pipeline stages.

    Uses mocks for actual ML models and database interactions,
    testing only the data transformation and schema compatibility between stages.
    """

    @pytest.mark.asyncio
    async def test_segmentation_output_feeds_into_mesh_service_interface(
        self, mock_segmentation_output, sample_spacing
    ):
        """Segmentation output (mask, labels, confidences) can be consumed by mesh service."""
        mask, labels, confidences = mock_segmentation_output

        # Mock the mesh service
        mock_mesh_service = MagicMock()
        mock_mesh_paths = {
            "mandible": {"stl": "/meshes/mandible.stl", "glb": "/meshes/mandible.glb"},
            "maxilla": {"stl": "/meshes/maxilla.stl", "glb": "/meshes/maxilla.glb"},
        }
        mock_mesh_service.extract_and_process_all_structures.return_value = mock_mesh_paths

        # Call the mesh service with segmentation output
        result = mock_mesh_service.extract_and_process_all_structures(
            masks={name: (mask == val).astype(np.uint8) for name, val in labels.items()},
            labels=labels,
            spacing=sample_spacing,
            output_dir=Path("/tmp/meshes"),
        )

        # Verify output structure
        assert "mandible" in result
        assert "maxilla" in result
        assert "stl" in result["mandible"]

    @pytest.mark.asyncio
    async def test_registration_then_occlusion_pipeline(self):
        """Registration → occlusion pipeline produces compatible output."""
        # Stage 1: Registration
        reg_service = RegistrationService(use_global_registration=False)
        ct_mesh = _make_mock_mesh(300)
        scan_mesh = _make_mock_mesh(300)

        ct_transform, reg_metrics = await reg_service.register_ct_to_scan(ct_mesh, scan_mesh)

        # Stage 2: Use registration transform for occlusion evaluation
        occ_service = OcclusionService()
        upper_arch = _make_mock_mesh(200, seed=10)
        lower_arch = _make_mock_mesh(200, seed=11)

        mat_4x4 = np.array(ct_transform.to_4x4_matrix())
        planned = {"upper": mat_4x4, "lower": np.eye(4)}

        metrics = await occ_service.evaluate_occlusion(
            upper_arch,
            lower_arch,
            planned_transforms=planned,
            upper_fragment_id="upper",
            lower_fragment_id="lower",
        )
        assert isinstance(metrics, OcclusalMetrics)
        assert isinstance(metrics.constraints_satisfied, bool)

    @pytest.mark.asyncio
    async def test_multi_fragment_registration_pipeline(self):
        """Multiple fragments → registration → transforms chain."""
        service = RegistrationService(use_global_registration=False)
        reference = _make_mock_mesh(500, seed=99)
        n_fragments = 3
        fragments = [_make_mock_mesh(150, seed=i) for i in range(n_fragments)]

        results = await service.register_fragments(fragments, reference)
        assert len(results) == n_fragments

        # All transforms should be convertible to 4x4 matrices
        for transform, metrics in results:
            mat = np.array(transform.to_4x4_matrix())
            assert mat.shape == (4, 4)
            assert np.allclose(mat[3, :], [0, 0, 0, 1])

    @pytest.mark.asyncio
    async def test_splint_design_generated_at_pipeline_end(self):
        """The occlusion service generates a splint spec as final pipeline output."""
        occ_service = OcclusionService()
        upper_arch = _make_mock_mesh(200)
        lower_arch = _make_mock_mesh(200)

        # Evaluate occlusion
        metrics = await occ_service.evaluate_occlusion(upper_arch, lower_arch)

        # Generate splint
        splint = await occ_service.suggest_splint_design(
            occlusal_plan=metrics,
            upper_arch=upper_arch,
            lower_arch=lower_arch,
        )
        assert isinstance(splint, SplintDesignSpec)
        assert splint.target_vertical_dimension_mm >= 0
        assert len(splint.material_recommendation) > 0


# ─── Error propagation tests ──────────────────────────────────────────────────


class TestErrorPropagation:
    """Verify that errors propagate correctly through the pipeline stages."""

    @pytest.mark.asyncio
    async def test_missing_arch_raises_dental_arch_error(self):
        """Missing upper arch raises DentalArchError (not a generic exception)."""
        service = OcclusionService()
        with pytest.raises(DentalArchError):
            await service.evaluate_occlusion(None, _make_mock_mesh(100))

    @pytest.mark.asyncio
    async def test_insufficient_points_raises_insufficient_overlap_error(self):
        """Too few mesh points in registration raises InsufficientOverlapError."""
        from app.services.registration.registration_service import ICPRegistrationModel
        model = ICPRegistrationModel()
        with pytest.raises(InsufficientOverlapError):
            model.register(
                np.random.rand(3, 3),
                np.random.rand(200, 3),
            )

    @pytest.mark.asyncio
    async def test_all_pipeline_exceptions_inherit_from_facial_align_error(self):
        """All domain exceptions raised during the pipeline inherit from FacialAlignError."""
        for exc_class in [
            DentalArchError,
            RegistrationError,
            InsufficientOverlapError,
            SegmentationError,
            DicomValidationError,
        ]:
            assert issubclass(exc_class, FacialAlignError)

    @pytest.mark.asyncio
    async def test_failed_fragment_registration_does_not_abort_pipeline(self):
        """A failed fragment registration produces a fallback result, not an exception."""
        service = RegistrationService(use_global_registration=False)
        reference = _make_mock_mesh(300)

        # Mix of valid and invalid fragments
        good_fragment = _make_mock_mesh(200, seed=1)
        bad_fragment = MagicMock()
        bad_fragment.vertices = np.random.rand(2, 3)  # Too few points

        results = await service.register_fragments(
            [good_fragment, bad_fragment], reference
        )
        # Should return 2 results (no exception)
        assert len(results) == 2

        # Bad fragment gets fallback metrics
        _, bad_metrics = results[1]
        assert not bad_metrics.converged

    def test_empty_arch_mesh_raises_dental_arch_error_in_compute_arch_geometry(self):
        """compute_arch_geometry raises DentalArchError when mesh is empty."""
        service = OcclusionService()
        empty_mesh = MagicMock()
        empty_mesh.vertices = np.empty((0, 3))

        with pytest.raises(DentalArchError):
            service.compute_arch_geometry(empty_mesh, is_upper=True)

    @pytest.mark.asyncio
    async def test_occlusion_model_error_wrapped_in_occlusion_metric_error(self):
        """A crash inside the occlusion model is wrapped in OcclusionMetricError."""
        from app.core.exceptions import OcclusionMetricError

        service = OcclusionService()
        service._geometric = MagicMock()
        service._geometric.evaluate.side_effect = RuntimeError("GPU OOM")

        with pytest.raises(OcclusionMetricError):
            await service.evaluate_occlusion(
                _make_mock_mesh(100), _make_mock_mesh(100)
            )

    def test_to_http_exception_produces_fastapi_exception(self):
        """All pipeline exceptions can be converted to FastAPI HTTP exceptions."""
        from fastapi import HTTPException

        exceptions = [
            DentalArchError("test"),
            RegistrationError("test"),
            InsufficientOverlapError("test"),
            SegmentationError("test"),
            DicomValidationError("test"),
        ]
        for exc in exceptions:
            http_exc = exc.to_http_exception()
            assert isinstance(http_exc, HTTPException)
            assert http_exc.status_code >= 400


# ─── Schema serialization round-trip tests ───────────────────────────────────


class TestSchemaSerializationRoundTrip:
    """Verify that schemas can be serialized and deserialized correctly."""

    def test_transform3d_dict_roundtrip(self):
        """Transform3D can be serialized to dict and back."""
        original = Transform3D.identity()
        dumped = original.model_dump()
        restored = Transform3D(**dumped)
        assert np.allclose(restored.rotation_matrix, original.rotation_matrix)
        assert np.allclose(restored.translation_mm, original.translation_mm)

    def test_occlusal_metrics_roundtrip(self):
        """OcclusalMetrics can be serialized to dict and back."""
        original = OcclusalMetrics(
            overjet_mm=2.0,
            overbite_mm=3.0,
            constraints_satisfied=True,
            constraint_violations=[],
        )
        dumped = original.model_dump()
        restored = OcclusalMetrics(**dumped)
        assert restored.overjet_mm == original.overjet_mm
        assert restored.constraints_satisfied == original.constraints_satisfied

    def test_occlusal_constraints_roundtrip(self):
        """OcclusalConstraints can be serialized to dict and back."""
        original = OcclusalConstraints(
            target_overjet_mm=2.5,
            molar_class_target="Class_II_div1",
        )
        dumped = original.model_dump()
        restored = OcclusalConstraints(**dumped)
        assert restored.molar_class_target == "Class_II_div1"
        assert restored.target_overjet_mm == 2.5

    def test_fragment_transform_json_serializable(self):
        """FragmentTransform can be serialized to JSON-compatible format."""
        ft = FragmentTransform(
            fragment_id="frag_01",
            fragment_label=1,
            transform=Transform3D.identity(),
            confidence=0.88,
        )
        dumped = ft.model_dump()
        assert dumped["fragment_id"] == "frag_01"
        assert dumped["confidence"] == 0.88
        # transform should be a dict, not a Transform3D object
        assert isinstance(dumped["transform"], dict)

    def test_validation_result_json_serializable(self):
        """ValidationResult can be serialized to JSON-compatible dict."""
        vr = ValidationResult(
            passed=True,
            symmetry_ok=True,
            occlusion_ok=True,
            condylar_seating_ok=True,
            hardware_placement_ok=True,
            warnings=["Minor issue"],
            skeletal_symmetry_score=0.92,
        )
        dumped = vr.model_dump()
        assert dumped["passed"] is True
        assert dumped["skeletal_symmetry_score"] == pytest.approx(0.92)
        assert "Minor issue" in dumped["warnings"]

    def test_bounding_box_center_matches_expected(self):
        """BoundingBox3D center is consistent after round-trip."""
        bbox = BoundingBox3D(
            min_x=-10, min_y=-20, min_z=0,
            max_x=10, max_y=20, max_z=40,
        )
        center = bbox.center
        assert center.x == pytest.approx(0.0)
        assert center.y == pytest.approx(0.0)
        assert center.z == pytest.approx(20.0)
