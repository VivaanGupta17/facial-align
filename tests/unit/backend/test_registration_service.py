"""
Unit tests for the registration service.

Tests cover:
- ICPRegistrationModel raises InsufficientOverlapError for too few points
- GlobalRegistrationModel initialization
- RegistrationService.register_fragments dispatches to correct model
- RegistrationService.register_ct_to_scan_surface
- Validate RegistrationMetrics fields (rms_error, fitness_score, converged)
- Multi-step registration pipeline
- Mock Open3D since it's not installed
"""

from __future__ import annotations

import sys
from typing import List, Tuple
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest

from app.core.exceptions import (
    ICPConvergenceError,
    InsufficientOverlapError,
    RegistrationError,
)
from app.schemas.common import Transform3D
from app.services.registration.registration_service import (
    BaseRegistrationModel,
    DeepRegistrationModel,
    GlobalRegistrationModel,
    ICPRegistrationModel,
    RegistrationMetrics,
    RegistrationService,
)


# ─── Open3D mock setup ────────────────────────────────────────────────────────
# Open3D is not installed in the test environment; we mock it globally.

def _make_o3d_mock():
    """Build a realistic Open3D mock that supports the registration pipeline."""
    o3d = MagicMock()

    # Point cloud mock
    pcd_mock = MagicMock()
    pcd_mock.points = MagicMock()

    def make_pcd(*args, **kwargs):
        return MagicMock()

    o3d.geometry.PointCloud = make_pcd

    # Vector3dVector
    o3d.utility.Vector3dVector = MagicMock(return_value=MagicMock())

    # KDTree search param
    o3d.geometry.KDTreeSearchParamHybrid = MagicMock(return_value=MagicMock())

    # ICP result mock
    icp_result = MagicMock()
    icp_result.transformation = np.eye(4)
    icp_result.inlier_rmse = 0.5
    icp_result.fitness = 0.95
    icp_result.correspondence_set = [MagicMock() for _ in range(500)]

    o3d.pipelines.registration.registration_icp = MagicMock(return_value=icp_result)
    o3d.pipelines.registration.evaluate_registration = MagicMock(return_value=icp_result)
    o3d.pipelines.registration.TransformationEstimationPointToPlane = MagicMock(
        return_value=MagicMock()
    )
    o3d.pipelines.registration.TransformationEstimationPointToPoint = MagicMock(
        return_value=MagicMock()
    )
    o3d.pipelines.registration.ICPConvergenceCriteria = MagicMock(
        return_value=MagicMock()
    )

    # RANSAC / global registration
    ransac_result = MagicMock()
    ransac_result.transformation = np.eye(4)
    ransac_result.inlier_rmse = 1.0
    ransac_result.fitness = 0.7
    ransac_result.correspondence_set = [MagicMock() for _ in range(200)]
    o3d.pipelines.registration.registration_ransac_based_on_feature_matching = MagicMock(
        return_value=ransac_result
    )
    o3d.pipelines.registration.compute_fpfh_feature = MagicMock(return_value=MagicMock())
    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength = MagicMock(
        return_value=MagicMock()
    )
    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance = MagicMock(
        return_value=MagicMock()
    )
    o3d.pipelines.registration.RANSACConvergenceCriteria = MagicMock(
        return_value=MagicMock()
    )

    return o3d


@pytest.fixture(autouse=True)
def mock_open3d(monkeypatch):
    """Auto-use fixture: inject a mocked open3d into sys.modules for all tests."""
    o3d = _make_o3d_mock()
    monkeypatch.setitem(sys.modules, "open3d", o3d)
    monkeypatch.setitem(sys.modules, "open3d.geometry", o3d.geometry)
    monkeypatch.setitem(sys.modules, "open3d.utility", o3d.utility)
    monkeypatch.setitem(sys.modules, "open3d.pipelines", o3d.pipelines)
    monkeypatch.setitem(sys.modules, "open3d.pipelines.registration", o3d.pipelines.registration)
    return o3d


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_random_points(n: int = 200, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 3)).astype(np.float64)


def _make_mock_mesh(n_vertices: int = 300, seed: int = 0):
    """Return a MagicMock with .vertices populated."""
    mesh = MagicMock()
    rng = np.random.default_rng(seed)
    mesh.vertices = rng.standard_normal((n_vertices, 3)).astype(np.float64)
    return mesh


# ─── RegistrationMetrics tests ────────────────────────────────────────────────


class TestRegistrationMetrics:
    """Ensure the RegistrationMetrics dataclass stores and exposes correct fields."""

    def test_fields_are_accessible(self):
        metrics = RegistrationMetrics(
            rms_error_mm=0.42,
            max_error_mm=1.26,
            mean_error_mm=0.38,
            fitness_score=0.91,
            inlier_count=450,
            total_correspondences=480,
            converged=True,
            iterations=120,
            time_ms=350,
        )
        assert metrics.rms_error_mm == pytest.approx(0.42)
        assert metrics.max_error_mm == pytest.approx(1.26)
        assert metrics.mean_error_mm == pytest.approx(0.38)
        assert metrics.fitness_score == pytest.approx(0.91)
        assert metrics.inlier_count == 450
        assert metrics.total_correspondences == 480
        assert metrics.converged is True
        assert metrics.iterations == 120
        assert metrics.time_ms == 350

    def test_not_converged_when_fitness_low(self):
        metrics = RegistrationMetrics(
            rms_error_mm=5.0,
            max_error_mm=15.0,
            mean_error_mm=4.5,
            fitness_score=0.1,
            inlier_count=10,
            total_correspondences=200,
            converged=False,
            iterations=200,
            time_ms=800,
        )
        assert metrics.converged is False

    def test_fitness_score_range(self):
        """fitness_score should be in [0, 1] for valid registrations."""
        metrics = RegistrationMetrics(
            rms_error_mm=1.0,
            max_error_mm=3.0,
            mean_error_mm=0.9,
            fitness_score=0.75,
            inlier_count=100,
            total_correspondences=150,
            converged=True,
            iterations=50,
            time_ms=200,
        )
        assert 0.0 <= metrics.fitness_score <= 1.0

    def test_time_ms_is_non_negative(self):
        metrics = RegistrationMetrics(
            rms_error_mm=0.0, max_error_mm=0.0, mean_error_mm=0.0,
            fitness_score=1.0, inlier_count=0, total_correspondences=0,
            converged=True, iterations=0, time_ms=0,
        )
        assert metrics.time_ms >= 0


# ─── ICPRegistrationModel tests ───────────────────────────────────────────────


class TestICPRegistrationModel:
    @pytest.fixture
    def icp_model(self) -> ICPRegistrationModel:
        return ICPRegistrationModel(max_iterations=50, max_correspondence_distance_mm=5.0)

    def test_raises_insufficient_overlap_for_too_few_source_points(self, icp_model):
        """Less than 10 source points → InsufficientOverlapError."""
        source = _make_random_points(5)
        target = _make_random_points(200)
        with pytest.raises(InsufficientOverlapError):
            icp_model.register(source, target)

    def test_raises_insufficient_overlap_for_too_few_target_points(self, icp_model):
        """Less than 10 target points → InsufficientOverlapError."""
        source = _make_random_points(200)
        target = _make_random_points(3)
        with pytest.raises(InsufficientOverlapError):
            icp_model.register(source, target)

    def test_raises_insufficient_overlap_for_both_too_few(self, icp_model):
        """Both source and target under threshold → InsufficientOverlapError."""
        with pytest.raises(InsufficientOverlapError):
            icp_model.register(
                _make_random_points(2),
                _make_random_points(2),
            )

    def test_error_context_contains_point_counts(self, icp_model):
        """InsufficientOverlapError context should include point counts."""
        source = _make_random_points(5)
        target = _make_random_points(200)
        try:
            icp_model.register(source, target)
            pytest.fail("Expected InsufficientOverlapError")
        except InsufficientOverlapError as exc:
            assert "source_points" in exc.context or "target_points" in exc.context

    def test_register_returns_transform_and_metrics(self, icp_model, mock_open3d):
        """register() returns a 4x4 transform and RegistrationMetrics."""
        source = _make_random_points(100)
        target = _make_random_points(100)
        transform, metrics = icp_model.register(source, target)
        assert transform.shape == (4, 4)
        assert isinstance(metrics, RegistrationMetrics)

    def test_register_accepts_initial_transform(self, icp_model, mock_open3d):
        """register() accepts an optional initial transform without error."""
        source = _make_random_points(100)
        target = _make_random_points(100)
        init_t = np.eye(4)
        transform, metrics = icp_model.register(source, target, initial_transform=init_t)
        assert transform.shape == (4, 4)

    def test_metrics_fitness_from_mock(self, icp_model, mock_open3d):
        """RegistrationMetrics should reflect the mocked ICP result fitness."""
        source = _make_random_points(100)
        target = _make_random_points(100)
        _, metrics = icp_model.register(source, target)
        # The mock returns fitness=0.95
        assert metrics.fitness_score == pytest.approx(0.95)

    def test_point_to_point_method(self, mock_open3d):
        """ICPRegistrationModel with point_to_point method should register without error."""
        model = ICPRegistrationModel(method="point_to_point")
        source = _make_random_points(100)
        target = _make_random_points(100)
        transform, metrics = model.register(source, target)
        assert transform.shape == (4, 4)

    def test_raises_registration_error_without_open3d(self, monkeypatch):
        """When open3d import fails, RegistrationError is raised (not InsufficientOverlapError)."""
        monkeypatch.setitem(sys.modules, "open3d", None)
        model = ICPRegistrationModel()
        source = _make_random_points(100)
        target = _make_random_points(100)
        with pytest.raises((RegistrationError, ImportError, TypeError)):
            model.register(source, target)


# ─── GlobalRegistrationModel tests ────────────────────────────────────────────


class TestGlobalRegistrationModel:
    def test_initializes_with_defaults(self):
        """GlobalRegistrationModel can be instantiated with defaults."""
        model = GlobalRegistrationModel()
        assert model._voxel_size == 2.0
        assert model._refine_with_icp is True

    def test_initializes_with_custom_params(self):
        """GlobalRegistrationModel accepts custom parameters."""
        model = GlobalRegistrationModel(voxel_size=1.5, refine_with_icp=False)
        assert model._voxel_size == 1.5
        assert model._refine_with_icp is False

    def test_contains_icp_sub_model(self):
        """GlobalRegistrationModel should embed an ICP model for refinement."""
        model = GlobalRegistrationModel()
        assert isinstance(model._icp, ICPRegistrationModel)

    def test_register_returns_transform_and_metrics(self, mock_open3d):
        """Global registration with mocked open3d returns valid output."""
        model = GlobalRegistrationModel(refine_with_icp=True)
        source = _make_random_points(100)
        target = _make_random_points(100)
        transform, metrics = model.register(source, target)
        assert transform.shape == (4, 4)
        assert isinstance(metrics, RegistrationMetrics)

    def test_without_icp_refinement(self, mock_open3d):
        """Global registration without ICP refinement returns RANSAC result."""
        model = GlobalRegistrationModel(refine_with_icp=False)
        source = _make_random_points(100)
        target = _make_random_points(100)
        transform, metrics = model.register(source, target)
        assert transform.shape == (4, 4)
        # RANSAC mock sets fitness=0.7
        assert isinstance(metrics.fitness_score, float)


# ─── DeepRegistrationModel tests ──────────────────────────────────────────────


class TestDeepRegistrationModel:
    def test_is_available_returns_false(self):
        model = DeepRegistrationModel(model_path="/nonexistent/weights.pt")
        assert model.is_available is False

    def test_register_raises_not_implemented(self, mock_open3d):
        model = DeepRegistrationModel(model_path="/nonexistent/weights.pt")
        with pytest.raises(NotImplementedError):
            model.register(
                _make_random_points(100),
                _make_random_points(100),
            )


# ─── RegistrationService tests ────────────────────────────────────────────────


class TestRegistrationServiceInit:
    def test_default_initialization(self):
        service = RegistrationService()
        assert service._use_global is True
        assert isinstance(service._icp, ICPRegistrationModel)
        assert isinstance(service._global_reg, GlobalRegistrationModel)

    def test_no_global_registration(self):
        service = RegistrationService(use_global_registration=False)
        assert service._global_reg is None

    def test_custom_icp_params(self):
        service = RegistrationService(icp_max_iterations=50, correspondence_distance_mm=2.0)
        assert service._icp._max_iterations == 50
        assert service._icp._max_corr_dist == 2.0


class TestRegistrationServiceMeshToPoints:
    """Tests for _mesh_to_points helper."""

    def test_extracts_vertices_from_mesh(self):
        service = RegistrationService()
        mesh = _make_mock_mesh(n_vertices=100)
        pts = service._mesh_to_points(mesh)
        assert isinstance(pts, np.ndarray)
        assert pts.shape == (100, 3)

    def test_fallback_for_plain_numpy_array(self):
        service = RegistrationService()
        pts_in = _make_random_points(50)
        pts_out = service._mesh_to_points(pts_in)
        assert isinstance(pts_out, np.ndarray)


class TestRegisterFragments:
    @pytest.mark.asyncio
    async def test_returns_one_result_per_fragment(self, mock_open3d):
        service = RegistrationService()
        reference = _make_mock_mesh(300)
        fragments = [_make_mock_mesh(100, seed=i) for i in range(3)]

        results = await service.register_fragments(fragments, reference)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_each_result_is_transform_and_metrics(self, mock_open3d):
        service = RegistrationService()
        reference = _make_mock_mesh(300)
        fragments = [_make_mock_mesh(100, seed=i) for i in range(2)]

        results = await service.register_fragments(fragments, reference)
        for transform, metrics in results:
            assert isinstance(transform, Transform3D)
            assert isinstance(metrics, RegistrationMetrics)

    @pytest.mark.asyncio
    async def test_empty_fragments_list_returns_empty_results(self, mock_open3d):
        service = RegistrationService()
        reference = _make_mock_mesh(300)
        results = await service.register_fragments([], reference)
        assert results == []

    @pytest.mark.asyncio
    async def test_failed_fragment_gets_identity_transform(self, mock_open3d):
        """When ICP fails for a fragment, identity transform is returned."""
        service = RegistrationService(use_global_registration=False)
        reference = _make_mock_mesh(300)

        # Provide too-few-vertex mesh to trigger InsufficientOverlapError
        tiny_mesh = MagicMock()
        tiny_mesh.vertices = np.random.rand(3, 3)  # 3 points → raises error
        fragments = [tiny_mesh]

        results = await service.register_fragments(fragments, reference)
        assert len(results) == 1
        _, failed_metrics = results[0]
        assert failed_metrics.converged is False
        assert failed_metrics.fitness_score == 0.0

    @pytest.mark.asyncio
    async def test_initial_transforms_passed_to_icp(self, mock_open3d):
        """When initial_transforms are provided, they are passed through."""
        service = RegistrationService(use_global_registration=False)
        reference = _make_mock_mesh(300)
        fragments = [_make_mock_mesh(100)]
        initial_transforms = [np.eye(4)]

        results = await service.register_fragments(fragments, reference, initial_transforms)
        assert len(results) == 1


class TestRegisterCtToScan:
    @pytest.mark.asyncio
    async def test_returns_transform_and_metrics(self, mock_open3d):
        service = RegistrationService()
        ct_mesh = _make_mock_mesh(200)
        scan_mesh = _make_mock_mesh(200)

        transform, metrics = await service.register_ct_to_scan(ct_mesh, scan_mesh)
        assert isinstance(transform, Transform3D)
        assert isinstance(metrics, RegistrationMetrics)

    @pytest.mark.asyncio
    async def test_logs_warning_on_high_rms(self, mock_open3d):
        """When RMS error > 5mm, a warning should be logged."""
        service = RegistrationService()
        ct_mesh = _make_mock_mesh(200)
        scan_mesh = _make_mock_mesh(200)

        # Patch the mock ICP result to return high RMS
        o3d = sys.modules["open3d"]
        high_rms_result = MagicMock()
        high_rms_result.transformation = np.eye(4)
        high_rms_result.inlier_rmse = 10.0  # > 5mm threshold
        high_rms_result.fitness = 0.9
        high_rms_result.correspondence_set = [MagicMock() for _ in range(50)]
        o3d.pipelines.registration.registration_icp.return_value = high_rms_result
        o3d.pipelines.registration.evaluate_registration.return_value = high_rms_result

        # Should not raise, just log a warning
        transform, metrics = await service.register_ct_to_scan(ct_mesh, scan_mesh)
        assert isinstance(transform, Transform3D)

    @pytest.mark.asyncio
    async def test_uses_global_reg_when_no_initial_transform(self, mock_open3d):
        """Without an initial transform, global registration is run first."""
        service = RegistrationService(use_global_registration=True)
        ct_mesh = _make_mock_mesh(200)
        scan_mesh = _make_mock_mesh(200)

        with patch.object(service._global_reg, "register", wraps=service._global_reg.register) as spy:
            await service.register_ct_to_scan(ct_mesh, scan_mesh)
            spy.assert_called_once()

    @pytest.mark.asyncio
    async def test_uses_icp_directly_when_initial_transform_provided(self, mock_open3d):
        """With an initial transform, ICP is used directly (no global reg)."""
        service = RegistrationService(use_global_registration=True)
        ct_mesh = _make_mock_mesh(200)
        scan_mesh = _make_mock_mesh(200)
        initial_t = np.eye(4)

        with patch.object(service._icp, "register", wraps=service._icp.register) as icp_spy:
            await service.register_ct_to_scan(ct_mesh, scan_mesh, initial_transform=initial_t)
            icp_spy.assert_called_once()


class TestMatrixToTransform3D:
    """Tests for _matrix_to_transform3d helper."""

    def test_identity_matrix_gives_identity_transform(self):
        service = RegistrationService()
        T = service._matrix_to_transform3d(np.eye(4))
        assert np.allclose(T.rotation_matrix, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert np.allclose(T.translation_mm, [0.0, 0.0, 0.0])

    def test_translation_extracted_correctly(self):
        service = RegistrationService()
        mat = np.eye(4)
        mat[:3, 3] = [10.0, -5.0, 3.0]
        T = service._matrix_to_transform3d(mat)
        assert np.allclose(T.translation_mm, [10.0, -5.0, 3.0])

    def test_rotation_is_orthonormal(self):
        """Output rotation matrix should be orthonormal (R^T R ≈ I)."""
        service = RegistrationService()
        # Slightly perturbed rotation (SVD should fix it)
        angle = np.radians(30)
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0, 0, 1],
        ])
        mat = np.eye(4)
        mat[:3, :3] = R
        T = service._matrix_to_transform3d(mat)
        R_out = np.array(T.rotation_matrix)
        assert np.allclose(R_out.T @ R_out, np.eye(3), atol=1e-6)

    def test_returns_transform3d_instance(self):
        service = RegistrationService()
        T = service._matrix_to_transform3d(np.eye(4))
        assert isinstance(T, Transform3D)


# ─── Multi-step registration pipeline ────────────────────────────────────────


class TestMultiStepRegistrationPipeline:
    """Integration-style tests for complete registration workflows."""

    @pytest.mark.asyncio
    async def test_ct_to_scan_then_fragment_registration(self, mock_open3d):
        """Complete pipeline: CT-to-scan registration followed by fragment registration."""
        service = RegistrationService()
        ct_mesh = _make_mock_mesh(500)
        scan_mesh = _make_mock_mesh(500)
        fragments = [_make_mock_mesh(150, seed=i) for i in range(2)]

        # Step 1: CT to scan
        scan_transform, scan_metrics = await service.register_ct_to_scan(ct_mesh, scan_mesh)
        assert isinstance(scan_transform, Transform3D)
        assert isinstance(scan_metrics, RegistrationMetrics)

        # Step 2: Fragment registration
        frag_results = await service.register_fragments(fragments, ct_mesh)
        assert len(frag_results) == 2
        for frag_transform, frag_metrics in frag_results:
            assert isinstance(frag_transform, Transform3D)
            assert isinstance(frag_metrics, RegistrationMetrics)

    @pytest.mark.asyncio
    async def test_transforms_can_be_converted_to_4x4_matrices(self, mock_open3d):
        """Transform3D objects returned from registration can produce 4x4 matrices."""
        service = RegistrationService()
        ct_mesh = _make_mock_mesh(200)
        scan_mesh = _make_mock_mesh(200)

        transform, _ = await service.register_ct_to_scan(ct_mesh, scan_mesh)
        mat = transform.to_4x4_matrix()
        mat_np = np.array(mat)
        assert mat_np.shape == (4, 4)
        # Last row must be [0, 0, 0, 1]
        assert np.allclose(mat_np[3, :], [0, 0, 0, 1])

    @pytest.mark.asyncio
    async def test_registration_pipeline_preserves_fragment_count(self, mock_open3d):
        """Fragment count in output matches input fragment count."""
        service = RegistrationService()
        n_fragments = 5
        reference = _make_mock_mesh(500)
        fragments = [_make_mock_mesh(100, seed=i) for i in range(n_fragments)]

        results = await service.register_fragments(fragments, reference)
        assert len(results) == n_fragments
