"""
Unit tests for the occlusion analysis service.

Tests cover:
- GeometricOcclusionModel.evaluate returns OcclusalMetrics
- OcclusionService._assess_constraint_satisfaction with normal and abnormal ranges
- OcclusionService.evaluate_occlusion raises DentalArchError when arches are None
- OcclusionService.compute_dental_constraints returns defaults when no pre-injury data
- OcclusionService.assess_molar_relationship classifies Class I, II, III correctly
- OcclusionService.compute_arch_geometry raises DentalArchError on empty mesh
- SplintDesignSpec generation
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.core.exceptions import DentalArchError, OcclusionMetricError
from app.schemas.plan import OcclusalConstraints, OcclusalMetrics
from app.services.occlusion.occlusion_service import (
    ArchGeometry,
    GeometricOcclusionModel,
    LearnedOcclusionModel,
    NORMAL_OCCLUSAL_RANGES,
    OcclusalContact,
    OcclusionService,
    SplintDesignSpec,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def occlusion_service() -> OcclusionService:
    """Default OcclusionService with geometric model."""
    return OcclusionService(use_learned_model=False)


@pytest.fixture
def geometric_model() -> GeometricOcclusionModel:
    return GeometricOcclusionModel()


@pytest.fixture
def mock_upper_arch():
    """Mock upper dental arch mesh with vertex data."""
    mesh = MagicMock()
    # Generate a plausible arch shape: an ellipse of points
    t = np.linspace(0, np.pi, 50)
    x = 25.0 * np.cos(t)
    y = 15.0 * np.sin(t)
    z = np.zeros(50)
    vertices = np.stack([x, y, z], axis=1).astype(np.float32)
    mesh.vertices = vertices
    return mesh


@pytest.fixture
def mock_lower_arch():
    """Mock lower dental arch mesh with vertex data."""
    mesh = MagicMock()
    t = np.linspace(0, np.pi, 50)
    x = 23.0 * np.cos(t)
    y = 13.0 * np.sin(t)
    z = np.full(50, -5.0)  # Lower arch is a few mm below
    vertices = np.stack([x, y, z], axis=1).astype(np.float32)
    mesh.vertices = vertices
    return mesh


@pytest.fixture
def empty_mesh():
    """Mock mesh with no vertices."""
    mesh = MagicMock()
    mesh.vertices = np.empty((0, 3), dtype=np.float32)
    return mesh


@pytest.fixture
def identity_transform() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


# ─── GeometricOcclusionModel tests ────────────────────────────────────────────


class TestGeometricOcclusionModel:
    def test_evaluate_returns_occlusal_metrics(
        self, geometric_model, mock_upper_arch, mock_lower_arch, identity_transform
    ):
        """GeometricOcclusionModel.evaluate should return an OcclusalMetrics instance."""
        metrics = geometric_model.evaluate(
            mock_upper_arch,
            mock_lower_arch,
            identity_transform,
            identity_transform,
        )
        assert isinstance(metrics, OcclusalMetrics)

    def test_evaluate_constraints_satisfied_by_default(
        self, geometric_model, mock_upper_arch, mock_lower_arch, identity_transform
    ):
        """Default geometric evaluation returns constraints_satisfied=True (placeholder)."""
        metrics = geometric_model.evaluate(
            mock_upper_arch,
            mock_lower_arch,
            identity_transform,
            identity_transform,
        )
        assert metrics.constraints_satisfied is True

    def test_evaluate_returns_empty_violations_by_default(
        self, geometric_model, mock_upper_arch, mock_lower_arch, identity_transform
    ):
        """Default geometric evaluation returns an empty constraint_violations list."""
        metrics = geometric_model.evaluate(
            mock_upper_arch,
            mock_lower_arch,
            identity_transform,
            identity_transform,
        )
        assert metrics.constraint_violations == []

    def test_evaluate_with_non_identity_transforms(
        self, geometric_model, mock_upper_arch, mock_lower_arch
    ):
        """Evaluation with non-identity transforms should still return valid OcclusalMetrics."""
        angle = np.radians(5)
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle),  np.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        metrics = geometric_model.evaluate(
            mock_upper_arch, mock_lower_arch, R, np.eye(4)
        )
        assert isinstance(metrics, OcclusalMetrics)

    def test_evaluate_optional_metric_fields_are_none(
        self, geometric_model, mock_upper_arch, mock_lower_arch, identity_transform
    ):
        """Geometric placeholder returns None for all optional metric fields."""
        metrics = geometric_model.evaluate(
            mock_upper_arch, mock_lower_arch, identity_transform, identity_transform
        )
        assert metrics.overjet_mm is None
        assert metrics.overbite_mm is None
        assert metrics.molar_relationship is None
        assert metrics.midline_deviation_mm is None
        assert metrics.cant_degrees is None


# ─── LearnedOcclusionModel tests ──────────────────────────────────────────────


class TestLearnedOcclusionModel:
    def test_is_available_returns_false(self):
        model = LearnedOcclusionModel()
        assert model.is_available is False

    def test_evaluate_raises_not_implemented(self, mock_upper_arch, mock_lower_arch, identity_transform):
        model = LearnedOcclusionModel()
        with pytest.raises(NotImplementedError):
            model.evaluate(mock_upper_arch, mock_lower_arch, identity_transform, identity_transform)


# ─── OcclusionService._assess_constraint_satisfaction ─────────────────────────


class TestAssessConstraintSatisfaction:
    """Tests for OcclusionService._assess_constraint_satisfaction."""

    def test_normal_values_produce_no_violations(self, occlusion_service):
        """Metrics within normal ranges produce no violations."""
        metrics = OcclusalMetrics(
            overjet_mm=2.0,
            overbite_mm=3.0,
            midline_deviation_mm=0.5,
            cant_degrees=1.0,
            constraints_satisfied=False,
        )
        occlusion_service._assess_constraint_satisfaction(metrics)
        assert metrics.constraints_satisfied is True
        assert metrics.constraint_violations == []

    def test_abnormal_overjet_produces_violation(self, occlusion_service):
        """Overjet outside [1, 3]mm should produce a violation."""
        metrics = OcclusalMetrics(overjet_mm=5.0)
        occlusion_service._assess_constraint_satisfaction(metrics)
        assert not metrics.constraints_satisfied
        assert any("Overjet" in v for v in metrics.constraint_violations)

    def test_negative_overjet_produces_violation(self, occlusion_service):
        """Negative overjet (underbite) should produce a violation."""
        metrics = OcclusalMetrics(overjet_mm=-1.0)
        occlusion_service._assess_constraint_satisfaction(metrics)
        assert not metrics.constraints_satisfied
        assert any("Overjet" in v for v in metrics.constraint_violations)

    def test_abnormal_overbite_produces_violation(self, occlusion_service):
        """Overbite outside [2, 4]mm should produce a violation."""
        metrics = OcclusalMetrics(overbite_mm=6.0)
        occlusion_service._assess_constraint_satisfaction(metrics)
        assert not metrics.constraints_satisfied
        assert any("Overbite" in v for v in metrics.constraint_violations)

    def test_excessive_midline_deviation_produces_violation(self, occlusion_service):
        """Midline deviation >1mm or <-1mm should produce a violation."""
        metrics = OcclusalMetrics(midline_deviation_mm=2.5)
        occlusion_service._assess_constraint_satisfaction(metrics)
        assert not metrics.constraints_satisfied
        assert any("Midline" in v or "midline" in v for v in metrics.constraint_violations)

    def test_excessive_cant_produces_violation(self, occlusion_service):
        """Cant outside [-2, 2] degrees should produce a violation."""
        metrics = OcclusalMetrics(cant_degrees=4.0)
        occlusion_service._assess_constraint_satisfaction(metrics)
        assert not metrics.constraints_satisfied
        assert any("cant" in v.lower() or "Cant" in v for v in metrics.constraint_violations)

    def test_multiple_violations_accumulated(self, occlusion_service):
        """Multiple out-of-range values produce multiple violations."""
        metrics = OcclusalMetrics(
            overjet_mm=8.0,
            overbite_mm=0.0,
            midline_deviation_mm=3.0,
            cant_degrees=5.0,
        )
        occlusion_service._assess_constraint_satisfaction(metrics)
        assert not metrics.constraints_satisfied
        assert len(metrics.constraint_violations) >= 4

    def test_none_metrics_are_skipped(self, occlusion_service):
        """None metric values should not produce violations."""
        metrics = OcclusalMetrics()  # All metrics are None by default
        occlusion_service._assess_constraint_satisfaction(metrics)
        assert metrics.constraints_satisfied is True
        assert metrics.constraint_violations == []

    def test_boundary_values_are_accepted(self, occlusion_service):
        """Exact boundary values (inclusive) should not produce violations."""
        lo_overjet, hi_overjet = NORMAL_OCCLUSAL_RANGES["overjet_mm"]
        metrics = OcclusalMetrics(overjet_mm=lo_overjet)
        occlusion_service._assess_constraint_satisfaction(metrics)
        assert metrics.constraints_satisfied is True

        metrics2 = OcclusalMetrics(overjet_mm=hi_overjet)
        occlusion_service._assess_constraint_satisfaction(metrics2)
        assert metrics2.constraints_satisfied is True


# ─── OcclusionService.evaluate_occlusion ──────────────────────────────────────


class TestEvaluateOcclusion:
    @pytest.mark.asyncio
    async def test_raises_dental_arch_error_when_upper_arch_none(self, occlusion_service, mock_lower_arch):
        """evaluate_occlusion raises DentalArchError when upper_arch is None."""
        with pytest.raises(DentalArchError):
            await occlusion_service.evaluate_occlusion(None, mock_lower_arch)

    @pytest.mark.asyncio
    async def test_raises_dental_arch_error_when_lower_arch_none(self, occlusion_service, mock_upper_arch):
        """evaluate_occlusion raises DentalArchError when lower_arch is None."""
        with pytest.raises(DentalArchError):
            await occlusion_service.evaluate_occlusion(mock_upper_arch, None)

    @pytest.mark.asyncio
    async def test_raises_dental_arch_error_when_both_none(self, occlusion_service):
        """evaluate_occlusion raises DentalArchError when both arches are None."""
        with pytest.raises(DentalArchError):
            await occlusion_service.evaluate_occlusion(None, None)

    @pytest.mark.asyncio
    async def test_returns_occlusal_metrics_with_valid_arches(
        self, occlusion_service, mock_upper_arch, mock_lower_arch
    ):
        """evaluate_occlusion returns OcclusalMetrics when both arches are provided."""
        metrics = await occlusion_service.evaluate_occlusion(
            mock_upper_arch, mock_lower_arch
        )
        assert isinstance(metrics, OcclusalMetrics)

    @pytest.mark.asyncio
    async def test_applies_planned_transforms_for_known_fragment_ids(
        self, occlusion_service, mock_upper_arch, mock_lower_arch
    ):
        """evaluate_occlusion applies transforms from planned_transforms dict."""
        planned_transforms = {
            "upper_frag": np.eye(4),
            "lower_frag": np.eye(4),
        }
        metrics = await occlusion_service.evaluate_occlusion(
            mock_upper_arch,
            mock_lower_arch,
            planned_transforms=planned_transforms,
            upper_fragment_id="upper_frag",
            lower_fragment_id="lower_frag",
        )
        assert isinstance(metrics, OcclusalMetrics)

    @pytest.mark.asyncio
    async def test_wraps_model_exception_in_occlusion_metric_error(
        self, occlusion_service, mock_upper_arch, mock_lower_arch
    ):
        """evaluate_occlusion wraps unexpected model errors in OcclusionMetricError."""
        occlusion_service._geometric = MagicMock()
        occlusion_service._geometric.evaluate.side_effect = RuntimeError("Model crashed")
        with pytest.raises(OcclusionMetricError):
            await occlusion_service.evaluate_occlusion(mock_upper_arch, mock_lower_arch)

    @pytest.mark.asyncio
    async def test_assess_constraints_is_called_after_evaluation(
        self, occlusion_service, mock_upper_arch, mock_lower_arch
    ):
        """_assess_constraint_satisfaction is invoked on the returned metrics."""
        called = []
        original = occlusion_service._assess_constraint_satisfaction

        def spy(metrics):
            called.append(metrics)
            return original(metrics)

        occlusion_service._assess_constraint_satisfaction = spy
        await occlusion_service.evaluate_occlusion(mock_upper_arch, mock_lower_arch)
        assert len(called) == 1


# ─── OcclusionService.compute_dental_constraints ──────────────────────────────


class TestComputeDentalConstraints:
    @pytest.mark.asyncio
    async def test_returns_defaults_when_no_pre_injury_data(self, occlusion_service):
        """compute_dental_constraints returns clinical defaults when pre_injury_occlusion is None."""
        constraints = await occlusion_service.compute_dental_constraints(
            pre_injury_occlusion=None,
            current_fragments=[],
        )
        assert isinstance(constraints, OcclusalConstraints)
        assert constraints.target_overjet_mm == 2.0
        assert constraints.target_overbite_mm == 3.0
        assert constraints.molar_class_target == "Class_I"

    @pytest.mark.asyncio
    async def test_default_midline_tolerance(self, occlusion_service):
        """Default midline tolerance is 1mm."""
        constraints = await occlusion_service.compute_dental_constraints(
            pre_injury_occlusion=None,
            current_fragments=[],
        )
        assert constraints.midline_tolerance_mm == 1.0

    @pytest.mark.asyncio
    async def test_default_cant_tolerance(self, occlusion_service):
        """Default cant tolerance is 2 degrees."""
        constraints = await occlusion_service.compute_dental_constraints(
            pre_injury_occlusion=None,
            current_fragments=[],
        )
        assert constraints.cant_tolerance_degrees == 2.0

    @pytest.mark.asyncio
    async def test_returns_occlusal_constraints_instance(self, occlusion_service):
        """Return type is always OcclusalConstraints."""
        mock_pre_injury = MagicMock()
        constraints = await occlusion_service.compute_dental_constraints(
            pre_injury_occlusion=mock_pre_injury,
            current_fragments=[],
        )
        assert isinstance(constraints, OcclusalConstraints)

    @pytest.mark.asyncio
    async def test_pre_injury_flag_set_when_reference_available(self, occlusion_service):
        """use_pre_injury_occlusion flag is True when pre-injury reference given."""
        mock_pre_injury = MagicMock()
        constraints = await occlusion_service.compute_dental_constraints(
            pre_injury_occlusion=mock_pre_injury,
            current_fragments=[],
        )
        assert constraints.use_pre_injury_occlusion is True

    @pytest.mark.asyncio
    async def test_no_pre_injury_flag_is_still_valid(self, occlusion_service):
        """Returned OcclusalConstraints passes Pydantic validation even when no pre-injury data."""
        constraints = await occlusion_service.compute_dental_constraints(
            pre_injury_occlusion=None,
            current_fragments=[MagicMock()],
        )
        # Pydantic model must be valid — this will raise if not
        assert constraints.model_dump() is not None


# ─── OcclusionService.assess_molar_relationship ───────────────────────────────


class TestAssessMolarRelationship:
    """Tests for Angle molar classification logic."""

    @pytest.fixture
    def service(self) -> OcclusionService:
        return OcclusionService()

    @pytest.mark.parametrize("ap_offset,expected", [
        (0.0, "Class_I"),     # Exactly aligned
        (1.5, "Class_I"),     # Within ±2mm window
        (-1.5, "Class_I"),    # Within ±2mm window
        (2.0, "Class_I"),     # Inclusive boundary
        (-2.0, "Class_I"),    # Inclusive boundary
    ])
    def test_class_i_classification(self, service, ap_offset, expected):
        """Lower M1 within ±2mm of upper M1 in AP direction → Class I."""
        upper = np.array([0.0, 0.0, 0.0])
        lower = np.array([ap_offset, 0.0, 0.0])
        result = service.assess_molar_relationship(upper, lower)
        assert result == expected

    @pytest.mark.parametrize("ap_offset", [2.5, 5.0, 10.0])
    def test_class_iii_when_lower_is_anterior(self, service, ap_offset):
        """Lower M1 more than 2mm anterior to upper M1 → Class III."""
        upper = np.array([0.0, 0.0, 0.0])
        lower = np.array([ap_offset, 0.0, 0.0])
        result = service.assess_molar_relationship(upper, lower)
        assert result == "Class_III"

    @pytest.mark.parametrize("ap_offset", [-2.5, -5.0, -10.0])
    def test_class_ii_when_lower_is_posterior(self, service, ap_offset):
        """Lower M1 more than 2mm posterior to upper M1 → Class II."""
        upper = np.array([0.0, 0.0, 0.0])
        lower = np.array([ap_offset, 0.0, 0.0])
        result = service.assess_molar_relationship(upper, lower)
        assert result in {"Class_II_div1", "Class_II_div2"}

    def test_only_ap_axis_matters(self, service):
        """Lateral and vertical offset should not affect molar class."""
        upper = np.array([0.0, 0.0, 0.0])
        lower = np.array([1.0, 100.0, -50.0])  # Large lateral/vertical offsets, small AP
        result = service.assess_molar_relationship(upper, lower)
        assert result == "Class_I"

    def test_result_is_string(self, service):
        """assess_molar_relationship always returns a string."""
        upper = np.array([0.0, 0.0, 0.0])
        lower = np.array([3.0, 0.0, 0.0])
        result = service.assess_molar_relationship(upper, lower)
        assert isinstance(result, str)

    def test_symmetric_input_gives_class_i(self, service):
        """Perfectly symmetric bilateral molars → Class I."""
        upper_R = np.array([30.0, -50.0, 0.0])
        lower_R = np.array([30.5, -50.0, 0.0])  # 0.5mm AP offset
        result = service.assess_molar_relationship(upper_R, lower_R)
        assert result == "Class_I"


# ─── OcclusionService.compute_arch_geometry ───────────────────────────────────


class TestComputeArchGeometry:
    def test_raises_dental_arch_error_on_empty_mesh(
        self, occlusion_service, empty_mesh
    ):
        """compute_arch_geometry raises DentalArchError when mesh has no vertices."""
        with pytest.raises(DentalArchError, match="Empty arch mesh"):
            occlusion_service.compute_arch_geometry(empty_mesh, is_upper=True)

    def test_raises_dental_arch_error_on_mesh_without_vertices_attribute(
        self, occlusion_service
    ):
        """compute_arch_geometry raises DentalArchError when mesh lacks vertices attr."""
        mesh = object()  # Plain object with no .vertices
        with pytest.raises((DentalArchError, AttributeError)):
            occlusion_service.compute_arch_geometry(mesh, is_upper=True)

    def test_returns_arch_geometry_for_valid_upper_mesh(
        self, occlusion_service, mock_upper_arch
    ):
        """compute_arch_geometry returns ArchGeometry for a valid upper arch mesh."""
        result = occlusion_service.compute_arch_geometry(mock_upper_arch, is_upper=True)
        assert isinstance(result, ArchGeometry)
        assert result.is_upper is True

    def test_returns_arch_geometry_for_valid_lower_mesh(
        self, occlusion_service, mock_lower_arch
    ):
        """compute_arch_geometry returns ArchGeometry for a valid lower arch mesh."""
        result = occlusion_service.compute_arch_geometry(mock_lower_arch, is_upper=False)
        assert isinstance(result, ArchGeometry)
        assert result.is_upper is False

    def test_centroid_shape(self, occlusion_service, mock_upper_arch):
        """centroid_mm should be a 3-element array."""
        result = occlusion_service.compute_arch_geometry(mock_upper_arch, is_upper=True)
        assert result.centroid_mm.shape == (3,)

    def test_midline_vector_is_unit_vector(self, occlusion_service, mock_upper_arch):
        """midline_vector should be approximately a unit vector."""
        result = occlusion_service.compute_arch_geometry(mock_upper_arch, is_upper=True)
        norm = np.linalg.norm(result.midline_vector)
        assert abs(norm - 1.0) < 1e-6

    def test_arch_width_is_positive(self, occlusion_service, mock_upper_arch):
        """arch_width_mm should be a positive value."""
        result = occlusion_service.compute_arch_geometry(mock_upper_arch, is_upper=True)
        assert result.arch_width_mm > 0

    def test_arch_length_is_positive(self, occlusion_service, mock_upper_arch):
        """arch_length_mm should be a positive value."""
        result = occlusion_service.compute_arch_geometry(mock_upper_arch, is_upper=True)
        assert result.arch_length_mm > 0

    def test_curve_of_spee_is_capped_at_10mm(self, occlusion_service):
        """curve_of_spee_depth_mm should not exceed 10mm."""
        # Create mesh with extreme z variation
        mesh = MagicMock()
        x = np.linspace(-30, 30, 100)
        y = np.zeros(100)
        z = np.linspace(-50, 50, 100)  # 100mm z range
        mesh.vertices = np.stack([x, y, z], axis=1)
        result = occlusion_service.compute_arch_geometry(mesh, is_upper=True)
        assert result.curve_of_spee_depth_mm <= 10.0

    def test_arch_points_stored_as_numpy_array(self, occlusion_service, mock_upper_arch):
        """arch_points in result should be a numpy array."""
        result = occlusion_service.compute_arch_geometry(mock_upper_arch, is_upper=True)
        assert isinstance(result.arch_points, np.ndarray)


# ─── SplintDesignSpec tests ───────────────────────────────────────────────────


class TestSplintDesignSpec:
    def test_default_values(self):
        """SplintDesignSpec defaults should be sensible."""
        spec = SplintDesignSpec()
        assert spec.upper_arch_path is None
        assert spec.lower_arch_path is None
        assert spec.target_vertical_dimension_mm == 0.0
        assert spec.contact_regions == []
        assert spec.material_recommendation == "acrylic_resin"
        assert spec.notes == ""

    def test_custom_values(self):
        """SplintDesignSpec accepts custom values."""
        spec = SplintDesignSpec(
            upper_arch_path="/output/upper.stl",
            lower_arch_path="/output/lower.stl",
            target_vertical_dimension_mm=3.5,
            material_recommendation="clear_thermoplastic",
            notes="Custom splint notes.",
        )
        assert spec.upper_arch_path == "/output/upper.stl"
        assert spec.lower_arch_path == "/output/lower.stl"
        assert spec.target_vertical_dimension_mm == 3.5

    @pytest.mark.asyncio
    async def test_suggest_splint_design_returns_spec(
        self, occlusion_service, mock_upper_arch, mock_lower_arch
    ):
        """suggest_splint_design returns a SplintDesignSpec."""
        occlusal_plan = OcclusalMetrics(constraints_satisfied=True)
        spec = await occlusion_service.suggest_splint_design(occlusal_plan)
        assert isinstance(spec, SplintDesignSpec)

    @pytest.mark.asyncio
    async def test_suggest_splint_small_vd_uses_thermoplastic(self, occlusion_service):
        """When vertical dimension < 5mm, material is clear thermoplastic."""
        plan = OcclusalMetrics(constraints_satisfied=True)
        spec = await occlusion_service.suggest_splint_design(plan)
        # Default VD is 3.0mm → should use clear_thermoplastic
        assert spec.material_recommendation == "clear_thermoplastic"

    @pytest.mark.asyncio
    async def test_suggest_splint_notes_not_empty(self, occlusion_service):
        """suggest_splint_design always produces non-empty notes."""
        plan = OcclusalMetrics()
        spec = await occlusion_service.suggest_splint_design(plan)
        assert len(spec.notes) > 0


# ─── OcclusalContact dataclass ────────────────────────────────────────────────


class TestOcclusalContact:
    def test_fields_are_stored(self):
        """OcclusalContact stores all fields correctly."""
        contact = OcclusalContact(
            upper_fdi=16,
            lower_fdi=46,
            contact_force_relative=0.8,
            location_mm=[10.0, -40.0, 5.0],
            contact_area_mm2=2.5,
        )
        assert contact.upper_fdi == 16
        assert contact.lower_fdi == 46
        assert contact.contact_force_relative == 0.8
        assert contact.contact_area_mm2 == 2.5

    def test_contact_force_range(self):
        """contact_force_relative should typically be in [0, 1]."""
        contact = OcclusalContact(
            upper_fdi=26,
            lower_fdi=36,
            contact_force_relative=0.0,
            location_mm=[0.0, 0.0, 0.0],
            contact_area_mm2=0.0,
        )
        assert 0.0 <= contact.contact_force_relative <= 1.0


# ─── OcclusionService._get_model ──────────────────────────────────────────────


class TestGetModel:
    def test_defaults_to_geometric_model(self):
        """With no learned model, _get_model returns GeometricOcclusionModel."""
        service = OcclusionService(use_learned_model=False)
        model = service._get_model()
        assert isinstance(model, GeometricOcclusionModel)

    def test_falls_back_to_geometric_when_learned_unavailable(self):
        """With unavailable learned model, _get_model returns GeometricOcclusionModel."""
        service = OcclusionService(use_learned_model=True)
        # LearnedOcclusionModel.is_available is always False currently
        model = service._get_model()
        assert isinstance(model, GeometricOcclusionModel)
