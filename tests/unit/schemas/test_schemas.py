"""
Unit tests for all Pydantic schemas in apps/backend/app/schemas/.

Tests cover:
- Transform3D.identity() produces correct values
- Transform3D validation rejects non-orthonormal rotation matrices
- Transform3D.to_4x4_matrix() correctness
- Vector3D.from_list and to_list round-trip
- BoundingBox3D.dimensions, center, volume_mm3 computed correctly
- PaginatedResponse.create computes pages correctly
- OcclusalConstraints.validate_molar_class rejects invalid classes
- FragmentTransform confidence bounds [0, 1]
- ReductionPlanRequest defaults
- ValidationResult fields
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from typing import List

import numpy as np
import pytest
from pydantic import ValidationError

from app.schemas.common import (
    BaseSchema,
    BoundingBox3D,
    ComponentHealth,
    ErrorResponse,
    HealthCheck,
    JobStatus,
    PaginatedResponse,
    PaginationParams,
    Transform3D,
    Vector3D,
)
from app.schemas.plan import (
    FragmentInfo,
    FragmentTransform,
    OcclusalConstraints,
    OcclusalMetrics,
    ReductionPlanRequest,
    SurgeonEditRequest,
    ValidationResult,
)


# ─── Vector3D tests ───────────────────────────────────────────────────────────


class TestVector3D:
    def test_stores_components(self):
        v = Vector3D(x=1.0, y=2.0, z=3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_to_list_returns_list(self):
        v = Vector3D(x=4.0, y=5.0, z=6.0)
        result = v.to_list()
        assert result == [4.0, 5.0, 6.0]
        assert isinstance(result, list)

    def test_from_list_constructs_vector(self):
        v = Vector3D.from_list([7.0, 8.0, 9.0])
        assert v.x == 7.0
        assert v.y == 8.0
        assert v.z == 9.0

    def test_from_list_to_list_round_trip(self):
        original = [10.5, -3.14, 0.001]
        v = Vector3D.from_list(original)
        result = v.to_list()
        assert result == pytest.approx(original)

    def test_from_list_to_list_with_zeros(self):
        v = Vector3D.from_list([0.0, 0.0, 0.0])
        assert v.to_list() == [0.0, 0.0, 0.0]

    def test_from_list_to_list_with_negative_values(self):
        original = [-1.0, -2.5, -100.0]
        assert Vector3D.from_list(original).to_list() == pytest.approx(original)

    @pytest.mark.parametrize("values", [
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
        [-5.0, 10.5, 0.001],
    ])
    def test_from_list_round_trip_parametrized(self, values):
        assert Vector3D.from_list(values).to_list() == pytest.approx(values)

    def test_model_dump_roundtrip(self):
        v = Vector3D(x=1.0, y=2.0, z=3.0)
        dumped = v.model_dump()
        restored = Vector3D(**dumped)
        assert restored == v


# ─── Transform3D tests ────────────────────────────────────────────────────────


class TestTransform3DIdentity:
    def test_identity_returns_transform3d(self):
        T = Transform3D.identity()
        assert isinstance(T, Transform3D)

    def test_identity_rotation_is_eye(self):
        T = Transform3D.identity()
        R = np.array(T.rotation_matrix)
        assert np.allclose(R, np.eye(3), atol=1e-9)

    def test_identity_translation_is_zero(self):
        T = Transform3D.identity()
        assert T.translation_mm == pytest.approx([0.0, 0.0, 0.0])

    def test_identity_is_idempotent(self):
        T1 = Transform3D.identity()
        T2 = Transform3D.identity()
        assert T1 == T2


class TestTransform3DValidation:
    def test_valid_identity_matrix_accepted(self):
        T = Transform3D(
            rotation_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            translation_mm=[0.0, 0.0, 0.0],
        )
        assert T is not None

    def test_valid_rotation_accepted(self):
        angle = math.radians(45)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        T = Transform3D(
            rotation_matrix=[
                [cos_a, -sin_a, 0.0],
                [sin_a,  cos_a, 0.0],
                [0.0,    0.0,   1.0],
            ],
            translation_mm=[10.0, 0.0, 0.0],
        )
        assert T is not None

    def test_rejects_non_orthonormal_rotation(self):
        """A matrix with random values that is not orthonormal should fail validation."""
        with pytest.raises(ValidationError):
            Transform3D(
                rotation_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
                translation_mm=[0.0, 0.0, 0.0],
            )

    def test_rejects_shear_matrix(self):
        """A shear matrix is not a rotation matrix and should fail validation."""
        with pytest.raises(ValidationError):
            Transform3D(
                rotation_matrix=[[1, 1, 0], [0, 1, 0], [0, 0, 1]],
                translation_mm=[0.0, 0.0, 0.0],
            )

    def test_rejects_wrong_rotation_shape_2x3(self):
        with pytest.raises(ValidationError):
            Transform3D(
                rotation_matrix=[[1, 0, 0], [0, 1, 0]],  # 2x3 not 3x3
                translation_mm=[0.0, 0.0, 0.0],
            )

    def test_rejects_translation_with_wrong_length(self):
        with pytest.raises(ValidationError):
            Transform3D(
                rotation_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                translation_mm=[0.0, 0.0],  # Only 2 components
            )

    def test_rejects_translation_with_four_components(self):
        with pytest.raises(ValidationError):
            Transform3D(
                rotation_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                translation_mm=[0.0, 0.0, 0.0, 0.0],
            )

    def test_accepts_near_orthonormal_rotation(self):
        """Slightly imprecise rotation matrix within tolerance should be accepted."""
        eps = 1e-5
        T = Transform3D(
            rotation_matrix=[
                [1.0, eps, 0.0],
                [-eps, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            translation_mm=[0.0, 0.0, 0.0],
        )
        assert T is not None


class TestTransform3DTo4x4Matrix:
    def test_identity_gives_eye4(self):
        T = Transform3D.identity()
        mat = np.array(T.to_4x4_matrix())
        assert np.allclose(mat, np.eye(4), atol=1e-9)

    def test_translation_in_last_column(self):
        T = Transform3D(
            rotation_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            translation_mm=[3.0, -1.0, 7.5],
        )
        mat = np.array(T.to_4x4_matrix())
        assert mat[0, 3] == pytest.approx(3.0)
        assert mat[1, 3] == pytest.approx(-1.0)
        assert mat[2, 3] == pytest.approx(7.5)

    def test_bottom_row_is_0001(self):
        T = Transform3D(
            rotation_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            translation_mm=[1.0, 2.0, 3.0],
        )
        mat = np.array(T.to_4x4_matrix())
        assert np.allclose(mat[3, :], [0, 0, 0, 1])

    def test_rotation_block_correct(self):
        angle = math.radians(30)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        R = [[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]]
        T = Transform3D(rotation_matrix=R, translation_mm=[0.0, 0.0, 0.0])
        mat = np.array(T.to_4x4_matrix())
        assert np.allclose(mat[:3, :3], R, atol=1e-9)

    def test_output_is_list_of_lists(self):
        T = Transform3D.identity()
        mat = T.to_4x4_matrix()
        assert isinstance(mat, list)
        assert len(mat) == 4
        assert all(isinstance(row, list) for row in mat)

    def test_matrix_shape_is_4x4(self):
        T = Transform3D.identity()
        mat = T.to_4x4_matrix()
        assert len(mat) == 4
        assert all(len(row) == 4 for row in mat)


# ─── BoundingBox3D tests ──────────────────────────────────────────────────────


class TestBoundingBox3D:
    @pytest.fixture
    def unit_box(self) -> BoundingBox3D:
        return BoundingBox3D(
            min_x=0, min_y=0, min_z=0,
            max_x=1, max_y=1, max_z=1,
        )

    @pytest.fixture
    def asymmetric_box(self) -> BoundingBox3D:
        return BoundingBox3D(
            min_x=-5, min_y=10, min_z=0,
            max_x=5, max_y=20, max_z=3,
        )

    def test_dimensions_unit_box(self, unit_box):
        d = unit_box.dimensions
        assert d.x == pytest.approx(1.0)
        assert d.y == pytest.approx(1.0)
        assert d.z == pytest.approx(1.0)

    def test_dimensions_asymmetric_box(self, asymmetric_box):
        d = asymmetric_box.dimensions
        assert d.x == pytest.approx(10.0)
        assert d.y == pytest.approx(10.0)
        assert d.z == pytest.approx(3.0)

    def test_dimensions_is_vector3d(self, unit_box):
        assert isinstance(unit_box.dimensions, Vector3D)

    def test_center_unit_box(self, unit_box):
        c = unit_box.center
        assert c.x == pytest.approx(0.5)
        assert c.y == pytest.approx(0.5)
        assert c.z == pytest.approx(0.5)

    def test_center_asymmetric_box(self, asymmetric_box):
        c = asymmetric_box.center
        assert c.x == pytest.approx(0.0)
        assert c.y == pytest.approx(15.0)
        assert c.z == pytest.approx(1.5)

    def test_center_is_vector3d(self, unit_box):
        assert isinstance(unit_box.center, Vector3D)

    def test_volume_mm3_unit_box(self, unit_box):
        assert unit_box.volume_mm3 == pytest.approx(1.0)

    def test_volume_mm3_asymmetric_box(self, asymmetric_box):
        # 10 * 10 * 3 = 300
        assert asymmetric_box.volume_mm3 == pytest.approx(300.0)

    def test_volume_is_positive(self, unit_box):
        assert unit_box.volume_mm3 > 0

    def test_zero_extent_gives_zero_volume(self):
        box = BoundingBox3D(min_x=0, min_y=0, min_z=0, max_x=0, max_y=0, max_z=0)
        assert box.volume_mm3 == pytest.approx(0.0)

    @pytest.mark.parametrize("min_v,max_v,expected_dim", [
        (0.0, 10.0, 10.0),
        (-5.0, 5.0, 10.0),
        (100.0, 200.0, 100.0),
    ])
    def test_dimension_parametrized(self, min_v, max_v, expected_dim):
        box = BoundingBox3D(
            min_x=min_v, min_y=min_v, min_z=min_v,
            max_x=max_v, max_y=max_v, max_z=max_v,
        )
        assert box.dimensions.x == pytest.approx(expected_dim)


# ─── PaginatedResponse tests ──────────────────────────────────────────────────


class TestPaginatedResponse:
    def test_create_computes_pages_correctly(self):
        result = PaginatedResponse.create(
            items=list(range(20)), total=100, page=1, page_size=20
        )
        assert result.pages == 5

    def test_create_rounds_pages_up(self):
        result = PaginatedResponse.create(
            items=list(range(10)), total=101, page=1, page_size=20
        )
        assert result.pages == 6  # ceil(101/20) = 6

    def test_create_with_total_zero_gives_one_page(self):
        result = PaginatedResponse.create(
            items=[], total=0, page=1, page_size=10
        )
        assert result.pages == 1

    def test_create_stores_page_and_page_size(self):
        result = PaginatedResponse.create(
            items=[1, 2, 3], total=3, page=2, page_size=5
        )
        assert result.page == 2
        assert result.page_size == 5

    def test_create_stores_total(self):
        result = PaginatedResponse.create(
            items=[], total=999, page=1, page_size=50
        )
        assert result.total == 999

    def test_single_item_per_page(self):
        result = PaginatedResponse.create(
            items=[42], total=42, page=1, page_size=1
        )
        assert result.pages == 42

    @pytest.mark.parametrize("total,page_size,expected_pages", [
        (0, 10, 1),
        (10, 10, 1),
        (11, 10, 2),
        (100, 10, 10),
        (101, 10, 11),
        (1, 100, 1),
    ])
    def test_pages_computation_parametrized(self, total, page_size, expected_pages):
        result = PaginatedResponse.create(
            items=[], total=total, page=1, page_size=page_size
        )
        assert result.pages == expected_pages


class TestPaginationParams:
    def test_default_page_is_1(self):
        params = PaginationParams()
        assert params.page == 1

    def test_default_page_size_is_20(self):
        params = PaginationParams()
        assert params.page_size == 20

    def test_offset_calculation(self):
        params = PaginationParams(page=3, page_size=20)
        assert params.offset == 40  # (3-1) * 20

    def test_first_page_offset_is_zero(self):
        params = PaginationParams(page=1, page_size=10)
        assert params.offset == 0

    def test_page_must_be_at_least_1(self):
        with pytest.raises(ValidationError):
            PaginationParams(page=0)

    def test_page_size_max_100(self):
        with pytest.raises(ValidationError):
            PaginationParams(page_size=101)

    def test_page_size_min_1(self):
        with pytest.raises(ValidationError):
            PaginationParams(page_size=0)


# ─── OcclusalConstraints tests ────────────────────────────────────────────────


class TestOcclusalConstraints:
    def test_valid_class_i_accepted(self):
        c = OcclusalConstraints(molar_class_target="Class_I")
        assert c.molar_class_target == "Class_I"

    def test_valid_class_ii_div1_accepted(self):
        c = OcclusalConstraints(molar_class_target="Class_II_div1")
        assert c.molar_class_target == "Class_II_div1"

    def test_valid_class_ii_div2_accepted(self):
        c = OcclusalConstraints(molar_class_target="Class_II_div2")
        assert c.molar_class_target == "Class_II_div2"

    def test_valid_class_iii_accepted(self):
        c = OcclusalConstraints(molar_class_target="Class_III")
        assert c.molar_class_target == "Class_III"

    def test_invalid_molar_class_rejected(self):
        with pytest.raises(ValidationError):
            OcclusalConstraints(molar_class_target="Class_IV")

    def test_invalid_molar_class_lowercase_rejected(self):
        with pytest.raises(ValidationError):
            OcclusalConstraints(molar_class_target="class_i")

    def test_invalid_molar_class_number_rejected(self):
        with pytest.raises(ValidationError):
            OcclusalConstraints(molar_class_target="I")

    def test_default_values(self):
        c = OcclusalConstraints()
        assert c.target_overjet_mm == 2.0
        assert c.target_overbite_mm == 3.0
        assert c.molar_class_target == "Class_I"
        assert c.midline_tolerance_mm == 1.0
        assert c.cant_tolerance_degrees == 2.0

    def test_midline_tolerance_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            OcclusalConstraints(midline_tolerance_mm=-0.1)

    def test_cant_tolerance_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            OcclusalConstraints(cant_tolerance_degrees=-1.0)


# ─── FragmentTransform tests ──────────────────────────────────────────────────


class TestFragmentTransform:
    def test_confidence_at_lower_bound(self):
        ft = FragmentTransform(
            fragment_id="frag_1",
            fragment_label=1,
            transform=Transform3D.identity(),
            confidence=0.0,
        )
        assert ft.confidence == 0.0

    def test_confidence_at_upper_bound(self):
        ft = FragmentTransform(
            fragment_id="frag_1",
            fragment_label=1,
            transform=Transform3D.identity(),
            confidence=1.0,
        )
        assert ft.confidence == 1.0

    def test_confidence_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            FragmentTransform(
                fragment_id="frag_1",
                fragment_label=1,
                transform=Transform3D.identity(),
                confidence=-0.1,
            )

    def test_confidence_above_one_rejected(self):
        with pytest.raises(ValidationError):
            FragmentTransform(
                fragment_id="frag_1",
                fragment_label=1,
                transform=Transform3D.identity(),
                confidence=1.01,
            )

    def test_is_reference_fragment_default_false(self):
        ft = FragmentTransform(
            fragment_id="frag_2",
            fragment_label=2,
            transform=Transform3D.identity(),
            confidence=0.8,
        )
        assert ft.is_reference_fragment is False

    def test_stores_fragment_id_and_label(self):
        ft = FragmentTransform(
            fragment_id="mandible_body",
            fragment_label=5,
            transform=Transform3D.identity(),
            confidence=0.92,
        )
        assert ft.fragment_id == "mandible_body"
        assert ft.fragment_label == 5

    def test_transform_field_is_transform3d(self):
        ft = FragmentTransform(
            fragment_id="frag_3",
            fragment_label=3,
            transform=Transform3D.identity(),
            confidence=0.5,
        )
        assert isinstance(ft.transform, Transform3D)


# ─── ReductionPlanRequest tests ───────────────────────────────────────────────


class TestReductionPlanRequest:
    def test_default_model_is_baseline_icp(self):
        req = ReductionPlanRequest(
            case_id=uuid.uuid4(),
            segmentation_id=uuid.uuid4(),
        )
        assert req.model_name == "baseline_icp"

    def test_default_use_intact_reference_is_true(self):
        req = ReductionPlanRequest(
            case_id=uuid.uuid4(),
            segmentation_id=uuid.uuid4(),
        )
        assert req.use_intact_reference is True

    def test_default_include_alternatives_is_false(self):
        req = ReductionPlanRequest(
            case_id=uuid.uuid4(),
            segmentation_id=uuid.uuid4(),
        )
        assert req.include_alternative_plans is False

    def test_default_max_fragments_is_none(self):
        req = ReductionPlanRequest(
            case_id=uuid.uuid4(),
            segmentation_id=uuid.uuid4(),
        )
        assert req.max_fragments is None

    def test_default_occlusal_constraints_is_none(self):
        req = ReductionPlanRequest(
            case_id=uuid.uuid4(),
            segmentation_id=uuid.uuid4(),
        )
        assert req.occlusal_constraints is None

    def test_case_id_must_be_uuid(self):
        with pytest.raises(ValidationError):
            ReductionPlanRequest(
                case_id="not-a-uuid",
                segmentation_id=uuid.uuid4(),
            )

    def test_custom_model_name_accepted(self):
        req = ReductionPlanRequest(
            case_id=uuid.uuid4(),
            segmentation_id=uuid.uuid4(),
            model_name="learned_v1",
        )
        assert req.model_name == "learned_v1"

    def test_with_occlusal_constraints(self):
        constraints = OcclusalConstraints()
        req = ReductionPlanRequest(
            case_id=uuid.uuid4(),
            segmentation_id=uuid.uuid4(),
            occlusal_constraints=constraints,
        )
        assert req.occlusal_constraints is not None
        assert isinstance(req.occlusal_constraints, OcclusalConstraints)


# ─── ValidationResult tests ───────────────────────────────────────────────────


class TestValidationResult:
    def test_passed_field(self):
        vr = ValidationResult(
            passed=True,
            symmetry_ok=True,
            occlusion_ok=True,
            condylar_seating_ok=True,
            hardware_placement_ok=True,
        )
        assert vr.passed is True

    def test_failed_validation(self):
        vr = ValidationResult(
            passed=False,
            symmetry_ok=False,
            occlusion_ok=True,
            condylar_seating_ok=True,
            hardware_placement_ok=True,
        )
        assert vr.passed is False

    def test_default_warnings_and_errors_empty(self):
        vr = ValidationResult(
            passed=True,
            symmetry_ok=True,
            occlusion_ok=True,
            condylar_seating_ok=True,
            hardware_placement_ok=True,
        )
        assert vr.warnings == []
        assert vr.errors == []

    def test_stores_warnings(self):
        vr = ValidationResult(
            passed=True,
            symmetry_ok=True,
            occlusion_ok=True,
            condylar_seating_ok=True,
            hardware_placement_ok=True,
            warnings=["Minor asymmetry detected"],
        )
        assert "Minor asymmetry detected" in vr.warnings

    def test_stores_errors(self):
        vr = ValidationResult(
            passed=False,
            symmetry_ok=False,
            occlusion_ok=False,
            condylar_seating_ok=False,
            hardware_placement_ok=False,
            errors=["Severe occlusal discrepancy"],
        )
        assert "Severe occlusal discrepancy" in vr.errors

    def test_symmetry_score_optional(self):
        vr = ValidationResult(
            passed=True,
            symmetry_ok=True,
            occlusion_ok=True,
            condylar_seating_ok=True,
            hardware_placement_ok=True,
        )
        assert vr.skeletal_symmetry_score is None

    def test_symmetry_score_bounds(self):
        vr = ValidationResult(
            passed=True,
            symmetry_ok=True,
            occlusion_ok=True,
            condylar_seating_ok=True,
            hardware_placement_ok=True,
            skeletal_symmetry_score=0.87,
        )
        assert 0.0 <= vr.skeletal_symmetry_score <= 1.0

    def test_symmetry_score_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            ValidationResult(
                passed=True,
                symmetry_ok=True,
                occlusion_ok=True,
                condylar_seating_ok=True,
                hardware_placement_ok=True,
                skeletal_symmetry_score=-0.1,
            )

    def test_symmetry_score_above_one_rejected(self):
        with pytest.raises(ValidationError):
            ValidationResult(
                passed=True,
                symmetry_ok=True,
                occlusion_ok=True,
                condylar_seating_ok=True,
                hardware_placement_ok=True,
                skeletal_symmetry_score=1.1,
            )


# ─── OcclusalMetrics tests ────────────────────────────────────────────────────


class TestOcclusalMetrics:
    def test_default_all_none(self):
        m = OcclusalMetrics()
        assert m.overjet_mm is None
        assert m.overbite_mm is None
        assert m.molar_relationship is None
        assert m.midline_deviation_mm is None
        assert m.cant_degrees is None

    def test_constraints_satisfied_default_false(self):
        m = OcclusalMetrics()
        assert m.constraints_satisfied is False

    def test_constraint_violations_default_empty(self):
        m = OcclusalMetrics()
        assert m.constraint_violations == []

    def test_stores_all_metrics(self):
        m = OcclusalMetrics(
            overjet_mm=2.0,
            overbite_mm=3.0,
            molar_relationship="Class_I",
            midline_deviation_mm=0.5,
            cant_degrees=1.0,
            constraints_satisfied=True,
        )
        assert m.overjet_mm == 2.0
        assert m.overbite_mm == 3.0
        assert m.molar_relationship == "Class_I"
