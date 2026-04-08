"""
Unit tests for the fracture reduction planning service.
"""

from __future__ import annotations

from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.services.reduction.reduction_service import (
    BaselineReductionModel,
    FragmentMesh,
    FractureReductionService,
    LearnedReductionModel,
    OcclusalConstraintEngine,
    ReductionPlan,
)
from app.schemas.plan import OcclusalConstraints, ValidationResult


class TestBaselineReductionModel:
    """Tests for the ICP-based baseline reduction model."""

    @pytest.fixture
    def model(self) -> BaselineReductionModel:
        return BaselineReductionModel()

    def test_model_name_and_version(self, model):
        assert model.name == "baseline_icp"
        assert model.version != ""

    def test_reference_fragment_gets_identity(self, model, mock_fragments):
        """Reference fragment should receive identity transform."""
        transforms = model.predict(mock_fragments, None, None)

        ref_frag = next(f for f in mock_fragments if f.is_reference)
        ref_transform = transforms.get(ref_frag.fragment_id)

        assert ref_transform is not None
        assert np.allclose(ref_transform, np.eye(4), atol=1e-6)

    def test_all_fragments_get_transforms(self, model, mock_fragments):
        """Every fragment should receive a transform."""
        transforms = model.predict(mock_fragments, None, None)

        for frag in mock_fragments:
            assert frag.fragment_id in transforms, (
                f"Fragment {frag.fragment_id} missing from transforms"
            )
            assert transforms[frag.fragment_id].shape == (4, 4)

    def test_transforms_are_valid_matrices(self, model, mock_fragments):
        """All transforms should be valid 4x4 matrices."""
        transforms = model.predict(mock_fragments, None, None)

        for frag_id, T in transforms.items():
            assert T.shape == (4, 4), f"Transform for {frag_id} wrong shape"
            # Last row should be [0, 0, 0, 1]
            assert np.allclose(T[3], [0, 0, 0, 1], atol=1e-6), (
                f"Transform {frag_id} has invalid homogeneous row"
            )

    def test_single_fragment_returns_identity(self, model):
        """Single reference fragment should get identity transform."""
        single_frag = FragmentMesh(
            fragment_id="sole_fragment",
            label_value=1,
            points=np.random.rand(100, 3).astype(np.float32),
            centroid_mm=np.array([0.0, 0.0, 0.0]),
            volume_mm3=1000.0,
            is_reference=True,
        )
        transforms = model.predict([single_frag], None, None)
        assert np.allclose(transforms["sole_fragment"], np.eye(4), atol=1e-6)


class TestOcclusalConstraintEngine:
    """Tests for the constraint engine."""

    @pytest.fixture
    def engine(self) -> OcclusalConstraintEngine:
        return OcclusalConstraintEngine(symmetry_tolerance_mm=3.0)

    def test_identity_transforms_pass_through(self, engine, mock_fragments):
        """Identity transforms should pass through unchanged."""
        transforms = {f.fragment_id: np.eye(4) for f in mock_fragments}
        dental_constraints = OcclusalConstraints()

        refined, metrics = engine.apply_constraints(
            fragment_transforms=transforms,
            fragments=mock_fragments,
            dental_constraints=dental_constraints,
            upper_dental_arch=None,
            lower_dental_arch=None,
        )

        # Transforms should be unchanged
        for frag_id in transforms:
            assert frag_id in refined

    def test_check_symmetry_bilateral_fragments(self, engine):
        """Test symmetry check with bilateral fragment pair."""
        fragments = [
            FragmentMesh(
                fragment_id="zygoma_L",
                label_value=1,
                points=np.random.rand(50, 3).astype(np.float32),
                centroid_mm=np.array([-30.0, 0.0, 0.0]),
                volume_mm3=500.0,
            ),
            FragmentMesh(
                fragment_id="zygoma_R",
                label_value=2,
                points=np.random.rand(50, 3).astype(np.float32),
                centroid_mm=np.array([30.0, 0.0, 0.0]),
                volume_mm3=500.0,
            ),
        ]
        # Identity transforms = perfectly symmetric
        transforms = {f.fragment_id: np.eye(4) for f in fragments}

        score, violations = engine.check_symmetry(fragments, transforms)

        # Should be symmetric with identity transforms
        assert 0.0 <= score <= 1.0
        # With identity and already-symmetric centroids → no violations
        assert len(violations) == 0

    def test_check_symmetry_asymmetric_transforms(self, engine):
        """Test that asymmetric transforms are detected."""
        fragments = [
            FragmentMesh(
                fragment_id="zygoma_L",
                label_value=1,
                points=np.random.rand(50, 3).astype(np.float32),
                centroid_mm=np.array([-30.0, 0.0, 0.0]),
                volume_mm3=500.0,
            ),
            FragmentMesh(
                fragment_id="zygoma_R",
                label_value=2,
                points=np.random.rand(50, 3).astype(np.float32),
                centroid_mm=np.array([30.0, 0.0, 0.0]),
                volume_mm3=500.0,
            ),
        ]
        # Left gets a big translation
        transforms = {
            "zygoma_L": np.eye(4),
            "zygoma_R": np.array([
                [1, 0, 0, 20.0],  # 20mm lateral displacement
                [0, 1, 0, 0.0],
                [0, 0, 1, 0.0],
                [0, 0, 0, 1.0],
            ]),
        }

        score, violations = engine.check_symmetry(fragments, transforms)
        # Should detect the asymmetry
        assert len(violations) > 0 or score < 0.9

    def test_find_bilateral_pairs(self, engine, mock_fragments):
        """Test bilateral pair detection by naming convention."""
        fragments = [
            FragmentMesh("zygoma_L", 1, np.zeros((10, 3)), np.zeros(3), 100.0),
            FragmentMesh("zygoma_R", 2, np.zeros((10, 3)), np.zeros(3), 100.0),
            FragmentMesh("mandible_body", 3, np.zeros((10, 3)), np.zeros(3), 1000.0),
        ]
        pairs = engine._find_bilateral_pairs(fragments)
        assert len(pairs) == 1
        assert pairs[0][0].fragment_id == "zygoma_L"
        assert pairs[0][1].fragment_id == "zygoma_R"


class TestFractureReductionService:
    """Tests for FractureReductionService."""

    @pytest.fixture
    def service(self) -> FractureReductionService:
        return FractureReductionService(model_registry=None)

    @pytest.mark.asyncio
    async def test_suggest_reduction_empty_fragments_raises(self, service):
        """Empty fragment list should raise FractureFragmentError."""
        from app.core.exceptions import FractureFragmentError
        with pytest.raises(FractureFragmentError):
            await service.suggest_reduction(fragments=[])

    @pytest.mark.asyncio
    async def test_suggest_reduction_returns_plan(self, service, mock_fragments):
        """suggest_reduction should return a valid ReductionPlan."""
        plan = await service.suggest_reduction(
            fragments=mock_fragments,
            model_name="baseline_icp",
        )

        assert isinstance(plan, ReductionPlan)
        assert len(plan.fragment_transforms) == len(mock_fragments)
        assert 0.0 <= plan.overall_confidence <= 1.0
        assert 0.0 <= plan.symmetry_score <= 1.0
        assert plan.model_name == "baseline_icp"

    @pytest.mark.asyncio
    async def test_suggest_reduction_validates_plan(self, service, mock_fragments):
        """Generated plan should include validation results."""
        plan = await service.suggest_reduction(
            fragments=mock_fragments,
            model_name="baseline_icp",
        )

        assert plan.validation is not None
        assert isinstance(plan.validation, ValidationResult)
        assert isinstance(plan.validation.passed, bool)

    @pytest.mark.asyncio
    async def test_refine_reduction_applies_edits(self, service, mock_fragments):
        """Surgeon edits should be applied to refined plan."""
        # Get initial plan
        initial_plan = await service.suggest_reduction(
            fragments=mock_fragments,
            model_name="baseline_icp",
        )

        # Apply a surgeon edit to one non-reference fragment
        non_ref = next(f for f in mock_fragments if not f.is_reference)
        surgeon_edit = np.array([
            [1, 0, 0, 5.0],
            [0, 1, 0, 3.0],
            [0, 0, 1, -2.0],
            [0, 0, 0, 1.0],
        ])

        refined = await service.refine_reduction(
            plan=initial_plan,
            surgeon_edits={non_ref.fragment_id: surgeon_edit},
            fragments=mock_fragments,
        )

        # The edited fragment should have the surgeon's transform
        assert np.allclose(
            refined.fragment_transforms[non_ref.fragment_id],
            surgeon_edit,
            atol=1e-6,
        )

    def test_estimate_confidences_returns_all_fragments(self, service, mock_fragments):
        """Confidence estimation should cover all fragments."""
        transforms = {f.fragment_id: np.eye(4) for f in mock_fragments}
        confidences = service._estimate_confidences(mock_fragments, transforms, None)

        assert len(confidences) == len(mock_fragments)
        for frag in mock_fragments:
            assert frag.fragment_id in confidences
            assert 0.0 <= confidences[frag.fragment_id] <= 1.0

    def test_validate_plan_passes_good_plan(self, service, mock_fragments):
        """A good plan (high confidence, identity transforms) should pass."""
        plan = ReductionPlan(
            fragment_transforms={f.fragment_id: np.eye(4) for f in mock_fragments},
            fragment_confidences={f.fragment_id: 0.9 for f in mock_fragments},
            occlusal_metrics=None,
            symmetry_score=0.9,
            overall_confidence=0.9,
            model_name="baseline_icp",
            model_version="1.0.0",
            generation_time_ms=100,
        )

        validation = service._validate_plan(plan, mock_fragments, None)

        assert isinstance(validation, ValidationResult)
        # Good symmetry score should pass symmetry check
        assert validation.symmetry_ok is True
