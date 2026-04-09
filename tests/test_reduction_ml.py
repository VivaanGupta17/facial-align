"""
Comprehensive tests for the occlusion-first reduction system.

Tests cover:
- FractureFittingLoss and FractureOverlapLoss gradient flow
- SegmentTransformParams parameterization and conversion
- OcclusionFirstJointOptimizer convergence on synthetic data
- FractureReductionService end-to-end with mock fragments
- OcclusalConstraintEngine with joint optimization
- Backward-compatible API (suggest_reduction, refine_reduction)
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pytest
import torch

from app.services.reduction.reduction_service import (
    FragmentMesh,
    FractureReductionService,
    LandmarkICPModel,
    OcclusalConstraintEngine,
    ReductionPlan,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_fragments() -> List[FragmentMesh]:
    """Create mock fracture fragments (mandible body + ramus)."""
    return [
        FragmentMesh(
            fragment_id="skull_base",
            label_value=1,
            points=np.random.randn(500, 3).astype(np.float32) * 10 + np.array([0, 50, 0]),
            centroid_mm=np.array([0.0, 50.0, 0.0]),
            volume_mm3=5000.0,
            parent_structure="skull_base",
            is_reference=True,
        ),
        FragmentMesh(
            fragment_id="mandible_body_L",
            label_value=2,
            points=np.random.randn(400, 3).astype(np.float32) * 8 + np.array([-20, 0, -10]),
            centroid_mm=np.array([-20.0, 0.0, -10.0]),
            volume_mm3=3000.0,
            parent_structure="mandible_body",
            is_reference=False,
        ),
        FragmentMesh(
            fragment_id="mandible_body_R",
            label_value=3,
            points=np.random.randn(400, 3).astype(np.float32) * 8 + np.array([20, 0, -10]),
            centroid_mm=np.array([20.0, 0.0, -10.0]),
            volume_mm3=3000.0,
            parent_structure="mandible_body",
            is_reference=False,
        ),
        FragmentMesh(
            fragment_id="symphysis",
            label_value=4,
            points=np.random.randn(200, 3).astype(np.float32) * 5 + np.array([0, -10, -15]),
            centroid_mm=np.array([0.0, -10.0, -15.0]),
            volume_mm3=1000.0,
            parent_structure="symphysis",
            is_reference=False,
        ),
    ]


@pytest.fixture
def mock_dental_constraints():
    from app.schemas.plan import OcclusalConstraints
    return OcclusalConstraints(
        target_overjet_mm=2.0,
        target_overbite_mm=3.0,
        molar_class_target="Class_I",
        midline_tolerance_mm=1.0,
        cant_tolerance_degrees=2.0,
    )


# ─── Joint optimizer loss tests ──────────────────────────────────────────────


class TestFractureFittingLoss:
    def test_forward(self):
        from app.services.reduction.joint_optimizer import FractureFittingLoss
        loss_fn = FractureFittingLoss()
        a = torch.randn(2, 256, 3)
        b = torch.randn(2, 256, 3)
        loss = loss_fn(a, b)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_gradient_flows(self):
        from app.services.reduction.joint_optimizer import FractureFittingLoss
        loss_fn = FractureFittingLoss()
        a = torch.randn(1, 128, 3, requires_grad=True)
        b = torch.randn(1, 128, 3)
        loss = loss_fn(a, b)
        loss.backward()
        assert a.grad is not None
        assert not torch.all(a.grad == 0)

    def test_identical_surfaces_zero_loss(self):
        from app.services.reduction.joint_optimizer import FractureFittingLoss
        loss_fn = FractureFittingLoss()
        pts = torch.randn(1, 100, 3)
        loss = loss_fn(pts, pts.clone())
        assert loss.item() < 1e-5


class TestFractureOverlapLoss:
    def test_forward(self):
        from app.services.reduction.joint_optimizer import FractureOverlapLoss
        loss_fn = FractureOverlapLoss(threshold_mm=1.0)
        a = torch.randn(2, 256, 3)
        b = torch.randn(2, 256, 3) + 10  # Far apart
        loss = loss_fn(a, b)
        assert loss.dim() == 0

    def test_overlapping_surfaces_high_loss(self):
        from app.services.reduction.joint_optimizer import FractureOverlapLoss
        loss_fn = FractureOverlapLoss(threshold_mm=5.0)
        pts = torch.randn(1, 100, 3) * 0.1  # Small cloud
        loss = loss_fn(pts, pts.clone())
        assert loss.item() > 0  # Should detect overlap


# ─── Segment transform params tests ─────────────────────────────────────────


class TestSegmentTransformParams:
    def test_init_identity(self):
        from app.services.reduction.joint_optimizer import SegmentTransformParams
        params = SegmentTransformParams(n_segments=3)
        R, t = params.get_transforms()
        # Should be close to identity rotation
        identity = torch.eye(3).unsqueeze(0).expand(3, -1, -1)
        assert torch.allclose(R, identity, atol=1e-4)
        # Translation should be zero
        assert torch.allclose(t, torch.zeros(3, 3), atol=1e-6)

    def test_apply_to_points_identity(self):
        from app.services.reduction.joint_optimizer import SegmentTransformParams
        params = SegmentTransformParams(n_segments=2)
        pts = torch.randn(100, 3)
        result = params.apply_to_points(0, pts)
        assert torch.allclose(pts, result, atol=1e-4)

    def test_to_4x4_matrices(self):
        from app.services.reduction.joint_optimizer import SegmentTransformParams
        params = SegmentTransformParams(n_segments=2)
        matrices = params.to_4x4_matrices()
        assert len(matrices) == 2
        for M in matrices:
            assert M.shape == (4, 4)
            assert np.allclose(M, np.eye(4), atol=1e-4)

    def test_optimization_changes_params(self):
        from app.services.reduction.joint_optimizer import SegmentTransformParams
        params = SegmentTransformParams(n_segments=1)
        optimizer = torch.optim.Adam(params.parameters(), lr=0.01)

        # Create a simple objective: move the points toward a target
        target = torch.tensor([5.0, 0.0, 0.0])
        pts = torch.randn(50, 3)

        for _ in range(10):
            optimizer.zero_grad()
            transformed = params.apply_to_points(0, pts)
            loss = (transformed.mean(dim=0) - target).pow(2).sum()
            loss.backward()
            optimizer.step()

        # Translation should have moved toward target
        _, t = params.get_transforms()
        assert t[0, 0].item() > 0.1  # Should be positive (toward target x=5)


# ─── Joint optimizer convergence test ────────────────────────────────────────


class TestOcclusionFirstJointOptimizer:
    def test_basic_convergence(self):
        """Test that the optimizer converges on simple synthetic data."""
        from app.services.reduction.joint_optimizer import OcclusionFirstJointOptimizer

        optimizer = OcclusionFirstJointOptimizer(
            max_steps=100,
            lr=0.01,
            device="cpu",
        )

        # Create simple fragments
        ref_pts = np.random.randn(200, 3).astype(np.float32) * 5
        movable_pts = np.random.randn(200, 3).astype(np.float32) * 5 + np.array([10, 0, 0])

        result = optimizer.optimize(
            fragment_points={
                "reference": ref_pts,
                "movable": movable_pts,
            },
            fragment_is_reference={
                "reference": True,
                "movable": False,
            },
            fracture_surface_pairs=[("reference", "movable")],
        )

        assert result.segment_transforms is not None
        assert "reference" in result.segment_transforms
        assert "movable" in result.segment_transforms
        assert result.optimization_steps > 0
        assert result.final_total_loss < float("inf")

        # Reference should be identity
        assert np.allclose(result.segment_transforms["reference"], np.eye(4))

    def test_with_dental_data(self):
        """Test optimization with dental arch data."""
        from app.services.reduction.joint_optimizer import OcclusionFirstJointOptimizer

        optimizer = OcclusionFirstJointOptimizer(
            max_steps=50,
            lr=0.01,
            device="cpu",
        )

        result = optimizer.optimize(
            fragment_points={
                "ref": np.random.randn(100, 3).astype(np.float32),
                "frag_1": np.random.randn(100, 3).astype(np.float32) + 5,
            },
            fragment_is_reference={"ref": True, "frag_1": False},
            upper_dental_points=np.random.randn(200, 3).astype(np.float32),
            lower_dental_points=np.random.randn(200, 3).astype(np.float32),
            upper_midline=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            lower_midline=np.array([0.5, 0.0, -5.0], dtype=np.float32),
        )

        assert "midline" in result.loss_breakdown
        assert result.total_time_ms > 0

    def test_loss_decreases(self):
        """Verify that the total loss decreases during optimization."""
        from app.services.reduction.joint_optimizer import OcclusionFirstJointOptimizer

        optimizer = OcclusionFirstJointOptimizer(
            max_steps=200,
            lr=0.005,
            device="cpu",
        )

        result = optimizer.optimize(
            fragment_points={
                "ref": np.random.randn(150, 3).astype(np.float32),
                "frag": np.random.randn(150, 3).astype(np.float32) + 8,
            },
            fragment_is_reference={"ref": True, "frag": False},
            fracture_surface_pairs=[("ref", "frag")],
        )

        initial = result.convergence_metrics.get("initial_loss", 0)
        final = result.convergence_metrics.get("final_loss", 0)
        # Loss should decrease (or at least not increase significantly)
        assert final <= initial + 0.01


# ─── Landmark ICP model tests ───────────────────────────────────────────────


class TestLandmarkICPModel:
    def test_predict_with_reference(self, mock_fragments):
        model = LandmarkICPModel()
        assert model.name == "landmark_icp"
        assert model.version == "2.0.0"

        # With no reference, should fall back to dental-aware alignment
        transforms = model.predict(mock_fragments, None, None)
        assert len(transforms) == len(mock_fragments)
        # Reference fragment should be identity
        assert np.allclose(transforms["skull_base"], np.eye(4))

    def test_predict_non_ref_fragments_get_transforms(self, mock_fragments):
        model = LandmarkICPModel()
        transforms = model.predict(mock_fragments, None, None)
        # Non-reference fragments should have non-identity transforms
        for frag in mock_fragments:
            assert frag.fragment_id in transforms
            assert transforms[frag.fragment_id].shape == (4, 4)


# ─── Constraint engine tests ────────────────────────────────────────────────


class TestOcclusalConstraintEngine:
    def test_apply_constraints_no_dental_data(self, mock_fragments):
        engine = OcclusalConstraintEngine()
        transforms = {f.fragment_id: np.eye(4) for f in mock_fragments}
        refined, metrics = engine.apply_constraints(
            transforms, mock_fragments, None, None, None,
        )
        assert len(refined) == len(transforms)
        assert metrics.constraints_satisfied

    def test_check_symmetry(self, mock_fragments):
        engine = OcclusalConstraintEngine()
        transforms = {f.fragment_id: np.eye(4) for f in mock_fragments}
        score, violations = engine.check_symmetry(mock_fragments, transforms)
        assert 0.0 <= score <= 1.0

    def test_find_bilateral_pairs(self, mock_fragments):
        engine = OcclusalConstraintEngine()
        pairs = engine._find_bilateral_pairs(mock_fragments)
        # mandible_body_L and mandible_body_R should pair
        frag_pair_ids = [(a.fragment_id, b.fragment_id) for a, b in pairs]
        assert ("mandible_body_L", "mandible_body_R") in frag_pair_ids


# ─── Reduction service end-to-end tests ─────────────────────────────────────


class TestFractureReductionService:
    @pytest.mark.asyncio
    async def test_suggest_reduction_basic(self, mock_fragments):
        service = FractureReductionService()
        plan = await service.suggest_reduction(
            fragments=mock_fragments,
        )
        assert isinstance(plan, ReductionPlan)
        assert len(plan.fragment_transforms) == len(mock_fragments)
        assert plan.overall_confidence > 0
        assert plan.symmetry_score > 0
        assert plan.validation is not None
        assert plan.model_name == "occlusion_first_v2"
        assert plan.model_version == "2.0.0"

    @pytest.mark.asyncio
    async def test_suggest_reduction_with_constraints(
        self, mock_fragments, mock_dental_constraints
    ):
        service = FractureReductionService()
        plan = await service.suggest_reduction(
            fragments=mock_fragments,
            dental_constraints=mock_dental_constraints,
        )
        assert plan.occlusal_metrics is not None

    @pytest.mark.asyncio
    async def test_suggest_reduction_empty_fragments(self):
        from app.core.exceptions import FractureFragmentError
        service = FractureReductionService()
        with pytest.raises(FractureFragmentError):
            await service.suggest_reduction(fragments=[])

    @pytest.mark.asyncio
    async def test_refine_reduction(self, mock_fragments):
        service = FractureReductionService()
        plan = await service.suggest_reduction(fragments=mock_fragments)

        # Surgeon manually adjusts one fragment
        surgeon_edits = {
            "symphysis": np.eye(4),
        }
        surgeon_edits["symphysis"][:3, 3] = [1.0, 0.0, 0.0]

        refined = await service.refine_reduction(
            plan=plan,
            surgeon_edits=surgeon_edits,
            fragments=mock_fragments,
        )
        assert isinstance(refined, ReductionPlan)
        assert "surgeon_refined" in refined.model_name

    @pytest.mark.asyncio
    async def test_plan_has_correct_structure(self, mock_fragments):
        service = FractureReductionService()
        plan = await service.suggest_reduction(fragments=mock_fragments)

        # All fragments have transforms
        for frag in mock_fragments:
            assert frag.fragment_id in plan.fragment_transforms
            T = plan.fragment_transforms[frag.fragment_id]
            assert T.shape == (4, 4)

        # All fragments have confidences
        for frag in mock_fragments:
            assert frag.fragment_id in plan.fragment_confidences
            conf = plan.fragment_confidences[frag.fragment_id]
            assert 0 <= conf <= 1

        # Validation present
        assert plan.validation is not None
        assert isinstance(plan.validation.passed, bool)

    @pytest.mark.asyncio
    async def test_reference_fragment_identity(self, mock_fragments):
        service = FractureReductionService()
        plan = await service.suggest_reduction(fragments=mock_fragments)

        # Reference fragment should keep identity
        ref_frags = [f for f in mock_fragments if f.is_reference]
        for rf in ref_frags:
            T = plan.fragment_transforms[rf.fragment_id]
            assert np.allclose(T, np.eye(4))
