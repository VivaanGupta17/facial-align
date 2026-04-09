"""
Comprehensive tests for the ML-native occlusion system.

Tests cover:
- Loss function forward passes and gradient flow
- SE(3) transform composition and application
- DentalArchEncoder forward pass with random point clouds
- OcclusionTransformer cross-attention
- OcclusionModel end-to-end forward pass
- Landmark detector forward pass
- Collision detection (differentiable and BVH)
- OcclusionService API compatibility
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def random_upper_cloud():
    """Random point cloud simulating upper arch (16 teeth, 512 pts each)."""
    teeth = {}
    for fdi in [11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28]:
        # Place teeth along an arch-shaped curve
        angle = (fdi % 20 - 4) * 0.2  # Spread along arch
        cx = np.sin(angle) * 30
        cy = np.cos(angle) * 30
        cz = 0.0
        pts = np.random.randn(512, 3).astype(np.float32) * 2.0
        pts[:, 0] += cx
        pts[:, 1] += cy
        pts[:, 2] += cz
        teeth[fdi] = pts
    return teeth


@pytest.fixture
def random_lower_cloud():
    """Random point cloud simulating lower arch."""
    teeth = {}
    for fdi in [31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]:
        angle = (fdi % 40 - 4) * 0.2
        cx = np.sin(angle) * 30
        cy = np.cos(angle) * 30
        cz = -5.0  # Below upper arch
        pts = np.random.randn(512, 3).astype(np.float32) * 2.0
        pts[:, 0] += cx
        pts[:, 1] += cy
        pts[:, 2] += cz
        teeth[fdi] = pts
    return teeth


@pytest.fixture
def batch_upper_lower():
    """Batched upper and lower point clouds (B=2, N=1024)."""
    B, N, M = 2, 1024, 1024
    upper = torch.randn(B, N, 3)
    lower = torch.randn(B, M, 3)
    lower[:, :, 2] -= 5.0  # Offset lower arch
    return upper, lower


# ─── Loss function tests ────────────────────────────────────────────────────


class TestChamferOcclusionLoss:
    def test_forward_returns_scalar(self, batch_upper_lower):
        from app.services.occlusion.occlusal_losses import ChamferOcclusionLoss
        upper, lower = batch_upper_lower
        loss_fn = ChamferOcclusionLoss()
        loss = loss_fn(upper, lower)
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0.0

    def test_gradient_flows(self, batch_upper_lower):
        from app.services.occlusion.occlusal_losses import ChamferOcclusionLoss
        upper, lower = batch_upper_lower
        upper.requires_grad_(True)
        loss_fn = ChamferOcclusionLoss()
        loss = loss_fn(upper, lower)
        loss.backward()
        assert upper.grad is not None
        assert not torch.all(upper.grad == 0)

    def test_identical_clouds_low_loss(self):
        from app.services.occlusion.occlusal_losses import ChamferOcclusionLoss
        pts = torch.randn(1, 256, 3)
        loss_fn = ChamferOcclusionLoss()
        loss = loss_fn(pts, pts.clone())
        assert loss.item() < 1e-5


class TestOcclusalProjectionOverlapLoss:
    def test_forward_returns_scalar(self, batch_upper_lower):
        from app.services.occlusion.occlusal_losses import OcclusalProjectionOverlapLoss
        upper, lower = batch_upper_lower
        loss_fn = OcclusalProjectionOverlapLoss(grid_resolution=32)
        loss = loss_fn(upper, lower)
        assert loss.dim() == 0
        assert loss.item() >= 0.0

    def test_gradient_flows(self, batch_upper_lower):
        from app.services.occlusion.occlusal_losses import OcclusalProjectionOverlapLoss
        upper, lower = batch_upper_lower
        upper.requires_grad_(True)
        loss_fn = OcclusalProjectionOverlapLoss(grid_resolution=32)
        loss = loss_fn(upper, lower)
        loss.backward()
        assert upper.grad is not None


class TestOcclusalDistanceUniformityLoss:
    def test_forward_returns_scalar(self, batch_upper_lower):
        from app.services.occlusion.occlusal_losses import OcclusalDistanceUniformityLoss
        upper, lower = batch_upper_lower
        loss_fn = OcclusalDistanceUniformityLoss()
        loss = loss_fn(upper, lower)
        assert loss.dim() == 0
        assert loss.item() >= 0.0

    def test_gradient_flows(self, batch_upper_lower):
        from app.services.occlusion.occlusal_losses import OcclusalDistanceUniformityLoss
        upper, lower = batch_upper_lower
        upper.requires_grad_(True)
        loss_fn = OcclusalDistanceUniformityLoss()
        loss = loss_fn(upper, lower)
        loss.backward()
        assert upper.grad is not None


class TestCollisionLoss:
    def test_forward_no_normals(self, batch_upper_lower):
        from app.services.occlusion.occlusal_losses import CollisionLoss
        upper, lower = batch_upper_lower
        loss_fn = CollisionLoss(penetration_threshold_mm=1.0)
        loss = loss_fn(upper, lower)
        assert loss.dim() == 0

    def test_gradient_flows(self, batch_upper_lower):
        from app.services.occlusion.occlusal_losses import CollisionLoss
        upper, lower = batch_upper_lower
        upper.requires_grad_(True)
        loss_fn = CollisionLoss()
        loss = loss_fn(upper, lower)
        loss.backward()
        assert upper.grad is not None


class TestMidlineDeviationLoss:
    def test_forward(self):
        from app.services.occlusion.occlusal_losses import MidlineDeviationLoss
        upper_mid = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        lower_mid = torch.tensor([[0.5, 0.0, -5.0], [1.5, 0.0, -5.0]])
        loss_fn = MidlineDeviationLoss()
        loss = loss_fn(upper_mid, lower_mid)
        assert loss.item() > 0

    def test_zero_deviation(self):
        from app.services.occlusion.occlusal_losses import MidlineDeviationLoss
        mid = torch.tensor([[0.0, 0.0, 0.0]])
        loss_fn = MidlineDeviationLoss()
        loss = loss_fn(mid, mid.clone())
        assert loss.item() < 1e-5


class TestMolarRelationLoss:
    def test_forward(self):
        from app.services.occlusion.occlusal_losses import MolarRelationLoss
        upper_mol = torch.randn(2, 5, 3)
        lower_mol = torch.randn(2, 5, 3)
        loss_fn = MolarRelationLoss()
        loss = loss_fn(upper_mol, lower_mol)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_gradient_flows(self):
        from app.services.occlusion.occlusal_losses import MolarRelationLoss
        upper_mol = torch.randn(2, 5, 3, requires_grad=True)
        lower_mol = torch.randn(2, 5, 3)
        loss_fn = MolarRelationLoss()
        loss = loss_fn(upper_mol, lower_mol)
        loss.backward()
        assert upper_mol.grad is not None


class TestDentalArchCurveLoss:
    def test_forward(self):
        from app.services.occlusion.occlusal_losses import DentalArchCurveLoss
        pred = torch.randn(1, 8, 3)
        target = torch.randn(1, 8, 3)
        loss_fn = DentalArchCurveLoss()
        loss = loss_fn(pred, target)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_identical_curves_low_loss(self):
        from app.services.occlusion.occlusal_losses import DentalArchCurveLoss
        curve = torch.randn(1, 8, 3)
        loss_fn = DentalArchCurveLoss()
        loss = loss_fn(curve, curve.clone())
        assert loss.item() < 1e-3


class TestCompositeDentalLoss:
    def test_forward_minimal(self, batch_upper_lower):
        from app.services.occlusion.occlusal_losses import CompositeDentalLoss
        upper, lower = batch_upper_lower
        loss_fn = CompositeDentalLoss()
        total, breakdown = loss_fn(upper, lower)
        assert total.dim() == 0
        assert "chamfer" in breakdown
        assert "overlap" in breakdown
        assert "uniformity" in breakdown
        assert "collision" in breakdown
        assert "total" in breakdown

    def test_forward_with_all_components(self, batch_upper_lower):
        from app.services.occlusion.occlusal_losses import CompositeDentalLoss
        upper, lower = batch_upper_lower
        B = upper.shape[0]
        loss_fn = CompositeDentalLoss()
        total, breakdown = loss_fn(
            upper, lower,
            upper_midline=torch.randn(B, 3),
            lower_midline=torch.randn(B, 3),
            upper_molar_landmarks=torch.randn(B, 5, 3),
            lower_molar_landmarks=torch.randn(B, 5, 3),
            predicted_centroids=torch.randn(B, 8, 3),
            target_centroids=torch.randn(B, 8, 3),
        )
        assert "midline" in breakdown
        assert "molar" in breakdown
        assert "arch_curve" in breakdown

    def test_gradient_flows_composite(self, batch_upper_lower):
        from app.services.occlusion.occlusal_losses import CompositeDentalLoss
        upper, lower = batch_upper_lower
        upper.requires_grad_(True)
        loss_fn = CompositeDentalLoss()
        total, _ = loss_fn(upper, lower)
        total.backward()
        assert upper.grad is not None


# ─── SE(3) transform tests ──────────────────────────────────────────────────


class TestSE3Transforms:
    def test_apply_se3_identity(self):
        from app.services.occlusion.se3_transforms import apply_se3_transform
        pts = torch.randn(2, 100, 3)
        params = torch.zeros(2, 6)  # Identity in Lie algebra
        result = apply_se3_transform(pts, params)
        assert result.shape == pts.shape
        assert torch.allclose(result, pts, atol=1e-5)

    def test_apply_rotation_translation(self):
        from app.services.occlusion.se3_transforms import apply_rotation_translation
        pts = torch.randn(1, 50, 3)
        # Identity rotation in 6D: [1,0,0, 0,1,0]
        rot_6d = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        trans = torch.tensor([[1.0, 2.0, 3.0]])
        result = apply_rotation_translation(pts, rot_6d, trans)
        expected = pts + trans.unsqueeze(1)
        assert torch.allclose(result, expected, atol=1e-4)

    def test_compose_transforms(self):
        from app.services.occlusion.se3_transforms import compose_transforms
        T1 = torch.eye(4).unsqueeze(0)
        T2 = torch.eye(4).unsqueeze(0)
        T2[0, :3, 3] = torch.tensor([1.0, 0.0, 0.0])
        composed = compose_transforms(T1, T2)
        assert torch.allclose(composed, T2, atol=1e-6)

    def test_invert_transform(self):
        from app.services.occlusion.se3_transforms import compose_transforms, invert_transform
        T = torch.eye(4).unsqueeze(0)
        T[0, :3, 3] = torch.tensor([5.0, -3.0, 1.0])
        T_inv = invert_transform(T)
        identity = compose_transforms(T, T_inv)
        assert torch.allclose(identity, torch.eye(4).unsqueeze(0), atol=1e-5)

    def test_rotation_6d_roundtrip(self):
        from app.services.occlusion.se3_transforms import (
            rotation_6d_to_mat,
            rotation_matrix_to_6d,
        )
        R = torch.eye(3).unsqueeze(0)
        r6d = rotation_matrix_to_6d(R)
        R_back = rotation_6d_to_mat(r6d)
        assert torch.allclose(R, R_back, atol=1e-5)

    def test_per_tooth_transform_identity(self):
        from app.services.occlusion.se3_transforms import per_tooth_transform
        teeth = [torch.randn(100, 3) for _ in range(4)]
        # Identity params: [1,0,0,0,1,0, 0,0,0]
        params = torch.zeros(4, 9)
        params[:, 0] = 1.0
        params[:, 4] = 1.0
        result = per_tooth_transform(teeth, params)
        for orig, transformed in zip(teeth, result):
            assert torch.allclose(orig, transformed, atol=1e-4)


class TestSE3TransformHead:
    def test_forward(self):
        from app.services.occlusion.se3_transforms import SE3TransformHead
        head = SE3TransformHead(input_dim=256)
        features = torch.randn(2, 16, 256)
        rot, trans = head(features)
        assert rot.shape == (2, 16, 6)
        assert trans.shape == (2, 16, 3)

    def test_to_transform_matrices(self):
        from app.services.occlusion.se3_transforms import SE3TransformHead
        head = SE3TransformHead(input_dim=128)
        features = torch.randn(1, 8, 128)
        rot, trans = head(features)
        T = head.to_transform_matrices(rot, trans)
        assert T.shape == (1, 8, 4, 4)
        # Check last row
        assert torch.allclose(T[:, :, 3, :3], torch.zeros(1, 8, 3))
        assert torch.allclose(T[:, :, 3, 3], torch.ones(1, 8))


# ─── Encoder tests ──────────────────────────────────────────────────────────


class TestDGCNNToothEncoder:
    def test_forward(self):
        from app.services.occlusion.arch_encoder import DGCNNToothEncoder
        from torch_geometric.data import Batch, Data
        encoder = DGCNNToothEncoder(k=10, embed_dim=256)
        # Create batch of 3 teeth, each 128 points
        data_list = [Data(pos=torch.randn(128, 3)) for _ in range(3)]
        batch = Batch.from_data_list(data_list)
        out = encoder(batch.pos, batch.batch)
        assert out.shape == (3, 256)


class TestDentalArchEncoder:
    def test_forward(self):
        from app.services.occlusion.arch_encoder import DentalArchEncoder
        encoder = DentalArchEncoder(k=10, points_per_tooth=128)
        tooth_clouds = [torch.randn(200, 3) for _ in range(4)]
        fdi_numbers = [11, 12, 13, 14]
        per_tooth, global_embed = encoder(tooth_clouds, fdi_numbers)
        assert per_tooth.shape == (4, 256)
        assert global_embed.shape == (1, 512)

    def test_empty_arch(self):
        from app.services.occlusion.arch_encoder import DentalArchEncoder
        encoder = DentalArchEncoder(k=10)
        per_tooth, global_embed = encoder([], [])
        assert per_tooth.shape[0] == 0
        assert global_embed.shape == (1, 512)

    def test_encode_arch_numpy(self, random_upper_cloud):
        from app.services.occlusion.arch_encoder import DentalArchEncoder
        encoder = DentalArchEncoder(k=10, points_per_tooth=128)
        encoder.eval()
        per_tooth, global_embed, fdi_order = encoder.encode_arch(random_upper_cloud)
        assert per_tooth.shape[0] == len(random_upper_cloud)
        assert len(fdi_order) == len(random_upper_cloud)


# ─── Landmark detector tests ────────────────────────────────────────────────


class TestDentalLandmarkDetector:
    def test_forward(self):
        from app.services.occlusion.landmark_detector import DentalLandmarkDetector
        detector = DentalLandmarkDetector(k=10, points_per_tooth=128)
        clouds = [torch.randn(200, 3) for _ in range(3)]
        fdi_numbers = [16, 36, 11]  # Molar, molar, incisor
        results = detector(clouds, fdi_numbers)
        assert 16 in results
        assert 36 in results
        assert 11 in results
        # Molar should have 5 landmarks
        assert results[16]["landmarks"].shape == (5, 3)
        # Incisor should have 3 landmarks
        assert results[11]["landmarks"].shape == (3, 3)

    def test_detect_arch_landmarks_numpy(self, random_upper_cloud):
        from app.services.occlusion.landmark_detector import DentalLandmarkDetector
        detector = DentalLandmarkDetector(k=10, points_per_tooth=128)
        detector.eval()
        # Use subset of teeth
        subset = {k: v for k, v in list(random_upper_cloud.items())[:4]}
        results = detector.detect_arch_landmarks(subset)
        for fdi, data in results.items():
            assert "landmarks" in data
            assert "confidence" in data
            assert isinstance(data["landmarks"], np.ndarray)


# ─── Collision detection tests ───────────────────────────────────────────────


class TestDifferentiableCollisionLoss:
    def test_forward(self, batch_upper_lower):
        from app.services.occlusion.collision_detection import DifferentiableCollisionLoss
        upper, lower = batch_upper_lower
        loss_fn = DifferentiableCollisionLoss()
        loss, info = loss_fn(upper, lower)
        assert loss.dim() == 0
        assert "n_penetrating" in info
        assert "max_depth_mm" in info

    def test_gradient_flows(self, batch_upper_lower):
        from app.services.occlusion.collision_detection import DifferentiableCollisionLoss
        upper, lower = batch_upper_lower
        upper.requires_grad_(True)
        loss_fn = DifferentiableCollisionLoss()
        loss, _ = loss_fn(upper, lower)
        loss.backward()
        assert upper.grad is not None


class TestBVHCollisionDetector:
    def test_no_collision(self):
        from app.services.occlusion.collision_detection import BVHCollisionDetector
        detector = BVHCollisionDetector(resolution_mm=0.5)
        pts_a = np.random.randn(100, 3).astype(np.float64)
        pts_b = np.random.randn(100, 3).astype(np.float64) + 100  # Far apart
        result = detector.check_collision(pts_a, pts_b)
        assert not result["colliding"]

    def test_collision_detected(self):
        from app.services.occlusion.collision_detection import BVHCollisionDetector
        detector = BVHCollisionDetector(resolution_mm=1.0)
        pts_a = np.random.randn(200, 3).astype(np.float64) * 0.1
        pts_b = np.random.randn(200, 3).astype(np.float64) * 0.1  # Same region
        result = detector.check_collision(pts_a, pts_b)
        assert result["colliding"]
        assert result["n_intersecting_points"] > 0


# ─── Occlusion transformer tests ────────────────────────────────────────────


class TestOcclusionTransformer:
    def test_forward(self):
        from app.services.occlusion.occlusion_service import OcclusionTransformer
        transformer = OcclusionTransformer(d_model=64, nhead=4, num_layers=2)
        upper = torch.randn(1, 8, 64)
        lower = torch.randn(1, 8, 64)
        fused_upper, fused_lower = transformer(upper, lower)
        assert fused_upper.shape == upper.shape
        assert fused_lower.shape == lower.shape


class TestOcclusionScoringHead:
    def test_forward(self):
        from app.services.occlusion.occlusion_service import OcclusionScoringHead
        head = OcclusionScoringHead(d_model=64)
        upper_global = torch.randn(2, 64)
        lower_global = torch.randn(2, 64)
        metrics = head(upper_global, lower_global)
        assert "overjet_mm" in metrics
        assert "quality_score" in metrics
        assert metrics["quality_score"].shape == (2,)
        assert (metrics["quality_score"] >= 0).all()
        assert (metrics["quality_score"] <= 1).all()


# ─── Full model end-to-end test ──────────────────────────────────────────────


class TestOcclusionModel:
    def test_forward_e2e(self):
        from app.services.occlusion.occlusion_service import OcclusionModel
        model = OcclusionModel(
            per_tooth_dim=64, global_dim=128,
            transformer_layers=2, transformer_heads=4,
        )
        model.eval()
        upper_clouds = [torch.randn(128, 3) for _ in range(4)]
        upper_fdi = [11, 12, 13, 14]
        lower_clouds = [torch.randn(128, 3) for _ in range(4)]
        lower_fdi = [31, 32, 33, 34]

        with torch.no_grad():
            output = model(upper_clouds, upper_fdi, lower_clouds, lower_fdi)

        assert "rotation_6d" in output
        assert "translation" in output
        assert "metrics" in output
        assert output["rotation_6d"].shape[-1] == 6
        assert output["translation"].shape[-1] == 3


# ─── Service API tests ──────────────────────────────────────────────────────


class TestOcclusionServiceAPI:
    """Test that the new ML service maintains backward-compatible API."""

    def test_instantiation(self):
        from app.services.occlusion.occlusion_service import OcclusionService
        service = OcclusionService()
        assert service is not None

    @pytest.mark.asyncio
    async def test_evaluate_occlusion_with_mock_arch(self):
        """Test with a simple mock arch mesh."""
        from app.services.occlusion.occlusion_service import OcclusionService

        class MockMesh:
            def __init__(self, n_pts=1000):
                self.vertices = np.random.randn(n_pts, 3).astype(np.float32) * 20
                self.metadata = None

        service = OcclusionService()
        metrics = await service.evaluate_occlusion(
            upper_arch=MockMesh(),
            lower_arch=MockMesh(),
        )
        assert metrics is not None
        assert hasattr(metrics, "constraints_satisfied")
        assert hasattr(metrics, "overjet_mm")

    @pytest.mark.asyncio
    async def test_evaluate_raises_on_none_arch(self):
        from app.services.occlusion.occlusion_service import OcclusionService
        from app.core.exceptions import DentalArchError
        service = OcclusionService()
        with pytest.raises(DentalArchError):
            await service.evaluate_occlusion(None, None)

    @pytest.mark.asyncio
    async def test_compute_dental_constraints_defaults(self):
        from app.services.occlusion.occlusion_service import OcclusionService
        service = OcclusionService()
        constraints = await service.compute_dental_constraints(None, [])
        assert constraints.target_overjet_mm == 2.0
        assert constraints.molar_class_target == "Class_I"

    @pytest.mark.asyncio
    async def test_suggest_splint_design(self):
        from app.services.occlusion.occlusion_service import OcclusionService
        from app.schemas.plan import OcclusalMetrics
        service = OcclusionService()
        metrics = OcclusalMetrics(constraints_satisfied=True)
        splint = await service.suggest_splint_design(metrics)
        assert splint is not None
        assert splint.target_vertical_dimension_mm > 0

    def test_assess_molar_relationship(self):
        from app.services.occlusion.occlusion_service import OcclusionService
        service = OcclusionService()
        result = service.assess_molar_relationship(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
        )
        assert result == "Class_I"

    def test_compute_arch_geometry(self):
        from app.services.occlusion.occlusion_service import OcclusionService

        class MockMesh:
            def __init__(self):
                self.vertices = np.random.randn(500, 3) * 20

        service = OcclusionService()
        geom = service.compute_arch_geometry(MockMesh(), is_upper=True)
        assert geom.arch_width_mm > 0
        assert geom.arch_length_mm > 0
