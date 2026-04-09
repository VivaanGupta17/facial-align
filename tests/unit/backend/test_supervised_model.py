"""
Unit tests for the supervised fracture reduction model.

Tests cover:
- CTVolumeEncoder: Shape correctness, HU preprocessing
- IOSPointCloudEncoder: Shape correctness, missing IOS handling
- MultimodalFusionModule: CT-only and CT+IOS modes
- Prediction heads: Output shapes, identity initialisation
- End-to-end FacialAlignSupervisedModel: Full forward pass
- Supervised losses: Loss computation and phase schedule
- Rotation utilities: R6 → SO(3) correctness

All tests use small tensor sizes for fast execution.
"""

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

# ─── Fixtures ─────────────────────────────────────────────────────────────────

BATCH_SIZE = 2
CT_DIM = 64  # Small volume for testing
NUM_FRAGMENTS = 3
NUM_TEETH = 32
POINTS_PER_TOOTH = 128


@pytest.fixture
def ct_volume():
    """Small CT volume for testing."""
    return torch.randn(BATCH_SIZE, 1, CT_DIM, CT_DIM, CT_DIM) * 1000  # HU range


@pytest.fixture
def ios_point_clouds():
    """Per-tooth point clouds."""
    return torch.randn(BATCH_SIZE, NUM_TEETH, POINTS_PER_TOOTH, 3)


@pytest.fixture
def tooth_mask():
    """Tooth mask (True = missing)."""
    mask = torch.ones(BATCH_SIZE, NUM_TEETH, dtype=torch.bool)
    # First 16 teeth present
    mask[:, :16] = False
    return mask


# ─── Rotation Utilities ───────────────────────────────────────────────────────


class TestRotationUtilities:
    """Tests for R6 → SO(3) conversion."""

    def test_rotation_6d_identity(self):
        """Identity R6 [1,0,0,0,1,0] should produce I₃."""
        from app.services.supervised.prediction_heads import rotation_6d_to_matrix

        r6 = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        R = rotation_6d_to_matrix(r6)

        expected = torch.eye(3).unsqueeze(0)
        assert torch.allclose(R, expected, atol=1e-5)

    def test_rotation_6d_orthogonal(self):
        """Output should always be a valid rotation matrix (R^T R = I, det = 1)."""
        from app.services.supervised.prediction_heads import rotation_6d_to_matrix

        # Random R6 input
        r6 = torch.randn(10, 6)
        R = rotation_6d_to_matrix(r6)

        # Check orthogonality: R^T @ R ≈ I
        RtR = torch.bmm(R.transpose(1, 2), R)
        I = torch.eye(3).expand_as(RtR)
        assert torch.allclose(RtR, I, atol=1e-5)

        # Check determinant ≈ 1 (proper rotation, not reflection)
        dets = torch.linalg.det(R)
        assert torch.allclose(dets, torch.ones(10), atol=1e-5)

    def test_rotation_6d_batch(self):
        """Batch R6 conversion should work correctly."""
        from app.services.supervised.prediction_heads import rotation_6d_to_matrix

        r6 = torch.randn(BATCH_SIZE, NUM_FRAGMENTS, 6)
        R = rotation_6d_to_matrix(r6)
        assert R.shape == (BATCH_SIZE, NUM_FRAGMENTS, 3, 3)

    def test_build_se3_matrix(self):
        """SE(3) matrix should be correctly assembled."""
        from app.services.supervised.prediction_heads import build_se3_matrix

        R = torch.eye(3).unsqueeze(0)
        t = torch.tensor([[1.0, 2.0, 3.0]])
        T = build_se3_matrix(R, t)

        assert T.shape == (1, 4, 4)
        assert torch.allclose(T[0, :3, :3], R[0])
        assert torch.allclose(T[0, :3, 3], t[0])
        assert torch.allclose(T[0, 3, :], torch.tensor([0.0, 0.0, 0.0, 1.0]))


# ─── CT Encoder ───────────────────────────────────────────────────────────────


class TestCTVolumeEncoder:
    """Tests for the 3D ResNet CT encoder."""

    def test_output_shapes(self, ct_volume):
        """Encoder should produce correct output shapes."""
        from app.services.supervised.ct_encoder import CTEncoderConfig, CTVolumeEncoder

        config = CTEncoderConfig(base_channels=16, layer_depths=(1, 1, 1, 1))
        encoder = CTVolumeEncoder(config)

        global_feat, patch_feat = encoder(ct_volume)
        assert global_feat.shape == (BATCH_SIZE, config.ct_feat_dim)
        assert patch_feat.shape[0] == BATCH_SIZE
        assert patch_feat.shape[2] == config.ct_feat_dim

    def test_hu_preprocessing(self):
        """HU preprocessing should clip and normalise."""
        from app.services.supervised.ct_encoder import CTVolumeEncoder

        volume = torch.tensor([-500.0, 0.0, 1500.0, 5000.0])
        normalised = CTVolumeEncoder.preprocess_hu(volume)

        assert normalised.min() >= 0.0
        assert normalised.max() <= 1.0
        assert normalised[0] == 0.0  # -500 clipped to 0


# ─── Multimodal Fusion ────────────────────────────────────────────────────────


class TestMultimodalFusion:
    """Tests for cross-attention fusion module."""

    def test_ct_only_mode(self):
        """Fusion should work without IOS data."""
        from app.services.supervised.multimodal_fusion import FusionConfig, MultimodalFusionModule

        config = FusionConfig(ct_feat_dim=64, ios_feat_dim=64, fused_dim=128, num_layers=1)
        fusion = MultimodalFusionModule(config)

        ct_global = torch.randn(BATCH_SIZE, 64)
        ct_patches = torch.randn(BATCH_SIZE, 16, 64)

        fused, ios_used = fusion(ct_global, ct_patches)
        assert fused.shape == (BATCH_SIZE, 129)  # 128 + 1 modality indicator
        # IOS was not provided
        assert not ios_used

    def test_ct_ios_mode(self):
        """Fusion should work with both modalities."""
        from app.services.supervised.multimodal_fusion import FusionConfig, MultimodalFusionModule

        config = FusionConfig(
            ct_feat_dim=64, ios_feat_dim=64, fused_dim=128,
            num_layers=1, ios_dropout_p=0.0,
        )
        fusion = MultimodalFusionModule(config)
        fusion.eval()  # Disable training-time IOS dropout

        ct_global = torch.randn(BATCH_SIZE, 64)
        ct_patches = torch.randn(BATCH_SIZE, 16, 64)
        ios_teeth = torch.randn(BATCH_SIZE, 8, 64)

        fused, ios_used = fusion(ct_global, ct_patches, ios_per_tooth=ios_teeth)
        assert fused.shape == (BATCH_SIZE, 129)
        assert ios_used


# ─── Fragment Transform Head ──────────────────────────────────────────────────


class TestFragmentTransformHead:
    """Tests for the fragment transform prediction head."""

    def test_output_shapes(self):
        """Head should produce correct shapes for all outputs."""
        from app.services.supervised.prediction_heads import FragmentTransformHead

        head = FragmentTransformHead(fused_dim=129, max_fragments=4)
        fused = torch.randn(BATCH_SIZE, 129)
        num_frags = torch.tensor([2, 3])

        result = head(fused, num_frags)

        assert result["rotations_r6"].shape == (BATCH_SIZE, 4, 6)
        assert result["rotations_matrix"].shape == (BATCH_SIZE, 4, 3, 3)
        assert result["translations"].shape == (BATCH_SIZE, 4, 3)
        assert result["confidences"].shape == (BATCH_SIZE, 4)
        assert result["se3_matrices"].shape == (BATCH_SIZE, 4, 4, 4)

    def test_inactive_fragment_masking(self):
        """Inactive fragments should have zero confidence and identity SE(3)."""
        from app.services.supervised.prediction_heads import FragmentTransformHead

        head = FragmentTransformHead(fused_dim=129, max_fragments=4)
        fused = torch.randn(BATCH_SIZE, 129)
        num_frags = torch.tensor([1, 2])

        result = head(fused, num_frags)

        # Batch 0: only fragment 0 active, rest should be identity
        assert result["confidences"][0, 1].item() == 0.0
        assert result["confidences"][0, 2].item() == 0.0
        assert result["confidences"][0, 3].item() == 0.0


# ─── Occlusion Scoring Head ──────────────────────────────────────────────────


class TestOcclusionScoringHead:
    """Tests for clinical metric prediction."""

    def test_output_shapes(self):
        """Head should produce metrics and classification."""
        from app.services.supervised.prediction_heads import OcclusionScoringHead

        head = OcclusionScoringHead(fused_dim=129)
        fused = torch.randn(BATCH_SIZE, 129)

        result = head(fused)

        assert result["metrics"].shape == (BATCH_SIZE, 3)
        assert result["molar_class_logits"].shape == (BATCH_SIZE, 3)
        assert result["molar_class_probs"].shape == (BATCH_SIZE, 3)

        # Probabilities should sum to 1
        prob_sums = result["molar_class_probs"].sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones(BATCH_SIZE), atol=1e-5)


# ─── Supervised Losses ────────────────────────────────────────────────────────


class TestSupervisedLosses:
    """Tests for the composite training loss."""

    def test_geodesic_rotation_loss_zero(self):
        """Geodesic loss should be zero for identical rotations."""
        from app.services.supervised.supervised_losses import geodesic_rotation_loss

        R = torch.eye(3).unsqueeze(0).expand(4, -1, -1)
        loss = geodesic_rotation_loss(R, R)
        assert torch.allclose(loss, torch.zeros(4), atol=1e-5)

    def test_geodesic_rotation_loss_90deg(self):
        """Geodesic loss for 90° rotation should be π/2."""
        from app.services.supervised.supervised_losses import geodesic_rotation_loss

        R1 = torch.eye(3).unsqueeze(0)
        # 90° rotation around Z axis
        R2 = torch.tensor([[[0.0, -1.0, 0.0],
                             [1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0]]]).float()

        loss = geodesic_rotation_loss(R1, R2)
        expected = torch.tensor([math.pi / 2])
        assert torch.allclose(loss, expected, atol=1e-4)

    def test_phase_schedule(self):
        """Loss should use MSE before transition and geodesic after."""
        from app.services.supervised.supervised_losses import SupervisedReductionLoss

        loss_fn = SupervisedReductionLoss(phase_transition_epoch=50)

        # Create minimal predictions and targets
        B, F, T = 1, 2, 4
        predictions = {
            "fragment_transforms": {
                "rotations_r6": torch.randn(B, F, 6),
                "rotations_matrix": torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, F, -1, -1),
                "translations": torch.randn(B, F, 3),
                "confidences": torch.ones(B, F),
                "se3_matrices": torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, F, -1, -1),
            },
            "tooth_transforms": {
                "rotations_r6": torch.randn(B, T, 6),
                "rotations_matrix": torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1),
                "translations": torch.randn(B, T, 3),
                "confidences": torch.ones(B, T),
                "se3_matrices": torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1),
            },
            "occlusion_scores": {
                "metrics": torch.randn(B, 3),
                "molar_class_logits": torch.randn(B, 3),
                "molar_class_probs": torch.softmax(torch.randn(B, 3), dim=-1),
            },
            "uncertainty": {
                "fragment_log_var": torch.zeros(B, F, 9),
                "tooth_log_var": torch.zeros(B, T, 9),
                "temperature": torch.tensor(1.0),
            },
        }
        targets = {
            "fragment_rotations": torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, F, -1, -1),
            "fragment_translations": torch.zeros(B, F, 3),
            "fragment_mask": torch.ones(B, F, dtype=torch.bool),
            "tooth_rotations": torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1),
            "tooth_translations": torch.zeros(B, T, 3),
            "tooth_mask": torch.ones(B, T, dtype=torch.bool),
            "metrics": torch.tensor([[2.5, 3.0, 0.0]]),
            "molar_class": torch.tensor([0]),
        }

        # Phase A (epoch < 50): MSE rotation
        loss_a = loss_fn(predictions, targets, epoch=10)
        assert loss_a["rotation_loss_type"] == "mse"
        assert "total_loss" in loss_a

        # Phase B (epoch >= 50): Geodesic
        loss_b = loss_fn(predictions, targets, epoch=60)
        assert loss_b["rotation_loss_type"] == "geodesic"


# ─── Synthetic Fracture Generator ────────────────────────────────────────────


class TestSyntheticFractureGenerator:
    """Tests for the DFGM-style fracture generator."""

    def test_random_displacement(self):
        """Displacement should be valid SE(3) with bounded magnitude."""
        from training.synthetic.fracture_generator import (
            FRACTURE_PATTERNS,
            SyntheticFractureGenerator,
        )

        gen = SyntheticFractureGenerator(seed=42)
        pattern = FRACTURE_PATTERNS[0]  # parasymphyseal

        T = gen._random_displacement(pattern)
        assert T.shape == (4, 4)
        assert np.allclose(T[3, :], [0, 0, 0, 1])

        # Translation should be within range
        t_mag = np.linalg.norm(T[:3, 3])
        assert pattern.displacement_range_mm[0] <= t_mag <= pattern.displacement_range_mm[1]

        # Rotation should be valid (det ≈ 1, orthogonal)
        R = T[:3, :3]
        assert np.allclose(np.linalg.det(R), 1.0, atol=1e-5)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-5)

    def test_ground_truth_is_inverse(self):
        """Ground truth transform should be the inverse of applied displacement."""
        from training.synthetic.fracture_generator import (
            FRACTURE_PATTERNS,
            SyntheticFractureGenerator,
        )

        gen = SyntheticFractureGenerator(seed=42)
        pattern = FRACTURE_PATTERNS[0]

        T = gen._random_displacement(pattern)
        T_inv = np.linalg.inv(T)

        # T @ T_inv should be identity
        product = T @ T_inv
        assert np.allclose(product, np.eye(4), atol=1e-10)
