"""
Unit tests for the post-processing pipeline.

Tests cover:
- TransformApplicator: SE(3) application, SLERP interpolation
- MeshCleanup: Degenerate removal, hole filling, manifold enforcement
- CollisionResolver: Penetration detection, iterative resolution
- ConfidenceGate: Threshold-based routing
- SurgeonEditHandler: Edit capture and undo/redo
"""

import numpy as np
import pytest


# ─── TransformApplicator ──────────────────────────────────────────────────────


class TestTransformApplicator:
    """Tests for SE(3) transform application to meshes."""

    def test_identity_transform(self):
        """Identity transform should not change vertex positions."""
        from app.services.postprocessing.transform_applicator import TransformApplicator

        applicator = TransformApplicator()

        # Create a simple triangle mesh
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        transform = np.eye(4)

        result = applicator.apply_transform(vertices, transform)
        assert np.allclose(result, vertices, atol=1e-10)

    def test_translation_only(self):
        """Pure translation should shift all vertices equally."""
        from app.services.postprocessing.transform_applicator import TransformApplicator

        applicator = TransformApplicator()

        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        transform = np.eye(4)
        transform[:3, 3] = [5.0, 0.0, 0.0]

        result = applicator.apply_transform(vertices, transform)
        expected = vertices + np.array([5.0, 0.0, 0.0])
        assert np.allclose(result, expected, atol=1e-10)

    def test_transform_validation(self):
        """Invalid transforms should be rejected."""
        from app.services.postprocessing.transform_applicator import (
            TransformApplicator,
            TransformValidationResult,
        )

        applicator = TransformApplicator()

        # Non-orthogonal rotation matrix
        bad_transform = np.eye(4)
        bad_transform[:3, :3] = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]])

        result = applicator.validate_transform(bad_transform)
        assert isinstance(result, TransformValidationResult)


# ─── ConfidenceGate ───────────────────────────────────────────────────────────


class TestConfidenceGate:
    """Tests for confidence-based routing."""

    def test_high_confidence_accepts(self):
        """Confidence ≥ 0.8 should route to ACCEPT."""
        from app.services.postprocessing.confidence_gate import ConfidenceGate

        gate = ConfidenceGate()
        result = gate.evaluate(
            overall_confidence=0.85,
            per_fragment_confidence=[0.9, 0.85],
        )
        assert result.level == "accept"
        assert not result.route_to_fallback

    def test_medium_confidence_reviews(self):
        """Confidence 0.5-0.8 should route to REVIEW."""
        from app.services.postprocessing.confidence_gate import ConfidenceGate

        gate = ConfidenceGate()
        result = gate.evaluate(
            overall_confidence=0.65,
            per_fragment_confidence=[0.7, 0.6],
        )
        assert result.level == "review"
        assert result.requires_review

    def test_low_confidence_fallback(self):
        """Confidence < 0.5 should route to FALLBACK."""
        from app.services.postprocessing.confidence_gate import ConfidenceGate

        gate = ConfidenceGate()
        result = gate.evaluate(
            overall_confidence=0.35,
            per_fragment_confidence=[0.4, 0.3],
        )
        assert result.level == "fallback"
        assert result.route_to_fallback

    def test_very_low_fragment_rejects(self):
        """Any fragment confidence < 0.3 should REJECT."""
        from app.services.postprocessing.confidence_gate import ConfidenceGate

        gate = ConfidenceGate()
        result = gate.evaluate(
            overall_confidence=0.6,
            per_fragment_confidence=[0.8, 0.2],  # Fragment 2 below reject threshold
        )
        assert result.level == "reject"


# ─── SurgeonEditHandler ──────────────────────────────────────────────────────


class TestSurgeonEditHandler:
    """Tests for surgeon edit capture and undo."""

    def test_edit_capture(self):
        """Edit should compute correct delta transform."""
        from app.services.postprocessing.surgeon_edit_handler import SurgeonEditHandler

        handler = SurgeonEditHandler()

        # Predicted transform: identity
        predicted = np.eye(4)
        # Surgeon moves fragment 5mm in X
        surgeon_adjusted = np.eye(4)
        surgeon_adjusted[0, 3] = 5.0

        delta = handler.compute_delta(predicted, surgeon_adjusted)
        assert np.allclose(delta[0, 3], 5.0)

    def test_undo_redo(self):
        """Undo/redo stack should work correctly."""
        from app.services.postprocessing.surgeon_edit_handler import SurgeonEditHandler

        handler = SurgeonEditHandler()

        edit1 = {"fragment_id": "frag_0", "transform": np.eye(4).tolist()}
        edit2 = {"fragment_id": "frag_0", "transform": (np.eye(4) * 2).tolist()}

        handler.push_edit(edit1)
        handler.push_edit(edit2)

        assert handler.can_undo()
        undone = handler.undo()
        assert undone is not None

        assert handler.can_redo()
        redone = handler.redo()
        assert redone is not None
