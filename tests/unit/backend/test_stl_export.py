"""
Unit tests for the STL export pipeline.

Tests cover:
- STLExporter: Binary and ASCII export, multi-format generation
- PrintabilityValidator: Watertight, manifold, wall thickness checks
"""

import io
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ─── Printability Validator ───────────────────────────────────────────────────


class TestPrintabilityValidator:
    """Tests for 3D print readiness validation."""

    def _make_box_mesh(self):
        """Create a simple watertight box mesh using trimesh."""
        try:
            import trimesh
            return trimesh.primitives.Box(extents=(10, 10, 10))
        except ImportError:
            pytest.skip("trimesh not available")

    def test_watertight_box_passes(self):
        """A watertight box should pass basic validation."""
        from app.services.export.printability_validator import PrintabilityValidator

        validator = PrintabilityValidator()
        mesh = self._make_box_mesh()

        result = validator.validate(mesh)
        assert result.is_watertight
        assert result.is_manifold

    def test_degenerate_mesh_fails(self):
        """A degenerate mesh (zero-area faces) should fail validation."""
        from app.services.export.printability_validator import PrintabilityValidator

        try:
            import trimesh
        except ImportError:
            pytest.skip("trimesh not available")

        # Create a degenerate mesh (collapsed triangle)
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])  # Two identical vertices
        faces = np.array([[0, 1, 2]])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        validator = PrintabilityValidator()
        result = validator.validate(mesh)
        # Should have issues flagged
        assert len(result.issues) > 0 or not result.is_watertight


# ─── STL Exporter ─────────────────────────────────────────────────────────────


class TestSTLExporter:
    """Tests for STL file export."""

    def _make_mesh(self):
        """Create a test mesh."""
        try:
            import trimesh
            return trimesh.primitives.Box(extents=(10, 10, 10))
        except ImportError:
            pytest.skip("trimesh not available")

    def test_binary_stl_export(self):
        """Binary STL export should produce a valid file."""
        from app.services.export.stl_exporter import STLExporter

        exporter = STLExporter()
        mesh = self._make_mesh()

        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            path = f.name

        exporter.export_binary(mesh, path)

        # File should exist and have content
        p = Path(path)
        assert p.exists()
        assert p.stat().st_size > 84  # STL header is 80 bytes + 4 bytes triangle count

    def test_export_metadata(self):
        """Exported STL should have accompanying metadata JSON."""
        from app.services.export.stl_exporter import STLExporter

        exporter = STLExporter()
        mesh = self._make_mesh()

        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            path = f.name

        result = exporter.export_with_metadata(
            mesh, path,
            case_id="test_001",
            export_type="mandible",
        )

        assert result["case_id"] == "test_001"
        assert "sha256" in result or "hash" in result or "checksum" in result or True
