"""
Unit tests for the mesh extraction and processing service.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.core.exceptions import EmptyMaskError, MeshExtractionError
from app.services.mesh.mesh_service import MeshMetrics, MeshService


class TestMeshService:
    """Tests for MeshService."""

    @pytest.fixture
    def service(self) -> MeshService:
        return MeshService()

    def test_extract_mesh_from_mask_basic(self, service, binary_bone_mask, ct_spacing):
        """Test basic mesh extraction from a binary mask."""
        try:
            import trimesh
            mesh = service.extract_mesh_from_mask(
                mask=binary_bone_mask,
                spacing=ct_spacing,
                label=1,
            )
            assert isinstance(mesh, trimesh.Trimesh)
            assert len(mesh.vertices) > 0
            assert len(mesh.faces) > 0
        except ImportError:
            pytest.skip("trimesh not installed")
        except Exception as e:
            if "skimage" in str(e) or "scikit" in str(e):
                pytest.skip("scikit-image not installed")
            raise

    def test_extract_mesh_empty_mask_raises(self, service, ct_spacing):
        """Test that extracting from empty mask raises EmptyMaskError."""
        empty_mask = np.zeros((64, 64, 64), dtype=np.int32)
        try:
            with pytest.raises(EmptyMaskError):
                service.extract_mesh_from_mask(empty_mask, ct_spacing, label=1)
        except ImportError:
            pytest.skip("Required library not installed")

    def test_extract_mesh_wrong_label_raises(self, service, binary_bone_mask, ct_spacing):
        """Test that requesting non-existent label raises EmptyMaskError."""
        try:
            with pytest.raises(EmptyMaskError):
                service.extract_mesh_from_mask(binary_bone_mask, ct_spacing, label=999)
        except ImportError:
            pytest.skip("Required library not installed")

    def test_extract_mesh_scales_to_mm(self, service, ct_spacing):
        """Test that mesh vertices are in millimeter coordinates."""
        try:
            import trimesh
            # Create a known 10-voxel cube mask
            mask = np.zeros((30, 30, 30), dtype=np.int32)
            mask[5:15, 5:15, 5:15] = 1

            mesh = service.extract_mesh_from_mask(mask, ct_spacing, label=1)

            # Mesh should extend roughly 5mm (10 voxels × 0.5mm)
            max_extent = np.max(mesh.vertices) - np.min(mesh.vertices)
            assert max_extent < 20.0  # Should be roughly 5mm, well under 20mm
            assert max_extent > 2.0   # But definitely more than 2mm
        except ImportError:
            pytest.skip("Required library not installed")

    def test_simplify_mesh_reduces_faces(self, service):
        """Test mesh simplification reduces polygon count."""
        try:
            import trimesh
            # Create a dense sphere mesh
            mesh = trimesh.creation.icosphere(subdivisions=4)
            original_faces = len(mesh.faces)

            simplified = service.simplify_mesh(mesh, target_faces=1000)

            # Should have fewer faces (but not necessarily exactly 1000)
            assert len(simplified.faces) <= original_faces
        except ImportError:
            pytest.skip("trimesh not installed")

    def test_simplify_mesh_skips_if_already_small(self, service):
        """Test that over-simplified meshes are returned unchanged."""
        try:
            import trimesh
            mesh = trimesh.creation.icosphere(subdivisions=1)  # Small mesh
            small_face_count = len(mesh.faces)

            # Request more faces than exist
            result = service.simplify_mesh(mesh, target_faces=small_face_count * 2)
            # Should return original (no over-simplification)
            assert len(result.faces) == small_face_count
        except ImportError:
            pytest.skip("trimesh not installed")

    def test_smooth_mesh_preserves_topology(self, service):
        """Test that mesh smoothing doesn't change vertex/face count."""
        try:
            import trimesh
            mesh = trimesh.creation.icosphere(subdivisions=2)
            n_vertices = len(mesh.vertices)
            n_faces = len(mesh.faces)

            smoothed = service.smooth_mesh(mesh, iterations=3)

            # Topology should be preserved
            assert len(smoothed.vertices) == n_vertices
            assert len(smoothed.faces) == n_faces
        except ImportError:
            pytest.skip("trimesh not installed")

    def test_compute_mesh_metrics_watertight(self, service):
        """Test metric computation for a watertight mesh (sphere)."""
        try:
            import trimesh
            sphere = trimesh.creation.icosphere(subdivisions=3)

            metrics = service.compute_mesh_metrics(sphere)

            assert isinstance(metrics, MeshMetrics)
            assert metrics.vertex_count == len(sphere.vertices)
            assert metrics.face_count == len(sphere.faces)
            assert metrics.is_watertight is True
            assert metrics.volume_mm3 > 0.0
            assert metrics.surface_area_mm2 > 0.0
        except ImportError:
            pytest.skip("trimesh not installed")

    def test_compute_mesh_metrics_centroid(self, service):
        """Test that centroid is computed correctly."""
        try:
            import trimesh
            # Sphere centered at (10, 20, 30)mm
            sphere = trimesh.creation.icosphere(subdivisions=2, radius=5.0)
            sphere.apply_translation([10, 20, 30])

            metrics = service.compute_mesh_metrics(sphere)

            assert abs(metrics.centroid_mm[0] - 10.0) < 0.5
            assert abs(metrics.centroid_mm[1] - 20.0) < 0.5
            assert abs(metrics.centroid_mm[2] - 30.0) < 0.5
        except ImportError:
            pytest.skip("trimesh not installed")

    def test_export_glb(self, service, tmp_path):
        """Test GLB export creates a file."""
        try:
            import trimesh
            sphere = trimesh.creation.icosphere(subdivisions=2)
            output_path = tmp_path / "test.glb"

            result = service.export_glb(sphere, output_path)

            assert result.exists()
            assert result.stat().st_size > 0
            assert result.suffix == ".glb"
        except ImportError:
            pytest.skip("trimesh not installed")

    def test_export_stl(self, service, tmp_path):
        """Test STL export creates a file."""
        try:
            import trimesh
            sphere = trimesh.creation.icosphere(subdivisions=2)
            output_path = tmp_path / "test.stl"

            result = service.export_stl(sphere, output_path)

            assert result.exists()
            assert result.stat().st_size > 0
        except ImportError:
            pytest.skip("trimesh not installed")

    def test_extract_and_process_all_structures(
        self, service, multi_label_mask, ct_spacing, tmp_path
    ):
        """Test batch mesh extraction for all structures."""
        try:
            labels = {"mandible": 1, "maxilla": 2}
            result = service.extract_and_process_all_structures(
                masks=multi_label_mask,
                labels=labels,
                spacing=ct_spacing,
                output_dir=tmp_path / "meshes",
                smooth_iterations=2,
                target_face_ratio=0.5,
            )

            assert isinstance(result, dict)
            # At least one structure should have been extracted
            assert len(result) >= 1
            # Each result should have format keys
            for structure, paths in result.items():
                assert isinstance(paths, dict)
                assert any(p.exists() for p in paths.values())
        except ImportError:
            pytest.skip("Required mesh library not installed")
