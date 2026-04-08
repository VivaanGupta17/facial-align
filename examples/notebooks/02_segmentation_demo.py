"""
Facial Align — Segmentation Pipeline Demo
===========================================

Demonstrates the ML segmentation pipeline using TotalSegmentator
for craniofacial structure segmentation.

Prerequisites:
    pip install totalsegmentator SimpleITK numpy trimesh

Usage:
    python examples/notebooks/02_segmentation_demo.py --nifti /path/to/ct.nii.gz --output-dir ./output
"""

# %% [markdown]
# # Segmentation Pipeline Demo
# 1. Load preprocessed CT volume (NIfTI)
# 2. Run TotalSegmentator for craniofacial structures
# 3. Extract meshes from segmentation masks
# 4. Export GLB meshes for web viewer

# %% Imports
import argparse
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("segmentation_demo")


# %% TotalSegmentator Wrapper
class TotalSegmentatorAdapter:
    """
    Wrapper around TotalSegmentator for craniofacial segmentation.

    Uses the `craniofacial_structures` and `teeth` subtasks for
    comprehensive CMF anatomy segmentation.
    """

    # CMF-relevant subtasks and their outputs
    SUBTASKS = {
        "craniofacial_structures": [
            "mandible",
            "upper_dental_arch",
            "lower_dental_arch",
            "skull",
            "maxillary_sinus_left",
            "maxillary_sinus_right",
            "frontal_sinus",
        ],
        "teeth": [
            f"tooth_{q}{n}"
            for q in [1, 2, 3, 4]
            for n in range(1, 9)
        ],
        "head_muscles": [
            "masseter_left",
            "masseter_right",
            "temporalis_left",
            "temporalis_right",
            "medial_pterygoid_left",
            "medial_pterygoid_right",
            "lateral_pterygoid_left",
            "lateral_pterygoid_right",
        ],
    }

    def __init__(self, device: str = "gpu", fast: bool = False):
        self.device = device
        self.fast = fast
        self._check_installation()

    def _check_installation(self):
        """Verify TotalSegmentator is installed."""
        try:
            import totalsegmentator  # noqa: F401
            logger.info("TotalSegmentator is available")
        except ImportError:
            logger.error(
                "TotalSegmentator not installed. "
                "Install with: pip install totalsegmentator"
            )
            raise

    def segment(
        self,
        input_path: str,
        output_dir: str,
        subtask: str = "craniofacial_structures",
    ) -> dict:
        """
        Run TotalSegmentator segmentation.

        Args:
            input_path: Path to input NIfTI file
            output_dir: Directory for segmentation outputs
            subtask: TotalSegmentator subtask name

        Returns:
            dict with structure names mapped to NIfTI mask paths
        """
        from totalsegmentator.python_api import totalsegmentator

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Running TotalSegmentator subtask: {subtask}")
        logger.info(f"Input: {input_path}")
        logger.info(f"Device: {self.device}")

        start_time = time.time()

        totalsegmentator(
            input=Path(input_path),
            output=output_path,
            task=subtask,
            fast=self.fast,
            device=self.device,
            verbose=True,
        )

        elapsed = time.time() - start_time
        logger.info(f"Segmentation completed in {elapsed:.1f}s")

        # Collect output masks
        results = {}
        for nifti_file in output_path.glob("*.nii.gz"):
            structure_name = nifti_file.stem.replace(".nii", "")
            results[structure_name] = str(nifti_file)

        logger.info(f"Segmented {len(results)} structures")
        return results


# %% Mesh Extraction
def extract_mesh_from_mask(
    mask_path: str,
    label: int = 1,
    smoothing_iterations: int = 15,
    target_faces: int = 50000,
) -> "trimesh.Trimesh":
    """
    Extract a 3D mesh from a binary segmentation mask using marching cubes.

    Args:
        mask_path: Path to NIfTI segmentation mask
        label: Label value to extract (default 1 for binary masks)
        smoothing_iterations: Laplacian smoothing passes
        target_faces: Target face count after simplification

    Returns:
        trimesh.Trimesh object
    """
    import SimpleITK as sitk
    from skimage.measure import marching_cubes
    import trimesh

    # Load mask
    mask_sitk = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(mask_sitk)
    spacing = mask_sitk.GetSpacing()
    origin = mask_sitk.GetOrigin()

    # Binary threshold
    binary_mask = (mask_array == label).astype(np.float32)

    if binary_mask.sum() == 0:
        logger.warning(f"No voxels found for label {label} in {mask_path}")
        return None

    # Apply light Gaussian smoothing before marching cubes
    from scipy.ndimage import gaussian_filter
    smoothed = gaussian_filter(binary_mask, sigma=0.5)

    # Marching cubes
    verts, faces, normals, _ = marching_cubes(
        smoothed, level=0.5, spacing=spacing
    )

    # Offset vertices to physical coordinates
    verts += np.array(origin)

    # Create trimesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

    logger.info(
        f"Raw mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces"
    )

    # Simplify if too many faces
    if len(mesh.faces) > target_faces:
        mesh = mesh.simplify_quadric_decimation(target_faces)
        logger.info(f"Simplified to {len(mesh.faces)} faces")

    # Laplacian smoothing
    trimesh.smoothing.filter_laplacian(mesh, iterations=smoothing_iterations)

    # Compute metrics
    volume = mesh.volume
    area = mesh.area
    logger.info(f"Mesh volume: {volume:.1f} mm³, surface area: {area:.1f} mm²")

    return mesh


def export_glb(mesh, output_path: str, structure_name: str, color: list = None):
    """Export a trimesh as GLB for the web viewer."""
    import trimesh

    if color is None:
        color = [200, 200, 200, 255]  # Default gray

    # Apply color as vertex colors
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        face_colors=np.tile(color, (len(mesh.faces), 1)),
    )

    # Export as GLB
    glb_data = mesh.export(file_type="glb")
    Path(output_path).write_bytes(glb_data)
    logger.info(f"Exported {structure_name} → {output_path} ({len(glb_data)} bytes)")


# %% Structure Colors (standard CMF visualization)
STRUCTURE_COLORS = {
    "mandible": [255, 230, 100, 255],      # Yellow
    "maxilla": [100, 150, 255, 255],        # Blue
    "skull": [240, 230, 220, 200],          # Ivory, semi-transparent
    "teeth": [255, 255, 255, 255],          # White
    "zygomatic": [200, 180, 160, 255],      # Tan
    "masseter": [255, 120, 120, 180],       # Red, semi-transparent
    "temporalis": [255, 140, 140, 180],     # Light red
    "sinus": [100, 200, 255, 120],          # Light blue, transparent
    "default": [200, 200, 200, 255],        # Gray
}

def get_color_for_structure(name: str) -> list:
    """Get visualization color for an anatomical structure."""
    for key, color in STRUCTURE_COLORS.items():
        if key in name.lower():
            return color
    return STRUCTURE_COLORS["default"]


# %% Main Pipeline
def main():
    parser = argparse.ArgumentParser(description="Segmentation Pipeline Demo")
    parser.add_argument("--nifti", type=str, required=True, help="Input NIfTI CT volume")
    parser.add_argument("--output-dir", type=str, default="./segmentation_output")
    parser.add_argument("--device", type=str, default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--fast", action="store_true", help="Use fast (lower res) mode")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    mask_dir = output_dir / "masks"
    mesh_dir = output_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Run segmentation
    logger.info("=" * 60)
    logger.info("Step 1: Running TotalSegmentator")
    logger.info("=" * 60)

    adapter = TotalSegmentatorAdapter(device=args.device, fast=args.fast)
    masks = adapter.segment(args.nifti, str(mask_dir), subtask="craniofacial_structures")

    # Step 2: Extract meshes
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Extracting meshes from masks")
    logger.info("=" * 60)

    mesh_manifest = {}
    for structure_name, mask_path in masks.items():
        logger.info(f"\nProcessing: {structure_name}")
        mesh = extract_mesh_from_mask(mask_path)
        if mesh is not None:
            color = get_color_for_structure(structure_name)
            glb_path = str(mesh_dir / f"{structure_name}.glb")
            export_glb(mesh, glb_path, structure_name, color)
            mesh_manifest[structure_name] = {
                "glb_path": glb_path,
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces),
                "volume_mm3": float(mesh.volume),
                "color": color,
            }

    # Step 3: Write manifest
    import json
    manifest_path = output_dir / "mesh_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(mesh_manifest, f, indent=2)
    logger.info(f"\nMesh manifest written to {manifest_path}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    for name, info in mesh_manifest.items():
        logger.info(f"  {name}: {info['faces']} faces, {info['volume_mm3']:.0f} mm³")
    logger.info(f"\nTotal structures: {len(mesh_manifest)}")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
