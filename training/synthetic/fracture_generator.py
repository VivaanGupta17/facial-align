"""
DFGM-style synthetic fracture generation from intact mandibles.

Generates training data by:
1. Taking an intact mandible mesh (from normal CT)
2. Defining fracture planes based on epidemiological patterns
3. Splitting the mesh into fragments along fracture planes
4. Applying random displacement + rotation to each fragment
5. Recording the ground truth transform (inverse of applied displacement)

This enables training the supervised model without any manually labelled
data.  FracFormer's DFGM approach achieves 1.85mm accuracy on real cases
when trained entirely on synthetic fractures.

Fracture patterns (epidemiologically weighted):
- Parasymphyseal (30%): median sagittal + lateral body
- Body (25%): lateral body fracture
- Angle (20%): posterior to last molar
- Condylar (15%): subcondylar neck fracture
- Symphysis (10%): midline fracture

References:
- FracFormer (IEEE TMI 2025): DFGM synthetic fracture generation
- Nardi et al. (2020): Mandibular fracture epidemiology and patterns
"""

from __future__ import annotations

import logging
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─── Fracture pattern definitions ─────────────────────────────────────────────

@dataclass
class FracturePattern:
    """Definition of an anatomical fracture pattern."""
    name: str
    probability: float
    num_fragments: int
    plane_offsets_mm: List[float]      # Offsets from midline or reference point
    plane_normals: List[List[float]]   # Normal vectors for cutting planes
    displacement_range_mm: Tuple[float, float] = (2.0, 15.0)
    rotation_range_deg: Tuple[float, float] = (2.0, 20.0)
    comminution_probability: float = 0.1  # Chance of additional small fragments


# Anatomically-informed fracture patterns
FRACTURE_PATTERNS = [
    FracturePattern(
        name="parasymphyseal",
        probability=0.30,
        num_fragments=2,
        plane_offsets_mm=[12.0],           # ~12mm lateral to midline
        plane_normals=[[0.9, 0.0, 0.4]],  # Oblique sagittal
        displacement_range_mm=(3.0, 12.0),
        rotation_range_deg=(3.0, 15.0),
    ),
    FracturePattern(
        name="body",
        probability=0.25,
        num_fragments=2,
        plane_offsets_mm=[25.0],
        plane_normals=[[0.8, 0.0, 0.6]],
        displacement_range_mm=(2.0, 10.0),
        rotation_range_deg=(2.0, 12.0),
    ),
    FracturePattern(
        name="angle",
        probability=0.20,
        num_fragments=2,
        plane_offsets_mm=[35.0],
        plane_normals=[[0.3, 0.0, 0.95]],
        displacement_range_mm=(2.0, 8.0),
        rotation_range_deg=(2.0, 10.0),
    ),
    FracturePattern(
        name="condylar",
        probability=0.15,
        num_fragments=2,
        plane_offsets_mm=[45.0],
        plane_normals=[[0.1, 0.0, 0.99]],
        displacement_range_mm=(1.0, 6.0),
        rotation_range_deg=(5.0, 25.0),
    ),
    FracturePattern(
        name="symphysis",
        probability=0.10,
        num_fragments=2,
        plane_offsets_mm=[0.0],
        plane_normals=[[1.0, 0.0, 0.0]],
        displacement_range_mm=(2.0, 8.0),
        rotation_range_deg=(2.0, 10.0),
    ),
]


@dataclass
class SyntheticFractureCase:
    """A single generated synthetic fracture case."""
    case_id: str
    pattern_name: str
    intact_mesh_path: str
    fragment_meshes: Dict[str, np.ndarray]       # fragment_id → (N, 3) vertices
    fragment_faces: Dict[str, np.ndarray]         # fragment_id → (M, 3) face indices
    applied_transforms: Dict[str, np.ndarray]     # fragment_id → 4x4 displacement SE(3)
    ground_truth_transforms: Dict[str, np.ndarray] # fragment_id → 4x4 correction (inverse)
    fracture_planes: List[Dict[str, Any]]         # Plane definitions used
    metadata: Dict[str, Any] = field(default_factory=dict)


class SyntheticFractureGenerator:
    """
    Generate synthetic mandibular fractures from intact anatomy.

    Usage:
        generator = SyntheticFractureGenerator(seed=42)
        cases = generator.generate_batch(
            intact_meshes=["/data/normals/mandible_001.stl", ...],
            num_cases_per_mesh=10,
        )

    Each generated case contains:
    - Fragment meshes (displaced from original position)
    - Ground truth SE(3) transforms to restore original position
    - Fracture pattern metadata
    - Anatomical landmark positions

    Args:
        seed: Random seed for reproducibility.
        patterns: Custom fracture patterns (default: epidemiological distribution).
    """

    def __init__(
        self,
        seed: int = 42,
        patterns: Optional[List[FracturePattern]] = None,
    ) -> None:
        self.rng = np.random.RandomState(seed)
        self.patterns = patterns or FRACTURE_PATTERNS
        self._validate_patterns()

    def _validate_patterns(self) -> None:
        """Ensure pattern probabilities sum to ~1.0."""
        total = sum(p.probability for p in self.patterns)
        if abs(total - 1.0) > 0.01:
            logger.warning("Fracture pattern probabilities sum to %.2f (expected 1.0)", total)

    def generate_batch(
        self,
        intact_mesh_paths: List[str],
        num_cases_per_mesh: int = 10,
        output_dir: Optional[str] = None,
    ) -> List[SyntheticFractureCase]:
        """
        Generate synthetic fractures from a batch of intact mandibles.

        Args:
            intact_mesh_paths: Paths to intact mandible STL files.
            num_cases_per_mesh: Number of fracture cases per input mesh.
            output_dir: If provided, save generated fragments as STL files.

        Returns:
            List of SyntheticFractureCase objects.
        """
        import trimesh

        cases = []
        for mesh_idx, mesh_path in enumerate(intact_mesh_paths):
            try:
                mesh = trimesh.load(mesh_path, force="mesh")
                logger.info(
                    "Generating %d cases from %s (%d vertices)",
                    num_cases_per_mesh, mesh_path, len(mesh.vertices),
                )
            except Exception as e:
                logger.error("Failed to load %s: %s", mesh_path, e)
                continue

            for case_idx in range(num_cases_per_mesh):
                case_id = f"synthetic_{mesh_idx:04d}_{case_idx:04d}"
                case = self._generate_single(mesh, mesh_path, case_id)
                cases.append(case)

                if output_dir:
                    self._save_case(case, output_dir)

        logger.info("Generated %d synthetic fracture cases", len(cases))
        return cases

    def _generate_single(
        self,
        intact_mesh: "trimesh.Trimesh",
        mesh_path: str,
        case_id: str,
    ) -> SyntheticFractureCase:
        """Generate a single synthetic fracture from an intact mesh."""
        import trimesh

        # Select fracture pattern
        pattern = self._sample_pattern()

        # Compute mesh centre and bounding box for plane placement
        centroid = intact_mesh.centroid
        bounds = intact_mesh.bounds  # (2, 3): min, max

        # Generate fracture planes
        planes = self._generate_fracture_planes(pattern, centroid, bounds)

        # Split mesh along fracture planes
        fragments = self._split_mesh(intact_mesh, planes)

        # Apply random displacements and record ground truth
        applied_transforms = {}
        ground_truth = {}
        displaced_verts = {}
        displaced_faces = {}

        for frag_id, frag_mesh in fragments.items():
            T_displace = self._random_displacement(pattern)
            T_restore = np.linalg.inv(T_displace)

            # Apply displacement to fragment vertices
            verts_homo = np.hstack([
                frag_mesh.vertices,
                np.ones((len(frag_mesh.vertices), 1)),
            ])
            verts_displaced = (T_displace @ verts_homo.T).T[:, :3]

            displaced_verts[frag_id] = verts_displaced
            displaced_faces[frag_id] = frag_mesh.faces
            applied_transforms[frag_id] = T_displace
            ground_truth[frag_id] = T_restore

        return SyntheticFractureCase(
            case_id=case_id,
            pattern_name=pattern.name,
            intact_mesh_path=mesh_path,
            fragment_meshes=displaced_verts,
            fragment_faces=displaced_faces,
            applied_transforms=applied_transforms,
            ground_truth_transforms=ground_truth,
            fracture_planes=[{
                "origin": p["origin"].tolist(),
                "normal": p["normal"].tolist(),
            } for p in planes],
            metadata={
                "pattern": pattern.name,
                "num_fragments": len(fragments),
                "intact_vertices": len(intact_mesh.vertices),
            },
        )

    def _sample_pattern(self) -> FracturePattern:
        """Sample a fracture pattern from the epidemiological distribution."""
        probs = [p.probability for p in self.patterns]
        idx = self.rng.choice(len(self.patterns), p=probs)
        return self.patterns[idx]

    def _generate_fracture_planes(
        self,
        pattern: FracturePattern,
        centroid: np.ndarray,
        bounds: np.ndarray,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Generate cutting planes based on fracture pattern.

        Adds random jitter to the plane position and orientation for variety.
        """
        planes = []
        for offset, normal in zip(pattern.plane_offsets_mm, pattern.plane_normals):
            # Base plane origin: centroid + offset along X axis
            origin = centroid.copy()
            # Randomise side (left vs right)
            side = self.rng.choice([-1, 1])
            origin[0] += side * (offset + self.rng.uniform(-3, 3))

            # Randomise normal direction slightly
            normal_arr = np.array(normal, dtype=np.float64)
            noise = self.rng.normal(0, 0.05, size=3)
            normal_arr = normal_arr * side + noise
            normal_arr = normal_arr / (np.linalg.norm(normal_arr) + 1e-8)

            planes.append({"origin": origin, "normal": normal_arr})

        return planes

    def _split_mesh(
        self,
        mesh: "trimesh.Trimesh",
        planes: List[Dict[str, np.ndarray]],
    ) -> Dict[str, "trimesh.Trimesh"]:
        """
        Split a mesh along fracture planes.

        Uses trimesh's slice_mesh_plane to cut the mesh. For multiple planes,
        applies cuts sequentially.

        Returns dict mapping fragment_id → trimesh.Trimesh.
        """
        import trimesh

        fragments = {}
        remaining = mesh.copy()

        for i, plane in enumerate(planes):
            try:
                # Slice: vertices on positive side of plane go to fragment_a
                frag_pos = remaining.slice_mesh_plane(
                    plane_origin=plane["origin"],
                    plane_normal=plane["normal"],
                    cached_dots=None,
                )
                frag_neg = remaining.slice_mesh_plane(
                    plane_origin=plane["origin"],
                    plane_normal=-plane["normal"],
                    cached_dots=None,
                )

                if frag_pos is not None and len(frag_pos.vertices) > 10:
                    fragments[f"fragment_{i}"] = frag_pos

                if frag_neg is not None and len(frag_neg.vertices) > 10:
                    remaining = frag_neg
                else:
                    remaining = None
                    break

            except Exception as e:
                logger.warning("Mesh split failed at plane %d: %s", i, e)
                break

        # Add the remaining piece
        if remaining is not None and len(remaining.vertices) > 10:
            fragments[f"fragment_{len(planes)}"] = remaining

        # Fallback: if splitting failed, create two fragments by splitting at centroid
        if len(fragments) < 2:
            logger.warning("Plane split produced <2 fragments, using centroid split")
            fragments = self._centroid_split(mesh)

        return fragments

    def _centroid_split(
        self,
        mesh: "trimesh.Trimesh",
    ) -> Dict[str, "trimesh.Trimesh"]:
        """Fallback: split mesh at its centroid along a random axis."""
        import trimesh

        centroid = mesh.centroid
        # Random axis
        axis = self.rng.choice(3)
        normal = np.zeros(3)
        normal[axis] = 1.0

        try:
            frag_a = mesh.slice_mesh_plane(centroid, normal)
            frag_b = mesh.slice_mesh_plane(centroid, -normal)
            return {"fragment_0": frag_a, "fragment_1": frag_b}
        except Exception:
            # Last resort: return the whole mesh as one fragment
            return {"fragment_0": mesh}

    def _random_displacement(self, pattern: FracturePattern) -> np.ndarray:
        """
        Generate a random SE(3) displacement for a fragment.

        Translation: uniform in [min, max] mm along a random direction.
        Rotation: uniform in [min, max] degrees around a random axis.

        Returns:
            4x4 SE(3) displacement matrix.
        """
        from scipy.spatial.transform import Rotation

        # Random translation
        t_mag = self.rng.uniform(*pattern.displacement_range_mm)
        t_dir = self.rng.randn(3)
        t_dir = t_dir / (np.linalg.norm(t_dir) + 1e-8)
        translation = t_dir * t_mag

        # Random rotation
        r_deg = self.rng.uniform(*pattern.rotation_range_deg)
        r_axis = self.rng.randn(3)
        r_axis = r_axis / (np.linalg.norm(r_axis) + 1e-8)
        r_rad = math.radians(r_deg)
        R = Rotation.from_rotvec(r_axis * r_rad).as_matrix()

        # Build SE(3) matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = translation
        return T

    def _save_case(
        self,
        case: SyntheticFractureCase,
        output_dir: str,
    ) -> None:
        """Save a generated case to disk."""
        import trimesh

        case_dir = Path(output_dir) / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        # Save fragment meshes
        for frag_id, verts in case.fragment_meshes.items():
            faces = case.fragment_faces[frag_id]
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            mesh.export(str(case_dir / f"{frag_id}.stl"))

        # Save ground truth transforms
        gt = {}
        for frag_id, T in case.ground_truth_transforms.items():
            gt[frag_id] = T.tolist()

        with open(case_dir / "ground_truth.json", "w") as f:
            json.dump({
                "case_id": case.case_id,
                "pattern": case.pattern_name,
                "ground_truth_transforms": gt,
                "fracture_planes": case.fracture_planes,
                "metadata": case.metadata,
            }, f, indent=2)

        logger.debug("Saved case %s to %s", case.case_id, case_dir)
