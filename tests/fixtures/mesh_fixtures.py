"""
Geometric mesh test fixtures for Facial Align unit and integration tests.

All objects use a simple duck-typed MeshLike container with .vertices (Nx3
float32 ndarray) and .faces (Mx3 int32 ndarray) attributes so that tests
remain independent of trimesh or any specific mesh library.

Anatomical proxies are generated parametrically and are intentionally simple
— accuracy is not the goal; exercising the processing pipelines is.

Usage:
    from tests.fixtures.mesh_fixtures import (
        make_cube_mesh,
        make_sphere_mesh,
        make_mandible_proxy,
        make_fragment_pair,
        make_degenerate_mesh,
        make_point_cloud,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Lightweight mesh container (no trimesh dependency)
# ---------------------------------------------------------------------------

@dataclass
class SimpleMesh:
    """
    Minimal triangle-mesh representation for testing purposes.

    Attributes
    ----------
    vertices : np.ndarray, shape (N, 3), float32
        XYZ vertex positions in millimeters.
    faces : np.ndarray, shape (M, 3), int32
        Triangular face definitions (zero-indexed vertex references).
    name : str
        Human-readable mesh identifier.
    metadata : dict
        Arbitrary metadata attached during construction (normals, curvature, etc.)
    """
    vertices: np.ndarray
    faces: np.ndarray
    name: str = "mesh"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float32)
        self.faces = np.asarray(self.faces, dtype=np.int32)

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def face_count(self) -> int:
        return len(self.faces)

    @property
    def centroid(self) -> np.ndarray:
        return self.vertices.mean(axis=0)

    @property
    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (min_xyz, max_xyz) bounding box corners."""
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    @property
    def volume_estimate_mm3(self) -> float:
        """Rough volume estimate via bounding-box (not watertight)."""
        lo, hi = self.bounding_box
        dims = hi - lo
        return float(dims[0] * dims[1] * dims[2])

    def face_areas(self) -> np.ndarray:
        """Return per-face areas in mm²."""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        return 0.5 * np.linalg.norm(cross, axis=1)

    @property
    def surface_area_mm2(self) -> float:
        return float(self.face_areas().sum())


# ---------------------------------------------------------------------------
# Cube mesh
# ---------------------------------------------------------------------------

def make_cube_mesh(size: float = 10.0) -> SimpleMesh:
    """
    Build a closed cube mesh centred at the origin.

    Parameters
    ----------
    size : float
        Edge length in millimetres. Default 10 mm.

    Returns
    -------
    SimpleMesh
        12 triangles, 8 unique vertices.
    """
    h = size / 2.0
    vertices = np.array([
        [-h, -h, -h], [ h, -h, -h], [ h,  h, -h], [-h,  h, -h],
        [-h, -h,  h], [ h, -h,  h], [ h,  h,  h], [-h,  h,  h],
    ], dtype=np.float32)

    # Two triangles per face × 6 faces = 12 triangles
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom   −Z
        [4, 6, 5], [4, 7, 6],  # top      +Z
        [0, 5, 1], [0, 4, 5],  # front    −Y
        [3, 2, 6], [3, 6, 7],  # back     +Y
        [0, 3, 7], [0, 7, 4],  # left     −X
        [1, 5, 6], [1, 6, 2],  # right    +X
    ], dtype=np.int32)

    return SimpleMesh(vertices=vertices, faces=faces, name=f"cube_{size}mm")


# ---------------------------------------------------------------------------
# Sphere mesh (UV parameterisation)
# ---------------------------------------------------------------------------

def make_sphere_mesh(radius: float = 5.0, resolution: int = 20) -> SimpleMesh:
    """
    Build a UV-parameterised sphere mesh.

    Parameters
    ----------
    radius : float
        Sphere radius in mm.
    resolution : int
        Number of latitude (and longitude) divisions. Higher = smoother.

    Returns
    -------
    SimpleMesh
        Approximately 2 × resolution² triangular faces.
    """
    vertices = []
    faces = []

    for lat in range(resolution + 1):
        theta = math.pi * lat / resolution          # 0 → π
        for lon in range(resolution):
            phi = 2 * math.pi * lon / resolution    # 0 → 2π
            x = radius * math.sin(theta) * math.cos(phi)
            y = radius * math.sin(theta) * math.sin(phi)
            z = radius * math.cos(theta)
            vertices.append([x, y, z])

    # Build faces by connecting adjacent latitude rings
    for lat in range(resolution):
        for lon in range(resolution):
            v0 = lat * resolution + lon
            v1 = lat * resolution + (lon + 1) % resolution
            v2 = (lat + 1) * resolution + (lon + 1) % resolution
            v3 = (lat + 1) * resolution + lon
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    return SimpleMesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
        name=f"sphere_r{radius}",
    )


# ---------------------------------------------------------------------------
# Mandible proxy
# ---------------------------------------------------------------------------

def make_mandible_proxy(
    arch_width_mm: float = 60.0,
    arch_depth_mm: float = 40.0,
    ramus_height_mm: float = 55.0,
    ramus_width_mm: float = 15.0,
    resolution: int = 24,
) -> SimpleMesh:
    """
    Parametric mandible-shaped mesh (U-shaped arch + bilateral rami).

    The body is a semi-elliptic tube in the XY plane.  Each ramus is a
    rectangular prism ascending in Z from the posterior arch ends.  This is
    sufficient to test fragment splitting, surface registration, and symmetry
    assessments without requiring a real CT-derived mesh.

    Parameters
    ----------
    arch_width_mm : float
        Full width (inter-condyle) of the dental arch in mm.
    arch_depth_mm : float
        Anterior-posterior depth of the dental arch in mm.
    ramus_height_mm : float
        Height of the vertical rami in mm.
    ramus_width_mm : float
        Thickness of each ramus in mm.
    resolution : int
        Number of segments around the arch semicircle.

    Returns
    -------
    SimpleMesh
        Coarse mandible proxy mesh.
    """
    vertices = []
    faces = []
    idx = 0

    a = arch_width_mm / 2.0     # semi-major (lateral)
    b = arch_depth_mm            # semi-minor (AP)
    tube_r = 4.0                 # tube cross-section radius

    # --- Dental arch (semi-ellipse tube) ------------------------------------
    # Sample the semi-ellipse from θ=0 (right) → θ=π (left), inferior view
    arch_verts_outer = []
    arch_verts_inner = []

    for i in range(resolution + 1):
        theta = math.pi * i / resolution
        cx = -a * math.cos(theta)
        cy = -b * math.sin(theta)

        # Normal to arch centreline (approximate in-plane)
        if i < resolution:
            theta_next = math.pi * (i + 1) / resolution
            dcx = a * math.sin(theta_next) - a * math.sin(theta)
            dcy = -b * math.cos(theta_next) + b * math.cos(theta)
            length = math.sqrt(dcx ** 2 + dcy ** 2) or 1e-6
            nx, ny = -dcy / length, dcx / length
        else:
            nx, ny = arch_verts_outer[-1][0] - cx, arch_verts_outer[-1][1] - cy
            ln = math.sqrt(nx ** 2 + ny ** 2) or 1e-6
            nx, ny = nx / ln, ny / ln

        arch_verts_outer.append([cx + nx * tube_r, cy + ny * tube_r, 0.0])
        arch_verts_inner.append([cx - nx * tube_r, cy - ny * tube_r, 0.0])

    # Bottom face (z=0) and top face (z=tube_r*2)
    n_arch = len(arch_verts_outer)
    for v in arch_verts_outer:
        vertices.append([v[0], v[1], 0.0])       # outer bottom
    for v in arch_verts_outer:
        vertices.append([v[0], v[1], tube_r * 2]) # outer top
    for v in arch_verts_inner:
        vertices.append([v[0], v[1], 0.0])        # inner bottom
    for v in arch_verts_inner:
        vertices.append([v[0], v[1], tube_r * 2]) # inner top

    ob = 0
    ot = n_arch
    ib = 2 * n_arch
    it_ = 3 * n_arch

    for i in range(n_arch - 1):
        # outer side
        faces.append([ob + i, ob + i + 1, ot + i + 1])
        faces.append([ob + i, ot + i + 1, ot + i])
        # inner side (reversed winding)
        faces.append([ib + i, it_ + i + 1, ib + i + 1])
        faces.append([ib + i, it_ + i, it_ + i + 1])
        # top cap
        faces.append([ot + i, ot + i + 1, it_ + i + 1])
        faces.append([ot + i, it_ + i + 1, it_ + i])

    idx = 4 * n_arch

    # --- Rami (simple boxes at arch endpoints) -------------------------------
    def add_ramus_box(base_x: float, base_y: float, flip: int = 1) -> None:
        nonlocal idx
        rx = ramus_width_mm / 2.0
        ry = ramus_width_mm / 2.0
        box_v = np.array([
            [base_x - rx, base_y - ry, 0.0],
            [base_x + rx * flip, base_y - ry, 0.0],
            [base_x + rx * flip, base_y + ry, 0.0],
            [base_x - rx, base_y + ry, 0.0],
            [base_x - rx, base_y - ry, ramus_height_mm],
            [base_x + rx * flip, base_y - ry, ramus_height_mm],
            [base_x + rx * flip, base_y + ry, ramus_height_mm],
            [base_x - rx, base_y + ry, ramus_height_mm],
        ], dtype=np.float32)
        box_f = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 5, 1], [0, 4, 5],
            [3, 2, 6], [3, 6, 7],
            [0, 3, 7], [0, 7, 4],
            [1, 5, 6], [1, 6, 2],
        ], dtype=np.int32) + idx

        for v in box_v:
            vertices.append(v.tolist())
        for f in box_f:
            faces.append(f.tolist())
        idx += 8

    # Right ramus at posterior right end of arch
    add_ramus_box(arch_verts_outer[0][0], arch_verts_outer[0][1])
    # Left ramus at posterior left end
    add_ramus_box(arch_verts_outer[-1][0], arch_verts_outer[-1][1], flip=-1)

    return SimpleMesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
        name="mandible_proxy",
        metadata={
            "arch_width_mm": arch_width_mm,
            "arch_depth_mm": arch_depth_mm,
            "ramus_height_mm": ramus_height_mm,
        },
    )


# ---------------------------------------------------------------------------
# Fragment pair (complementary split cube)
# ---------------------------------------------------------------------------

def make_fragment_pair(
    size: float = 10.0,
    split_axis: str = "x",
    gap_mm: float = 0.0,
) -> Tuple[SimpleMesh, SimpleMesh]:
    """
    Return two complementary mesh fragments that fit together when the gap is 0.

    The split is an axis-aligned planar cut through the centre of a cube.

    Parameters
    ----------
    size : float
        Cube edge length in mm.
    split_axis : str
        Axis perpendicular to the cut plane: ``"x"``, ``"y"``, or ``"z"``.
    gap_mm : float
        Separation added between the two halves along the split axis (simulates
        fracture displacement).

    Returns
    -------
    (fragment_a, fragment_b) : tuple[SimpleMesh, SimpleMesh]
        ``fragment_a`` is the negative-axis half; ``fragment_b`` the positive half.
    """
    h = size / 2.0
    axis_index = {"x": 0, "y": 1, "z": 2}[split_axis]

    full_vertices = np.array([
        [-h, -h, -h], [ h, -h, -h], [ h,  h, -h], [-h,  h, -h],
        [-h, -h,  h], [ h, -h,  h], [ h,  h,  h], [-h,  h,  h],
    ], dtype=np.float32)

    half_a_mask = full_vertices[:, axis_index] <= 0.0
    half_b_mask = full_vertices[:, axis_index] >= 0.0

    def _half_mesh(mask: np.ndarray, shift: float, name: str) -> SimpleMesh:
        # Build a minimal mesh from the masked vertices + a centre face at split plane
        selected = full_vertices.copy()
        selected[mask, axis_index] -= gap_mm / 2.0
        selected[~mask, axis_index] += gap_mm / 2.0

        # Only keep vertices on one side (re-use cube topology, accept degenerate faces)
        all_faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 5, 1], [0, 4, 5],
            [3, 2, 6], [3, 6, 7],
            [0, 3, 7], [0, 7, 4],
            [1, 5, 6], [1, 6, 2],
        ], dtype=np.int32)

        # Keep faces where at least two vertices belong to this half
        keep = []
        for tri in all_faces:
            in_half = sum(mask[v] for v in tri)
            if in_half >= 2:
                keep.append(tri)

        return SimpleMesh(
            vertices=selected,
            faces=np.array(keep, dtype=np.int32),
            name=name,
        )

    a = _half_mesh(half_a_mask, -gap_mm / 2.0, f"fragment_a_{split_axis}")
    b = _half_mesh(half_b_mask,  gap_mm / 2.0, f"fragment_b_{split_axis}")
    return a, b


# ---------------------------------------------------------------------------
# Degenerate mesh
# ---------------------------------------------------------------------------

def make_degenerate_mesh() -> SimpleMesh:
    """
    Construct a mesh with known pathological features for robustness testing:

    - Zero-area (degenerate) triangles (two shared vertices)
    - Duplicate vertices
    - Non-manifold edge (three faces sharing one edge)
    - Isolated vertex (referenced by no face)

    Returns
    -------
    SimpleMesh
        Mesh with known issues; processing code must handle or report them.
    """
    vertices = np.array([
        [0.0, 0.0, 0.0],   # 0
        [1.0, 0.0, 0.0],   # 1
        [0.0, 1.0, 0.0],   # 2
        [1.0, 1.0, 0.0],   # 3
        [0.5, 0.5, 0.0],   # 4  degenerate: same plane as 0,1,2
        [0.0, 0.0, 0.0],   # 5  duplicate of vertex 0
        [2.0, 2.0, 5.0],   # 6  isolated — referenced by no face
        [0.0, 0.0, 1.0],   # 7
    ], dtype=np.float32)

    faces = np.array([
        [0, 1, 2],    # valid
        [1, 3, 2],    # valid
        [0, 0, 1],    # degenerate (repeated vertex → zero area)
        [0, 1, 4],    # degenerate (coplanar, near-zero area)
        [0, 1, 7],    # non-manifold edge shared with [0, 1, 2]
        [5, 1, 2],    # duplicate-vertex face (5 == 0 in 3D)
    ], dtype=np.int32)

    return SimpleMesh(
        vertices=vertices,
        faces=faces,
        name="degenerate_mesh",
        metadata={
            "known_issues": [
                "degenerate_faces",
                "non_manifold_edges",
                "duplicate_vertices",
                "isolated_vertices",
            ]
        },
    )


# ---------------------------------------------------------------------------
# Point cloud
# ---------------------------------------------------------------------------

def make_point_cloud(
    n_points: int = 500,
    noise_mm: float = 0.0,
    shape: str = "sphere",
    radius: float = 10.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a random point cloud with optional Gaussian noise.

    Parameters
    ----------
    n_points : int
        Number of points.
    noise_mm : float
        Standard deviation of isotropic Gaussian noise added to each point (mm).
    shape : str
        Base geometry: ``"sphere"`` (surface samples) or ``"box"`` (volume fill).
    radius : float
        Characteristic size of the cloud in mm.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray, shape (n_points, 3), float32
        Point coordinates in mm.
    """
    rng = np.random.default_rng(seed)

    if shape == "sphere":
        # Uniform random points on a sphere surface
        phi = rng.uniform(0, 2 * np.pi, n_points)
        cos_theta = rng.uniform(-1, 1, n_points)
        sin_theta = np.sqrt(1.0 - cos_theta ** 2)
        pts = radius * np.column_stack([
            sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            cos_theta,
        ])
    elif shape == "box":
        pts = rng.uniform(-radius, radius, (n_points, 3))
    else:
        raise ValueError(f"Unsupported shape {shape!r}; choose 'sphere' or 'box'")

    if noise_mm > 0.0:
        pts += rng.normal(0.0, noise_mm, pts.shape)

    return pts.astype(np.float32)
