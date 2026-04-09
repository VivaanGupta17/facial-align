"""
Occlusion-first joint optimizer for simultaneous dental occlusion
AND fracture section fitting.

Per PMC11574221: "Optimized Jawbone-Reduction Model for Mandibular Fracture Surgery"
— uses gradient-based optimization (Adam) on a composite objective that balances
dental landmark distance, molar relation, midline deviation, fracture fitting,
and overlap penalties.

References:
- PMC11574221: Objective function with 6 weighted terms
- arxiv 2410.20806: BVH collision detection + occlusal projection overlap
- arxiv 2312.15139 (TADPM): SE(3) transform prediction, arch curve Fréchet distance
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points
from pytorch3d.transforms import rotation_6d_to_matrix

logger = logging.getLogger(__name__)


# ─── Optimization result ────────────────────────────────────────────────────


@dataclass
class JointOptimizationResult:
    """
    Result of joint fracture + occlusion optimization.

    Per PMC11574221: returns per-segment transforms, convergence info,
    and a full breakdown of each loss component.
    """
    segment_transforms: Dict[str, np.ndarray]  # fragment_id → 4x4 transform
    convergence_metrics: Dict[str, float] = field(default_factory=dict)
    loss_breakdown: Dict[str, float] = field(default_factory=dict)
    optimization_steps: int = 0
    converged: bool = False
    total_time_ms: int = 0
    final_total_loss: float = float("inf")


# ─── Fracture fitting loss ──────────────────────────────────────────────────


class FractureFittingLoss(nn.Module):
    """
    Chamfer distance between fracture surfaces of adjacent fragments.

    Measures how well the fracture edges of two fragments align when
    brought together. Lower = better anatomical reduction.

    Uses pytorch3d.loss.chamfer_distance.
    """

    def forward(
        self,
        frag_a_surface: torch.Tensor,
        frag_b_surface: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            frag_a_surface: (B, N, 3) fracture surface points of fragment A.
            frag_b_surface: (B, M, 3) fracture surface points of fragment B.

        Returns:
            Scalar fracture fitting loss.
        """
        loss, _ = chamfer_distance(frag_a_surface, frag_b_surface)
        return loss


class FractureOverlapLoss(nn.Module):
    """
    Penalizes overlap (interpenetration) between fracture fragments.

    Per PMC11574221: fragments should fit together without overlapping.
    Uses knn_points to detect and penalize points that are inside
    the opposing fragment.
    """

    def __init__(self, threshold_mm: float = 0.5) -> None:
        super().__init__()
        self.threshold = threshold_mm

    def forward(
        self,
        frag_a_points: torch.Tensor,
        frag_b_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            frag_a_points: (B, N, 3) all surface points of fragment A.
            frag_b_points: (B, M, 3) all surface points of fragment B.

        Returns:
            Scalar overlap penalty.
        """
        knn_result = knn_points(frag_a_points, frag_b_points, K=1)
        dists = knn_result.dists.squeeze(-1).sqrt()  # (B, N)

        # Penalize points closer than threshold
        penetration = torch.relu(self.threshold - dists)
        return penetration.pow(2).mean()


# ─── Per-segment transform parameterization ─────────────────────────────────


class SegmentTransformParams(nn.Module):
    """
    Learnable transform parameters for mandible segments.

    Per PMC11574221: 6 DoF per segment (3 rotation + 3 translation).
    Uses 6D continuous rotation representation for smooth optimization.

    Each segment (e.g., left body, right body, symphysis) gets independent
    transform parameters that are optimized jointly.
    """

    def __init__(self, n_segments: int) -> None:
        super().__init__()
        self.n_segments = n_segments

        # 6D rotation representation (initialized to identity [1,0,0,0,1,0])
        rot_init = torch.zeros(n_segments, 6)
        rot_init[:, 0] = 1.0
        rot_init[:, 4] = 1.0
        self.rotation_6d = nn.Parameter(rot_init)

        # Translation (initialized to zero)
        self.translation = nn.Parameter(torch.zeros(n_segments, 3))

    def get_transforms(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get current rotation matrices and translations.

        Returns:
            rotation: (N, 3, 3) rotation matrices.
            translation: (N, 3) translation vectors in mm.
        """
        R = rotation_6d_to_matrix(self.rotation_6d)  # (N, 3, 3)
        return R, self.translation

    def apply_to_points(
        self,
        segment_idx: int,
        points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply this segment's transform to a point cloud.

        Args:
            segment_idx: Which segment's transform to apply.
            points: (N, 3) point cloud.

        Returns:
            (N, 3) transformed points.
        """
        R = rotation_6d_to_matrix(self.rotation_6d[segment_idx].unsqueeze(0))  # (1, 3, 3)
        t = self.translation[segment_idx]  # (3,)
        return (points @ R.squeeze(0).T) + t

    def to_4x4_matrices(self) -> List[np.ndarray]:
        """Convert current parameters to list of 4x4 numpy matrices."""
        R, t = self.get_transforms()
        matrices = []
        for i in range(self.n_segments):
            T = np.eye(4)
            T[:3, :3] = R[i].detach().cpu().numpy()
            T[:3, 3] = t[i].detach().cpu().numpy()
            matrices.append(T)
        return matrices


# ─── Joint optimizer ────────────────────────────────────────────────────────


class OcclusionFirstJointOptimizer:
    """
    Simultaneously optimizes dental occlusion AND fracture section fitting.

    Per PMC11574221 objective function:
    f(θ) = w1 * landmark_dist
         + w2 * molar_relation
         + w3 * midline_deviation
         + w4 * fracture_fitting
         + w5 * dental_overlap
         + w6 * fracture_overlap

    Uses gradient-based optimization (Adam) on differentiable losses.
    Per-segment: 6 DoF (3 rotation + 3 translation) per mandible segment.

    The key insight from PMC11574221 is that fracture reduction should be
    OCCLUSION-FIRST: dental alignment quality is the primary objective,
    with fracture fitting as a secondary (but important) constraint.
    """

    def __init__(
        self,
        w_landmark: float = 2.0,
        w_molar: float = 2.0,
        w_midline: float = 1.5,
        w_fracture_fit: float = 1.0,
        w_dental_overlap: float = 3.0,
        w_fracture_overlap: float = 3.0,
        lr: float = 1e-3,
        max_steps: int = 1000,
        convergence_threshold: float = 1e-6,
        device: str = "cpu",
    ) -> None:
        self.weights = {
            "landmark": w_landmark,
            "molar": w_molar,
            "midline": w_midline,
            "fracture_fit": w_fracture_fit,
            "dental_overlap": w_dental_overlap,
            "fracture_overlap": w_fracture_overlap,
        }
        self.lr = lr
        self.max_steps = max_steps
        self.convergence_threshold = convergence_threshold
        self.device = device

        # Loss modules
        self.fracture_fitting_loss = FractureFittingLoss()
        self.fracture_overlap_loss = FractureOverlapLoss()

    def optimize(
        self,
        fragment_points: Dict[str, np.ndarray],
        fragment_is_reference: Dict[str, bool],
        upper_dental_points: Optional[np.ndarray] = None,
        lower_dental_points: Optional[np.ndarray] = None,
        dental_landmarks_upper: Optional[np.ndarray] = None,
        dental_landmarks_lower: Optional[np.ndarray] = None,
        upper_midline: Optional[np.ndarray] = None,
        lower_midline: Optional[np.ndarray] = None,
        upper_molar_landmarks: Optional[np.ndarray] = None,
        lower_molar_landmarks: Optional[np.ndarray] = None,
        fracture_surface_pairs: Optional[List[Tuple[str, str]]] = None,
        initial_transforms: Optional[Dict[str, np.ndarray]] = None,
    ) -> JointOptimizationResult:
        """
        Run joint optimization of dental occlusion + fracture fitting.

        Args:
            fragment_points: Dict of fragment_id → (N, 3) point cloud.
            fragment_is_reference: Dict of fragment_id → bool (reference stays fixed).
            upper_dental_points: (N, 3) upper arch surface points (optional).
            lower_dental_points: (M, 3) lower arch surface points (optional).
            dental_landmarks_upper: (K, 3) upper dental landmarks (optional).
            dental_landmarks_lower: (K, 3) lower dental landmarks (optional).
            upper_midline: (3,) upper midline point (optional).
            lower_midline: (3,) lower midline point (optional).
            upper_molar_landmarks: (K, 3) upper molar landmarks (optional).
            lower_molar_landmarks: (K, 3) lower molar landmarks (optional).
            fracture_surface_pairs: List of (frag_a_id, frag_b_id) pairs to fit.
            initial_transforms: Optional initial transforms from ICP/ML.

        Returns:
            JointOptimizationResult with optimized transforms and metrics.
        """
        start_time = time.perf_counter()
        device = self.device

        # Identify movable (non-reference) segments
        movable_ids = [
            fid for fid, is_ref in fragment_is_reference.items() if not is_ref
        ]
        n_segments = len(movable_ids)

        if n_segments == 0:
            logger.warning("No movable segments — returning identity transforms")
            return JointOptimizationResult(
                segment_transforms={
                    fid: np.eye(4) for fid in fragment_points
                },
                converged=True,
            )

        # Initialize transform parameters
        transform_params = SegmentTransformParams(n_segments).to(device)

        # Apply initial transforms if provided
        if initial_transforms:
            with torch.no_grad():
                for i, fid in enumerate(movable_ids):
                    if fid in initial_transforms:
                        T = initial_transforms[fid]
                        R = torch.tensor(T[:3, :3], dtype=torch.float32)
                        # Convert 3x3 to 6D: take first two columns
                        rot_6d = torch.cat([R[:, 0], R[:, 1]])
                        transform_params.rotation_6d.data[i] = rot_6d
                        transform_params.translation.data[i] = torch.tensor(
                            T[:3, 3], dtype=torch.float32
                        )

        # Convert point clouds to tensors
        frag_tensors = {
            fid: torch.tensor(pts, dtype=torch.float32, device=device)
            for fid, pts in fragment_points.items()
        }

        # Optimizer
        optimizer = torch.optim.Adam(transform_params.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=50, factor=0.5,
        )

        # Optimization loop
        loss_history = []
        prev_loss = float("inf")

        for step in range(self.max_steps):
            optimizer.zero_grad()

            # Apply transforms to movable fragments
            transformed_frags = {}
            for fid in fragment_points:
                if fid in movable_ids:
                    idx = movable_ids.index(fid)
                    transformed_frags[fid] = transform_params.apply_to_points(
                        idx, frag_tensors[fid]
                    )
                else:
                    transformed_frags[fid] = frag_tensors[fid]

            total_loss, loss_dict = self._compute_total_loss(
                transformed_frags=transformed_frags,
                upper_dental=upper_dental_points,
                lower_dental=lower_dental_points,
                dental_landmarks_upper=dental_landmarks_upper,
                dental_landmarks_lower=dental_landmarks_lower,
                upper_midline=upper_midline,
                lower_midline=lower_midline,
                upper_molar_landmarks=upper_molar_landmarks,
                lower_molar_landmarks=lower_molar_landmarks,
                fracture_pairs=fracture_surface_pairs,
            )

            total_loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(transform_params.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step(total_loss.item())

            loss_val = total_loss.item()
            loss_history.append(loss_val)

            # Convergence check
            if abs(prev_loss - loss_val) < self.convergence_threshold:
                logger.info(
                    "Joint optimization converged at step %d (loss=%.6f)", step, loss_val
                )
                break

            prev_loss = loss_val

            if step % 100 == 0:
                logger.debug(
                    "Step %d: total_loss=%.4f, components=%s",
                    step, loss_val,
                    {k: f"{v:.4f}" for k, v in loss_dict.items()},
                )

        # Build result
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        # Extract final transforms
        final_transforms: Dict[str, np.ndarray] = {}
        transform_matrices = transform_params.to_4x4_matrices()
        for i, fid in enumerate(movable_ids):
            final_transforms[fid] = transform_matrices[i]
        for fid in fragment_points:
            if fid not in final_transforms:
                final_transforms[fid] = np.eye(4)

        # Loss breakdown (detach for logging)
        final_loss_dict = {k: float(v) for k, v in loss_dict.items()}

        return JointOptimizationResult(
            segment_transforms=final_transforms,
            convergence_metrics={
                "initial_loss": loss_history[0] if loss_history else 0.0,
                "final_loss": loss_history[-1] if loss_history else 0.0,
                "loss_reduction_pct": (
                    (1.0 - loss_history[-1] / (loss_history[0] + 1e-8)) * 100
                    if loss_history else 0.0
                ),
                "learning_rate": optimizer.param_groups[0]["lr"],
            },
            loss_breakdown=final_loss_dict,
            optimization_steps=step + 1,
            converged=step + 1 < self.max_steps,
            total_time_ms=elapsed_ms,
            final_total_loss=loss_history[-1] if loss_history else float("inf"),
        )

    def _compute_total_loss(
        self,
        transformed_frags: Dict[str, torch.Tensor],
        upper_dental: Optional[np.ndarray],
        lower_dental: Optional[np.ndarray],
        dental_landmarks_upper: Optional[np.ndarray],
        dental_landmarks_lower: Optional[np.ndarray],
        upper_midline: Optional[np.ndarray],
        lower_midline: Optional[np.ndarray],
        upper_molar_landmarks: Optional[np.ndarray],
        lower_molar_landmarks: Optional[np.ndarray],
        fracture_pairs: Optional[List[Tuple[str, str]]],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the PMC11574221 composite objective.

        f(θ) = w1*landmark + w2*molar + w3*midline + w4*frac_fit + w5*dental_overlap + w6*frac_overlap
        """
        device = self.device
        total = torch.tensor(0.0, device=device, requires_grad=True)
        loss_dict: Dict[str, float] = {}

        # ── Dental landmark distance loss ──
        if dental_landmarks_upper is not None and dental_landmarks_lower is not None:
            upper_lm = torch.tensor(
                dental_landmarks_upper, dtype=torch.float32, device=device
            ).unsqueeze(0)
            lower_lm = torch.tensor(
                dental_landmarks_lower, dtype=torch.float32, device=device
            ).unsqueeze(0)
            lm_loss, _ = chamfer_distance(upper_lm, lower_lm)
            loss_dict["landmark"] = lm_loss.item()
            total = total + self.weights["landmark"] * lm_loss

        # ── Molar relation loss ──
        if upper_molar_landmarks is not None and lower_molar_landmarks is not None:
            upper_mol = torch.tensor(
                upper_molar_landmarks, dtype=torch.float32, device=device
            ).unsqueeze(0)
            lower_mol = torch.tensor(
                lower_molar_landmarks, dtype=torch.float32, device=device
            ).unsqueeze(0)
            mol_loss, _ = chamfer_distance(upper_mol, lower_mol)
            loss_dict["molar"] = mol_loss.item()
            total = total + self.weights["molar"] * mol_loss

        # ── Midline deviation loss ──
        if upper_midline is not None and lower_midline is not None:
            um = torch.tensor(upper_midline, dtype=torch.float32, device=device)
            lm = torch.tensor(lower_midline, dtype=torch.float32, device=device)
            mid_loss = (um[:2] - lm[:2]).pow(2).sum()  # XY deviation
            loss_dict["midline"] = mid_loss.item()
            total = total + self.weights["midline"] * mid_loss

        # ── Fracture fitting loss ──
        if fracture_pairs:
            fit_total = torch.tensor(0.0, device=device)
            for fid_a, fid_b in fracture_pairs:
                if fid_a in transformed_frags and fid_b in transformed_frags:
                    pts_a = transformed_frags[fid_a].unsqueeze(0)
                    pts_b = transformed_frags[fid_b].unsqueeze(0)
                    fit_loss = self.fracture_fitting_loss(pts_a, pts_b)
                    fit_total = fit_total + fit_loss
            if len(fracture_pairs) > 0:
                fit_total = fit_total / len(fracture_pairs)
            loss_dict["fracture_fit"] = fit_total.item()
            total = total + self.weights["fracture_fit"] * fit_total

        # ── Dental overlap (collision) loss ──
        if upper_dental is not None and lower_dental is not None:
            upper_t = torch.tensor(
                upper_dental, dtype=torch.float32, device=device
            ).unsqueeze(0)
            # Collect all transformed lower (mandible) fragment points
            lower_pts_list = []
            for fid, pts in transformed_frags.items():
                if pts.shape[0] > 0:
                    lower_pts_list.append(pts)

            if lower_pts_list:
                all_lower = torch.cat(lower_pts_list, dim=0).unsqueeze(0)
                # Subsample for efficiency
                max_pts = 4096
                if all_lower.shape[1] > max_pts:
                    idx = torch.randperm(all_lower.shape[1], device=device)[:max_pts]
                    all_lower = all_lower[:, idx]
                if upper_t.shape[1] > max_pts:
                    idx = torch.randperm(upper_t.shape[1], device=device)[:max_pts]
                    upper_t = upper_t[:, idx]

                knn_result = knn_points(all_lower, upper_t, K=1)
                dists = knn_result.dists.squeeze(-1).sqrt()
                dental_pen = torch.relu(0.5 - dists).pow(2).mean()
                loss_dict["dental_overlap"] = dental_pen.item()
                total = total + self.weights["dental_overlap"] * dental_pen

        # ── Fracture overlap (interpenetration) loss ──
        frag_ids = list(transformed_frags.keys())
        overlap_total = torch.tensor(0.0, device=device)
        n_pairs = 0
        for i, fid_a in enumerate(frag_ids):
            for fid_b in frag_ids[i + 1:]:
                pts_a = transformed_frags[fid_a].unsqueeze(0)
                pts_b = transformed_frags[fid_b].unsqueeze(0)
                # Subsample for speed
                if pts_a.shape[1] > 2048:
                    idx = torch.randperm(pts_a.shape[1], device=device)[:2048]
                    pts_a = pts_a[:, idx]
                if pts_b.shape[1] > 2048:
                    idx = torch.randperm(pts_b.shape[1], device=device)[:2048]
                    pts_b = pts_b[:, idx]
                overlap_total = overlap_total + self.fracture_overlap_loss(pts_a, pts_b)
                n_pairs += 1

        if n_pairs > 0:
            overlap_total = overlap_total / n_pairs
        loss_dict["fracture_overlap"] = overlap_total.item()
        total = total + self.weights["fracture_overlap"] * overlap_total

        loss_dict["total"] = total.item()
        return total, loss_dict
