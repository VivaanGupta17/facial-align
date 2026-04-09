"""
Complete loss module for training the supervised facial alignment model.

Combines supervised losses (geodesic SO(3) rotation + L2 translation), self-supervised
dental losses (Chamfer, overlap, uniformity, collision), clinical metric losses
(overjet, overbite, midline), and regularization (arch form, bilateral symmetry).

The total loss is a weighted sum configurable per training phase:
    L_total = w_geo * L_geodesic(R)
            + w_trans * L_L2(t)
            + w_clinical * L_clinical(metrics)
            + w_dental * L_composite_dental
            + w_symmetry * L_symmetry
            + w_evidential * L_evidential(uncertainty)

Uses R6 continuous rotation representation. Geodesic distance on SO(3):
    d(R1, R2) = arccos((tr(R1^T R2) - 1) / 2)

References:
- Huynh, "Metrics for 3D Rotations" (JMIV 2009) — geodesic rotation distance
- Zhou et al., "On the Continuity of Rotation Representations" (CVPR 2019)
- Amini et al., "Deep Evidential Regression" (NeurIPS 2020)
- PMC11574221: Composite objective for CMF surgery
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.services.occlusion.occlusal_losses import CompositeDentalLoss
from app.services.supervised.prediction_heads import r6_to_rotation_matrix_svd

logger = logging.getLogger(__name__)


# ─── Configuration ────────────────────────────────────────────────────────────


@dataclass
class LossConfig:
    """Weights and hyperparameters for the supervised loss."""
    # Supervised transform losses
    w_geodesic: float = 5.0
    w_translation: float = 2.0

    # Clinical metric losses
    w_overjet: float = 1.0
    w_overbite: float = 1.0
    w_midline: float = 1.5
    w_molar_class: float = 1.0

    # Self-supervised dental composite loss
    w_dental_composite: float = 1.0

    # Regularization
    w_symmetry: float = 0.5
    w_arch_form: float = 0.3

    # Uncertainty (evidential)
    w_evidential: float = 0.1
    evidential_lambda: float = 0.05

    # Confidence weighting
    use_confidence_weighting: bool = True
    confidence_floor: float = 0.1


# ─── Geodesic rotation loss ──────────────────────────────────────────────────


class GeodesicRotationLoss(nn.Module):
    """
    Geodesic distance on SO(3) between predicted and target rotation matrices.

    The geodesic distance is the angle of the relative rotation R_rel = R_pred^T @ R_target:
        d(R1, R2) = arccos( clamp( (tr(R1^T R2) - 1) / 2, -1, 1 ) )

    This is the true intrinsic distance on the rotation manifold and is
    preferred over Frobenius norm or quaternion distances because it:
    1. Directly measures the rotation angle (in radians)
    2. Is invariant to the choice of rotation representation
    3. Has well-defined gradients everywhere

    Args:
        eps: Small epsilon for numerical stability in arccos.
        reduction: 'mean', 'sum', or 'none'.

    References:
    - Huynh, "Metrics for 3D Rotations: Comparison and Analysis" (JMIV 2009)
    - Hartley et al., "Rotation Averaging" (IJCV 2013)
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean") -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(
        self,
        R_pred: torch.Tensor,
        R_target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute geodesic distance between predicted and target rotations.

        Args:
            R_pred: (..., 3, 3) predicted rotation matrices.
            R_target: (..., 3, 3) target rotation matrices.
            weights: (...,) optional per-element weights (e.g., confidence scores).

        Returns:
            Scalar geodesic loss (or per-element if reduction='none').
        """
        # Relative rotation: R_rel = R_pred^T @ R_target
        R_rel = torch.matmul(R_pred.transpose(-1, -2), R_target)

        # Trace of relative rotation
        trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]

        # Geodesic angle: arccos((tr(R_rel) - 1) / 2)
        cos_angle = (trace - 1.0) / 2.0
        cos_angle = cos_angle.clamp(-1.0 + self.eps, 1.0 - self.eps)
        angle = torch.acos(cos_angle)  # in [0, pi]

        if weights is not None:
            angle = angle * weights

        if self.reduction == "mean":
            return angle.mean()
        elif self.reduction == "sum":
            return angle.sum()
        return angle


# ─── Translation loss ─────────────────────────────────────────────────────────


class TranslationLoss(nn.Module):
    """
    L2 (Euclidean) distance between predicted and target translations.

    Measures the displacement error in mm between where the model predicts
    a fragment/tooth should move and where it should actually go.

    Supports optional per-element weighting (e.g., by fragment confidence)
    and Huber smoothing for robustness to outliers.

    Args:
        use_huber: If True, uses Smooth L1 (Huber) loss instead of L2.
        huber_delta: Threshold for Huber loss transition from L2 to L1.
        reduction: 'mean', 'sum', or 'none'.

    References:
    - Huber, "Robust Statistics" (1981) — Huber loss
    """

    def __init__(
        self, use_huber: bool = False, huber_delta: float = 5.0, reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.use_huber = use_huber
        self.huber_delta = huber_delta
        self.reduction = reduction

    def forward(
        self,
        t_pred: torch.Tensor,
        t_target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute translation loss.

        Args:
            t_pred: (..., 3) predicted translations in mm.
            t_target: (..., 3) target translations in mm.
            weights: (...,) optional per-element weights.

        Returns:
            Scalar translation loss.
        """
        if self.use_huber:
            per_element = F.smooth_l1_loss(
                t_pred, t_target, beta=self.huber_delta, reduction="none",
            )
            loss = per_element.sum(dim=-1)  # sum over xyz
        else:
            diff = t_pred - t_target
            loss = (diff ** 2).sum(dim=-1).sqrt()  # L2 norm per element

        if weights is not None:
            loss = loss * weights

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ─── Clinical metric losses ──────────────────────────────────────────────────


class ClinicalMetricLoss(nn.Module):
    """
    Loss for clinical occlusion metrics (overjet, overbite, midline, molar class).

    Combines:
    - MSE for continuous metrics (overjet, overbite, midline deviation)
    - Cross-entropy for categorical metrics (molar class)

    These losses ensure the model's predictions align with clinically
    meaningful measurements, not just geometric accuracy.

    Args:
        w_overjet: Weight for overjet MSE.
        w_overbite: Weight for overbite MSE.
        w_midline: Weight for midline MSE.
        w_molar_class: Weight for molar class cross-entropy.

    References:
    - PMC11574221: Clinical occlusion objectives for CMF surgery
    """

    def __init__(
        self,
        w_overjet: float = 1.0,
        w_overbite: float = 1.0,
        w_midline: float = 1.5,
        w_molar_class: float = 1.0,
    ) -> None:
        super().__init__()
        self.w_overjet = w_overjet
        self.w_overbite = w_overbite
        self.w_midline = w_midline
        self.w_molar_class = w_molar_class
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        pred_scores: Dict[str, torch.Tensor],
        target_overjet: Optional[torch.Tensor] = None,
        target_overbite: Optional[torch.Tensor] = None,
        target_midline: Optional[torch.Tensor] = None,
        target_molar_class: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute clinical metric losses.

        Args:
            pred_scores: Dict from OcclusionScoringHead.forward().
            target_overjet: (B,) target overjet in mm.
            target_overbite: (B,) target overbite in mm.
            target_midline: (B,) target midline deviation in mm.
            target_molar_class: (B,) target molar class indices (0=I, 1=II, 2=III).

        Returns:
            total_loss: Weighted sum of clinical losses.
            breakdown: Dict of individual named losses.
        """
        device = pred_scores["overjet_mm"].device
        total = torch.tensor(0.0, device=device)
        breakdown: Dict[str, torch.Tensor] = {}

        if target_overjet is not None:
            l_overjet = self.mse(pred_scores["overjet_mm"], target_overjet)
            breakdown["overjet_mse"] = l_overjet
            total = total + self.w_overjet * l_overjet

        if target_overbite is not None:
            l_overbite = self.mse(pred_scores["overbite_mm"], target_overbite)
            breakdown["overbite_mse"] = l_overbite
            total = total + self.w_overbite * l_overbite

        if target_midline is not None:
            l_midline = self.mse(pred_scores["midline_deviation_mm"], target_midline)
            breakdown["midline_mse"] = l_midline
            total = total + self.w_midline * l_midline

        if target_molar_class is not None:
            l_molar = self.ce(pred_scores["molar_class_logits"], target_molar_class)
            breakdown["molar_class_ce"] = l_molar
            total = total + self.w_molar_class * l_molar

        breakdown["clinical_total"] = total
        return total, breakdown


# ─── Bilateral symmetry loss ──────────────────────────────────────────────────


class BilateralSymmetryLoss(nn.Module):
    """
    Regularization loss encouraging bilateral symmetry of predicted transforms.

    For bone fragments that have bilateral counterparts (e.g., left/right
    condyle, left/right ramus), penalizes asymmetry in the predicted
    rotation and translation.

    Symmetry is computed by reflecting the transform across the midsagittal
    plane (X=0 by default) and computing the geodesic distance between the
    original and reflected transforms.

    Args:
        symmetry_axis: Index of the axis perpendicular to the sagittal plane
                      (0=X for left/right symmetry).
        w_rotation: Weight for rotation symmetry.
        w_translation: Weight for translation symmetry.

    References:
    - PMC11574221: Bilateral symmetry as a surgical planning constraint
    """

    def __init__(
        self, symmetry_axis: int = 0, w_rotation: float = 1.0, w_translation: float = 1.0,
    ) -> None:
        super().__init__()
        self.symmetry_axis = symmetry_axis
        self.w_rotation = w_rotation
        self.w_translation = w_translation
        self.geodesic = GeodesicRotationLoss(reduction="mean")

    def forward(
        self,
        rotation_matrices: torch.Tensor,
        translations: torch.Tensor,
        bilateral_pairs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute bilateral symmetry loss for paired fragments.

        Args:
            rotation_matrices: (B, F, 3, 3) predicted rotations.
            translations: (B, F, 3) predicted translations.
            bilateral_pairs: (P, 2) indices of bilateral fragment pairs.

        Returns:
            total_loss: Scalar symmetry loss.
            breakdown: Dict with rotation and translation symmetry losses.
        """
        if bilateral_pairs.shape[0] == 0:
            device = rotation_matrices.device
            zero = torch.tensor(0.0, device=device)
            return zero, {"sym_rotation": zero, "sym_translation": zero}

        left_idx = bilateral_pairs[:, 0]
        right_idx = bilateral_pairs[:, 1]

        # Reflection matrix across sagittal plane
        # Reflects the symmetry_axis column and row of the rotation
        S = torch.eye(3, device=rotation_matrices.device)
        S[self.symmetry_axis, self.symmetry_axis] = -1.0

        # Get left and right transforms
        R_left = rotation_matrices[:, left_idx]    # (B, P, 3, 3)
        R_right = rotation_matrices[:, right_idx]  # (B, P, 3, 3)
        t_left = translations[:, left_idx]          # (B, P, 3)
        t_right = translations[:, right_idx]        # (B, P, 3)

        # Reflect left rotation: R_reflected = S @ R_left @ S
        R_left_reflected = S @ R_left @ S

        # Reflect left translation: negate the symmetry axis component
        t_left_reflected = t_left.clone()
        t_left_reflected[..., self.symmetry_axis] = -t_left_reflected[..., self.symmetry_axis]

        # Rotation symmetry: geodesic distance between reflected left and right
        l_rot_sym = self.geodesic(R_left_reflected, R_right)

        # Translation symmetry: L2 distance
        l_trans_sym = (t_left_reflected - t_right).pow(2).sum(dim=-1).sqrt().mean()

        total = self.w_rotation * l_rot_sym + self.w_translation * l_trans_sym

        return total, {
            "sym_rotation": l_rot_sym,
            "sym_translation": l_trans_sym,
        }


# ─── Arch form regularization ────────────────────────────────────────────────


class ArchFormRegularization(nn.Module):
    """
    Regularization penalizing deviations from the expected dental arch curve.

    After applying predicted per-tooth transforms, the resulting tooth centroids
    should still trace a smooth parabolic arch form. This loss penalizes
    large deviations from the original arch shape.

    Computed as the mean L2 displacement of tooth centroids after transform,
    weighted by distance from the arch midline (central teeth should move less).

    References:
    - arxiv 2312.15139 (TADPM): Dental arch form preservation
    """

    def __init__(self, max_displacement_mm: float = 5.0) -> None:
        super().__init__()
        self.max_disp = max_displacement_mm

    def forward(
        self,
        original_centroids: torch.Tensor,
        transformed_centroids: torch.Tensor,
        fdi_numbers: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute arch form preservation loss.

        Args:
            original_centroids: (B, T, 3) original tooth centroids before transform.
            transformed_centroids: (B, T, 3) tooth centroids after predicted transform.
            fdi_numbers: (B, T) FDI numbers for position-dependent weighting.

        Returns:
            Scalar arch form regularization loss.
        """
        displacement = (transformed_centroids - original_centroids).pow(2).sum(dim=-1).sqrt()

        # Position-dependent weighting: central teeth penalized more
        if fdi_numbers is not None:
            tooth_num = fdi_numbers % 10  # Tooth number within quadrant (1-8)
            # Weight: incisors (1,2) → 2.0, canines (3) → 1.5, premolars (4,5) → 1.0, molars (6-8) → 0.5
            weights = torch.ones_like(displacement)
            weights = torch.where(tooth_num <= 2, 2.0 * torch.ones_like(weights), weights)
            weights = torch.where(tooth_num == 3, 1.5 * torch.ones_like(weights), weights)
            weights = torch.where(tooth_num >= 6, 0.5 * torch.ones_like(weights), weights)
            weights = torch.where(fdi_numbers == 0, torch.zeros_like(weights), weights)
            displacement = displacement * weights

        # Penalize displacements beyond max threshold
        excess = F.relu(displacement - self.max_disp)
        loss = displacement.mean() + 2.0 * excess.mean()

        return loss


# ─── Evidential loss ──────────────────────────────────────────────────────────


class EvidentialRegressionLoss(nn.Module):
    """
    Loss for Deep Evidential Regression with Normal-Inverse-Gamma prior.

    Combines:
    1. NLL of the target under the predicted NIG distribution
    2. Regularization penalizing evidence on errors (prevents trivially
       high uncertainty to explain all targets)

    The NIG distribution parameterized by (mu, v, alpha, beta) represents:
    - Gaussian likelihood: y ~ N(mu, sigma^2)
    - Inverse-Gamma prior: sigma^2 ~ IG(alpha, beta)
    - Normal prior on mu: mu ~ N(gamma, sigma^2/v)

    Args:
        lambda_reg: Weight for evidence regularization term.

    References:
    - Amini et al., "Deep Evidential Regression" (NeurIPS 2020)
    """

    def __init__(self, lambda_reg: float = 0.05) -> None:
        super().__init__()
        self.lambda_reg = lambda_reg

    def forward(
        self,
        nig_params: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute evidential regression loss.

        Args:
            nig_params: Dict with 'mu', 'v', 'alpha', 'beta' tensors, each (B, D).
            targets: (B, D) regression targets.

        Returns:
            total_loss: Scalar loss.
            breakdown: Dict with NLL and regularization terms.
        """
        mu = nig_params["mu"]
        v = nig_params["v"]
        alpha = nig_params["alpha"]
        beta = nig_params["beta"]

        # NLL of NIG distribution
        # log p(y | mu, v, alpha, beta)
        omega = 2.0 * beta * (1.0 + v)
        nll = (
            0.5 * torch.log(torch.tensor(torch.pi, device=mu.device) / v)
            - alpha * torch.log(omega)
            + (alpha + 0.5) * torch.log((targets - mu).pow(2) * v + omega)
            + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
        )
        nll = nll.mean()

        # Evidence regularization: penalize evidence (2v + alpha) on errors
        # This prevents the model from trivially increasing uncertainty to explain errors
        error = (targets - mu).abs()
        evidence = 2.0 * v + alpha
        reg = (error * evidence).mean()

        total = nll + self.lambda_reg * reg

        return total, {"evidential_nll": nll, "evidential_reg": reg}


# ─── Composite supervised loss ────────────────────────────────────────────────


class SupervisedReductionLoss(nn.Module):
    """
    Full training loss combining all components for the supervised model.

    Components:
    1. Supervised: geodesic SO(3) + L2 translation per fragment/tooth
    2. Self-supervised: composite dental loss (Chamfer, overlap, uniformity, collision)
    3. Clinical: overjet/overbite/midline MSE + molar class CE
    4. Regularization: arch form + bilateral symmetry
    5. Uncertainty: evidential regression loss

    L_total = w_geo * L_geodesic(R)
            + w_trans * L_L2(t)
            + w_clinical * L_clinical(metrics)
            + w_dental * L_composite_dental
            + w_symmetry * L_symmetry
            + w_evidential * L_evidential

    All weights are configurable via LossConfig.

    Args:
        config: LossConfig with loss weights and hyperparameters.

    References:
    - Huynh, "Metrics for 3D Rotations" (JMIV 2009)
    - Zhou et al., "On the Continuity of Rotation Representations" (CVPR 2019)
    - Amini et al., "Deep Evidential Regression" (NeurIPS 2020)
    - PMC11574221: Composite objective for CMF surgery
    """

    def __init__(self, config: Optional[LossConfig] = None) -> None:
        super().__init__()
        if config is None:
            config = LossConfig()
        self.config = config

        # Component losses
        self.geodesic_loss = GeodesicRotationLoss(reduction="none")
        self.translation_loss = TranslationLoss(use_huber=True, huber_delta=5.0, reduction="none")
        self.clinical_loss = ClinicalMetricLoss(
            w_overjet=config.w_overjet,
            w_overbite=config.w_overbite,
            w_midline=config.w_midline,
            w_molar_class=config.w_molar_class,
        )
        self.dental_composite = CompositeDentalLoss()
        self.symmetry_loss = BilateralSymmetryLoss()
        self.arch_form_loss = ArchFormRegularization()
        self.evidential_loss = EvidentialRegressionLoss(lambda_reg=config.evidential_lambda)

    def forward(
        self,
        predictions: Dict[str, dict],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the complete training loss.

        Args:
            predictions: Dict from FacialAlignSupervisedModel.forward(), containing:
                - fragment_transforms: {rotation_r6, rotation_matrix, translation, confidence}
                - tooth_transforms: {rotation_r6, rotation_matrix, translation, confidence, valid_mask}
                - occlusion_scores: {overjet_mm, overbite_mm, midline_deviation_mm, molar_class_logits, ...}
                - uncertainty: {mu, nig_params, ...}

            targets: Dict containing any subset of:
                - fragment_rotation: (B, F, 3, 3) target rotation matrices
                - fragment_translation: (B, F, 3) target translations in mm
                - tooth_rotation: (B, T, 3, 3) target rotation matrices
                - tooth_translation: (B, T, 3) target translations in mm
                - overjet_mm: (B,) target overjet
                - overbite_mm: (B,) target overbite
                - midline_deviation_mm: (B,) target midline deviation
                - molar_class: (B,) target molar class (0/1/2)
                - upper_points: (B, N, 3) upper arch points (for dental composite)
                - lower_points: (B, M, 3) lower arch points
                - bilateral_pairs: (P, 2) bilateral fragment pair indices
                - original_centroids: (B, T, 3) original tooth centroids
                - transformed_centroids: (B, T, 3) transformed tooth centroids
                - fdi_numbers: (B, T) FDI numbers
                - uncertainty_targets: (B, D) targets for uncertainty regression

        Returns:
            total_loss: Scalar weighted total loss.
            breakdown: Dict of all named loss components for logging.
        """
        device = next(iter(predictions.values()))
        if isinstance(device, dict):
            # Get device from first tensor in nested dict
            for v in device.values():
                if isinstance(v, torch.Tensor):
                    device = v.device
                    break
        elif isinstance(device, torch.Tensor):
            device = device.device

        total = torch.tensor(0.0, device=device)
        breakdown: Dict[str, torch.Tensor] = {}

        # ── 1. Fragment transform losses ──
        if "fragment_transforms" in predictions and "fragment_rotation" in targets:
            frag_pred = predictions["fragment_transforms"]
            frag_R = frag_pred["rotation_matrix"]  # (B, F, 3, 3)
            frag_t = frag_pred["translation"]       # (B, F, 3)
            frag_conf = frag_pred["confidence"]     # (B, F)

            target_R = targets["fragment_rotation"]   # (B, F, 3, 3)
            target_t = targets["fragment_translation"]  # (B, F, 3)

            # Confidence-weighted losses
            weights = None
            if self.config.use_confidence_weighting:
                weights = frag_conf.clamp(min=self.config.confidence_floor)

            l_geo_frag = self.geodesic_loss(frag_R, target_R, weights)
            l_geo_frag = l_geo_frag.mean()
            breakdown["fragment_geodesic"] = l_geo_frag
            total = total + self.config.w_geodesic * l_geo_frag

            l_trans_frag = self.translation_loss(frag_t, target_t, weights)
            l_trans_frag = l_trans_frag.mean()
            breakdown["fragment_translation"] = l_trans_frag
            total = total + self.config.w_translation * l_trans_frag

        # ── 2. Tooth transform losses ──
        if "tooth_transforms" in predictions and "tooth_rotation" in targets:
            tooth_pred = predictions["tooth_transforms"]
            tooth_R = tooth_pred["rotation_matrix"]  # (B, T, 3, 3)
            tooth_t = tooth_pred["translation"]       # (B, T, 3)
            tooth_conf = tooth_pred["confidence"]     # (B, T)
            valid_mask = tooth_pred["valid_mask"]     # (B, T)

            target_R_t = targets["tooth_rotation"]     # (B, T, 3, 3)
            target_t_t = targets["tooth_translation"]  # (B, T, 3)

            # Only compute loss on valid (non-padding) teeth
            if valid_mask.any():
                weights = None
                if self.config.use_confidence_weighting:
                    weights = tooth_conf.clamp(min=self.config.confidence_floor)
                    weights = weights * valid_mask.float()
                else:
                    weights = valid_mask.float()

                l_geo_tooth = self.geodesic_loss(tooth_R, target_R_t, weights)
                l_geo_tooth = l_geo_tooth.sum() / valid_mask.float().sum().clamp(min=1.0)
                breakdown["tooth_geodesic"] = l_geo_tooth
                total = total + self.config.w_geodesic * l_geo_tooth

                l_trans_tooth = self.translation_loss(tooth_t, target_t_t, weights)
                l_trans_tooth = l_trans_tooth.sum() / valid_mask.float().sum().clamp(min=1.0)
                breakdown["tooth_translation"] = l_trans_tooth
                total = total + self.config.w_translation * l_trans_tooth

        # ── 3. Clinical metric losses ──
        if "occlusion_scores" in predictions:
            l_clinical, clinical_breakdown = self.clinical_loss(
                predictions["occlusion_scores"],
                target_overjet=targets.get("overjet_mm"),
                target_overbite=targets.get("overbite_mm"),
                target_midline=targets.get("midline_deviation_mm"),
                target_molar_class=targets.get("molar_class"),
            )
            breakdown.update(clinical_breakdown)
            total = total + l_clinical

        # ── 4. Self-supervised dental composite loss ──
        if "upper_points" in targets and "lower_points" in targets:
            l_dental, dental_breakdown = self.dental_composite(
                upper_points=targets["upper_points"],
                lower_points=targets["lower_points"],
                upper_midline=targets.get("upper_midline"),
                lower_midline=targets.get("lower_midline"),
                upper_molar_landmarks=targets.get("upper_molar_landmarks"),
                lower_molar_landmarks=targets.get("lower_molar_landmarks"),
                predicted_centroids=targets.get("predicted_centroids"),
                target_centroids=targets.get("target_centroids"),
            )
            for k, v in dental_breakdown.items():
                breakdown[f"dental_{k}"] = v
            total = total + self.config.w_dental_composite * l_dental

        # ── 5. Bilateral symmetry regularization ──
        if (
            "fragment_transforms" in predictions
            and "bilateral_pairs" in targets
            and targets["bilateral_pairs"].shape[0] > 0
        ):
            frag_pred = predictions["fragment_transforms"]
            l_sym, sym_breakdown = self.symmetry_loss(
                frag_pred["rotation_matrix"],
                frag_pred["translation"],
                targets["bilateral_pairs"],
            )
            breakdown.update(sym_breakdown)
            total = total + self.config.w_symmetry * l_sym

        # ── 6. Arch form regularization ──
        if "original_centroids" in targets and "transformed_centroids" in targets:
            l_arch = self.arch_form_loss(
                targets["original_centroids"],
                targets["transformed_centroids"],
                targets.get("fdi_numbers"),
            )
            breakdown["arch_form"] = l_arch
            total = total + self.config.w_arch_form * l_arch

        # ── 7. Evidential uncertainty loss ──
        if (
            "uncertainty" in predictions
            and predictions["uncertainty"].get("nig_params") is not None
            and "uncertainty_targets" in targets
        ):
            l_evid, evid_breakdown = self.evidential_loss(
                predictions["uncertainty"]["nig_params"],
                targets["uncertainty_targets"],
            )
            breakdown.update(evid_breakdown)
            total = total + self.config.w_evidential * l_evid

        breakdown["total"] = total
        return total, breakdown
