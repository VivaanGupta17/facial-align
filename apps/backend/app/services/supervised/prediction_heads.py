"""
Prediction heads for the supervised facial alignment model.

Each head takes fused multimodal features and produces a specific prediction:
- FragmentTransformHead: per-fragment SE(3) bone repositioning transforms
- ToothTransformHead: per-tooth SE(3) dental occlusion correction
- OcclusionScoringHead: clinical occlusion quality metrics
- UncertaintyHead: aleatoric + epistemic uncertainty via MC dropout + evidential DL

All rotation predictions use R6 continuous representation (Zhou et al. CVPR 2019),
projected to SO(3) via SVD orthogonalization.

References:
- Zhou et al., "On the Continuity of Rotation Representations" (CVPR 2019)
- Amini et al., "Deep Evidential Regression" (NeurIPS 2020)
- Gal & Ghahramani, "Dropout as a Bayesian Approximation" (ICML 2016)
- PMC11574221: Fragment transform prediction for CMF surgery
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import rotation_6d_to_matrix

logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────

MAX_FRAGMENTS = 8
MAX_TEETH = 32
NUM_FDI_ENTRIES = 49  # 0=padding, 11-48 for teeth


# ─── R6 rotation utilities ────────────────────────────────────────────────────


def r6_to_rotation_matrix(r6: torch.Tensor) -> torch.Tensor:
    """
    Convert R6 continuous rotation representation to SO(3) rotation matrix
    via Gram-Schmidt orthogonalization (pytorch3d implementation).

    The R6 representation uses the first two columns of the rotation matrix
    (6 values), then recovers the third column via cross product.
    This ensures continuity of the representation.

    Args:
        r6: (..., 6) R6 rotation parameters.

    Returns:
        (..., 3, 3) rotation matrices in SO(3).

    References:
    - Zhou et al., "On the Continuity of Rotation Representations" (CVPR 2019)
    """
    shape = r6.shape[:-1]
    r6_flat = r6.reshape(-1, 6)
    R = rotation_6d_to_matrix(r6_flat)
    return R.reshape(*shape, 3, 3)


def r6_to_rotation_matrix_svd(r6: torch.Tensor) -> torch.Tensor:
    """
    Convert R6 to SO(3) via SVD projection (more robust than Gram-Schmidt).

    Constructs a 3x3 matrix from R6 (first two columns + cross product),
    then projects onto SO(3) via SVD: R = U @ V^T, ensuring det(R)=+1.

    This is more numerically stable than Gram-Schmidt for backpropagation
    through rotation predictions.

    Args:
        r6: (..., 6) R6 rotation parameters.

    Returns:
        (..., 3, 3) rotation matrices in SO(3) with det=+1.

    References:
    - Levinson et al., "An Analysis of SVD for Deep Rotation Estimation" (NeurIPS 2020)
    """
    shape = r6.shape[:-1]
    r6_flat = r6.reshape(-1, 6)

    # Build approximate rotation matrix from R6
    col1 = r6_flat[:, :3]
    col2 = r6_flat[:, 3:6]

    # Normalize first column
    col1 = F.normalize(col1, dim=-1)
    # Orthogonalize second column
    col2 = col2 - (col1 * col2).sum(dim=-1, keepdim=True) * col1
    col2 = F.normalize(col2, dim=-1)
    # Third column via cross product
    col3 = torch.cross(col1, col2, dim=-1)

    M = torch.stack([col1, col2, col3], dim=-1)  # (N, 3, 3)

    # SVD projection to SO(3)
    U, _, Vh = torch.linalg.svd(M)
    R = U @ Vh

    # Ensure det(R) = +1 (not reflection)
    det = torch.det(R)
    sign = det.sign().unsqueeze(-1).unsqueeze(-1)
    # Flip last column of U if det is negative
    U_corrected = U.clone()
    U_corrected[:, :, -1] = U[:, :, -1] * sign.squeeze(-1)
    R = U_corrected @ Vh

    return R.reshape(*shape, 3, 3)


def identity_r6(batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Return R6 representation of the identity rotation.

    Identity rotation matrix columns: [1,0,0], [0,1,0]
    → R6 = [1, 0, 0, 0, 1, 0]

    Args:
        batch_size: Number of identity rotations to generate.
        device: Target device.

    Returns:
        (batch_size, 6) identity R6 vectors.
    """
    r6 = torch.zeros(batch_size, 6, device=device)
    r6[:, 0] = 1.0  # First column: [1, 0, 0]
    r6[:, 4] = 1.0  # Second column: [0, 1, 0]
    return r6


# ─── Fragment Transform Head ─────────────────────────────────────────────────


class FragmentTransformHead(nn.Module):
    """
    Predicts per-fragment SE(3) transforms from fused features.

    For each bone fragment, predicts:
    - Rotation: R6 continuous representation (6D) → SVD → SO(3)
    - Translation: t in R^3 in mm
    - Confidence: sigma in [0,1] via sigmoid

    Uses R6 representation per "On the Continuity of Rotation Representations"
    (Zhou et al. CVPR 2019). The SVD projection ensures the output is always
    a valid rotation matrix.

    Architecture:
        fused_features → shared MLP → per-fragment branch (repeated per fragment)
        Each branch: Linear → ReLU → [rotation_head (→6), translation_head (→3), confidence_head (→1)]

    The head uses a fragment embedding table to distinguish between fragments
    (since fragment identity matters for surgical planning — e.g., condylar
    fragments need different handling than symphyseal fragments).

    Args:
        input_dim: Dimension of fused features.
        hidden_dim: Hidden layer dimension.
        max_fragments: Maximum number of bone fragments supported.

    References:
    - Zhou et al., "On the Continuity of Rotation Representations" (CVPR 2019)
    - PMC11574221: Per-fragment transform prediction
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        max_fragments: int = MAX_FRAGMENTS,
    ) -> None:
        super().__init__()
        self.max_fragments = max_fragments

        # Fragment identity embedding
        self.fragment_embedding = nn.Embedding(max_fragments, input_dim)

        # Shared feature processing
        self.shared_mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Rotation head: predicts R6 (6D continuous rotation)
        self.rotation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 6),
        )

        # Translation head: predicts translation in mm
        self.translation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),
        )

        # Confidence head: sigmoid output in [0, 1]
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, 1),
        )

        # Initialize to identity transform
        self._init_to_identity()

    def _init_to_identity(self) -> None:
        """Initialize rotation head bias to identity R6 [1,0,0,0,1,0]."""
        nn.init.zeros_(self.rotation_head[-1].weight)
        self.rotation_head[-1].bias.data.copy_(
            torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        )
        nn.init.zeros_(self.translation_head[-1].weight)
        nn.init.zeros_(self.translation_head[-1].bias)
        # Initialize confidence to 0.5 (logit = 0)
        nn.init.zeros_(self.confidence_head[-1].weight)
        nn.init.zeros_(self.confidence_head[-1].bias)

    def forward(
        self,
        fused_features: torch.Tensor,
        num_fragments: int,
        fragment_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict per-fragment SE(3) transforms.

        Args:
            fused_features: (B, fused_dim) fused multimodal features.
            num_fragments: Number of fragments to predict transforms for.
            fragment_indices: (num_fragments,) optional fragment identity indices
                            (0 to max_fragments-1). If None, uses sequential 0..N-1.

        Returns:
            Dict with:
                rotation_r6: (B, num_fragments, 6) R6 rotation parameters.
                rotation_matrix: (B, num_fragments, 3, 3) SO(3) rotation matrices.
                translation: (B, num_fragments, 3) translations in mm.
                confidence: (B, num_fragments) confidence scores in [0, 1].
        """
        B = fused_features.shape[0]
        device = fused_features.device

        if fragment_indices is None:
            fragment_indices = torch.arange(num_fragments, device=device)

        # Get fragment identity embeddings
        frag_embed = self.fragment_embedding(fragment_indices)  # (F, input_dim)

        # Expand fused features for each fragment
        fused_expanded = fused_features.unsqueeze(1).expand(-1, num_fragments, -1)  # (B, F, d)
        frag_embed_expanded = frag_embed.unsqueeze(0).expand(B, -1, -1)  # (B, F, d)

        # Concat and process
        combined = torch.cat([fused_expanded, frag_embed_expanded], dim=-1)  # (B, F, 2*d)
        h = self.shared_mlp(combined)  # (B, F, hidden)

        # Predict transforms
        rot_r6 = self.rotation_head(h)               # (B, F, 6)
        rot_mat = r6_to_rotation_matrix_svd(rot_r6)  # (B, F, 3, 3)
        trans = self.translation_head(h)              # (B, F, 3)
        conf = torch.sigmoid(self.confidence_head(h).squeeze(-1))  # (B, F)

        return {
            "rotation_r6": rot_r6,
            "rotation_matrix": rot_mat,
            "translation": trans,
            "confidence": conf,
        }


# ─── Tooth Transform Head ────────────────────────────────────────────────────


class ToothTransformHead(nn.Module):
    """
    Predicts per-tooth SE(3) transforms for dental occlusion correction.

    Same R6 rotation representation as FragmentTransformHead, but with
    FDI-aware positional encoding so the network knows which tooth each
    prediction corresponds to (important for anatomically correct movement
    constraints — e.g., molars have more limited movement than incisors).

    Includes FDI-aware positional encoding that captures:
    - Tooth type (incisor, canine, premolar, molar)
    - Arch side (left/right for symmetry)
    - Arch (upper/lower)

    Architecture:
        fused_features + ios_tooth_features → concat → shared MLP
        → per-tooth: [rotation_head (→6), translation_head (→3), confidence_head (→1)]

    Args:
        fused_dim: Dimension of fused features.
        tooth_feat_dim: Dimension of per-tooth IOS features.
        hidden_dim: Hidden layer dimension.
        max_teeth: Maximum number of teeth.

    References:
    - Zhou et al., "On the Continuity of Rotation Representations" (CVPR 2019)
    - arxiv 2312.15139 (TADPM): Per-tooth SE(3) prediction
    """

    def __init__(
        self,
        fused_dim: int = 512,
        tooth_feat_dim: int = 256,
        hidden_dim: int = 256,
        max_teeth: int = MAX_TEETH,
    ) -> None:
        super().__init__()
        self.max_teeth = max_teeth

        # FDI positional encoding — encodes tooth identity
        self.fdi_embedding = nn.Embedding(NUM_FDI_ENTRIES, 64)

        # Tooth type encoding (0=padding, 1=incisor, 2=canine, 3=premolar, 4=molar)
        self.tooth_type_embedding = nn.Embedding(5, 32)

        # Arch side encoding (0=padding, 1=right, 2=left)
        self.arch_side_embedding = nn.Embedding(3, 16)

        # Input projection: fused + tooth_feat + FDI + type + side → hidden
        combined_dim = fused_dim + tooth_feat_dim + 64 + 32 + 16
        self.input_proj = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Rotation head: R6 representation
        self.rotation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 6),
        )

        # Translation head
        self.translation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),
        )

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, 1),
        )

        self._init_to_identity()

    def _init_to_identity(self) -> None:
        """Initialize rotation head bias to identity R6."""
        nn.init.zeros_(self.rotation_head[-1].weight)
        self.rotation_head[-1].bias.data.copy_(
            torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        )
        nn.init.zeros_(self.translation_head[-1].weight)
        nn.init.zeros_(self.translation_head[-1].bias)
        nn.init.zeros_(self.confidence_head[-1].weight)
        nn.init.zeros_(self.confidence_head[-1].bias)

    @staticmethod
    def fdi_to_tooth_type(fdi: int) -> int:
        """Map FDI tooth number to tooth type index.

        FDI system: quadrant digit (1-4) + tooth digit (1-8).
        Tooth digit: 1-2=incisor, 3=canine, 4-5=premolar, 6-8=molar.
        """
        if fdi == 0:
            return 0  # padding
        tooth_num = fdi % 10
        if tooth_num in (1, 2):
            return 1  # incisor
        elif tooth_num == 3:
            return 2  # canine
        elif tooth_num in (4, 5):
            return 3  # premolar
        else:
            return 4  # molar

    @staticmethod
    def fdi_to_arch_side(fdi: int) -> int:
        """Map FDI tooth number to arch side index.

        Quadrant 1 (upper right), 4 (lower right) → right (1)
        Quadrant 2 (upper left), 3 (lower left) → left (2)
        """
        if fdi == 0:
            return 0  # padding
        quadrant = fdi // 10
        if quadrant in (1, 4):
            return 1  # right
        else:
            return 2  # left

    def forward(
        self,
        fused_features: torch.Tensor,
        tooth_features: torch.Tensor,
        fdi_numbers: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict per-tooth SE(3) transforms.

        Args:
            fused_features: (B, fused_dim) fused multimodal features.
            tooth_features: (B, T, tooth_feat_dim) per-tooth features from IOS encoder.
            fdi_numbers: (B, T) FDI tooth numbers (0 for padding/absent teeth).

        Returns:
            Dict with:
                rotation_r6: (B, T, 6) R6 rotation parameters.
                rotation_matrix: (B, T, 3, 3) SO(3) rotation matrices.
                translation: (B, T, 3) translations in mm.
                confidence: (B, T) confidence scores in [0, 1].
                valid_mask: (B, T) True for real teeth (FDI > 0).
        """
        B, T, _ = tooth_features.shape
        device = fused_features.device

        # Compute positional encodings
        fdi_embed = self.fdi_embedding(fdi_numbers)  # (B, T, 64)

        # Compute tooth type and arch side from FDI numbers
        fdi_flat = fdi_numbers.view(-1).cpu().tolist()
        type_ids = torch.tensor(
            [self.fdi_to_tooth_type(f) for f in fdi_flat],
            dtype=torch.long, device=device,
        ).view(B, T)
        side_ids = torch.tensor(
            [self.fdi_to_arch_side(f) for f in fdi_flat],
            dtype=torch.long, device=device,
        ).view(B, T)

        type_embed = self.tooth_type_embedding(type_ids)  # (B, T, 32)
        side_embed = self.arch_side_embedding(side_ids)    # (B, T, 16)

        # Expand fused features to per-tooth
        fused_expanded = fused_features.unsqueeze(1).expand(-1, T, -1)  # (B, T, fused_dim)

        # Combine all inputs
        combined = torch.cat([
            fused_expanded, tooth_features, fdi_embed, type_embed, side_embed,
        ], dim=-1)

        h = self.input_proj(combined)  # (B, T, hidden)

        # Predict transforms
        rot_r6 = self.rotation_head(h)               # (B, T, 6)
        rot_mat = r6_to_rotation_matrix_svd(rot_r6)  # (B, T, 3, 3)
        trans = self.translation_head(h)              # (B, T, 3)
        conf = torch.sigmoid(self.confidence_head(h).squeeze(-1))  # (B, T)

        valid_mask = fdi_numbers > 0  # (B, T)

        return {
            "rotation_r6": rot_r6,
            "rotation_matrix": rot_mat,
            "translation": trans,
            "confidence": conf,
            "valid_mask": valid_mask,
        }


# ─── Occlusion Scoring Head ──────────────────────────────────────────────────


class OcclusionScoringHead(nn.Module):
    """
    Predicts clinical occlusion metrics from fused features.

    Outputs:
    - overjet_mm: horizontal incisal overjet (normal: 1-3mm)
    - overbite_mm: vertical incisal overbite (normal: 2-4mm)
    - midline_deviation_mm: lateral midline offset (normal: <1mm)
    - molar_class: Angle classification (Class I/II/III as 3-way softmax)
    - overall_quality_score in [0, 1]

    These metrics directly correspond to the clinical measurements used
    by surgeons to evaluate occlusion quality, allowing the model to
    provide interpretable feedback alongside the predicted transforms.

    Architecture:
        fused_features → shared backbone → [regression_head, classification_head, quality_head]

    Args:
        input_dim: Dimension of fused features.
        hidden_dim: Hidden layer dimension.

    References:
    - PMC11574221: Clinical occlusion metrics for CMF surgery
    - arxiv 2410.20806: Occlusion quality scoring
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Regression heads for continuous metrics
        self.overjet_head = nn.Linear(hidden_dim, 1)
        self.overbite_head = nn.Linear(hidden_dim, 1)
        self.midline_head = nn.Linear(hidden_dim, 1)

        # Classification head for Angle molar class (3 classes)
        self.molar_class_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),  # Class I, II, III logits
        )

        # Overall quality score in [0, 1]
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        # Initialize regression heads to typical normal values
        self._init_regression_heads()

    def _init_regression_heads(self) -> None:
        """Initialize regression heads to predict normal clinical values."""
        # Overjet: ~2.0 mm (normal)
        nn.init.zeros_(self.overjet_head.weight)
        self.overjet_head.bias.data.fill_(2.0)

        # Overbite: ~3.0 mm (normal)
        nn.init.zeros_(self.overbite_head.weight)
        self.overbite_head.bias.data.fill_(3.0)

        # Midline deviation: ~0.0 mm (ideal)
        nn.init.zeros_(self.midline_head.weight)
        nn.init.zeros_(self.midline_head.bias)

    def forward(self, fused_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict clinical occlusion metrics.

        Args:
            fused_features: (B, input_dim) fused multimodal features.

        Returns:
            Dict with:
                overjet_mm: (B,) predicted overjet in mm.
                overbite_mm: (B,) predicted overbite in mm.
                midline_deviation_mm: (B,) predicted midline deviation in mm.
                molar_class_logits: (B, 3) logits for Class I/II/III.
                molar_class_probs: (B, 3) softmax probabilities.
                overall_quality_score: (B,) quality score in [0, 1].
        """
        h = self.backbone(fused_features)

        overjet = self.overjet_head(h).squeeze(-1)           # (B,)
        overbite = self.overbite_head(h).squeeze(-1)         # (B,)
        midline = self.midline_head(h).squeeze(-1)           # (B,)
        molar_logits = self.molar_class_head(h)              # (B, 3)
        molar_probs = F.softmax(molar_logits, dim=-1)        # (B, 3)
        quality = torch.sigmoid(self.quality_head(h).squeeze(-1))  # (B,)

        return {
            "overjet_mm": overjet,
            "overbite_mm": overbite,
            "midline_deviation_mm": midline,
            "molar_class_logits": molar_logits,
            "molar_class_probs": molar_probs,
            "overall_quality_score": quality,
        }


# ─── Uncertainty Head ─────────────────────────────────────────────────────────


class UncertaintyHead(nn.Module):
    """
    Monte Carlo dropout + evidential deep learning for uncertainty estimation.

    Returns aleatoric + epistemic uncertainty estimates:
    - Aleatoric: learned per-prediction noise (data uncertainty) via evidential DL
    - Epistemic: model uncertainty via MC dropout at inference time

    Evidential deep learning models the output distribution as a Normal-Inverse-Gamma
    (NIG) prior, predicting (mu, v, alpha, beta) for each regression target.
    This allows decomposing total uncertainty into aleatoric and epistemic
    components in a single forward pass.

    For MC dropout, the head maintains dropout enabled during inference
    and aggregates predictions over T stochastic forward passes.

    Args:
        input_dim: Dimension of fused features.
        hidden_dim: Hidden layer dimension.
        output_dim: Number of regression targets to estimate uncertainty for.
        mc_dropout_rate: Dropout probability for MC dropout.
        mc_samples: Number of MC samples for epistemic uncertainty.

    References:
    - Amini et al., "Deep Evidential Regression" (NeurIPS 2020)
    - Gal & Ghahramani, "Dropout as a Bayesian Approximation" (ICML 2016)
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        output_dim: int = 9,
        mc_dropout_rate: float = 0.2,
        mc_samples: int = 20,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.mc_dropout_rate = mc_dropout_rate
        self.mc_samples = mc_samples

        # Shared backbone with MC dropout (stays active during inference)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(mc_dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(mc_dropout_rate),
        )

        # Evidential regression heads: predict (mu, v, alpha, beta) per target
        # mu: predicted mean
        # v: virtual evidence (>0, softplus)
        # alpha: shape parameter (>1, softplus + 1)
        # beta: scale parameter (>0, softplus)
        self.mu_head = nn.Linear(hidden_dim, output_dim)
        self.v_head = nn.Linear(hidden_dim, output_dim)
        self.alpha_head = nn.Linear(hidden_dim, output_dim)
        self.beta_head = nn.Linear(hidden_dim, output_dim)

        self._init_evidential()

    def _init_evidential(self) -> None:
        """Initialize evidential parameters for stable early training."""
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        # v should start near 1 (moderate virtual evidence)
        nn.init.zeros_(self.v_head.weight)
        self.v_head.bias.data.fill_(0.5)
        # alpha should start > 1 for valid NIG distribution
        nn.init.zeros_(self.alpha_head.weight)
        self.alpha_head.bias.data.fill_(1.0)
        # beta should start moderate
        nn.init.zeros_(self.beta_head.weight)
        self.beta_head.bias.data.fill_(0.5)

    def _single_forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Single forward pass through the evidential network."""
        h = self.backbone(features)

        mu = self.mu_head(h)                                  # (B, D)
        v = F.softplus(self.v_head(h)) + 1e-6                # (B, D) > 0
        alpha = F.softplus(self.alpha_head(h)) + 1.0 + 1e-6  # (B, D) > 1
        beta = F.softplus(self.beta_head(h)) + 1e-6          # (B, D) > 0

        return {"mu": mu, "v": v, "alpha": alpha, "beta": beta}

    def forward(
        self, fused_features: torch.Tensor, mc_inference: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate uncertainty via evidential DL and optional MC dropout.

        Args:
            fused_features: (B, input_dim) fused multimodal features.
            mc_inference: If True, run MC dropout sampling for epistemic uncertainty.
                         Requires calling model.eval() first but keeps dropout active.

        Returns:
            Dict with:
                mu: (B, output_dim) predicted mean.
                aleatoric: (B, output_dim) aleatoric uncertainty (from NIG: beta / (alpha - 1)).
                epistemic: (B, output_dim) epistemic uncertainty.
                    - If mc_inference: variance across MC samples.
                    - Otherwise: from NIG: beta / (v * (alpha - 1)).
                total_uncertainty: (B, output_dim) sum of aleatoric + epistemic.
                nig_params: Dict with raw NIG parameters (mu, v, alpha, beta).
        """
        if mc_inference:
            return self._mc_inference(fused_features)

        nig = self._single_forward(fused_features)
        mu = nig["mu"]
        v = nig["v"]
        alpha = nig["alpha"]
        beta = nig["beta"]

        # Aleatoric uncertainty: expected data noise (from NIG distribution)
        # E[sigma^2] = beta / (alpha - 1) when alpha > 1
        aleatoric = beta / (alpha - 1.0)

        # Epistemic uncertainty: model uncertainty (from NIG distribution)
        # Var[mu] = beta / (v * (alpha - 1))
        epistemic = beta / (v * (alpha - 1.0))

        total = aleatoric + epistemic

        return {
            "mu": mu,
            "aleatoric": aleatoric,
            "epistemic": epistemic,
            "total_uncertainty": total,
            "nig_params": nig,
        }

    def _mc_inference(self, fused_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run Monte Carlo dropout sampling for epistemic uncertainty.

        Performs self.mc_samples stochastic forward passes with dropout active,
        then computes mean and variance across samples.

        Args:
            fused_features: (B, input_dim) fused features.

        Returns:
            Same structure as forward(), with MC-estimated epistemic uncertainty.
        """
        # Enable dropout even in eval mode by temporarily switching
        was_training = self.training
        self.train()  # Enable dropout

        mc_mus = []
        mc_aleatorics = []

        for _ in range(self.mc_samples):
            nig = self._single_forward(fused_features)
            mc_mus.append(nig["mu"])
            aleatoric_i = nig["beta"] / (nig["alpha"] - 1.0)
            mc_aleatorics.append(aleatoric_i)

        if not was_training:
            self.eval()

        # Stack MC samples: (T, B, D)
        mc_mus_stacked = torch.stack(mc_mus, dim=0)
        mc_aleatoric_stacked = torch.stack(mc_aleatorics, dim=0)

        # Mean prediction across MC samples
        mu = mc_mus_stacked.mean(dim=0)  # (B, D)

        # Mean aleatoric (average per-sample noise estimate)
        aleatoric = mc_aleatoric_stacked.mean(dim=0)  # (B, D)

        # Epistemic: variance of means across MC samples
        epistemic = mc_mus_stacked.var(dim=0)  # (B, D)

        total = aleatoric + epistemic

        return {
            "mu": mu,
            "aleatoric": aleatoric,
            "epistemic": epistemic,
            "total_uncertainty": total,
            "nig_params": None,
        }
