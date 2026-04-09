"""
Cross-modal attention fusion for CT volumes and IOS point clouds.

Implements the CMAF-Net pattern (PMC11250309) adapted for maxillofacial
CT + intraoral scan fusion.  The module performs bidirectional cross-attention
between CT patch features and IOS tooth features, producing a fused
representation that captures both volumetric bone structure and dental
surface detail.

Missing-modality robustness
---------------------------
When IOS data is unavailable (CT-only mode), the module substitutes a
learned null embedding for the IOS tokens.  During training, IOS tokens
are randomly dropped (p=0.3) so the network learns to function in both
modes.  A binary modality indicator is concatenated to the fused output
so downstream heads know whether IOS was available.

References:
- CMAF-Net (PMC11250309): Cross-Modal Attention Fusion for missing modality
- ShaSpec (CVPR 2023): Shared spectral encoders for missing modality
- Vaswani et al., "Attention Is All You Need" (NeurIPS 2017)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ─── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_CT_DIM = 512
DEFAULT_IOS_DIM = 512
DEFAULT_FUSED_DIM = 1024
DEFAULT_NUM_HEADS = 8
DEFAULT_NUM_LAYERS = 4
DEFAULT_DROPOUT = 0.1
DEFAULT_IOS_DROPOUT_P = 0.3


@dataclass
class FusionConfig:
    """Configuration for MultimodalFusionModule."""
    ct_feat_dim: int = DEFAULT_CT_DIM
    ios_feat_dim: int = DEFAULT_IOS_DIM
    fused_dim: int = DEFAULT_FUSED_DIM
    num_heads: int = DEFAULT_NUM_HEADS
    num_layers: int = DEFAULT_NUM_LAYERS
    dropout: float = DEFAULT_DROPOUT
    ios_dropout_p: float = DEFAULT_IOS_DROPOUT_P


# ─── Cross-attention block ────────────────────────────────────────────────────


class CrossAttentionBlock(nn.Module):
    """
    Bidirectional cross-attention between two modalities.

    Given query tokens from modality A and key/value tokens from modality B,
    performs multi-head cross-attention followed by a feed-forward network
    with residual connections and layer normalisation (pre-norm architecture).

    Architecture:
        query_A → LN → CrossAttn(Q=A, K=B, V=B) → +residual → LN → FFN → +residual

    This is applied in both directions (CT→IOS and IOS→CT) per fusion layer.

    Args:
        d_model: Token embedding dimension.
        nhead: Number of attention heads.
        dim_feedforward: Hidden dimension of the FFN.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, N_q, D) query tokens from one modality.
            key_value: (B, N_kv, D) key/value tokens from the other modality.
            key_padding_mask: (B, N_kv) True for positions to ignore.

        Returns:
            (B, N_q, D) updated query tokens.
        """
        # Pre-norm cross-attention
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)
        attended, _ = self.cross_attn(q, kv, kv, key_padding_mask=key_padding_mask)
        query = query + attended

        # Pre-norm FFN
        query = query + self.ffn(self.norm_ffn(query))
        return query


class BidirectionalCrossAttentionLayer(nn.Module):
    """
    One layer of bidirectional cross-attention: CT attends to IOS, then IOS
    attends to CT.  Both directions share no parameters (separate blocks).

    Args:
        d_model: Token embedding dimension.
        nhead: Number of attention heads.
        dim_feedforward: Hidden FFN dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ct_attends_ios = CrossAttentionBlock(d_model, nhead, dim_feedforward, dropout)
        self.ios_attends_ct = CrossAttentionBlock(d_model, nhead, dim_feedforward, dropout)

    def forward(
        self,
        ct_tokens: torch.Tensor,
        ios_tokens: torch.Tensor,
        ios_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ct_tokens: (B, N_ct, D) CT patch tokens.
            ios_tokens: (B, N_ios, D) IOS tooth tokens.
            ios_mask: (B, N_ios) True for padded/missing IOS positions.

        Returns:
            Updated (ct_tokens, ios_tokens).
        """
        ct_out = self.ct_attends_ios(ct_tokens, ios_tokens, key_padding_mask=ios_mask)
        ios_out = self.ios_attends_ct(ios_tokens, ct_tokens)
        return ct_out, ios_out


# ─── Main fusion module ──────────────────────────────────────────────────────


class MultimodalFusionModule(nn.Module):
    """
    Cross-modal attention fusion for CT volume + IOS point cloud features.

    Architecture:
    1. Project CT and IOS features into a shared dimension
    2. Apply N layers of bidirectional cross-attention
    3. Pool both modalities into a single fused vector
    4. Concatenate a modality indicator (1-dim: IOS present or absent)

    Missing IOS is handled by:
    - Replacing IOS tokens with a learned null embedding
    - Setting a modality indicator flag to 0
    - During training: randomly dropping IOS with p=ios_dropout_p

    The output fused vector has dimension `fused_dim + 1` (the +1 is the
    modality indicator).

    Args:
        config: FusionConfig with architecture hyperparameters.

    References:
    - CMAF-Net (PMC11250309): Missing modality cross-attention
    - ShaSpec (CVPR 2023): Shared spectral encoder approach
    """

    def __init__(self, config: Optional[FusionConfig] = None) -> None:
        super().__init__()
        if config is None:
            config = FusionConfig()
        self.config = config

        d = config.ct_feat_dim  # Internal dimension for cross-attention

        # Project CT and IOS to shared dimension if needed
        self.ct_proj = (
            nn.Identity()
            if config.ct_feat_dim == d
            else nn.Sequential(nn.Linear(config.ct_feat_dim, d), nn.LayerNorm(d))
        )
        self.ios_proj = (
            nn.Sequential(nn.Linear(config.ios_feat_dim, d), nn.LayerNorm(d))
            if config.ios_feat_dim != d
            else nn.Identity()
        )

        # Learned null embedding for missing IOS — shape (1, 1, d)
        self.ios_null_embedding = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        # Bidirectional cross-attention layers
        self.fusion_layers = nn.ModuleList([
            BidirectionalCrossAttentionLayer(
                d_model=d,
                nhead=config.num_heads,
                dim_feedforward=d * 4,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])

        # Pooling: attention-weighted pooling for each modality
        self.ct_pool_attn = nn.Linear(d, 1)
        self.ios_pool_attn = nn.Linear(d, 1)

        # Final projection to fused_dim
        # CT pooled (d) + IOS pooled (d) → fused_dim
        self.output_proj = nn.Sequential(
            nn.Linear(d * 2, config.fused_dim),
            nn.LayerNorm(config.fused_dim),
            nn.GELU(),
        )

    def _attention_pool(
        self, tokens: torch.Tensor, attn_proj: nn.Linear, mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Attention-weighted pooling over tokens.

        Args:
            tokens: (B, N, D) token features.
            attn_proj: Linear(D, 1) for attention scores.
            mask: (B, N) True for positions to ignore.

        Returns:
            (B, D) pooled feature.
        """
        scores = attn_proj(tokens).squeeze(-1)  # (B, N)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)  # (B, N)
        pooled = (tokens * weights.unsqueeze(-1)).sum(dim=1)  # (B, D)
        return pooled

    def forward(
        self,
        ct_global: torch.Tensor,
        ct_patches: torch.Tensor,
        ios_per_tooth: Optional[torch.Tensor] = None,
        ios_arch: Optional[torch.Tensor] = None,
        ios_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Fuse CT and IOS features.

        Args:
            ct_global: (B, ct_feat_dim) global CT feature (unused in cross-attn,
                       but available for skip connections).
            ct_patches: (B, N_ct, ct_feat_dim) CT patch features from stage-4 map.
            ios_per_tooth: (B, N_teeth, ios_feat_dim) per-tooth IOS features.
                           None if IOS unavailable.
            ios_arch: (B, ios_feat_dim) global arch feature (optional, unused currently).
            ios_mask: (B, N_teeth) True for missing/padded teeth. None = all present.

        Returns:
            fused: (B, fused_dim + 1) fused feature vector.
                   The last dimension is the modality indicator (1.0 if IOS present, 0.0 otherwise).
            ios_available: bool indicating whether IOS was used.
        """
        B = ct_patches.shape[0]
        device = ct_patches.device

        # Project CT patches to shared dimension
        ct_tokens = self.ct_proj(ct_patches)  # (B, N_ct, d)

        # Determine IOS availability
        ios_available = ios_per_tooth is not None

        # Training-time IOS dropout: randomly mask IOS to build robustness
        if self.training and ios_available and torch.rand(1).item() < self.config.ios_dropout_p:
            ios_available = False

        if ios_available:
            ios_tokens = self.ios_proj(ios_per_tooth)  # (B, N_teeth, d)
        else:
            # Use learned null embedding — broadcast to batch
            # Use a single null token per batch element
            ios_tokens = self.ios_null_embedding.expand(B, -1, -1)  # (B, 1, d)
            ios_mask = None  # No masking needed for single null token

        # Apply bidirectional cross-attention
        for layer in self.fusion_layers:
            ct_tokens, ios_tokens = layer(ct_tokens, ios_tokens, ios_mask)

        # Attention pooling
        ct_pooled = self._attention_pool(ct_tokens, self.ct_pool_attn)  # (B, d)
        ios_pooled = self._attention_pool(ios_tokens, self.ios_pool_attn, ios_mask)  # (B, d)

        # Concatenate and project
        combined = torch.cat([ct_pooled, ios_pooled], dim=-1)  # (B, 2d)
        fused = self.output_proj(combined)  # (B, fused_dim)

        # Append modality indicator
        indicator = torch.ones(B, 1, device=device) if (ios_per_tooth is not None and not (
            self.training and not ios_available
        )) else torch.zeros(B, 1, device=device)
        fused = torch.cat([fused, indicator], dim=-1)  # (B, fused_dim + 1)

        return fused, ios_per_tooth is not None
