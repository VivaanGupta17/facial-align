"""
Point cloud encoder for intraoral scan (IOS) dental meshes.

Wraps the existing DGCNNToothEncoder from the occlusion module and adds
cross-tooth attention for inter-tooth relationship modeling and global
arch-level feature extraction via attention pooling.

When IOS data is unavailable (CT-only mode), returns a learned zero-embedding
so the downstream fusion module receives a consistent tensor shape.

References:
- Wang et al., "Dynamic Graph CNN for Learning on Point Clouds" (2019)
- MICCAI TAPoseNet: DGCNN for tooth pose estimation
- arxiv 2312.15139 (TADPM): Dental arch-level aggregation with cross-tooth attention
- Vaswani et al., "Attention Is All You Need" (NeurIPS 2017) — transformer layers
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from app.services.occlusion.arch_encoder import (
    DEFAULT_POINTS_PER_TOOTH,
    DGCNNToothEncoder,
    FDI_LOWER,
    FDI_UPPER,
    GLOBAL_ARCH_EMBED_DIM,
    NUM_TEETH_MAX,
    PER_TOOTH_EMBED_DIM,
)

logger = logging.getLogger(__name__)


@dataclass
class IOSEncoderConfig:
    """Configuration for IOSPointCloudEncoder."""
    per_tooth_dim: int = PER_TOOTH_EMBED_DIM
    arch_dim: int = GLOBAL_ARCH_EMBED_DIM
    dgcnn_k: int = 20
    max_teeth: int = NUM_TEETH_MAX
    points_per_tooth: int = DEFAULT_POINTS_PER_TOOTH
    num_cross_tooth_layers: int = 4
    num_attention_heads: int = 8
    feedforward_dim: int = 1024
    dropout_rate: float = 0.1
    fdi_embed_dim: int = 32


# ─── Cross-tooth attention ────────────────────────────────────────────────────


class CrossToothTransformerLayer(nn.Module):
    """
    Transformer encoder layer for modeling inter-tooth spatial relationships.

    Each tooth embedding attends to all other teeth in the arch, allowing the
    network to learn relative positioning constraints (e.g., adjacent teeth
    should maintain contact, contralateral teeth should be symmetric).

    Uses pre-norm architecture (LayerNorm before attention/FFN) for more
    stable training with dental point cloud data.

    Args:
        d_model: Token (per-tooth) embedding dimension.
        nhead: Number of attention heads.
        dim_feedforward: Hidden dim of the FFN.
        dropout: Dropout rate.

    References:
    - Vaswani et al., "Attention Is All You Need" (NeurIPS 2017)
    - Xiong et al., "On Layer Normalization in the Transformer Architecture" (ICML 2020)
    """

    def __init__(
        self,
        d_model: int = PER_TOOTH_EMBED_DIM,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) per-tooth embeddings.
            key_padding_mask: (B, T) True where tokens are padding (absent teeth).

        Returns:
            (B, T, D) updated tooth embeddings.
        """
        # Pre-norm self-attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed, key_padding_mask=key_padding_mask,
        )
        x = x + attn_out

        # Pre-norm FFN
        normed = self.norm2(x)
        x = x + self.ffn(normed)
        return x


# ─── Attention pooling ────────────────────────────────────────────────────────


class AttentionPooling(nn.Module):
    """
    Attention-weighted pooling over a variable-length set of token embeddings.

    Learns a query vector that attends to all tokens, producing a single
    fixed-size output. Used to aggregate per-tooth features into a global
    arch-level representation.

    This avoids the information loss of simple mean/max pooling by learning
    which teeth are most informative for the global representation.

    Args:
        embed_dim: Dimension of input token embeddings.
        output_dim: Dimension of the pooled output.

    References:
    - Lee et al., "Set Transformer" (ICML 2019) — attention-based set pooling
    """

    def __init__(self, embed_dim: int, output_dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) token embeddings.
            key_padding_mask: (B, T) True where tokens are padding.

        Returns:
            (B, output_dim) pooled global feature.
        """
        B = x.shape[0]
        query = self.query.expand(B, -1, -1)  # (B, 1, D)
        pooled, _ = self.attn(query, x, x, key_padding_mask=key_padding_mask)
        pooled = pooled.squeeze(1)  # (B, D)
        return self.projection(pooled)


# ─── Main encoder ─────────────────────────────────────────────────────────────


class IOSPointCloudEncoder(nn.Module):
    """
    Encodes intraoral scan (IOS) point clouds into per-tooth and global features.

    Wraps the existing DGCNNToothEncoder (from occlusion/arch_encoder.py) and adds:
    - Per-tooth encoding via DGCNN
    - FDI-aware positional encoding (learned embedding per tooth number)
    - Cross-tooth attention transformer for inter-tooth relationships
    - Global arch feature via attention pooling
    - Outputs: per_tooth_features (B, T, tooth_dim), arch_global (B, arch_dim)

    When IOS is unavailable, returns learned zero-embedding so downstream
    modules receive tensors of the expected shape.

    The encoder handles variable numbers of teeth (missing teeth are masked)
    and supports both upper and lower arches.

    Args:
        config: IOSEncoderConfig with architecture hyperparameters.

    References:
    - Wang et al., "Dynamic Graph CNN" (2019) — DGCNN backbone
    - MICCAI TAPoseNet — per-tooth DGCNN architecture
    - arxiv 2312.15139 (TADPM) — dental arch cross-tooth attention
    """

    def __init__(self, config: Optional[IOSEncoderConfig] = None) -> None:
        super().__init__()
        if config is None:
            config = IOSEncoderConfig()
        self.config = config

        # Reuse existing DGCNN tooth encoder from occlusion module
        self.tooth_encoder = DGCNNToothEncoder(
            k=config.dgcnn_k, embed_dim=config.per_tooth_dim,
        )

        # FDI positional encoding — maps FDI number to a learned embedding
        # FDI range: 11-18, 21-28, 31-38, 41-48 → index via raw FDI number (0=padding)
        self.fdi_embedding = nn.Embedding(49, config.fdi_embed_dim)

        # Fuse per-tooth DGCNN embedding with FDI positional encoding
        self.tooth_fusion = nn.Sequential(
            nn.Linear(config.per_tooth_dim + config.fdi_embed_dim, config.per_tooth_dim),
            nn.LayerNorm(config.per_tooth_dim),
            nn.ReLU(inplace=True),
        )

        # Cross-tooth transformer layers
        self.cross_tooth_layers = nn.ModuleList([
            CrossToothTransformerLayer(
                d_model=config.per_tooth_dim,
                nhead=config.num_attention_heads,
                dim_feedforward=config.feedforward_dim,
                dropout=config.dropout_rate,
            )
            for _ in range(config.num_cross_tooth_layers)
        ])
        self.final_norm = nn.LayerNorm(config.per_tooth_dim)

        # Global arch pooling
        self.arch_pool = AttentionPooling(config.per_tooth_dim, config.arch_dim)

        # Learned null embeddings for CT-only mode (no IOS)
        self.null_tooth_embedding = nn.Parameter(
            torch.zeros(1, 1, config.per_tooth_dim),
        )
        self.null_arch_embedding = nn.Parameter(
            torch.zeros(1, config.arch_dim),
        )

        self.points_per_tooth = config.points_per_tooth

    def forward(
        self,
        tooth_point_clouds: Optional[List[List[torch.Tensor]]] = None,
        fdi_numbers: Optional[List[List[int]]] = None,
        batch_size: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode IOS dental scans into per-tooth and global arch features.

        Args:
            tooth_point_clouds: List of B items, each a list of (P, 3) tensors
                               for each tooth. None if IOS unavailable.
            fdi_numbers: List of B items, each a list of FDI tooth numbers.
                        Same structure as tooth_point_clouds.
            batch_size: Batch size (used when IOS is None to shape null embeddings).

        Returns:
            per_tooth_features: (B, max_teeth, per_tooth_dim) — per-tooth features
                               with zero-padding for missing teeth. When IOS is None,
                               returns learned null embedding.
            arch_global: (B, arch_dim) — global arch feature vector.
        """
        device = next(self.parameters()).device

        # IOS unavailable → return learned null embeddings
        if tooth_point_clouds is None or fdi_numbers is None:
            null_teeth = self.null_tooth_embedding.expand(
                batch_size, self.config.max_teeth, -1,
            )
            null_arch = self.null_arch_embedding.expand(batch_size, -1)
            return null_teeth, null_arch

        B = len(tooth_point_clouds)
        T_max = self.config.max_teeth

        # Encode each sample in the batch
        all_tooth_features = []
        all_padding_masks = []

        for b_idx in range(B):
            pcs = tooth_point_clouds[b_idx]
            fdis = fdi_numbers[b_idx]
            n_teeth = len(pcs)

            if n_teeth == 0:
                # No teeth in this sample
                feat = self.null_tooth_embedding.expand(1, T_max, -1).squeeze(0)
                mask = torch.ones(T_max, dtype=torch.bool, device=device)
                all_tooth_features.append(feat)
                all_padding_masks.append(mask)
                continue

            # Build PyG batch for DGCNN encoding
            data_list = []
            for pc in pcs:
                pc_t = pc.to(device) if isinstance(pc, torch.Tensor) else torch.tensor(
                    pc, dtype=torch.float32, device=device,
                )
                pc_t = self._normalize_point_count(pc_t)
                data_list.append(Data(pos=pc_t))

            batch_pyg = Batch.from_data_list(data_list)
            tooth_embeds = self.tooth_encoder(
                batch_pyg.pos, batch_pyg.batch,
            )  # (n_teeth, per_tooth_dim)

            # FDI positional encoding
            fdi_tensor = torch.tensor(fdis, dtype=torch.long, device=device)
            fdi_embeds = self.fdi_embedding(fdi_tensor)  # (n_teeth, fdi_embed_dim)

            # Fuse DGCNN features with FDI position
            fused = torch.cat([tooth_embeds, fdi_embeds], dim=-1)
            tooth_feat = self.tooth_fusion(fused)  # (n_teeth, per_tooth_dim)

            # Pad to max_teeth
            if n_teeth < T_max:
                padding = torch.zeros(
                    T_max - n_teeth, self.config.per_tooth_dim, device=device,
                )
                tooth_feat = torch.cat([tooth_feat, padding], dim=0)

            mask = torch.zeros(T_max, dtype=torch.bool, device=device)
            mask[n_teeth:] = True  # True = padding position

            all_tooth_features.append(tooth_feat)
            all_padding_masks.append(mask)

        # Stack batch: (B, T_max, per_tooth_dim)
        per_tooth = torch.stack(all_tooth_features, dim=0)
        padding_mask = torch.stack(all_padding_masks, dim=0)  # (B, T_max)

        # Cross-tooth transformer attention
        for layer in self.cross_tooth_layers:
            per_tooth = layer(per_tooth, key_padding_mask=padding_mask)
        per_tooth = self.final_norm(per_tooth)

        # Global arch feature via attention pooling
        arch_global = self.arch_pool(per_tooth, key_padding_mask=padding_mask)

        return per_tooth, arch_global

    def _normalize_point_count(self, points: torch.Tensor) -> torch.Tensor:
        """Subsample or pad a point cloud to self.points_per_tooth."""
        n = points.shape[0]
        target = self.points_per_tooth
        if n == target:
            return points
        elif n > target:
            idx = torch.randperm(n, device=points.device)[:target]
            return points[idx]
        else:
            pad_idx = torch.randint(0, n, (target - n,), device=points.device)
            return torch.cat([points, points[pad_idx]], dim=0)

    def encode_arch_numpy(
        self,
        tooth_meshes: Dict[int, np.ndarray],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method: encode from numpy arrays (single sample, no batch dim).

        Args:
            tooth_meshes: Dict mapping FDI number → (N, 3) numpy point cloud.

        Returns:
            per_tooth: (1, max_teeth, per_tooth_dim)
            arch_global: (1, arch_dim)
        """
        fdi_order = sorted(tooth_meshes.keys())
        point_clouds = [
            torch.tensor(tooth_meshes[fdi], dtype=torch.float32)
            for fdi in fdi_order
        ]
        with torch.no_grad():
            return self.forward(
                tooth_point_clouds=[point_clouds],
                fdi_numbers=[fdi_order],
                batch_size=1,
            )
