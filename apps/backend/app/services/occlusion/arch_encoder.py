"""
Dental arch point cloud encoder using PyG's built-in DGCNN backbone.

Encodes per-tooth point clouds into feature embeddings for downstream
occlusion analysis, landmark detection, and SE(3) transform prediction.

References:
- Wang et al., "Dynamic Graph CNN for Learning on Point Clouds" (2019)
- MICCAI TAPoseNet: DGCNN for tooth pose estimation
- arxiv 2312.15139 (TADPM): PointNet++ encoder for dental SE(3) transforms
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import DynamicEdgeConv, MLP, global_max_pool

logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────

DEFAULT_POINTS_PER_TOOTH = 1024
PER_TOOTH_EMBED_DIM = 256
GLOBAL_ARCH_EMBED_DIM = 512
FDI_UPPER = list(range(11, 29))  # 11-18, 21-28
FDI_LOWER = list(range(31, 49))  # 31-38, 41-48
NUM_TEETH_MAX = 32  # Full dentition


class DGCNNToothEncoder(nn.Module):
    """
    Per-tooth point cloud encoder using DynamicEdgeConv (DGCNN).

    Takes a single tooth point cloud (P x 3) and produces a D-dimensional
    feature vector. Uses torch_geometric's DynamicEdgeConv layers which
    dynamically recompute k-NN graphs at each layer.

    Architecture (per MICCAI TAPoseNet):
        Input (P x 3) → EdgeConv(3→64) → EdgeConv(64→128) → EdgeConv(128→256)
        → global_max_pool → MLP(256→256) → per-tooth embedding

    Args:
        k: Number of nearest neighbors for dynamic graph construction.
        embed_dim: Output embedding dimension.
    """

    def __init__(self, k: int = 20, embed_dim: int = PER_TOOTH_EMBED_DIM) -> None:
        super().__init__()
        self.k = k
        self.embed_dim = embed_dim

        # DynamicEdgeConv layers — each takes (xi || xj - xi) as input
        # so input channels to each MLP are 2 * in_channels
        self.conv1 = DynamicEdgeConv(
            nn=MLP([2 * 3, 64, 64]),
            k=k,
        )
        self.conv2 = DynamicEdgeConv(
            nn=MLP([2 * 64, 128, 128]),
            k=k,
        )
        self.conv3 = DynamicEdgeConv(
            nn=MLP([2 * 128, 256, 256]),
            k=k,
        )

        # Final projection to embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of tooth point clouds.

        Args:
            x: (N_total, 3) — concatenated point positions for all teeth in batch.
            batch: (N_total,) — batch index per point (from torch_geometric Batch).

        Returns:
            (B, embed_dim) — per-tooth embeddings, one per tooth in the batch.
        """
        x = self.conv1(x, batch)  # (N_total, 64)
        x = self.conv2(x, batch)  # (N_total, 128)
        x = self.conv3(x, batch)  # (N_total, 256)

        # Global max pool per tooth
        x = global_max_pool(x, batch)  # (B, 256)

        # Project to final embedding
        x = self.projection(x)  # (B, embed_dim)
        return x


class DentalArchEncoder(nn.Module):
    """
    Full dental arch encoder: per-tooth DGCNN → tooth-level features
    → arch-level aggregation with positional encoding.

    Input: per-tooth point clouds (batch of N teeth, each P points x 3)
    Output: per-tooth embeddings (N x D) + global arch embedding (1 x D_global)

    The encoder adds FDI positional encoding so the network knows which
    tooth each point cloud represents (spatial position in the arch).

    Architecture:
        1. DGCNNToothEncoder: per-tooth point cloud → 256-dim embedding
        2. FDI positional encoding (learned, 32-dim)
        3. Arch aggregation MLP: concat all tooth embeddings → 512-dim global

    Reference:
    - Wang et al., "Dynamic Graph CNN" (2019) — backbone
    - MICCAI TAPoseNet — per-tooth DGCNN architecture
    - arxiv 2312.15139 (TADPM) — dental arch-level aggregation
    """

    def __init__(
        self,
        k: int = 20,
        per_tooth_dim: int = PER_TOOTH_EMBED_DIM,
        global_dim: int = GLOBAL_ARCH_EMBED_DIM,
        max_teeth: int = NUM_TEETH_MAX,
        points_per_tooth: int = DEFAULT_POINTS_PER_TOOTH,
    ) -> None:
        super().__init__()
        self.per_tooth_dim = per_tooth_dim
        self.global_dim = global_dim
        self.max_teeth = max_teeth
        self.points_per_tooth = points_per_tooth

        # Per-tooth DGCNN encoder
        self.tooth_encoder = DGCNNToothEncoder(k=k, embed_dim=per_tooth_dim)

        # Learned FDI positional encoding (48 FDI numbers → 32-dim)
        self.fdi_embedding = nn.Embedding(49, 32)  # FDI 0-48 (0 = padding)

        # Fusion: per-tooth embedding + FDI position → refined per-tooth
        self.tooth_fusion = nn.Sequential(
            nn.Linear(per_tooth_dim + 32, per_tooth_dim),
            nn.LayerNorm(per_tooth_dim),
            nn.ReLU(),
        )

        # Arch-level aggregation: attention-weighted pool over teeth
        self.arch_attention = nn.Sequential(
            nn.Linear(per_tooth_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.arch_projection = nn.Sequential(
            nn.Linear(per_tooth_dim, global_dim),
            nn.LayerNorm(global_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        tooth_point_clouds: List[torch.Tensor],
        fdi_numbers: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a dental arch from per-tooth point clouds.

        Args:
            tooth_point_clouds: List of (P, 3) tensors, one per tooth.
            fdi_numbers: List of FDI tooth numbers (11-48), same length.

        Returns:
            per_tooth_embeddings: (N_teeth, per_tooth_dim) — per-tooth features.
            global_embedding: (1, global_dim) — arch-level feature.
        """
        if len(tooth_point_clouds) == 0:
            device = next(self.parameters()).device
            return (
                torch.zeros(0, self.per_tooth_dim, device=device),
                torch.zeros(1, self.global_dim, device=device),
            )

        device = next(self.parameters()).device

        # Build a PyG Batch from per-tooth point clouds
        data_list = []
        for pc in tooth_point_clouds:
            pc_tensor = pc.to(device) if isinstance(pc, torch.Tensor) else torch.tensor(
                pc, dtype=torch.float32, device=device
            )
            # Subsample or pad to fixed point count
            pc_tensor = self._normalize_point_count(pc_tensor)
            data_list.append(Data(pos=pc_tensor))

        batch = Batch.from_data_list(data_list)
        x = batch.pos  # (N_total, 3)
        batch_idx = batch.batch  # (N_total,)

        # Per-tooth encoding via DGCNN
        tooth_embeds = self.tooth_encoder(x, batch_idx)  # (N_teeth, per_tooth_dim)

        # FDI positional encoding
        fdi_tensor = torch.tensor(fdi_numbers, dtype=torch.long, device=device)
        fdi_embeds = self.fdi_embedding(fdi_tensor)  # (N_teeth, 32)

        # Fuse tooth features with FDI position
        fused = torch.cat([tooth_embeds, fdi_embeds], dim=-1)  # (N_teeth, per_tooth_dim + 32)
        per_tooth = self.tooth_fusion(fused)  # (N_teeth, per_tooth_dim)

        # Arch-level aggregation via attention-weighted pooling
        attn_weights = self.arch_attention(per_tooth)  # (N_teeth, 1)
        attn_weights = torch.softmax(attn_weights, dim=0)
        weighted = (per_tooth * attn_weights).sum(dim=0, keepdim=True)  # (1, per_tooth_dim)
        global_embed = self.arch_projection(weighted)  # (1, global_dim)

        return per_tooth, global_embed

    def _normalize_point_count(self, points: torch.Tensor) -> torch.Tensor:
        """Subsample or pad a point cloud to self.points_per_tooth."""
        n = points.shape[0]
        target = self.points_per_tooth

        if n == target:
            return points
        elif n > target:
            # Random subsample
            idx = torch.randperm(n, device=points.device)[:target]
            return points[idx]
        else:
            # Pad by repeating random points
            pad_idx = torch.randint(0, n, (target - n,), device=points.device)
            return torch.cat([points, points[pad_idx]], dim=0)

    def encode_arch(
        self,
        tooth_meshes: Dict[int, np.ndarray],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Convenience method: encode a dental arch from numpy point clouds.

        Args:
            tooth_meshes: Dict mapping FDI number → (N, 3) numpy point cloud.

        Returns:
            per_tooth_embeddings: (N_teeth, per_tooth_dim)
            global_embedding: (1, global_dim)
            fdi_order: List of FDI numbers in the order of per-tooth embeddings.
        """
        fdi_order = sorted(tooth_meshes.keys())
        point_clouds = [
            torch.tensor(tooth_meshes[fdi], dtype=torch.float32)
            for fdi in fdi_order
        ]

        with torch.no_grad():
            per_tooth, global_embed = self.forward(point_clouds, fdi_order)

        return per_tooth, global_embed, fdi_order

    def load_weights(self, path: Path) -> None:
        """Load pretrained weights, falling back to random init."""
        if path.exists():
            state = torch.load(path, map_location="cpu", weights_only=True)
            self.load_state_dict(state, strict=False)
            logger.info("Loaded DentalArchEncoder weights from %s", path)
        else:
            logger.info(
                "No pretrained weights at %s — using random initialization", path
            )

    def save_weights(self, path: Path) -> None:
        """Save current weights."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info("Saved DentalArchEncoder weights to %s", path)
