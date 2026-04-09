"""
3D CNN encoder for maxillofacial CT volumes.

Extracts volumetric features from CBCT/CT scans using a 3D ResNet backbone.
Architecture follows MONAI's ResNet3D with modifications for dental HU range
(clip to [0, 3000] for bone visualization).

References:
- He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016)
- Chen et al., "Med3D: Transfer Learning for 3D Medical Image Analysis" (2019)
- Hatamizadeh et al., "Swin UNETR" (CVPR 2022) — optional pretrained init
- MONAI: Medical Open Network for AI (Project MONAI, 2020)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────

DEFAULT_CT_FEAT_DIM = 512
HU_MIN = 0.0
HU_MAX = 3000.0
ISOTROPIC_SPACING_MM = 0.4


@dataclass
class CTEncoderConfig:
    """Configuration for CTVolumeEncoder."""
    ct_feat_dim: int = DEFAULT_CT_FEAT_DIM
    in_channels: int = 1
    base_channels: int = 64
    layer_depths: Tuple[int, ...] = (3, 4, 6, 3)
    hu_min: float = HU_MIN
    hu_max: float = HU_MAX
    dropout_rate: float = 0.0


# ─── Building blocks ─────────────────────────────────────────────────────────


class ResidualBlock3D(nn.Module):
    """
    3D residual block with two 3x3x3 convolutions and a skip connection.

    Follows the standard ResNet bottleneck pattern adapted for 3D volumes.
    Uses GroupNorm instead of BatchNorm for stability with small batch sizes
    typical in medical imaging (B=1-4).

    Architecture:
        x → Conv3d(3x3x3) → GroupNorm → ReLU → Conv3d(3x3x3) → GroupNorm → + x → ReLU

    If in_channels != out_channels, a 1x1x1 projection shortcut is applied.
    If stride > 1, the shortcut uses strided convolution for downsampling.

    References:
    - He et al., "Deep Residual Learning" (CVPR 2016)
    - Wu & He, "Group Normalization" (ECCV 2018)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        # Ensure num_groups divides both in and out channels
        groups_1 = min(num_groups, out_channels)
        while out_channels % groups_1 != 0:
            groups_1 -= 1
        groups_2 = groups_1

        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.gn1 = nn.GroupNorm(groups_1, out_channels)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.gn2 = nn.GroupNorm(groups_2, out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Shortcut projection when dimensions change
        if stride != 1 or in_channels != out_channels:
            shortcut_groups = min(num_groups, out_channels)
            while out_channels % shortcut_groups != 0:
                shortcut_groups -= 1
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(shortcut_groups, out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: residual connection around two conv layers."""
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        out = out + identity
        out = self.relu(out)
        return out


class BottleneckBlock3D(nn.Module):
    """
    3D bottleneck residual block with 1x1 → 3x3x3 → 1x1 convolutions.

    Reduces computational cost by using a narrow 3x3x3 convolution between
    two 1x1 projections. Expansion factor is 4 (standard ResNet bottleneck).

    Architecture:
        x → Conv3d(1x1) → GN → ReLU → Conv3d(3x3x3) → GN → ReLU → Conv3d(1x1) → GN → + x → ReLU

    References:
    - He et al., "Deep Residual Learning" (CVPR 2016)
    """

    EXPANSION = 4

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        stride: int = 1,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        out_channels = mid_channels * self.EXPANSION

        def _groups(ch: int) -> int:
            g = min(num_groups, ch)
            while ch % g != 0:
                g -= 1
            return g

        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(_groups(mid_channels), mid_channels)

        self.conv2 = nn.Conv3d(
            mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.gn2 = nn.GroupNorm(_groups(mid_channels), mid_channels)

        self.conv3 = nn.Conv3d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.gn3 = nn.GroupNorm(_groups(out_channels), out_channels)

        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(_groups(out_channels), out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.relu(self.gn1(self.conv1(x)))
        out = self.relu(self.gn2(self.conv2(out)))
        out = self.gn3(self.conv3(out))

        out = out + identity
        out = self.relu(out)
        return out


# ─── Main encoder ─────────────────────────────────────────────────────────────


class CTVolumeEncoder(nn.Module):
    """
    3D ResNet encoder for maxillofacial CT volumes.

    Extracts volumetric features from CBCT/CT. Architecture follows
    MONAI's ResNet3D with modifications for dental HU range.

    Input: (B, 1, D, H, W) CT volume (resampled to 0.4mm isotropic, HU clipped [0, 3000])
    Output: (B, ct_feat_dim) global CT feature vector + (B, num_patches, ct_feat_dim) patch features

    Uses 3D ResNet-50 backbone with:
    - 7x7x7 stem convolution with stride 2
    - 4 residual stages with increasing channels (64→128→256→512)
    - Global average pooling → ct_feat_dim=512
    - Patch features from stage 4 feature maps (for cross-attention with IOS)

    The patch features are obtained by flattening the spatial dimensions of the
    stage-4 feature map, providing spatially-localized volumetric features that
    the multimodal fusion module uses as keys/values for cross-attention.

    Args:
        config: CTEncoderConfig with architecture hyperparameters.

    References:
    - He et al., "Deep Residual Learning" (CVPR 2016)
    - Chen et al., "Med3D" (2019) — 3D ResNet for medical volumes
    - Hatamizadeh et al., "Swin UNETR" (CVPR 2022) — pretrained 3D encoder
    """

    def __init__(self, config: Optional[CTEncoderConfig] = None) -> None:
        super().__init__()
        if config is None:
            config = CTEncoderConfig()
        self.config = config

        base = config.base_channels
        depths = config.layer_depths

        # ── Stem: 7x7x7 conv with stride 2, then max pool ──
        def _groups(ch: int) -> int:
            g = min(8, ch)
            while ch % g != 0:
                g -= 1
            return g

        self.stem = nn.Sequential(
            nn.Conv3d(
                config.in_channels, base, kernel_size=7, stride=2, padding=3, bias=False,
            ),
            nn.GroupNorm(_groups(base), base),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        # ── Residual stages ──
        # Stage 1: base channels, no downsample
        self.stage1 = self._make_stage(base, base, depths[0], stride=1)
        # Stage 2: 2x channels, stride-2 downsample
        self.stage2 = self._make_stage(base, base * 2, depths[1], stride=2)
        # Stage 3: 4x channels, stride-2 downsample
        self.stage3 = self._make_stage(base * 2, base * 4, depths[2], stride=2)
        # Stage 4: 8x channels, stride-2 downsample
        self.stage4 = self._make_stage(base * 4, base * 8, depths[3], stride=2)

        # ── Global feature extraction ──
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.global_projection = nn.Sequential(
            nn.Linear(base * 8, config.ct_feat_dim),
            nn.LayerNorm(config.ct_feat_dim),
            nn.ReLU(inplace=True),
        )

        # ── Patch feature projection (stage 4 feature maps → ct_feat_dim) ──
        self.patch_projection = nn.Sequential(
            nn.Linear(base * 8, config.ct_feat_dim),
            nn.LayerNorm(config.ct_feat_dim),
            nn.ReLU(inplace=True),
        )

        if config.dropout_rate > 0:
            self.dropout = nn.Dropout(config.dropout_rate)
        else:
            self.dropout = nn.Identity()

        # Initialize weights
        self._init_weights()

    def _make_stage(
        self, in_channels: int, out_channels: int, num_blocks: int, stride: int,
    ) -> nn.Sequential:
        """Build a residual stage with `num_blocks` ResidualBlock3D layers."""
        layers = [ResidualBlock3D(in_channels, out_channels, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock3D(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """Kaiming initialization for conv layers, zeros for norms."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def preprocess_hu(volume: torch.Tensor, hu_min: float = HU_MIN, hu_max: float = HU_MAX) -> torch.Tensor:
        """
        Preprocess CT volume: clip HU values and normalize to [0, 1].

        Dental/craniofacial CT is clipped to [0, 3000] HU to focus on bone
        structures while suppressing soft tissue and air.

        Args:
            volume: (B, 1, D, H, W) raw CT volume in Hounsfield units.
            hu_min: Lower HU clip bound (default 0 — excludes air/soft tissue).
            hu_max: Upper HU clip bound (default 3000 — includes dense bone).

        Returns:
            (B, 1, D, H, W) normalized volume in [0, 1].
        """
        volume = volume.clamp(hu_min, hu_max)
        volume = (volume - hu_min) / (hu_max - hu_min)
        return volume

    def forward(
        self, volume: torch.Tensor, preprocess: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a CT volume into global and patch-level features.

        Args:
            volume: (B, 1, D, H, W) CT volume. If preprocess=True, raw HU values
                    are expected and will be clipped/normalized. If preprocess=False,
                    volume should already be in [0, 1].
            preprocess: Whether to apply HU clipping and normalization.

        Returns:
            global_features: (B, ct_feat_dim) — global CT feature vector from GAP.
            patch_features: (B, num_patches, ct_feat_dim) — spatially-localized
                           features from stage-4 feature map, flattened over D'×H'×W'.
                           num_patches = D'×H'×W' where D',H',W' are the stage-4
                           spatial dimensions (input_size / 32 per axis).
        """
        if preprocess:
            volume = self.preprocess_hu(volume, self.config.hu_min, self.config.hu_max)

        # Stem: (B, 1, D, H, W) → (B, base, D/4, H/4, W/4)
        x = self.stem(volume)

        # Residual stages
        x = self.stage1(x)   # (B, base, D/4, H/4, W/4)
        x = self.stage2(x)   # (B, base*2, D/8, H/8, W/8)
        x = self.stage3(x)   # (B, base*4, D/16, H/16, W/16)
        x = self.stage4(x)   # (B, base*8, D/32, H/32, W/32)

        # Global feature via adaptive average pooling
        global_feat = self.global_pool(x)              # (B, base*8, 1, 1, 1)
        global_feat = global_feat.flatten(1)            # (B, base*8)
        global_feat = self.dropout(global_feat)
        global_feat = self.global_projection(global_feat)  # (B, ct_feat_dim)

        # Patch features: flatten spatial dims of stage-4 feature map
        B, C, D_p, H_p, W_p = x.shape
        patch_feat = x.permute(0, 2, 3, 4, 1)           # (B, D', H', W', C)
        patch_feat = patch_feat.reshape(B, -1, C)         # (B, num_patches, base*8)
        patch_feat = self.patch_projection(patch_feat)     # (B, num_patches, ct_feat_dim)

        return global_feat, patch_feat

    @classmethod
    def from_pretrained(
        cls,
        weights_path: Optional[str] = None,
        config: Optional[CTEncoderConfig] = None,
        map_location: str = "cpu",
    ) -> "CTVolumeEncoder":
        """
        Load a CTVolumeEncoder from pretrained weights.

        Supports loading from:
        1. Custom checkpoint (.pt/.pth) — direct state_dict load.
        2. MONAI SwinUNETR pretrained weights — extracts the encoder portion
           and maps feature dimensions to our architecture. Only the overlapping
           layers are loaded (strict=False).

        Args:
            weights_path: Path to .pt/.pth checkpoint. If None, returns randomly
                         initialized model.
            config: Encoder configuration. If None, uses default.
            map_location: Device to map weights to.

        Returns:
            Initialized CTVolumeEncoder with loaded weights.
        """
        if config is None:
            config = CTEncoderConfig()

        model = cls(config)

        if weights_path is None:
            logger.info("CTVolumeEncoder initialized with random weights.")
            return model

        path = Path(weights_path)
        if not path.exists():
            logger.warning("Weights file %s not found — using random initialization.", path)
            return model

        state_dict = torch.load(str(path), map_location=map_location, weights_only=True)

        # Handle MONAI SwinUNETR pretrained weights
        if any(k.startswith("swinViT.") for k in state_dict.keys()):
            logger.info("Detected MONAI SwinUNETR checkpoint — extracting encoder weights.")
            state_dict = cls._remap_monai_swin_weights(state_dict, config)

        # Handle wrapper checkpoint with 'model_state_dict' key
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.info("CTVolumeEncoder: %d missing keys (fine-tune layers).", len(missing))
        if unexpected:
            logger.info("CTVolumeEncoder: %d unexpected keys (skipped).", len(unexpected))

        return model

    @staticmethod
    def _remap_monai_swin_weights(
        swin_state: Dict[str, torch.Tensor], config: CTEncoderConfig,
    ) -> Dict[str, torch.Tensor]:
        """
        Map MONAI SwinUNETR encoder weights to our 3D ResNet architecture.

        Since SwinUNETR uses a Swin Transformer backbone and we use 3D ResNet,
        only the patch embedding and early downsampling layers have compatible
        shapes. We extract what we can and let strict=False handle the rest.

        Args:
            swin_state: State dict from a MONAI SwinUNETR checkpoint.
            config: Our encoder config.

        Returns:
            Remapped state dict with compatible keys.
        """
        remapped = {}
        # Map the patch embedding conv to our stem conv
        stem_key = "swinViT.patch_embed.proj.weight"
        if stem_key in swin_state:
            w = swin_state[stem_key]  # (embed_dim, in_ch, P, P, P)
            # Average over the patch dimension to get a 7x7x7 approximation
            target_shape = (config.base_channels, config.in_channels, 7, 7, 7)
            if w.shape != target_shape:
                w = F.interpolate(
                    w.unsqueeze(0),
                    size=(7, 7, 7),
                    mode="trilinear",
                    align_corners=False,
                ).squeeze(0)
                # Adjust channel dimensions if needed
                if w.shape[0] != config.base_channels:
                    w = w[:config.base_channels]
                if w.shape[1] != config.in_channels:
                    w = w[:, :config.in_channels]
            remapped["stem.0.weight"] = w

        return remapped
