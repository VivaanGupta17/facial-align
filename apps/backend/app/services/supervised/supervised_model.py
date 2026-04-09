"""
End-to-end supervised model for fracture reduction and occlusion planning.

FacialAlignSupervisedModel assembles the complete inference pipeline:
  CT Volume → CT Encoder → ┐
                           ├→ Multimodal Fusion → Fragment Transforms
  IOS Scans → IOS Encoder → ┘                  → Tooth Transforms
                                                → Occlusion Metrics
                                                → Uncertainty Estimates

The model is designed to replace the optimisation-based pipeline as the
primary prediction path, with the optimiser preserved as a fallback for
low-confidence cases.

Key design decisions:
- R6 continuous rotation representation (no quaternion discontinuities)
- 30% IOS dropout during training (CT-only robustness)
- Identity-biased output initialisation (safe default predictions)
- MC Dropout for epistemic uncertainty at inference

References:
- FracFormer (IEEE TMI 2025): 1.85mm / 3.40° fragment accuracy
- Swin-T Tooth (ICCV 2025): 1.16mm / 2.77° tooth accuracy
- CMAF-Net: Cross-modal attention for missing modality robustness
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from app.services.supervised.ct_encoder import CTEncoderConfig, CTVolumeEncoder
from app.services.supervised.ios_encoder import IOSEncoderConfig, IOSPointCloudEncoder
from app.services.supervised.multimodal_fusion import FusionConfig, MultimodalFusionModule
from app.services.supervised.prediction_heads import (
    FragmentTransformHead,
    OcclusionScoringHead,
    ToothTransformHead,
    UncertaintyHead,
)

logger = logging.getLogger(__name__)


# ─── Configuration ────────────────────────────────────────────────────────────


@dataclass
class SupervisedModelConfig:
    """
    Full configuration for FacialAlignSupervisedModel.

    All sub-module configs can be overridden individually or left at defaults.
    """
    # Encoder configs
    ct_encoder: CTEncoderConfig = field(default_factory=CTEncoderConfig)
    ios_encoder: IOSEncoderConfig = field(default_factory=IOSEncoderConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)

    # Head configs
    max_fragments: int = 8
    max_teeth: int = 32

    # Uncertainty
    mc_dropout_p: float = 0.1
    mc_dropout_passes: int = 10

    # General
    use_gradient_checkpointing: bool = False
    use_mixed_precision: bool = True

    @property
    def fused_dim(self) -> int:
        """Fused feature dimension including modality indicator."""
        return self.fusion.fused_dim + 1


# ─── Main model ───────────────────────────────────────────────────────────────


class FacialAlignSupervisedModel(nn.Module):
    """
    End-to-end supervised model for craniomaxillofacial fracture reduction
    and dental occlusion planning.

    Input:
        - CT volume: (B, 1, D, H, W) CBCT/CT scan resampled to 0.4mm isotropic
        - IOS point clouds (optional): per-tooth point clouds (B, T, P, 3)

    Output (dict):
        - fragment_transforms: Per-fragment SE(3) predictions
        - tooth_transforms: Per-tooth SE(3) predictions
        - occlusion_scores: Clinical metric predictions
        - uncertainty: Aleatoric variance estimates
        - metadata: Modality flags, confidence summary

    Training:
        - Use SupervisedReductionLoss from supervised_losses.py
        - Two-stage schedule: MSE rotation (epochs 1-50) → geodesic (51+)
        - IOS randomly dropped with p=0.3 during training

    Inference:
        - Single forward pass for fast prediction
        - MC Dropout (T=10 passes) for uncertainty estimation
        - Confidence-based routing to fallback optimiser

    Args:
        config: SupervisedModelConfig with all hyperparameters.
    """

    def __init__(self, config: Optional[SupervisedModelConfig] = None) -> None:
        super().__init__()
        if config is None:
            config = SupervisedModelConfig()
        self.config = config

        # ── Encoders ──
        self.ct_encoder = CTVolumeEncoder(config.ct_encoder)
        self.ios_encoder = IOSPointCloudEncoder(config.ios_encoder)

        # ── Fusion ──
        self.fusion = MultimodalFusionModule(config.fusion)

        # ── Prediction heads ──
        fused_dim = config.fused_dim
        self.fragment_head = FragmentTransformHead(
            fused_dim=fused_dim,
            max_fragments=config.max_fragments,
        )
        self.tooth_head = ToothTransformHead(
            fused_dim=fused_dim,
            max_teeth=config.max_teeth,
        )
        self.scoring_head = OcclusionScoringHead(fused_dim=fused_dim)
        self.uncertainty_head = UncertaintyHead(
            fused_dim=fused_dim,
            max_fragments=config.max_fragments,
            max_teeth=config.max_teeth,
        )

        # MC Dropout for epistemic uncertainty at inference
        self.mc_dropout = nn.Dropout(p=config.mc_dropout_p)

        # Gradient checkpointing for memory efficiency
        if config.use_gradient_checkpointing:
            self.ct_encoder.set_grad_checkpointing(True)

        # Log model size
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "FacialAlignSupervisedModel: %.1fM params (%.1fM trainable)",
            total_params / 1e6,
            trainable_params / 1e6,
        )

    def forward(
        self,
        ct_volume: torch.Tensor,
        ios_point_clouds: Optional[torch.Tensor] = None,
        ios_tooth_ids: Optional[torch.Tensor] = None,
        num_fragments: Optional[torch.Tensor] = None,
        tooth_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Full forward pass.

        Args:
            ct_volume: (B, 1, D, H, W) CT volume in Hounsfield units.
            ios_point_clouds: (B, T, P, 3) per-tooth IOS point clouds.
                              None if IOS unavailable.
            ios_tooth_ids: (B, T) FDI tooth IDs for each IOS tooth.
            num_fragments: (B,) number of active bone fragments per sample.
            tooth_mask: (B, max_teeth) True for missing teeth.

        Returns:
            Dict containing all predictions:
            - "fragment_transforms": Dict from FragmentTransformHead
            - "tooth_transforms": Dict from ToothTransformHead
            - "occlusion_scores": Dict from OcclusionScoringHead
            - "uncertainty": Dict from UncertaintyHead
            - "fused_features": (B, fused_dim) for auxiliary use
            - "ios_available": bool
        """
        # ── Encode CT ──
        ct_global, ct_patches = self.ct_encoder(ct_volume)

        # ── Encode IOS (if available) ──
        ios_per_tooth = None
        ios_arch = None
        ios_padding_mask = None

        if ios_point_clouds is not None:
            ios_per_tooth, ios_arch = self.ios_encoder(
                ios_point_clouds, ios_tooth_ids,
            )
            # Build padding mask from tooth_mask if provided
            if tooth_mask is not None:
                # tooth_mask is already (B, max_teeth) with True = missing
                ios_padding_mask = tooth_mask

        # ── Fuse modalities ──
        fused, ios_used = self.fusion(
            ct_global=ct_global,
            ct_patches=ct_patches,
            ios_per_tooth=ios_per_tooth,
            ios_arch=ios_arch,
            ios_mask=ios_padding_mask,
        )

        # Apply MC dropout (active during both training and MC-inference)
        fused_for_heads = self.mc_dropout(fused)

        # ── Prediction heads ──
        fragment_pred = self.fragment_head(fused_for_heads, num_fragments)
        tooth_pred = self.tooth_head(fused_for_heads, tooth_mask)
        scoring_pred = self.scoring_head(fused_for_heads)
        uncertainty_pred = self.uncertainty_head(fused_for_heads)

        return {
            "fragment_transforms": fragment_pred,
            "tooth_transforms": tooth_pred,
            "occlusion_scores": scoring_pred,
            "uncertainty": uncertainty_pred,
            "fused_features": fused,
            "ios_available": ios_used,
        }

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        ct_volume: torch.Tensor,
        ios_point_clouds: Optional[torch.Tensor] = None,
        ios_tooth_ids: Optional[torch.Tensor] = None,
        num_fragments: Optional[torch.Tensor] = None,
        tooth_mask: Optional[torch.Tensor] = None,
        num_mc_passes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Inference with MC Dropout uncertainty estimation.

        Runs T forward passes with dropout enabled, then computes:
        - Mean predictions (more robust than single pass)
        - Epistemic variance (model uncertainty from prediction spread)
        - Combined confidence (aleatoric + epistemic)

        Args:
            ct_volume: (B, 1, D, H, W) CT volume.
            ios_point_clouds: Optional (B, T, P, 3) IOS point clouds.
            ios_tooth_ids: Optional (B, T) FDI IDs.
            num_fragments: Optional (B,) fragment counts.
            tooth_mask: Optional (B, max_teeth) tooth mask.
            num_mc_passes: Number of MC dropout passes (default from config).

        Returns:
            Dict with averaged predictions + uncertainty estimates.
        """
        T = num_mc_passes or self.config.mc_dropout_passes

        # Enable dropout for MC sampling
        self.mc_dropout.train()

        predictions: List[Dict[str, Any]] = []
        for _ in range(T):
            pred = self.forward(
                ct_volume, ios_point_clouds, ios_tooth_ids,
                num_fragments, tooth_mask,
            )
            predictions.append(pred)

        # Disable dropout
        self.mc_dropout.eval()

        # Average predictions
        result = self._average_predictions(predictions)

        # Compute epistemic uncertainty
        frag_epistemic = UncertaintyHead.compute_epistemic_uncertainty(
            [p["fragment_transforms"] for p in predictions], key="translations",
        )
        tooth_epistemic = UncertaintyHead.compute_epistemic_uncertainty(
            [p["tooth_transforms"] for p in predictions], key="translations",
        )

        result["epistemic_uncertainty"] = {
            "fragment_translation_var": frag_epistemic,
            "tooth_translation_var": tooth_epistemic,
        }

        # Combined confidence
        aleatoric = result["uncertainty"]
        temperature = aleatoric["temperature"]

        result["combined_confidence"] = {
            "fragment": UncertaintyHead.combine_confidence(
                aleatoric["fragment_log_var"], frag_epistemic.unsqueeze(-1), temperature,
            ),
            "tooth": UncertaintyHead.combine_confidence(
                aleatoric["tooth_log_var"], tooth_epistemic.unsqueeze(-1), temperature,
            ),
        }

        return result

    def _average_predictions(
        self, predictions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Average predictions from multiple MC dropout passes."""
        T = len(predictions)
        base = predictions[0]

        # Average fragment transforms
        avg_frag_t = torch.stack(
            [p["fragment_transforms"]["translations"] for p in predictions],
        ).mean(dim=0)
        avg_frag_conf = torch.stack(
            [p["fragment_transforms"]["confidences"] for p in predictions],
        ).mean(dim=0)

        # Average tooth transforms
        avg_tooth_t = torch.stack(
            [p["tooth_transforms"]["translations"] for p in predictions],
        ).mean(dim=0)
        avg_tooth_conf = torch.stack(
            [p["tooth_transforms"]["confidences"] for p in predictions],
        ).mean(dim=0)

        # Average metrics
        avg_metrics = torch.stack(
            [p["occlusion_scores"]["metrics"] for p in predictions],
        ).mean(dim=0)
        avg_class_probs = torch.stack(
            [p["occlusion_scores"]["molar_class_probs"] for p in predictions],
        ).mean(dim=0)

        # For rotations, use the last pass (averaging R6 then converting is
        # better than averaging rotation matrices which may not be valid SO(3))
        result = {
            "fragment_transforms": {
                **base["fragment_transforms"],
                "translations": avg_frag_t,
                "confidences": avg_frag_conf,
            },
            "tooth_transforms": {
                **base["tooth_transforms"],
                "translations": avg_tooth_t,
                "confidences": avg_tooth_conf,
            },
            "occlusion_scores": {
                "metrics": avg_metrics,
                "molar_class_probs": avg_class_probs,
                "molar_class_logits": base["occlusion_scores"]["molar_class_logits"],
            },
            "uncertainty": base["uncertainty"],
            "fused_features": base["fused_features"],
            "ios_available": base["ios_available"],
        }
        return result

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config: Optional[SupervisedModelConfig] = None,
        map_location: str = "cpu",
    ) -> "FacialAlignSupervisedModel":
        """
        Load model from a training checkpoint.

        Args:
            checkpoint_path: Path to .pt checkpoint file.
            config: Model config. If None, uses config saved in checkpoint.
            map_location: Device to map weights.

        Returns:
            Loaded FacialAlignSupervisedModel.
        """
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

        if config is None and "config" in checkpoint:
            config = checkpoint["config"]
        if config is None:
            config = SupervisedModelConfig()

        model = cls(config)

        state_key = "model_state_dict" if "model_state_dict" in checkpoint else "state_dict"
        if state_key in checkpoint:
            missing, unexpected = model.load_state_dict(checkpoint[state_key], strict=False)
            if missing:
                logger.warning("Missing keys: %d", len(missing))
            if unexpected:
                logger.warning("Unexpected keys: %d", len(unexpected))
        else:
            # Assume checkpoint IS the state dict
            model.load_state_dict(checkpoint, strict=False)

        logger.info("Loaded model from %s", checkpoint_path)
        return model

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata for registry integration."""
        return {
            "name": "facial_align_supervised",
            "architecture": "CT-ResNet50 + DGCNN-IOS + CMAF-Fusion + SE3-Heads",
            "ct_encoder": "3D ResNet-50 (MONAI-style)",
            "ios_encoder": "DGCNN + Cross-Tooth Transformer",
            "fusion": "Bidirectional Cross-Attention (CMAF-Net)",
            "rotation_repr": "R6 continuous (Gram-Schmidt)",
            "max_fragments": self.config.max_fragments,
            "max_teeth": self.config.max_teeth,
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
