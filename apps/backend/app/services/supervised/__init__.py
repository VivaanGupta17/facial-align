"""
Supervised-learning-first surgical planning module.

Replaces the optimization-based pipeline with an end-to-end supervised model
that predicts per-fragment and per-tooth SE(3) transforms from CT + optional IOS.

Modules:
- ct_encoder: 3D ResNet encoder for maxillofacial CT volumes
- ios_encoder: DGCNN-based IOS point cloud encoder (wraps occlusion.arch_encoder)
- multimodal_fusion: Cross-attention CT+IOS fusion (CMAF-Net pattern)
- prediction_heads: Fragment/tooth transform, occlusion scoring, uncertainty heads
- supervised_model: End-to-end FacialAlignSupervisedModel
- supervised_losses: Geodesic SO(3) + clinical + composite dental losses
- inference_service: Production inference with confidence-based fallback

References:
- Zhou et al., "On the Continuity of Rotation Representations" (CVPR 2019)
- Wang et al., "Dynamic Graph CNN for Learning on Point Clouds" (2019)
- CMAF-Net: Cross-Modal Attention Fusion Network for medical imaging
- PMC11574221: Composite objective for simultaneous dental occlusion + fracture fitting
"""

from app.services.supervised.ct_encoder import CTVolumeEncoder
from app.services.supervised.ios_encoder import IOSPointCloudEncoder
from app.services.supervised.multimodal_fusion import MultimodalFusionModule
from app.services.supervised.prediction_heads import (
    FragmentTransformHead,
    OcclusionScoringHead,
    ToothTransformHead,
    UncertaintyHead,
)
from app.services.supervised.supervised_losses import SupervisedReductionLoss
from app.services.supervised.supervised_model import (
    FacialAlignSupervisedModel,
    SupervisedModelConfig,
)
from app.services.supervised.inference_service import SupervisedInferenceService

__all__ = [
    "CTVolumeEncoder",
    "IOSPointCloudEncoder",
    "MultimodalFusionModule",
    "FragmentTransformHead",
    "ToothTransformHead",
    "OcclusionScoringHead",
    "UncertaintyHead",
    "FacialAlignSupervisedModel",
    "SupervisedModelConfig",
    "SupervisedReductionLoss",
    "SupervisedInferenceService",
]
