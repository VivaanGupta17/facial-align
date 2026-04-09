"""
Training loop implementations.

- BaseTrainer: shared infrastructure (mixed precision, checkpointing, etc.)
- OcclusionTrainer: OcclusionModel supervised + self-supervised training
- ReductionTrainer: reduction pipeline end-to-end training
- LandmarkTrainer: DentalLandmarkDetector supervised training
- ScoringTrainer: OcclusionScoringHead supervised training
"""

from training.trainers.base_trainer import BaseTrainer
from training.trainers.landmark_trainer import LandmarkTrainer
from training.trainers.occlusion_trainer import OcclusionTrainer
from training.trainers.reduction_trainer import ReductionTrainer
from training.trainers.scoring_trainer import ScoringTrainer

__all__ = [
    "BaseTrainer",
    "LandmarkTrainer",
    "OcclusionTrainer",
    "ReductionTrainer",
    "ScoringTrainer",
]
