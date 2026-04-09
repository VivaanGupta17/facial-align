"""
PyTorch datasets for supervised training.

- FractureDataset: pre/post-op CT pair dataset
- DentalDataset: IOS-only dental occlusion dataset
- NormalReferenceDataset: normal anatomy reference
- data_augmentation: 3D-specific augmentations
"""

from training.datasets.data_augmentation import AugmentationConfig, augment_point_cloud
from training.datasets.dental_dataset import DentalDataset
from training.datasets.fracture_dataset import FractureDataset
from training.datasets.normal_reference_dataset import NormalReferenceDataset

__all__ = [
    "AugmentationConfig",
    "DentalDataset",
    "FractureDataset",
    "NormalReferenceDataset",
    "augment_point_cloud",
]
