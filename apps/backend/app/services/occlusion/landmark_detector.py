"""
Learned dental landmark detection from tooth point clouds.

Uses a small DGCNN (via torch_geometric) per tooth to predict anatomical
landmark positions: cusp tips, fossa centers, marginal ridges, incisal edges.

Landmarks are critical for:
- Molar relation assessment (cusp-fossa distances)
- Midline computation (incisor edge midpoints)
- Overjet/overbite measurement (incisal edge positions)
- Occlusal contact analysis

References:
- MICCAI TAPoseNet: DGCNN for tooth pose estimation
- PMC11574221: Dental landmark-based molar relation + midline metrics
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

# ─── Landmark definitions ────────────────────────────────────────────────────

# Standard dental landmarks per tooth type
MOLAR_LANDMARKS = [
    "mesial_cusp",
    "distal_cusp",
    "central_fossa",
    "mesial_marginal_ridge",
    "distal_marginal_ridge",
]
PREMOLAR_LANDMARKS = [
    "buccal_cusp",
    "lingual_cusp",
    "central_fossa",
    "mesial_marginal_ridge",
    "distal_marginal_ridge",
]
INCISOR_LANDMARKS = [
    "incisal_edge_center",
    "mesial_corner",
    "distal_corner",
]
CANINE_LANDMARKS = [
    "cusp_tip",
    "mesial_ridge",
    "distal_ridge",
]

# FDI tooth type mapping
FDI_TOOTH_TYPE = {}
for fdi in [11, 12, 21, 22, 31, 32, 41, 42]:
    FDI_TOOTH_TYPE[fdi] = "incisor"
for fdi in [13, 23, 33, 43]:
    FDI_TOOTH_TYPE[fdi] = "canine"
for fdi in [14, 15, 24, 25, 34, 35, 44, 45]:
    FDI_TOOTH_TYPE[fdi] = "premolar"
for fdi in [16, 17, 18, 26, 27, 28, 36, 37, 38, 46, 47, 48]:
    FDI_TOOTH_TYPE[fdi] = "molar"

LANDMARKS_BY_TYPE = {
    "molar": MOLAR_LANDMARKS,
    "premolar": PREMOLAR_LANDMARKS,
    "incisor": INCISOR_LANDMARKS,
    "canine": CANINE_LANDMARKS,
}

MAX_LANDMARKS = 5  # Maximum landmarks per tooth


class DGCNNLandmarkHead(nn.Module):
    """
    Small DGCNN network for per-tooth landmark regression.

    Takes a single tooth point cloud and predicts K landmark 3D positions.
    Uses DynamicEdgeConv from torch_geometric as backbone, with a regression
    head that outputs landmark coordinates.

    Architecture:
        EdgeConv(3→64) → EdgeConv(64→128) → global_max_pool →
        MLP(128→64→K*3) → reshape to (K, 3)
    """

    def __init__(
        self,
        k: int = 16,
        max_landmarks: int = MAX_LANDMARKS,
    ) -> None:
        super().__init__()
        self.k = k
        self.max_landmarks = max_landmarks

        self.conv1 = DynamicEdgeConv(
            nn=MLP([2 * 3, 64, 64]),
            k=k,
        )
        self.conv2 = DynamicEdgeConv(
            nn=MLP([2 * 64, 128, 128]),
            k=k,
        )

        self.regression_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, max_landmarks * 3),
        )

        # Landmark confidence head (predicts which landmarks are valid)
        self.confidence_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, max_landmarks),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (N_total, 3) concatenated tooth point positions.
            batch: (N_total,) batch index.

        Returns:
            landmarks: (B, max_landmarks, 3) predicted landmark positions.
            confidence: (B, max_landmarks) landmark confidence scores.
        """
        x = self.conv1(x, batch)  # (N_total, 64)
        x = self.conv2(x, batch)  # (N_total, 128)
        x = global_max_pool(x, batch)  # (B, 128)

        landmarks = self.regression_head(x)  # (B, K*3)
        landmarks = landmarks.view(-1, self.max_landmarks, 3)  # (B, K, 3)

        confidence = self.confidence_head(global_max_pool(
            self.conv2.nn[0](torch.zeros_like(x[:1]).expand(x.shape[0], -1))
            if False else x,
            torch.arange(x.shape[0], device=x.device),
        ))

        # Re-run confidence from the pooled features directly
        confidence = self.confidence_head(x)  # (B, K)

        return landmarks, confidence


class DentalLandmarkDetector(nn.Module):
    """
    Full dental landmark detection system.

    Detects anatomical landmarks on each tooth in a dental arch.
    Uses a shared DGCNN backbone with tooth-type-specific output heads.

    Landmark count per tooth type:
    - Molars (FDI 16-18, 26-28, 36-38, 46-48): 5 landmarks
    - Premolars (FDI 14-15, 24-25, 34-35, 44-45): 5 landmarks
    - Canines (FDI 13, 23, 33, 43): 3 landmarks
    - Incisors (FDI 11-12, 21-22, 31-32, 41-42): 3 landmarks

    References:
    - MICCAI TAPoseNet: DGCNN for dental feature extraction
    - PMC11574221: Landmark-based occlusal metrics
    """

    def __init__(
        self,
        k: int = 16,
        points_per_tooth: int = 512,
    ) -> None:
        super().__init__()
        self.k = k
        self.points_per_tooth = points_per_tooth

        # Shared DGCNN backbone for all tooth types
        self.landmark_head = DGCNNLandmarkHead(k=k, max_landmarks=MAX_LANDMARKS)

        # Tooth-type embedding (4 types)
        self.type_embedding = nn.Embedding(4, 16)
        self._type_to_idx = {"incisor": 0, "canine": 1, "premolar": 2, "molar": 3}

    def forward(
        self,
        tooth_point_clouds: List[torch.Tensor],
        fdi_numbers: List[int],
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Detect landmarks on all teeth.

        Args:
            tooth_point_clouds: List of (P, 3) tensors, one per tooth.
            fdi_numbers: List of FDI numbers corresponding to each tooth.

        Returns:
            Dict mapping FDI number → {
                'landmarks': (K, 3) landmark positions,
                'confidence': (K,) confidence scores,
                'names': List[str] landmark names,
            }
        """
        if not tooth_point_clouds:
            return {}

        device = next(self.parameters()).device
        results: Dict[int, Dict[str, torch.Tensor]] = {}

        # Build batch for DGCNN
        data_list = []
        for pc in tooth_point_clouds:
            pc_tensor = pc.to(device) if isinstance(pc, torch.Tensor) else torch.tensor(
                pc, dtype=torch.float32, device=device,
            )
            pc_tensor = self._normalize_points(pc_tensor)
            data_list.append(Data(pos=pc_tensor))

        batch = Batch.from_data_list(data_list)
        x = batch.pos
        batch_idx = batch.batch

        # Run landmark detection
        landmarks, confidence = self.landmark_head(x, batch_idx)

        # Unpack per-tooth results
        for i, fdi in enumerate(fdi_numbers):
            tooth_type = FDI_TOOTH_TYPE.get(fdi, "molar")
            landmark_names = LANDMARKS_BY_TYPE[tooth_type]
            n_landmarks = len(landmark_names)

            # Take only the relevant landmarks for this tooth type
            tooth_landmarks = landmarks[i, :n_landmarks]  # (K, 3)
            tooth_confidence = confidence[i, :n_landmarks]  # (K,)

            results[fdi] = {
                "landmarks": tooth_landmarks,
                "confidence": tooth_confidence,
                "names": landmark_names,
            }

        return results

    def detect_arch_landmarks(
        self,
        tooth_meshes: Dict[int, np.ndarray],
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Convenience method: detect landmarks from numpy point clouds.

        Args:
            tooth_meshes: Dict mapping FDI number → (N, 3) numpy array.

        Returns:
            Dict mapping FDI → {
                'landmarks': (K, 3) numpy positions,
                'confidence': (K,) numpy scores,
                'names': List[str] names,
            }
        """
        fdi_order = sorted(tooth_meshes.keys())
        point_clouds = [
            torch.tensor(tooth_meshes[fdi], dtype=torch.float32)
            for fdi in fdi_order
        ]

        with torch.no_grad():
            results = self.forward(point_clouds, fdi_order)

        # Convert to numpy
        np_results = {}
        for fdi, data in results.items():
            np_results[fdi] = {
                "landmarks": data["landmarks"].cpu().numpy(),
                "confidence": data["confidence"].cpu().numpy(),
                "names": data["names"],
            }
        return np_results

    def extract_midline_points(
        self,
        landmarks: Dict[int, Dict[str, torch.Tensor]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Compute upper and lower midline points from detected landmarks.

        Upper midline: midpoint of FDI 11 and 21 incisal edges.
        Lower midline: midpoint of FDI 31 and 41 incisal edges.

        Returns:
            (upper_midline (3,), lower_midline (3,)) or None if teeth missing.
        """
        upper_midline = None
        lower_midline = None

        # Upper midline: average of 11 and 21 incisal edge centers
        if 11 in landmarks and 21 in landmarks:
            ie_11 = landmarks[11]["landmarks"][0]  # incisal_edge_center
            ie_21 = landmarks[21]["landmarks"][0]
            upper_midline = (ie_11 + ie_21) / 2

        # Lower midline: average of 31 and 41
        if 31 in landmarks and 41 in landmarks:
            ie_31 = landmarks[31]["landmarks"][0]
            ie_41 = landmarks[41]["landmarks"][0]
            lower_midline = (ie_31 + ie_41) / 2

        return upper_midline, lower_midline

    def extract_molar_landmarks(
        self,
        landmarks: Dict[int, Dict[str, torch.Tensor]],
        side: str = "right",
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Extract first molar landmarks for molar relation assessment.

        Args:
            landmarks: Output of forward().
            side: "right" (FDI 16/46) or "left" (FDI 26/36).

        Returns:
            (upper_molar_landmarks (K, 3), lower_molar_landmarks (K, 3))
        """
        if side == "right":
            upper_fdi, lower_fdi = 16, 46
        else:
            upper_fdi, lower_fdi = 26, 36

        upper = landmarks.get(upper_fdi, {}).get("landmarks")
        lower = landmarks.get(lower_fdi, {}).get("landmarks")
        return upper, lower

    def _normalize_points(self, points: torch.Tensor) -> torch.Tensor:
        """Subsample/pad to fixed size and center."""
        n = points.shape[0]
        target = self.points_per_tooth

        if n > target:
            idx = torch.randperm(n, device=points.device)[:target]
            points = points[idx]
        elif n < target:
            pad_idx = torch.randint(0, n, (target - n,), device=points.device)
            points = torch.cat([points, points[pad_idx]], dim=0)

        # Center the point cloud (tooth-local frame)
        centroid = points.mean(dim=0, keepdim=True)
        points = points - centroid
        return points

    def load_weights(self, path: Path) -> None:
        """Load pretrained weights with graceful fallback."""
        if path.exists():
            state = torch.load(path, map_location="cpu", weights_only=True)
            self.load_state_dict(state, strict=False)
            logger.info("Loaded DentalLandmarkDetector weights from %s", path)
        else:
            logger.info(
                "No pretrained weights at %s — using random initialization", path
            )

    def save_weights(self, path: Path) -> None:
        """Save current weights."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info("Saved DentalLandmarkDetector weights to %s", path)
