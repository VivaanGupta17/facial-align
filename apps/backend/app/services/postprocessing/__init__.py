"""
Post-processing pipeline for ML-predicted surgical transforms.

Modules
-------
- transform_applicator : Apply predicted SE(3) transforms to bone/tooth meshes
- mesh_cleanup         : Clinical-grade mesh cleaning (hole fill, smooth, remesh)
- collision_resolver   : Resolve interpenetrations between repositioned fragments
- confidence_gate      : Route predictions by model confidence
- surgeon_edit_handler : Handle interactive surgeon edits with undo/redo
"""

from app.services.postprocessing.collision_resolver import CollisionResolver
from app.services.postprocessing.confidence_gate import (
    ClinicalDecision,
    ConfidenceGate,
    DecisionType,
)
from app.services.postprocessing.mesh_cleanup import MeshCleanup
from app.services.postprocessing.surgeon_edit_handler import SurgeonEditHandler
from app.services.postprocessing.transform_applicator import TransformApplicator

__all__ = [
    "TransformApplicator",
    "MeshCleanup",
    "CollisionResolver",
    "ConfidenceGate",
    "ClinicalDecision",
    "DecisionType",
    "SurgeonEditHandler",
]
