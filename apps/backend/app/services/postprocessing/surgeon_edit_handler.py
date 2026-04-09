"""
Handle interactive surgeon edits to ML-predicted fragment positions.

When a clinician adjusts a fragment position in the 3D viewer, this module:
1. Captures the edit as a delta SE(3) transform
2. Re-validates occlusion with the edited position
3. Regenerates STL output with the edit applied
4. Logs the edit for audit trail and future model fine-tuning

Undo/redo
---------
A bounded undo/redo stack tracks every edit per fragment.  The stack is
capped at 50 entries per fragment to prevent unbounded memory growth.
Undo pops the last edit and reverts the mesh; redo re-applies it.

Audit trail
-----------
Every edit is logged to an ``AuditRecord`` that captures the surgeon ID,
timestamp, before/after transforms, and rationale.  This data feeds the
offline fine-tuning pipeline that learns from surgeon corrections.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as ScipyRotation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_UNDO_STACK_SIZE: int = 50


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EditDelta:
    """A single edit expressed as a delta SE(3) transform."""

    fragment_id: str
    delta_rotation: np.ndarray  # 3x3
    delta_translation_mm: np.ndarray  # (3,)
    rotation_angle_deg: float
    translation_magnitude_mm: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    rationale: Optional[str] = None


@dataclass
class AuditRecord:
    """Immutable audit record for a single edit."""

    record_id: str
    surgeon_id: str
    fragment_id: str
    timestamp: datetime
    transform_before: np.ndarray  # 4x4
    transform_after: np.ndarray  # 4x4
    delta_rotation_deg: float
    delta_translation_mm: float
    rationale: Optional[str]
    edit_tool: str


@dataclass
class OcclusionValidationResult:
    """Result of validating occlusion after an edit."""

    is_valid: bool
    max_gap_mm: float
    max_interference_mm: float
    contact_points: int
    warnings: List[str]


@dataclass
class EditResult:
    """Result of applying a surgeon edit."""

    fragment_id: str
    updated_mesh: trimesh.Trimesh
    updated_transform: np.ndarray
    delta: EditDelta
    occlusion_valid: bool
    audit_record: AuditRecord
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# SurgeonEditHandler
# ---------------------------------------------------------------------------

class SurgeonEditHandler:
    """
    Handles interactive surgeon edits to model predictions.

    Manages an undo/redo stack, validates occlusion constraints after
    each edit, and maintains a complete audit trail.

    Note: This class maintains state (undo/redo stacks and audit log)
    and should be instantiated per-session rather than shared globally.
    """

    def __init__(
        self,
        surgeon_id: str,
        case_id: str,
        plan_id: str,
        edit_tool: str = "web_viewer",
        max_undo_size: int = MAX_UNDO_STACK_SIZE,
    ) -> None:
        """
        Initialise the edit handler for a session.

        Args:
            surgeon_id: Identifier for the editing surgeon.
            case_id: Surgical case ID.
            plan_id: Plan ID being edited.
            edit_tool: Interface used for editing.
            max_undo_size: Maximum undo stack depth per fragment.
        """
        self._surgeon_id = surgeon_id
        self._case_id = case_id
        self._plan_id = plan_id
        self._edit_tool = edit_tool
        self._max_undo = max_undo_size

        # Per-fragment undo/redo stacks: fragment_id -> list of (mesh_snapshot, transform_snapshot)
        self._undo_stacks: Dict[str, List[Tuple[trimesh.Trimesh, np.ndarray]]] = defaultdict(list)
        self._redo_stacks: Dict[str, List[Tuple[trimesh.Trimesh, np.ndarray]]] = defaultdict(list)

        # Audit trail
        self._audit_log: List[AuditRecord] = []
        self._edit_counter: int = 0

    # ------------------------------------------------------------------
    # Public: Apply edit
    # ------------------------------------------------------------------

    def apply_edit(
        self,
        fragment_id: str,
        current_mesh: trimesh.Trimesh,
        current_transform: np.ndarray,
        new_transform: np.ndarray,
        opposing_mesh: Optional[trimesh.Trimesh] = None,
        rationale: Optional[str] = None,
    ) -> EditResult:
        """
        Apply a surgeon edit to a fragment.

        Captures the delta transform, pushes undo state, applies the new
        position, validates occlusion, and logs the audit record.

        Args:
            fragment_id: Fragment being edited.
            current_mesh: Current mesh (will be transformed to new position).
            current_transform: Current 4x4 transform of the fragment.
            new_transform: New 4x4 transform (surgeon's desired position).
            opposing_mesh: Opposing arch/fragment for occlusion validation.
            rationale: Surgeon's rationale for the edit.

        Returns:
            EditResult with updated mesh and validation status.
        """
        t0 = time.monotonic()
        current_transform = np.asarray(current_transform, dtype=np.float64)
        new_transform = np.asarray(new_transform, dtype=np.float64)

        # Compute delta transform: T_new = T_delta @ T_current
        # → T_delta = T_new @ T_current^{-1}
        T_current_inv = _invert_se3(current_transform)
        T_delta = new_transform @ T_current_inv

        # Extract delta rotation and translation
        delta_R = T_delta[:3, :3]
        delta_t = T_delta[:3, 3]
        rot_angle = _rotation_angle_deg(delta_R)
        trans_mag = float(np.linalg.norm(delta_t))

        delta = EditDelta(
            fragment_id=fragment_id,
            delta_rotation=delta_R,
            delta_translation_mm=delta_t,
            rotation_angle_deg=rot_angle,
            translation_magnitude_mm=trans_mag,
            rationale=rationale,
        )

        # Push current state to undo stack
        self._push_undo(fragment_id, current_mesh, current_transform)

        # Clear redo stack (new edit invalidates redo history)
        self._redo_stacks[fragment_id].clear()

        # Apply the new transform
        relative_transform = new_transform @ T_current_inv
        updated_mesh = current_mesh.copy()
        updated_mesh.apply_transform(relative_transform)

        # Validate occlusion
        occlusion_result = self._validate_occlusion(updated_mesh, opposing_mesh)

        # Create audit record
        self._edit_counter += 1
        record = AuditRecord(
            record_id=f"edit-{self._case_id}-{self._edit_counter:04d}",
            surgeon_id=self._surgeon_id,
            fragment_id=fragment_id,
            timestamp=datetime.now(tz=timezone.utc),
            transform_before=current_transform.copy(),
            transform_after=new_transform.copy(),
            delta_rotation_deg=rot_angle,
            delta_translation_mm=trans_mag,
            rationale=rationale,
            edit_tool=self._edit_tool,
        )
        self._audit_log.append(record)

        elapsed = time.monotonic() - t0
        logger.info(
            "Edit applied: fragment '%s', rot=%.2f°, trans=%.2fmm, occlusion=%s (%.3fs)",
            fragment_id, rot_angle, trans_mag,
            "valid" if occlusion_result.is_valid else "INVALID",
            elapsed,
        )

        return EditResult(
            fragment_id=fragment_id,
            updated_mesh=updated_mesh,
            updated_transform=new_transform,
            delta=delta,
            occlusion_valid=occlusion_result.is_valid,
            audit_record=record,
            elapsed_seconds=elapsed,
        )

    # ------------------------------------------------------------------
    # Public: Undo / Redo
    # ------------------------------------------------------------------

    def undo(
        self,
        fragment_id: str,
        current_mesh: trimesh.Trimesh,
        current_transform: np.ndarray,
    ) -> Optional[Tuple[trimesh.Trimesh, np.ndarray]]:
        """
        Undo the last edit for a fragment.

        Pushes the current state to redo before restoring the previous state.

        Args:
            fragment_id: Fragment to undo.
            current_mesh: Current mesh state.
            current_transform: Current transform.

        Returns:
            (restored_mesh, restored_transform) or None if nothing to undo.
        """
        stack = self._undo_stacks.get(fragment_id, [])
        if not stack:
            logger.info("Nothing to undo for fragment '%s'", fragment_id)
            return None

        # Push current to redo
        self._push_redo(fragment_id, current_mesh, current_transform)

        # Pop previous state
        prev_mesh, prev_transform = stack.pop()

        # Log undo as an audit record
        self._edit_counter += 1
        record = AuditRecord(
            record_id=f"undo-{self._case_id}-{self._edit_counter:04d}",
            surgeon_id=self._surgeon_id,
            fragment_id=fragment_id,
            timestamp=datetime.now(tz=timezone.utc),
            transform_before=np.asarray(current_transform),
            transform_after=prev_transform,
            delta_rotation_deg=_rotation_angle_deg(
                (prev_transform @ _invert_se3(np.asarray(current_transform)))[:3, :3]
            ),
            delta_translation_mm=float(np.linalg.norm(
                prev_transform[:3, 3] - np.asarray(current_transform)[:3, 3]
            )),
            rationale="undo",
            edit_tool=self._edit_tool,
        )
        self._audit_log.append(record)

        logger.info("Undo: fragment '%s' restored to previous state", fragment_id)
        return prev_mesh, prev_transform

    def redo(
        self,
        fragment_id: str,
        current_mesh: trimesh.Trimesh,
        current_transform: np.ndarray,
    ) -> Optional[Tuple[trimesh.Trimesh, np.ndarray]]:
        """
        Redo the last undone edit for a fragment.

        Args:
            fragment_id: Fragment to redo.
            current_mesh: Current mesh state.
            current_transform: Current transform.

        Returns:
            (restored_mesh, restored_transform) or None if nothing to redo.
        """
        stack = self._redo_stacks.get(fragment_id, [])
        if not stack:
            logger.info("Nothing to redo for fragment '%s'", fragment_id)
            return None

        # Push current to undo
        self._push_undo(fragment_id, current_mesh, current_transform)

        # Pop redo state
        redo_mesh, redo_transform = stack.pop()

        # Log redo
        self._edit_counter += 1
        record = AuditRecord(
            record_id=f"redo-{self._case_id}-{self._edit_counter:04d}",
            surgeon_id=self._surgeon_id,
            fragment_id=fragment_id,
            timestamp=datetime.now(tz=timezone.utc),
            transform_before=np.asarray(current_transform),
            transform_after=redo_transform,
            delta_rotation_deg=_rotation_angle_deg(
                (redo_transform @ _invert_se3(np.asarray(current_transform)))[:3, :3]
            ),
            delta_translation_mm=float(np.linalg.norm(
                redo_transform[:3, 3] - np.asarray(current_transform)[:3, 3]
            )),
            rationale="redo",
            edit_tool=self._edit_tool,
        )
        self._audit_log.append(record)

        logger.info("Redo: fragment '%s' restored to redo state", fragment_id)
        return redo_mesh, redo_transform

    def can_undo(self, fragment_id: str) -> bool:
        """Check if undo is available for a fragment."""
        return len(self._undo_stacks.get(fragment_id, [])) > 0

    def can_redo(self, fragment_id: str) -> bool:
        """Check if redo is available for a fragment."""
        return len(self._redo_stacks.get(fragment_id, [])) > 0

    def undo_depth(self, fragment_id: str) -> int:
        """Return the number of available undo steps for a fragment."""
        return len(self._undo_stacks.get(fragment_id, []))

    def redo_depth(self, fragment_id: str) -> int:
        """Return the number of available redo steps for a fragment."""
        return len(self._redo_stacks.get(fragment_id, []))

    # ------------------------------------------------------------------
    # Public: Reset to ML prediction
    # ------------------------------------------------------------------

    def reset_to_ml(
        self,
        fragment_id: str,
        current_mesh: trimesh.Trimesh,
        current_transform: np.ndarray,
        original_mesh: trimesh.Trimesh,
        original_transform: np.ndarray,
    ) -> EditResult:
        """
        Reset a fragment to its original ML-predicted position.

        Saves the current state for undo, then restores the original.

        Args:
            fragment_id: Fragment to reset.
            current_mesh: Current mesh.
            current_transform: Current transform.
            original_mesh: Original ML-predicted mesh.
            original_transform: Original ML-predicted transform.

        Returns:
            EditResult with the restored original position.
        """
        return self.apply_edit(
            fragment_id=fragment_id,
            current_mesh=current_mesh,
            current_transform=current_transform,
            new_transform=original_transform,
            rationale="reset_to_ml",
        )

    # ------------------------------------------------------------------
    # Public: Audit trail
    # ------------------------------------------------------------------

    def get_audit_log(self) -> List[AuditRecord]:
        """
        Return the complete audit trail for this session.

        Returns:
            List of AuditRecord in chronological order.
        """
        return list(self._audit_log)

    def get_fragment_audit_log(self, fragment_id: str) -> List[AuditRecord]:
        """
        Return audit records for a specific fragment.

        Args:
            fragment_id: Fragment to query.

        Returns:
            List of AuditRecord for the fragment.
        """
        return [r for r in self._audit_log if r.fragment_id == fragment_id]

    def get_session_summary(self) -> Dict[str, Any]:
        """
        Compute summary statistics for the editing session.

        Returns:
            Dict with session stats (n_edits, fragments edited, magnitudes).
        """
        if not self._audit_log:
            return {
                "case_id": self._case_id,
                "plan_id": self._plan_id,
                "surgeon_id": self._surgeon_id,
                "n_edits": 0,
                "n_fragments_edited": 0,
            }

        edited_fragments = {r.fragment_id for r in self._audit_log}
        rot_deltas = [r.delta_rotation_deg for r in self._audit_log]
        trans_deltas = [r.delta_translation_mm for r in self._audit_log]

        return {
            "case_id": self._case_id,
            "plan_id": self._plan_id,
            "surgeon_id": self._surgeon_id,
            "n_edits": len(self._audit_log),
            "n_fragments_edited": len(edited_fragments),
            "edited_fragment_ids": sorted(edited_fragments),
            "mean_rotation_delta_deg": float(np.mean(rot_deltas)),
            "max_rotation_delta_deg": float(np.max(rot_deltas)),
            "mean_translation_delta_mm": float(np.mean(trans_deltas)),
            "max_translation_delta_mm": float(np.max(trans_deltas)),
            "session_start": self._audit_log[0].timestamp.isoformat(),
            "session_end": self._audit_log[-1].timestamp.isoformat(),
        }

    def export_audit_log_json(self) -> List[Dict[str, Any]]:
        """
        Export the audit log as JSON-serialisable dicts.

        Returns:
            List of dicts representing each audit record.
        """
        records = []
        for r in self._audit_log:
            records.append({
                "record_id": r.record_id,
                "surgeon_id": r.surgeon_id,
                "fragment_id": r.fragment_id,
                "timestamp": r.timestamp.isoformat(),
                "transform_before": r.transform_before.tolist(),
                "transform_after": r.transform_after.tolist(),
                "delta_rotation_deg": round(r.delta_rotation_deg, 4),
                "delta_translation_mm": round(r.delta_translation_mm, 4),
                "rationale": r.rationale,
                "edit_tool": r.edit_tool,
            })
        return records

    # ------------------------------------------------------------------
    # Internal: Stack management
    # ------------------------------------------------------------------

    def _push_undo(
        self,
        fragment_id: str,
        mesh: trimesh.Trimesh,
        transform: np.ndarray,
    ) -> None:
        """
        Push current state onto the undo stack.

        Trims the stack if it exceeds max size.

        Args:
            fragment_id: Fragment ID.
            mesh: Current mesh (deep copied).
            transform: Current 4x4 transform (copied).
        """
        stack = self._undo_stacks[fragment_id]
        stack.append((mesh.copy(), np.array(transform, dtype=np.float64)))
        while len(stack) > self._max_undo:
            stack.pop(0)

    def _push_redo(
        self,
        fragment_id: str,
        mesh: trimesh.Trimesh,
        transform: np.ndarray,
    ) -> None:
        """
        Push current state onto the redo stack.

        Args:
            fragment_id: Fragment ID.
            mesh: Current mesh (deep copied).
            transform: Current 4x4 transform (copied).
        """
        stack = self._redo_stacks[fragment_id]
        stack.append((mesh.copy(), np.array(transform, dtype=np.float64)))
        while len(stack) > self._max_undo:
            stack.pop(0)

    # ------------------------------------------------------------------
    # Internal: Occlusion validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_occlusion(
        edited_mesh: trimesh.Trimesh,
        opposing_mesh: Optional[trimesh.Trimesh],
    ) -> OcclusionValidationResult:
        """
        Validate occlusion between an edited fragment and the opposing structure.

        If no opposing mesh is provided, validation passes trivially.

        Args:
            edited_mesh: The edited fragment mesh.
            opposing_mesh: Opposing arch or bone mesh for contact checking.

        Returns:
            OcclusionValidationResult.
        """
        if opposing_mesh is None:
            return OcclusionValidationResult(
                is_valid=True,
                max_gap_mm=0.0,
                max_interference_mm=0.0,
                contact_points=0,
                warnings=["No opposing mesh provided; occlusion not validated"],
            )

        # Compute closest point distances
        closest_pts, distances, _ = trimesh.proximity.closest_point(
            opposing_mesh, edited_mesh.vertices
        )

        # Contact analysis
        contact_threshold_mm = 0.5
        interference_threshold_mm = 0.1
        contact_mask = distances < contact_threshold_mm
        n_contact = int(np.sum(contact_mask))

        max_gap = float(np.max(distances)) if len(distances) > 0 else 0.0

        # Estimate interference by checking if points are inside opposing mesh
        max_interference = 0.0
        warnings: List[str] = []

        if opposing_mesh.is_watertight:
            inside = opposing_mesh.contains(edited_mesh.vertices)
            if np.any(inside):
                inside_pts = edited_mesh.vertices[inside]
                _, inside_dists, _ = trimesh.proximity.closest_point(
                    opposing_mesh, inside_pts
                )
                max_interference = float(np.max(inside_dists))
                n_interfering = int(np.sum(inside))
                warnings.append(
                    f"{n_interfering} vertices penetrate opposing mesh "
                    f"(max depth {max_interference:.2f}mm)"
                )

        is_valid = max_interference < 2.0 and n_contact > 0

        if n_contact == 0:
            warnings.append("No occlusal contact points detected after edit")
            is_valid = False

        return OcclusionValidationResult(
            is_valid=is_valid,
            max_gap_mm=max_gap,
            max_interference_mm=max_interference,
            contact_points=n_contact,
            warnings=warnings,
        )


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------

def _invert_se3(T: np.ndarray) -> np.ndarray:
    """
    Invert a 4x4 SE(3) transform.

    Args:
        T: 4x4 homogeneous transform.

    Returns:
        4x4 inverted transform.
    """
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv


def _rotation_angle_deg(R: np.ndarray) -> float:
    """
    Compute rotation angle in degrees from a 3x3 rotation matrix.

    Args:
        R: 3x3 rotation matrix.

    Returns:
        Angle in degrees.
    """
    trace = float(np.trace(R))
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))
