"""Capability registry helpers and provenance builders."""

from __future__ import annotations

from datetime import datetime, timezone
from importlib.util import find_spec
from pathlib import Path
from typing import List, Optional

from app.core.config import get_settings
from app.schemas.capabilities import CapabilitySnapshotResponse, CapabilityStatus, ProvenanceInfo


def _path_ready(path: Optional[Path]) -> bool:
    return path is not None and path.exists()


def _package_ready(module_name: str) -> bool:
    return find_spec(module_name) is not None


def build_provenance(
    *,
    algorithm_used: str,
    validation_tier: str,
    beta_status: str,
    warnings: Optional[List[str]] = None,
    fallback_reason: Optional[str] = None,
    model_version: Optional[str] = None,
) -> dict:
    """Create a canonical provenance payload."""

    return ProvenanceInfo(
        algorithm_used=algorithm_used,
        validation_tier=validation_tier,
        beta_status=beta_status,
        warnings=list(warnings or []),
        fallback_reason=fallback_reason,
        model_version=model_version,
    ).model_dump(exclude_none=True)


def get_capability_snapshot() -> CapabilitySnapshotResponse:
    """Describe what the current deployment can do truthfully."""

    settings = get_settings()

    cmf_ready = _path_ready(settings.model_registry.cmf_segmentation_model_path)
    dental_ready = _path_ready(settings.model_registry.dental_segmentation_model_path)
    learned_reduction_ready = _path_ready(settings.model_registry.fracture_reduction_model_path)
    learned_landmarks_ready = _path_ready(settings.model_registry.deep_registration_model_path)

    capabilities = [
        CapabilityStatus(
            name="dicom_ingestion",
            category="ingestion",
            status="available",
            baseline_available=True,
            validation_tier="deterministic_baseline",
            beta_status="not_beta",
        ),
        CapabilityStatus(
            name="segmentation_baseline",
            category="segmentation",
            status="available" if _package_ready("totalsegmentator") else "degraded",
            baseline_available=True,
            validation_tier="deterministic_baseline",
            beta_status="not_beta",
            warnings=[] if _package_ready("totalsegmentator") else [
                "TotalSegmentator is not installed in this environment.",
            ],
        ),
        CapabilityStatus(
            name="segmentation_learned_cmf",
            category="segmentation",
            status="available" if cmf_ready else "unavailable",
            learned_available=cmf_ready,
            artifact_required=True,
            artifact_ready=cmf_ready,
            validation_tier="learned_beta",
            beta_status="beta_available" if cmf_ready else "beta_unavailable",
            warnings=[] if cmf_ready else ["Custom CMF model weights are not available."],
        ),
        CapabilityStatus(
            name="dental_segmentation",
            category="segmentation",
            status="available" if dental_ready else "unavailable",
            learned_available=dental_ready,
            artifact_required=True,
            artifact_ready=dental_ready,
            validation_tier="learned_beta",
            beta_status="beta_available" if dental_ready else "beta_unavailable",
            warnings=[] if dental_ready else [
                "Dental segmentation artifacts are not available; this capability is disabled.",
            ],
        ),
        CapabilityStatus(
            name="reduction_baseline_icp",
            category="planning",
            status="available",
            baseline_available=True,
            validation_tier="deterministic_baseline",
            beta_status="not_beta",
        ),
        CapabilityStatus(
            name="reduction_learned_v1",
            category="planning",
            status="available" if learned_reduction_ready else "unavailable",
            learned_available=learned_reduction_ready,
            artifact_required=True,
            artifact_ready=learned_reduction_ready,
            validation_tier="learned_beta",
            beta_status="beta_available" if learned_reduction_ready else "beta_unavailable",
            warnings=[] if learned_reduction_ready else [
                "Learned fracture-reduction artifacts are not available.",
            ],
        ),
        CapabilityStatus(
            name="landmarks_heuristic",
            category="landmarks",
            status="available",
            baseline_available=True,
            validation_tier="deterministic_baseline",
            beta_status="not_beta",
        ),
        CapabilityStatus(
            name="landmarks_learned",
            category="landmarks",
            status="available" if learned_landmarks_ready else "unavailable",
            learned_available=learned_landmarks_ready,
            artifact_required=True,
            artifact_ready=learned_landmarks_ready,
            validation_tier="learned_beta",
            beta_status="beta_available" if learned_landmarks_ready else "beta_unavailable",
            warnings=[] if learned_landmarks_ready else [
                "Learned landmark artifacts are not available.",
            ],
        ),
    ]

    return CapabilitySnapshotResponse(
        generated_at=datetime.now(timezone.utc),
        capabilities=capabilities,
    )
