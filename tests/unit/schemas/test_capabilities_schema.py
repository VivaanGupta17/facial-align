"""Schema coverage for capability registry and provenance metadata."""

from __future__ import annotations

from datetime import datetime, timezone

from app.schemas.capabilities import CapabilitySnapshotResponse, CapabilityStatus, ProvenanceInfo
from app.schemas.plan import MetricOverrideRequest


class TestProvenanceInfo:
    def test_provenance_info_accepts_fallback_metadata(self):
        provenance = ProvenanceInfo(
            algorithm_used="baseline_icp",
            validation_tier="deterministic_baseline",
            beta_status="fallback",
            warnings=["Learned model unavailable; using baseline."],
            fallback_reason="Artifact missing",
            model_version="baseline-2026.05",
        )

        assert provenance.algorithm_used == "baseline_icp"
        assert provenance.fallback_reason == "Artifact missing"
        assert provenance.model_version == "baseline-2026.05"


class TestCapabilitySnapshotResponse:
    def test_snapshot_contains_capability_entries(self):
        snapshot = CapabilitySnapshotResponse(
            generated_at=datetime.now(timezone.utc),
            capabilities=[
                CapabilityStatus(
                    name="segmentation_baseline",
                    category="segmentation",
                    status="available",
                    baseline_available=True,
                    validation_tier="deterministic_baseline",
                    beta_status="not_beta",
                )
            ],
        )

        assert len(snapshot.capabilities) == 1
        assert snapshot.capabilities[0].name == "segmentation_baseline"
        assert snapshot.capabilities[0].baseline_available is True


class TestMetricOverrideRequest:
    def test_metric_override_request_accepts_molar_class(self):
        request = MetricOverrideRequest(
            metric_name="molar_class",
            target_value=0,
            notes="Switch to Class I target.",
        )

        assert request.metric_name == "molar_class"
