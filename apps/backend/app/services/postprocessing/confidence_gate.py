"""
Route ML predictions based on model confidence.

The confidence gate sits between the ML prediction and the STL export
pipeline.  It evaluates per-fragment and overall prediction confidence
and routes to one of four outcomes:

- ACCEPT   — confidence sufficient, proceed to STL export
- REVIEW   — display prediction with warnings, require clinician sign-off
- FALLBACK — run optimisation-based pipeline as alternative
- REJECT   — do not produce output, display error

Clinical safety
---------------
In a surgical planning context, false confidence is more dangerous than low
confidence.  The gate is deliberately conservative: borderline cases are
routed to REVIEW rather than ACCEPT.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class DecisionType(str, Enum):
    """Clinical routing decision."""

    ACCEPT = "accept"
    REVIEW = "review"
    FALLBACK = "fallback"
    REJECT = "reject"


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_ROTATION_UNCERTAINTY_THRESHOLD_DEG: float = 5.0
DEFAULT_TRANSLATION_UNCERTAINTY_THRESHOLD_MM: float = 1.5
DEFAULT_OVERALL_CONFIDENCE_THRESHOLD: float = 0.7
DEFAULT_FRAGMENT_CONFIDENCE_BLOCK_THRESHOLD: float = 0.5
DEFAULT_ACCEPT_CONFIDENCE_THRESHOLD: float = 0.85


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FragmentConfidence:
    """Confidence metrics for a single fragment prediction."""

    fragment_id: str
    overall_confidence: float
    rotation_uncertainty_deg: float
    translation_uncertainty_mm: float
    is_surgeon_edited: bool = False


@dataclass
class PredictionConfidence:
    """Aggregated confidence for the full prediction."""

    case_id: str
    plan_id: str
    model_version: str
    fragments: List[FragmentConfidence]
    overall_confidence: float
    mean_rotation_uncertainty_deg: float
    mean_translation_uncertainty_mm: float


@dataclass
class ClinicalDecision:
    """Routing decision from the confidence gate."""

    decision: DecisionType
    case_id: str
    plan_id: str
    overall_confidence: float
    reasons: List[str]
    fragment_decisions: Dict[str, DecisionType]
    flagged_fragments: List[str]
    blocked_fragments: List[str]
    recommendations: List[str]
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# ConfidenceGate
# ---------------------------------------------------------------------------

class ConfidenceGate:
    """
    Routes predictions based on model confidence.

    Evaluates rotation uncertainty, translation uncertainty, per-fragment
    confidence, and overall confidence against configurable thresholds.

    Thread-safe: no mutable instance state beyond configuration.
    """

    def __init__(
        self,
        rotation_uncertainty_threshold_deg: float = DEFAULT_ROTATION_UNCERTAINTY_THRESHOLD_DEG,
        translation_uncertainty_threshold_mm: float = DEFAULT_TRANSLATION_UNCERTAINTY_THRESHOLD_MM,
        overall_confidence_threshold: float = DEFAULT_OVERALL_CONFIDENCE_THRESHOLD,
        fragment_block_threshold: float = DEFAULT_FRAGMENT_CONFIDENCE_BLOCK_THRESHOLD,
        accept_threshold: float = DEFAULT_ACCEPT_CONFIDENCE_THRESHOLD,
    ) -> None:
        """
        Initialise the confidence gate.

        Args:
            rotation_uncertainty_threshold_deg: Flag if rotation uncertainty exceeds this.
            translation_uncertainty_threshold_mm: Flag if translation uncertainty exceeds this.
            overall_confidence_threshold: Route to fallback if overall confidence below this.
            fragment_block_threshold: Block fragment if confidence below this.
            accept_threshold: Accept without review if all confidences above this.
        """
        self._rot_threshold = rotation_uncertainty_threshold_deg
        self._trans_threshold = translation_uncertainty_threshold_mm
        self._overall_threshold = overall_confidence_threshold
        self._block_threshold = fragment_block_threshold
        self._accept_threshold = accept_threshold

    # ------------------------------------------------------------------
    # Public: Gate evaluation
    # ------------------------------------------------------------------

    def evaluate(self, prediction: PredictionConfidence) -> ClinicalDecision:
        """
        Evaluate a prediction and produce a clinical routing decision.

        Decision logic:
        1. Any fragment confidence < block_threshold → REJECT that fragment
        2. Overall confidence < overall_threshold → FALLBACK
        3. Any uncertainty above thresholds → REVIEW
        4. All good → ACCEPT

        Args:
            prediction: Aggregated prediction confidence.

        Returns:
            ClinicalDecision with routing and per-fragment decisions.
        """
        t0 = time.monotonic()

        reasons: List[str] = []
        recommendations: List[str] = []
        flagged: List[str] = []
        blocked: List[str] = []
        fragment_decisions: Dict[str, DecisionType] = {}

        # Evaluate each fragment
        for frag in prediction.fragments:
            frag_decision = self._evaluate_fragment(frag)
            fragment_decisions[frag.fragment_id] = frag_decision

            if frag_decision == DecisionType.REJECT:
                blocked.append(frag.fragment_id)
                reasons.append(
                    f"Fragment '{frag.fragment_id}' confidence {frag.overall_confidence:.2f} "
                    f"below block threshold {self._block_threshold}"
                )
                recommendations.append(
                    f"Fragment '{frag.fragment_id}' requires manual positioning — "
                    "model prediction not reliable enough"
                )
            elif frag_decision == DecisionType.REVIEW:
                flagged.append(frag.fragment_id)
                frag_reasons = self._get_fragment_flag_reasons(frag)
                reasons.extend(frag_reasons)

        # Determine overall decision
        overall_decision = self._determine_overall_decision(
            prediction, fragment_decisions, blocked, flagged
        )

        # Add overall-level reasons
        if prediction.overall_confidence < self._overall_threshold:
            reasons.append(
                f"Overall confidence {prediction.overall_confidence:.2f} "
                f"below threshold {self._overall_threshold}"
            )
            recommendations.append(
                "Consider running optimisation-based pipeline as alternative"
            )

        if overall_decision == DecisionType.ACCEPT:
            recommendations.append("Prediction meets all confidence thresholds; safe to proceed")
        elif overall_decision == DecisionType.REVIEW:
            recommendations.append(
                "Display prediction with highlighted warnings for clinician review"
            )
        elif overall_decision == DecisionType.FALLBACK:
            recommendations.append(
                "Route to optimisation-based pipeline; ML prediction unreliable"
            )

        elapsed = time.monotonic() - t0

        decision = ClinicalDecision(
            decision=overall_decision,
            case_id=prediction.case_id,
            plan_id=prediction.plan_id,
            overall_confidence=prediction.overall_confidence,
            reasons=reasons,
            fragment_decisions=fragment_decisions,
            flagged_fragments=flagged,
            blocked_fragments=blocked,
            recommendations=recommendations,
            elapsed_seconds=elapsed,
        )

        logger.info(
            "Confidence gate: %s (confidence=%.2f, %d flagged, %d blocked)",
            overall_decision.value, prediction.overall_confidence,
            len(flagged), len(blocked),
        )
        return decision

    # ------------------------------------------------------------------
    # Public: Convenience builders
    # ------------------------------------------------------------------

    @staticmethod
    def build_prediction_confidence(
        case_id: str,
        plan_id: str,
        model_version: str,
        fragment_data: List[Dict[str, Any]],
    ) -> PredictionConfidence:
        """
        Build a PredictionConfidence from raw fragment data.

        Args:
            case_id: Case identifier.
            plan_id: Plan identifier.
            model_version: Model version string.
            fragment_data: List of dicts with keys: fragment_id, confidence,
                           rotation_uncertainty_deg, translation_uncertainty_mm,
                           is_surgeon_edited (optional).

        Returns:
            PredictionConfidence instance.
        """
        fragments = []
        for fd in fragment_data:
            fragments.append(FragmentConfidence(
                fragment_id=fd["fragment_id"],
                overall_confidence=fd["confidence"],
                rotation_uncertainty_deg=fd.get("rotation_uncertainty_deg", 0.0),
                translation_uncertainty_mm=fd.get("translation_uncertainty_mm", 0.0),
                is_surgeon_edited=fd.get("is_surgeon_edited", False),
            ))

        confidences = [f.overall_confidence for f in fragments]
        rot_uncertainties = [f.rotation_uncertainty_deg for f in fragments]
        trans_uncertainties = [f.translation_uncertainty_mm for f in fragments]

        overall = float(np.mean(confidences)) if confidences else 0.0
        mean_rot = float(np.mean(rot_uncertainties)) if rot_uncertainties else 0.0
        mean_trans = float(np.mean(trans_uncertainties)) if trans_uncertainties else 0.0

        return PredictionConfidence(
            case_id=case_id,
            plan_id=plan_id,
            model_version=model_version,
            fragments=fragments,
            overall_confidence=overall,
            mean_rotation_uncertainty_deg=mean_rot,
            mean_translation_uncertainty_mm=mean_trans,
        )

    # ------------------------------------------------------------------
    # Public: Threshold queries
    # ------------------------------------------------------------------

    def is_acceptable(self, confidence: float) -> bool:
        """
        Check if a single confidence value meets the accept threshold.

        Args:
            confidence: Confidence score (0-1).

        Returns:
            True if above accept threshold.
        """
        return confidence >= self._accept_threshold

    def requires_review(self, confidence: float) -> bool:
        """
        Check if a confidence value falls in the review range.

        Args:
            confidence: Confidence score (0-1).

        Returns:
            True if in the review zone (between block and accept).
        """
        return self._block_threshold <= confidence < self._accept_threshold

    def should_block(self, confidence: float) -> bool:
        """
        Check if a confidence value is too low and should be blocked.

        Args:
            confidence: Confidence score (0-1).

        Returns:
            True if below block threshold.
        """
        return confidence < self._block_threshold

    # ------------------------------------------------------------------
    # Internal: Fragment evaluation
    # ------------------------------------------------------------------

    def _evaluate_fragment(self, frag: FragmentConfidence) -> DecisionType:
        """
        Evaluate a single fragment's confidence and return its decision.

        Surgeon-edited fragments are always ACCEPT (the clinician has
        already approved the position).

        Args:
            frag: Fragment confidence data.

        Returns:
            DecisionType for this fragment.
        """
        if frag.is_surgeon_edited:
            return DecisionType.ACCEPT

        if frag.overall_confidence < self._block_threshold:
            return DecisionType.REJECT

        needs_review = False
        if frag.rotation_uncertainty_deg > self._rot_threshold:
            needs_review = True
        if frag.translation_uncertainty_mm > self._trans_threshold:
            needs_review = True
        if frag.overall_confidence < self._accept_threshold:
            needs_review = True

        if needs_review:
            return DecisionType.REVIEW

        return DecisionType.ACCEPT

    def _get_fragment_flag_reasons(self, frag: FragmentConfidence) -> List[str]:
        """
        Get human-readable reasons for flagging a fragment.

        Args:
            frag: Fragment confidence data.

        Returns:
            List of reason strings.
        """
        reasons: List[str] = []
        if frag.rotation_uncertainty_deg > self._rot_threshold:
            reasons.append(
                f"Fragment '{frag.fragment_id}' rotation uncertainty "
                f"{frag.rotation_uncertainty_deg:.1f}° > {self._rot_threshold}°"
            )
        if frag.translation_uncertainty_mm > self._trans_threshold:
            reasons.append(
                f"Fragment '{frag.fragment_id}' translation uncertainty "
                f"{frag.translation_uncertainty_mm:.1f}mm > {self._trans_threshold}mm"
            )
        if frag.overall_confidence < self._accept_threshold:
            reasons.append(
                f"Fragment '{frag.fragment_id}' confidence "
                f"{frag.overall_confidence:.2f} < {self._accept_threshold}"
            )
        return reasons

    def _determine_overall_decision(
        self,
        prediction: PredictionConfidence,
        fragment_decisions: Dict[str, DecisionType],
        blocked: List[str],
        flagged: List[str],
    ) -> DecisionType:
        """
        Determine the overall routing decision from fragment-level results.

        Priority: REJECT > FALLBACK > REVIEW > ACCEPT.

        Args:
            prediction: Overall prediction data.
            fragment_decisions: Per-fragment decisions.
            blocked: List of blocked fragment IDs.
            flagged: List of flagged fragment IDs.

        Returns:
            Overall DecisionType.
        """
        # Any blocked fragments → check if majority are blocked → REJECT vs REVIEW
        if blocked:
            blocked_fraction = len(blocked) / len(prediction.fragments)
            if blocked_fraction > 0.5:
                return DecisionType.REJECT
            # Some blocked, but not majority — still need review
            return DecisionType.REVIEW

        # Overall confidence too low → FALLBACK
        if prediction.overall_confidence < self._overall_threshold:
            return DecisionType.FALLBACK

        # Any flagged fragments → REVIEW
        if flagged:
            return DecisionType.REVIEW

        # All fragments accepted and overall confidence sufficient
        all_accept = all(d == DecisionType.ACCEPT for d in fragment_decisions.values())
        if all_accept and prediction.overall_confidence >= self._accept_threshold:
            return DecisionType.ACCEPT

        return DecisionType.REVIEW
