"""
Surgical sequence optimizer.

Determines the optimal order of fragment reduction and fixation
based on anatomical constraints, access limitations, and biomechanical
principles. This is a critical planning output — the order in which
fragments are reduced significantly affects achievable accuracy.

Algorithm:
1. Build a dependency graph (fragment adjacency and reference relationships)
2. Identify the reference fragment (anchor)
3. Topological sort with priority based on:
   - Clinical importance (occlusion-bearing fragments first)
   - Proximity to reference fragment
   - Size (larger fragments provide more stability)
   - Access difficulty (easier access first when possible)

Clinical Principles (from Ellis & Zide, AO CMF):
- "Bottom-up" mandible reduction: body → angle → ramus → condyle
- "Top-down" midface: frontal bar → NOE → zygoma → maxilla
- Occlusion established first whenever possible (IMF or splint)
- Load-bearing segments reduced before non-load-bearing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


class ClinicalPriority(str, Enum):
    """Clinical importance categories for sequencing."""
    CRITICAL = "critical"        # Must be reduced first (reference, occlusion-bearing)
    HIGH = "high"                # Important for functional outcome
    STANDARD = "standard"        # Standard priority
    LOW = "low"                  # Can be deferred or reduced last
    COSMETIC_ONLY = "cosmetic"   # Cosmetic improvement only


class AccessDifficulty(str, Enum):
    """Surgical access difficulty."""
    EASY = "easy"          # Direct transoral or direct skin incision
    MODERATE = "moderate"  # Requires retraction or limited visibility
    DIFFICULT = "difficult"  # Requires specialized approach (endoscopic, etc.)


@dataclass
class FragmentNode:
    """A fragment in the surgical sequencing graph."""
    fragment_id: str
    parent_structure: str
    volume_mm3: float
    centroid_mm: np.ndarray
    is_reference: bool = False
    is_occlusion_bearing: bool = False
    is_load_bearing: bool = False
    clinical_priority: ClinicalPriority = ClinicalPriority.STANDARD
    access_difficulty: AccessDifficulty = AccessDifficulty.MODERATE
    adjacent_fragments: List[str] = field(default_factory=list)

    @property
    def priority_score(self) -> float:
        """Numerical priority score for sorting (higher = reduce first)."""
        base = {
            ClinicalPriority.CRITICAL: 100,
            ClinicalPriority.HIGH: 75,
            ClinicalPriority.STANDARD: 50,
            ClinicalPriority.LOW: 25,
            ClinicalPriority.COSMETIC_ONLY: 10,
        }[self.clinical_priority]

        if self.is_reference:
            base += 50  # Reference always first
        if self.is_occlusion_bearing:
            base += 30
        if self.is_load_bearing:
            base += 20

        # Larger fragments provide more stability → reduce earlier
        volume_bonus = min(20, self.volume_mm3 / 500.0)
        base += volume_bonus

        # Easier access → slight preference for earlier reduction
        access_modifier = {
            AccessDifficulty.EASY: 5,
            AccessDifficulty.MODERATE: 0,
            AccessDifficulty.DIFFICULT: -5,
        }[self.access_difficulty]
        base += access_modifier

        return base


@dataclass
class SequenceStep:
    """A single step in the surgical sequence."""
    step_number: int
    fragment_id: str
    action: str              # "reduce", "fixate", "verify", "splint"
    instructions: str        # Clinical instructions for this step
    estimated_time_minutes: int = 0
    hardware: Optional[str] = None
    verification_required: bool = False
    notes: List[str] = field(default_factory=list)


@dataclass
class SurgicalSequence:
    """Complete surgical sequence for fracture reduction."""
    steps: List[SequenceStep]
    total_fragments: int
    estimated_total_time_minutes: int
    sequence_rationale: str
    critical_steps: List[int] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class SurgicalSequenceOptimizer:
    """
    Determines optimal surgical reduction and fixation sequence.

    Uses anatomical knowledge + graph-based optimization to produce
    a step-by-step surgical plan.
    """

    # Anatomical reduction order templates
    MANDIBLE_ORDER = [
        "symphysis", "parasymphysis", "body", "angle", "ramus",
        "subcondylar", "condyle",
    ]
    MIDFACE_ORDER = [
        "frontal_bar", "noe", "orbital_rim", "zygoma", "maxilla",
        "orbital_floor", "nasal",
    ]

    # Structure classification
    OCCLUSION_BEARING = {"mandible_body", "maxilla", "mandible_symphysis", "mandible_parasymphysis"}
    LOAD_BEARING = {"mandible_body", "mandible_angle", "mandible_ramus", "maxilla"}

    def __init__(self) -> None:
        pass

    def optimize(
        self,
        fragments: List[FragmentNode],
    ) -> SurgicalSequence:
        """
        Generate the optimal surgical sequence.

        Args:
            fragments: List of fragment nodes with adjacency information

        Returns:
            SurgicalSequence with ordered steps
        """
        if not fragments:
            return SurgicalSequence(
                steps=[], total_fragments=0,
                estimated_total_time_minutes=0,
                sequence_rationale="No fragments provided.",
            )

        # Classify fragments
        for frag in fragments:
            self._classify_fragment(frag)

        # Sort by priority
        sorted_fragments = sorted(fragments, key=lambda f: f.priority_score, reverse=True)

        # Build sequence
        steps: List[SequenceStep] = []
        step_num = 1

        # Step 0: Pre-reduction verification
        steps.append(SequenceStep(
            step_number=step_num,
            fragment_id="__preop__",
            action="verify",
            instructions=(
                "Verify patient identity, mark surgical site, confirm CT imaging matches "
                "clinical findings. Perform pre-reduction occlusal assessment."
            ),
            estimated_time_minutes=10,
            verification_required=True,
        ))
        step_num += 1

        # Determine if IMF should be placed first
        has_mandible = any("mandible" in f.parent_structure.lower() for f in fragments)
        has_maxilla = any("maxilla" in f.parent_structure.lower() for f in fragments)
        if has_mandible or has_maxilla:
            steps.append(SequenceStep(
                step_number=step_num,
                fragment_id="__imf__",
                action="splint",
                instructions=(
                    "Establish dental occlusion using arch bars, IMF screws, or "
                    "pre-fabricated occlusal splint. Verify with planned occlusal targets. "
                    "If using AI-planned splint, verify fit against actual anatomy."
                ),
                estimated_time_minutes=20,
                hardware="arch_bars_or_imf_screws",
                verification_required=True,
                notes=["Occlusion is the primary reference for mandible and maxilla reduction"],
            ))
            step_num += 1

        # Generate fragment-specific steps
        for frag in sorted_fragments:
            if frag.is_reference:
                # Reference fragment: verify position, don't move
                steps.append(SequenceStep(
                    step_number=step_num,
                    fragment_id=frag.fragment_id,
                    action="verify",
                    instructions=(
                        f"Verify reference fragment '{frag.fragment_id}' position. "
                        f"This fragment serves as the anatomical anchor. "
                        f"Do not mobilize — confirm position is acceptable."
                    ),
                    estimated_time_minutes=5,
                    verification_required=True,
                ))
                step_num += 1
                continue

            # Reduction step
            reduction_instructions = self._generate_reduction_instructions(frag)
            steps.append(SequenceStep(
                step_number=step_num,
                fragment_id=frag.fragment_id,
                action="reduce",
                instructions=reduction_instructions,
                estimated_time_minutes=self._estimate_reduction_time(frag),
                notes=self._get_reduction_notes(frag),
            ))
            step_num += 1

            # Fixation step
            fixation_instructions = self._generate_fixation_instructions(frag)
            steps.append(SequenceStep(
                step_number=step_num,
                fragment_id=frag.fragment_id,
                action="fixate",
                instructions=fixation_instructions,
                estimated_time_minutes=self._estimate_fixation_time(frag),
                hardware=self._determine_hardware(frag),
            ))
            step_num += 1

            # Verification after critical fragments
            if frag.clinical_priority in (ClinicalPriority.CRITICAL, ClinicalPriority.HIGH):
                steps.append(SequenceStep(
                    step_number=step_num,
                    fragment_id=frag.fragment_id,
                    action="verify",
                    instructions=(
                        f"Verify reduction of {frag.fragment_id}: "
                        f"check cortical continuity, palpate for step-off, "
                        f"assess occlusion if applicable."
                    ),
                    estimated_time_minutes=5,
                    verification_required=True,
                ))
                step_num += 1

        # Final verification
        steps.append(SequenceStep(
            step_number=step_num,
            fragment_id="__final__",
            action="verify",
            instructions=(
                "Final verification: release IMF (if placed), assess occlusion in "
                "centric relation and lateral excursions. Verify facial symmetry. "
                "Obtain intraoperative imaging if indicated. "
                "Document final occlusal status."
            ),
            estimated_time_minutes=15,
            verification_required=True,
        ))

        total_time = sum(s.estimated_time_minutes for s in steps)
        critical = [s.step_number for s in steps if s.verification_required]

        rationale = self._build_rationale(sorted_fragments)

        return SurgicalSequence(
            steps=steps,
            total_fragments=len(fragments),
            estimated_total_time_minutes=total_time,
            sequence_rationale=rationale,
            critical_steps=critical,
        )

    # ── Classification ────────────────────────────────────────────────────────

    def _classify_fragment(self, frag: FragmentNode) -> None:
        """Classify a fragment's clinical properties from its ID and structure."""
        lower_id = frag.fragment_id.lower()
        lower_struct = frag.parent_structure.lower()

        # Occlusion-bearing
        for pattern in self.OCCLUSION_BEARING:
            if pattern in lower_id or pattern in lower_struct:
                frag.is_occlusion_bearing = True
                break

        # Load-bearing
        for pattern in self.LOAD_BEARING:
            if pattern in lower_id or pattern in lower_struct:
                frag.is_load_bearing = True
                break

        # Clinical priority
        if frag.is_reference:
            frag.clinical_priority = ClinicalPriority.CRITICAL
        elif frag.is_occlusion_bearing:
            frag.clinical_priority = ClinicalPriority.HIGH
        elif "condyle" in lower_id:
            frag.clinical_priority = ClinicalPriority.HIGH
        elif "nasal" in lower_id and "noe" not in lower_id:
            frag.clinical_priority = ClinicalPriority.LOW
        else:
            frag.clinical_priority = ClinicalPriority.STANDARD

        # Access difficulty
        if "condyle" in lower_id:
            frag.access_difficulty = AccessDifficulty.DIFFICULT
        elif "orbital_floor" in lower_id:
            frag.access_difficulty = AccessDifficulty.MODERATE
        elif "noe" in lower_id:
            frag.access_difficulty = AccessDifficulty.DIFFICULT
        elif "body" in lower_id or "symphysis" in lower_id:
            frag.access_difficulty = AccessDifficulty.EASY
        else:
            frag.access_difficulty = AccessDifficulty.MODERATE

    # ── Instruction generation ────────────────────────────────────────────────

    def _generate_reduction_instructions(self, frag: FragmentNode) -> str:
        """Generate fragment-specific reduction instructions."""
        structure = frag.parent_structure.lower()
        fid = frag.fragment_id.lower()

        if "mandible" in structure:
            if "condyle" in fid:
                return (
                    f"Reduce {frag.fragment_id} to planned position. "
                    "Approach via preauricular or retromandibular incision (or endoscopic). "
                    "Use planned transform to guide condylar head into fossa. "
                    "Verify condylar seating with gentle mandibular manipulation. "
                    "Assess condyle-fossa relationship before fixation."
                )
            return (
                f"Reduce {frag.fragment_id} to planned position. "
                "Align inferior border cortex first, then verify dental occlusion. "
                "Use reduction forceps for temporary stabilization. "
                "Check for step-off at inferior border and buccal cortex."
            )

        if "maxilla" in structure:
            return (
                f"Reduce {frag.fragment_id} to planned position. "
                "Mobilize with Rowe disimpaction forceps if needed. "
                "Guide into position using planned occlusal splint. "
                "Verify maxillary buttress alignment bilaterally."
            )

        if "zygoma" in structure:
            return (
                f"Reduce {frag.fragment_id} to planned position. "
                "Verify malar projection symmetry. "
                "Assess at zygomaticofrontal suture, inferior orbital rim, "
                "and zygomatic arch. Use Carroll-Girard screw for controlled manipulation."
            )

        return (
            f"Reduce {frag.fragment_id} to planned position as shown in the "
            f"3D planning viewer. Verify cortical continuity and alignment."
        )

    def _generate_fixation_instructions(self, frag: FragmentNode) -> str:
        """Generate fragment-specific fixation instructions."""
        structure = frag.parent_structure.lower()
        hw = self._determine_hardware(frag)

        if "mandible" in structure:
            return (
                f"Apply {hw} fixation to {frag.fragment_id}. "
                "Place plate at ideal biomechanical position (Champy line for angle, "
                "superior and inferior border for body/symphysis). "
                "Ensure bicortical screw purchase. "
                "Avoid tooth roots and inferior alveolar canal."
            )

        if "maxilla" in structure:
            return (
                f"Apply {hw} fixation to {frag.fragment_id} at maxillary buttresses. "
                "Fixate at nasomaxillary and zygomaticomaxillary buttresses. "
                "Avoid infraorbital nerve."
            )

        return f"Apply {hw} fixation to {frag.fragment_id} per planned hardware placement."

    def _get_reduction_notes(self, frag: FragmentNode) -> List[str]:
        """Get clinical notes for a reduction step."""
        notes = []
        fid = frag.fragment_id.lower()

        if "condyle" in fid:
            notes.append("Risk: facial nerve (marginal mandibular branch)")
            notes.append("Use nerve stimulator if retromandibular approach")
        if "body" in fid or "angle" in fid:
            notes.append("Preserve inferior alveolar nerve bundle")
        if "orbital" in fid:
            notes.append("Perform forced duction test before and after")
        if "noe" in fid:
            notes.append("Assess medial canthal tendon integrity")
            notes.append("CSF leak protocol if posterior table involved")

        return notes

    # ── Time estimation ───────────────────────────────────────────────────────

    def _estimate_reduction_time(self, frag: FragmentNode) -> int:
        """Estimate reduction time in minutes."""
        base = 15
        if frag.access_difficulty == AccessDifficulty.DIFFICULT:
            base = 30
        elif frag.access_difficulty == AccessDifficulty.EASY:
            base = 10
        if frag.volume_mm3 > 2000:
            base += 5
        return base

    def _estimate_fixation_time(self, frag: FragmentNode) -> int:
        """Estimate fixation time in minutes."""
        base = 15
        if frag.is_load_bearing:
            base = 20  # More hardware
        if frag.access_difficulty == AccessDifficulty.DIFFICULT:
            base += 10
        return base

    def _determine_hardware(self, frag: FragmentNode) -> str:
        """Determine hardware system for a fragment."""
        structure = frag.parent_structure.lower()
        if "mandible" in structure:
            if "condyle" in frag.fragment_id.lower():
                return "2.0mm condylar plate"
            if "symphysis" in frag.fragment_id.lower():
                return "2.4mm reconstruction plate"
            return "2.0mm mandible plate"
        if "maxilla" in structure:
            return "1.5mm midface plate"
        if "zygoma" in structure:
            return "1.5mm midface plate"
        if "orbit" in structure:
            return "0.4mm titanium mesh"
        if "noe" in structure or "naso" in structure:
            return "1.3mm micro plate"
        return "1.5mm universal plate"

    def _build_rationale(self, sorted_fragments: List[FragmentNode]) -> str:
        """Build a clinical rationale for the chosen sequence."""
        parts = ["Sequence rationale:"]

        ref = [f for f in sorted_fragments if f.is_reference]
        if ref:
            parts.append(
                f"- Reference fragment ({ref[0].fragment_id}) verified first as the anatomical anchor."
            )

        occ = [f for f in sorted_fragments if f.is_occlusion_bearing and not f.is_reference]
        if occ:
            parts.append(
                f"- Occlusion-bearing fragments ({', '.join(f.fragment_id for f in occ)}) "
                f"reduced early to establish dental relationship."
            )

        difficult = [f for f in sorted_fragments if f.access_difficulty == AccessDifficulty.DIFFICULT]
        if difficult:
            parts.append(
                f"- Difficult-access fragments ({', '.join(f.fragment_id for f in difficult)}) "
                f"sequenced to avoid interference with previously placed hardware."
            )

        parts.append(
            "- Sequence follows established AO CMF principles: "
            "establish occlusion → reduce load-bearing segments → "
            "address non-load-bearing and cosmetic segments."
        )

        return "\n".join(parts)
