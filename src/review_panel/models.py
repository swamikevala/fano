"""
Data models for the automated review panel.

Models for tracking review rounds, responses, mind changes,
and final review outcomes.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


class ReviewDecision(Enum):
    """Possible review decisions."""
    BLESS = "⚡"          # Approve as axiom
    UNCERTAIN = "?"       # Interesting but needs work
    REJECT = "✗"          # Discard


@dataclass
class VerificationResult:
    """Result of DeepSeek mathematical verification."""

    # Core verdict
    verdict: str  # "verified" | "refuted" | "unclear" | "not_applicable"

    # The precise mathematical statement evaluated
    precise_statement: str

    # Evidence
    formal_proof: Optional[str] = None      # If verified
    counterexample: Optional[str] = None    # If refuted
    proof_sketch: Optional[str] = None      # If unclear but partially analyzed

    # Confidence in verdict (0.0 to 1.0)
    confidence: float = 0.0

    # Specific concerns or caveats
    concerns: list[str] = field(default_factory=list)

    # What was checked
    claims_extracted: list[str] = field(default_factory=list)
    claims_verified: list[str] = field(default_factory=list)
    claims_refuted: list[str] = field(default_factory=list)
    claims_unclear: list[str] = field(default_factory=list)

    # Metadata
    model_used: str = ""
    verification_time_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "precise_statement": self.precise_statement,
            "formal_proof": self.formal_proof,
            "counterexample": self.counterexample,
            "proof_sketch": self.proof_sketch,
            "confidence": self.confidence,
            "concerns": self.concerns,
            "claims_extracted": self.claims_extracted,
            "claims_verified": self.claims_verified,
            "claims_refuted": self.claims_refuted,
            "claims_unclear": self.claims_unclear,
            "model_used": self.model_used,
            "verification_time_seconds": self.verification_time_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VerificationResult":
        return cls(
            verdict=data["verdict"],
            precise_statement=data["precise_statement"],
            formal_proof=data.get("formal_proof"),
            counterexample=data.get("counterexample"),
            proof_sketch=data.get("proof_sketch"),
            confidence=data.get("confidence", 0.0),
            concerns=data.get("concerns", []),
            claims_extracted=data.get("claims_extracted", []),
            claims_verified=data.get("claims_verified", []),
            claims_refuted=data.get("claims_refuted", []),
            claims_unclear=data.get("claims_unclear", []),
            model_used=data.get("model_used", ""),
            verification_time_seconds=data.get("verification_time_seconds", 0.0),
        )

    def summary_for_reviewers(self) -> str:
        """Format for inclusion in Round 2 prompts."""
        if self.verdict == "verified":
            proof_preview = self.formal_proof[:500] + "..." if self.formal_proof and len(self.formal_proof) > 500 else self.formal_proof
            return f"✓ MATHEMATICALLY VERIFIED (confidence: {self.confidence:.0%})\n\nProof:\n{proof_preview or 'See full proof in artifacts'}"
        elif self.verdict == "refuted":
            return f"✗ MATHEMATICALLY REFUTED (confidence: {self.confidence:.0%})\n\nCounterexample:\n{self.counterexample}"
        elif self.verdict == "unclear":
            concerns_str = "\n".join(f"  • {c}" for c in self.concerns)
            return f"? VERIFICATION UNCLEAR\n\nConcerns:\n{concerns_str}"
        else:
            return "○ No mathematical claims requiring verification"

    @property
    def should_auto_reject(self) -> bool:
        """Check if this result warrants automatic rejection."""
        return self.verdict == "refuted" and self.confidence >= 0.8

    @property
    def has_proof_artifact(self) -> bool:
        """Check if there's a formal proof to attach as artifact."""
        return self.verdict == "verified" and bool(self.formal_proof)


@dataclass
class ReviewResponse:
    """
    A single reviewer's response in a review round.
    """
    llm: str                              # "gemini" | "chatgpt" | "claude"
    mode: str                             # "standard" | "deep_think" | "pro" | "extended_thinking"
    rating: str                           # "⚡" | "?" | "✗" | "ABANDON"
    mathematical_verification: str         # Verify or refute specific claims
    structural_analysis: str               # Is connection deep or superficial?
    naturalness_assessment: str            # Does it feel inevitable?
    reasoning: str                         # Overall justification (2-4 sentences)
    confidence: str                        # "high" | "medium" | "low"

    # Modification proposal (any round)
    proposed_modification: Optional[str] = None    # Proposed rewritten insight
    modification_rationale: Optional[str] = None   # Why the modification is needed

    # Round 2+ fields (when reviewing others' responses)
    new_information: Optional[str] = None  # What did others point out?
    changed_mind: Optional[bool] = None    # Did this change things?
    previous_rating: Optional[str] = None  # Rating before reconsideration

    # Round 3 fields (deliberation)
    strongest_argument: Optional[str] = None    # Minority's best case
    response_to_argument: Optional[str] = None  # Majority's response
    final_stance: Optional[str] = None          # "concede" | "maintain"

    def to_dict(self) -> dict:
        return {
            "llm": self.llm,
            "mode": self.mode,
            "rating": self.rating,
            "mathematical_verification": self.mathematical_verification,
            "structural_analysis": self.structural_analysis,
            "naturalness_assessment": self.naturalness_assessment,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "proposed_modification": self.proposed_modification,
            "modification_rationale": self.modification_rationale,
            "new_information": self.new_information,
            "changed_mind": self.changed_mind,
            "previous_rating": self.previous_rating,
            "strongest_argument": self.strongest_argument,
            "response_to_argument": self.response_to_argument,
            "final_stance": self.final_stance,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReviewResponse":
        return cls(
            llm=data["llm"],
            mode=data["mode"],
            rating=data["rating"],
            mathematical_verification=data["mathematical_verification"],
            structural_analysis=data["structural_analysis"],
            naturalness_assessment=data["naturalness_assessment"],
            reasoning=data["reasoning"],
            confidence=data["confidence"],
            proposed_modification=data.get("proposed_modification"),
            modification_rationale=data.get("modification_rationale"),
            new_information=data.get("new_information"),
            changed_mind=data.get("changed_mind"),
            previous_rating=data.get("previous_rating"),
            strongest_argument=data.get("strongest_argument"),
            response_to_argument=data.get("response_to_argument"),
            final_stance=data.get("final_stance"),
        )


@dataclass
class MindChange:
    """
    Record of a reviewer changing their position.
    """
    llm: str                # Which model changed
    round_number: int       # When they changed (2 or 3)
    from_rating: str        # Original rating
    to_rating: str          # New rating
    reason: str             # What convinced them

    def to_dict(self) -> dict:
        return {
            "llm": self.llm,
            "round_number": self.round_number,
            "from_rating": self.from_rating,
            "to_rating": self.to_rating,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MindChange":
        return cls(
            llm=data["llm"],
            round_number=data["round_number"],
            from_rating=data["from_rating"],
            to_rating=data["to_rating"],
            reason=data["reason"],
        )


@dataclass
class ReviewRound:
    """
    A single round of the review process.
    """
    round_number: int                      # 1, 2, or 3
    mode: str                              # "standard" or "deep"
    responses: dict[str, ReviewResponse]   # Keyed by LLM name
    outcome: str                           # "unanimous" | "split" | "resolved"
    timestamp: datetime

    def to_dict(self) -> dict:
        return {
            "round_number": self.round_number,
            "mode": self.mode,
            "responses": {k: v.to_dict() for k, v in self.responses.items()},
            "outcome": self.outcome,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReviewRound":
        return cls(
            round_number=data["round_number"],
            mode=data["mode"],
            responses={k: ReviewResponse.from_dict(v) for k, v in data["responses"].items()},
            outcome=data["outcome"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )

    def get_ratings(self) -> dict[str, str]:
        """Get rating from each reviewer."""
        return {llm: resp.rating for llm, resp in self.responses.items()}

    def is_unanimous(self) -> bool:
        """Check if all reviewers agree."""
        ratings = list(self.get_ratings().values())
        return len(set(ratings)) == 1

    def get_majority_rating(self) -> Optional[str]:
        """Get the majority rating (2 out of 3)."""
        ratings = list(self.get_ratings().values())
        for rating in ["⚡", "?", "✗"]:
            if ratings.count(rating) >= 2:
                return rating
        return None

    def get_minority_llms(self) -> list[str]:
        """Get LLMs that hold minority position."""
        majority = self.get_majority_rating()
        if not majority:
            return []
        return [llm for llm, resp in self.responses.items() if resp.rating != majority]

    def get_majority_llms(self) -> list[str]:
        """Get LLMs that hold majority position."""
        majority = self.get_majority_rating()
        if not majority:
            return list(self.responses.keys())
        return [llm for llm, resp in self.responses.items() if resp.rating == majority]


@dataclass
class RefinementRecord:
    """
    Record of a refinement/modification made to an insight during review.

    This can represent either:
    1. Old-style refinement: Claude Opus rewrites based on critiques
    2. New-style modification: An LLM proposes a fix during review
    """
    from_version: int                      # Original version number
    to_version: int                        # New version number
    original_insight: str                  # Text before refinement
    refined_insight: str                   # Text after refinement
    changes_made: list[str]                # What was changed
    addressed_critiques: list[str]         # Which concerns were addressed
    unresolved_issues: list[str]           # Issues that couldn't be fixed
    refinement_confidence: str             # "high" | "medium" | "low"
    triggered_by_ratings: dict[str, str]   # Ratings that prompted this
    timestamp: datetime
    # New fields for modification workflow
    proposer: Optional[str] = None         # Which LLM proposed this ("gemini", "claude", "chatgpt")
    round_proposed: Optional[int] = None   # Which round (1, 2, 3)

    def to_dict(self) -> dict:
        result = {
            "from_version": self.from_version,
            "to_version": self.to_version,
            "original_insight": self.original_insight,
            "refined_insight": self.refined_insight,
            "changes_made": self.changes_made,
            "addressed_critiques": self.addressed_critiques,
            "unresolved_issues": self.unresolved_issues,
            "refinement_confidence": self.refinement_confidence,
            "triggered_by_ratings": self.triggered_by_ratings,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.proposer:
            result["proposer"] = self.proposer
        if self.round_proposed:
            result["round_proposed"] = self.round_proposed
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "RefinementRecord":
        return cls(
            from_version=data["from_version"],
            to_version=data["to_version"],
            original_insight=data["original_insight"],
            refined_insight=data["refined_insight"],
            changes_made=data.get("changes_made", []),
            addressed_critiques=data.get("addressed_critiques", []),
            unresolved_issues=data.get("unresolved_issues", []),
            refinement_confidence=data.get("refinement_confidence", "medium"),
            triggered_by_ratings=data.get("triggered_by_ratings", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            proposer=data.get("proposer"),
            round_proposed=data.get("round_proposed"),
        )


@dataclass
class ChunkReview:
    """
    Complete review record for an insight.
    """
    chunk_id: str                          # ID of insight being reviewed
    rounds: list[ReviewRound] = field(default_factory=list)
    refinements: list[RefinementRecord] = field(default_factory=list)  # Refinement history
    final_version: int = 1                 # Which version was blessed/rejected
    was_refined: bool = False              # Did the insight go through refinement?
    final_rating: Optional[str] = None     # "⚡" | "?" | "✗"
    is_unanimous: bool = False             # Did all reviewers agree?
    is_disputed: bool = False              # Persistent 2-1 split?
    mind_changes: list[MindChange] = field(default_factory=list)
    total_tokens_used: int = 0             # For cost tracking
    review_duration_seconds: float = 0.0   # How long the review took
    reviewed_at: Optional[datetime] = None

    # Final insight text (if modified during deliberation, this is the modified version)
    final_insight_text: Optional[str] = None

    # DeepSeek mathematical verification (between Round 1 and Round 2)
    math_verification: Optional[VerificationResult] = None
    math_verification_skipped: bool = False
    math_verification_skip_reason: str = ""
    rejection_reason: Optional[str] = None  # If auto-rejected by DeepSeek

    # Priority-based pausing
    is_paused: bool = False                # True if review was paused for higher priority item
    paused_for_id: Optional[str] = None    # ID of the higher priority item we switched to

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "rounds": [r.to_dict() for r in self.rounds],
            "refinements": [r.to_dict() for r in self.refinements],
            "final_version": self.final_version,
            "was_refined": self.was_refined,
            "final_rating": self.final_rating,
            "is_unanimous": self.is_unanimous,
            "is_disputed": self.is_disputed,
            "mind_changes": [m.to_dict() for m in self.mind_changes],
            "total_tokens_used": self.total_tokens_used,
            "review_duration_seconds": self.review_duration_seconds,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "final_insight_text": self.final_insight_text,
            "math_verification": self.math_verification.to_dict() if self.math_verification else None,
            "math_verification_skipped": self.math_verification_skipped,
            "math_verification_skip_reason": self.math_verification_skip_reason,
            "rejection_reason": self.rejection_reason,
            "is_paused": self.is_paused,
            "paused_for_id": self.paused_for_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChunkReview":
        math_verification = None
        if data.get("math_verification"):
            math_verification = VerificationResult.from_dict(data["math_verification"])

        return cls(
            chunk_id=data["chunk_id"],
            rounds=[ReviewRound.from_dict(r) for r in data.get("rounds", [])],
            refinements=[RefinementRecord.from_dict(r) for r in data.get("refinements", [])],
            final_version=data.get("final_version", 1),
            was_refined=data.get("was_refined", False),
            final_rating=data.get("final_rating"),
            is_unanimous=data.get("is_unanimous", False),
            is_disputed=data.get("is_disputed", False),
            mind_changes=[MindChange.from_dict(m) for m in data.get("mind_changes", [])],
            total_tokens_used=data.get("total_tokens_used", 0),
            review_duration_seconds=data.get("review_duration_seconds", 0.0),
            reviewed_at=datetime.fromisoformat(data["reviewed_at"]) if data.get("reviewed_at") else None,
            final_insight_text=data.get("final_insight_text"),
            math_verification=math_verification,
            math_verification_skipped=data.get("math_verification_skipped", False),
            math_verification_skip_reason=data.get("math_verification_skip_reason", ""),
            rejection_reason=data.get("rejection_reason"),
            is_paused=data.get("is_paused", False),
            paused_for_id=data.get("paused_for_id"),
        )

    def save(self, base_dir: Path):
        """Save review to disk."""
        # Determine subdirectory based on disputed status
        if self.is_disputed:
            review_dir = base_dir / "reviews" / "disputed"
        else:
            review_dir = base_dir / "reviews" / "completed"

        review_dir.mkdir(parents=True, exist_ok=True)
        filepath = review_dir / f"{self.chunk_id}.json"

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: Path) -> "ChunkReview":
        """Load review from disk."""
        with open(filepath, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def add_round(self, round: ReviewRound):
        """Add a review round."""
        self.rounds.append(round)

    def add_refinement(self, refinement: RefinementRecord):
        """Add a refinement record."""
        self.refinements.append(refinement)
        self.was_refined = True
        self.final_version = refinement.to_version

    def finalize(self, rating: str, unanimous: bool, disputed: bool):
        """Finalize the review with outcome."""
        self.final_rating = rating
        self.is_unanimous = unanimous
        self.is_disputed = disputed
        self.reviewed_at = datetime.now()


def detect_mind_changes(
    round1: ReviewRound,
    round2: ReviewRound,
) -> list[MindChange]:
    """
    Detect which reviewers changed their minds between rounds.

    Args:
        round1: Earlier round
        round2: Later round

    Returns:
        List of MindChange records
    """
    changes = []

    for llm in round1.responses:
        if llm not in round2.responses:
            continue

        r1_rating = round1.responses[llm].rating
        r2_rating = round2.responses[llm].rating

        if r1_rating != r2_rating:
            # Get the reason from Round 2 response
            r2_resp = round2.responses[llm]
            reason = r2_resp.new_information or r2_resp.reasoning or "No reason given"

            changes.append(MindChange(
                llm=llm,
                round_number=round2.round_number,
                from_rating=r1_rating,
                to_rating=r2_rating,
                reason=reason[:500],  # Truncate if too long
            ))

    return changes


# Patterns in critique text that suggest refinement vs deliberation
REFINEMENT_PATTERNS = [
    "vague wording",
    "unclear",
    "imprecise",
    "could be clearer",
    "needs sharpening",
    "hedging",
    "could be stated more",
    "framing is",
    "missing precision",
    "articulation",
    "language could",
    "could be more specific",
]

FUNDAMENTAL_FLAW_PATTERNS = [
    "core claim is wrong",
    "fundamentally flawed",
    "numerology",
    "unfalsifiable",
    "too vague to evaluate",
    "restates input",
    "no actual insight",
    "circular reasoning",
    "doesn't actually say anything",
    "not a valid",
    "mathematically incorrect",
]


def should_refine_vs_deliberate(round1: ReviewRound) -> tuple[bool, str]:
    """
    Determine whether to refine the insight or proceed to deliberation.

    Based on the decision criteria from chunk-authorship-refinement.md:
    - Refine: articulation/precision/framing issues that can be fixed
    - Deliberate: fundamental disagreement about the claim itself

    Args:
        round1: The completed Round 1 review

    Returns:
        Tuple of (should_refine, reason)
    """
    ratings = round1.get_ratings()
    rating_values = list(ratings.values())

    # If unanimous, no need for refinement or deliberation
    if len(set(rating_values)) == 1:
        return False, "unanimous"

    # Collect all reasoning text
    all_reasoning = ""
    for resp in round1.responses.values():
        all_reasoning += " " + resp.reasoning.lower()
        all_reasoning += " " + resp.mathematical_verification.lower()
        all_reasoning += " " + resp.structural_analysis.lower()
        all_reasoning += " " + resp.naturalness_assessment.lower()

    # Check for fundamental flaws (should deliberate, not refine)
    for pattern in FUNDAMENTAL_FLAW_PATTERNS:
        if pattern in all_reasoning:
            return False, f"fundamental flaw detected: {pattern}"

    # Check for refinement opportunities
    refinement_signals = 0
    for pattern in REFINEMENT_PATTERNS:
        if pattern in all_reasoning:
            refinement_signals += 1

    # Decision matrix based on rating pattern
    bless_count = rating_values.count("⚡")
    uncertain_count = rating_values.count("?")
    reject_count = rating_values.count("✗")

    # Strong candidates for refinement:
    # 2×⚡ + 1×? : Strong with minor hesitation - may refine to resolve hesitation
    # 1×⚡ + 2×? : Promising but underdeveloped - may refine to clarify
    # 1×⚡ + 1×? + 1×✗ : Full spread - analyze critiques to decide
    if bless_count >= 1 and reject_count == 0 and refinement_signals >= 1:
        return True, f"promising with {refinement_signals} refinement signals"

    # Mixed with refinement signals - try refinement first
    if refinement_signals >= 2:
        return True, f"{refinement_signals} refinement signals detected"

    # Default: go to deliberation for fundamental disagreement
    if bless_count >= 1 and reject_count >= 1:
        return False, "fundamental disagreement (bless vs reject)"

    # 2×? + 1×anything - needs development, may benefit from refinement
    if uncertain_count >= 2:
        return True, "mostly uncertain, try refinement"

    return False, "default to deliberation"


def get_rating_pattern(round: ReviewRound) -> str:
    """
    Get a human-readable pattern of ratings.

    Returns strings like "2×⚡ + 1×?" for logging.
    """
    ratings = list(round.get_ratings().values())
    bless = ratings.count("⚡")
    uncertain = ratings.count("?")
    reject = ratings.count("✗")

    parts = []
    if bless > 0:
        parts.append(f"{bless}×⚡")
    if uncertain > 0:
        parts.append(f"{uncertain}×?")
    if reject > 0:
        parts.append(f"{reject}×✗")

    return " + ".join(parts) if parts else "no ratings"
