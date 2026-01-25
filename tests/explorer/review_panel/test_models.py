"""
Tests for review panel models.

Tests cover:
- ReviewResponse serialization
- ReviewRound operations (ratings, majority, minority)
- ChunkReview lifecycle
- MindChange detection
- VerificationResult
- RefinementRecord
"""

import json
from datetime import datetime

import pytest

from explorer.src.review_panel.models import (
    ReviewResponse,
    ReviewRound,
    ChunkReview,
    MindChange,
    RefinementRecord,
    VerificationResult,
    ReviewDecision,
    detect_mind_changes,
    should_refine_vs_deliberate,
    get_rating_pattern,
)


class TestReviewResponse:
    """Tests for ReviewResponse model."""

    def test_create_response(self):
        """Can create a ReviewResponse with required fields."""
        response = ReviewResponse(
            llm="gemini",
            mode="standard",
            rating="⚡",
            mathematical_verification="Verified correct",
            structural_analysis="Deep structural connection",
            naturalness_assessment="Feels natural",
            reasoning="Solid insight",
            confidence="high",
        )
        assert response.llm == "gemini"
        assert response.rating == "⚡"

    def test_response_with_modification(self):
        """Can create a response with modification proposal."""
        response = ReviewResponse(
            llm="chatgpt",
            mode="thinking",
            rating="?",
            mathematical_verification="Needs clarification",
            structural_analysis="Superficial connection",
            naturalness_assessment="Somewhat forced",
            reasoning="Could be improved",
            confidence="medium",
            proposed_modification="The Fano plane has exactly 7 points.",
            modification_rationale="Made more precise",
        )
        assert response.proposed_modification is not None
        assert response.modification_rationale is not None

    def test_response_serialization(self):
        """ReviewResponse serializes and deserializes correctly."""
        response = ReviewResponse(
            llm="claude",
            mode="extended_thinking",
            rating="⚡",
            mathematical_verification="Proof verified",
            structural_analysis="Deep",
            naturalness_assessment="Natural",
            reasoning="Excellent",
            confidence="high",
            changed_mind=True,
            previous_rating="?",
        )

        data = response.to_dict()
        restored = ReviewResponse.from_dict(data)

        assert restored.llm == response.llm
        assert restored.rating == response.rating
        assert restored.changed_mind == response.changed_mind
        assert restored.previous_rating == response.previous_rating


class TestReviewRound:
    """Tests for ReviewRound model."""

    @pytest.fixture
    def unanimous_bless_round(self):
        """Create a round with unanimous blessing."""
        return ReviewRound(
            round_number=1,
            mode="standard",
            responses={
                "gemini": ReviewResponse(
                    llm="gemini", mode="standard", rating="⚡",
                    mathematical_verification="OK", structural_analysis="OK",
                    naturalness_assessment="OK", reasoning="Good", confidence="high"
                ),
                "chatgpt": ReviewResponse(
                    llm="chatgpt", mode="thinking", rating="⚡",
                    mathematical_verification="OK", structural_analysis="OK",
                    naturalness_assessment="OK", reasoning="Good", confidence="high"
                ),
                "claude": ReviewResponse(
                    llm="claude", mode="standard", rating="⚡",
                    mathematical_verification="OK", structural_analysis="OK",
                    naturalness_assessment="OK", reasoning="Good", confidence="high"
                ),
            },
            outcome="unanimous",
            timestamp=datetime.now(),
        )

    @pytest.fixture
    def split_round(self):
        """Create a round with split ratings."""
        return ReviewRound(
            round_number=1,
            mode="standard",
            responses={
                "gemini": ReviewResponse(
                    llm="gemini", mode="standard", rating="⚡",
                    mathematical_verification="OK", structural_analysis="OK",
                    naturalness_assessment="OK", reasoning="Good", confidence="high"
                ),
                "chatgpt": ReviewResponse(
                    llm="chatgpt", mode="thinking", rating="?",
                    mathematical_verification="OK", structural_analysis="OK",
                    naturalness_assessment="OK", reasoning="Unsure", confidence="medium"
                ),
                "claude": ReviewResponse(
                    llm="claude", mode="standard", rating="⚡",
                    mathematical_verification="OK", structural_analysis="OK",
                    naturalness_assessment="OK", reasoning="Good", confidence="high"
                ),
            },
            outcome="split",
            timestamp=datetime.now(),
        )

    def test_get_ratings(self, unanimous_bless_round):
        """get_ratings returns all ratings."""
        ratings = unanimous_bless_round.get_ratings()
        assert ratings == {"gemini": "⚡", "chatgpt": "⚡", "claude": "⚡"}

    def test_is_unanimous_true(self, unanimous_bless_round):
        """is_unanimous returns True when all ratings match."""
        assert unanimous_bless_round.is_unanimous() is True

    def test_is_unanimous_false(self, split_round):
        """is_unanimous returns False when ratings differ."""
        assert split_round.is_unanimous() is False

    def test_get_majority_rating(self, split_round):
        """get_majority_rating returns rating with 2+ votes."""
        assert split_round.get_majority_rating() == "⚡"

    def test_get_minority_llms(self, split_round):
        """get_minority_llms returns LLMs with minority rating."""
        minority = split_round.get_minority_llms()
        assert minority == ["chatgpt"]

    def test_get_majority_llms(self, split_round):
        """get_majority_llms returns LLMs with majority rating."""
        majority = split_round.get_majority_llms()
        assert set(majority) == {"gemini", "claude"}

    def test_round_serialization(self, split_round):
        """ReviewRound serializes and deserializes correctly."""
        data = split_round.to_dict()
        restored = ReviewRound.from_dict(data)

        assert restored.round_number == split_round.round_number
        assert restored.mode == split_round.mode
        assert restored.outcome == split_round.outcome
        assert len(restored.responses) == len(split_round.responses)


class TestChunkReview:
    """Tests for ChunkReview model."""

    def test_create_review(self):
        """Can create a ChunkReview."""
        review = ChunkReview(chunk_id="insight-123")
        assert review.chunk_id == "insight-123"
        assert review.rounds == []
        assert review.final_rating is None

    def test_add_round(self):
        """Can add rounds to a review."""
        review = ChunkReview(chunk_id="test")
        round1 = ReviewRound(
            round_number=1,
            mode="standard",
            responses={},
            outcome="split",
            timestamp=datetime.now(),
        )
        review.add_round(round1)
        assert len(review.rounds) == 1

    def test_add_refinement(self):
        """Adding refinement updates was_refined and final_version."""
        review = ChunkReview(chunk_id="test")
        refinement = RefinementRecord(
            from_version=1,
            to_version=2,
            original_insight="Original",
            refined_insight="Refined",
            changes_made=["Made clearer"],
            addressed_critiques=["Vagueness"],
            unresolved_issues=[],
            refinement_confidence="high",
            triggered_by_ratings={"claude": "?"},
            timestamp=datetime.now(),
        )
        review.add_refinement(refinement)

        assert review.was_refined is True
        assert review.final_version == 2

    def test_finalize(self):
        """finalize sets final state."""
        review = ChunkReview(chunk_id="test")
        review.finalize(rating="⚡", unanimous=True, disputed=False)

        assert review.final_rating == "⚡"
        assert review.is_unanimous is True
        assert review.is_disputed is False
        assert review.reviewed_at is not None

    def test_review_serialization(self, temp_dir):
        """ChunkReview serializes and deserializes correctly."""
        review = ChunkReview(chunk_id="test-123")
        review.finalize(rating="⚡", unanimous=True, disputed=False)

        data = review.to_dict()
        restored = ChunkReview.from_dict(data)

        assert restored.chunk_id == review.chunk_id
        assert restored.final_rating == review.final_rating
        assert restored.is_unanimous == review.is_unanimous

    def test_save_and_load(self, temp_dir):
        """ChunkReview can save and load from disk."""
        review = ChunkReview(chunk_id="test-save")
        review.finalize(rating="?", unanimous=False, disputed=True)
        review.save(temp_dir)

        # Should save to disputed directory
        expected_path = temp_dir / "reviews" / "disputed" / "test-save.json"
        assert expected_path.exists()

        loaded = ChunkReview.load(expected_path)
        assert loaded.chunk_id == "test-save"
        assert loaded.is_disputed is True


class TestMindChange:
    """Tests for MindChange tracking."""

    def test_create_mind_change(self):
        """Can create a MindChange record."""
        change = MindChange(
            llm="gemini",
            round_number=2,
            from_rating="?",
            to_rating="⚡",
            reason="Convinced by Claude's argument",
        )
        assert change.llm == "gemini"
        assert change.from_rating == "?"
        assert change.to_rating == "⚡"

    def test_mind_change_serialization(self):
        """MindChange serializes correctly."""
        change = MindChange(
            llm="chatgpt",
            round_number=3,
            from_rating="✗",
            to_rating="?",
            reason="New information",
        )
        data = change.to_dict()
        restored = MindChange.from_dict(data)

        assert restored.llm == change.llm
        assert restored.from_rating == change.from_rating
        assert restored.to_rating == change.to_rating


class TestDetectMindChanges:
    """Tests for detect_mind_changes function."""

    def test_detect_no_changes(self):
        """Returns empty list when no changes."""
        round1 = ReviewRound(
            round_number=1, mode="standard",
            responses={
                "gemini": ReviewResponse(
                    llm="gemini", mode="standard", rating="⚡",
                    mathematical_verification="", structural_analysis="",
                    naturalness_assessment="", reasoning="", confidence="high"
                ),
            },
            outcome="split", timestamp=datetime.now(),
        )
        round2 = ReviewRound(
            round_number=2, mode="deep",
            responses={
                "gemini": ReviewResponse(
                    llm="gemini", mode="deep_think", rating="⚡",
                    mathematical_verification="", structural_analysis="",
                    naturalness_assessment="", reasoning="", confidence="high"
                ),
            },
            outcome="split", timestamp=datetime.now(),
        )

        changes = detect_mind_changes(round1, round2)
        assert changes == []

    def test_detect_rating_change(self):
        """Detects when a reviewer changes their rating."""
        round1 = ReviewRound(
            round_number=1, mode="standard",
            responses={
                "gemini": ReviewResponse(
                    llm="gemini", mode="standard", rating="?",
                    mathematical_verification="", structural_analysis="",
                    naturalness_assessment="", reasoning="Unsure", confidence="medium"
                ),
            },
            outcome="split", timestamp=datetime.now(),
        )
        round2 = ReviewRound(
            round_number=2, mode="deep",
            responses={
                "gemini": ReviewResponse(
                    llm="gemini", mode="deep_think", rating="⚡",
                    mathematical_verification="", structural_analysis="",
                    naturalness_assessment="", reasoning="Now convinced", confidence="high",
                    new_information="Claude's proof was convincing"
                ),
            },
            outcome="resolved", timestamp=datetime.now(),
        )

        changes = detect_mind_changes(round1, round2)
        assert len(changes) == 1
        assert changes[0].llm == "gemini"
        assert changes[0].from_rating == "?"
        assert changes[0].to_rating == "⚡"


class TestVerificationResult:
    """Tests for VerificationResult model."""

    def test_verified_result(self):
        """Create a verified result."""
        result = VerificationResult(
            verdict="verified",
            precise_statement="The Fano plane has 7 points",
            formal_proof="By construction...",
            confidence=0.95,
        )
        assert result.verdict == "verified"
        assert result.should_auto_reject is False
        assert result.has_proof_artifact is True

    def test_refuted_result_auto_reject(self):
        """Refuted result with high confidence triggers auto-reject."""
        result = VerificationResult(
            verdict="refuted",
            precise_statement="The Fano plane has 8 points",
            counterexample="Counter: it has exactly 7",
            confidence=0.9,
        )
        assert result.verdict == "refuted"
        assert result.should_auto_reject is True

    def test_unclear_result(self):
        """Unclear result doesn't auto-reject."""
        result = VerificationResult(
            verdict="unclear",
            precise_statement="Some claim",
            confidence=0.5,
            concerns=["Ambiguous statement", "Need more context"],
        )
        assert result.should_auto_reject is False
        assert len(result.concerns) == 2

    def test_verification_serialization(self):
        """VerificationResult serializes correctly."""
        result = VerificationResult(
            verdict="verified",
            precise_statement="Test",
            formal_proof="Proof",
            confidence=0.8,
            model_used="deepseek",
            verification_time_seconds=5.2,
        )

        data = result.to_dict()
        restored = VerificationResult.from_dict(data)

        assert restored.verdict == result.verdict
        assert restored.confidence == result.confidence
        assert restored.model_used == result.model_used

    def test_summary_for_reviewers(self):
        """summary_for_reviewers formats correctly."""
        result = VerificationResult(
            verdict="verified",
            precise_statement="Test",
            formal_proof="Short proof",
            confidence=0.9,
        )
        summary = result.summary_for_reviewers()
        assert "VERIFIED" in summary
        assert "90%" in summary


class TestRefinementRecord:
    """Tests for RefinementRecord model."""

    def test_create_refinement(self):
        """Can create a RefinementRecord."""
        record = RefinementRecord(
            from_version=1,
            to_version=2,
            original_insight="Original text",
            refined_insight="Refined text",
            changes_made=["Added precision"],
            addressed_critiques=["Was vague"],
            unresolved_issues=[],
            refinement_confidence="high",
            triggered_by_ratings={"gemini": "?"},
            timestamp=datetime.now(),
            proposer="claude",
            round_proposed=2,
        )
        assert record.from_version == 1
        assert record.to_version == 2
        assert record.proposer == "claude"

    def test_refinement_serialization(self):
        """RefinementRecord serializes correctly."""
        record = RefinementRecord(
            from_version=1,
            to_version=2,
            original_insight="Old",
            refined_insight="New",
            changes_made=["Change"],
            addressed_critiques=["Critique"],
            unresolved_issues=["Issue"],
            refinement_confidence="medium",
            triggered_by_ratings={"chatgpt": "✗"},
            timestamp=datetime.now(),
        )

        data = record.to_dict()
        restored = RefinementRecord.from_dict(data)

        assert restored.from_version == record.from_version
        assert restored.refined_insight == record.refined_insight


class TestGetRatingPattern:
    """Tests for get_rating_pattern utility."""

    def test_all_bless(self):
        """Pattern for all blessings."""
        round_obj = ReviewRound(
            round_number=1, mode="standard",
            responses={
                "gemini": ReviewResponse(
                    llm="gemini", mode="standard", rating="⚡",
                    mathematical_verification="", structural_analysis="",
                    naturalness_assessment="", reasoning="", confidence="high"
                ),
                "chatgpt": ReviewResponse(
                    llm="chatgpt", mode="standard", rating="⚡",
                    mathematical_verification="", structural_analysis="",
                    naturalness_assessment="", reasoning="", confidence="high"
                ),
            },
            outcome="unanimous", timestamp=datetime.now(),
        )
        pattern = get_rating_pattern(round_obj)
        assert pattern == "2×⚡"

    def test_mixed_pattern(self):
        """Pattern for mixed ratings."""
        round_obj = ReviewRound(
            round_number=1, mode="standard",
            responses={
                "gemini": ReviewResponse(
                    llm="gemini", mode="standard", rating="⚡",
                    mathematical_verification="", structural_analysis="",
                    naturalness_assessment="", reasoning="", confidence="high"
                ),
                "chatgpt": ReviewResponse(
                    llm="chatgpt", mode="standard", rating="?",
                    mathematical_verification="", structural_analysis="",
                    naturalness_assessment="", reasoning="", confidence="medium"
                ),
                "claude": ReviewResponse(
                    llm="claude", mode="standard", rating="✗",
                    mathematical_verification="", structural_analysis="",
                    naturalness_assessment="", reasoning="", confidence="high"
                ),
            },
            outcome="split", timestamp=datetime.now(),
        )
        pattern = get_rating_pattern(round_obj)
        assert "⚡" in pattern
        assert "?" in pattern
        assert "✗" in pattern


class TestReviewDecision:
    """Tests for ReviewDecision enum."""

    def test_decision_values(self):
        """ReviewDecision has expected values."""
        assert ReviewDecision.BLESS.value == "⚡"
        assert ReviewDecision.UNCERTAIN.value == "?"
        assert ReviewDecision.REJECT.value == "✗"
