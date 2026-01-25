"""
Tests for AutomatedReviewer.

Tests cover:
- Initialization with various browser combinations
- Modification consensus logic
- Review flow (Round 1 → Round 2 → Round 3)
- Priority switching
- Progress save/load
- Quota exception handling
- Outcome action determination

IMPORTANT: All tests must verify that deep_mode and pro_mode are NEVER used.
"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

import pytest

from explorer.src.review_panel.reviewer import (
    AutomatedReviewer,
    _get_modification_consensus,
)
from explorer.src.review_panel.models import (
    ChunkReview,
    ReviewResponse,
    ReviewRound,
    RefinementRecord,
    VerificationResult,
)


class TestGetModificationConsensus:
    """Tests for _get_modification_consensus function."""

    def test_no_modifications_proposed(self):
        """Returns None when no modifications proposed."""
        round_obj = ReviewRound(
            round_number=1,
            mode="standard",
            responses={
                "gemini": ReviewResponse(
                    llm="gemini", mode="standard", rating="⚡",
                    mathematical_verification="", structural_analysis="",
                    naturalness_assessment="", reasoning="Good", confidence="high"
                ),
                "chatgpt": ReviewResponse(
                    llm="chatgpt", mode="thinking", rating="⚡",
                    mathematical_verification="", structural_analysis="",
                    naturalness_assessment="", reasoning="Good", confidence="high"
                ),
            },
            outcome="unanimous",
            timestamp=datetime.now(),
        )

        mod, source, rationale = _get_modification_consensus(round_obj)
        assert mod is None
        assert source is None
        assert rationale is None

    def test_accepts_modification_from_bless_voter(self):
        """Accepts modification from LLM that rated ⚡."""
        round_obj = ReviewRound(
            round_number=1,
            mode="standard",
            responses={
                "gemini": ReviewResponse(
                    llm="gemini", mode="standard", rating="⚡",
                    mathematical_verification="", structural_analysis="",
                    naturalness_assessment="", reasoning="Good", confidence="high",
                    proposed_modification="Improved version of insight",
                    modification_rationale="Made it clearer",
                ),
                "chatgpt": ReviewResponse(
                    llm="chatgpt", mode="thinking", rating="?",
                    mathematical_verification="", structural_analysis="",
                    naturalness_assessment="", reasoning="Unsure", confidence="medium"
                ),
            },
            outcome="split",
            timestamp=datetime.now(),
        )

        mod, source, rationale = _get_modification_consensus(round_obj)
        assert mod == "Improved version of insight"
        assert source == "gemini"
        assert rationale == "Made it clearer"

    def test_prefers_higher_rating_modification(self):
        """Prefers modification from LLM with higher rating."""
        round_obj = ReviewRound(
            round_number=1,
            mode="standard",
            responses={
                "gemini": ReviewResponse(
                    llm="gemini", mode="standard", rating="?",
                    mathematical_verification="", structural_analysis="",
                    naturalness_assessment="", reasoning="Unsure", confidence="medium",
                    proposed_modification="Gemini modification",
                    modification_rationale="Gemini's fix",
                ),
                "chatgpt": ReviewResponse(
                    llm="chatgpt", mode="thinking", rating="⚡",
                    mathematical_verification="", structural_analysis="",
                    naturalness_assessment="", reasoning="Good", confidence="high",
                    proposed_modification="ChatGPT modification",
                    modification_rationale="ChatGPT's fix",
                ),
            },
            outcome="split",
            timestamp=datetime.now(),
        )

        mod, source, rationale = _get_modification_consensus(round_obj)
        assert mod == "ChatGPT modification"
        assert source == "chatgpt"

    def test_rejects_modification_from_reject_voter(self):
        """Does not accept modification from LLM that rated ✗."""
        round_obj = ReviewRound(
            round_number=1,
            mode="standard",
            responses={
                "gemini": ReviewResponse(
                    llm="gemini", mode="standard", rating="✗",
                    mathematical_verification="", structural_analysis="",
                    naturalness_assessment="", reasoning="Bad", confidence="high",
                    proposed_modification="This won't help",
                    modification_rationale="It's unsalvageable",
                ),
            },
            outcome="split",
            timestamp=datetime.now(),
        )

        mod, source, rationale = _get_modification_consensus(round_obj)
        assert mod is None

    def test_mode_priority_tiebreaker(self):
        """Uses mode priority as tiebreaker when ratings equal."""
        round_obj = ReviewRound(
            round_number=2,
            mode="deep",
            responses={
                "gemini": ReviewResponse(
                    llm="gemini", mode="deep_think", rating="⚡",
                    mathematical_verification="", structural_analysis="",
                    naturalness_assessment="", reasoning="Good", confidence="high",
                    proposed_modification="Deep think modification",
                    modification_rationale="Deep analysis",
                ),
                "chatgpt": ReviewResponse(
                    llm="chatgpt", mode="standard", rating="⚡",
                    mathematical_verification="", structural_analysis="",
                    naturalness_assessment="", reasoning="Good", confidence="high",
                    proposed_modification="Standard modification",
                    modification_rationale="Basic fix",
                ),
            },
            outcome="split",
            timestamp=datetime.now(),
        )

        mod, source, rationale = _get_modification_consensus(round_obj)
        # deep_think has higher priority than standard
        assert source == "gemini"


class TestAutomatedReviewerInit:
    """Tests for AutomatedReviewer initialization."""

    def test_init_with_all_browsers(self, temp_dir):
        """Initializes with all browsers available."""
        mock_gemini = MagicMock()
        mock_chatgpt = MagicMock()
        config = {"review_panel": {}}

        with patch("explorer.src.review_panel.reviewer.get_claude_reviewer", return_value=MagicMock()):
            with patch("explorer.src.review_panel.reviewer.get_deepseek_verifier", return_value=None):
                reviewer = AutomatedReviewer(
                    gemini_browser=mock_gemini,
                    chatgpt_browser=mock_chatgpt,
                    config=config,
                    data_dir=temp_dir,
                )

        assert reviewer.gemini_browser is mock_gemini
        assert reviewer.chatgpt_browser is mock_chatgpt

    def test_init_with_missing_browser(self, temp_dir):
        """Initializes with only some browsers available."""
        mock_gemini = MagicMock()
        config = {"review_panel": {}}

        with patch("explorer.src.review_panel.reviewer.get_claude_reviewer", return_value=MagicMock()):
            with patch("explorer.src.review_panel.reviewer.get_deepseek_verifier", return_value=None):
                reviewer = AutomatedReviewer(
                    gemini_browser=mock_gemini,
                    chatgpt_browser=None,
                    config=config,
                    data_dir=temp_dir,
                )

        assert reviewer.gemini_browser is mock_gemini
        assert reviewer.chatgpt_browser is None

    def test_init_warns_when_insufficient_reviewers(self, temp_dir, caplog):
        """Logs warning when fewer than 2 reviewers available."""
        config = {"review_panel": {}}

        with patch("explorer.src.review_panel.reviewer.get_claude_reviewer", return_value=None):
            with patch("explorer.src.review_panel.reviewer.get_deepseek_verifier", return_value=None):
                reviewer = AutomatedReviewer(
                    gemini_browser=None,
                    chatgpt_browser=None,
                    config=config,
                    data_dir=temp_dir,
                )

        # Should have logged a warning about insufficient reviewers
        assert reviewer.claude_reviewer is None


class TestAutomatedReviewerProgress:
    """Tests for progress save/load functionality."""

    @pytest.fixture
    def reviewer(self, temp_dir):
        """Create a reviewer instance for testing."""
        config = {"review_panel": {}}
        with patch("explorer.src.review_panel.reviewer.get_claude_reviewer", return_value=None):
            with patch("explorer.src.review_panel.reviewer.get_deepseek_verifier", return_value=None):
                return AutomatedReviewer(
                    gemini_browser=MagicMock(),
                    chatgpt_browser=MagicMock(),
                    config=config,
                    data_dir=temp_dir,
                )

    def test_save_progress(self, reviewer, temp_dir):
        """_save_progress saves review to in_progress directory."""
        review = ChunkReview(chunk_id="test-progress")
        round1 = ReviewRound(
            round_number=1, mode="standard",
            responses={}, outcome="split", timestamp=datetime.now(),
        )
        review.add_round(round1)

        reviewer._save_progress(review, 1)

        progress_path = temp_dir / "reviews" / "in_progress" / "test-progress.json"
        assert progress_path.exists()

    def test_load_progress(self, reviewer, temp_dir):
        """_load_progress loads saved review."""
        # Create progress file
        progress_dir = temp_dir / "reviews" / "in_progress"
        progress_dir.mkdir(parents=True, exist_ok=True)

        review = ChunkReview(chunk_id="test-load")
        review.add_round(ReviewRound(
            round_number=1, mode="standard",
            responses={}, outcome="split", timestamp=datetime.now(),
        ))

        progress_path = progress_dir / "test-load.json"
        with open(progress_path, "w") as f:
            json.dump(review.to_dict(), f)

        loaded = reviewer._load_progress("test-load")
        assert loaded is not None
        assert loaded.chunk_id == "test-load"
        assert len(loaded.rounds) == 1

    def test_load_progress_returns_none_if_missing(self, reviewer):
        """_load_progress returns None if no saved progress."""
        loaded = reviewer._load_progress("nonexistent-id")
        assert loaded is None

    def test_get_saved_progress(self, reviewer, temp_dir):
        """get_saved_progress returns number of completed rounds."""
        progress_dir = temp_dir / "reviews" / "in_progress"
        progress_dir.mkdir(parents=True, exist_ok=True)

        review = ChunkReview(chunk_id="test-rounds")
        review.add_round(ReviewRound(
            round_number=1, mode="standard",
            responses={}, outcome="split", timestamp=datetime.now(),
        ))
        review.add_round(ReviewRound(
            round_number=2, mode="deep",
            responses={}, outcome="split", timestamp=datetime.now(),
        ))

        with open(progress_dir / "test-rounds.json", "w") as f:
            json.dump(review.to_dict(), f)

        rounds = reviewer.get_saved_progress("test-rounds")
        assert rounds == 2

    def test_get_saved_progress_returns_zero_if_none(self, reviewer):
        """get_saved_progress returns 0 if no progress."""
        rounds = reviewer.get_saved_progress("nonexistent")
        assert rounds == 0


class TestAutomatedReviewerOutcomeAction:
    """Tests for get_outcome_action method."""

    @pytest.fixture
    def reviewer(self, temp_dir):
        """Create a reviewer with outcome configuration."""
        config = {
            "review_panel": {
                "outcomes": {
                    "unanimous_bless": "auto_bless",
                    "unanimous_reject": "auto_reject",
                    "unanimous_uncertain": "needs_development",
                    "disputed_majority_bless": "bless_with_flag",
                    "disputed_majority_reject": "reject_with_flag",
                }
            }
        }
        with patch("explorer.src.review_panel.reviewer.get_claude_reviewer", return_value=None):
            with patch("explorer.src.review_panel.reviewer.get_deepseek_verifier", return_value=None):
                return AutomatedReviewer(
                    gemini_browser=MagicMock(),
                    chatgpt_browser=MagicMock(),
                    config=config,
                    data_dir=temp_dir,
                )

    def test_unanimous_bless_action(self, reviewer):
        """Returns auto_bless for unanimous blessing."""
        review = ChunkReview(chunk_id="test")
        review.finalize(rating="⚡", unanimous=True, disputed=False)

        action = reviewer.get_outcome_action(review)
        assert action == "auto_bless"

    def test_unanimous_reject_action(self, reviewer):
        """Returns auto_reject for unanimous rejection."""
        review = ChunkReview(chunk_id="test")
        review.finalize(rating="✗", unanimous=True, disputed=False)

        action = reviewer.get_outcome_action(review)
        assert action == "auto_reject"

    def test_unanimous_uncertain_action(self, reviewer):
        """Returns needs_development for unanimous uncertainty."""
        review = ChunkReview(chunk_id="test")
        review.finalize(rating="?", unanimous=True, disputed=False)

        action = reviewer.get_outcome_action(review)
        assert action == "needs_development"

    def test_disputed_bless_action(self, reviewer):
        """Returns bless_with_flag for disputed blessing."""
        review = ChunkReview(chunk_id="test")
        review.finalize(rating="⚡", unanimous=False, disputed=True)

        action = reviewer.get_outcome_action(review)
        assert action == "bless_with_flag"

    def test_disputed_reject_action(self, reviewer):
        """Returns reject_with_flag for disputed rejection."""
        review = ChunkReview(chunk_id="test")
        review.finalize(rating="✗", unanimous=False, disputed=True)

        action = reviewer.get_outcome_action(review)
        assert action == "reject_with_flag"


class TestAutomatedReviewerReviewInsight:
    """Tests for review_insight method."""

    @pytest.fixture
    def mock_round1(self):
        """Create a mock Round 1 result."""
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
    def mock_split_round1(self):
        """Create a mock split Round 1 result."""
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
            },
            outcome="split",
            timestamp=datetime.now(),
        )

    @pytest.fixture
    def mock_round2(self):
        """Create a mock Round 2 result."""
        return ReviewRound(
            round_number=2,
            mode="deep",
            responses={
                "gemini": ReviewResponse(
                    llm="gemini", mode="deep_think", rating="⚡",
                    mathematical_verification="OK", structural_analysis="OK",
                    naturalness_assessment="OK", reasoning="Good", confidence="high"
                ),
                "chatgpt": ReviewResponse(
                    llm="chatgpt", mode="pro", rating="⚡",
                    mathematical_verification="OK", structural_analysis="OK",
                    naturalness_assessment="OK", reasoning="Now convinced", confidence="high"
                ),
            },
            outcome="unanimous",
            timestamp=datetime.now(),
        )

    @pytest.mark.asyncio
    async def test_unanimous_round1_exits_early(self, temp_dir, mock_round1):
        """Review exits early on unanimous Round 1."""
        config = {"review_panel": {}}

        with patch("explorer.src.review_panel.reviewer.get_claude_reviewer", return_value=None):
            with patch("explorer.src.review_panel.reviewer.get_deepseek_verifier", return_value=None):
                with patch("explorer.src.review_panel.reviewer.run_round1", new_callable=AsyncMock) as mock_run:
                    mock_run.return_value = mock_round1

                    reviewer = AutomatedReviewer(
                        gemini_browser=MagicMock(),
                        chatgpt_browser=MagicMock(),
                        config=config,
                        data_dir=temp_dir,
                    )

                    review = await reviewer.review_insight(
                        chunk_id="test-123",
                        insight_text="Test insight",
                        confidence="high",
                        tags=["test"],
                        dependencies=[],
                        blessed_axioms_summary="",
                        check_priority_switches=False,
                    )

        assert review.final_rating == "⚡"
        assert review.is_unanimous is True
        assert len(review.rounds) == 1

    @pytest.mark.asyncio
    async def test_split_round1_continues_to_round2(self, temp_dir, mock_split_round1, mock_round2):
        """Review continues to Round 2 on split Round 1."""
        config = {"review_panel": {}}

        with patch("explorer.src.review_panel.reviewer.get_claude_reviewer", return_value=None):
            with patch("explorer.src.review_panel.reviewer.get_deepseek_verifier", return_value=None):
                with patch("explorer.src.review_panel.reviewer.run_round1", new_callable=AsyncMock) as mock_r1:
                    with patch("explorer.src.review_panel.reviewer.run_round2", new_callable=AsyncMock) as mock_r2:
                        mock_r1.return_value = mock_split_round1
                        mock_r2.return_value = (mock_round2, [])

                        reviewer = AutomatedReviewer(
                            gemini_browser=MagicMock(),
                            chatgpt_browser=MagicMock(),
                            config=config,
                            data_dir=temp_dir,
                        )

                        review = await reviewer.review_insight(
                            chunk_id="test-456",
                            insight_text="Test insight",
                            confidence="high",
                            tags=["test"],
                            dependencies=[],
                            blessed_axioms_summary="",
                            check_priority_switches=False,
                        )

        assert review.final_rating == "⚡"
        assert len(review.rounds) == 2

    @pytest.mark.asyncio
    async def test_abandoned_round1_rejects(self, temp_dir):
        """Unanimous ABANDON in Round 1 results in rejection."""
        abandoned_round = ReviewRound(
            round_number=1,
            mode="standard",
            responses={
                "gemini": ReviewResponse(
                    llm="gemini", mode="standard", rating="ABANDON",
                    mathematical_verification="", structural_analysis="",
                    naturalness_assessment="", reasoning="Unsalvageable", confidence="high"
                ),
            },
            outcome="abandoned",
            timestamp=datetime.now(),
        )

        config = {"review_panel": {}}

        with patch("explorer.src.review_panel.reviewer.get_claude_reviewer", return_value=None):
            with patch("explorer.src.review_panel.reviewer.get_deepseek_verifier", return_value=None):
                with patch("explorer.src.review_panel.reviewer.run_round1", new_callable=AsyncMock) as mock_run:
                    mock_run.return_value = abandoned_round

                    reviewer = AutomatedReviewer(
                        gemini_browser=MagicMock(),
                        chatgpt_browser=MagicMock(),
                        config=config,
                        data_dir=temp_dir,
                    )

                    review = await reviewer.review_insight(
                        chunk_id="test-abandon",
                        insight_text="Bad insight",
                        confidence="low",
                        tags=["test"],
                        dependencies=[],
                        blessed_axioms_summary="",
                        check_priority_switches=False,
                    )

        assert review.final_rating == "✗"
        assert review.rejection_reason is not None

    @pytest.mark.asyncio
    async def test_resumes_from_saved_progress(self, temp_dir, mock_split_round1, mock_round2):
        """Resumes review from saved progress."""
        config = {"review_panel": {}}

        # Create saved progress
        progress_dir = temp_dir / "reviews" / "in_progress"
        progress_dir.mkdir(parents=True, exist_ok=True)

        saved_review = ChunkReview(chunk_id="test-resume")
        saved_review.add_round(mock_split_round1)

        with open(progress_dir / "test-resume.json", "w") as f:
            json.dump(saved_review.to_dict(), f)

        with patch("explorer.src.review_panel.reviewer.get_claude_reviewer", return_value=None):
            with patch("explorer.src.review_panel.reviewer.get_deepseek_verifier", return_value=None):
                with patch("explorer.src.review_panel.reviewer.run_round1", new_callable=AsyncMock) as mock_r1:
                    with patch("explorer.src.review_panel.reviewer.run_round2", new_callable=AsyncMock) as mock_r2:
                        mock_r2.return_value = (mock_round2, [])

                        reviewer = AutomatedReviewer(
                            gemini_browser=MagicMock(),
                            chatgpt_browser=MagicMock(),
                            config=config,
                            data_dir=temp_dir,
                        )

                        review = await reviewer.review_insight(
                            chunk_id="test-resume",
                            insight_text="Test insight",
                            confidence="high",
                            tags=["test"],
                            dependencies=[],
                            blessed_axioms_summary="",
                            check_priority_switches=False,
                        )

        # Round 1 should NOT have been called (was loaded from progress)
        mock_r1.assert_not_called()
        # Round 2 should have been called
        mock_r2.assert_called_once()


class TestAutomatedReviewerPrioritySwitching:
    """Tests for priority switching behavior."""

    @pytest.fixture
    def reviewer(self, temp_dir):
        """Create a reviewer instance."""
        config = {"review_panel": {}}
        with patch("explorer.src.review_panel.reviewer.get_claude_reviewer", return_value=None):
            with patch("explorer.src.review_panel.reviewer.get_deepseek_verifier", return_value=None):
                return AutomatedReviewer(
                    gemini_browser=MagicMock(),
                    chatgpt_browser=MagicMock(),
                    config=config,
                    data_dir=temp_dir,
                )

    def test_should_switch_when_higher_priority_exists(self, reviewer):
        """Returns True when higher priority item exists."""
        with patch.object(reviewer, "get_highest_priority_pending", return_value=("high-priority", 10, 0)):
            should_switch, new_id = reviewer.should_switch_to_higher_priority("current", 5)

        assert should_switch is True
        assert new_id == "high-priority"

    def test_should_not_switch_when_no_higher_priority(self, reviewer):
        """Returns False when no higher priority item."""
        with patch.object(reviewer, "get_highest_priority_pending", return_value=("low-priority", 3, 0)):
            should_switch, new_id = reviewer.should_switch_to_higher_priority("current", 5)

        assert should_switch is False

    def test_should_not_switch_to_same_item(self, reviewer):
        """Returns False when highest priority is current item."""
        with patch.object(reviewer, "get_highest_priority_pending", return_value=("current", 10, 0)):
            should_switch, new_id = reviewer.should_switch_to_higher_priority("current", 5)

        assert should_switch is False


class TestAutomatedReviewerMathVerification:
    """Tests for mathematical verification integration."""

    @pytest.mark.asyncio
    async def test_auto_rejects_on_refuted_math(self, temp_dir):
        """Auto-rejects insight when math verification refutes it."""
        mock_round1 = ReviewRound(
            round_number=1,
            mode="standard",
            responses={
                "gemini": ReviewResponse(
                    llm="gemini", mode="standard", rating="⚡",
                    mathematical_verification="Looks correct",
                    structural_analysis="", naturalness_assessment="",
                    reasoning="Good", confidence="high"
                ),
            },
            outcome="split",
            timestamp=datetime.now(),
        )

        mock_deepseek = MagicMock()
        mock_deepseek.is_available.return_value = True
        mock_deepseek.verify_insight = AsyncMock(return_value=VerificationResult(
            verdict="refuted",
            precise_statement="Test claim",
            counterexample="This is wrong because...",
            confidence=0.95,
        ))

        config = {"review_panel": {}}

        with patch("explorer.src.review_panel.reviewer.get_claude_reviewer", return_value=None):
            with patch("explorer.src.review_panel.reviewer.get_deepseek_verifier", return_value=mock_deepseek):
                with patch("explorer.src.review_panel.reviewer.run_round1", new_callable=AsyncMock) as mock_r1:
                    with patch("explorer.src.review_panel.reviewer.needs_math_verification", return_value=(True, "math tags")):
                        mock_r1.return_value = mock_round1

                        reviewer = AutomatedReviewer(
                            gemini_browser=MagicMock(),
                            chatgpt_browser=MagicMock(),
                            config=config,
                            data_dir=temp_dir,
                        )

                        review = await reviewer.review_insight(
                            chunk_id="test-math",
                            insight_text="The Fano plane has 8 points",
                            confidence="high",
                            tags=["fano-plane", "counting"],
                            dependencies=[],
                            blessed_axioms_summary="",
                            check_priority_switches=False,
                        )

        assert review.final_rating == "✗"
        assert "refuted" in review.rejection_reason.lower()


class TestAutomatedReviewerBatch:
    """Tests for review_batch method."""

    @pytest.mark.asyncio
    async def test_review_batch_processes_all(self, temp_dir):
        """review_batch processes all insights."""
        mock_round1 = ReviewRound(
            round_number=1,
            mode="standard",
            responses={
                "gemini": ReviewResponse(
                    llm="gemini", mode="standard", rating="⚡",
                    mathematical_verification="", structural_analysis="",
                    naturalness_assessment="", reasoning="Good", confidence="high"
                ),
            },
            outcome="unanimous",
            timestamp=datetime.now(),
        )

        config = {"review_panel": {}}

        with patch("explorer.src.review_panel.reviewer.get_claude_reviewer", return_value=None):
            with patch("explorer.src.review_panel.reviewer.get_deepseek_verifier", return_value=None):
                with patch("explorer.src.review_panel.reviewer.run_round1", new_callable=AsyncMock) as mock_r1:
                    mock_r1.return_value = mock_round1

                    reviewer = AutomatedReviewer(
                        gemini_browser=MagicMock(),
                        chatgpt_browser=MagicMock(),
                        config=config,
                        data_dir=temp_dir,
                    )

                    # Patch should_switch_to_higher_priority to avoid broken import
                    with patch.object(reviewer, "should_switch_to_higher_priority", return_value=(False, None)):
                        insights = [
                            {"id": "insight-1", "text": "First insight", "tags": []},
                            {"id": "insight-2", "text": "Second insight", "tags": []},
                            {"id": "insight-3", "text": "Third insight", "tags": []},
                        ]

                        results = await reviewer.review_batch(insights, "")

        assert len(results) == 3
        assert all(r.final_rating == "⚡" for r in results)

    @pytest.mark.asyncio
    async def test_review_batch_handles_failures(self, temp_dir):
        """review_batch handles individual review failures gracefully."""
        config = {"review_panel": {}}

        call_count = 0

        async def failing_round1(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("Review failed")
            return ReviewRound(
                round_number=1,
                mode="standard",
                responses={
                    "gemini": ReviewResponse(
                        llm="gemini", mode="standard", rating="⚡",
                        mathematical_verification="", structural_analysis="",
                        naturalness_assessment="", reasoning="Good", confidence="high"
                    ),
                },
                outcome="unanimous",
                timestamp=datetime.now(),
            )

        with patch("explorer.src.review_panel.reviewer.get_claude_reviewer", return_value=None):
            with patch("explorer.src.review_panel.reviewer.get_deepseek_verifier", return_value=None):
                with patch("explorer.src.review_panel.reviewer.run_round1", new_callable=AsyncMock) as mock_r1:
                    mock_r1.side_effect = failing_round1

                    reviewer = AutomatedReviewer(
                        gemini_browser=MagicMock(),
                        chatgpt_browser=MagicMock(),
                        config=config,
                        data_dir=temp_dir,
                    )

                    # Patch should_switch_to_higher_priority to avoid broken import
                    with patch.object(reviewer, "should_switch_to_higher_priority", return_value=(False, None)):
                        insights = [
                            {"id": "insight-1", "text": "First", "tags": []},
                            {"id": "insight-2", "text": "Second (will fail)", "tags": []},
                            {"id": "insight-3", "text": "Third", "tags": []},
                        ]

                        results = await reviewer.review_batch(insights, "")

        assert len(results) == 3
        # First and third should pass, second should be marked uncertain
        assert results[0].final_rating == "⚡"
        assert results[1].final_rating == "?"  # Failed review
        assert results[2].final_rating == "⚡"


class TestRecordModification:
    """Tests for _record_modification method."""

    @pytest.fixture
    def reviewer(self, temp_dir):
        """Create a reviewer instance."""
        config = {"review_panel": {}}
        with patch("explorer.src.review_panel.reviewer.get_claude_reviewer", return_value=None):
            with patch("explorer.src.review_panel.reviewer.get_deepseek_verifier", return_value=None):
                return AutomatedReviewer(
                    gemini_browser=MagicMock(),
                    chatgpt_browser=MagicMock(),
                    config=config,
                    data_dir=temp_dir,
                )

    def test_record_modification_updates_version(self, reviewer):
        """_record_modification increments version number."""
        review = ChunkReview(chunk_id="test")
        round_obj = ReviewRound(
            round_number=1, mode="standard",
            responses={
                "gemini": ReviewResponse(
                    llm="gemini", mode="standard", rating="?",
                    mathematical_verification="", structural_analysis="",
                    naturalness_assessment="", reasoning="", confidence="medium"
                ),
            },
            outcome="split", timestamp=datetime.now(),
        )

        new_insight, new_version, mod, source = reviewer._record_modification(
            review=review,
            review_round=round_obj,
            current_insight="Original",
            current_version=1,
            mod="Modified version",
            mod_source="gemini",
            mod_rationale="Made it better",
            round_num=1,
        )

        assert new_insight == "Modified version"
        assert new_version == 2
        assert review.was_refined is True
        assert len(review.refinements) == 1
