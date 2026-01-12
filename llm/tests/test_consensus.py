"""Tests for LLM consensus reviewer."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from llm.src.consensus import ConsensusReviewer
from llm.src.consensus.prompts import build_review_round1_prompt, build_review_round2_prompt
from llm.src.consensus.response_parser import parse_review_response
from llm.src.client import LLMClient
from llm.src.models import LLMResponse, ConsensusResult, ReviewResponse


@pytest.fixture
def mock_client():
    """Create a mock LLMClient."""
    client = MagicMock(spec=LLMClient)
    client.send = AsyncMock()
    client.send_parallel = AsyncMock()
    client.get_available_backends = AsyncMock()
    return client


@pytest.fixture
def reviewer(mock_client):
    """Create a ConsensusReviewer with mock client."""
    return ConsensusReviewer(mock_client)


@pytest.fixture
def sample_round1_response():
    """Sample structured response for Round 1."""
    return """RATING: bless
MATHEMATICAL_VERIFICATION: The number 84 is indeed significant in yoga traditions.
STRUCTURAL_ANALYSIS: This represents a deep structural connection.
NATURALNESS: This feels discovered, not invented.
REASONING: The insight correctly identifies the relationship between 84 asanas and other occurrences of 84 in yoga.
CONFIDENCE: high"""


@pytest.fixture
def sample_round2_response():
    """Sample structured response for Round 2."""
    return """RATING: bless
NEW_INFORMATION: The other reviewer's mathematical verification confirms the claim.
CHANGED_MIND: no
REASONING: My assessment remains the same after seeing other reviews.
CONFIDENCE: high"""


class TestConsensusReviewerInit:
    """Tests for ConsensusReviewer initialization."""

    def test_init_with_client(self, mock_client):
        reviewer = ConsensusReviewer(mock_client)

        assert reviewer.client == mock_client
        assert reviewer.config == {}

    def test_init_with_config(self, mock_client):
        config = {"some": "config"}
        reviewer = ConsensusReviewer(mock_client, config=config)

        assert reviewer.config == config


class TestConsensusReviewerReview:
    """Tests for the main review method."""

    @pytest.mark.asyncio
    async def test_review_insufficient_backends(self, reviewer, mock_client):
        """review() returns error when fewer than 2 backends available."""
        mock_client.get_available_backends.return_value = ["claude"]

        result = await reviewer.review("Test insight")

        assert result.success is False
        assert result.final_rating == "uncertain"
        assert result.is_disputed is True
        assert "Need at least 2 backends" in result.rounds[0]["error"]

    @pytest.mark.asyncio
    async def test_review_unanimous_round1(self, reviewer, mock_client, sample_round1_response):
        """review() exits early on unanimous Round 1."""
        mock_client.get_available_backends.return_value = ["gemini", "claude"]

        # All reviewers rate "bless"
        mock_client.send_parallel.return_value = {
            "gemini": LLMResponse(success=True, text=sample_round1_response),
            "claude": LLMResponse(success=True, text=sample_round1_response),
        }

        result = await reviewer.review("Test insight")

        assert result.success is True
        assert result.final_rating == "bless"
        assert result.is_unanimous is True
        assert result.is_disputed is False
        assert len(result.rounds) == 1  # Only Round 1

    @pytest.mark.asyncio
    async def test_review_unanimous_round2(self, reviewer, mock_client):
        """review() exits early on unanimous Round 2."""
        mock_client.get_available_backends.return_value = ["gemini", "claude", "chatgpt"]

        # Round 1: mixed ratings
        round1_responses = {
            "gemini": LLMResponse(success=True, text="RATING: bless\nREASONING: Valid\nCONFIDENCE: high"),
            "claude": LLMResponse(success=True, text="RATING: uncertain\nREASONING: Needs work\nCONFIDENCE: medium"),
            "chatgpt": LLMResponse(success=True, text="RATING: bless\nREASONING: Good\nCONFIDENCE: high"),
        }

        # Round 2: all agree on bless
        round2_responses = {
            "gemini": LLMResponse(success=True, text="RATING: bless\nCHANGED_MIND: no\nREASONING: Still valid\nCONFIDENCE: high"),
            "claude": LLMResponse(success=True, text="RATING: bless\nCHANGED_MIND: yes\nREASONING: Convinced\nCONFIDENCE: high"),
            "chatgpt": LLMResponse(success=True, text="RATING: bless\nCHANGED_MIND: no\nREASONING: Confirmed\nCONFIDENCE: high"),
        }

        mock_client.send_parallel.side_effect = [round1_responses, round2_responses]

        result = await reviewer.review("Test insight")

        assert result.success is True
        assert result.final_rating == "bless"
        assert result.is_unanimous is True
        assert len(result.rounds) == 2  # Rounds 1 and 2

    @pytest.mark.asyncio
    async def test_review_goes_to_round3(self, reviewer, mock_client):
        """review() goes to Round 3 when Round 2 is not unanimous."""
        mock_client.get_available_backends.return_value = ["gemini", "claude", "chatgpt"]

        # Round 1: mixed
        round1_responses = {
            "gemini": LLMResponse(success=True, text="RATING: bless\nREASONING: Valid\nCONFIDENCE: high"),
            "claude": LLMResponse(success=True, text="RATING: reject\nREASONING: Flawed\nCONFIDENCE: high"),
            "chatgpt": LLMResponse(success=True, text="RATING: uncertain\nREASONING: Maybe\nCONFIDENCE: medium"),
        }

        # Round 2: still not unanimous
        round2_responses = {
            "gemini": LLMResponse(success=True, text="RATING: bless\nREASONING: Still valid\nCONFIDENCE: high"),
            "claude": LLMResponse(success=True, text="RATING: reject\nREASONING: Still flawed\nCONFIDENCE: high"),
            "chatgpt": LLMResponse(success=True, text="RATING: bless\nREASONING: Convinced by gemini\nCONFIDENCE: high"),
        }

        mock_client.send_parallel.side_effect = [round1_responses, round2_responses]

        result = await reviewer.review("Test insight")

        assert result.success is True
        assert result.final_rating == "bless"  # 2 out of 3
        assert result.is_unanimous is True  # Majority = no dispute
        assert result.is_disputed is False
        assert len(result.rounds) == 3

    @pytest.mark.asyncio
    async def test_review_disputed_no_majority(self, reviewer, mock_client):
        """review() marks as disputed when no majority exists."""
        mock_client.get_available_backends.return_value = ["gemini", "claude", "chatgpt"]

        # All three different ratings
        round1_responses = {
            "gemini": LLMResponse(success=True, text="RATING: bless\nREASONING: Valid\nCONFIDENCE: high"),
            "claude": LLMResponse(success=True, text="RATING: reject\nREASONING: Flawed\nCONFIDENCE: high"),
            "chatgpt": LLMResponse(success=True, text="RATING: uncertain\nREASONING: Maybe\nCONFIDENCE: medium"),
        }

        # Round 2: still all different
        round2_responses = {
            "gemini": LLMResponse(success=True, text="RATING: bless\nREASONING: Still valid\nCONFIDENCE: high"),
            "claude": LLMResponse(success=True, text="RATING: reject\nREASONING: Still flawed\nCONFIDENCE: high"),
            "chatgpt": LLMResponse(success=True, text="RATING: uncertain\nREASONING: Still unsure\nCONFIDENCE: medium"),
        }

        mock_client.send_parallel.side_effect = [round1_responses, round2_responses]

        result = await reviewer.review("Test insight")

        assert result.success is True
        assert result.final_rating == "uncertain"  # Default when no majority
        assert result.is_disputed is True

    @pytest.mark.asyncio
    async def test_review_handles_error_responses(self, reviewer, mock_client):
        """review() handles error responses from backends."""
        mock_client.get_available_backends.return_value = ["gemini", "claude"]

        # One success, one error
        round1_responses = {
            "gemini": LLMResponse(success=True, text="RATING: bless\nREASONING: Valid\nCONFIDENCE: high"),
            "claude": LLMResponse(success=False, error="api_error", message="Service unavailable"),
        }

        mock_client.send_parallel.return_value = round1_responses

        result = await reviewer.review("Test insight")

        # Should still work with one valid response
        assert len(result.rounds) >= 1


class TestBuildRound1Prompt:
    """Tests for build_review_round1_prompt function."""

    def test_includes_insight_text(self):
        prompt = build_review_round1_prompt(
            "Test insight about math",
            tags=[], context="", confidence="medium", dependencies=[]
        )

        assert "Test insight about math" in prompt

    def test_includes_tags(self):
        prompt = build_review_round1_prompt(
            "Test", tags=["yoga", "numbers"], context="", confidence="medium", dependencies=[]
        )

        assert "yoga" in prompt
        assert "numbers" in prompt

    def test_includes_confidence(self):
        prompt = build_review_round1_prompt(
            "Test", tags=[], context="", confidence="high", dependencies=[]
        )

        assert "high" in prompt

    def test_includes_context(self):
        prompt = build_review_round1_prompt(
            "Test", tags=[], context="Some blessed axioms here", confidence="medium", dependencies=[]
        )

        assert "Some blessed axioms here" in prompt

    def test_includes_dependencies(self):
        prompt = build_review_round1_prompt(
            "Test", tags=[], context="", confidence="medium", dependencies=["insight-1", "insight-2"]
        )

        assert "insight-1" in prompt
        assert "insight-2" in prompt

    def test_includes_rating_instructions(self):
        prompt = build_review_round1_prompt(
            "Test", tags=[], context="", confidence="medium", dependencies=[]
        )

        assert "bless" in prompt
        assert "uncertain" in prompt
        assert "reject" in prompt

    def test_includes_format_instructions(self):
        prompt = build_review_round1_prompt(
            "Test", tags=[], context="", confidence="medium", dependencies=[]
        )

        assert "RATING:" in prompt
        assert "REASONING:" in prompt
        assert "CONFIDENCE:" in prompt
        assert "MATHEMATICAL_VERIFICATION:" in prompt
        assert "STRUCTURAL_ANALYSIS:" in prompt
        assert "NATURALNESS:" in prompt


class TestBuildRound2Prompt:
    """Tests for build_review_round2_prompt function."""

    def test_includes_insight_text(self):
        round1 = {
            "gemini": ReviewResponse(
                llm="gemini", mode="standard", rating="bless",
                reasoning="Valid", confidence="high"
            )
        }

        prompt = build_review_round2_prompt("Test insight", round1, "")

        assert "Test insight" in prompt

    def test_includes_round1_responses(self):
        round1 = {
            "gemini": ReviewResponse(
                llm="gemini", mode="standard", rating="bless",
                reasoning="This is valid", confidence="high"
            ),
            "claude": ReviewResponse(
                llm="claude", mode="standard", rating="uncertain",
                reasoning="Needs more work", confidence="medium"
            ),
        }

        prompt = build_review_round2_prompt("Test", round1, "")

        assert "GEMINI" in prompt
        assert "bless" in prompt
        assert "CLAUDE" in prompt
        assert "uncertain" in prompt

    def test_includes_round2_format(self):
        round1 = {
            "gemini": ReviewResponse(
                llm="gemini", mode="standard", rating="bless",
                reasoning="Valid", confidence="high"
            )
        }

        prompt = build_review_round2_prompt("Test", round1, "")

        assert "NEW_INFORMATION:" in prompt
        assert "CHANGED_MIND:" in prompt


class TestParseReviewResponse:
    """Tests for parse_review_response function."""

    def test_parses_complete_response(self, sample_round1_response):
        result = parse_review_response("gemini", sample_round1_response, "standard")

        assert result.llm == "gemini"
        assert result.mode == "standard"
        assert result.rating == "bless"
        assert "84" in result.mathematical_verification
        assert result.confidence == "high"

    def test_parses_rating_bless(self):
        result = parse_review_response(
            "gemini", "RATING: bless\nREASONING: Valid\nCONFIDENCE: high", "standard"
        )
        assert result.rating == "bless"

    def test_parses_rating_reject(self):
        result = parse_review_response(
            "gemini", "RATING: reject\nREASONING: Invalid\nCONFIDENCE: high", "standard"
        )
        assert result.rating == "reject"

    def test_parses_rating_uncertain(self):
        result = parse_review_response(
            "gemini", "RATING: uncertain\nREASONING: Maybe\nCONFIDENCE: medium", "standard"
        )
        assert result.rating == "uncertain"

    def test_defaults_to_uncertain(self):
        """Defaults to uncertain when rating cannot be parsed."""
        result = parse_review_response(
            "gemini", "Some unstructured response", "standard"
        )
        assert result.rating == "uncertain"

    def test_parses_reasoning(self):
        result = parse_review_response(
            "gemini", "RATING: bless\nREASONING: This is the reason why.\nCONFIDENCE: high", "standard"
        )
        assert "This is the reason" in result.reasoning

    def test_parses_confidence(self):
        result = parse_review_response(
            "gemini", "RATING: bless\nREASONING: Valid\nCONFIDENCE: low", "standard"
        )
        assert result.confidence == "low"

    def test_truncates_long_fields(self):
        long_text = "A" * 1000
        result = parse_review_response(
            "gemini", f"RATING: bless\nREASONING: {long_text}\nCONFIDENCE: high", "standard"
        )
        assert len(result.reasoning) <= 500

    def test_handles_multiline_reasoning(self):
        response = """RATING: bless
REASONING: First line of reasoning.
Second line of reasoning.
Third line.
CONFIDENCE: high"""

        result = parse_review_response("gemini", response, "standard")
        assert "First line" in result.reasoning
        assert "Second line" in result.reasoning


class TestQuickCheck:
    """Tests for quick_check method."""

    @pytest.mark.asyncio
    async def test_quick_check_no_backends(self, reviewer, mock_client):
        """quick_check returns uncertain when no backends available."""
        mock_client.get_available_backends.return_value = []

        rating, reason = await reviewer.quick_check("Test")

        assert rating == "uncertain"
        assert "No backends available" in reason

    @pytest.mark.asyncio
    async def test_quick_check_prefers_claude(self, reviewer, mock_client):
        """quick_check prefers Claude when available."""
        mock_client.get_available_backends.return_value = ["gemini", "claude"]
        mock_client.send.return_value = LLMResponse(
            success=True,
            text="RATING: bless\nREASON: Looks good"
        )

        await reviewer.quick_check("Test")

        mock_client.send.assert_called_once()
        call_args = mock_client.send.call_args
        assert call_args[0][0] == "claude"

    @pytest.mark.asyncio
    async def test_quick_check_uses_first_available(self, reviewer, mock_client):
        """quick_check uses first available when Claude not available."""
        mock_client.get_available_backends.return_value = ["gemini"]
        mock_client.send.return_value = LLMResponse(
            success=True,
            text="RATING: bless\nREASON: Valid"
        )

        await reviewer.quick_check("Test")

        mock_client.send.assert_called_once()
        call_args = mock_client.send.call_args
        assert call_args[0][0] == "gemini"

    @pytest.mark.asyncio
    async def test_quick_check_parses_bless(self, reviewer, mock_client):
        """quick_check parses bless rating."""
        mock_client.get_available_backends.return_value = ["claude"]
        mock_client.send.return_value = LLMResponse(
            success=True,
            text="RATING: bless\nREASON: This is valid"
        )

        rating, reason = await reviewer.quick_check("Test")

        assert rating == "bless"
        assert "valid" in reason.lower()

    @pytest.mark.asyncio
    async def test_quick_check_parses_reject(self, reviewer, mock_client):
        """quick_check parses reject rating."""
        mock_client.get_available_backends.return_value = ["claude"]
        mock_client.send.return_value = LLMResponse(
            success=True,
            text="RATING: reject\nREASON: This is flawed"
        )

        rating, reason = await reviewer.quick_check("Test")

        assert rating == "reject"

    @pytest.mark.asyncio
    async def test_quick_check_handles_error(self, reviewer, mock_client):
        """quick_check handles error response."""
        mock_client.get_available_backends.return_value = ["claude"]
        mock_client.send.return_value = LLMResponse(
            success=False,
            error="api_error",
            message="Service unavailable"
        )

        rating, reason = await reviewer.quick_check("Test")

        assert rating == "uncertain"
        assert "Error" in reason


class TestRunRounds:
    """Tests for round execution methods (via ReviewTask)."""

    @pytest.mark.asyncio
    async def test_run_round1_sends_to_all_backends(self, reviewer, mock_client):
        """_run_round1 sends to all available backends."""
        backends = ["gemini", "claude", "chatgpt"]

        mock_client.send_parallel.return_value = {
            b: LLMResponse(success=True, text="RATING: bless\nREASONING: Valid\nCONFIDENCE: high")
            for b in backends
        }

        result = await reviewer._review_task._run_round1(
            "Test", ["tag"], "", "medium", [], backends
        )

        mock_client.send_parallel.assert_called_once()
        call_args = mock_client.send_parallel.call_args
        prompts = call_args[0][0]
        assert set(prompts.keys()) == set(backends)

    @pytest.mark.asyncio
    async def test_run_round2_uses_deep_mode(self, reviewer, mock_client):
        """_run_round2 uses deep mode when enabled."""
        backends = ["gemini", "claude"]
        round1 = {
            "gemini": ReviewResponse(
                llm="gemini", mode="standard", rating="bless",
                reasoning="Valid", confidence="high"
            )
        }

        mock_client.send_parallel.return_value = {
            b: LLMResponse(success=True, text="RATING: bless\nREASONING: Valid\nCONFIDENCE: high", deep_mode_used=True)
            for b in backends
        }

        await reviewer._review_task._run_round2("Test", round1, "", backends, use_deep_mode=True)

        mock_client.send_parallel.assert_called_once()
        call_kwargs = mock_client.send_parallel.call_args[1]
        assert call_kwargs["deep_mode"] is True

    def test_run_round3_finds_majority(self, reviewer):
        """_run_round3 correctly identifies majority rating."""
        round2 = {
            "gemini": ReviewResponse(llm="gemini", mode="deep", rating="bless", reasoning="", confidence="high"),
            "claude": ReviewResponse(llm="claude", mode="deep", rating="bless", reasoning="", confidence="high"),
            "chatgpt": ReviewResponse(llm="chatgpt", mode="deep", rating="reject", reasoning="", confidence="high"),
        }

        final_rating, is_disputed = reviewer._review_task._run_round3(round2)

        assert final_rating == "bless"
        assert is_disputed is False

    def test_run_round3_no_majority(self, reviewer):
        """_run_round3 returns uncertain when no majority."""
        round2 = {
            "gemini": ReviewResponse(llm="gemini", mode="deep", rating="bless", reasoning="", confidence="high"),
            "claude": ReviewResponse(llm="claude", mode="deep", rating="reject", reasoning="", confidence="high"),
            "chatgpt": ReviewResponse(llm="chatgpt", mode="deep", rating="uncertain", reasoning="", confidence="high"),
        }

        final_rating, is_disputed = reviewer._review_task._run_round3(round2)

        assert final_rating == "uncertain"
        assert is_disputed is True


class TestConsensusReviewerTiming:
    """Tests for timing and duration tracking."""

    @pytest.mark.asyncio
    async def test_tracks_review_duration(self, reviewer, mock_client, sample_round1_response):
        """review() tracks duration in result."""
        mock_client.get_available_backends.return_value = ["gemini", "claude"]
        mock_client.send_parallel.return_value = {
            "gemini": LLMResponse(success=True, text=sample_round1_response),
            "claude": LLMResponse(success=True, text=sample_round1_response),
        }

        result = await reviewer.review("Test")

        # Duration is tracked (may be 0.0 with fast mocks but field should exist)
        assert result.review_duration_seconds >= 0
        assert isinstance(result.review_duration_seconds, float)
