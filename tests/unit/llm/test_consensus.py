"""Tests for llm.src.consensus.engine and llm.src.consensus.parsing."""

from collections import Counter
from unittest.mock import AsyncMock

import pytest

from shared.errors import InsufficientResponsesError
from shared.models import (
    ConsensusConfig,
    ConsensusResult,
    ConsensusTaskInterface,
    EvaluationCriterion,
    LLMClientInterface,
    LLMResponse,
    ParsedResponse,
    RoundResult,
    ValidatedResponse,
    Verdict,
)
from llm.src.consensus.engine import ConsensusEngine
from llm.src.consensus.parsing import normalize_verdict, parse_review_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_response(text: str, backend: str = "gemini") -> LLMResponse:
    """Create an LLMResponse with sensible defaults."""
    return LLMResponse(
        success=True,
        text=text,
        backend=backend,
        model=f"{backend}-model",
        token_usage=None,
        error=None,
    )


def _make_config(
    backends: list[str] | None = None,
    **overrides: object,
) -> ConsensusConfig:
    """Create a ConsensusConfig with sensible defaults for testing."""
    defaults = dict(
        backends=backends or ["gemini", "chatgpt", "claude"],
        max_rounds=4,
        min_valid_responses=2,
        max_retries_per_backend=2,
        convergence_threshold=0.7,
        decision_method="majority",
        minimum_agreement=0.66,
    )
    defaults.update(overrides)
    return ConsensusConfig(**defaults)


class StubTask(ConsensusTaskInterface):
    """Simple task implementation for testing the engine."""

    def __init__(
        self,
        criteria: list[EvaluationCriterion] | None = None,
        parse_fn=None,
    ):
        self._criteria = criteria or [
            EvaluationCriterion(name="rigor", description="Logical soundness", weight=1.0),
            EvaluationCriterion(name="depth", description="Structural depth", weight=0.8),
        ]
        self._parse_fn = parse_fn

    def get_prompt(
        self,
        round_num: int,
        prior_rounds: list[RoundResult],
        backend: str,
    ) -> str:
        if round_num == 1:
            return f"Evaluate this insight. Backend: {backend}"
        return f"Round {round_num} for {backend}. Prior rounds: {len(prior_rounds)}"

    def parse_response(self, text: str) -> ParsedResponse:
        if self._parse_fn:
            return self._parse_fn(text)
        criteria_names = [c.name for c in self._criteria]
        return parse_review_response(text, criteria_names)

    def get_evaluation_criteria(self) -> list[EvaluationCriterion]:
        return self._criteria


def _mock_client(responses: dict[str, list[str]]) -> LLMClientInterface:
    """Create a mock LLMClient that returns pre-configured responses.

    Args:
        responses: Maps backend name to a list of response texts (one per round).
    """
    call_counts: dict[str, int] = {}

    async def send_side_effect(backend: str, prompt: str, **kwargs: object) -> LLMResponse:
        idx = call_counts.get(backend, 0)
        call_counts[backend] = idx + 1
        texts = responses.get(backend, [])
        text = texts[idx] if idx < len(texts) else texts[-1] if texts else ""
        return _make_llm_response(text, backend)

    client = AsyncMock(spec=LLMClientInterface)
    client.send = AsyncMock(side_effect=send_side_effect)
    return client


# ---------------------------------------------------------------------------
# Engine Tests
# ---------------------------------------------------------------------------


class TestConsensusEngine:
    """Tests from Design Spec Section 7.7."""

    # -- test_unanimous_accept_round1 ---------------------------------------

    async def test_unanimous_accept_round1(self) -> None:
        """All backends accept -> attested in round 1."""
        accept_json = '```json\n{"verdict": "accept", "scores": {"rigor": 8, "depth": 7}, "reasoning": "Solid."}\n```'
        client = _mock_client({
            "gemini": [accept_json],
            "chatgpt": [accept_json],
            "claude": [accept_json],
        })
        config = _make_config(decision_method="unanimous")
        engine = ConsensusEngine(client, config)

        result = await engine.run(StubTask())

        assert result.verdict == Verdict.ACCEPT
        assert result.rounds_completed == 1
        assert result.confidence == 1.0
        assert len(result.round_history) == 1
        assert result.round_history[0].is_converged is True

    # -- test_unanimous_reject_round1 ---------------------------------------

    async def test_unanimous_reject_round1(self) -> None:
        """All backends reject -> discarded in round 1."""
        reject_json = '```json\n{"verdict": "reject", "scores": {"rigor": 2, "depth": 1}, "reasoning": "Flawed."}\n```'
        client = _mock_client({
            "gemini": [reject_json],
            "chatgpt": [reject_json],
            "claude": [reject_json],
        })
        config = _make_config(decision_method="unanimous")
        engine = ConsensusEngine(client, config)

        result = await engine.run(StubTask())

        assert result.verdict == Verdict.REJECT
        assert result.rounds_completed == 1
        assert result.confidence == 1.0

    # -- test_split_proceeds_to_round2 --------------------------------------

    async def test_split_proceeds_to_round2(self) -> None:
        """Mixed verdicts -> round 2 with prior context."""
        accept_json = '```json\n{"verdict": "accept", "scores": {"rigor": 8}, "reasoning": "Good."}\n```'
        reject_json = '```json\n{"verdict": "reject", "scores": {"rigor": 3}, "reasoning": "Bad."}\n```'
        uncertain_json = '```json\n{"verdict": "uncertain", "scores": {"rigor": 5}, "reasoning": "Hmm."}\n```'

        # Round 1: split. Round 2: all accept (convergence).
        client = _mock_client({
            "gemini": [accept_json, accept_json],
            "chatgpt": [reject_json, accept_json],
            "claude": [uncertain_json, accept_json],
        })
        config = _make_config(decision_method="unanimous")
        engine = ConsensusEngine(client, config)

        result = await engine.run(StubTask())

        assert result.rounds_completed == 2
        assert result.verdict == Verdict.ACCEPT
        assert len(result.round_history) == 2
        assert result.round_history[0].is_converged is False
        assert result.round_history[1].is_converged is True

    # -- test_majority_converges --------------------------------------------

    async def test_majority_converges(self) -> None:
        """2/3 agree in round 1 with majority method -> converged."""
        accept_json = '```json\n{"verdict": "accept", "scores": {"rigor": 8}, "reasoning": "Good."}\n```'
        reject_json = '```json\n{"verdict": "reject", "scores": {"rigor": 3}, "reasoning": "Bad."}\n```'

        client = _mock_client({
            "gemini": [accept_json],
            "chatgpt": [accept_json],
            "claude": [reject_json],
        })
        config = _make_config(decision_method="majority", minimum_agreement=0.66)
        engine = ConsensusEngine(client, config)

        result = await engine.run(StubTask())

        assert result.verdict == Verdict.ACCEPT
        assert result.rounds_completed == 1
        # 2/3 ~= 0.667
        assert result.confidence >= 0.66

    # -- test_error_response_excluded ---------------------------------------

    async def test_error_response_excluded(self) -> None:
        """Backend returns error/unparseable -> not counted as vote."""
        accept_json = '```json\n{"verdict": "accept", "scores": {"rigor": 8}, "reasoning": "Good."}\n```'

        client = _mock_client({
            "gemini": [accept_json],
            "chatgpt": [accept_json],
            "claude": ["I apologize, I cannot process this request."],
        })
        config = _make_config(decision_method="majority", minimum_agreement=0.5)
        engine = ConsensusEngine(client, config)

        result = await engine.run(StubTask())

        assert result.verdict == Verdict.ACCEPT
        # Only 2 valid votes
        assert result.valid_vote_count == 2

    # -- test_insufficient_responses_raises ---------------------------------

    async def test_insufficient_responses_raises(self) -> None:
        """All retries exhausted -> InsufficientResponsesError."""
        client = _mock_client({
            "gemini": ["I apologize, I cannot help."],
            "chatgpt": ["I apologize, I cannot help."],
            "claude": ["I apologize, I cannot help."],
        })
        config = _make_config(min_valid_responses=2, max_retries_per_backend=1)
        engine = ConsensusEngine(client, config)

        with pytest.raises(InsufficientResponsesError):
            await engine.run(StubTask())

    # -- test_insufficient_responses_retries --------------------------------

    async def test_insufficient_responses_retries(self) -> None:
        """Too many failures in first attempt -> retried, second attempt succeeds."""
        accept_json = '```json\n{"verdict": "accept", "scores": {"rigor": 8}, "reasoning": "OK."}\n```'

        # First call: error. Second call (retry): valid.
        client = _mock_client({
            "gemini": ["I apologize", accept_json],
            "chatgpt": ["I apologize", accept_json],
            "claude": [accept_json, accept_json],
        })
        config = _make_config(
            min_valid_responses=2,
            max_retries_per_backend=2,
            decision_method="majority",
            minimum_agreement=0.5,
        )
        engine = ConsensusEngine(client, config)

        result = await engine.run(StubTask())

        assert result.verdict == Verdict.ACCEPT

    # -- test_score_aggregation ---------------------------------------------

    async def test_score_aggregation(self) -> None:
        """Scores averaged correctly across valid responses."""
        r1 = '```json\n{"verdict": "accept", "scores": {"rigor": 8, "depth": 6}, "reasoning": "A."}\n```'
        r2 = '```json\n{"verdict": "accept", "scores": {"rigor": 6, "depth": 8}, "reasoning": "B."}\n```'
        r3 = '```json\n{"verdict": "accept", "scores": {"rigor": 10, "depth": 4}, "reasoning": "C."}\n```'

        client = _mock_client({
            "gemini": [r1],
            "chatgpt": [r2],
            "claude": [r3],
        })
        config = _make_config(decision_method="unanimous")
        engine = ConsensusEngine(client, config)

        result = await engine.run(StubTask())

        assert result.scores["rigor"] == pytest.approx(8.0, abs=0.01)
        assert result.scores["depth"] == pytest.approx(6.0, abs=0.01)

    # -- test_max_rounds_reached --------------------------------------------

    async def test_max_rounds_reached(self) -> None:
        """4 rounds without convergence -> uncertain verdict."""
        accept_json = '```json\n{"verdict": "accept", "scores": {"rigor": 8}, "reasoning": "Yes."}\n```'
        reject_json = '```json\n{"verdict": "reject", "scores": {"rigor": 3}, "reasoning": "No."}\n```'

        # Permanent split: never converges under unanimous
        client = _mock_client({
            "gemini": [accept_json] * 4,
            "chatgpt": [reject_json] * 4,
            "claude": [accept_json] * 4,
        })
        config = _make_config(decision_method="unanimous", max_rounds=4)
        engine = ConsensusEngine(client, config)

        result = await engine.run(StubTask())

        assert result.verdict == Verdict.UNCERTAIN
        assert result.rounds_completed == 4

    # -- test_variable_backend_count ----------------------------------------

    async def test_variable_backend_count(self) -> None:
        """Works with 2, 3, 4, 5 backends."""
        accept_json = '```json\n{"verdict": "accept", "scores": {"rigor": 7}, "reasoning": "OK."}\n```'

        for n in (2, 3, 4, 5):
            names = [f"backend_{i}" for i in range(n)]
            responses = {name: [accept_json] for name in names}
            client = _mock_client(responses)
            config = _make_config(
                backends=names,
                decision_method="unanimous",
                min_valid_responses=2,
            )
            engine = ConsensusEngine(client, config)
            result = await engine.run(StubTask())

            assert result.verdict == Verdict.ACCEPT
            assert result.valid_vote_count == n

    # -- test_supermajority -------------------------------------------------

    async def test_supermajority_converges(self) -> None:
        """Supermajority method: 2/3 >= 0.66 -> converges."""
        accept_json = '```json\n{"verdict": "accept", "scores": {"rigor": 8}, "reasoning": "Good."}\n```'
        reject_json = '```json\n{"verdict": "reject", "scores": {"rigor": 2}, "reasoning": "Bad."}\n```'

        client = _mock_client({
            "gemini": [accept_json],
            "chatgpt": [accept_json],
            "claude": [reject_json],
        })
        config = _make_config(decision_method="supermajority")
        engine = ConsensusEngine(client, config)

        result = await engine.run(StubTask())

        assert result.verdict == Verdict.ACCEPT
        assert result.rounds_completed == 1

    # -- test_backends_override ---------------------------------------------

    async def test_backends_override_in_run(self) -> None:
        """Passing backends to run() overrides config.backends."""
        accept_json = '```json\n{"verdict": "accept", "scores": {"rigor": 8}, "reasoning": "OK."}\n```'

        client = _mock_client({
            "gemini": [accept_json],
            "chatgpt": [accept_json],
        })
        config = _make_config(
            backends=["gemini", "chatgpt", "claude"],
            decision_method="unanimous",
            min_valid_responses=2,
        )
        engine = ConsensusEngine(client, config)

        result = await engine.run(StubTask(), backends=["gemini", "chatgpt"])

        assert result.verdict == Verdict.ACCEPT
        assert result.valid_vote_count == 2


# ---------------------------------------------------------------------------
# Parsing Tests
# ---------------------------------------------------------------------------


class TestParsing:
    """Tests for parse_review_response and normalize_verdict."""

    # -- test_parse_json_response -------------------------------------------

    def test_parse_json_response(self) -> None:
        """JSON block parsed correctly."""
        text = '```json\n{"verdict": "accept", "scores": {"rigor": 8, "depth": 7}, "reasoning": "Solid work."}\n```'
        result = parse_review_response(text, ["rigor", "depth"])

        assert result.is_valid is True
        assert result.verdict == Verdict.ACCEPT
        assert result.scores == {"rigor": 8.0, "depth": 7.0}
        assert result.reasoning == "Solid work."

    # -- test_parse_regex_response ------------------------------------------

    def test_parse_regex_response(self) -> None:
        """Symbol/text verdict + scores parsed via regex fallback."""
        text = "Verdict: ACCEPT\nrigor: 8/10\ndepth: 7/10\nThis is solid reasoning."
        result = parse_review_response(text, ["rigor", "depth"])

        assert result.is_valid is True
        assert result.verdict == Verdict.ACCEPT
        assert result.scores["rigor"] == pytest.approx(8.0)
        assert result.scores["depth"] == pytest.approx(7.0)

    # -- test_parse_empty_response ------------------------------------------

    def test_parse_empty_response(self) -> None:
        """Empty string -> is_valid=False."""
        result = parse_review_response("", ["rigor"])
        assert result.is_valid is False

    def test_parse_whitespace_response(self) -> None:
        """Whitespace-only string -> is_valid=False."""
        result = parse_review_response("   \n\t  ", ["rigor"])
        assert result.is_valid is False

    # -- test_parse_error_message -------------------------------------------

    def test_parse_error_message(self) -> None:
        """'I apologize...' -> is_valid=False."""
        result = parse_review_response(
            "I apologize, but I cannot evaluate this content.", ["rigor"]
        )
        assert result.is_valid is False

    def test_parse_i_cannot_message(self) -> None:
        """'I cannot...' -> is_valid=False."""
        result = parse_review_response(
            "I cannot process this request at this time.", ["rigor"]
        )
        assert result.is_valid is False

    # -- test_parse_case_insensitive ----------------------------------------

    def test_parse_case_insensitive(self) -> None:
        """'ACCEPT', 'accept', 'Accept' all -> Verdict.ACCEPT."""
        for variant in ["ACCEPT", "accept", "Accept", "aCcEpT"]:
            text = f'```json\n{{"verdict": "{variant}", "scores": {{"rigor": 5}}, "reasoning": "OK."}}\n```'
            result = parse_review_response(text, ["rigor"])
            assert result.is_valid is True
            assert result.verdict == Verdict.ACCEPT, f"Failed for variant: {variant}"

    # -- test_normalize_verdict ---------------------------------------------

    def test_normalize_verdict_accept_variants(self) -> None:
        """All accept-like strings map to ACCEPT."""
        for raw in ["accept", "ACCEPT", "Accept", "approve", "bless"]:
            assert normalize_verdict(raw) == Verdict.ACCEPT, f"Failed for: {raw}"

    def test_normalize_verdict_reject_variants(self) -> None:
        """All reject-like strings map to REJECT."""
        for raw in ["reject", "REJECT", "Reject", "discard"]:
            assert normalize_verdict(raw) == Verdict.REJECT, f"Failed for: {raw}"

    def test_normalize_verdict_uncertain_variants(self) -> None:
        """All uncertain-like strings map to UNCERTAIN."""
        for raw in ["uncertain", "UNCERTAIN", "Uncertain", "unsure"]:
            assert normalize_verdict(raw) == Verdict.UNCERTAIN, f"Failed for: {raw}"

    def test_normalize_verdict_unknown(self) -> None:
        """Unrecognized string returns None."""
        assert normalize_verdict("gobbledygook") is None

    def test_normalize_verdict_emoji(self) -> None:
        """Emoji verdict symbols map correctly."""
        assert normalize_verdict("\u26a1") == Verdict.ACCEPT
        assert normalize_verdict("\u2717") == Verdict.REJECT
        assert normalize_verdict("?") == Verdict.UNCERTAIN

    # -- test_parse_json_with_extra_text ------------------------------------

    def test_parse_json_surrounded_by_text(self) -> None:
        """JSON block embedded in prose is still parsed."""
        text = (
            "Here is my evaluation:\n"
            '```json\n{"verdict": "reject", "scores": {"rigor": 2}, "reasoning": "Flawed logic."}\n```\n'
            "I hope this helps."
        )
        result = parse_review_response(text, ["rigor"])
        assert result.is_valid is True
        assert result.verdict == Verdict.REJECT

    # -- test_parse_no_verdict_in_freetext ----------------------------------

    def test_parse_no_verdict_in_freetext(self) -> None:
        """Freetext without any identifiable verdict -> is_valid=False."""
        result = parse_review_response(
            "The mathematical structure is interesting and worth further study.",
            ["rigor"],
        )
        assert result.is_valid is False

    # -- test_regex_score_formats -------------------------------------------

    def test_parse_regex_score_without_denominator(self) -> None:
        """Regex parser handles 'criterion: N' without /10."""
        text = "Verdict: reject\nrigor: 3\nSome reasoning."
        result = parse_review_response(text, ["rigor"])
        assert result.is_valid is True
        assert result.scores["rigor"] == pytest.approx(3.0)
