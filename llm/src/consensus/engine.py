"""ConsensusEngine - Multi-LLM agreement engine.

Runs 1-N rounds of structured evaluation to reach consensus on a task.
Each use case provides a ConsensusTaskInterface that controls prompts and parsing.
"""

from __future__ import annotations

import asyncio
from collections import Counter

from shared.errors import InsufficientResponsesError
from shared.logging import get_logger
from shared.models import (
    ConsensusConfig,
    ConsensusEngineInterface,
    ConsensusResult,
    ConsensusTaskInterface,
    LLMClientInterface,
    RoundResult,
    ValidatedResponse,
    Verdict,
)

log = get_logger("llm", "consensus_engine")


class ConsensusEngine(ConsensusEngineInterface):
    """Multi-LLM agreement engine implementing ConsensusEngineInterface."""

    def __init__(self, llm_client: LLMClientInterface, config: ConsensusConfig) -> None:
        self._client = llm_client
        self._config = config

    async def run(
        self,
        task: ConsensusTaskInterface,
        backends: list[str] | None = None,
    ) -> ConsensusResult:
        """Execute multi-round consensus and return the result."""
        active_backends = backends or list(self._config.backends)
        round_history: list[RoundResult] = []

        for round_num in range(1, self._config.max_rounds + 1):
            valid = await self._run_round(task, round_num, round_history, active_backends)
            converged, verdict = self._check_convergence(valid, round_num)
            round_history.append(RoundResult(
                round_num=round_num, responses=valid,
                is_converged=converged, verdict=verdict,
            ))
            if converged:
                break

        return self._compile_result(round_history)

    async def _run_round(
        self,
        task: ConsensusTaskInterface,
        round_num: int,
        prior_rounds: list[RoundResult],
        backends: list[str],
    ) -> list[ValidatedResponse]:
        """Run one round: send, parse, filter, retry if insufficient."""
        valid: list[ValidatedResponse] = []
        for attempt in range(1 + self._config.max_retries_per_backend):
            prompts = {b: task.get_prompt(round_num, prior_rounds, b) for b in backends}
            raw = await asyncio.gather(
                *(self._client.send(b, prompts[b]) for b in backends),
                return_exceptions=True,
            )
            valid = []
            for backend, resp in zip(backends, raw):
                if isinstance(resp, BaseException):
                    log.warning("consensus.backend.error",
                                backend=backend, round_num=round_num, error=str(resp))
                    continue
                parsed = task.parse_response(resp.text)
                if not parsed.is_valid or parsed.verdict is None:
                    continue
                valid.append(ValidatedResponse(
                    backend=backend, verdict=parsed.verdict,
                    scores=parsed.scores, reasoning=parsed.reasoning or "",
                    raw_text=resp.text,
                ))
            if len(valid) >= self._config.min_valid_responses:
                return valid
            log.warning("consensus.insufficient_responses",
                        round_num=round_num, attempt=attempt + 1,
                        valid_count=len(valid),
                        required=self._config.min_valid_responses)

        raise InsufficientResponsesError(
            f"Only {len(valid)} valid responses after retries "
            f"(need {self._config.min_valid_responses})",
            rounds_completed=round_num,
        )

    def _check_convergence(
        self,
        responses: list[ValidatedResponse],
        round_num: int,
    ) -> tuple[bool, Verdict | None]:
        """Determine if consensus has been reached based on decision_method."""
        if not responses:
            return False, None

        counts = Counter(r.verdict for r in responses)
        total = len(responses)
        top_verdict, top_count = counts.most_common(1)[0]
        method = self._config.decision_method

        if method == "unanimous":
            if top_count == total:
                return True, top_verdict
        elif method == "majority":
            if top_count >= self._config.minimum_agreement * total:
                return True, top_verdict
        elif method == "supermajority":
            if top_count >= 0.66 * total:
                return True, top_verdict

        return False, None

    def _aggregate_scores(self, responses: list[ValidatedResponse]) -> dict[str, float]:
        """Average scores across valid responses, per criterion."""
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for resp in responses:
            for name, score in resp.scores.items():
                totals[name] = totals.get(name, 0.0) + score
                counts[name] = counts.get(name, 0) + 1
        return {name: totals[name] / counts[name] for name in totals}

    def _compile_result(self, round_history: list[RoundResult]) -> ConsensusResult:
        """Build final ConsensusResult from round history."""
        last = round_history[-1]

        if last.is_converged and last.verdict is not None:
            verdict = last.verdict
            responses = last.responses
            vote_count = len(responses)
            counts = Counter(r.verdict for r in responses)
            confidence = counts[verdict] / vote_count if vote_count else 0.0
        else:
            responses = last.responses
            vote_count = len(responses)
            verdict = Verdict.UNCERTAIN
            if responses:
                _, top_count = Counter(r.verdict for r in responses).most_common(1)[0]
                confidence = top_count / vote_count
            else:
                confidence = 0.0

        return ConsensusResult(
            verdict=verdict,
            confidence=confidence,
            scores=self._aggregate_scores(responses),
            rounds_completed=len(round_history),
            round_history=round_history,
            valid_vote_count=vote_count,
        )
