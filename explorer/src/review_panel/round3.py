"""
Round 3: Structured Deliberation with Collaborative Modification

When Round 2 still has a split (2-1), we engage in structured deliberation:
1. Minority states their strongest single argument AND may propose a modification
2. Majority responds to argument AND evaluates any proposed modification
3. If modification is accepted by majority, final votes are on the modified insight
4. If still 2-1, majority wins but result is flagged as "disputed"

Key insight: Sometimes a small modification to definitions/wording reveals a
profound mathematical truth that the original wording obscured. LLMs should
be able to collaboratively refine the insight during deliberation.
"""

import asyncio
from datetime import datetime
from typing import Optional, Tuple

from shared.logging import get_logger

from .models import ReviewResponse, ReviewRound, MindChange, RefinementRecord
from .prompts import (
    build_round3_minority_prompt,
    build_round3_majority_response_prompt,
    build_round3_final_prompt,
    parse_round3_response,
)
from .claude_api import ClaudeReviewer
from .llm_executor import LLMExecutor, create_executors, send_to_llm

# Import quota exception for special handling
try:
    from ..browser.gemini import GeminiQuotaExhausted
except ImportError:
    GeminiQuotaExhausted = None

log = get_logger("explorer", "review_panel.round3")


async def run_round3(
    chunk_insight: str,
    round2: ReviewRound,
    gemini_browser,
    chatgpt_browser,
    claude_reviewer: Optional[ClaudeReviewer],
    config: dict,
) -> Tuple[ReviewRound, list[MindChange], bool, Optional[str], Optional[RefinementRecord]]:
    """
    Run Round 3: Structured deliberation with collaborative modification.

    Args:
        chunk_insight: The insight text being reviewed
        round2: The completed Round 2
        gemini_browser: GeminiBrowser instance
        chatgpt_browser: ChatGPTBrowser instance
        claude_reviewer: ClaudeReviewer instance
        config: Review panel configuration

    Returns:
        Tuple of (ReviewRound, list of MindChanges, is_disputed, modified_insight, refinement_record)
        - modified_insight: The new insight text if modification was accepted, else None
        - refinement_record: Record of the modification if one was accepted, else None
    """
    log.info("round3.started")

    # Create executors for available LLMs
    executors = create_executors(gemini_browser, chatgpt_browser, claude_reviewer)

    # Identify majority and minority positions
    majority_rating = round2.get_majority_rating()
    minority_llms = round2.get_minority_llms()
    majority_llms = round2.get_majority_llms()

    if not majority_rating or not minority_llms:
        log.warning("round3.no_clear_split")
        return round2, [], False, None, None

    minority_llm = minority_llms[0]  # Should be exactly one
    minority_rating = round2.responses[minority_llm].rating

    log.info("round3.positions", majority=majority_llms, minority=minority_llms,
             majority_rating=majority_rating, minority_rating=minority_rating)

    # Build summaries for prompts
    majority_reasoning = _summarize_reasoning(round2, majority_llms)
    minority_reasoning = _summarize_reasoning(round2, minority_llms)

    # Step 1: Get minority's strongest argument (and optional modification proposal)
    minority_result = await _get_minority_argument(
        chunk_insight=chunk_insight,
        majority_rating=majority_rating,
        majority_count=len(majority_llms),
        majority_reasoning=majority_reasoning,
        minority_rating=minority_rating,
        minority_count=len(minority_llms),
        minority_reasoning=minority_reasoning,
        minority_llm=minority_llm,
        executors=executors,
    )

    if not minority_result:
        log.warning("round3.minority_argument_failed")
        return round2, [], True, None, None

    minority_argument = minority_result.get("strongest_argument", "")
    proposed_modification = minority_result.get("proposed_modification", "")
    modification_rationale = minority_result.get("modification_rationale", "")

    log.info("round3.minority_argument", preview=minority_argument[:100])
    if proposed_modification:
        log.info("round3.minority_modification", preview=proposed_modification[:100])

    # Step 2: Get majority's response (including evaluation of any proposed modification)
    majority_result = await _get_majority_response(
        chunk_insight=chunk_insight,
        majority_rating=majority_rating,
        minority_argument=minority_argument,
        majority_llms=majority_llms,
        proposed_modification=proposed_modification,
        modification_rationale=modification_rationale,
        executors=executors,
    )

    if not majority_result:
        log.warning("round3.majority_response_failed")
        return round2, [], True, None, None

    majority_response = majority_result.get("response_to_argument", "")
    modification_accepted = majority_result.get("accept_modification", False)

    log.info("round3.majority_response", preview=majority_response[:100])
    if proposed_modification:
        log.info("round3.modification_decision", accepted=modification_accepted)

    # Track accepted modification
    modified_insight = None
    refinement_record = None

    if proposed_modification and modification_accepted:
        log.info("round3.modification_accepted")
        modified_insight = proposed_modification
        refinement_record = RefinementRecord(
            from_version=1,
            to_version=2,
            original_insight=chunk_insight,
            refined_insight=proposed_modification,
            changes_made=[f"Modified during Round 3 deliberation by {minority_llm}"],
            addressed_critiques=[minority_argument],
            unresolved_issues=[],
            refinement_confidence="high",
            triggered_by_ratings=round2.get_ratings(),
            timestamp=datetime.now(),
        )

    # Step 3: Get final votes from all (on original or modified insight)
    try:
        final_responses, mind_changes = await _get_final_votes(
            chunk_insight=chunk_insight,
            minority_argument=minority_argument,
            majority_response=majority_response,
            round2=round2,
            minority_llm=minority_llm,
            executors=executors,
            modified_insight=modified_insight,
            modification_accepted=modification_accepted,
        )
    except Exception as e:
        if GeminiQuotaExhausted and isinstance(e, GeminiQuotaExhausted):
            log.error("round3.quota_exhausted")
            final_responses = getattr(e, 'partial_responses', {})
            mind_changes = getattr(e, 'mind_changes', [])
            partial_round = ReviewRound(
                round_number=3,
                mode="deliberation",
                responses=final_responses,
                outcome="quota_exhausted",
                timestamp=datetime.now(),
            )
            e.partial_round = partial_round
            e.mind_changes = mind_changes
            e.is_disputed = True
            e.modified_insight = modified_insight
            e.refinement_record = refinement_record
            raise
        else:
            raise

    # Determine final outcome
    final_ratings = [r.rating for r in final_responses.values()]
    unique_ratings = set(final_ratings)

    if len(unique_ratings) == 1:
        outcome = "resolved"
        is_disputed = False
        log.info("round3.resolved", rating=final_ratings[0], modified=bool(modified_insight))
    else:
        outcome = "disputed"
        is_disputed = True
        log.info("round3.disputed", ratings=final_ratings)

    review_round = ReviewRound(
        round_number=3,
        mode="deliberation",
        responses=final_responses,
        outcome=outcome,
        timestamp=datetime.now(),
    )

    return review_round, mind_changes, is_disputed, modified_insight, refinement_record


def _summarize_reasoning(round2: ReviewRound, llms: list[str]) -> str:
    """Summarize the reasoning from a group of reviewers."""
    parts = []
    for llm in llms:
        if llm in round2.responses:
            resp = round2.responses[llm]
            reasoning = resp.reasoning or resp.new_information or "No reasoning given"
            parts.append(f"{llm}: {reasoning[:200]}")
    return " | ".join(parts)


async def _get_minority_argument(
    chunk_insight: str,
    majority_rating: str,
    majority_count: int,
    majority_reasoning: str,
    minority_rating: str,
    minority_count: int,
    minority_reasoning: str,
    minority_llm: str,
    executors: dict[str, LLMExecutor],
) -> Optional[dict]:
    """Get the minority's strongest argument and optional modification proposal."""
    prompt = build_round3_minority_prompt(
        chunk_insight=chunk_insight,
        majority_rating=majority_rating,
        majority_count=majority_count,
        majority_reasoning_summary=majority_reasoning,
        minority_rating=minority_rating,
        minority_count=minority_count,
        minority_reasoning_summary=minority_reasoning,
    )

    try:
        response_text = await send_to_llm(minority_llm, prompt, executors, thinking_mode="deep")
        parsed = parse_round3_response(response_text, is_minority=True)
        return {
            "strongest_argument": parsed.get("strongest_argument", ""),
            "proposed_modification": parsed.get("proposed_modification", ""),
            "modification_rationale": parsed.get("modification_rationale", ""),
        }
    except Exception as e:
        log.error("round3.minority_argument_error", error=str(e))
        return None


async def _get_majority_response(
    chunk_insight: str,
    majority_rating: str,
    minority_argument: str,
    majority_llms: list[str],
    proposed_modification: str,
    modification_rationale: str,
    executors: dict[str, LLMExecutor],
) -> Optional[dict]:
    """Get the majority's response to the minority argument and modification proposal."""
    prompt = build_round3_majority_response_prompt(
        chunk_insight=chunk_insight,
        majority_rating=majority_rating,
        minority_strongest_argument=minority_argument,
        proposed_modification=proposed_modification,
        modification_rationale=modification_rationale,
    )

    # Get response from first available majority LLM
    for llm in majority_llms:
        if llm not in executors:
            continue
        try:
            response_text = await send_to_llm(llm, prompt, executors, thinking_mode="deep")
            parsed = parse_round3_response(response_text, is_minority=False)
            return {
                "response_to_argument": parsed.get("response_to_argument", ""),
                "accept_modification": parsed.get("accept_modification", False),
                "modification_assessment": parsed.get("modification_assessment", ""),
            }
        except Exception as e:
            log.warning("round3.majority_response_error", llm=llm, error=str(e))
            continue

    return None


async def _get_final_votes(
    chunk_insight: str,
    minority_argument: str,
    majority_response: str,
    round2: ReviewRound,
    minority_llm: str,
    executors: dict[str, LLMExecutor],
    modified_insight: Optional[str] = None,
    modification_accepted: bool = False,
) -> tuple[dict[str, ReviewResponse], list[MindChange]]:
    """Get final votes from all reviewers after deliberation."""
    all_llms = [llm for llm in round2.responses.keys() if llm in executors]

    tasks = []
    for llm in all_llms:
        is_minority = (llm == minority_llm)
        prompt = build_round3_final_prompt(
            chunk_insight=chunk_insight,
            minority_strongest_argument=minority_argument,
            majority_response=majority_response,
            is_minority=is_minority,
            modified_insight=modified_insight or "",
            modification_accepted=modification_accepted,
        )
        tasks.append((llm, is_minority, _final_vote_with_executor(
            executors[llm], prompt, is_minority, minority_argument, majority_response
        )))

    # Execute in parallel
    results = await asyncio.gather(*[t[2] for t in tasks], return_exceptions=True)

    responses = {}
    mind_changes = []
    quota_exhausted_exception = None

    for (llm, is_minority, _), result in zip(tasks, results):
        if isinstance(result, Exception):
            if GeminiQuotaExhausted and isinstance(result, GeminiQuotaExhausted):
                log.error("round3.vote_quota_exhausted", llm=llm)
                quota_exhausted_exception = result
            else:
                log.error("round3.vote_failed", llm=llm, error=str(result))

            # Carry forward Round 2 response
            if llm in round2.responses:
                responses[llm] = round2.responses[llm]
        else:
            responses[llm] = result

            # Check for mind change
            if llm in round2.responses:
                r2_rating = round2.responses[llm].rating
                r3_rating = result.rating

                if r2_rating != r3_rating:
                    suffix = " (on modified insight)" if modification_accepted else ""
                    log.info("round3.mind_changed", llm=llm, from_rating=r2_rating,
                             to_rating=r3_rating, modified=modification_accepted)
                    stance = result.final_stance or ("conceded" if is_minority else "persuaded")
                    mind_changes.append(MindChange(
                        llm=llm,
                        round_number=3,
                        from_rating=r2_rating,
                        to_rating=r3_rating,
                        reason=f"After deliberation{suffix}: {stance}",
                    ))

    if quota_exhausted_exception:
        log.warning("round3.reraise_quota_exhaustion")
        quota_exhausted_exception.partial_responses = responses
        quota_exhausted_exception.mind_changes = mind_changes
        raise quota_exhausted_exception

    return responses, mind_changes


async def _final_vote_with_executor(
    executor: LLMExecutor,
    prompt: str,
    is_minority: bool,
    minority_argument: str,
    majority_response: str,
) -> ReviewResponse:
    """Get a final vote from a single LLM using its executor."""
    response_text = await executor.send(prompt, thinking_mode="deep")
    parsed = parse_round3_response(response_text, is_minority=is_minority)

    strongest_arg = minority_argument if is_minority else None
    response_arg = majority_response if not is_minority else None
    reasoning = parsed.get("reasoning", "") or parsed.get("reason", "") or parsed.get("justification", "")

    return ReviewResponse(
        llm=executor.name,
        mode="deliberation",
        rating=parsed["rating"],
        mathematical_verification="",
        structural_analysis="",
        naturalness_assessment="",
        reasoning=reasoning,
        confidence="high",
        strongest_argument=strongest_arg,
        response_to_argument=response_arg,
        final_stance=parsed.get("final_stance"),
    )
