"""
Round 3: Structured Deliberation

When Round 2 still has a split (2-1), we engage in structured deliberation:
1. Minority states their strongest single argument
2. Majority responds to that specific argument
3. Final votes are cast
4. If still 2-1, majority wins but result is flagged as "disputed"
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from .models import ReviewResponse, ReviewRound, MindChange
from .prompts import (
    build_round3_minority_prompt,
    build_round3_majority_response_prompt,
    build_round3_final_prompt,
    parse_round3_response,
)
from .claude_api import ClaudeReviewer

logger = logging.getLogger(__name__)


async def run_round3(
    chunk_insight: str,
    round2: ReviewRound,
    gemini_browser,
    chatgpt_browser,
    claude_reviewer: Optional[ClaudeReviewer],
    config: dict,
) -> tuple[ReviewRound, list[MindChange], bool]:
    """
    Run Round 3: Structured deliberation.

    Args:
        chunk_insight: The insight text being reviewed
        round2: The completed Round 2
        gemini_browser: GeminiBrowser instance
        chatgpt_browser: ChatGPTBrowser instance
        claude_reviewer: ClaudeReviewer instance
        config: Review panel configuration

    Returns:
        Tuple of (ReviewRound, list of MindChanges, is_disputed)
    """
    logger.info("[round3] Starting structured deliberation")

    # Identify majority and minority positions
    majority_rating = round2.get_majority_rating()
    minority_llms = round2.get_minority_llms()
    majority_llms = round2.get_majority_llms()

    if not majority_rating or not minority_llms:
        # This shouldn't happen - Round 3 only runs on 2-1 splits
        logger.warning("[round3] No clear majority/minority, skipping deliberation")
        return round2, [], False

    minority_llm = minority_llms[0]  # Should be exactly one
    minority_rating = round2.responses[minority_llm].rating

    logger.info(f"[round3] Majority ({majority_rating}): {majority_llms}")
    logger.info(f"[round3] Minority ({minority_rating}): {minority_llms}")

    # Build summaries for prompts
    majority_reasoning = _summarize_reasoning(round2, majority_llms)
    minority_reasoning = _summarize_reasoning(round2, minority_llms)

    # Step 1: Get minority's strongest argument
    minority_argument = await _get_minority_argument(
        chunk_insight=chunk_insight,
        majority_rating=majority_rating,
        majority_count=len(majority_llms),
        majority_reasoning=majority_reasoning,
        minority_rating=minority_rating,
        minority_count=len(minority_llms),
        minority_reasoning=minority_reasoning,
        minority_llm=minority_llm,
        gemini_browser=gemini_browser,
        chatgpt_browser=chatgpt_browser,
        claude_reviewer=claude_reviewer,
    )

    if not minority_argument:
        logger.warning("[round3] Could not get minority argument")
        return round2, [], True  # Flag as disputed

    logger.info(f"[round3] Minority argument: {minority_argument[:100]}...")

    # Step 2: Get majority's response
    majority_response = await _get_majority_response(
        chunk_insight=chunk_insight,
        majority_rating=majority_rating,
        minority_argument=minority_argument,
        majority_llms=majority_llms,
        gemini_browser=gemini_browser,
        chatgpt_browser=chatgpt_browser,
        claude_reviewer=claude_reviewer,
    )

    if not majority_response:
        logger.warning("[round3] Could not get majority response")
        return round2, [], True  # Flag as disputed

    logger.info(f"[round3] Majority response: {majority_response[:100]}...")

    # Step 3: Get final votes from all
    final_responses, mind_changes = await _get_final_votes(
        chunk_insight=chunk_insight,
        minority_argument=minority_argument,
        majority_response=majority_response,
        round2=round2,
        minority_llm=minority_llm,
        gemini_browser=gemini_browser,
        chatgpt_browser=chatgpt_browser,
        claude_reviewer=claude_reviewer,
    )

    # Determine final outcome
    final_ratings = [r.rating for r in final_responses.values()]
    unique_ratings = set(final_ratings)

    if len(unique_ratings) == 1:
        outcome = "resolved"
        is_disputed = False
        logger.info(f"[round3] Resolved to unanimous: {final_ratings[0]}")
    else:
        outcome = "disputed"
        is_disputed = True
        logger.info(f"[round3] Still disputed: {final_ratings}")

    review_round = ReviewRound(
        round_number=3,
        mode="deliberation",
        responses=final_responses,
        outcome=outcome,
        timestamp=datetime.now(),
    )

    return review_round, mind_changes, is_disputed


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
    gemini_browser,
    chatgpt_browser,
    claude_reviewer: Optional[ClaudeReviewer],
) -> Optional[str]:
    """Get the minority's strongest argument."""
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
        response_text = await _send_to_llm(
            minority_llm, prompt, gemini_browser, chatgpt_browser, claude_reviewer
        )
        parsed = parse_round3_response(response_text, is_minority=True)
        return parsed.get("strongest_argument", "")
    except Exception as e:
        logger.error(f"[round3] Error getting minority argument: {e}")
        return None


async def _get_majority_response(
    chunk_insight: str,
    majority_rating: str,
    minority_argument: str,
    majority_llms: list[str],
    gemini_browser,
    chatgpt_browser,
    claude_reviewer: Optional[ClaudeReviewer],
) -> Optional[str]:
    """Get the majority's response to the minority argument."""
    prompt = build_round3_majority_response_prompt(
        chunk_insight=chunk_insight,
        majority_rating=majority_rating,
        minority_strongest_argument=minority_argument,
    )

    # Get response from first available majority LLM
    for llm in majority_llms:
        try:
            response_text = await _send_to_llm(
                llm, prompt, gemini_browser, chatgpt_browser, claude_reviewer
            )
            parsed = parse_round3_response(response_text, is_minority=False)
            return parsed.get("response_to_argument", "")
        except Exception as e:
            logger.warning(f"[round3] Error getting majority response from {llm}: {e}")
            continue

    return None


async def _get_final_votes(
    chunk_insight: str,
    minority_argument: str,
    majority_response: str,
    round2: ReviewRound,
    minority_llm: str,
    gemini_browser,
    chatgpt_browser,
    claude_reviewer: Optional[ClaudeReviewer],
) -> tuple[dict[str, ReviewResponse], list[MindChange]]:
    """Get final votes from all reviewers after deliberation."""

    # Build prompts for each LLM
    tasks = []
    all_llms = list(round2.responses.keys())

    for llm in all_llms:
        is_minority = (llm == minority_llm)
        prompt = build_round3_final_prompt(
            chunk_insight=chunk_insight,
            minority_strongest_argument=minority_argument,
            majority_response=majority_response,
            is_minority=is_minority,
        )

        # Determine which browser/API to use
        if llm == "gemini" and gemini_browser:
            tasks.append((llm, is_minority, _final_vote(llm, prompt, gemini_browser, None, None)))
        elif llm == "chatgpt" and chatgpt_browser:
            tasks.append((llm, is_minority, _final_vote(llm, prompt, None, chatgpt_browser, None)))
        elif llm == "claude" and claude_reviewer:
            tasks.append((llm, is_minority, _final_vote(llm, prompt, None, None, claude_reviewer)))
        else:
            logger.warning(f"[round3] {llm} not available for final vote")

    # Execute in parallel
    results = await asyncio.gather(*[t[2] for t in tasks], return_exceptions=True)

    responses = {}
    mind_changes = []

    for (llm, is_minority, _), result in zip(tasks, results):
        if isinstance(result, Exception):
            logger.error(f"[round3] Final vote failed for {llm}: {result}")
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
                    logger.info(f"[round3] {llm} changed mind: {r2_rating} -> {r3_rating}")
                    stance = result.final_stance or ("conceded" if is_minority else "persuaded")
                    mind_changes.append(MindChange(
                        llm=llm,
                        round_number=3,
                        from_rating=r2_rating,
                        to_rating=r3_rating,
                        reason=f"After deliberation: {stance}",
                    ))

    return responses, mind_changes


async def _final_vote(
    llm: str,
    prompt: str,
    gemini_browser,
    chatgpt_browser,
    claude_reviewer: Optional[ClaudeReviewer],
) -> ReviewResponse:
    """Get a final vote from a single LLM."""
    response_text = await _send_to_llm(
        llm, prompt, gemini_browser, chatgpt_browser, claude_reviewer
    )
    parsed = parse_round3_response(response_text, is_minority=True)

    return ReviewResponse(
        llm=llm,
        mode="deliberation",
        rating=parsed["rating"],
        mathematical_verification="",
        structural_analysis="",
        naturalness_assessment="",
        reasoning=parsed.get("response_to_argument", "") or parsed.get("strongest_argument", ""),
        confidence="high",
        final_stance=parsed.get("final_stance"),
    )


async def _send_to_llm(
    llm: str,
    prompt: str,
    gemini_browser,
    chatgpt_browser,
    claude_reviewer: Optional[ClaudeReviewer],
) -> str:
    """Send a prompt to the specified LLM and get response."""
    if llm == "gemini" and gemini_browser:
        # Start fresh chat to ensure clean state
        await gemini_browser.start_new_chat()
        return await gemini_browser.send_message(prompt)
    elif llm == "chatgpt" and chatgpt_browser:
        # Start fresh chat to ensure clean state
        await chatgpt_browser.start_new_chat()
        return await chatgpt_browser.send_message(prompt)
    elif llm == "claude" and claude_reviewer:
        return await claude_reviewer.send_message(prompt, extended_thinking=False)
    else:
        raise ValueError(f"LLM {llm} not available")
