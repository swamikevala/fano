"""
Round 4: Modification Focus

For disputed insights that went through all 3 rounds without consensus.
This round shows the full deliberation history and focuses specifically
on finding a modification that could resolve the disagreement.

Phase 1: All LLMs see full history and propose modifications
Phase 2: If a modification is accepted, all LLMs vote on the modified insight
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Tuple

from .models import ReviewResponse, ReviewRound, RefinementRecord
from .prompts import (
    build_round4_modification_prompt,
    build_round4_final_vote_prompt,
    build_round_summary,
    parse_round4_modification_response,
    parse_round4_final_vote_response,
)
from .claude_api import ClaudeReviewer

logger = logging.getLogger(__name__)


async def run_round4(
    chunk_insight: str,
    review_rounds: list[ReviewRound],
    gemini_browser,
    chatgpt_browser,
    claude_reviewer: Optional[ClaudeReviewer],
    config: dict,
) -> Tuple[ReviewRound, Optional[str], Optional[RefinementRecord], bool]:
    """
    Run Round 4: Modification-focused review for disputed insights.

    Args:
        chunk_insight: The insight text being reviewed
        review_rounds: List of previous ReviewRounds (1, 2, 3)
        gemini_browser: GeminiBrowser instance
        chatgpt_browser: ChatGPTBrowser instance
        claude_reviewer: ClaudeReviewer instance
        config: Review panel configuration

    Returns:
        Tuple of (ReviewRound, modified_insight, refinement_record, is_intractable)
        - modified_insight: The new insight text if modification was accepted, else None
        - refinement_record: Record of the modification if one was accepted, else None
        - is_intractable: True if no LLM could propose a valid fix
    """
    logger.info("[round4] Starting modification-focused review")

    # Build summaries of previous rounds
    round1 = next((r for r in review_rounds if r.round_number == 1), None)
    round2 = next((r for r in review_rounds if r.round_number == 2), None)
    round3 = next((r for r in review_rounds if r.round_number == 3), None)

    round1_summary = build_round_summary(round1.responses, 1) if round1 else "Not available"
    round2_summary = build_round_summary(round2.responses, 2) if round2 else "Not available"
    round3_summary = build_round_summary(round3.responses, 3) if round3 else "Not available"

    # Get final ratings from Round 3 (or Round 2 if no Round 3)
    final_round = round3 or round2 or round1
    final_ratings = {llm: r.rating for llm, r in final_round.responses.items()} if final_round else {}

    # Phase 1: Get modification proposals from all LLMs
    logger.info("[round4] Phase 1: Getting modification proposals")

    prompt = build_round4_modification_prompt(
        chunk_insight=chunk_insight,
        round1_summary=round1_summary,
        round2_summary=round2_summary,
        round3_summary=round3_summary,
        final_ratings=final_ratings,
    )

    # Run all proposals in parallel
    tasks = []
    if gemini_browser:
        tasks.append(("gemini", _get_modification_proposal(gemini_browser, prompt, "gemini")))
    if chatgpt_browser:
        tasks.append(("chatgpt", _get_modification_proposal(chatgpt_browser, prompt, "chatgpt")))
    if claude_reviewer and claude_reviewer.is_available():
        tasks.append(("claude", _get_modification_proposal_claude(claude_reviewer, prompt)))

    if not tasks:
        raise RuntimeError("No LLMs available for Round 4")

    task_names = [t[0] for t in tasks]
    task_coros = [t[1] for t in tasks]

    logger.info(f"[round4] Running modification proposals: {task_names}")
    results = await asyncio.gather(*task_coros, return_exceptions=True)

    # Collect proposals
    proposals = {}
    for name, result in zip(task_names, results):
        if isinstance(result, Exception):
            logger.error(f"[round4] {name} proposal failed: {result}")
        else:
            proposals[name] = result
            if result.get("can_be_fixed"):
                logger.info(f"[round4] {name} proposed a fix (expected rating: {result.get('expected_rating')})")
            else:
                logger.info(f"[round4] {name} says insight cannot be fixed")

    # Select best modification using priority logic
    best_mod, mod_source, mod_rationale = _select_best_modification(proposals)

    if not best_mod:
        # No valid modification proposed - mark as intractable
        logger.info("[round4] No valid modifications proposed - insight is intractable")

        # Build response round with diagnoses
        responses = {}
        for name, proposal in proposals.items():
            responses[name] = ReviewResponse(
                llm=name,
                mode="modification_focus",
                rating="?",  # Can't rate without modification
                mathematical_verification="",
                structural_analysis="",
                naturalness_assessment="",
                reasoning=proposal.get("diagnosis", "No diagnosis"),
                confidence="high",
                proposed_modification=None,
                modification_rationale=proposal.get("modification_rationale", ""),
            )

        review_round = ReviewRound(
            round_number=4,
            mode="modification_focus",
            responses=responses,
            outcome="intractable",
            timestamp=datetime.now(),
        )

        return review_round, None, None, True

    # Phase 2: Get final votes on the modified insight
    logger.info(f"[round4] Phase 2: Getting final votes on modification from {mod_source}")

    vote_prompt = build_round4_final_vote_prompt(
        original_insight=chunk_insight,
        modified_insight=best_mod,
        modification_source=mod_source,
        modification_rationale=mod_rationale,
        round3_summary=round3_summary,
    )

    # Run all votes in parallel
    vote_tasks = []
    if gemini_browser:
        vote_tasks.append(("gemini", _get_final_vote(gemini_browser, vote_prompt, "gemini")))
    if chatgpt_browser:
        vote_tasks.append(("chatgpt", _get_final_vote(chatgpt_browser, vote_prompt, "chatgpt")))
    if claude_reviewer and claude_reviewer.is_available():
        vote_tasks.append(("claude", _get_final_vote_claude(claude_reviewer, vote_prompt)))

    vote_names = [t[0] for t in vote_tasks]
    vote_coros = [t[1] for t in vote_tasks]

    logger.info(f"[round4] Running final votes: {vote_names}")
    vote_results = await asyncio.gather(*vote_coros, return_exceptions=True)

    # Build responses
    responses = {}
    for name, result in zip(vote_names, vote_results):
        if isinstance(result, Exception):
            logger.error(f"[round4] {name} vote failed: {result}")
            responses[name] = ReviewResponse(
                llm=name,
                mode="final_vote",
                rating="?",
                mathematical_verification="",
                structural_analysis="",
                naturalness_assessment="",
                reasoning=f"Vote failed: {result}",
                confidence="low",
            )
        else:
            responses[name] = ReviewResponse(
                llm=name,
                mode="final_vote",
                rating=result["rating"],
                mathematical_verification="",
                structural_analysis="",
                naturalness_assessment="",
                reasoning=result["reasoning"],
                confidence=result["confidence"],
            )
            logger.info(f"[round4] {name} voted {result['rating']} (resolves: {result.get('resolves_dispute', '?')})")

    # Determine outcome
    ratings = [r.rating for r in responses.values()]
    unique_ratings = set(ratings)

    if len(unique_ratings) == 1:
        outcome = "resolved"
        logger.info(f"[round4] Unanimous on modified insight: {ratings[0]}")
    else:
        # Check for majority
        rating_counts = {r: ratings.count(r) for r in unique_ratings}
        max_count = max(rating_counts.values())
        if max_count >= 2:
            outcome = "majority"
            majority_rating = [r for r, c in rating_counts.items() if c == max_count][0]
            logger.info(f"[round4] Majority on modified insight: {majority_rating} ({max_count}/{len(ratings)})")
        else:
            outcome = "still_disputed"
            logger.info(f"[round4] Still disputed after modification: {ratings}")

    # Create refinement record
    refinement_record = RefinementRecord(
        from_version=1,  # Will be updated by caller
        to_version=2,    # Will be updated by caller
        original_insight=chunk_insight,
        refined_insight=best_mod,
        changes_made=[mod_rationale] if mod_rationale else ["Modification proposed in Round 4"],
        addressed_critiques=["Disputed insight modified to resolve disagreement"],
        unresolved_issues=[],
        refinement_confidence="medium",
        triggered_by_ratings=final_ratings,
        timestamp=datetime.now(),
        proposer=mod_source,
        round_proposed=4,
    )

    review_round = ReviewRound(
        round_number=4,
        mode="modification_focus",
        responses=responses,
        outcome=outcome,
        timestamp=datetime.now(),
    )

    return review_round, best_mod, refinement_record, False


def _select_best_modification(proposals: dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Select the best modification from proposals.

    Priority: Gemini Deep Think > ChatGPT Pro > Claude
    Only accept if can_be_fixed is True and expected_rating is good.

    Returns:
        Tuple of (modification, source, rationale) or (None, None, None)
    """
    # Priority order for Round 4 (all using deep modes)
    priority = ["gemini", "chatgpt", "claude"]

    # First pass: prefer modifications with expected_rating of Profound
    for llm in priority:
        if llm in proposals:
            p = proposals[llm]
            if p.get("can_be_fixed") and p.get("proposed_modification"):
                if p.get("expected_rating") == "⚡":
                    logger.info(f"[round4] Accepting Profound modification from {llm}")
                    return p["proposed_modification"], llm, p.get("modification_rationale", "")

    # Second pass: accept any valid modification
    for llm in priority:
        if llm in proposals:
            p = proposals[llm]
            if p.get("can_be_fixed") and p.get("proposed_modification"):
                logger.info(f"[round4] Accepting modification from {llm} (expected: {p.get('expected_rating')})")
                return p["proposed_modification"], llm, p.get("modification_rationale", "")

    return None, None, None


async def _get_modification_proposal(browser, prompt: str, llm_name: str) -> dict:
    """Get modification proposal from Gemini or ChatGPT."""
    logger.info(f"[round4] Getting modification proposal from {llm_name}")

    try:
        # Start fresh chat
        await browser.start_new_chat()

        # Enable deep thinking mode
        if llm_name == "gemini":
            await browser.enable_deep_think()
        elif llm_name == "chatgpt":
            try:
                await browser.enable_pro_mode()
            except Exception:
                pass  # Pro mode may not be available

        # Send prompt and get response
        response_text = await browser.send_message(prompt)

        # Parse response
        return parse_round4_modification_response(response_text)

    except Exception as e:
        logger.error(f"[round4] {llm_name} modification proposal failed: {e}")
        raise


async def _get_modification_proposal_claude(claude_reviewer: ClaudeReviewer, prompt: str) -> dict:
    """Get modification proposal from Claude."""
    logger.info("[round4] Getting modification proposal from Claude")

    try:
        # Use extended thinking for deep analysis
        response_text = await claude_reviewer.send_message(
            prompt,
            extended_thinking=True,
        )

        # Parse response
        return parse_round4_modification_response(response_text)

    except Exception as e:
        logger.error(f"[round4] Claude modification proposal failed: {e}")
        raise


async def _get_final_vote(browser, prompt: str, llm_name: str) -> dict:
    """Get final vote from Gemini or ChatGPT."""
    logger.info(f"[round4] Getting final vote from {llm_name}")

    try:
        # Start fresh chat (don't need deep mode for simple vote)
        await browser.start_new_chat()

        # Send prompt and get response
        response_text = await browser.send_message(prompt)

        # Parse response
        return parse_round4_final_vote_response(response_text)

    except Exception as e:
        logger.error(f"[round4] {llm_name} final vote failed: {e}")
        raise


async def _get_final_vote_claude(claude_reviewer: ClaudeReviewer, prompt: str) -> dict:
    """Get final vote from Claude."""
    logger.info("[round4] Getting final vote from Claude")

    try:
        # Standard mode for simple vote
        response_text = await claude_reviewer.send_message(
            prompt,
            extended_thinking=False,
        )

        # Parse response
        return parse_round4_final_vote_response(response_text)

    except Exception as e:
        logger.error(f"[round4] Claude final vote failed: {e}")
        raise
