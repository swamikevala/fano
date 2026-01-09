"""
Automated Review Panel Coordinator

Orchestrates the review process with collaborative modification:
1. Round 1: Independent parallel review (standard modes)
   - Any LLM can propose modifications
   - ABANDON vote allows early exit for unsalvageable chunks
2. Round 2: Deep analysis with all Round 1 responses visible
   - Can review modified insight if consensus reached
   - Further modifications can be proposed
3. Round 3: Structured deliberation (if still split)
   - Final collaborative decision

Modification consensus: When modifications are proposed, they are evaluated
for acceptance between rounds. If accepted, subsequent rounds review the
modified version.
"""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from .models import (
    ChunkReview,
    ReviewDecision,
    ReviewResponse,
    ReviewRound,
    RefinementRecord,
    VerificationResult,
    get_rating_pattern,
)
from .round1 import run_round1
from .round2 import run_round2
from .round3 import run_round3

# Import quota exception for special handling
try:
    from ..browser.gemini import GeminiQuotaExhausted
except ImportError:
    GeminiQuotaExhausted = None
from .round4 import _get_final_vote, _get_final_vote_claude
from .prompts import build_round4_final_vote_prompt, build_round_summary
from .claude_api import get_claude_reviewer, ClaudeReviewer
from .deepseek_verifier import get_deepseek_verifier, DeepSeekVerifier
from .math_triggers import needs_math_verification

logger = logging.getLogger(__name__)


def _get_modification_consensus(
    review_round: ReviewRound,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Evaluate modification proposals and determine consensus.

    Logic:
    1. If an LLM rated ⚡ and proposed a modification, prefer their modification
       (they see value and know how to improve it)
    2. If an LLM rated ? with a modification, they think it's fixable
    3. If multiple modifications, prefer from higher-rating LLM
    4. Tiebreaker by mode: Gemini Deep Think > ChatGPT Pro > ChatGPT Thinking > Gemini standard > Claude

    Args:
        review_round: The completed round with responses

    Returns:
        Tuple of (accepted_modification, modification_source, modification_rationale)
        or (None, None, None) if no modification accepted
    """
    # Collect all proposed modifications with their sources and ratings
    modifications = []

    rating_priority = {"⚡": 3, "?": 2, "✗": 1, "ABANDON": 0}

    # Mode-based priority (higher = better)
    # Gemini Deep Think > ChatGPT Pro > ChatGPT Thinking > Gemini standard > Claude
    mode_priority = {
        "deep_think": 5,      # Gemini Deep Think (Round 2)
        "pro": 4,             # ChatGPT Pro (Round 2)
        "thinking": 3,        # ChatGPT Thinking (Round 1)
        "standard": 2,        # Gemini standard (Round 1) - will be adjusted for Claude
        "extended_thinking": 1,  # Claude Extended Thinking (Round 2)
    }

    for llm_name, response in review_round.responses.items():
        if response.proposed_modification:
            # Get mode priority, with special handling for Claude's standard mode
            mode = response.mode or "standard"
            mode_score = mode_priority.get(mode, 1)

            # Claude standard mode should be lowest priority
            if llm_name == "claude" and mode == "standard":
                mode_score = 1

            modifications.append({
                "llm": llm_name,
                "mode": mode,
                "modification": response.proposed_modification,
                "rationale": response.modification_rationale or "No rationale provided",
                "rating": response.rating,
                "rating_score": rating_priority.get(response.rating, 0),
                "mode_score": mode_score,
            })

    if not modifications:
        return None, None, None

    # Sort by rating score (descending), then mode_score (descending)
    modifications.sort(key=lambda x: (x["rating_score"], x["mode_score"]), reverse=True)

    # Accept the best modification
    best = modifications[0]

    # Only accept if rating is ? or better (don't accept from ABANDON or ✗ voters)
    if best["rating_score"] >= 2:  # ? or ⚡
        logger.info(f"[consensus] Accepting modification from {best['llm']} ({best['mode']}, rated {best['rating']})")
        return best["modification"], best["llm"], best["rationale"]

    # If only low-rating LLMs proposed modifications, log but don't accept
    logger.info(f"[consensus] Modifications proposed but from low-rating reviewers, not accepting")
    return None, None, None


class AutomatedReviewer:
    """
    Coordinates the three-round automated review process.

    Uses browser automation for Gemini/ChatGPT and API for Claude.
    """

    def __init__(
        self,
        gemini_browser,
        chatgpt_browser,
        config: dict,
        data_dir: Path,
    ):
        """
        Initialize the automated reviewer.

        Args:
            gemini_browser: GeminiBrowser instance (or None)
            chatgpt_browser: ChatGPTBrowser instance (or None)
            config: Full configuration dict
            data_dir: Base data directory for saving reviews
        """
        self.gemini_browser = gemini_browser
        self.chatgpt_browser = chatgpt_browser
        self.config = config
        self.data_dir = data_dir

        # Get review panel config
        self.panel_config = config.get("review_panel", {})

        # Initialize Claude reviewer
        self.claude_reviewer = get_claude_reviewer(self.panel_config)
        if self.claude_reviewer:
            logger.info("[reviewer] Claude API available")
        else:
            logger.warning("[reviewer] Claude API not available")

        # Initialize DeepSeek math verifier
        self.deepseek = get_deepseek_verifier(config)
        if self.deepseek and self.deepseek.is_available():
            logger.info("[reviewer] DeepSeek math verifier available")
        else:
            self.deepseek = None
            logger.info("[reviewer] DeepSeek math verifier not configured")

        # Check we have at least 2 reviewers
        available = sum([
            1 if gemini_browser else 0,
            1 if chatgpt_browser else 0,
            1 if self.claude_reviewer else 0,
        ])
        if available < 2:
            logger.warning(f"[reviewer] Only {available} reviewers available, need at least 2")

    async def review_insight(
        self,
        chunk_id: str,
        insight_text: str,
        confidence: str,
        tags: list[str],
        dependencies: list[str],
        blessed_axioms_summary: str,
        priority: int = 5,
        check_priority_switches: bool = True,
    ) -> ChunkReview:
        """
        Run the full review process on an insight.

        Flow:
        1. Round 1: Independent review (with modification proposals)
           - If unanimous → Done
           - If unanimous ABANDON → Reject and done
           - Check for modification consensus
        2. Round 2: Deep analysis (sees Round 1, may use modified insight)
           - If unanimous → Done
           - If unanimous ABANDON → Reject and done
           - Check for modification consensus
        3. Round 3: Structured deliberation (if still split)
           - Final collaborative decision with majority rule

        After each round, checks if a higher-priority item needs attention.
        If so, saves progress and returns early with is_paused=True.

        Args:
            chunk_id: Unique ID for this insight
            insight_text: The insight text to review
            confidence: Confidence level from extraction
            tags: Tags assigned to the insight
            dependencies: Dependencies listed for the insight
            blessed_axioms_summary: Summary of blessed axioms for context
            priority: Priority of this insight (1-10, used for switching check)
            check_priority_switches: If True, check for higher priority items after each round

        Returns:
            ChunkReview with complete review history and final rating.
            Check review.is_paused to see if review was interrupted for higher priority.
        """
        start_time = time.time()
        logger.info(f"[reviewer] Starting review for {chunk_id}")

        # Check for existing progress (paused or interrupted review)
        existing_progress = self._load_progress(chunk_id)
        if existing_progress and len(existing_progress.rounds) > 0:
            logger.info(f"[reviewer] Resuming {chunk_id} from Round {len(existing_progress.rounds)}")
            print(f"  [↻] Resuming review from Round {len(existing_progress.rounds)}")
            review = existing_progress
            current_insight = existing_progress.final_insight_text or insight_text
            current_version = existing_progress.final_version
            # Reset pause status since we're now processing this item
            review.is_paused = False
            review.paused_for_id = None
        else:
            review = ChunkReview(chunk_id=chunk_id)
            current_insight = insight_text
            current_version = 1

        accepted_modification = ""
        modification_source = ""

        try:
            # Skip Round 1 if already completed
            if len(review.rounds) >= 1:
                round1 = review.rounds[0]
                logger.info(f"[reviewer] Skipping Round 1 (already completed)")
            else:
                # Round 1: Independent review
                round1 = await run_round1(
                    chunk_insight=current_insight,
                    confidence=confidence,
                    tags=tags,
                    dependencies=dependencies,
                    blessed_axioms_summary=blessed_axioms_summary,
                    gemini_browser=self.gemini_browser,
                    chatgpt_browser=self.chatgpt_browser,
                    claude_reviewer=self.claude_reviewer,
                    config=self.panel_config,
                )
                review.add_round(round1)

                pattern = get_rating_pattern(round1)
                logger.info(f"[reviewer] Round 1 complete: {pattern}")

                # Save progress after Round 1 (safe to interrupt after this)
                self._save_progress(review, 1)

            # Check if we should switch to a higher priority item
            if check_priority_switches:
                should_switch, higher_id = self.should_switch_to_higher_priority(chunk_id, priority)
                if should_switch:
                    logger.info(f"[reviewer] Pausing {chunk_id} for higher priority item {higher_id}")
                    print(f"  [!] Higher priority item found ({higher_id}) - pausing review")
                    review.is_paused = True
                    review.paused_for_id = higher_id
                    return review

            # === MATHEMATICAL VERIFICATION GATE ===
            if self.deepseek:
                # Collect reviewer reasoning for concern detection
                reviewer_reasoning = [
                    (r.reasoning or "") + " " + (r.mathematical_verification or "")
                    for r in round1.responses.values()
                ]

                # Check if verification needed
                should_verify, verify_reason = needs_math_verification(
                    insight=current_insight,
                    tags=tags,
                    reviewer_responses=reviewer_reasoning,
                )

                if should_verify:
                    logger.info(f"[reviewer] Math verification triggered: {verify_reason}")

                    verification = await self.deepseek.verify_insight(
                        insight=current_insight,
                        context=blessed_axioms_summary,
                    )
                    review.math_verification = verification

                    logger.info(f"[reviewer] DeepSeek verdict: {verification.verdict} "
                               f"(confidence: {verification.confidence:.0%})")

                    # Check for auto-rejection
                    if verification.should_auto_reject:
                        logger.info(f"[reviewer] Auto-rejecting: mathematical claim refuted")
                        review.rejection_reason = (
                            f"Mathematical claim refuted by DeepSeek:\n"
                            f"{verification.counterexample}"
                        )
                        review.finalize(
                            rating="✗",
                            unanimous=True,
                            disputed=False,
                        )
                        self._save_review(review, start_time)
                        return review
                else:
                    review.math_verification_skipped = True
                    review.math_verification_skip_reason = verify_reason

            # Check for early exit: unanimous ABANDON
            if round1.outcome == "abandoned":
                review.rejection_reason = "Insight unanimously abandoned as unsalvageable"
                review.finalize(
                    rating="✗",
                    unanimous=True,
                    disputed=False,
                )
                logger.info(f"[reviewer] Abandoned after Round 1 unanimously")
                self._save_review(review, start_time)
                return review

            # Check for early exit: unanimous
            if round1.outcome == "unanimous":
                final_rating = list(round1.get_ratings().values())[0]
                review.finalize(
                    rating=final_rating,
                    unanimous=True,
                    disputed=False,
                )
                logger.info(f"[reviewer] Unanimous after Round 1: {final_rating}")
                self._save_review(review, start_time)
                return review

            # Check for modification consensus after Round 1 (only if we just ran it)
            if len(review.rounds) == 1:  # Just completed Round 1
                mod, mod_source, mod_rationale = _get_modification_consensus(round1)
                if mod:
                    logger.info(f"[reviewer] Modification accepted from {mod_source} after Round 1")
                    # Record the modification
                    refinement_record = RefinementRecord(
                        from_version=current_version,
                        to_version=current_version + 1,
                        original_insight=current_insight,
                        refined_insight=mod,
                        changes_made=[mod_rationale] if mod_rationale else ["Modification proposed during review"],
                        addressed_critiques=["Modification proposed during Round 1 review"],
                        unresolved_issues=[],
                        refinement_confidence="medium",
                        triggered_by_ratings={llm: r.rating for llm, r in round1.responses.items()},
                        timestamp=datetime.now(),
                        proposer=mod_source,
                        round_proposed=1,
                    )
                    review.add_refinement(refinement_record)
                    current_insight = mod
                    current_version += 1
                    accepted_modification = mod
                    modification_source = mod_source
                    logger.info(f"[reviewer] Insight modified to v{current_version}")

            # Skip Round 2 if already completed
            if len(review.rounds) >= 2:
                round2 = review.rounds[1]
                logger.info(f"[reviewer] Skipping Round 2 (already completed)")
            else:
                # Round 2: Deep analysis with all Round 1 responses visible
                try:
                    round2, mind_changes_r2 = await run_round2(
                        chunk_insight=current_insight,
                        blessed_axioms_summary=blessed_axioms_summary,
                        round1=round1,
                        gemini_browser=self.gemini_browser,
                        chatgpt_browser=self.chatgpt_browser,
                        claude_reviewer=self.claude_reviewer,
                        config=self.panel_config,
                        math_verification=review.math_verification,
                        accepted_modification=accepted_modification,
                        modification_source=modification_source,
                    )
                except Exception as e:
                    # Check for Gemini quota exhaustion
                    if GeminiQuotaExhausted and isinstance(e, GeminiQuotaExhausted):
                        logger.error(f"[reviewer] Gemini Deep Think quota exhausted: {e}")
                        # Get partial round from exception if available
                        if hasattr(e, 'partial_round'):
                            round2 = e.partial_round
                            mind_changes_r2 = getattr(e, 'mind_changes', [])
                            review.add_round(round2)
                            review.mind_changes.extend(mind_changes_r2)
                            self._save_progress(review, 2)
                        # Mark as paused with quota exhaustion reason
                        review.is_paused = True
                        review.paused_for_id = f"QUOTA_EXHAUSTED:{e.resume_time}"
                        logger.info(f"[reviewer] Review paused due to quota exhaustion, resume at: {e.resume_time}")
                        # Re-raise so the orchestrator can stop processing
                        raise
                    else:
                        # Other errors - re-raise
                        raise

                review.add_round(round2)
                review.mind_changes.extend(mind_changes_r2)

                pattern = get_rating_pattern(round2)
                logger.info(f"[reviewer] Round 2 (deep analysis): {pattern}")

                # Save progress after Round 2 (safe to interrupt after this)
                self._save_progress(review, 2)

                # Check if we should switch to a higher priority item
                if check_priority_switches:
                    should_switch, higher_id = self.should_switch_to_higher_priority(chunk_id, priority)
                    if should_switch:
                        logger.info(f"[reviewer] Pausing {chunk_id} after Round 2 for higher priority item {higher_id}")
                        print(f"  [!] Higher priority item found ({higher_id}) - pausing review")
                        review.is_paused = True
                        review.paused_for_id = higher_id
                        return review

            # Check for early exit: unanimous ABANDON
            if round2.outcome == "abandoned":
                review.rejection_reason = "Insight unanimously abandoned during deep analysis"
                review.finalize(
                    rating="✗",
                    unanimous=True,
                    disputed=False,
                )
                logger.info(f"[reviewer] Abandoned after Round 2 unanimously")
                self._save_review(review, start_time)
                return review

            # Check for resolution: unanimous
            if round2.outcome == "unanimous":
                final_rating = list(round2.get_ratings().values())[0]
                review.finalize(
                    rating=final_rating,
                    unanimous=True,
                    disputed=False,
                )
                # Store final insight text if modified
                if current_insight != insight_text:
                    review.final_insight_text = current_insight
                logger.info(f"[reviewer] Unanimous after Round 2: {final_rating}")
                self._save_review(review, start_time)
                return review

            # Check for modification consensus after Round 2 (only if we just ran it)
            if len(review.rounds) == 2:  # Just completed Round 2
                mod2, mod2_source, mod2_rationale = _get_modification_consensus(round2)
                if mod2:
                    logger.info(f"[reviewer] Modification accepted from {mod2_source} after Round 2")
                    # Record the modification
                    refinement_record = RefinementRecord(
                        from_version=current_version,
                        to_version=current_version + 1,
                        original_insight=current_insight,
                        refined_insight=mod2,
                        changes_made=[mod2_rationale] if mod2_rationale else ["Modification proposed during deep analysis"],
                        addressed_critiques=["Modification proposed during Round 2 deep analysis"],
                        unresolved_issues=[],
                        refinement_confidence="medium",
                        triggered_by_ratings={llm: r.rating for llm, r in round2.responses.items()},
                        timestamp=datetime.now(),
                        proposer=mod2_source,
                        round_proposed=2,
                    )
                    review.add_refinement(refinement_record)
                    current_insight = mod2
                    current_version += 1
                    accepted_modification = mod2
                    modification_source = mod2_source
                    logger.info(f"[reviewer] Insight modified to v{current_version}")

            # Round 3: Structured deliberation with collaborative modification
            if self.panel_config.get("round3", {}).get("enabled", True):
                # Skip Round 3 if already completed
                if len(review.rounds) >= 3:
                    round3 = review.rounds[2]
                    logger.info(f"[reviewer] Skipping Round 3 (already completed)")
                    # Just finalize with existing data
                    final_rating = round3.get_majority_rating() or "?"
                    review.finalize(
                        rating=final_rating,
                        unanimous=(round3.outcome == "resolved"),
                        disputed=review.is_disputed,
                    )
                else:
                    try:
                        round3, mind_changes_r3, is_disputed, modified_insight, refinement_record = await run_round3(
                            chunk_insight=current_insight,
                            round2=round2,
                            gemini_browser=self.gemini_browser,
                            chatgpt_browser=self.chatgpt_browser,
                            claude_reviewer=self.claude_reviewer,
                            config=self.panel_config,
                        )
                    except Exception as e:
                        # Check for Gemini quota exhaustion
                        if GeminiQuotaExhausted and isinstance(e, GeminiQuotaExhausted):
                            logger.error(f"[reviewer] Gemini Deep Think quota exhausted in Round 3: {e}")
                            # Get partial results from exception if available
                            if hasattr(e, 'partial_round'):
                                round3 = e.partial_round
                                mind_changes_r3 = getattr(e, 'mind_changes', [])
                                is_disputed = getattr(e, 'is_disputed', True)
                                modified_insight = getattr(e, 'modified_insight', None)
                                refinement_record = getattr(e, 'refinement_record', None)
                                review.add_round(round3)
                                review.mind_changes.extend(mind_changes_r3)
                                self._save_progress(review, 3)
                            # Mark as paused with quota exhaustion reason
                            review.is_paused = True
                            review.paused_for_id = f"QUOTA_EXHAUSTED:{e.resume_time}"
                            logger.info(f"[reviewer] Review paused due to quota exhaustion, resume at: {e.resume_time}")
                            # Re-raise so the orchestrator can stop processing
                            raise
                        else:
                            raise

                    review.add_round(round3)
                    review.mind_changes.extend(mind_changes_r3)

                    # If a modification was accepted during deliberation, record it
                    if modified_insight and refinement_record:
                        # Update version numbers
                        refinement_record.from_version = current_version
                        refinement_record.to_version = current_version + 1
                        review.add_refinement(refinement_record)
                        current_insight = modified_insight
                        current_version = refinement_record.to_version
                        logger.info(f"[reviewer] Insight modified during deliberation, now v{current_version}")

                        # Check if we need a verification vote on the modified insight
                        # If not all LLMs voted ⚡, we need them to vote on the modified version
                        round3_ratings = list(round3.get_ratings().values())
                        if round3.outcome != "resolved" or "?" in round3_ratings or "✗" in round3_ratings:
                            logger.info("[reviewer] Running verification vote on modified insight")
                            print("  [⚡] Modification proposed - running verification vote...")

                            # Build verification prompt
                            round3_summary = build_round_summary(round3.responses, 3)
                            vote_prompt = build_round4_final_vote_prompt(
                                original_insight=insight_text,
                                modified_insight=current_insight,
                                modification_source=refinement_record.proposer,
                                modification_rationale=refinement_record.changes_made[0] if refinement_record.changes_made else "",
                                round3_summary=round3_summary,
                            )

                            # Run verification votes in parallel
                            vote_tasks = []
                            if self.gemini_browser:
                                vote_tasks.append(("gemini", _get_final_vote(self.gemini_browser, vote_prompt, "gemini")))
                            if self.chatgpt_browser:
                                vote_tasks.append(("chatgpt", _get_final_vote(self.chatgpt_browser, vote_prompt, "chatgpt")))
                            if self.claude_reviewer and self.claude_reviewer.is_available():
                                vote_tasks.append(("claude", _get_final_vote_claude(self.claude_reviewer, vote_prompt)))

                            vote_names = [t[0] for t in vote_tasks]
                            vote_coros = [t[1] for t in vote_tasks]

                            logger.info(f"[reviewer] Running verification votes: {vote_names}")
                            vote_results = await asyncio.gather(*vote_coros, return_exceptions=True)

                            # Build verification round responses
                            verification_responses = {}
                            for name, result in zip(vote_names, vote_results):
                                if isinstance(result, Exception):
                                    logger.error(f"[reviewer] {name} verification vote failed: {result}")
                                    verification_responses[name] = ReviewResponse(
                                        llm=name,
                                        mode="verification",
                                        rating="?",
                                        reasoning=f"Vote failed: {result}",
                                        confidence="low",
                                    )
                                else:
                                    verification_responses[name] = ReviewResponse(
                                        llm=name,
                                        mode="verification",
                                        rating=result["rating"],
                                        reasoning=result["reasoning"],
                                        confidence=result["confidence"],
                                    )
                                    logger.info(f"[reviewer] {name} verification vote: {result['rating']}")

                            # Create verification round
                            verification_ratings = [r.rating for r in verification_responses.values()]
                            unique_ratings = set(verification_ratings)
                            if len(unique_ratings) == 1:
                                verification_outcome = "resolved"
                            elif verification_ratings.count("⚡") >= 2 or verification_ratings.count("✗") >= 2 or verification_ratings.count("?") >= 2:
                                verification_outcome = "majority"
                            else:
                                verification_outcome = "split"

                            verification_round = ReviewRound(
                                round_number=4,
                                mode="verification",
                                responses=verification_responses,
                                outcome=verification_outcome,
                                timestamp=datetime.now(),
                            )
                            review.add_round(verification_round)

                            # Use verification round for final rating
                            final_rating = verification_round.get_majority_rating() or "?"
                            is_disputed = verification_outcome not in ["resolved"]
                            logger.info(f"[reviewer] Verification complete: {verification_ratings} -> {final_rating}")
                            print(f"  [✓] Verification: {verification_ratings} -> {final_rating}")

                            review.finalize(
                                rating=final_rating,
                                unanimous=(verification_outcome == "resolved"),
                                disputed=is_disputed,
                            )
                            review.final_insight_text = current_insight
                        else:
                            # All voted ⚡ in Round 3, no verification needed
                            final_rating = round3.get_majority_rating() or "?"
                            review.finalize(
                                rating=final_rating,
                                unanimous=(round3.outcome == "resolved"),
                                disputed=is_disputed,
                            )
                            review.final_insight_text = current_insight
                    else:
                        # No modification - use Round 3 result directly
                        final_rating = round3.get_majority_rating() or "?"
                        review.finalize(
                            rating=final_rating,
                            unanimous=(round3.outcome == "resolved"),
                            disputed=is_disputed,
                        )
                        # Store final insight text if modified during earlier rounds
                        if current_insight != insight_text:
                            review.final_insight_text = current_insight
            else:
                # Skip Round 3, use Round 2 majority
                final_rating = round2.get_majority_rating() or "?"
                review.finalize(
                    rating=final_rating,
                    unanimous=False,
                    disputed=True,
                )
                # Store final insight text if modified during earlier rounds
                if current_insight != insight_text:
                    review.final_insight_text = current_insight

            logger.info(f"[reviewer] Final rating: {final_rating} "
                       f"(disputed={review.is_disputed}, refined={review.was_refined})")
            self._save_review(review, start_time)
            return review

        except Exception as e:
            logger.error(f"[reviewer] Review failed: {e}")
            review.finalize(rating="?", unanimous=False, disputed=True)
            self._save_review(review, start_time)
            raise

    def _save_review(self, review: ChunkReview, start_time: float):
        """Save review to disk and update duration."""
        review.review_duration_seconds = time.time() - start_time
        review.save(self.data_dir)
        logger.info(f"[reviewer] ✓ SAVED review for {review.chunk_id} ({review.review_duration_seconds:.1f}s)")

        # Clean up progress file if it exists
        progress_path = self.data_dir / "reviews" / "in_progress" / f"{review.chunk_id}.json"
        if progress_path.exists():
            progress_path.unlink()
            logger.info(f"[reviewer] Cleaned up progress file for {review.chunk_id}")

    def _save_progress(self, review: ChunkReview, round_num: int):
        """Save intermediate progress after a round completes (safe to interrupt after this)."""
        # Save to a progress file so work isn't lost if process is killed
        progress_dir = self.data_dir / "reviews" / "in_progress"
        progress_dir.mkdir(parents=True, exist_ok=True)
        progress_path = progress_dir / f"{review.chunk_id}.json"

        import json
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(review.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"[reviewer] ✓ PROGRESS SAVED after Round {round_num} - safe to interrupt")
        # Also print to console so user knows
        print(f"  [✓] Round {round_num} complete - progress saved (safe to interrupt)")

    def _load_progress(self, chunk_id: str) -> ChunkReview:
        """Load saved progress for a chunk if it exists."""
        progress_path = self.data_dir / "reviews" / "in_progress" / f"{chunk_id}.json"
        if progress_path.exists():
            import json
            with open(progress_path, encoding="utf-8") as f:
                data = json.load(f)
            return ChunkReview.from_dict(data)
        return None

    def get_saved_progress(self, chunk_id: str) -> int:
        """Check if there's saved progress for a chunk. Returns number of rounds completed, or 0."""
        review = self._load_progress(chunk_id)
        if review:
            return len(review.rounds)
        return 0

    def get_highest_priority_pending(self) -> tuple[str, int, int]:
        """
        Get the highest priority insight that needs review.

        Checks pending insights, interesting insights, and in-progress reviews.
        Items in 'interesting' (rated ?) also need review/re-exploration.

        Returns:
            Tuple of (insight_id, priority, rounds_completed) or (None, 0, 0) if none found
        """
        from chunking import AtomicInsight

        candidates = []

        # Check pending AND interesting insights (both need work)
        for status in ["pending", "interesting"]:
            status_dir = self.data_dir / "chunks" / "insights" / status
            if status_dir.exists():
                for json_file in status_dir.glob("*.json"):
                    try:
                        insight = AtomicInsight.load(json_file)
                        # Check if there's in-progress review for this
                        rounds = self.get_saved_progress(insight.id)
                        candidates.append((insight.id, insight.priority, rounds))
                    except Exception as e:
                        logger.warning(f"Could not load {json_file}: {e}")

        # Check in-progress reviews that might not have pending insight files
        # (e.g., if insight was moved but review is partial)
        progress_dir = self.data_dir / "reviews" / "in_progress"
        if progress_dir.exists():
            for json_file in progress_dir.glob("*.json"):
                chunk_id = json_file.stem
                # Only add if not already in candidates
                if not any(c[0] == chunk_id for c in candidates):
                    try:
                        review = self._load_progress(chunk_id)
                        if review:
                            # Try to get priority from insight file
                            priority = 5  # default
                            for status in ["pending", "interesting"]:
                                insight_path = self.data_dir / "chunks" / "insights" / status / f"{chunk_id}.json"
                                if insight_path.exists():
                                    insight = AtomicInsight.load(insight_path)
                                    priority = insight.priority
                                    break
                            candidates.append((chunk_id, priority, len(review.rounds)))
                    except Exception as e:
                        logger.warning(f"Could not load progress {json_file}: {e}")

        if not candidates:
            return (None, 0, 0)

        # Sort by priority (descending), then by rounds completed (ascending - prefer fresh starts)
        candidates.sort(key=lambda x: (-x[1], x[2]))

        return candidates[0]

    def should_switch_to_higher_priority(self, current_id: str, current_priority: int) -> tuple[bool, str]:
        """
        Check if there's a higher priority item we should switch to.

        Args:
            current_id: ID of the insight currently being reviewed
            current_priority: Priority of the current insight

        Returns:
            Tuple of (should_switch, new_insight_id)
        """
        highest_id, highest_priority, _ = self.get_highest_priority_pending()

        if highest_id and highest_id != current_id and highest_priority > current_priority:
            return (True, highest_id)

        return (False, None)

    def get_outcome_action(self, review: ChunkReview) -> str:
        """
        Determine what action to take based on review outcome.

        Uses the outcomes config to map results to actions.

        Args:
            review: Completed review

        Returns:
            Action string: "auto_bless", "auto_reject", "needs_development",
                          "bless_with_flag", "reject_with_flag"
        """
        outcomes_config = self.panel_config.get("outcomes", {})

        if review.is_unanimous:
            if review.final_rating == "⚡":
                return outcomes_config.get("unanimous_bless", "auto_bless")
            elif review.final_rating == "✗":
                return outcomes_config.get("unanimous_reject", "auto_reject")
            else:
                return outcomes_config.get("unanimous_uncertain", "needs_development")
        else:
            # Disputed
            if review.final_rating == "⚡":
                return outcomes_config.get("disputed_majority_bless", "bless_with_flag")
            elif review.final_rating == "✗":
                return outcomes_config.get("disputed_majority_reject", "reject_with_flag")
            else:
                return "needs_development"

    async def review_batch(
        self,
        insights: list[dict],
        blessed_axioms_summary: str,
    ) -> list[ChunkReview]:
        """
        Review a batch of insights sequentially.

        Args:
            insights: List of insight dicts with id, text, confidence, tags, dependencies
            blessed_axioms_summary: Summary of blessed axioms

        Returns:
            List of ChunkReview results
        """
        results = []

        for insight in insights:
            try:
                review = await self.review_insight(
                    chunk_id=insight["id"],
                    insight_text=insight["text"],
                    confidence=insight.get("confidence", "medium"),
                    tags=insight.get("tags", []),
                    dependencies=insight.get("dependencies", []),
                    blessed_axioms_summary=blessed_axioms_summary,
                )
                results.append(review)
            except Exception as e:
                logger.error(f"[reviewer] Failed to review {insight['id']}: {e}")
                # Create a failed review
                review = ChunkReview(chunk_id=insight["id"])
                review.finalize(rating="?", unanimous=False, disputed=True)
                results.append(review)

        return results
