"""
Automated review panel module.

Three-round LLM review process (Gemini, ChatGPT, Claude) with
consensus-based rating for atomic insights.

Now supports collaborative modification during all rounds:
- Any LLM can propose modifications in Round 1, 2, or 3
- ABANDON vote allows early exit for unsalvageable chunks
- Modifications are evaluated for consensus between rounds
"""

from .models import (
    ReviewResponse,
    ReviewRound,
    ChunkReview,
    MindChange,
    ReviewDecision,
    RefinementRecord,
    VerificationResult,
    detect_mind_changes,
    get_rating_pattern,
)
from .reviewer import AutomatedReviewer
from .claude_api import ClaudeReviewer, get_claude_reviewer
from .round1 import run_round1
from .round2 import run_round2
from .round3 import run_round3
from .round4 import run_round4
from .deepseek_verifier import DeepSeekVerifier, get_deepseek_verifier
from .math_triggers import needs_math_verification, get_verification_priority

__all__ = [
    # Models
    "ReviewResponse",
    "ReviewRound",
    "ChunkReview",
    "MindChange",
    "ReviewDecision",
    "RefinementRecord",
    "VerificationResult",
    "detect_mind_changes",
    "get_rating_pattern",
    # Main coordinator
    "AutomatedReviewer",
    # Claude API
    "ClaudeReviewer",
    "get_claude_reviewer",
    # DeepSeek math verification
    "DeepSeekVerifier",
    "get_deepseek_verifier",
    "needs_math_verification",
    "get_verification_priority",
    # Rounds (for advanced usage)
    "run_round1",
    "run_round2",
    "run_round3",
    "run_round4",
]
