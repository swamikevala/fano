"""Data models for the LLM library."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class Backend(str, Enum):
    """Available LLM backends."""
    GEMINI = "gemini"
    CHATGPT = "chatgpt"
    CLAUDE = "claude"


class Priority(str, Enum):
    """Request priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


@dataclass
class ImageAttachment:
    """An image attachment to include with a prompt."""
    filename: str  # Original filename
    data: str  # Base64-encoded image data
    media_type: str  # MIME type, e.g., "image/png", "image/jpeg"

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "data": self.data,
            "media_type": self.media_type,
        }

    @classmethod
    def from_file(cls, filepath: str) -> "ImageAttachment":
        """Create from a file path."""
        import base64
        from pathlib import Path

        path = Path(filepath)
        ext = path.suffix.lower()
        media_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }

        data = base64.b64encode(path.read_bytes()).decode("utf-8")
        return cls(
            filename=path.name,
            data=data,
            media_type=media_types.get(ext, "application/octet-stream"),
        )


@dataclass
class LLMResponse:
    """Response from an LLM request."""
    success: bool
    text: Optional[str] = None
    error: Optional[str] = None
    message: Optional[str] = None

    # Metadata
    backend: Optional[str] = None
    deep_mode_used: bool = False
    response_time_seconds: float = 0.0
    session_id: Optional[str] = None

    # Rate limiting
    retry_after_seconds: Optional[int] = None

    @classmethod
    def from_pool_response(cls, data: dict) -> "LLMResponse":
        """Create from pool service response."""
        metadata = data.get("metadata", {}) or {}
        return cls(
            success=data.get("success", False),
            text=data.get("response"),
            error=data.get("error"),
            message=data.get("message"),
            backend=metadata.get("backend"),
            deep_mode_used=metadata.get("deep_mode_used", False),
            response_time_seconds=metadata.get("response_time_seconds", 0.0),
            session_id=metadata.get("session_id"),
            retry_after_seconds=data.get("retry_after_seconds"),
        )


@dataclass
class BackendStatus:
    """Status of a single backend."""
    available: bool
    authenticated: bool
    rate_limited: bool
    rate_limit_resets_at: Optional[datetime] = None
    queue_depth: int = 0
    deep_mode_uses_today: Optional[int] = None
    deep_mode_limit: Optional[int] = None
    pro_mode_uses_today: Optional[int] = None
    pro_mode_limit: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict) -> "BackendStatus":
        """Create from dict."""
        reset_at = None
        if data.get("rate_limit_resets_at"):
            try:
                reset_at = datetime.fromisoformat(data["rate_limit_resets_at"])
            except (ValueError, TypeError):
                pass

        return cls(
            available=data.get("available", False),
            authenticated=data.get("authenticated", False),
            rate_limited=data.get("rate_limited", False),
            rate_limit_resets_at=reset_at,
            queue_depth=data.get("queue_depth", 0),
            deep_mode_uses_today=data.get("deep_mode_uses_today"),
            deep_mode_limit=data.get("deep_mode_limit"),
            pro_mode_uses_today=data.get("pro_mode_uses_today"),
            pro_mode_limit=data.get("pro_mode_limit"),
        )


@dataclass
class PoolStatus:
    """Status of the entire pool."""
    gemini: Optional[BackendStatus] = None
    chatgpt: Optional[BackendStatus] = None
    claude: Optional[BackendStatus] = None

    @classmethod
    def from_dict(cls, data: dict) -> "PoolStatus":
        """Create from dict."""
        return cls(
            gemini=BackendStatus.from_dict(data["gemini"]) if data.get("gemini") else None,
            chatgpt=BackendStatus.from_dict(data["chatgpt"]) if data.get("chatgpt") else None,
            claude=BackendStatus.from_dict(data["claude"]) if data.get("claude") else None,
        )

    def get_available_backends(self) -> list[str]:
        """Get list of available backend names."""
        available = []
        if self.gemini and self.gemini.available:
            available.append("gemini")
        if self.chatgpt and self.chatgpt.available:
            available.append("chatgpt")
        if self.claude and self.claude.available:
            available.append("claude")
        return available


# --- Consensus Models ---

@dataclass
class ReviewResponse:
    """A single reviewer's response in a consensus review."""
    llm: str  # "gemini" | "chatgpt" | "claude"
    mode: str  # "standard" | "deep_think" | "pro" | "thinking"
    rating: str  # "bless" | "uncertain" | "reject"
    reasoning: str
    confidence: str  # "high" | "medium" | "low"

    # Analysis fields
    mathematical_verification: str = ""
    structural_analysis: str = ""
    naturalness_assessment: str = ""

    # Modification proposal
    proposed_modification: Optional[str] = None
    modification_rationale: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "llm": self.llm,
            "mode": self.mode,
            "rating": self.rating,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "mathematical_verification": self.mathematical_verification,
            "structural_analysis": self.structural_analysis,
            "naturalness_assessment": self.naturalness_assessment,
            "proposed_modification": self.proposed_modification,
            "modification_rationale": self.modification_rationale,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReviewResponse":
        return cls(
            llm=data["llm"],
            mode=data["mode"],
            rating=data["rating"],
            reasoning=data["reasoning"],
            confidence=data["confidence"],
            mathematical_verification=data.get("mathematical_verification", ""),
            structural_analysis=data.get("structural_analysis", ""),
            naturalness_assessment=data.get("naturalness_assessment", ""),
            proposed_modification=data.get("proposed_modification"),
            modification_rationale=data.get("modification_rationale"),
        )


@dataclass
class ConsensusResult:
    """Result of a multi-LLM consensus review."""
    success: bool
    final_rating: str  # "bless" | "uncertain" | "reject"
    is_unanimous: bool
    is_disputed: bool

    # Responses from each round
    rounds: list[dict] = field(default_factory=list)

    # Final insight text (if modified during review)
    final_text: Optional[str] = None

    # Mind changes during deliberation
    mind_changes: list[dict] = field(default_factory=list)

    # Timing
    review_duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "final_rating": self.final_rating,
            "is_unanimous": self.is_unanimous,
            "is_disputed": self.is_disputed,
            "rounds": self.rounds,
            "final_text": self.final_text,
            "mind_changes": self.mind_changes,
            "review_duration_seconds": self.review_duration_seconds,
        }


@dataclass
class ConsensusRunResult:
    """Result of a general-purpose consensus task."""
    success: bool
    outcome: str  # The synthesized answer/decision/content
    converged: bool  # Did all LLMs agree?
    confidence: float  # 0.0-1.0 based on agreement strength

    # Full transcript of all rounds
    rounds: list[dict] = field(default_factory=list)

    # Minority view if not converged
    dissent: Optional[str] = None

    # Selection stats (when select_best=True was used)
    selection_stats: Optional[dict] = None

    # Timing
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "outcome": self.outcome,
            "converged": self.converged,
            "confidence": self.confidence,
            "rounds": self.rounds,
            "dissent": self.dissent,
            "selection_stats": self.selection_stats,
            "duration_seconds": self.duration_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConsensusRunResult":
        return cls(
            success=data["success"],
            outcome=data["outcome"],
            converged=data["converged"],
            confidence=data["confidence"],
            rounds=data.get("rounds", []),
            dissent=data.get("dissent"),
            selection_stats=data.get("selection_stats"),
            duration_seconds=data.get("duration_seconds", 0.0),
        )
