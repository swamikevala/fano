"""
LLM Library - Unified interface for LLM access.

This library provides a clean API for all LLM interactions:
- Single LLM calls (via Pool service)
- Multi-LLM consensus for reliable validation
- Automatic rate limiting and queueing
- Deep mode management

Usage:
    from llm import LLMClient, ConsensusReviewer

    # Single LLM call
    client = LLMClient()
    response = await client.send("gemini", "Your prompt here")

    # Multi-LLM consensus
    reviewer = ConsensusReviewer(client)
    result = await reviewer.review(insight_text, tags=["math"])
"""

from .src.client import LLMClient, PoolUnavailableError
from .src.models import (
    LLMResponse,
    Backend,
    Priority,
    PoolStatus,
    BackendStatus,
)
from .src.consensus import ConsensusReviewer
from .src.adapters import (
    BrowserAdapter,
    GeminiAdapter,
    ChatGPTAdapter,
    ClaudeAdapter,
    create_adapters,
)

__all__ = [
    "LLMClient",
    "PoolUnavailableError",
    "LLMResponse",
    "Backend",
    "Priority",
    "PoolStatus",
    "BackendStatus",
    "ConsensusReviewer",
    "BrowserAdapter",
    "GeminiAdapter",
    "ChatGPTAdapter",
    "ClaudeAdapter",
    "create_adapters",
]

__version__ = "0.1.0"
