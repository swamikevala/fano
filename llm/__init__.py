"""
LLM Library - Unified interface for LLM access via OpenRouter.

This library provides a clean API for all LLM interactions:
- Single LLM calls via OpenRouter API
- Multi-LLM consensus for reliable validation
- Automatic rate limiting
- Support for multiple backends (Gemini, ChatGPT, Claude, DeepSeek)

Usage:
    from llm import LLMClient, ConsensusReviewer

    # Single LLM call
    client = LLMClient()
    response = await client.send("gemini", "Your prompt here")

    # Multi-LLM consensus
    reviewer = ConsensusReviewer(client)
    result = await reviewer.review(insight_text, tags=["math"])
"""

from .src.client import LLMClient
from .src.models import (
    LLMResponse,
    Backend,
    ImageAttachment,
)
from .src.consensus import ConsensusReviewer
from .src.adapters import (
    APIAdapter,
    GeminiAdapter,
    ChatGPTAdapter,
    ClaudeAdapter,
    DeepSeekAdapter,
    create_adapters,
    # Legacy aliases
    BrowserAdapter,
)

__all__ = [
    "LLMClient",
    "LLMResponse",
    "Backend",
    "ImageAttachment",
    "ConsensusReviewer",
    "APIAdapter",
    "GeminiAdapter",
    "ChatGPTAdapter",
    "ClaudeAdapter",
    "DeepSeekAdapter",
    "create_adapters",
    # Legacy
    "BrowserAdapter",
]

__version__ = "0.2.0"
