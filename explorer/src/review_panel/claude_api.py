"""
Claude API stub for backward compatibility.

This module previously provided ClaudeReviewer for direct Anthropic API access.
Now all LLM access goes through OpenRouter, so this is just a compatibility stub.
"""

from typing import Any, Optional


# Type alias for backward compatibility
ClaudeReviewer = Any


def get_claude_reviewer(config: dict = None) -> Optional[Any]:
    """
    Get Claude reviewer instance.

    Deprecated: Claude access is now via OpenRouter API.
    This function returns None - use LLMClient with ClaudeAdapter instead.

    Args:
        config: Ignored

    Returns:
        None (Claude access is via OpenRouter now)
    """
    return None
