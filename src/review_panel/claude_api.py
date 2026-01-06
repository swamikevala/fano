"""
Claude API client for the review panel.

Provides Claude access via the Anthropic API for review panel operations.
"""

import asyncio
import logging
import os
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


class ClaudeReviewer:
    """
    Claude API client for review panel.

    Uses the Anthropic Python SDK to communicate with Claude.
    Supports extended thinking for deep analysis rounds.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-opus-4-20250514",
    ):
        """
        Initialize the Claude reviewer.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Claude model to use
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.client = None
        self._initialized = False

    def _ensure_client(self):
        """Lazily initialize the Anthropic client."""
        if self._initialized:
            return

        if not self.api_key:
            raise ValueError(
                "Claude API key not provided. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key to ClaudeReviewer."
            )

        try:
            import anthropic
            import httpx

            # Check for SSL bypass flag (for corporate proxies)
            disable_ssl = os.environ.get("ANTHROPIC_DISABLE_SSL_VERIFY", "").lower() in ("1", "true", "yes")

            if disable_ssl:
                logger.warning("[claude] SSL verification DISABLED - use only for corporate proxy issues")
                # Create custom httpx client without SSL verification
                http_client = httpx.Client(verify=False)
                self.client = anthropic.Anthropic(api_key=self.api_key, http_client=http_client)
            else:
                self.client = anthropic.Anthropic(api_key=self.api_key)

            self._initialized = True
            logger.info(f"[claude] Initialized with model {self.model}")
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            )

    async def _stream_extended_thinking(self, prompt: str) -> str:
        """
        Stream an extended thinking request.

        Streaming is required for operations that may take longer than 10 minutes.
        """
        def _do_stream():
            text_content = ""
            with self.client.messages.stream(
                model=self.model,
                max_tokens=16000,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 10000,
                },
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for event in stream:
                    # Collect text content from content_block_delta events
                    if hasattr(event, "type"):
                        if event.type == "content_block_delta":
                            delta = event.delta
                            if hasattr(delta, "text"):
                                text_content += delta.text
            return text_content

        return await asyncio.to_thread(_do_stream)

    async def send_message(
        self,
        prompt: str,
        extended_thinking: bool = False,
        max_tokens: int = 4096,
    ) -> str:
        """
        Send a message to Claude and get a response.

        Args:
            prompt: The prompt to send
            extended_thinking: Whether to use extended thinking mode
            max_tokens: Maximum tokens in response

        Returns:
            Claude's response text
        """
        self._ensure_client()

        logger.info(f"[claude] Sending message ({len(prompt)} chars, extended_thinking={extended_thinking})")

        # Retry logic for connection errors
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                # Recreate client on retry to get fresh connection
                if attempt > 0:
                    import anthropic
                    self.client = anthropic.Anthropic(api_key=self.api_key)
                    logger.info(f"[claude] Recreated client for retry attempt {attempt + 1}")

                if extended_thinking:
                    # Use streaming for extended thinking (required for long operations)
                    text_content = await self._stream_extended_thinking(prompt)
                else:
                    # Standard message (non-streaming)
                    response = await asyncio.to_thread(
                        self.client.messages.create,
                        model=self.model,
                        max_tokens=max_tokens,
                        messages=[{"role": "user", "content": prompt}],
                    )

                    # Extract text from response
                    text_content = ""
                    for block in response.content:
                        if hasattr(block, "text"):
                            text_content += block.text

                logger.info(f"[claude] Got response ({len(text_content)} chars)")
                return text_content

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Only retry on connection errors
                if "connection" in error_str or "timeout" in error_str:
                    wait_time = 2 ** attempt  # 1, 2, 4 seconds
                    logger.warning(f"[claude] Connection error (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    # Non-connection error, don't retry
                    logger.error(f"[claude] Error: {type(e).__name__}: {e}")
                    raise

        # All retries exhausted
        logger.error(f"[claude] All {max_retries} attempts failed: {last_error}")
        raise last_error

    async def review(
        self,
        prompt: str,
        extended_thinking: bool = False,
    ) -> str:
        """
        Alias for send_message for review operations.

        Args:
            prompt: Review prompt
            extended_thinking: Whether to use extended thinking

        Returns:
            Review response text
        """
        return await self.send_message(prompt, extended_thinking=extended_thinking)

    def is_available(self) -> bool:
        """Check if Claude API is available."""
        try:
            self._ensure_client()
            return True
        except (ValueError, ImportError) as e:
            logger.warning(f"[claude] is_available check failed: {e}")
            return False
        except Exception as e:
            logger.error(f"[claude] Unexpected error in is_available: {e}")
            return False


def get_claude_reviewer(config: dict = None) -> Optional[ClaudeReviewer]:
    """
    Factory function to create a ClaudeReviewer from config.

    Args:
        config: Review panel configuration

    Returns:
        ClaudeReviewer instance or None if not configured
    """
    config = config or {}

    # Get API key from environment variable specified in config
    api_key_env = config.get("claude_api_key_env", "ANTHROPIC_API_KEY")
    api_key = os.environ.get(api_key_env)

    if not api_key:
        logger.warning(f"[claude] {api_key_env} not set, Claude review unavailable")
        return None

    model = config.get("claude_model", "claude-opus-4-20250514")

    return ClaudeReviewer(api_key=api_key, model=model)
