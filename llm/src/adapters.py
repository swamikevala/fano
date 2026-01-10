"""
Adapters that mimic browser interfaces but use LLMClient.

These adapters allow existing code (orchestrator, review_panel, etc.)
to work with the new LLM library without major refactoring.
"""

import sys
from pathlib import Path
from typing import Optional

# Add shared module to path
SHARED_PATH = Path(__file__).resolve().parent.parent.parent / "shared"
sys.path.insert(0, str(SHARED_PATH.parent))

from shared.logging import get_logger

from .client import LLMClient

log = get_logger("llm", "adapters")


class BrowserAdapter:
    """
    Base adapter that mimics browser interface methods.

    Existing code expects objects with:
    - start_new_chat()
    - send_message(prompt, use_deep_think=False, ...)
    - disconnect()
    - last_deep_mode_used attribute

    This adapter translates those calls to LLMClient.
    """

    def __init__(self, client: LLMClient, backend: str):
        """
        Initialize adapter.

        Args:
            client: LLMClient instance
            backend: Backend name ("gemini", "chatgpt", "claude")
        """
        self.client = client
        self.backend = backend
        self.last_deep_mode_used = False
        self._connected = False

    async def connect(self):
        """Mark as connected. Pool handles actual connection."""
        self._connected = True
        log.info("llm.adapter.lifecycle", action="connected", backend=self.backend)

    async def disconnect(self):
        """Mark as disconnected."""
        self._connected = False
        log.info("llm.adapter.lifecycle", action="disconnected", backend=self.backend)

    async def start_new_chat(self):
        """Start new chat - passed through to pool."""
        # Pool handles new_chat via the request options
        pass

    async def send_message(
        self,
        prompt: str,
        use_deep_think: bool = False,
        use_pro_mode: bool = False,
        use_thinking_mode: bool = False,
    ) -> str:
        """
        Send message to LLM.

        Args:
            prompt: The prompt text
            use_deep_think: Use Gemini Deep Think mode
            use_pro_mode: Use ChatGPT Pro mode
            use_thinking_mode: Use ChatGPT Thinking mode (ignored if pro_mode)

        Returns:
            Response text
        """
        # Determine deep mode based on backend-specific flags
        deep_mode = use_deep_think or use_pro_mode

        response = await self.client.send(
            self.backend,
            prompt,
            deep_mode=deep_mode,
            new_chat=True,  # Each send starts fresh
        )

        # Track whether deep mode was actually used
        self.last_deep_mode_used = response.deep_mode_used

        if not response.success:
            error_msg = f"{response.error}: {response.message}"
            log.error("llm.adapter.request_failed", backend=self.backend, error=response.error, message=response.message)
            raise RuntimeError(error_msg)

        return response.text or ""

    def is_available(self) -> bool:
        """Check if backend is available."""
        # This is sync, so we can't check pool status
        # Assume available if connected
        return self._connected


class GeminiAdapter(BrowserAdapter):
    """Adapter that mimics GeminiInterface."""

    model_name = "gemini"

    def __init__(self, client: LLMClient):
        super().__init__(client, "gemini")

    async def enable_deep_think(self):
        """Deep think is enabled via send_message flag."""
        pass

    def _check_rate_limit(self, response: str) -> bool:
        """Check for rate limit signals in response."""
        rate_limit_signals = [
            "try again tomorrow",
            "quota exceeded",
            "rate limit",
            "too many requests",
        ]
        response_lower = response.lower()
        return any(signal in response_lower for signal in rate_limit_signals)


class ChatGPTAdapter(BrowserAdapter):
    """Adapter that mimics ChatGPTInterface."""

    model_name = "chatgpt"

    def __init__(self, client: LLMClient):
        super().__init__(client, "chatgpt")

    async def enable_pro_mode(self):
        """Pro mode is enabled via send_message flag."""
        pass

    async def enable_thinking_mode(self):
        """Thinking mode is enabled via send_message flag."""
        pass

    def _check_rate_limit(self, response: str) -> bool:
        """Check for rate limit signals in response."""
        rate_limit_signals = [
            "usage cap",
            "rate limit",
            "too many requests",
            "reached the limit",
        ]
        response_lower = response.lower()
        return any(signal in response_lower for signal in rate_limit_signals)


class ClaudeAdapter(BrowserAdapter):
    """Adapter for Claude API (direct, not via pool)."""

    model_name = "claude"

    def __init__(self, client: LLMClient, model: str = "claude-sonnet-4-20250514"):
        super().__init__(client, "claude")
        self.model = model

    async def send_message(
        self,
        prompt: str,
        use_deep_think: bool = False,
        use_pro_mode: bool = False,
        use_thinking_mode: bool = False,
    ) -> str:
        """Send message to Claude API."""
        response = await self.client.send(
            "claude",
            prompt,
            model=self.model,
        )

        self.last_deep_mode_used = False  # Claude doesn't have deep mode

        if not response.success:
            error_msg = f"{response.error}: {response.message}"
            log.error("llm.adapter.request_failed", backend="claude", error=response.error, message=response.message)
            raise RuntimeError(error_msg)

        return response.text or ""


def create_adapters(client: LLMClient) -> dict:
    """
    Create all adapters from an LLMClient.

    Returns:
        Dict with 'gemini', 'chatgpt', 'claude' adapters
    """
    return {
        "gemini": GeminiAdapter(client),
        "chatgpt": ChatGPTAdapter(client),
        "claude": ClaudeAdapter(client),
    }
