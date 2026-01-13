"""
Adapters that mimic browser interfaces but use LLMClient.

These adapters allow existing code (orchestrator, review_panel, etc.)
to work with the new LLM library without major refactoring.
"""

import uuid
from pathlib import Path
from typing import Optional

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

    This adapter translates those calls to LLMClient using the async job system
    for robust handling of timeouts and pool restarts.
    """

    # Override in subclasses with backend-specific rate limit signals
    rate_limit_signals: list[str] = []

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
        thread_id: Optional[str] = None,
        images: Optional[list] = None,
    ) -> str:
        """
        Send message to LLM using the async job system.

        This uses the robust async job flow which handles pool restarts
        and long-running requests gracefully.

        Args:
            prompt: The prompt text
            use_deep_think: Use Gemini Deep Think mode
            use_pro_mode: Use ChatGPT Pro mode
            use_thinking_mode: Use ChatGPT Thinking mode (ignored if pro_mode)
            thread_id: Thread ID for recovery correlation
            images: Optional list of ImageAttachment objects to include

        Returns:
            Response text
        """
        # Determine deep mode based on backend-specific flags
        deep_mode = use_deep_think or use_pro_mode

        # Generate a unique job ID
        # Use thread_id as prefix if available for easier correlation
        job_id = f"{thread_id or 'job'}-{uuid.uuid4().hex[:8]}"

        log.info("llm.adapter.send_async",
                 backend=self.backend,
                 job_id=job_id,
                 thread_id=thread_id,
                 deep_mode=deep_mode,
                 prompt_length=len(prompt),
                 image_count=len(images) if images else 0)

        response = await self.client.send_async(
            self.backend,
            prompt,
            job_id=job_id,
            thread_id=thread_id,
            deep_mode=deep_mode,
            new_chat=True,  # Each send starts fresh
            poll_interval=5.0,  # Check every 5 seconds
            timeout_seconds=3600,  # 1 hour max wait
            images=images,
        )

        # Track whether deep mode was actually used
        self.last_deep_mode_used = response.deep_mode_used

        if not response.success:
            error_msg = f"{response.error}: {response.message}"
            log.error("llm.adapter.request_failed",
                     backend=self.backend,
                     job_id=job_id,
                     error=response.error,
                     message=response.message)
            raise RuntimeError(error_msg)

        log.info("llm.adapter.send_complete",
                 backend=self.backend,
                 job_id=job_id,
                 response_length=len(response.text or ""))

        return response.text or ""

    def is_available(self) -> bool:
        """Check if backend is available."""
        # This is sync, so we can't check pool status
        # Assume available if connected
        return self._connected

    def _check_rate_limit(self, response: str) -> bool:
        """Check for rate limit signals in response."""
        response_lower = response.lower()
        return any(signal in response_lower for signal in self.rate_limit_signals)


class GeminiAdapter(BrowserAdapter):
    """Adapter that mimics GeminiInterface."""

    model_name = "gemini"
    rate_limit_signals = [
        "try again tomorrow",
        "quota exceeded",
        "rate limit",
        "too many requests",
    ]

    def __init__(self, client: LLMClient):
        super().__init__(client, "gemini")

    async def enable_deep_think(self):
        """Deep think is enabled via send_message flag."""
        pass


class ChatGPTAdapter(BrowserAdapter):
    """Adapter that mimics ChatGPTInterface."""

    model_name = "chatgpt"
    rate_limit_signals = [
        "usage cap",
        "rate limit",
        "too many requests",
        "reached the limit",
    ]

    def __init__(self, client: LLMClient):
        super().__init__(client, "chatgpt")

    async def enable_pro_mode(self):
        """Pro mode is enabled via send_message flag."""
        pass

    async def enable_thinking_mode(self):
        """Thinking mode is enabled via send_message flag."""
        pass


class ClaudeAdapter(BrowserAdapter):
    """Adapter that mimics ClaudeInterface (via pool browser)."""

    model_name = "claude"
    rate_limit_signals = [
        "rate limit",
        "too many requests",
        "usage limit",
    ]

    def __init__(self, client: LLMClient):
        super().__init__(client, "claude")

    async def enable_extended_thinking(self):
        """Extended thinking is enabled via send_message flag."""
        pass


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
