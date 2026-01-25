"""
API-based adapters for LLM access.

These adapters provide a consistent interface for different LLM backends,
all routing through OpenRouter API.
"""

from typing import Optional

from shared.logging import get_logger

from .client import LLMClient
from .models import ImageAttachment

log = get_logger("llm", "adapters")


class APIAdapter:
    """
    Base adapter for API-based LLM access.

    Provides a simple interface compatible with existing code that expects:
    - send_message(prompt, ...)
    - connect() / disconnect()
    - last_deep_mode_used attribute
    """

    # Override in subclasses
    model_name: str = "unknown"
    backend: str = "unknown"

    def __init__(self, client: LLMClient, model: Optional[str] = None):
        """
        Initialize adapter.

        Args:
            client: LLMClient instance
            model: Override model (uses client's default for backend if not specified)
        """
        self.client = client
        self._model = model
        self._connected = False
        self.last_deep_mode_used = False  # Legacy compatibility

    async def connect(self):
        """Mark as connected (no-op for API access)."""
        self._connected = True
        log.info("llm.adapter.connected", backend=self.backend)

    async def disconnect(self):
        """Mark as disconnected (no-op for API access)."""
        self._connected = False
        log.info("llm.adapter.disconnected", backend=self.backend)

    async def start_new_chat(self):
        """Start new chat (no-op for API - each request is independent)."""
        pass

    async def send_message(
        self,
        prompt: str,
        use_deep_think: bool = False,
        use_pro_mode: bool = False,
        use_thinking_mode: bool = False,
        thread_id: Optional[str] = None,
        task_type: Optional[str] = None,
        images: Optional[list[ImageAttachment]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        timeout_seconds: int = 300,
    ) -> str:
        """
        Send message to LLM.

        Args:
            prompt: The prompt text
            use_deep_think: Legacy flag (ignored)
            use_pro_mode: Legacy flag (ignored)
            use_thinking_mode: Legacy flag (ignored)
            thread_id: Legacy flag (ignored)
            task_type: Legacy flag (ignored)
            images: Optional image attachments
            system_prompt: Optional system message
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            timeout_seconds: Request timeout

        Returns:
            Response text

        Raises:
            RuntimeError: If the request fails
        """
        log.info("llm.adapter.send",
                 backend=self.backend,
                 prompt_length=len(prompt),
                 image_count=len(images) if images else 0)

        response = await self.client.send(
            self.backend,
            prompt,
            model=self._model,
            system_prompt=system_prompt,
            images=images,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )

        if not response.success:
            error_msg = f"{response.error}: {response.message}"
            log.error("llm.adapter.request_failed",
                     backend=self.backend,
                     error=response.error,
                     message=response.message)
            raise RuntimeError(error_msg)

        log.info("llm.adapter.send_complete",
                 backend=self.backend,
                 response_length=len(response.text or ""),
                 duration_seconds=response.response_time_seconds)

        return response.text or ""

    def is_available(self) -> bool:
        """Check if backend is available (has API key configured)."""
        return bool(self.client._openrouter_key)


class GeminiAdapter(APIAdapter):
    """Adapter for Gemini via OpenRouter."""

    model_name = "gemini"
    backend = "gemini"

    def __init__(self, client: LLMClient, model: Optional[str] = None):
        super().__init__(client, model)

    async def enable_deep_think(self):
        """Legacy method (no-op)."""
        pass


class ChatGPTAdapter(APIAdapter):
    """Adapter for ChatGPT/GPT-4 via OpenRouter."""

    model_name = "chatgpt"
    backend = "chatgpt"

    def __init__(self, client: LLMClient, model: Optional[str] = None):
        super().__init__(client, model)

    async def enable_pro_mode(self):
        """Legacy method (no-op)."""
        pass

    async def enable_thinking_mode(self):
        """Legacy method (no-op)."""
        pass


class ClaudeAdapter(APIAdapter):
    """Adapter for Claude via OpenRouter."""

    model_name = "claude"
    backend = "claude"

    def __init__(self, client: LLMClient, model: Optional[str] = None):
        super().__init__(client, model)

    async def enable_extended_thinking(self):
        """Legacy method (no-op)."""
        pass


class DeepSeekAdapter(APIAdapter):
    """Adapter for DeepSeek via OpenRouter."""

    model_name = "deepseek"
    backend = "deepseek"

    def __init__(self, client: LLMClient, model: Optional[str] = None):
        super().__init__(client, model)


# Legacy aliases for backward compatibility
BrowserAdapter = APIAdapter
GeminiAPIAdapter = GeminiAdapter
ChatGPTAPIAdapter = ChatGPTAdapter
ClaudeAPIAdapter = ClaudeAdapter


def create_adapters(client: LLMClient) -> dict[str, APIAdapter]:
    """
    Create all adapters from an LLMClient.

    Returns:
        Dict with 'gemini', 'chatgpt', 'claude', 'deepseek' adapters
    """
    return {
        "gemini": GeminiAdapter(client),
        "chatgpt": ChatGPTAdapter(client),
        "claude": ClaudeAdapter(client),
        "deepseek": DeepSeekAdapter(client),
    }
