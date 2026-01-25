"""
LLM Executor abstraction for review panel.

Provides a unified interface for sending prompts to different LLMs
via the OpenRouter API.

Thinking modes:
- "standard": Normal mode, no special reasoning
- "thinking": Light reasoning (same as standard for API access)
- "deep": Deep reasoning (same as standard for API access)

Note: With API access, thinking modes are handled by model selection
rather than runtime toggles.
"""

from abc import ABC, abstractmethod
from typing import Optional

from shared.logging import get_logger
from llm import LLMClient, APIAdapter, GeminiAdapter, ChatGPTAdapter, ClaudeAdapter, DeepSeekAdapter

log = get_logger("explorer", "review_panel.llm_executor")


class LLMExecutor(ABC):
    """Abstract base class for LLM execution."""

    name: str = "unknown"

    @abstractmethod
    async def send(self, prompt: str, thinking_mode: str = "standard") -> str:
        """
        Send a prompt to the LLM and return the response text.

        Args:
            prompt: The prompt to send
            thinking_mode: One of "standard", "thinking", or "deep"
                          (Note: with API access, these are treated the same)

        Returns:
            The LLM's response text
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this executor is available for use."""
        ...

    async def start_fresh(self) -> None:
        """Start a fresh conversation. No-op for API access."""
        pass


class APIExecutor(LLMExecutor):
    """Executor using API adapter."""

    def __init__(self, adapter: APIAdapter, name: str = None):
        """
        Initialize API executor.

        Args:
            adapter: API adapter instance
            name: Optional name override
        """
        self._adapter = adapter
        self.name = name or adapter.backend

    def is_available(self) -> bool:
        return self._adapter is not None and self._adapter.is_available()

    async def send(self, prompt: str, thinking_mode: str = "standard") -> str:
        """Send prompt via API."""
        if not self._adapter:
            raise RuntimeError(f"{self.name} adapter not available")

        log.info(
            f"llm_executor.{self.name}.send",
            thinking_mode=thinking_mode,
            prompt_length=len(prompt),
        )

        response = await self._adapter.send_message(prompt)

        log.debug(
            f"llm_executor.{self.name}.response",
            response_length=len(response),
        )

        return response


class GeminiExecutor(APIExecutor):
    """Executor for Gemini via API."""

    name = "gemini"

    def __init__(self, adapter: GeminiAdapter = None, client: LLMClient = None):
        """
        Initialize Gemini executor.

        Args:
            adapter: GeminiAdapter instance, or
            client: LLMClient to create adapter from
        """
        if adapter is None and client is not None:
            adapter = GeminiAdapter(client)
        super().__init__(adapter, "gemini")


class ChatGPTExecutor(APIExecutor):
    """Executor for ChatGPT via API."""

    name = "chatgpt"

    def __init__(self, adapter: ChatGPTAdapter = None, client: LLMClient = None):
        """
        Initialize ChatGPT executor.

        Args:
            adapter: ChatGPTAdapter instance, or
            client: LLMClient to create adapter from
        """
        if adapter is None and client is not None:
            adapter = ChatGPTAdapter(client)
        super().__init__(adapter, "chatgpt")


class ClaudeExecutor(APIExecutor):
    """Executor for Claude via API."""

    name = "claude"

    def __init__(self, adapter: ClaudeAdapter = None, client: LLMClient = None, reviewer=None):
        """
        Initialize Claude executor.

        Args:
            adapter: ClaudeAdapter instance, or
            client: LLMClient to create adapter from, or
            reviewer: Legacy ClaudeReviewer (ignored, kept for compatibility)
        """
        if adapter is None and client is not None:
            adapter = ClaudeAdapter(client)
        super().__init__(adapter, "claude")


class DeepSeekExecutor(APIExecutor):
    """Executor for DeepSeek via API."""

    name = "deepseek"

    def __init__(self, adapter: DeepSeekAdapter = None, client: LLMClient = None):
        """
        Initialize DeepSeek executor.

        Args:
            adapter: DeepSeekAdapter instance, or
            client: LLMClient to create adapter from
        """
        if adapter is None and client is not None:
            adapter = DeepSeekAdapter(client)
        super().__init__(adapter, "deepseek")


def create_executors(
    client: LLMClient = None,
    # Legacy parameters (kept for compatibility)
    gemini_browser=None,
    chatgpt_browser=None,
    claude_reviewer=None,
) -> dict[str, LLMExecutor]:
    """
    Create executor instances for available LLMs.

    Args:
        client: LLMClient instance (recommended)
        gemini_browser: Legacy parameter (ignored)
        chatgpt_browser: Legacy parameter (ignored)
        claude_reviewer: Legacy parameter (ignored)

    Returns:
        Dict mapping LLM name to executor instance
    """
    if client is None:
        return {}

    executors = {}

    gemini = GeminiExecutor(client=client)
    if gemini.is_available():
        executors["gemini"] = gemini

    chatgpt = ChatGPTExecutor(client=client)
    if chatgpt.is_available():
        executors["chatgpt"] = chatgpt

    claude = ClaudeExecutor(client=client)
    if claude.is_available():
        executors["claude"] = claude

    deepseek = DeepSeekExecutor(client=client)
    if deepseek.is_available():
        executors["deepseek"] = deepseek

    return executors


async def send_to_llm(
    llm_name: str,
    prompt: str,
    executors: dict[str, LLMExecutor],
    thinking_mode: str = "standard",
) -> str:
    """
    Send a prompt to a specific LLM.

    Convenience function for when you need to target a specific LLM.

    Args:
        llm_name: Name of the LLM ("gemini", "chatgpt", "claude", "deepseek")
        prompt: The prompt to send
        executors: Dict of available executors
        thinking_mode: One of "standard", "thinking", or "deep"

    Returns:
        The LLM's response text

    Raises:
        RuntimeError: If the specified LLM is not available
    """
    if llm_name not in executors:
        raise RuntimeError(f"LLM '{llm_name}' not available")

    return await executors[llm_name].send(prompt, thinking_mode=thinking_mode)
