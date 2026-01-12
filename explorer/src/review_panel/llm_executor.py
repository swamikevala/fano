"""
LLM Executor abstraction for review panel.

Provides a unified interface for sending prompts to different LLMs,
abstracting away the differences between browser-based (Gemini, ChatGPT)
and API-based (Claude) implementations.

Thinking modes:
- "standard": Normal mode, no special reasoning
- "thinking": Light reasoning (ChatGPT Thinking mode)
- "deep": Deep reasoning (Gemini Deep Think, ChatGPT Pro, Claude Extended Thinking)
"""

from abc import ABC, abstractmethod
from typing import Optional

from shared.logging import get_logger

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

        Returns:
            The LLM's response text
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this executor is available for use."""
        ...

    async def start_fresh(self) -> None:
        """Start a fresh conversation. Override if needed."""
        pass


class GeminiExecutor(LLMExecutor):
    """Executor for Gemini via browser automation."""

    name = "gemini"

    def __init__(self, browser):
        """
        Initialize Gemini executor.

        Args:
            browser: GeminiBrowser instance
        """
        self._browser = browser

    def is_available(self) -> bool:
        return self._browser is not None

    async def start_fresh(self) -> None:
        """Start a new chat to clear context."""
        if self._browser:
            await self._browser.start_new_chat()

    async def send(self, prompt: str, thinking_mode: str = "standard") -> str:
        """Send prompt to Gemini."""
        if not self._browser:
            raise RuntimeError("Gemini browser not available")

        log.info(
            "llm_executor.gemini.send",
            thinking_mode=thinking_mode,
            prompt_length=len(prompt),
        )

        # Start fresh chat
        await self._browser.start_new_chat()

        # Enable deep think mode if requested
        if thinking_mode == "deep":
            await self._browser.enable_deep_think()

        # Send and return response
        response = await self._browser.send_message(prompt)

        log.debug(
            "llm_executor.gemini.response",
            response_length=len(response),
        )

        return response


class ChatGPTExecutor(LLMExecutor):
    """Executor for ChatGPT via browser automation."""

    name = "chatgpt"

    def __init__(self, browser):
        """
        Initialize ChatGPT executor.

        Args:
            browser: ChatGPTBrowser instance
        """
        self._browser = browser

    def is_available(self) -> bool:
        return self._browser is not None

    async def start_fresh(self) -> None:
        """Start a new chat to clear context."""
        if self._browser:
            await self._browser.start_new_chat()

    async def send(self, prompt: str, thinking_mode: str = "standard") -> str:
        """Send prompt to ChatGPT."""
        if not self._browser:
            raise RuntimeError("ChatGPT browser not available")

        log.info(
            "llm_executor.chatgpt.send",
            thinking_mode=thinking_mode,
            prompt_length=len(prompt),
        )

        # Start fresh chat
        await self._browser.start_new_chat()

        # Configure mode based on thinking_mode
        use_thinking = thinking_mode in ("thinking", "deep")

        if thinking_mode == "deep":
            # Try to enable Pro mode for deep thinking
            try:
                await self._browser.enable_pro_mode()
            except Exception as e:
                log.warning(
                    "llm_executor.chatgpt.pro_mode_failed",
                    error=str(e),
                )

        # Send with appropriate thinking mode
        response = await self._browser.send_message(prompt, use_thinking_mode=use_thinking)

        log.debug(
            "llm_executor.chatgpt.response",
            response_length=len(response),
        )

        return response


class ClaudeExecutor(LLMExecutor):
    """Executor for Claude via API."""

    name = "claude"

    def __init__(self, reviewer):
        """
        Initialize Claude executor.

        Args:
            reviewer: ClaudeReviewer instance
        """
        self._reviewer = reviewer

    def is_available(self) -> bool:
        return self._reviewer is not None and self._reviewer.is_available()

    async def send(self, prompt: str, thinking_mode: str = "standard") -> str:
        """Send prompt to Claude."""
        if not self._reviewer:
            raise RuntimeError("Claude reviewer not available")

        log.info(
            "llm_executor.claude.send",
            thinking_mode=thinking_mode,
            prompt_length=len(prompt),
        )

        # Use extended thinking for deep mode
        extended_thinking = thinking_mode == "deep"

        response = await self._reviewer.send_message(
            prompt,
            extended_thinking=extended_thinking,
        )

        log.debug(
            "llm_executor.claude.response",
            response_length=len(response),
        )

        return response


def create_executors(
    gemini_browser=None,
    chatgpt_browser=None,
    claude_reviewer=None,
) -> dict[str, LLMExecutor]:
    """
    Create executor instances for available LLMs.

    Args:
        gemini_browser: Optional GeminiBrowser instance
        chatgpt_browser: Optional ChatGPTBrowser instance
        claude_reviewer: Optional ClaudeReviewer instance

    Returns:
        Dict mapping LLM name to executor instance
    """
    executors = {}

    if gemini_browser:
        executors["gemini"] = GeminiExecutor(gemini_browser)

    if chatgpt_browser:
        executors["chatgpt"] = ChatGPTExecutor(chatgpt_browser)

    if claude_reviewer:
        executor = ClaudeExecutor(claude_reviewer)
        if executor.is_available():
            executors["claude"] = executor

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
        llm_name: Name of the LLM ("gemini", "chatgpt", "claude")
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
