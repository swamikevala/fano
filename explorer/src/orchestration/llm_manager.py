"""
LLM Manager - Handles LLM connections and communication.

This module centralizes:
- LLM client initialization
- Model availability checking
- Unified message sending interface
"""

import asyncio
import os
from pathlib import Path
from typing import Optional, Any

import yaml

from shared.logging import get_logger

from llm import LLMClient, GeminiAdapter, ChatGPTAdapter, ClaudeAdapter, DeepSeekAdapter
from explorer.src.models import ExplorationThread
from explorer.src.storage import ExplorerPaths

log = get_logger("explorer", "orchestration.llm")


def _load_llm_config() -> dict:
    """Load LLM config from config.yaml."""
    config_path = Path(__file__).resolve().parent.parent.parent.parent / "config.yaml"
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return config.get("llm", {})
    return {}


class LLMManager:
    """
    Manages LLM connections and provides a unified interface for message sending.

    All LLMs are accessed via OpenRouter API - no browser automation needed.
    """

    def __init__(self, config: dict, paths: ExplorerPaths):
        """
        Initialize LLM manager.

        Args:
            config: Full configuration dict
            paths: ExplorerPaths instance for data directories
        """
        self.config = config
        self.paths = paths

        # LLM client and adapters
        self.llm_client: Optional[LLMClient] = None
        self.chatgpt: Optional[ChatGPTAdapter] = None
        self.gemini: Optional[GeminiAdapter] = None
        self.claude: Optional[ClaudeAdapter] = None
        self.deepseek: Optional[DeepSeekAdapter] = None

    async def connect(self) -> bool:
        """
        Initialize LLM client and adapters.

        Returns:
            True if API key is configured and client is ready.
        """
        # Load LLM config
        llm_config = _load_llm_config()
        models = llm_config.get("models", {})

        # Initialize client with configured models
        self.llm_client = LLMClient(
            openrouter_api_key=os.environ.get("OPENROUTER_API_KEY"),
            models=models if models else None,
        )

        # Check if API key is configured
        if not self.llm_client._openrouter_key:
            log.warning("llm.manager.no_api_key",
                       message="OPENROUTER_API_KEY not set. LLMs will not be available.")
            return False

        # Create adapters
        self.chatgpt = ChatGPTAdapter(self.llm_client)
        self.gemini = GeminiAdapter(self.llm_client)
        self.claude = ClaudeAdapter(self.llm_client)
        self.deepseek = DeepSeekAdapter(self.llm_client)

        # Mark as connected
        await self.chatgpt.connect()
        await self.gemini.connect()
        await self.claude.connect()
        await self.deepseek.connect()

        log.info("llm.manager.connected",
                 models=list(self.llm_client.list_models().keys()))
        return True

    async def disconnect(self) -> None:
        """Disconnect from all LLM services."""
        if self.chatgpt:
            await self.chatgpt.disconnect()
        if self.gemini:
            await self.gemini.disconnect()
        if self.claude:
            await self.claude.disconnect()
        if self.deepseek:
            await self.deepseek.disconnect()
        if self.llm_client:
            await self.llm_client.close()

    async def ensure_connected(self) -> bool:
        """
        Ensure LLM client is ready.

        Returns:
            True if at least one model is available.
        """
        if self.llm_client is None:
            return await self.connect()
        return self.chatgpt is not None or self.gemini is not None

    def get_available_models(self, check_rate_limits: bool = True) -> dict[str, Any]:
        """
        Get available models as a dict.

        Args:
            check_rate_limits: Ignored (kept for compatibility).

        Returns:
            Dict mapping model name to model instance.
        """
        models = {}
        if self.chatgpt:
            models["chatgpt"] = self.chatgpt
        if self.gemini:
            models["gemini"] = self.gemini
        if self.claude:
            models["claude"] = self.claude
        if self.deepseek:
            models["deepseek"] = self.deepseek
        return models

    def get_backlog_model(self) -> tuple[Optional[str], Optional[Any]]:
        """Get an available model for backlog processing (prefers Gemini)."""
        if self.gemini:
            return ("gemini", self.gemini)
        if self.chatgpt:
            return ("chatgpt", self.chatgpt)
        if self.claude:
            return ("claude", self.claude)
        return (None, None)

    def get_other_model(self, current: str) -> Optional[tuple[str, Any]]:
        """Get a different model than the current one."""
        models = self.get_available_models()
        for name, model in models.items():
            if name != current:
                return (name, model)
        return None

    def select_model_for_task(
        self, task: str, available_models: dict[str, Any] = None
    ) -> Optional[str]:
        """
        Select a model for a specific task.

        Args:
            task: Task type ('exploration', 'critique', 'synthesis')
            available_models: Dict of available models. If None, uses get_available_models().

        Returns:
            Selected model name, or None if no models available.
        """
        if available_models is None:
            available_models = self.get_available_models()

        if not available_models:
            return None

        # Simple selection: prefer gemini for exploration, chatgpt for critique
        if task == "exploration" and "gemini" in available_models:
            return "gemini"
        if task == "critique" and "chatgpt" in available_models:
            return "chatgpt"

        # Default: return first available
        return next(iter(available_models.keys()))

    async def send_message(
        self,
        model_name: str,
        model: Any,
        prompt: str,
        thread: ExplorationThread = None,
        task_type: str = "exploration",
        images: list = None,
    ) -> tuple[str, bool]:
        """
        Send a message to an LLM.

        Args:
            model_name: Name of the model ('chatgpt', 'gemini', 'claude', 'deepseek')
            model: The model adapter instance
            prompt: The prompt to send
            thread: Optional thread for context (used for logging)
            task_type: Type of task ('exploration', 'critique', 'synthesis')
            images: Optional list of ImageAttachment objects

        Returns:
            Tuple of (response_text, deep_mode_used)
            Note: deep_mode_used is always False with API access (kept for compatibility)
        """
        thread_id = thread.id if thread else None

        log.info("llm.manager.send_message",
                 model=model_name,
                 thread_id=thread_id,
                 task_type=task_type,
                 prompt_length=len(prompt),
                 image_count=len(images) if images else 0)

        response = await model.send_message(
            prompt,
            images=images,
        )

        log.info("llm.manager.response_received",
                 model=model_name,
                 thread_id=thread_id,
                 response_length=len(response))

        # deep_mode_used is always False with API access
        return response, False
