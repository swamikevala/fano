"""
LLM Manager - Handles LLM connections and communication.

This module centralizes:
- LLM client initialization and connection
- Model availability checking
- Deep mode handling
- Response recovery from pool restarts
- Unified message sending with deep mode support
"""

import json
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

import yaml

from shared.logging import get_logger

from llm import LLMClient, GeminiAdapter, ChatGPTAdapter


def _get_pool_url() -> str:
    """Load pool URL from config.yaml."""
    config_path = Path(__file__).resolve().parent.parent.parent.parent / "config.yaml"
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        pool_config = config.get("llm", {}).get("pool", {})
        host = pool_config.get("host", "127.0.0.1")
        port = pool_config.get("port", 9000)
        return f"http://{host}:{port}"
    return "http://127.0.0.1:9000"

from explorer.src.browser import (
    rate_tracker,
    deep_mode_tracker,
    select_model,
    should_use_deep_mode,
)
from explorer.src.models import ExplorationThread, ExchangeRole
from explorer.src.storage import ExplorerPaths

log = get_logger("explorer", "orchestration.llm")


class LLMManager:
    """
    Manages LLM connections and provides a unified interface for message sending.

    Handles:
    - Connection/disconnection of browser-based LLMs (ChatGPT, Gemini)
    - Rate limit checking
    - Deep mode selection and tracking
    - Response recovery from pool restarts
    """

    def __init__(self, config: dict, paths: ExplorerPaths):
        """
        Initialize LLM manager.

        Args:
            config: Full configuration dict (needs 'llm' and 'review_panel' sections)
            paths: ExplorerPaths instance for data directories
        """
        self.config = config
        self.paths = paths

        # LLM client and adapters
        self.llm_client: Optional[LLMClient] = None
        self.chatgpt: Optional[ChatGPTAdapter] = None
        self.gemini: Optional[GeminiAdapter] = None

    async def connect(self) -> bool:
        """
        Connect to LLMs via the pool service.

        Returns:
            True if at least one model is available.
        """
        pool_url = _get_pool_url()
        self.llm_client = LLMClient(pool_url=pool_url)

        # Check if pool service is available
        pool_available = await self.llm_client.is_pool_available()
        if not pool_available:
            log.warning(
                "Pool service not available. Start it with: cd pool && python browser_pool.py start"
            )
            log.warning("Browser-based LLMs (Gemini, ChatGPT) will not be available.")
            return False

        # Connect to ChatGPT
        try:
            self.chatgpt = ChatGPTAdapter(self.llm_client)
            await self.chatgpt.connect()
            log.info("Connected to ChatGPT (via pool)")
        except Exception as e:
            log.warning(f"Could not connect to ChatGPT: {e}")
            self.chatgpt = None

        # Connect to Gemini
        try:
            self.gemini = GeminiAdapter(self.llm_client)
            await self.gemini.connect()
            log.info("Connected to Gemini (via pool)")
        except Exception as e:
            log.warning(f"Could not connect to Gemini: {e}")
            self.gemini = None

        return self.chatgpt is not None or self.gemini is not None

    async def disconnect(self) -> None:
        """Disconnect from all LLM services."""
        if self.chatgpt:
            await self.chatgpt.disconnect()
        if self.gemini:
            await self.gemini.disconnect()
        if self.llm_client:
            await self.llm_client.close()

    def get_available_models(self, check_rate_limits: bool = True) -> dict[str, Any]:
        """
        Get available models as a dict.

        Args:
            check_rate_limits: If True, exclude rate-limited models.

        Returns:
            Dict mapping model name to model instance.
        """
        models = {}
        if self.chatgpt and (not check_rate_limits or rate_tracker.is_available("chatgpt")):
            models["chatgpt"] = self.chatgpt
        if self.gemini and (not check_rate_limits or rate_tracker.is_available("gemini")):
            models["gemini"] = self.gemini
        return models

    def get_backlog_model(self) -> tuple[Optional[str], Optional[Any]]:
        """Get an available model for backlog processing (prefers Gemini)."""
        if self.gemini and rate_tracker.is_available("gemini"):
            return ("gemini", self.gemini)
        if self.chatgpt and rate_tracker.is_available("chatgpt"):
            return ("chatgpt", self.chatgpt)
        return (None, None)

    def get_other_model(self, current: str) -> Optional[tuple[str, Any]]:
        """Get a different model than the current one."""
        if current == "chatgpt" and self.gemini and rate_tracker.is_available("gemini"):
            return ("gemini", self.gemini)
        if current == "gemini" and self.chatgpt and rate_tracker.is_available("chatgpt"):
            return ("chatgpt", self.chatgpt)
        return None

    def select_model_for_task(
        self, task: str, available_models: dict[str, Any] = None
    ) -> Optional[str]:
        """
        Select a model for a specific task using weighted selection.

        Args:
            task: Task type ('exploration', 'critique', 'synthesis')
            available_models: Dict of available models. If None, uses get_available_models().

        Returns:
            Selected model name, or None if no models available.
        """
        if available_models is None:
            available_models = self.get_available_models()
        return select_model(task, available_models)

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
        Send a message to an LLM with deep mode handling.

        This is the unified entry point for all LLM communication, handling:
        - Deep mode selection based on task and model
        - Different parameter names for ChatGPT vs Gemini
        - Deep mode usage tracking
        - Thread ID passing for recovery
        - Image attachment passing

        Args:
            model_name: Name of the model ('chatgpt' or 'gemini')
            model: The model adapter instance
            prompt: The prompt to send
            thread: Optional thread for context (used for deep mode decision and recovery)
            task_type: Type of task ('exploration', 'critique', 'synthesis')
            images: Optional list of ImageAttachment objects to include with the prompt

        Returns:
            Tuple of (response_text, deep_mode_used)
        """
        # Determine if we should use deep mode
        use_deep = should_use_deep_mode(model_name, thread, task_type) if thread else False
        thread_id = thread.id if thread else None

        await model.start_new_chat()

        # Send with model-specific parameters
        if model_name == "chatgpt":
            response = await model.send_message(
                prompt,
                use_pro_mode=use_deep,
                use_thinking_mode=not use_deep,
                thread_id=thread_id,
                images=images,
            )
        else:
            response = await model.send_message(
                prompt,
                use_deep_think=use_deep,
                thread_id=thread_id,
                images=images,
            )

        # Check if deep mode was actually used and record it
        deep_mode_used = getattr(model, "last_deep_mode_used", False)
        if deep_mode_used:
            mode_key = "gemini_deep_think" if model_name == "gemini" else "chatgpt_pro"
            deep_mode_tracker.record_usage(mode_key)

        return response, deep_mode_used

    async def check_recovered_responses(
        self,
        load_thread_fn,
    ) -> None:
        """
        Check for any recovered responses from pool restart.

        When the pool restarts while an LLM was generating a response,
        it attempts to recover the response by navigating back to the chat.
        Those recovered responses are stored and can be retrieved here.

        Args:
            load_thread_fn: Function to load a thread by ID (thread_id -> ExplorationThread)
        """
        if not self.llm_client:
            return

        try:
            # Get pool config
            pool_config = self.config.get("llm", {}).get("pool", {})
            pool_host = pool_config.get("host", "127.0.0.1")
            pool_port = pool_config.get("port", 9000)

            # Query pool for recovered responses
            url = f"http://{pool_host}:{pool_port}/recovered"
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read().decode())

            responses = data.get("responses", [])
            if not responses:
                return

            log.info(f"[recovery] Found {len(responses)} recovered responses from pool")

            for item in responses:
                request_id = item.get("request_id")
                thread_id = item.get("thread_id")
                response_text = item.get("response", "")
                backend = item.get("backend")

                log.info(
                    f"[recovery] Processing recovered response from {backend}",
                    extra={
                        "request_id": request_id,
                        "thread_id": thread_id,
                        "response_length": len(response_text),
                    },
                )

                # Try to find the associated thread and add the response
                if thread_id:
                    try:
                        thread = load_thread_fn(thread_id)
                        if thread:
                            from explorer.src.models import Exchange

                            thread.add_exchange(
                                Exchange(
                                    role=ExchangeRole.LLM,
                                    content=response_text,
                                    model=backend,
                                    timestamp=datetime.now(),
                                )
                            )
                            thread.save(self.paths.explorations_dir)
                            log.info(f"[recovery] Added recovered response to thread {thread_id}")
                    except Exception as e:
                        log.warning(
                            f"[recovery] Could not add response to thread {thread_id}: {e}"
                        )

                # Clear the recovered response from pool
                try:
                    clear_url = f"http://{pool_host}:{pool_port}/recovered/{request_id}"
                    req = urllib.request.Request(clear_url, method="DELETE")
                    urllib.request.urlopen(req, timeout=5)
                    log.info(f"[recovery] Cleared recovered response {request_id}")
                except Exception as e:
                    log.warning(
                        f"[recovery] Could not clear recovered response {request_id}: {e}"
                    )

        except urllib.error.URLError:
            # Pool not running or not reachable - that's fine
            pass
        except Exception as e:
            log.warning(f"[recovery] Error checking for recovered responses: {e}")
