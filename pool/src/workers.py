"""
Backend workers for the Browser Pool Service.

Each worker manages a browser instance and processes requests from its queue.
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Add explorer's src to path so we can reuse browser automation
EXPLORER_SRC = Path(__file__).resolve().parent.parent.parent / "explorer" / "src"
sys.path.insert(0, str(EXPLORER_SRC))

from .models import SendRequest, SendResponse, ResponseMetadata, Backend
from .state import StateManager
from .queue import RequestQueue, QueuedRequest

logger = logging.getLogger(__name__)


class BaseWorker:
    """Base class for backend workers."""

    backend_name: str = "base"

    def __init__(self, config: dict, state: StateManager, queue: RequestQueue):
        self.config = config
        self.state = state
        self.queue = queue
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the worker."""
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"[{self.backend_name}] Worker started")

    async def stop(self):
        """Stop the worker."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(f"[{self.backend_name}] Worker stopped")

    async def _run_loop(self):
        """Main worker loop - process requests from queue."""
        while self._running:
            try:
                # Check if we're available
                if not self.state.is_available(self.backend_name):
                    await asyncio.sleep(1)
                    continue

                # Try to get a request
                queued = await self.queue.dequeue()
                if not queued:
                    await asyncio.sleep(0.1)  # Small sleep when idle
                    continue

                # Process the request
                logger.info(f"[{self.backend_name}] Processing request {queued.request_id}")
                try:
                    response = await self._process_request(queued.request)
                    queued.future.set_result(response)
                except Exception as e:
                    logger.error(f"[{self.backend_name}] Request failed: {e}")
                    error_response = SendResponse(
                        success=False,
                        error="processing_error",
                        message=str(e),
                    )
                    queued.future.set_result(error_response)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[{self.backend_name}] Worker loop error: {e}")
                await asyncio.sleep(1)

    async def _process_request(self, request: SendRequest) -> SendResponse:
        """Process a single request. Override in subclasses."""
        raise NotImplementedError

    async def connect(self):
        """Connect to the backend. Override in subclasses."""
        raise NotImplementedError

    async def disconnect(self):
        """Disconnect from the backend. Override in subclasses."""
        pass

    async def authenticate(self):
        """Trigger interactive authentication. Override in subclasses."""
        raise NotImplementedError


class GeminiWorker(BaseWorker):
    """Worker for Gemini backend."""

    backend_name = "gemini"

    def __init__(self, config: dict, state: StateManager, queue: RequestQueue):
        super().__init__(config, state, queue)
        self.browser = None
        self._session_id = None

    async def connect(self):
        """Connect to Gemini."""
        try:
            # Import from explorer's browser module
            from browser.gemini import GeminiInterface

            self.browser = GeminiInterface()

            # Override browser data dir to use pool's directory
            pool_browser_data = Path(__file__).parent.parent / "browser_data" / "gemini"
            pool_browser_data.mkdir(parents=True, exist_ok=True)

            await self.browser.connect()
            self.state.mark_authenticated(self.backend_name, True)
            logger.info(f"[{self.backend_name}] Connected")

        except Exception as e:
            logger.error(f"[{self.backend_name}] Connection failed: {e}")
            self.state.mark_authenticated(self.backend_name, False)
            raise

    async def disconnect(self):
        """Disconnect from Gemini."""
        if self.browser:
            await self.browser.disconnect()
            self.browser = None

    async def authenticate(self):
        """Open browser for manual authentication."""
        try:
            from browser.gemini import GeminiInterface

            # Create browser for auth
            browser = GeminiInterface()
            await browser.connect()

            logger.info(f"[{self.backend_name}] Auth browser opened - waiting for user to log in")
            return True

        except Exception as e:
            logger.error(f"[{self.backend_name}] Auth failed: {e}")
            return False

    async def _process_request(self, request: SendRequest) -> SendResponse:
        """Process a Gemini request."""
        if not self.browser:
            return SendResponse(
                success=False,
                error="unavailable",
                message="Gemini browser not connected",
            )

        start_time = time.time()
        deep_mode_used = False

        try:
            # Start new chat if requested
            if request.options.new_chat:
                await self.browser.start_new_chat()

            # Enable deep mode if requested and available
            if request.options.deep_mode:
                if self.state.can_use_deep_mode(self.backend_name):
                    try:
                        await self.browser.enable_deep_think()
                        deep_mode_used = True
                        self.state.increment_deep_mode_usage(self.backend_name)
                    except Exception as e:
                        logger.warning(f"[{self.backend_name}] Could not enable deep mode: {e}")
                else:
                    logger.warning(f"[{self.backend_name}] Deep mode limit reached, using standard mode")

            # Send the prompt
            response_text = await self.browser.send_message(request.prompt)

            # Check for rate limiting in response
            if self.browser._check_rate_limit(response_text):
                self.state.mark_rate_limited(self.backend_name)

            elapsed = time.time() - start_time

            return SendResponse(
                success=True,
                response=response_text,
                metadata=ResponseMetadata(
                    backend=self.backend_name,
                    deep_mode_used=deep_mode_used,
                    response_time_seconds=elapsed,
                    session_id=self.browser.chat_logger.get_session_id(),
                ),
            )

        except Exception as e:
            logger.error(f"[{self.backend_name}] Request error: {e}")
            return SendResponse(
                success=False,
                error="processing_error",
                message=str(e),
            )


class ChatGPTWorker(BaseWorker):
    """Worker for ChatGPT backend."""

    backend_name = "chatgpt"

    def __init__(self, config: dict, state: StateManager, queue: RequestQueue):
        super().__init__(config, state, queue)
        self.browser = None

    async def connect(self):
        """Connect to ChatGPT."""
        try:
            from browser.chatgpt import ChatGPTInterface

            self.browser = ChatGPTInterface()
            await self.browser.connect()
            self.state.mark_authenticated(self.backend_name, True)
            logger.info(f"[{self.backend_name}] Connected")

        except Exception as e:
            logger.error(f"[{self.backend_name}] Connection failed: {e}")
            self.state.mark_authenticated(self.backend_name, False)
            raise

    async def disconnect(self):
        """Disconnect from ChatGPT."""
        if self.browser:
            await self.browser.disconnect()
            self.browser = None

    async def authenticate(self):
        """Open browser for manual authentication."""
        try:
            from browser.chatgpt import ChatGPTInterface

            browser = ChatGPTInterface()
            await browser.connect()
            logger.info(f"[{self.backend_name}] Auth browser opened")
            return True

        except Exception as e:
            logger.error(f"[{self.backend_name}] Auth failed: {e}")
            return False

    async def _process_request(self, request: SendRequest) -> SendResponse:
        """Process a ChatGPT request."""
        if not self.browser:
            return SendResponse(
                success=False,
                error="unavailable",
                message="ChatGPT browser not connected",
            )

        start_time = time.time()
        pro_mode_used = False

        try:
            if request.options.new_chat:
                await self.browser.start_new_chat()

            # Enable pro mode if requested
            if request.options.deep_mode:
                if self.state.can_use_deep_mode(self.backend_name):
                    try:
                        await self.browser.enable_pro_mode()
                        pro_mode_used = True
                        self.state.increment_deep_mode_usage(self.backend_name)
                    except Exception as e:
                        logger.warning(f"[{self.backend_name}] Could not enable pro mode: {e}")

            response_text = await self.browser.send_message(request.prompt)

            if self.browser._check_rate_limit(response_text):
                self.state.mark_rate_limited(self.backend_name)

            elapsed = time.time() - start_time

            return SendResponse(
                success=True,
                response=response_text,
                metadata=ResponseMetadata(
                    backend=self.backend_name,
                    deep_mode_used=pro_mode_used,
                    response_time_seconds=elapsed,
                    session_id=self.browser.chat_logger.get_session_id(),
                ),
            )

        except Exception as e:
            logger.error(f"[{self.backend_name}] Request error: {e}")
            return SendResponse(
                success=False,
                error="processing_error",
                message=str(e),
            )


class ClaudeWorker(BaseWorker):
    """Worker for Claude API backend."""

    backend_name = "claude"

    def __init__(self, config: dict, state: StateManager, queue: RequestQueue):
        super().__init__(config, state, queue)
        self.client = None
        self._model = config.get("backends", {}).get("claude", {}).get("model", "claude-sonnet-4-20250514")

    async def connect(self):
        """Initialize Claude API client."""
        try:
            import anthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")

            self.client = anthropic.Anthropic(api_key=api_key)
            self.state.mark_authenticated(self.backend_name, True)
            logger.info(f"[{self.backend_name}] API client initialized")

        except Exception as e:
            logger.error(f"[{self.backend_name}] Connection failed: {e}")
            self.state.mark_authenticated(self.backend_name, False)
            raise

    async def disconnect(self):
        """Cleanup Claude client."""
        self.client = None

    async def authenticate(self):
        """Claude uses API key, no interactive auth needed."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            self.state.mark_authenticated(self.backend_name, True)
            return True
        return False

    async def _process_request(self, request: SendRequest) -> SendResponse:
        """Process a Claude API request."""
        if not self.client:
            return SendResponse(
                success=False,
                error="unavailable",
                message="Claude API client not initialized",
            )

        start_time = time.time()

        try:
            # Run synchronous API call in thread pool
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self._model,
                max_tokens=8192,
                messages=[{"role": "user", "content": request.prompt}],
            )

            response_text = response.content[0].text
            elapsed = time.time() - start_time

            return SendResponse(
                success=True,
                response=response_text,
                metadata=ResponseMetadata(
                    backend=self.backend_name,
                    deep_mode_used=False,
                    response_time_seconds=elapsed,
                ),
            )

        except Exception as e:
            error_str = str(e)
            if "rate" in error_str.lower() or "429" in error_str:
                self.state.mark_rate_limited(self.backend_name, 60)

            logger.error(f"[{self.backend_name}] API error: {e}")
            return SendResponse(
                success=False,
                error="api_error",
                message=str(e),
            )
