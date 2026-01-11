"""
Backend workers for the Browser Pool Service.

Each worker manages a browser instance and processes requests from its queue.
"""

import asyncio
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

# Add explorer's src to path so we can reuse browser automation
EXPLORER_SRC = Path(__file__).resolve().parent.parent.parent / "explorer" / "src"
sys.path.insert(0, str(EXPLORER_SRC))

# Add shared module to path
SHARED_PATH = Path(__file__).resolve().parent.parent.parent / "shared"
sys.path.insert(0, str(SHARED_PATH.parent))

from shared.logging import get_logger, set_session_id
from .models import SendRequest, SendResponse, ResponseMetadata, Backend
from .state import StateManager
from .queue import RequestQueue, QueuedRequest

log = get_logger("pool", "workers")


class BaseWorker:
    """Base class for backend workers."""

    backend_name: str = "base"
    deep_mode_method: Optional[str] = None  # Override in subclass: "enable_deep_think", "enable_pro_mode"

    def __init__(self, config: dict, state: StateManager, queue: RequestQueue):
        self.config = config
        self.state = state
        self.queue = queue
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self.browser = None  # Set by subclasses

        # Track current work
        self._current_request_id: Optional[str] = None
        self._current_prompt: Optional[str] = None
        self._current_start_time: Optional[float] = None

        # Request history (circular buffer of last N requests)
        self._request_history: list[dict] = []
        self._max_history = 20

    async def start(self):
        """Start the worker."""
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        log.info("pool.worker.lifecycle", action="started", backend=self.backend_name)

    async def stop(self):
        """Stop the worker."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info("pool.worker.lifecycle", action="stopped", backend=self.backend_name)

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
                start_time = log.request_start(
                    queued.request_id, self.backend_name, queued.request.prompt,
                    priority=queued.request.options.priority.value,
                    deep_mode_requested=queued.request.options.deep_mode,
                )

                # Track current work
                self._current_request_id = queued.request_id
                self._current_prompt = queued.request.prompt
                self._current_start_time = time.time()

                try:
                    response = await self._process_request(queued.request)
                    log.request_complete(
                        queued.request_id, self.backend_name,
                        response.response or "", start_time,
                        success=response.success,
                        deep_mode_used=response.metadata.deep_mode_used if response.metadata else False,
                    )
                    queued.future.set_result(response)
                except Exception as e:
                    log.request_error(
                        queued.request_id, self.backend_name,
                        str(e), type(e).__name__, start_time,
                    )
                    error_response = SendResponse(
                        success=False,
                        error="processing_error",
                        message=str(e),
                    )
                    queued.future.set_result(error_response)
                finally:
                    # Save to history before clearing
                    if self._current_request_id:
                        elapsed = time.time() - self._current_start_time if self._current_start_time else 0
                        history_entry = {
                            "request_id": self._current_request_id,
                            "prompt": self._current_prompt,
                            "prompt_length": len(self._current_prompt) if self._current_prompt else 0,
                            "started_at": self._current_start_time,
                            "completed_at": time.time(),
                            "elapsed_seconds": round(elapsed, 2),
                            "success": response.success if 'response' in dir() else False,
                            "backend": self.backend_name,
                        }
                        self._request_history.append(history_entry)
                        # Keep only last N entries
                        if len(self._request_history) > self._max_history:
                            self._request_history.pop(0)

                    # Clear current work tracking
                    self._current_request_id = None
                    self._current_prompt = None
                    self._current_start_time = None

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.exception(e, "pool.worker.loop_error", {"backend": self.backend_name})
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

    async def check_health(self) -> tuple[bool, str]:
        """
        Check if the backend is actually healthy and responsive.

        Returns:
            Tuple of (is_healthy, reason)
        """
        # Default implementation - subclasses should override
        return True, "ok"

    async def try_reconnect(self) -> bool:
        """
        Attempt to reconnect a crashed browser.

        Returns True if reconnection succeeded.
        """
        log.info("pool.worker.reconnect.attempt", backend=self.backend_name)
        try:
            await self.disconnect()
            await self.connect()
            log.info("pool.worker.reconnect.success", backend=self.backend_name)
            return True
        except Exception as e:
            log.error("pool.worker.reconnect.failed", backend=self.backend_name, error=str(e))
            return False

    def get_current_work(self) -> Optional[dict]:
        """
        Get information about the current work being processed.

        Returns None if idle, or a dict with request details.
        """
        if self._current_request_id is None:
            return None

        elapsed = time.time() - self._current_start_time if self._current_start_time else 0

        # Truncate prompt for display (first 200 chars)
        prompt_preview = self._current_prompt[:200] if self._current_prompt else ""
        if self._current_prompt and len(self._current_prompt) > 200:
            prompt_preview += "..."

        return {
            "request_id": self._current_request_id,
            "prompt_preview": prompt_preview,
            "prompt": self._current_prompt,  # Full prompt for detail view
            "prompt_length": len(self._current_prompt) if self._current_prompt else 0,
            "elapsed_seconds": round(elapsed, 1),
            "backend": self.backend_name,
        }

    def get_request_history(self, limit: int = 10) -> list[dict]:
        """
        Get recent request history.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of recent requests (newest first)
        """
        # Return newest first
        history = list(reversed(self._request_history))[:limit]
        # Add prompt preview to each entry
        for entry in history:
            if entry.get("prompt"):
                preview = entry["prompt"][:300]
                if len(entry["prompt"]) > 300:
                    preview += "..."
                entry["prompt_preview"] = preview
        return history

    async def _try_enable_deep_mode(self) -> bool:
        """
        Try to enable deep/pro mode if available and within limits.

        Returns True if deep mode was successfully enabled.
        """
        if not self.deep_mode_method or not self.browser:
            return False

        if not self.state.can_use_deep_mode(self.backend_name):
            log.warning("pool.deep_mode.limit_reached", backend=self.backend_name)
            return False

        try:
            enable_fn = getattr(self.browser, self.deep_mode_method)
            await enable_fn()
            self.state.increment_deep_mode_usage(self.backend_name)
            return True
        except Exception as e:
            log.warning("pool.deep_mode.enable_failed", backend=self.backend_name, error=str(e))
            return False

    def _check_and_mark_rate_limit(self, response_text: str):
        """Check response for rate limiting and update state if detected."""
        if self.browser and hasattr(self.browser, '_check_rate_limit'):
            if self.browser._check_rate_limit(response_text):
                self.state.mark_rate_limited(self.backend_name)

    async def check_and_recover_work(self) -> Optional[SendResponse]:
        """
        Check for and recover any in-progress work from before a restart.

        If there's active work in state and we can navigate back to the chat,
        try to collect the response and store it for later pickup.

        Returns SendResponse if work was recovered, None otherwise.
        """
        if not self.browser or not hasattr(self.browser, 'page'):
            return None

        active = self.state.get_active_work(self.backend_name)
        if not active:
            return None

        chat_url = active.get("chat_url")
        if not chat_url:
            # No chat URL to recover from
            self.state.clear_active_work(self.backend_name)
            return None

        log.info("pool.worker.recovery.attempting",
                 backend=self.backend_name,
                 request_id=active.get("request_id"),
                 chat_url=chat_url)

        try:
            # Navigate back to the chat
            await self.browser.page.goto(chat_url)
            await self.browser.page.wait_for_load_state("networkidle")
            await asyncio.sleep(2)  # Let page settle

            # Check if response is ready or still generating
            if hasattr(self.browser, 'is_generating') and await self.browser.is_generating():
                log.info("pool.worker.recovery.still_generating", backend=self.backend_name)
                # Still generating - wait for it
                response_text = await self.browser._wait_for_response()
            elif hasattr(self.browser, 'try_get_response'):
                response_text = await self.browser.try_get_response()
                if not response_text:
                    log.warning("pool.worker.recovery.no_response", backend=self.backend_name)
                    self.state.clear_active_work(self.backend_name)
                    return None
            else:
                log.warning("pool.worker.recovery.no_method", backend=self.backend_name)
                self.state.clear_active_work(self.backend_name)
                return None

            # Store the recovered response for later pickup
            self.state.add_recovered_response(
                backend=self.backend_name,
                request_id=active.get("request_id", str(uuid.uuid4())),
                prompt=active.get("prompt", ""),
                response=response_text,
                thread_id=active.get("thread_id"),
                options=active.get("options"),
            )

            # Clear active work
            self.state.clear_active_work(self.backend_name)

            log.info("pool.worker.recovery.success",
                     backend=self.backend_name,
                     request_id=active.get("request_id"),
                     response_length=len(response_text))

            return SendResponse(
                success=True,
                response=response_text,
                recovered=True,
                metadata=ResponseMetadata(
                    backend=self.backend_name,
                    deep_mode_used=active.get("options", {}).get("deep_mode", False),
                ),
            )

        except Exception as e:
            log.error("pool.worker.recovery.failed",
                      backend=self.backend_name,
                      error=str(e),
                      error_type=type(e).__name__)
            # Clear the stale active work
            self.state.clear_active_work(self.backend_name)
            return None


class GeminiWorker(BaseWorker):
    """Worker for Gemini backend."""

    backend_name = "gemini"
    deep_mode_method = "enable_deep_think"

    def __init__(self, config: dict, state: StateManager, queue: RequestQueue):
        super().__init__(config, state, queue)
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
            log.info("pool.worker.lifecycle", action="connected", backend=self.backend_name)

            # Check for and recover any in-progress work from before restart
            await self.check_and_recover_work()

        except Exception as e:
            log.exception(e, "pool.worker.connection_failed", {"backend": self.backend_name})
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

            log.info("pool.worker.auth", action="browser_opened", backend=self.backend_name)
            return True

        except Exception as e:
            log.exception(e, "pool.worker.auth_failed", {"backend": self.backend_name})
            return False

    async def check_health(self) -> tuple[bool, str]:
        """Check if Gemini browser is healthy by verifying page is responsive."""
        if not self.browser:
            return False, "browser_not_initialized"

        try:
            # Check if page exists and is not closed
            if not self.browser.page:
                return False, "page_not_initialized"

            # Try a simple page query to verify page is responsive
            # This will throw TargetClosedError if browser crashed
            await asyncio.wait_for(
                self.browser.page.evaluate("() => document.readyState"),
                timeout=5.0
            )
            return True, "ok"

        except asyncio.TimeoutError:
            log.warning("pool.health.timeout", backend=self.backend_name)
            self.state.mark_authenticated(self.backend_name, False)
            return False, "page_unresponsive"

        except Exception as e:
            error_name = type(e).__name__
            log.warning("pool.health.failed", backend=self.backend_name, error=error_name, message=str(e))
            self.state.mark_authenticated(self.backend_name, False)
            return False, f"page_error:{error_name}"

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
        request_id = self._current_request_id or str(uuid.uuid4())

        try:
            if request.options.new_chat:
                await self.browser.start_new_chat()

            if request.options.deep_mode:
                deep_mode_used = await self._try_enable_deep_mode()

            # Capture chat URL before sending message
            chat_url = self.browser.page.url

            # Record active work so we can recover if pool restarts
            self.state.set_active_work(
                backend=self.backend_name,
                request_id=request_id,
                prompt=request.prompt,
                chat_url=chat_url,
                thread_id=getattr(request, 'thread_id', None),
                options={"deep_mode": deep_mode_used, "new_chat": request.options.new_chat},
            )

            response_text = await self.browser.send_message(request.prompt)

            # Update chat URL after sending (it may have changed to include conversation ID)
            final_chat_url = self.browser.page.url
            if final_chat_url != chat_url:
                self.state.set_active_work(
                    backend=self.backend_name,
                    request_id=request_id,
                    prompt=request.prompt,
                    chat_url=final_chat_url,
                    thread_id=getattr(request, 'thread_id', None),
                    options={"deep_mode": deep_mode_used, "new_chat": request.options.new_chat},
                )

            self._check_and_mark_rate_limit(response_text)

            # Clear active work on successful completion
            self.state.clear_active_work(self.backend_name)

            return SendResponse(
                success=True,
                response=response_text,
                metadata=ResponseMetadata(
                    backend=self.backend_name,
                    deep_mode_used=deep_mode_used,
                    response_time_seconds=time.time() - start_time,
                    session_id=self.browser.chat_logger.get_session_id(),
                ),
            )

        except Exception as e:
            log.exception(e, "pool.request.processing_error", {"backend": self.backend_name})
            # Don't clear active work on error - we might be able to recover
            return SendResponse(
                success=False,
                error="processing_error",
                message=str(e),
            )


class ChatGPTWorker(BaseWorker):
    """Worker for ChatGPT backend."""

    backend_name = "chatgpt"
    deep_mode_method = "enable_pro_mode"

    def __init__(self, config: dict, state: StateManager, queue: RequestQueue):
        super().__init__(config, state, queue)

    async def connect(self):
        """Connect to ChatGPT."""
        try:
            from browser.chatgpt import ChatGPTInterface

            self.browser = ChatGPTInterface()
            await self.browser.connect()
            self.state.mark_authenticated(self.backend_name, True)
            log.info("pool.worker.lifecycle", action="connected", backend=self.backend_name)

            # Check for and recover any in-progress work from before restart
            await self.check_and_recover_work()

        except Exception as e:
            log.exception(e, "pool.worker.connection_failed", {"backend": self.backend_name})
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
            log.info("pool.worker.auth", action="browser_opened", backend=self.backend_name)
            return True

        except Exception as e:
            log.exception(e, "pool.worker.auth_failed", {"backend": self.backend_name})
            return False

    async def check_health(self) -> tuple[bool, str]:
        """Check if ChatGPT browser is healthy by verifying page is responsive."""
        if not self.browser:
            return False, "browser_not_initialized"

        try:
            # Check if page exists and is not closed
            if not self.browser.page:
                return False, "page_not_initialized"

            # Try a simple page query to verify page is responsive
            # This will throw TargetClosedError if browser crashed
            await asyncio.wait_for(
                self.browser.page.evaluate("() => document.readyState"),
                timeout=5.0
            )
            return True, "ok"

        except asyncio.TimeoutError:
            log.warning("pool.health.timeout", backend=self.backend_name)
            self.state.mark_authenticated(self.backend_name, False)
            return False, "page_unresponsive"

        except Exception as e:
            error_name = type(e).__name__
            log.warning("pool.health.failed", backend=self.backend_name, error=error_name, message=str(e))
            self.state.mark_authenticated(self.backend_name, False)
            return False, f"page_error:{error_name}"

    async def _process_request(self, request: SendRequest) -> SendResponse:
        """Process a ChatGPT request."""
        if not self.browser:
            return SendResponse(
                success=False,
                error="unavailable",
                message="ChatGPT browser not connected",
            )

        start_time = time.time()
        deep_mode_used = False
        request_id = self._current_request_id or str(uuid.uuid4())

        try:
            if request.options.new_chat:
                await self.browser.start_new_chat()

            if request.options.deep_mode:
                deep_mode_used = await self._try_enable_deep_mode()
            else:
                # Explicitly ensure we're NOT in Pro mode
                # (browser may remember Pro mode from previous session)
                await self.browser.enable_thinking_mode()

            # Capture chat URL before sending message
            chat_url = self.browser.page.url

            # Record active work so we can recover if pool restarts
            self.state.set_active_work(
                backend=self.backend_name,
                request_id=request_id,
                prompt=request.prompt,
                chat_url=chat_url,
                thread_id=getattr(request, 'thread_id', None),
                options={"deep_mode": deep_mode_used, "new_chat": request.options.new_chat},
            )

            response_text = await self.browser.send_message(request.prompt)

            # Update chat URL after sending (it may have changed to include conversation ID)
            final_chat_url = self.browser.page.url
            if final_chat_url != chat_url:
                self.state.set_active_work(
                    backend=self.backend_name,
                    request_id=request_id,
                    prompt=request.prompt,
                    chat_url=final_chat_url,
                    thread_id=getattr(request, 'thread_id', None),
                    options={"deep_mode": deep_mode_used, "new_chat": request.options.new_chat},
                )

            self._check_and_mark_rate_limit(response_text)

            # Clear active work on successful completion
            self.state.clear_active_work(self.backend_name)

            return SendResponse(
                success=True,
                response=response_text,
                metadata=ResponseMetadata(
                    backend=self.backend_name,
                    deep_mode_used=deep_mode_used,
                    response_time_seconds=time.time() - start_time,
                    session_id=self.browser.chat_logger.get_session_id(),
                ),
            )

        except Exception as e:
            log.exception(e, "pool.request.processing_error", {"backend": self.backend_name})
            # Don't clear active work on error - we might be able to recover
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
            log.info("pool.worker.lifecycle", action="connected", backend=self.backend_name)

        except Exception as e:
            log.exception(e, "pool.worker.connection_failed", {"backend": self.backend_name})
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

            log.exception(e, "pool.request.api_error", {"backend": self.backend_name})
            return SendResponse(
                success=False,
                error="api_error",
                message=str(e),
            )
