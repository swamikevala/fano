"""
Backend workers for the Browser Pool Service.

Each worker manages a browser instance and processes requests from its queue.
"""

import asyncio
import os
import time
import uuid
from pathlib import Path
from typing import Optional

from shared.logging import get_logger, set_session_id

# Import browser modules from explorer - now available via proper package structure
from explorer.src.browser.gemini import GeminiInterface
from explorer.src.browser.chatgpt import ChatGPTInterface
from explorer.src.browser.claude import ClaudeInterface
from explorer.src.browser.base import AuthenticationRequired
from .models import SendRequest, SendResponse, ResponseMetadata, Backend
from .state import StateManager
from .queue import RequestQueue, QueuedRequest
from .jobs import JobStore, Job, JobStatus

log = get_logger("pool", "workers")


class BaseWorker:
    """Base class for backend workers."""

    backend_name: str = "base"
    deep_mode_method: Optional[str] = None  # Override in subclass: "enable_deep_think", "enable_pro_mode"

    def __init__(self, config: dict, state: StateManager, queue: RequestQueue, jobs: Optional[JobStore] = None):
        self.config = config
        self.state = state
        self.queue = queue
        self.jobs = jobs  # JobStore for async jobs
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self.browser = None  # Set by subclasses

        # Track current work
        self._current_request_id: Optional[str] = None
        self._current_prompt: Optional[str] = None
        self._current_start_time: Optional[float] = None
        self._current_job: Optional[Job] = None  # For async jobs

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
        """Main worker loop - process jobs and legacy queue requests."""
        while self._running:
            try:
                # Check if we're available
                if not self.state.is_available(self.backend_name):
                    await asyncio.sleep(1)
                    continue

                # First, check JobStore for async jobs (new system)
                if self.jobs:
                    job = self.jobs.get_next_job(self.backend_name)
                    if job:
                        await self._process_job(job)
                        continue

                # Fall back to legacy queue (sync system)
                queued = await self.queue.dequeue()
                if not queued:
                    await asyncio.sleep(0.1)  # Small sleep when idle
                    continue

                # Process the legacy request
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
                        # Check if response was assigned (handles both success and exception cases)
                        was_successful = 'response' in locals() and response.success
                        history_entry = {
                            "request_id": self._current_request_id,
                            "prompt": self._current_prompt,
                            "prompt_length": len(self._current_prompt) if self._current_prompt else 0,
                            "started_at": self._current_start_time,
                            "completed_at": time.time(),
                            "elapsed_seconds": round(elapsed, 2),
                            "success": was_successful,
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

    async def _process_job(self, job: Job):
        """
        Process an async job from JobStore.

        This converts the job to a SendRequest and processes it,
        then updates the JobStore with the result.
        """
        log.info("pool.job.processing",
                 job_id=job.job_id,
                 backend=self.backend_name,
                 thread_id=job.thread_id,
                 deep_mode=job.deep_mode)

        # Track current job
        self._current_job = job
        self._current_request_id = job.job_id
        self._current_prompt = job.prompt
        self._current_start_time = time.time()

        try:
            # Convert job to SendRequest for processing
            from .models import SendOptions, Priority, ImageAttachment
            request = SendRequest(
                backend=Backend(self.backend_name),
                prompt=job.prompt,
                options=SendOptions(
                    deep_mode=job.deep_mode,
                    new_chat=job.new_chat,
                    priority=Priority(job.priority),
                    timeout_seconds=3600,  # Jobs have long timeout
                ),
                thread_id=job.thread_id,
                images=[ImageAttachment(**img) for img in (job.images or [])],
            )

            # Process the request
            response = await self._process_request(request)

            if response.success:
                # Complete the job successfully
                self.jobs.complete(
                    job_id=job.job_id,
                    result=response.response or "",
                    deep_mode_used=response.metadata.deep_mode_used if response.metadata else False,
                )
                log.info("pool.job.completed",
                         job_id=job.job_id,
                         backend=self.backend_name,
                         result_length=len(response.response or ""))
            else:
                # Job failed
                self.jobs.fail(
                    job_id=job.job_id,
                    error=f"{response.error}: {response.message}",
                )
                log.error("pool.job.failed",
                          job_id=job.job_id,
                          backend=self.backend_name,
                          error=response.error,
                          message=response.message)

        except Exception as e:
            # Unexpected error
            self.jobs.fail(
                job_id=job.job_id,
                error=f"processing_error: {str(e)}",
            )
            log.exception(e, "pool.job.error", {"job_id": job.job_id, "backend": self.backend_name})

        finally:
            # Save to history
            elapsed = time.time() - self._current_start_time if self._current_start_time else 0
            history_entry = {
                "request_id": job.job_id,
                "prompt_length": len(job.prompt),
                "started_at": self._current_start_time,
                "completed_at": time.time(),
                "elapsed_seconds": round(elapsed, 2),
                "success": job.status == JobStatus.COMPLETE if hasattr(job, 'status') else False,
                "backend": self.backend_name,
                "is_async_job": True,
            }
            self._request_history.append(history_entry)
            if len(self._request_history) > self._max_history:
                self._request_history.pop(0)

            # Clear tracking
            self._current_job = None
            self._current_request_id = None
            self._current_prompt = None
            self._current_start_time = None

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
        Check if the backend browser is healthy and responsive.

        Returns:
            Tuple of (is_healthy, reason)
        """
        if not self.browser:
            return False, "browser_not_initialized"

        try:
            # Check if page exists - try reconnect if not
            if not self.browser.page:
                log.warning("pool.health.page_none", backend=self.backend_name, action="attempting_reconnect")
                if await self.try_reconnect():
                    log.info("pool.health.reconnected", backend=self.backend_name)
                else:
                    return False, "page_not_initialized_reconnect_failed"

            # Verify page is responsive (throws TargetClosedError if browser crashed)
            await asyncio.wait_for(
                self.browser.page.evaluate("() => document.readyState"),
                timeout=5.0
            )
            return True, "ok"

        except asyncio.TimeoutError:
            log.warning("pool.health.timeout", backend=self.backend_name)
            return False, "page_unresponsive"

        except Exception as e:
            error_name = type(e).__name__
            log.warning("pool.health.failed", backend=self.backend_name, error=error_name, message=str(e))
            return False, f"page_error:{error_name}"

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
        Check for and recover any in-progress async jobs from before a restart.

        Checks JobStore for PROCESSING jobs and attempts to navigate back to
        the chat URL to collect the response.

        Returns SendResponse if work was recovered, None otherwise.
        """
        if not self.browser or not hasattr(self.browser, 'page'):
            return None

        # Check JobStore for processing jobs
        if not self.jobs:
            self.state.clear_active_work(self.backend_name)
            return None

        processing_job = self.jobs.get_processing_job(self.backend_name)
        if not processing_job:
            # No async job to recover - just clear any stale active work
            self.state.clear_active_work(self.backend_name)
            return None

        chat_url = processing_job.chat_url
        if not chat_url:
            log.warning("pool.worker.recovery.no_chat_url",
                       backend=self.backend_name,
                       job_id=processing_job.job_id)
            self.jobs.fail(processing_job.job_id, "No chat URL to recover from")
            self.state.clear_active_work(self.backend_name)
            return None

        log.info("pool.worker.recovery.attempting",
                 backend=self.backend_name,
                 job_id=processing_job.job_id,
                 thread_id=processing_job.thread_id,
                 chat_url=chat_url)

        try:
            # Navigate back to the chat
            await self.browser.page.goto(chat_url)
            await self.browser.page.wait_for_load_state("networkidle")
            await asyncio.sleep(5)  # Give page more time to settle

            response_text = None

            # Exponential backoff delays: 3, 6, 12, 24, 48 seconds
            backoff_delays = [3, 6, 12, 24, 48]
            max_attempts = 5

            # Try multiple times to detect state and get response
            for attempt in range(max_attempts):
                log.info("pool.worker.recovery.check_attempt",
                         backend=self.backend_name,
                         attempt=attempt + 1,
                         max_attempts=max_attempts)

                # Try scrolling the page to trigger lazy-loaded content
                try:
                    await self.browser.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await asyncio.sleep(1)
                    await self.browser.page.evaluate("window.scrollTo(0, 0)")
                    await asyncio.sleep(1)
                except Exception:
                    pass  # Best effort scroll

                # Check if still generating
                is_generating = False
                if hasattr(self.browser, 'is_generating'):
                    is_generating = await self.browser.is_generating()

                if is_generating:
                    log.info("pool.worker.recovery.still_generating", backend=self.backend_name)
                    response_text = await self.browser._wait_for_response()
                    break

                # Try to get existing response
                if hasattr(self.browser, 'try_get_response'):
                    response_text = await self.browser.try_get_response()
                    if response_text:
                        log.info("pool.worker.recovery.found_response",
                                 backend=self.backend_name,
                                 response_length=len(response_text))
                        break

                # Neither generating nor response found - wait with exponential backoff
                if attempt < max_attempts - 1:
                    delay = backoff_delays[attempt]
                    log.info("pool.worker.recovery.waiting_retry",
                            backend=self.backend_name,
                            delay_seconds=delay)
                    await asyncio.sleep(delay)

            # Last resort: wait for response with long timeout
            if not response_text:
                log.info("pool.worker.recovery.last_resort_wait", backend=self.backend_name)
                try:
                    response_text = await asyncio.wait_for(
                        self.browser._wait_for_response(),
                        timeout=1800  # 30 minute timeout
                    )
                except asyncio.TimeoutError:
                    log.warning("pool.worker.recovery.timeout", backend=self.backend_name)
                    response_text = None

            if not response_text:
                log.warning("pool.worker.recovery.no_response", backend=self.backend_name)

                # Save a debug screenshot on failure
                try:
                    screenshot_path = Path(__file__).parent.parent / "debug" / f"recovery_fail_{self.backend_name}_{int(time.time())}.png"
                    screenshot_path.parent.mkdir(parents=True, exist_ok=True)
                    await self.browser.page.screenshot(path=str(screenshot_path))
                    log.info("pool.worker.recovery.screenshot_saved",
                            backend=self.backend_name,
                            path=str(screenshot_path))
                except Exception as e:
                    log.warning("pool.worker.recovery.screenshot_failed", error=str(e))

                self.jobs.fail(processing_job.job_id, "Recovery failed - no response found")
                self.state.clear_active_work(self.backend_name)
                return None

            # Complete the job in JobStore
            self.jobs.complete(
                job_id=processing_job.job_id,
                result=response_text,
                deep_mode_used=processing_job.deep_mode,
            )

            # Clear active work
            self.state.clear_active_work(self.backend_name)

            log.info("pool.worker.recovery.success",
                     backend=self.backend_name,
                     job_id=processing_job.job_id,
                     thread_id=processing_job.thread_id,
                     response_length=len(response_text))

            return SendResponse(
                success=True,
                response=response_text,
                recovered=True,
                metadata=ResponseMetadata(
                    backend=self.backend_name,
                    deep_mode_used=processing_job.deep_mode,
                    response_time_seconds=0.0,
                ),
            )

        except Exception as e:
            log.error("pool.worker.recovery.failed",
                      backend=self.backend_name,
                      error=str(e),
                      error_type=type(e).__name__)
            self.jobs.fail(processing_job.job_id, f"Recovery error: {str(e)}")
            self.state.clear_active_work(self.backend_name)
            return None


class GeminiWorker(BaseWorker):
    """Worker for Gemini backend."""

    backend_name = "gemini"
    deep_mode_method = "enable_deep_think"

    def __init__(self, config: dict, state: StateManager, queue: RequestQueue, jobs: Optional[JobStore] = None):
        super().__init__(config, state, queue, jobs)
        self._session_id = None

    async def connect(self):
        """Connect to Gemini."""
        try:
            self.browser = GeminiInterface()

            # Override browser data dir to use pool's directory
            pool_browser_data = Path(__file__).parent.parent / "browser_data" / "gemini"
            pool_browser_data.mkdir(parents=True, exist_ok=True)

            await self.browser.connect()
            self.state.mark_authenticated(self.backend_name, True)
            log.info("pool.worker.lifecycle", action="connected", backend=self.backend_name)

            # Check for and recover any in-progress work from before restart
            await self.check_and_recover_work()

        except AuthenticationRequired as e:
            # Genuine auth failure - mark as needing auth
            log.warning("pool.worker.auth_required", backend=self.backend_name, message=str(e))
            self.state.mark_authenticated(self.backend_name, False)
            raise

        except Exception as e:
            # Connection issue - don't change auth state, just log and raise
            print(f"[gemini] Connection failed: {type(e).__name__}: {e}")
            log.warning("pool.worker.connection_failed",
                       backend=self.backend_name,
                       error=str(e),
                       error_type=type(e).__name__,
                       auth_state_unchanged=True)
            # Don't mark as unauthenticated - might just be transient
            raise

    async def disconnect(self):
        """Disconnect from Gemini."""
        if self.browser:
            await self.browser.disconnect()
            self.browser = None

    async def authenticate(self):
        """Open browser for manual authentication."""
        try:
            # Create browser for auth
            browser = GeminiInterface()
            await browser.connect()

            log.info("pool.worker.auth", action="browser_opened", backend=self.backend_name)
            return True

        except Exception as e:
            log.exception(e, "pool.worker.auth_failed", {"backend": self.backend_name})
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
        request_id = self._current_request_id or str(uuid.uuid4())

        try:
            if request.options.new_chat:
                await self.browser.start_new_chat()

            if request.options.deep_mode:
                deep_mode_used = await self._try_enable_deep_mode()

            # Capture chat URL before sending message - try reconnect if page is None
            if self.browser.page is None:
                log.warning("pool.worker.page_none", backend=self.backend_name, action="attempting_reconnect")
                if not await self.try_reconnect():
                    raise RuntimeError("Browser page not available and reconnection failed")
            chat_url = self.browser.page.url if self.browser.page else "unknown"

            # Set up URL update callback - browser will call this when URL changes
            def on_url_change(new_url: str):
                log.info("pool.worker.url_changed", backend=self.backend_name, new_url=new_url)
                self.state.set_active_work(
                    backend=self.backend_name,
                    request_id=request_id,
                    prompt=request.prompt,
                    chat_url=new_url,
                    thread_id=getattr(request, 'thread_id', None),
                    options={"deep_mode": deep_mode_used, "new_chat": request.options.new_chat},
                )
                # Also update job if processing async job
                if self._current_job and self.jobs:
                    self.jobs.set_chat_url(self._current_job.job_id, new_url)

            if self.browser.page:
                self.browser.set_url_update_callback(on_url_change)

            # Record active work so we can recover if pool restarts
            self.state.set_active_work(
                backend=self.backend_name,
                request_id=request_id,
                prompt=request.prompt,
                chat_url=chat_url,
                thread_id=getattr(request, 'thread_id', None),
                options={"deep_mode": deep_mode_used, "new_chat": request.options.new_chat},
            )
            # Also set initial chat URL on job if processing async job
            if self._current_job and self.jobs:
                self.jobs.set_chat_url(self._current_job.job_id, chat_url)

            # Convert ImageAttachment models to dicts for browser interface
            images = None
            if request.images:
                images = [
                    {"filename": img.filename, "data": img.data, "media_type": img.media_type}
                    for img in request.images
                ]

            response_text = await self.browser.send_message(request.prompt, images=images)

            # Clear callback
            self.browser.set_url_update_callback(None)

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
            # Clear callback on error too
            if self.browser:
                self.browser.set_url_update_callback(None)
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

    def __init__(self, config: dict, state: StateManager, queue: RequestQueue, jobs: Optional[JobStore] = None):
        super().__init__(config, state, queue, jobs)

    async def connect(self):
        """Connect to ChatGPT."""
        try:
            self.browser = ChatGPTInterface()
            await self.browser.connect()
            self.state.mark_authenticated(self.backend_name, True)
            log.info("pool.worker.lifecycle", action="connected", backend=self.backend_name)

            # Check for and recover any in-progress work from before restart
            await self.check_and_recover_work()

        except AuthenticationRequired as e:
            # Genuine auth failure - mark as needing auth
            log.warning("pool.worker.auth_required", backend=self.backend_name, message=str(e))
            self.state.mark_authenticated(self.backend_name, False)
            raise

        except Exception as e:
            # Connection issue - don't change auth state, just log and raise
            print(f"[chatgpt] Connection failed: {type(e).__name__}: {e}")
            log.warning("pool.worker.connection_failed",
                       backend=self.backend_name,
                       error=str(e),
                       error_type=type(e).__name__,
                       auth_state_unchanged=True)
            # Don't mark as unauthenticated - might just be transient
            raise

    async def disconnect(self):
        """Disconnect from ChatGPT."""
        if self.browser:
            await self.browser.disconnect()
            self.browser = None

    async def authenticate(self):
        """Open browser for manual authentication."""
        try:
            browser = ChatGPTInterface()
            await browser.connect()
            log.info("pool.worker.auth", action="browser_opened", backend=self.backend_name)
            return True

        except Exception as e:
            log.exception(e, "pool.worker.auth_failed", {"backend": self.backend_name})
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

            # Capture chat URL before sending message - try reconnect if page is None
            if self.browser.page is None:
                log.warning("pool.worker.page_none", backend=self.backend_name, action="attempting_reconnect")
                if not await self.try_reconnect():
                    raise RuntimeError("Browser page not available and reconnection failed")
            chat_url = self.browser.page.url if self.browser.page else "unknown"

            # Set up URL update callback - browser will call this when URL changes
            def on_url_change(new_url: str):
                log.info("pool.worker.url_changed", backend=self.backend_name, new_url=new_url)
                self.state.set_active_work(
                    backend=self.backend_name,
                    request_id=request_id,
                    prompt=request.prompt,
                    chat_url=new_url,
                    thread_id=getattr(request, 'thread_id', None),
                    options={"deep_mode": deep_mode_used, "new_chat": request.options.new_chat},
                )
                # Also update job if processing async job
                if self._current_job and self.jobs:
                    self.jobs.set_chat_url(self._current_job.job_id, new_url)

            if self.browser.page:
                self.browser.set_url_update_callback(on_url_change)

            # Record active work so we can recover if pool restarts
            self.state.set_active_work(
                backend=self.backend_name,
                request_id=request_id,
                prompt=request.prompt,
                chat_url=chat_url,
                thread_id=getattr(request, 'thread_id', None),
                options={"deep_mode": deep_mode_used, "new_chat": request.options.new_chat},
            )
            # Also set initial chat URL on job if processing async job
            if self._current_job and self.jobs:
                self.jobs.set_chat_url(self._current_job.job_id, chat_url)

            # Convert ImageAttachment models to dicts for browser interface
            images = None
            if request.images:
                images = [
                    {"filename": img.filename, "data": img.data, "media_type": img.media_type}
                    for img in request.images
                ]

            response_text = await self.browser.send_message(request.prompt, images=images)

            # Clear callback
            self.browser.set_url_update_callback(None)

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
            # Clear callback on error too
            if self.browser:
                self.browser.set_url_update_callback(None)
            log.exception(e, "pool.request.processing_error", {"backend": self.backend_name})
            # Don't clear active work on error - we might be able to recover
            return SendResponse(
                success=False,
                error="processing_error",
                message=str(e),
            )


class ClaudeWorker(BaseWorker):
    """Worker for Claude browser backend."""

    backend_name = "claude"
    deep_mode_method = "enable_extended_thinking"

    def __init__(self, config: dict, state: StateManager, queue: RequestQueue, jobs: Optional[JobStore] = None):
        super().__init__(config, state, queue, jobs)

    async def connect(self):
        """Connect to Claude browser."""
        try:
            self.browser = ClaudeInterface()
            await self.browser.connect()
            self.state.mark_authenticated(self.backend_name, True)
            log.info("pool.worker.lifecycle", action="connected", backend=self.backend_name)

            # Check for and recover any in-progress work from before restart
            await self.check_and_recover_work()

        except AuthenticationRequired as e:
            # Genuine auth failure - mark as needing auth
            log.warning("pool.worker.auth_required", backend=self.backend_name, message=str(e))
            self.state.mark_authenticated(self.backend_name, False)
            raise

        except Exception as e:
            # Connection issue - don't change auth state, just log and raise
            print(f"[claude] Connection failed: {type(e).__name__}: {e}")
            log.warning("pool.worker.connection_failed",
                       backend=self.backend_name,
                       error=str(e),
                       error_type=type(e).__name__,
                       auth_state_unchanged=True)
            # Don't mark as unauthenticated - might just be transient
            raise

    async def disconnect(self):
        """Disconnect from Claude browser."""
        if self.browser:
            await self.browser.disconnect()
            self.browser = None

    async def authenticate(self):
        """Open browser for manual authentication."""
        try:
            browser = ClaudeInterface()
            await browser.connect()
            log.info("pool.worker.auth", action="browser_opened", backend=self.backend_name)
            return True

        except Exception as e:
            log.exception(e, "pool.worker.auth_failed", {"backend": self.backend_name})
            return False

    async def _process_request(self, request: SendRequest) -> SendResponse:
        """Process a Claude browser request."""
        if not self.browser:
            return SendResponse(
                success=False,
                error="unavailable",
                message="Claude browser not connected",
            )

        start_time = time.time()
        deep_mode_used = False
        request_id = self._current_request_id or str(uuid.uuid4())

        try:
            if request.options.new_chat:
                await self.browser.start_new_chat()

            # Claude Extended Thinking is lighter than ChatGPT Pro or Gemini Deep Think,
            # so we enable it by default (not just when deep_mode is requested)
            if request.options.deep_mode:
                deep_mode_used = await self._try_enable_deep_mode()
            else:
                # Enable Extended Thinking as default mode (it's not as heavy as Pro/Deep Think)
                await self.browser.enable_extended_thinking()

            # Capture chat URL before sending message - try reconnect if page is None
            if self.browser.page is None:
                log.warning("pool.worker.page_none", backend=self.backend_name, action="attempting_reconnect")
                if not await self.try_reconnect():
                    raise RuntimeError("Browser page not available and reconnection failed")
            chat_url = self.browser.page.url if self.browser.page else "unknown"

            # Set up URL update callback - browser will call this when URL changes
            def on_url_change(new_url: str):
                log.info("pool.worker.url_changed", backend=self.backend_name, new_url=new_url)
                self.state.set_active_work(
                    backend=self.backend_name,
                    request_id=request_id,
                    prompt=request.prompt,
                    chat_url=new_url,
                    thread_id=getattr(request, 'thread_id', None),
                    options={"deep_mode": deep_mode_used, "new_chat": request.options.new_chat},
                )
                # Also update job if processing async job
                if self._current_job and self.jobs:
                    self.jobs.set_chat_url(self._current_job.job_id, new_url)

            if self.browser.page:
                self.browser.set_url_update_callback(on_url_change)

            # Record active work so we can recover if pool restarts
            self.state.set_active_work(
                backend=self.backend_name,
                request_id=request_id,
                prompt=request.prompt,
                chat_url=chat_url,
                thread_id=getattr(request, 'thread_id', None),
                options={"deep_mode": deep_mode_used, "new_chat": request.options.new_chat},
            )
            # Also set initial chat URL on job if processing async job
            if self._current_job and self.jobs:
                self.jobs.set_chat_url(self._current_job.job_id, chat_url)

            response_text = await self.browser.send_message(request.prompt)

            # Clear callback
            self.browser.set_url_update_callback(None)

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
            # Clear callback on error too
            if self.browser:
                self.browser.set_url_update_callback(None)
            log.exception(e, "pool.request.processing_error", {"backend": self.backend_name})
            # Don't clear active work on error - we might be able to recover
            return SendResponse(
                success=False,
                error="processing_error",
                message=str(e),
            )
