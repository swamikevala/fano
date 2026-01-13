"""
LLM Client - Unified interface for LLM access.

Routes requests appropriately:
- Browser backends (Gemini, ChatGPT) → Pool service
- API backends (Claude, OpenRouter) → Direct API calls
"""

import asyncio
import os
from pathlib import Path
from typing import Optional

import aiohttp

from shared.logging import get_logger

from .models import (
    LLMResponse,
    Backend,
    Priority,
    PoolStatus,
    BackendStatus,
    ImageAttachment,
)

log = get_logger("llm", "client")

# Backends that require browser automation (go through pool)
BROWSER_BACKENDS = {"gemini", "chatgpt", "claude"}

# Backends that use direct API calls
API_BACKENDS = {"openrouter"}


class PoolUnavailableError(Exception):
    """Raised when the pool service is not available."""
    pass


class LLMClient:
    """
    Unified client for LLM access.

    Routes requests to the appropriate backend:
    - Gemini/ChatGPT: Via Pool service (browser automation)
    - Claude: Direct API call
    - OpenRouter: Direct API call (for DeepSeek, etc.)

    Usage:
        client = LLMClient()

        # Send to browser-based LLM (uses pool)
        response = await client.send("gemini", "Hello!")

        # Send to API-based LLM (direct call)
        response = await client.send("claude", "Hello!")
    """

    def __init__(
        self,
        pool_url: str = "http://127.0.0.1:9000",
        anthropic_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
    ):
        """
        Initialize the client.

        Args:
            pool_url: URL of the pool service (for browser backends)
            anthropic_api_key: API key for Claude (or uses ANTHROPIC_API_KEY env var)
            openrouter_api_key: API key for OpenRouter (or uses OPENROUTER_API_KEY env var)
        """
        self.pool_url = pool_url.rstrip("/")
        self._http_session: Optional[aiohttp.ClientSession] = None

        # API keys (from args or environment)
        self._anthropic_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._openrouter_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")

        # Lazy-loaded API clients
        self._anthropic_client = None

    async def _get_http_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session for pool/API calls."""
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()
        return self._http_session

    def _get_anthropic_client(self):
        """Get or create Anthropic client."""
        if self._anthropic_client is None and self._anthropic_key:
            import anthropic
            self._anthropic_client = anthropic.Anthropic(api_key=self._anthropic_key)
        return self._anthropic_client

    async def close(self):
        """Close HTTP session."""
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
            self._http_session = None

    # --- Pool Service Methods (for browser backends) ---

    async def is_pool_available(self) -> bool:
        """Check if the pool service is running."""
        try:
            session = await self._get_http_session()
            async with session.get(
                f"{self.pool_url}/health",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                return resp.status == 200
        except Exception:
            return False

    async def get_pool_status(self) -> PoolStatus:
        """Get status of browser backends from pool."""
        try:
            session = await self._get_http_session()
            async with session.get(
                f"{self.pool_url}/status",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    raise PoolUnavailableError(f"Pool returned status {resp.status}")
                data = await resp.json()
                return PoolStatus.from_dict(data)
        except aiohttp.ClientError as e:
            raise PoolUnavailableError(f"Could not connect to pool: {e}")

    async def _send_via_pool(
        self,
        backend: str,
        prompt: str,
        deep_mode: bool,
        new_chat: bool,
        timeout_seconds: int,
        priority: str,
        thread_id: Optional[str] = None,
        images: Optional[list[ImageAttachment]] = None,
    ) -> LLMResponse:
        """
        Send request via pool service (legacy sync mode).

        Note: For robust handling of long-running requests, use send_async() instead.
        This method is kept for backward compatibility.
        """
        request_data = {
            "backend": backend,
            "prompt": prompt,
            "options": {
                "deep_mode": deep_mode,
                "new_chat": new_chat,
                "timeout_seconds": timeout_seconds,
                "priority": priority,
            },
            "thread_id": thread_id,
            "images": [img.to_dict() for img in images] if images else [],
        }

        max_retries = 2
        retry_delay = 5

        for attempt in range(max_retries + 1):
            try:
                session = await self._get_http_session()
                async with session.post(
                    f"{self.pool_url}/send",
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=timeout_seconds + 30),
                ) as resp:
                    data = await resp.json()
                    return LLMResponse.from_pool_response(data)

            except aiohttp.ClientError as e:
                log.warning("llm.pool.connection_error",
                           backend=backend,
                           attempt=attempt + 1,
                           max_retries=max_retries,
                           error=str(e))

                # Retry if attempts remain
                if attempt < max_retries:
                    log.info("llm.pool.retry_attempt",
                            attempt=attempt + 1,
                            delay=retry_delay)
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue

                # Out of retries
                log.error("llm.pool.request_failed", backend=backend, error=str(e))
                raise PoolUnavailableError(f"Could not connect to pool after {max_retries + 1} attempts: {e}")

            except asyncio.TimeoutError:
                return LLMResponse(
                    success=False,
                    error="timeout",
                    message=f"Request timed out after {timeout_seconds} seconds",
                )

    # --- Direct API Methods ---

    async def _send_to_claude(
        self,
        prompt: str,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 8192,
        timeout_seconds: int = 300,
    ) -> LLMResponse:
        """Send request directly to Claude API."""
        import time

        client = self._get_anthropic_client()
        if not client:
            return LLMResponse(
                success=False,
                error="auth_required",
                message="ANTHROPIC_API_KEY not configured",
            )

        start_time = time.time()

        try:
            # Run sync API call in thread pool
            response = await asyncio.to_thread(
                client.messages.create,
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )

            elapsed = time.time() - start_time

            return LLMResponse(
                success=True,
                text=response.content[0].text,
                backend="claude",
                response_time_seconds=elapsed,
            )

        except Exception as e:
            error_str = str(e)
            if "rate" in error_str.lower() or "429" in error_str:
                return LLMResponse(
                    success=False,
                    error="rate_limited",
                    message=str(e),
                    retry_after_seconds=60,
                )
            return LLMResponse(
                success=False,
                error="api_error",
                message=str(e),
            )

    async def _send_to_openrouter(
        self,
        prompt: str,
        model: str = "deepseek/deepseek-r1",
        timeout_seconds: int = 300,
    ) -> LLMResponse:
        """Send request directly to OpenRouter API."""
        import time

        if not self._openrouter_key:
            return LLMResponse(
                success=False,
                error="auth_required",
                message="OPENROUTER_API_KEY not configured",
            )

        start_time = time.time()

        try:
            session = await self._get_http_session()
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._openrouter_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=aiohttp.ClientTimeout(total=timeout_seconds),
            ) as resp:
                data = await resp.json()

                if resp.status != 200:
                    return LLMResponse(
                        success=False,
                        error="api_error",
                        message=data.get("error", {}).get("message", str(data)),
                    )

                elapsed = time.time() - start_time
                text = data["choices"][0]["message"]["content"]

                return LLMResponse(
                    success=True,
                    text=text,
                    backend="openrouter",
                    response_time_seconds=elapsed,
                )

        except asyncio.TimeoutError:
            return LLMResponse(
                success=False,
                error="timeout",
                message=f"Request timed out after {timeout_seconds} seconds",
            )
        except Exception as e:
            return LLMResponse(
                success=False,
                error="api_error",
                message=str(e),
            )

    # --- Unified Send Method ---

    async def send(
        self,
        backend: str,
        prompt: str,
        *,
        deep_mode: bool = False,
        new_chat: bool = True,
        timeout_seconds: int = 300,
        priority: str = "normal",
        model: Optional[str] = None,
        thread_id: Optional[str] = None,
        images: Optional[list[ImageAttachment]] = None,
    ) -> LLMResponse:
        """
        Send a prompt to an LLM backend.

        Automatically routes to pool (browser) or direct API as appropriate.

        Args:
            backend: Which LLM ("gemini", "chatgpt", "claude", "openrouter")
            prompt: The prompt text
            deep_mode: Use deep/pro mode (browser backends only)
            new_chat: Start new session (browser backends only)
            timeout_seconds: Request timeout
            priority: Request priority (browser backends only)
            model: Specific model to use (API backends only)
            thread_id: Thread ID for recovery correlation (browser backends only)
            images: Optional list of ImageAttachment objects to include

        Returns:
            LLMResponse with the result
        """
        backend = backend.lower()

        if backend in BROWSER_BACKENDS:
            # Route to pool service (gemini, chatgpt, claude)
            return await self._send_via_pool(
                backend, prompt, deep_mode, new_chat, timeout_seconds, priority, thread_id, images
            )

        elif backend == "openrouter":
            # Direct API call
            return await self._send_to_openrouter(
                prompt,
                model=model or "deepseek/deepseek-r1",
                timeout_seconds=timeout_seconds,
            )

        else:
            return LLMResponse(
                success=False,
                error="unknown_backend",
                message=f"Unknown backend: {backend}",
            )

    async def send_parallel(
        self,
        prompts: dict[str, str],
        *,
        deep_mode: bool = False,
        new_chat: bool = True,
        timeout_seconds: int = 300,
        priority: str = "normal",
    ) -> dict[str, LLMResponse]:
        """
        Send prompts to multiple backends in parallel.

        Args:
            prompts: Dict mapping backend name to prompt text
            deep_mode: Use deep/pro mode where available
            new_chat: Start new sessions
            timeout_seconds: Request timeout per backend
            priority: Request priority

        Returns:
            Dict mapping backend name to LLMResponse
        """
        tasks = {}
        for backend, prompt in prompts.items():
            tasks[backend] = self.send(
                backend,
                prompt,
                deep_mode=deep_mode,
                new_chat=new_chat,
                timeout_seconds=timeout_seconds,
                priority=priority,
            )

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        responses = {}
        for backend, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                responses[backend] = LLMResponse(
                    success=False,
                    error="exception",
                    message=str(result),
                    backend=backend,
                )
            else:
                responses[backend] = result

        return responses

    async def get_available_backends(self) -> list[str]:
        """
        Get list of currently available backends.

        Checks pool for browser backends and API keys for API backends.
        """
        available = []

        # Check browser backends via pool (gemini, chatgpt, claude)
        try:
            status = await self.get_pool_status()
            available.extend(status.get_available_backends())
        except PoolUnavailableError:
            pass  # Pool not running, browser backends unavailable

        # Check API backends
        if self._openrouter_key:
            available.append("openrouter")

        return available

    # --- Convenience Methods ---

    async def gemini(
        self,
        prompt: str,
        *,
        deep_think: bool = False,
        new_chat: bool = True,
        timeout_seconds: int = 300,
    ) -> LLMResponse:
        """Send prompt to Gemini (via pool)."""
        return await self.send(
            "gemini", prompt,
            deep_mode=deep_think,
            new_chat=new_chat,
            timeout_seconds=timeout_seconds,
        )

    async def chatgpt(
        self,
        prompt: str,
        *,
        pro_mode: bool = False,
        new_chat: bool = True,
        timeout_seconds: int = 300,
    ) -> LLMResponse:
        """Send prompt to ChatGPT (via pool)."""
        return await self.send(
            "chatgpt", prompt,
            deep_mode=pro_mode,
            new_chat=new_chat,
            timeout_seconds=timeout_seconds,
        )

    async def claude(
        self,
        prompt: str,
        *,
        extended_thinking: bool = False,
        new_chat: bool = True,
        timeout_seconds: int = 300,
    ) -> LLMResponse:
        """Send prompt to Claude (via pool browser)."""
        return await self.send(
            "claude", prompt,
            deep_mode=extended_thinking,
            new_chat=new_chat,
            timeout_seconds=timeout_seconds,
        )

    async def deepseek(
        self,
        prompt: str,
        *,
        timeout_seconds: int = 300,
    ) -> LLMResponse:
        """Send prompt to DeepSeek via OpenRouter (direct API)."""
        return await self.send(
            "openrouter", prompt,
            model="deepseek/deepseek-r1",
            timeout_seconds=timeout_seconds,
        )

    # --- Async Job Methods (new robust submission pattern) ---

    async def submit_job(
        self,
        backend: str,
        prompt: str,
        job_id: str,
        *,
        thread_id: Optional[str] = None,
        deep_mode: bool = False,
        new_chat: bool = True,
        priority: str = "normal",
        images: Optional[list[ImageAttachment]] = None,
    ) -> dict:
        """
        Submit a job for async processing.

        Returns immediately with job submission status.
        Use get_job_status() or wait_for_job() to check completion.

        Args:
            backend: Which LLM ("gemini", "chatgpt", "claude")
            prompt: The prompt text
            job_id: Unique job identifier (for deduplication and tracking)
            thread_id: Optional thread ID for correlation
            deep_mode: Use deep/pro mode
            new_chat: Start new session
            priority: Request priority
            images: Optional list of ImageAttachment objects to include

        Returns:
            {"status": "queued" | "exists" | "cached", "job_id": str, "cached_job_id"?: str}

        Raises:
            PoolUnavailableError: If pool is not available or backend unavailable
        """
        if backend not in BROWSER_BACKENDS:
            raise ValueError(f"Async jobs only support browser backends: {BROWSER_BACKENDS}")

        request_data = {
            "backend": backend,
            "prompt": prompt,
            "job_id": job_id,
            "thread_id": thread_id,
            "deep_mode": deep_mode,
            "new_chat": new_chat,
            "priority": priority,
            "images": [img.to_dict() for img in images] if images else [],
        }

        try:
            session = await self._get_http_session()
            async with session.post(
                f"{self.pool_url}/job/submit",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 503:
                    data = await resp.json()
                    raise PoolUnavailableError(data.get("detail", "Backend unavailable"))
                if resp.status == 400:
                    data = await resp.json()
                    raise ValueError(data.get("detail", "Bad request"))
                resp.raise_for_status()
                return await resp.json()

        except aiohttp.ClientError as e:
            raise PoolUnavailableError(f"Could not connect to pool: {e}")

    async def get_job_status(self, job_id: str) -> Optional[dict]:
        """
        Get the status of a job.

        Args:
            job_id: The job ID

        Returns:
            {job_id, status, queue_position, backend, created_at, started_at, completed_at}
            or None if job not found
        """
        try:
            session = await self._get_http_session()
            async with session.get(
                f"{self.pool_url}/job/{job_id}/status",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 404:
                    return None
                resp.raise_for_status()
                return await resp.json()

        except aiohttp.ClientError as e:
            log.warning("llm.job.status_error", job_id=job_id, error=str(e))
            return None

    async def get_job_result(self, job_id: str) -> Optional[dict]:
        """
        Get the result of a completed job.

        Args:
            job_id: The job ID

        Returns:
            {job_id, status, result?, error?, deep_mode_used, backend, thread_id}
            or None if job not found
        """
        try:
            session = await self._get_http_session()
            async with session.get(
                f"{self.pool_url}/job/{job_id}/result",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 404:
                    return None
                resp.raise_for_status()
                return await resp.json()

        except aiohttp.ClientError as e:
            log.warning("llm.job.result_error", job_id=job_id, error=str(e))
            return None

    async def wait_for_job(
        self,
        job_id: str,
        *,
        poll_interval: float = 3.0,
        timeout_seconds: int = 3600,
    ) -> LLMResponse:
        """
        Wait for a job to complete by polling.

        Args:
            job_id: The job ID to wait for
            poll_interval: Seconds between poll attempts
            timeout_seconds: Maximum wait time before timing out

        Returns:
            LLMResponse with the result
        """
        import time
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                return LLMResponse(
                    success=False,
                    error="timeout",
                    message=f"Job {job_id} did not complete within {timeout_seconds} seconds",
                )

            result = await self.get_job_result(job_id)

            if result is None:
                return LLMResponse(
                    success=False,
                    error="not_found",
                    message=f"Job {job_id} not found",
                )

            status = result.get("status")

            if status == "complete":
                return LLMResponse(
                    success=True,
                    text=result.get("result", ""),
                    backend=result.get("backend"),
                    deep_mode_used=result.get("deep_mode_used", False),
                )

            if status == "failed":
                return LLMResponse(
                    success=False,
                    error="job_failed",
                    message=result.get("error", "Unknown error"),
                    backend=result.get("backend"),
                )

            # Still queued or processing - wait and poll again
            await asyncio.sleep(poll_interval)

    async def send_async(
        self,
        backend: str,
        prompt: str,
        job_id: str,
        *,
        thread_id: Optional[str] = None,
        deep_mode: bool = False,
        new_chat: bool = True,
        priority: str = "normal",
        poll_interval: float = 3.0,
        timeout_seconds: int = 3600,
        images: Optional[list[ImageAttachment]] = None,
    ) -> LLMResponse:
        """
        Submit a job and wait for completion.

        This is a convenience method that combines submit_job() and wait_for_job().
        Unlike send(), this uses the async job system which is more robust
        to pool restarts and timeouts.

        Args:
            backend: Which LLM ("gemini", "chatgpt", "claude")
            prompt: The prompt text
            job_id: Unique job identifier
            thread_id: Optional thread ID for correlation
            deep_mode: Use deep/pro mode
            new_chat: Start new session
            priority: Request priority
            poll_interval: Seconds between status polls
            timeout_seconds: Maximum wait time
            images: Optional list of ImageAttachment objects to include

        Returns:
            LLMResponse with the result
        """
        # Handle cached result
        try:
            submit_result = await self.submit_job(
                backend, prompt, job_id,
                thread_id=thread_id,
                deep_mode=deep_mode,
                new_chat=new_chat,
                priority=priority,
                images=images,
            )
        except PoolUnavailableError as e:
            return LLMResponse(
                success=False,
                error="pool_unavailable",
                message=str(e),
                backend=backend,
            )

        # If we got a cached result, fetch from the cached job
        if submit_result.get("status") == "cached":
            cached_job_id = submit_result.get("cached_job_id")
            log.info("llm.job.cache_hit", job_id=job_id, cached_job_id=cached_job_id)
            return await self.wait_for_job(cached_job_id, poll_interval=poll_interval, timeout_seconds=timeout_seconds)

        # Wait for our job to complete
        return await self.wait_for_job(job_id, poll_interval=poll_interval, timeout_seconds=timeout_seconds)
