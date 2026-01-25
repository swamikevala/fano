"""
LLM Client - Unified interface for LLM access via OpenRouter.

All backends route through OpenRouter API for simplicity.
Supports: gemini, chatgpt, claude, deepseek, and any other OpenRouter model.
"""

import asyncio
import os
import time
from typing import Optional

import aiohttp

from shared.logging import get_logger

from .models import (
    LLMResponse,
    ImageAttachment,
)

log = get_logger("llm", "client")


# Default model mappings (backend alias -> OpenRouter model ID)
DEFAULT_MODELS = {
    "gemini": "google/gemini-2.0-flash-thinking-exp-01-21",
    "chatgpt": "openai/gpt-4o",
    "claude": "anthropic/claude-sonnet-4-20250514",
    "deepseek": "deepseek/deepseek-r1",
}

# Rate limits (requests per minute) - conservative defaults
DEFAULT_RATE_LIMITS = {
    "gemini": 10,
    "chatgpt": 60,
    "claude": 50,
    "deepseek": 10,
}


class RateLimiter:
    """Simple token bucket rate limiter per backend."""

    def __init__(self, limits: dict[str, int] | None = None):
        self._limits = limits or DEFAULT_RATE_LIMITS
        self._tokens: dict[str, float] = {}
        self._last_update: dict[str, float] = {}

    async def acquire(self, backend: str) -> None:
        """Wait until we can make a request to this backend."""
        limit = self._limits.get(backend, 60)  # Default 60 rpm
        tokens_per_second = limit / 60.0

        now = time.time()

        # Initialize if first request
        if backend not in self._tokens:
            self._tokens[backend] = limit
            self._last_update[backend] = now

        # Refill tokens based on time elapsed
        elapsed = now - self._last_update[backend]
        self._tokens[backend] = min(limit, self._tokens[backend] + elapsed * tokens_per_second)
        self._last_update[backend] = now

        # Wait if no tokens available
        if self._tokens[backend] < 1:
            wait_time = (1 - self._tokens[backend]) / tokens_per_second
            log.info("llm.rate_limit.waiting", backend=backend, wait_seconds=wait_time)
            await asyncio.sleep(wait_time)
            self._tokens[backend] = 0
        else:
            self._tokens[backend] -= 1


class LLMClient:
    """
    Unified client for LLM access via OpenRouter.

    All backends route through OpenRouter API using model mappings.
    No browser automation or pool service required.

    Usage:
        client = LLMClient()

        # Send using backend alias
        response = await client.send("gemini", "Hello!")
        response = await client.send("claude", "Explain this...")

        # Send using specific model
        response = await client.send("claude", "Hello!", model="anthropic/claude-3-opus")
    """

    def __init__(
        self,
        openrouter_api_key: Optional[str] = None,
        models: Optional[dict[str, str]] = None,
        rate_limits: Optional[dict[str, int]] = None,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        """
        Initialize the client.

        Args:
            openrouter_api_key: API key for OpenRouter (or uses OPENROUTER_API_KEY env var)
            models: Override default model mappings (backend -> OpenRouter model ID)
            rate_limits: Override rate limits per backend (requests per minute)
            base_url: OpenRouter API base URL
        """
        self._openrouter_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        self._base_url = base_url.rstrip("/")
        self._http_session: Optional[aiohttp.ClientSession] = None

        # Model mappings
        self._models = {**DEFAULT_MODELS}
        if models:
            self._models.update(models)

        # Rate limiter
        self._rate_limiter = RateLimiter(rate_limits)

    async def _get_http_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()
        return self._http_session

    async def close(self):
        """Close HTTP session."""
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
            self._http_session = None

    def get_model(self, backend: str) -> str:
        """Get OpenRouter model ID for a backend alias."""
        return self._models.get(backend.lower(), backend)

    def set_model(self, backend: str, model: str) -> None:
        """Set the model for a backend alias."""
        self._models[backend.lower()] = model

    # --- OpenRouter API Method ---

    async def _send_to_openrouter(
        self,
        prompt: str,
        model: str,
        backend: str,
        system_prompt: Optional[str] = None,
        images: Optional[list[ImageAttachment]] = None,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        timeout_seconds: int = 300,
    ) -> LLMResponse:
        """
        Send request to OpenRouter API.

        Args:
            prompt: User message
            model: OpenRouter model ID (e.g., "anthropic/claude-sonnet-4-20250514")
            backend: Backend alias for logging/tracking
            system_prompt: Optional system message
            images: Optional image attachments
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            timeout_seconds: Request timeout

        Returns:
            LLMResponse with result or error
        """
        if not self._openrouter_key:
            return LLMResponse(
                success=False,
                error="auth_required",
                message="OPENROUTER_API_KEY not configured",
            )

        # Apply rate limiting
        await self._rate_limiter.acquire(backend)

        start_time = time.time()

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Handle images if present
        if images:
            content = [{"type": "text", "text": prompt}]
            for img in images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{img.media_type};base64,{img.data}"
                    }
                })
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})

        request_body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        log.info("llm.openrouter.request",
                 backend=backend,
                 model=model,
                 prompt_length=len(prompt),
                 image_count=len(images) if images else 0)

        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                session = await self._get_http_session()
                async with session.post(
                    f"{self._base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._openrouter_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/fano-project",
                        "X-Title": "Fano Mathematical Explorer",
                    },
                    json=request_body,
                    timeout=aiohttp.ClientTimeout(total=timeout_seconds),
                ) as resp:
                    data = await resp.json()

                    # Handle rate limiting
                    if resp.status == 429:
                        retry_after = int(resp.headers.get("Retry-After", 60))
                        if attempt < max_retries - 1:
                            log.warning("llm.openrouter.rate_limited",
                                       backend=backend,
                                       retry_after=retry_after,
                                       attempt=attempt + 1)
                            await asyncio.sleep(min(retry_after, 60))
                            continue
                        return LLMResponse(
                            success=False,
                            error="rate_limited",
                            message="Rate limited by OpenRouter",
                            backend=backend,
                            retry_after_seconds=retry_after,
                        )

                    # Handle other errors
                    if resp.status != 200:
                        error_msg = data.get("error", {}).get("message", str(data))
                        log.error("llm.openrouter.error",
                                 backend=backend,
                                 status=resp.status,
                                 error=error_msg)
                        return LLMResponse(
                            success=False,
                            error="api_error",
                            message=error_msg,
                            backend=backend,
                        )

                    # Extract response
                    elapsed = time.time() - start_time
                    text = data["choices"][0]["message"]["content"]

                    log.info("llm.openrouter.success",
                            backend=backend,
                            model=model,
                            response_length=len(text),
                            duration_seconds=round(elapsed, 2))

                    return LLMResponse(
                        success=True,
                        text=text,
                        backend=backend,
                        response_time_seconds=elapsed,
                    )

            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    log.warning("llm.openrouter.timeout_retry",
                               backend=backend,
                               attempt=attempt + 1)
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                return LLMResponse(
                    success=False,
                    error="timeout",
                    message=f"Request timed out after {timeout_seconds} seconds",
                    backend=backend,
                )

            except aiohttp.ClientError as e:
                if attempt < max_retries - 1:
                    log.warning("llm.openrouter.connection_error",
                               backend=backend,
                               error=str(e),
                               attempt=attempt + 1)
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                return LLMResponse(
                    success=False,
                    error="connection_error",
                    message=str(e),
                    backend=backend,
                )

            except Exception as e:
                log.error("llm.openrouter.unexpected_error",
                         backend=backend,
                         error=str(e))
                return LLMResponse(
                    success=False,
                    error="api_error",
                    message=str(e),
                    backend=backend,
                )

    # --- Unified Send Method ---

    async def send(
        self,
        backend: str,
        prompt: str,
        *,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        images: Optional[list[ImageAttachment]] = None,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        timeout_seconds: int = 300,
        # Legacy parameters (ignored but accepted for compatibility)
        deep_mode: bool = False,
        new_chat: bool = True,
        priority: str = "normal",
        thread_id: Optional[str] = None,
    ) -> LLMResponse:
        """
        Send a prompt to an LLM backend via OpenRouter.

        Args:
            backend: Backend alias ("gemini", "chatgpt", "claude", "deepseek")
                     or full OpenRouter model ID
            prompt: The prompt text
            model: Override model (uses backend's default if not specified)
            system_prompt: Optional system message
            images: Optional list of ImageAttachment objects
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum response tokens
            timeout_seconds: Request timeout

            # Legacy (accepted but ignored - for backward compatibility):
            deep_mode, new_chat, priority, thread_id

        Returns:
            LLMResponse with the result
        """
        backend = backend.lower()

        # Resolve model: explicit model > backend mapping > backend as model ID
        resolved_model = model or self._models.get(backend, backend)

        return await self._send_to_openrouter(
            prompt=prompt,
            model=resolved_model,
            backend=backend,
            system_prompt=system_prompt,
            images=images,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )

    async def send_parallel(
        self,
        prompts: dict[str, str],
        *,
        timeout_seconds: int = 300,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> dict[str, LLMResponse]:
        """
        Send prompts to multiple backends in parallel.

        Args:
            prompts: Dict mapping backend name to prompt text
            timeout_seconds: Request timeout per backend
            temperature: Sampling temperature
            max_tokens: Maximum response tokens

        Returns:
            Dict mapping backend name to LLMResponse
        """
        tasks = {}
        for backend, prompt in prompts.items():
            tasks[backend] = self.send(
                backend,
                prompt,
                timeout_seconds=timeout_seconds,
                temperature=temperature,
                max_tokens=max_tokens,
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
        Get list of available backends.

        Returns backends that have API key configured.
        """
        if self._openrouter_key:
            return list(self._models.keys())
        return []

    def list_models(self) -> dict[str, str]:
        """Get current backend -> model mappings."""
        return dict(self._models)

    # --- Convenience Methods ---

    async def gemini(
        self,
        prompt: str,
        *,
        timeout_seconds: int = 300,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Send prompt to Gemini via OpenRouter."""
        return await self.send(
            "gemini", prompt,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
        )

    async def chatgpt(
        self,
        prompt: str,
        *,
        timeout_seconds: int = 300,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Send prompt to ChatGPT/GPT-4 via OpenRouter."""
        return await self.send(
            "chatgpt", prompt,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
        )

    async def claude(
        self,
        prompt: str,
        *,
        timeout_seconds: int = 300,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Send prompt to Claude via OpenRouter."""
        return await self.send(
            "claude", prompt,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
        )

    async def deepseek(
        self,
        prompt: str,
        *,
        timeout_seconds: int = 300,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Send prompt to DeepSeek via OpenRouter."""
        return await self.send(
            "deepseek", prompt,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
        )

    # --- Async Send (compatibility alias) ---

    async def send_async(
        self,
        backend: str,
        prompt: str,
        job_id: str = "",
        *,
        thread_id: Optional[str] = None,
        task_type: Optional[str] = None,
        deep_mode: bool = False,
        new_chat: bool = True,
        priority: str = "normal",
        poll_interval: float = 3.0,
        timeout_seconds: int = 3600,
        images: Optional[list[ImageAttachment]] = None,
    ) -> LLMResponse:
        """
        Send a prompt (compatibility method).

        With API-based access, this is equivalent to send() since all calls
        are naturally async. The job_id and polling parameters are ignored.

        Args:
            backend: Which LLM backend
            prompt: The prompt text
            job_id: Ignored (was for pool job tracking)
            images: Optional image attachments
            timeout_seconds: Request timeout

            # Legacy parameters (ignored):
            thread_id, task_type, deep_mode, new_chat, priority, poll_interval

        Returns:
            LLMResponse with the result
        """
        return await self.send(
            backend,
            prompt,
            images=images,
            timeout_seconds=timeout_seconds,
        )
