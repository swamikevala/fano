"""
LLM Client - Unified interface for LLM access.

Routes requests appropriately:
- Browser backends (Gemini, ChatGPT) → Pool service
- API backends (Claude, OpenRouter) → Direct API calls
"""

import asyncio
import logging
import os
from typing import Optional

import aiohttp

from .models import (
    LLMResponse,
    Backend,
    Priority,
    PoolStatus,
    BackendStatus,
)

logger = logging.getLogger(__name__)

# Backends that require browser automation (go through pool)
BROWSER_BACKENDS = {"gemini", "chatgpt"}

# Backends that use direct API calls
API_BACKENDS = {"claude", "openrouter"}


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
    ) -> LLMResponse:
        """Send request via pool service (for browser backends)."""
        request_data = {
            "backend": backend,
            "prompt": prompt,
            "options": {
                "deep_mode": deep_mode,
                "new_chat": new_chat,
                "timeout_seconds": timeout_seconds,
                "priority": priority,
            },
        }

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
            logger.error(f"Pool request failed: {e}")
            raise PoolUnavailableError(f"Could not connect to pool: {e}")

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

        Returns:
            LLMResponse with the result
        """
        backend = backend.lower()

        if backend in BROWSER_BACKENDS:
            # Route to pool service
            return await self._send_via_pool(
                backend, prompt, deep_mode, new_chat, timeout_seconds, priority
            )

        elif backend == "claude":
            # Direct API call
            return await self._send_to_claude(
                prompt,
                model=model or "claude-sonnet-4-20250514",
                timeout_seconds=timeout_seconds,
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

        # Check browser backends via pool
        try:
            status = await self.get_pool_status()
            available.extend(status.get_available_backends())
        except PoolUnavailableError:
            pass  # Pool not running, browser backends unavailable

        # Check API backends
        if self._anthropic_key:
            available.append("claude")
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
        model: str = "claude-sonnet-4-20250514",
        timeout_seconds: int = 300,
    ) -> LLMResponse:
        """Send prompt to Claude (direct API)."""
        return await self.send(
            "claude", prompt,
            model=model,
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
