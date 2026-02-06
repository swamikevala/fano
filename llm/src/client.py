"""LLMClient v2 — OpenRouter API client with rate limiting, circuit breaker,
retry with exponential backoff, and token/cost tracking.

Implements LLMClientInterface from shared.models.
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import aiohttp

from shared.config import Config
from shared.errors import LLMError
from shared.logging import get_logger
from shared.models import (
    BackendState, BackendStatus, Event, EventBusInterface,
    LLMClientInterface, LLMResponse, TokenUsage,
)

log = get_logger("llm", "client")


class RateLimiter:
    """Token bucket rate limiter, per-backend."""

    def __init__(self, limits: dict[str, int]) -> None:
        self._limits = dict(limits)
        self._tokens: dict[str, float] = {}
        self._last_update: dict[str, float] = {}

    async def acquire(self, backend: str) -> None:
        """Wait until a request slot is available for *backend*."""
        limit = self._limits.get(backend, 60)
        tps = limit / 60.0
        now = time.time()
        if backend not in self._tokens:
            self._tokens[backend] = float(limit)
            self._last_update[backend] = now
        elapsed = now - self._last_update[backend]
        self._tokens[backend] = min(float(limit), self._tokens[backend] + elapsed * tps)
        self._last_update[backend] = now
        if self._tokens[backend] < 1.0:
            wait = (1.0 - self._tokens[backend]) / tps
            log.info("llm.rate_limit.waiting", backend=backend, wait_seconds=round(wait, 3))
            await asyncio.sleep(wait)
            self._tokens[backend] = 0.0
        else:
            self._tokens[backend] -= 1.0


class CircuitBreaker:
    """Per-backend circuit breaker: closed -> open -> half_open -> closed."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout_seconds: int = 60) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout_seconds
        self._backends: dict[str, dict[str, Any]] = {}

    def _ensure(self, backend: str) -> dict[str, Any]:
        if backend not in self._backends:
            self._backends[backend] = {
                "state": BackendState.CLOSED, "failure_count": 0,
                "opened_at": 0.0, "last_failure_at": None,
            }
        return self._backends[backend]

    def can_request(self, backend: str) -> bool:
        """Return True if requests should be attempted."""
        info = self._ensure(backend)
        if info["state"] == BackendState.CLOSED:
            return True
        if info["state"] == BackendState.OPEN:
            if time.time() - info["opened_at"] >= self._recovery_timeout:
                info["state"] = BackendState.HALF_OPEN
                return True
            return False
        return True  # half_open: allow test request

    def record_success(self, backend: str) -> None:
        info = self._ensure(backend)
        info["state"] = BackendState.CLOSED
        info["failure_count"] = 0

    def record_failure(self, backend: str) -> None:
        info = self._ensure(backend)
        info["failure_count"] += 1
        info["last_failure_at"] = datetime.now(timezone.utc)
        if info["failure_count"] >= self._failure_threshold:
            info["state"] = BackendState.OPEN
            info["opened_at"] = time.time()

    def get_state(self, backend: str) -> BackendState:
        return self._ensure(backend)["state"]

    def get_failure_count(self, backend: str) -> int:
        return self._ensure(backend)["failure_count"]

    def get_last_failure_at(self, backend: str) -> datetime | None:
        return self._ensure(backend)["last_failure_at"]


class CostEstimator:
    """Estimates USD cost from token counts. Pricing per 1M tokens: (input, output)."""

    MODEL_PRICING: dict[str, tuple[float, float]] = {
        "google/gemini-flash": (0.075, 0.30),
        "google/gemini-2.0-flash-thinking-exp-01-21": (0.0, 0.0),
        "google/gemini-pro": (0.125, 0.375),
        "openai/gpt-4o": (2.50, 10.00),
        "openai/gpt-4o-mini": (0.15, 0.60),
        "openai/gpt-4-turbo": (10.00, 30.00),
        "anthropic/claude-sonnet": (3.00, 15.00),
        "anthropic/claude-sonnet-4-20250514": (3.00, 15.00),
        "anthropic/claude-haiku": (0.25, 1.25),
        "anthropic/claude-opus": (15.00, 75.00),
        "deepseek/deepseek-r1": (0.55, 2.19),
        "deepseek/deepseek-chat": (0.14, 0.28),
    }

    def estimate(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Return estimated cost in USD. Returns 0.0 for unknown models."""
        pricing = self.MODEL_PRICING.get(model)
        if pricing is None:
            return 0.0
        inp, out = pricing
        return (prompt_tokens * inp + completion_tokens * out) / 1_000_000


class LLMClient(LLMClientInterface):
    """OpenRouter API client implementing LLMClientInterface."""

    def __init__(self, config: Config, event_bus: EventBusInterface) -> None:
        self._config = config
        self._event_bus = event_bus
        self._api_key = os.environ.get(config.get("llm.api_key_env", "OPENROUTER_API_KEY"), "")
        self._endpoint = config.get("llm.endpoint", "https://openrouter.ai/api/v1").rstrip("/")
        self._models: dict[str, str] = dict(config.get("llm.models", {}))
        self._default_timeout: int = int(config.get("llm.default_timeout_seconds", 30))
        self._max_retries: int = int(config.get("llm.max_retries", 2))
        self._rate_limiter = RateLimiter(config.get("llm.rate_limits", {}))
        self._circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout_seconds=60)
        self._cost_estimator = CostEstimator()
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # -- Public API --------------------------------------------------------

    async def send(self, backend: str, prompt: str, **kwargs: object) -> LLMResponse:
        """Send prompt to *backend* via OpenRouter.

        Kwargs: module, temperature, max_tokens, timeout.
        """
        return await self._do_send(
            backend=backend, prompt=prompt,
            module=str(kwargs.get("module", "unknown")),
            temperature=float(kwargs.get("temperature", 0.7)),
            max_tokens=kwargs.get("max_tokens"),  # type: ignore[arg-type]
            timeout=int(kwargs.get("timeout") or self._default_timeout),
        )

    async def send_structured(self, backend: str, prompt: str, schema: dict, **kwargs: object) -> LLMResponse:
        """Send prompt expecting JSON response. Adds response_format to API call."""
        return await self._do_send(
            backend=backend, prompt=prompt,
            module=str(kwargs.get("module", "unknown")),
            temperature=0.7, max_tokens=None, timeout=self._default_timeout,
            extra_body={"response_format": {"type": "json_object"}},
        )

    def get_available_backends(self) -> list[str]:
        return [n for n in self._models if self._circuit_breaker.can_request(n)]

    def get_backend_status(self, backend: str) -> BackendStatus:
        rpm = int(self._config.get("llm.rate_limits", {}).get(backend, 60))
        return BackendStatus(
            name=backend, state=self._circuit_breaker.get_state(backend),
            requests_per_minute=rpm, avg_latency_ms=0.0,
            failure_count=self._circuit_breaker.get_failure_count(backend),
            last_failure_at=self._circuit_breaker.get_last_failure_at(backend),
        )

    # -- Internal ----------------------------------------------------------

    async def _do_send(
        self, *, backend: str, prompt: str, module: str,
        temperature: float, max_tokens: int | None, timeout: int,
        extra_body: dict | None = None,
    ) -> LLMResponse:
        if not self._circuit_breaker.can_request(backend):
            raise LLMError(f"Circuit breaker open for {backend}", backend=backend, is_transient=True)

        model = self._models.get(backend, backend)
        last_error = ""
        total_attempts = 1 + self._max_retries

        for attempt in range(total_attempts):
            await self._rate_limiter.acquire(backend)
            body: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
            }
            if max_tokens is not None:
                body["max_tokens"] = max_tokens
            if extra_body:
                body.update(extra_body)

            start = time.time()
            try:
                session = await self._get_session()
                async with session.post(
                    f"{self._endpoint}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/fano-project",
                        "X-Title": "Fano Research Assistant",
                    },
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    data = await resp.json()
                    elapsed_ms = (time.time() - start) * 1000

                    if resp.status == 200:
                        text = data["choices"][0]["message"]["content"]
                        usage = data.get("usage", {})
                        p_tok = usage.get("prompt_tokens", 0)
                        c_tok = usage.get("completion_tokens", 0)
                        cost = self._cost_estimator.estimate(model, p_tok, c_tok)
                        token_usage = TokenUsage(
                            prompt_tokens=p_tok, completion_tokens=c_tok,
                            total_tokens=p_tok + c_tok, estimated_cost_usd=cost,
                        )
                        self._circuit_breaker.record_success(backend)
                        response = LLMResponse(
                            success=True, text=text, backend=backend,
                            model=model, token_usage=token_usage, error=None,
                        )
                        await self._publish_completed(backend, model, module, elapsed_ms, token_usage)
                        return response

                    error_msg = data.get("error", {}).get("message", str(data))
                    last_error = f"HTTP {resp.status}: {error_msg}"
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_error = str(exc)

            if attempt < total_attempts - 1:
                delay = 2 ** attempt * 0.5
                log.warning("llm.request.retrying", backend=backend, attempt=attempt + 1,
                            delay_seconds=delay, error=last_error)
                await asyncio.sleep(delay)

        self._circuit_breaker.record_failure(backend)
        await self._publish_failed(backend, model, module, last_error)
        raise LLMError(
            f"All {total_attempts} attempts failed for {backend}: {last_error}",
            backend=backend, is_transient=True,
        )

    # -- Event helpers -----------------------------------------------------

    async def _publish_completed(
        self, backend: str, model: str, module: str,
        elapsed_ms: float, token_usage: TokenUsage,
    ) -> None:
        await self._event_bus.publish(Event(
            topic="llm.request.completed", timestamp=datetime.now(timezone.utc),
            source="llm.client", correlation_id=str(uuid.uuid4()),
            payload={
                "backend": backend, "model": model, "module": module, "success": True,
                "elapsed_ms": round(elapsed_ms, 1),
                "prompt_tokens": token_usage.prompt_tokens,
                "completion_tokens": token_usage.completion_tokens,
                "estimated_cost_usd": token_usage.estimated_cost_usd,
            },
        ))

    async def _publish_failed(self, backend: str, model: str, module: str, error: str) -> None:
        await self._event_bus.publish(Event(
            topic="llm.request.failed", timestamp=datetime.now(timezone.utc),
            source="llm.client", correlation_id=str(uuid.uuid4()),
            payload={
                "backend": backend, "model": model, "module": module,
                "success": False, "error": error,
            },
        ))
