"""Tests for LLMClient v2 — OpenRouter client with circuit breaker, rate limiter, cost tracking.

Tests follow the Design Spec Section 6.5 test matrix.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.config import Config
from shared.errors import LLMError
from shared.events import EventBus
from shared.models import BackendState, LLMResponse, TokenUsage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config() -> Config:
    """Build a Config object for tests (requires OPENROUTER_API_KEY in env)."""
    return Config.from_dict({
        "llm": {
            "api_key_env": "OPENROUTER_API_KEY",
            "endpoint": "https://openrouter.ai/api/v1",
            "models": {
                "gemini": "google/gemini-flash",
                "chatgpt": "openai/gpt-4o",
                "claude": "anthropic/claude-sonnet",
            },
            "rate_limits": {"gemini": 10, "chatgpt": 60, "claude": 50},
            "default_timeout_seconds": 30,
            "max_retries": 2,
        },
        "consensus": {"backends": ["gemini", "chatgpt", "claude"]},
    })


def _ok_json(text: str = "Hello world", prompt_tok: int = 10, comp_tok: int = 20) -> dict:
    """Simulate a successful OpenRouter JSON response body."""
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {
            "prompt_tokens": prompt_tok,
            "completion_tokens": comp_tok,
            "total_tokens": prompt_tok + comp_tok,
        },
    }


def _mock_response(status: int = 200, json_data: dict | None = None) -> MagicMock:
    """Create a mock aiohttp response."""
    resp = MagicMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_data or _ok_json())
    resp.headers = {}
    return resp


def _mock_session(responses: list[MagicMock]) -> MagicMock:
    """Create a mock aiohttp.ClientSession whose post() yields *responses* in order."""
    session = AsyncMock()
    call_idx = {"i": 0}

    def _post(*args, **kwargs):
        ctx = AsyncMock()
        idx = call_idx["i"]
        call_idx["i"] += 1
        resp = responses[idx] if idx < len(responses) else responses[-1]
        ctx.__aenter__ = AsyncMock(return_value=resp)
        ctx.__aexit__ = AsyncMock(return_value=False)
        return ctx

    session.post = MagicMock(side_effect=_post)
    session.close = AsyncMock()
    return session


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


@pytest.fixture
def config() -> Config:
    return _make_config()


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus(store=None)


@pytest.fixture
def client(config: Config, event_bus: EventBus):
    from llm.src.client import LLMClient
    return LLMClient(config, event_bus)


# ---------------------------------------------------------------------------
# test_send_success
# ---------------------------------------------------------------------------

async def test_send_success(client, event_bus: EventBus):
    """Mock HTTP 200 -> returns LLMResponse with text."""
    from llm.src.client import LLMClient

    session = _mock_session([_mock_response(200, _ok_json("Math is great"))])

    with patch("aiohttp.ClientSession", return_value=session):
        resp = await client.send("gemini", "What is math?")

    assert isinstance(resp, LLMResponse)
    assert resp.success is True
    assert resp.text == "Math is great"
    assert resp.backend == "gemini"
    assert resp.model == "google/gemini-flash"
    assert resp.error is None


# ---------------------------------------------------------------------------
# test_send_retry_on_500
# ---------------------------------------------------------------------------

async def test_send_retry_on_500(client):
    """Mock HTTP 500 -> retries -> succeeds on 2nd try."""
    fail = _mock_response(500, {"error": {"message": "Internal error"}})
    ok = _mock_response(200, _ok_json("Recovered"))

    session = _mock_session([fail, ok])

    with patch("aiohttp.ClientSession", return_value=session):
        with patch("asyncio.sleep", new_callable=AsyncMock):
            resp = await client.send("gemini", "Retry test")

    assert resp.success is True
    assert resp.text == "Recovered"


# ---------------------------------------------------------------------------
# test_send_raises_after_max_retries
# ---------------------------------------------------------------------------

async def test_send_raises_after_max_retries(client):
    """All retries fail -> LLMError raised."""
    fail = _mock_response(500, {"error": {"message": "Server error"}})
    # max_retries=2 means initial + 2 retries = 3 total attempts
    session = _mock_session([fail, fail, fail])

    with patch("aiohttp.ClientSession", return_value=session):
        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(LLMError) as exc_info:
                await client.send("gemini", "Fail test")

    assert exc_info.value.backend == "gemini"
    assert exc_info.value.is_transient is True


# ---------------------------------------------------------------------------
# test_rate_limiter_blocks
# ---------------------------------------------------------------------------

async def test_rate_limiter_blocks():
    """Exceeding rate limit causes wait (sleep is called)."""
    from llm.src.client import RateLimiter

    # 1 request per minute — after consuming the initial token, the next must wait
    limiter = RateLimiter({"slow": 1})

    # First acquire should succeed without waiting
    await limiter.acquire("slow")

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        await limiter.acquire("slow")
        mock_sleep.assert_called_once()
        # The wait should be a positive duration
        assert mock_sleep.call_args[0][0] > 0


# ---------------------------------------------------------------------------
# test_circuit_breaker_opens
# ---------------------------------------------------------------------------

async def test_circuit_breaker_opens():
    """5 failures -> circuit opens -> can_request returns False."""
    from llm.src.client import CircuitBreaker

    cb = CircuitBreaker(failure_threshold=5, recovery_timeout_seconds=60)

    for _ in range(5):
        cb.record_failure("gemini")

    assert cb.can_request("gemini") is False


# ---------------------------------------------------------------------------
# test_circuit_breaker_half_open
# ---------------------------------------------------------------------------

async def test_circuit_breaker_half_open():
    """After recovery timeout, allows a test request (half_open)."""
    from llm.src.client import CircuitBreaker

    cb = CircuitBreaker(failure_threshold=5, recovery_timeout_seconds=1)

    for _ in range(5):
        cb.record_failure("gemini")
    assert cb.can_request("gemini") is False

    # Simulate time passing beyond recovery timeout
    cb._backends["gemini"]["opened_at"] = time.time() - 2

    assert cb.can_request("gemini") is True
    assert cb._backends["gemini"]["state"] == BackendState.HALF_OPEN


# ---------------------------------------------------------------------------
# test_circuit_breaker_recovers
# ---------------------------------------------------------------------------

async def test_circuit_breaker_recovers():
    """Success in half_open -> transitions back to closed."""
    from llm.src.client import CircuitBreaker

    cb = CircuitBreaker(failure_threshold=5, recovery_timeout_seconds=1)

    # Open the circuit
    for _ in range(5):
        cb.record_failure("gemini")

    # Move to half_open
    cb._backends["gemini"]["opened_at"] = time.time() - 2
    assert cb.can_request("gemini") is True  # transitions to half_open

    # Record success -> should go back to closed
    cb.record_success("gemini")
    assert cb._backends["gemini"]["state"] == BackendState.CLOSED
    assert cb._backends["gemini"]["failure_count"] == 0


# ---------------------------------------------------------------------------
# test_token_usage_tracked
# ---------------------------------------------------------------------------

async def test_token_usage_tracked(client):
    """Response includes TokenUsage with correct counts."""
    data = _ok_json("Token test", prompt_tok=15, comp_tok=25)
    session = _mock_session([_mock_response(200, data)])

    with patch("aiohttp.ClientSession", return_value=session):
        resp = await client.send("gemini", "Token test")

    assert resp.token_usage is not None
    assert isinstance(resp.token_usage, TokenUsage)
    assert resp.token_usage.prompt_tokens == 15
    assert resp.token_usage.completion_tokens == 25
    assert resp.token_usage.total_tokens == 40


# ---------------------------------------------------------------------------
# test_cost_estimation
# ---------------------------------------------------------------------------

async def test_cost_estimation():
    """Estimated cost matches expected for known model pricing."""
    from llm.src.client import CostEstimator

    estimator = CostEstimator()
    # google/gemini-flash should be in the pricing table
    cost = estimator.estimate("google/gemini-flash", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert cost > 0.0

    # Unknown model should return 0.0
    unknown = estimator.estimate("unknown/model", prompt_tokens=100, completion_tokens=100)
    assert unknown == 0.0


# ---------------------------------------------------------------------------
# test_event_published_on_success
# ---------------------------------------------------------------------------

async def test_event_published_on_success(client, event_bus: EventBus):
    """llm.request.completed event published on success."""
    captured: list = []

    async def handler(event):
        captured.append(event)

    event_bus.subscribe("llm.request.completed", handler)

    session = _mock_session([_mock_response(200, _ok_json("Event test"))])
    with patch("aiohttp.ClientSession", return_value=session):
        await client.send("gemini", "Event test")

    assert len(captured) == 1
    assert captured[0].topic == "llm.request.completed"
    assert captured[0].payload["backend"] == "gemini"
    assert captured[0].payload["success"] is True


# ---------------------------------------------------------------------------
# test_event_published_on_failure
# ---------------------------------------------------------------------------

async def test_event_published_on_failure(client, event_bus: EventBus):
    """llm.request.failed event published on failure."""
    captured: list = []

    async def handler(event):
        captured.append(event)

    event_bus.subscribe("llm.request.failed", handler)

    fail = _mock_response(500, {"error": {"message": "Boom"}})
    session = _mock_session([fail, fail, fail])

    with patch("aiohttp.ClientSession", return_value=session):
        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(LLMError):
                await client.send("gemini", "Fail event test")

    assert len(captured) == 1
    assert captured[0].topic == "llm.request.failed"
    assert captured[0].payload["backend"] == "gemini"


# ---------------------------------------------------------------------------
# test_available_backends_excludes_open_circuits
# ---------------------------------------------------------------------------

async def test_available_backends_excludes_open_circuits(client):
    """get_available_backends() skips backends with open circuits."""
    # All should be available initially
    available = client.get_available_backends()
    assert "gemini" in available
    assert "chatgpt" in available
    assert "claude" in available

    # Open the gemini circuit
    for _ in range(5):
        client._circuit_breaker.record_failure("gemini")

    available = client.get_available_backends()
    assert "gemini" not in available
    assert "chatgpt" in available
    assert "claude" in available


# ---------------------------------------------------------------------------
# test_send_structured
# ---------------------------------------------------------------------------

async def test_send_structured(client):
    """send_structured adds response_format to the API call."""
    json_body = _ok_json('{"answer": 42}')
    session = _mock_session([_mock_response(200, json_body)])

    with patch("aiohttp.ClientSession", return_value=session):
        resp = await client.send_structured(
            "gemini", "Return JSON", schema={"type": "object"}
        )

    assert resp.success is True
    assert resp.text == '{"answer": 42}'

    # Verify that the request body included response_format
    call_args = session.post.call_args
    request_body = call_args.kwargs.get("json") or call_args[1].get("json")
    assert request_body["response_format"] == {"type": "json_object"}


# ---------------------------------------------------------------------------
# test_get_backend_status
# ---------------------------------------------------------------------------

async def test_get_backend_status(client):
    """get_backend_status returns a BackendStatus with correct fields."""
    from shared.models import BackendStatus

    status = client.get_backend_status("gemini")
    assert isinstance(status, BackendStatus)
    assert status.name == "gemini"
    assert status.state == BackendState.CLOSED
    assert status.failure_count == 0


# ---------------------------------------------------------------------------
# test_circuit_breaker_blocks_send
# ---------------------------------------------------------------------------

async def test_circuit_breaker_blocks_send(client):
    """When circuit is open, send raises LLMError(is_transient=True) immediately."""
    # Open the circuit
    for _ in range(5):
        client._circuit_breaker.record_failure("gemini")

    with pytest.raises(LLMError) as exc_info:
        await client.send("gemini", "Should not reach API")

    assert exc_info.value.is_transient is True
    assert exc_info.value.backend == "gemini"
