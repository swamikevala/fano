"""Tests for LLM client."""

import asyncio
import os
import pytest
import aiohttp
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.client import LLMClient, PoolUnavailableError, BROWSER_BACKENDS, API_BACKENDS
from src.models import LLMResponse, PoolStatus, BackendStatus


class TestLLMClientInit:
    """Tests for LLMClient initialization."""

    def test_default_pool_url(self):
        client = LLMClient()
        assert client.pool_url == "http://127.0.0.1:9000"

    def test_custom_pool_url(self):
        client = LLMClient(pool_url="http://custom:8080/")
        assert client.pool_url == "http://custom:8080"  # Trailing slash stripped

    def test_anthropic_key_from_arg(self):
        client = LLMClient(anthropic_api_key="test-key")
        assert client._anthropic_key == "test-key"

    def test_anthropic_key_from_env(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-key"}):
            client = LLMClient()
            assert client._anthropic_key == "env-key"

    def test_openrouter_key_from_arg(self):
        client = LLMClient(openrouter_api_key="or-key")
        assert client._openrouter_key == "or-key"

    def test_openrouter_key_from_env(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-env-key"}):
            client = LLMClient()
            assert client._openrouter_key == "or-env-key"

    def test_lazy_http_session(self):
        client = LLMClient()
        assert client._http_session is None

    def test_lazy_anthropic_client(self):
        client = LLMClient()
        assert client._anthropic_client is None


class TestLLMClientHttpSession:
    """Tests for HTTP session management."""

    @pytest.mark.asyncio
    async def test_get_http_session_creates_session(self):
        client = LLMClient()

        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.closed = False
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            session = await client._get_http_session()

            mock_session_cls.assert_called_once()
            assert session == mock_session

            await client.close()

    @pytest.mark.asyncio
    async def test_get_http_session_reuses_session(self):
        client = LLMClient()

        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.closed = False
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            session1 = await client._get_http_session()
            session2 = await client._get_http_session()

            assert mock_session_cls.call_count == 1
            assert session1 == session2

            await client.close()

    @pytest.mark.asyncio
    async def test_close_closes_session(self):
        client = LLMClient()

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()
        client._http_session = mock_session

        await client.close()

        mock_session.close.assert_called_once()
        assert client._http_session is None


class TestLLMClientAnthropicClient:
    """Tests for Anthropic client management."""

    def test_get_anthropic_client_creates_client(self):
        client = LLMClient(anthropic_api_key="test-key")

        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client

            result = client._get_anthropic_client()

            mock_anthropic.assert_called_once_with(api_key="test-key")
            assert result == mock_client

    def test_get_anthropic_client_returns_none_without_key(self):
        client = LLMClient()
        client._anthropic_key = None

        result = client._get_anthropic_client()

        assert result is None


class TestLLMClientPoolMethods:
    """Tests for pool service methods."""

    @pytest.mark.asyncio
    async def test_is_pool_available_true(self):
        client = LLMClient()

        mock_response = MagicMock()
        mock_response.status = 200

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)))

        with patch.object(client, "_get_http_session", return_value=mock_session):
            result = await client.is_pool_available()

        assert result is True

    @pytest.mark.asyncio
    async def test_is_pool_available_false_on_error(self):
        client = LLMClient()

        with patch.object(client, "_get_http_session", side_effect=Exception("Connection failed")):
            result = await client.is_pool_available()

        assert result is False

    @pytest.mark.asyncio
    async def test_get_pool_status_success(self):
        client = LLMClient()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "gemini": {"available": True, "authenticated": True, "rate_limited": False},
            "chatgpt": {"available": True, "authenticated": True, "rate_limited": False},
            "claude": {"available": True, "authenticated": True, "rate_limited": False},
        })

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_context)

        with patch.object(client, "_get_http_session", return_value=mock_session):
            status = await client.get_pool_status()

        assert isinstance(status, PoolStatus)
        assert status.gemini.available is True

    @pytest.mark.asyncio
    async def test_get_pool_status_error(self):
        client = LLMClient()

        mock_response = MagicMock()
        mock_response.status = 500

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_context)

        with patch.object(client, "_get_http_session", return_value=mock_session):
            with pytest.raises(PoolUnavailableError):
                await client.get_pool_status()


class TestLLMClientSend:
    """Tests for the unified send method."""

    @pytest.mark.asyncio
    async def test_send_routes_gemini_to_pool(self):
        client = LLMClient()

        expected_response = LLMResponse(success=True, text="Hello")

        with patch.object(client, "_send_via_pool", return_value=expected_response) as mock_pool:
            response = await client.send("gemini", "Test prompt")

            mock_pool.assert_called_once_with(
                "gemini", "Test prompt", False, True, 300, "normal", None
            )
            assert response == expected_response

    @pytest.mark.asyncio
    async def test_send_routes_chatgpt_to_pool(self):
        client = LLMClient()

        expected_response = LLMResponse(success=True, text="Hello")

        with patch.object(client, "_send_via_pool", return_value=expected_response) as mock_pool:
            response = await client.send("chatgpt", "Test prompt", deep_mode=True)

            mock_pool.assert_called_once_with(
                "chatgpt", "Test prompt", True, True, 300, "normal", None
            )

    @pytest.mark.asyncio
    async def test_send_routes_claude_to_api(self):
        client = LLMClient()

        expected_response = LLMResponse(success=True, text="Hello from Claude")

        with patch.object(client, "_send_to_claude", return_value=expected_response) as mock_api:
            response = await client.send("claude", "Test prompt")

            mock_api.assert_called_once_with(
                "Test prompt",
                model="claude-sonnet-4-20250514",
                timeout_seconds=300,
            )
            assert response == expected_response

    @pytest.mark.asyncio
    async def test_send_routes_openrouter_to_api(self):
        client = LLMClient()

        expected_response = LLMResponse(success=True, text="Hello from OpenRouter")

        with patch.object(client, "_send_to_openrouter", return_value=expected_response) as mock_api:
            response = await client.send("openrouter", "Test prompt")

            mock_api.assert_called_once_with(
                "Test prompt",
                model="deepseek/deepseek-r1",
                timeout_seconds=300,
            )

    @pytest.mark.asyncio
    async def test_send_unknown_backend(self):
        client = LLMClient()

        response = await client.send("unknown", "Test")

        assert response.success is False
        assert response.error == "unknown_backend"

    @pytest.mark.asyncio
    async def test_send_with_custom_model(self):
        client = LLMClient()

        expected_response = LLMResponse(success=True, text="Hello")

        with patch.object(client, "_send_to_claude", return_value=expected_response) as mock_api:
            await client.send("claude", "Test", model="claude-opus-4-20250514")

            mock_api.assert_called_once_with(
                "Test",
                model="claude-opus-4-20250514",
                timeout_seconds=300,
            )


class TestLLMClientSendViaPool:
    """Tests for _send_via_pool method."""

    @pytest.mark.asyncio
    async def test_send_via_pool_success(self, mock_pool_response):
        client = LLMClient()

        mock_response = MagicMock()
        mock_response.json = AsyncMock(return_value=mock_pool_response)

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_context)

        with patch.object(client, "_get_http_session", return_value=mock_session):
            response = await client._send_via_pool(
                "gemini", "Test", False, True, 300, "normal"
            )

        assert response.success is True
        assert response.text == "Hello from pool!"
        assert response.backend == "gemini"


class TestLLMClientSendToClaude:
    """Tests for _send_to_claude method."""

    @pytest.mark.asyncio
    async def test_send_to_claude_success(self, mock_anthropic_client):
        client = LLMClient(anthropic_api_key="test-key")

        with patch.object(client, "_get_anthropic_client", return_value=mock_anthropic_client):
            response = await client._send_to_claude("Test prompt")

        assert response.success is True
        assert response.text == "Claude response"
        assert response.backend == "claude"

    @pytest.mark.asyncio
    async def test_send_to_claude_no_api_key(self):
        client = LLMClient()
        client._anthropic_key = None

        response = await client._send_to_claude("Test prompt")

        assert response.success is False
        assert response.error == "auth_required"

    @pytest.mark.asyncio
    async def test_send_to_claude_rate_limit(self):
        client = LLMClient(anthropic_api_key="test-key")

        mock_client = MagicMock()
        mock_client.messages.create = MagicMock(
            side_effect=Exception("Rate limit exceeded (429)")
        )

        with patch.object(client, "_get_anthropic_client", return_value=mock_client):
            response = await client._send_to_claude("Test prompt")

        assert response.success is False
        assert response.error == "rate_limited"
        assert response.retry_after_seconds == 60


class TestLLMClientSendToOpenRouter:
    """Tests for _send_to_openrouter method."""

    @pytest.mark.asyncio
    async def test_send_to_openrouter_no_api_key(self):
        client = LLMClient()
        client._openrouter_key = None

        response = await client._send_to_openrouter("Test prompt")

        assert response.success is False
        assert response.error == "auth_required"

    @pytest.mark.asyncio
    async def test_send_to_openrouter_success(self):
        client = LLMClient(openrouter_api_key="or-key")

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{"message": {"content": "DeepSeek response"}}]
        })

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_context)

        with patch.object(client, "_get_http_session", return_value=mock_session):
            response = await client._send_to_openrouter("Test prompt")

        assert response.success is True
        assert response.text == "DeepSeek response"
        assert response.backend == "openrouter"


class TestLLMClientSendParallel:
    """Tests for send_parallel method."""

    @pytest.mark.asyncio
    async def test_send_parallel_all_success(self):
        client = LLMClient()

        responses = {
            "gemini": LLMResponse(success=True, text="Gemini says hi"),
            "claude": LLMResponse(success=True, text="Claude says hi"),
        }

        async def mock_send(backend, prompt, **kwargs):
            return responses[backend]

        with patch.object(client, "send", side_effect=mock_send):
            results = await client.send_parallel({
                "gemini": "Test",
                "claude": "Test",
            })

        assert results["gemini"].text == "Gemini says hi"
        assert results["claude"].text == "Claude says hi"

    @pytest.mark.asyncio
    async def test_send_parallel_handles_exceptions(self):
        client = LLMClient()

        async def mock_send(backend, prompt, **kwargs):
            if backend == "gemini":
                raise Exception("Gemini failed")
            return LLMResponse(success=True, text="Claude says hi")

        with patch.object(client, "send", side_effect=mock_send):
            results = await client.send_parallel({
                "gemini": "Test",
                "claude": "Test",
            })

        assert results["gemini"].success is False
        assert results["gemini"].error == "exception"
        assert "Gemini failed" in results["gemini"].message
        assert results["claude"].success is True


class TestLLMClientGetAvailableBackends:
    """Tests for get_available_backends method."""

    @pytest.mark.asyncio
    async def test_includes_pool_backends(self):
        client = LLMClient()

        mock_status = PoolStatus(
            gemini=BackendStatus(available=True, authenticated=True, rate_limited=False),
            chatgpt=BackendStatus(available=False, authenticated=True, rate_limited=True),
            claude=BackendStatus(available=True, authenticated=True, rate_limited=False),
        )

        with patch.object(client, "get_pool_status", return_value=mock_status):
            available = await client.get_available_backends()

        assert "gemini" in available
        assert "chatgpt" not in available

    @pytest.mark.asyncio
    async def test_includes_api_backends_with_keys(self):
        client = LLMClient(anthropic_api_key="key", openrouter_api_key="key")

        with patch.object(client, "get_pool_status", side_effect=PoolUnavailableError()):
            available = await client.get_available_backends()

        assert "claude" in available
        assert "openrouter" in available

    @pytest.mark.asyncio
    async def test_handles_pool_unavailable(self):
        client = LLMClient(anthropic_api_key="key")

        with patch.object(client, "get_pool_status", side_effect=PoolUnavailableError()):
            available = await client.get_available_backends()

        # Pool backends not available, but Claude is
        assert "gemini" not in available
        assert "claude" in available


class TestLLMClientConvenienceMethods:
    """Tests for convenience methods."""

    @pytest.mark.asyncio
    async def test_gemini_method(self):
        client = LLMClient()
        expected = LLMResponse(success=True, text="Hello")

        with patch.object(client, "send", return_value=expected) as mock_send:
            result = await client.gemini("Test", deep_think=True)

            mock_send.assert_called_once_with(
                "gemini", "Test",
                deep_mode=True,
                new_chat=True,
                timeout_seconds=300,
            )
            assert result == expected

    @pytest.mark.asyncio
    async def test_chatgpt_method(self):
        client = LLMClient()
        expected = LLMResponse(success=True, text="Hello")

        with patch.object(client, "send", return_value=expected) as mock_send:
            result = await client.chatgpt("Test", pro_mode=True)

            mock_send.assert_called_once_with(
                "chatgpt", "Test",
                deep_mode=True,
                new_chat=True,
                timeout_seconds=300,
            )

    @pytest.mark.asyncio
    async def test_claude_method(self):
        client = LLMClient()
        expected = LLMResponse(success=True, text="Hello")

        with patch.object(client, "send", return_value=expected) as mock_send:
            result = await client.claude("Test", model="claude-opus-4-20250514")

            mock_send.assert_called_once_with(
                "claude", "Test",
                model="claude-opus-4-20250514",
                timeout_seconds=300,
            )

    @pytest.mark.asyncio
    async def test_deepseek_method(self):
        client = LLMClient()
        expected = LLMResponse(success=True, text="Hello")

        with patch.object(client, "send", return_value=expected) as mock_send:
            result = await client.deepseek("Test")

            mock_send.assert_called_once_with(
                "openrouter", "Test",
                model="deepseek/deepseek-r1",
                timeout_seconds=300,
            )


class TestConstants:
    """Tests for module constants."""

    def test_browser_backends(self):
        assert "gemini" in BROWSER_BACKENDS
        assert "chatgpt" in BROWSER_BACKENDS
        assert "claude" not in BROWSER_BACKENDS

    def test_api_backends(self):
        assert "claude" in API_BACKENDS
        assert "openrouter" in API_BACKENDS
        assert "gemini" not in API_BACKENDS


# =============================================================================
# Recovery Tests
# =============================================================================

class TestWaitForPool:
    """Tests for _wait_for_pool method."""

    @pytest.mark.asyncio
    async def test_wait_for_pool_immediately_available(self):
        """Returns True immediately when pool is available."""
        client = LLMClient()

        with patch.object(client, "is_pool_available", return_value=True):
            result = await client._wait_for_pool(timeout_seconds=10)

        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_pool_becomes_available(self):
        """Returns True when pool becomes available within timeout."""
        client = LLMClient()

        call_count = 0

        async def mock_is_available():
            nonlocal call_count
            call_count += 1
            return call_count >= 3  # Available on 3rd call

        with patch.object(client, "is_pool_available", side_effect=mock_is_available):
            result = await client._wait_for_pool(timeout_seconds=30)

        assert result is True
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_wait_for_pool_timeout(self):
        """Returns False when pool doesn't become available within timeout."""
        client = LLMClient()

        with patch.object(client, "is_pool_available", return_value=False):
            # Use very short timeout for test speed
            result = await client._wait_for_pool(timeout_seconds=0.1)

        assert result is False


class TestPollForRecovered:
    """Tests for _poll_for_recovered method."""

    @pytest.mark.asyncio
    async def test_finds_matching_response(self):
        """Returns response when matching thread_id is found."""
        client = LLMClient()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "responses": [
                {
                    "request_id": "gemini-123",
                    "thread_id": "target-thread",
                    "backend": "gemini",
                    "response": "Recovered response text",
                    "options": {"deep_mode": True},
                },
                {
                    "request_id": "chatgpt-456",
                    "thread_id": "other-thread",
                    "backend": "chatgpt",
                    "response": "Other response",
                },
            ]
        })

        delete_response = MagicMock()
        delete_response.__aenter__ = AsyncMock(return_value=delete_response)
        delete_response.__aexit__ = AsyncMock(return_value=None)

        get_context = AsyncMock()
        get_context.__aenter__ = AsyncMock(return_value=mock_response)
        get_context.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=get_context)
        mock_session.delete = MagicMock(return_value=delete_response)

        with patch.object(client, "_get_http_session", return_value=mock_session):
            result = await client._poll_for_recovered("target-thread", timeout_seconds=10)

        assert result is not None
        assert result.success is True
        assert result.text == "Recovered response text"
        assert result.backend == "gemini"
        assert result.deep_mode_used is True

    @pytest.mark.asyncio
    async def test_no_matching_response(self):
        """Returns None when no matching thread_id is found."""
        client = LLMClient()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "responses": [
                {
                    "request_id": "chatgpt-456",
                    "thread_id": "other-thread",
                    "backend": "chatgpt",
                    "response": "Other response",
                },
            ]
        })

        get_context = AsyncMock()
        get_context.__aenter__ = AsyncMock(return_value=mock_response)
        get_context.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=get_context)

        with patch.object(client, "_get_http_session", return_value=mock_session):
            result = await client._poll_for_recovered(
                "nonexistent-thread",
                timeout_seconds=0.1,
                poll_interval=0.05,
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_empty_responses_list(self):
        """Returns None when responses list is empty."""
        client = LLMClient()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"responses": []})

        get_context = AsyncMock()
        get_context.__aenter__ = AsyncMock(return_value=mock_response)
        get_context.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=get_context)

        with patch.object(client, "_get_http_session", return_value=mock_session):
            result = await client._poll_for_recovered(
                "any-thread",
                timeout_seconds=0.1,
                poll_interval=0.05,
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_clears_recovered_after_pickup(self):
        """Attempts to DELETE recovered response after picking it up."""
        client = LLMClient()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "responses": [
                {
                    "request_id": "gemini-to-clear",
                    "thread_id": "clear-thread",
                    "backend": "gemini",
                    "response": "Response to clear",
                },
            ]
        })

        delete_response = MagicMock()
        delete_response.__aenter__ = AsyncMock(return_value=delete_response)
        delete_response.__aexit__ = AsyncMock(return_value=None)

        get_context = AsyncMock()
        get_context.__aenter__ = AsyncMock(return_value=mock_response)
        get_context.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=get_context)
        mock_session.delete = MagicMock(return_value=delete_response)

        with patch.object(client, "_get_http_session", return_value=mock_session):
            await client._poll_for_recovered("clear-thread", timeout_seconds=10)

        # Verify DELETE was called with correct URL
        mock_session.delete.assert_called()
        delete_url = mock_session.delete.call_args[0][0]
        assert "gemini-to-clear" in delete_url

    @pytest.mark.asyncio
    async def test_handles_poll_errors_gracefully(self):
        """Continues polling despite individual request errors."""
        client = LLMClient()

        call_count = 0

        async def mock_get_session():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Connection error")
            # Return success on 3rd call
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "responses": [
                    {
                        "request_id": "found",
                        "thread_id": "retry-thread",
                        "backend": "gemini",
                        "response": "Found after retries",
                    }
                ]
            })

            get_context = AsyncMock()
            get_context.__aenter__ = AsyncMock(return_value=mock_response)
            get_context.__aexit__ = AsyncMock(return_value=None)

            mock_session = MagicMock()
            mock_session.get = MagicMock(return_value=get_context)
            mock_session.delete = MagicMock(return_value=AsyncMock(
                __aenter__=AsyncMock(),
                __aexit__=AsyncMock()
            ))
            return mock_session

        with patch.object(client, "_get_http_session", side_effect=mock_get_session):
            result = await client._poll_for_recovered(
                "retry-thread",
                timeout_seconds=10,
                poll_interval=0.01,
            )

        assert result is not None
        assert result.text == "Found after retries"


class TestSendViaPoolRecovery:
    """Tests for recovery behavior in _send_via_pool."""

    @pytest.mark.asyncio
    async def test_recovery_on_connection_error_with_thread_id(self):
        """Attempts recovery when connection fails and thread_id is provided."""
        client = LLMClient()

        recovered_response = LLMResponse(
            success=True,
            text="Recovered from pool restart",
            backend="gemini",
        )

        # Create a context manager that raises on entry
        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(side_effect=aiohttp.ClientError("Connection refused"))
        mock_context.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_context)

        with patch.object(client, "_get_http_session", return_value=mock_session):
            with patch.object(client, "_wait_for_pool", return_value=True):
                with patch.object(client, "_poll_for_recovered", return_value=recovered_response):
                    result = await client._send_via_pool(
                        backend="gemini",
                        prompt="Test prompt",
                        deep_mode=False,
                        new_chat=True,
                        timeout_seconds=60,
                        priority="normal",
                        thread_id="recovery-thread",
                    )

        assert result.success is True
        assert result.text == "Recovered from pool restart"

    @pytest.mark.asyncio
    async def test_no_recovery_without_thread_id(self):
        """Does not attempt recovery when thread_id not provided."""
        client = LLMClient()

        # Create a context manager that raises on entry
        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(side_effect=aiohttp.ClientError("Connection refused"))
        mock_context.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_context)

        with patch.object(client, "_get_http_session", return_value=mock_session):
            with patch.object(client, "_wait_for_pool") as mock_wait:
                with patch.object(client, "_poll_for_recovered") as mock_poll:
                    with pytest.raises(PoolUnavailableError):
                        await client._send_via_pool(
                            backend="gemini",
                            prompt="Test prompt",
                            deep_mode=False,
                            new_chat=True,
                            timeout_seconds=60,
                            priority="normal",
                            thread_id=None,  # No thread_id
                        )

        # Should not have called recovery methods
        mock_wait.assert_not_called()
        mock_poll.assert_not_called()

    @pytest.mark.asyncio
    async def test_retry_when_pool_down_no_recovery(self):
        """Retries request when pool unavailable and no recovery found."""
        client = LLMClient()

        call_count = 0

        def create_context(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                # Return context that fails on entry
                mock_context = MagicMock()
                mock_context.__aenter__ = AsyncMock(side_effect=aiohttp.ClientError("Connection refused"))
                mock_context.__aexit__ = AsyncMock(return_value=None)
                return mock_context
            # Success on 3rd attempt
            mock_response = MagicMock()
            mock_response.json = AsyncMock(return_value={
                "success": True,
                "response": "Success after retry",
                "metadata": {"backend": "gemini"},
            })
            context = MagicMock()
            context.__aenter__ = AsyncMock(return_value=mock_response)
            context.__aexit__ = AsyncMock(return_value=None)
            return context

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=create_context)

        with patch.object(client, "_get_http_session", return_value=mock_session):
            with patch.object(client, "_wait_for_pool", return_value=True):
                with patch.object(client, "_poll_for_recovered", return_value=None):
                    result = await client._send_via_pool(
                        backend="gemini",
                        prompt="Test prompt",
                        deep_mode=False,
                        new_chat=True,
                        timeout_seconds=60,
                        priority="normal",
                        thread_id="retry-thread",
                    )

        assert result.success is True
        assert result.text == "Success after retry"

    @pytest.mark.asyncio
    async def test_thread_id_passed_in_request(self):
        """Thread ID is included in the request data."""
        client = LLMClient()

        captured_request = None

        def capture_post(url, json=None, **kwargs):
            nonlocal captured_request
            captured_request = json
            mock_response = MagicMock()
            mock_response.json = AsyncMock(return_value={
                "success": True,
                "response": "Response",
                "metadata": {"backend": "gemini"},
            })
            context = MagicMock()
            context.__aenter__ = AsyncMock(return_value=mock_response)
            context.__aexit__ = AsyncMock(return_value=None)
            return context

        mock_session = MagicMock()
        mock_session.post = capture_post

        with patch.object(client, "_get_http_session", return_value=mock_session):
            await client._send_via_pool(
                backend="gemini",
                prompt="Test prompt",
                deep_mode=False,
                new_chat=True,
                timeout_seconds=60,
                priority="normal",
                thread_id="my-thread-id",
            )

        assert captured_request is not None
        assert captured_request["thread_id"] == "my-thread-id"


class TestSendWithThreadId:
    """Tests for thread_id parameter in send method."""

    @pytest.mark.asyncio
    async def test_send_passes_thread_id_to_pool(self):
        """send() passes thread_id through to _send_via_pool."""
        client = LLMClient()

        with patch.object(client, "_send_via_pool") as mock_pool:
            mock_pool.return_value = LLMResponse(success=True, text="Response")

            await client.send(
                "gemini",
                "Test prompt",
                thread_id="explicit-thread",
            )

            # Verify thread_id was passed
            call_kwargs = mock_pool.call_args
            assert call_kwargs[1].get("thread_id") == "explicit-thread" or \
                   (len(call_kwargs[0]) > 6 and call_kwargs[0][6] == "explicit-thread")
