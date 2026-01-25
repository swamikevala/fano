"""Tests for LLM client (OpenRouter-based)."""

import os
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from llm.src.client import LLMClient, RateLimiter, DEFAULT_MODELS
from llm.src.models import LLMResponse


class TestLLMClientInit:
    """Tests for LLMClient initialization."""

    def test_default_base_url(self):
        client = LLMClient()
        assert client._base_url == "https://openrouter.ai/api/v1"

    def test_custom_base_url(self):
        client = LLMClient(base_url="https://custom.api/v1/")
        assert client._base_url == "https://custom.api/v1"  # Trailing slash stripped

    def test_openrouter_key_from_arg(self):
        client = LLMClient(openrouter_api_key="or-key")
        assert client._openrouter_key == "or-key"

    def test_openrouter_key_from_env(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-env-key"}):
            client = LLMClient()
            assert client._openrouter_key == "or-env-key"

    def test_default_models(self):
        client = LLMClient()
        assert "gemini" in client._models
        assert "chatgpt" in client._models
        assert "claude" in client._models
        assert "deepseek" in client._models

    def test_custom_models(self):
        custom = {"gemini": "custom/gemini-model"}
        client = LLMClient(models=custom)
        assert client._models["gemini"] == "custom/gemini-model"
        # Others should still have defaults
        assert client._models["chatgpt"] == DEFAULT_MODELS["chatgpt"]

    def test_lazy_http_session(self):
        client = LLMClient()
        assert client._http_session is None


class TestLLMClientModelMapping:
    """Tests for model mapping functionality."""

    def test_get_model_with_alias(self):
        client = LLMClient()
        assert client.get_model("gemini") == DEFAULT_MODELS["gemini"]
        assert client.get_model("claude") == DEFAULT_MODELS["claude"]

    def test_get_model_passthrough(self):
        client = LLMClient()
        # Unknown alias returns the input as-is (assumed to be a full model ID)
        assert client.get_model("anthropic/claude-3-opus") == "anthropic/claude-3-opus"

    def test_set_model(self):
        client = LLMClient()
        client.set_model("gemini", "google/custom-model")
        assert client.get_model("gemini") == "google/custom-model"

    def test_list_models(self):
        client = LLMClient()
        models = client.list_models()
        assert "gemini" in models
        assert "chatgpt" in models
        assert "claude" in models


class TestRateLimiter:
    """Tests for the rate limiter."""

    @pytest.mark.asyncio
    async def test_acquire_first_request(self):
        limiter = RateLimiter({"test": 60})
        # First request should not wait
        await limiter.acquire("test")
        # Token should be consumed
        assert limiter._tokens["test"] < 60

    @pytest.mark.asyncio
    async def test_acquire_uses_default_limit(self):
        limiter = RateLimiter()
        await limiter.acquire("unknown_backend")
        # Should use default of 60 rpm
        assert "unknown_backend" in limiter._tokens


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
    async def test_close_closes_session(self):
        client = LLMClient()

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()
        client._http_session = mock_session

        await client.close()

        mock_session.close.assert_called_once()
        assert client._http_session is None


class TestLLMClientSend:
    """Tests for the unified send method."""

    @pytest.mark.asyncio
    async def test_send_uses_model_mapping(self):
        client = LLMClient(openrouter_api_key="test-key")

        expected_response = LLMResponse(success=True, text="Hello", backend="gemini")

        with patch.object(client, "_send_to_openrouter", return_value=expected_response) as mock_api:
            response = await client.send("gemini", "Test prompt")

            # Should use the mapped model
            call_args = mock_api.call_args
            assert call_args.kwargs["model"] == DEFAULT_MODELS["gemini"]
            assert call_args.kwargs["backend"] == "gemini"

    @pytest.mark.asyncio
    async def test_send_with_explicit_model(self):
        client = LLMClient(openrouter_api_key="test-key")

        expected_response = LLMResponse(success=True, text="Hello")

        with patch.object(client, "_send_to_openrouter", return_value=expected_response) as mock_api:
            await client.send("claude", "Test", model="anthropic/claude-3-opus")

            call_args = mock_api.call_args
            assert call_args.kwargs["model"] == "anthropic/claude-3-opus"

    @pytest.mark.asyncio
    async def test_send_passthrough_full_model_id(self):
        client = LLMClient(openrouter_api_key="test-key")

        expected_response = LLMResponse(success=True, text="Hello")

        with patch.object(client, "_send_to_openrouter", return_value=expected_response) as mock_api:
            # Using a full model ID as backend (not an alias)
            await client.send("meta-llama/llama-3-70b", "Test")

            call_args = mock_api.call_args
            assert call_args.kwargs["model"] == "meta-llama/llama-3-70b"


class TestLLMClientSendToOpenRouter:
    """Tests for _send_to_openrouter method."""

    @pytest.mark.asyncio
    async def test_send_to_openrouter_no_api_key(self):
        client = LLMClient()
        client._openrouter_key = None

        response = await client._send_to_openrouter(
            "Test prompt", model="test/model", backend="test"
        )

        assert response.success is False
        assert response.error == "auth_required"

    @pytest.mark.asyncio
    async def test_send_to_openrouter_success(self):
        client = LLMClient(openrouter_api_key="or-key")

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{"message": {"content": "API response"}}]
        })
        mock_response.headers = {}

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_context)

        with patch.object(client, "_get_http_session", return_value=mock_session):
            with patch.object(client._rate_limiter, "acquire", return_value=None):
                response = await client._send_to_openrouter(
                    "Test prompt", model="test/model", backend="test"
                )

        assert response.success is True
        assert response.text == "API response"
        assert response.backend == "test"


class TestLLMClientSendParallel:
    """Tests for send_parallel method."""

    @pytest.mark.asyncio
    async def test_send_parallel_all_success(self):
        client = LLMClient(openrouter_api_key="test-key")

        responses = {
            "gemini": LLMResponse(success=True, text="Gemini says hi", backend="gemini"),
            "claude": LLMResponse(success=True, text="Claude says hi", backend="claude"),
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
        client = LLMClient(openrouter_api_key="test-key")

        async def mock_send(backend, prompt, **kwargs):
            if backend == "gemini":
                raise Exception("Gemini failed")
            return LLMResponse(success=True, text="Claude says hi", backend="claude")

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
    async def test_returns_all_backends_with_api_key(self):
        client = LLMClient(openrouter_api_key="key")
        available = await client.get_available_backends()

        assert "gemini" in available
        assert "chatgpt" in available
        assert "claude" in available
        assert "deepseek" in available

    @pytest.mark.asyncio
    async def test_returns_empty_without_api_key(self):
        client = LLMClient()
        client._openrouter_key = None
        available = await client.get_available_backends()

        assert len(available) == 0


class TestLLMClientConvenienceMethods:
    """Tests for convenience methods."""

    @pytest.mark.asyncio
    async def test_gemini_method(self):
        client = LLMClient(openrouter_api_key="key")
        expected = LLMResponse(success=True, text="Hello")

        with patch.object(client, "send", return_value=expected) as mock_send:
            result = await client.gemini("Test")

            mock_send.assert_called_once()
            call_args = mock_send.call_args
            assert call_args[0][0] == "gemini"
            assert result == expected

    @pytest.mark.asyncio
    async def test_chatgpt_method(self):
        client = LLMClient(openrouter_api_key="key")
        expected = LLMResponse(success=True, text="Hello")

        with patch.object(client, "send", return_value=expected) as mock_send:
            result = await client.chatgpt("Test")

            mock_send.assert_called_once()
            call_args = mock_send.call_args
            assert call_args[0][0] == "chatgpt"

    @pytest.mark.asyncio
    async def test_claude_method(self):
        client = LLMClient(openrouter_api_key="key")
        expected = LLMResponse(success=True, text="Hello")

        with patch.object(client, "send", return_value=expected) as mock_send:
            result = await client.claude("Test")

            mock_send.assert_called_once()
            call_args = mock_send.call_args
            assert call_args[0][0] == "claude"

    @pytest.mark.asyncio
    async def test_deepseek_method(self):
        client = LLMClient(openrouter_api_key="key")
        expected = LLMResponse(success=True, text="Hello")

        with patch.object(client, "send", return_value=expected) as mock_send:
            result = await client.deepseek("Test")

            mock_send.assert_called_once()
            call_args = mock_send.call_args
            assert call_args[0][0] == "deepseek"


class TestLLMClientSendAsync:
    """Tests for send_async compatibility method."""

    @pytest.mark.asyncio
    async def test_send_async_calls_send(self):
        client = LLMClient(openrouter_api_key="key")
        expected = LLMResponse(success=True, text="Hello")

        with patch.object(client, "send", return_value=expected) as mock_send:
            result = await client.send_async("gemini", "Test", job_id="test-job")

            mock_send.assert_called_once()
            assert result == expected
