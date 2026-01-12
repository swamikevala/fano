"""Tests for LLM adapters."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapters import (
    BrowserAdapter,
    GeminiAdapter,
    ChatGPTAdapter,
    ClaudeAdapter,
    create_adapters,
)
from src.client import LLMClient
from src.models import LLMResponse


@pytest.fixture
def mock_client():
    """Create a mock LLMClient."""
    client = MagicMock(spec=LLMClient)
    client.send = AsyncMock()
    return client


class TestBrowserAdapter:
    """Tests for BrowserAdapter base class."""

    def test_init(self, mock_client):
        adapter = BrowserAdapter(mock_client, "gemini")

        assert adapter.client == mock_client
        assert adapter.backend == "gemini"
        assert adapter.last_deep_mode_used is False
        assert adapter._connected is False

    @pytest.mark.asyncio
    async def test_connect(self, mock_client):
        adapter = BrowserAdapter(mock_client, "gemini")

        await adapter.connect()

        assert adapter._connected is True

    @pytest.mark.asyncio
    async def test_disconnect(self, mock_client):
        adapter = BrowserAdapter(mock_client, "gemini")
        adapter._connected = True

        await adapter.disconnect()

        assert adapter._connected is False

    @pytest.mark.asyncio
    async def test_start_new_chat(self, mock_client):
        """start_new_chat is a no-op (pool handles it)."""
        adapter = BrowserAdapter(mock_client, "gemini")

        # Should not raise
        await adapter.start_new_chat()

    @pytest.mark.asyncio
    async def test_send_message_success(self, mock_client):
        adapter = BrowserAdapter(mock_client, "gemini")

        mock_client.send.return_value = LLMResponse(
            success=True,
            text="Hello world",
            deep_mode_used=False,
        )

        result = await adapter.send_message("Test prompt")

        assert result == "Hello world"
        mock_client.send.assert_called_once_with(
            "gemini",
            "Test prompt",
            deep_mode=False,
            new_chat=True,
            thread_id=None,
        )

    @pytest.mark.asyncio
    async def test_send_message_with_deep_think(self, mock_client):
        adapter = BrowserAdapter(mock_client, "gemini")

        mock_client.send.return_value = LLMResponse(
            success=True,
            text="Deep response",
            deep_mode_used=True,
        )

        result = await adapter.send_message("Test", use_deep_think=True)

        assert result == "Deep response"
        assert adapter.last_deep_mode_used is True
        mock_client.send.assert_called_once_with(
            "gemini",
            "Test",
            deep_mode=True,
            new_chat=True,
            thread_id=None,
        )

    @pytest.mark.asyncio
    async def test_send_message_with_pro_mode(self, mock_client):
        adapter = BrowserAdapter(mock_client, "chatgpt")

        mock_client.send.return_value = LLMResponse(
            success=True,
            text="Pro response",
            deep_mode_used=True,
        )

        result = await adapter.send_message("Test", use_pro_mode=True)

        mock_client.send.assert_called_once_with(
            "chatgpt",
            "Test",
            deep_mode=True,
            new_chat=True,
            thread_id=None,
        )

    @pytest.mark.asyncio
    async def test_send_message_error_raises(self, mock_client):
        adapter = BrowserAdapter(mock_client, "gemini")

        mock_client.send.return_value = LLMResponse(
            success=False,
            error="rate_limited",
            message="Too many requests",
        )

        with pytest.raises(RuntimeError) as exc_info:
            await adapter.send_message("Test")

        assert "rate_limited" in str(exc_info.value)
        assert "Too many requests" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_send_message_empty_response(self, mock_client):
        adapter = BrowserAdapter(mock_client, "gemini")

        mock_client.send.return_value = LLMResponse(
            success=True,
            text=None,
        )

        result = await adapter.send_message("Test")

        assert result == ""

    def test_is_available_when_connected(self, mock_client):
        adapter = BrowserAdapter(mock_client, "gemini")
        adapter._connected = True

        assert adapter.is_available() is True

    def test_is_available_when_disconnected(self, mock_client):
        adapter = BrowserAdapter(mock_client, "gemini")
        adapter._connected = False

        assert adapter.is_available() is False


class TestGeminiAdapter:
    """Tests for GeminiAdapter."""

    def test_init(self, mock_client):
        adapter = GeminiAdapter(mock_client)

        assert adapter.backend == "gemini"
        assert adapter.model_name == "gemini"

    @pytest.mark.asyncio
    async def test_enable_deep_think(self, mock_client):
        """enable_deep_think is a no-op."""
        adapter = GeminiAdapter(mock_client)

        # Should not raise
        await adapter.enable_deep_think()

    def test_check_rate_limit_detects_quota(self, mock_client):
        adapter = GeminiAdapter(mock_client)

        assert adapter._check_rate_limit("Your quota exceeded for today") is True
        assert adapter._check_rate_limit("Please try again tomorrow") is True
        assert adapter._check_rate_limit("Rate limit reached") is True
        assert adapter._check_rate_limit("Too many requests") is True

    def test_check_rate_limit_normal_response(self, mock_client):
        adapter = GeminiAdapter(mock_client)

        assert adapter._check_rate_limit("Hello, how can I help?") is False
        assert adapter._check_rate_limit("Here is the answer...") is False

    def test_check_rate_limit_case_insensitive(self, mock_client):
        adapter = GeminiAdapter(mock_client)

        assert adapter._check_rate_limit("QUOTA EXCEEDED") is True
        assert adapter._check_rate_limit("Quota Exceeded") is True


class TestChatGPTAdapter:
    """Tests for ChatGPTAdapter."""

    def test_init(self, mock_client):
        adapter = ChatGPTAdapter(mock_client)

        assert adapter.backend == "chatgpt"
        assert adapter.model_name == "chatgpt"

    @pytest.mark.asyncio
    async def test_enable_pro_mode(self, mock_client):
        """enable_pro_mode is a no-op."""
        adapter = ChatGPTAdapter(mock_client)

        # Should not raise
        await adapter.enable_pro_mode()

    @pytest.mark.asyncio
    async def test_enable_thinking_mode(self, mock_client):
        """enable_thinking_mode is a no-op."""
        adapter = ChatGPTAdapter(mock_client)

        # Should not raise
        await adapter.enable_thinking_mode()

    def test_check_rate_limit_detects_cap(self, mock_client):
        adapter = ChatGPTAdapter(mock_client)

        assert adapter._check_rate_limit("You have reached your usage cap") is True
        assert adapter._check_rate_limit("Rate limit exceeded") is True
        assert adapter._check_rate_limit("Too many requests") is True
        assert adapter._check_rate_limit("You've reached the limit") is True

    def test_check_rate_limit_normal_response(self, mock_client):
        adapter = ChatGPTAdapter(mock_client)

        assert adapter._check_rate_limit("Hello, I can help with that!") is False


class TestClaudeAdapter:
    """Tests for ClaudeAdapter."""

    def test_init(self, mock_client):
        adapter = ClaudeAdapter(mock_client)

        assert adapter.backend == "claude"
        assert adapter.model_name == "claude"
        assert adapter.model == "claude-sonnet-4-20250514"

    def test_init_custom_model(self, mock_client):
        adapter = ClaudeAdapter(mock_client, model="claude-opus-4-20250514")

        assert adapter.model == "claude-opus-4-20250514"

    @pytest.mark.asyncio
    async def test_send_message_success(self, mock_client):
        adapter = ClaudeAdapter(mock_client)

        mock_client.send.return_value = LLMResponse(
            success=True,
            text="Claude response",
        )

        result = await adapter.send_message("Test prompt")

        assert result == "Claude response"
        assert adapter.last_deep_mode_used is False
        mock_client.send.assert_called_once_with(
            "claude",
            "Test prompt",
            model="claude-sonnet-4-20250514",
        )

    @pytest.mark.asyncio
    async def test_send_message_ignores_deep_mode_flags(self, mock_client):
        """Claude doesn't support deep mode, so flags are ignored."""
        adapter = ClaudeAdapter(mock_client)

        mock_client.send.return_value = LLMResponse(
            success=True,
            text="Response",
        )

        await adapter.send_message(
            "Test",
            use_deep_think=True,
            use_pro_mode=True,
            use_thinking_mode=True,
        )

        # deep_mode should not be passed to send
        mock_client.send.assert_called_once_with(
            "claude",
            "Test",
            model="claude-sonnet-4-20250514",
        )

    @pytest.mark.asyncio
    async def test_send_message_error_raises(self, mock_client):
        adapter = ClaudeAdapter(mock_client)

        mock_client.send.return_value = LLMResponse(
            success=False,
            error="api_error",
            message="API error occurred",
        )

        with pytest.raises(RuntimeError) as exc_info:
            await adapter.send_message("Test")

        assert "api_error" in str(exc_info.value)


class TestCreateAdapters:
    """Tests for create_adapters factory function."""

    def test_creates_all_adapters(self, mock_client):
        adapters = create_adapters(mock_client)

        assert "gemini" in adapters
        assert "chatgpt" in adapters
        assert "claude" in adapters

    def test_gemini_adapter_type(self, mock_client):
        adapters = create_adapters(mock_client)

        assert isinstance(adapters["gemini"], GeminiAdapter)

    def test_chatgpt_adapter_type(self, mock_client):
        adapters = create_adapters(mock_client)

        assert isinstance(adapters["chatgpt"], ChatGPTAdapter)

    def test_claude_adapter_type(self, mock_client):
        adapters = create_adapters(mock_client)

        assert isinstance(adapters["claude"], ClaudeAdapter)

    def test_adapters_share_client(self, mock_client):
        adapters = create_adapters(mock_client)

        assert adapters["gemini"].client is mock_client
        assert adapters["chatgpt"].client is mock_client
        assert adapters["claude"].client is mock_client


class TestAdapterCompatibility:
    """Tests for backward compatibility with browser interfaces."""

    @pytest.mark.asyncio
    async def test_gemini_interface_compatibility(self, mock_client):
        """GeminiAdapter has all methods expected by existing code."""
        adapter = GeminiAdapter(mock_client)

        # All these methods should exist
        assert hasattr(adapter, "connect")
        assert hasattr(adapter, "disconnect")
        assert hasattr(adapter, "start_new_chat")
        assert hasattr(adapter, "send_message")
        assert hasattr(adapter, "enable_deep_think")
        assert hasattr(adapter, "last_deep_mode_used")
        assert hasattr(adapter, "_check_rate_limit")

    @pytest.mark.asyncio
    async def test_chatgpt_interface_compatibility(self, mock_client):
        """ChatGPTAdapter has all methods expected by existing code."""
        adapter = ChatGPTAdapter(mock_client)

        assert hasattr(adapter, "connect")
        assert hasattr(adapter, "disconnect")
        assert hasattr(adapter, "start_new_chat")
        assert hasattr(adapter, "send_message")
        assert hasattr(adapter, "enable_pro_mode")
        assert hasattr(adapter, "enable_thinking_mode")
        assert hasattr(adapter, "last_deep_mode_used")
        assert hasattr(adapter, "_check_rate_limit")

    @pytest.mark.asyncio
    async def test_typical_usage_flow(self, mock_client):
        """Test typical usage flow as expected by orchestrator."""
        adapter = GeminiAdapter(mock_client)

        mock_client.send.return_value = LLMResponse(
            success=True,
            text="Response text",
            deep_mode_used=True,
        )

        # Typical flow
        await adapter.connect()
        await adapter.start_new_chat()
        result = await adapter.send_message("Test prompt", use_deep_think=True)
        deep_mode_used = adapter.last_deep_mode_used
        await adapter.disconnect()

        assert result == "Response text"
        assert deep_mode_used is True
