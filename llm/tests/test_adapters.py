"""Tests for LLM adapters (API-based)."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from llm.src.adapters import (
    APIAdapter,
    GeminiAdapter,
    ChatGPTAdapter,
    ClaudeAdapter,
    DeepSeekAdapter,
    create_adapters,
    # Legacy alias
    BrowserAdapter,
)
from llm.src.client import LLMClient
from llm.src.models import LLMResponse


@pytest.fixture
def mock_client():
    """Create a mock LLMClient."""
    client = MagicMock(spec=LLMClient)
    client.send = AsyncMock()
    client._openrouter_key = "test-key"
    return client


class TestAPIAdapter:
    """Tests for APIAdapter base class."""

    def test_init(self, mock_client):
        adapter = APIAdapter(mock_client)
        adapter.backend = "test"

        assert adapter.client == mock_client
        assert adapter.last_deep_mode_used is False
        assert adapter._connected is False

    @pytest.mark.asyncio
    async def test_connect(self, mock_client):
        adapter = APIAdapter(mock_client)

        await adapter.connect()

        assert adapter._connected is True

    @pytest.mark.asyncio
    async def test_disconnect(self, mock_client):
        adapter = APIAdapter(mock_client)
        adapter._connected = True

        await adapter.disconnect()

        assert adapter._connected is False

    @pytest.mark.asyncio
    async def test_start_new_chat(self, mock_client):
        """start_new_chat is a no-op (API calls are stateless)."""
        adapter = APIAdapter(mock_client)

        # Should not raise
        await adapter.start_new_chat()

    @pytest.mark.asyncio
    async def test_send_message_success(self, mock_client):
        adapter = APIAdapter(mock_client)
        adapter.backend = "test"

        mock_client.send.return_value = LLMResponse(
            success=True,
            text="Hello world",
        )

        result = await adapter.send_message("Test prompt")

        assert result == "Hello world"
        mock_client.send.assert_called_once()
        call_args = mock_client.send.call_args
        assert call_args[0][0] == "test"  # backend
        assert call_args[0][1] == "Test prompt"  # prompt

    @pytest.mark.asyncio
    async def test_send_message_error_raises(self, mock_client):
        adapter = APIAdapter(mock_client)
        adapter.backend = "test"

        mock_client.send.return_value = LLMResponse(
            success=False,
            error="api_error",
            message="Something went wrong",
        )

        with pytest.raises(RuntimeError) as exc_info:
            await adapter.send_message("Test")

        assert "api_error" in str(exc_info.value)

    def test_is_available_with_key(self, mock_client):
        adapter = APIAdapter(mock_client)

        assert adapter.is_available() is True

    def test_is_available_without_key(self, mock_client):
        mock_client._openrouter_key = None
        adapter = APIAdapter(mock_client)

        assert adapter.is_available() is False


class TestGeminiAdapter:
    """Tests for GeminiAdapter."""

    def test_init(self, mock_client):
        adapter = GeminiAdapter(mock_client)

        assert adapter.backend == "gemini"
        assert adapter.model_name == "gemini"

    def test_init_with_custom_model(self, mock_client):
        adapter = GeminiAdapter(mock_client, model="google/custom-model")

        assert adapter._model == "google/custom-model"

    @pytest.mark.asyncio
    async def test_enable_deep_think(self, mock_client):
        """enable_deep_think is a no-op for API access."""
        adapter = GeminiAdapter(mock_client)

        # Should not raise
        await adapter.enable_deep_think()


class TestChatGPTAdapter:
    """Tests for ChatGPTAdapter."""

    def test_init(self, mock_client):
        adapter = ChatGPTAdapter(mock_client)

        assert adapter.backend == "chatgpt"
        assert adapter.model_name == "chatgpt"

    @pytest.mark.asyncio
    async def test_enable_pro_mode(self, mock_client):
        """enable_pro_mode is a no-op for API access."""
        adapter = ChatGPTAdapter(mock_client)

        # Should not raise
        await adapter.enable_pro_mode()

    @pytest.mark.asyncio
    async def test_enable_thinking_mode(self, mock_client):
        """enable_thinking_mode is a no-op for API access."""
        adapter = ChatGPTAdapter(mock_client)

        # Should not raise
        await adapter.enable_thinking_mode()


class TestClaudeAdapter:
    """Tests for ClaudeAdapter."""

    def test_init(self, mock_client):
        adapter = ClaudeAdapter(mock_client)

        assert adapter.backend == "claude"
        assert adapter.model_name == "claude"

    @pytest.mark.asyncio
    async def test_enable_extended_thinking(self, mock_client):
        """enable_extended_thinking is a no-op for API access."""
        adapter = ClaudeAdapter(mock_client)

        # Should not raise
        await adapter.enable_extended_thinking()

    @pytest.mark.asyncio
    async def test_send_message_success(self, mock_client):
        adapter = ClaudeAdapter(mock_client)

        mock_client.send.return_value = LLMResponse(
            success=True,
            text="Claude response",
        )

        result = await adapter.send_message("Test prompt")

        assert result == "Claude response"
        mock_client.send.assert_called_once()
        call_args = mock_client.send.call_args
        assert call_args[0][0] == "claude"


class TestDeepSeekAdapter:
    """Tests for DeepSeekAdapter."""

    def test_init(self, mock_client):
        adapter = DeepSeekAdapter(mock_client)

        assert adapter.backend == "deepseek"
        assert adapter.model_name == "deepseek"


class TestCreateAdapters:
    """Tests for create_adapters function."""

    def test_creates_all_adapters(self, mock_client):
        adapters = create_adapters(mock_client)

        assert "gemini" in adapters
        assert "chatgpt" in adapters
        assert "claude" in adapters
        assert "deepseek" in adapters

    def test_gemini_adapter_type(self, mock_client):
        adapters = create_adapters(mock_client)

        assert isinstance(adapters["gemini"], GeminiAdapter)

    def test_chatgpt_adapter_type(self, mock_client):
        adapters = create_adapters(mock_client)

        assert isinstance(adapters["chatgpt"], ChatGPTAdapter)

    def test_claude_adapter_type(self, mock_client):
        adapters = create_adapters(mock_client)

        assert isinstance(adapters["claude"], ClaudeAdapter)

    def test_deepseek_adapter_type(self, mock_client):
        adapters = create_adapters(mock_client)

        assert isinstance(adapters["deepseek"], DeepSeekAdapter)

    def test_adapters_share_client(self, mock_client):
        adapters = create_adapters(mock_client)

        assert adapters["gemini"].client is mock_client
        assert adapters["chatgpt"].client is mock_client
        assert adapters["claude"].client is mock_client


class TestLegacyAlias:
    """Tests for legacy BrowserAdapter alias."""

    def test_browser_adapter_is_api_adapter(self):
        assert BrowserAdapter is APIAdapter


class TestAdapterUsageFlow:
    """Tests for typical adapter usage patterns."""

    @pytest.mark.asyncio
    async def test_typical_usage_flow(self, mock_client):
        """Test the typical connect -> send -> disconnect flow."""
        adapter = GeminiAdapter(mock_client)

        mock_client.send.return_value = LLMResponse(
            success=True,
            text="Response text",
        )

        # Connect
        await adapter.connect()
        assert adapter._connected is True

        # Send message
        result = await adapter.send_message("Hello")
        assert result == "Response text"

        # Disconnect
        await adapter.disconnect()
        assert adapter._connected is False

    @pytest.mark.asyncio
    async def test_send_with_images(self, mock_client):
        """Test sending with image attachments."""
        from llm.src.models import ImageAttachment

        adapter = ClaudeAdapter(mock_client)

        mock_client.send.return_value = LLMResponse(
            success=True,
            text="I see the image",
        )

        images = [ImageAttachment(filename="test.png", data="base64data", media_type="image/png")]
        result = await adapter.send_message("Describe this", images=images)

        assert result == "I see the image"
        call_args = mock_client.send.call_args
        assert call_args.kwargs["images"] == images
