"""
Shared LLM connection management for Explorer commands.

Provides a unified interface for connecting to LLMs via OpenRouter API.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Callable

from shared.logging import get_logger

log = get_logger("explorer", "llm_connections")


@dataclass
class ConnectionStatus:
    """Status of an LLM connection attempt."""
    name: str
    connected: bool
    message: str


@dataclass
class LLMConnections:
    """
    Manages connections to LLM providers via API.

    Uses OpenRouter for unified access to Gemini, ChatGPT, Claude, and DeepSeek.
    """
    client: Optional[object] = None
    gemini: Optional[object] = None
    chatgpt: Optional[object] = None
    claude: Optional[object] = None
    deepseek: Optional[object] = None
    statuses: list[ConnectionStatus] = field(default_factory=list)

    def has_any(self) -> bool:
        """Check if any LLM is available."""
        return bool(self.gemini or self.chatgpt or self.claude or self.deepseek)

    def available_names(self) -> list[str]:
        """Get names of available LLMs."""
        names = []
        if self.gemini:
            names.append("Gemini")
        if self.chatgpt:
            names.append("ChatGPT")
        if self.claude:
            names.append("Claude")
        if self.deepseek:
            names.append("DeepSeek")
        return names


async def connect_llms(
    on_status: Optional[Callable[[ConnectionStatus], None]] = None
) -> LLMConnections:
    """
    Connect to all available LLM providers via OpenRouter API.

    Args:
        on_status: Optional callback for status updates during connection.
                   Called with ConnectionStatus for each provider.

    Returns:
        LLMConnections with connected providers (None for failed ones)
    """
    from llm import LLMClient, GeminiAdapter, ChatGPTAdapter, ClaudeAdapter, DeepSeekAdapter

    connections = LLMConnections()

    def report(status: ConnectionStatus):
        connections.statuses.append(status)
        if on_status:
            on_status(status)

    # Check for API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        report(ConnectionStatus(
            "OpenRouter", False,
            "OPENROUTER_API_KEY not set - run 'python fano_explorer.py auth' for info"
        ))
        log.error("llm.connection.no_api_key", provider="openrouter")
        return connections

    # Initialize client
    client = LLMClient(openrouter_api_key=api_key)
    connections.client = client

    # Create adapters (all available if API key is set)
    connections.gemini = GeminiAdapter(client)
    await connections.gemini.connect()
    report(ConnectionStatus("Gemini", True, "API ready"))
    log.info("llm.connection.success", provider="gemini")

    connections.chatgpt = ChatGPTAdapter(client)
    await connections.chatgpt.connect()
    report(ConnectionStatus("ChatGPT", True, "API ready"))
    log.info("llm.connection.success", provider="chatgpt")

    connections.claude = ClaudeAdapter(client)
    await connections.claude.connect()
    report(ConnectionStatus("Claude", True, "API ready"))
    log.info("llm.connection.success", provider="claude")

    connections.deepseek = DeepSeekAdapter(client)
    await connections.deepseek.connect()
    report(ConnectionStatus("DeepSeek", True, "API ready"))
    log.info("llm.connection.success", provider="deepseek")

    return connections


async def disconnect_llms(connections: LLMConnections):
    """
    Disconnect all LLM providers.

    Args:
        connections: The LLMConnections to disconnect
    """
    if connections.gemini:
        await connections.gemini.disconnect()
        log.info("llm.disconnected", provider="gemini")

    if connections.chatgpt:
        await connections.chatgpt.disconnect()
        log.info("llm.disconnected", provider="chatgpt")

    if connections.claude:
        await connections.claude.disconnect()
        log.info("llm.disconnected", provider="claude")

    if connections.deepseek:
        await connections.deepseek.disconnect()
        log.info("llm.disconnected", provider="deepseek")

    if connections.client:
        await connections.client.close()
        log.info("llm.client.closed")
