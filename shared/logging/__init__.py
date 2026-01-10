"""
Structured logging for the Fano platform.

Provides JSON Lines logging with correlation IDs for tracing
requests across components (pool, llm, explorer).

Usage:
    from shared.logging import get_logger, correlation_context

    log = get_logger("pool", "workers")

    with correlation_context() as cid:
        log.event("pool.request.process",
                  action="started",
                  backend="gemini",
                  prompt=request.prompt)
"""

from .logger import get_logger, FanoLogger
from .context import (
    correlation_context,
    get_correlation_id,
    set_correlation_id,
    get_session_id,
    set_session_id,
)

__all__ = [
    "get_logger",
    "FanoLogger",
    "correlation_context",
    "get_correlation_id",
    "set_correlation_id",
    "get_session_id",
    "set_session_id",
]
