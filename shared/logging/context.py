"""
Correlation context management for distributed tracing.

Uses ContextVar for async-safe context propagation.
"""

import uuid
from contextvars import ContextVar
from contextlib import contextmanager
from typing import Optional, Generator

# Context variables (async-safe)
_correlation_id: ContextVar[str] = ContextVar('correlation_id', default='')
_session_id: ContextVar[str] = ContextVar('session_id', default='')
_request_id: ContextVar[str] = ContextVar('request_id', default='')


def get_correlation_id() -> str:
    """Get current correlation ID, generating one if none exists."""
    cid = _correlation_id.get()
    if not cid:
        cid = str(uuid.uuid4())
        _correlation_id.set(cid)
    return cid


def set_correlation_id(cid: str) -> None:
    """Set correlation ID for current context."""
    _correlation_id.set(cid)


def get_session_id() -> Optional[str]:
    """Get current session ID."""
    return _session_id.get() or None


def set_session_id(sid: str) -> None:
    """Set session ID for current context."""
    _session_id.set(sid)


def get_request_id() -> Optional[str]:
    """Get current request ID."""
    return _request_id.get() or None


def set_request_id(rid: str) -> None:
    """Set request ID for current context."""
    _request_id.set(rid)


@contextmanager
def correlation_context(
    correlation_id: Optional[str] = None,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> Generator[str, None, None]:
    """
    Context manager for setting correlation context.

    Args:
        correlation_id: Optional correlation ID (generated if not provided)
        session_id: Optional session ID
        request_id: Optional request ID

    Yields:
        The correlation ID being used

    Example:
        with correlation_context() as cid:
            log.event("request.started", correlation_id=cid)
    """
    old_cid = _correlation_id.get()
    old_sid = _session_id.get()
    old_rid = _request_id.get()

    try:
        if correlation_id:
            _correlation_id.set(correlation_id)
        elif not old_cid:
            _correlation_id.set(str(uuid.uuid4()))

        if session_id:
            _session_id.set(session_id)

        if request_id:
            _request_id.set(request_id)

        yield get_correlation_id()
    finally:
        _correlation_id.set(old_cid)
        _session_id.set(old_sid)
        _request_id.set(old_rid)
