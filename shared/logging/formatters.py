"""
JSON Lines formatter for structured logging.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any

from .context import get_correlation_id, get_session_id, get_request_id


class JsonLinesFormatter(logging.Formatter):
    """
    Formats log records as JSON Lines (one JSON object per line).

    Each log entry includes:
    - timestamp: ISO 8601 with microseconds in UTC
    - level: Log level (DEBUG, INFO, WARNING, ERROR)
    - event_type: Structured event type identifier
    - module: Fano module (pool, llm, explorer)
    - component: Component within module
    - correlation_id: Request tracing ID
    - session_id: Browser/conversation session (if set)
    - request_id: Specific request ID (if set)
    - Additional event-specific fields
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Build base log entry
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event_type": getattr(record, 'event_type', 'log'),
            "module": getattr(record, 'fano_module', record.module),
            "component": getattr(record, 'component', record.funcName),
            "correlation_id": get_correlation_id(),
        }

        # Add optional context
        session_id = get_session_id()
        if session_id:
            log_entry["session_id"] = session_id

        request_id = get_request_id()
        if request_id:
            log_entry["request_id"] = request_id

        # Add event-specific data
        if hasattr(record, 'event_data'):
            log_entry.update(record.event_data)

        # Add message if it's not just the event type
        if record.getMessage() and record.getMessage() != log_entry.get("event_type"):
            log_entry["message"] = record.getMessage()

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=self._json_serializer)

    def _json_serializer(self, obj: Any) -> Any:
        """Handle non-serializable objects."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, '__dict__'):
            return str(obj)
        return repr(obj)


class ConsoleFormatter(logging.Formatter):
    """
    Human-readable console formatter.

    Format: timestamp [LEVEL] [module.component] message
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record for console output."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        module = getattr(record, 'fano_module', record.module)
        component = getattr(record, 'component', '')
        event_type = getattr(record, 'event_type', '')

        prefix = f"{timestamp} [{record.levelname}]"

        if module and component:
            prefix += f" [{module}.{component}]"

        message = record.getMessage()
        if event_type and event_type != message:
            message = f"{event_type}: {message}" if message else event_type

        return f"{prefix} {message}"
