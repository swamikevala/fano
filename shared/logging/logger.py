"""
FanoLogger - Structured logging for Fano platform components.
"""

import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any, Optional
from logging.handlers import RotatingFileHandler

from .formatters import JsonLinesFormatter, ConsoleFormatter
from .context import get_correlation_id, set_request_id

# Cache of loggers by module.component
_loggers: dict[str, "FanoLogger"] = {}

# Default log directory (relative to project root)
_log_dir: Optional[Path] = None


def _get_log_dir() -> Path:
    """Get or create log directory."""
    global _log_dir
    if _log_dir is None:
        # Try to find project root (look for shared/ directory)
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "shared").exists():
                _log_dir = parent / "logs"
                break
        else:
            _log_dir = Path("logs")

        _log_dir.mkdir(parents=True, exist_ok=True)
    return _log_dir


def get_logger(module: str, component: str, console: bool = True) -> "FanoLogger":
    """
    Get or create a FanoLogger for a module/component.

    Args:
        module: Module name (pool, llm, explorer)
        component: Component within module (workers, consensus, etc.)
        console: Whether to also output to console

    Returns:
        FanoLogger instance
    """
    key = f"{module}.{component}"
    if key not in _loggers:
        _loggers[key] = FanoLogger(module, component, console)
    return _loggers[key]


class FanoLogger:
    """
    Structured logger for Fano components.

    Outputs JSON Lines to file and optionally human-readable to console.
    All events include correlation ID for request tracing.
    """

    def __init__(self, module: str, component: str, console: bool = True):
        """
        Initialize logger.

        Args:
            module: Module name (pool, llm, explorer)
            component: Component within module
            console: Whether to output to console
        """
        self.module = module
        self.component = component
        self._logger = logging.getLogger(f"fano.{module}.{component}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False  # Don't propagate to root logger

        # Clear existing handlers
        self._logger.handlers.clear()

        # Add JSON Lines file handler
        log_dir = _get_log_dir()
        log_file = log_dir / f"{module}.jsonl"

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10,
            encoding='utf-8',
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JsonLinesFormatter())
        self._logger.addHandler(file_handler)

        # Add console handler if requested
        if console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(ConsoleFormatter())
            self._logger.addHandler(console_handler)

    def event(
        self,
        event_type: str,
        level: str = "INFO",
        **data: Any,
    ) -> None:
        """
        Log a structured event.

        Args:
            event_type: Event type identifier (e.g., "pool.request.process")
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            **data: Event-specific data fields
        """
        log_level = getattr(logging, level.upper(), logging.INFO)

        # Prepare event data
        event_data = {
            "event_type": event_type,
            "fano_module": self.module,
            "component": self.component,
            **data,
        }

        # Create log record with extra data
        self._logger.log(
            log_level,
            event_type,
            extra={
                "event_type": event_type,
                "fano_module": self.module,
                "component": self.component,
                "event_data": event_data,
            },
        )

    def debug(self, event_type: str, **data: Any) -> None:
        """Log a DEBUG level event."""
        self.event(event_type, level="DEBUG", **data)

    def info(self, event_type: str, **data: Any) -> None:
        """Log an INFO level event."""
        self.event(event_type, level="INFO", **data)

    def warning(self, event_type: str, **data: Any) -> None:
        """Log a WARNING level event."""
        self.event(event_type, level="WARNING", **data)

    def error(self, event_type: str, **data: Any) -> None:
        """Log an ERROR level event."""
        self.event(event_type, level="ERROR", **data)

    def exception(
        self,
        error: Exception,
        event_type: str = "error",
        context: Optional[dict] = None,
    ) -> None:
        """
        Log an exception with full stack trace.

        Args:
            error: The exception to log
            event_type: Event type (default: "error")
            context: Additional context about what was happening
        """
        self.event(
            event_type,
            level="ERROR",
            error_class=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            context=context or {},
        )

    # Convenience methods for common patterns

    def request_start(
        self,
        request_id: str,
        backend: str,
        prompt: str,
        **kwargs: Any,
    ) -> float:
        """
        Log request start and return start time for duration calculation.

        Args:
            request_id: Unique request identifier
            backend: Backend name
            prompt: Full prompt text
            **kwargs: Additional fields

        Returns:
            Start time (for duration calculation)
        """
        set_request_id(request_id)
        self.event(
            f"{self.module}.request.start",
            action="started",
            request_id=request_id,
            backend=backend,
            prompt=prompt,
            prompt_length=len(prompt),
            **kwargs,
        )
        return time.time()

    def request_complete(
        self,
        request_id: str,
        backend: str,
        response: str,
        start_time: float,
        success: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Log request completion with duration.

        Args:
            request_id: Unique request identifier
            backend: Backend name
            response: Full response text
            start_time: Time from request_start()
            success: Whether request succeeded
            **kwargs: Additional fields
        """
        duration_ms = (time.time() - start_time) * 1000
        self.event(
            f"{self.module}.request.complete",
            action="completed",
            request_id=request_id,
            backend=backend,
            response=response,
            response_length=len(response) if response else 0,
            duration_ms=round(duration_ms, 2),
            success=success,
            **kwargs,
        )

    def request_error(
        self,
        request_id: str,
        backend: str,
        error: str,
        error_type: str,
        start_time: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Log request error.

        Args:
            request_id: Unique request identifier
            backend: Backend name
            error: Error message
            error_type: Error type/category
            start_time: Optional start time for duration
            **kwargs: Additional fields
        """
        event_data = {
            "action": "failed",
            "request_id": request_id,
            "backend": backend,
            "error": error,
            "error_type": error_type,
            **kwargs,
        }
        if start_time:
            event_data["duration_ms"] = round((time.time() - start_time) * 1000, 2)

        self.event(f"{self.module}.request.error", level="ERROR", **event_data)
