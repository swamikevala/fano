"""Minimal structured logging helper for v2 modules.

Provides a logger that accepts arbitrary keyword arguments as
structured context (e.g. ``log.info("event", key=value)``).
Extra kwargs are stored in the LogRecord's ``extra`` dict.
"""

from __future__ import annotations

import logging as _logging


class _StructuredLogger(_logging.Logger):
    """Logger subclass that accepts structured keyword arguments."""

    def _log(
        self,
        level: int,
        msg: object,
        args: tuple,  # type: ignore[override]
        exc_info: object = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: dict | None = None,
        **kwargs: object,
    ) -> None:
        if kwargs:
            extra = extra or {}
            extra.update(kwargs)
        super()._log(
            level, msg, args,
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel,
            extra=extra,
        )


_logging.setLoggerClass(_StructuredLogger)


def get_logger(component: str, module: str) -> _StructuredLogger:
    """Return a structured logger namespaced as ``component.module``."""
    return _logging.getLogger(f"{component}.{module}")  # type: ignore[return-value]
