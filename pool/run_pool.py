#!/usr/bin/env python3
"""Runner script for the pool service."""

import asyncio
import logging
import signal
import sys
import warnings
from pathlib import Path

# Add parent directory to path so 'pool' package can be imported
_fano_root = Path(__file__).parent.parent
if str(_fano_root) not in sys.path:
    sys.path.insert(0, str(_fano_root))

# Suppress ResourceWarning about unclosed transports during shutdown
# These are harmless on Windows when the process is being terminated
warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed transport")

# Suppress "Exception ignored in __del__" errors during interpreter shutdown on Windows
# These occur when asyncio transports are garbage collected after pipes are closed
if sys.platform == "win32":
    def _quiet_unraisablehook(unraisable):
        # Suppress ValueError from closed pipes during shutdown
        if unraisable.exc_type is ValueError and "closed pipe" in str(unraisable.exc_value):
            return
        # For other unraisable exceptions, print them normally
        sys.__unraisablehook__(unraisable)

    sys.unraisablehook = _quiet_unraisablehook

# Now import and run
from pool.src.api import create_app, load_config
import uvicorn

from shared.logging import get_logger

log = get_logger("pool", "run_pool")


class HealthCheckFilter(logging.Filter):
    """Filter out health check and status endpoint logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        # Filter out health and status endpoint logs
        if "/health" in message or "/status" in message:
            return False
        return True


def handle_exception(loop, context):
    """Handle asyncio exceptions, suppressing Windows connection reset noise."""
    exception = context.get("exception")
    if exception:
        # Suppress Windows connection reset errors (common with proactor event loop)
        if isinstance(exception, ConnectionResetError):
            return
        if "ConnectionResetError" in str(exception):
            return
    # Log other exceptions normally
    msg = context.get("message", "Unhandled exception in event loop")
    log.error("asyncio.error", message=msg)


if __name__ == "__main__":
    config = load_config()
    server_config = config.get("server", {})
    host = server_config.get("host", "127.0.0.1")
    port = server_config.get("port", 9000)

    print(f"\n  Browser Pool Service")
    print(f"  =====================")
    print(f"  Running on http://{host}:{port}")
    print(f"  Press Ctrl+C to stop\n")

    # Add filter to suppress health check logs
    logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())

    # Suppress asyncio connection reset noise on Windows
    if sys.platform == "win32":
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.set_exception_handler(handle_exception)

    app = create_app(config)
    uvicorn.run(app, host=host, port=port, log_level="info")
