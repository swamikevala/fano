"""
Control Panel Server - Flask app for managing Fano components.

This module provides the main Flask application factory and server startup.
Route handlers are organized into blueprints in the blueprints/ directory.
"""

import logging
import subprocess
import sys
import warnings
from typing import Optional

from flask import Flask

from .services import FANO_ROOT
from .services.process_manager import ProcessManager
from .blueprints import (
    ui_bp,
    status_bp,
    components_bp,
    documenter_bp,
    annotations_bp,
    explorer_bp,
    researcher_bp,
    pool_bp,
)

# Suppress ResourceWarning about unclosed transports during shutdown
# These are harmless on Windows when child processes are being terminated
warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed transport")

# Suppress "Exception ignored in __del__" errors during interpreter shutdown on Windows
if sys.platform == "win32":
    def _quiet_unraisablehook(unraisable):
        # Suppress ValueError from closed pipes during shutdown
        if unraisable.exc_type is ValueError and "closed pipe" in str(unraisable.exc_value):
            return
        sys.__unraisablehook__(unraisable)

    sys.unraisablehook = _quiet_unraisablehook


class StatusLogFilter(logging.Filter):
    """Filter out frequent polling requests from Flask logs."""

    NOISY_ENDPOINTS = [
        "/api/status",
        "/health",
        "/api/documenter/activity",
        "/api/explorer/stats",
        "/api/pool/activity",
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for endpoint in self.NOISY_ENDPOINTS:
            if endpoint in message:
                return False
        return True


class AsyncioNoiseFilter(logging.Filter):
    """Filter out noisy asyncio connection reset errors on Windows."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        # Filter out Windows connection reset noise
        if "ConnectionResetError" in message or "_call_connection_lost" in message:
            return False
        return True


# Apply filters
logging.getLogger("werkzeug").addFilter(StatusLogFilter())
logging.getLogger("asyncio").addFilter(AsyncioNoiseFilter())


def create_app(process_manager: Optional[ProcessManager] = None) -> Flask:
    """
    Create the Flask application.

    Args:
        process_manager: Optional ProcessManager instance. If not provided,
                        a new one will be created.

    Returns:
        Configured Flask application.
    """
    app = Flask(
        __name__,
        template_folder=str(FANO_ROOT / "control" / "templates"),
        static_folder=str(FANO_ROOT / "control" / "static"),
    )

    # Store process manager in app config for blueprint access
    if process_manager is None:
        process_manager = ProcessManager()
    app.config["process_manager"] = process_manager

    # Register blueprints
    app.register_blueprint(ui_bp)
    app.register_blueprint(status_bp)
    app.register_blueprint(components_bp)
    app.register_blueprint(documenter_bp)
    app.register_blueprint(annotations_bp)
    app.register_blueprint(explorer_bp)
    app.register_blueprint(researcher_bp)
    app.register_blueprint(pool_bp)

    return app


def start_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    debug: bool = True,
    pool_process: Optional[subprocess.Popen] = None
):
    """
    Start the control panel server.

    Args:
        host: Host to bind to
        port: Port to bind to
        debug: Enable debug mode
        pool_process: Optional externally-started pool process to register
    """
    # Create process manager and register any external processes
    pm = ProcessManager()
    if pool_process is not None:
        pm.register_external("pool", pool_process)

    app = create_app(process_manager=pm)

    print(f"\n  Fano Control Panel")
    print(f"  ==================")
    print(f"  Open http://{host}:{port} in your browser\n")

    app.run(host=host, port=port, debug=debug, use_reloader=False)


if __name__ == "__main__":
    start_server()
