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

from shared.logging import get_logger
from .services import FANO_ROOT, load_config
from .services.process_manager import ProcessManager

log = get_logger("control", "server")
from .blueprints import (
    ui_bp,
    status_bp,
    components_bp,
    documenter_bp,
    annotations_bp,
    explorer_bp,
    researcher_bp,
    pool_bp,
    orchestrator_bp,
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


def _check_pool_health(host: str, port: int) -> bool:
    """Quick check if pool is responding."""
    import urllib.request
    import urllib.error

    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=2) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _ensure_pool_running(pm: ProcessManager, config: dict) -> None:
    """
    Ensure pool is running and healthy.

    If pool is not running, starts it and waits for it to become healthy.
    Logs warnings but does not raise - server can function without pool.
    """
    pool_cfg = config.get("llm", {}).get("pool", {})
    host = pool_cfg.get("host", "127.0.0.1")
    port = pool_cfg.get("port", 9000)

    # Check if already running (external or managed)
    if _check_pool_health(host, port):
        log.info("control.pool.already_running", host=host, port=port)
        return

    # Check if pool process exists but not healthy yet
    if pm.is_running("pool"):
        log.info("control.pool.waiting_for_healthy", host=host, port=port)
    else:
        # Start pool
        log.info("control.pool.auto_starting")
        pm.start_pool()

    # Wait for healthy
    timeout = config.get("services", {}).get("pool", {}).get("health_timeout_seconds", 30)
    if pm.wait_for_pool_health(host, port, timeout):
        log.info("control.pool.started", host=host, port=port)
    else:
        log.warning("control.pool.startup_timeout", timeout=timeout)


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

    # Auto-start pool if configured
    config = load_config()
    services_config = config.get("services", {})
    pool_service_config = services_config.get("pool", {})

    if pool_service_config.get("auto_start", True):
        _ensure_pool_running(process_manager, config)

    # Register blueprints
    app.register_blueprint(ui_bp)
    app.register_blueprint(status_bp)
    app.register_blueprint(components_bp)
    app.register_blueprint(documenter_bp)
    app.register_blueprint(annotations_bp)
    app.register_blueprint(explorer_bp)
    app.register_blueprint(researcher_bp)
    app.register_blueprint(pool_bp)
    app.register_blueprint(orchestrator_bp)

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
