"""
Components Blueprint - Process start/stop/restart routes.
"""

import os
import subprocess
import sys
import threading
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request

from ..services import FANO_ROOT, load_config

bp = Blueprint("components", __name__, url_prefix="/api")


def get_process_manager():
    """Get the ProcessManager from the app context."""
    return current_app.config.get("process_manager")


def check_pool_health(host: str = "127.0.0.1", port: int = 9000) -> bool:
    """Check if pool is responding to health checks."""
    try:
        import urllib.request
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


@bp.route("/start/<component>", methods=["POST"])
def api_start(component: str):
    """Start a component."""
    if component not in ["pool", "explorer", "documenter", "researcher"]:
        return jsonify({"error": f"Unknown component: {component}"}), 400

    config = load_config()
    pm = get_process_manager()

    if pm is None:
        return jsonify({"error": "Process manager not available"}), 500

    # Check if already running
    if component == "pool":
        pool_host = config.get("llm", {}).get("pool", {}).get("host", "127.0.0.1")
        pool_port = config.get("llm", {}).get("pool", {}).get("port", 9000)
        if check_pool_health(pool_host, pool_port):
            return jsonify({"error": "Pool is already running (detected on port)", "already_running": True}), 400
    elif pm.is_running(component):
        return jsonify({"error": f"{component} is already running"}), 400

    try:
        if component == "pool":
            pm.start_pool()
        elif component == "explorer":
            options = request.json or {}
            pm.start_explorer(options.get("mode", "start"))
        elif component == "documenter":
            pm.start_documenter()
        elif component == "researcher":
            pm.start_researcher()

        return jsonify({"status": "started", "component": component})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/stop/<component>", methods=["POST"])
def api_stop(component: str):
    """Stop a component."""
    if component not in ["pool", "explorer", "documenter", "researcher"]:
        return jsonify({"error": f"Unknown component: {component}"}), 400

    pm = get_process_manager()
    if pm is None:
        return jsonify({"error": "Process manager not available"}), 500

    if not pm.is_running(component):
        return jsonify({"error": f"{component} is not running"}), 400

    try:
        pm.stop(component)
        return jsonify({"status": "stopped", "component": component})
    except Exception as e:
        return jsonify({"status": "killed", "component": component, "error": str(e)})


@bp.route("/server/restart", methods=["POST"])
def api_server_restart():
    """Restart the server."""
    pm = get_process_manager()

    def delayed_restart():
        import time
        time.sleep(0.5)  # Let response send first

        # Terminate child processes cleanly before restart
        # Suppress stderr during shutdown to hide harmless asyncio transport warnings
        old_stderr = sys.stderr
        sys.stderr = open(os.devnull, "w")
        try:
            if pm:
                pm.cleanup_all()
        finally:
            sys.stderr.close()
            sys.stderr = old_stderr

        # Start new server process (detached on Windows)
        server_script = FANO_ROOT / "control" / "server.py"
        logs_dir = FANO_ROOT / "logs"
        logs_dir.mkdir(exist_ok=True)
        restart_log = logs_dir / "control.log"

        # Wait for port to be released
        time.sleep(1)

        if sys.platform == "win32":
            # Windows: use CREATE_NEW_PROCESS_GROUP and DETACHED_PROCESS
            # Note: close_fds must NOT be used with DETACHED_PROCESS on Windows
            # Run as module (-m control.server) to handle relative imports
            DETACHED_PROCESS = 0x00000008
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            log_file = open(restart_log, "w")
            subprocess.Popen(
                [sys.executable, "-m", "control.server"],
                cwd=str(FANO_ROOT),
                creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        else:
            # Unix: use nohup-like behavior
            # Run as module (-m control.server) to handle relative imports
            subprocess.Popen(
                [sys.executable, "-m", "control.server"],
                cwd=str(FANO_ROOT),
                start_new_session=True,
                close_fds=True,
            )

        # Exit current process
        os._exit(0)

    # Start restart in background thread
    thread = threading.Thread(target=delayed_restart, daemon=False)
    thread.start()

    return jsonify({"success": True, "message": "Server restarting..."})


@bp.route("/health", methods=["GET"])
def health():
    """Health check endpoint for restart polling."""
    return jsonify({"status": "ok"})
