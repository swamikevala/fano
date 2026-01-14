"""
HTTP API for the Orchestrator.

Provides REST endpoints for status monitoring and configuration.
Designed to be run alongside the orchestrator in the same process.
"""

import asyncio
from functools import wraps
from typing import TYPE_CHECKING

from flask import Flask, jsonify, request

from shared.logging import get_logger

if TYPE_CHECKING:
    from .main import Orchestrator

log = get_logger("orchestrator", "api")

# Global reference to orchestrator instance (set by run_with_api)
_orchestrator: "Orchestrator" = None


def get_orchestrator() -> "Orchestrator":
    """Get the global orchestrator instance."""
    return _orchestrator


def require_orchestrator(f):
    """Decorator to require orchestrator to be running."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if _orchestrator is None:
            return jsonify({"error": "Orchestrator not initialized"}), 503
        if not _orchestrator._running:
            return jsonify({"error": "Orchestrator not running"}), 503
        return f(*args, **kwargs)
    return decorated


def create_app() -> Flask:
    """Create Flask app for orchestrator API."""
    app = Flask(__name__)

    @app.route("/health")
    def health():
        """Health check endpoint."""
        if _orchestrator and _orchestrator._running:
            return jsonify({"status": "healthy", "running": True})
        return jsonify({"status": "starting", "running": False})

    @app.route("/status")
    @require_orchestrator
    def status():
        """Get full orchestrator status."""
        return jsonify(_orchestrator.get_status())

    @app.route("/tasks")
    @require_orchestrator
    def list_tasks():
        """List all tasks with optional filtering."""
        state_filter = request.args.get("state")
        module_filter = request.args.get("module")
        limit = request.args.get("limit", 100, type=int)

        all_tasks = _orchestrator.state.get_all_tasks()

        # Apply filters
        tasks = []
        for task in all_tasks:
            if state_filter and task.state.value != state_filter:
                continue
            if module_filter and task.module != module_filter:
                continue
            tasks.append(task.to_dict())
            if len(tasks) >= limit:
                break

        return jsonify({
            "tasks": tasks,
            "total": len(all_tasks),
            "filtered": len(tasks),
        })

    @app.route("/tasks/<task_id>")
    @require_orchestrator
    def get_task(task_id: str):
        """Get task details by ID."""
        task = _orchestrator.get_task(task_id)
        if not task:
            return jsonify({"error": "Task not found"}), 404
        return jsonify(task.to_dict())

    @app.route("/queue")
    @require_orchestrator
    def queue_stats():
        """Get queue statistics."""
        return jsonify(_orchestrator.scheduler.get_queue_stats())

    @app.route("/quotas")
    @require_orchestrator
    def quota_status():
        """Get quota usage for all backends."""
        quota_type = request.args.get("type")
        return jsonify(_orchestrator.allocator.get_status(quota_type))

    @app.route("/quotas", methods=["POST"])
    @require_orchestrator
    def update_quotas():
        """Update quota budgets."""
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        quota_type = data.get("quota_type")
        if not quota_type:
            return jsonify({"error": "quota_type required"}), 400

        _orchestrator.allocator.set_budget(
            quota_type=quota_type,
            explorer=data.get("explorer"),
            documenter=data.get("documenter"),
            buffer=data.get("buffer"),
        )

        log.info("api.quotas_updated", quota_type=quota_type)
        return jsonify({"success": True, "quota_type": quota_type})

    @app.route("/weights")
    @require_orchestrator
    def get_weights():
        """Get current module weights."""
        return jsonify(_orchestrator.scheduler.module_weights)

    @app.route("/weights", methods=["POST"])
    @require_orchestrator
    def update_weights():
        """Update module priority weights."""
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        _orchestrator.update_module_weights(data)
        log.info("api.weights_updated", weights=data)
        return jsonify({"success": True, "weights": data})

    @app.route("/failure-cache/clear", methods=["POST"])
    @require_orchestrator
    def clear_failure_cache():
        """Clear failure cache for a key or all keys."""
        data = request.get_json() or {}
        key = data.get("key")

        _orchestrator.clear_failure_cache(key)
        return jsonify({"success": True, "key": key or "all"})

    @app.route("/workers")
    @require_orchestrator
    def worker_status():
        """Get worker status by backend."""
        if _orchestrator.workers:
            return jsonify(_orchestrator.workers.get_status())
        return jsonify({})

    return app


def run_api_server(host: str = "127.0.0.1", port: int = 9001):
    """Run the Flask API server (blocking)."""
    app = create_app()
    app.run(host=host, port=port, threaded=True)


async def run_with_api(
    orchestrator: "Orchestrator",
    host: str = "127.0.0.1",
    port: int = 9001,
):
    """
    Run orchestrator with HTTP API.

    Starts the orchestrator and runs Flask API in a thread.
    """
    global _orchestrator
    _orchestrator = orchestrator

    # Start orchestrator
    await orchestrator.start()

    # Run Flask in a thread
    import threading
    api_thread = threading.Thread(
        target=run_api_server,
        kwargs={"host": host, "port": port},
        daemon=True,
    )
    api_thread.start()
    log.info("api.server_started", host=host, port=port)

    # Keep running until orchestrator stops
    try:
        while orchestrator._running:
            await asyncio.sleep(1)
    finally:
        await orchestrator.stop()
