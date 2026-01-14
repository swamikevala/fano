"""Orchestrator Blueprint - Proxy routes for orchestrator service."""

import requests
from flask import Blueprint, jsonify, request

bp = Blueprint("orchestrator", __name__, url_prefix="/api/orchestrator")

ORCHESTRATOR_URL = "http://127.0.0.1:9001"


def _proxy_get(endpoint: str, timeout: int = 5):
    """Proxy GET request to orchestrator."""
    try:
        resp = requests.get(f"{ORCHESTRATOR_URL}{endpoint}", timeout=timeout)
        resp.raise_for_status()
        return jsonify(resp.json())
    except requests.RequestException as e:
        return jsonify({"error": f"Orchestrator not available: {e}"}), 503


def _proxy_post(endpoint: str, data: dict = None, timeout: int = 5):
    """Proxy POST request to orchestrator."""
    try:
        resp = requests.post(
            f"{ORCHESTRATOR_URL}{endpoint}",
            json=data,
            timeout=timeout,
        )
        resp.raise_for_status()
        return jsonify(resp.json())
    except requests.RequestException as e:
        return jsonify({"error": f"Orchestrator not available: {e}"}), 503


@bp.route("/health")
def api_orchestrator_health():
    """Check orchestrator health."""
    return _proxy_get("/health", timeout=2)


@bp.route("/status")
def api_orchestrator_status():
    """Get full orchestrator status (queue, quotas, workers, modules)."""
    return _proxy_get("/status")


@bp.route("/tasks")
def api_orchestrator_tasks():
    """List tasks with optional filtering."""
    # Forward query params
    params = []
    if request.args.get("state"):
        params.append(f"state={request.args['state']}")
    if request.args.get("module"):
        params.append(f"module={request.args['module']}")
    if request.args.get("limit"):
        params.append(f"limit={request.args['limit']}")

    query = "?" + "&".join(params) if params else ""
    return _proxy_get(f"/tasks{query}")


@bp.route("/tasks/<task_id>")
def api_orchestrator_task(task_id: str):
    """Get task details by ID."""
    return _proxy_get(f"/tasks/{task_id}")


@bp.route("/queue")
def api_orchestrator_queue():
    """Get queue statistics."""
    return _proxy_get("/queue")


@bp.route("/quotas")
def api_orchestrator_quotas():
    """Get quota usage for all backends."""
    query = ""
    if request.args.get("type"):
        query = f"?type={request.args['type']}"
    return _proxy_get(f"/quotas{query}")


@bp.route("/quotas", methods=["POST"])
def api_orchestrator_update_quotas():
    """Update quota budgets."""
    return _proxy_post("/quotas", request.get_json())


@bp.route("/weights")
def api_orchestrator_weights():
    """Get current module weights."""
    return _proxy_get("/weights")


@bp.route("/weights", methods=["POST"])
def api_orchestrator_update_weights():
    """Update module priority weights."""
    return _proxy_post("/weights", request.get_json())


@bp.route("/failure-cache/clear", methods=["POST"])
def api_orchestrator_clear_failure_cache():
    """Clear failure cache."""
    return _proxy_post("/failure-cache/clear", request.get_json())


@bp.route("/workers")
def api_orchestrator_workers():
    """Get worker status by backend."""
    return _proxy_get("/workers")
