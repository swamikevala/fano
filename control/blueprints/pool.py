"""Pool Blueprint - Proxy routes for pool service data."""

import requests
from flask import Blueprint, jsonify, current_app

bp = Blueprint("pool", __name__, url_prefix="/api/pool")


def get_pool_url() -> str:
    """Get pool URL from config or fall back to default."""
    config = current_app.config.get("fano_config", {})
    pool_config = config.get("pool", {})
    host = pool_config.get("host", "127.0.0.1")
    port = pool_config.get("port", 9000)
    return f"http://{host}:{port}"


@bp.route("/activity")
def api_pool_activity():
    """Proxy to pool /activity endpoint."""
    try:
        resp = requests.get(f"{get_pool_url()}/activity", timeout=2)
        resp.raise_for_status()
        return jsonify(resp.json())
    except requests.RequestException:
        return jsonify({"error": "Pool not available"}), 503


@bp.route("/activity/detail")
def api_pool_activity_detail():
    """Proxy to pool /activity/detail endpoint."""
    try:
        resp = requests.get(f"{get_pool_url()}/activity/detail", timeout=2)
        resp.raise_for_status()
        return jsonify(resp.json())
    except requests.RequestException:
        return jsonify({"error": "Pool not available"}), 503


@bp.route("/status")
def api_pool_status():
    """Proxy to pool /status endpoint."""
    try:
        resp = requests.get(f"{get_pool_url()}/status", timeout=2)
        resp.raise_for_status()
        return jsonify(resp.json())
    except requests.RequestException:
        return jsonify({"error": "Pool not available"}), 503


@bp.route("/kick/<backend>", methods=["POST"])
def api_pool_kick(backend):
    """Kick a stuck worker - takes screenshot, fails job, reconnects browser."""
    try:
        resp = requests.post(f"{get_pool_url()}/kick/{backend}", timeout=30)
        resp.raise_for_status()
        return jsonify(resp.json())
    except requests.RequestException as e:
        return jsonify({"error": f"Pool not available: {e}"}), 503


@bp.route("/recovery/status")
def api_pool_recovery_status():
    """Proxy to pool /recovery/status endpoint."""
    try:
        resp = requests.get(f"{get_pool_url()}/recovery/status", timeout=2)
        resp.raise_for_status()
        return jsonify(resp.json())
    except requests.RequestException:
        return jsonify({"error": "Pool not available"}), 503
