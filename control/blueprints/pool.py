"""Pool Blueprint - Proxy routes for pool service data."""

import requests
from flask import Blueprint, jsonify

bp = Blueprint("pool", __name__, url_prefix="/api/pool")

POOL_URL = "http://127.0.0.1:9000"


@bp.route("/activity")
def api_pool_activity():
    """Proxy to pool /activity endpoint."""
    try:
        resp = requests.get(f"{POOL_URL}/activity", timeout=2)
        resp.raise_for_status()
        return jsonify(resp.json())
    except requests.RequestException:
        return jsonify({"error": "Pool not available"}), 503


@bp.route("/activity/detail")
def api_pool_activity_detail():
    """Proxy to pool /activity/detail endpoint."""
    try:
        resp = requests.get(f"{POOL_URL}/activity/detail", timeout=2)
        resp.raise_for_status()
        return jsonify(resp.json())
    except requests.RequestException:
        return jsonify({"error": "Pool not available"}), 503


@bp.route("/status")
def api_pool_status():
    """Proxy to pool /status endpoint."""
    try:
        resp = requests.get(f"{POOL_URL}/status", timeout=2)
        resp.raise_for_status()
        return jsonify(resp.json())
    except requests.RequestException:
        return jsonify({"error": "Pool not available"}), 503
