"""Pool Blueprint - Deprecated, pool is no longer used.

All LLM access is now via OpenRouter API directly.
These endpoints are kept for API compatibility but return deprecation notices.
"""

from flask import Blueprint, jsonify

bp = Blueprint("pool", __name__, url_prefix="/api/pool")


def _deprecated_response():
    """Return a deprecation notice."""
    return jsonify({
        "error": "Pool is deprecated",
        "message": "LLM access is now via OpenRouter API. Pool is no longer needed.",
    }), 410  # 410 Gone


@bp.route("/activity")
def api_pool_activity():
    """Pool activity endpoint - deprecated."""
    return _deprecated_response()


@bp.route("/activity/detail")
def api_pool_activity_detail():
    """Pool activity detail endpoint - deprecated."""
    return _deprecated_response()


@bp.route("/status")
def api_pool_status():
    """Pool status endpoint - deprecated."""
    return _deprecated_response()


@bp.route("/kick/<backend>", methods=["POST"])
def api_pool_kick(backend):
    """Pool kick endpoint - deprecated."""
    return _deprecated_response()


@bp.route("/recovery/status")
def api_pool_recovery_status():
    """Pool recovery status endpoint - deprecated."""
    return _deprecated_response()
