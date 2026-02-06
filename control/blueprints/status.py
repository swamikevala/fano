"""Status blueprint — system health, metrics, events (v2 API)."""

from __future__ import annotations

from flask import Blueprint, request

from control.async_utils import run_async

from .helpers import err, get_store, ok, serialize

bp = Blueprint("status_v2", __name__, url_prefix="/api/status")


@bp.route("", methods=["GET"])
def status():
    """Return module health overview."""
    # In v2, modules register health via events. For now, return a stub
    # indicating the control panel itself is healthy.
    data = {
        "modules": {
            "control": {"healthy": True, "message": "Control panel running"},
            "explorer": {"healthy": False, "message": "Not connected"},
            "documenter": {"healthy": False, "message": "Not connected"},
            "researcher": {"healthy": False, "message": "Not connected"},
        },
    }
    return ok(data)


@bp.route("/metrics", methods=["GET"])
def metrics():
    name = request.args.get("name")
    if not name:
        return err("name query parameter is required", "VALIDATION_ERROR")

    store = get_store()
    rows = run_async(store.query_metrics(name))
    return ok(rows)


@bp.route("/events", methods=["GET"])
def events():
    store = get_store()
    topic = request.args.get("topic")
    rows = run_async(store.list_events(topic=topic))
    return ok([serialize(e) for e in rows])
