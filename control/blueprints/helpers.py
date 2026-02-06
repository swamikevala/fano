"""Shared helpers for v2 control panel blueprints.

Provides access to dependencies (store, bus, config) and standard
JSON envelope formatting.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from flask import current_app, jsonify, request

from control.async_utils import run_async
from shared.models import Event, generate_id


# ── Dependency access ────────────────────────────────────────


def get_store():
    """Return the StateStore from the current Flask app."""
    return current_app.config["state_store"]


def get_bus():
    """Return the EventBus from the current Flask app."""
    return current_app.config["event_bus"]


def get_config():
    """Return the Config from the current Flask app."""
    return current_app.config["app_config"]


# ── JSON envelope helpers ────────────────────────────────────


def ok(data: Any, status: int = 200, **extra) -> tuple:
    """Return a success envelope: {"ok": true, "data": ...}."""
    body: dict[str, Any] = {"ok": True, "data": data}
    body.update(extra)
    return jsonify(body), status


def ok_list(items: list[Any], total: int, offset: int, limit: int) -> tuple:
    """Return a paginated list envelope."""
    return jsonify({
        "ok": True,
        "data": items,
        "total": total,
        "offset": offset,
        "limit": limit,
    }), 200


def err(message: str, code: str, status: int = 400) -> tuple:
    """Return an error envelope: {"ok": false, "error": ..., "code": ...}."""
    return jsonify({"ok": False, "error": message, "code": code}), status


# ── Pagination ───────────────────────────────────────────────


def get_pagination() -> tuple[int, int]:
    """Extract offset and limit from query string (defaults: 0, 20)."""
    offset = request.args.get("offset", 0, type=int)
    limit = request.args.get("limit", 20, type=int)
    return offset, limit


def paginate(items: list, offset: int, limit: int) -> list:
    """Slice a list according to offset/limit."""
    return items[offset: offset + limit]


# ── Serialization ────────────────────────────────────────────


def _serialize_value(v: Any) -> Any:
    """Recursively convert dataclass-unfriendly types to JSON-safe values."""
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, dict):
        return {k: _serialize_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_serialize_value(item) for item in v]
    if hasattr(v, "value"):  # Enum
        return v.value
    return v


def serialize(obj: Any) -> dict:
    """Convert a frozen dataclass to a JSON-safe dict."""
    raw = asdict(obj)
    return {k: _serialize_value(v) for k, v in raw.items()}


# ── Event publishing ─────────────────────────────────────────


def publish_event(topic: str, payload: dict, source: str = "control") -> None:
    """Publish an event through the EventBus (sync wrapper)."""
    bus = get_bus()
    event = Event(
        topic=topic,
        timestamp=datetime.now(timezone.utc),
        source=source,
        payload=payload,
        correlation_id=generate_id(),
    )
    run_async(bus.publish(event))
