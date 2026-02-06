"""Control Panel Server v2 — Flask app translating HTTP to events and state reads.

This module provides the application factory. The control panel contains
NO business logic; it translates HTTP requests to EventBus events and
StateStore reads to JSON envelopes.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from typing import TYPE_CHECKING

from flask import Flask, jsonify

if TYPE_CHECKING:
    from shared.config import Config
    from shared.models import EventBusInterface, StateStoreInterface


# Suppress noisy warnings on Windows
warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed transport")
if sys.platform == "win32":
    def _quiet_unraisablehook(unraisable):
        if unraisable.exc_type is ValueError and "closed pipe" in str(unraisable.exc_value):
            return
        sys.__unraisablehook__(unraisable)
    sys.unraisablehook = _quiet_unraisablehook


class _StatusLogFilter(logging.Filter):
    _NOISY = ["/api/status", "/health"]

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(ep in msg for ep in self._NOISY)


logging.getLogger("werkzeug").addFilter(_StatusLogFilter())


# ── Application factory ─────────────────────────────────────


def create_app(
    store: StateStoreInterface,
    event_bus: EventBusInterface,
    config: Config,
) -> Flask:
    """Create Flask app with all v2 blueprints registered.

    Args:
        store: Initialised StateStore (already connected).
        event_bus: EventBus instance for publishing user actions.
        config: Validated Config object.

    Returns:
        Configured Flask application.
    """
    app = Flask(__name__)

    # Secret key for session cookies
    secret = config.get("auth.secret_key", "") if hasattr(config, "get") else ""
    app.secret_key = secret or os.urandom(32)

    # Stash dependencies where blueprints can reach them
    app.config["state_store"] = store
    app.config["event_bus"] = event_bus
    app.config["app_config"] = config

    # Import blueprints here (after module fully loads) to avoid circular deps
    from control.blueprints import (
        annotations_bp,
        auth_bp,
        document_bp,
        insights_bp,
        projects_bp,
        research_bp,
        seeds_bp,
        status_bp,
        ui_bp,
    )

    # Register auth blueprint first
    app.register_blueprint(auth_bp)

    # Register v2 API blueprints
    app.register_blueprint(projects_bp)
    app.register_blueprint(seeds_bp)
    app.register_blueprint(insights_bp)
    app.register_blueprint(document_bp)
    app.register_blueprint(annotations_bp)
    app.register_blueprint(research_bp)
    app.register_blueprint(status_bp)

    # Register UI page routes
    app.register_blueprint(ui_bp)

    # Auth middleware — enforce login on all routes except /login and /static
    @app.before_request
    def enforce_auth():
        from flask import request
        skip_prefixes = ("/login", "/static")
        if any(request.path.startswith(p) for p in skip_prefixes):
            return None
        from control.blueprints.helpers import require_auth
        return require_auth()

    # Global error handler for unhandled exceptions
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"ok": False, "error": "Resource not found", "code": "NOT_FOUND"}), 404

    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({"ok": False, "error": "Internal server error", "code": "INTERNAL_ERROR"}), 500

    return app
