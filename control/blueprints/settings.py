"""Settings blueprint — manage AD/LDAP and other system settings."""

from __future__ import annotations

import ldap3
from flask import Blueprint, request

from control.async_utils import run_async

from .helpers import err, get_config, get_store, ok

bp = Blueprint("settings", __name__, url_prefix="/api/settings")

_AUTH_KEYS = ("auth.ldap_server", "auth.ldap_domain", "auth.secret_key")


@bp.route("", methods=["GET"])
def get_settings():
    """Return all auth.* settings (DB values, falling back to config)."""
    store = get_store()
    config = get_config()
    saved = run_async(store.list_settings("auth."))
    result = {}
    for key in _AUTH_KEYS:
        result[key] = saved.get(key) or config.get(key, "")
    return ok(result)


@bp.route("", methods=["PUT"])
def put_settings():
    """Save auth.* settings to the database."""
    body = request.get_json(silent=True) or {}
    store = get_store()

    if "auth.ldap_server" not in body:
        return err("auth.ldap_server is required", "VALIDATION_ERROR")

    for key in _AUTH_KEYS:
        if key in body:
            run_async(store.set_setting(key, body[key]))

    # Re-read saved values to return
    saved = run_async(store.list_settings("auth."))
    config = get_config()
    result = {}
    for key in _AUTH_KEYS:
        result[key] = saved.get(key) or config.get(key, "")
    return ok(result)


@bp.route("/test-connection", methods=["POST"])
def test_connection():
    """Test LDAP connection with provided or saved settings."""
    body = request.get_json(silent=True) or {}
    store = get_store()
    config = get_config()

    saved = run_async(store.list_settings("auth."))
    ldap_server = body.get("auth.ldap_server") or saved.get("auth.ldap_server") or config.get("auth.ldap_server", "")
    ldap_domain = body.get("auth.ldap_domain") or saved.get("auth.ldap_domain") or config.get("auth.ldap_domain", "")

    if not ldap_server:
        return err("No LDAP server configured", "VALIDATION_ERROR")

    try:
        server = ldap3.Server(ldap_server, get_info=ldap3.NONE, connect_timeout=5)
        conn = ldap3.Connection(server, auto_bind=True)
        conn.unbind()
        return ok({"success": True, "server": ldap_server, "domain": ldap_domain})
    except Exception as exc:
        return ok({"success": False, "error": str(exc)})
