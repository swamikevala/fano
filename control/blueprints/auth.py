"""Auth blueprint — LDAP login/logout (AD authentication)."""

from __future__ import annotations

from datetime import datetime, timezone

import ldap3
from flask import Blueprint, flash, redirect, render_template, request, session, url_for

from control.async_utils import run_async
from shared.models import User, generate_id

from .helpers import get_config, get_store

bp = Blueprint("auth", __name__, url_prefix="")


@bp.route("/login", methods=["GET"])
def login_page() -> str:
    return render_template("login.html")


@bp.route("/login", methods=["POST"])
def login_submit():
    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""

    if not username or not password:
        flash("Username and password are required.", "error")
        return redirect(url_for("auth.login_page"))

    config = get_config()
    store = get_store()
    dev_mode = config.get("auth.dev_mode", False)

    if not dev_mode:
        # DB settings take priority over config.yaml
        ldap_server = run_async(store.get_setting("auth.ldap_server")) or config.get("auth.ldap_server", "")
        ldap_domain = run_async(store.get_setting("auth.ldap_domain")) or config.get("auth.ldap_domain", "")

        if not ldap_server:
            flash("LDAP server not configured.", "error")
            return redirect(url_for("auth.login_page"))

        # Attempt LDAP bind
        display_name = username
        try:
            server = ldap3.Server(ldap_server, get_info=ldap3.NONE)
            bind_user = f"{ldap_domain}\\{username}" if ldap_domain else username
            conn = ldap3.Connection(server, user=bind_user, password=password, auto_bind=True)

            # Try to fetch displayName from AD
            try:
                base_dn = ",".join(f"DC={p}" for p in ldap_domain.split(".")) if "." in ldap_domain else f"DC={ldap_domain}"
                conn.search(base_dn, f"(sAMAccountName={username})",
                            attributes=["displayName"])
                if conn.entries:
                    dn = conn.entries[0].displayName.value
                    if dn:
                        display_name = dn
            except Exception:
                pass  # displayName lookup is best-effort

            conn.unbind()
        except Exception as exc:
            err_msg = str(exc)
            if "invalidCredentials" in err_msg or "INVALID_CREDENTIALS" in err_msg:
                flash("Invalid username or password.", "error")
            else:
                flash("Could not reach AD server. Please try again later.", "error")
            return redirect(url_for("auth.login_page"))
    else:
        display_name = username

    # Upsert local user
    user = run_async(store.get_user_by_username(username))
    if user is None:
        user = User(
            id=generate_id(),
            username=username,
            display_name=display_name,
            created_at=datetime.now(timezone.utc),
        )
        run_async(store.create_user(user))

    session["user_id"] = user.id
    return redirect(url_for("ui.dashboard"))


@bp.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("auth.login_page"))
