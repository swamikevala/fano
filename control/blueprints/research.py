"""Research blueprint — findings, sources, research requests (v2 API)."""

from __future__ import annotations

from flask import Blueprint, request

from control.async_utils import run_async

from .helpers import check_project_access, err, get_store, ok, publish_event, serialize

bp = Blueprint("research_v2", __name__, url_prefix="/api/research")


@bp.route("/findings", methods=["GET"])
def list_findings():
    project_id = request.args.get("project_id")
    if not project_id:
        return err("project_id query parameter is required", "VALIDATION_ERROR")
    denied = check_project_access(project_id)
    if denied:
        return denied

    store = get_store()
    findings = run_async(store.list_findings(project_id))
    return ok([serialize(f) for f in findings])


@bp.route("/sources", methods=["GET"])
def list_sources():
    project_id = request.args.get("project_id")
    if not project_id:
        return err("project_id query parameter is required", "VALIDATION_ERROR")
    denied = check_project_access(project_id)
    if denied:
        return denied

    store = get_store()
    sources = run_async(store.list_sources(project_id))
    return ok([serialize(s) for s in sources])


@bp.route("/request", methods=["POST"])
def research_request():
    body = request.get_json(silent=True) or {}
    project_id = body.get("project_id")
    query = body.get("query")
    if not project_id or not query:
        return err("project_id and query are required", "VALIDATION_ERROR")
    denied = check_project_access(project_id)
    if denied:
        return denied

    publish_event("user.research.requested", {
        "project_id": project_id,
        "query": query,
        "domain": body.get("domain"),
    })

    return ok({"requested": True, "query": query}, status=201)
