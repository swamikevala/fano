"""Document blueprint — read document + sections (v2 API)."""

from __future__ import annotations

from flask import Blueprint, request

from control.async_utils import run_async

from .helpers import err, get_store, ok, serialize

bp = Blueprint("document_v2", __name__, url_prefix="/api")


@bp.route("/document", methods=["GET"])
def get_document():
    """Return the full document as rendered markdown."""
    project_id = request.args.get("project_id")
    if not project_id:
        return err("project_id query parameter is required", "VALIDATION_ERROR")

    store = get_store()
    sections = run_async(store.list_sections(project_id))

    # Assemble markdown from ordered sections
    parts: list[str] = []
    for sec in sections:
        parts.append(sec.content)

    markdown = "\n\n".join(parts) if parts else ""
    return ok({
        "markdown": markdown,
        "section_count": len(sections),
        "project_id": project_id,
    })


@bp.route("/document/sections", methods=["GET"])
def list_sections():
    project_id = request.args.get("project_id")
    if not project_id:
        return err("project_id query parameter is required", "VALIDATION_ERROR")

    store = get_store()
    sections = run_async(store.list_sections(project_id))
    return ok([serialize(s) for s in sections])
