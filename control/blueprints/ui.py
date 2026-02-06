"""UI blueprint — serves HTML template pages."""

from __future__ import annotations

from flask import Blueprint, g, render_template, request

from control.async_utils import run_async
from shared.models import AnnotationStatus, ProjectStatus, SectionStatus

from .helpers import check_project_access, get_store

bp = Blueprint("ui", __name__, url_prefix="")


@bp.route("/")
@bp.route("/dashboard")
def dashboard() -> str:
    """Serve the main dashboard page."""
    return render_template("dashboard.html")


@bp.route("/settings")
def settings() -> str:
    """Serve the settings page."""
    return render_template("settings.html")


@bp.route("/document")
def document() -> str:
    """Serve the document viewer with data from the store."""
    store = get_store()

    # Resolve project — explicit param or first active project (scoped to user)
    project_id = request.args.get("project_id")
    project = None
    owner_id = g.user.id if hasattr(g, "user") and g.user else None
    if project_id:
        project = run_async(store.get_project(project_id))
        # Verify ownership
        if project and project.owner_id is not None and owner_id and project.owner_id != owner_id:
            project = None
    if not project:
        projects = run_async(store.list_projects(status=ProjectStatus.ACTIVE, owner_id=owner_id))
        if projects:
            project = projects[0]
            project_id = project.id

    # Fetch sections and annotations
    sections = []
    annotations = []
    if project_id:
        sections = run_async(store.list_sections(project_id))
        annotations = run_async(store.list_annotations(project_id))

    sections.sort(key=lambda s: s.order_index)

    # Build lookup for annotation → section title
    section_map = {s.id: s for s in sections}

    # Compute stats
    stats = {
        "sections": len(sections),
        "stable": sum(1 for s in sections if s.status == SectionStatus.STABLE),
        "provisional": sum(
            1 for s in sections if s.status == SectionStatus.PROVISIONAL
        ),
        "needs_work": sum(
            1 for s in sections if s.status == SectionStatus.NEEDS_WORK
        ),
        "annotations": len(annotations),
        "reviews": sum(s.review_count for s in sections),
    }

    # Annotation filter tab counts
    ann_counts = {
        "all": len(annotations),
        "open": sum(
            1 for a in annotations if a.status == AnnotationStatus.OPEN
        ),
        "attempted": sum(
            1 for a in annotations if a.status == AnnotationStatus.ATTEMPTED
        ),
        "resolved": sum(
            1 for a in annotations if a.status == AnnotationStatus.RESOLVED
        ),
    }

    return render_template(
        "document.html",
        project=project,
        sections=sections,
        annotations=annotations,
        section_map=section_map,
        stats=stats,
        ann_counts=ann_counts,
    )
