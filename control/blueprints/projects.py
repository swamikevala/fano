"""Projects blueprint — CRUD for projects (v2 API)."""

from __future__ import annotations

from datetime import datetime, timezone

from flask import Blueprint, g, request

from control.async_utils import run_async
from shared.models import (
    EvaluationCriterion,
    Project,
    ProjectStatus,
    ResearchDomain,
    generate_id,
)

from .helpers import check_project_access, err, get_store, ok, serialize

bp = Blueprint("projects_v2", __name__, url_prefix="/api")


@bp.route("/projects", methods=["GET"])
def list_projects():
    store = get_store()
    status_filter = request.args.get("status")
    ps = None
    if status_filter:
        try:
            ps = ProjectStatus(status_filter)
        except ValueError:
            return err(f"Invalid status: {status_filter}", "VALIDATION_ERROR")
    owner_id = g.user.id if hasattr(g, "user") and g.user else None
    projects = run_async(store.list_projects(status=ps, owner_id=owner_id))
    data = [serialize(p) for p in projects]
    return ok(data)


@bp.route("/projects", methods=["POST"])
def create_project():
    body = request.get_json(silent=True) or {}
    name = body.get("name")
    goal = body.get("goal")
    if not name or not goal:
        return err("name and goal are required", "VALIDATION_ERROR")

    now = datetime.now(timezone.utc)
    criteria_raw = body.get("evaluation_criteria", [
        {"name": "rigor", "description": "Logical soundness", "weight": 1.0},
    ])
    criteria = [
        EvaluationCriterion(
            name=c.get("name", "rigor"),
            description=c.get("description", ""),
            weight=float(c.get("weight", 1.0)),
        )
        for c in criteria_raw
    ]
    domains_raw = body.get("research_domains", [])
    domains = [
        ResearchDomain(
            name=d.get("name", "default"),
            keywords=d.get("keywords", []),
            source_types=d.get("source_types", []),
        )
        for d in domains_raw
    ]

    owner_id = g.user.id if hasattr(g, "user") and g.user else None
    project = Project(
        id=generate_id(),
        owner_id=owner_id,
        name=name,
        goal=goal,
        context=body.get("context", ""),
        evaluation_criteria=criteria,
        exploration_guidance=body.get("exploration_guidance", ""),
        document_guidance=body.get("document_guidance", ""),
        seed_modification_enabled=bool(body.get("seed_modification_enabled", True)),
        seed_modification_require_approval=bool(body.get("seed_modification_require_approval", False)),
        research_domains=domains,
        status=ProjectStatus.ACTIVE,
        created_at=now,
        updated_at=now,
    )
    store = get_store()
    run_async(store.create_project(project))
    return ok(serialize(project), status=201)


@bp.route("/projects/<project_id>", methods=["GET"])
def get_project(project_id: str):
    denied = check_project_access(project_id)
    if denied:
        return denied
    store = get_store()
    project = run_async(store.get_project(project_id))
    return ok(serialize(project))


@bp.route("/projects/<project_id>", methods=["PUT"])
def update_project(project_id: str):
    denied = check_project_access(project_id)
    if denied:
        return denied
    store = get_store()
    existing = run_async(store.get_project(project_id))

    body = request.get_json(silent=True) or {}
    allowed = {"name", "goal", "context", "exploration_guidance",
               "document_guidance", "status", "seed_modification_enabled",
               "seed_modification_require_approval"}
    fields = {k: v for k, v in body.items() if k in allowed}
    if not fields:
        return err("No updatable fields provided", "VALIDATION_ERROR")

    run_async(store.update_project(project_id, **fields))
    updated = run_async(store.get_project(project_id))
    return ok(serialize(updated))
