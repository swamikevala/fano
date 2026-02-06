"""Seeds blueprint — CRUD for seeds + user action events (v2 API)."""

from __future__ import annotations

from datetime import datetime, timezone

from flask import Blueprint, request

from control.async_utils import run_async
from shared.models import (
    Confidence,
    Seed,
    SeedStatus,
    SeedType,
    generate_id,
)

from .helpers import (
    err,
    get_pagination,
    get_store,
    ok,
    ok_list,
    paginate,
    publish_event,
    serialize,
)

bp = Blueprint("seeds_v2", __name__, url_prefix="/api")


@bp.route("/seeds", methods=["GET"])
def list_seeds():
    project_id = request.args.get("project_id")
    if not project_id:
        return err("project_id query parameter is required", "VALIDATION_ERROR")

    store = get_store()
    status_filter = request.args.get("status")
    ss = None
    if status_filter:
        try:
            ss = SeedStatus(status_filter)
        except ValueError:
            return err(f"Invalid status: {status_filter}", "VALIDATION_ERROR")

    seeds = run_async(store.list_seeds(project_id, status=ss))
    total = len(seeds)
    offset, limit = get_pagination()
    page = paginate(seeds, offset, limit)
    return ok_list([serialize(s) for s in page], total, offset, limit)


@bp.route("/seeds", methods=["POST"])
def create_seed():
    body = request.get_json(silent=True) or {}
    project_id = body.get("project_id")
    text = body.get("text")
    if not project_id or not text:
        return err("project_id and text are required", "VALIDATION_ERROR")

    now = datetime.now(timezone.utc)
    seed_type_str = body.get("type", "conjecture")
    try:
        seed_type = SeedType(seed_type_str)
    except ValueError:
        return err(f"Invalid seed type: {seed_type_str}", "VALIDATION_ERROR")

    conf_str = body.get("confidence")
    confidence = Confidence(conf_str) if conf_str else None

    seed = Seed(
        id=generate_id(),
        project_id=project_id,
        text=text,
        type=seed_type,
        priority=int(body.get("priority", 5)),
        tags=body.get("tags", []),
        confidence=confidence,
        source=body.get("source"),
        notes=body.get("notes"),
        status=SeedStatus.ACTIVE,
        parent_seed_id=body.get("parent_seed_id"),
        modification_reason=None,
        exploration_count=0,
        created_at=now,
        updated_at=now,
    )
    store = get_store()
    run_async(store.create_seed(seed))

    publish_event("user.seed.created", {
        "seed_id": seed.id,
        "project_id": project_id,
        "text": text,
    })

    return ok(serialize(seed), status=201)


@bp.route("/seeds/<seed_id>", methods=["GET"])
def get_seed(seed_id: str):
    store = get_store()
    seed = run_async(store.get_seed(seed_id))
    if seed is None:
        return err("Seed not found", "NOT_FOUND", 404)
    return ok(serialize(seed))


@bp.route("/seeds/<seed_id>", methods=["PUT"])
def update_seed(seed_id: str):
    store = get_store()
    existing = run_async(store.get_seed(seed_id))
    if existing is None:
        return err("Seed not found", "NOT_FOUND", 404)

    body = request.get_json(silent=True) or {}
    allowed = {"text", "priority", "tags", "confidence", "notes", "status"}
    fields = {k: v for k, v in body.items() if k in allowed}
    if not fields:
        return err("No updatable fields provided", "VALIDATION_ERROR")

    run_async(store.update_seed(seed_id, **fields))
    return ok({"updated": True})


@bp.route("/seeds/<seed_id>", methods=["DELETE"])
def delete_seed(seed_id: str):
    """Soft-delete: set status to retired."""
    store = get_store()
    existing = run_async(store.get_seed(seed_id))
    if existing is None:
        return err("Seed not found", "NOT_FOUND", 404)

    run_async(store.update_seed(seed_id, status=SeedStatus.RETIRED.value))
    return ok({"deleted": True})


@bp.route("/seeds/<seed_id>/approve", methods=["POST"])
def approve_seed(seed_id: str):
    store = get_store()
    existing = run_async(store.get_seed(seed_id))
    if existing is None:
        return err("Seed not found", "NOT_FOUND", 404)

    body = request.get_json(silent=True) or {}
    reason = body.get("modification_reason", "")
    run_async(store.update_seed(seed_id, modification_reason=reason))

    publish_event("user.seed.approved", {
        "seed_id": seed_id,
        "project_id": existing.project_id,
        "reason": reason,
    })

    return ok({"approved": True})


@bp.route("/seeds/<seed_id>/reject", methods=["POST"])
def reject_seed(seed_id: str):
    store = get_store()
    existing = run_async(store.get_seed(seed_id))
    if existing is None:
        return err("Seed not found", "NOT_FOUND", 404)

    body = request.get_json(silent=True) or {}
    reason = body.get("reason", "")

    publish_event("user.seed.rejected", {
        "seed_id": seed_id,
        "project_id": existing.project_id,
        "reason": reason,
    })

    return ok({"rejected": True})
