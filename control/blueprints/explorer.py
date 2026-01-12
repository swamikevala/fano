"""
Explorer Blueprint - Insights management routes.
"""

import json
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from flask import Blueprint, jsonify, request

from explorer.src.models.axiom import AxiomStore, SeedAphorism

from ..services import (
    EXPLORER_DATA_DIR,
    get_explorer_stats,
    get_insight_by_id,
    get_insights_by_status,
    get_review_for_insight,
    load_insight_json,
)

# Initialize AxiomStore for seeds management
_axiom_store: AxiomStore | None = None


def get_axiom_store() -> AxiomStore:
    """Get or create the AxiomStore instance."""
    global _axiom_store
    if _axiom_store is None:
        _axiom_store = AxiomStore(EXPLORER_DATA_DIR)
    return _axiom_store

bp = Blueprint("explorer", __name__, url_prefix="/api/explorer")


@bp.route("/stats")
def api_explorer_stats():
    """Get explorer insight stats."""
    return jsonify(get_explorer_stats())


@bp.route("/insights/<status>")
def api_explorer_insights(status: str):
    """Get insights by status."""
    valid_statuses = ["pending", "blessed", "interesting", "rejected", "reviewing", "disputed"]
    if status not in valid_statuses:
        return jsonify({"error": f"Invalid status: {status}"}), 400

    if status == "reviewing":
        # Get insights currently being reviewed
        reviewing_dir = EXPLORER_DATA_DIR / "chunks" / "reviewing"
        insights = []
        if reviewing_dir.exists():
            for json_file in reviewing_dir.glob("*.json"):
                data = load_insight_json(json_file)
                if data:
                    review = get_review_for_insight(data.get("id", ""))
                    insights.append({
                        "insight": data,
                        "review": review,
                        "rounds_completed": len(review.get("rounds", [])) if review else 0,
                    })
        return jsonify({"insights": insights, "status": status})

    elif status == "disputed":
        # Get disputed reviews
        disputed_dir = EXPLORER_DATA_DIR / "reviews" / "disputed"
        insights = []
        if disputed_dir.exists():
            for json_file in disputed_dir.glob("*.json"):
                review = load_insight_json(json_file)
                if review:
                    insight_id = review.get("chunk_id", "")
                    insight_data, _ = get_insight_by_id(insight_id)
                    if insight_data:
                        insights.append({
                            "insight": insight_data,
                            "review": review,
                        })
        return jsonify({"insights": insights, "status": status})

    else:
        # Regular status
        raw_insights = get_insights_by_status(status)
        insights = []
        for data in raw_insights:
            review = get_review_for_insight(data.get("id", ""))
            insights.append({
                "insight": data,
                "review": review,
            })
        return jsonify({"insights": insights, "status": status})


@bp.route("/insight/<insight_id>")
def api_explorer_insight(insight_id: str):
    """Get a specific insight."""
    insight, status = get_insight_by_id(insight_id)
    if not insight:
        return jsonify({"error": "Insight not found"}), 404

    review = get_review_for_insight(insight_id)
    return jsonify({
        "insight": insight,
        "review": review,
        "status": status,
    })


@bp.route("/feedback", methods=["POST"])
def api_explorer_feedback():
    """Submit feedback for an insight."""
    data = request.json
    insight_id = data.get("insight_id")
    feedback = data.get("feedback")  # "bless", "interesting", "reject"
    notes = data.get("notes", "")

    if not insight_id or not feedback:
        return jsonify({"error": "Missing insight_id or feedback"}), 400

    # Map feedback to rating and new status
    rating_map = {"bless": "⚡", "interesting": "?", "reject": "✗"}
    status_map = {"bless": "blessed", "interesting": "interesting", "reject": "rejected"}

    if feedback not in rating_map:
        return jsonify({"error": "Invalid feedback value"}), 400

    insight_data, current_status = get_insight_by_id(insight_id)
    if not insight_data:
        return jsonify({"error": "Insight not found"}), 404

    # Update the insight data
    insight_data["rating"] = rating_map[feedback]
    insight_data["status"] = status_map[feedback]
    insight_data["reviewed_at"] = datetime.now().isoformat()
    if notes:
        insight_data["review_notes"] = f"Manual review: {notes}"

    new_status = status_map[feedback]

    # Save to new location
    new_dir = EXPLORER_DATA_DIR / "chunks" / "insights" / new_status
    new_dir.mkdir(parents=True, exist_ok=True)
    new_path = new_dir / f"{insight_id}.json"

    with open(new_path, "w", encoding="utf-8") as f:
        json.dump(insight_data, f, indent=2, default=str)

    # Remove from old location
    old_path = EXPLORER_DATA_DIR / "chunks" / "insights" / current_status / f"{insight_id}.json"
    if old_path.exists():
        old_path.unlink()
    old_md = EXPLORER_DATA_DIR / "chunks" / "insights" / current_status / f"{insight_id}.md"
    if old_md.exists():
        old_md.unlink()

    return jsonify({"success": True, "new_status": new_status})


@bp.route("/priority", methods=["POST"])
def api_explorer_priority():
    """Update insight priority."""
    data = request.json
    insight_id = data.get("insight_id")
    priority = data.get("priority")

    if not insight_id or priority is None:
        return jsonify({"error": "Missing insight_id or priority"}), 400

    try:
        priority = int(priority)
        if not 1 <= priority <= 10:
            return jsonify({"error": "Priority must be between 1 and 10"}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid priority value"}), 400

    insight_data, status = get_insight_by_id(insight_id)
    if not insight_data:
        return jsonify({"error": "Insight not found"}), 404

    # Update priority
    insight_data["priority"] = priority

    # Save in place
    json_path = EXPLORER_DATA_DIR / "chunks" / "insights" / status / f"{insight_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(insight_data, f, indent=2, default=str)

    return jsonify({"success": True, "priority": priority})


# ============ Seeds API ============


@bp.route("/seeds")
def api_get_seeds():
    """Get all seeds/axioms."""
    store = get_axiom_store()
    type_filter = request.args.get("type")
    sort_by_priority = request.args.get("sort", "true").lower() == "true"

    seeds = store.get_seed_aphorisms(type_filter=type_filter, sort_by_priority=sort_by_priority)

    # Convert to dict for JSON serialization
    seeds_data = [asdict(seed) for seed in seeds]

    # Get counts by type
    all_seeds = store.get_seed_aphorisms(sort_by_priority=False)
    counts = {
        "total": len(all_seeds),
        "axiom": len([s for s in all_seeds if s.type == "axiom"]),
        "conjecture": len([s for s in all_seeds if s.type == "conjecture"]),
        "question": len([s for s in all_seeds if s.type == "question"]),
    }

    return jsonify({"seeds": seeds_data, "counts": counts})


@bp.route("/seed/<seed_id>")
def api_get_seed(seed_id: str):
    """Get a specific seed by ID."""
    store = get_axiom_store()
    seed = store.get_seed_by_id(seed_id)

    if not seed:
        return jsonify({"error": "Seed not found"}), 404

    return jsonify({"seed": asdict(seed)})


@bp.route("/seed", methods=["POST"])
def api_create_seed():
    """Create a new seed."""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Seed text is required"}), 400

    # Generate a unique ID
    seed_id = f"seed-{uuid.uuid4().hex[:8]}"

    # Parse priority
    priority = data.get("priority", 5)
    if isinstance(priority, str):
        priority_map = {"high": 8, "medium": 5, "low": 2}
        priority = priority_map.get(priority.lower(), 5)
    try:
        priority = max(1, min(10, int(priority)))
    except (ValueError, TypeError):
        priority = 5

    # Parse tags
    tags = data.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]

    seed = SeedAphorism(
        id=seed_id,
        text=text,
        type=data.get("type", "conjecture"),
        priority=priority,
        tags=tags,
        confidence=data.get("confidence", "high"),
        source=data.get("source", "user"),
        notes=data.get("notes", ""),
    )

    store = get_axiom_store()
    store.add_seed(seed)

    return jsonify({"success": True, "seed": asdict(seed)})


@bp.route("/seed/<seed_id>", methods=["PUT"])
def api_update_seed(seed_id: str):
    """Update an existing seed."""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    store = get_axiom_store()

    # Check if seed exists
    existing = store.get_seed_by_id(seed_id)
    if not existing:
        return jsonify({"error": "Seed not found"}), 404

    # Parse tags if provided as string
    if "tags" in data and isinstance(data["tags"], str):
        data["tags"] = [t.strip() for t in data["tags"].split(",") if t.strip()]

    success = store.update_seed(seed_id, data)
    if success:
        updated_seed = store.get_seed_by_id(seed_id)
        return jsonify({"success": True, "seed": asdict(updated_seed)})
    else:
        return jsonify({"error": "Failed to update seed"}), 500


@bp.route("/seed/<seed_id>", methods=["DELETE"])
def api_delete_seed(seed_id: str):
    """Delete a seed."""
    store = get_axiom_store()

    success = store.delete_seed(seed_id)
    if success:
        return jsonify({"success": True})
    else:
        return jsonify({"error": "Seed not found"}), 404


@bp.route("/seed/<seed_id>/priority", methods=["POST"])
def api_update_seed_priority(seed_id: str):
    """Update seed priority."""
    data = request.json
    priority = data.get("priority")

    if priority is None:
        return jsonify({"error": "Missing priority"}), 400

    try:
        priority = int(priority)
        if not 1 <= priority <= 10:
            return jsonify({"error": "Priority must be between 1 and 10"}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid priority value"}), 400

    store = get_axiom_store()
    success = store.update_seed(seed_id, {"priority": priority})

    if success:
        return jsonify({"success": True, "priority": priority})
    else:
        return jsonify({"error": "Seed not found"}), 404
