"""
Review Server - Web UI for reviewing research insights.

Provides a local web interface to view:
- Atomic insights (pending, blessed, interesting, rejected)
- Review panel decisions and deliberation
- Refinement history
- Augmentations (diagrams, tables, proofs, code)
"""

import json
import re
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
import markdown
import yaml

# Import models
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from chunking import AtomicInsight, InsightStatus
from review_panel.models import ChunkReview, ReviewRound, ReviewResponse
from augmentation import AugmentedInsight, Augmentation, AugmentationType
from models.axiom import AxiomStore, SeedAphorism


def render_markdown(text: str) -> str:
    """Convert markdown to HTML while preserving LaTeX math."""
    if not text:
        return ""

    # Protect LaTeX blocks from markdown processing
    math_blocks = []

    def save_math(match):
        math_blocks.append(match.group(0))
        return f"%%MATH_{len(math_blocks)-1}%%"

    # Protect display math \[...\] and $$...$$
    text = re.sub(r'\\\[[\s\S]*?\\\]', save_math, text)
    text = re.sub(r'\$\$[\s\S]*?\$\$', save_math, text)

    # Protect inline math \(...\) and $...$
    text = re.sub(r'\\\(.*?\\\)', save_math, text)
    text = re.sub(r'\$[^\$\n]+\$', save_math, text)

    # Process markdown
    md = markdown.Markdown(
        extensions=[
            'fenced_code',
            'tables',
            'nl2br',
        ]
    )
    html = md.convert(text)

    # Restore math blocks
    for i, math in enumerate(math_blocks):
        html = html.replace(f"%%MATH_{i}%%", math)

    return html


# Load config
CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"
with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
TEMPLATE_DIR = Path(__file__).parent.parent.parent / "templates"

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))


# Register markdown filter for templates
@app.template_filter('markdown')
def markdown_filter(text):
    """Jinja2 filter to render markdown as HTML."""
    if not text:
        return ""
    return render_markdown(text)


def get_insights_by_status(status: str) -> list[AtomicInsight]:
    """Load all insights with a given status, sorted by most recently worked on first."""
    insights_dir = DATA_DIR / "chunks" / "insights" / status
    insights = []

    if insights_dir.exists():
        for json_file in insights_dir.glob("*.json"):
            try:
                insight = AtomicInsight.load(json_file)
                insights.append(insight)
            except Exception as e:
                print(f"[review] Error loading {json_file}: {e}")

    # Sort by most recent activity first (reviewed_at if available, else extracted_at)
    # Secondary sort by priority (descending)
    def sort_key(i):
        # Use reviewed_at if available, otherwise extracted_at
        last_activity = i.reviewed_at if i.reviewed_at else i.extracted_at
        return (last_activity, i.priority)

    return sorted(insights, key=sort_key, reverse=True)


def get_insight_by_id(insight_id: str) -> tuple[AtomicInsight, str]:
    """Load a specific insight by ID, return (insight, status)."""
    for status in ["pending", "blessed", "interesting", "rejected"]:
        json_path = DATA_DIR / "chunks" / "insights" / status / f"{insight_id}.json"
        if json_path.exists():
            return AtomicInsight.load(json_path), status
    return None, None


def get_seed_by_id(seed_id: str) -> SeedAphorism:
    """Load a seed aphorism by ID (e.g., 'seed-002')."""
    axiom_store = AxiomStore(DATA_DIR)
    seeds = axiom_store.get_seed_aphorisms()
    for seed in seeds:
        if seed.id == seed_id:
            return seed
    return None


def get_review_for_insight(insight_id: str) -> ChunkReview:
    """Load review data for an insight."""
    # Check completed reviews
    review_path = DATA_DIR / "reviews" / "completed" / f"{insight_id}.json"
    if review_path.exists():
        with open(review_path, encoding="utf-8") as f:
            return ChunkReview.from_dict(json.load(f))

    # Check disputed reviews
    review_path = DATA_DIR / "reviews" / "disputed" / f"{insight_id}.json"
    if review_path.exists():
        with open(review_path, encoding="utf-8") as f:
            return ChunkReview.from_dict(json.load(f))

    return None


def get_augmentations_for_insight(insight_id: str) -> AugmentedInsight:
    """Load augmentations for an insight."""
    aug_dir = DATA_DIR / "augmentations" / f"chunk_{insight_id}"
    if aug_dir.exists():
        return AugmentedInsight.load(aug_dir)
    return None


def get_stats() -> dict:
    """Get insight counts by status."""
    stats = {}
    for status in ["pending", "blessed", "interesting", "rejected"]:
        dir_path = DATA_DIR / "chunks" / "insights" / status
        if dir_path.exists():
            stats[status] = len(list(dir_path.glob("*.json")))
        else:
            stats[status] = 0

    # Count insights currently being reviewed (in reviewing directory)
    reviewing_dir = DATA_DIR / "chunks" / "reviewing"
    if reviewing_dir.exists():
        stats["reviewing"] = len(list(reviewing_dir.glob("*.json")))
    else:
        stats["reviewing"] = 0

    # Count disputed reviews
    disputed_dir = DATA_DIR / "reviews" / "disputed"
    if disputed_dir.exists():
        stats["disputed"] = len(list(disputed_dir.glob("*.json")))
    else:
        stats["disputed"] = 0

    return stats


def get_reviewing_insights() -> list[dict]:
    """Get insights currently being reviewed with their review progress."""
    reviewing_dir = DATA_DIR / "chunks" / "reviewing"
    insights = []

    if not reviewing_dir.exists():
        return insights

    for json_file in reviewing_dir.glob("*.json"):
        try:
            insight = AtomicInsight.load(json_file)

            # Try to get in-progress review data
            review = get_review_for_insight(insight.id)
            rounds_completed = len(review.rounds) if review else 0

            insights.append({
                "insight": insight,
                "review": review,
                "rounds_completed": rounds_completed,
                "augmentations": None,
            })
        except Exception as e:
            print(f"[review] Error loading {json_file}: {e}")

    # Sort by most recent activity (last round timestamp or extracted_at)
    def sort_key(item):
        review = item.get("review")
        if review and review.rounds:
            return review.rounds[-1].timestamp
        return item["insight"].extracted_at

    return sorted(insights, key=sort_key, reverse=True)


@app.route("/")
def index():
    """Main page - show pending insights."""
    insights = get_insights_by_status("pending")
    stats = get_stats()

    # Enrich insights with review and augmentation data
    enriched = []
    for insight in insights:
        enriched.append({
            "insight": insight,
            "review": get_review_for_insight(insight.id),
            "augmentations": get_augmentations_for_insight(insight.id),
        })

    return render_template(
        "review.html",
        insights=enriched,
        stats=stats,
        current_status="pending",
        page_title="Pending Review",
    )


@app.route("/blessed")
def blessed():
    """Show blessed insights with augmentations."""
    insights = get_insights_by_status("blessed")
    stats = get_stats()

    enriched = []
    for insight in insights:
        enriched.append({
            "insight": insight,
            "review": get_review_for_insight(insight.id),
            "augmentations": get_augmentations_for_insight(insight.id),
        })

    return render_template(
        "review.html",
        insights=enriched,
        stats=stats,
        current_status="blessed",
        page_title="⚡ Blessed Insights",
    )


@app.route("/interesting")
def interesting():
    """Show interesting insights."""
    insights = get_insights_by_status("interesting")
    stats = get_stats()

    enriched = []
    for insight in insights:
        enriched.append({
            "insight": insight,
            "review": get_review_for_insight(insight.id),
            "augmentations": get_augmentations_for_insight(insight.id),
        })

    return render_template(
        "review.html",
        insights=enriched,
        stats=stats,
        current_status="interesting",
        page_title="? Interesting",
    )


@app.route("/rejected")
def rejected():
    """Show rejected insights."""
    insights = get_insights_by_status("rejected")
    stats = get_stats()

    enriched = []
    for insight in insights:
        enriched.append({
            "insight": insight,
            "review": get_review_for_insight(insight.id),
            "augmentations": None,  # No augmentations for rejected
        })

    return render_template(
        "review.html",
        insights=enriched,
        stats=stats,
        current_status="rejected",
        page_title="✗ Rejected",
    )


@app.route("/reviewing")
def reviewing():
    """Show insights currently being reviewed (in deliberation)."""
    enriched = get_reviewing_insights()
    stats = get_stats()

    return render_template(
        "review.html",
        insights=enriched,
        stats=stats,
        current_status="reviewing",
        page_title="🔄 In Review",
    )


@app.route("/disputed")
def disputed():
    """Show disputed insights that need Round 4."""
    disputed_dir = DATA_DIR / "reviews" / "disputed"
    stats = get_stats()

    enriched = []
    if disputed_dir.exists():
        for json_file in disputed_dir.glob("*.json"):
            try:
                with open(json_file, encoding="utf-8") as f:
                    review = ChunkReview.from_dict(json.load(f))

                # Try to find the corresponding insight
                insight = None
                insight_id = review.chunk_id
                for status in ["pending", "blessed", "interesting", "rejected"]:
                    json_path = DATA_DIR / "chunks" / "insights" / status / f"{insight_id}.json"
                    if json_path.exists():
                        insight = AtomicInsight.load(json_path)
                        break

                if insight:
                    enriched.append({
                        "insight": insight,
                        "review": review,
                        "augmentations": get_augmentations_for_insight(insight_id),
                    })
            except Exception as e:
                print(f"[review] Error loading disputed {json_file}: {e}")

    # Sort by most recent
    enriched.sort(key=lambda x: x["review"].rounds[-1].timestamp if x["review"].rounds else x["insight"].extracted_at, reverse=True)

    return render_template(
        "review.html",
        insights=enriched,
        stats=stats,
        current_status="disputed",
        page_title="⚠️ Disputed Insights",
    )


@app.route("/insight/<insight_id>")
def view_insight(insight_id: str):
    """View a single insight with full details."""
    # Check if this is a seed ID
    if insight_id.startswith("seed-"):
        seed = get_seed_by_id(insight_id)
        if not seed:
            return f"Seed {insight_id} not found", 404

        return render_template(
            "seed_view.html",
            seed=seed,
            stats=get_stats(),
            page_title=f"Seed: {insight_id}",
        )

    insight, status = get_insight_by_id(insight_id)
    if not insight:
        return "Insight not found", 404

    enriched = [{
        "insight": insight,
        "review": get_review_for_insight(insight_id),
        "augmentations": get_augmentations_for_insight(insight_id),
    }]

    return render_template(
        "review.html",
        insights=enriched,
        stats=get_stats(),
        current_status=status,
        page_title=f"Insight {insight_id}",
        viewing_single=True,
    )


@app.route("/api/insight/<insight_id>/review")
def get_review_api(insight_id: str):
    """API endpoint to get review data for an insight."""
    review = get_review_for_insight(insight_id)
    if not review:
        return jsonify({"error": "Review not found"}), 404
    return jsonify(review.to_dict())


@app.route("/api/insight/<insight_id>/augmentations")
def get_augmentations_api(insight_id: str):
    """API endpoint to get augmentations for an insight."""
    aug = get_augmentations_for_insight(insight_id)
    if not aug:
        return jsonify({"error": "No augmentations found"}), 404
    return jsonify(aug.to_dict())


@app.route("/api/augmentation/<insight_id>/diagram")
def get_diagram(insight_id: str):
    """Serve generated diagram image."""
    # Check for generated image files
    aug_dir = DATA_DIR / "augmentations" / f"chunk_{insight_id}"

    for ext in ["svg", "png", "jpg"]:
        img_path = aug_dir / f"diagram.{ext}"
        if img_path.exists():
            return send_file(img_path)

    return "Diagram not found", 404


@app.route("/api/stats")
def stats_api():
    """API endpoint to get stats."""
    return jsonify(get_stats())


@app.route("/api/feedback", methods=["POST"])
def submit_feedback():
    """Submit manual feedback for a pending insight."""
    data = request.json
    insight_id = data.get("insight_id")
    feedback = data.get("feedback")  # "bless", "interesting", "reject"
    notes = data.get("notes", "")

    if not insight_id or not feedback:
        return jsonify({"error": "Missing insight_id or feedback"}), 400

    insight, current_status = get_insight_by_id(insight_id)
    if not insight:
        return jsonify({"error": "Insight not found"}), 404

    # Map feedback to rating
    rating_map = {
        "bless": "⚡",
        "interesting": "?",
        "reject": "✗",
    }

    if feedback not in rating_map:
        return jsonify({"error": "Invalid feedback value"}), 400

    # Apply rating
    insight.apply_rating(rating_map[feedback], notes=f"Manual review: {notes}")

    # Remove from old location
    old_path = DATA_DIR / "chunks" / "insights" / current_status / f"{insight_id}.json"
    old_md_path = DATA_DIR / "chunks" / "insights" / current_status / f"{insight_id}.md"

    # Save to new location
    insight.save(DATA_DIR / "chunks")

    # Clean up old files
    if old_path.exists():
        old_path.unlink()
    if old_md_path.exists():
        old_md_path.unlink()

    return jsonify({"success": True, "new_status": insight.status.value})


@app.route("/api/priority", methods=["POST"])
def update_priority():
    """Update the priority of an insight."""
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

    insight, current_status = get_insight_by_id(insight_id)
    if not insight:
        return jsonify({"error": "Insight not found"}), 404

    # Update priority
    insight.set_priority(priority)

    # Save in place (same status directory)
    insight.save(DATA_DIR / "chunks")

    return jsonify({"success": True, "priority": insight.priority})


def start_server():
    """Start the review server."""
    host = CONFIG["review_server"]["host"]
    port = CONFIG["review_server"]["port"]
    debug = CONFIG["review_server"].get("debug", False)

    print(f"\n  Fano Explorer Review Server")
    print(f"  ===========================")
    print(f"  Running at: http://{host}:{port}")
    if debug:
        print(f"  Mode: DEBUG (auto-reload enabled)")
    print(f"")
    print(f"  Pages:")
    print(f"    /            - Pending insights")
    print(f"    /reviewing   - Insights in review")
    print(f"    /disputed    - Disputed insights (needs Round 4)")
    print(f"    /blessed     - Blessed insights with augmentations")
    print(f"    /interesting - Interesting insights")
    print(f"    /rejected    - Rejected insights")
    print(f"")
    print(f"  Press Ctrl+C to stop\n")

    # In debug mode, Flask auto-reloads on Python and template changes
    # Extra files can be watched with extra_files parameter
    extra_files = None
    if debug:
        # Watch template files for changes
        extra_files = list(TEMPLATE_DIR.glob("*.html"))

    app.run(host=host, port=port, debug=debug, extra_files=extra_files)


if __name__ == "__main__":
    start_server()
