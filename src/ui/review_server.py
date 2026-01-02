"""
Review Server - Web UI for reviewing research chunks.

Provides a simple local web interface for the human oracle to
review pending chunks and provide feedback:
- ⚡ Profound ("This is real")
- ? Interesting ("Not sure yet")  
- ✗ Wrong ("This isn't right")
"""

import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for
import yaml

from models import Chunk, ChunkStatus, ChunkFeedback, AxiomStore, BlessedInsight


# Load config
CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
TEMPLATE_DIR = Path(__file__).parent.parent.parent / "templates"

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))


def get_pending_chunks() -> list[Chunk]:
    """Load all pending chunks."""
    chunks_dir = DATA_DIR / "chunks" / "pending"
    chunks = []
    if chunks_dir.exists():
        for json_file in chunks_dir.glob("*.json"):
            try:
                chunks.append(Chunk.load(json_file))
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
    return sorted(chunks, key=lambda c: c.created_at, reverse=True)


def get_chunk_by_id(chunk_id: str) -> Chunk:
    """Load a specific chunk by ID."""
    for status in ["pending", "profound", "interesting", "rejected"]:
        json_path = DATA_DIR / "chunks" / status / f"{chunk_id}.json"
        if json_path.exists():
            return Chunk.load(json_path)
    return None


@app.route("/")
def index():
    """Main review page showing pending chunks."""
    chunks = get_pending_chunks()
    
    # Get stats
    stats = {
        "pending": len(list((DATA_DIR / "chunks" / "pending").glob("*.json"))) if (DATA_DIR / "chunks" / "pending").exists() else 0,
        "profound": len(list((DATA_DIR / "chunks" / "profound").glob("*.json"))) if (DATA_DIR / "chunks" / "profound").exists() else 0,
        "interesting": len(list((DATA_DIR / "chunks" / "interesting").glob("*.json"))) if (DATA_DIR / "chunks" / "interesting").exists() else 0,
        "rejected": len(list((DATA_DIR / "chunks" / "rejected").glob("*.json"))) if (DATA_DIR / "chunks" / "rejected").exists() else 0,
    }
    
    return render_template("review.html", chunks=chunks, stats=stats)


@app.route("/chunk/<chunk_id>")
def view_chunk(chunk_id: str):
    """View a specific chunk."""
    chunk = get_chunk_by_id(chunk_id)
    if not chunk:
        return "Chunk not found", 404
    return render_template("review.html", chunks=[chunk], viewing_single=True, stats={})


@app.route("/api/feedback", methods=["POST"])
def submit_feedback():
    """Submit feedback for a chunk."""
    data = request.json
    chunk_id = data.get("chunk_id")
    feedback = data.get("feedback")  # profound, interesting, rejected
    notes = data.get("notes", "")
    
    if not chunk_id or not feedback:
        return jsonify({"error": "Missing chunk_id or feedback"}), 400
    
    if feedback not in ["profound", "interesting", "rejected"]:
        return jsonify({"error": "Invalid feedback value"}), 400
    
    # Load chunk
    chunk = get_chunk_by_id(chunk_id)
    if not chunk:
        return jsonify({"error": "Chunk not found"}), 404
    
    # Apply feedback
    feedback_enum = ChunkFeedback(feedback)
    chunk.apply_feedback(feedback_enum, notes)
    
    # Move to appropriate directory
    old_status = ChunkStatus.PENDING
    new_status = {
        ChunkFeedback.PROFOUND: ChunkStatus.PROFOUND,
        ChunkFeedback.INTERESTING: ChunkStatus.INTERESTING,
        ChunkFeedback.REJECTED: ChunkStatus.REJECTED,
    }[feedback_enum]
    
    chunk.move_to_status(DATA_DIR, new_status)
    chunk.save(DATA_DIR)
    
    # If profound, also add to blessed insights
    if feedback_enum == ChunkFeedback.PROFOUND:
        axioms = AxiomStore(DATA_DIR)
        insight = BlessedInsight.from_chunk(chunk)
        axioms.add_blessed_insight(insight)
    
    return jsonify({"success": True, "new_status": new_status.value})


@app.route("/api/stats")
def get_stats():
    """Get exploration statistics."""
    stats = {
        "pending": len(list((DATA_DIR / "chunks" / "pending").glob("*.json"))) if (DATA_DIR / "chunks" / "pending").exists() else 0,
        "profound": len(list((DATA_DIR / "chunks" / "profound").glob("*.json"))) if (DATA_DIR / "chunks" / "profound").exists() else 0,
        "interesting": len(list((DATA_DIR / "chunks" / "interesting").glob("*.json"))) if (DATA_DIR / "chunks" / "interesting").exists() else 0,
        "rejected": len(list((DATA_DIR / "chunks" / "rejected").glob("*.json"))) if (DATA_DIR / "chunks" / "rejected").exists() else 0,
    }
    return jsonify(stats)


def start_server():
    """Start the review server."""
    host = CONFIG["review_server"]["host"]
    port = CONFIG["review_server"]["port"]
    
    print(f"\n  Fano Explorer Review Server")
    print(f"  ===========================")
    print(f"  Running at: http://{host}:{port}")
    print(f"  Press Ctrl+C to stop\n")
    
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    start_server()
