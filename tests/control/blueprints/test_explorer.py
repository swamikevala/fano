"""
Tests for the explorer blueprint API endpoints.

Tests cover:
- GET /api/explorer/stats - Get explorer statistics
- GET /api/explorer/pipeline - Get pipeline status
- GET /api/explorer/threads - Get exploration threads
- GET /api/explorer/insights/<status> - Get insights by status
- POST /api/explorer/feedback - Submit insight feedback
- Seeds API (CRUD operations)
- Thread safety for AxiomStore initialization
"""

import json
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class TestExplorerStats:
    """Tests for GET /api/explorer/stats."""

    def test_stats_returns_structure(self, flask_client, temp_dir):
        """Stats endpoint returns expected structure."""
        with patch("control.services.EXPLORER_DATA_DIR", temp_dir):
            with patch("control.blueprints.explorer.EXPLORER_DATA_DIR", temp_dir):
                response = flask_client.get("/api/explorer/stats")

        assert response.status_code == 200
        data = response.get_json()
        assert "stats" in data or "pending" in data or "total" in data


class TestExplorerPipeline:
    """Tests for GET /api/explorer/pipeline."""

    def test_pipeline_returns_counts(self, flask_client, temp_dir):
        """Pipeline endpoint returns insight and thread counts."""
        # Create directory structure
        (temp_dir / "explorations").mkdir(parents=True, exist_ok=True)
        (temp_dir / "chunks" / "insights" / "pending").mkdir(parents=True, exist_ok=True)
        (temp_dir / "axioms").mkdir(parents=True, exist_ok=True)
        (temp_dir / "axioms" / "axioms.json").write_text('{"seeds": []}')

        with patch("control.services.EXPLORER_DATA_DIR", temp_dir):
            with patch("control.blueprints.explorer.EXPLORER_DATA_DIR", temp_dir):
                response = flask_client.get("/api/explorer/pipeline")

        assert response.status_code == 200
        data = response.get_json()
        assert "threads" in data or "pipeline" in data


class TestExplorerThreads:
    """Tests for thread listing endpoints."""

    def test_active_threads_returns_list(self, flask_client, temp_dir, sample_thread_files):
        """Active threads endpoint returns thread list."""
        with patch("control.services.EXPLORER_DATA_DIR", sample_thread_files):
            with patch("control.blueprints.explorer.EXPLORER_DATA_DIR", sample_thread_files):
                response = flask_client.get("/api/explorer/threads/active")

        assert response.status_code == 200
        data = response.get_json()
        assert "threads" in data
        assert isinstance(data["threads"], list)

    def test_all_threads_returns_list(self, flask_client, temp_dir, sample_thread_files):
        """All threads endpoint returns all threads."""
        with patch("control.services.EXPLORER_DATA_DIR", sample_thread_files):
            with patch("control.blueprints.explorer.EXPLORER_DATA_DIR", sample_thread_files):
                response = flask_client.get("/api/explorer/threads")

        assert response.status_code == 200
        data = response.get_json()
        assert "threads" in data

    def test_thread_detail(self, flask_client, temp_dir, sample_thread_files):
        """Thread detail endpoint returns full thread data."""
        # Create axioms.json for AxiomStore
        (sample_thread_files / "axioms").mkdir(parents=True, exist_ok=True)
        (sample_thread_files / "axioms" / "axioms.json").write_text('{"seeds": []}')

        with patch("control.services.EXPLORER_DATA_DIR", sample_thread_files):
            with patch("control.blueprints.explorer.EXPLORER_DATA_DIR", sample_thread_files):
                # Reset the global axiom store for this test
                import control.blueprints.explorer as explorer_bp
                explorer_bp._axiom_store = None

                response = flask_client.get("/api/explorer/threads/thread-001")

        assert response.status_code == 200
        data = response.get_json()
        assert "thread" in data
        assert "exchanges" in data


class TestExplorerInsights:
    """Tests for insight listing and management."""

    def test_insights_by_status(self, flask_client, sample_insight_files):
        """Insights endpoint returns insights filtered by status."""
        with patch("control.services.EXPLORER_DATA_DIR", sample_insight_files):
            with patch("control.blueprints.explorer.EXPLORER_DATA_DIR", sample_insight_files):
                response = flask_client.get("/api/explorer/insights/pending")

        assert response.status_code == 200
        data = response.get_json()
        assert "insights" in data
        assert data["status"] == "pending"

    def test_insights_invalid_status(self, flask_client):
        """Insights endpoint returns 400 for invalid status."""
        response = flask_client.get("/api/explorer/insights/invalid_status")

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_single_insight(self, flask_client, sample_insight_files):
        """Single insight endpoint returns insight data."""
        # Also patch the get_insight_by_id function
        from control.services import get_insight_by_id

        with patch("control.services.EXPLORER_DATA_DIR", sample_insight_files):
            with patch("control.blueprints.explorer.EXPLORER_DATA_DIR", sample_insight_files):
                with patch("control.blueprints.explorer.get_insight_by_id") as mock_get:
                    mock_get.return_value = ({"id": "insight-001", "insight": "Test"}, "pending")
                    response = flask_client.get("/api/explorer/insight/insight-001")

        assert response.status_code == 200
        data = response.get_json()
        assert "insight" in data

    def test_insight_not_found(self, flask_client, sample_insight_files):
        """Single insight endpoint returns 404 for missing insight."""
        with patch("control.services.EXPLORER_DATA_DIR", sample_insight_files):
            with patch("control.blueprints.explorer.EXPLORER_DATA_DIR", sample_insight_files):
                response = flask_client.get("/api/explorer/insight/nonexistent")

        assert response.status_code == 404


class TestInsightFeedback:
    """Tests for insight feedback submission."""

    def test_feedback_bless(self, flask_client, temp_dir):
        """Feedback endpoint blesses an insight."""
        # Create the directory structure
        pending_dir = temp_dir / "chunks" / "insights" / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)
        blessed_dir = temp_dir / "chunks" / "insights" / "blessed"
        blessed_dir.mkdir(parents=True, exist_ok=True)

        with patch("control.services.EXPLORER_DATA_DIR", temp_dir):
            with patch("control.blueprints.explorer.EXPLORER_DATA_DIR", temp_dir):
                with patch("control.blueprints.explorer.get_insight_by_id") as mock_get:
                    mock_get.return_value = ({"id": "insight-001", "insight": "Test"}, "pending")
                    response = flask_client.post(
                        "/api/explorer/feedback",
                        json={"insight_id": "insight-001", "feedback": "bless"},
                        content_type="application/json"
                    )

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert data["new_status"] == "blessed"

    def test_feedback_reject(self, flask_client, temp_dir):
        """Feedback endpoint rejects an insight."""
        # Create the directory structure
        pending_dir = temp_dir / "chunks" / "insights" / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)
        rejected_dir = temp_dir / "chunks" / "insights" / "rejected"
        rejected_dir.mkdir(parents=True, exist_ok=True)

        with patch("control.services.EXPLORER_DATA_DIR", temp_dir):
            with patch("control.blueprints.explorer.EXPLORER_DATA_DIR", temp_dir):
                with patch("control.blueprints.explorer.get_insight_by_id") as mock_get:
                    mock_get.return_value = ({"id": "insight-001", "insight": "Test"}, "pending")
                    response = flask_client.post(
                        "/api/explorer/feedback",
                        json={"insight_id": "insight-001", "feedback": "reject"},
                        content_type="application/json"
                    )

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert data["new_status"] == "rejected"

    def test_feedback_interesting(self, flask_client, temp_dir):
        """Feedback endpoint marks an insight as interesting."""
        # Create the directory structure
        pending_dir = temp_dir / "chunks" / "insights" / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)
        interesting_dir = temp_dir / "chunks" / "insights" / "interesting"
        interesting_dir.mkdir(parents=True, exist_ok=True)

        with patch("control.services.EXPLORER_DATA_DIR", temp_dir):
            with patch("control.blueprints.explorer.EXPLORER_DATA_DIR", temp_dir):
                with patch("control.blueprints.explorer.get_insight_by_id") as mock_get:
                    mock_get.return_value = ({"id": "insight-001", "insight": "Test"}, "pending")
                    response = flask_client.post(
                        "/api/explorer/feedback",
                        json={"insight_id": "insight-001", "feedback": "interesting"},
                        content_type="application/json"
                    )

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert data["new_status"] == "interesting"

    def test_feedback_missing_fields(self, flask_client):
        """Feedback endpoint returns 400 for missing fields."""
        response = flask_client.post(
            "/api/explorer/feedback",
            json={"insight_id": "test"},
            content_type="application/json"
        )

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_feedback_invalid_value(self, flask_client):
        """Feedback endpoint returns 400 for invalid feedback value."""
        response = flask_client.post(
            "/api/explorer/feedback",
            json={"insight_id": "test", "feedback": "invalid"},
            content_type="application/json"
        )

        assert response.status_code == 400


class TestInsightPriority:
    """Tests for insight priority updates."""

    def test_update_priority(self, flask_client, temp_dir):
        """Priority endpoint updates insight priority."""
        # Create the directory structure
        pending_dir = temp_dir / "chunks" / "insights" / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)
        (pending_dir / "insight-001.json").write_text(json.dumps({
            "id": "insight-001",
            "insight": "Test insight",
            "priority": 5,
        }))

        with patch("control.services.EXPLORER_DATA_DIR", temp_dir):
            with patch("control.blueprints.explorer.EXPLORER_DATA_DIR", temp_dir):
                with patch("control.blueprints.explorer.get_insight_by_id") as mock_get:
                    mock_get.return_value = ({"id": "insight-001", "insight": "Test", "priority": 5}, "pending")
                    response = flask_client.post(
                        "/api/explorer/priority",
                        json={"insight_id": "insight-001", "priority": 8},
                        content_type="application/json"
                    )

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert data["priority"] == 8

    def test_update_priority_out_of_range(self, flask_client, temp_dir):
        """Priority endpoint rejects out-of-range values."""
        with patch("control.services.EXPLORER_DATA_DIR", temp_dir):
            with patch("control.blueprints.explorer.EXPLORER_DATA_DIR", temp_dir):
                response = flask_client.post(
                    "/api/explorer/priority",
                    json={"insight_id": "insight-001", "priority": 15},
                    content_type="application/json"
                )

        assert response.status_code == 400


class TestSeedsAPI:
    """Tests for seeds/axioms CRUD operations."""

    def test_get_seeds(self, flask_client, sample_seed_files):
        """GET /api/explorer/seeds returns seed list."""
        with patch("control.services.EXPLORER_DATA_DIR", sample_seed_files):
            with patch("control.blueprints.explorer.EXPLORER_DATA_DIR", sample_seed_files):
                # Reset global axiom store
                import control.blueprints.explorer as explorer_bp
                explorer_bp._axiom_store = None

                response = flask_client.get("/api/explorer/seeds")

        assert response.status_code == 200
        data = response.get_json()
        assert "seeds" in data
        assert "counts" in data

    def test_create_seed(self, flask_client, sample_seed_files):
        """POST /api/explorer/seed creates a new seed."""
        with patch("control.services.EXPLORER_DATA_DIR", sample_seed_files):
            with patch("control.blueprints.explorer.EXPLORER_DATA_DIR", sample_seed_files):
                # Reset global axiom store
                import control.blueprints.explorer as explorer_bp
                explorer_bp._axiom_store = None

                response = flask_client.post(
                    "/api/explorer/seed",
                    json={
                        "text": "New test seed",
                        "type": "conjecture",
                        "priority": 7,
                    },
                    content_type="application/json"
                )

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert "seed" in data
        assert data["seed"]["text"] == "New test seed"

    def test_create_seed_missing_text(self, flask_client, sample_seed_files):
        """POST /api/explorer/seed returns 400 for missing text."""
        with patch("control.services.EXPLORER_DATA_DIR", sample_seed_files):
            with patch("control.blueprints.explorer.EXPLORER_DATA_DIR", sample_seed_files):
                import control.blueprints.explorer as explorer_bp
                explorer_bp._axiom_store = None

                response = flask_client.post(
                    "/api/explorer/seed",
                    json={"type": "axiom"},
                    content_type="application/json"
                )

        assert response.status_code == 400

    def test_update_seed(self, flask_client):
        """PUT /api/explorer/seed/<id> updates seed."""
        mock_store = MagicMock()
        mock_seed = MagicMock()
        mock_seed.text = "Original text"
        mock_store.get_seed_by_id.return_value = mock_seed
        mock_store.update_seed.return_value = True

        from dataclasses import asdict
        updated_seed_dict = {
            "id": "seed-001",
            "text": "Updated text",
            "type": "axiom",
            "priority": 9,
            "tags": [],
            "confidence": "high",
            "source": "user",
            "notes": "",
            "images": [],
        }

        with patch("control.blueprints.explorer.get_axiom_store", return_value=mock_store):
            with patch("control.blueprints.explorer.asdict", return_value=updated_seed_dict):
                response = flask_client.put(
                    "/api/explorer/seed/seed-001",
                    json={"text": "Updated text", "priority": 9},
                    content_type="application/json"
                )

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True

    def test_delete_seed(self, flask_client):
        """DELETE /api/explorer/seed/<id> deletes seed."""
        mock_store = MagicMock()
        mock_store.delete_seed.return_value = True

        with patch("control.blueprints.explorer.get_axiom_store", return_value=mock_store):
            response = flask_client.delete("/api/explorer/seed/seed-001")

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True

    def test_update_seed_priority(self, flask_client):
        """POST /api/explorer/seed/<id>/priority updates seed priority."""
        mock_store = MagicMock()
        mock_store.update_seed.return_value = True

        with patch("control.blueprints.explorer.get_axiom_store", return_value=mock_store):
            response = flask_client.post(
                "/api/explorer/seed/seed-001/priority",
                json={"priority": 10},
                content_type="application/json"
            )

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert data["priority"] == 10


class TestAxiomStoreThreadSafety:
    """Tests for thread safety of AxiomStore initialization."""

    def test_concurrent_get_axiom_store_returns_same_instance(self, temp_dir):
        """Multiple threads get the same AxiomStore instance."""
        # Create axioms directory and file
        axioms_dir = temp_dir / "axioms"
        axioms_dir.mkdir(parents=True, exist_ok=True)
        (axioms_dir / "axioms.json").write_text('{"seeds": []}')

        results = []
        errors = []

        def get_store():
            try:
                from control.blueprints.explorer import get_axiom_store, _axiom_store_lock
                with patch("control.blueprints.explorer.EXPLORER_DATA_DIR", temp_dir):
                    store = get_axiom_store()
                    results.append(id(store))
            except Exception as e:
                errors.append(e)

        # Reset global state
        import control.blueprints.explorer as explorer_bp
        explorer_bp._axiom_store = None

        # Start multiple threads simultaneously
        threads = [threading.Thread(target=get_store) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # All threads should get the same instance
        assert len(set(results)) == 1

    def test_axiom_store_double_check_locking(self):
        """AxiomStore uses proper double-check locking pattern."""
        from control.blueprints.explorer import get_axiom_store

        # Verify the implementation has proper locking
        import inspect
        source = inspect.getsource(get_axiom_store)

        # Should have lock acquisition
        assert "_axiom_store_lock" in source
        # Should have double-check pattern (check before and after lock)
        assert source.count("_axiom_store is None") >= 2


class TestValidStatusValues:
    """Tests for valid status values in API."""

    def test_all_valid_statuses_accepted(self, flask_client, temp_dir):
        """All valid status values are accepted."""
        valid_statuses = ["pending", "blessed", "interesting", "rejected", "reviewing", "disputed"]

        for status in valid_statuses:
            with patch("control.services.EXPLORER_DATA_DIR", temp_dir):
                with patch("control.blueprints.explorer.EXPLORER_DATA_DIR", temp_dir):
                    response = flask_client.get(f"/api/explorer/insights/{status}")

            assert response.status_code == 200, f"Status {status} should be valid"
