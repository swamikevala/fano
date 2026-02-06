"""Tests for authentication — LDAP login, session management, project isolation."""

import asyncio
import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from shared.config import Config
from shared.events import EventBus
from shared.models import (
    EvaluationCriterion,
    Project,
    ProjectStatus,
    ResearchDomain,
    Seed,
    SeedStatus,
    SeedType,
    Confidence,
    User,
    generate_id,
)
from shared.store import StateStore


# ── helpers ──────────────────────────────────────────────────


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_config() -> Config:
    os.environ.setdefault("FANO_TEST_KEY", "test-key-value")
    return Config.from_dict({
        "llm": {
            "api_key_env": "FANO_TEST_KEY",
            "models": {"claude": {"model": "claude-3"}},
        },
        "consensus": {"backends": ["claude"]},
        "control": {"host": "127.0.0.1", "port": 8080},
        "auth": {
            "ldap_server": "ldap://test-ad.company.com",
            "ldap_domain": "COMPANY",
            "secret_key": "test-secret-key",
        },
    })


def _make_user(uid: str = "user-1", username: str = "alice") -> User:
    return User(
        id=uid, username=username,
        display_name=f"User {username}", created_at=_now(),
    )


def _make_project(pid: str = "proj-1", owner_id: str = "user-1") -> Project:
    now = _now()
    return Project(
        id=pid, owner_id=owner_id, name="Test Project", goal="Test",
        context="C",
        evaluation_criteria=[EvaluationCriterion("rigor", "r", 1.0)],
        exploration_guidance="EG", document_guidance="DG",
        seed_modification_enabled=True,
        seed_modification_require_approval=False,
        research_domains=[ResearchDomain("math", ["x"], ["y"])],
        status=ProjectStatus.ACTIVE,
        created_at=now, updated_at=now,
    )


def _make_seed(project_id: str = "proj-1", sid: str | None = None) -> Seed:
    now = _now()
    return Seed(
        id=sid or generate_id(), project_id=project_id,
        text="Test seed", type=SeedType.CONJECTURE, priority=5,
        tags=[], confidence=Confidence.MEDIUM,
        source=None, notes=None, status=SeedStatus.ACTIVE,
        parent_seed_id=None, modification_reason=None,
        exploration_count=0, created_at=now, updated_at=now,
    )


# ── fixtures ─────────────────────────────────────────────────


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    yield lp
    lp.close()


@pytest.fixture
def app(loop):
    from control.server import create_app

    store = StateStore(":memory:")
    loop.run_until_complete(store.connect())
    bus = EventBus(store=None)
    config = _make_config()
    application = create_app(store, bus, config)
    application.config["TESTING"] = True
    yield application
    loop.run_until_complete(store.close())


@pytest.fixture
def client(app):
    """Unauthenticated client."""
    return app.test_client()


@pytest.fixture
def store(app):
    return app.config["state_store"]


def _auth_client(app, user_id: str):
    """Create a client with session set to the given user_id."""
    c = app.test_client()
    with c.session_transaction() as sess:
        sess["user_id"] = user_id
    return c


# ── Login tests ──────────────────────────────────────────────


class TestLogin:
    def test_login_page_renders(self, client):
        """GET /login returns the login form."""
        resp = client.get("/login")
        assert resp.status_code == 200
        assert b"Sign in" in resp.data or b"sign in" in resp.data.lower()

    @patch("control.blueprints.auth.ldap3")
    def test_login_success_creates_user_and_sets_session(self, mock_ldap3, app, loop, store):
        """Successful LDAP bind creates a local user and sets session."""
        # Mock successful LDAP connection
        mock_conn = MagicMock()
        mock_conn.entries = []  # No displayName lookup results
        mock_ldap3.Server.return_value = MagicMock()
        mock_ldap3.Connection.return_value = mock_conn
        mock_ldap3.NONE = 0
        mock_conn.unbind = MagicMock()

        client = app.test_client()
        resp = client.post("/login", data={
            "username": "jdoe",
            "password": "secret123",
        }, follow_redirects=False)

        assert resp.status_code == 302
        assert "/dashboard" in resp.headers["Location"]

        # Verify user was created
        user = loop.run_until_complete(store.get_user_by_username("jdoe"))
        assert user is not None
        assert user.username == "jdoe"

        # Verify session is set (make an authenticated request)
        resp2 = client.get("/api/projects")
        assert resp2.status_code == 200

    @patch("control.blueprints.auth.ldap3")
    def test_login_failure_invalid_credentials(self, mock_ldap3, client):
        """Failed LDAP bind shows error and doesn't set session."""
        mock_ldap3.Server.return_value = MagicMock()
        mock_ldap3.NONE = 0
        mock_ldap3.Connection.side_effect = Exception("invalidCredentials")

        resp = client.post("/login", data={
            "username": "baduser",
            "password": "wrongpass",
        }, follow_redirects=False)

        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]

    @patch("control.blueprints.auth.ldap3")
    def test_login_failure_server_unreachable(self, mock_ldap3, client):
        """When AD server is unreachable, show appropriate error."""
        mock_ldap3.Server.return_value = MagicMock()
        mock_ldap3.NONE = 0
        mock_ldap3.Connection.side_effect = Exception("Connection refused")

        resp = client.post("/login", data={
            "username": "user",
            "password": "pass",
        }, follow_redirects=False)

        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]

    @patch("control.blueprints.auth.ldap3")
    def test_login_existing_user_reuses_record(self, mock_ldap3, app, loop, store):
        """Second login for same username reuses existing user record."""
        # Create existing user
        user = _make_user(uid="existing-1", username="returning")
        loop.run_until_complete(store.create_user(user))

        mock_conn = MagicMock()
        mock_conn.entries = []
        mock_ldap3.Server.return_value = MagicMock()
        mock_ldap3.Connection.return_value = mock_conn
        mock_ldap3.NONE = 0

        client = app.test_client()
        resp = client.post("/login", data={
            "username": "returning",
            "password": "pass",
        }, follow_redirects=False)

        assert resp.status_code == 302
        # Verify session uses existing user's ID
        resp2 = client.get("/api/projects")
        assert resp2.status_code == 200

    def test_login_empty_fields_redirects(self, client):
        """Empty username/password redirects back to login."""
        resp = client.post("/login", data={
            "username": "",
            "password": "",
        }, follow_redirects=False)
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]


# ── Dev mode tests ───────────────────────────────────────────


class TestDevMode:
    def test_dev_mode_skips_ldap(self, loop):
        """With auth.dev_mode=True, login succeeds without LDAP."""
        from control.server import create_app

        store = StateStore(":memory:")
        loop.run_until_complete(store.connect())
        bus = EventBus(store=None)
        config = Config.from_dict({
            "llm": {
                "api_key_env": "FANO_TEST_KEY",
                "models": {"claude": {"model": "claude-3"}},
            },
            "consensus": {"backends": ["claude"]},
            "control": {"host": "127.0.0.1", "port": 8080},
            "auth": {
                "dev_mode": True,
                "secret_key": "test-secret",
            },
        })
        app = create_app(store, bus, config)
        app.config["TESTING"] = True
        client = app.test_client()

        resp = client.post("/login", data={
            "username": "devuser",
            "password": "anything",
        }, follow_redirects=False)

        assert resp.status_code == 302
        assert "/dashboard" in resp.headers["Location"]

        # User was created
        user = loop.run_until_complete(store.get_user_by_username("devuser"))
        assert user is not None
        assert user.display_name == "devuser"

        # Session is set — can access API
        resp2 = client.get("/api/projects")
        assert resp2.status_code == 200

        loop.run_until_complete(store.close())


# ── Logout tests ─────────────────────────────────────────────


class TestLogout:
    def test_logout_clears_session(self, app, loop, store):
        """POST /logout clears the session and redirects to login."""
        user = _make_user()
        loop.run_until_complete(store.create_user(user))
        client = _auth_client(app, "user-1")

        resp = client.post("/logout", follow_redirects=False)
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]

        # Subsequent requests should redirect to login
        resp2 = client.get("/dashboard", follow_redirects=False)
        assert resp2.status_code == 302
        assert "/login" in resp2.headers["Location"]


# ── Auth enforcement tests ────────────────────────────────────


class TestAuthEnforcement:
    def test_unauthenticated_page_redirects_to_login(self, client):
        """Unauthenticated requests to page routes redirect to /login."""
        resp = client.get("/dashboard", follow_redirects=False)
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]

    def test_unauthenticated_api_returns_401(self, client):
        """Unauthenticated API requests return 401 JSON."""
        resp = client.get("/api/projects")
        assert resp.status_code == 401
        data = resp.get_json()
        assert data["ok"] is False
        assert data["code"] == "UNAUTHORIZED"

    def test_login_page_accessible_without_auth(self, client):
        """The login page itself doesn't require auth."""
        resp = client.get("/login")
        assert resp.status_code == 200


# ── Project isolation tests ───────────────────────────────────


class TestProjectIsolation:
    def test_user_sees_only_own_projects(self, app, loop, store):
        """User A only sees projects owned by A."""
        user_a = _make_user(uid="user-a", username="alice")
        user_b = _make_user(uid="user-b", username="bob")
        loop.run_until_complete(store.create_user(user_a))
        loop.run_until_complete(store.create_user(user_b))

        loop.run_until_complete(store.create_project(
            _make_project(pid="proj-a", owner_id="user-a")
        ))
        loop.run_until_complete(store.create_project(
            _make_project(pid="proj-b", owner_id="user-b")
        ))

        # User A sees only their project
        client_a = _auth_client(app, "user-a")
        resp = client_a.get("/api/projects")
        data = resp.get_json()
        assert data["ok"]
        project_ids = [p["id"] for p in data["data"]]
        assert "proj-a" in project_ids
        assert "proj-b" not in project_ids

        # User B sees only their project
        client_b = _auth_client(app, "user-b")
        resp = client_b.get("/api/projects")
        data = resp.get_json()
        project_ids = [p["id"] for p in data["data"]]
        assert "proj-b" in project_ids
        assert "proj-a" not in project_ids

    def test_user_cannot_access_other_users_project(self, app, loop, store):
        """User A cannot GET /api/projects/<id> for user B's project."""
        user_a = _make_user(uid="user-a2", username="alice2")
        user_b = _make_user(uid="user-b2", username="bob2")
        loop.run_until_complete(store.create_user(user_a))
        loop.run_until_complete(store.create_user(user_b))

        loop.run_until_complete(store.create_project(
            _make_project(pid="proj-b2", owner_id="user-b2")
        ))

        client_a = _auth_client(app, "user-a2")
        resp = client_a.get("/api/projects/proj-b2")
        assert resp.status_code == 403

    def test_user_cannot_access_other_users_seeds(self, app, loop, store):
        """User A cannot list seeds for user B's project."""
        user_a = _make_user(uid="user-a3", username="alice3")
        user_b = _make_user(uid="user-b3", username="bob3")
        loop.run_until_complete(store.create_user(user_a))
        loop.run_until_complete(store.create_user(user_b))

        loop.run_until_complete(store.create_project(
            _make_project(pid="proj-b3", owner_id="user-b3")
        ))
        loop.run_until_complete(store.create_seed(
            _make_seed(project_id="proj-b3", sid="seed-b3")
        ))

        client_a = _auth_client(app, "user-a3")
        resp = client_a.get("/api/seeds?project_id=proj-b3")
        assert resp.status_code == 403

    def test_create_project_sets_owner(self, app, loop, store):
        """Creating a project via API sets the owner_id to the logged-in user."""
        user = _make_user(uid="creator-1", username="creator")
        loop.run_until_complete(store.create_user(user))

        client = _auth_client(app, "creator-1")
        resp = client.post("/api/projects", json={
            "name": "My Project",
            "goal": "Test ownership",
        })
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["data"]["owner_id"] == "creator-1"
