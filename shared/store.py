"""StateStore — SQLite-backed persistence for all system state.

Implements StateStoreInterface with aiosqlite, WAL mode, and JSON serialization.
Schema defined in store_schema.py, hydration in store_hydrate.py.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

import aiosqlite

from shared.errors import StoreError
from shared.models import (
    Annotation,
    AnnotationStatus,
    Concept,
    Event,
    Exchange,
    Finding,
    Insight,
    InsightComment,
    InsightStatus,
    Project,
    ProjectStatus,
    Section,
    Seed,
    SeedModification,
    SeedStatus,
    Source,
    StateStoreInterface,
    Thread,
    ThreadStatus,
    User,
)
from shared.store_hydrate import (
    _json_load,
    hydrate_annotation,
    hydrate_concept,
    hydrate_event,
    hydrate_exchange,
    hydrate_finding,
    hydrate_insight,
    hydrate_insight_comment,
    hydrate_project,
    hydrate_section,
    hydrate_seed,
    hydrate_seed_modification,
    hydrate_source,
    hydrate_thread,
    hydrate_user,
)
from shared.store_schema import DT_COLS, JSON_COLS, SCHEMA, TABLES_WITH_UPDATED_AT


# ── Serialization helpers ────────────────────────────────────

def _to_iso(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.isoformat()
    return str(v)


def _json_dump(v: Any) -> str:
    if isinstance(v, str):
        return v
    return json.dumps(v, default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o))


def _enum_val(v: Any) -> Any:
    return v.value if hasattr(v, "value") else v


# ── StateStore ───────────────────────────────────────────────

class StateStore(StateStoreInterface):
    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)
        self._db: aiosqlite.Connection | None = None
        self._in_transaction: bool = False

    async def connect(self) -> None:
        if self._db is None:
            self._db = await aiosqlite.connect(self._db_path)
            self._db.row_factory = aiosqlite.Row
            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._db.execute("PRAGMA foreign_keys=ON")
            await self._db.execute("PRAGMA busy_timeout=5000")
        await self._db.executescript(SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        assert self._db is not None
        await self._db.execute("BEGIN")
        self._in_transaction = True
        try:
            yield
            await self._db.commit()
        except Exception:
            await self._db.rollback()
            raise
        finally:
            self._in_transaction = False

    async def _maybe_commit(self) -> None:
        if not self._in_transaction:
            await self._db.commit()

    # ── generic helpers ──────────────────────────────────────

    async def _insert(self, table: str, data: dict[str, Any]) -> None:
        json_cols = JSON_COLS.get(table, set())
        dt_cols = DT_COLS.get(table, set())
        cols, vals = [], []
        for k, v in data.items():
            cols.append(k)
            if k in json_cols:
                vals.append(_json_dump(v))
            elif k in dt_cols:
                vals.append(_to_iso(v))
            else:
                vals.append(_enum_val(v))
        placeholders = ",".join("?" for _ in cols)
        sql = f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders})"
        try:
            await self._db.execute(sql, vals)
            await self._maybe_commit()
        except StoreError:
            raise
        except Exception as exc:
            raise StoreError(str(exc)) from exc

    async def _get_row(self, table: str, where: str, params: tuple) -> dict | None:
        sql = f"SELECT * FROM {table} WHERE {where}"
        async with self._db.execute(sql, params) as cur:
            row = await cur.fetchone()
        return dict(row) if row else None

    async def _get_rows(self, table: str, where: str = "1=1",
                        params: tuple = (), order: str = "") -> list[dict]:
        sql = f"SELECT * FROM {table} WHERE {where}"
        if order:
            sql += f" ORDER BY {order}"
        async with self._db.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def _update(self, table: str, id_col: str, id_val: str,
                      fields: dict[str, Any]) -> None:
        if not fields:
            return
        json_cols = JSON_COLS.get(table, set())
        dt_cols = DT_COLS.get(table, set())
        sets, vals = [], []
        for k, v in fields.items():
            sets.append(f"{k}=?")
            if k in json_cols:
                vals.append(_json_dump(v))
            elif k in dt_cols:
                vals.append(_to_iso(v))
            else:
                vals.append(_enum_val(v))
        if table in TABLES_WITH_UPDATED_AT and "updated_at" not in fields:
            sets.append("updated_at=?")
            vals.append(datetime.now(timezone.utc).isoformat())
        vals.append(id_val)
        sql = f"UPDATE {table} SET {','.join(sets)} WHERE {id_col}=?"
        try:
            cur = await self._db.execute(sql, vals)
            if cur.rowcount == 0:
                raise StoreError(f"{table} {id_val} not found")
            await self._maybe_commit()
        except StoreError:
            raise
        except Exception as exc:
            raise StoreError(str(exc)) from exc

    # ── Users ───────────────────────────────────────────────

    async def create_user(self, user: User) -> None:
        await self._insert("users", asdict(user))

    async def get_user(self, user_id: str) -> User | None:
        row = await self._get_row("users", "id=?", (user_id,))
        return hydrate_user(row) if row else None

    async def get_user_by_username(self, username: str) -> User | None:
        row = await self._get_row("users", "username=?", (username,))
        return hydrate_user(row) if row else None

    # ── Projects ─────────────────────────────────────────────

    async def create_project(self, project: Project) -> None:
        await self._insert("projects", asdict(project))

    async def get_project(self, project_id: str) -> Project | None:
        row = await self._get_row("projects", "id=?", (project_id,))
        return hydrate_project(row) if row else None

    async def update_project(self, project_id: str, **fields: object) -> None:
        await self._update("projects", "id", project_id, fields)

    async def list_projects(self, status: ProjectStatus | None = None,
                            owner_id: str | None = None) -> list[Project]:
        parts: list[str] = []
        params: list[Any] = []
        if status:
            parts.append("status=?")
            params.append(_enum_val(status))
        if owner_id:
            parts.append("owner_id=?")
            params.append(owner_id)
        where = " AND ".join(parts) if parts else "1=1"
        rows = await self._get_rows("projects", where, tuple(params))
        return [hydrate_project(r) for r in rows]

    # ── Seeds ────────────────────────────────────────────────

    async def create_seed(self, seed: Seed) -> None:
        await self._insert("seeds", asdict(seed))

    async def get_seed(self, seed_id: str) -> Seed | None:
        row = await self._get_row("seeds", "id=?", (seed_id,))
        return hydrate_seed(row) if row else None

    async def update_seed(self, seed_id: str, **fields: object) -> None:
        await self._update("seeds", "id", seed_id, fields)

    async def list_seeds(self, project_id: str, status: SeedStatus | None = None) -> list[Seed]:
        if status:
            rows = await self._get_rows(
                "seeds", "project_id=? AND status=?",
                (project_id, _enum_val(status)),
                order="priority DESC, created_at ASC",
            )
        else:
            rows = await self._get_rows(
                "seeds", "project_id=?", (project_id,),
                order="priority DESC, created_at ASC",
            )
        return [hydrate_seed(r) for r in rows]

    async def get_seed_lineage(self, seed_id: str) -> list[Seed]:
        chain: list[Seed] = []
        current_id: str | None = seed_id
        while current_id:
            seed = await self.get_seed(current_id)
            if seed is None:
                break
            chain.append(seed)
            current_id = seed.parent_seed_id
        return chain

    # ── Threads ──────────────────────────────────────────────

    async def create_thread(self, thread: Thread) -> None:
        await self._insert("threads", asdict(thread))

    async def get_thread(self, thread_id: str) -> Thread | None:
        row = await self._get_row("threads", "id=?", (thread_id,))
        return hydrate_thread(row) if row else None

    async def update_thread(self, thread_id: str, **fields: object) -> None:
        await self._update("threads", "id", thread_id, fields)

    async def list_threads(self, project_id: str, status: ThreadStatus | None = None) -> list[Thread]:
        if status:
            rows = await self._get_rows(
                "threads", "project_id=? AND status=?",
                (project_id, _enum_val(status)),
            )
        else:
            rows = await self._get_rows("threads", "project_id=?", (project_id,))
        return [hydrate_thread(r) for r in rows]

    # ── Exchanges ────────────────────────────────────────────

    async def create_exchange(self, exchange: Exchange) -> None:
        await self._insert("exchanges", asdict(exchange))

    async def get_exchanges(self, thread_id: str) -> list[Exchange]:
        rows = await self._get_rows("exchanges", "thread_id=?", (thread_id,), order="sequence ASC")
        return [hydrate_exchange(r) for r in rows]

    # ── Insights ─────────────────────────────────────────────

    async def create_insight(self, insight: Insight) -> None:
        await self._insert("insights", asdict(insight))

    async def get_insight(self, insight_id: str) -> Insight | None:
        row = await self._get_row("insights", "id=?", (insight_id,))
        return hydrate_insight(row) if row else None

    async def update_insight(self, insight_id: str, **fields: object) -> None:
        await self._update("insights", "id", insight_id, fields)

    async def list_insights(self, project_id: str, status: InsightStatus | None = None) -> list[Insight]:
        if status:
            rows = await self._get_rows(
                "insights", "project_id=? AND status=?",
                (project_id, _enum_val(status)),
            )
        else:
            rows = await self._get_rows("insights", "project_id=?", (project_id,))
        return [hydrate_insight(r) for r in rows]

    async def get_insights_for_thread(self, thread_id: str) -> list[Insight]:
        rows = await self._get_rows("insights", "source_thread_id=?", (thread_id,))
        return [hydrate_insight(r) for r in rows]

    # ── Insight Comments ─────────────────────────────────────

    async def create_insight_comment(self, comment: InsightComment) -> None:
        await self._insert("insight_comments", asdict(comment))

    async def get_insight_comments(self, insight_id: str) -> list[InsightComment]:
        rows = await self._get_rows("insight_comments", "insight_id=?", (insight_id,), order="created_at ASC")
        return [hydrate_insight_comment(r) for r in rows]

    # ── Seed Modifications ───────────────────────────────────

    async def create_seed_modification(self, modification: SeedModification) -> None:
        await self._insert("seed_modifications", asdict(modification))

    async def get_seed_modification(self, modification_id: str) -> SeedModification | None:
        row = await self._get_row("seed_modifications", "id=?", (modification_id,))
        return hydrate_seed_modification(row) if row else None

    async def update_seed_modification(self, modification_id: str, **fields: object) -> None:
        await self._update("seed_modifications", "id", modification_id, fields)

    async def list_pending_modifications(self, project_id: str) -> list[SeedModification]:
        sql = (
            "SELECT sm.* FROM seed_modifications sm "
            "JOIN seeds s ON sm.seed_id = s.id "
            "WHERE s.project_id=? AND sm.status='pending'"
        )
        async with self._db.execute(sql, (project_id,)) as cur:
            rows = await cur.fetchall()
        return [hydrate_seed_modification(dict(r)) for r in rows]

    # ── Sections ─────────────────────────────────────────────

    async def create_section(self, section: Section) -> None:
        await self._insert("sections", asdict(section))

    async def get_section(self, section_id: str) -> Section | None:
        row = await self._get_row("sections", "id=?", (section_id,))
        return hydrate_section(row) if row else None

    async def update_section(self, section_id: str, **fields: object) -> None:
        await self._update("sections", "id", section_id, fields)

    async def list_sections(self, project_id: str) -> list[Section]:
        rows = await self._get_rows("sections", "project_id=?", (project_id,), order="order_index ASC")
        return [hydrate_section(r) for r in rows]

    async def get_recent_sections(self, project_id: str, limit: int = 2) -> list[Section]:
        sql = "SELECT * FROM sections WHERE project_id=? ORDER BY created_at DESC LIMIT ?"
        async with self._db.execute(sql, (project_id, limit)) as cur:
            rows = await cur.fetchall()
        return [hydrate_section(dict(r)) for r in rows]

    async def get_sections_establishing(self, concept_names: list[str]) -> list[Section]:
        rows = await self._get_rows("sections")
        results = []
        for r in rows:
            establishes = _json_load(r["establishes"]) or []
            if any(cn in establishes for cn in concept_names):
                results.append(hydrate_section(r))
        return results

    async def get_sections_by_tags(self, tags: list[str], limit: int = 3) -> list[Section]:
        rows = await self._get_rows("sections")
        results: list[Section] = []
        tag_set = set(tags)
        for r in rows:
            establishes = _json_load(r["establishes"]) or []
            if tag_set & set(establishes):
                results.append(hydrate_section(r))
                if len(results) >= limit:
                    break
        return results

    # ── Concepts ─────────────────────────────────────────────

    async def create_concept(self, concept: Concept) -> None:
        await self._insert("concepts", asdict(concept))

    async def get_concept(self, canonical_name: str, project_id: str) -> Concept | None:
        row = await self._get_row("concepts", "canonical_name=? AND project_id=?", (canonical_name, project_id))
        return hydrate_concept(row) if row else None

    async def list_concepts(self, project_id: str) -> list[Concept]:
        rows = await self._get_rows("concepts", "project_id=?", (project_id,))
        return [hydrate_concept(r) for r in rows]

    # ── Annotations ──────────────────────────────────────────

    async def create_annotation(self, annotation: Annotation) -> None:
        await self._insert("annotations", asdict(annotation))

    async def get_annotation(self, annotation_id: str) -> Annotation | None:
        row = await self._get_row("annotations", "id=?", (annotation_id,))
        return hydrate_annotation(row) if row else None

    async def update_annotation(self, annotation_id: str, **fields: object) -> None:
        await self._update("annotations", "id", annotation_id, fields)

    async def list_annotations(self, project_id: str, status: AnnotationStatus | None = None) -> list[Annotation]:
        if status:
            rows = await self._get_rows("annotations", "project_id=? AND status=?", (project_id, _enum_val(status)))
        else:
            rows = await self._get_rows("annotations", "project_id=?", (project_id,))
        return [hydrate_annotation(r) for r in rows]

    # ── Sources ──────────────────────────────────────────────

    async def create_source(self, source: Source) -> None:
        await self._insert("sources", asdict(source))

    async def get_source(self, source_id: str) -> Source | None:
        row = await self._get_row("sources", "id=?", (source_id,))
        return hydrate_source(row) if row else None

    async def get_source_by_url(self, url: str, project_id: str) -> Source | None:
        row = await self._get_row("sources", "url=? AND project_id=?", (url, project_id))
        return hydrate_source(row) if row else None

    async def list_sources(self, project_id: str) -> list[Source]:
        rows = await self._get_rows("sources", "project_id=?", (project_id,))
        return [hydrate_source(r) for r in rows]

    # ── Findings ─────────────────────────────────────────────

    async def create_finding(self, finding: Finding) -> None:
        await self._insert("findings", asdict(finding))

    async def get_finding(self, finding_id: str) -> Finding | None:
        row = await self._get_row("findings", "id=?", (finding_id,))
        return hydrate_finding(row) if row else None

    async def list_findings(self, project_id: str, related_insight_id: str | None = None) -> list[Finding]:
        if related_insight_id:
            rows = await self._get_rows("findings", "project_id=? AND related_insight_id=?", (project_id, related_insight_id))
        else:
            rows = await self._get_rows("findings", "project_id=?", (project_id,))
        return [hydrate_finding(r) for r in rows]

    # ── Events ───────────────────────────────────────────────

    async def persist_event(self, event: Event) -> None:
        data = {
            "topic": event.topic,
            "timestamp": _to_iso(event.timestamp),
            "source": event.source,
            "payload": _json_dump(event.payload),
            "correlation_id": event.correlation_id,
            "created_at": _to_iso(event.timestamp),
        }
        cols = ",".join(data.keys())
        placeholders = ",".join("?" for _ in data)
        sql = f"INSERT INTO events ({cols}) VALUES ({placeholders})"
        try:
            await self._db.execute(sql, list(data.values()))
            await self._maybe_commit()
        except Exception as exc:
            raise StoreError(str(exc)) from exc

    async def list_events(self, since: datetime | None = None,
                          topic: str | None = None) -> list[Event]:
        parts: list[str] = []
        params: list[Any] = []
        if since:
            parts.append("created_at >= ?")
            params.append(_to_iso(since))
        if topic:
            parts.append("topic = ?")
            params.append(topic)
        where = " AND ".join(parts) if parts else "1=1"
        rows = await self._get_rows("events", where, tuple(params), order="created_at ASC")
        return [hydrate_event(r) for r in rows]

    # ── Metrics ──────────────────────────────────────────────

    async def record_metric(self, project_id: str, name: str,
                            value: float, labels: dict | None = None) -> None:
        data = {
            "project_id": project_id,
            "name": name,
            "value": value,
            "labels": _json_dump(labels or {}),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        cols = ",".join(data.keys())
        placeholders = ",".join("?" for _ in data)
        sql = f"INSERT INTO metrics ({cols}) VALUES ({placeholders})"
        try:
            await self._db.execute(sql, list(data.values()))
            await self._maybe_commit()
        except Exception as exc:
            raise StoreError(str(exc)) from exc

    async def query_metrics(self, name: str,
                            since: datetime | None = None) -> list[dict]:
        where = "name=?"
        params: list[Any] = [name]
        if since:
            where += " AND recorded_at >= ?"
            params.append(_to_iso(since))
        rows = await self._get_rows("metrics", where, tuple(params), order="recorded_at ASC")
        return [
            {
                "name": r["name"],
                "value": r["value"],
                "labels": _json_load(r["labels"]),
                "project_id": r["project_id"],
                "recorded_at": r["recorded_at"],
            }
            for r in rows
        ]
