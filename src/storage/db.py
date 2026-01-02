"""
Storage module - Database operations for Fano Explorer.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional


class Database:
    """SQLite database for tracking exploration state."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    @contextmanager
    def _get_conn(self):
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def _init_db(self):
        """Initialize database schema."""
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS threads (
                    id TEXT PRIMARY KEY,
                    topic TEXT,
                    status TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    exchange_count INTEGER DEFAULT 0
                );
                
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    thread_id TEXT,
                    title TEXT,
                    status TEXT,
                    created_at TEXT,
                    reviewed_at TEXT,
                    feedback TEXT,
                    profundity_score REAL
                );
                
                CREATE TABLE IF NOT EXISTS rate_limits (
                    model TEXT PRIMARY KEY,
                    limited INTEGER,
                    retry_at TEXT
                );
                
                CREATE TABLE IF NOT EXISTS stats (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_threads_status ON threads(status);
                CREATE INDEX IF NOT EXISTS idx_chunks_status ON chunks(status);
            """)
    
    def get_active_threads(self) -> list[dict]:
        """Get all active threads."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM threads WHERE status = 'active' ORDER BY updated_at DESC"
            ).fetchall()
            return [dict(row) for row in rows]
    
    def update_thread(self, thread_id: str, **kwargs):
        """Update thread metadata."""
        with self._get_conn() as conn:
            sets = ", ".join(f"{k} = ?" for k in kwargs.keys())
            values = list(kwargs.values()) + [thread_id]
            conn.execute(
                f"INSERT OR REPLACE INTO threads (id, {', '.join(kwargs.keys())}) VALUES (?, {', '.join('?' * len(kwargs))})",
                [thread_id] + list(kwargs.values())
            )
    
    def get_pending_chunks(self) -> list[dict]:
        """Get chunks pending review."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE status = 'pending' ORDER BY created_at DESC"
            ).fetchall()
            return [dict(row) for row in rows]
    
    def update_chunk_feedback(self, chunk_id: str, feedback: str, notes: str = ""):
        """Update chunk with feedback."""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE chunks SET feedback = ?, status = ?, reviewed_at = ? WHERE id = ?",
                (feedback, feedback, datetime.now().isoformat(), chunk_id)
            )
    
    def get_stats(self) -> dict:
        """Get exploration statistics."""
        with self._get_conn() as conn:
            stats = {}
            
            # Thread counts by status
            rows = conn.execute(
                "SELECT status, COUNT(*) as count FROM threads GROUP BY status"
            ).fetchall()
            stats["threads"] = {row["status"]: row["count"] for row in rows}
            
            # Chunk counts by status
            rows = conn.execute(
                "SELECT status, COUNT(*) as count FROM chunks GROUP BY status"
            ).fetchall()
            stats["chunks"] = {row["status"]: row["count"] for row in rows}
            
            return stats
    
    def record_stat(self, key: str, value):
        """Record a statistic."""
        with self._get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO stats (key, value, updated_at) VALUES (?, ?, ?)",
                (key, json.dumps(value), datetime.now().isoformat())
            )
    
    def get_stat(self, key: str) -> Optional[any]:
        """Get a recorded statistic."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT value FROM stats WHERE key = ?", (key,)
            ).fetchone()
            if row:
                return json.loads(row["value"])
            return None
