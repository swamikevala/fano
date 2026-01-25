"""
Status display data gathering for Explorer.

Collects and returns status information about the explorer state
without handling presentation (that's left to the CLI).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from shared.logging import get_logger

log = get_logger("explorer", "status_display")


@dataclass
class ThreadInfo:
    """Information about an exploration thread."""
    topic: str
    exchange_count: int


@dataclass
class RateLimitInfo:
    """Rate limit status for a model."""
    model: str
    limited: bool
    retry_at: Optional[str] = None


@dataclass
class ExplorerStatus:
    """Complete status of the explorer."""
    active_threads: list[ThreadInfo] = field(default_factory=list)
    pending_chunks: int = 0
    profound_count: int = 0
    interesting_count: int = 0
    rejected_count: int = 0
    rate_limits: list[RateLimitInfo] = field(default_factory=list)


def get_status(data_dir: Path) -> ExplorerStatus:
    """
    Gather current explorer status.

    Args:
        data_dir: Path to the explorer data directory

    Returns:
        ExplorerStatus with all current information
    """
    from explorer.src.storage.db import Database

    status = ExplorerStatus()

    # Get active threads from database
    try:
        db = Database(data_dir / "fano_explorer.db")
        threads = db.get_active_threads()
        status.active_threads = [
            ThreadInfo(topic=t.topic, exchange_count=t.exchange_count)
            for t in threads
        ]
    except Exception as e:
        log.warning("status.db_error", error=str(e))

    # Count pending chunks
    pending_dir = data_dir / "chunks" / "pending"
    if pending_dir.exists():
        status.pending_chunks = len(list(pending_dir.glob("*.md")))

    # Count chunks by category
    chunks_dir = data_dir / "chunks"

    profound_dir = chunks_dir / "profound"
    if profound_dir.exists():
        status.profound_count = len(list(profound_dir.glob("*.md")))

    interesting_dir = chunks_dir / "interesting"
    if interesting_dir.exists():
        status.interesting_count = len(list(interesting_dir.glob("*.md")))

    rejected_dir = chunks_dir / "rejected"
    if rejected_dir.exists():
        status.rejected_count = len(list(rejected_dir.glob("*.md")))

    # Note: Rate limits are now managed by the LLMClient via OpenRouter API
    # No local rate limit tracking needed

    log.info(
        "status.gathered",
        active_threads=len(status.active_threads),
        pending=status.pending_chunks,
        profound=status.profound_count,
        interesting=status.interesting_count,
        rejected=status.rejected_count,
    )

    return status
