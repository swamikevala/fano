"""
Async Job Store for the browser pool.

Manages jobs through their lifecycle:
- Submission (with deduplication)
- Processing
- Completion
- Result retrieval

Jobs persist in memory and optionally to disk for crash recovery.
"""

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional, Any

from shared.logging import get_logger

log = get_logger("pool", "jobs")


class JobStatus(str, Enum):
    """Job lifecycle states."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class Job:
    """Represents a single async job."""

    job_id: str
    backend: str
    prompt: str
    thread_id: Optional[str] = None
    task_type: Optional[str] = None  # "exploration" or "critique"

    # Options
    deep_mode: bool = False
    new_chat: bool = True
    priority: str = "normal"
    images: list = field(default_factory=list)  # Image attachments as dicts

    # State
    status: JobStatus = JobStatus.QUEUED
    queue_position: int = 0

    # Result (populated on completion)
    result: Optional[str] = None
    error: Optional[str] = None
    deep_mode_used: bool = False

    # Browser state (for recovery)
    chat_url: Optional[str] = None

    # Timestamps
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    # Content hash for deduplication
    content_hash: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Job":
        """Create from dictionary."""
        data = data.copy()
        data["status"] = JobStatus(data["status"])
        return cls(**data)


class JobStore:
    """
    Thread-safe store for async jobs.

    Features:
    - Job submission with ID-based deduplication
    - Content-hash based caching (optional)
    - Persistent storage for crash recovery
    - Queue position tracking
    """

    def __init__(self, persist_path: Optional[Path] = None, cache_ttl: int = 3600):
        """
        Initialize job store.

        Args:
            persist_path: Path to persist jobs (for crash recovery)
            cache_ttl: Time-to-live for content-hash cache in seconds
        """
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        self._content_cache: dict[str, str] = {}  # hash -> job_id
        self._cache_times: dict[str, float] = {}  # hash -> timestamp
        self._backend_queues: dict[str, list[str]] = {}  # backend -> [job_ids]

        self.persist_path = persist_path
        self.cache_ttl = cache_ttl

        if persist_path:
            self._load_from_disk()

    def _compute_content_hash(self, backend: str, prompt: str) -> str:
        """Compute hash for content-based deduplication."""
        content = f"{backend}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _clean_expired_cache(self):
        """Remove expired entries from content cache."""
        now = time.time()
        expired = [h for h, t in self._cache_times.items() if now - t > self.cache_ttl]
        for h in expired:
            self._content_cache.pop(h, None)
            self._cache_times.pop(h, None)

    def submit(
        self,
        job_id: str,
        backend: str,
        prompt: str,
        thread_id: Optional[str] = None,
        task_type: Optional[str] = None,
        deep_mode: bool = False,
        new_chat: bool = True,
        priority: str = "normal",
        images: list = None,
    ) -> dict:
        """
        Submit a new job.

        Returns:
            {
                "status": "queued" | "exists" | "cached",
                "job_id": str,
                "cached_job_id": str (if cached)
            }
        """
        with self._lock:
            # Check if job_id already exists (exact dedup)
            if job_id in self._jobs:
                log.info("pool.jobs.duplicate", job_id=job_id)
                return {"status": "exists", "job_id": job_id}

            # Check content cache (content-based dedup)
            self._clean_expired_cache()
            content_hash = self._compute_content_hash(backend, prompt)

            if content_hash in self._content_cache:
                cached_job_id = self._content_cache[content_hash]
                cached_job = self._jobs.get(cached_job_id)
                if cached_job and cached_job.status == JobStatus.COMPLETE:
                    log.info("pool.jobs.cache_hit",
                             job_id=job_id,
                             cached_job_id=cached_job_id)
                    return {
                        "status": "cached",
                        "job_id": job_id,
                        "cached_job_id": cached_job_id,
                    }

            # Create new job
            job = Job(
                job_id=job_id,
                backend=backend,
                prompt=prompt,
                thread_id=thread_id,
                task_type=task_type,
                deep_mode=deep_mode,
                new_chat=new_chat,
                priority=priority,
                content_hash=content_hash,
                images=images or [],
            )

            # Add to store
            self._jobs[job_id] = job

            # Add to backend queue
            if backend not in self._backend_queues:
                self._backend_queues[backend] = []
            self._backend_queues[backend].append(job_id)
            job.queue_position = len(self._backend_queues[backend])

            # Add to content cache
            self._content_cache[content_hash] = job_id
            self._cache_times[content_hash] = time.time()

            log.info("pool.jobs.submitted",
                     job_id=job_id,
                     backend=backend,
                     thread_id=thread_id,
                     queue_position=job.queue_position)

            self._persist()

            return {"status": "queued", "job_id": job_id}

    def get_next_job(self, backend: str) -> Optional[Job]:
        """
        Get the next queued job for a backend.

        Marks it as PROCESSING and removes from queue.
        """
        with self._lock:
            queue = self._backend_queues.get(backend, [])

            for job_id in queue:
                job = self._jobs.get(job_id)
                if job and job.status == JobStatus.QUEUED:
                    job.status = JobStatus.PROCESSING
                    job.started_at = time.time()
                    queue.remove(job_id)
                    self._update_queue_positions(backend)
                    self._persist()

                    log.info("pool.jobs.processing",
                             job_id=job_id,
                             backend=backend)
                    return job

            return None

    def _update_queue_positions(self, backend: str):
        """Update queue positions after removal."""
        queue = self._backend_queues.get(backend, [])
        for i, job_id in enumerate(queue):
            job = self._jobs.get(job_id)
            if job:
                job.queue_position = i + 1

    def set_chat_url(self, job_id: str, chat_url: str):
        """Update the chat URL for a processing job (for recovery)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.chat_url = chat_url
                self._persist()

    def complete(
        self,
        job_id: str,
        result: str,
        deep_mode_used: bool = False,
    ):
        """Mark a job as complete with its result."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                log.warning("pool.jobs.complete_unknown", job_id=job_id)
                return

            job.status = JobStatus.COMPLETE
            job.result = result
            job.deep_mode_used = deep_mode_used
            job.completed_at = time.time()

            log.info("pool.jobs.completed",
                     job_id=job_id,
                     backend=job.backend,
                     result_length=len(result),
                     deep_mode_used=deep_mode_used)

            self._persist()

    def fail(self, job_id: str, error: str):
        """Mark a job as failed."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                log.warning("pool.jobs.fail_unknown", job_id=job_id)
                return

            job.status = JobStatus.FAILED
            job.error = error
            job.completed_at = time.time()

            log.error("pool.jobs.failed",
                      job_id=job_id,
                      backend=job.backend,
                      error=error)

            self._persist()

    def get_status(self, job_id: str) -> Optional[dict]:
        """Get the status of a job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None

            return {
                "job_id": job_id,
                "status": job.status.value,
                "queue_position": job.queue_position if job.status == JobStatus.QUEUED else 0,
                "backend": job.backend,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "completed_at": job.completed_at,
            }

    def get_result(self, job_id: str) -> Optional[dict]:
        """Get the result of a completed job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None

            if job.status == JobStatus.COMPLETE:
                return {
                    "job_id": job_id,
                    "status": "complete",
                    "result": job.result,
                    "deep_mode_used": job.deep_mode_used,
                    "backend": job.backend,
                    "thread_id": job.thread_id,
                }
            elif job.status == JobStatus.FAILED:
                return {
                    "job_id": job_id,
                    "status": "failed",
                    "error": job.error,
                    "backend": job.backend,
                    "thread_id": job.thread_id,
                }
            else:
                return {
                    "job_id": job_id,
                    "status": job.status.value,
                }

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        with self._lock:
            return self._jobs.get(job_id)

    def get_processing_job(self, backend: str) -> Optional[Job]:
        """Get the currently processing job for a backend."""
        with self._lock:
            for job in self._jobs.values():
                if job.backend == backend and job.status == JobStatus.PROCESSING:
                    return job
            return None

    def get_queue_depth(self, backend: str) -> int:
        """Get the number of queued jobs for a backend."""
        with self._lock:
            return sum(
                1 for job_id in self._backend_queues.get(backend, [])
                if (job := self._jobs.get(job_id)) and job.status == JobStatus.QUEUED
            )

    def get_all_queues(self) -> dict[str, int]:
        """Get queue depths for all backends."""
        with self._lock:
            return {
                backend: self.get_queue_depth(backend)
                for backend in self._backend_queues
            }

    def cleanup_old_jobs(self, max_age: int = 7200):
        """Remove completed/failed jobs older than max_age seconds."""
        with self._lock:
            now = time.time()
            to_remove = []

            for job_id, job in self._jobs.items():
                if job.status in (JobStatus.COMPLETE, JobStatus.FAILED):
                    if job.completed_at and now - job.completed_at > max_age:
                        to_remove.append(job_id)

            for job_id in to_remove:
                job = self._jobs.pop(job_id)
                if job.content_hash:
                    self._content_cache.pop(job.content_hash, None)
                    self._cache_times.pop(job.content_hash, None)

            if to_remove:
                log.info("pool.jobs.cleanup", removed_count=len(to_remove))
                self._persist()

    def get_completed_jobs(self) -> list[dict]:
        """
        Get all completed jobs (for recovery after orchestrator restart).

        Returns:
            List of completed job info with results.
        """
        with self._lock:
            results = []
            for job_id, job in self._jobs.items():
                if job.status == JobStatus.COMPLETE:
                    results.append({
                        "job_id": job.job_id,
                        "backend": job.backend,
                        "thread_id": job.thread_id,
                        "task_type": job.task_type,
                        "result": job.result,
                        "response": job.result,  # Alias for orchestrator compatibility
                        "deep_mode_used": job.deep_mode_used,
                        "completed_at": job.completed_at,
                        "request_id": job.job_id,  # For compatibility
                    })
            return results

    def remove_job(self, job_id: str) -> bool:
        """
        Remove a job from the store (after recovery pickup).

        Returns:
            True if job was removed, False if not found.
        """
        with self._lock:
            job = self._jobs.pop(job_id, None)
            if job:
                if job.content_hash:
                    self._content_cache.pop(job.content_hash, None)
                    self._cache_times.pop(job.content_hash, None)
                log.info("pool.jobs.removed", job_id=job_id)
                self._persist()
                return True
            return False

    def _persist(self):
        """
        Persist jobs to disk atomically.

        Uses write-to-temp-file + rename pattern to ensure atomic updates.
        This prevents data corruption if the process crashes during write.
        """
        if not self.persist_path:
            return

        try:
            data = {
                "jobs": {jid: j.to_dict() for jid, j in self._jobs.items()},
                "queues": self._backend_queues,
            }
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to temp file first, then atomically rename
            temp_path = self.persist_path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(data, f)
                f.flush()
                # Ensure data is written to disk before rename
                import os
                os.fsync(f.fileno())

            # Atomic rename (on POSIX this is atomic, on Windows it replaces)
            temp_path.replace(self.persist_path)
        except Exception as e:
            log.error("pool.jobs.persist_failed", error=str(e))
            # Clean up temp file if it exists
            try:
                temp_path = self.persist_path.with_suffix(".tmp")
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass

    def _load_from_disk(self):
        """Load jobs from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return

        try:
            with open(self.persist_path) as f:
                data = json.load(f)

            for job_id, job_data in data.get("jobs", {}).items():
                self._jobs[job_id] = Job.from_dict(job_data)
                if self._jobs[job_id].content_hash:
                    self._content_cache[self._jobs[job_id].content_hash] = job_id
                    self._cache_times[self._jobs[job_id].content_hash] = self._jobs[job_id].created_at

            self._backend_queues = data.get("queues", {})

            # Count jobs that need recovery (were processing when we crashed)
            processing_count = sum(
                1 for j in self._jobs.values()
                if j.status == JobStatus.PROCESSING
            )

            log.info("pool.jobs.loaded",
                     total_jobs=len(self._jobs),
                     processing_jobs=processing_count,
                     queued_jobs=sum(self.get_queue_depth(b) for b in self._backend_queues))

        except Exception as e:
            log.error("pool.jobs.load_failed", error=str(e))
