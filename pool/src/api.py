"""FastAPI HTTP API for the Browser Pool Service."""

import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

# Add shared module to path
SHARED_PATH = Path(__file__).resolve().parent.parent.parent / "shared"
sys.path.insert(0, str(SHARED_PATH.parent))

from shared.logging import get_logger

from .models import (
    SendRequest, SendResponse, Backend, Priority,
    BackendStatus, PoolStatus, HealthResponse, AuthResponse,
)
from .state import StateManager
from .queue import QueueManager, QueueFullError
from .workers import GeminiWorker, ChatGPTWorker, ClaudeWorker

log = get_logger("pool", "api")


class BrowserPool:
    """
    The main Browser Pool service.

    Manages workers, queues, and state for all backends.
    """

    def __init__(self, config: dict):
        self.config = config
        self.start_time = time.time()

        # Initialize state manager
        state_file = Path(__file__).parent.parent / "pool_state.json"
        self.state = StateManager(state_file, config)

        # Initialize queue manager
        self.queues = QueueManager(config)

        # Initialize workers (but don't start yet)
        self.workers = {}
        backends_config = config.get("backends", {})

        if backends_config.get("gemini", {}).get("enabled", True):
            self.workers["gemini"] = GeminiWorker(
                config, self.state, self.queues.get_queue("gemini")
            )

        if backends_config.get("chatgpt", {}).get("enabled", True):
            self.workers["chatgpt"] = ChatGPTWorker(
                config, self.state, self.queues.get_queue("chatgpt")
            )

        if backends_config.get("claude", {}).get("enabled", True):
            self.workers["claude"] = ClaudeWorker(
                config, self.state, self.queues.get_queue("claude")
            )

    async def startup(self):
        """Start all workers and connect to backends."""
        log.info("pool.service.lifecycle", action="starting", backends=list(self.workers.keys()))

        for name, worker in self.workers.items():
            try:
                await worker.connect()
            except Exception as e:
                log.warning("pool.backend.connect_failed", backend=name, error=str(e), will_retry=True)
            # Start worker loop regardless - it will wait for availability
            try:
                await worker.start()
            except Exception as e:
                log.error("pool.backend.worker_start_failed", backend=name, error=str(e))

        log.info("pool.service.lifecycle", action="started", backends=list(self.workers.keys()))

    async def shutdown(self):
        """Stop all workers and disconnect."""
        log.info("pool.service.lifecycle", action="stopping")

        for name, worker in self.workers.items():
            try:
                await worker.stop()
                await worker.disconnect()
            except Exception as e:
                log.error("pool.backend.stop_failed", backend=name, error=str(e))

        log.info("pool.service.lifecycle", action="stopped")

    async def send(self, request: SendRequest) -> SendResponse:
        """Send a prompt to a backend and wait for response."""
        backend = request.backend.value

        # Check if backend exists
        if backend not in self.workers:
            return SendResponse(
                success=False,
                error="unavailable",
                message=f"Backend '{backend}' not enabled",
            )

        # Check if backend is available
        if not self.state.is_available(backend):
            state = self.state.get_backend_state(backend)
            if state.get("rate_limited"):
                return SendResponse(
                    success=False,
                    error="rate_limited",
                    message=f"{backend} is rate limited",
                    retry_after_seconds=3600,
                )
            else:
                return SendResponse(
                    success=False,
                    error="auth_required",
                    message=f"{backend} requires authentication",
                )

        # Enqueue the request
        try:
            queue = self.queues.get_queue(backend)
            future = await queue.enqueue(request)
        except QueueFullError as e:
            return SendResponse(
                success=False,
                error="queue_full",
                message=str(e),
            )

        # Wait for response with timeout
        timeout = request.options.timeout_seconds
        try:
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            return SendResponse(
                success=False,
                error="timeout",
                message=f"Request timed out after {timeout} seconds",
            )

    def get_status(self) -> PoolStatus:
        """Get status of all backends."""
        backends_config = self.config.get("backends", {})
        depths = self.queues.get_depths()

        status = PoolStatus()

        if "gemini" in self.workers:
            state = self.state.get_backend_state("gemini")
            gemini_config = backends_config.get("gemini", {})
            status.gemini = BackendStatus(
                available=self.state.is_available("gemini"),
                authenticated=state.get("authenticated", False),
                rate_limited=state.get("rate_limited", False),
                rate_limit_resets_at=state.get("rate_limit_resets_at"),
                queue_depth=depths.get("gemini", 0),
                deep_mode_uses_today=state.get("deep_mode_uses_today", 0),
                deep_mode_limit=gemini_config.get("deep_mode", {}).get("daily_limit", 20),
            )

        if "chatgpt" in self.workers:
            state = self.state.get_backend_state("chatgpt")
            chatgpt_config = backends_config.get("chatgpt", {})
            status.chatgpt = BackendStatus(
                available=self.state.is_available("chatgpt"),
                authenticated=state.get("authenticated", False),
                rate_limited=state.get("rate_limited", False),
                rate_limit_resets_at=state.get("rate_limit_resets_at"),
                queue_depth=depths.get("chatgpt", 0),
                pro_mode_uses_today=state.get("pro_mode_uses_today", 0),
                pro_mode_limit=chatgpt_config.get("pro_mode", {}).get("daily_limit", 100),
            )

        if "claude" in self.workers:
            state = self.state.get_backend_state("claude")
            status.claude = BackendStatus(
                available=self.state.is_available("claude"),
                authenticated=state.get("authenticated", False),
                rate_limited=state.get("rate_limited", False),
                rate_limit_resets_at=state.get("rate_limit_resets_at"),
                queue_depth=depths.get("claude", 0),
            )

        return status

    async def authenticate(self, backend: str) -> bool:
        """Trigger interactive authentication for a backend."""
        if backend not in self.workers:
            return False

        return await self.workers[backend].authenticate()


def create_app(config: dict) -> FastAPI:
    """Create the FastAPI application."""

    app = FastAPI(
        title="Browser Pool Service",
        description="Shared LLM access layer for Fano platform",
        version="0.1.0",
    )

    pool = BrowserPool(config)

    @app.on_event("startup")
    async def startup():
        await pool.startup()

    @app.on_event("shutdown")
    async def shutdown():
        await pool.shutdown()

    @app.post("/send", response_model=SendResponse)
    async def send(request: SendRequest):
        """Send a prompt to an LLM and wait for response."""
        return await pool.send(request)

    @app.get("/status", response_model=PoolStatus)
    async def status():
        """Get status of all backends."""
        return pool.get_status()

    @app.post("/auth/{backend}", response_model=AuthResponse)
    async def auth(backend: str):
        """Trigger interactive authentication for a backend."""
        if backend not in ["gemini", "chatgpt", "claude"]:
            raise HTTPException(status_code=400, detail=f"Unknown backend: {backend}")

        success = await pool.authenticate(backend)
        if success:
            return AuthResponse(
                success=True,
                message=f"Authentication window opened for {backend}. Please log in manually.",
            )
        else:
            return AuthResponse(
                success=False,
                message=f"Failed to start authentication for {backend}",
            )

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        return HealthResponse(
            status="ok",
            uptime_seconds=time.time() - pool.start_time,
            version="0.1.0",
        )

    return app
