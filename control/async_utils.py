"""Async-to-sync helper for Flask routes.

Extracted to its own module to avoid circular imports between
control.server and control.blueprints.
"""

from __future__ import annotations

import asyncio


def run_async(coro):
    """Run an async coroutine from synchronous Flask context.

    Attempts to use the running loop (if any); otherwise creates a new one.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an already-running loop (e.g. during tests with
        # pytest-asyncio). Use a new thread to avoid "cannot run nested".
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)
