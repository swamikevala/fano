"""
Thread-safe process management for Fano components.
"""

import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional, TextIO

from .config import FANO_ROOT

# Central logs directory
LOGS_DIR = FANO_ROOT / "logs"


class ProcessManager:
    """
    Thread-safe manager for component processes (pool, explorer, documenter).

    Uses a lock to protect state access from concurrent requests and
    background threads (e.g., server restart).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._processes = {
            "pool": None,
            "explorer": None,
            "documenter": None,
            "researcher": None,
        }
        self._log_files: dict[str, TextIO] = {}

        # Ensure logs directory exists
        LOGS_DIR.mkdir(exist_ok=True)

    def get(self, component: str) -> Optional[subprocess.Popen]:
        """Get a process by component name."""
        with self._lock:
            return self._processes.get(component)

    def set(self, component: str, proc: Optional[subprocess.Popen]):
        """Set a process for a component."""
        with self._lock:
            self._processes[component] = proc

    def is_running(self, component: str) -> bool:
        """Check if a component is running."""
        with self._lock:
            proc = self._processes.get(component)
            return proc is not None and proc.poll() is None

    def get_pid(self, component: str) -> Optional[int]:
        """Get the PID of a running component."""
        with self._lock:
            proc = self._processes.get(component)
            if proc is not None and proc.poll() is None:
                return proc.pid
            return None

    def _open_log_file(self, component: str) -> TextIO:
        """Open (or truncate) a log file for a component."""
        # Close existing log file if any
        if component in self._log_files:
            try:
                self._log_files[component].close()
            except Exception:
                pass

        log_path = LOGS_DIR / f"{component}.log"
        # Truncate on start to avoid unbounded growth
        log_file = open(log_path, "w", encoding="utf-8", buffering=1)  # Line buffered
        self._log_files[component] = log_file
        return log_file

    def start_pool(self) -> subprocess.Popen:
        """Start the pool service."""
        log_file = self._open_log_file("pool")
        pool_script = FANO_ROOT / "pool" / "run_pool.py"
        proc = subprocess.Popen(
            [sys.executable, str(pool_script)],
            cwd=str(FANO_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        self.set("pool", proc)
        return proc

    def start_explorer(self, mode: str = "start") -> subprocess.Popen:
        """Start the explorer."""
        log_file = self._open_log_file("explorer")
        explorer_script = FANO_ROOT / "explorer" / "fano_explorer.py"
        proc = subprocess.Popen(
            [sys.executable, str(explorer_script), mode],
            cwd=str(FANO_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        self.set("explorer", proc)
        return proc

    def start_documenter(self) -> subprocess.Popen:
        """Start the documenter."""
        log_file = self._open_log_file("documenter")
        documenter_script = FANO_ROOT / "documenter" / "fano_documenter.py"
        proc = subprocess.Popen(
            [sys.executable, str(documenter_script), "start"],
            cwd=str(FANO_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        self.set("documenter", proc)
        return proc

    def start_researcher(self) -> subprocess.Popen:
        """Start the researcher."""
        log_file = self._open_log_file("researcher")
        researcher_script = FANO_ROOT / "researcher" / "main.py"
        proc = subprocess.Popen(
            [sys.executable, str(researcher_script), "start"],
            cwd=str(FANO_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        self.set("researcher", proc)
        return proc

    def _close_log_file(self, component: str):
        """Close the log file for a component if open."""
        if component in self._log_files:
            try:
                self._log_files[component].close()
            except Exception:
                pass
            del self._log_files[component]

    def stop(self, component: str) -> bool:
        """
        Stop a component. Returns True if stopped successfully.
        """
        with self._lock:
            proc = self._processes.get(component)
            if proc is None or proc.poll() is not None:
                return False

            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

            self._processes[component] = None
            self._close_log_file(component)
            return True

    def cleanup_all(self):
        """Stop all managed processes and close log files."""
        with self._lock:
            for name, proc in self._processes.items():
                if proc is not None and proc.poll() is None:
                    try:
                        proc.terminate()
                        proc.wait(timeout=3)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass

            # Clear all references
            for name in self._processes:
                self._processes[name] = None

            # Close all log files
            for name in list(self._log_files.keys()):
                self._close_log_file(name)

    def register_external(self, component: str, proc: subprocess.Popen):
        """Register an externally started process."""
        self.set(component, proc)
