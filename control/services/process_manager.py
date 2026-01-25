"""
Thread-safe process management for Fano components.
"""

import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, TextIO

from shared.logging import get_logger
from .config import FANO_ROOT

log = get_logger("control", "process_manager")

# Central logs directory
LOGS_DIR = FANO_ROOT / "logs"


class ProcessManager:
    """
    Thread-safe manager for component processes (pool, explorer, documenter).

    Uses a lock to protect state access from concurrent requests and
    background threads (e.g., server restart).
    """

    # Service dependencies - which services must be running before others can start
    # Note: Pool is no longer needed - all LLM access is via API
    SERVICE_DEPENDENCIES = {
        "pool": [],  # Deprecated - kept for compatibility
        "orchestrator": [],
        "explorer": [],
        "documenter": [],
        "researcher": [],
    }

    def __init__(self):
        self._lock = threading.Lock()
        self._processes = {
            "pool": None,
            "orchestrator": None,
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

    def _validate_script(self, script_path: Path, component: str) -> None:
        """Validate that a script file exists before starting."""
        if not script_path.exists():
            raise FileNotFoundError(
                f"Script for {component} not found: {script_path}"
            )

    def start_pool(self) -> subprocess.Popen:
        """Start the pool service."""
        pool_script = FANO_ROOT / "pool" / "run_pool.py"
        self._validate_script(pool_script, "pool")

        log_file = self._open_log_file("pool")
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
        explorer_script = FANO_ROOT / "explorer" / "fano_explorer.py"
        self._validate_script(explorer_script, "explorer")

        log_file = self._open_log_file("explorer")
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
        documenter_script = FANO_ROOT / "documenter" / "fano_documenter.py"
        self._validate_script(documenter_script, "documenter")

        log_file = self._open_log_file("documenter")
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
        researcher_script = FANO_ROOT / "researcher" / "main.py"
        self._validate_script(researcher_script, "researcher")

        log_file = self._open_log_file("researcher")
        proc = subprocess.Popen(
            [sys.executable, str(researcher_script), "start"],
            cwd=str(FANO_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        self.set("researcher", proc)
        return proc

    def start_orchestrator(self) -> subprocess.Popen:
        """Start the unified orchestrator."""
        orchestrator_script = FANO_ROOT / "run_orchestrator.py"
        self._validate_script(orchestrator_script, "orchestrator")

        log_file = self._open_log_file("orchestrator")
        proc = subprocess.Popen(
            [sys.executable, str(orchestrator_script)],
            cwd=str(FANO_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        self.set("orchestrator", proc)
        return proc

    def wait_for_orchestrator_health(
        self,
        host: str = "127.0.0.1",
        port: int = 9001,
        timeout_seconds: int = 30,
        poll_interval: float = 1.0,
    ) -> bool:
        """
        Wait for orchestrator /health endpoint to respond.

        Args:
            host: Orchestrator host address
            port: Orchestrator port
            timeout_seconds: Max time to wait
            poll_interval: Time between checks

        Returns:
            True if orchestrator became healthy within timeout, False otherwise.
        """
        start = time.time()
        url = f"http://{host}:{port}/health"

        while time.time() - start < timeout_seconds:
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        return True
            except (urllib.error.URLError, TimeoutError, OSError):
                pass
            time.sleep(poll_interval)

        return False

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

    def wait_for_pool_health(
        self,
        host: str = "127.0.0.1",
        port: int = 9000,
        timeout_seconds: int = 30,
        poll_interval: float = 1.0,
    ) -> bool:
        """
        Wait for pool /health endpoint to respond.

        Args:
            host: Pool host address
            port: Pool port
            timeout_seconds: Max time to wait
            poll_interval: Time between checks

        Returns:
            True if pool became healthy within timeout, False otherwise.
        """
        start = time.time()
        url = f"http://{host}:{port}/health"

        while time.time() - start < timeout_seconds:
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        return True
            except (urllib.error.URLError, TimeoutError, OSError):
                pass
            time.sleep(poll_interval)

        return False

    def check_dependencies_ready(self, component: str) -> tuple[bool, list[str]]:
        """
        Check if all dependencies for a component are running.

        Args:
            component: The component to check dependencies for

        Returns:
            Tuple of (all_ready, list_of_missing_dependencies)
        """
        deps = self.SERVICE_DEPENDENCIES.get(component, [])
        missing = [d for d in deps if not self.is_running(d)]
        return (len(missing) == 0, missing)

    def start_with_deps(self, component: str, config: dict = None) -> bool:
        """
        Start a component after ensuring its dependencies are ready and healthy.

        This method will:
        1. Auto-start any missing dependencies
        2. Wait for pool to be healthy (if it's a dependency)
        3. Start the requested component

        Args:
            component: The component to start
            config: Optional config dict for pool host/port/timeout settings

        Returns:
            True if component started successfully, False if dependencies failed.
        """
        deps = self.SERVICE_DEPENDENCIES.get(component, [])

        for dep in deps:
            if not self.is_running(dep):
                # Auto-start the dependency
                log.info(
                    "process.dependency.auto_starting",
                    component=component,
                    dependency=dep,
                )
                if dep == "pool":
                    self.start_pool()
                elif dep == "orchestrator":
                    self.start_orchestrator()
                elif dep == "explorer":
                    self.start_explorer()
                elif dep == "documenter":
                    self.start_documenter()
                elif dep == "researcher":
                    self.start_researcher()

            # Wait for pool to be healthy
            if dep == "pool":
                pool_config = (config or {}).get("llm", {}).get("pool", {})
                host = pool_config.get("host", "127.0.0.1")
                port = pool_config.get("port", 9000)
                timeout = (
                    (config or {})
                    .get("services", {})
                    .get("pool", {})
                    .get("health_timeout_seconds", 30)
                )

                log.info(
                    "process.dependency.waiting_for_health",
                    component=component,
                    dependency=dep,
                    timeout=timeout,
                )

                if not self.wait_for_pool_health(host, port, timeout):
                    log.error(
                        "process.dependency.unhealthy",
                        component=component,
                        dependency=dep,
                    )
                    return False

        # Now start the component itself
        try:
            if component == "pool":
                return self.start_pool() is not None
            elif component == "orchestrator":
                return self.start_orchestrator() is not None
            elif component == "explorer":
                return self.start_explorer() is not None
            elif component == "documenter":
                return self.start_documenter() is not None
            elif component == "researcher":
                return self.start_researcher() is not None
            else:
                log.warning("process.unknown_component", component=component)
                return False
        except Exception as e:
            log.error("process.start_failed", component=component, error=str(e))
            return False
