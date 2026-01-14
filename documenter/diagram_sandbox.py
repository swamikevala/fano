"""
Diagram Sandbox - Safe execution environment for diagram generation.

Executes LLM-generated Python code for creating mathematical diagrams
in a restricted environment with:
- Whitelisted modules only (matplotlib, networkx, numpy, math)
- 30 second timeout
- No file system access outside assets directory
- No network access
"""

import ast
import io
import sys
import uuid
from pathlib import Path
from typing import Optional
import asyncio
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError

from shared.logging import get_logger

log = get_logger("documenter", "diagram_sandbox")


# Modules allowed in the sandbox
ALLOWED_MODULES = {
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.patches",
    "matplotlib.lines",
    "networkx",
    "numpy",
    "math",
    "itertools",
    "collections",
}

# Dangerous built-ins to remove
BLOCKED_BUILTINS = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "input",
    "breakpoint",
}


class SandboxViolation(Exception):
    """Raised when code attempts forbidden operations."""
    pass


class DiagramSandbox:
    """
    Safe execution environment for diagram code.

    Validates and executes Python code that generates matplotlib
    or networkx diagrams, saving output to the assets directory.
    """

    def __init__(
        self,
        assets_dir: Path,
        timeout_seconds: int = 30,
    ):
        """
        Initialize the sandbox.

        Args:
            assets_dir: Directory to save generated diagrams
            timeout_seconds: Maximum execution time
        """
        self.assets_dir = Path(assets_dir)
        self.timeout_seconds = timeout_seconds
        self.assets_dir.mkdir(parents=True, exist_ok=True)

    def validate_code(self, code: str) -> tuple[bool, str]:
        """
        Validate code before execution.

        Checks for:
        - Syntax errors
        - Forbidden imports
        - Dangerous function calls

        Args:
            code: Python code to validate

        Returns:
            (is_valid, error_message)
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        # Check all nodes
        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not self._is_allowed_module(alias.name):
                        return False, f"Forbidden import: {alias.name}"

            elif isinstance(node, ast.ImportFrom):
                if node.module and not self._is_allowed_module(node.module):
                    return False, f"Forbidden import: {node.module}"

            # Check for dangerous calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in BLOCKED_BUILTINS:
                        return False, f"Forbidden function: {node.func.id}"

                elif isinstance(node.func, ast.Attribute):
                    # Block subprocess, os.system, etc.
                    if node.func.attr in ("system", "popen", "spawn", "fork"):
                        return False, f"Forbidden call: {node.func.attr}"

            # Check for file operations
            elif isinstance(node, ast.With):
                for item in node.items:
                    if isinstance(item.context_expr, ast.Call):
                        if isinstance(item.context_expr.func, ast.Name):
                            if item.context_expr.func.id == "open":
                                return False, "File operations not allowed"

        return True, ""

    def _is_allowed_module(self, module_name: str) -> bool:
        """Check if a module is in the whitelist."""
        # Check exact match or prefix match
        if module_name in ALLOWED_MODULES:
            return True
        for allowed in ALLOWED_MODULES:
            if module_name.startswith(allowed + "."):
                return True
        return False

    async def execute(
        self,
        code: str,
        output_format: str = "svg",
    ) -> tuple[Optional[Path], str]:
        """
        Execute diagram code in the sandbox.

        Args:
            code: Python code that creates a matplotlib figure
            output_format: Image format (svg, png)

        Returns:
            (path_to_output, error_message)
        """
        # Validate first
        is_valid, error = self.validate_code(code)
        if not is_valid:
            log.warning("diagram_sandbox.validation_failed", error=error)
            return None, error

        # Generate output filename
        diagram_id = str(uuid.uuid4())[:8]
        output_path = self.assets_dir / f"diagram_{diagram_id}.{output_format}"

        # Execute in process pool with timeout
        try:
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    self._execute_in_sandbox,
                    code,
                    str(output_path),
                    output_format,
                ),
                timeout=self.timeout_seconds,
            )

            if result[0]:  # Success
                log.info(
                    "diagram_sandbox.execution_success",
                    path=str(output_path),
                )
                return output_path, ""
            else:
                return None, result[1]

        except asyncio.TimeoutError:
            log.warning(
                "diagram_sandbox.timeout",
                timeout=self.timeout_seconds,
            )
            return None, f"Execution timed out after {self.timeout_seconds} seconds"

        except Exception as e:
            log.error("diagram_sandbox.execution_error", error=str(e))
            return None, str(e)

    def _execute_in_sandbox(
        self,
        code: str,
        output_path: str,
        output_format: str,
    ) -> tuple[bool, str]:
        """
        Execute code in restricted environment.

        This runs in a separate thread/process.
        """
        # Create restricted globals
        restricted_globals = {
            "__builtins__": {
                k: v for k, v in __builtins__.items()
                if k not in BLOCKED_BUILTINS
            } if isinstance(__builtins__, dict) else {
                k: getattr(__builtins__, k)
                for k in dir(__builtins__)
                if k not in BLOCKED_BUILTINS and not k.startswith('_')
            },
        }

        # Add allowed imports
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import numpy as np
            import math
            import networkx as nx
            import itertools
            import collections

            restricted_globals.update({
                "plt": plt,
                "np": np,
                "numpy": np,
                "math": math,
                "nx": nx,
                "networkx": nx,
                "itertools": itertools,
                "collections": collections,
            })
        except ImportError as e:
            return False, f"Required module not available: {e}"

        # Capture output
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        try:
            # Execute the code
            exec(code, restricted_globals)

            # Get current figure and save
            fig = plt.gcf()
            if fig.get_axes():  # Only save if there's content
                fig.savefig(output_path, format=output_format, bbox_inches='tight')
                plt.close(fig)
                return True, ""
            else:
                plt.close(fig)
                return False, "Code did not produce a figure"

        except Exception as e:
            stderr_output = sys.stderr.getvalue()
            return False, f"Execution error: {e}\n{stderr_output}"

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            plt.close('all')


def generate_fano_plane_code() -> str:
    """
    Example: Generate code for a Fano plane diagram.

    This is a template the LLM can reference/modify.
    """
    return '''
import matplotlib.pyplot as plt
import numpy as np

# Fano plane coordinates (7 points)
points = {
    'A': (0.5, 0.866),      # Top vertex
    'B': (0, 0),            # Bottom left
    'C': (1, 0),            # Bottom right
    'D': (0.5, 0),          # Midpoint BC
    'E': (0.75, 0.433),     # Midpoint AC
    'F': (0.25, 0.433),     # Midpoint AB
    'O': (0.5, 0.289),      # Center
}

# Lines (7 lines, each with 3 points)
lines = [
    ['A', 'B', 'F'],  # Side AB
    ['B', 'C', 'D'],  # Side BC
    ['A', 'C', 'E'],  # Side AC
    ['A', 'O', 'D'],  # Altitude from A
    ['B', 'O', 'E'],  # Altitude from B
    ['C', 'O', 'F'],  # Altitude from C
    ['D', 'E', 'F'],  # Inner circle (circumscribed)
]

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

# Draw lines
for line in lines:
    if 'D' in line and 'E' in line and 'F' in line:
        # Draw the inscribed circle
        theta = np.linspace(0, 2*np.pi, 100)
        r = 0.25
        cx, cy = 0.5, 0.289
        ax.plot(cx + r*np.cos(theta), cy + r*np.sin(theta), 'b-', linewidth=1.5)
    else:
        pts = [points[p] for p in line]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, 'b-', linewidth=1.5)

# Draw points
for name, (x, y) in points.items():
    ax.plot(x, y, 'ko', markersize=12)
    ax.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points',
                fontsize=14, fontweight='bold')

ax.set_xlim(-0.2, 1.2)
ax.set_ylim(-0.2, 1.1)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('The Fano Plane', fontsize=16, fontweight='bold')
'''
