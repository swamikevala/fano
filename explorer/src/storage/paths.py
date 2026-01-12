"""
Centralized path management for the explorer.

All path construction logic is consolidated here to avoid
scattered hardcoded paths throughout the codebase.
"""

from pathlib import Path


class ExplorerPaths:
    """Centralized path management for explorer data directories."""

    def __init__(self, data_dir: Path = None):
        """
        Initialize path manager.

        Args:
            data_dir: Root data directory. Defaults to explorer/data.
        """
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent.parent.parent / "data"
        self.data_dir = Path(data_dir)

    # === Core Directories ===

    @property
    def explorations_dir(self) -> Path:
        """Directory containing exploration thread JSON files."""
        return self.data_dir / "explorations"

    @property
    def chunks_dir(self) -> Path:
        """Directory containing chunk files."""
        return self.data_dir / "chunks"

    @property
    def insights_dir(self) -> Path:
        """Directory containing insight files."""
        return self.chunks_dir / "insights"

    @property
    def axioms_dir(self) -> Path:
        """Directory containing axiom/seed files."""
        return self.data_dir / "axioms"

    # === Insight Status Directories ===

    @property
    def pending_insights_dir(self) -> Path:
        """Directory for pending (unreviewed) insights."""
        return self.insights_dir / "pending"

    @property
    def blessed_insights_dir(self) -> Path:
        """Directory for blessed insights."""
        return self.insights_dir / "blessed"

    @property
    def interesting_insights_dir(self) -> Path:
        """Directory for interesting (but not blessed) insights."""
        return self.insights_dir / "interesting"

    @property
    def rejected_insights_dir(self) -> Path:
        """Directory for rejected insights."""
        return self.insights_dir / "rejected"

    @property
    def reviewing_insights_dir(self) -> Path:
        """Directory for insights currently under review."""
        return self.chunks_dir / "reviewing"

    # === Special Files ===

    @property
    def blessed_insights_file(self) -> Path:
        """JSON file containing all blessed insights."""
        return self.data_dir / "blessed_insights.json"

    @property
    def database_file(self) -> Path:
        """SQLite database file."""
        return self.data_dir / "fano_explorer.db"

    # === Helper Methods ===

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        directories = [
            self.explorations_dir,
            self.chunks_dir,
            self.insights_dir,
            self.pending_insights_dir,
            self.blessed_insights_dir,
            self.interesting_insights_dir,
            self.rejected_insights_dir,
            self.reviewing_insights_dir,
            self.axioms_dir,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_thread_path(self, thread_id: str) -> Path:
        """Get the path for a specific thread file."""
        return self.explorations_dir / f"{thread_id}.json"

    def get_insight_path(self, insight_id: str, status: str = "pending") -> Path:
        """
        Get the path for a specific insight file.

        Args:
            insight_id: The insight ID
            status: One of 'pending', 'blessed', 'interesting', 'rejected'
        """
        status_dirs = {
            "pending": self.pending_insights_dir,
            "blessed": self.blessed_insights_dir,
            "interesting": self.interesting_insights_dir,
            "rejected": self.rejected_insights_dir,
        }
        base_dir = status_dirs.get(status, self.pending_insights_dir)
        return base_dir / f"{insight_id}.json"

    def find_insight_path(self, insight_id: str) -> Path | None:
        """
        Find an insight file by ID, searching all status directories.

        Returns:
            Path to the insight file, or None if not found.
        """
        for status in ["pending", "blessed", "interesting", "rejected"]:
            path = self.get_insight_path(insight_id, status)
            if path.exists():
                return path
        return None

    def get_chunk_path(self, chunk_id: str) -> Path:
        """Get the path for a specific chunk file."""
        return self.chunks_dir / f"{chunk_id}.json"


# Default instance for convenience
_default_paths: ExplorerPaths | None = None


def get_paths(data_dir: Path = None) -> ExplorerPaths:
    """
    Get a paths instance.

    Args:
        data_dir: Optional data directory. If None, uses default.

    Returns:
        ExplorerPaths instance.
    """
    global _default_paths
    if data_dir is not None:
        return ExplorerPaths(data_dir)
    if _default_paths is None:
        _default_paths = ExplorerPaths()
    return _default_paths
