"""Flask blueprints for the control panel."""

from .ui import bp as ui_bp
from .status import bp as status_bp
from .components import bp as components_bp
from .documenter import bp as documenter_bp
from .annotations import bp as annotations_bp
from .explorer import bp as explorer_bp
from .researcher import bp as researcher_bp
from .pool import bp as pool_bp
from .orchestrator import bp as orchestrator_bp

__all__ = [
    "ui_bp",
    "status_bp",
    "components_bp",
    "documenter_bp",
    "annotations_bp",
    "explorer_bp",
    "researcher_bp",
    "pool_bp",
    "orchestrator_bp",
]
