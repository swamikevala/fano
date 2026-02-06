"""Flask blueprints for the control panel v2 API."""

from .auth import bp as auth_bp
from .projects import bp as projects_bp
from .seeds import bp as seeds_bp
from .insights import bp as insights_bp
from .document import bp as document_bp
from .annotations import bp as annotations_bp
from .research import bp as research_bp
from .status import bp as status_bp
from .ui import bp as ui_bp

__all__ = [
    "auth_bp",
    "projects_bp",
    "seeds_bp",
    "insights_bp",
    "document_bp",
    "annotations_bp",
    "research_bp",
    "status_bp",
    "ui_bp",
]
