"""Two-level configuration: infrastructure (config.yaml) + project (projects/*.yaml).

Supports environment variable overrides with FANO_ prefix, dotted key
access, type coercion, and validation.  See docs/DESIGN_SPEC.md Section 5.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from shared.errors import ConfigError
from shared.models import (
    EvaluationCriterion, Project, ProjectStatus, ResearchDomain,
)

FANO_ROOT = Path(__file__).resolve().parent.parent


def _coerce(value: str) -> bool | int | float | str:
    """Coerce a string env var to bool / int / float / str."""
    low = value.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _deep_set(data: dict, keys: list[str], value: Any) -> None:
    for key in keys[:-1]:
        data = data.setdefault(key, {})
    data[keys[-1]] = value


def _deep_get(data: dict, keys: list[str], default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


class Config:
    """Two-level configuration with env-var overrides and validation."""

    _REQUIRED_KEYS = ["llm.api_key_env", "llm.models", "consensus.backends"]
    _THRESHOLD_KEYS = [
        "consensus.convergence_threshold",
        "consensus.minimum_agreement",
    ]

    def __init__(
        self,
        config_path: Path | None = None,
        project_path: Path | None = None,
    ) -> None:
        self._config_path = config_path or (FANO_ROOT / "config.yaml")
        self._project_path = project_path
        self._data: dict = {}
        self._project: Project | None = None

    # -- Loading --------------------------------------------------------

    def load(self) -> None:
        """Load and validate config files. Apply env-var overrides.
        Raises ConfigError if validation fails."""
        self._data = self._load_yaml(self._config_path)
        self._apply_env_overrides()
        self._validate_infra()
        if self._project_path is not None:
            proj_data = self._load_yaml(self._project_path)
            self._validate_project(proj_data)
            self._project = self._parse_project(proj_data)

    @classmethod
    def from_dict(cls, data: dict, project_data: dict | None = None) -> Config:
        """Create a Config from dicts (useful for testing)."""
        cfg = cls.__new__(cls)
        cfg._config_path = None
        cfg._project_path = None
        cfg._data = dict(data)
        cfg._project = None
        cfg._apply_env_overrides()
        cfg._validate_infra()
        if project_data is not None:
            cfg._validate_project(project_data)
            cfg._project = cfg._parse_project(project_data)
        return cfg

    # -- Access ---------------------------------------------------------

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Retrieve a value by dotted key (e.g. 'llm.models.gemini')."""
        return _deep_get(self._data, dotted_key.split("."), default)

    @property
    def project(self) -> Project | None:
        """Parsed project config, or None if no project loaded."""
        return self._project

    # -- Internal -------------------------------------------------------

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        try:
            with open(path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        except FileNotFoundError as exc:
            raise ConfigError(f"Config file not found: {path}") from exc
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"Expected mapping at top level of {path}")
        return data

    def _apply_env_overrides(self) -> None:
        prefix = "FANO_"
        for key, raw in os.environ.items():
            if not key.startswith(prefix):
                continue
            parts = key[len(prefix):].lower().split("__")
            _deep_set(self._data, parts, _coerce(raw))

    def _validate_infra(self) -> None:
        for dotted in self._REQUIRED_KEYS:
            if self.get(dotted) is None:
                raise ConfigError(f"Required config key missing: {dotted}")
        api_key_env: str = self.get("llm.api_key_env")
        if not os.environ.get(api_key_env):
            raise ConfigError(
                f"Environment variable {api_key_env} (from llm.api_key_env) is not set"
            )
        models: dict = self.get("llm.models")
        for backend in self.get("consensus.backends"):
            if backend not in models:
                raise ConfigError(
                    f"Consensus backend '{backend}' not found in llm.models"
                )
        for dotted in self._THRESHOLD_KEYS:
            val = self.get(dotted)
            if val is not None and not (0 <= float(val) <= 1):
                name = dotted.split(".")[-1]
                raise ConfigError(f"{name} must be between 0 and 1, got {val}")

    @staticmethod
    def _validate_project(data: dict) -> None:
        if not data.get("goal"):
            raise ConfigError("Project config: goal must be non-empty")
        if not data.get("evaluation_criteria"):
            raise ConfigError("Project config: evaluation_criteria must be non-empty")

    @staticmethod
    def _parse_project(data: dict) -> Project:
        """Build a Project model from raw project YAML data."""
        criteria = [
            EvaluationCriterion(name=c["name"], description=c["description"],
                                weight=float(c["weight"]))
            for c in data.get("evaluation_criteria", [])
        ]
        domains = [
            ResearchDomain(name=d["name"], keywords=d.get("keywords", []),
                           source_types=d.get("source_types", []),
                           extraction_patterns=d.get("extraction_patterns", []))
            for d in data.get("research_domains", [])
        ]

        def _ts(val: Any) -> datetime:
            return val if isinstance(val, datetime) else datetime.fromisoformat(str(val))

        return Project(
            id=data["id"],
            name=data["name"],
            goal=data["goal"],
            context=data.get("context", ""),
            evaluation_criteria=criteria,
            exploration_guidance=data.get("exploration_guidance", ""),
            document_guidance=data.get("document_guidance", ""),
            seed_modification_enabled=bool(data.get("seed_modification_enabled", False)),
            seed_modification_require_approval=bool(
                data.get("seed_modification_require_approval", True)),
            research_domains=domains,
            status=ProjectStatus(data.get("status", "active")),
            created_at=_ts(data.get("created_at", datetime.min)),
            updated_at=_ts(data.get("updated_at", datetime.min)),
        )
