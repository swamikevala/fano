"""Tests for shared.config — Config module.

Tests the two-level configuration system: infrastructure (config.yaml)
+ project (projects/*.yaml), environment variable overrides, dotted
key access, type coercion, and validation.
"""

from pathlib import Path

import pytest
import yaml

from shared.config import Config
from shared.errors import ConfigError


# ---------------------------------------------------------------------------
# Fixtures — minimal valid YAML data
# ---------------------------------------------------------------------------

def _minimal_infra() -> dict:
    """Return a minimal valid infrastructure config dict."""
    return {
        "llm": {
            "api_key_env": "OPENROUTER_API_KEY",
            "endpoint": "https://openrouter.ai/api/v1",
            "models": {
                "gemini": "google/gemini-2.0-flash-thinking-exp-01-21",
                "chatgpt": "openai/gpt-4o",
                "claude": "anthropic/claude-sonnet-4-20250514",
            },
            "rate_limits": {"gemini": 10, "chatgpt": 60, "claude": 50},
            "default_timeout_seconds": 300,
            "max_retries": 3,
        },
        "consensus": {
            "backends": ["gemini", "chatgpt", "claude"],
            "max_rounds": 4,
            "min_valid_responses": 2,
            "convergence_threshold": 0.7,
            "decision_method": "majority",
            "minimum_agreement": 0.66,
        },
        "explorer": {
            "max_active_threads": 3,
        },
    }


def _minimal_project() -> dict:
    """Return a minimal valid project config dict."""
    return {
        "id": "proj-fano-math",
        "name": "Fano Mathematics",
        "goal": "Explore the Fano plane and its connections to combinatorics.",
        "context": "The Fano plane is the smallest finite projective plane.",
        "evaluation_criteria": [
            {
                "name": "mathematical_rigor",
                "description": "Logical correctness and proof quality",
                "weight": 0.4,
            },
            {
                "name": "novelty",
                "description": "Degree of new insight or perspective",
                "weight": 0.3,
            },
        ],
        "exploration_guidance": "Focus on symmetry groups and collineation.",
        "document_guidance": "Write for advanced undergraduates.",
        "seed_modification_enabled": True,
        "seed_modification_require_approval": True,
        "research_domains": [
            {
                "name": "finite_geometry",
                "keywords": ["Fano plane", "projective plane"],
                "source_types": ["arxiv", "mathworld"],
            },
        ],
        "status": "active",
        "created_at": "2025-01-01T00:00:00",
        "updated_at": "2025-01-01T00:00:00",
    }


def _write_yaml(path: Path, data: dict) -> Path:
    """Write *data* as YAML to *path* and return the path."""
    path.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")
    return path


@pytest.fixture()
def infra_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write a valid infrastructure config and set the env var it references."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key-12345")
    return _write_yaml(tmp_path / "config.yaml", _minimal_infra())


@pytest.fixture()
def project_yaml(tmp_path: Path) -> Path:
    """Write a valid project config."""
    return _write_yaml(tmp_path / "fano-mathematics.yaml", _minimal_project())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadInfrastructureConfig:
    """test_load_infrastructure_config — Loads config.yaml, all keys accessible."""

    def test_load_infrastructure_config(
        self, infra_yaml: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = Config(config_path=infra_yaml)
        cfg.load()

        assert cfg.get("llm.api_key_env") == "OPENROUTER_API_KEY"
        assert cfg.get("llm.models.gemini") == "google/gemini-2.0-flash-thinking-exp-01-21"
        assert cfg.get("consensus.backends") == ["gemini", "chatgpt", "claude"]
        assert cfg.get("explorer.max_active_threads") == 3
        assert cfg.get("llm.default_timeout_seconds") == 300


class TestLoadProjectConfig:
    """test_load_project_config — Project model parsed correctly."""

    def test_load_project_config(
        self,
        infra_yaml: Path,
        project_yaml: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cfg = Config(config_path=infra_yaml, project_path=project_yaml)
        cfg.load()

        proj = cfg.project
        assert proj is not None
        assert proj.id == "proj-fano-math"
        assert proj.name == "Fano Mathematics"
        assert proj.goal.startswith("Explore the Fano plane")
        assert len(proj.evaluation_criteria) == 2
        assert proj.evaluation_criteria[0].name == "mathematical_rigor"
        assert proj.evaluation_criteria[0].weight == 0.4
        assert proj.seed_modification_enabled is True
        assert len(proj.research_domains) == 1
        assert proj.research_domains[0].name == "finite_geometry"


class TestDottedKeyAccess:
    """test_dotted_key_access — Nested keys via get('a.b.c')."""

    def test_dotted_key_access(
        self, infra_yaml: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = Config(config_path=infra_yaml)
        cfg.load()

        assert cfg.get("llm.models.chatgpt") == "openai/gpt-4o"
        assert cfg.get("consensus.convergence_threshold") == 0.7
        assert cfg.get("llm.rate_limits.gemini") == 10


class TestDefaultValue:
    """test_default_value — Missing key returns default."""

    def test_default_value(
        self, infra_yaml: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = Config(config_path=infra_yaml)
        cfg.load()

        assert cfg.get("nonexistent.key") is None
        assert cfg.get("nonexistent.key", 42) == 42
        assert cfg.get("llm.nonexistent", "fallback") == "fallback"


class TestEnvOverride:
    """test_env_override — Environment variable overrides YAML value."""

    def test_env_override(
        self,
        infra_yaml: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # FANO_LLM__MODELS__GEMINI  →  overrides llm.models.gemini
        monkeypatch.setenv("FANO_LLM__MODELS__GEMINI", "google/gemini-pro")
        monkeypatch.setenv("FANO_EXPLORER__MAX_ACTIVE_THREADS", "5")

        cfg = Config(config_path=infra_yaml)
        cfg.load()

        assert cfg.get("llm.models.gemini") == "google/gemini-pro"
        assert cfg.get("explorer.max_active_threads") == 5


class TestEnvTypeCoercion:
    """test_env_type_coercion — 'true' → bool, '42' → int, '3.14' → float."""

    def test_env_type_coercion(
        self,
        infra_yaml: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("FANO_EXPLORER__MAX_ACTIVE_THREADS", "42")
        # Use a field without range validation to test float coercion
        monkeypatch.setenv("FANO_LLM__DEFAULT_TIMEOUT_SECONDS", "3.14")
        monkeypatch.setenv("FANO_EXPLORER__ENABLED", "true")

        cfg = Config(config_path=infra_yaml)
        cfg.load()

        val_int = cfg.get("explorer.max_active_threads")
        assert val_int == 42
        assert isinstance(val_int, int)

        val_float = cfg.get("llm.default_timeout_seconds")
        assert val_float == pytest.approx(3.14)
        assert isinstance(val_float, float)

        val_bool = cfg.get("explorer.enabled")
        assert val_bool is True
        assert isinstance(val_bool, bool)

    def test_false_coercion(
        self,
        infra_yaml: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("FANO_EXPLORER__ENABLED", "false")

        cfg = Config(config_path=infra_yaml)
        cfg.load()

        val = cfg.get("explorer.enabled")
        assert val is False
        assert isinstance(val, bool)


class TestValidationMissingRequired:
    """test_validation_missing_required — ConfigError on missing llm.api_key_env."""

    def test_missing_api_key_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        data = _minimal_infra()
        del data["llm"]["api_key_env"]
        config_path = _write_yaml(tmp_path / "config.yaml", data)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

        cfg = Config(config_path=config_path)
        with pytest.raises(ConfigError, match="llm.api_key_env"):
            cfg.load()

    def test_missing_llm_models(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        data = _minimal_infra()
        del data["llm"]["models"]
        config_path = _write_yaml(tmp_path / "config.yaml", data)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

        cfg = Config(config_path=config_path)
        with pytest.raises(ConfigError, match="llm.models"):
            cfg.load()

    def test_missing_consensus_backends(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        data = _minimal_infra()
        del data["consensus"]["backends"]
        config_path = _write_yaml(tmp_path / "config.yaml", data)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

        cfg = Config(config_path=config_path)
        with pytest.raises(ConfigError, match="consensus.backends"):
            cfg.load()

    def test_api_key_env_var_not_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """api_key_env references an env var that is NOT set."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        data = _minimal_infra()
        config_path = _write_yaml(tmp_path / "config.yaml", data)

        cfg = Config(config_path=config_path)
        with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
            cfg.load()


class TestValidationInvalidBackend:
    """test_validation_invalid_backend — ConfigError if consensus backend not in llm.models."""

    def test_invalid_backend(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        data = _minimal_infra()
        data["consensus"]["backends"] = ["gemini", "nonexistent_model"]
        config_path = _write_yaml(tmp_path / "config.yaml", data)

        cfg = Config(config_path=config_path)
        with pytest.raises(ConfigError, match="nonexistent_model"):
            cfg.load()


class TestNoProjectConfig:
    """test_no_project_config — config.project is None when no project loaded."""

    def test_no_project_config(
        self, infra_yaml: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = Config(config_path=infra_yaml)
        cfg.load()
        assert cfg.project is None


class TestFromDict:
    """Config.from_dict class method for testing convenience."""

    def test_from_dict_infra_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        cfg = Config.from_dict(_minimal_infra())
        assert cfg.get("llm.models.gemini") == "google/gemini-2.0-flash-thinking-exp-01-21"
        assert cfg.project is None

    def test_from_dict_with_project(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        cfg = Config.from_dict(_minimal_infra(), project_data=_minimal_project())
        assert cfg.project is not None
        assert cfg.project.name == "Fano Mathematics"


class TestProjectValidation:
    """Project config validation: goal and evaluation_criteria non-empty."""

    def test_empty_goal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        infra_path = _write_yaml(tmp_path / "config.yaml", _minimal_infra())
        proj = _minimal_project()
        proj["goal"] = ""
        proj_path = _write_yaml(tmp_path / "project.yaml", proj)

        cfg = Config(config_path=infra_path, project_path=proj_path)
        with pytest.raises(ConfigError, match="goal"):
            cfg.load()

    def test_empty_evaluation_criteria(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        infra_path = _write_yaml(tmp_path / "config.yaml", _minimal_infra())
        proj = _minimal_project()
        proj["evaluation_criteria"] = []
        proj_path = _write_yaml(tmp_path / "project.yaml", proj)

        cfg = Config(config_path=infra_path, project_path=proj_path)
        with pytest.raises(ConfigError, match="evaluation_criteria"):
            cfg.load()


class TestNumericRangeValidation:
    """Numeric values validated: thresholds 0-1."""

    def test_threshold_out_of_range(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        data = _minimal_infra()
        data["consensus"]["convergence_threshold"] = 1.5
        config_path = _write_yaml(tmp_path / "config.yaml", data)

        cfg = Config(config_path=config_path)
        with pytest.raises(ConfigError, match="convergence_threshold"):
            cfg.load()

    def test_minimum_agreement_out_of_range(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        data = _minimal_infra()
        data["consensus"]["minimum_agreement"] = -0.1
        config_path = _write_yaml(tmp_path / "config.yaml", data)

        cfg = Config(config_path=config_path)
        with pytest.raises(ConfigError, match="minimum_agreement"):
            cfg.load()
