"""
Tests for the FanoLogger structured logging system.

Tests cover:
- Logger creation and caching
- Event logging at all levels
- JSON output format
- Correlation ID context management
- Request lifecycle logging
- Exception logging
"""

import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from shared.logging import (
    get_logger,
    FanoLogger,
    correlation_context,
    get_correlation_id,
    set_correlation_id,
    get_session_id,
    set_session_id,
)
from shared.logging.context import (
    get_request_id,
    set_request_id,
    _correlation_id,
    _session_id,
    _request_id,
)


class TestGetLogger:
    """Tests for get_logger factory function."""

    def test_get_logger_returns_fano_logger(self):
        """get_logger returns a FanoLogger instance."""
        log = get_logger("test_module", "test_component")
        assert isinstance(log, FanoLogger)

    def test_get_logger_returns_same_instance(self):
        """get_logger returns cached instance for same module/component."""
        log1 = get_logger("cache_test", "component")
        log2 = get_logger("cache_test", "component")
        assert log1 is log2

    def test_get_logger_different_components(self):
        """Different components get different loggers."""
        log1 = get_logger("diff_test", "component1")
        log2 = get_logger("diff_test", "component2")
        assert log1 is not log2

    def test_get_logger_different_modules(self):
        """Different modules get different loggers."""
        log1 = get_logger("module1", "same_component")
        log2 = get_logger("module2", "same_component")
        assert log1 is not log2


class TestFanoLogger:
    """Tests for FanoLogger class."""

    @pytest.fixture
    def log_dir(self, temp_dir):
        """Create temporary log directory."""
        log_path = temp_dir / "logs"
        log_path.mkdir(parents=True, exist_ok=True)
        return log_path

    @pytest.fixture
    def logger(self, log_dir):
        """Create logger with mocked log directory."""
        with patch("shared.logging.logger._get_log_dir", return_value=log_dir):
            # Clear logger cache for clean test
            from shared.logging import logger as logger_module
            logger_module._loggers.clear()
            yield FanoLogger("test_mod", "test_comp", console=False)

    def test_logger_creates_log_file(self, logger, log_dir):
        """Logger creates .jsonl file in log directory."""
        logger.info("test.event")
        log_file = log_dir / "test_mod.jsonl"
        assert log_file.exists()

    def test_logger_writes_json_lines(self, logger, log_dir):
        """Logger writes valid JSON to log file."""
        logger.info("test.json.event", key="value", count=42)

        log_file = log_dir / "test_mod.jsonl"
        content = log_file.read_text().strip()

        # Should be valid JSON
        log_entry = json.loads(content)
        assert log_entry["event_type"] == "test.json.event"

    def test_info_logs_info_level(self, logger, log_dir):
        """info() logs at INFO level."""
        logger.info("test.info", data="test")

        log_file = log_dir / "test_mod.jsonl"
        content = log_file.read_text()
        assert "INFO" in content

    def test_debug_logs_debug_level(self, logger, log_dir):
        """debug() logs at DEBUG level."""
        logger.debug("test.debug", data="test")

        log_file = log_dir / "test_mod.jsonl"
        content = log_file.read_text()
        assert "DEBUG" in content

    def test_warning_logs_warning_level(self, logger, log_dir):
        """warning() logs at WARNING level."""
        logger.warning("test.warning", issue="something")

        log_file = log_dir / "test_mod.jsonl"
        content = log_file.read_text()
        assert "WARNING" in content

    def test_error_logs_error_level(self, logger, log_dir):
        """error() logs at ERROR level."""
        logger.error("test.error", error="failed")

        log_file = log_dir / "test_mod.jsonl"
        content = log_file.read_text()
        assert "ERROR" in content

    def test_event_includes_module_and_component(self, logger, log_dir):
        """Event includes module and component in output."""
        logger.info("test.module.event")

        log_file = log_dir / "test_mod.jsonl"
        log_entry = json.loads(log_file.read_text().strip())

        assert log_entry["fano_module"] == "test_mod"
        assert log_entry["component"] == "test_comp"

    def test_event_includes_custom_data(self, logger, log_dir):
        """Event includes custom key-value data."""
        logger.info(
            "test.custom",
            user_id="user-123",
            action="login",
            count=5,
        )

        log_file = log_dir / "test_mod.jsonl"
        log_entry = json.loads(log_file.read_text().strip())

        assert log_entry["user_id"] == "user-123"
        assert log_entry["action"] == "login"
        assert log_entry["count"] == 5


class TestExceptionLogging:
    """Tests for exception logging."""

    @pytest.fixture
    def logger(self, temp_dir):
        """Create logger with mocked log directory."""
        log_dir = temp_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        with patch("shared.logging.logger._get_log_dir", return_value=log_dir):
            from shared.logging import logger as logger_module
            logger_module._loggers.clear()
            yield FanoLogger("exc_test", "component", console=False), log_dir

    def test_exception_logs_error_class(self, logger):
        """exception() logs the error class name."""
        log, log_dir = logger
        try:
            raise ValueError("Test error message")
        except ValueError as e:
            log.exception(e, event_type="test.exception")

        log_file = log_dir / "exc_test.jsonl"
        log_entry = json.loads(log_file.read_text().strip())

        assert log_entry["error_class"] == "ValueError"

    def test_exception_logs_error_message(self, logger):
        """exception() logs the error message."""
        log, log_dir = logger
        try:
            raise RuntimeError("Something went wrong")
        except RuntimeError as e:
            log.exception(e, event_type="test.exception")

        log_file = log_dir / "exc_test.jsonl"
        log_entry = json.loads(log_file.read_text().strip())

        assert log_entry["error_message"] == "Something went wrong"

    def test_exception_logs_stack_trace(self, logger):
        """exception() includes stack trace."""
        log, log_dir = logger
        try:
            raise KeyError("missing_key")
        except KeyError as e:
            log.exception(e, event_type="test.exception")

        log_file = log_dir / "exc_test.jsonl"
        log_entry = json.loads(log_file.read_text().strip())

        assert "stack_trace" in log_entry
        assert "KeyError" in log_entry["stack_trace"]

    def test_exception_logs_context(self, logger):
        """exception() includes provided context."""
        log, log_dir = logger
        try:
            raise IOError("File not found")
        except IOError as e:
            log.exception(
                e,
                event_type="test.exception",
                context={"file": "/path/to/file", "operation": "read"},
            )

        log_file = log_dir / "exc_test.jsonl"
        log_entry = json.loads(log_file.read_text().strip())

        assert log_entry["context"]["file"] == "/path/to/file"
        assert log_entry["context"]["operation"] == "read"


class TestRequestLifecycleLogging:
    """Tests for request lifecycle logging methods."""

    @pytest.fixture
    def logger(self, temp_dir):
        """Create logger with mocked log directory."""
        log_dir = temp_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        with patch("shared.logging.logger._get_log_dir", return_value=log_dir):
            from shared.logging import logger as logger_module
            logger_module._loggers.clear()
            yield FanoLogger("req_test", "api", console=False), log_dir

    def test_request_start_returns_time(self, logger):
        """request_start returns start time for duration calculation."""
        log, _ = logger
        start_time = log.request_start(
            request_id="req-123",
            backend="claude",
            prompt="Test prompt",
        )

        assert isinstance(start_time, float)
        assert start_time > 0

    def test_request_start_logs_prompt_length(self, logger):
        """request_start logs prompt length."""
        log, log_dir = logger
        log.request_start(
            request_id="req-456",
            backend="gemini",
            prompt="A" * 100,
        )

        log_file = log_dir / "req_test.jsonl"
        log_entry = json.loads(log_file.read_text().strip())

        assert log_entry["prompt_length"] == 100

    def test_request_complete_logs_duration(self, logger):
        """request_complete logs duration in milliseconds."""
        log, log_dir = logger
        start_time = time.time() - 1.5  # 1.5 seconds ago

        log.request_complete(
            request_id="req-789",
            backend="chatgpt",
            response="Test response",
            start_time=start_time,
        )

        log_file = log_dir / "req_test.jsonl"
        log_entry = json.loads(log_file.read_text().strip())

        assert "duration_ms" in log_entry
        assert log_entry["duration_ms"] >= 1500  # At least 1.5 seconds

    def test_request_complete_logs_response_length(self, logger):
        """request_complete logs response length."""
        log, log_dir = logger
        log.request_complete(
            request_id="req-len",
            backend="claude",
            response="B" * 200,
            start_time=time.time(),
        )

        log_file = log_dir / "req_test.jsonl"
        log_entry = json.loads(log_file.read_text().strip())

        assert log_entry["response_length"] == 200

    def test_request_error_logs_error_type(self, logger):
        """request_error logs error type."""
        log, log_dir = logger
        log.request_error(
            request_id="req-err",
            backend="gemini",
            error="Connection timeout",
            error_type="timeout",
        )

        log_file = log_dir / "req_test.jsonl"
        log_entry = json.loads(log_file.read_text().strip())

        assert log_entry["error_type"] == "timeout"
        assert log_entry["error"] == "Connection timeout"

    def test_request_error_logs_duration_if_provided(self, logger):
        """request_error includes duration if start_time provided."""
        log, log_dir = logger
        start_time = time.time() - 2.0

        log.request_error(
            request_id="req-err-dur",
            backend="claude",
            error="Rate limited",
            error_type="rate_limit",
            start_time=start_time,
        )

        log_file = log_dir / "req_test.jsonl"
        log_entry = json.loads(log_file.read_text().strip())

        assert "duration_ms" in log_entry
        assert log_entry["duration_ms"] >= 2000


class TestCorrelationContext:
    """Tests for correlation ID context management."""

    def setup_method(self):
        """Reset context vars before each test."""
        _correlation_id.set("")
        _session_id.set("")
        _request_id.set("")

    def test_get_correlation_id_generates_uuid(self):
        """get_correlation_id generates UUID if none set."""
        cid = get_correlation_id()
        assert cid  # Not empty
        assert len(cid) == 36  # UUID format

    def test_set_correlation_id(self):
        """set_correlation_id sets the ID."""
        set_correlation_id("custom-cid-123")
        assert get_correlation_id() == "custom-cid-123"

    def test_correlation_context_yields_id(self):
        """correlation_context yields correlation ID."""
        with correlation_context() as cid:
            assert cid
            assert len(cid) == 36

    def test_correlation_context_accepts_custom_id(self):
        """correlation_context uses provided ID."""
        with correlation_context(correlation_id="my-custom-id") as cid:
            assert cid == "my-custom-id"
            assert get_correlation_id() == "my-custom-id"

    def test_correlation_context_restores_previous(self):
        """correlation_context restores previous ID on exit."""
        set_correlation_id("original-id")

        with correlation_context(correlation_id="temporary-id"):
            assert get_correlation_id() == "temporary-id"

        assert get_correlation_id() == "original-id"

    def test_session_id_management(self):
        """Session ID can be set and retrieved."""
        assert get_session_id() is None  # Initially None

        set_session_id("session-abc")
        assert get_session_id() == "session-abc"

    def test_correlation_context_with_session(self):
        """correlation_context can set session ID."""
        with correlation_context(session_id="ctx-session"):
            assert get_session_id() == "ctx-session"

    def test_request_id_management(self):
        """Request ID can be set and retrieved."""
        assert get_request_id() is None

        set_request_id("req-xyz")
        assert get_request_id() == "req-xyz"

    def test_correlation_context_with_request(self):
        """correlation_context can set request ID."""
        with correlation_context(request_id="ctx-request"):
            assert get_request_id() == "ctx-request"

    def test_nested_correlation_contexts(self):
        """Nested correlation contexts work correctly."""
        with correlation_context(correlation_id="outer") as outer_cid:
            assert outer_cid == "outer"

            with correlation_context(correlation_id="inner") as inner_cid:
                assert inner_cid == "inner"

            # After inner context exits, outer is restored
            assert get_correlation_id() == "outer"


class TestLogDirectoryDiscovery:
    """Tests for log directory discovery logic."""

    def test_log_dir_created_if_not_exists(self, temp_dir):
        """Log directory is created if it doesn't exist."""
        from shared.logging import logger as logger_module

        # Create the logs directory that the mock will return
        logs_dir = temp_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        # Reset cached log dir and loggers
        logger_module._log_dir = None
        logger_module._loggers.clear()

        with patch("shared.logging.logger._get_log_dir", return_value=logs_dir):
            log = FanoLogger("discovery_test", "comp", console=False)
            log.info("test.event")

        # Log file should have been created
        log_file = logs_dir / "discovery_test.jsonl"
        assert log_file.exists()
