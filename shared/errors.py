"""Shared error types for the Research Project Assistant."""


class ResearchAssistantError(Exception):
    """Base error for all project errors."""


class StoreError(ResearchAssistantError):
    """Database operation failed."""


class ConfigError(ResearchAssistantError):
    """Configuration invalid or missing."""


class LLMError(ResearchAssistantError):
    """LLM call failed after retries."""

    def __init__(self, message: str, backend: str, is_transient: bool):
        super().__init__(message)
        self.backend = backend
        self.is_transient = is_transient


class ConsensusError(ResearchAssistantError):
    """Consensus could not be reached."""

    def __init__(
        self,
        message: str,
        rounds_completed: int,
        partial_result: dict | None = None,
    ):
        super().__init__(message)
        self.rounds_completed = rounds_completed
        self.partial_result = partial_result


class InsufficientResponsesError(ConsensusError):
    """Too many backends failed to get a valid consensus round."""


class BudgetExhaustedError(ResearchAssistantError):
    """Module has exceeded its LLM budget."""

    def __init__(self, module: str, spent: float, limit: float):
        super().__init__(f"{module} budget exhausted: ${spent:.2f} / ${limit:.2f}")
        self.module = module
        self.spent = spent
        self.limit = limit
