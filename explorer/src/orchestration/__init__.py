"""
Orchestration modules for the explorer.

This package contains the refactored orchestration components:
- LLMManager: LLM connection and communication management
- ThreadManager: Thread loading, selection, and spawning
- ExplorationEngine: Exploration and critique operations
- SynthesisEngine: Chunk synthesis from threads
- InsightProcessor: Insight extraction and automated review
- BlessedStore: Blessed insights management
"""

from .llm_manager import LLMManager
from .thread_manager import ThreadManager
from .exploration import ExplorationEngine
from .synthesis import SynthesisEngine
from .insight_processor import InsightProcessor
from .blessed_store import BlessedStore

__all__ = [
    "LLMManager",
    "ThreadManager",
    "ExplorationEngine",
    "SynthesisEngine",
    "InsightProcessor",
    "BlessedStore",
]
