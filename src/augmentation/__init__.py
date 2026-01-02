"""
Chunk Augmentation Module.

Provides functionality to augment blessed insights with:
- Diagrams: Visual representations of structures
- Tables: Systematic enumerations
- Proofs: Formal mathematical verification
- Code: Executable demonstrations
"""

from .models import (
    Augmentation,
    AugmentationAnalysis,
    AugmentationType,
    AugmentedInsight,
    DiagramType,
    TableType,
    CodePurpose,
)

from .augmenter import Augmenter, get_augmenter

__all__ = [
    "Augmentation",
    "AugmentationAnalysis",
    "AugmentationType",
    "AugmentedInsight",
    "DiagramType",
    "TableType",
    "CodePurpose",
    "Augmenter",
    "get_augmenter",
]
