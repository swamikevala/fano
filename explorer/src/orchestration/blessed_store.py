"""
Blessed Store - Manages blessed insights.

This module centralizes:
- Loading and storing blessed insights
- Blessing new insights
- Augmentation of blessed insights
- Deduplication tracking
"""

import json
import logging
from datetime import datetime
from typing import Optional

from explorer.src.models import AxiomStore
from explorer.src.chunking import AtomicInsight, DeduplicationChecker
from explorer.src.augmentation import Augmenter
from explorer.src.storage import ExplorerPaths

logger = logging.getLogger(__name__)


class BlessedStore:
    """
    Manages the blessed insights store.

    Responsible for:
    - Loading blessed insights from file and directory
    - Blessing new insights
    - Augmenting blessed insights
    - Providing blessed summaries for prompts
    """

    def __init__(
        self,
        config: dict,
        paths: ExplorerPaths,
        axioms: AxiomStore,
        augmenter: Optional[Augmenter] = None,
        dedup_checker: Optional[DeduplicationChecker] = None,
    ):
        """
        Initialize blessed store.

        Args:
            config: Full configuration dict
            paths: ExplorerPaths instance
            axioms: AxiomStore for seed access
            augmenter: Optional Augmenter for generating augmentations
            dedup_checker: Optional DeduplicationChecker for tracking
        """
        self.config = config
        self.paths = paths
        self.axioms = axioms
        self.augmenter = augmenter
        self.dedup_checker = dedup_checker

    def get_blessed_insights(self) -> list[dict]:
        """
        Get list of blessed insights for dependency matching.

        Returns:
            List of dicts with 'id', 'text', 'tags', and optionally 'is_seed'.
        """
        blessed = []

        # Load seed aphorisms first (user-provided starting points)
        seeds = self.axioms.get_seed_aphorisms()
        for seed in seeds:
            blessed.append({
                "id": seed.id,
                "text": seed.text,
                "tags": seed.tags,
                "is_seed": True,
            })

        # Load from blessed_insights.json if exists
        if self.paths.blessed_insights_file.exists():
            with open(self.paths.blessed_insights_file, encoding="utf-8") as f:
                data = json.load(f)
                blessed.extend(data.get("insights", []))

        # Also include insights with BLESSED status from insights directory
        if self.paths.blessed_insights_dir.exists():
            for filepath in self.paths.blessed_insights_dir.glob("*.json"):
                try:
                    insight = AtomicInsight.load(filepath)
                    blessed.append({
                        "id": insight.id,
                        "text": insight.insight,
                        "tags": insight.tags,
                    })
                except Exception:
                    pass

        return blessed

    def get_blessed_summary(self) -> str:
        """
        Get a summary of blessed axioms/insights for prompts.

        Returns:
            Context string for exploration prompts.
        """
        return self.axioms.get_context_for_exploration()

    async def bless_insight(self, insight: AtomicInsight, review_summary: str = "") -> None:
        """
        Add an insight to the blessed insights store and augment it.

        Args:
            insight: The insight to bless
            review_summary: Summary of review findings for augmentation context
        """
        # Load existing data
        if self.paths.blessed_insights_file.exists():
            with open(self.paths.blessed_insights_file, encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {"insights": []}

        # Add new insight
        data["insights"].append({
            "id": insight.id,
            "text": insight.insight,
            "confidence": insight.confidence,
            "tags": insight.tags,
            "depends_on": insight.depends_on,
            "source_thread_id": insight.source_thread_id,
            "blessed_at": datetime.now().isoformat(),
        })

        # Save
        with open(self.paths.blessed_insights_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Blessed insight [{insight.id}] added to axiom store")

        # Add to deduplication checker for future duplicate detection
        if self.dedup_checker:
            self.dedup_checker.add_known_insight(insight.id, insight.insight)

        # Augment the blessed insight (generate diagrams, tables, proofs, code)
        if self.augmenter:
            try:
                augmented = await self.augmenter.augment_insight(
                    insight_id=insight.id,
                    insight_text=insight.insight,
                    tags=insight.tags,
                    dependencies=insight.depends_on,
                    review_summary=review_summary,
                )
                aug_count = len(augmented.augmentations)
                if aug_count > 0:
                    logger.info(f"[{insight.id}] Generated {aug_count} augmentations")
            except Exception as e:
                logger.warning(f"[{insight.id}] Augmentation failed: {e}")

    def load_blessed_into_dedup(self) -> int:
        """
        Load all blessed insights into the deduplication checker.

        Returns:
            Number of insights loaded.
        """
        if not self.dedup_checker:
            return 0

        blessed_insights = self.get_blessed_insights()
        self.dedup_checker.load_known_insights(blessed_insights)
        return len(blessed_insights)
