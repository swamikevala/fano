"""
Opportunity finder - identifying what to work on next.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from shared.logging import get_logger
from .document import Document
from .concepts import ConceptTracker

log = get_logger("documenter", "opportunities")


@dataclass
class Opportunity:
    """An opportunity to add content to the document."""
    # Currently only "blessed_insight" is implemented.
    # "bridge" and "extension" are reserved for future features.
    type: str  # "blessed_insight" (| "bridge" | "extension" - reserved)
    source_file: Optional[str] = None
    insight_id: Optional[str] = None
    text: str = ""
    requires: list[str] = None
    priority: int = 0  # Higher = more important
    dispute_count: int = 0  # Times this was disputed

    def __post_init__(self):
        if self.requires is None:
            self.requires = []


class OpportunityFinder:
    """
    Finds opportunities for document growth.

    Scans blessed insights, checks for ready items, and
    prioritizes based on dependencies and document state.
    """

    def __init__(
        self,
        document: Document,
        concept_tracker: ConceptTracker,
        blessed_dir: Path,
        max_disputes: int = 3,
        disputes_file: Optional[Path] = None,
    ):
        """
        Initialize opportunity finder.

        Args:
            document: The document being built
            concept_tracker: Concept dependency tracker
            blessed_dir: Directory containing blessed insight files
            max_disputes: Max times to retry a disputed item
            disputes_file: Path to persist dispute counts (prevents reset between sessions)
        """
        self.document = document
        self.concept_tracker = concept_tracker
        self.blessed_dir = Path(blessed_dir)
        self.max_disputes = max_disputes
        self._disputes_file = disputes_file or (Path(blessed_dir).parent / ".dispute_counts.json")
        self._dispute_counts: dict[str, int] = self._load_disputes()

    def _load_disputes(self) -> dict[str, int]:
        """Load dispute counts from persistent storage."""
        if self._disputes_file and self._disputes_file.exists():
            try:
                data = json.loads(self._disputes_file.read_text(encoding="utf-8"))
                return {k: int(v) for k, v in data.items()}
            except Exception as e:
                log.warning("documenter.disputes.load_error", error=str(e))
        return {}

    def _save_disputes(self):
        """Save dispute counts to persistent storage."""
        if self._disputes_file:
            try:
                self._disputes_file.parent.mkdir(parents=True, exist_ok=True)
                self._disputes_file.write_text(
                    json.dumps(self._dispute_counts, indent=2),
                    encoding="utf-8"
                )
            except Exception as e:
                log.warning("documenter.disputes.save_error", error=str(e))

    def find_next(self) -> Optional[Opportunity]:
        """
        Find the next opportunity to work on.

        Returns the highest priority opportunity whose dependencies are met,
        or None if no opportunities are ready.
        """
        opportunities = self._gather_all_opportunities()

        if not opportunities:
            log.info("documenter.opportunities.none_found")
            return None

        # Filter to those whose dependencies are met
        established = self.concept_tracker.get_established_concepts()
        ready = []

        for opp in opportunities:
            missing = [r for r in opp.requires if r not in established]
            if not missing:
                ready.append(opp)

        if not ready:
            log.info(
                "documenter.opportunities.none_ready",
                total=len(opportunities),
                blocked_by_deps=len(opportunities),
            )
            return None

        # Sort by priority (higher first)
        ready.sort(key=lambda o: o.priority, reverse=True)

        log.info(
            "documenter.opportunities.found",
            ready_count=len(ready),
            selected_type=ready[0].type,
        )
        return ready[0]

    def _gather_all_opportunities(self) -> list[Opportunity]:
        """Gather all potential opportunities."""
        opportunities = []

        # Blessed insights
        opportunities.extend(self._load_blessed_insights())

        return opportunities

    def _load_blessed_insights(self) -> list[Opportunity]:
        """Load unincorporated blessed insights."""
        opportunities = []

        if not self.blessed_dir.exists():
            return opportunities

        for file_path in self.blessed_dir.glob("*.json"):
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))

                # Skip already incorporated (check both old and explorer formats)
                if data.get("incorporated", False):
                    continue
                # Explorer format: status might be "incorporated" after documenter processes it
                if data.get("status") == "incorporated":
                    continue

                # Skip if disputed too many times
                insight_id = data.get("id", file_path.stem)
                if self._dispute_counts.get(insight_id, 0) >= self.max_disputes:
                    log.debug(
                        "documenter.opportunities.skipped_disputed",
                        insight_id=insight_id,
                    )
                    continue

                # Handle both old format (text/requires) and explorer format (insight/depends_on)
                text = data.get("text") or data.get("insight", "")
                requires = data.get("requires") or data.get("depends_on", [])

                opp = Opportunity(
                    type="blessed_insight",
                    source_file=str(file_path),
                    insight_id=insight_id,
                    text=text,
                    requires=requires,
                    priority=self._calculate_priority(data),
                    dispute_count=self._dispute_counts.get(insight_id, 0),
                )
                opportunities.append(opp)

            except Exception as e:
                log.warning(
                    "documenter.opportunities.load_error",
                    file=str(file_path),
                    error=str(e),
                )

        return opportunities

    def _calculate_priority(self, insight_data: dict) -> int:
        """Calculate priority for an insight."""
        priority = 0

        # Higher confidence = higher priority
        confidence = insight_data.get("confidence", "medium")
        if confidence == "high":
            priority += 10
        elif confidence == "medium":
            priority += 5

        # Fewer dependencies = higher priority (easier to add)
        requires = insight_data.get("requires", [])
        priority -= len(requires) * 2

        # Tags might indicate importance
        tags = insight_data.get("tags", [])
        if "core" in tags or "fundamental" in tags:
            priority += 5

        return priority

    def mark_incorporated(self, opportunity: Opportunity):
        """Mark an opportunity as incorporated."""
        if opportunity.source_file and Path(opportunity.source_file).exists():
            try:
                file_path = Path(opportunity.source_file)
                data = json.loads(file_path.read_text(encoding="utf-8"))
                data["incorporated"] = True
                data["incorporated_at"] = datetime.now().isoformat()
                # Also update explorer format status field
                if "status" in data:
                    data["status"] = "incorporated"
                file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

                log.info(
                    "documenter.opportunities.marked_incorporated",
                    insight_id=opportunity.insight_id,
                )
            except Exception as e:
                log.warning(
                    "documenter.opportunities.mark_error",
                    insight_id=opportunity.insight_id,
                    error=str(e),
                )

    def mark_disputed(self, opportunity: Opportunity):
        """Mark an opportunity as disputed (consensus failed)."""
        if opportunity.insight_id:
            self._dispute_counts[opportunity.insight_id] = (
                self._dispute_counts.get(opportunity.insight_id, 0) + 1
            )

            # Persist dispute counts to survive session restarts
            self._save_disputes()

            count = self._dispute_counts[opportunity.insight_id]
            log.info(
                "documenter.opportunities.marked_disputed",
                insight_id=opportunity.insight_id,
                dispute_count=count,
                max_disputes=self.max_disputes,
            )

            # If max disputes reached, flag for human review
            if count >= self.max_disputes:
                self._flag_for_human_review(opportunity)

    def _flag_for_human_review(self, opportunity: Opportunity):
        """Flag an opportunity for human review."""
        if opportunity.source_file and Path(opportunity.source_file).exists():
            try:
                file_path = Path(opportunity.source_file)
                data = json.loads(file_path.read_text(encoding="utf-8"))
                data["needs_human_review"] = True
                data["dispute_count"] = self._dispute_counts.get(opportunity.insight_id, 0)
                file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

                log.warning(
                    "documenter.opportunities.flagged_for_review",
                    insight_id=opportunity.insight_id,
                )
            except Exception as e:
                log.warning(
                    "documenter.opportunities.flag_error",
                    insight_id=opportunity.insight_id,
                    error=str(e),
                )

    def get_pending_count(self) -> int:
        """Get count of pending (unincorporated) opportunities."""
        return len(self._gather_all_opportunities())

    def get_ready_count(self) -> int:
        """Get count of opportunities ready to be worked on."""
        opportunities = self._gather_all_opportunities()
        established = self.concept_tracker.get_established_concepts()

        ready = 0
        for opp in opportunities:
            if all(r in established for r in opp.requires):
                ready += 1

        return ready
