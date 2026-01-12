"""
Documenter main module - orchestration and entry point.

This module coordinates the document growth process by delegating to:
- SessionManager: initialization, cleanup, configuration
- WorkPlanner: intelligent work planning
- OpportunityProcessor: insight evaluation pipeline
- CommentHandler: author comment and review handling
"""

import asyncio
from pathlib import Path
from typing import Optional

from shared.logging import get_logger, correlation_context

from .session import SessionManager
from .planning import WorkPlanner
from .opportunity_processor import OpportunityProcessor
from .comments import CommentHandler

log = get_logger("documenter", "main")


class Orchestrator:
    """
    Main documenter orchestrator - coordinates document growth.

    Creates and maintains a living mathematical document by:
    - Incorporating blessed insights from Explorer
    - Validating all additions through LLM consensus
    - Reviewing and improving existing content
    - Creating daily snapshots
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize orchestrator.

        Args:
            config_path: Path to config file (default: config.yaml in project root)
        """
        self.session = SessionManager(config_path)
        self.planner: Optional[WorkPlanner] = None
        self.processor: Optional[OpportunityProcessor] = None
        self.comment_handler: Optional[CommentHandler] = None

    async def run(self):
        """Main loop - grow the document."""
        # Initialize session and get guidance text
        guidance_text = await self.session.initialize()

        # Initialize component handlers
        self.planner = WorkPlanner(self.session, guidance_text)
        self.processor = OpportunityProcessor(self.session)
        self.comment_handler = CommentHandler(self.session)

        log.info("orchestrator.started")

        try:
            with correlation_context() as cid:
                while not self.session.exhausted:
                    # Check for snapshot
                    self.session.snapshot_manager.check_and_snapshot()

                    # Check consensus call budget
                    if self.session.check_budget():
                        break

                    # Check for author comments first (highest priority)
                    comments = self.session.document.find_unresolved_comments()
                    if comments:
                        comment_text, line_num = comments[0]
                        await self.comment_handler.address_comment(comment_text, line_num)
                        continue

                    # Decide: new work or review?
                    if self.session.review_manager.should_do_review():
                        section = self.session.review_manager.select_section_for_review()
                        if section:
                            await self.comment_handler.review_section(section)
                            continue

                    # Work on new material - use intelligent planning
                    work_plan = await self.planner.plan_next_work()
                    if work_plan:
                        if work_plan["type"] == "prerequisite":
                            # Generate foundational content first
                            establishes = work_plan.get("establishes", [])
                            await self.processor.write_prerequisite(work_plan["content"], establishes)
                        elif work_plan["type"] == "insight":
                            # Work on the selected insight
                            await self.processor.process_opportunity(work_plan["opportunity"])
                        continue

                    # Nothing to do - LLMs said WAIT or no clear decision
                    log.info("orchestrator.exhausted", reason="nothing_to_do")
                    self.session.exhausted = True

        finally:
            # Always cleanup
            await self.session.cleanup()

            # Final snapshot
            self.session.snapshot_manager.create_snapshot()

            # Log summary
            self.session.log_summary()


# Backward compatibility alias
Documenter = Orchestrator


async def main():
    """Entry point for running the documenter."""
    orchestrator = Orchestrator()
    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
