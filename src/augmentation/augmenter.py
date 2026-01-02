"""
Augmenter - Main coordinator for chunk augmentation.

Analyzes blessed insights and generates appropriate augmentations
(diagrams, tables, proofs, code) using Claude Opus.
"""

import asyncio
import logging
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import (
    Augmentation,
    AugmentationAnalysis,
    AugmentationType,
    AugmentedInsight,
)
from .prompts import (
    build_analysis_prompt,
    parse_analysis_response,
    build_diagram_prompt,
    parse_diagram_response,
    build_table_prompt,
    parse_table_response,
    build_proof_prompt,
    parse_proof_response,
    build_code_prompt,
    parse_code_response,
    build_proof_verification_prompt,
    parse_verification_response,
)

logger = logging.getLogger(__name__)


class Augmenter:
    """
    Coordinates the augmentation of blessed insights.

    Uses Claude Opus for analysis and generation.
    """

    def __init__(
        self,
        claude_reviewer,
        config: dict,
        data_dir: Path,
    ):
        """
        Initialize the augmenter.

        Args:
            claude_reviewer: ClaudeReviewer instance for API calls
            config: Augmentation configuration
            data_dir: Base data directory for saving augmentations
        """
        self.claude = claude_reviewer
        self.config = config.get("augmentation", {})
        self.data_dir = data_dir / "augmentations"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Configuration
        self.enabled = self.config.get("enabled", True)
        self.auto_generate = self.config.get("auto_generate", True)
        self.require_verification = self.config.get("require_verification", True)

        # Type-specific config
        self.types_config = self.config.get("types", {})

        logger.info(f"[augmenter] Initialized (enabled={self.enabled})")

    async def augment_insight(
        self,
        insight_id: str,
        insight_text: str,
        tags: list[str],
        dependencies: list[str],
        review_summary: str,
    ) -> AugmentedInsight:
        """
        Analyze and augment a blessed insight.

        Args:
            insight_id: Unique ID of the insight
            insight_text: The insight text
            tags: Tags assigned to the insight
            dependencies: Dependencies of the insight
            review_summary: Summary of what reviewers found compelling

        Returns:
            AugmentedInsight with generated augmentations
        """
        if not self.enabled:
            logger.info(f"[augmenter] Augmentation disabled, skipping {insight_id}")
            return AugmentedInsight(
                insight_id=insight_id,
                insight_text=insight_text,
            )

        logger.info(f"[augmenter] Analyzing insight {insight_id}")

        # Step 1: Analyze what augmentations would help
        analysis = await self._analyze_insight(
            insight_text, tags, dependencies, review_summary
        )

        augmented = AugmentedInsight(
            insight_id=insight_id,
            insight_text=insight_text,
            analysis=analysis,
        )

        if not self.auto_generate:
            logger.info(f"[augmenter] Auto-generate disabled, returning analysis only")
            return augmented

        # Step 2: Generate recommended augmentations
        augmentations = []

        for rec in analysis.recommendations:
            rec_lower = rec.lower()

            if "diagram" in rec_lower and analysis.diagram_helpful:
                if self._type_enabled("diagram"):
                    aug = await self._generate_diagram(
                        insight_text,
                        analysis.diagram_type.value,
                        analysis.diagram_description,
                    )
                    if aug:
                        augmentations.append(aug)

            elif "table" in rec_lower and analysis.table_helpful:
                if self._type_enabled("table"):
                    aug = await self._generate_table(
                        insight_text,
                        analysis.table_type.value,
                        analysis.table_description,
                    )
                    if aug:
                        augmentations.append(aug)

            elif "proof" in rec_lower and analysis.proof_possible in ("yes", "partial"):
                if self._type_enabled("proof"):
                    aug = await self._generate_proof(
                        insight_text,
                        analysis.proof_strategy,
                        analysis.proof_dependencies,
                    )
                    if aug:
                        # Verify proof if required
                        if self.types_config.get("proof", {}).get("require_review", True):
                            aug = await self._verify_proof(aug)
                        augmentations.append(aug)

            elif "code" in rec_lower and analysis.code_helpful:
                if self._type_enabled("code"):
                    aug = await self._generate_code(
                        insight_text,
                        analysis.code_purpose.value,
                        analysis.code_description,
                    )
                    if aug:
                        # Execute code if configured
                        if self.types_config.get("code", {}).get("execute", True):
                            aug = await self._execute_code(aug)
                        augmentations.append(aug)

        augmented.augmentations = augmentations
        augmented.augmented_at = datetime.now()

        # Save to disk
        augmented.save(self.data_dir)

        logger.info(
            f"[augmenter] Generated {len(augmentations)} augmentations for {insight_id}"
        )

        return augmented

    def _type_enabled(self, aug_type: str) -> bool:
        """Check if an augmentation type is enabled."""
        return self.types_config.get(aug_type, {}).get("enabled", True)

    async def _analyze_insight(
        self,
        insight: str,
        tags: list[str],
        dependencies: list[str],
        review_summary: str,
    ) -> AugmentationAnalysis:
        """Analyze what augmentations would help an insight."""
        prompt = build_analysis_prompt(insight, tags, dependencies, review_summary)

        try:
            response = await self.claude.send_message(prompt, extended_thinking=False)
            analysis = parse_analysis_response(response)
            logger.info(
                f"[augmenter] Analysis: recommendations={analysis.recommendations}"
            )
            return analysis
        except Exception as e:
            logger.error(f"[augmenter] Analysis failed: {e}")
            return AugmentationAnalysis()

    async def _generate_diagram(
        self,
        insight: str,
        diagram_type: str,
        diagram_description: str,
    ) -> Optional[Augmentation]:
        """Generate a diagram for an insight."""
        logger.info(f"[augmenter] Generating diagram ({diagram_type})")

        prompt = build_diagram_prompt(insight, diagram_type, diagram_description)

        try:
            response = await self.claude.send_message(prompt, extended_thinking=False)
            code, caption = parse_diagram_response(response)

            if not code:
                logger.warning("[augmenter] No diagram code generated")
                return None

            return Augmentation(
                id=str(uuid.uuid4())[:12],
                type=AugmentationType.DIAGRAM,
                content=code,
                caption=caption,
                generated_by="claude",
            )
        except Exception as e:
            logger.error(f"[augmenter] Diagram generation failed: {e}")
            return None

    async def _generate_table(
        self,
        insight: str,
        table_type: str,
        table_description: str,
    ) -> Optional[Augmentation]:
        """Generate a table for an insight."""
        logger.info(f"[augmenter] Generating table ({table_type})")

        prompt = build_table_prompt(insight, table_type, table_description)

        try:
            response = await self.claude.send_message(prompt, extended_thinking=False)
            table, notes, verification = parse_table_response(response)

            if not table:
                logger.warning("[augmenter] No table generated")
                return None

            # Combine table with notes
            content = table
            if notes:
                content += f"\n\n**Notes:** {notes}"
            if verification:
                content += f"\n\n**Verification:** {verification}"

            return Augmentation(
                id=str(uuid.uuid4())[:12],
                type=AugmentationType.TABLE,
                content=content,
                caption=notes or "Systematic enumeration",
                generated_by="claude",
            )
        except Exception as e:
            logger.error(f"[augmenter] Table generation failed: {e}")
            return None

    async def _generate_proof(
        self,
        insight: str,
        proof_strategy: str,
        proof_dependencies: list[str],
    ) -> Optional[Augmentation]:
        """Generate a formal proof for an insight."""
        logger.info(f"[augmenter] Generating proof")

        prompt = build_proof_prompt(insight, proof_strategy, proof_dependencies)

        try:
            response = await self.claude.send_message(prompt, extended_thinking=False)
            proof, notes = parse_proof_response(response)

            if not proof:
                logger.warning("[augmenter] No proof generated")
                return None

            return Augmentation(
                id=str(uuid.uuid4())[:12],
                type=AugmentationType.PROOF,
                content=proof,
                caption=notes or "Formal mathematical proof",
                generated_by="claude",
            )
        except Exception as e:
            logger.error(f"[augmenter] Proof generation failed: {e}")
            return None

    async def _verify_proof(self, aug: Augmentation) -> Augmentation:
        """Verify a generated proof using another LLM call."""
        logger.info(f"[augmenter] Verifying proof {aug.id}")

        # Extract theorem from proof
        lines = aug.content.split("\n")
        theorem = ""
        for line in lines:
            if line.strip().startswith("THEOREM:"):
                theorem = line.replace("THEOREM:", "").strip()
                break

        prompt = build_proof_verification_prompt(theorem, aug.content)

        try:
            response = await self.claude.send_message(prompt, extended_thinking=False)
            verdict, issues, suggestions = parse_verification_response(response)

            aug.verified = verdict == "valid"
            aug.verification_notes = f"Verdict: {verdict}"
            if issues and issues.lower() != "none":
                aug.verification_notes += f"\nIssues: {issues}"
            if suggestions and suggestions.lower() != "none":
                aug.verification_notes += f"\nSuggestions: {suggestions}"

            logger.info(f"[augmenter] Proof verification: {verdict}")
            return aug
        except Exception as e:
            logger.error(f"[augmenter] Proof verification failed: {e}")
            aug.verification_notes = f"Verification failed: {e}"
            return aug

    async def _generate_code(
        self,
        insight: str,
        code_purpose: str,
        code_description: str,
    ) -> Optional[Augmentation]:
        """Generate verification code for an insight."""
        logger.info(f"[augmenter] Generating code ({code_purpose})")

        prompt = build_code_prompt(insight, code_purpose, code_description)

        try:
            response = await self.claude.send_message(prompt, extended_thinking=False)
            code, expected, proves = parse_code_response(response)

            if not code:
                logger.warning("[augmenter] No code generated")
                return None

            caption = proves or expected or "Verification code"

            return Augmentation(
                id=str(uuid.uuid4())[:12],
                type=AugmentationType.CODE,
                content=code,
                caption=caption,
                generated_by="claude",
            )
        except Exception as e:
            logger.error(f"[augmenter] Code generation failed: {e}")
            return None

    async def _execute_code(self, aug: Augmentation) -> Augmentation:
        """Execute generated code and capture output."""
        logger.info(f"[augmenter] Executing code {aug.id}")

        timeout = self.types_config.get("code", {}).get("timeout_seconds", 30)

        try:
            # Write code to temp file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as f:
                f.write(aug.content)
                temp_path = f.name

            # Execute with timeout
            result = await asyncio.to_thread(
                self._run_python_code, temp_path, timeout
            )

            aug.execution_output = result["output"]
            aug.execution_success = result["success"]
            aug.verified = result["success"]

            if result["success"]:
                aug.verification_notes = "Code executed successfully"
                logger.info(f"[augmenter] Code execution successful")
            else:
                aug.verification_notes = f"Code execution failed: {result['error']}"
                logger.warning(f"[augmenter] Code execution failed: {result['error']}")

            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)

            return aug
        except Exception as e:
            logger.error(f"[augmenter] Code execution error: {e}")
            aug.execution_success = False
            aug.verification_notes = f"Execution error: {e}"
            return aug

    def _run_python_code(self, script_path: str, timeout: int) -> dict:
        """Run Python code and capture output."""
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode == 0:
                return {
                    "success": True,
                    "output": result.stdout,
                    "error": None,
                }
            else:
                return {
                    "success": False,
                    "output": result.stdout,
                    "error": result.stderr,
                }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": f"Timeout after {timeout} seconds",
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
            }


def get_augmenter(
    claude_reviewer,
    config: dict,
    data_dir: Path,
) -> Optional[Augmenter]:
    """
    Factory function to create an Augmenter.

    Args:
        claude_reviewer: ClaudeReviewer instance
        config: Full configuration dict
        data_dir: Base data directory

    Returns:
        Augmenter instance or None if not configured
    """
    aug_config = config.get("augmentation", {})

    if not aug_config.get("enabled", True):
        logger.info("[augmenter] Augmentation disabled in config")
        return None

    if not claude_reviewer or not claude_reviewer.is_available():
        logger.warning("[augmenter] Claude not available, augmentation disabled")
        return None

    return Augmenter(claude_reviewer, config, data_dir)
