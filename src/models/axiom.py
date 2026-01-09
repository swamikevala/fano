"""
Axiom model.

Axioms are the foundational truths that seed and constrain exploration:
- Sadhguru excerpts (source texts)
- Target numbers (to be decoded)
- Blessed insights (⚡ chunks that have been validated)
"""

import uuid
import yaml
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class SourceExcerpt:
    """A source text from Sadhguru."""
    id: str
    title: str
    content: str
    source: str  # Book, video, talk, etc.
    tags: list[str] = field(default_factory=list)
    numbers_mentioned: list[str] = field(default_factory=list)
    
    @classmethod
    def from_markdown(cls, filepath: Path) -> "SourceExcerpt":
        """
        Load from markdown file with YAML frontmatter.
        
        Expected format:
        ---
        title: On the Five Elements
        source: Inner Engineering, Ch. 4
        tags: [pancha_bhuta, body, elements]
        numbers: [72, 12, 4, 6, 6]
        ---
        
        The actual excerpt content here...
        """
        text = filepath.read_text(encoding="utf-8")
        
        # Parse YAML frontmatter
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                frontmatter = yaml.safe_load(parts[1])
                content = parts[2].strip()
            else:
                frontmatter = {}
                content = text
        else:
            frontmatter = {}
            content = text
        
        return cls(
            id=filepath.stem,
            title=frontmatter.get("title", filepath.stem),
            content=content,
            source=frontmatter.get("source", "Unknown"),
            tags=frontmatter.get("tags", []),
            numbers_mentioned=[str(n) for n in frontmatter.get("numbers", [])],
        )


@dataclass
class TargetNumberSet:
    """A set of numbers to decode from the teachings."""
    id: str
    description: str
    source: str
    numbers: dict[str, float]  # e.g. {"water": 72, "earth": 12}
    total: Optional[float] = None
    notes: str = ""
    
    @classmethod
    def from_dict(cls, id: str, data: dict) -> "TargetNumberSet":
        return cls(
            id=id,
            description=data.get("description", ""),
            source=data.get("source", ""),
            numbers=data.get("numbers", {}),
            total=data.get("total"),
            notes=data.get("notes", ""),
        )


@dataclass
class SeedAphorism:
    """
    A seed entry provided by the user to guide exploration.

    Types:
    - axiom: Assumed true facts that don't need to be re-discovered
    - conjecture: Ideas to explore and verify (default)
    - question: Specific questions to answer
    """
    id: str
    text: str
    type: str = "conjecture"  # axiom, conjecture, question
    tags: list[str] = field(default_factory=list)
    confidence: str = "high"  # high, medium, low (for conjectures)
    source: str = "user"  # Where this seed came from
    notes: str = ""  # Additional context

    @classmethod
    def from_dict(cls, data: dict, index: int = 0) -> "SeedAphorism":
        # Determine type - default to conjecture for backward compatibility
        entry_type = data.get("type", "conjecture")

        return cls(
            id=data.get("id", f"seed-{index:03d}"),
            text=data.get("text", ""),
            type=entry_type,
            tags=data.get("tags", []),
            confidence=data.get("confidence", "high"),
            source=data.get("source", "user"),
            notes=data.get("notes", ""),
        )


@dataclass
class BlessedInsight:
    """
    A chunk that has been marked as ⚡ Profound.
    These become part of the axiom store for future exploration.
    """
    id: str
    title: str
    summary: str
    content: str
    source_chunk_id: str
    blessed_at: datetime
    numbers_explained: list[str]
    
    @classmethod
    def from_chunk(cls, chunk) -> "BlessedInsight":
        """Create blessed insight from a profound chunk."""
        return cls(
            id=str(uuid.uuid4())[:12],
            title=chunk.title,
            summary=chunk.summary,
            content=chunk.content,
            source_chunk_id=chunk.id,
            blessed_at=datetime.now(),
            numbers_explained=chunk.target_numbers_addressed,
        )


class AxiomStore:
    """
    Manager for all axioms: excerpts, numbers, and blessed insights.
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.excerpts_dir = data_dir / "axioms" / "sadhguru_excerpts"
        self.numbers_file = data_dir / "axioms" / "target_numbers.yaml"
        self.blessed_dir = data_dir / "axioms" / "blessed_insights"
        self.seeds_file = data_dir / "axioms" / "seeds.yaml"

        # Ensure directories exist
        self.excerpts_dir.mkdir(parents=True, exist_ok=True)
        self.blessed_dir.mkdir(parents=True, exist_ok=True)
    
    def get_excerpts(self) -> list[SourceExcerpt]:
        """Load all source excerpts."""
        excerpts = []
        for filepath in self.excerpts_dir.glob("*.md"):
            try:
                excerpts.append(SourceExcerpt.from_markdown(filepath))
            except Exception as e:
                print(f"Warning: Could not load {filepath}: {e}")
        return excerpts
    
    def get_target_numbers(self) -> list[TargetNumberSet]:
        """Load target numbers configuration."""
        if not self.numbers_file.exists():
            return []

        with open(self.numbers_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        numbers = []
        for key, value in data.items():
            if isinstance(value, dict) and "numbers" in value:
                numbers.append(TargetNumberSet.from_dict(key, value))

        return numbers

    def get_seed_aphorisms(self, type_filter: str = None) -> list[SeedAphorism]:
        """
        Load seed aphorisms from seeds.yaml.

        Args:
            type_filter: If provided, only return entries of this type
                        ('axiom', 'conjecture', 'question')
        """
        if not self.seeds_file.exists():
            return []

        with open(self.seeds_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            return []

        seeds = []
        seed_list = data.get("seeds", [])
        for i, seed_data in enumerate(seed_list):
            if isinstance(seed_data, dict) and seed_data.get("text"):
                seed = SeedAphorism.from_dict(seed_data, i)
                if type_filter is None or seed.type == type_filter:
                    seeds.append(seed)

        return seeds

    def get_axioms(self) -> list[SeedAphorism]:
        """Get all axioms (assumed true facts)."""
        return self.get_seed_aphorisms(type_filter="axiom")

    def get_conjectures(self) -> list[SeedAphorism]:
        """Get all conjectures (to explore and verify)."""
        return self.get_seed_aphorisms(type_filter="conjecture")

    def get_questions(self) -> list[SeedAphorism]:
        """Get all questions (to answer)."""
        return self.get_seed_aphorisms(type_filter="question")
    
    def get_blessed_insights(self) -> list[BlessedInsight]:
        """Load all blessed insights."""
        insights = []
        for filepath in self.blessed_dir.glob("*.md"):
            # Parse similar to excerpts
            text = filepath.read_text(encoding="utf-8")
            if text.startswith("---"):
                parts = text.split("---", 2)
                if len(parts) >= 3:
                    fm = yaml.safe_load(parts[1])
                    content = parts[2].strip()
                    insights.append(BlessedInsight(
                        id=filepath.stem,
                        title=fm.get("title", ""),
                        summary=fm.get("summary", ""),
                        content=content,
                        source_chunk_id=fm.get("source_chunk", ""),
                        blessed_at=datetime.fromisoformat(fm.get("blessed_at", datetime.now().isoformat())),
                        numbers_explained=fm.get("numbers_explained", []),
                    ))
        return insights
    
    def add_blessed_insight(self, insight: BlessedInsight):
        """Save a new blessed insight."""
        filepath = self.blessed_dir / f"{insight.id}.md"
        
        frontmatter = {
            "title": insight.title,
            "summary": insight.summary,
            "source_chunk": insight.source_chunk_id,
            "blessed_at": insight.blessed_at.isoformat(),
            "numbers_explained": insight.numbers_explained,
        }
        
        content = f"---\n{yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)}---\n\n{insight.content}"
        filepath.write_text(content, encoding="utf-8")
    
    def get_context_for_exploration(self, max_seeds: int = 10) -> str:
        """
        Build context string for exploration prompts.

        Includes three types of entries:
        - Axioms: Assumed true facts (always included as given)
        - Conjectures: Ideas to explore and verify
        - Questions: Specific questions to answer
        """
        lines = []

        # 1. AXIOMS - Assumed true facts (always included)
        axioms = self.get_axioms()
        if axioms:
            lines.append("=== AXIOMS (ASSUMED TRUE) ===")
            lines.append("These are established facts. Do NOT re-derive or question these - take them as given:\n")
            for axiom in axioms:
                lines.append(f"• {axiom.text}")
                if axiom.notes:
                    lines.append(f"   Note: {axiom.notes}")
            lines.append("")

        # 2. CONJECTURES - To explore and verify
        conjectures = self.get_conjectures()[:max_seeds]
        if conjectures:
            lines.append("=== CONJECTURES (TO EXPLORE) ===")
            lines.append("These are conjectured connections to explore, verify, and build upon:\n")
            for seed in conjectures:
                confidence_marker = {"high": "⚡", "medium": "?", "low": "○"}.get(seed.confidence, "?")
                lines.append(f"{confidence_marker} {seed.text}")
                if seed.tags:
                    lines.append(f"   [Tags: {', '.join(seed.tags)}]")
                if seed.notes:
                    lines.append(f"   Note: {seed.notes}")
            lines.append("")

        # 3. QUESTIONS - Specific questions to answer
        questions = self.get_questions()
        if questions:
            lines.append("=== QUESTIONS (TO ANSWER) ===")
            lines.append("These are specific questions that need answers:\n")
            for q in questions:
                lines.append(f"❓ {q.text}")
                if q.tags:
                    lines.append(f"   [Tags: {', '.join(q.tags)}]")
                if q.notes:
                    lines.append(f"   Context: {q.notes}")
            lines.append("")

        return "\n".join(lines)
    
    def get_unexplained_numbers(self) -> list[str]:
        """Get number sets that haven't been explained yet."""
        all_numbers = {ns.id for ns in self.get_target_numbers()}
        explained = set()
        for insight in self.get_blessed_insights():
            explained.update(insight.numbers_explained)
        return list(all_numbers - explained)
