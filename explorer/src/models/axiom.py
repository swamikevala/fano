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

    Priority:
    - 1-10 scale (10 = highest priority, explore first)
    - Default is 5 (medium priority)

    Images:
    - Optional list of image filenames attached to this seed
    - Images are stored in data/axioms/images/
    - Will be passed to LLMs during exploration (for vision-capable models)
    """
    id: str
    text: str
    type: str = "conjecture"  # axiom, conjecture, question
    priority: int = 5  # 1-10, higher = explore first
    tags: list[str] = field(default_factory=list)
    confidence: str = "high"  # high, medium, low (for conjectures)
    source: str = "user"  # Where this seed came from
    notes: str = ""  # Additional context
    images: list[str] = field(default_factory=list)  # Image filenames

    @classmethod
    def from_dict(cls, data: dict, index: int = 0) -> "SeedAphorism":
        # Determine type - default to conjecture for backward compatibility
        entry_type = data.get("type", "conjecture")

        # Parse priority (1-10 scale, default 5)
        priority = data.get("priority", 5)
        if isinstance(priority, str):
            # Support "high", "medium", "low" as aliases
            priority_map = {"high": 8, "medium": 5, "low": 2}
            priority = priority_map.get(priority.lower(), 5)
        priority = max(1, min(10, int(priority)))  # Clamp to 1-10

        return cls(
            id=data.get("id", f"seed-{index:03d}"),
            text=data.get("text", ""),
            type=entry_type,
            priority=priority,
            tags=data.get("tags", []),
            confidence=data.get("confidence", "high"),
            source=data.get("source", "user"),
            notes=data.get("notes", ""),
            images=data.get("images", []),
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
        self.images_dir = data_dir / "axioms" / "images"

        # Ensure directories exist
        self.excerpts_dir.mkdir(parents=True, exist_ok=True)
        self.blessed_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
    
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

    def get_seed_aphorisms(self, type_filter: str = None, sort_by_priority: bool = True) -> list[SeedAphorism]:
        """
        Load seed aphorisms from seeds.yaml.

        Args:
            type_filter: If provided, only return entries of this type
                        ('axiom', 'conjecture', 'question')
            sort_by_priority: If True, sort by priority (highest first)
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

        # Sort by priority (highest first)
        if sort_by_priority:
            seeds.sort(key=lambda s: s.priority, reverse=True)

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
        text_context, _ = self.get_context_with_images(max_seeds)
        return text_context

    def get_context_with_images(self, max_seeds: int = 10) -> tuple[str, list[Path]]:
        """
        Build context string and collect image paths for exploration prompts.

        Returns:
            tuple of (text_context, image_paths)
            - text_context: Formatted text for prompts
            - image_paths: List of Path objects for all seed images
        """
        lines = []
        all_images: list[Path] = []

        def add_seed_to_context(seed: SeedAphorism, marker: str, show_confidence: bool = False):
            """Helper to add a seed and its images to the context."""
            prefix = marker
            if show_confidence:
                confidence_marker = {"high": "⚡", "medium": "?", "low": "○"}.get(seed.confidence, "?")
                prefix = confidence_marker
            priority_marker = f"[P{seed.priority}]" if seed.priority != 5 else ""

            # Add text
            line = f"{prefix} {priority_marker} {seed.text}".strip()
            if seed.images:
                line += f" [📷 {len(seed.images)} image(s) attached]"
            lines.append(line)

            if seed.tags:
                lines.append(f"   [Tags: {', '.join(seed.tags)}]")
            if seed.notes:
                lines.append(f"   Note: {seed.notes}")

            # Collect images
            for filename in seed.images:
                image_path = self.images_dir / filename
                if image_path.exists():
                    all_images.append(image_path)

        # 1. AXIOMS - Assumed true facts (always included)
        axioms = self.get_axioms()
        if axioms:
            lines.append("=== AXIOMS (ASSUMED TRUE) ===")
            lines.append("These are established facts. Do NOT re-derive or question these - take them as given:\n")
            for axiom in axioms:
                add_seed_to_context(axiom, "•")
            lines.append("")

        # 2. CONJECTURES - To explore and verify (sorted by priority)
        conjectures = self.get_conjectures()[:max_seeds]
        if conjectures:
            lines.append("=== CONJECTURES (TO EXPLORE) ===")
            lines.append("These are conjectured connections to explore, verify, and build upon:\n")
            for seed in conjectures:
                add_seed_to_context(seed, "", show_confidence=True)
            lines.append("")

        # 3. QUESTIONS - Specific questions to answer (sorted by priority)
        questions = self.get_questions()
        if questions:
            lines.append("=== QUESTIONS (TO ANSWER) ===")
            lines.append("These are specific questions that need answers (highest priority first):\n")
            for q in questions:
                add_seed_to_context(q, "❓")
            lines.append("")

        # Add note about images if any exist
        if all_images:
            lines.append(f"[Note: {len(all_images)} reference image(s) are attached to provide visual context for the seeds above.]")
            lines.append("")

        return "\n".join(lines), all_images
    
    def get_unexplained_numbers(self) -> list[str]:
        """Get number sets that haven't been explained yet."""
        all_numbers = {ns.id for ns in self.get_target_numbers()}
        explained = set()
        for insight in self.get_blessed_insights():
            explained.update(insight.numbers_explained)
        return list(all_numbers - explained)

    def save_seeds(self, seeds: list[SeedAphorism]) -> None:
        """
        Save all seeds to the seeds.yaml file.

        Args:
            seeds: List of SeedAphorism objects to save
        """
        # Build the YAML structure
        seeds_data = []
        for seed in seeds:
            seed_dict = {"text": seed.text}
            # Always save ID to ensure persistence across loads
            seed_dict["id"] = seed.id
            if seed.type != "conjecture":
                seed_dict["type"] = seed.type
            if seed.priority != 5:
                seed_dict["priority"] = seed.priority
            if seed.tags:
                seed_dict["tags"] = seed.tags
            if seed.confidence != "high":
                seed_dict["confidence"] = seed.confidence
            if seed.source and seed.source != "user":
                seed_dict["source"] = seed.source
            if seed.notes:
                seed_dict["notes"] = seed.notes
            if seed.images:
                seed_dict["images"] = seed.images
            seeds_data.append(seed_dict)

        # Create the full YAML content with header comments
        header = """# Seeds Configuration
# ===================
# This file contains three types of entries:
#
# 1. AXIOMS (type: axiom)
#    - Assumed TRUE facts that don't need to be re-discovered
#    - Always included in every exploration as "given"
#    - The system will NOT question these
#
# 2. CONJECTURES (type: conjecture) [DEFAULT]
#    - Ideas to explore and verify
#    - The system will try to develop and validate these
#    - Use confidence: high/medium/low to indicate certainty
#
# 3. QUESTIONS (type: question)
#    - Specific questions you want answered
#    - The system will focus exploration on finding answers
#
# Format:
#   - text: The statement or question
#     type: axiom | conjecture | question  (default: conjecture)
#     tags: [domain1, domain2]  # Helps with categorization
#     confidence: high/medium/low  # For conjectures only
#     source: Where this idea came from (optional)
#     notes: Additional context (optional)
#     images: [image1.png, image2.jpg]  # Attached images (optional)

"""
        yaml_content = yaml.dump(
            {"seeds": seeds_data},
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=120,
        )

        self.seeds_file.parent.mkdir(parents=True, exist_ok=True)
        self.seeds_file.write_text(header + yaml_content, encoding="utf-8")

    def add_seed(self, seed: SeedAphorism) -> None:
        """
        Add a new seed to the seeds file.

        Args:
            seed: The SeedAphorism to add
        """
        seeds = self.get_seed_aphorisms(sort_by_priority=False)
        seeds.append(seed)
        self.save_seeds(seeds)

    def update_seed(self, seed_id: str, updates: dict) -> bool:
        """
        Update an existing seed.

        Args:
            seed_id: The ID of the seed to update
            updates: Dictionary of fields to update

        Returns:
            True if seed was found and updated, False otherwise
        """
        seeds = self.get_seed_aphorisms(sort_by_priority=False)
        for i, seed in enumerate(seeds):
            if seed.id == seed_id:
                # Apply updates
                if "text" in updates:
                    seed.text = updates["text"]
                if "type" in updates:
                    seed.type = updates["type"]
                if "priority" in updates:
                    priority = updates["priority"]
                    if isinstance(priority, str):
                        priority_map = {"high": 8, "medium": 5, "low": 2}
                        priority = priority_map.get(priority.lower(), 5)
                    seed.priority = max(1, min(10, int(priority)))
                if "tags" in updates:
                    seed.tags = updates["tags"]
                if "confidence" in updates:
                    seed.confidence = updates["confidence"]
                if "source" in updates:
                    seed.source = updates["source"]
                if "notes" in updates:
                    seed.notes = updates["notes"]
                if "images" in updates:
                    seed.images = updates["images"]

                seeds[i] = seed
                self.save_seeds(seeds)
                return True
        return False

    def delete_seed(self, seed_id: str) -> bool:
        """
        Delete a seed by its ID.
        Also removes any associated images.

        Args:
            seed_id: The ID of the seed to delete

        Returns:
            True if seed was found and deleted, False otherwise
        """
        # First, clean up any associated images
        self.delete_seed_images(seed_id)

        seeds = self.get_seed_aphorisms(sort_by_priority=False)
        original_count = len(seeds)
        seeds = [s for s in seeds if s.id != seed_id]

        if len(seeds) < original_count:
            self.save_seeds(seeds)
            return True
        return False

    def get_seed_by_id(self, seed_id: str) -> Optional[SeedAphorism]:
        """
        Get a specific seed by its ID.

        Args:
            seed_id: The ID of the seed to retrieve

        Returns:
            The SeedAphorism if found, None otherwise
        """
        seeds = self.get_seed_aphorisms(sort_by_priority=False)
        for seed in seeds:
            if seed.id == seed_id:
                return seed
        return None

    # --- Image Management Methods ---

    def get_seed_image_path(self, seed_id: str, filename: str) -> Path:
        """
        Get the full path to a seed image file.

        Args:
            seed_id: The seed ID
            filename: The image filename

        Returns:
            Full Path to the image file
        """
        return self.images_dir / f"{seed_id}_{filename}"

    def add_seed_image(self, seed_id: str, filename: str, image_data: bytes) -> Optional[str]:
        """
        Add an image to a seed.

        Args:
            seed_id: The seed ID to add the image to
            filename: Original filename of the image
            image_data: Raw image bytes

        Returns:
            The stored filename if successful, None if seed not found
        """
        seed = self.get_seed_by_id(seed_id)
        if not seed:
            return None

        # Create a unique filename: seed_id + original filename
        stored_filename = f"{seed_id}_{filename}"
        image_path = self.images_dir / stored_filename

        # Write the image file
        image_path.write_bytes(image_data)

        # Update the seed's images list
        if stored_filename not in seed.images:
            seed.images.append(stored_filename)
            self.update_seed(seed_id, {"images": seed.images})

        return stored_filename

    def remove_seed_image(self, seed_id: str, filename: str) -> bool:
        """
        Remove an image from a seed.

        Args:
            seed_id: The seed ID
            filename: The stored filename to remove

        Returns:
            True if image was removed, False otherwise
        """
        seed = self.get_seed_by_id(seed_id)
        if not seed:
            return False

        if filename not in seed.images:
            return False

        # Remove the file
        image_path = self.images_dir / filename
        if image_path.exists():
            image_path.unlink()

        # Update the seed's images list
        seed.images.remove(filename)
        self.update_seed(seed_id, {"images": seed.images})

        return True

    def get_seed_images(self, seed_id: str) -> list[Path]:
        """
        Get all image paths for a seed.

        Args:
            seed_id: The seed ID

        Returns:
            List of Path objects to existing image files
        """
        seed = self.get_seed_by_id(seed_id)
        if not seed:
            return []

        paths = []
        for filename in seed.images:
            image_path = self.images_dir / filename
            if image_path.exists():
                paths.append(image_path)

        return paths

    def delete_seed_images(self, seed_id: str) -> int:
        """
        Delete all images associated with a seed.
        Called when deleting a seed to clean up orphaned images.

        Args:
            seed_id: The seed ID

        Returns:
            Number of images deleted
        """
        seed = self.get_seed_by_id(seed_id)
        if not seed:
            return 0

        deleted = 0
        for filename in seed.images:
            image_path = self.images_dir / filename
            if image_path.exists():
                image_path.unlink()
                deleted += 1

        return deleted

    def get_all_seed_images(self) -> list[tuple[str, str, Path]]:
        """
        Get all images across all seeds.

        Returns:
            List of (seed_id, filename, path) tuples
        """
        result = []
        for seed in self.get_seed_aphorisms(sort_by_priority=False):
            for filename in seed.images:
                image_path = self.images_dir / filename
                if image_path.exists():
                    result.append((seed.id, filename, image_path))
        return result
