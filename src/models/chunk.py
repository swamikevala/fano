"""
Research Chunk model.

A chunk is a synthesized, written-up piece of research that
is ready for human review. It represents a coherent finding
or insight that emerged from exploration.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


class ChunkStatus(Enum):
    PENDING = "pending"          # Awaiting human review
    PROFOUND = "profound"        # ⚡ Human marked as profound
    INTERESTING = "interesting"  # ? Human marked as interesting but uncertain
    REJECTED = "rejected"        # ✗ Human rejected


class ChunkFeedback(Enum):
    PROFOUND = "profound"        # ⚡ "This is real"
    INTERESTING = "interesting"  # ? "Not sure yet"
    REJECTED = "rejected"        # ✗ "This isn't right"


@dataclass
class Chunk:
    """
    A synthesized research chunk ready for human review.
    """
    id: str
    title: str
    thread_id: str  # Source thread
    content: str    # The actual write-up (markdown)
    summary: str    # One-line summary
    
    # What this chunk attempts to explain
    target_numbers_addressed: list[str]
    
    # Status tracking
    status: ChunkStatus
    created_at: datetime
    reviewed_at: Optional[datetime] = None
    
    # Human feedback
    feedback: Optional[ChunkFeedback] = None
    feedback_notes: str = ""  # Optional human notes
    
    # Metrics from orchestrator
    profundity_score: float = 0.0  # 0-1 based on signal words
    critique_rounds: int = 0
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "thread_id": self.thread_id,
            "content": self.content,
            "summary": self.summary,
            "target_numbers_addressed": self.target_numbers_addressed,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "feedback": self.feedback.value if self.feedback else None,
            "feedback_notes": self.feedback_notes,
            "profundity_score": self.profundity_score,
            "critique_rounds": self.critique_rounds,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(
            id=data["id"],
            title=data["title"],
            thread_id=data["thread_id"],
            content=data["content"],
            summary=data["summary"],
            target_numbers_addressed=data["target_numbers_addressed"],
            status=ChunkStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            reviewed_at=datetime.fromisoformat(data["reviewed_at"]) if data.get("reviewed_at") else None,
            feedback=ChunkFeedback(data["feedback"]) if data.get("feedback") else None,
            feedback_notes=data.get("feedback_notes", ""),
            profundity_score=data.get("profundity_score", 0.0),
            critique_rounds=data.get("critique_rounds", 0),
        )
    
    @classmethod
    def create_from_thread(
        cls,
        thread_id: str,
        title: str,
        content: str,
        summary: str,
        target_numbers: list[str],
        profundity_score: float = 0.0,
        critique_rounds: int = 0,
    ) -> "Chunk":
        """Create a new chunk from synthesis."""
        return cls(
            id=str(uuid.uuid4())[:12],
            title=title,
            thread_id=thread_id,
            content=content,
            summary=summary,
            target_numbers_addressed=target_numbers,
            status=ChunkStatus.PENDING,
            created_at=datetime.now(),
            profundity_score=profundity_score,
            critique_rounds=critique_rounds,
        )
    
    def apply_feedback(self, feedback: ChunkFeedback, notes: str = ""):
        """Apply human feedback to this chunk."""
        self.feedback = feedback
        self.feedback_notes = notes
        self.reviewed_at = datetime.now()
        
        # Update status based on feedback
        status_map = {
            ChunkFeedback.PROFOUND: ChunkStatus.PROFOUND,
            ChunkFeedback.INTERESTING: ChunkStatus.INTERESTING,
            ChunkFeedback.REJECTED: ChunkStatus.REJECTED,
        }
        self.status = status_map[feedback]
    
    def save(self, base_dir: Path):
        """Save chunk to appropriate directory based on status."""
        status_dirs = {
            ChunkStatus.PENDING: "pending",
            ChunkStatus.PROFOUND: "profound",
            ChunkStatus.INTERESTING: "interesting",
            ChunkStatus.REJECTED: "rejected",
        }
        
        chunk_dir = base_dir / "chunks" / status_dirs[self.status]
        chunk_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as markdown for readability
        md_path = chunk_dir / f"{self.id}.md"
        with open(md_path, "w") as f:
            f.write(self.to_markdown())
        
        # Also save JSON for structured access
        json_path = chunk_dir / f"{self.id}.json"
        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def move_to_status(self, base_dir: Path, new_status: ChunkStatus):
        """Move chunk files when status changes."""
        old_dir = base_dir / "chunks" / self.status.value
        new_dir = base_dir / "chunks" / new_status.value
        new_dir.mkdir(parents=True, exist_ok=True)
        
        for ext in [".md", ".json"]:
            old_path = old_dir / f"{self.id}{ext}"
            new_path = new_dir / f"{self.id}{ext}"
            if old_path.exists():
                old_path.rename(new_path)
        
        self.status = new_status
    
    @classmethod
    def load(cls, json_path: Path) -> "Chunk":
        """Load chunk from JSON file."""
        with open(json_path) as f:
            return cls.from_dict(json.load(f))
    
    def to_markdown(self) -> str:
        """Export chunk as markdown."""
        feedback_emoji = {
            ChunkFeedback.PROFOUND: "⚡",
            ChunkFeedback.INTERESTING: "?",
            ChunkFeedback.REJECTED: "✗",
        }
        
        lines = [
            f"# {self.title}",
            "",
            f"**ID:** `{self.id}`",
            f"**Thread:** `{self.thread_id}`",
            f"**Created:** {self.created_at.strftime('%Y-%m-%d %H:%M')}",
            f"**Status:** {self.status.value}",
        ]
        
        if self.feedback:
            emoji = feedback_emoji.get(self.feedback, "")
            lines.append(f"**Feedback:** {emoji} {self.feedback.value}")
            if self.feedback_notes:
                lines.append(f"**Notes:** {self.feedback_notes}")
        
        lines.extend([
            "",
            f"**Summary:** {self.summary}",
            "",
            f"**Numbers Addressed:** {', '.join(self.target_numbers_addressed)}",
            "",
            f"**Profundity Score:** {self.profundity_score:.2f}",
            f"**Critique Rounds:** {self.critique_rounds}",
            "",
            "---",
            "",
            self.content,
        ])
        
        return "\n".join(lines)
