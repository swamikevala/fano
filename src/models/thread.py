"""
Exploration Thread model.

An exploration thread is a single line of inquiry that may span
multiple exchanges with multiple LLMs. Threads can be:
- active (being explored)
- paused (waiting for something)
- chunk_ready (mature enough for synthesis)
- archived (done)
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


class ThreadStatus(Enum):
    ACTIVE = "active"
    PAUSED = "paused"  
    CHUNK_READY = "chunk_ready"
    ARCHIVED = "archived"


class ExchangeRole(Enum):
    EXPLORER = "explorer"  # Generating ideas
    CRITIC = "critic"      # Critiquing ideas
    SYNTHESIZER = "synthesizer"  # Combining/summarizing


@dataclass
class Exchange:
    """A single exchange in the thread."""
    id: str
    timestamp: datetime
    role: ExchangeRole
    model: str  # chatgpt, gemini, claude
    prompt: str
    response: str
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "role": self.role.value,
            "model": self.model,
            "prompt": self.prompt,
            "response": self.response,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Exchange":
        return cls(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            role=ExchangeRole(data["role"]),
            model=data["model"],
            prompt=data["prompt"],
            response=data["response"],
        )


@dataclass
class ExplorationThread:
    """
    An exploration thread tracking a line of mathematical inquiry.
    """
    id: str
    topic: str  # Brief description of what's being explored
    seed_axioms: list[str]  # IDs of axioms that seeded this thread
    target_numbers: list[str]  # Which number sets we're trying to decode
    status: ThreadStatus
    created_at: datetime
    updated_at: datetime
    exchanges: list[Exchange] = field(default_factory=list)
    notes: str = ""  # Orchestrator notes
    
    @property
    def exchange_count(self) -> int:
        return len(self.exchanges)
    
    @property
    def needs_exploration(self) -> bool:
        """Does this thread need more exploration?"""
        if self.status != ThreadStatus.ACTIVE:
            return False
        # Alternate between exploration and critique
        if not self.exchanges:
            return True
        last = self.exchanges[-1]
        return last.role == ExchangeRole.CRITIC
    
    @property
    def needs_critique(self) -> bool:
        """Does this thread need critique?"""
        if self.status != ThreadStatus.ACTIVE:
            return False
        if not self.exchanges:
            return False
        last = self.exchanges[-1]
        return last.role == ExchangeRole.EXPLORER
    
    def add_exchange(self, role: ExchangeRole, model: str, prompt: str, response: str):
        """Add an exchange to the thread."""
        exchange = Exchange(
            id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(),
            role=role,
            model=model,
            prompt=prompt,
            response=response,
        )
        self.exchanges.append(exchange)
        self.updated_at = datetime.now()
    
    def get_context_for_prompt(self, max_exchanges: int = 6) -> str:
        """
        Get recent context for building the next prompt.
        Returns a formatted string of recent exchanges.
        """
        recent = self.exchanges[-max_exchanges:]
        lines = []
        for ex in recent:
            role_label = {
                ExchangeRole.EXPLORER: "EXPLORATION",
                ExchangeRole.CRITIC: "CRITIQUE",
                ExchangeRole.SYNTHESIZER: "SYNTHESIS",
            }[ex.role]
            lines.append(f"=== {role_label} ({ex.model}) ===")
            lines.append(ex.response)
            lines.append("")
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "topic": self.topic,
            "seed_axioms": self.seed_axioms,
            "target_numbers": self.target_numbers,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "exchanges": [e.to_dict() for e in self.exchanges],
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ExplorationThread":
        return cls(
            id=data["id"],
            topic=data["topic"],
            seed_axioms=data["seed_axioms"],
            target_numbers=data["target_numbers"],
            status=ThreadStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            exchanges=[Exchange.from_dict(e) for e in data["exchanges"]],
            notes=data.get("notes", ""),
        )
    
    def save(self, base_dir: Path):
        """Save thread to disk."""
        thread_dir = base_dir / "explorations"
        thread_dir.mkdir(parents=True, exist_ok=True)
        filepath = thread_dir / f"{self.id}.json"
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Path) -> "ExplorationThread":
        """Load thread from disk."""
        with open(filepath) as f:
            return cls.from_dict(json.load(f))
    
    @classmethod
    def create_new(
        cls,
        topic: str,
        seed_axioms: list[str],
        target_numbers: list[str],
    ) -> "ExplorationThread":
        """Create a new exploration thread."""
        now = datetime.now()
        return cls(
            id=str(uuid.uuid4())[:12],
            topic=topic,
            seed_axioms=seed_axioms,
            target_numbers=target_numbers,
            status=ThreadStatus.ACTIVE,
            created_at=now,
            updated_at=now,
        )
    
    def to_markdown(self) -> str:
        """Export thread as markdown for human reading."""
        lines = [
            f"# Exploration: {self.topic}",
            "",
            f"**ID:** {self.id}",
            f"**Status:** {self.status.value}",
            f"**Created:** {self.created_at.strftime('%Y-%m-%d %H:%M')}",
            f"**Updated:** {self.updated_at.strftime('%Y-%m-%d %H:%M')}",
            "",
            f"**Seed Axioms:** {', '.join(self.seed_axioms)}",
            f"**Target Numbers:** {', '.join(self.target_numbers)}",
            "",
            "---",
            "",
        ]
        
        for ex in self.exchanges:
            role_emoji = {
                ExchangeRole.EXPLORER: "🔭",
                ExchangeRole.CRITIC: "🔍",
                ExchangeRole.SYNTHESIZER: "✨",
            }[ex.role]
            
            lines.append(f"## {role_emoji} {ex.role.value.title()} ({ex.model})")
            lines.append(f"*{ex.timestamp.strftime('%Y-%m-%d %H:%M')}*")
            lines.append("")
            lines.append(f"**Prompt:**")
            lines.append(f"> {ex.prompt[:200]}..." if len(ex.prompt) > 200 else f"> {ex.prompt}")
            lines.append("")
            lines.append(f"**Response:**")
            lines.append(ex.response)
            lines.append("")
            lines.append("---")
            lines.append("")
        
        if self.notes:
            lines.append("## Orchestrator Notes")
            lines.append(self.notes)
        
        return "\n".join(lines)
