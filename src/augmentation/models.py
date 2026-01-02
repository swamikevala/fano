"""
Augmentation data models.

Augmentations are supplementary materials (diagrams, tables, proofs, code)
that clarify and strengthen blessed insights.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
import json


class AugmentationType(str, Enum):
    """Types of augmentation."""
    DIAGRAM = "diagram"
    TABLE = "table"
    PROOF = "proof"
    CODE = "code"


class DiagramType(str, Enum):
    """Types of diagrams."""
    GRAPH = "graph"
    GEOMETRY = "geometry"
    MAPPING = "mapping"
    HIERARCHY = "hierarchy"
    FLOW = "flow"
    NONE = "none"


class TableType(str, Enum):
    """Types of tables."""
    MAPPING = "mapping"
    ENUMERATION = "enumeration"
    COMPARISON = "comparison"
    CORRESPONDENCE = "correspondence"
    NONE = "none"


class CodePurpose(str, Enum):
    """Purpose of generated code."""
    VERIFY = "verify"
    GENERATE = "generate"
    DEMONSTRATE = "demonstrate"
    NONE = "none"


@dataclass
class AugmentationAnalysis:
    """Analysis of what augmentations would help an insight."""

    # Diagram analysis
    diagram_helpful: bool = False
    diagram_type: DiagramType = DiagramType.NONE
    diagram_description: str = ""

    # Table analysis
    table_helpful: bool = False
    table_type: TableType = TableType.NONE
    table_description: str = ""

    # Proof analysis
    proof_possible: str = "no"  # "yes" / "partial" / "no"
    proof_strategy: str = ""
    proof_dependencies: list[str] = field(default_factory=list)

    # Code analysis
    code_helpful: bool = False
    code_purpose: CodePurpose = CodePurpose.NONE
    code_description: str = ""

    # Recommendations
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "diagram_helpful": self.diagram_helpful,
            "diagram_type": self.diagram_type.value if isinstance(self.diagram_type, DiagramType) else self.diagram_type,
            "diagram_description": self.diagram_description,
            "table_helpful": self.table_helpful,
            "table_type": self.table_type.value if isinstance(self.table_type, TableType) else self.table_type,
            "table_description": self.table_description,
            "proof_possible": self.proof_possible,
            "proof_strategy": self.proof_strategy,
            "proof_dependencies": self.proof_dependencies,
            "code_helpful": self.code_helpful,
            "code_purpose": self.code_purpose.value if isinstance(self.code_purpose, CodePurpose) else self.code_purpose,
            "code_description": self.code_description,
            "recommendations": self.recommendations,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AugmentationAnalysis":
        return cls(
            diagram_helpful=data.get("diagram_helpful", False),
            diagram_type=DiagramType(data.get("diagram_type", "none")),
            diagram_description=data.get("diagram_description", ""),
            table_helpful=data.get("table_helpful", False),
            table_type=TableType(data.get("table_type", "none")),
            table_description=data.get("table_description", ""),
            proof_possible=data.get("proof_possible", "no"),
            proof_strategy=data.get("proof_strategy", ""),
            proof_dependencies=data.get("proof_dependencies", []),
            code_helpful=data.get("code_helpful", False),
            code_purpose=CodePurpose(data.get("code_purpose", "none")),
            code_description=data.get("code_description", ""),
            recommendations=data.get("recommendations", []),
        )


@dataclass
class Augmentation:
    """A single augmentation attached to an insight."""

    id: str
    type: AugmentationType
    content: str  # The actual content (code, markdown, proof text)
    caption: str  # Brief description
    generated_by: str  # Which LLM generated it
    generated_at: datetime = field(default_factory=datetime.now)
    verified: bool = False
    verification_notes: str = ""
    file_path: Optional[str] = None  # For diagrams: path to generated image
    execution_output: Optional[str] = None  # For code: what it printed
    execution_success: Optional[bool] = None  # For code: did it run without error

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value if isinstance(self.type, AugmentationType) else self.type,
            "content": self.content,
            "caption": self.caption,
            "generated_by": self.generated_by,
            "generated_at": self.generated_at.isoformat(),
            "verified": self.verified,
            "verification_notes": self.verification_notes,
            "file_path": self.file_path,
            "execution_output": self.execution_output,
            "execution_success": self.execution_success,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Augmentation":
        return cls(
            id=data["id"],
            type=AugmentationType(data["type"]),
            content=data["content"],
            caption=data.get("caption", ""),
            generated_by=data.get("generated_by", "unknown"),
            generated_at=datetime.fromisoformat(data.get("generated_at", datetime.now().isoformat())),
            verified=data.get("verified", False),
            verification_notes=data.get("verification_notes", ""),
            file_path=data.get("file_path"),
            execution_output=data.get("execution_output"),
            execution_success=data.get("execution_success"),
        )

    def save_to_file(self, chunk_dir: Path):
        """Save augmentation content to appropriate file."""
        chunk_dir.mkdir(parents=True, exist_ok=True)

        if self.type == AugmentationType.DIAGRAM:
            # Save diagram generation code
            code_path = chunk_dir / "diagram.py"
            code_path.write_text(self.content, encoding="utf-8")

        elif self.type == AugmentationType.TABLE:
            # Save markdown table
            table_path = chunk_dir / "table.md"
            table_path.write_text(self.content, encoding="utf-8")

        elif self.type == AugmentationType.PROOF:
            # Save proof
            proof_path = chunk_dir / "proof.md"
            proof_path.write_text(self.content, encoding="utf-8")

        elif self.type == AugmentationType.CODE:
            # Save verification code
            code_path = chunk_dir / "verification.py"
            code_path.write_text(self.content, encoding="utf-8")


@dataclass
class AugmentedInsight:
    """An insight with its augmentations."""

    insight_id: str
    insight_text: str
    augmentations: list[Augmentation] = field(default_factory=list)
    analysis: Optional[AugmentationAnalysis] = None
    augmented_at: Optional[datetime] = None

    @property
    def has_diagram(self) -> bool:
        return any(a.type == AugmentationType.DIAGRAM for a in self.augmentations)

    @property
    def has_table(self) -> bool:
        return any(a.type == AugmentationType.TABLE for a in self.augmentations)

    @property
    def has_proof(self) -> bool:
        return any(a.type == AugmentationType.PROOF for a in self.augmentations)

    @property
    def has_code(self) -> bool:
        return any(a.type == AugmentationType.CODE for a in self.augmentations)

    @property
    def all_verified(self) -> bool:
        return all(a.verified for a in self.augmentations)

    def to_dict(self) -> dict:
        return {
            "insight_id": self.insight_id,
            "insight_text": self.insight_text,
            "augmentations": [a.to_dict() for a in self.augmentations],
            "analysis": self.analysis.to_dict() if self.analysis else None,
            "augmented_at": self.augmented_at.isoformat() if self.augmented_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AugmentedInsight":
        return cls(
            insight_id=data["insight_id"],
            insight_text=data["insight_text"],
            augmentations=[Augmentation.from_dict(a) for a in data.get("augmentations", [])],
            analysis=AugmentationAnalysis.from_dict(data["analysis"]) if data.get("analysis") else None,
            augmented_at=datetime.fromisoformat(data["augmented_at"]) if data.get("augmented_at") else None,
        )

    def save(self, base_dir: Path):
        """Save augmented insight to disk."""
        chunk_dir = base_dir / f"chunk_{self.insight_id}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        # Save main data
        data_path = chunk_dir / "augmentation.json"
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        # Save individual augmentation files
        for aug in self.augmentations:
            aug.save_to_file(chunk_dir)

    @classmethod
    def load(cls, chunk_dir: Path) -> Optional["AugmentedInsight"]:
        """Load augmented insight from disk."""
        data_path = chunk_dir / "augmentation.json"
        if not data_path.exists():
            return None

        with open(data_path, encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)
