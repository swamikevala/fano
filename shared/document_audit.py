"""
Document Audit Module for Fano.

Analyzes an existing document for:
- Duplicate/redundant sections
- User comments (resolved and unresolved)
- Formatting issues (math delimiters, markdown)
- Structural problems (numbering, flow)

This is separate from the deduplication module which prevents NEW duplicates.
The audit module analyzes EXISTING content.

Usage:
    from shared.document_audit import DocumentAuditor, AuditReport

    auditor = DocumentAuditor(llm_callback=my_llm_callback)
    report = await auditor.audit_file("document/main.md")

    # Or audit raw content
    report = await auditor.audit_content(markdown_text)

    print(report.to_json())
"""

import hashlib
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Callable, Awaitable

# Add shared module to path
SHARED_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(SHARED_PATH.parent))

from shared.logging import get_logger
from shared.deduplication import (
    DeduplicationChecker,
    ContentItem,
    ContentType,
    LLMCallback,
)

log = get_logger("shared", "document_audit")


# =============================================================================
# Data Classes
# =============================================================================


class IssueSeverity(Enum):
    """Severity levels for audit issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class IssueType(Enum):
    """Types of issues the auditor can find."""
    DUPLICATE_CONTENT = "duplicate_content"
    SIMILAR_CONTENT = "similar_content"
    UNRESOLVED_COMMENT = "unresolved_comment"
    RESOLVED_COMMENT = "resolved_comment"
    MATH_FORMATTING = "math_formatting"
    MARKDOWN_FORMATTING = "markdown_formatting"
    SECTION_NUMBERING = "section_numbering"
    MISSING_CONTENT = "missing_content"
    STRUCTURAL = "structural"


@dataclass
class DocumentSection:
    """A section extracted from the document."""
    id: str
    title: str
    content: str
    level: int  # Header level (1-6)
    start_line: int
    end_line: int
    word_count: int = 0

    def __post_init__(self):
        if self.word_count == 0:
            self.word_count = len(self.content.split())


@dataclass
class UserComment:
    """A user comment found in the document."""
    id: str
    text: str
    line_number: int
    context: str  # Surrounding text
    resolved: bool = False
    comment_type: str = "inline"  # "inline" or "annotation"


@dataclass
class AuditIssue:
    """A single issue found during audit."""
    type: IssueType
    severity: IssueSeverity
    message: str
    location: Optional[str] = None  # Section ID or line number
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "severity": self.severity.value,
            "message": self.message,
            "location": self.location,
            "details": self.details,
        }


@dataclass
class DuplicatePair:
    """A pair of sections identified as duplicates."""
    section1_id: str
    section2_id: str
    section1_title: str
    section2_title: str
    similarity_reason: str
    recommendation: str  # "merge", "remove_first", "remove_second", "review"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AuditReport:
    """Complete audit report for a document."""
    document_path: str
    audit_time: str
    total_sections: int
    total_words: int

    # Findings
    duplicates: list[DuplicatePair] = field(default_factory=list)
    comments: list[UserComment] = field(default_factory=list)
    issues: list[AuditIssue] = field(default_factory=list)

    # Summary counts
    duplicate_count: int = 0
    unresolved_comment_count: int = 0
    formatting_issue_count: int = 0

    def to_dict(self) -> dict:
        return {
            "document_path": self.document_path,
            "audit_time": self.audit_time,
            "summary": {
                "total_sections": self.total_sections,
                "total_words": self.total_words,
                "duplicate_pairs": self.duplicate_count,
                "unresolved_comments": self.unresolved_comment_count,
                "formatting_issues": self.formatting_issue_count,
            },
            "duplicates": [d.to_dict() for d in self.duplicates],
            "comments": [asdict(c) for c in self.comments],
            "issues": [i.to_dict() for i in self.issues],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def has_issues(self) -> bool:
        return bool(self.duplicates or self.issues or
                    any(not c.resolved for c in self.comments))


# =============================================================================
# Document Parser
# =============================================================================


class DocumentParser:
    """Parses markdown documents into sections and extracts comments."""

    # Patterns for comments
    INLINE_COMMENT_PATTERN = re.compile(
        r'<!--\s*COMMENT:\s*(.+?)\s*(?:\(attempted:\s*\w+\))?\s*-->',
        re.IGNORECASE
    )
    ANNOTATION_MARKER_PATTERN = re.compile(r'<!--\s*@ann:(\w+)\s*-->')

    # Pattern for section headers
    HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    # Pattern for section metadata
    SECTION_META_PATTERN = re.compile(
        r'<!--\s*SECTION\s*\n(.*?)\n\s*-->',
        re.DOTALL
    )

    def parse(self, content: str) -> tuple[list[DocumentSection], list[UserComment]]:
        """
        Parse document content into sections and comments.

        Returns:
            Tuple of (sections, comments)
        """
        sections = self._extract_sections(content)
        comments = self._extract_comments(content)
        return sections, comments

    def _extract_sections(self, content: str) -> list[DocumentSection]:
        """Extract sections based on headers."""
        sections = []
        lines = content.split('\n')

        # Find all headers
        headers = []
        for i, line in enumerate(lines):
            match = self.HEADER_PATTERN.match(line)
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                headers.append((i, level, title))

        # Create sections from headers
        for idx, (line_num, level, title) in enumerate(headers):
            # Section content goes until next header or end
            start = line_num
            if idx + 1 < len(headers):
                end = headers[idx + 1][0]
            else:
                end = len(lines)

            content_lines = lines[start:end]
            section_content = '\n'.join(content_lines)

            # Generate section ID from title
            section_id = self._generate_section_id(title, idx)

            sections.append(DocumentSection(
                id=section_id,
                title=title,
                content=section_content,
                level=level,
                start_line=start + 1,  # 1-indexed
                end_line=end,
            ))

        return sections

    def _generate_section_id(self, title: str, index: int) -> str:
        """Generate a section ID from title."""
        # Extract number prefix if present (e.g., "1.2" from "1.2 The Title")
        num_match = re.match(r'^([\d.]+)\s*', title)
        if num_match:
            return f"section_{num_match.group(1).replace('.', '_')}"
        # Otherwise use index
        return f"section_{index + 1}"

    def _extract_comments(self, content: str) -> list[UserComment]:
        """Extract user comments from document."""
        comments = []
        lines = content.split('\n')

        for i, line in enumerate(lines):
            # Check for inline comments
            for match in self.INLINE_COMMENT_PATTERN.finditer(line):
                comment_text = match.group(1).strip()
                # Check if marked as attempted (resolved)
                resolved = 'attempted:' in match.group(0).lower()

                # Get context (surrounding lines)
                context_start = max(0, i - 2)
                context_end = min(len(lines), i + 3)
                context = '\n'.join(lines[context_start:context_end])

                comments.append(UserComment(
                    id=f"comment_{i}",
                    text=comment_text,
                    line_number=i + 1,
                    context=context,
                    resolved=resolved,
                    comment_type="inline",
                ))

            # Check for annotation markers
            for match in self.ANNOTATION_MARKER_PATTERN.finditer(line):
                annotation_id = match.group(1)
                # These are markers - actual content is in annotations.json
                # Mark as placeholder for now
                context_start = max(0, i - 2)
                context_end = min(len(lines), i + 3)
                context = '\n'.join(lines[context_start:context_end])

                comments.append(UserComment(
                    id=annotation_id,
                    text=f"[Annotation marker: {annotation_id}]",
                    line_number=i + 1,
                    context=context,
                    resolved=False,
                    comment_type="annotation",
                ))

        return comments


# =============================================================================
# Formatting Checker
# =============================================================================


class FormattingChecker:
    """Checks for formatting issues in markdown/math content."""

    # Math delimiter patterns
    DISPLAY_MATH_OPEN = re.compile(r'\$\$(?!\$)')
    DISPLAY_MATH_CLOSE = re.compile(r'(?<!\$)\$\$')
    INLINE_MATH = re.compile(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)')

    # Common issues
    UNCLOSED_MATH = re.compile(r'\$\$[^$]+$', re.MULTILINE)
    DOUBLE_DOLLAR_IN_INLINE = re.compile(r'(?<!\$)\$[^$]*\$\$')

    def check(self, content: str) -> list[AuditIssue]:
        """Check content for formatting issues."""
        issues = []
        lines = content.split('\n')

        # Check math delimiters
        issues.extend(self._check_math_delimiters(content, lines))

        # Check markdown structure
        issues.extend(self._check_markdown_structure(lines))

        return issues

    def _check_math_delimiters(
        self,
        content: str,
        lines: list[str]
    ) -> list[AuditIssue]:
        """Check for math delimiter issues."""
        issues = []

        # Count $$ pairs
        display_opens = len(self.DISPLAY_MATH_OPEN.findall(content))
        display_closes = len(self.DISPLAY_MATH_CLOSE.findall(content))

        if display_opens != display_closes:
            issues.append(AuditIssue(
                type=IssueType.MATH_FORMATTING,
                severity=IssueSeverity.ERROR,
                message=f"Unbalanced display math delimiters: {display_opens} opens, {display_closes} closes",
                details={"opens": display_opens, "closes": display_closes},
            ))

        # Check each line for issues
        in_display_math = False
        for i, line in enumerate(lines):
            # Track display math blocks
            opens_in_line = len(self.DISPLAY_MATH_OPEN.findall(line))
            closes_in_line = len(self.DISPLAY_MATH_CLOSE.findall(line))

            # Check for $$ used incorrectly
            if '$$' in line and '$$$' in line:
                issues.append(AuditIssue(
                    type=IssueType.MATH_FORMATTING,
                    severity=IssueSeverity.WARNING,
                    message="Triple dollar sign found - likely delimiter error",
                    location=f"line {i + 1}",
                ))

            # Check for common LaTeX issues
            if '\\' in line and '$' in line:
                # Check for unescaped underscores in math
                if re.search(r'\$[^$]*(?<!\\)_[^_]*[^$]*\$', line):
                    # This is actually valid LaTeX, skip
                    pass

        return issues

    def _check_markdown_structure(self, lines: list[str]) -> list[AuditIssue]:
        """Check markdown structural issues."""
        issues = []

        # Check header hierarchy
        prev_level = 0
        for i, line in enumerate(lines):
            match = re.match(r'^(#{1,6})\s+', line)
            if match:
                level = len(match.group(1))
                # Warn if skipping levels (e.g., # to ###)
                if level > prev_level + 1 and prev_level > 0:
                    issues.append(AuditIssue(
                        type=IssueType.MARKDOWN_FORMATTING,
                        severity=IssueSeverity.WARNING,
                        message=f"Header level jumps from {prev_level} to {level}",
                        location=f"line {i + 1}",
                    ))
                prev_level = level

        # Check for duplicate section numbers
        section_numbers = []
        for i, line in enumerate(lines):
            match = re.match(r'^#{1,6}\s+([\d.]+)', line)
            if match:
                num = match.group(1)
                if num in section_numbers:
                    issues.append(AuditIssue(
                        type=IssueType.SECTION_NUMBERING,
                        severity=IssueSeverity.WARNING,
                        message=f"Duplicate section number: {num}",
                        location=f"line {i + 1}",
                    ))
                section_numbers.append(num)

        return issues


# =============================================================================
# Main Auditor
# =============================================================================


class DocumentAuditor:
    """
    Main document auditor class.

    Performs comprehensive analysis of a document including:
    - Duplicate section detection (using LLM)
    - Comment extraction and status
    - Formatting issue detection
    - Structural analysis
    """

    def __init__(
        self,
        llm_callback: Optional[LLMCallback] = None,
        *,
        similarity_threshold: float = 0.7,
        check_formatting: bool = True,
        check_duplicates: bool = True,
    ):
        """
        Initialize the document auditor.

        Args:
            llm_callback: Async function for LLM calls (for duplicate detection).
                         Use a cheap/fast model.
            similarity_threshold: Threshold for flagging similar sections
            check_formatting: Whether to check formatting issues
            check_duplicates: Whether to check for duplicates
        """
        self.llm_callback = llm_callback
        self.similarity_threshold = similarity_threshold
        self.check_formatting = check_formatting
        self.check_duplicates = check_duplicates

        self.parser = DocumentParser()
        self.formatter_checker = FormattingChecker()

    async def audit_file(self, file_path: str | Path) -> AuditReport:
        """
        Audit a document file.

        Args:
            file_path: Path to the markdown file

        Returns:
            AuditReport with findings
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        content = path.read_text(encoding='utf-8')
        report = await self.audit_content(content, str(path))

        # Also check for annotations.json
        annotations_path = path.parent / "annotations.json"
        if annotations_path.exists():
            await self._load_annotations(annotations_path, report)

        return report

    async def audit_content(
        self,
        content: str,
        source_path: str = "unknown"
    ) -> AuditReport:
        """
        Audit document content.

        Args:
            content: The markdown content to audit
            source_path: Source path for reporting

        Returns:
            AuditReport with findings
        """
        log.info("document_audit.started", source=source_path)

        # Parse document
        sections, comments = self.parser.parse(content)

        # Initialize report
        report = AuditReport(
            document_path=source_path,
            audit_time=datetime.now().isoformat(),
            total_sections=len(sections),
            total_words=sum(s.word_count for s in sections),
            comments=comments,
        )

        # Check for duplicates
        if self.check_duplicates and len(sections) > 1:
            duplicates = await self._find_duplicates(sections)
            report.duplicates = duplicates
            report.duplicate_count = len(duplicates)

        # Check formatting
        if self.check_formatting:
            formatting_issues = self.formatter_checker.check(content)
            report.issues.extend(formatting_issues)
            report.formatting_issue_count = len(formatting_issues)

        # Count unresolved comments
        report.unresolved_comment_count = sum(
            1 for c in comments if not c.resolved
        )

        # Add comment issues
        for comment in comments:
            if not comment.resolved:
                report.issues.append(AuditIssue(
                    type=IssueType.UNRESOLVED_COMMENT,
                    severity=IssueSeverity.INFO,
                    message=f"Unresolved comment: {comment.text[:50]}...",
                    location=f"line {comment.line_number}",
                    details={"comment_id": comment.id, "full_text": comment.text},
                ))

        log.info(
            "document_audit.completed",
            sections=len(sections),
            duplicates=report.duplicate_count,
            comments=len(comments),
            issues=len(report.issues),
        )

        return report

    async def _find_duplicates(
        self,
        sections: list[DocumentSection]
    ) -> list[DuplicatePair]:
        """Find duplicate or highly similar sections."""
        duplicates = []

        if not self.llm_callback:
            log.warning("document_audit.no_llm", message="Skipping duplicate check - no LLM callback")
            return duplicates

        # Compare each pair of sections
        checked_pairs = set()

        for i, section1 in enumerate(sections):
            for j, section2 in enumerate(sections):
                if i >= j:
                    continue

                pair_key = (section1.id, section2.id)
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)

                # Skip very short sections
                if section1.word_count < 50 or section2.word_count < 50:
                    continue

                # Check for duplicates using LLM
                is_dup, reason = await self._compare_sections(section1, section2)

                if is_dup:
                    # Determine recommendation
                    recommendation = self._get_merge_recommendation(section1, section2)

                    duplicates.append(DuplicatePair(
                        section1_id=section1.id,
                        section2_id=section2.id,
                        section1_title=section1.title,
                        section2_title=section2.title,
                        similarity_reason=reason,
                        recommendation=recommendation,
                    ))

                    log.info(
                        "document_audit.duplicate_found",
                        section1=section1.id,
                        section2=section2.id,
                        reason=reason[:100],
                    )

        return duplicates

    async def _compare_sections(
        self,
        section1: DocumentSection,
        section2: DocumentSection
    ) -> tuple[bool, str]:
        """Compare two sections for semantic similarity."""
        prompt = f"""Compare these two document sections and determine if they cover the SAME material (duplicate content that should be merged).

SECTION A: "{section1.title}"
{section1.content[:1500]}

---

SECTION B: "{section2.title}"
{section2.content[:1500]}

Consider:
1. Do they explain the same concepts?
2. Do they cover the same mathematical structures?
3. Would a reader find them redundant?

Respond in this format:
IS_DUPLICATE: [yes/no]
OVERLAP_LEVEL: [none/low/medium/high/complete]
REASON: [1-2 sentences explaining what content overlaps]
UNIQUE_IN_A: [what section A has that B doesn't, or "nothing"]
UNIQUE_IN_B: [what section B has that A doesn't, or "nothing"]"""

        try:
            response = await self.llm_callback(prompt)

            is_duplicate = False
            reason = ""

            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('IS_DUPLICATE:'):
                    value = line.replace('IS_DUPLICATE:', '').strip().lower()
                    is_duplicate = value in ('yes', 'true', 'y')
                elif line.startswith('REASON:'):
                    reason = line.replace('REASON:', '').strip()
                elif line.startswith('OVERLAP_LEVEL:'):
                    level = line.replace('OVERLAP_LEVEL:', '').strip().lower()
                    # Also flag high overlap even if not marked as duplicate
                    if level in ('high', 'complete') and not is_duplicate:
                        is_duplicate = True
                        reason = reason or f"High content overlap ({level})"

            return is_duplicate, reason

        except Exception as e:
            log.warning(
                "document_audit.compare_failed",
                section1=section1.id,
                section2=section2.id,
                error=str(e),
            )
            return False, ""

    def _get_merge_recommendation(
        self,
        section1: DocumentSection,
        section2: DocumentSection
    ) -> str:
        """Determine merge recommendation based on sections."""
        # Prefer earlier section (lower line number)
        if section1.start_line < section2.start_line:
            if section1.word_count >= section2.word_count:
                return "remove_second"
            else:
                return "merge"
        else:
            if section2.word_count >= section1.word_count:
                return "remove_first"
            else:
                return "merge"

    async def _load_annotations(
        self,
        annotations_path: Path,
        report: AuditReport
    ) -> None:
        """Load and incorporate annotations from JSON file."""
        try:
            data = json.loads(annotations_path.read_text(encoding='utf-8'))
            annotations = data.get('annotations', {})

            for ann_id, ann_data in annotations.items():
                # Find matching comment in report and update
                for comment in report.comments:
                    if comment.id == ann_id:
                        comment.text = ann_data.get('content', comment.text)
                        break
                else:
                    # Add annotation not found in document
                    report.comments.append(UserComment(
                        id=ann_id,
                        text=ann_data.get('content', ''),
                        line_number=0,  # Unknown - marker might be missing
                        context='[from annotations.json]',
                        resolved=False,
                        comment_type=ann_data.get('type', 'annotation'),
                    ))

        except Exception as e:
            log.warning(
                "document_audit.annotations_load_failed",
                path=str(annotations_path),
                error=str(e),
            )


# =============================================================================
# Convenience Functions
# =============================================================================


async def audit_document(
    file_path: str | Path,
    llm_callback: Optional[LLMCallback] = None,
) -> AuditReport:
    """
    Convenience function to audit a document.

    Args:
        file_path: Path to the document
        llm_callback: Optional LLM callback for duplicate detection

    Returns:
        AuditReport with findings
    """
    auditor = DocumentAuditor(llm_callback=llm_callback)
    return await auditor.audit_file(file_path)


def print_audit_report(report: AuditReport) -> None:
    """Print a human-readable audit report."""
    print(f"\n{'='*60}")
    print(f"DOCUMENT AUDIT REPORT")
    print(f"{'='*60}")
    print(f"Document: {report.document_path}")
    print(f"Audit time: {report.audit_time}")
    print(f"Sections: {report.total_sections}")
    print(f"Total words: {report.total_words}")

    if report.duplicates:
        print(f"\n{'─'*60}")
        print(f"DUPLICATE CONTENT ({len(report.duplicates)} pairs)")
        print(f"{'─'*60}")
        for dup in report.duplicates:
            print(f"\n  • {dup.section1_title}")
            print(f"    ≈ {dup.section2_title}")
            print(f"    Reason: {dup.similarity_reason}")
            print(f"    Recommendation: {dup.recommendation}")

    unresolved = [c for c in report.comments if not c.resolved]
    if unresolved:
        print(f"\n{'─'*60}")
        print(f"UNRESOLVED COMMENTS ({len(unresolved)})")
        print(f"{'─'*60}")
        for comment in unresolved:
            print(f"\n  Line {comment.line_number}: {comment.text[:80]}...")

    if report.issues:
        print(f"\n{'─'*60}")
        print(f"OTHER ISSUES ({len(report.issues)})")
        print(f"{'─'*60}")
        for issue in report.issues:
            if issue.type not in (IssueType.UNRESOLVED_COMMENT,):
                print(f"\n  [{issue.severity.value.upper()}] {issue.type.value}")
                print(f"    {issue.message}")
                if issue.location:
                    print(f"    Location: {issue.location}")

    print(f"\n{'='*60}\n")
