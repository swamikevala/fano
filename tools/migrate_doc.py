#!/usr/bin/env python3
"""
Migration tool for Documenter v3.0 "Living Textbook" architecture.

Converts existing single-file document.md to split storage format:
- Parses SECTION markers and extracts content
- Creates sections/*.md files
- Generates structure.json with chapter/section hierarchy
- Extracts definitions.json from ESTABLISHES markers
- Validates roundtrip integrity

Usage:
    python tools/migrate_doc.py                    # Dry run
    python tools/migrate_doc.py --execute         # Perform migration
    python tools/migrate_doc.py --validate        # Validate existing migration
"""

import argparse
import hashlib
import io
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Fix Windows console encoding for Unicode
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from documenter.repository import DocumentRepository, SectionContent


# -----------------------------------------------------------------------------
# Section Parser
# -----------------------------------------------------------------------------

# Pattern for section metadata blocks
SECTION_PATTERN = re.compile(
    r'<!-- SECTION\s*\n(.*?)-->',
    re.DOTALL
)

# Pattern for chapter/section headers
HEADER_PATTERN = re.compile(r'^(#{1,3})\s+(.+?)$', re.MULTILINE)


def parse_section_metadata(metadata_text: str) -> dict:
    """Parse section metadata comment into dict."""
    data = {}
    for line in metadata_text.strip().split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()

            # Parse lists like [concept1, concept2]
            if value.startswith('[') and value.endswith(']'):
                inner = value[1:-1].strip()
                if inner:
                    # Handle ** markers in concept names
                    items = [item.strip().replace('**', '').strip() for item in inner.split(',')]
                    value = [i for i in items if i]
                else:
                    value = []

            data[key] = value

    return data


def extract_sections_from_document(content: str) -> list[dict]:
    """
    Extract all sections from a document.

    Returns list of dicts with:
        - id: Section ID
        - title: Inferred from content header
        - content: Section content (without metadata)
        - metadata: Parsed metadata dict
        - start_pos: Character position in source
        - end_pos: Character position in source
    """
    sections = []
    matches = list(SECTION_PATTERN.finditer(content))

    for i, match in enumerate(matches):
        metadata = parse_section_metadata(match.group(1))

        # Find end of section (start of next section or end of document)
        start_pos = match.end()
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(content)

        section_content = content[start_pos:end_pos].strip()

        # Extract title from first header in content
        title = "Untitled Section"
        header_match = HEADER_PATTERN.search(section_content)
        if header_match:
            title = header_match.group(2).strip()
            # Remove annotation markers from title
            title = re.sub(r'<!--.*?-->', '', title).strip()

        sections.append({
            'id': metadata.get('id', f'section_{i:03d}'),
            'title': title,
            'content': section_content,
            'metadata': metadata,
            'start_pos': match.start(),
            'end_pos': end_pos,
        })

    return sections


def extract_preamble(content: str) -> str:
    """Extract content before the first SECTION marker."""
    match = SECTION_PATTERN.search(content)
    if match:
        return content[:match.start()].strip()
    return content.strip()


def generate_summary(content: str, max_length: int = 150) -> str:
    """Generate a brief summary from section content."""
    # Remove markdown formatting
    text = re.sub(r'[#*_`]', '', content)
    text = re.sub(r'\$.*?\$', '[math]', text)  # Replace LaTeX
    text = re.sub(r'<!--.*?-->', '', text)  # Remove comments
    text = re.sub(r'\s+', ' ', text).strip()

    # Take first sentence or truncate
    sentences = re.split(r'[.!?]', text)
    if sentences and sentences[0]:
        summary = sentences[0].strip()[:max_length]
        if len(summary) < len(sentences[0].strip()):
            summary += "..."
        return summary

    return text[:max_length] + "..." if len(text) > max_length else text


def infer_chapter_structure(sections: list[dict]) -> list[dict]:
    """
    Infer chapter structure from section headers.

    Groups sections under chapters based on H1 headers.
    """
    chapters = []
    current_chapter = None

    for section in sections:
        content = section['content']

        # Check if section starts with H1 (chapter header)
        first_line = content.split('\n')[0].strip()
        is_chapter_start = first_line.startswith('# ') and not first_line.startswith('## ')

        if is_chapter_start or current_chapter is None:
            # Extract chapter title
            if is_chapter_start:
                chapter_title = first_line[2:].strip()
                # Parse chapter number if present (e.g., "1. Foundations")
                num_match = re.match(r'^(\d+)\.\s*(.+)$', chapter_title)
                if num_match:
                    chapter_num = num_match.group(1)
                    chapter_title = num_match.group(2)
                else:
                    chapter_num = str(len(chapters) + 1)
            else:
                chapter_num = str(len(chapters) + 1)
                chapter_title = section['title']

            current_chapter = {
                'id': f'ch_{int(chapter_num):02d}',
                'title': chapter_title,
                'summary': generate_summary(content),
                'sections': [],
            }
            chapters.append(current_chapter)

        current_chapter['sections'].append(section)

    return chapters


# -----------------------------------------------------------------------------
# Migration Functions
# -----------------------------------------------------------------------------

def migrate_document(
    source_path: Path,
    target_dir: Path,
    dry_run: bool = True,
) -> dict:
    """
    Migrate a single-file document to split storage format.

    Args:
        source_path: Path to source document.md
        target_dir: Path to target document directory
        dry_run: If True, don't write files

    Returns:
        Migration report dict
    """
    report = {
        'source': str(source_path),
        'target': str(target_dir),
        'dry_run': dry_run,
        'sections_found': 0,
        'chapters_inferred': 0,
        'concepts_extracted': 0,
        'errors': [],
    }

    # Read source document
    if not source_path.exists():
        report['errors'].append(f"Source file not found: {source_path}")
        return report

    content = source_path.read_text(encoding='utf-8')
    print(f"Read {len(content):,} characters from {source_path}")

    # Extract preamble
    preamble = extract_preamble(content)
    print(f"Extracted preamble: {len(preamble):,} characters")

    # Extract sections
    sections = extract_sections_from_document(content)
    report['sections_found'] = len(sections)
    print(f"Found {len(sections)} sections")

    if not sections:
        # No SECTION markers - treat entire document as one section
        sections = [{
            'id': 'sec_00_01',
            'title': 'Document Content',
            'content': content,
            'metadata': {
                'created': datetime.now().strftime('%Y-%m-%d'),
                'status': 'provisional',
                'establishes': [],
                'requires': [],
            },
            'start_pos': 0,
            'end_pos': len(content),
        }]

    # Infer chapter structure
    chapters = infer_chapter_structure(sections)
    report['chapters_inferred'] = len(chapters)
    print(f"Inferred {len(chapters)} chapters")

    # Extract concepts from establishes markers
    all_concepts = {}
    for section in sections:
        establishes = section['metadata'].get('establishes', [])
        if isinstance(establishes, list):
            for concept in establishes:
                if concept and concept not in all_concepts:
                    all_concepts[concept] = section['id']

    report['concepts_extracted'] = len(all_concepts)
    print(f"Extracted {len(all_concepts)} concepts from ESTABLISHES markers")

    if dry_run:
        print("\n=== DRY RUN - No files written ===")
        print_migration_plan(chapters, all_concepts)
        return report

    # Execute migration
    print("\n=== Executing Migration ===")

    repo = DocumentRepository(target_dir)
    repo.initialize()

    # Add chapters and sections
    for chapter in chapters:
        # Add chapter
        repo.add_chapter(
            chapter_id=chapter['id'],
            title=chapter['title'],
            summary=chapter['summary'],
        )

        # Add sections
        for i, section_data in enumerate(chapter['sections']):
            section = SectionContent(
                id=section_data['id'],
                title=section_data['title'],
                content=section_data['content'],
                summary=generate_summary(section_data['content']),
                concepts_defined=section_data['metadata'].get('establishes', []) or [],
                concepts_required=section_data['metadata'].get('requires', []) or [],
                chapter_id=chapter['id'],
                created=parse_date(section_data['metadata'].get('created', '')),
                status=section_data['metadata'].get('status', 'provisional'),
            )
            repo.add_section(chapter['id'], section)

    # Compile book to verify
    compiled = repo.compile_book()
    print(f"Compiled book: {len(compiled):,} characters")

    report['success'] = True
    return report


def parse_date(date_str: str) -> datetime:
    """Parse date string or return current datetime."""
    if not date_str:
        return datetime.now()
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        return datetime.now()


def print_migration_plan(chapters: list[dict], concepts: dict):
    """Print the planned migration structure."""
    print("\n--- Migration Plan ---\n")

    for chapter in chapters:
        print(f"Chapter: {chapter['id']} - {chapter['title']}")
        print(f"  Summary: {chapter['summary'][:80]}...")
        print(f"  Sections ({len(chapter['sections'])}):")

        for section in chapter['sections']:
            establishes = section['metadata'].get('establishes', [])
            est_str = ', '.join(establishes[:3]) if establishes else '—'
            if len(establishes) > 3:
                est_str += f" (+{len(establishes)-3} more)"
            print(f"    - {section['id']}: {section['title'][:50]}")
            print(f"      Establishes: {est_str}")
        print()

    print(f"\n--- Concepts Registry ({len(concepts)} concepts) ---\n")
    for concept, section_id in sorted(concepts.items()):
        print(f"  {concept}: {section_id}")


def validate_migration(
    original_path: Path,
    migrated_dir: Path,
) -> dict:
    """
    Validate that migration preserved content.

    Compares original document with recompiled version.
    """
    report = {
        'original_path': str(original_path),
        'migrated_dir': str(migrated_dir),
        'valid': False,
        'errors': [],
        'warnings': [],
    }

    # Load original
    if not original_path.exists():
        report['errors'].append(f"Original file not found: {original_path}")
        return report

    original = original_path.read_text(encoding='utf-8')

    # Load compiled
    repo = DocumentRepository(migrated_dir)
    if not repo.structure_path.exists():
        report['errors'].append(f"Migration not found: {migrated_dir}")
        return report

    compiled = repo.compile_book()

    # Compare content (normalize whitespace)
    def normalize(text: str) -> str:
        # Remove metadata comments
        text = re.sub(r'<!-- SECTION\n.*?-->', '', text, flags=re.DOTALL)
        text = re.sub(r'<!--.*?-->', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    original_norm = normalize(original)
    compiled_norm = normalize(compiled)

    # Check key content is preserved
    # Extract all mathematical expressions
    original_math = set(re.findall(r'\$[^$]+\$', original))
    compiled_math = set(re.findall(r'\$[^$]+\$', compiled))

    missing_math = original_math - compiled_math
    if missing_math:
        report['warnings'].append(f"Missing {len(missing_math)} mathematical expressions")

    # Check section count
    original_sections = len(SECTION_PATTERN.findall(original))
    structure = repo.load_structure()
    migrated_sections = sum(len(ch['sections']) for ch in structure['chapters'])

    if original_sections != migrated_sections:
        report['warnings'].append(
            f"Section count mismatch: {original_sections} original vs {migrated_sections} migrated"
        )

    # Basic content length check
    len_diff = abs(len(original_norm) - len(compiled_norm)) / max(len(original_norm), 1)
    if len_diff > 0.1:  # More than 10% difference
        report['warnings'].append(f"Content length differs by {len_diff*100:.1f}%")

    report['valid'] = len(report['errors']) == 0
    report['original_length'] = len(original)
    report['compiled_length'] = len(compiled)
    report['original_sections'] = original_sections
    report['migrated_sections'] = migrated_sections

    return report


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Migrate document.md to split storage format',
    )
    parser.add_argument(
        '--source',
        type=Path,
        default=PROJECT_ROOT / 'documenter' / 'document' / 'main.md',
        help='Source document path',
    )
    parser.add_argument(
        '--target',
        type=Path,
        default=PROJECT_ROOT / 'data' / 'document',
        help='Target directory for split storage',
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Execute migration (default is dry run)',
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate existing migration',
    )

    args = parser.parse_args()

    if args.validate:
        print("Validating migration...")
        report = validate_migration(args.source, args.target)
        print(f"\nValidation {'PASSED' if report['valid'] else 'FAILED'}")
        if report['errors']:
            print("Errors:")
            for e in report['errors']:
                print(f"  - {e}")
        if report['warnings']:
            print("Warnings:")
            for w in report['warnings']:
                print(f"  - {w}")
        print(f"\nOriginal: {report.get('original_length', 0):,} chars, "
              f"{report.get('original_sections', 0)} sections")
        print(f"Compiled: {report.get('compiled_length', 0):,} chars, "
              f"{report.get('migrated_sections', 0)} sections")
        return 0 if report['valid'] else 1

    print(f"Migrating: {args.source}")
    print(f"Target: {args.target}")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY RUN'}")
    print()

    report = migrate_document(
        source_path=args.source,
        target_dir=args.target,
        dry_run=not args.execute,
    )

    if report.get('errors'):
        print("\nErrors:")
        for e in report['errors']:
            print(f"  - {e}")
        return 1

    print(f"\nMigration {'complete' if args.execute else 'plan ready'}.")
    print(f"  Sections: {report['sections_found']}")
    print(f"  Chapters: {report['chapters_inferred']}")
    print(f"  Concepts: {report['concepts_extracted']}")

    if not args.execute:
        print("\nRun with --execute to perform migration.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
