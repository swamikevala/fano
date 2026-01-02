# Chunk Augmentation: Diagrams, Tables, and Proofs

## Overview

After refinement, blessed chunks should be evaluated for supplementary content that would clarify and strengthen the insight. A chunk may be augmented with:

- **Diagrams** — visual representation of structures, mappings, relationships
- **Tables** — systematic enumeration of all cases, mappings, correspondences
- **Proofs** — formal mathematical verification where applicable
- **Code** — executable demonstration of a claim

---

## Augmentation Flow

```
Chunk passes review (⚡)
    │
    ▼
Augmentation Analysis (Claude Opus)
    │
    ├── Would a diagram clarify this? → Generate diagram
    ├── Would a table systematize this? → Generate table
    ├── Can this be formally proven? → Generate proof
    ├── Would code demonstrate this? → Generate code
    └── Text is sufficient → No augmentation needed
    │
    ▼
Augmented Chunk (insight + supporting materials)
```

---

## Augmentation Analysis Prompt

```
A mathematical insight has been blessed. Evaluate whether supplementary 
materials would strengthen or clarify it.

BLESSED INSIGHT:
{refined_insight}

TAGS: {tags}
DEPENDS ON: {dependencies}

REVIEW CONTEXT:
{summary_of_what_reviewers_found_compelling}

EVALUATE EACH AUGMENTATION TYPE:

1. DIAGRAM
   Would a visual representation clarify this?
   Consider: graph structures, geometric relationships, mappings between sets,
   flow diagrams, hierarchies, symmetry illustrations
   
   DIAGRAM_HELPFUL: [yes/no]
   DIAGRAM_TYPE: [graph / geometry / mapping / hierarchy / flow / none]
   DIAGRAM_DESCRIPTION: [what it would show]

2. TABLE
   Would a systematic enumeration strengthen this?
   Consider: complete mappings, correspondences, case analysis,
   numerical relationships, comparison matrices
   
   TABLE_HELPFUL: [yes/no]
   TABLE_TYPE: [mapping / enumeration / comparison / correspondence / none]
   TABLE_DESCRIPTION: [rows, columns, what it demonstrates]

3. PROOF
   Can this be formally verified mathematically?
   Consider: the claim is precise enough to prove, there's a proof strategy,
   it's not just an observation but a theorem
   
   PROOF_POSSIBLE: [yes/partial/no]
   PROOF_STRATEGY: [approach if yes]
   PROOF_DEPENDENCIES: [what lemmas or known results it would use]

4. CODE
   Would executable code demonstrate or verify this?
   Consider: computation that confirms numerical claims, generation of 
   structures, verification of properties
   
   CODE_HELPFUL: [yes/no]
   CODE_PURPOSE: [verify / generate / demonstrate / none]
   CODE_DESCRIPTION: [what it would do]

RECOMMENDATION:
[List which augmentations to generate, in priority order]
```

---

## Diagram Generation

**Tools:** Python with matplotlib, networkx, graphviz, or SVG generation

**Prompt:**

```
Generate a diagram to illustrate this mathematical insight.

INSIGHT: {insight}

DIAGRAM SPECIFICATION:
Type: {diagram_type}
Description: {diagram_description}

REQUIREMENTS:
- Clear, minimal, mathematically precise
- Labels must be readable
- Use color meaningfully (not decoratively)
- Include legend if multiple elements
- Export as SVG or PNG

Generate Python code using matplotlib/networkx/graphviz that produces this diagram.
The code must be self-contained and executable.

Respond with:
CODE: [complete Python code]
CAPTION: [1-2 sentence description for the diagram]
```

**Example outputs:**
- Fano plane with labeled points and lines
- Bipartite graph showing chakra-to-dharana mapping
- Hierarchical tree of 7 dimensions containing 112 chakras
- Geometric illustration of ellipsoid/linga form

---

## Table Generation

**Format:** Markdown table (renders well, versionable)

**Prompt:**

```
Generate a table to systematize this mathematical insight.

INSIGHT: {insight}

TABLE SPECIFICATION:
Type: {table_type}
Description: {table_description}

REQUIREMENTS:
- Complete enumeration (all cases, not just examples)
- Clear column headers
- Sorted logically (by number, by category, by relationship)
- Include totals/summaries if applicable
- Markdown format

Respond with:
TABLE: [complete markdown table]
NOTES: [any observations about patterns in the table]
VERIFICATION: [how to verify the table is complete/correct]
```

**Example outputs:**

| Element | Body % | Chakras | Notes |
|---------|--------|---------|-------|
| Water | 72% | 1-48 | Lowest dimensions |
| Earth | 12% | 49-72 | ... |
| ... | ... | ... | ... |

---

## Proof Generation

**Format:** Structured proof with clear steps

**Prompt:**

```
Generate a formal mathematical proof for this insight.

INSIGHT: {insight}

PROOF SPECIFICATION:
Strategy: {proof_strategy}
Dependencies: {proof_dependencies}

REQUIREMENTS:
- State the theorem precisely
- List all assumptions/axioms used
- Each step must follow logically from previous
- Cite known results where used
- QED or explicit conclusion

FORMAT:
THEOREM: [precise statement]

ASSUMPTIONS:
- [list each assumption]

PROOF:
1. [first step with justification]
2. [second step with justification]
...
n. [conclusion] ∎

NOTES: [any caveats, alternative approaches, or open questions]
```

**Proof validation:** After generation, the proof should be reviewed by at least one other LLM for correctness.

---

## Code Generation

**Language:** Python (most versatile)

**Prompt:**

```
Generate executable code to demonstrate or verify this insight.

INSIGHT: {insight}

CODE SPECIFICATION:
Purpose: {code_purpose}
Description: {code_description}

REQUIREMENTS:
- Self-contained (no external dependencies beyond numpy/standard lib)
- Clear comments explaining each step
- Print output that demonstrates the claim
- Include verification/assertion where applicable

Respond with:
CODE: [complete Python code]
EXPECTED_OUTPUT: [what running this should show]
WHAT_IT_PROVES: [how the output supports the insight]
```

**Example:**
```python
# Verify that 72 + 12 = 84 and this equals the cosmic cycle count
body_water = 72
body_earth = 12
solid_matter = body_water + body_earth
cosmic_cycles = 84

assert solid_matter == cosmic_cycles, f"Mismatch: {solid_matter} != {cosmic_cycles}"
print(f"✓ Body solid matter ({solid_matter}%) = Cosmic cycles ({cosmic_cycles})")
```

---

## Augmented Chunk Data Model

```python
@dataclass
class Augmentation:
    type: str  # "diagram" / "table" / "proof" / "code"
    content: str  # The actual content (code, markdown, proof text)
    caption: str  # Brief description
    generated_by: str  # Which LLM generated it
    verified: bool  # Has it been checked?
    file_path: Optional[str]  # For diagrams: path to generated image
    execution_output: Optional[str]  # For code: what it printed

@dataclass
class Chunk:
    id: str
    insight: str
    versions: List[ChunkVersion]
    augmentations: List[Augmentation]  # NEW
    # ... rest of fields
    
    @property
    def has_diagram(self) -> bool:
        return any(a.type == "diagram" for a in self.augmentations)
    
    @property
    def has_proof(self) -> bool:
        return any(a.type == "proof" for a in self.augmentations)
```

---

## Augmentation Storage

```
data/
├── chunks/
│   └── chunk_042/
│       ├── chunk.json          # Main chunk data
│       ├── diagram.svg         # Generated diagram
│       ├── diagram.py          # Code that generated it
│       ├── table.md            # Markdown table
│       ├── proof.md            # Formal proof
│       └── verification.py     # Verification code + output
```

---

## Augmentation Review

Generated augmentations should be verified:

| Type | Verification Method |
|------|---------------------|
| Diagram | LLM confirms it accurately represents the insight |
| Table | LLM confirms completeness and accuracy |
| Proof | Different LLM checks each step |
| Code | Actually execute it, verify output matches claim |

**Proof verification prompt:**
```
Review this mathematical proof for correctness.

THEOREM: {theorem}
PROOF: {proof_steps}

CHECK:
1. Is the theorem statement precise and unambiguous?
2. Are all assumptions explicitly stated?
3. Does each step follow logically from previous steps?
4. Are cited results used correctly?
5. Does the conclusion actually follow?

VERDICT: [valid / flawed / needs revision]
ISSUES: [list any problems found]
SUGGESTIONS: [how to fix if flawed]
```

---

## When to Augment

Not every chunk needs augmentation. Guidelines:

| Chunk Type | Likely Augmentation |
|------------|---------------------|
| Structural mapping (A ↔ B) | Table showing full mapping |
| Geometric claim | Diagram |
| Numerical relationship | Code verification |
| Precise theorem | Formal proof |
| Enumeration claim ("there are N...") | Table of all N items |
| Process/flow | Flow diagram |
| Hierarchy | Tree diagram |
| Simple observation | None needed |

---

## Configuration

```yaml
augmentation:
  enabled: true
  auto_generate: true  # Generate for all blessed chunks
  require_verification: true  # Must pass verification before attaching
  
  types:
    diagram:
      enabled: true
      format: "svg"  # or "png"
      tool: "matplotlib"  # or "graphviz"
    table:
      enabled: true
      format: "markdown"
    proof:
      enabled: true
      require_review: true  # Second LLM must verify
    code:
      enabled: true
      execute: true  # Actually run the code
      timeout_seconds: 30
```

---

## Summary

Blessed chunks can be augmented with supporting materials:

| Augmentation | Purpose | Format |
|--------------|---------|--------|
| Diagram | Visualize structure | SVG/PNG |
| Table | Systematize mappings | Markdown |
| Proof | Formally verify | Structured text |
| Code | Demonstrate/verify | Python |

This transforms chunks from isolated statements into rich, self-documenting units of knowledge.
