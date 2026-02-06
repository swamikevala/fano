# Research Project Assistant — v2 Architecture

> **Status**: Proposal (pending review)
> **Date**: 2026-02-06
> **Scope**: Full system redesign — generalized from single-domain tool to multi-domain research platform

---

## Table of Contents

1. [Vision: What This System Is](#1-vision-what-this-system-is)
2. [Diagnosis: What's Wrong with v1](#2-diagnosis-whats-wrong-with-v1)
3. [Design Principles](#3-design-principles)
4. [System Architecture](#4-system-architecture)
5. [The Project Model](#5-the-project-model)
6. [Explorer: Pure LLM Reasoning Engine](#6-explorer-pure-llm-reasoning-engine)
   - 6.7 [v1 Code Gap Analysis and Hardening Plan](#v1-code-gap-analysis-and-hardening-plan)
7. [Researcher: External Evidence Engine](#7-researcher-external-evidence-engine)
   - 7.5 [v1 Code Gap Analysis and Hardening Plan](#v1-code-gap-analysis-and-hardening-plan-1)
8. [Documenter: Synthesis Engine](#8-documenter-synthesis-engine)
   - 8.7 [v1 Code Gap Analysis and Hardening Plan](#v1-code-gap-analysis-and-hardening-plan-2)
9. [User Interaction Model](#9-user-interaction-model)
10. [Infrastructure: Event Bus](#10-infrastructure-event-bus)
11. [Infrastructure: State Store](#11-infrastructure-state-store)
12. [Infrastructure: LLM Client](#12-infrastructure-llm-client)
    - 12.5 [Consensus Engine Architecture](#125-consensus-engine-architecture)
13. [Infrastructure: Configuration](#13-infrastructure-configuration)
14. [Observability and Operations](#14-observability-and-operations)
15. [Testing Strategy](#15-testing-strategy)
16. [Migration Strategy](#16-migration-strategy)
17. [Appendix: File Layout](#17-appendix-file-layout)

---

## 1. Vision: What This System Is

### The Core Idea

This is a **research project assistant** — a platform where a human researcher defines a goal, provides seed ideas, and then collaborates with a team of LLM agents to explore, validate, research, and document findings into a living document that addresses the goal.

The system is domain-agnostic. The same engine that explores connections between the Fano plane and yogic traditions can explore business opportunities, synthesize content research, or investigate any other open-ended intellectual problem. The difference between domains is **configuration** — the prompts, evaluation criteria, and seed inputs — not the machinery.

### Three Engines, One Goal

```
USER defines: Research Goal + Seed Inputs
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                        EXPLORER                              │
│  Pure LLM reasoning. Takes seeds, debates them via           │
│  multi-round consensus. Produces insights:                   │
│  attested / discarded / disputed / interesting               │
│                                                              │
│  Seeds can be MODIFIED by LLMs if they discover              │
│  something more profound than the original premise.          │
└────────────────────────┬────────────────────────────────────┘
                         │ attested insights
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                       RESEARCHER                             │
│  Goes into the world. Finds resources that                   │
│  back up / refute / extend exploration insights.             │
│  Feeds evidence back to Explorer and Documenter.             │
└────────────────────────┬────────────────────────────────────┘
                         │ evidence-backed insights
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                       DOCUMENTER                             │
│  Synthesizes attested insights into a coherent document      │
│  that addresses the research goal. Incorporates user         │
│  annotations and comments.                                   │
└─────────────────────────────────────────────────────────────┘
```

### The User Is Always In The Loop

The user dynamically interacts with the system at **every stage**:

| Stage | User Actions |
|-------|-------------|
| **Seeds** | Add new seeds, prioritize them, retire exhausted ones |
| **Exploration** | Comment on insights ("this looks promising", "not useful"), steer direction |
| **Research** | Request specific research topics, evaluate source credibility |
| **Document** | Leave annotations that the documenter incorporates, mark sections as protected or needing revision |

The system is not a black box. It's a collaborative workspace where LLMs do the heavy lifting but the human researcher retains full creative control.

### Example Projects

**Mathematical Research** (current Fano project):
- Goal: "Explore the mathematical structure of 'three' and its appearance across Indian traditions"
- Seeds: "The Fano plane's 7 points correspond to the 7 chakras...", "The 112 chakras map to anti-incidences..."
- Evaluation criteria: mathematical rigor, structural depth, naturalness, alignment with traditional teachings
- Seed modification: Allowed — LLMs may discover a more elegant formulation

**Business Planning**:
- Goal: "Develop a comprehensive business plan for entering the European EV charging market"
- Seeds: "Urban fast-charging hubs in city centers", "Highway corridor networks", "Residential building integration"
- Evaluation criteria: market viability, financial soundness, competitive advantage, regulatory feasibility
- Seed modification: Allowed — LLMs may discover a more viable market entry angle

**Content Research**:
- Goal: "Produce a comprehensive analysis of the impact of AI on journalism"
- Seeds: "Automated news generation", "AI fact-checking tools", "Revenue model disruption", "Deepfake detection"
- Evaluation criteria: factual accuracy, source credibility, narrative coherence, balanced perspective
- Seed modification: Allowed — LLMs may identify a more important angle the researcher hadn't considered

All three use the same Explorer → Researcher → Documenter pipeline. The difference is entirely in the project configuration.

---

## 2. Diagnosis: What's Wrong with v1

### 2.1 Domain Lock-in

v1 is hardcoded to a single research domain. The coupling runs deep:

| Location | Hardcoded Domain Reference |
|----------|---------------------------|
| `explorer/config.yaml` lines 132-144 | Exploration intro references Fano plane, chakras, Sanskrit grammar, Indian music |
| `explorer/config.yaml` lines 150-161 | Goals reference "these specific numbers across traditions" |
| `explorer/axioms/seeds.yaml` | All seeds reference Fano plane / yogic traditions |
| `explorer/axioms/target_numbers.yaml` | Hardcoded number-to-concept mappings (7→chakras, 72→melakartas) |
| `explorer/src/chunking/prompts.py` lines 40-44 | Extraction prompt lists specific cross-domain bridges |
| `explorer/src/chunking/prompts.py` lines 95-100 | `SADHGURU_ALIGNMENT` scoring dimension |
| `explorer/src/review_panel/prompts/round1.py` line 36 | "Sadhguru's teachings on yogic science" |
| `researcher/src/analysis/extractor.py` lines 282-292 | Hardcoded number→concept mappings |
| `document/guidance.md` | Domain-specific document guidance |

This means standing up a new research project requires forking the entire codebase and rewriting prompts — rather than just creating a new project configuration.

### 2.2 No Research Goal as First-Class Entity

There is no explicit, machine-readable research goal. The closest is a Sadhguru quote in the document preamble and some qualitative criteria buried in `explorer/config.yaml`. The goal should drive everything: which seeds are relevant, how insights are evaluated, what the documenter is trying to produce.

### 2.3 Limited User Interaction

The user's ability to steer the system at runtime is minimal:

- **No UI to add/edit/prioritize seeds** — seeds are manually edited YAML files
- **No way to comment on insights** — no endpoint exists for insight-level feedback
- **Annotations exist but are disconnected** — `annotations.json` and inline `<!-- COMMENT: -->` markers are separate systems that can diverge
- **No way to redirect exploration** — can't say "this thread is going nowhere, try a different angle"

### 2.4 Seed Rigidity

Seeds are static YAML entries that never change once loaded. But the user explicitly wants seeds to be **mutable** — if the LLM reasoning process discovers that a slightly different formulation of the seed premise is more profound, the system should be able to propose a modification. This is a fundamentally different model from "seeds are immutable inputs."

### 2.5 Infrastructure Issues (Carried Forward from v1 Analysis)

These problems exist independently of the domain lock-in and are addressed in the infrastructure sections:

| # | Problem | Impact |
|---|---------|--------|
| 1 | File-based inter-component communication | Race conditions, 30s+ latency |
| 2 | Blessed insights can be corrupted (non-atomic writes) | Data loss on concurrent read/write |
| 3 | Four config files with conflicting values | Silent misconfigurations |
| 4 | Researcher runs unsupervised (not in orchestrator) | Unmanaged LLM spend |
| 5 | No cost tracking or circuit breakers | Unbounded spending |
| 6 | Threads never retire; data grows unbounded | 1GB explorer dir, stale threads block new work |
| 7 | Documenter conflates API failures with content rejection | Good insights killed by network jitter |
| 8 | Static thread priority | Promising threads can't be boosted |
| 9 | No operational visibility (logs only, no metrics) | Failures discovered manually |
| 10 | 983MB dead browser cache + unbounded log growth | Disk exhaustion |

---

## 3. Design Principles

1. **Project-first**: Everything is scoped to a research project. The project defines the goal, the domains, the evaluation criteria, and the prompts. The engine is generic; the project is specific.

2. **User as creative director**: The system proposes, the user disposes. Seeds can be added, modified, retired. Insights can be endorsed or dismissed. The document can be annotated. The user's input is not an afterthought — it's a core data flow.

3. **Seeds are living hypotheses**: Seeds are not immutable inputs. They're starting points that can evolve as exploration reveals deeper formulations. The system tracks seed lineage (original → modified → why).

4. **Events over polling**: Components publish events; interested parties subscribe. No file scanning.

5. **Single source of truth**: One database for state, one config hierarchy for settings.

6. **Fail gracefully**: Distinguish transient failures (API timeout) from substantive rejections (insight lacks rigor). Never kill a good insight because of network jitter.

7. **Track every token**: Know what each LLM call costs, enforce budgets, surface trends.

8. **Lifecycle everything**: Seeds, threads, insights, sections, and research questions all have explicit state machines with terminal states.

9. **Test-driven**: Every behavioral change starts with a failing test. The test suite is the acceptance criteria for migration phases. No phase is complete until all tests pass.

---

## 4. System Architecture

### Component Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                         CONTROL PANEL (Flask)                          │
│                                                                        │
│  Project Setup • Seed Management • Insight Feed • Document Viewer      │
│  Research Monitor • Metrics Dashboard • User Comments & Annotations    │
│                                                                        │
└───────────────────────────────┬────────────────────────────────────────┘
                                │ HTTP (status, control, user actions)
                                ▼
┌────────────────────────────────────────────────────────────────────────┐
│                           ORCHESTRATOR                                 │
│                                                                        │
│  Project Registry • Module Registry • Task Scheduler                   │
│  Quota Allocator • WAL-based State Recovery                            │
│                                                                        │
│  EVENT BUS: routes events between all modules and the user             │
│                                                                        │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐                │
│  │  Explorer    │  │  Documenter  │  │  Researcher    │                │
│  │  Adapter     │  │  Adapter     │  │  Adapter       │                │
│  └──────┬──────┘  └──────┬───────┘  └───────┬────────┘                │
└─────────┼────────────────┼──────────────────┼──────────────────────────┘
          │                │                  │
          ▼                ▼                  ▼
   ┌────────────┐   ┌────────────┐   ┌────────────────┐
   │  Explorer   │   │ Documenter │   │  Researcher    │
   │  Engine     │   │  Engine    │   │  Engine        │
   │             │   │            │   │                │
   │ Pure LLM    │   │ Synthesis  │   │ External       │
   │ reasoning   │   │ & document │   │ evidence       │
   │ + consensus │   │ generation │   │ gathering      │
   └──────┬─────┘   └─────┬──────┘   └───────┬────────┘
          │                │                  │
          └────────────────┼──────────────────┘
                           ▼
              ┌──────────────────────┐
              │     STATE STORE      │
              │     (SQLite + WAL)   │
              │                      │
              │  Projects, Seeds,    │
              │  Threads, Insights,  │
              │  Sections, Research, │
              │  User Comments,      │
              │  Events, Metrics     │
              └──────────────────────┘
                           │
                           ▼
              ┌──────────────────────┐
              │     LLM CLIENT       │
              │     OpenRouter API   │
              │                      │
              │  Rate Limiter        │
              │  Token Tracker       │
              │  Circuit Breaker     │
              │  Cost Estimator      │
              └──────────────────────┘
```

### Key Structural Changes from v1

| Aspect | v1 | v2 |
|--------|----|----|
| Domain scope | Hardcoded to Fano/Sadhguru | Project-configurable (any domain) |
| Research goal | Implicit (buried in prompts) | First-class entity driving all evaluation |
| Seeds | Static YAML, immutable | Living hypotheses, modifiable by LLMs and user |
| User interaction | View-only dashboard + manual YAML editing | Full CRUD on seeds, insight comments, document annotations |
| Prompts | Hardcoded in Python | Template-based, loaded from project config |
| Inter-component communication | File polling + HTTP + subprocess | Event bus (in-process pub/sub) |
| State storage | JSON files scattered across directories | Single SQLite database with WAL |
| Configuration | 4 YAML files with duplication | 1 hierarchical YAML + project overrides |
| Researcher integration | Standalone process, unmanaged | First-class module, quota-managed, directed by other components |
| Insight lifecycle | Directory-as-status, no failure classification | Explicit state machine, transient vs substantive failures |
| Thread lifecycle | Born, never dies | Explicit state machine with retirement |
| Cost tracking | None | Per-request token counting with budgets |

---

## 5. The Project Model

### What Is a Project?

A project is the unit of research. It defines:

1. **Goal** — what the research is trying to achieve or answer
2. **Seeds** — starting hypotheses, questions, or premises to explore
3. **Evaluation criteria** — what makes an insight valuable in this domain
4. **Prompt context** — domain framing injected into all LLM interactions
5. **Research domains** — where the Researcher should look for evidence

Everything else — the Explorer engine, consensus mechanism, document pipeline, event bus — is shared infrastructure that operates identically across projects.

### Project Configuration

```yaml
# projects/fano-mathematics.yaml

project:
  id: "fano-mathematics"
  name: "The Mathematics of Three"

  goal: |
    Explore the mathematical structure that emerges from the principle
    of "three" — the idea that the first manifestation is always in the
    form of three. Investigate its appearance across Indian traditions
    (yogic systems, Sanskrit grammar, classical music, cosmology) and
    modern mathematics (finite geometry, group theory, combinatorics).

    We seek what is natural, elegant, beautiful, interesting, and
    inevitable — structure that must exist, not structure we impose.

  # Domain context injected into all LLM prompts
  context: |
    This research explores connections between:
    - Fano plane incidence geometry (7 points, 7 lines, 3 points per line)
    - The Yogic energy system (chakras, nadis, 5 elements)
    - Sanskrit grammar (Panini's system, 14 Maheshvara Sutras)
    - The Sriyantra / Srichakra
    - Indian classical music theory (12 swaras, 22 shrutis, 72 melakartas)
    - Yogic/Tantric cosmology and practice systems

  # What makes an insight valuable in THIS project?
  evaluation_criteria:
    - name: "rigor"
      description: "Are the claims logically sound and verifiable?"
      weight: 1.0
    - name: "structural_depth"
      description: "Is this a genuine structural connection, not surface-level pattern matching?"
      weight: 1.0
    - name: "naturalness"
      description: "Does this feel discovered rather than invented? Inevitable rather than forced?"
      weight: 1.0
    - name: "cross_domain_bridging"
      description: "Does this connect multiple domains in a meaningful way?"
      weight: 0.8
    - name: "tradition_alignment"
      description: "Does this align with how these traditions understand themselves?"
      weight: 0.7

  # Controls whether LLMs can propose modifications to seed premises
  seed_modification:
    enabled: true
    require_user_approval: false  # Auto-accept if consensus agrees
    preserve_original: true       # Always keep the original seed text for reference

  # Exploration guidance — injected into explorer prompts
  exploration_guidance: |
    Follow your mathematical curiosity. Let the structure reveal itself.
    If something feels forced, abandon it. If something feels inevitable,
    pursue it — even if it seems unrelated to anything practical.

    The standard to aspire to: "This HAS to be true. It would be
    bizarre if it weren't."

  # Document guidance — injected into documenter prompts
  document_guidance: |
    The document should build from first principles. Start with the
    simplest structures and develop toward the profound. Each section
    should feel like a natural next step from the previous one.

    Use precise mathematical language but remain accessible to someone
    with undergraduate mathematics background.

  # Where the Researcher should look for evidence
  research_domains:
    - name: "finite_geometry"
      keywords: ["Fano plane", "projective plane", "incidence geometry"]
      source_types: ["academic_paper", "textbook", "lecture_notes"]
    - name: "yogic_systems"
      keywords: ["chakra", "nadi", "kundalini", "tantra"]
      source_types: ["traditional_text", "commentary", "academic_study"]
    - name: "sanskrit_grammar"
      keywords: ["Panini", "Maheshvara Sutras", "Ashtadhyayi"]
      source_types: ["academic_paper", "traditional_text"]
    - name: "indian_music"
      keywords: ["melakarta", "svara", "shruti", "raga"]
      source_types: ["musicology", "traditional_treatise"]
```

Compare with a business planning project:

```yaml
# projects/ev-charging-europe.yaml

project:
  id: "ev-charging-europe"
  name: "European EV Charging Market Entry"

  goal: |
    Develop a comprehensive business plan for entering the European
    EV charging market. Identify the most viable market entry strategy,
    target segments, competitive positioning, financial projections,
    and regulatory considerations.

  context: |
    This analysis covers the European EV charging infrastructure market,
    including urban fast-charging, highway corridors, residential/commercial
    charging, and emerging technologies (V2G, wireless charging).
    Key markets: Germany, France, Netherlands, Norway, UK.

  evaluation_criteria:
    - name: "market_viability"
      description: "Is there demonstrable market demand and willingness to pay?"
      weight: 1.0
    - name: "financial_soundness"
      description: "Are the unit economics realistic? Is the business model sustainable?"
      weight: 1.0
    - name: "competitive_advantage"
      description: "Does this create a defensible position against incumbents?"
      weight: 0.9
    - name: "regulatory_feasibility"
      description: "Is this compatible with current and anticipated EU/national regulations?"
      weight: 0.8
    - name: "scalability"
      description: "Can this approach scale across multiple European markets?"
      weight: 0.7

  seed_modification:
    enabled: true
    require_user_approval: true   # Business decisions need human sign-off
    preserve_original: true

  exploration_guidance: |
    Think like a strategy consultant. Challenge assumptions rigorously.
    Look for non-obvious competitive angles. Consider second-order effects
    of regulatory changes. Be skeptical of "obvious" market opportunities —
    if they were obvious, incumbents would already own them.

  document_guidance: |
    Structure as a professional business plan: Executive Summary, Market
    Analysis, Competitive Landscape, Strategy, Operations, Financial
    Projections, Risk Assessment. Use concrete numbers where possible.

  research_domains:
    - name: "ev_market_data"
      keywords: ["EV charging", "electric vehicle infrastructure", "EVSE"]
      source_types: ["market_report", "industry_analysis", "government_data"]
    - name: "regulatory"
      keywords: ["AFIR", "EU green deal", "charging regulation"]
      source_types: ["legislation", "policy_document", "regulatory_analysis"]
    - name: "competitive"
      keywords: ["Ionity", "Tesla Supercharger", "Fastned", "Allego"]
      source_types: ["company_report", "news", "financial_filing"]
```

### Why Project-Level Configuration?

The alternative is "configure everything in one big config.yaml." But research projects are fundamentally different from infrastructure settings:

- **Infrastructure** (LLM models, rate limits, database path) rarely changes and applies system-wide.
- **Project** (goal, seeds, evaluation criteria, prompts) changes per research effort and is the user's primary creative input.

Separating them means:
1. A user can switch between projects without touching infrastructure config.
2. Project files can be version-controlled, shared, and forked independently.
3. The system can eventually support multiple concurrent projects (future).

---

## 6. Explorer: Pure LLM Reasoning Engine

### What the Explorer Does

The Explorer takes seed inputs and uses multi-LLM consensus to reason about them in the context of the research goal. It is **100% pure reasoning** — no web searches, no external data, just LLMs thinking deeply about ideas and debating each other.

The output is a stream of insights, each classified as:
- **Attested** — multiple LLMs agree this is valid and valuable
- **Discarded** — consensus rejects it (lacks rigor, not relevant, too shallow)
- **Disputed** — LLMs disagree; needs more exploration or user input
- **Interesting** — shows potential but needs development before attestation

### Seed Lifecycle

Seeds are not static inputs. They are **living hypotheses** that evolve:

```
                    ┌──────────┐
     user creates   │  ACTIVE  │ ◄── user can edit, reprioritize
                    └────┬─────┘
                         │
            ┌────────────┼────────────┐
            │            │            │
    explored by LLMs     │     user retires
            │            │            │
            ▼            │            ▼
     ┌────────────┐      │     ┌───────────┐
     │  EXPLORED  │      │     │  RETIRED  │
     └─────┬──────┘      │     └───────────┘
           │             │
    LLMs propose         │
    modification         │
           │             │
           ▼             │
     ┌────────────┐      │
     │  EVOLVED   │──────┘ can be re-explored with new formulation
     └────────────┘
         │
         │ original preserved in lineage
         ▼
     ┌────────────┐
     │  (new seed │  ← child seed with reference to parent
     │   created) │
     └────────────┘
```

**Seed modification** is a key feature: when the Explorer's LLM debate reveals that a slightly different formulation of the original premise is more profound, the system can:

1. Propose the modification (always logged with reasoning)
2. Either auto-accept (if `require_user_approval: false`) or queue for user review
3. Create a new child seed linked to the original (lineage preserved)
4. The original seed text is always preserved for reference

```python
# Example: seed evolution

# Original seed (user-provided):
# "The 7 chakras map to the 7 points of the Fano plane"

# After exploration, LLMs discover:
# "The 7 chakras map to the 7 LINES of the Fano plane (not points),
#  because lines represent relationships between fundamental elements,
#  which is what chakras actually are — junction points of nadis."

# System creates:
# - Original seed status: EXPLORED
# - New child seed: "The 7 chakras map to the 7 lines of the Fano plane..."
#   with parent_seed_id pointing to original
#   with modification_reason: "Lines better represent chakras because..."
```

### Thread Model

A thread is an exploration session around a seed. Each thread is a multi-turn LLM conversation with rotating roles:

```
┌─────────────────────────────────────────────────────┐
│ THREAD: Exploring seed "7 chakras ↔ Fano lines"    │
│                                                      │
│ Exchange 1: Explorer (Gemini) — initial reasoning    │
│ Exchange 2: Critic (ChatGPT) — challenges/pushback  │
│ Exchange 3: Explorer (Gemini) — addresses critique   │
│ Exchange 4: Synthesizer (Claude) — identifies core   │
│ Exchange 5: Explorer (Gemini) — develops further     │
│ Exchange 6: Critic (ChatGPT) — final assessment      │
│ ...                                                  │
│                                                      │
│ → Chunk ready: extract atomic insights               │
│ → Review panel: multi-round attestation              │
│ → Output: attested/discarded/disputed/interesting    │
└─────────────────────────────────────────────────────┘
```

### Thread Lifecycle

```
                    ┌─────────┐
          spawn()   │  ACTIVE  │ ◄── dynamic priority adjustment
                    └────┬────┘
                         │
            ┌────────────┼────────────┐
            │            │            │
    chunk_ready()        │     idle_too_long()
            │            │            │
            ▼            │            ▼
     ┌────────────┐      │     ┌───────────┐
     │SYNTHESIZING│      │     │  STALLED  │
     └─────┬──────┘      │     └─────┬─────┘
           │             │           │
           │ extraction  │    retry? │ no → RETIRED
           │ complete    │           │
           ▼             │           ▼
     ┌───────────┐       │    ┌───────────┐
     │ EXTRACTED  │       │    │  RETIRED  │
     └─────┬─────┘       │    └───────────┘
           │             │
           │ review      │
           │ complete    │
           ▼             │
     ┌───────────┐       │
     │  RETIRED  │ ◄─────┘ max_exchanges reached
     └───────────┘
```

**Dynamic priority**: Thread priority adjusts based on the quality of reasoning emerging in the debate. Signals like "this necessarily follows," "elegant," "inevitable" boost priority. Signals like "forcing," "arbitrary," "circular" reduce it. User comments on insights from a thread also influence its priority.

**Retirement**: Threads are retired when they've been synthesized, stalled for too long, or hit the exchange limit. Retired threads free their slot (max 3 active by default) for new exploration.

**Seed re-exploration cap**: A seed can only be re-explored a configurable number of times (default 3) before it's considered exhausted. This prevents the degenerate loop in v1 where the system endlessly re-explores the same seed.

### Review Panel (Multi-Round Attestation)

After a thread generates enough exchanges, insights are extracted and put through the review panel — a multi-round debate among LLMs about whether each insight meets the project's evaluation criteria.

```
Round 1: Independent Rating
  Each LLM independently rates the insight on the project's
  evaluation criteria. No knowledge of other ratings.
  → Unanimous accept: ATTESTED
  → Unanimous reject: DISCARDED
  → Mixed: proceed to Round 2

Round 2: Deep Analysis
  LLMs see each other's Round 1 ratings and reasoning.
  Engage in structured debate. May change their minds.
  → Consensus reached: ATTESTED or DISCARDED
  → Still split: proceed to Round 3

Round 3: Deliberation
  Focused exchange on specific points of disagreement.
  Structured argument and counter-argument.
  → Majority decides: ATTESTED or DISPUTED

Round 4 (rare): Tiebreaker
  If Round 3 is perfectly split, a final model breaks the tie.
  → ATTESTED, DISCARDED, or INTERESTING (promising but unresolved)
```

**The review criteria come from the project configuration**, not from hardcoded Python strings. The review panel prompts are templates:

```python
# explorer/src/review_panel/prompts/templates.py

ROUND1_TEMPLATE = """You are reviewing a proposed insight for a research project.

PROJECT GOAL:
{project_goal}

PROJECT CONTEXT:
{project_context}

EVALUATION CRITERIA:
{evaluation_criteria_formatted}

The insight to review:
{insight_text}

Rate this insight on each criterion (1-10) with reasoning.
Then give an overall verdict: ⚡ (accept), ? (uncertain), ✗ (reject).
"""
```

The `{evaluation_criteria_formatted}` block is generated from the project's `evaluation_criteria` list, so it automatically adapts to whatever criteria the user defined.

### Prompt Architecture

All Explorer prompts follow the same pattern:

```
┌──────────────────────────────┐
│ SYSTEM CONTEXT               │
│ (from project.context)       │
├──────────────────────────────┤
│ RESEARCH GOAL                │
│ (from project.goal)          │
├──────────────────────────────┤
│ EVALUATION CRITERIA          │
│ (from project.evaluation_    │
│  criteria, formatted)        │
├──────────────────────────────┤
│ EXPLORATION GUIDANCE         │
│ (from project.exploration_   │
│  guidance)                   │
├──────────────────────────────┤
│ TASK-SPECIFIC INSTRUCTIONS   │
│ (template for this specific  │
│  operation: explore, review, │
│  extract, etc.)              │
├──────────────────────────────┤
│ CONTENT                      │
│ (the actual seed, thread     │
│  history, insight, etc.)     │
└──────────────────────────────┘
```

The top four sections come from the project configuration. The bottom two are engine-specific templates. This means changing the project configuration changes how every LLM interaction is framed — without touching any code.

### v1 Code Gap Analysis and Hardening Plan

The following issues were identified by deep code review of the v1 Explorer implementation. Each must be addressed during migration.

#### 6.7.1 Dual Consensus Implementation

v1 has **two separate consensus implementations** that don't share code:

| Implementation | Location | Used By |
|---------------|----------|---------|
| Generic `ConsensusTask` / `ReviewTask` | `llm/src/consensus/` | Documenter (planning) |
| Custom 4-round review flow | `explorer/src/review_panel/reviewer.py` (880 lines) | Explorer (insight attestation) |

The Explorer's review panel has its own round progression, vote counting, and convergence detection — completely bypassing the generic framework. This means bug fixes to one don't apply to the other.

**v2 requirement**: One `ConsensusEngine` class that both Explorer and Documenter parameterize with their specific prompts and evaluation criteria. The review panel's 4-round structure becomes a configuration of the shared engine, not a separate implementation.

#### 6.7.2 Response Parsing Robustness

v1 uses fragile parsing throughout:

- **Case-sensitive string matching** for verdict markers (`⚡`, `?`, `✗`) — fails if model uses words instead of symbols
- **First-100-characters comparison** for structural convergence — misses semantic similarity in longer responses
- **Jaccard similarity threshold hardcoded at 0.5** — too lenient for meaningful convergence detection
- **"Longest response = best response"** fallback when parsing fails — a bad heuristic

**v2 requirement**: Multi-strategy response parsing with fallback chain:

```
1. Try structured JSON parsing (for models that support response_format)
2. Try regex extraction with case-insensitive matching
3. Fall back to LLM re-extraction ("Given this review text, extract the verdict and scores as JSON")
4. If all fail, mark as PARSE_ERROR (not as a valid vote)
```

Convergence detection must use full-text comparison with configurable threshold, not truncated string matching.

#### 6.7.3 Error Response Filtering (CRITICAL BUG)

v1 treats error responses as valid votes in decision-based convergence. If 2 out of 3 backends return the same generic error message ("I cannot process this request"), that counts as "consensus."

**v2 requirement**:
- Validate responses before counting as votes
- Error responses are excluded from convergence calculation
- Minimum valid response threshold: at least `consensus.min_valid_responses` backends must return valid responses for any round to count
- If too many errors, the round is retried (up to `max_retries`), not declared "converged"

#### 6.7.4 Variable Reviewer Count

v1 panel extraction is hardcoded for exactly 3 LLMs. The `_run_panel_review()` method assumes indices 0, 1, 2.

**v2 requirement**:
- Reviewer count driven by `config.yaml` `consensus.backends` list (2–5 reviewers)
- Tiebreaker rules adapt to odd/even reviewer count
- Round progression works with any number of reviewers
- If a reviewer is unavailable (circuit breaker open), continue with remaining panel members

#### 6.7.5 Profundity Scoring

v1 uses **keyword-based profundity scoring** — counting occurrences of words like "elegant," "profound," "inevitable" to score insight depth. This is gameable and superficial.

**v2 requirement**:
- Profundity is derived from the review panel's evaluation scores, not keyword counting
- Project-specific scoring dimensions come from `project.evaluation_criteria`
- Optional: LLM-based meta-evaluation where a separate call rates the quality of the review panel's reasoning itself

#### 6.7.6 Error Handling in Review Rounds

v1 has **no retry logic** when LLM calls fail during review. A timeout on Round 2 means the insight is stuck.

**v2 requirement**:
- Each review round retries failed calls up to `config.llm.max_retries`
- If a reviewer fails consistently, exclude that reviewer and continue with remaining panel
- Never treat an error response as a valid review verdict (see 6.7.3)
- Partial round results are saved — if 2 of 3 reviewers complete, those results are preserved even if the third fails

#### 6.7.7 Modification Consensus

v1's seed modification flow is complex and underspecified. The code path for proposing, voting on, and applying modifications exists but is tangled with the main review flow.

**v2 requirement**:
- Modification proposals require supermajority (>66%) agreement among reviewers
- Proposals are structured: `{original_text, proposed_text, reasoning}`
- If `project.seed_modification.require_user_approval: true`, proposal queues in the database for user review
- Modification history tracked in seed lineage table (`parent_seed_id` + `modification_reason`)
- Clear separation: review flow produces insights; modification proposals are a side-channel output

#### 6.7.8 Thread Priority and Resume

v1 has unclear priority-switching logic. When the Explorer switches to a higher-priority thread mid-conversation, it's ambiguous whether the interrupted thread resumes from its last exchange or restarts.

**v2 requirement**:
- Thread state (including last exchange sequence number) saved atomically to database before any switch
- Resume always continues from last completed exchange
- Priority recalculated after each exchange batch, not just on user action
- Thread preemption only happens at exchange boundaries, never mid-LLM-call

#### 6.7.9 Deduplication Across Extraction Runs

v1 has no deduplication between extraction runs for the same thread. If a thread is chunked twice (e.g., after additional exchanges), the same insights can be extracted again.

**v2 requirement**:
- Content-hash check against all existing insights from the same thread before entering review
- Cross-thread deduplication via the shared dedup pipeline (existing, but needs to be applied at extraction time)
- Dedup uses `hashlib.sha256` (not Python's non-deterministic `hash()`)

---

## 7. Researcher: External Evidence Engine

### What the Researcher Does

The Researcher goes outside the system to find external evidence that backs up, refutes, or extends what the Explorer discovered. It is the system's connection to the real world.

The Researcher operates in two modes:

1. **Autonomous**: Monitors attested insights and generates research questions about them. Searches for relevant sources, evaluates their credibility, extracts findings.

2. **Directed**: Responds to specific requests from the Documenter ("I need background on X to write this section") or the user ("research topic Y").

### Integration with Orchestrator

In v1, the Researcher is a standalone process that nobody supervises. In v2, it's a first-class module:

```python
class ResearcherAdapter(ModuleInterface):
    """Integrates Researcher into the Orchestrator's task system."""

    @property
    def module_name(self) -> str:
        return "researcher"

    @property
    def supported_task_types(self) -> list[str]:
        return [
            "research_question",      # Autonomous question generation
            "directed_research",      # Respond to specific topic request
            "trust_evaluation",       # Evaluate a source's trustworthiness
            "cross_reference_scan",   # Detect patterns across findings
        ]

    async def initialize(self) -> bool:
        self.event_bus.subscribe("explorer.insight.attested", self._on_new_insight)
        self.event_bus.subscribe("documenter.section.added", self._on_new_section)
        self.event_bus.subscribe("documenter.research.requested", self._on_research_request)
        self.event_bus.subscribe("user.research.requested", self._on_user_research_request)
        return True
```

### Research Domain Configuration

Research domains come from the project configuration (see `research_domains` in the project YAML above). This replaces the hardcoded keyword patterns in `researcher/src/analysis/extractor.py`.

The Researcher loads its domain keywords, source type preferences, and trust evaluation rules from the project config. Different projects can have radically different research strategies:

- A math research project searches academic papers and traditional texts
- A business planning project searches market reports, filings, and news
- A content research project searches journalism, reports, and primary sources

### Directed Research Flow

```
Documenter is writing about a concept and needs background
  → Publishes: documenter.research.requested
    payload: {topic, context, urgency: "high"}

Researcher receives event
  → Generates targeted search queries from topic + project context
  → Searches, evaluates sources, extracts findings
  → Publishes: researcher.finding.stored
    payload: {finding_id, topic, summary, sources, confidence}

Documenter receives finding
  → Incorporates into section content
```

This closes the feedback loop that doesn't exist in v1, where the Documenter has no way to ask for specific research.

### Evidence Feed-Back to Explorer

When the Researcher finds external evidence that contradicts or strongly supports an Explorer insight, it publishes an event:

```
researcher.evidence.contradicts → Explorer can de-prioritize related threads
researcher.evidence.supports    → Explorer can boost priority of related threads
```

This creates a virtuous cycle: Explorer proposes → Researcher validates → Explorer focuses on validated directions.

### v1 Code Gap Analysis and Hardening Plan

#### 7.5.1 Extraction Plugin Architecture

v1 has only **2 hardcoded regex extraction patterns** for rule-based extraction in `researcher/src/analysis/extractor.py`. This is barely functional and completely domain-specific.

**v2 requirement**: Pluggable extraction pipeline:

```
1. Fast pre-filter: regex patterns from project config (research_domains[].extraction_patterns)
2. LLM-based extraction (primary method): prompt model to extract structured findings
3. Domain-specific extractors (optional): custom Python classes registered per project
```

The pipeline is configurable per project. A math project might rely heavily on LLM extraction. A business project might have regex patterns for financial figures, market sizes, etc.

#### 7.5.2 Trust Evaluation

v1 **promises consensus-based trust voting** but actually does a single LLM call. The code comments say "multi-model verification" but only one model is invoked.

**v2 requirement**:
- Trust evaluation uses the shared `ConsensusEngine` (same as Explorer review)
- Minimum 2 LLM evaluations per source, with configurable threshold in `config.yaml`
- Trust scores are durable and cached — same URL doesn't get re-evaluated unless content changes (checked via `content_hash`)
- Trust tiers (from v1's existing schema) are derived from consensus scores, not arbitrary single-model ratings

#### 7.5.3 Configurable Processing Limits

v1 hardcodes: **5 questions per insight, 3 searches per question**. These are buried in Python code, not configurable.

**v2 requirement**:
- All limits configurable in `config.yaml` under `researcher:` section:

```yaml
researcher:
  max_questions_per_insight: 10
  max_searches_per_question: 5
  max_findings_per_source: 20
  idle_polling_interval_seconds: 300
```

- Per-project overrides possible in project config
- Adaptive limits based on quota remaining (automatically reduce searches when budget is tight, increase when budget is plentiful)

#### 7.5.4 Domain Fallback Removal

v1 has hardcoded domain fallbacks in the question generator: `"Hindu tradition"`, `"Indian philosophy"` appear as literal strings in `researcher/src/questions/generator.py`. The `number-to-meaning` mapping is also duplicated between `extractor.py` and `domains.yaml`.

**v2 requirement**:
- **Zero hardcoded domain references** in any Researcher code
- All domain context comes from `project.context` and `project.research_domains`
- Fallback behavior: use `project.goal` as generic context, never domain-specific strings
- Number-to-meaning mappings (if needed) live in project config, not code

#### 7.5.5 Observer Race Conditions

v1 uses file-based polling to detect new insights ready for research. The observer watches the filesystem and can miss events or process partial writes.

**v2 requirement**:
- EventBus subscription (`explorer.insight.attested`) replaces file polling entirely
- No race conditions possible — events are ordered and persisted before delivery
- Researcher receives `finding_id` in event payload and reads full data from StateStore

#### 7.5.6 LLM Client Silent Degradation

v1's researcher LLM client checks for `None` responses but logs no warning and silently skips. Failed research produces no output and no visibility.

**v2 requirement**:
- All LLM failures in Researcher are logged with structured context (insight_id, question, backend, error_type)
- Failed research questions are re-queued with exponential backoff
- Dashboard shows research completion rate and failure rate per domain

---

## 8. Documenter: Synthesis Engine

### What the Documenter Does

The Documenter takes attested insights and synthesizes them into a coherent document that addresses the research goal. It's not just concatenating insights — it's building a narrative that:

1. Addresses the project's stated goal
2. Builds from prerequisites to conclusions
3. Incorporates user annotations and comments
4. Integrates Researcher evidence where relevant
5. Maintains internal consistency

### Document Structure

The document remains a markdown file (the primary output artifact), but its structural metadata lives in the database:

```
┌─────────────────────────────────────────────┐
│ document/main.md (rendered output)           │
│                                              │
│ # [Project Name]                             │
│ > [Project goal excerpt or epigraph]         │
│                                              │
│ ## 1. [First Major Theme]                    │
│ ### 1.1 [Section Title]                      │
│ Content synthesized from attested insights.  │
│                                              │
│ ### 1.2 [Section Title]                      │
│ Content with integrated research evidence.   │
│                                              │
│ ## 2. [Second Major Theme]                   │
│ ...                                          │
└─────────────────────────────────────────────┘
```

Section metadata (establishes, requires, status, source insight) lives in SQLite. The markdown file is the rendered output, regenerated from the database.

### User Annotations

The user can leave three types of annotations on the document, and the Documenter actively incorporates them:

| Type | Purpose | Documenter Action |
|------|---------|-------------------|
| **Comment** | "This section needs more detail on X" | Prioritizes addressing the comment; uses consensus to revise the section |
| **Protected** | "Don't change this paragraph" | Preserves the marked content during revisions |
| **Suggestion** | "Consider connecting this to Y" | Treats as a hint for the next planning cycle |

All annotations are stored in the database (not in divergent inline markers and JSON files):

```sql
CREATE TABLE annotations (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id),
    type TEXT NOT NULL CHECK(type IN ('comment', 'protected', 'suggestion')),
    section_id TEXT REFERENCES sections(id),
    content TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('open', 'attempted', 'resolved')),
    attempt_count INTEGER DEFAULT 0,
    last_attempted_at TEXT,
    created_at TEXT NOT NULL,
    resolved_at TEXT
);
```

### Comment Processing Priority

User comments are the **highest priority** work item for the Documenter. When the user says "this section needs more detail on X," that takes precedence over incorporating new insights. This preserves v1's good pattern where comments are prioritized.

### Sliding Context Window

Instead of showing ALL existing sections to LLMs (which doesn't scale), the Documenter builds focused context:

```python
async def get_context_for_insight(self, insight, max_tokens: int = 4000) -> str:
    """Build focused context relevant to the insight being incorporated."""

    # 1. Always include the research goal
    goal_section = self.project.goal

    # 2. Include the 2 most recent sections (narrative continuity)
    recent = await self.store.get_recent_sections(limit=2)

    # 3. Include sections that establish concepts this insight depends on
    dependencies = await self.store.get_insight_dependencies(insight.id)
    dep_sections = await self.store.get_sections_establishing(dependencies)

    # 4. Include sections with overlapping tags (thematic relevance)
    related = await self.store.get_sections_by_tags(insight.tags, limit=3)

    # 5. Deduplicate and truncate to token budget
    sections = deduplicate(recent + dep_sections + related)
    return self._truncate_to_tokens(sections, max_tokens)
```

### Concept Canonicalization

Concept names are normalized to prevent false "missing prerequisite" errors:

```python
def canonicalize(concept_name: str) -> str:
    """Normalize concept names for reliable matching.

    'Fano plane' → 'fano_plane'
    'The Fano Plane' → 'fano_plane'
    'market entry strategy' → 'market_entry_strategy'
    'Market-Entry Strategy' → 'market_entry_strategy'
    """
    name = concept_name.strip().lower()
    name = re.sub(r'^(the|a|an)\s+', '', name)
    name = re.sub(r'[^a-z0-9]+', '_', name)
    name = name.strip('_')
    return name
```

### Document Guidance

The Documenter receives project-specific guidance from `project.document_guidance`. This replaces the hardcoded `document/guidance.md` file. For a math project, it might say "build from first principles, use precise mathematical language." For a business plan, it might say "structure as Executive Summary, Market Analysis, Strategy, Financial Projections."

### v1 Code Gap Analysis and Hardening Plan

#### 8.7.1 Dispute/Failure Conflation (CRITICAL BUG)

v1's `opportunity_processor.py` uses **a single counter** for both API failures and content rejections. If the LLM call times out during drafting, that increments the same `dispute_count` as a genuine content rejection ("this insight lacks rigor"). After `max_consecutive_disputes` (default 3), the insight is permanently shelved — meaning 3 network glitches can kill a perfectly good insight.

**v2 requirement**:
- `InsightLifecycle` class (see Section 14) with separate `transient_failure_count` and `dispute_count`
- The 5-stage opportunity processor pipeline (dedup → eval → draft → exposition → add) must distinguish transient failures at **each stage independently**
- Transient failures trigger retry with backoff; substantive disputes trigger the dispute → shelved flow
- The distinction is made at the LLM client level: timeout/rate-limit/connection-error = transient; successful LLM response containing rejection = substantive

#### 8.7.2 Comment Retry Logic (BUG)

v1's comment retry in `comments.py` is broken: the code checks for `'attempted: true'` (boolean) but writes `'attempted: 2026-01-15'` (date string). This means the retry-detection logic **never matches**, so comments are either processed once or retried indefinitely depending on the code path.

**v2 requirement**:
- Comment status tracked in the `annotations` table with:
  - `attempt_count` (INTEGER) — how many times the system tried to address this comment
  - `last_attempted_at` (TEXT/datetime) — when the last attempt was made
  - `status` (TEXT) — `'open'` → `'attempted'` → `'resolved'`
- Retry logic: `WHERE status = 'attempted' AND attempt_count < max_attempts`
- After `max_attempts` (default 3), status moves to `'needs_human_review'` (user notified)

#### 8.7.3 Transaction Semantics

v1's `_add_to_document()` has rollback logic on failure, but `write_prerequisite()` has **no error handling at all**. A crash during prerequisite writing leaves the document in an inconsistent state.

**v2 requirement**:
- ALL document write operations use StateStore transactions
- Failure mid-write rolls back to last consistent state (database-level rollback)
- Document markdown file is rendered **after** the database transaction commits, never before
- Automatic snapshots before any structural change (section add, major revision, reorder)

#### 8.7.4 Seed Document Generation

v1 hardcodes the initial document content to mathematics domain. The preamble, section structure, and initial guidance are all Fano-specific.

**v2 requirement**:
- Initial document structure generated from `project.document_guidance`
- Project goal becomes the document epigraph/preamble
- Initial section suggestions generated by LLM from `project.goal` + initial seeds
- No hardcoded domain content in any Documenter code

#### 8.7.5 Annotation Convergence

v1 stores annotations in **two divergent locations**: inline `<!-- COMMENT: -->` markers in the markdown file AND a separate `annotations.json` file. These can (and do) get out of sync — a comment resolved in the JSON may still appear as a marker in the markdown, and vice versa.

**v2 requirement**:
- Single source of truth: `annotations` table in SQLite (see schema in Section 11)
- Markdown rendering reads from database; **no metadata is stored inline** in the markdown
- Annotations reference sections by `section_id`, not by text position (positions break when content is edited)
- When rendering the document, annotations are injected at display time by the Control Panel, not baked into the file

#### 8.7.6 Token Estimation

v1 uses `len(text) // 4` for token estimation. This is rough — off by 30-50% for non-English text, code, or structured content.

**v2 requirement**:
- Use `tiktoken` (or equivalent) for accurate token counting where precision matters (budget tracking, context window management)
- Fall back to character-based estimation only when tokenizer is unavailable or for quick approximations
- Actual token counts from LLM response `usage` field are always preferred over estimates
- Budget calculations must use actual consumed tokens, not estimated tokens

#### 8.7.7 Dead Code Removal

v1 has dead code paths in the Documenter: `suggest_order()` and `get_retriable_comments()` are defined but never called. Section metadata parsing uses fragile regexes that break on edge cases.

**v2 requirement**:
- Remove all dead code during migration
- Section metadata lives in database, not parsed from markdown comments
- Regex-based parsing replaced with database queries

#### 8.7.8 Review Scheduling Drift

v1's review allocation (`new_material: 70% / review_existing: 30%`) drifts from target because the scheduling doesn't account for actual time spent — only for planned task count.

**v2 requirement**:
- Track actual LLM calls spent on new material vs. review
- Rebalance allocation periodically based on actual spend
- Make allocation configurable in `config.yaml` under `documenter.work_allocation`

---

## 9. User Interaction Model

### Overview

The user is the creative director of the research. The system proposes; the user disposes, redirects, and enriches. Every user action is an event that the system reacts to.

### 9.1 Seed Management

**Current (v1)**: Edit YAML files manually. No UI.

**Proposed (v2)**:

```
Control Panel: Seed Management

┌──────────────────────────────────────────────────────────┐
│  SEEDS                                        [+ Add]    │
│                                                          │
│  ⚡ P:9  "7 chakras map to Fano lines"        [Edit]    │
│          Status: EXPLORED (2 threads)          [Retire]  │
│          └─ Child: "...as junction points..."  [View]    │
│                                                          │
│  ● P:7  "112 chakras as anti-incidences"      [Edit]    │
│          Status: ACTIVE                        [Retire]  │
│          3 user comments                                 │
│                                                          │
│  ○ P:5  "Maheshvara Sutras ↔ Fano coloring"  [Edit]    │
│          Status: PENDING                       [Retire]  │
│                                                          │
│  ◆ P:3  "Melakarta ragas ↔ Steiner system"   [Edit]    │
│          Status: EVOLVED                       [View]    │
│          └─ Original: "72 melakartas ↔ ..."             │
│                                                          │
│  [Drag to reorder priority]                              │
└──────────────────────────────────────────────────────────┘
```

API endpoints:

```
POST   /api/seeds              — Add a new seed
PUT    /api/seeds/:id          — Edit seed text, priority, tags
DELETE /api/seeds/:id          — Retire a seed
GET    /api/seeds              — List all seeds with status
GET    /api/seeds/:id/lineage  — View seed evolution history
POST   /api/seeds/:id/comment  — Comment on a seed
PUT    /api/seeds/:id/priority — Change priority
POST   /api/seeds/:id/approve-modification — Approve LLM-proposed modification
POST   /api/seeds/:id/reject-modification  — Reject, keep original
```

### 9.2 Insight Feed

The user sees a live feed of insights as they're produced, and can interact:

```
Control Panel: Insight Feed

┌──────────────────────────────────────────────────────────┐
│  INSIGHTS                              [Filter] [Sort]   │
│                                                          │
│  ⚡ ATTESTED  "The 7 Fano lines encode..."              │
│    Rigor: 8/10  Depth: 9/10  Natural: 8/10              │
│    From: Thread #12 → Seed "7 chakras..."               │
│    [👍 Promising] [👎 Not useful] [💬 Comment]           │
│                                                          │
│  ? DISPUTED  "The 22 shrutis map to..."                 │
│    Rigor: 6/10  Depth: 7/10  Natural: 5/10              │
│    Dispute: "Mapping is numerically forced"              │
│    [👍 Explore further] [👎 Discard] [💬 Comment]       │
│                                                          │
│  ✗ DISCARDED  "Chakra colors follow..."                 │
│    Reason: "Surface-level pattern, no structural basis"  │
│    [🔄 Reconsider] [💬 Comment]                         │
│                                                          │
│  ★ INTERESTING  "The Heawood graph and..."              │
│    Potential: 7/10  Needs: "Further exploration of..."   │
│    [📌 Prioritize] [💬 Comment]                          │
└──────────────────────────────────────────────────────────┘
```

API endpoints:

```
GET    /api/insights              — List insights with filters
GET    /api/insights/:id          — Full insight detail with review record
POST   /api/insights/:id/comment  — Add user comment
PUT    /api/insights/:id/verdict  — User override (endorse/dismiss/reconsider)
GET    /api/insights/:id/lineage  — Trace back to seed and thread
```

**User comments on insights** flow back into the system:

- **"Promising"** on an insight → boosts the priority of its source thread and related seeds
- **"Not useful"** → reduces priority, informs Explorer to avoid similar directions
- **A text comment** → stored and visible to the Documenter when incorporating the insight
- **"Reconsider"** on a discarded insight → re-queues it for review with user's reasoning as additional context

### 9.3 Document Annotations

The user interacts with the living document through annotations:

```
Control Panel: Document Viewer

┌──────────────────────────────────────────────────────────┐
│  THE MATHEMATICS OF THREE                                │
│                                                          │
│  ## 1. The Primary Triad                                 │
│                                                          │
│  The Fano plane is the simplest finite projective        │
│  plane...                                                │
│       ┌──────────────────────────────────┐               │
│       │ 📝 "Expand on why 3 points per  │               │
│       │     line is significant"         │               │
│       │     Status: Open                 │               │
│       └──────────────────────────────────┘               │
│                                                          │
│  ### 1.1 Incidence Structure                             │
│                                                          │
│  Each point lies on exactly 3 lines...                   │
│  ┌────────────────────────────────────────┐              │
│  │ 🔒 PROTECTED — don't modify this      │              │
│  │    paragraph (user-verified correct)   │              │
│  └────────────────────────────────────────┘              │
│                                                          │
│  [+ Add Comment]  [🔒 Protect Selection]                 │
│  [💡 Suggest Connection]                                 │
└──────────────────────────────────────────────────────────┘
```

API endpoints:

```
POST   /api/annotations           — Add comment/protection/suggestion
DELETE /api/annotations/:id       — Remove annotation
GET    /api/annotations           — List all annotations with status
PUT    /api/annotations/:id       — Update annotation content
GET    /api/document              — Current document with annotation markers
GET    /api/document/history      — Version history
```

### 9.4 Event Flow from User Actions

Every user action becomes an event that the system reacts to:

| User Action | Event | System Reaction |
|-------------|-------|-----------------|
| Add seed | `user.seed.created` | Explorer queues for exploration |
| Prioritize seed | `user.seed.prioritized` | Explorer adjusts thread selection |
| Comment on insight: "promising" | `user.insight.endorsed` | Explorer boosts source thread priority |
| Comment on insight: "not useful" | `user.insight.dismissed` | Explorer reduces thread priority, avoids similar |
| Add document comment | `user.annotation.created` | Documenter prioritizes addressing it |
| Protect document section | `user.annotation.protected` | Documenter marks section as immutable |
| Request research on topic | `user.research.requested` | Researcher creates high-priority task |
| Approve seed modification | `user.seed.modification.approved` | Explorer creates child seed, queues exploration |
| Reject seed modification | `user.seed.modification.rejected` | Explorer notes rejection, continues with original |

---

## 10. Infrastructure: Event Bus

### Problem

Components communicate through three incompatible mechanisms: HTTP proxying, file system polling, and subprocess control. Events take 30+ seconds to propagate. File polling creates race conditions.

### Design

An in-process async event bus with topic-based routing:

```python
# shared/events.py

@dataclass
class Event:
    topic: str          # e.g., "explorer.insight.attested"
    timestamp: datetime
    source: str         # Component that published
    payload: dict       # Event-specific data
    correlation_id: str # For tracing

class EventBus:
    """In-process async event bus with topic-based routing."""

    def __init__(self):
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)

    def subscribe(self, topic: str, handler: Callable[[Event], Awaitable[None]]):
        """Subscribe to events matching a topic pattern.

        Topics use dot-separated hierarchy: 'explorer.insight.attested'
        Wildcards: 'explorer.*' matches all explorer events.
        """
        self._subscribers[topic].append(handler)

    async def publish(self, event: Event):
        """Publish an event to all matching subscribers.

        Events are persisted to the state store before delivery,
        ensuring at-least-once delivery even after crashes.
        """
        await self._persist(event)
        for topic, handlers in self._subscribers.items():
            if self._matches(topic, event.topic):
                for handler in handlers:
                    try:
                        await handler(event)
                    except Exception as e:
                        log.error("event.handler.failed",
                                  topic=event.topic,
                                  handler=handler.__name__,
                                  error=str(e))
```

### Event Catalog

| Topic | Publisher | Subscribers | Payload |
|-------|-----------|-------------|---------|
| `explorer.thread.created` | Explorer | Control, Metrics | `{thread_id, seed_id, priority}` |
| `explorer.thread.retired` | Explorer | Control, Metrics | `{thread_id, reason, insights_produced}` |
| `explorer.insight.extracted` | Explorer | Dedup checker | `{insight_id, text, confidence}` |
| `explorer.insight.attested` | Explorer | Documenter, Researcher, Control | `{insight_id, text, tags, scores}` |
| `explorer.insight.discarded` | Explorer | Metrics | `{insight_id, reason, round}` |
| `explorer.insight.disputed` | Explorer | Control (user can weigh in) | `{insight_id, text, disagreement}` |
| `explorer.seed.modification.proposed` | Explorer | Control (user review) | `{seed_id, original, proposed, reason}` |
| `documenter.section.added` | Documenter | Researcher, Control | `{section_id, establishes}` |
| `documenter.insight.incorporated` | Documenter | Explorer, Control | `{insight_id, section_id}` |
| `documenter.research.requested` | Documenter | Researcher | `{topic, context, urgency}` |
| `researcher.finding.stored` | Researcher | Documenter, Explorer | `{finding_id, topic, summary}` |
| `researcher.evidence.supports` | Researcher | Explorer | `{insight_id, evidence_summary}` |
| `researcher.evidence.contradicts` | Researcher | Explorer | `{insight_id, evidence_summary}` |
| `user.seed.created` | Control | Explorer | `{seed_id, text, priority}` |
| `user.seed.prioritized` | Control | Explorer | `{seed_id, new_priority}` |
| `user.insight.endorsed` | Control | Explorer | `{insight_id, comment}` |
| `user.insight.dismissed` | Control | Explorer | `{insight_id, comment}` |
| `user.annotation.created` | Control | Documenter | `{annotation_id, type, content}` |
| `user.research.requested` | Control | Researcher | `{topic, context}` |
| `llm.request.completed` | LLM Client | Metrics, Quota | `{backend, tokens, cost, latency_ms}` |
| `llm.request.failed` | LLM Client | Circuit Breaker | `{backend, error_type}` |

### Why Not a Message Broker?

All components run on the same machine. Message volume is low (~100 events/hour). An in-process bus keeps operational complexity zero. If the system later scales to multiple machines, the `EventBus` interface stays the same — only the transport changes.

---

## 11. Infrastructure: State Store

### Problem

State is scattered across 12+ filesystem locations. JSON files suffer from race conditions, non-atomic writes, and no query capability.

### Design

A single SQLite database with WAL mode:

```sql
-- =============================================
-- PROJECT
-- =============================================

CREATE TABLE projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    goal TEXT NOT NULL,
    context TEXT NOT NULL,
    evaluation_criteria TEXT NOT NULL,   -- JSON array
    exploration_guidance TEXT,
    document_guidance TEXT,
    seed_modification_enabled BOOLEAN DEFAULT TRUE,
    seed_modification_require_approval BOOLEAN DEFAULT FALSE,
    research_domains TEXT DEFAULT '[]',  -- JSON array
    status TEXT NOT NULL CHECK(status IN ('active', 'paused', 'archived')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- =============================================
-- SEEDS
-- =============================================

CREATE TABLE seeds (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id),
    text TEXT NOT NULL,
    type TEXT NOT NULL CHECK(type IN ('axiom', 'conjecture', 'question')),
    priority INTEGER NOT NULL DEFAULT 5 CHECK(priority BETWEEN 1 AND 10),
    tags TEXT NOT NULL DEFAULT '[]',      -- JSON array
    confidence TEXT CHECK(confidence IN ('high', 'medium', 'low')),
    source TEXT,                          -- Where this idea came from
    notes TEXT,
    status TEXT NOT NULL CHECK(status IN ('active', 'explored', 'evolved', 'retired')),
    parent_seed_id TEXT REFERENCES seeds(id),  -- For evolved seeds
    modification_reason TEXT,                   -- Why the LLM proposed this change
    exploration_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- =============================================
-- THREADS
-- =============================================

CREATE TABLE threads (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id),
    seed_id TEXT NOT NULL REFERENCES seeds(id),
    status TEXT NOT NULL CHECK(status IN (
        'active', 'synthesizing', 'extracted', 'stalled', 'retired'
    )),
    priority INTEGER NOT NULL DEFAULT 5,
    exchange_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    retired_at TEXT,
    retire_reason TEXT
);

CREATE TABLE exchanges (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL REFERENCES threads(id),
    sequence INTEGER NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('explorer', 'critic', 'synthesizer')),
    model TEXT NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- =============================================
-- INSIGHTS
-- =============================================

CREATE TABLE insights (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id),
    text TEXT NOT NULL,
    confidence TEXT NOT NULL CHECK(confidence IN ('high', 'medium', 'low')),
    tags TEXT NOT NULL DEFAULT '[]',
    source_thread_id TEXT REFERENCES threads(id),
    extraction_model TEXT,
    status TEXT NOT NULL CHECK(status IN (
        'extracted',       -- Just pulled from thread
        'reviewing',       -- In review panel
        'attested',        -- Consensus: valid and valuable
        'discarded',       -- Consensus: rejected on merit
        'disputed',        -- LLMs disagree
        'interesting',     -- Shows potential, needs development
        'incorporating',   -- Documenter is working on it
        'incorporated',    -- Written into document
        'shelved',         -- Too many disputes, needs human review
        'transient_failure' -- API error, will retry
    )),
    evaluation_scores TEXT DEFAULT '{}',  -- JSON: {criterion_name: score}
    dispute_count INTEGER DEFAULT 0,
    transient_failure_count INTEGER DEFAULT 0,
    review_record TEXT,       -- JSON blob of full review history
    blessed_at TEXT,
    incorporated_at TEXT,
    incorporated_in_section TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE insight_dependencies (
    insight_id TEXT NOT NULL REFERENCES insights(id),
    depends_on_id TEXT NOT NULL REFERENCES insights(id),
    PRIMARY KEY (insight_id, depends_on_id)
);

-- =============================================
-- USER COMMENTS ON INSIGHTS
-- =============================================

CREATE TABLE insight_comments (
    id TEXT PRIMARY KEY,
    insight_id TEXT NOT NULL REFERENCES insights(id),
    comment_type TEXT NOT NULL CHECK(comment_type IN (
        'endorse', 'dismiss', 'reconsider', 'general'
    )),
    content TEXT,       -- Optional text explanation
    created_at TEXT NOT NULL
);

-- =============================================
-- DOCUMENT
-- =============================================

CREATE TABLE sections (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id),
    content TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('provisional', 'stable', 'needs_work')),
    establishes TEXT NOT NULL DEFAULT '[]',
    requires TEXT NOT NULL DEFAULT '[]',
    source_insight_id TEXT REFERENCES insights(id),
    review_count INTEGER DEFAULT 0,
    last_reviewed_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE concepts (
    name TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    established_in_section TEXT REFERENCES sections(id),
    project_id TEXT NOT NULL REFERENCES projects(id),
    domain TEXT
);

CREATE TABLE annotations (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id),
    type TEXT NOT NULL CHECK(type IN ('comment', 'protected', 'suggestion')),
    section_id TEXT REFERENCES sections(id),
    content TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('open', 'attempted', 'resolved')),
    attempt_count INTEGER DEFAULT 0,
    last_attempted_at TEXT,
    created_at TEXT NOT NULL,
    resolved_at TEXT
);

-- =============================================
-- RESEARCH
-- =============================================

CREATE TABLE sources (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id),
    url TEXT NOT NULL,
    domain TEXT,
    title TEXT,
    trust_score INTEGER DEFAULT 0,
    trust_tier TEXT,
    content_hash TEXT,
    evaluated_at TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE findings (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id),
    source_id TEXT REFERENCES sources(id),
    finding_type TEXT,
    summary TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    domain TEXT,
    related_insight_id TEXT REFERENCES insights(id),
    created_at TEXT NOT NULL
);

-- =============================================
-- EVENTS AND METRICS
-- =============================================

CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT REFERENCES projects(id),
    topic TEXT NOT NULL,
    source TEXT NOT NULL,
    payload TEXT NOT NULL,
    correlation_id TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT REFERENCES projects(id),
    name TEXT NOT NULL,
    value REAL NOT NULL,
    labels TEXT DEFAULT '{}',
    recorded_at TEXT NOT NULL
);

-- =============================================
-- INDEXES
-- =============================================

CREATE INDEX idx_seeds_project ON seeds(project_id, status);
CREATE INDEX idx_threads_project ON threads(project_id, status);
CREATE INDEX idx_insights_project ON insights(project_id, status);
CREATE INDEX idx_insights_status ON insights(status);
CREATE INDEX idx_sections_project ON sections(project_id);
CREATE INDEX idx_concepts_canonical ON concepts(canonical_name);
CREATE INDEX idx_annotations_project ON annotations(project_id, status);
CREATE INDEX idx_events_topic ON events(topic);
CREATE INDEX idx_events_created ON events(created_at);
CREATE INDEX idx_metrics_name ON metrics(name, recorded_at);
CREATE INDEX idx_insight_comments ON insight_comments(insight_id);
CREATE INDEX idx_findings_insight ON findings(related_insight_id);
```

### Data Access Layer

```python
# shared/store.py

class StateStore:
    """Centralized state management with ACID transactions."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None

    async def connect(self):
        self._conn = await aiosqlite.connect(self._db_path)
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")

    @asynccontextmanager
    async def transaction(self):
        """Atomic transaction context manager."""
        await self._conn.execute("BEGIN")
        try:
            yield self._conn
            await self._conn.commit()
        except Exception:
            await self._conn.rollback()
            raise

    async def attest_insight(self, insight_id: str, scores: dict) -> None:
        """Atomically transition insight to attested status."""
        async with self.transaction() as conn:
            await conn.execute(
                """UPDATE insights
                   SET status='attested', blessed_at=?, evaluation_scores=?
                   WHERE id=?""",
                (datetime.now().isoformat(), json.dumps(scores), insight_id)
            )
```

### Why SQLite?

- **Single machine** — no network overhead
- **ACID transactions** — atomic insight status transitions (no more partial writes)
- **WAL mode** — concurrent reads during writes (already used by the Orchestrator)
- **Zero deployment** — no server to install or manage
- **Rich queries** — SQL for complex filtering (`WHERE status='attested' AND project_id=?`)
- **Proven in this codebase** — the Researcher and Orchestrator already use SQLite successfully

---

## 12. Infrastructure: LLM Client

### Token Tracking and Cost Estimation

Every LLM call records token usage and estimated cost:

```python
async def _send_to_openrouter(self, backend, prompt, **kwargs) -> LLMResponse:
    # ... existing request logic ...

    if response.status == 200:
        data = await response.json()
        usage = data.get("usage", {})

        token_record = TokenUsage(
            backend=backend,
            model=resolved_model,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            estimated_cost=self._estimate_cost(resolved_model, usage),
        )

        await self._event_bus.publish(Event(
            topic="llm.request.completed",
            payload=asdict(token_record),
        ))

        return LLMResponse(success=True, text=text, token_usage=token_record)
```

### Circuit Breaker

Prevents cascading failures when a backend is down:

```python
class CircuitBreaker:
    """
    States:
    - CLOSED: Normal operation.
    - OPEN: Backend down, requests fail immediately.
    - HALF_OPEN: After cooldown, allow one test request.

    In v1, if OpenRouter returns 500s, all three components independently
    retry with exponential backoff — 3x the retry traffic. The circuit
    breaker says "backend X is down, stop trying for 60 seconds."
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._state: dict[str, str] = {}
        self._failure_counts: dict[str, int] = {}
        self._last_failure: dict[str, float] = {}

    def can_request(self, backend: str) -> bool:
        state = self._state.get(backend, "closed")
        if state == "closed":
            return True
        if state == "open":
            if time.time() - self._last_failure[backend] > self._recovery_timeout:
                self._state[backend] = "half_open"
                return True
            return False
        return True  # half_open: allow test request

    def record_success(self, backend: str):
        self._failure_counts[backend] = 0
        self._state[backend] = "closed"

    def record_failure(self, backend: str):
        self._failure_counts[backend] = self._failure_counts.get(backend, 0) + 1
        self._last_failure[backend] = time.time()
        if self._failure_counts[backend] >= self._failure_threshold:
            self._state[backend] = "open"
```

### Per-Module Budget Allocation

```yaml
# config.yaml
quotas:
  daily_budget_usd: 10.00
  per_module_weights:
    explorer: 50      # Gets 50% = $5.00/day
    documenter: 35    # Gets 35% = $3.50/day
    researcher: 15    # Gets 15% = $1.50/day
  alert_at_percent: 80
```

The `QuotaAllocator` subscribes to `llm.request.completed` events and accumulates per-module spend. When a module exceeds its budget, requests are queued until the next day or until budget is manually increased.

### 12.5 Consensus Engine Architecture

The consensus engine is the system's decision-making backbone. It is used by:
- **Explorer**: Review panel attestation (is this insight valid and valuable?)
- **Documenter**: Planning consensus (what work should we do next?)
- **Researcher**: Trust evaluation (is this source credible?)

v1 has this split across two incompatible implementations. v2 unifies them.

#### Unified Interface

```python
# llm/src/consensus/engine.py

class ConsensusEngine:
    """Shared multi-LLM agreement engine.

    Configurable for different use cases via ConsensusConfig.
    """

    def __init__(self, llm_client: LLMClient, config: ConsensusConfig):
        self.llm_client = llm_client
        self.config = config

    async def run(
        self,
        task: ConsensusTask,
        backends: list[str] | None = None,
    ) -> ConsensusResult:
        """Execute multi-round consensus on a task.

        Args:
            task: The task to evaluate (insight review, trust eval, planning, etc.)
            backends: Override default backend list

        Returns:
            ConsensusResult with verdict, scores, round history, and vote breakdown
        """
        backends = backends or self.config.backends
        rounds_completed = []

        for round_num in range(1, self.config.max_rounds + 1):
            round_result = await self._execute_round(
                task, backends, round_num, rounds_completed
            )
            rounds_completed.append(round_result)

            if round_result.is_converged:
                break

        return self._compile_result(rounds_completed)

    async def _execute_round(
        self,
        task: ConsensusTask,
        backends: list[str],
        round_num: int,
        prior_rounds: list[RoundResult],
    ) -> RoundResult:
        """Execute a single consensus round with error handling."""
        responses = []

        for backend in backends:
            for attempt in range(self.config.max_retries + 1):
                try:
                    prompt = task.get_prompt(round_num, prior_rounds, backend)
                    response = await self.llm_client.send(backend, prompt)
                    parsed = task.parse_response(response.text)

                    if parsed.is_valid:
                        responses.append(ValidatedResponse(
                            backend=backend, parsed=parsed, raw=response.text
                        ))
                        break
                    else:
                        log.warning("consensus.response.invalid",
                                    backend=backend, round=round_num,
                                    reason=parsed.error)
                except Exception as e:
                    if attempt < self.config.max_retries:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        log.error("consensus.round.backend_failed",
                                  backend=backend, round=round_num,
                                  error=str(e))

        if len(responses) < self.config.min_valid_responses:
            raise InsufficientResponsesError(
                f"Only {len(responses)} valid responses "
                f"(need {self.config.min_valid_responses})"
            )

        return self._evaluate_round(responses, round_num)
```

#### Configuration

```python
@dataclass
class ConsensusConfig:
    backends: list[str]                # ["gemini", "chatgpt", "claude"]
    max_rounds: int = 4                # 1-4 rounds
    min_valid_responses: int = 2       # Minimum backends that must respond validly
    max_retries: int = 2               # Per-backend retry count
    convergence_method: str = "semantic"  # "semantic", "jaccard", "exact"
    convergence_threshold: float = 0.7
    decision_method: str = "majority"  # "majority", "unanimous", "supermajority"
    minimum_agreement: float = 0.66
```

These values come from `config.yaml` → `consensus:` section, making all magic numbers configurable.

#### Response Validation

```python
@dataclass
class ParsedResponse:
    """Result of parsing an LLM response."""
    is_valid: bool
    verdict: str | None          # "accept", "reject", "uncertain"
    scores: dict[str, float]     # {criterion_name: score}
    reasoning: str | None
    error: str | None            # Why parsing failed, if invalid

def parse_review_response(text: str, criteria_names: list[str]) -> ParsedResponse:
    """Multi-strategy response parser.

    Tries in order:
    1. JSON block extraction (```json ... ```)
    2. Regex pattern matching (case-insensitive)
    3. Returns invalid with error description (for LLM re-extraction fallback)
    """
    # Strategy 1: JSON
    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return ParsedResponse(
                is_valid=True,
                verdict=_normalize_verdict(data.get("verdict")),
                scores=_extract_scores(data, criteria_names),
                reasoning=data.get("reasoning"),
                error=None,
            )
        except json.JSONDecodeError:
            pass

    # Strategy 2: Regex (case-insensitive)
    verdict = _regex_extract_verdict(text)
    scores = _regex_extract_scores(text, criteria_names)
    if verdict:
        return ParsedResponse(
            is_valid=True, verdict=verdict, scores=scores,
            reasoning=text, error=None,
        )

    # Strategy 3: Failed — return invalid for potential LLM re-extraction
    return ParsedResponse(
        is_valid=False, verdict=None, scores={},
        reasoning=None, error=f"Could not parse verdict from response ({len(text)} chars)",
    )
```

#### Deterministic Hashing

All content hashing uses SHA-256, not Python's non-deterministic `hash()`:

```python
import hashlib

def content_hash(text: str) -> str:
    """Deterministic content hash for dedup and comparison."""
    normalized = text.strip().lower()
    normalized = re.sub(r'\s+', ' ', normalized)
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]
```

---

## 13. Infrastructure: Configuration

### The Problem

v1 has four config files with overlapping and sometimes contradictory settings. The explorer config silently overrides root values through dict merging.

### The Solution

Two-level configuration:

1. **`config.yaml`** — Infrastructure settings (LLM models, rate limits, database path, ports). Rarely changes. Shared across all projects.

2. **`projects/<name>.yaml`** — Project settings (goal, seeds, evaluation criteria, prompts). Changes per research effort. The user's primary creative input.

```yaml
# config.yaml — Infrastructure (shared)

llm:
  api_key_env: "OPENROUTER_API_KEY"
  endpoint: "https://openrouter.ai/api/v1"
  models:
    gemini: "google/gemini-2.0-flash-thinking-exp-01-21"
    chatgpt: "openai/gpt-4o"
    claude: "anthropic/claude-sonnet-4-20250514"
    deepseek: "deepseek/deepseek-r1"
  rate_limits:
    gemini: 10
    chatgpt: 60
    claude: 50
    deepseek: 10
  default_timeout_seconds: 300
  max_retries: 3

consensus:
  backends: [gemini, chatgpt, claude]
  max_rounds: 3
  confidence_threshold: 0.8

deduplication:
  enabled: true
  use_signature_check: true
  use_heuristic_check: false
  use_llm_check: true
  batch_size: 20
  model: "claude"

logging:
  level: "INFO"
  directory: "./logs"
  max_bytes: 10485760
  backup_count: 10

explorer:
  max_active_threads: 3
  min_exchanges_for_chunk: 4
  max_exchanges_per_thread: 12
  thread_retirement:
    max_idle_hours: 48
    max_reexplore_count: 3
  review_panel:
    enabled: true
    max_refinement_rounds: 2
  model_weights:
    exploration: { gemini: 60, chatgpt: 40 }
    critique: { gemini: 30, chatgpt: 70 }

documenter:
  document_dir: "data/document"
  snapshot_time: "00:00"
  work_allocation:
    new_material: 70
    review_existing: 30
  context:
    max_tokens: 8000
  termination:
    max_consecutive_disputes: 3
    max_transient_failures_before_retry: 5
    max_consensus_calls_per_session: 100

researcher:
  idle_polling_interval_seconds: 300
  max_questions_per_cycle: 20
  trust:
    min_trust_score: 50

control:
  host: "127.0.0.1"
  port: 8080

quotas:
  daily_budget_usd: 10.00
  per_module_weights:
    explorer: 50
    documenter: 35
    researcher: 15
  alert_at_percent: 80

retention:
  logs:
    max_age_days: 30
    max_size_mb: 500
  events:
    max_age_days: 90
  metrics:
    max_age_days: 365
  snapshots:
    max_count: 90

# Active project (or set via CLI: --project fano-mathematics)
active_project: "fano-mathematics"
```

### Config Loader

```python
# shared/config.py

class Config:
    """Hierarchical configuration with environment variable overrides."""

    def __init__(self, config_path: Path = None, project_path: Path = None):
        self._config_path = config_path or FANO_ROOT / "config.yaml"
        self._project_path = project_path
        self._data: dict = {}
        self._project: dict = {}

    def load(self) -> None:
        with open(self._config_path) as f:
            self._data = yaml.safe_load(f)

        if self._project_path:
            with open(self._project_path) as f:
                self._project = yaml.safe_load(f).get("project", {})

        self._apply_env_overrides()
        self._validate()

    def get(self, dotted_key: str, default=None):
        """Access nested config: config.get('explorer.max_active_threads')"""
        keys = dotted_key.split('.')
        value = self._data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    @property
    def project(self) -> dict:
        """Access project configuration."""
        return self._project

    def _apply_env_overrides(self):
        """Override config via FANO_EXPLORER_MAX_ACTIVE_THREADS=5, etc."""
        prefix = "FANO_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                dotted = key[len(prefix):].lower().replace("__", ".")
                self._set_nested(dotted, self._coerce_type(value))

    def _validate(self):
        required = ["llm.api_key_env", "llm.models", "consensus.backends"]
        for key in required:
            if self.get(key) is None:
                raise ConfigError(f"Required config key missing: {key}")
```

---

## 14. Observability and Operations

### Insight Lifecycle State Machine

The insight lifecycle cleanly separates substantive rejection from transient failure:

```
                    ┌──────────────┐
                    │  extracted   │
                    └──────┬───────┘
                           │ dedup passes
                           ▼
                    ┌──────────────┐
                    │  reviewing   │
                    └──────┬───────┘
                           │
              ┌────────────┼──────────────┬──────────────┐
              │            │              │              │
              ▼            ▼              ▼              ▼
       ┌──────────┐ ┌───────────┐ ┌───────────┐ ┌─────────────┐
       │ attested │ │ discarded │ │ disputed  │ │ interesting │
       └────┬─────┘ └───────────┘ └─────┬─────┘ └──────┬──────┘
            │                           │               │
            │ documenter picks up       │ user/explorer  │ further
            ▼                           │ revisits       │ development
     ┌──────────────┐                   └──→ reviewing   └──→ reviewing
     │incorporating │
     └──────┬───────┘
            │
   ┌────────┼──────────┐
   │        │          │
   ▼        ▼          ▼
┌────────┐ ┌────────┐ ┌──────────────────┐
│incorpo-│ │disputed│ │ transient_failure │
│rated   │ │(merit) │ │ (API error)      │
└────────┘ └───┬────┘ └────────┬─────────┘
               │               │ retry with backoff
               │               └──→ incorporating
               │
               │ after max_disputes
               ▼
        ┌──────────┐
        │ shelved  │ (needs human review)
        └──────────┘
```

The critical distinction: **transient failures** (API timeout, rate limit) increment a separate counter and trigger retries with backoff. **Substantive disputes** (LLM says "this lacks rigor") increment the dispute counter and eventually shelve the insight. In v1, both hit the same counter, so network jitter can permanently kill good insights.

```python
class InsightLifecycle:
    TRANSIENT_ERRORS = {"timeout", "rate_limited", "connection_error", "api_error"}

    async def record_failure(self, insight_id: str, error_type: str, details: str):
        if error_type in self.TRANSIENT_ERRORS:
            await self._record_transient_failure(insight_id, error_type, details)
        else:
            await self._record_dispute(insight_id, details)

    async def _record_transient_failure(self, insight_id, error_type, details):
        async with self.store.transaction() as conn:
            await conn.execute("""
                UPDATE insights
                SET transient_failure_count = transient_failure_count + 1,
                    status = 'transient_failure', updated_at = ?
                WHERE id = ?
            """, (now(), insight_id))

        # Schedule retry with exponential backoff
        count = await self._get_transient_count(insight_id)
        backoff = min(300 * (2 ** count), 3600)  # Max 1 hour

    async def _record_dispute(self, insight_id, reason):
        async with self.store.transaction() as conn:
            await conn.execute("""
                UPDATE insights
                SET dispute_count = dispute_count + 1,
                    status = CASE
                        WHEN dispute_count + 1 >= ? THEN 'shelved'
                        ELSE 'disputed'
                    END, updated_at = ?
                WHERE id = ?
            """, (self.max_disputes, now(), insight_id))

        await self.event_bus.publish(Event(
            topic="documenter.insight.disputed",
            payload={"insight_id": insight_id, "reason": reason}
        ))
```

### Metrics

Lightweight counters and histograms stored in SQLite, exposed via the Control Panel:

| Metric | Type | Purpose |
|--------|------|---------|
| `insights.attested.total` | Counter | How many insights attested |
| `insights.discarded.total` | Counter | Rejection rate |
| `insights.disputed.total` | Counter | Disagreement rate |
| `llm.request.duration_seconds` | Histogram | Latency tracking |
| `llm.request.tokens` | Counter | Token consumption by backend |
| `llm.request.cost_usd` | Counter | Cost tracking by module |
| `llm.circuit_breaker.state` | Gauge | Backend health |
| `consensus.rounds_needed` | Histogram | How often Round 2/3 needed? |
| `explorer.threads.active` | Gauge | Current thread count |
| `documenter.sections.total` | Gauge | Document growth |
| `researcher.findings.stored` | Counter | Research output by domain |
| `user.interactions.total` | Counter | User engagement by type |

### Dashboard

```
┌──────────────── RESEARCH ASSISTANT ─────────────────┐
│  Project: The Mathematics of Three                   │
│                                                      │
│  TODAY'S ACTIVITY                                    │
│  ├─ Insights attested: 3                            │
│  ├─ Insights disputed: 1  (user weighed in on 1)   │
│  ├─ Sections added: 2                               │
│  ├─ Research findings: 7                             │
│  └─ User interactions: 4                             │
│                                                      │
│  LLM COSTS (today)                                   │
│  ├─ Explorer:   $2.14  (43% of budget)              │
│  ├─ Documenter: $1.87  (53% of budget)              │
│  ├─ Researcher: $0.32  (21% of budget)              │
│  └─ Total:      $4.33  (43% of $10.00)              │
│                                                      │
│  BACKEND HEALTH                                      │
│  ├─ gemini:    ● UP   (10 rpm, 3.2s avg)            │
│  ├─ chatgpt:   ● UP   (60 rpm, 2.1s avg)            │
│  ├─ claude:    ● UP   (50 rpm, 1.8s avg)             │
│  └─ deepseek:  ○ DOWN (since 14:32)                  │
│                                                      │
│  SEED STATUS                                         │
│  ├─ Active: 4   Explored: 8   Evolved: 2            │
│  └─ Pending modifications: 1 (awaiting approval)    │
│                                                      │
│  CONSENSUS QUALITY                                   │
│  ├─ Round 1 unanimous: 68%                           │
│  ├─ Round 2 resolved:  24%                           │
│  └─ Round 3 needed:     8%                           │
└──────────────────────────────────────────────────────┘
```

### Operational Hygiene

**Immediate cleanup**:
- Delete `explorer/browser_data/` (983MB dead cache)
- Remove browser config sections from explorer config
- Add to `.gitignore`

**Data retention** (configured in `config.yaml`):
- Events: 90 days
- Metrics: 365 days
- Document snapshots: 90 most recent
- Logs: 30 days, 500MB max

**CLI entry points** in `pyproject.toml`:
```toml
[project.scripts]
research-assistant = "control.cli:main"
ra-explorer = "explorer.fano_explorer:main"
ra-documenter = "documenter.main:main"
ra-researcher = "researcher.main:main"
ra-control = "control.server:main"
```

---

## 15. Testing Strategy

The system must be developed test-first. v1 has 576 tests across 26 modules — a decent foundation — but with critical coverage gaps. v2 requires comprehensive testing at every layer.

### 15.1 Current Test Coverage (v1 Baseline)

**Well-tested** (keep and extend):
- Consensus rounds (`tests/llm/test_consensus_rounds.py`) — round progression, vote counting
- Orchestrator state management (`tests/orchestrator/`) — state transitions, checkpoint/restore
- Control panel endpoints (`tests/control/`) — API routes, process management
- Structured logging (`tests/shared/test_logging.py`) — JSON format, field validation
- Deduplication pipeline (`tests/shared/test_dedup*.py`) — 3-layer dedup logic

**NOT tested** (critical gaps for v2):
- Insight lifecycle transitions (extracted → reviewing → attested/discarded/disputed)
- Error recovery (what happens when an LLM call fails mid-review?)
- Concurrent access patterns (multiple writers to insight state)
- Documenter main loop and opportunity processor pipeline
- Researcher pipeline end-to-end
- Response parsing robustness (malformed LLM output, edge cases)
- Seed modification workflow (propose → approve/reject → child creation)
- Thread priority adjustment and retirement
- Event bus delivery guarantees
- Cross-component integration (Explorer attests → Documenter picks up)

### 15.2 Test Architecture

```
tests/
├── unit/                           # Pure logic, no I/O, fast
│   ├── consensus/
│   │   ├── test_convergence.py     # Convergence detection algorithms
│   │   ├── test_vote_counting.py   # Vote tallying, majority rules
│   │   ├── test_error_filtering.py # Error responses excluded from votes
│   │   └── test_response_parsing.py # Verdict extraction from LLM text
│   ├── explorer/
│   │   ├── test_thread_lifecycle.py
│   │   ├── test_seed_lifecycle.py
│   │   ├── test_priority_scoring.py
│   │   └── test_insight_extraction.py
│   ├── documenter/
│   │   ├── test_annotation_state.py
│   │   ├── test_comment_retry.py
│   │   ├── test_concept_canonicalization.py
│   │   └── test_context_window.py
│   ├── researcher/
│   │   ├── test_question_generation.py
│   │   ├── test_trust_scoring.py
│   │   └── test_extraction_pipeline.py
│   └── shared/
│       ├── test_store.py           # StateStore CRUD, transactions
│       ├── test_events.py          # EventBus pub/sub, ordering
│       ├── test_config.py          # Config loading, validation
│       └── test_deduplication.py   # Existing (keep)
├── integration/                    # Component interactions, uses real DB
│   ├── test_insight_lifecycle.py   # Full flow: extract → review → attest → incorporate
│   ├── test_event_routing.py       # Events flow between components correctly
│   ├── test_seed_modification.py   # Propose → approve → child seed → re-explore
│   ├── test_directed_research.py   # Documenter requests → Researcher delivers
│   ├── test_user_actions.py        # User comment → priority adjustment → thread change
│   └── test_failure_recovery.py    # Transient failures retry, disputes shelve
├── contract/                       # LLM response format validation
│   ├── test_review_response.py     # Validates LLM output matches expected schema
│   ├── test_extraction_response.py # Validates insight extraction output format
│   └── test_research_response.py   # Validates research question/answer format
└── conftest.py                     # Shared fixtures, mock LLM factory
```

### 15.3 Testing Principles

**1. Mock LLMs at the boundary, not inside algorithms.**

```python
# Good: mock the LLM client, test the consensus logic
@pytest.fixture
def mock_llm():
    """Returns a function that creates predictable LLM responses."""
    async def _mock(backend, prompt, **kwargs):
        return LLMResponse(success=True, text="⚡ Accept\nRigor: 8/10\n...")
    return _mock

async def test_unanimous_accept_attests(mock_llm, store):
    engine = ConsensusEngine(llm_client=mock_llm, store=store)
    result = await engine.review(insight, criteria)
    assert result.status == "attested"

# Bad: mock internal convergence detection
```

**2. Property-based testing for convergence logic.**

```python
from hypothesis import given, strategies as st

@given(
    votes=st.lists(st.sampled_from(["accept", "reject", "uncertain"]), min_size=2, max_size=5),
    threshold=st.floats(min_value=0.5, max_value=1.0)
)
def test_majority_decision_is_deterministic(votes, threshold):
    result1 = compute_majority(votes, threshold)
    result2 = compute_majority(votes, threshold)
    assert result1 == result2

@given(votes=st.lists(st.just("accept"), min_size=2, max_size=5))
def test_unanimous_accept_always_attests(votes):
    assert compute_majority(votes, 0.5) == "accept"
```

**3. Contract tests for LLM response parsing.**

Since LLM output is non-deterministic, test the parser against a corpus of real and adversarial responses:

```python
RESPONSE_CORPUS = [
    # Happy path
    ("⚡ Accept\nRigor: 8/10", {"verdict": "accept", "scores": {"rigor": 8}}),
    # Case variations
    ("⚡ ACCEPT\nrigor: 8/10", {"verdict": "accept", "scores": {"rigor": 8}}),
    # Missing verdict marker
    ("I think this is strong.\nRigor: 9/10", {"verdict": "uncertain", "scores": {"rigor": 9}}),
    # Verdict buried in prose
    ("After careful review... ⚡ Accept", {"verdict": "accept", "scores": {}}),
    # Error/empty response
    ("", {"verdict": "error", "scores": {}}),
    ("I apologize, I cannot...", {"verdict": "error", "scores": {}}),
]

@pytest.mark.parametrize("response_text,expected", RESPONSE_CORPUS)
def test_parse_review_response(response_text, expected):
    result = parse_review_response(response_text, criteria_names=["rigor"])
    assert result["verdict"] == expected["verdict"]
```

**4. Integration tests use real SQLite (in-memory).**

```python
@pytest.fixture
async def store():
    """In-memory SQLite store for integration tests."""
    s = StateStore(":memory:")
    await s.connect()
    await s.create_tables()
    yield s
    await s.close()
```

**5. Failure injection tests.**

```python
async def test_llm_failure_during_review_excluded_from_vote(store):
    """An LLM that errors out should not count as a vote."""
    async def failing_llm(backend, prompt, **kwargs):
        if backend == "gemini":
            raise TimeoutError("API timeout")
        return LLMResponse(success=True, text="⚡ Accept\nRigor: 8/10")

    engine = ConsensusEngine(llm_client=failing_llm, store=store)
    result = await engine.review(insight, criteria, backends=["gemini", "chatgpt", "claude"])

    assert result.valid_vote_count == 2  # gemini excluded
    assert result.status == "attested"   # remaining 2 agree

async def test_too_many_failures_retries_round(store):
    """If >50% of backends fail, the round should be retried, not declared converged."""
    call_count = 0
    async def intermittent_llm(backend, prompt, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise ConnectionError("Network error")
        return LLMResponse(success=True, text="⚡ Accept\nRigor: 8/10")

    engine = ConsensusEngine(llm_client=intermittent_llm, store=store)
    result = await engine.review(insight, criteria)
    assert call_count > 3  # Retried
```

### 15.4 Test-Driven Migration Approach

Each migration phase begins with tests:

1. **Write failing tests** for the new behavior
2. **Implement** until tests pass
3. **Run existing tests** to ensure no regressions
4. **Add edge case tests** discovered during implementation

The test suite is the acceptance criteria for each phase. A phase is not complete until all its tests pass AND all pre-existing tests still pass.

---

## 16. Implementation Strategy

### Why Clean Rewrite, Not Incremental Migration

After deep code review of the v1 codebase (~24,000 LOC across all modules), the original "incremental migration" plan has been replaced with a **clean rewrite** that ports valuable algorithms.

**The case for rewrite:**

| Factor | Detail |
|--------|--------|
| v1 production status | **Not in production** — no users to migrate, no uptime to preserve |
| Domain coupling | **300+ hardcoded references** to Fano/Sadhguru/Yoga scattered across 15+ files — not localized, not easy to extract |
| Architectural gap | v2 introduces EventBus, StateStore, Project Model, unified ConsensusEngine — none exist in v1, so "modifying" means building these while working around v1's file-polling and JSON-scattered patterns |
| Technical debt | v1 was designed and developed simultaneously with frequent idea changes — dead code (800+ LOC), incomplete refactors (adapter_v3), broken logic (comment retry, dispute/failure conflation) |
| Code reuse reality | The 52% "reusable" code is mostly small, clean modules (`shared/logging`, `orchestrator/`, `llm/client.py`) that are faster to reimplement cleanly than to wire into v2's new architecture |

**What's NOT a rewrite:**

This is not "throw everything away and start over." It's "implement v2's architecture from the detailed blueprint, porting the algorithms that v1 proved work." The v2 architecture document IS the design — 2700+ lines of schemas, interfaces, state machines, and code examples.

### What to Port (Algorithms, Not Code)

These v1 ideas are proven through use and should be reimplemented in v2's clean architecture:

| Algorithm | Source in v1 | What to Port |
|-----------|-------------|--------------|
| OpenRouter API patterns | `llm/src/client.py` | Rate limiting, retry with backoff, model mapping, session management |
| 4-round consensus escalation | `explorer/src/review_panel/reviewer.py` | Round progression: independent → debate → deliberation → tiebreaker |
| Response parsing | `llm/src/consensus/response_parser.py` | Regex patterns for verdict/score extraction (extend with JSON and LLM fallback) |
| 3-layer deduplication | `shared/deduplication/` | Pipeline: content-hash → heuristic text similarity → LLM semantic check |
| WAL-based state recovery | `orchestrator/state.py` | Write-ahead log + checkpoint pattern for crash recovery |
| Priority queue scheduling | `orchestrator/scheduler.py` | Dynamic priority with JIT worker spawning |
| Structured JSON logging | `shared/logging/` | JSON Lines format, correlation IDs, component/module naming |
| Convergence detection | `llm/src/consensus/convergence.py` | Text similarity algorithm for determining if LLMs are converging |

### What to Drop (Not Worth Porting)

| v1 Code | LOC | Why Drop |
|---------|-----|----------|
| All prompt files | ~2,000 | Hardcoded to Fano/Sadhguru domain. v2 uses project-config templates |
| `explorer/config.yaml` | 342 | Domain-specific. Replaced by 2-level config |
| `documenter/config.yaml` | 33 | Duplicate. Replaced by 2-level config |
| `researcher/config/domains.yaml` | ~100 | Hardcoded domain keywords. Replaced by project config |
| `explorer/src/review_panel/reviewer.py` | 880 | Tangled logic. Rebuild as parameterized ConsensusEngine |
| `documenter/adapter_v3.py` + related | ~800 | Dead code (architect.py, mason.py, consensus_board.py — unused experiment) |
| All file-based state management | ~500 | Replaced by SQLite StateStore |
| All file-based polling/observers | ~300 | Replaced by EventBus |
| `explorer/axioms/` (seeds.yaml, target_numbers.yaml) | ~200 | Static domain-specific seeds. Replaced by database-backed seed lifecycle |
| `researcher/sources/browser.py` | ~150 | Dead Playwright code |
| `explorer/browser_data/` | 983MB | Dead Playwright cache |

### Build Phases

The system is built bottom-up. Each phase produces a working, tested layer that the next phase builds on. v1 code is referenced only as algorithmic inspiration.

#### Phase 1: Foundation

Build the infrastructure layer that everything else depends on.

| Component | Description | Tests |
|-----------|-------------|-------|
| `shared/store.py` | SQLite StateStore with WAL, transactions, full schema | CRUD for all tables, transaction rollback, concurrent reads |
| `shared/events.py` | In-process async EventBus with topic routing | Pub/sub ordering, wildcard matching, handler errors don't crash bus |
| `shared/config.py` | 2-level config loader (infrastructure + project) | Loading, validation, env overrides, project switching |
| `projects/fano-mathematics.yaml` | First project config | Validates against schema |

**Deliverable**: Foundation infrastructure, fully tested, zero domain coupling.

#### Phase 2: LLM Client + Consensus Engine

Build the LLM access layer and the decision-making backbone.

| Component | Description | Tests |
|-----------|-------------|-------|
| `llm/src/client.py` | OpenRouter client with rate limiting, retry, circuit breaker, token tracking | Rate limit enforcement, retry backoff, circuit breaker states, cost estimation |
| `llm/src/consensus/engine.py` | Unified ConsensusEngine (replaces both v1 implementations) | Error filtering, convergence detection, vote counting, variable reviewer count |
| `llm/src/consensus/parsing.py` | Multi-strategy response parser (JSON → regex → LLM fallback) | Response corpus (happy path, edge cases, adversarial), property-based tests |

**Port from v1**: Rate limiting patterns, convergence algorithm, parsing regexes.

**Deliverable**: Can make LLM calls, run multi-round consensus, parse responses robustly.

#### Phase 3: Explorer Engine

Build the pure-reasoning engine.

| Component | Description | Tests |
|-----------|-------------|-------|
| Seed lifecycle | CRUD + modification proposals + lineage tracking | State transitions, modification workflow, lineage queries |
| Thread management | Create/resume/retire threads with dynamic priority | Priority adjustment, retirement conditions, exchange-boundary preemption |
| Insight extraction | Chunk threads into atomic insights, dedup | Extraction from thread history, cross-thread dedup |
| Review panel | 4-round attestation using ConsensusEngine | Full flow: extract → review → attest/discard/dispute, failure injection |
| Prompt templates | All prompts parameterized by project config | Template rendering with different project configs |

**Port from v1**: 4-round escalation pattern, extraction chunking logic, dedup pipeline approach.

**Deliverable**: Can explore seeds, debate insights, produce attested/discarded/disputed results.

#### Phase 4: Documenter Engine

Build the synthesis engine.

| Component | Description | Tests |
|-----------|-------------|-------|
| Document structure | Sections, concepts, prerequisite tracking in database | Section CRUD, concept canonicalization, dependency queries |
| Opportunity processor | Pipeline: dedup → evaluate → draft → add (with transient/dispute separation) | Each stage independently, failure at each stage, rollback |
| Annotation system | Comments, protected sections, suggestions — single source of truth in DB | State transitions, priority ordering, rendering |
| Context window | Sliding window based on relevance, not "show everything" | Token budget enforcement, relevance ranking |
| Planning | Work allocation (new material vs. review) with actual-spend tracking | Allocation balance, rebalancing triggers |

**Port from v1**: Work planning algorithm, concept tracking approach, document structure patterns.

**Deliverable**: Can take attested insights and build a coherent document with user annotations.

#### Phase 5: Researcher Engine + Integration

Build the external evidence layer and wire everything together.

| Component | Description | Tests |
|-----------|-------------|-------|
| Research orchestrator | Question generation, search, extraction, trust evaluation | Question quality, search execution, finding extraction |
| Trust evaluation | Uses ConsensusEngine for multi-LLM source credibility | Consensus-based scoring, caching, content-change detection |
| Directed research | Documenter requests → Researcher delivers | Event-driven request/response flow |
| Evidence feedback | Researcher findings influence Explorer priorities | Supports → boost priority, contradicts → reduce priority |
| User interaction | Seed CRUD API, insight comments, document annotations | Each user action → correct system reaction |

**Port from v1**: Source evaluation patterns, question generation approach.

**Deliverable**: Full pipeline working end-to-end: seeds → explore → research → document.

#### Phase 6: Control Panel + Polish

Build the UI layer and operational maturity.

| Component | Description | Tests |
|-----------|-------------|-------|
| Control panel | Flask app with seed management, insight feed, document viewer | API endpoints, status display, user action routing |
| Metrics dashboard | LLM costs, consensus quality, pipeline throughput | Metric collection, aggregation, display |
| CLI entry points | `research-assistant`, `ra-explorer`, etc. | Startup, project selection, graceful shutdown |
| Second project config | Business planning template to validate generalization | Same engine code, different project config → different behavior |

**Deliverable**: Complete, operational system validated on two different research domains.

---

## 17. Appendix: File Layout

### Current (v1)

```
fano/
├── config.yaml                          # Platform config (171 lines)
├── explorer/
│   ├── config.yaml                      # DUPLICATE config (342 lines)
│   ├── browser_data/                    # DEAD: 983MB Playwright cache
│   ├── axioms/
│   │   ├── seeds.yaml                   # HARDCODED domain seeds
│   │   └── target_numbers.yaml          # HARDCODED domain numbers
│   ├── data/
│   │   ├── explorations/*.json          # Thread state (file-per-thread)
│   │   ├── chunks/insights/blessed/     # Blessed insights (file-per-insight)
│   │   ├── chunks/insights/pending/     # Pending insights
│   │   ├── blessed_insights.json        # Index (can diverge from files!)
│   │   ├── reviews/                     # Review records
│   │   ├── chat_logs/                   # Unbounded (14MB)
│   │   └── fano_explorer.db             # SQLite (barely used)
│   └── src/
│       ├── chunking/prompts.py          # HARDCODED domain prompts
│       └── review_panel/prompts/        # HARDCODED domain prompts
├── documenter/
│   ├── config.yaml                      # DUPLICATE config (33 lines)
│   ├── document/
│   │   ├── main.md                      # Generated document
│   │   ├── annotations.json             # Diverges from inline markers
│   │   ├── archive/                     # Unbounded snapshots
│   │   └── .versions/
│   └── src/
├── researcher/
│   ├── config/
│   │   ├── settings.yaml                # ISOLATED config
│   │   ├── domains.yaml                 # HARDCODED domain keywords
│   │   └── templates.yaml
│   └── data/
│       └── researcher.db                # Separate SQLite
├── orchestrator/
│   └── data/
│       ├── checkpoint.json
│       └── wal.jsonl
├── control/
├── shared/
├── llm/
└── logs/                                # Unbounded (124MB+)
```

### Proposed (v2)

```
fano/
├── config.yaml                          # Infrastructure config (shared)
├── projects/
│   ├── fano-mathematics.yaml            # Project: goal, seeds, criteria, prompts
│   ├── ev-charging-europe.yaml          # Another project (example)
│   └── ai-journalism.yaml              # Another project (example)
├── data/
│   ├── fano.db                          # Single SQLite database (all state)
│   └── document/
│       ├── main.md                      # Rendered document output
│       └── archive/                     # Snapshots (with retention)
├── explorer/
│   └── src/
│       ├── orchestration/
│       ├── chunking/
│       │   └── prompts.py              # TEMPLATES reading from project config
│       └── review_panel/
│           └── prompts/
│               └── templates.py         # TEMPLATES reading from project config
├── documenter/
│   └── src/
├── researcher/
│   └── src/
├── orchestrator/
│   └── src/
├── control/
│   ├── server.py
│   ├── blueprints/
│   │   ├── seeds.py                     # NEW: seed management API
│   │   ├── insights.py                  # NEW: insight feed + comments API
│   │   ├── annotations.py              # REWORKED: unified annotation API
│   │   ├── projects.py                  # NEW: project management API
│   │   └── ...
│   └── templates/
├── shared/
│   ├── store.py                         # NEW: centralized state store
│   ├── events.py                        # NEW: event bus
│   ├── config.py                        # NEW: config + project loader
│   ├── metrics.py                       # NEW: metrics collector
│   ├── logging/                         # Existing (kept)
│   └── deduplication/                   # Existing (kept)
├── llm/
│   └── src/
│       ├── client.py                    # + token tracking, circuit breaker
│       └── consensus/
│           ├── engine.py                # REWORKED: unified ConsensusEngine
│           ├── parsing.py               # NEW: multi-strategy response parser
│           └── config.py                # NEW: ConsensusConfig dataclass
├── tests/
│   ├── unit/
│   │   ├── consensus/                   # Convergence, vote counting, error filtering
│   │   ├── explorer/                    # Thread/seed lifecycle, priority, extraction
│   │   ├── documenter/                  # Annotations, comment retry, context window
│   │   ├── researcher/                  # Questions, trust scoring, extraction
│   │   └── shared/                      # Store, events, config, dedup
│   ├── integration/                     # Cross-component flows with real DB
│   ├── contract/                        # LLM response format validation
│   └── conftest.py                      # Shared fixtures, mock LLM factory
├── logs/                                # With rotation and retention
└── scripts/
    └── cleanup.sh
```

### Key Differences

| Aspect | v1 | v2 |
|--------|----|----|
| Domain scope | Single (Fano/Sadhguru) | Any research domain via project config |
| Config files | 4 (duplicating, conflicting) | 1 infrastructure + 1 per project |
| Data locations | 12+ scattered | 1 database + 1 document dir |
| Prompts | Hardcoded Python strings | Templates parameterized by project |
| Seeds | Static YAML, no UI | Database-backed, full CRUD, modifiable |
| User interaction | View-only + manual YAML | Active: comments, annotations, seed management |
| Communication | File polling + HTTP | Event bus + database |
| State format | JSON files + partial SQLite | SQLite everywhere |
| Dead files | 983MB browser cache | None |
| Log retention | Unbounded | Configured rotation |
