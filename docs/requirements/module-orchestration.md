# Fano Orchestration Layer: Revised Design (Claude + Gemini + ChatGPT Review)

## 0. Context, Inputs, and Constraints

This document is the **implementation-facing** orchestration design. It integrates:

- The current system architecture and codebase boundaries.
- The issues cataloged in `TASK_ORCHESTRATION_REVIEW.md`.
- Gemini DT and ChatGPT 5.2 feedback focused on **correctness, recovery, and long‑run stability**.

### Current System Boundaries

- **Explorer** runs its own orchestrator loop and uses `LLMManager` to talk to the Pool (`explorer/src/orchestrator.py`, `explorer/src/orchestration/llm_manager.py`).
- **Documenter** has a separate orchestrator that delegates to `SessionManager`, `WorkPlanner`, and `OpportunityProcessor` (`documenter/main.py`).
- **Pool** is the single LLM access point, with both sync (`RequestQueue`) and async (`JobStore`) APIs (`pool/src/queue.py`, `pool/src/jobs.py`).
- **Deep/Pro quotas** are tracked in both Explorer and Pool today, which is a known split-brain risk.

### Goals (Updated)

1. **System-level coordination** across Explorer, Documenter, Researcher
2. **Priority correctness** across sync/async Pool work
3. **Crash-safe persistence** for state and logs
4. **Quota-aware LLM allocation** with a single source of truth
5. **Work-stealing** to keep LLMs utilized
6. **Deterministic recovery** after process or Pool restarts
7. **Async reliability** under CPU-heavy workloads (no event-loop blocking)

### Non-Goals

- Multi-instance LLM scaling (single instance per backend)
- Removing the Pool service
- Rewriting Explorer/Documenter internals beyond integration points

---

## 1. Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR                           │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│  │ Task Scheduler    │  │ State Manager    │  │ LLM Allocator  │ │
│  └──────────────────┘  └──────────────────┘  └────────────────┘ │
└───────────┬──────────────────────┬──────────────────────┬───────┘
            │                      │                      │
            ▼                      ▼                      ▼
      ┌─────────┐            ┌───────────┐           ┌──────────┐
      │Explorer │            │Documenter │           │Researcher│
      └─────────┘            └───────────┘           └──────────┘
            │                      │                      │
            └─────────────┬────────┴─────────────┬────────┘
                          ▼                      ▼
                ┌────────────────────────────────────┐
                │            POOL SERVICE            │
                │  - RequestQueue (sync, priority)   │
                │  - JobStore (async, priority)      │
                │  - StateManager (quota + recovery) │
                └────────────────────────────────────┘
```

**Design principle:** the orchestrator is a **coordinator**, not a monolith. Module implementations remain intact; orchestration defines scheduling, allocation, persistence, and recovery policy.

---

## 2. Task Model

### 2.1 Task Definition

```python
@dataclass
class Task:
    id: str
    key: str  # stable idempotency key
    module: Literal["explorer", "documenter", "researcher"]
    task_type: str
    priority: int
    state: TaskState  # pending | running | paused | completed | failed
    context: dict
    created_at: datetime
    updated_at: datetime
    attempts: int
    llm_requests: int
    llm_preference: Optional[str]
    conversation_state: Optional[ConversationState]
```

### 2.2 Stable Task Keys (Duplicate Prevention)

Modules **must not** emit new tasks every poll tick. Use a deterministic key:

```
key = f"{module}:{task_type}:{stable_context_hash}"
```

The scheduler stores `active_task_keys` for states in `{PENDING, RUNNING, PAUSED}`. `submit()` becomes **submit‑if‑absent**, preventing runaway task duplication.

### 2.3 Conversation State

```python
@dataclass
class ConversationState:
    backend: str
    thread_id: Optional[str]
    messages: list[dict]
    turn_count: int
    context: dict
```

### 2.4 Task Types and Module Mapping

| Module | Task Type | Existing Flow |
|--------|-----------|---------------|
| Explorer | exploration | `ExplorationEngine.explore()` |
| Explorer | synthesis | `SynthesisEngine` + chunking |
| Explorer | review | `InsightProcessor` + review panel |
| Documenter | address_comment | `CommentHandler.address_comment()` |
| Documenter | review_section | `CommentHandler.review_section()` |
| Documenter | incorporate_insight | `OpportunityProcessor.process_opportunity()` |
| Researcher | evaluate_source | (existing pipeline) |
| Researcher | extract_content | (existing pipeline) |

---

## 3. Concurrency Model and CPU-Bound Safety

The orchestration runtime is **single-process asyncio**, but CPU-heavy work must **never** block the event loop.

### 3.1 Offloading CPU-Bound Work

Deduplication checks, document scans, and large JSON serialization **must** be offloaded:

```python
async def run_cpu_bound(self, fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(self.proc_pool, fn, *args)
```

**Rationale:** prevents event-loop stalls that would break browser heartbeats, trigger false health restarts, or stall LLM requests.

---

## 4. Scheduling and Work-Stealing

### 4.1 Priority Scoring

Priority reflects:

1. Human feedback (comments)
2. Backlog pressure (undocumented blessed insights)
3. Staleness (starvation prevention)
4. Phase criticality (synthesis vs. exploration)
5. Quota feasibility (avoid scheduling unrunnable work)

```python
BASE_PRIORITIES = {
    "address_comment": 80,
    "synthesis": 65,
    "review": 60,
    "incorporate_insight": 58,
    "exploration": 50,
    "review_section": 45,
    "research": 40,
}
```

### 4.2 Lazy Priority Recompute

Dynamic priority updates are **not** recomputed on every `get_next_task()` call. Use a periodic refresh (e.g., every 60s) or recompute only the top‑K tasks to avoid scheduler bottlenecks.

### 4.3 Work-Stealing Model

Each backend has its own worker loop. A free backend pulls the highest‑priority runnable task:

```python
async def llm_worker_loop(backend: str):
    while running:
        task = scheduler.next_task(backend)
        if not task:
            await asyncio.sleep(IDLE_SLEEP)
            continue
        await executor.run(task, backend)
```

This avoids the “gather and wait” pattern and prevents idle LLMs.

---

## 5. LLM Allocation and Quota Tracking

### 5.1 Single Source of Truth (Pool)

Pool state is authoritative for **deep/pro quotas** and rate‑limit usage. Explorer should **not** persist deep mode usage locally.

### 5.2 Allocation Policy (Allocator is the Gate)

Allocation must occur through a single gate:

1. Task preference
2. Thread affinity
3. Pool‑reported quota availability
4. Utilization (best remaining quota)

Scheduler must **not** pre‑filter available backends. Only `allocator.allocate(task, backends)` decides runnable-ness.

### 5.3 Persist Quota Usage

Quota usage must be persisted so crashes do not reset usage counters. Persist in WAL or checkpoint state.

---

## 6. Pool Scheduling: Unified Priority

### 6.1 Problem

`RequestQueue` is priority‑aware, but `JobStore` is FIFO. Workers currently process async jobs before sync requests, causing priority inversion.

### 6.2 Required Fix

1. Convert `JobStore` to a priority queue.
2. Add `peek_next_job(backend)`.
3. Worker picks the **higher priority** item across sync + async queues.

---

## 7. State Management, WAL, and Recovery

### 7.1 Atomic Persistence

All JSON persistence must be atomic:

- write `*.tmp`
- `fsync`
- `os.replace()`

Applies to: pool state, job store, queue state, blessed insights, document output, orchestrator state.

### 7.2 WAL Strategy (Delta Logging)

Logging the **entire conversation** on every turn causes write amplification. Use **delta entries**:

```python
@dataclass
class ConversationDelta:
    task_id: str
    added_messages: list[dict]
    turn: int
```

### 7.3 WAL Recovery Semantics (Low-Scope Safe Option)

Use a **redo‑log** model:

1. Checkpoint stores `last_wal_sequence`.
2. On recovery: load checkpoint, replay WAL entries with `sequence > last_wal_sequence`.
3. Stop on first JSON decode error (likely partial last line).
4. Set `self.sequence` to max observed sequence to prevent collisions.

This avoids fragile commit markers and prevents replaying pre‑checkpoint entries.

### 7.4 Durable State Repair

If recovery changes task states (e.g., `RUNNING → PAUSED`), it must **persist** those fixes immediately via checkpoint or `update_task()`.

### 7.5 Recovery Order

1. Restore module state
2. Resume/requeue tasks

This prevents tasks running against uninitialized modules.

---

## 8. Task Execution Semantics

### 8.1 Context Truncation Safety

Do **not** drop system/seed prompts. If truncating, keep head+tail:

```
messages = messages[:2] + messages[-48:]
```

For long threads, add a summary message pinned to context.

### 8.2 Failure Hooks

`TaskExecutor` must call `module.handle_failure(task, error)` on exception before marking failed.

---

## 9. Process Hygiene and Browser Lifecycle

### 9.1 PID Lifecycle Manager

On startup, read a `gateway.pid` file, terminate stale browser PIDs, then write current PIDs. Prevents “zombie” browser processes after crashes.

### 9.2 Per-Backend Locks

`LLMGateway` must use **one lock per backend** (not a single global lock) so multiple LLMs can work concurrently.

---

## 10. Module Integration Contracts (Polling Model)

**Polling** is the canonical interface. A message bus is out-of-scope and should not be implemented.

```python
class ModuleInterface(ABC):
    def get_pending_tasks(self) -> list[Task]:
        ...

    async def execute_task(self, task: Task, llm_backend: str) -> TaskResult:
        ...

    async def handle_failure(self, task: Task, error: str) -> None:
        ...

    def handle_task_result(self, task: Task, result: TaskResult):
        ...

    def get_state(self) -> dict:
        ...

    def restore_state(self, state: dict):
        ...
```

---

## 11. Insight and Document Flow Corrections

Required fixes:

1. Pending insights live under `chunks/insights/pending/`.
2. Documenter must load blessed insights into its dedup checker.
3. Insight status saves must be awaited.
4. `blessed_insights.json` writes must be atomic.

---

## 12. Configuration Updates

```yaml
orchestrator:
  state_path: "orchestrator/state.json"
  checkpoint_interval_seconds: 60

  scheduling:
    idle_sleep_seconds: 3
    priority_refresh_seconds: 60

  quotas:
    source_of_truth: "pool"

  recovery:
    pool_active_work_recover: true
    pool_active_work_timeout_seconds: 900
```

---

## 13. Migration Plan

1. Add orchestrator skeleton (state + scheduler)
2. Add module adapters (task interface)
3. Implement pool priority merge
4. Move deep/pro tracking to pool
5. Implement WAL redo + delta logging
6. Make persistence atomic everywhere
7. Add per‑backend worker loops
8. Add PID lifecycle management

---

## 14. Success Criteria

1. Priority scheduling respects task urgency across sync + async requests.
2. No loss or duplication of insights after restart.
3. LLMs remain utilized without idle waiting.
4. Deep/pro quotas never diverge after restart.
5. Recovery is deterministic and does not regress state.
6. Orchestrator runs for days without event-loop stalls.

