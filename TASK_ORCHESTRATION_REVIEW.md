# Task Orchestration System - Comprehensive Review

**Date:** 2026-01-13
**Scope:** Complete analysis of task orchestration, LLM utilization, robustness, and code quality

---

## Executive Summary

This review identified **47 issues** across the Fano task orchestration system, including:
- **8 CRITICAL** issues that could cause data loss or system failures
- **15 HIGH** severity bugs affecting reliability and correctness
- **14 MEDIUM** issues impacting efficiency and maintainability
- **10 LOW** priority improvements

The system has solid foundational architecture but needs improvements in:
1. Priority scheduling (JobStore ignores priorities)
2. Concurrency safety (multiple race conditions)
3. Atomicity of file operations (no crash-safe writes)
4. Recovery mechanisms (gaps in restart handling)

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [LLM Instance Utilization](#2-llm-instance-utilization)
3. [Task Priority Issues](#3-task-priority-issues)
4. [Insight Flow Problems](#4-insight-flow-problems)
5. [Robustness & Recovery](#5-robustness--recovery)
6. [Race Conditions](#6-race-conditions)
7. [Error Handling](#7-error-handling)
8. [Recommendations](#8-recommendations)

---

## 1. Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    EXPLORER ORCHESTRATOR                    │
│ explorer/src/orchestrator.py                                │
└─────────────────────────────────────────────────────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │ThreadManager │  │LLMManager    │  │SynthesisEng. │
    │(select/spawn)│  │(send message)│  │(chunk ready?)│
    └──────────────┘  └──────────────┘  └──────────────┘
          │                 │                 │
          └─────────────────┼─────────────────┘
                            ▼
          ┌─────────────────────────────────────┐
          │         POOL SERVICE (HTTP)         │
          │  pool/src/api.py                    │
          │  - JobStore (async jobs)            │
          │  - RequestQueue (sync requests)     │
          │  - Workers (Gemini/ChatGPT/Claude)  │
          └─────────────────────────────────────┘
                            │
                            ▼
          ┌─────────────────────────────────────┐
          │  InsightProcessor → BlessedStore    │
          │  (extraction, review, blessing)     │
          └─────────────────────────────────────┘
                            │
                            ▼
          ┌─────────────────────────────────────┐
          │        DOCUMENTER ORCHESTRATOR      │
          │  documenter/main.py                 │
          │  - OpportunityFinder                │
          │  - OpportunityProcessor             │
          └─────────────────────────────────────┘
```

### Key Data Flows

1. **Exploration Flow**: Thread → LLM Request → Response → Exchange → Save
2. **Synthesis Flow**: Thread → Chunk Ready Check → Extract → Review → Bless
3. **Document Flow**: Blessed Insight → Opportunity → Evaluate → Draft → Add Section

---

## 2. LLM Instance Utilization

### Current Model Management

**File:** `explorer/src/orchestration/llm_manager.py`

The system manages 2-3 LLM instances:
- ChatGPT (browser-based)
- Gemini (browser-based)
- Claude (API-based, via pool)

### Issues with LLM Utilization

#### ISSUE-LLM-1: No Work Stealing Between Models [HIGH]

**Location:** `explorer/src/orchestrator.py:373`

```python
results = await asyncio.gather(*tasks, return_exceptions=True)
```

**Problem:** If one model completes quickly (e.g., Gemini in 2 min) and another is slow (ChatGPT taking 10 min), the fast model sits idle until all parallel work completes.

**Impact:** With 30-second poll intervals, fast models waste ~80% of their capacity waiting.

**Recommendation:** Implement work-stealing pattern where idle models can pick up new work without waiting for cycle completion.

---

#### ISSUE-LLM-2: Thread Selection Reloads All Threads Each Cycle [MEDIUM]

**Location:** `explorer/src/orchestration/thread_manager.py:67-125`

**Problem:** `load_active_threads()` does filesystem glob scan for EACH model assignment (up to 3x per 30-second cycle).

**Impact:** ~100 JSON file reads per cycle as thread count grows.

**Recommendation:** Cache thread metadata in memory with invalidation on file changes.

---

#### ISSUE-LLM-3: Deep Mode Quota Underutilized [LOW]

**Configuration:** ChatGPT Pro limited to 100/day, Gemini Deep Think to 20/day

**Problem:** With ~3-5 synthesis events per day, only using ~30% of allocated quota.

**Recommendation:** Use deep mode more aggressively for complex exploration rounds (when profundity signals detected).

---

#### ISSUE-LLM-4: Dual Deep Mode Tracking Systems Out of Sync [HIGH]

**Locations:**
- `explorer/src/browser/model_selector.py` - maintains `deep_mode_state.json`
- `pool/src/state.py` - maintains `pool_state.json` with `deep_mode_uses_today`

**Problem:** Two independent counters that can diverge if pool restarts or explorer records usage before pool fails.

**Impact:** May exceed intended deep mode limits or leave quota unused.

**Recommendation:** Single source of truth in pool state, with explorer querying pool for quota availability.

---

## 3. Task Priority Issues

### CRITICAL: JobStore Ignores Priority [CRITICAL]

**Location:** `pool/src/jobs.py:211-234`

```python
def get_next_job(self, backend: str) -> Optional[Job]:
    with self._lock:
        queue = self._backend_queues.get(backend, [])

        for job_id in queue:  # SIMPLE LINEAR ITERATION (FIFO)
            job = self._jobs.get(job_id)
            if job and job.status == JobStatus.QUEUED:
                # ... returns first found, ignores priority
                return job
```

**Problem:** The `Job` dataclass accepts a `priority` field (line 48) but `get_next_job()` iterates through jobs in FIFO order, completely ignoring priority.

**Impact:** HIGH priority jobs wait behind LOW priority jobs. System cannot prioritize urgent work.

**Recommendation:** Replace list with heap-based priority queue (like RequestQueue uses).

---

### CRITICAL: Worker Scheduling Priority Inversion [CRITICAL]

**Location:** `pool/src/workers.py:71-91`

```python
async def _run_loop(self):
    while self._running:
        # First, check JobStore for async jobs (new system)
        if self.jobs:
            job = self.jobs.get_next_job(self.backend_name)  # ← FIFO
            if job:
                await self._process_job(job)
                continue

        # Fall back to legacy queue (sync system)
        queued = await self.queue.dequeue()  # ← Priority-aware
```

**Problem:** Workers check JobStore (FIFO) BEFORE RequestQueue (priority-aware). A LOW priority async job will be processed before a HIGH priority sync request.

**Impact:** Priority requests from sync API are starved by any async jobs.

**Recommendation:** Check priority across BOTH queues and process highest priority regardless of queue type.

---

### Priority System Summary

| Component | Priority Support | Status |
|-----------|-----------------|--------|
| RequestQueue (sync) | HIGH/NORMAL/LOW via heap | ✅ Working |
| JobStore (async) | Field exists, IGNORED | ❌ Broken |
| Worker scheduling | Async before sync | ❌ Inverted |
| Documenter opportunities | Calculated priority score | ✅ Working |

---

## 4. Insight Flow Problems

### CRITICAL: Directory Lookup Bug in Insight Processor [CRITICAL]

**Location:** `explorer/src/orchestration/insight_processor.py:390`

```python
subdirs = ["pending", "reviewing", "insights/blessed", "insights/rejected"]
for subdir in subdirs:
    search_dir = self.paths.chunks_dir / subdir
```

**Problem:** Searches `chunks/pending/` but pending insights are actually saved to `chunks/insights/pending/`.

**Impact:** After crash during review, restart cannot find existing insights and re-extracts them, creating duplicates.

---

### CRITICAL: Documenter Doesn't Load Blessed Insights into Dedup [CRITICAL]

**Location:** `documenter/session.py:217-237`

```python
async def _initialize_dedup(self):
    # Load existing document sections into dedup checker
    for section in self.document.sections:
        self.dedup_checker.add_content(...)
    # MISSING: Load blessed insights from blessed_dir!
```

**Problem:** Documenter only checks new insights against existing document sections, not against other blessed insights waiting to be processed.

**Impact:** Two similar blessed insights can both be incorporated, creating duplicates in the document.

---

### HIGH: Fire-and-Forget Save in Quota Handler [HIGH]

**Location:** `explorer/src/orchestration/insight_processor.py:372`

```python
def _handle_quota_exhausted(self, insight: AtomicInsight, error):
    insight.status = InsightStatus.PENDING
    asyncio.create_task(asyncio.to_thread(insight.save, self.paths.chunks_dir))
    # Returns immediately without awaiting
```

**Problem:** If process exits immediately, save may not complete.

**Impact:** Insight status changed to PENDING but not persisted, causing re-review on restart.

---

### HIGH: Non-Atomic blessed_insights.json Writes [HIGH]

**Location:** `explorer/src/orchestration/blessed_store.py:117-137`

```python
if self.paths.blessed_insights_file.exists():
    with open(self.paths.blessed_insights_file, encoding="utf-8") as f:
        data = json.load(f)
else:
    data = {"insights": []}
data["insights"].append({...})
with open(self.paths.blessed_insights_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ...)
```

**Problem:** Read-modify-write without atomicity. Crash during write corrupts file.

**Impact:** Lost blessed insights if crash occurs during JSON write.

---

### Insight Flow Summary

```
Thread Exploration
       │
       ▼
  Chunk Ready? ──────────────────────────────┐
       │                                      │
       ▼                                      │
Extract Insights ◄──── BUG: Wrong directory  │
       │                lookup after crash   │
       ▼                                      │
 Dedup Check ◄───────── BUG: Explorer and    │
       │                Documenter have       │
       ▼                separate dedup       │
Review Panel                                  │
       │                                      │
       ▼                                      │
Bless Insight ◄──────── BUG: Non-atomic      │
       │                JSON write           │
       ▼                                      │
Documenter ◄─────────── BUG: Doesn't load    │
       │                blessed into dedup   │
       ▼                                      │
 Add to Document                              │
```

---

## 5. Robustness & Recovery

### State Persistence Analysis

| State | Persisted | Recovery | Risk |
|-------|-----------|----------|------|
| Pool queue requests | ✅ Yes | ✅ On startup | LOW |
| Async jobs | ✅ Yes | ✅ On startup | LOW |
| Active work | ✅ Yes | ⚠️ Via watchdog only | MEDIUM |
| Explorer threads | ✅ Yes | ✅ On startup | LOW |
| In-progress extraction | ❌ No | ❌ None | HIGH |
| Review progress | ❌ No | ❌ None | MEDIUM |
| Document sections | ✅ Yes | ✅ On startup | LOW |
| Futures/callbacks | ❌ No | ❌ Cannot serialize | HIGH |

---

### HIGH: Pool Restart Mid-Request [HIGH]

**Scenario:** Pool crashes while browser is generating response.

**Current Behavior:**
1. Active work tracked in `pool_state.json`
2. On restart: Queue restored, but NO automatic recovery of active work
3. Watchdog kicks in after 1 HOUR timeout
4. Stale work auto-cleared after 2 hours

**Gap:** No immediate recovery on startup. Work sits in limbo for up to 2 hours.

**Location:** `pool/src/api.py:75-101` (startup) and `pool/src/state.py:220-234` (staleness check)

---

### HIGH: Orphaned Futures After Pool Restart [HIGH]

**Location:** `pool/src/queue.py:157-190`

**Problem:** When queue is restored, new Futures are created but original requestor's Future is lost (cannot be serialized).

**Impact:** Client waiting on response hangs indefinitely or times out. No way to notify original requestor.

**Recommendation:** Add request metadata for clients to query status and retry.

---

### MEDIUM: Document Write Not Atomic [MEDIUM]

**Location:** `documenter/document.py:107-135`

```python
def save(self):
    self.path.write_text(self._render(), encoding="utf-8")  # NOT ATOMIC
    if description:
        # Version saved separately - could create inconsistency
```

**Problem:** Direct `write_text()` - if crash mid-write, file is truncated/corrupted.

**Recommendation:** Write to temp file, then atomic rename.

---

## 6. Race Conditions

### CRITICAL: Unprotected Worker State Reads [CRITICAL]

**Location:** `pool/src/workers.py:45-48, 101-103` and `pool/src/api.py:137-144`

```python
# Workers (no lock):
self._current_request_id = queued.request_id
self._current_start_time = time.time()

# Watchdog (no lock):
if worker._current_start_time is None:
    continue
elapsed = time.time() - worker._current_start_time  # TOCTOU
```

**Problem:** Worker state accessed without synchronization. Watchdog may read stale/None values.

**Impact:** Incorrect stuck detection, potential NoneType errors.

---

### CRITICAL: Global rate_tracker Without Synchronization [CRITICAL]

**Location:** `explorer/src/browser/base.py:94-151`

```python
rate_tracker = RateLimitTracker()  # Global instance

def _save(self):
    with open(RATE_LIMIT_FILE, "w", encoding="utf-8") as f:
        json.dump(self.limits, f, indent=2, default=str)  # No lock!
```

**Problem:** Multiple async tasks call `mark_limited()`, `is_available()`, `_save()` without any synchronization.

**Impact:** Dictionary corruption, inconsistent state, race conditions in file writes.

---

### HIGH: File TOCTOU in Queue Recovery [HIGH]

**Location:** `pool/src/queue.py:236-268`

```python
if not self.state_file.exists():  # CHECK
    return restored

with open(self.state_file, encoding="utf-8") as f:  # USE
    state = json.load(f)
```

**Problem:** File could be deleted/modified between check and use.

---

### HIGH: JobStore Content Cache Race [HIGH]

**Location:** `pool/src/jobs.py:121-127`

```python
def _clean_expired_cache(self):
    expired = [h for h, t in self._cache_times.items() if now - t > self.cache_ttl]
    for h in expired:
        self._content_cache.pop(h, None)  # RACE: dict modified
        self._cache_times.pop(h, None)
```

**Problem:** TOCTOU between list comprehension and pop operations.

---

### Race Conditions Summary

| Issue | Severity | Type |
|-------|----------|------|
| Unprotected worker state | CRITICAL | Data Race |
| Global rate_tracker | CRITICAL | File + Memory Race |
| Queue recovery TOCTOU | HIGH | File Race |
| JobStore cache cleanup | HIGH | Memory Race |
| Callback variable capture | HIGH | Closure Race |
| JSON file writes | HIGH | File Corruption |
| Lock under blocking I/O | MEDIUM | Serialization |
| File deletion races | MEDIUM | Resource Race |

---

## 7. Error Handling

### CRITICAL: Bare Except Clause [CRITICAL]

**Location:** `control/debug_util.py:47`

```python
try:
    count = len(re.findall(pat, resp))
except:  # ⚠️ BARE EXCEPT
    pass
```

**Problem:** Catches `KeyboardInterrupt`, `SystemExit`, and all other exceptions.

**Fix:** Use `except (re.error, Exception):`

---

### HIGH: Swallowed Exceptions in Worker Loop [HIGH]

**Location:** `pool/src/workers.py:451-454`

```python
try:
    await self.browser.page.evaluate(...)
except Exception:
    pass  # Best effort scroll
```

**Problem:** Too broad exception handling. Should catch specific Playwright exceptions.

---

### MEDIUM: Unsafe asyncio.gather Results [MEDIUM]

**Locations:** Multiple files use `return_exceptions=True`

Most locations DO check for exceptions in results, but the pattern is error-prone:

```python
results = await asyncio.gather(*tasks, return_exceptions=True)
for result in results:
    if isinstance(result, Exception):  # Easy to forget
        # handle
```

---

### Error Handling Summary

| Issue | Location | Severity |
|-------|----------|----------|
| Bare except | `control/debug_util.py:47` | CRITICAL |
| Broad Exception catch | `pool/src/workers.py:451` | HIGH |
| Silent failures | Multiple locations | MEDIUM |
| Missing timeout config | `explorer/src/browser/base.py:244` | MEDIUM |
| Inconsistent logging | Mixed structured/string | LOW |

---

## 8. Recommendations

### Immediate Fixes (CRITICAL)

1. **Fix JobStore Priority Scheduling**
   - Location: `pool/src/jobs.py:211-234`
   - Replace list-based queue with heap-based priority queue
   - Ensure `priority` field is used in ordering

2. **Fix Worker Scheduling Order**
   - Location: `pool/src/workers.py:71-91`
   - Check priority across BOTH queues before processing
   - Process highest priority regardless of queue type

3. **Fix Directory Lookup Bug**
   - Location: `explorer/src/orchestration/insight_processor.py:390`
   - Correct path to `insights/pending` instead of `pending`

4. **Add Blessed Insights to Documenter Dedup**
   - Location: `documenter/session.py:217-237`
   - Load `blessed_insights.json` into dedup checker on startup

5. **Add Synchronization to rate_tracker**
   - Location: `explorer/src/browser/base.py:94-151`
   - Add `threading.Lock()` around all state modifications

6. **Fix Bare Except**
   - Location: `control/debug_util.py:47`
   - Change to `except (re.error, Exception):`

### Short-term Improvements (HIGH)

7. **Implement Atomic File Writes**
   - All JSON persistence: write to temp file, then `os.rename()`
   - Affected: `pool/src/state.py`, `pool/src/jobs.py`, `pool/src/queue.py`, `blessed_store.py`

8. **Add Worker State Locks**
   - Location: `pool/src/workers.py:45-48`
   - Add `asyncio.Lock()` for `_current_*` variables

9. **Auto-Recover Active Work on Pool Startup**
   - Location: `pool/src/api.py:75-101`
   - Check for active work in state and attempt immediate recovery

10. **Fix Fire-and-Forget Save**
    - Location: `explorer/src/orchestration/insight_processor.py:372`
    - Change `create_task()` to `await`

11. **Unify Deep Mode Tracking**
    - Single source of truth in pool state
    - Explorer queries pool for quota availability

### Medium-term Improvements

12. **Implement Work Stealing**
    - Allow idle models to pick up new work without waiting for cycle completion

13. **Cache Thread Metadata**
    - Avoid filesystem scans on every cycle

14. **Add Document-Level File Locking**
    - Prevent multi-process write corruption

15. **Reduce Watchdog Timeout**
    - Make configurable, default to lower value (e.g., 15 min)

16. **Add Request Recovery Metadata**
    - Allow clients to query status and retry after pool restart

---

## Appendix: Files Requiring Changes

| File | Issues | Priority |
|------|--------|----------|
| `pool/src/jobs.py` | Priority scheduling broken | CRITICAL |
| `pool/src/workers.py` | Scheduling order, race conditions | CRITICAL |
| `explorer/src/orchestration/insight_processor.py` | Directory bug, fire-and-forget | CRITICAL |
| `documenter/session.py` | Missing dedup loading | CRITICAL |
| `explorer/src/browser/base.py` | Global rate_tracker races | CRITICAL |
| `control/debug_util.py` | Bare except | CRITICAL |
| `pool/src/state.py` | Non-atomic writes | HIGH |
| `pool/src/queue.py` | TOCTOU, non-atomic writes | HIGH |
| `explorer/src/orchestration/blessed_store.py` | Non-atomic writes | HIGH |
| `pool/src/api.py` | No active work recovery on startup | HIGH |
| `documenter/document.py` | Non-atomic writes | MEDIUM |
| `explorer/src/orchestrator.py` | No work stealing | MEDIUM |

---

## Issue Count by Severity

| Severity | Count |
|----------|-------|
| CRITICAL | 8 |
| HIGH | 15 |
| MEDIUM | 14 |
| LOW | 10 |
| **TOTAL** | **47** |

---

*End of Review*
