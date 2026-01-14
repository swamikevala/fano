# Fano Module Orchestration: Design Document

## Overview

The Orchestrator is a top-level coordinator that manages the three Fano modules (Explorer, Documenter, Researcher) and allocates shared LLM resources intelligently. It enables autonomous operation for extended periods (days to weeks) without human intervention.

### Goals

1. **Autonomous operation**: Run continuously, recovering from failures, until explicitly stopped
2. **Intelligent allocation**: Dynamically decide which module gets LLM time based on system state
3. **Resource efficiency**: Maximize utilization of fixed-subscription LLMs without hitting rate limits
4. **State persistence**: Resume from saved state after restarts/crashes
5. **Human feedback integration**: Incorporate comments on docs/insights without blocking

### Non-Goals

- API-based LLM management (we use browser-based subscriptions only)
- Cost optimization (fixed monthly subscription)
- Multi-instance scaling (single instance of each LLM)

---

## Part 1: Architecture

### 1.1 System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Scheduler │  │   State     │  │   LLM Allocator         │  │
│  │   (decides  │  │   Manager   │  │   (routes requests,     │  │
│  │    what)    │  │   (tracks)  │  │    manages quotas)      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└────────┬────────────────┬─────────────────────┬─────────────────┘
         │                │                     │
         ▼                ▼                     ▼
    ┌─────────┐     ┌───────────┐        ┌───────────┐
    │Explorer │     │Documenter │        │Researcher │
    │ Module  │     │  Module   │        │  Module   │
    └─────────┘     └───────────┘        └───────────┘
         │                │                     │
         └────────────────┼─────────────────────┘
                          ▼
              ┌───────────────────────┐
              │      LLM Pool         │
              │  ┌───────┐ ┌───────┐  │
              │  │ChatGPT│ │Gemini │  │
              │  └───────┘ └───────┘  │
              │       ┌───────┐       │
              │       │Claude │       │
              │       └───────┘       │
              └───────────────────────┘
```

### 1.2 Core Components

| Component | Responsibility |
|-----------|----------------|
| **Scheduler** | Decides which module/task gets next LLM slot |
| **State Manager** | Tracks progress, metrics, persists state |
| **LLM Allocator** | Routes requests to LLMs, tracks quotas/limits |
| **Task Queue** | Holds preemptible work units from all modules |

### 1.3 Execution Model

Modules no longer run as independent processes. Instead:

1. Modules register **tasks** with the orchestrator
2. Orchestrator picks highest-priority task
3. Orchestrator allocates an LLM and executes the task
4. Task runs until it needs LLM response, then yields
5. Orchestrator can preempt between LLM requests
6. Task resumes when orchestrator schedules it again

This allows fine-grained control: a long Explorer conversation can be paused mid-way if Documenter has urgent work.

---

## Part 2: Task Model

### 2.1 What is a Task?

A **Task** is the smallest preemptible unit of work. It represents one logical operation that may require multiple LLM interactions.

```python
@dataclass
class Task:
    id: str                          # Unique identifier
    module: str                      # "explorer" | "documenter" | "researcher"
    task_type: str                   # e.g., "exploration", "synthesis", "document_section"
    priority: int                    # Higher = more urgent
    state: TaskState                 # pending | running | paused | completed | failed
    context: dict                    # Task-specific data (thread_id, insight_id, etc.)
    created_at: datetime
    updated_at: datetime
    llm_requests: int                # Count of LLM calls made
    llm_preference: Optional[str]    # Required LLM (e.g., "gemini" for deep think)
    conversation_state: Optional[dict]  # For resumable multi-turn conversations
```

### 2.2 Task Types by Module

**Explorer:**
- `exploration`: Multi-turn conversation exploring a seed/topic
- `synthesis`: Synthesizing insights from exploration (uses deep mode)
- `review`: Reviewing/blessing insights
- `critique`: Critical analysis of findings

**Documenter:**
- `incorporate_insight`: Adding a blessed insight to document
- `review_section`: Reviewing existing section
- `address_comment`: Responding to author comment
- `draft_section`: Writing new content

**Researcher:**
- `evaluate_source`: Assessing trustworthiness of a source
- `extract_content`: Pulling relevant info from source
- `generate_questions`: Creating research questions from activity
- `synthesize_findings`: Combining research into actionable insights

### 2.3 Task Lifecycle

```
                    ┌─────────┐
                    │ PENDING │
                    └────┬────┘
                         │ scheduled
                         ▼
    ┌────────────────────────────────────────┐
    │               RUNNING                   │
    │  ┌──────────────────────────────────┐  │
    │  │  execute → LLM request → yield   │  │
    │  │       ↑                    │      │  │
    │  │       └────── resume ──────┘      │  │
    │  └──────────────────────────────────┘  │
    └───────┬────────────────────┬───────────┘
            │ preempt            │ complete
            ▼                    ▼
       ┌────────┐          ┌───────────┐
       │ PAUSED │          │ COMPLETED │
       └────────┘          └───────────┘
            │ reschedule
            └──────→ RUNNING
```

### 2.4 Preemption

Tasks yield control after each LLM request:

```python
class TaskExecutor:
    async def run_task(self, task: Task) -> TaskResult:
        """Execute task, yielding after each LLM call."""
        while not task.is_complete():
            # Get next action from task
            action = task.get_next_action()

            if action.type == "llm_request":
                # Check if we should preempt
                if self.scheduler.should_preempt(task):
                    task.state = TaskState.PAUSED
                    task.save_conversation_state()
                    return TaskResult(preempted=True)

                # Execute LLM request
                llm = self.allocator.get_llm(task)
                response = await llm.send(action.prompt)
                task.process_response(response)
                task.llm_requests += 1

            elif action.type == "file_operation":
                # Non-LLM actions don't yield
                await action.execute()

        return TaskResult(completed=True, output=task.output)
```

### 2.5 Conversation State Persistence

For multi-turn conversations (especially Explorer), we save:

```python
@dataclass
class ConversationState:
    llm: str                    # Which LLM this conversation is with
    thread_id: Optional[str]    # Browser thread ID for continuity
    messages: list[dict]        # Message history
    turn_count: int
    context: dict               # Task-specific context
```

When resuming a paused task:
1. Load conversation state
2. Request same LLM if available (for thread continuity)
3. If LLM unavailable, either wait or start fresh (configurable per task type)

---

## Part 3: Scheduling Algorithm

### 3.1 Priority Factors

The scheduler computes a priority score for each pending/paused task:

```python
def compute_priority(task: Task, state: SystemState) -> float:
    score = task.base_priority  # From task type defaults

    # Factor 1: Backlog pressure
    if task.module == "documenter":
        backlog = state.blessed_insights_pending
        if backlog > 20:
            score += 30  # Heavy boost
        elif backlog > 10:
            score += 15

    # Factor 2: Starvation prevention
    time_waiting = now() - task.updated_at
    score += min(time_waiting.hours * 2, 20)  # Cap at +20

    # Factor 3: Seed priority (for explorer)
    if task.task_type == "exploration":
        score += task.context.get("seed_priority", 0) * 10

    # Factor 4: Comment responsiveness
    if task.task_type == "address_comment":
        score += 25  # Comments get priority

    # Factor 5: Deep mode availability
    if task.requires_deep_mode:
        if not state.quota_available(task.llm_preference):
            score -= 100  # Defer if quota exhausted

    return score
```

### 3.2 Base Priorities by Task Type

```python
BASE_PRIORITIES = {
    # Explorer
    "exploration": 50,
    "synthesis": 60,      # Higher: produces blessed insights
    "review": 55,
    "critique": 45,

    # Documenter
    "address_comment": 70,  # Highest: human feedback
    "incorporate_insight": 55,
    "review_section": 40,
    "draft_section": 50,

    # Researcher
    "evaluate_source": 45,
    "extract_content": 40,
    "generate_questions": 35,
    "synthesize_findings": 50,
}
```

### 3.3 Scheduling Loop

```python
async def scheduling_loop(self):
    while self.running:
        # Get available LLMs
        available_llms = self.allocator.get_available_llms()

        if not available_llms:
            # All LLMs busy or rate-limited
            await asyncio.sleep(10)
            continue

        # Get highest priority task that can run
        task = self.get_best_task(available_llms)

        if not task:
            # No tasks ready
            await self.maybe_generate_tasks()
            await asyncio.sleep(5)
            continue

        # Allocate LLM and run task
        llm = self.allocator.allocate(task, available_llms)
        result = await self.executor.run_task(task, llm)

        # Handle result
        if result.completed:
            self.handle_completion(task)
        elif result.preempted:
            self.requeue(task)
        elif result.failed:
            self.handle_failure(task, result.error)

        # Persist state
        self.state_manager.save()
```

### 3.4 LLM Consultation for Priority (Hourly)

Once per hour (or on significant state change), ask an LLM for priority guidance:

```python
async def consult_llm_for_priorities(self):
    """Use ~5% of LLM budget for meta-decisions."""

    context = f"""
Current system state:
- Blessed insights pending documentation: {self.state.blessed_pending}
- Document sections needing review: {self.state.sections_stale}
- Active explorations in progress: {self.state.explorations_active}
- Research questions open: {self.state.research_questions}
- Recent comments: {self.state.recent_comments}

Recent activity:
- Insights generated (24h): {self.state.insights_24h}
- Sections documented (24h): {self.state.sections_24h}
- Research items completed (24h): {self.state.research_24h}

Rate limit status:
- ChatGPT: {self.quotas.chatgpt_remaining} requests remaining
- Gemini: {self.quotas.gemini_remaining} requests remaining
- Gemini Deep Think: {self.quotas.gemini_deep_remaining} remaining today
- Claude: {self.quotas.claude_remaining} requests remaining
"""

    task = """
Analyze the current state and recommend priority allocation.
Consider:
1. Is there a backlog building up anywhere?
2. Are we exploring but not documenting (or vice versa)?
3. Should we shift focus to clear a bottleneck?

Respond with:
EXPLORER_WEIGHT: [0-100]
DOCUMENTER_WEIGHT: [0-100]
RESEARCHER_WEIGHT: [0-100]
REASONING: [brief explanation]
SPECIFIC_ACTIONS: [any specific tasks to prioritize]
"""

    response = await self.llm.send(context + task)
    self.apply_priority_weights(parse_response(response))
```

---

## Part 4: LLM Allocation

### 4.1 LLM Pool

Single browser instance per LLM, managed by the existing pool service:

```python
@dataclass
class LLMInstance:
    name: str                  # "chatgpt" | "gemini" | "claude"
    status: str                # "available" | "busy" | "rate_limited"
    current_task: Optional[str]
    thread_id: Optional[str]   # For conversation continuity

    # Quota tracking
    requests_this_hour: int
    requests_today: int
    deep_mode_used_today: int
    rate_limited_until: Optional[datetime]
```

### 4.2 Quota Tracking

Track usage proactively to avoid hitting limits:

```python
@dataclass
class LLMQuotas:
    """Learned limits per LLM. Updated when we hit actual limits."""

    # Requests per hour (learned from rate limit responses)
    hourly_limits: dict[str, int] = field(default_factory=lambda: {
        "chatgpt": 50,   # Conservative defaults
        "gemini": 50,
        "claude": 50,
    })

    # Deep mode daily limits
    deep_mode_limits: dict[str, int] = field(default_factory=lambda: {
        "gemini_deep_think": 5,   # Very limited
        "chatgpt_pro": 25,
        "claude_extended": 100,   # More generous
    })

    # Current usage
    hourly_usage: dict[str, int]
    daily_deep_usage: dict[str, int]

    def can_use(self, llm: str, deep_mode: bool = False) -> bool:
        """Check if we have quota for this request."""
        if self.hourly_usage.get(llm, 0) >= self.hourly_limits[llm] * 0.9:
            return False  # Leave 10% buffer

        if deep_mode:
            mode_key = f"{llm}_deep"
            if self.daily_deep_usage.get(mode_key, 0) >= self.deep_mode_limits.get(mode_key, 0):
                return False

        return True

    def record_usage(self, llm: str, deep_mode: bool = False):
        """Record a request."""
        self.hourly_usage[llm] = self.hourly_usage.get(llm, 0) + 1
        if deep_mode:
            mode_key = f"{llm}_deep"
            self.daily_deep_usage[mode_key] = self.daily_deep_usage.get(mode_key, 0) + 1

    def record_rate_limit(self, llm: str, retry_after: int):
        """Learn from rate limit. Adjust our estimates."""
        # Reduce estimated limit by 20%
        self.hourly_limits[llm] = int(self.hourly_usage[llm] * 0.8)
        log.warning("quota.limit_hit", llm=llm,
                    new_estimate=self.hourly_limits[llm],
                    retry_after=retry_after)
```

### 4.3 LLM Selection

When a task needs an LLM:

```python
def allocate_llm(self, task: Task, available: list[LLMInstance]) -> Optional[LLMInstance]:
    """Select best LLM for this task."""

    # If task requires specific LLM (e.g., for deep mode)
    if task.llm_preference:
        preferred = self.get_llm(task.llm_preference)
        if preferred and preferred.status == "available":
            if self.quotas.can_use(preferred.name, task.requires_deep_mode):
                return preferred
        # Preferred not available - task must wait
        return None

    # If task has conversation state, prefer same LLM
    if task.conversation_state:
        same_llm = self.get_llm(task.conversation_state.llm)
        if same_llm and same_llm.status == "available":
            return same_llm

    # Otherwise, pick any available LLM with quota
    candidates = [
        llm for llm in available
        if self.quotas.can_use(llm.name, task.requires_deep_mode)
    ]

    if not candidates:
        return None

    # Prefer LLM with most remaining quota
    return max(candidates, key=lambda l: self.quotas.remaining(l.name))
```

### 4.4 Rate Limit Recovery

When an LLM hits a rate limit:

```python
async def handle_rate_limit(self, llm: LLMInstance, retry_after: int):
    """Handle rate limit gracefully."""

    log.warning("llm.rate_limited",
                llm=llm.name,
                retry_after=retry_after,
                current_task=llm.current_task)

    # Mark LLM as rate-limited
    llm.status = "rate_limited"
    llm.rate_limited_until = datetime.now() + timedelta(seconds=retry_after)

    # Update quota estimates
    self.quotas.record_rate_limit(llm.name, retry_after)

    # Pause the current task (will be rescheduled)
    if llm.current_task:
        task = self.get_task(llm.current_task)
        task.state = TaskState.PAUSED
        task.save_conversation_state()

    # Schedule recovery check
    asyncio.create_task(self.check_llm_recovery(llm, retry_after))

async def check_llm_recovery(self, llm: LLMInstance, wait_seconds: int):
    """Check if LLM is available again after rate limit."""
    await asyncio.sleep(wait_seconds + 60)  # Add buffer

    # Test with a simple request
    try:
        await self.pool.health_check(llm.name)
        llm.status = "available"
        llm.rate_limited_until = None
        log.info("llm.recovered", llm=llm.name)
    except Exception as e:
        # Still limited, try again later
        log.warning("llm.still_limited", llm=llm.name, error=str(e))
        asyncio.create_task(self.check_llm_recovery(llm, 300))
```

---

## Part 5: State Management

### 5.1 Persistent State

All state is persisted to enable crash recovery:

```python
@dataclass
class OrchestratorState:
    # Task state
    tasks: dict[str, Task]
    task_queue: list[str]  # Task IDs in priority order

    # LLM state
    llm_instances: dict[str, LLMInstance]
    quotas: LLMQuotas

    # Module state
    explorer_state: dict      # Current explorations, seeds, etc.
    documenter_state: dict    # Document progress, pending insights
    researcher_state: dict    # Research queue, findings

    # Metrics
    metrics: OrchestratorMetrics

    # Scheduler state
    priority_weights: dict[str, float]  # From LLM consultation
    last_priority_consultation: datetime

    # Timestamps
    started_at: datetime
    last_checkpoint: datetime
```

### 5.2 Checkpointing

```python
class StateManager:
    def __init__(self, state_path: str = "orchestrator/state.json"):
        self.state_path = state_path
        self.checkpoint_interval = 60  # seconds

    async def checkpoint_loop(self):
        """Periodically save state."""
        while self.running:
            await asyncio.sleep(self.checkpoint_interval)
            self.save()

    def save(self):
        """Atomic save of current state."""
        state = self.orchestrator.get_state()

        # Write to temp file first
        temp_path = f"{self.state_path}.tmp"
        with open(temp_path, 'w') as f:
            json.dump(asdict(state), f, indent=2, default=str)

        # Atomic rename
        os.rename(temp_path, self.state_path)

        log.debug("state.checkpointed",
                  tasks=len(state.tasks),
                  path=self.state_path)

    def load(self) -> Optional[OrchestratorState]:
        """Load state from disk."""
        if not os.path.exists(self.state_path):
            return None

        with open(self.state_path) as f:
            data = json.load(f)

        return OrchestratorState(**data)
```

### 5.3 Recovery on Startup

```python
async def startup(self):
    """Initialize orchestrator, recovering state if available."""

    # Try to load saved state
    saved_state = self.state_manager.load()

    if saved_state:
        log.info("orchestrator.recovering",
                 tasks=len(saved_state.tasks),
                 last_checkpoint=saved_state.last_checkpoint)

        self.restore_state(saved_state)

        # Resume paused tasks
        for task in self.tasks.values():
            if task.state == TaskState.RUNNING:
                # Was running when crashed - mark as paused
                task.state = TaskState.PAUSED

        # Re-check LLM availability
        await self.refresh_llm_status()

    else:
        log.info("orchestrator.fresh_start")
        self.initialize_fresh()

    # Start the scheduling loop
    asyncio.create_task(self.scheduling_loop())
    asyncio.create_task(self.state_manager.checkpoint_loop())
    asyncio.create_task(self.hourly_consultation_loop())
```

---

## Part 6: Module Integration

### 6.1 Module Interface

Each module implements this interface to work with the orchestrator:

```python
class ModuleInterface(ABC):
    @abstractmethod
    def get_pending_tasks(self) -> list[Task]:
        """Return tasks this module wants to run."""
        pass

    @abstractmethod
    async def execute_task(self, task: Task, llm: LLMClient) -> TaskResult:
        """Execute a task using the provided LLM."""
        pass

    @abstractmethod
    def handle_task_result(self, task: Task, result: TaskResult):
        """Process completed/failed task."""
        pass

    @abstractmethod
    def get_state(self) -> dict:
        """Return module state for persistence."""
        pass

    @abstractmethod
    def restore_state(self, state: dict):
        """Restore module from saved state."""
        pass
```

### 6.2 Explorer Integration

```python
class ExplorerModule(ModuleInterface):
    def get_pending_tasks(self) -> list[Task]:
        tasks = []

        # Check for seeds needing exploration
        for seed in self.seed_queue:
            if not self.has_active_exploration(seed):
                tasks.append(Task(
                    module="explorer",
                    task_type="exploration",
                    priority=BASE_PRIORITIES["exploration"],
                    context={"seed_id": seed.id, "seed_priority": seed.priority}
                ))

        # Check for insights needing review
        for insight in self.pending_review:
            tasks.append(Task(
                module="explorer",
                task_type="review",
                priority=BASE_PRIORITIES["review"],
                context={"insight_id": insight.id}
            ))

        # Check for synthesis opportunities
        if self.chunks_ready_for_synthesis():
            tasks.append(Task(
                module="explorer",
                task_type="synthesis",
                priority=BASE_PRIORITIES["synthesis"],
                requires_deep_mode=True,
                context={"chunks": self.get_synthesis_chunks()}
            ))

        return tasks
```

### 6.3 Documenter Integration

```python
class DocumenterModule(ModuleInterface):
    def get_pending_tasks(self) -> list[Task]:
        tasks = []

        # Highest priority: unresolved comments
        for comment in self.find_unresolved_comments():
            tasks.append(Task(
                module="documenter",
                task_type="address_comment",
                priority=BASE_PRIORITIES["address_comment"],
                context={"comment": comment, "section_id": comment.section_id}
            ))

        # Blessed insights waiting to be documented
        for insight in self.get_undocumented_insights():
            tasks.append(Task(
                module="documenter",
                task_type="incorporate_insight",
                priority=BASE_PRIORITIES["incorporate_insight"],
                context={"insight_id": insight.id}
            ))

        # Sections needing review (based on work allocation ratio)
        if self.should_do_review():
            section = self.select_section_for_review()
            if section:
                tasks.append(Task(
                    module="documenter",
                    task_type="review_section",
                    priority=BASE_PRIORITIES["review_section"],
                    context={"section_id": section.id}
                ))

        return tasks
```

### 6.4 Researcher Integration

```python
class ResearcherModule(ModuleInterface):
    def get_pending_tasks(self) -> list[Task]:
        tasks = []

        # Generate research questions from recent activity
        if self.should_generate_questions():
            tasks.append(Task(
                module="researcher",
                task_type="generate_questions",
                priority=BASE_PRIORITIES["generate_questions"],
                context={"activity": self.get_recent_activity()}
            ))

        # Evaluate pending sources
        for source in self.sources_pending_evaluation:
            tasks.append(Task(
                module="researcher",
                task_type="evaluate_source",
                priority=BASE_PRIORITIES["evaluate_source"],
                context={"source_id": source.id, "url": source.url}
            ))

        # Extract content from trusted sources
        for source in self.sources_ready_for_extraction:
            tasks.append(Task(
                module="researcher",
                task_type="extract_content",
                priority=BASE_PRIORITIES["extract_content"],
                context={"source_id": source.id}
            ))

        return tasks
```

---

## Part 7: Metrics and Monitoring

### 7.1 Metrics Tracked

```python
@dataclass
class OrchestratorMetrics:
    # Throughput
    tasks_completed_24h: dict[str, int]   # By task type
    insights_generated_24h: int
    sections_documented_24h: int
    research_items_24h: int

    # LLM usage
    llm_requests_24h: dict[str, int]      # By LLM
    deep_mode_usage_24h: dict[str, int]
    rate_limits_hit_24h: dict[str, int]

    # Efficiency
    avg_task_duration: dict[str, float]   # By task type
    preemptions_24h: int
    task_failures_24h: int

    # Queue health
    queue_depth: dict[str, int]           # By module
    oldest_pending_task: datetime

    # Backlog
    blessed_insights_pending: int
    comments_pending: int
    seeds_unexplored: int
```

### 7.2 Logging

All orchestrator actions logged in structured format:

```python
# Task lifecycle
log.info("task.scheduled", task_id=task.id, module=task.module,
         task_type=task.task_type, priority=score)

log.info("task.started", task_id=task.id, llm=llm.name)

log.info("task.completed", task_id=task.id,
         duration_ms=duration, llm_requests=task.llm_requests)

log.info("task.preempted", task_id=task.id, reason="higher_priority",
         preempting_task=other.id)

log.error("task.failed", task_id=task.id, error=str(e))

# LLM allocation
log.info("llm.allocated", llm=llm.name, task_id=task.id)

log.warning("llm.rate_limited", llm=llm.name, retry_after=seconds)

log.info("llm.recovered", llm=llm.name)

# Scheduler decisions
log.info("scheduler.priority_consultation",
         explorer_weight=weights["explorer"],
         documenter_weight=weights["documenter"],
         researcher_weight=weights["researcher"])
```

---

## Part 8: Configuration

```yaml
orchestrator:
  # State persistence
  state_path: orchestrator/state.json
  checkpoint_interval_seconds: 60

  # Scheduling
  scheduling:
    tick_interval_ms: 100
    preemption_enabled: true
    starvation_prevention:
      max_wait_hours: 4
      boost_per_hour: 5

  # Priority consultation
  llm_consultation:
    enabled: true
    interval_minutes: 60
    on_significant_change: true
    significant_change_threshold: 10  # Tasks completed since last consultation

  # LLM quotas (initial estimates, learned over time)
  quotas:
    chatgpt:
      hourly_limit: 50
      deep_mode_daily: 25
    gemini:
      hourly_limit: 50
      deep_mode_daily: 5    # Very limited!
    claude:
      hourly_limit: 50
      deep_mode_daily: 100

  # Module weights (default, overridden by LLM consultation)
  default_weights:
    explorer: 40
    documenter: 40
    researcher: 20

  # Failure handling
  failure:
    max_retries: 3
    retry_delay_seconds: 60
    backoff_multiplier: 2

  # Logging
  logging:
    component: orchestrator
```

---

## Part 9: File Structure

```
fano/
├── orchestrator/
│   ├── __init__.py
│   ├── main.py              # Entry point
│   ├── scheduler.py         # Priority computation, task selection
│   ├── allocator.py         # LLM allocation, quota tracking
│   ├── executor.py          # Task execution, preemption
│   ├── state.py             # State management, persistence
│   ├── metrics.py           # Metrics collection
│   ├── consultation.py      # LLM priority consultation
│   ├── models.py            # Task, LLMInstance, etc.
│   └── state.json           # Persisted state (auto-generated)
│
├── explorer/
│   └── src/
│       └── module.py        # Implements ModuleInterface
│
├── documenter/
│   └── module.py            # Implements ModuleInterface
│
├── researcher/
│   └── module.py            # Implements ModuleInterface
│
└── config.yaml              # Orchestrator config section
```

---

## Part 10: Migration Path

### 10.1 Current State

- Modules run as independent processes via ProcessManager
- Each module has its own orchestrator/main loop
- LLMs accessed directly through pool service
- No cross-module coordination

### 10.2 Migration Steps

1. **Create orchestrator skeleton** with state management
2. **Add ModuleInterface** to each module without changing their logic
3. **Refactor LLM access** to go through orchestrator's allocator
4. **Add task model** - modules register tasks instead of running loops
5. **Implement scheduler** with basic priority rules
6. **Add preemption support** to task execution
7. **Add LLM consultation** for intelligent prioritization
8. **Update control panel** to start/stop orchestrator instead of individual modules

### 10.3 Backward Compatibility

During migration, modules can still run independently. The orchestrator becomes the preferred way to run but doesn't break existing scripts.

---

## Part 11: Success Criteria

The orchestrator is working correctly when:

1. **Autonomous operation**: Runs for days without intervention
2. **Balanced progress**: All three modules make progress, no starvation
3. **Intelligent allocation**: Responds to backlogs by shifting priorities
4. **Rate limit resilience**: Hits limits gracefully, recovers automatically
5. **Crash recovery**: Resumes from checkpoint after restart
6. **Quota efficiency**: Uses 80%+ of available LLM capacity
7. **Low overhead**: LLM consultation uses <10% of total LLM time
8. **Observable**: Rich logging enables debugging and monitoring
9. **Comment responsiveness**: Author comments processed within 1-2 hours
10. **Seed throughput**: High-priority seeds explored within 24 hours

---

## Part 12: Open Questions

1. **Browser pool simplification**: With single-instance LLMs, do we still need the full pool service, or can we simplify?

2. **Researcher scope**: The researcher module seems underspecified. What sources does it access? How does it find them?

3. **Cross-module insights**: Should researcher findings feed directly into explorer seeds, or go through human review first?

4. **Conversation continuity**: When preempting a multi-turn exploration, how important is it to resume with the same LLM vs. starting fresh?

5. **Document conflicts**: If documenter and explorer both want the same LLM for deep mode, who wins?

---

## Part 13: Design Review Addendum

**Date:** 2026-01-14
**Reviewers:** Gemini DT, ChatGPT Pro 5.2
**Status:** Validated against codebase

This section addresses critical feedback from external design reviews. Each issue has been validated against the actual Fano codebase to determine applicability.

### 13.1 Validation Summary

| ID | Issue | Reviewer | Verdict | Severity |
|----|-------|----------|---------|----------|
| A | GIL Blocking / Event Loop Blocker | Gemini | **PARTIALLY VALID** | HIGH |
| B | WAL Write Amplification | Gemini | **VALID** | CRITICAL |
| C | Quota Amnesia on Crash | Gemini | **PARTIALLY VALID** | HIGH |
| D | O(n) Scheduler Priority Recomputation | Gemini | **VALID** | MEDIUM |
| E | Context Truncation Loses System Prompt | Both | **VALID** | CRITICAL |
| F | Actor Model vs Polling Conflict | Gemini | **NOT APPLICABLE** | - |
| G | Zombie Browser Processes | Gemini | **VALID** | HIGH |
| 1 | WAL Recovery Logic Unsafe | ChatGPT | **VALID** | CRITICAL |
| 2 | Infinite Task Duplication | ChatGPT | **VALID** | CRITICAL |
| 3 | Shared State Contradicts Actor Model | ChatGPT | **VALID** | HIGH |
| 4 | LLM Gateway Global Lock | ChatGPT | **VALID** | HIGH |
| 5 | Quota Availability Not Consistently Gated | ChatGPT | **VALID** | HIGH |
| 6 | WAL Compaction Not Implemented | ChatGPT | **VALID** | HIGH |
| 7 | Module Restore Order Wrong | ChatGPT | **VALID** | MEDIUM |
| 8 | RUNNING→PAUSED Not Persisted | ChatGPT | **VALID** | MEDIUM |
| 9 | TaskExecutor Doesn't Call handle_failure | ChatGPT | **VALID** | MEDIUM |
| 10 | Scheduler API Inconsistencies | ChatGPT | **VALID** | LOW |
| 11 | READY State Unused | ChatGPT | **VALID** | LOW |
| 12 | Conversation Truncation Breaks Continuity | ChatGPT | **VALID** | HIGH |

### 13.2 Issue Details and Validation

#### Issue A: GIL Blocking (Gemini)

**Claim:** Deduplication and WAL serialization are CPU-bound, blocking the event loop.

**Validation:**
- **Deduplication**: Reviewed `shared/deduplication/checker.py`. The dedup checker uses async LLM callbacks for semantic checking, NOT CPU-intensive operations. Text processing (`calculate_similarity`) is lightweight O(n) string operations.
- **WAL Serialization**: Valid concern. Serializing large conversation states (50+ messages with code/math) via `json.dump()` can block for 100-500ms.

**Verdict:** PARTIALLY VALID - WAL serialization needs offloading, but dedup is already async.

**Resolution:** Use `loop.run_in_executor()` for JSON serialization of large states (>10KB).

---

#### Issue B: WAL Write Amplification (Gemini)

**Claim:** Logging entire conversation state before every LLM request creates O(n²) I/O growth.

**Validation:** The design shows `_save_conversation_state` writing full message history. With 50-turn conversations and math content, this becomes gigabytes per day.

**Verdict:** VALID

**Resolution:** Implement delta logging - only log new messages added per turn.

---

#### Issue C: Quota Amnesia (Gemini)

**Claim:** Rate limit state lost on crash, causing quota drain on crash loops.

**Validation:** Reviewed `pool/src/state.py`. The pool DOES persist quota state to `pool_state.json` with a lock. However:
- The orchestrator's own quota tracking (in design) is in-memory only
- There's a window between quota updates where state could be lost

**Verdict:** PARTIALLY VALID - Pool persists quota, but orchestrator needs to either use pool as source-of-truth OR persist its own quota to WAL.

**Resolution:** Single source of truth - orchestrator queries pool for quota status, does not maintain separate counters.

---

#### Issue D: O(n) Scheduler Bottleneck (Gemini)

**Claim:** Recomputing priorities for all tasks on each `get_next_task` call is expensive.

**Validation:** The pseudocode shows `_recompute_all_priorities()` called inside the scheduling lock. With 500+ tasks, this adds significant latency.

**Verdict:** VALID

**Resolution:** Lazy priority recomputation - only recompute if >60s since last recompute OR if task count changed significantly.

---

#### Issue E: Context Truncation Lobotomy (Both Reviewers)

**Claim:** `messages = messages[-50:]` drops the system prompt (index 0) and seed question (index 1).

**Validation:** CONFIRMED. The design's truncation logic is destructive.

**Verdict:** CRITICAL - VALID

**Resolution:** Head + Tail retention: `messages = messages[:2] + messages[-48:]`

---

#### Issue F: Actor Model vs Polling Conflict (Gemini)

**Claim:** Design describes Actor Model with MessageBus but implements Polling.

**Validation:** Reviewed the design document - there is NO MessageBus section. The design consistently uses polling (`get_pending_tasks`).

**Verdict:** NOT APPLICABLE - Gemini may have reviewed a different version.

---

#### Issue G: Zombie Browser Processes (Gemini)

**Claim:** If orchestrator crashes hard (SIGKILL/OOM), browser processes are orphaned.

**Validation:** CONFIRMED. The pool's Playwright browsers run as child processes. Hard crashes leave them running.

**Verdict:** VALID

**Resolution:** PID lifecycle manager - on startup, read `gateway.pid` file, kill listed PIDs, write new PIDs.

---

#### Issue 1: WAL Recovery Logic Unsafe (ChatGPT)

**Claim:** Recovery can replay old entries and regress state. Checkpoint sequence stored but not used for filtering.

**Validation:** CONFIRMED. The recovery pseudocode has multiple issues:
- Commit markers processed after entries, so entries stay in uncommitted set
- Checkpoint stores `wal_sequence` but recovery doesn't filter by it
- Replaying old task updates can revert state (COMPLETED → PENDING)
- datetime/enum serialization mismatch (writes as string, reads expect typed)

**Verdict:** CRITICAL - VALID

**Resolution:** Adopt "Simple Redo Log" approach (Option A from ChatGPT):
1. Remove explicit commit markers
2. Each WAL entry is a full, idempotent state upsert
3. On checkpoint: record `last_wal_sequence`
4. On recovery: replay only entries with `sequence > checkpoint.last_wal_sequence`
5. On WAL read: stop at first JSON decode error (partial line from crash)

---

#### Issue 2: Infinite Task Duplication (ChatGPT)

**Claim:** `get_pending_tasks()` generates new UUIDs each call, flooding queue with duplicates.

**Validation:** CONFIRMED. Looking at the design:
```python
tasks.append(Task(
    module="explorer",
    task_type="exploration",
    context={"seed_id": seed.id, ...}
))
```
No ID is specified, so each call creates new Task objects.

**Verdict:** CRITICAL - VALID

**Resolution:** Stable task keys (idempotency):
```python
@dataclass
class Task:
    task_key: str  # Stable: f"{module}:{task_type}:{context_hash}"
    ...
```
StateManager maintains `active_task_keys: set[str]` for non-completed tasks. Scheduler's `submit()` becomes "submit-if-absent by key".

---

#### Issue 3: Shared State Contradicts Actor Model (ChatGPT)

**Claim:** Design claims "actor isolation" but uses shared StateManager and shared dedup checker.

**Validation:** CONFIRMED. The design mixes patterns.

**Verdict:** VALID

**Resolution:** Adopt "Single-threaded async + disciplined locking" model:
- Drop "actor model" terminology
- StateManager is sole owner of task/module state, all access through async methods with lock
- DedupService provides async access to shared dedup checker

---

#### Issue 4: LLM Gateway Global Lock (ChatGPT)

**Claim:** Single lock on LLMGateway serializes all LLM requests across backends.

**Validation:** The design shows:
```python
async with self._lock:  # Only one request per browser at a time
```
This is at the gateway level, not per-backend.

**Verdict:** VALID - defeats the purpose of having multiple LLM subscriptions.

**Resolution:** Per-backend locks:
```python
class LLMGateway:
    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = {
            "gemini": asyncio.Lock(),
            "chatgpt": asyncio.Lock(),
            "claude": asyncio.Lock(),
        }

    async def send(self, backend: str, ...):
        async with self._locks[backend]:
            ...
```

---

#### Issue 5: Quota Not Consistently Gated (ChatGPT)

**Claim:** Multiple places check quota availability differently, causing scheduler churn.

**Validation:** The design shows:
- Scheduler checks backend availability
- Allocator checks quota separately
- Deep mode checked in yet another place

**Verdict:** VALID

**Resolution:** Single gate - `allocator.can_execute(task, available_backends) -> bool` is THE decision point. Scheduler calls this, not its own checks.

---

#### Issue 6: WAL Compaction Not Implemented (ChatGPT)

**Claim:** WAL will grow unbounded, filling disk.

**Validation:** The design mentions `compact_threshold` in config but `compact()` is not implemented.

**Verdict:** VALID

**Resolution:** WAL rotation on checkpoint:
1. After successful checkpoint, close current WAL file
2. Start new WAL file with incremented suffix
3. Delete WAL files whose max sequence ≤ checkpoint sequence
4. Hard cap: if WAL dir > 500MB, force checkpoint + rotation

---

#### Issue 7: Module Restore Order Wrong (ChatGPT)

**Claim:** Recovery resumes tasks before restoring module state.

**Validation:** The startup pseudocode shows:
```python
# Resume paused tasks
for task in self.tasks.values():
    ...
# THEN restore modules
for module in self.modules.values():
    module.restore_state(...)
```

**Verdict:** VALID - tasks could execute against uninitialized modules.

**Resolution:** Swap order: restore modules FIRST, then resume/resubmit tasks.

---

#### Issue 8: RUNNING→PAUSED Not Persisted (ChatGPT)

**Claim:** Recovery marks RUNNING tasks as PAUSED but doesn't persist, so another crash repeats it.

**Validation:** The recovery logic only logs the correction:
```python
if task.state == TaskState.RUNNING:
    task.state = TaskState.PAUSED  # Only in memory!
```

**Verdict:** VALID

**Resolution:** After state corrections during recovery, persist via `checkpoint()` before resuming operation.

---

#### Issue 9: TaskExecutor Doesn't Call handle_failure (ChatGPT)

**Claim:** Exception handling marks task FAILED but never calls module's cleanup hook.

**Validation:** CONFIRMED. The executor catches exceptions but doesn't invoke `module.handle_failure()`.

**Verdict:** VALID

**Resolution:** In failure path, call `await module.handle_failure(task, error)` before returning.

---

#### Issue 10: Scheduler API Inconsistencies (ChatGPT)

**Claim:** `peek_next()` and `_recompute_all_priorities()` referenced but not defined.

**Validation:** The preemption check calls `scheduler.peek_next()` which isn't in the Scheduler class.

**Verdict:** VALID

**Resolution:** Either:
- Remove preemption based on peek (use time-slice policy), OR
- Define `peek_next()` that returns highest-priority task without removing it

---

#### Issue 11: READY State Unused (ChatGPT)

**Claim:** TaskState.READY is defined but never used in transitions.

**Validation:** CONFIRMED. Tasks go PENDING → RUNNING → COMPLETED, READY never appears.

**Verdict:** VALID

**Resolution:** Remove READY from TaskState. Simplify to: PENDING, RUNNING, PAUSED, COMPLETED, FAILED.

---

#### Issue 12: Conversation Truncation Breaks Reasoning (ChatGPT)

**Claim:** Keeping last 50 messages loses critical early context in math work.

**Validation:** Mathematical explorations build on definitions, assumptions, and symbols established early. Dropping them breaks reasoning.

**Verdict:** VALID (same as Issue E)

**Resolution:** In addition to head+tail retention, maintain a running summary:
- Every 10 turns, generate a "context summary" of key definitions/progress
- Keep summary as pinned message after system prompt
- Truncation becomes: `[system, summary, ...last 46 messages]`

---

## Part 14: Updated Core Components

This section provides corrected pseudocode for components affected by the review.

### 14.1 WAL Manager (Corrected)

Adopts the "Simple Redo Log" approach - no commit markers, sequence-filtered recovery.

```python
@dataclass
class WALEntry:
    sequence: int
    timestamp: datetime
    entry_type: str  # "task_update" | "module_state" | "allocator_state"
    data: dict

class WALManager:
    """Write-Ahead Log with simple redo semantics."""

    def __init__(self, wal_dir: Path, max_size_mb: int = 100):
        self.wal_dir = wal_dir
        self.max_size_mb = max_size_mb
        self.sequence = 0
        self._file_handle: Optional[IO] = None
        self._current_wal_path: Optional[Path] = None
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _get_current_wal_path(self) -> Path:
        """Get path for current WAL file."""
        files = sorted(self.wal_dir.glob("wal_*.jsonl"))
        if files:
            return files[-1]
        return self.wal_dir / "wal_000001.jsonl"

    async def append(self, entry_type: str, data: dict) -> int:
        """
        Append entry to WAL. Returns sequence number.

        Uses executor for large data to avoid GIL blocking.
        """
        self.sequence += 1
        entry = WALEntry(
            sequence=self.sequence,
            timestamp=datetime.now(),
            entry_type=entry_type,
            data=data,
        )

        # Serialize (potentially blocking for large data)
        loop = asyncio.get_running_loop()
        line = await loop.run_in_executor(
            self._executor,
            lambda: json.dumps(asdict(entry), default=str) + "\n"
        )

        # Append to file
        self._ensure_file()
        self._file_handle.write(line)
        self._file_handle.flush()
        os.fsync(self._file_handle.fileno())

        return self.sequence

    async def append_delta(self, task_id: str, new_messages: list[dict]) -> int:
        """
        Append conversation delta (not full state) to WAL.

        This is the preferred method for conversation updates - avoids O(n²) growth.
        """
        return await self.append("conversation_delta", {
            "task_id": task_id,
            "messages": new_messages,  # Only NEW messages this turn
            "message_count": len(new_messages),
        })

    def recover(self, checkpoint_sequence: int) -> list[WALEntry]:
        """
        Recover entries after checkpoint.

        Only returns entries with sequence > checkpoint_sequence.
        Stops at first parse error (partial write from crash).
        """
        entries = []

        for wal_file in sorted(self.wal_dir.glob("wal_*.jsonl")):
            try:
                with open(wal_file, encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            entry = WALEntry(
                                sequence=data["sequence"],
                                timestamp=datetime.fromisoformat(data["timestamp"]),
                                entry_type=data["entry_type"],
                                data=data["data"],
                            )

                            # Only include entries after checkpoint
                            if entry.sequence > checkpoint_sequence:
                                entries.append(entry)

                            # Track max sequence seen
                            self.sequence = max(self.sequence, entry.sequence)

                        except (json.JSONDecodeError, KeyError) as e:
                            # Partial line from crash - stop here
                            log.warning("wal.recovery.partial_entry",
                                       file=wal_file.name,
                                       line=line_num,
                                       error=str(e))
                            break
            except FileNotFoundError:
                continue

        log.info("wal.recovery.complete",
                entries_recovered=len(entries),
                max_sequence=self.sequence)

        return sorted(entries, key=lambda e: e.sequence)

    def rotate(self, checkpoint_sequence: int):
        """
        Rotate WAL after checkpoint.

        1. Close current file
        2. Delete files with max sequence <= checkpoint
        3. Start new file
        """
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

        # Delete old WAL files
        for wal_file in self.wal_dir.glob("wal_*.jsonl"):
            try:
                # Check max sequence in file
                max_seq = 0
                with open(wal_file, encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            max_seq = max(max_seq, data.get("sequence", 0))
                        except json.JSONDecodeError:
                            break

                if max_seq <= checkpoint_sequence:
                    wal_file.unlink()
                    log.info("wal.rotated.deleted", file=wal_file.name)
            except Exception as e:
                log.warning("wal.rotation.error", file=wal_file.name, error=str(e))

        # Start new file
        new_name = f"wal_{self.sequence + 1:06d}.jsonl"
        self._current_wal_path = self.wal_dir / new_name

    def _ensure_file(self):
        """Ensure WAL file is open for writing."""
        if self._file_handle is None:
            if self._current_wal_path is None:
                self._current_wal_path = self._get_current_wal_path()
            self.wal_dir.mkdir(parents=True, exist_ok=True)
            self._file_handle = open(self._current_wal_path, "a", encoding="utf-8")
```

### 14.2 Task Model (Corrected)

Adds stable task keys and simplifies state machine.

```python
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import hashlib

class TaskState(Enum):
    """Simplified task states - READY removed."""
    PENDING = "pending"      # Waiting to be scheduled
    RUNNING = "running"      # Currently executing
    PAUSED = "paused"        # Preempted, waiting to resume
    COMPLETED = "completed"  # Successfully finished
    FAILED = "failed"        # Encountered unrecoverable error

@dataclass
class Task:
    """Represents a unit of work with stable identity."""

    # Identity
    id: str                          # UUID for this instance
    task_key: str                    # Stable key: f"{module}:{task_type}:{context_hash}"

    # Classification
    module: str                      # "explorer" | "documenter" | "researcher"
    task_type: str                   # e.g., "exploration", "synthesis"

    # Scheduling
    priority: int                    # Higher = more urgent
    state: TaskState

    # Context
    context: dict                    # Task-specific data

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Execution tracking
    llm_requests: int = 0
    retry_count: int = 0
    llm_preference: Optional[str] = None
    requires_deep_mode: bool = False

    # Conversation state (for multi-turn tasks)
    conversation_state: Optional[dict] = None

    @staticmethod
    def generate_task_key(module: str, task_type: str, context: dict) -> str:
        """
        Generate stable task key from module, type, and context.

        The key uniquely identifies the WORK to be done, not the instance.
        Multiple calls with same args return same key.
        """
        # Extract stable identifiers from context
        stable_parts = []
        for key in sorted(context.keys()):
            value = context[key]
            # Only include primitive values that identify the work
            if isinstance(value, (str, int, bool)):
                stable_parts.append(f"{key}={value}")

        context_str = "|".join(stable_parts)
        context_hash = hashlib.md5(context_str.encode()).hexdigest()[:12]

        return f"{module}:{task_type}:{context_hash}"

    @classmethod
    def create(
        cls,
        module: str,
        task_type: str,
        context: dict,
        priority: int = 50,
        **kwargs
    ) -> "Task":
        """Factory method that auto-generates task_key."""
        import uuid
        task_key = cls.generate_task_key(module, task_type, context)
        return cls(
            id=uuid.uuid4().hex[:12],
            task_key=task_key,
            module=module,
            task_type=task_type,
            context=context,
            priority=priority,
            state=TaskState.PENDING,
            **kwargs
        )
```

### 14.3 Scheduler (Corrected)

Implements lazy priority recomputation and task key deduplication.

```python
import heapq
from typing import Optional
import asyncio

class Scheduler:
    """
    Task scheduler with lazy priority recomputation.

    Key improvements:
    - Deduplicates by task_key (prevents infinite task creation)
    - Lazy priority recomputation (not on every get_next_task)
    - Proper heap-based priority queue
    """

    def __init__(self, state_manager: "StateManager"):
        self.state = state_manager
        self._lock = asyncio.Lock()

        # Priority queue: (negative_priority, created_at, task_id)
        # Negative because heapq is min-heap but we want max priority
        self._heap: list[tuple[int, datetime, str]] = []

        # Active task keys (for deduplication)
        self._active_keys: set[str] = set()

        # Priority recomputation tracking
        self._last_priority_recompute = datetime.min
        self._priority_recompute_interval = timedelta(seconds=60)
        self._task_count_at_last_recompute = 0

    async def submit(self, task: Task) -> bool:
        """
        Submit task if not already active (by task_key).

        Returns True if task was added, False if duplicate.
        """
        async with self._lock:
            # Check for duplicate by task_key
            if task.task_key in self._active_keys:
                log.debug("scheduler.duplicate_skipped",
                         task_key=task.task_key,
                         task_type=task.task_type)
                return False

            # Add to state manager
            self.state.tasks[task.id] = task

            # Track active key
            self._active_keys.add(task.task_key)

            # Add to heap
            heapq.heappush(
                self._heap,
                (-task.priority, task.created_at, task.id)
            )

            log.info("scheduler.task_submitted",
                    task_id=task.id,
                    task_key=task.task_key,
                    priority=task.priority)

            return True

    async def get_next_task(
        self,
        available_backends: list[str],
        allocator: "Allocator"
    ) -> Optional[Task]:
        """
        Get highest-priority runnable task.

        A task is runnable if:
        - State is PENDING or PAUSED
        - Required backend is available (or no preference)
        - Allocator approves (quota available)
        """
        async with self._lock:
            # Maybe recompute priorities
            await self._maybe_recompute_priorities()

            # Pop from heap until we find a runnable task
            skipped = []
            result = None

            while self._heap:
                neg_priority, created_at, task_id = heapq.heappop(self._heap)
                task = self.state.tasks.get(task_id)

                if task is None:
                    # Task was removed
                    continue

                if task.state == TaskState.COMPLETED:
                    # Clean up completed task key
                    self._active_keys.discard(task.task_key)
                    continue

                if task.state == TaskState.FAILED:
                    # Check retry eligibility
                    if task.retry_count >= 3:
                        self._active_keys.discard(task.task_key)
                        continue

                if task.state not in (TaskState.PENDING, TaskState.PAUSED):
                    # Task is RUNNING - skip but keep in queue
                    skipped.append((neg_priority, created_at, task_id))
                    continue

                # Check if required backend is available
                if task.llm_preference and task.llm_preference not in available_backends:
                    skipped.append((neg_priority, created_at, task_id))
                    continue

                # Check allocator approval (single gate for quota/availability)
                if not allocator.can_execute(task, available_backends):
                    skipped.append((neg_priority, created_at, task_id))
                    continue

                # Found a runnable task
                result = task
                break

            # Put skipped tasks back
            for item in skipped:
                heapq.heappush(self._heap, item)

            return result

    async def _maybe_recompute_priorities(self):
        """Recompute priorities if stale (lazy recomputation)."""
        now = datetime.now()
        task_count = len(self.state.tasks)

        # Recompute if:
        # 1. More than 60s since last recompute, OR
        # 2. Task count changed by more than 20%
        should_recompute = (
            now - self._last_priority_recompute > self._priority_recompute_interval or
            abs(task_count - self._task_count_at_last_recompute) > task_count * 0.2
        )

        if not should_recompute:
            return

        log.debug("scheduler.recomputing_priorities", task_count=task_count)

        # Rebuild heap with updated priorities
        new_heap = []
        for task_id, task in self.state.tasks.items():
            if task.state in (TaskState.COMPLETED, TaskState.FAILED):
                continue

            priority = self._compute_priority(task)
            task.priority = priority  # Update task's priority
            heapq.heappush(new_heap, (-priority, task.created_at, task_id))

        self._heap = new_heap
        self._last_priority_recompute = now
        self._task_count_at_last_recompute = task_count

    def _compute_priority(self, task: Task) -> int:
        """Compute dynamic priority for task."""
        score = BASE_PRIORITIES.get(task.task_type, 50)

        # Factor 1: Backlog pressure
        if task.module == "documenter":
            backlog = self.state.metrics.get("blessed_insights_pending", 0)
            if backlog > 20:
                score += 30
            elif backlog > 10:
                score += 15

        # Factor 2: Starvation prevention
        age_hours = (datetime.now() - task.updated_at).total_seconds() / 3600
        score += min(int(age_hours * 2), 20)

        # Factor 3: Comments get priority
        if task.task_type == "address_comment":
            score += 25

        return score

    def mark_completed(self, task_id: str):
        """Mark task completed and clean up."""
        task = self.state.tasks.get(task_id)
        if task:
            task.state = TaskState.COMPLETED
            task.updated_at = datetime.now()
            self._active_keys.discard(task.task_key)
```

### 14.4 Task Executor (Corrected)

Fixes context truncation and calls module.handle_failure.

```python
class TaskExecutor:
    """
    Executes tasks with proper context management.

    Key improvements:
    - Head + tail context retention (preserves system prompt)
    - Calls module.handle_failure() on errors
    - Running summary for long conversations
    """

    def __init__(
        self,
        gateway: "LLMGateway",
        allocator: "Allocator",
        scheduler: "Scheduler",
        modules: dict[str, "ModuleInterface"],
        wal: "WALManager",
    ):
        self.gateway = gateway
        self.allocator = allocator
        self.scheduler = scheduler
        self.modules = modules
        self.wal = wal

        # Context management
        self.max_context_messages = 50
        self.summary_interval = 10  # Generate summary every N turns

    async def execute(self, task: Task) -> TaskResult:
        """Execute task with full error handling."""
        module = self.modules.get(task.module)
        if not module:
            return TaskResult(failed=True, error=f"Unknown module: {task.module}")

        try:
            task.state = TaskState.RUNNING
            task.updated_at = datetime.now()

            # Allocate LLM
            llm = self.allocator.allocate(task)
            if not llm:
                task.state = TaskState.PAUSED
                return TaskResult(preempted=True, reason="No LLM available")

            # Execute with module
            result = await module.execute_task(task, llm)

            if result.completed:
                task.state = TaskState.COMPLETED
                self.scheduler.mark_completed(task.id)
            elif result.preempted:
                task.state = TaskState.PAUSED
                await self._save_conversation_state(task)

            return result

        except Exception as e:
            log.error("executor.task_failed",
                     task_id=task.id,
                     error=str(e),
                     exc_info=True)

            task.state = TaskState.FAILED
            task.retry_count += 1
            task.updated_at = datetime.now()

            # Call module cleanup hook
            try:
                await module.handle_failure(task, str(e))
            except Exception as cleanup_error:
                log.warning("executor.cleanup_failed",
                           task_id=task.id,
                           error=str(cleanup_error))

            return TaskResult(failed=True, error=str(e))

    async def _save_conversation_state(self, task: Task):
        """
        Save conversation state with delta logging.

        Only logs NEW messages, not entire history.
        """
        if not task.conversation_state:
            return

        messages = task.conversation_state.get("messages", [])
        last_saved = task.conversation_state.get("_last_saved_count", 0)

        # Only log new messages (delta)
        new_messages = messages[last_saved:]
        if new_messages:
            await self.wal.append_delta(task.id, new_messages)
            task.conversation_state["_last_saved_count"] = len(messages)

    def _truncate_context(self, messages: list[dict]) -> list[dict]:
        """
        Truncate context while preserving critical messages.

        CRITICAL: Always keep:
        - Index 0: System prompt
        - Index 1: Initial seed/question (or summary if present)
        - Last N-2 messages: Recent context
        """
        if len(messages) <= self.max_context_messages:
            return messages

        # Keep first 2 (system prompt + seed/summary) + last 48
        preserved_head = messages[:2]
        preserved_tail = messages[-(self.max_context_messages - 2):]

        log.info("executor.context_truncated",
                original_count=len(messages),
                preserved_count=len(preserved_head) + len(preserved_tail),
                dropped_count=len(messages) - len(preserved_head) - len(preserved_tail))

        return preserved_head + preserved_tail

    async def _maybe_generate_summary(
        self,
        task: Task,
        llm: "LLMClient"
    ) -> Optional[str]:
        """
        Generate running summary for long conversations.

        Called every summary_interval turns to maintain context.
        """
        messages = task.conversation_state.get("messages", [])
        turn_count = len(messages) // 2  # Approximate turn count

        if turn_count % self.summary_interval != 0:
            return None

        # Generate summary of key points
        summary_prompt = """Summarize the key definitions, assumptions, symbols,
        and progress made so far in this mathematical exploration.
        Keep it concise (max 200 words) but include all critical context
        needed to continue the discussion."""

        try:
            summary = await llm.send(summary_prompt)

            # Update message at index 1 with current summary
            if len(messages) > 1:
                messages[1] = {
                    "role": "system",
                    "content": f"[Running Summary]\n{summary}"
                }

            return summary
        except Exception as e:
            log.warning("executor.summary_failed", task_id=task.id, error=str(e))
            return None
```

### 14.5 LLM Gateway (Corrected)

Implements per-backend locks for true parallelism.

```python
class LLMGateway:
    """
    Gateway to LLM backends with per-backend locking.

    Key improvement: Per-backend locks allow parallel requests
    to different LLMs (Gemini + ChatGPT + Claude simultaneously).
    """

    def __init__(self, pool_client: "PoolClient"):
        self.pool = pool_client

        # Per-backend locks (not a single global lock!)
        self._locks: dict[str, asyncio.Lock] = {
            "gemini": asyncio.Lock(),
            "chatgpt": asyncio.Lock(),
            "claude": asyncio.Lock(),
        }

        # Backend status tracking
        self._backends: dict[str, LLMInstance] = {}

        # PID tracking for orphan cleanup
        self._pid_file = Path("orchestrator/gateway.pid")

    async def startup(self):
        """Initialize gateway with orphan cleanup."""
        # Kill any orphaned browser processes from previous crash
        await self._cleanup_orphans()

        # Initialize backends
        await self._refresh_backend_status()

        # Write current PIDs
        await self._write_pids()

    async def _cleanup_orphans(self):
        """Kill orphaned browser processes from previous run."""
        if not self._pid_file.exists():
            return

        try:
            with open(self._pid_file) as f:
                old_pids = json.load(f)

            for pid in old_pids:
                try:
                    os.kill(pid, signal.SIGTERM)
                    log.info("gateway.orphan_killed", pid=pid)
                except ProcessLookupError:
                    pass  # Already dead
                except PermissionError:
                    log.warning("gateway.orphan_permission_denied", pid=pid)

            self._pid_file.unlink()
        except Exception as e:
            log.warning("gateway.orphan_cleanup_failed", error=str(e))

    async def _write_pids(self):
        """Write current browser process PIDs for crash recovery."""
        try:
            pids = await self.pool.get_browser_pids()
            with open(self._pid_file, "w") as f:
                json.dump(pids, f)
        except Exception as e:
            log.warning("gateway.pid_write_failed", error=str(e))

    async def send(
        self,
        backend: str,
        prompt: str,
        **options
    ) -> str:
        """
        Send request to backend with per-backend locking.

        Multiple backends can process in parallel.
        """
        if backend not in self._locks:
            raise ValueError(f"Unknown backend: {backend}")

        # Lock ONLY this backend, not all backends
        async with self._locks[backend]:
            instance = self._backends.get(backend)
            if not instance or instance.status != "available":
                raise LLMUnavailableError(f"{backend} not available")

            instance.status = "busy"
            instance.current_task = options.get("task_id")

            try:
                response = await self.pool.send(backend, prompt, **options)
                return response
            finally:
                instance.status = "available"
                instance.current_task = None

    def get_available_backends(self) -> list[str]:
        """Get list of currently available backends."""
        return [
            name for name, instance in self._backends.items()
            if instance.status == "available"
        ]
```

### 14.6 Recovery Manager (Corrected)

Fixes restore order and persists state corrections.

```python
class RecoveryManager:
    """
    Manages crash recovery with correct ordering.

    Key improvements:
    - Restores modules BEFORE resuming tasks
    - Persists RUNNING→PAUSED corrections
    - Uses sequence-filtered WAL replay
    """

    def __init__(
        self,
        state_manager: "StateManager",
        wal: "WALManager",
        modules: dict[str, "ModuleInterface"],
        scheduler: "Scheduler",
    ):
        self.state = state_manager
        self.wal = wal
        self.modules = modules
        self.scheduler = scheduler

    async def recover(self) -> bool:
        """
        Recover from crash.

        Returns True if recovery was performed, False if fresh start.
        """
        checkpoint = self.state.load_checkpoint()
        if checkpoint is None:
            log.info("recovery.fresh_start")
            return False

        log.info("recovery.starting",
                checkpoint_time=checkpoint.timestamp,
                checkpoint_sequence=checkpoint.wal_sequence)

        # Step 1: Restore base state from checkpoint
        self.state.restore_from_checkpoint(checkpoint)

        # Step 2: Replay WAL entries after checkpoint
        entries = self.wal.recover(checkpoint.wal_sequence)
        await self._replay_entries(entries)

        # Step 3: Fix RUNNING tasks (they were interrupted)
        corrections_made = self._fix_interrupted_tasks()

        # Step 4: PERSIST corrections before doing anything else
        if corrections_made:
            await self.state.checkpoint()
            log.info("recovery.corrections_persisted", count=corrections_made)

        # Step 5: Restore MODULE state (BEFORE resuming tasks!)
        for module_name, module in self.modules.items():
            module_state = self.state.get_module_state(module_name)
            if module_state:
                try:
                    module.restore_state(module_state)
                    log.info("recovery.module_restored", module=module_name)
                except Exception as e:
                    log.error("recovery.module_restore_failed",
                             module=module_name, error=str(e))

        # Step 6: NOW resume tasks (modules are ready)
        await self._resume_tasks()

        log.info("recovery.complete",
                entries_replayed=len(entries),
                tasks_resumed=len([t for t in self.state.tasks.values()
                                   if t.state == TaskState.PAUSED]))

        return True

    async def _replay_entries(self, entries: list[WALEntry]):
        """Replay WAL entries in sequence order."""
        for entry in entries:
            try:
                if entry.entry_type == "task_update":
                    self._apply_task_update(entry.data)
                elif entry.entry_type == "conversation_delta":
                    self._apply_conversation_delta(entry.data)
                elif entry.entry_type == "module_state":
                    self._apply_module_state(entry.data)
                elif entry.entry_type == "allocator_state":
                    self._apply_allocator_state(entry.data)
            except Exception as e:
                log.warning("recovery.entry_replay_failed",
                           sequence=entry.sequence,
                           type=entry.entry_type,
                           error=str(e))

    def _apply_conversation_delta(self, data: dict):
        """Apply conversation delta (append messages)."""
        task_id = data["task_id"]
        new_messages = data["messages"]

        task = self.state.tasks.get(task_id)
        if task and task.conversation_state:
            existing = task.conversation_state.get("messages", [])
            task.conversation_state["messages"] = existing + new_messages

    def _fix_interrupted_tasks(self) -> int:
        """Mark RUNNING tasks as PAUSED (they were interrupted by crash)."""
        count = 0
        for task in self.state.tasks.values():
            if task.state == TaskState.RUNNING:
                task.state = TaskState.PAUSED
                task.updated_at = datetime.now()
                count += 1
                log.info("recovery.task_paused", task_id=task.id)
        return count

    async def _resume_tasks(self):
        """Resubmit paused tasks to scheduler."""
        for task in self.state.tasks.values():
            if task.state == TaskState.PAUSED:
                await self.scheduler.submit(task)
```

---

## Part 15: Allocator as Single Gate

The allocator becomes the ONLY place that decides if a task can execute.

```python
class Allocator:
    """
    Central gate for LLM allocation decisions.

    All quota checking, availability checking, and deep mode decisions
    go through here. Scheduler and other components do NOT duplicate this logic.
    """

    def __init__(self, gateway: "LLMGateway", pool_client: "PoolClient"):
        self.gateway = gateway
        self.pool = pool_client  # Source of truth for quotas

        # Cache quota status (refreshed periodically)
        self._quota_cache: dict[str, LLMQuota] = {}
        self._cache_ttl = timedelta(seconds=30)
        self._last_cache_refresh = datetime.min

    async def can_execute(
        self,
        task: Task,
        available_backends: list[str]
    ) -> bool:
        """
        Single decision point: can this task execute now?

        Checks:
        1. Required backend available (or any backend if no preference)
        2. Quota available for the request type
        3. Deep mode available if required
        """
        await self._maybe_refresh_cache()

        # Determine candidate backends
        if task.llm_preference:
            candidates = [task.llm_preference] if task.llm_preference in available_backends else []
        else:
            candidates = available_backends

        if not candidates:
            return False

        # Check at least one candidate has quota
        for backend in candidates:
            quota = self._quota_cache.get(backend)
            if quota is None:
                continue

            # Check hourly quota
            if quota.hourly_used >= quota.hourly_limit * 0.9:
                continue

            # Check deep mode if required
            if task.requires_deep_mode:
                if quota.deep_mode_used >= quota.deep_mode_limit:
                    continue

            return True

        return False

    def allocate(self, task: Task) -> Optional[str]:
        """
        Allocate a backend for the task.

        Returns backend name or None if none available.
        Called AFTER can_execute() returns True.
        """
        available = self.gateway.get_available_backends()

        if task.llm_preference:
            if task.llm_preference in available:
                return task.llm_preference
            return None

        # Prefer backend with most remaining quota
        best = None
        best_remaining = -1

        for backend in available:
            quota = self._quota_cache.get(backend)
            if quota:
                remaining = quota.hourly_limit - quota.hourly_used
                if remaining > best_remaining:
                    best = backend
                    best_remaining = remaining

        return best

    def record_usage(self, backend: str, deep_mode: bool = False):
        """Record usage after successful request."""
        # Update local cache
        quota = self._quota_cache.get(backend)
        if quota:
            quota.hourly_used += 1
            if deep_mode:
                quota.deep_mode_used += 1

        # Also update pool (source of truth)
        # This is fire-and-forget since pool persists its own state
        asyncio.create_task(self._update_pool_usage(backend, deep_mode))

    async def _maybe_refresh_cache(self):
        """Refresh quota cache from pool if stale."""
        if datetime.now() - self._last_cache_refresh < self._cache_ttl:
            return

        try:
            status = await self.pool.get_status()
            for backend, info in status.backends.items():
                self._quota_cache[backend] = LLMQuota(
                    hourly_limit=info.get("hourly_limit", 50),
                    hourly_used=info.get("hourly_used", 0),
                    deep_mode_limit=info.get("deep_mode_limit", 20),
                    deep_mode_used=info.get("deep_mode_used", 0),
                )
            self._last_cache_refresh = datetime.now()
        except Exception as e:
            log.warning("allocator.cache_refresh_failed", error=str(e))
```

---

## Part 16: Module Interface Updates

Modules must return stable task keys, not generate new UUIDs.

```python
class ExplorerModule(ModuleInterface):
    """Explorer module with stable task keys."""

    def get_pending_tasks(self) -> list[Task]:
        """
        Return tasks this module wants to run.

        CRITICAL: Use Task.create() which auto-generates stable task_key.
        Do NOT generate random IDs!
        """
        tasks = []

        # Check for seeds needing exploration
        for seed in self.seed_queue:
            if not self._has_active_task_for_seed(seed.id):
                tasks.append(Task.create(
                    module="explorer",
                    task_type="exploration",
                    context={"seed_id": seed.id},  # Stable context
                    priority=BASE_PRIORITIES["exploration"] + seed.priority * 10,
                ))

        # Check for insights needing review
        for insight in self.pending_review:
            tasks.append(Task.create(
                module="explorer",
                task_type="review",
                context={"insight_id": insight.id},  # Stable context
                priority=BASE_PRIORITIES["review"],
            ))

        return tasks

    def _has_active_task_for_seed(self, seed_id: str) -> bool:
        """Check if there's already an active task for this seed."""
        task_key = Task.generate_task_key(
            "explorer", "exploration", {"seed_id": seed_id}
        )
        return task_key in self.state.active_task_keys

    async def handle_failure(self, task: Task, error: str):
        """
        Handle task failure with proper cleanup.

        This is called by TaskExecutor on any exception.
        """
        log.error("explorer.task_failed",
                 task_id=task.id,
                 task_type=task.task_type,
                 error=error)

        if task.task_type == "exploration":
            # Mark thread as needing attention
            thread_id = task.context.get("thread_id")
            if thread_id:
                self.thread_manager.mark_needs_review(thread_id)

        elif task.task_type == "review":
            # Return insight to pending queue
            insight_id = task.context.get("insight_id")
            if insight_id:
                self.pending_review.add(insight_id)
```

---

## Part 17: Implementation Checklist

### Critical (Must Fix Before Production)

| ID | Fix | Component | Effort |
|----|-----|-----------|--------|
| 1 | WAL recovery with sequence filtering | WALManager | 2h |
| 2 | Delta logging for conversations | WALManager, TaskExecutor | 2h |
| E | Head+tail context truncation | TaskExecutor | 30m |
| 2 | Stable task keys | Task model, Scheduler | 2h |
| 4 | Per-backend locks | LLMGateway | 1h |

### High Priority (Fix in First Sprint)

| ID | Fix | Component | Effort |
|----|-----|-----------|--------|
| A | Offload large JSON to executor | WALManager | 1h |
| D | Lazy priority recomputation | Scheduler | 1h |
| G | PID lifecycle manager | LLMGateway | 1h |
| 5 | Allocator as single gate | Allocator, Scheduler | 2h |
| 6 | WAL rotation on checkpoint | WALManager, StateManager | 1h |
| 7 | Correct recovery order | RecoveryManager | 30m |
| 8 | Persist state corrections | RecoveryManager | 30m |
| 12 | Running summary for context | TaskExecutor | 2h |

### Medium Priority (Fix in Second Sprint)

| ID | Fix | Component | Effort |
|----|-----|-----------|--------|
| C | Query pool for quota (not duplicate) | Allocator | 1h |
| 9 | Call module.handle_failure | TaskExecutor | 30m |
| 10 | Define or remove peek_next | Scheduler | 30m |
| 11 | Remove READY state | Task model | 15m |
| 3 | Document concurrency model clearly | Design doc | 30m |

---

## Part 18: Concurrency Model Clarification

**Model: Single-threaded async with disciplined locking**

The orchestrator runs in a single Python process using `asyncio`. This is NOT an actor model. Instead:

1. **StateManager** is the sole owner of global state (tasks, module states)
2. All state modifications go through async methods with locks
3. CPU-bound operations (JSON serialization) are offloaded to executor
4. Multiple LLM requests can proceed in parallel (per-backend locks)
5. Deduplication uses async LLM callbacks, not CPU-intensive processing

```
┌──────────────────────────────────────────────────────────────┐
│                    ASYNCIO EVENT LOOP                         │
│                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │
│  │  Scheduler  │  │  Executor   │  │  Gateway (locks)    │   │
│  │  (lock)     │  │  (lock)     │  │  gemini  │ chatgpt  │   │
│  └──────┬──────┘  └──────┬──────┘  └────┬─────┴────┬─────┘   │
│         │                │              │          │          │
│         └────────────────┼──────────────┼──────────┘          │
│                          │              │                     │
│                          ▼              ▼                     │
│                   ┌─────────────────────────┐                 │
│                   │     StateManager        │                 │
│                   │     (single lock)       │                 │
│                   │     - tasks             │                 │
│                   │     - module_states     │                 │
│                   │     - metrics           │                 │
│                   └─────────────────────────┘                 │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                ThreadPoolExecutor                        │ │
│  │  (for CPU-bound: JSON serialize, file I/O)              │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

**Key invariants:**
- Only one task RUNNING per backend at a time (enforced by gateway locks)
- StateManager lock protects all task state mutations
- WAL appends are serialized but non-blocking (use executor for large data)
- Module `get_pending_tasks()` must be idempotent (same result for same state)
