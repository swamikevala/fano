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
