"""Data models for the Browser Pool Service."""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Priority(str, Enum):
    """Request priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class Backend(str, Enum):
    """Available backends."""
    GEMINI = "gemini"
    CHATGPT = "chatgpt"
    CLAUDE = "claude"


class ImageAttachment(BaseModel):
    """An image attachment to include with a prompt."""
    filename: str  # Original filename
    data: str  # Base64-encoded image data
    media_type: str  # MIME type, e.g., "image/png", "image/jpeg"


class SendOptions(BaseModel):
    """Options for send request."""
    deep_mode: bool = False
    timeout_seconds: int = 300
    priority: Priority = Priority.NORMAL
    new_chat: bool = True


class SendRequest(BaseModel):
    """Request to send a prompt to an LLM."""
    backend: Backend
    prompt: str
    options: SendOptions = Field(default_factory=SendOptions)
    thread_id: Optional[str] = None  # For tracking/recovery purposes
    images: list[ImageAttachment] = Field(default_factory=list)  # Optional image attachments


class JobSubmitRequest(BaseModel):
    """Request to submit an async job."""
    backend: str
    prompt: str
    job_id: str
    thread_id: Optional[str] = None
    task_type: Optional[str] = None  # "exploration" or "critique" for recovery
    deep_mode: bool = False
    new_chat: bool = True
    priority: str = "normal"
    images: list[ImageAttachment] = Field(default_factory=list)  # Optional image attachments


class ResponseMetadata(BaseModel):
    """Metadata about an LLM response."""
    backend: str
    deep_mode_used: bool = False
    response_time_seconds: float
    session_id: Optional[str] = None


class SendResponse(BaseModel):
    """Response from send request."""
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None
    message: Optional[str] = None
    retry_after_seconds: Optional[int] = None
    metadata: Optional[ResponseMetadata] = None
    recovered: bool = False  # True if this response was recovered after a restart


class BackendStatus(BaseModel):
    """Status of a single backend."""
    available: bool
    authenticated: bool
    rate_limited: bool
    rate_limit_resets_at: Optional[datetime] = None
    queue_depth: int = 0
    deep_mode_uses_today: Optional[int] = None
    deep_mode_limit: Optional[int] = None
    pro_mode_uses_today: Optional[int] = None
    pro_mode_limit: Optional[int] = None


class PoolStatus(BaseModel):
    """Status of the entire pool."""
    gemini: Optional[BackendStatus] = None
    chatgpt: Optional[BackendStatus] = None
    claude: Optional[BackendStatus] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    uptime_seconds: float
    version: str


class AuthResponse(BaseModel):
    """Response from auth request."""
    success: bool
    message: str


# ==================== JIT ORCHESTRATOR API MODELS ====================


class SubmitImmediateRequest(BaseModel):
    """Request for JIT immediate submission (bypasses queue)."""
    backend: str
    prompt: str
    idempotency_token: str  # Format: "{task_id}:{attempt}" for deduplication
    thread_id: Optional[str] = None  # URL for thread navigation
    thread_title: Optional[str] = None  # Title for sidebar fallback
    deep_mode: bool = False
    images: list[ImageAttachment] = Field(default_factory=list)


class SubmitImmediateResponse(BaseModel):
    """Response from submit_immediate."""
    success: bool
    request_id: Optional[str] = None
    error: Optional[str] = None
    message: Optional[str] = None
    # If idempotency token matched existing request
    existing_request_id: Optional[str] = None
    # Captured after execution - for Orchestrator to save for thread navigation
    thread_url: Optional[str] = None
    thread_title: Optional[str] = None
    # The actual response content
    response: Optional[str] = None
    deep_mode_used: bool = False


class ActiveRequest(BaseModel):
    """Information about an active request (for recovery)."""
    request_id: str
    backend: str
    thread_id: Optional[str] = None
    thread_title: Optional[str] = None
    chat_url: Optional[str] = None
    deep_mode: bool = False
    started_at: float
    prompt_preview: Optional[str] = None


class BackendBusyResponse(BaseModel):
    """Response from is_backend_busy check."""
    busy: bool
    current_request_id: Optional[str] = None
    elapsed_seconds: Optional[float] = None
