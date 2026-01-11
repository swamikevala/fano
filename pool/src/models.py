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
