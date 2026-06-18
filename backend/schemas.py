"""
Pydantic request/response schemas.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ── Requests ──────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10_000)
    thread_id: str = Field(..., pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


# ── Responses ─────────────────────────────────────────────────────────────────

class ThreadResponse(BaseModel):
    thread_id: str
    created_at: datetime
    status: str

    class Config:
        from_attributes = True


class MessageResponse(BaseModel):
    id: str
    thread_id: str
    role: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True


class MessagesListResponse(BaseModel):
    messages: list[MessageResponse]
    has_more: bool
    thread_id: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str


class MetricsResponse(BaseModel):
    active_streams: int
    total_requests: int
    uptime_seconds: int
