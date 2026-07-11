"""
backend/schemas.py - Pydantic Request & Response Models

Defines all data contracts used by the FastAPI routes.
Strict typing ensures input validation and clear API documentation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Shared / Common
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Liveness probe response."""

    status: str = "ok"
    version: str


# ---------------------------------------------------------------------------
# Event Analysis
# ---------------------------------------------------------------------------


class AnalyzeEventRequest(BaseModel):
    """Request body for /analyze-event."""

    event_description: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="A short description of the networking event.",
        examples=["AI for Sustainable Cities summit bringing together tech leaders."],
    )

    @field_validator("event_description")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class AnalyzeEventResponse(BaseModel):
    """Response for /analyze-event."""

    themes: list[str] = Field(..., description="Top extracted event themes.")
    scores: list[float] = Field(..., description="Confidence scores per theme.")


# ---------------------------------------------------------------------------
# Conversation Generation
# ---------------------------------------------------------------------------


class GenerateConversationRequest(BaseModel):
    """Request body for /generate-conversation."""

    event_description: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Networking event description.",
    )
    user_interests: list[str] = Field(
        default=[],
        description="List of personal interests to personalize starters.",
        examples=[["climate change", "urban planning"]],
    )

    @field_validator("event_description")
    @classmethod
    def strip_event(cls, v: str) -> str:
        return v.strip()

    @field_validator("user_interests")
    @classmethod
    def clean_interests(cls, v: list[str]) -> list[str]:
        return [i.strip() for i in v if i.strip()]


class GenerateConversationResponse(BaseModel):
    """Response for /generate-conversation."""

    conversation_id: str = Field(..., description="Unique ID for this generation.")
    themes: list[str]
    scores: list[float]
    starters: list[str] = Field(..., description="Generated conversation starters.")
    timestamp: datetime


# ---------------------------------------------------------------------------
# Fact Checking
# ---------------------------------------------------------------------------


class FactCheckResponse(BaseModel):
    """Response for /fact-check."""

    topic: str
    title: str
    summary: str
    verified: bool
    url: Optional[str] = None


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


class HistoryEntry(BaseModel):
    """A single record stored in history.json."""

    conversation_id: str
    timestamp: datetime
    event_description: str
    themes: list[str]
    scores: list[float]
    starters: list[str]
    user_interests: list[str] = []
    favorite: bool = False


class HistoryResponse(BaseModel):
    """Response for GET /history."""

    total: int
    entries: list[HistoryEntry]


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------


class FeedbackRequest(BaseModel):
    """Request body for POST /feedback."""

    conversation_id: str = Field(..., description="ID of the conversation to rate.")
    thumbs_up: bool = Field(default=False)
    thumbs_down: bool = Field(default=False)
    comment: Optional[str] = Field(default=None, max_length=500)

    @field_validator("thumbs_up", "thumbs_down")
    @classmethod
    def not_both(cls, v: bool) -> bool:
        return v


class FeedbackEntry(BaseModel):
    """A single record stored in feedback.json."""

    feedback_id: str
    conversation_id: str
    thumbs_up: bool
    thumbs_down: bool
    comment: Optional[str]
    timestamp: datetime


class FeedbackResponse(BaseModel):
    """Response for GET /feedback."""

    total: int
    entries: list[FeedbackEntry]
