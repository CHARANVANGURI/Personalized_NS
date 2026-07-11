"""
backend/routes.py - FastAPI Route Handlers

Defines all API endpoints. Business logic is delegated to service modules
to keep routes thin and focused on HTTP concerns only.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status

from backend.schemas import (
    AnalyzeEventRequest,
    AnalyzeEventResponse,
    FeedbackEntry,
    FeedbackRequest,
    FeedbackResponse,
    FactCheckResponse,
    GenerateConversationRequest,
    GenerateConversationResponse,
    HealthResponse,
    HistoryEntry,
    HistoryResponse,
)
from backend.config import API_VERSION
from services.event_analyzer import analyze_event
from services.topic_generator import generate_starters
from services.fact_checker import fact_check_topic
from services.history_logger import save_history, load_history
from services.feedback_logger import save_feedback, load_feedback

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Liveness probe",
)
async def health_check() -> HealthResponse:
    """Return API health status."""
    return HealthResponse(status="ok", version=API_VERSION)


# ---------------------------------------------------------------------------
# Event Analysis
# ---------------------------------------------------------------------------


@router.post(
    "/analyze-event",
    response_model=AnalyzeEventResponse,
    tags=["Analysis"],
    summary="Analyze an event description to extract themes",
    status_code=status.HTTP_200_OK,
)
async def analyze_event_route(body: AnalyzeEventRequest) -> AnalyzeEventResponse:
    """
    Run zero-shot classification on the event description and return
    the top extracted themes with their confidence scores.
    """
    logger.info("POST /analyze-event | event_description length=%d", len(body.event_description))
    try:
        result = analyze_event(body.event_description)
        return AnalyzeEventResponse(
            themes=result["themes"],
            scores=result["scores"],
        )
    except Exception as exc:
        logger.error("Event analysis failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Event analysis failed: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Conversation Generation
# ---------------------------------------------------------------------------


@router.post(
    "/generate-conversation",
    response_model=GenerateConversationResponse,
    tags=["Generation"],
    summary="Generate personalized conversation starters for a networking event",
    status_code=status.HTTP_201_CREATED,
)
async def generate_conversation_route(
    body: GenerateConversationRequest,
) -> GenerateConversationResponse:
    """
    1. Analyze the event description to extract themes (DistilBERT zero-shot).
    2. Generate 3 personalized networking conversation starters (GPT-2).
    3. Persist the result to history.json.
    4. Return the generated starters with metadata.
    """
    logger.info(
        "POST /generate-conversation | interests=%s", body.user_interests
    )
    try:
        # Step 1: Extract themes
        analysis = analyze_event(body.event_description)

        # Step 2: Generate starters
        starters = generate_starters(
            themes=analysis["themes"],
            user_interests=body.user_interests,
        )

        # Step 3: Persist
        conversation_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        entry = HistoryEntry(
            conversation_id=conversation_id,
            timestamp=now,
            event_description=body.event_description,
            themes=analysis["themes"],
            scores=analysis["scores"],
            starters=starters,
            user_interests=body.user_interests,
            favorite=False,
        )
        save_history(entry.model_dump(mode="json"))

        return GenerateConversationResponse(
            conversation_id=conversation_id,
            themes=analysis["themes"],
            scores=analysis["scores"],
            starters=starters,
            timestamp=now,
        )
    except Exception as exc:
        logger.error("Conversation generation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Conversation generation failed: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Fact Checking
# ---------------------------------------------------------------------------


@router.get(
    "/fact-check",
    response_model=FactCheckResponse,
    tags=["Fact Check"],
    summary="Verify a topic using Wikipedia",
    status_code=status.HTTP_200_OK,
)
async def fact_check_route(
    topic: str = Query(..., min_length=2, max_length=200, description="Topic to verify"),
) -> FactCheckResponse:
    """
    Search Wikipedia for the given topic, retrieve the article summary,
    and return a structured fact-check response.
    """
    logger.info("GET /fact-check | topic=%s", topic)
    try:
        result = fact_check_topic(topic)
        return FactCheckResponse(**result)
    except Exception as exc:
        logger.error("Fact check failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fact check failed: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


@router.get(
    "/history",
    response_model=HistoryResponse,
    tags=["History"],
    summary="Retrieve conversation history",
    status_code=status.HTTP_200_OK,
)
async def get_history_route(
    sort: str = Query(default="newest", description="Sort order: 'newest' or 'oldest'"),
    favorites_only: bool = Query(default=False, description="Filter to favorites only"),
) -> HistoryResponse:
    """Return stored conversation history sorted by timestamp."""
    logger.info("GET /history | sort=%s favorites_only=%s", sort, favorites_only)
    try:
        raw = load_history()
        if favorites_only:
            raw = [r for r in raw if r.get("favorite", False)]

        reverse = sort != "oldest"
        raw.sort(key=lambda x: x.get("timestamp", ""), reverse=reverse)

        entries = [HistoryEntry(**r) for r in raw]
        return HistoryResponse(total=len(entries), entries=entries)
    except Exception as exc:
        logger.error("History retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"History retrieval failed: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------


@router.post(
    "/feedback",
    response_model=FeedbackEntry,
    tags=["Feedback"],
    summary="Submit feedback for a conversation",
    status_code=status.HTTP_201_CREATED,
)
async def post_feedback_route(body: FeedbackRequest) -> FeedbackEntry:
    """Store thumbs-up/down and optional comment for a conversation."""
    logger.info("POST /feedback | conversation_id=%s", body.conversation_id)
    if body.thumbs_up and body.thumbs_down:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="thumbs_up and thumbs_down cannot both be true.",
        )
    try:
        feedback_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        entry = FeedbackEntry(
            feedback_id=feedback_id,
            conversation_id=body.conversation_id,
            thumbs_up=body.thumbs_up,
            thumbs_down=body.thumbs_down,
            comment=body.comment,
            timestamp=now,
        )
        save_feedback(entry.model_dump(mode="json"))
        return entry
    except Exception as exc:
        logger.error("Feedback save failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feedback save failed: {exc}",
        ) from exc


@router.get(
    "/feedback",
    response_model=FeedbackResponse,
    tags=["Feedback"],
    summary="Retrieve all feedback",
    status_code=status.HTTP_200_OK,
)
async def get_feedback_route(
    conversation_id: Optional[str] = Query(
        default=None, description="Filter by conversation ID"
    ),
) -> FeedbackResponse:
    """Return all stored feedback, optionally filtered by conversation ID."""
    logger.info("GET /feedback | conversation_id=%s", conversation_id)
    try:
        raw = load_feedback()
        if conversation_id:
            raw = [r for r in raw if r.get("conversation_id") == conversation_id]
        raw.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        entries = [FeedbackEntry(**r) for r in raw]
        return FeedbackResponse(total=len(entries), entries=entries)
    except Exception as exc:
        logger.error("Feedback retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feedback retrieval failed: {exc}",
        ) from exc
