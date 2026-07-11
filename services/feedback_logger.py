"""
services/feedback_logger.py - Feedback Persistence

Provides functions to save and retrieve user feedback (thumbs-up/down
and comments) for generated conversations. Uses atomic JSON writes for
data integrity. Mirrors the pattern from history_logger.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from backend.config import FEEDBACK_FILE
from services.history_logger import _read_json, _write_json  # shared helpers

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_feedback(entry: dict[str, Any]) -> None:
    """
    Append a new feedback entry to the JSON store.

    Parameters
    ----------
    entry : dict
        Serializable feedback record. Expected keys:
        feedback_id, conversation_id, thumbs_up, thumbs_down,
        comment, timestamp.

    Raises
    ------
    IOError
        If the file cannot be written.
    """
    logger.debug("Saving feedback for conversation_id=%s", entry.get("conversation_id"))
    records = _read_json(FEEDBACK_FILE)
    records.append(entry)
    _write_json(FEEDBACK_FILE, records)
    logger.info("Feedback saved. Total records: %d", len(records))


def load_feedback() -> list[dict[str, Any]]:
    """
    Load all feedback records.

    Returns
    -------
    list[dict]
        All stored feedback entries (may be empty).
    """
    records = _read_json(FEEDBACK_FILE)
    logger.debug("Loaded %d feedback records.", len(records))
    return records


def get_feedback_for_conversation(conversation_id: str) -> list[dict[str, Any]]:
    """
    Retrieve feedback records associated with a specific conversation.

    Parameters
    ----------
    conversation_id : str

    Returns
    -------
    list[dict]
        Feedback records matching the conversation ID.
    """
    return [
        r for r in load_feedback()
        if r.get("conversation_id") == conversation_id
    ]
