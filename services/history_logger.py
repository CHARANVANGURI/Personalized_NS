"""
services/history_logger.py - Conversation History Persistence

Provides functions to save and load conversation history to/from a local
JSON file. Thread-safe via file-level atomic writes using a temp-then-rename
strategy. All timestamps are stored in ISO 8601 format (UTC).
"""

from __future__ import annotations

import json
import logging
import tempfile
import os
from pathlib import Path
from typing import Any

from backend.config import HISTORY_FILE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_history(entry: dict[str, Any]) -> None:
    """
    Append a new history entry to the JSON store.

    Parameters
    ----------
    entry : dict
        A serializable dictionary representing a conversation history entry.
        Expected keys: conversation_id, timestamp, event_description,
        themes, scores, starters, user_interests, favorite.

    Raises
    ------
    IOError
        If the file cannot be written.
    """
    logger.debug("Saving history entry: %s", entry.get("conversation_id"))
    records = _read_json(HISTORY_FILE)
    records.append(entry)
    _write_json(HISTORY_FILE, records)
    logger.info("History saved. Total records: %d", len(records))


def load_history() -> list[dict[str, Any]]:
    """
    Load all conversation history records.

    Returns
    -------
    list[dict]
        All stored history entries (may be empty).
    """
    records = _read_json(HISTORY_FILE)
    logger.debug("Loaded %d history records.", len(records))
    return records


def update_history_favorite(conversation_id: str, favorite: bool) -> bool:
    """
    Toggle the favorite flag on a specific history entry.

    Parameters
    ----------
    conversation_id : str
        The ID of the conversation to update.
    favorite : bool
        New favorite value.

    Returns
    -------
    bool
        True if the record was found and updated, False otherwise.
    """
    records = _read_json(HISTORY_FILE)
    updated = False
    for record in records:
        if record.get("conversation_id") == conversation_id:
            record["favorite"] = favorite
            updated = True
            break

    if updated:
        _write_json(HISTORY_FILE, records)
        logger.info("Favorite updated for conversation_id=%s", conversation_id)
    else:
        logger.warning("conversation_id=%s not found in history.", conversation_id)

    return updated


# ---------------------------------------------------------------------------
# Private Helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> list[dict[str, Any]]:
    """Read and parse JSON array from file; return empty list if missing."""
    try:
        if path.exists() and path.stat().st_size > 0:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read %s: %s – starting fresh.", path, exc)
    return []


def _write_json(path: Path, data: list[dict[str, Any]]) -> None:
    """
    Atomically write JSON data to a file using a temp file + rename.

    Parameters
    ----------
    path : Path
        Target file path.
    data : list[dict]
        Data to serialize.
    """
    dir_path = path.parent
    dir_path.mkdir(parents=True, exist_ok=True)

    try:
        fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)
            os.replace(tmp_path, path)
        except Exception:
            os.unlink(tmp_path)
            raise
    except OSError as exc:
        logger.error("Failed to write %s: %s", path, exc)
        raise IOError(f"Cannot write to {path}: {exc}") from exc
