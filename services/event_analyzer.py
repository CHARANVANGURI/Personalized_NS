"""
services/event_analyzer.py - Event Theme Extractor

Uses a zero-shot classification pipeline (DistilRoBERTa-based NLI model)
to identify the most relevant themes in a networking event description.

The pipeline is initialized once at module import time and reused across
all requests to avoid repeated model loading overhead.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from transformers import pipeline

from backend.config import CANDIDATE_LABELS, ZERO_SHOT_MODEL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model Loading (singleton pattern via module-level variable)
# ---------------------------------------------------------------------------

_classifier: Any = None


def _get_classifier() -> Any:
    """
    Lazily load and cache the zero-shot classification pipeline.

    Returns
    -------
    transformers.Pipeline
        Loaded zero-shot classification pipeline.
    """
    global _classifier
    if _classifier is None:
        logger.info("Loading zero-shot classification model: %s", ZERO_SHOT_MODEL)
        _classifier = pipeline(
            "zero-shot-classification",
            model=ZERO_SHOT_MODEL,
        )
        logger.info("Zero-shot classifier ready.")
    return _classifier


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_event(event_description: str) -> dict[str, list]:
    """
    Classify the event description against predefined candidate labels
    and return the top themes with their confidence scores.

    Parameters
    ----------
    event_description : str
        Raw text describing a networking event.

    Returns
    -------
    dict
        {
            "themes": ["AI", "Urban Planning", ...],
            "scores": [0.92, 0.85, ...]
        }

    Raises
    ------
    ValueError
        If the event_description is empty.
    RuntimeError
        If the classification pipeline fails.
    """
    if not event_description or not event_description.strip():
        raise ValueError("event_description must not be empty.")

    logger.debug("Analyzing event: %.80s...", event_description)

    try:
        classifier = _get_classifier()
        result = classifier(
            event_description.strip(),
            candidate_labels=CANDIDATE_LABELS,
            multi_label=True,
        )
    except Exception as exc:
        logger.error("Zero-shot classification error: %s", exc, exc_info=True)
        raise RuntimeError(f"Event analysis pipeline failed: {exc}") from exc

    # Sort by score descending and return top-5 themes
    paired = sorted(
        zip(result["labels"], result["scores"]),
        key=lambda x: x[1],
        reverse=True,
    )
    top_themes = [label for label, _ in paired[:5]]
    top_scores = [round(score, 4) for _, score in paired[:5]]

    logger.info("Extracted themes: %s", top_themes)
    return {"themes": top_themes, "scores": top_scores}
