"""
services/fact_checker.py - Wikipedia Fact Verification Service

Queries the Wikipedia API to retrieve a concise summary of a given topic.
Handles disambiguation, missing pages, and network failures gracefully.
"""

from __future__ import annotations

import logging
from typing import Any

import wikipediaapi

from backend.config import WIKIPEDIA_LANGUAGE, WIKIPEDIA_USER_AGENT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wikipedia Client (singleton)
# ---------------------------------------------------------------------------

_wiki: Any = None


def _get_wiki() -> Any:
    """
    Lazily initialize and cache the Wikipedia API client.

    Returns
    -------
    wikipediaapi.Wikipedia
        Configured Wikipedia client.
    """
    global _wiki
    if _wiki is None:
        _wiki = wikipediaapi.Wikipedia(
            language=WIKIPEDIA_LANGUAGE,
            user_agent=WIKIPEDIA_USER_AGENT,
        )
        logger.info("Wikipedia client initialized (lang=%s).", WIKIPEDIA_LANGUAGE)
    return _wiki


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fact_check_topic(topic: str) -> dict[str, Any]:
    """
    Search Wikipedia for the given topic and return a structured summary.

    Parameters
    ----------
    topic : str
        The topic string to look up (e.g., "Blockchain in Healthcare").

    Returns
    -------
    dict
        {
            "topic": str,       # original query
            "title": str,       # Wikipedia page title (or "N/A")
            "summary": str,     # first paragraph summary
            "verified": bool,   # True if a page was found
            "url": str | None   # Wikipedia URL if found
        }

    Notes
    -----
    - Disambiguation pages are detected and handled.
    - Network failures return ``verified=False`` with a descriptive message.
    """
    if not topic or not topic.strip():
        return _not_found(topic, "Topic must not be empty.")

    topic = topic.strip()
    logger.info("Fact-checking topic: %s", topic)

    try:
        wiki = _get_wiki()
        page = wiki.page(topic)

        if not page.exists():
            logger.warning("Wikipedia page not found for: %s", topic)
            return _not_found(topic, "No reliable reference available.")

        # Detect disambiguation pages
        if _is_disambiguation(page):
            logger.warning("Disambiguation page for topic: %s", topic)
            return _not_found(
                topic,
                f"'{topic}' is a disambiguation term. Please be more specific.",
            )

        summary = _extract_summary(page.summary)
        logger.info("Found Wikipedia page: %s", page.title)

        return {
            "topic": topic,
            "title": page.title,
            "summary": summary,
            "verified": True,
            "url": page.fullurl,
        }

    except Exception as exc:
        logger.error("Wikipedia lookup failed for '%s': %s", topic, exc, exc_info=True)
        return _not_found(topic, f"Lookup failed due to a network error: {exc}")


# ---------------------------------------------------------------------------
# Private Helpers
# ---------------------------------------------------------------------------


def _not_found(topic: str, message: str) -> dict[str, Any]:
    """Return a standard 'not found' response dict."""
    return {
        "topic": topic,
        "title": "N/A",
        "summary": message,
        "verified": False,
        "url": None,
    }


def _is_disambiguation(page: Any) -> bool:
    """
    Heuristically detect Wikipedia disambiguation pages.

    Parameters
    ----------
    page : wikipediaapi.WikipediaPage

    Returns
    -------
    bool
    """
    categories = [c.lower() for c in page.categories.keys()]
    return any("disambiguation" in c for c in categories)


def _extract_summary(full_summary: str, max_sentences: int = 3) -> str:
    """
    Extract a concise summary from the full Wikipedia article text.

    Parameters
    ----------
    full_summary : str
        The full summary text from the Wikipedia page.
    max_sentences : int
        Maximum number of sentences to return.

    Returns
    -------
    str
        Truncated summary.
    """
    if not full_summary:
        return "No summary available."

    sentences = full_summary.split(". ")
    trimmed = ". ".join(sentences[:max_sentences])
    if trimmed and not trimmed.endswith("."):
        trimmed += "."
    return trimmed
