"""
services/event_analyzer.py - Event Theme Extractor

Uses the OpenRouter LLM API to extract and classify the most relevant
themes from a networking event description.

Returns top themes with confidence-like scores (inferred from ranked
ordering in the model's structured JSON response).
"""

from __future__ import annotations

import json
import logging
import re

from backend.config import CANDIDATE_LABELS
from services.openrouter_client import chat_completion

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an expert event classifier. Your task is to analyze event descriptions "
    "and identify the most relevant themes from a predefined list. "
    "Always respond with valid JSON only — no markdown, no explanations."
)

_USER_PROMPT_TEMPLATE = """\
Analyze the following networking event description and classify it against these candidate themes:
{labels}

Event description:
\"\"\"{description}\"\"\"

Return ONLY a JSON object in this exact format (top 5 themes, ordered by relevance):
{{
  "themes": ["Theme1", "Theme2", "Theme3", "Theme4", "Theme5"],
  "scores": [0.95, 0.87, 0.76, 0.65, 0.54]
}}

Rules:
- themes must be chosen ONLY from the candidate list above
- scores must be between 0.0 and 1.0, descending order
- return exactly 5 themes even if some are low relevance
- no explanation, just the JSON object
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_event(event_description: str) -> dict[str, list]:
    """
    Classify the event description against predefined candidate labels
    using the OpenRouter LLM and return the top themes with scores.

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
        If the LLM call fails or returns malformed JSON.
    """
    if not event_description or not event_description.strip():
        raise ValueError("event_description must not be empty.")

    logger.info("Analyzing event themes via OpenRouter...")

    labels_str = "\n".join(f"- {label}" for label in CANDIDATE_LABELS)
    user_prompt = _USER_PROMPT_TEMPLATE.format(
        labels=labels_str,
        description=event_description.strip(),
    )

    raw_response = chat_completion(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.2,   # low temperature for consistent classification
        max_tokens=200,
    )

    return _parse_analysis_response(raw_response, event_description)


# ---------------------------------------------------------------------------
# Private Helpers
# ---------------------------------------------------------------------------


def _parse_analysis_response(raw: str, original_description: str) -> dict[str, list]:
    """
    Parse the JSON response from the LLM and validate theme/score fields.

    Parameters
    ----------
    raw : str
        Raw string response from the model.
    original_description : str
        Used for fallback context if parsing fails.

    Returns
    -------
    dict with 'themes' and 'scores' keys.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

    try:
        data = json.loads(cleaned)
        themes = data.get("themes", [])
        scores = data.get("scores", [])

        # Validate themes are from candidate list (case-insensitive)
        valid_labels_lower = {l.lower(): l for l in CANDIDATE_LABELS}
        validated_themes = []
        for t in themes:
            canonical = valid_labels_lower.get(t.lower())
            if canonical:
                validated_themes.append(canonical)

        # If we got valid themes, return them
        if validated_themes:
            scores = [round(float(s), 4) for s in scores[:len(validated_themes)]]
            # Pad scores if needed
            while len(scores) < len(validated_themes):
                scores.append(round(max(0.1, scores[-1] - 0.05), 4) if scores else 0.5)

            logger.info("Extracted themes: %s", validated_themes[:5])
            return {
                "themes": validated_themes[:5],
                "scores": scores[:5],
            }

    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("JSON parse error in event analysis: %s | raw=%s", exc, raw[:200])

    # Fallback: keyword-based detection from the description
    logger.warning("Using keyword fallback for theme extraction.")
    return _keyword_fallback(original_description)


def _keyword_fallback(description: str) -> dict[str, list]:
    """Simple keyword-based theme detection as a last resort."""
    desc_lower = description.lower()
    keyword_map = {
        "AI": ["ai", "artificial intelligence", "machine learning", "deep learning", "neural"],
        "Sustainability": ["sustain", "green", "eco", "renewable", "circular"],
        "Climate": ["climate", "carbon", "emission", "warming", "environment"],
        "Healthcare": ["health", "medical", "hospital", "pharma", "biotech"],
        "Finance": ["finance", "fintech", "bank", "invest", "trading"],
        "Robotics": ["robot", "automation", "autonomous", "drone"],
        "Data Science": ["data", "analytics", "statistics", "insight"],
        "Urban Planning": ["urban", "city", "cities", "infrastructure", "planning"],
        "Blockchain": ["blockchain", "crypto", "web3", "decentralized", "token"],
        "Cybersecurity": ["security", "cyber", "hack", "privacy", "encryption"],
        "Startups": ["startup", "start-up", "venture", "founder"],
        "Entrepreneurship": ["entrepreneur", "business", "innovation", "disrupt"],
    }
    found: list[tuple[str, float]] = []
    for label, keywords in keyword_map.items():
        if any(kw in desc_lower for kw in keywords):
            found.append((label, 0.7))

    if not found:
        found = [("AI", 0.5), ("Startups", 0.4)]

    found = found[:5]
    themes = [f[0] for f in found]
    scores = [round(f[1] - i * 0.05, 4) for i, f in enumerate(found)]
    return {"themes": themes, "scores": scores}
