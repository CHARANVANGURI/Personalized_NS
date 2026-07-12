"""
services/topic_generator.py - Conversation Starter Generator

Uses the OpenRouter LLM API to produce personalized networking
conversation starters based on extracted event themes and
user-provided interests.

Replaces the previous GPT-2 local pipeline for faster, higher-quality output.
"""

from __future__ import annotations

import json
import logging
import re

from backend.config import NUM_STARTERS, TEMPERATURE
from services.openrouter_client import chat_completion

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a professional networking coach who creates engaging, natural, "
    "and personalized conversation starters for professional networking events. "
    "Your starters are thoughtful, open-ended, and encourage meaningful dialogue. "
    "Always respond with valid JSON only — no markdown, no explanations."
)

_USER_PROMPT_TEMPLATE = """\
Generate exactly {num} unique, engaging networking conversation starters for the following context:

Event Themes: {themes}
User's Personal Interests: {interests}

Requirements:
- Each starter must be a single, natural-sounding open-ended question or statement
- Personalize them using the user's interests when possible
- Make them specific to the event themes, not generic
- Vary the style: some can be questions, some can be observations that invite discussion
- Each starter should be 15–40 words long
- Do NOT repeat similar starters

Return ONLY a JSON object in this exact format:
{{
  "starters": [
    "Starter 1 text here?",
    "Starter 2 text here.",
    "Starter 3 text here?"
  ]
}}
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_starters(
    themes: list[str],
    user_interests: list[str] | None = None,
) -> list[str]:
    """
    Generate personalized networking conversation starters via OpenRouter.

    Parameters
    ----------
    themes : list[str]
        Event themes extracted by the event analyzer.
    user_interests : list[str], optional
        User-specified personal interests for personalization.

    Returns
    -------
    list[str]
        A list of ``NUM_STARTERS`` unique conversation starters.

    Raises
    ------
    ValueError
        If no themes are provided.
    RuntimeError
        If the API call fails and fallback also fails.
    """
    if not themes:
        raise ValueError("At least one theme is required to generate starters.")

    interests = user_interests or []
    theme_str = ", ".join(themes[:5])
    interest_str = ", ".join(interests[:5]) if interests else "professional innovation and technology"

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        num=NUM_STARTERS,
        themes=theme_str,
        interests=interest_str,
    )

    logger.info(
        "Generating %d conversation starters via OpenRouter (themes=%s)...",
        NUM_STARTERS,
        themes[:3],
    )

    try:
        raw_response = chat_completion(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=TEMPERATURE,
            max_tokens=600,
        )
        starters = _parse_starters_response(raw_response)

        if len(starters) >= NUM_STARTERS:
            logger.info("Generated %d starters successfully.", len(starters))
            return starters[:NUM_STARTERS]
        elif starters:
            # We have some valid starters, but not enough. Pad with fallbacks.
            logger.warning("Only got %d valid starters, padding with fallbacks.", len(starters))
            fallbacks = _fallback_starters(themes, interests)
            for fb in fallbacks:
                if fb not in starters:
                    starters.append(fb)
                if len(starters) == NUM_STARTERS:
                    break
            return starters

    except Exception as exc:
        logger.warning("Primary generation failed: %s – using fallback.", exc)

    # Fallback starters
    return _fallback_starters(themes, interests)


# ---------------------------------------------------------------------------
# Private Helpers
# ---------------------------------------------------------------------------


def _parse_starters_response(raw: str) -> list[str]:
    """
    Parse the JSON response from the LLM and extract conversation starters.

    Parameters
    ----------
    raw : str
        Raw string response from the model.

    Returns
    -------
    list[str]
        Parsed list of conversation starters.
    """
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

    try:
        data = json.loads(cleaned)
        starters = data.get("starters", [])
        if isinstance(starters, list) and starters:
            # Filter out empty/too-short starters
            valid = [s.strip() for s in starters if isinstance(s, str) and len(s.strip()) > 10]
            return valid
    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("Starters JSON parse error: %s | raw=%.200s", exc, raw)

    # Try to extract bullet/numbered list as fallback
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    starters = []
    for line in lines:
        # Remove numbering/bullets
        cleaned_line = re.sub(r"^[\d\.\-\*\•]+\s*", "", line).strip().strip('"')
        if len(cleaned_line) > 20:
            starters.append(cleaned_line)
    return starters


def _fallback_starters(themes: list[str], interests: list[str]) -> list[str]:
    """Generate template-based fallback starters when the API fails."""
    theme = themes[0] if themes else "technology"
    interest = interests[0] if interests else "innovation"

    fallbacks = [
        f"How are you currently applying {theme} in your work, and what challenges have you encountered?",
        f"As someone passionate about {interest}, what excites you most about where {theme} is heading?",
        f"Have you seen any {theme} projects recently that you think are going to make a real difference?",
    ]
    return fallbacks[:NUM_STARTERS]
