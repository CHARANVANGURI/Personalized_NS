"""
services/topic_generator.py - Conversation Starter Generator

Uses the GPT-2 Small text-generation pipeline to produce personalized
networking conversation starters based on extracted event themes and
user-provided interests.

The pipeline is initialized once at module import time (singleton pattern)
to avoid repeated model loading overhead across API requests.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from transformers import pipeline

from backend.config import (
    MAX_NEW_TOKENS,
    NUM_STARTERS,
    TEMPERATURE,
    TEXT_GEN_MODEL,
    TOP_P,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model Loading (singleton)
# ---------------------------------------------------------------------------

_generator: Any = None


def _get_generator() -> Any:
    """
    Lazily load and cache the text-generation pipeline.

    Returns
    -------
    transformers.Pipeline
        Loaded text-generation pipeline.
    """
    global _generator
    if _generator is None:
        logger.info("Loading text-generation model: %s", TEXT_GEN_MODEL)
        _generator = pipeline(
            "text-generation",
            model=TEXT_GEN_MODEL,
            pad_token_id=50256,  # EOS token used as pad for GPT-2
        )
        logger.info("Text-generation pipeline ready.")
    return _generator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Prompt templates to encourage diversity in generated starters
_PROMPT_TEMPLATES: list[str] = [
    "As someone interested in {interests}, at an event about {themes}, "
    "a great conversation starter would be: \"",
    "Networking question about {themes} for someone passionate about {interests}: \"",
    "At a {themes} conference, an engaging opening question related to {interests} is: \"",
]


def _build_prompt(themes: list[str], user_interests: list[str], template_idx: int) -> str:
    """
    Build a prompt string from themes and interests using a template.

    Parameters
    ----------
    themes : list[str]
        Top event themes.
    user_interests : list[str]
        User's personal interests.
    template_idx : int
        Index to select which prompt template to use.

    Returns
    -------
    str
        Formatted prompt string.
    """
    theme_str = ", ".join(themes[:3]) if themes else "technology"
    interest_str = ", ".join(user_interests[:3]) if user_interests else "innovation"
    template = _PROMPT_TEMPLATES[template_idx % len(_PROMPT_TEMPLATES)]
    return template.format(themes=theme_str, interests=interest_str)


def _clean_output(raw_text: str, prompt: str) -> str:
    """
    Extract and clean the generated continuation from the raw model output.

    Parameters
    ----------
    raw_text : str
        Full generated text (prompt + continuation).
    prompt : str
        The original prompt used for generation.

    Returns
    -------
    str
        Cleaned conversation starter sentence.
    """
    # Remove the prompt prefix
    continuation = raw_text[len(prompt):].strip()

    # Extract text up to the first closing quote or sentence boundary
    for delimiter in ['"', "?", "!", ".\n", "\n"]:
        idx = continuation.find(delimiter)
        if idx > 0:
            continuation = continuation[: idx + 1]
            break

    # Remove residual quotes and excess whitespace
    continuation = re.sub(r'"+', '"', continuation).strip()
    continuation = continuation.strip('"').strip()

    # Ensure it ends with proper punctuation
    if continuation and continuation[-1] not in ".!?":
        continuation += "?"

    return continuation if len(continuation) > 10 else ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_starters(
    themes: list[str],
    user_interests: list[str] | None = None,
) -> list[str]:
    """
    Generate personalized networking conversation starters.

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
        If the text-generation pipeline fails.
    """
    if not themes:
        raise ValueError("At least one theme is required to generate starters.")

    interests = user_interests or []
    gen = _get_generator()
    starters: list[str] = []
    seen: set[str] = set()

    attempts = 0
    max_attempts = NUM_STARTERS * 3  # Allow retries for duplicates

    while len(starters) < NUM_STARTERS and attempts < max_attempts:
        template_idx = attempts % len(_PROMPT_TEMPLATES)
        prompt = _build_prompt(themes, interests, template_idx)

        try:
            outputs = gen(
                prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                num_return_sequences=1,
            )
            raw_text = outputs[0]["generated_text"]
            cleaned = _clean_output(raw_text, prompt)

            if cleaned and cleaned not in seen:
                starters.append(cleaned)
                seen.add(cleaned)
                logger.debug("Generated starter: %s", cleaned)

        except Exception as exc:
            logger.warning("Generation attempt %d failed: %s", attempts, exc)

        attempts += 1

    # Fallback starters if generation didn't produce enough
    fallbacks = [
        f"How are you applying {themes[0]} in your current work?",
        f"What excites you most about the future of {themes[0]}?",
        f"Have you seen any interesting {themes[0]} projects recently?",
    ]
    while len(starters) < NUM_STARTERS:
        fb = fallbacks[len(starters) % len(fallbacks)]
        if fb not in seen:
            starters.append(fb)
            seen.add(fb)

    logger.info("Generated %d conversation starters.", len(starters))
    return starters[:NUM_STARTERS]
