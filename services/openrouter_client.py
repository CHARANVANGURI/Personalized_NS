"""
services/openrouter_client.py - Shared OpenRouter API Client

Provides a singleton OpenAI-compatible client pointed at OpenRouter.
All service modules (event_analyzer, topic_generator) use this shared
client to avoid creating multiple HTTP connection pools.
"""

from __future__ import annotations

import logging
from typing import Any

from openai import OpenAI

from backend.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    YOUR_SITE_NAME,
    YOUR_SITE_URL,
)

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """
    Return (or create) the singleton OpenRouter client.

    The client is OpenAI SDK-compatible and configured to route requests
    through OpenRouter's unified API gateway.

    Returns
    -------
    OpenAI
        Configured OpenAI client pointing at OpenRouter.

    Raises
    ------
    RuntimeError
        If OPENROUTER_API_KEY is not set in the environment.
    """
    global _client
    if _client is None:
        if not OPENROUTER_API_KEY:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. "
                "Copy .env.example to .env and add your key from https://openrouter.ai/keys"
            )
        _client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": YOUR_SITE_URL,
                "X-Title": YOUR_SITE_NAME,
            },
        )
        logger.info("OpenRouter client initialized (model base: %s).", OPENROUTER_BASE_URL)
    return _client


def chat_completion(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    temperature: float = 0.8,
    max_tokens: int = 500,
) -> str:
    """
    Send a chat completion request to OpenRouter and return the response text.

    Parameters
    ----------
    system_prompt : str
        Instruction/persona for the model.
    user_prompt : str
        User-facing input prompt.
    model : str, optional
        OpenRouter model ID. Defaults to OPENROUTER_MODEL from config.
    temperature : float
        Sampling temperature (0=deterministic, 1=creative).
    max_tokens : int
        Maximum tokens in the response.

    Returns
    -------
    str
        The model's response text.

    Raises
    ------
    RuntimeError
        On API failure.
    """
    from backend.config import OPENROUTER_MODEL, MAX_TOKENS, TEMPERATURE

    _model = model or OPENROUTER_MODEL
    _temperature = temperature
    _max_tokens = max_tokens or MAX_TOKENS

    client = get_client()

    try:
        response = client.chat.completions.create(
            model=_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=_temperature,
            max_tokens=_max_tokens,
        )
        content = response.choices[0].message.content or ""
        logger.debug("OpenRouter response (%s): %.120s...", _model, content)
        return content.strip()
    except Exception as exc:
        logger.error("OpenRouter API error: %s", exc, exc_info=True)
        raise RuntimeError(f"OpenRouter API call failed: {exc}") from exc
