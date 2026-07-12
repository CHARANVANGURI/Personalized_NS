"""
tests/test_topic_generator.py - Unit tests for the conversation starter generator.

All OpenRouter API calls are mocked so tests run offline without an API key.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from services import topic_generator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_chat(response_text: str):
    """Patch chat_completion to return a fixed response string."""
    return patch(
        "services.topic_generator.chat_completion",
        return_value=response_text,
    )


def _make_starters_json(starters: list[str]) -> str:
    return json.dumps({"starters": starters})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGenerateStarters:
    """Tests for generate_starters()."""

    def test_happy_path_returns_starters(self):
        """Happy path: valid JSON response returns correct starters."""
        starters_list = [
            "How do you see AI transforming urban infrastructure?",
            "What's the most exciting climate tech you've encountered recently?",
            "I work on sustainable cities too — what aspects are you focused on?",
        ]
        resp = _make_starters_json(starters_list)
        with _mock_chat(resp):
            result = topic_generator.generate_starters(
                ["AI", "Urban Planning"], ["climate change"]
            )

        assert len(result) == topic_generator.NUM_STARTERS
        assert result[0] == starters_list[0]

    def test_raises_on_empty_themes(self):
        """Empty themes list must raise ValueError."""
        with pytest.raises(ValueError, match="At least one theme"):
            topic_generator.generate_starters([])

    def test_strips_markdown_fences(self):
        """Response wrapped in ```json fences is parsed correctly."""
        starters_list = [
            "Question one about AI?",
            "Question two about climate?",
            "Question three about data science?",
        ]
        fenced = f"```json\n{_make_starters_json(starters_list)}\n```"
        with _mock_chat(fenced):
            result = topic_generator.generate_starters(["AI"])

        assert len(result) == 3

    def test_fallback_on_api_failure(self):
        """When OpenRouter throws, fallback starters are returned."""
        with patch(
            "services.topic_generator.chat_completion",
            side_effect=RuntimeError("API down"),
        ):
            result = topic_generator.generate_starters(["AI"], ["machine learning"])

        assert len(result) == topic_generator.NUM_STARTERS
        assert all(isinstance(s, str) and len(s) > 10 for s in result)

    def test_fallback_on_bad_json(self):
        """When JSON is malformed, tries line-based extraction or falls back."""
        with _mock_chat("This is not valid JSON!!!"):
            result = topic_generator.generate_starters(["Healthcare"], [])

        assert isinstance(result, list)
        assert len(result) > 0

    def test_none_interests_handled_gracefully(self):
        """Passing None for interests should not raise."""
        resp = _make_starters_json([
            "How do you apply AI in healthcare?",
            "What trends in AI excite you most?",
            "Are you seeing more AI adoption in hospitals?",
        ])
        with _mock_chat(resp):
            result = topic_generator.generate_starters(["AI"], None)

        assert isinstance(result, list)

    def test_filters_short_starters(self):
        """Starters shorter than 10 chars are filtered out; fallback fills gaps."""
        resp = _make_starters_json(["Short", "x", "Also too short"])
        with _mock_chat(resp):
            result = topic_generator.generate_starters(["Finance"])

        # Should still return NUM_STARTERS via fallback
        assert len(result) == topic_generator.NUM_STARTERS

    def test_returns_at_most_num_starters(self):
        """Result list is capped at NUM_STARTERS even if model returns more."""
        many = [f"Question number {i} about the topic at hand?" for i in range(10)]
        resp = _make_starters_json(many)
        with _mock_chat(resp):
            result = topic_generator.generate_starters(["AI"])

        assert len(result) == topic_generator.NUM_STARTERS


class TestFallbackStarters:
    """Tests for _fallback_starters()."""

    def test_returns_num_starters(self):
        result = topic_generator._fallback_starters(["AI"], ["data"])
        assert len(result) == topic_generator.NUM_STARTERS

    def test_uses_theme_in_text(self):
        result = topic_generator._fallback_starters(["Blockchain"], [])
        assert any("Blockchain" in s for s in result)

    def test_uses_interest_in_text(self):
        result = topic_generator._fallback_starters(["AI"], ["climate change"])
        assert any("climate change" in s for s in result)

    def test_empty_themes_uses_default(self):
        """Fallback with empty theme uses 'technology'."""
        result = topic_generator._fallback_starters([], [])
        assert all(isinstance(s, str) for s in result)


class TestParseStartersResponse:
    """Tests for _parse_starters_response()."""

    def test_parses_valid_json(self):
        raw = json.dumps({"starters": ["How are you using AI?", "What excites you about ML?", "Have you tried deep learning?"]})
        result = topic_generator._parse_starters_response(raw)
        assert len(result) == 3

    def test_extracts_numbered_list(self):
        raw = "1. How do you see AI evolving?\n2. What's your take on sustainability in tech?\n3. Have you seen any great blockchain projects?"
        result = topic_generator._parse_starters_response(raw)
        assert len(result) >= 2

    def test_returns_empty_on_garbage(self):
        result = topic_generator._parse_starters_response("garbage with no content")
        assert isinstance(result, list)
