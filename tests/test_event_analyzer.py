"""
tests/test_event_analyzer.py - Unit tests for the event analysis service.

All OpenRouter API calls are mocked so tests run offline without an API key.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from services import event_analyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_chat(response_text: str):
    """Patch chat_completion to return a fixed response string."""
    return patch(
        "services.event_analyzer.chat_completion",
        return_value=response_text,
    )


def _make_valid_response(themes: list[str], scores: list[float]) -> str:
    return json.dumps({"themes": themes, "scores": scores})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAnalyzeEvent:
    """Tests for analyze_event()."""

    def test_returns_themes_and_scores_from_valid_json(self):
        """Happy path: parses valid JSON response."""
        resp = _make_valid_response(
            ["AI", "Urban Planning", "Climate", "Sustainability", "Data Science"],
            [0.92, 0.85, 0.78, 0.65, 0.55],
        )
        with _mock_chat(resp):
            result = event_analyzer.analyze_event("AI for Sustainable Cities summit")

        assert "themes" in result
        assert "scores" in result
        assert result["themes"][0] == "AI"
        assert result["scores"][0] == pytest.approx(0.92, abs=1e-4)

    def test_returns_at_most_five_themes(self):
        """Only top-5 themes are returned."""
        resp = _make_valid_response(
            ["AI", "Climate", "Finance", "Healthcare", "Robotics", "Blockchain"],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        )
        with _mock_chat(resp):
            result = event_analyzer.analyze_event("General tech conference")

        assert len(result["themes"]) <= 5

    def test_strips_markdown_fences(self):
        """Response wrapped in ```json fences is parsed correctly."""
        payload = _make_valid_response(["Healthcare", "AI", "Data Science", "Finance", "Robotics"], [0.9, 0.8, 0.7, 0.6, 0.5])
        fenced = f"```json\n{payload}\n```"
        with _mock_chat(fenced):
            result = event_analyzer.analyze_event("Healthcare AI conference")

        assert "Healthcare" in result["themes"]

    def test_empty_description_raises_value_error(self):
        """Empty or whitespace-only input should raise ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            event_analyzer.analyze_event("   ")

    def test_invalid_json_falls_back_to_keywords(self):
        """If JSON is malformed, keyword fallback is used."""
        with _mock_chat("This is not JSON at all!"):
            result = event_analyzer.analyze_event(
                "A blockchain and cryptocurrency event"
            )

        assert isinstance(result["themes"], list)
        assert len(result["themes"]) > 0
        assert "Blockchain" in result["themes"]

    def test_unknown_themes_filtered_out(self):
        """Themes not in CANDIDATE_LABELS are filtered."""
        resp = _make_valid_response(
            ["AI", "Cooking", "Music", "Urban Planning", "Finance"],
            [0.9, 0.8, 0.75, 0.7, 0.65],
        )
        with _mock_chat(resp):
            result = event_analyzer.analyze_event("Mixed event")

        assert "Cooking" not in result["themes"]
        assert "Music" not in result["themes"]

    def test_api_failure_raises_runtime_error(self):
        """If OpenRouter throws, RuntimeError propagates."""
        with patch(
            "services.event_analyzer.chat_completion",
            side_effect=RuntimeError("OpenRouter error"),
        ):
            with pytest.raises(RuntimeError):
                event_analyzer.analyze_event("A valid event description text")

    def test_scores_are_float(self):
        """All scores should be floats."""
        resp = _make_valid_response(["AI", "Climate", "Finance", "Robotics", "Healthcare"], [0.9, 0.8, 0.7, 0.6, 0.5])
        with _mock_chat(resp):
            result = event_analyzer.analyze_event("AI Climate Finance summit")

        assert all(isinstance(s, float) for s in result["scores"])


class TestKeywordFallback:
    """Tests for the keyword fallback logic."""

    def test_detects_ai_keywords(self):
        result = event_analyzer._keyword_fallback("deep learning and neural networks summit")
        assert "AI" in result["themes"]

    def test_detects_blockchain_keywords(self):
        result = event_analyzer._keyword_fallback("decentralized crypto blockchain conference")
        assert "Blockchain" in result["themes"]

    def test_returns_default_on_no_match(self):
        result = event_analyzer._keyword_fallback("random words that match nothing specific")
        assert len(result["themes"]) > 0
