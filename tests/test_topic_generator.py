"""
tests/test_topic_generator.py - Unit tests for the conversation starter generator.

All GPT-2 pipeline calls are mocked to avoid GPU dependency in CI/CD.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from services import topic_generator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_generator():
    """Reset the module-level generator singleton before each test."""
    original = topic_generator._generator
    topic_generator._generator = None
    yield
    topic_generator._generator = original


def _make_mock_generator(generated_text: str):
    """Return a mock pipeline callable that produces the given text."""
    mock = MagicMock()
    mock.return_value = [{"generated_text": generated_text}]
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    """Tests for the _build_prompt helper."""

    def test_includes_themes(self):
        prompt = topic_generator._build_prompt(["AI", "Climate"], [], template_idx=0)
        assert "AI" in prompt

    def test_includes_interests(self):
        prompt = topic_generator._build_prompt(["AI"], ["climate change"], template_idx=0)
        assert "climate change" in prompt

    def test_defaults_when_no_themes(self):
        prompt = topic_generator._build_prompt([], [], template_idx=0)
        assert "technology" in prompt

    def test_defaults_when_no_interests(self):
        prompt = topic_generator._build_prompt(["AI"], [], template_idx=0)
        assert "innovation" in prompt

    def test_template_rotation(self):
        """Different template_idx values produce different prompts."""
        p0 = topic_generator._build_prompt(["AI"], ["ML"], template_idx=0)
        p1 = topic_generator._build_prompt(["AI"], ["ML"], template_idx=1)
        assert p0 != p1


class TestCleanOutput:
    """Tests for the _clean_output helper."""

    def test_removes_prompt_prefix(self):
        prompt = "Start of prompt "
        raw = prompt + "This is the continuation?"
        result = topic_generator._clean_output(raw, prompt)
        assert "Start of prompt" not in result

    def test_stops_at_question_mark(self):
        prompt = "Prompt: "
        raw = prompt + 'How do you use AI? And more text...'
        result = topic_generator._clean_output(raw, prompt)
        assert result.endswith("?")
        assert "And more text" not in result

    def test_adds_question_mark_if_missing(self):
        prompt = "Prompt: "
        raw = prompt + "Have you worked on sustainability"
        result = topic_generator._clean_output(raw, prompt)
        assert result.endswith("?")

    def test_returns_empty_for_short_output(self):
        prompt = "Prompt: "
        raw = prompt + "Hi"
        result = topic_generator._clean_output(raw, prompt)
        assert result == ""


class TestGenerateStarters:
    """Tests for the generate_starters() function."""

    def test_returns_num_starters(self):
        """Should return exactly NUM_STARTERS conversation starters."""
        prompt_prefix = topic_generator._build_prompt(["AI"], ["data"], template_idx=0)
        mock_gen = _make_mock_generator(
            prompt_prefix + 'How are you applying AI in your work?'
        )

        with patch.object(topic_generator, "_get_generator", return_value=mock_gen):
            starters = topic_generator.generate_starters(["AI"], ["data science"])

        assert len(starters) == topic_generator.NUM_STARTERS

    def test_raises_on_empty_themes(self):
        """Empty themes list must raise ValueError."""
        with pytest.raises(ValueError, match="At least one theme"):
            topic_generator.generate_starters([])

    def test_fallback_starters_on_failure(self):
        """When generation consistently fails, fallback starters are returned."""
        mock_gen = MagicMock(side_effect=Exception("GPU error"))

        with patch.object(topic_generator, "_get_generator", return_value=mock_gen):
            starters = topic_generator.generate_starters(["AI"], [])

        assert len(starters) == topic_generator.NUM_STARTERS
        assert all(isinstance(s, str) and len(s) > 5 for s in starters)

    def test_no_duplicate_starters(self):
        """All generated starters should be unique."""
        # Alternate between different outputs to simulate diversity
        outputs = [
            [{"generated_text": topic_generator._build_prompt(["AI"], [], template_idx=i % 3) + f" Unique question number {i}?"}]
            for i in range(10)
        ]
        mock_gen = MagicMock(side_effect=outputs)

        with patch.object(topic_generator, "_get_generator", return_value=mock_gen):
            starters = topic_generator.generate_starters(["AI"])

        assert len(starters) == len(set(starters))

    def test_none_interests_defaults_gracefully(self):
        """Passing None for interests should not raise."""
        prompt_prefix = topic_generator._build_prompt(["AI"], [], template_idx=0)
        mock_gen = _make_mock_generator(prompt_prefix + "How do you apply AI?")

        with patch.object(topic_generator, "_get_generator", return_value=mock_gen):
            starters = topic_generator.generate_starters(["AI"], None)

        assert isinstance(starters, list)

    def test_generator_initialized_once(self):
        """The generator pipeline should only be loaded once."""
        prompt_prefix = topic_generator._build_prompt(["AI"], [], template_idx=0)
        mock_gen_instance = _make_mock_generator(prompt_prefix + "Test question?")

        with patch("services.topic_generator.pipeline", return_value=mock_gen_instance) as mock_pipeline:
            topic_generator._generator = None
            topic_generator.generate_starters(["AI"])
            topic_generator.generate_starters(["Climate"])

        assert mock_pipeline.call_count == 1
