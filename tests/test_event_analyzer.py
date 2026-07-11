"""
tests/test_event_analyzer.py - Unit tests for the event analysis service.

All transformer pipeline calls are mocked so tests run without GPU and
without downloading models.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from services import event_analyzer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_classifier():
    """Reset the module-level classifier singleton before each test."""
    original = event_analyzer._classifier
    event_analyzer._classifier = None
    yield
    event_analyzer._classifier = original


def _make_mock_classifier(labels: list[str], scores: list[float]):
    """Return a mock pipeline callable that returns a classification result."""
    mock = MagicMock()
    mock.return_value = {"labels": labels, "scores": scores}
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAnalyzeEvent:
    """Tests for the analyze_event() function."""

    def test_returns_themes_and_scores(self):
        """Happy path: returns top themes sorted by score."""
        labels = ["AI", "Climate", "Urban Planning", "Sustainability", "Data Science"]
        scores = [0.92, 0.85, 0.78, 0.65, 0.55]
        mock_clf = _make_mock_classifier(labels, scores)

        with patch.object(event_analyzer, "_get_classifier", return_value=mock_clf):
            result = event_analyzer.analyze_event("AI for Sustainable Cities summit")

        assert "themes" in result
        assert "scores" in result
        assert result["themes"][0] == "AI"
        assert result["scores"][0] == pytest.approx(0.92, abs=1e-4)

    def test_returns_max_five_themes(self):
        """Only top-5 themes are returned even if more labels exist."""
        labels = [f"Label{i}" for i in range(10)]
        scores = [0.9 - i * 0.05 for i in range(10)]
        mock_clf = _make_mock_classifier(labels, scores)

        with patch.object(event_analyzer, "_get_classifier", return_value=mock_clf):
            result = event_analyzer.analyze_event("Some networking event")

        assert len(result["themes"]) <= 5
        assert len(result["scores"]) <= 5

    def test_themes_sorted_by_score_descending(self):
        """Themes must be ordered from highest to lowest confidence."""
        labels = ["B", "A", "C"]
        scores = [0.5, 0.9, 0.3]
        mock_clf = _make_mock_classifier(labels, scores)

        with patch.object(event_analyzer, "_get_classifier", return_value=mock_clf):
            result = event_analyzer.analyze_event("Healthcare blockchain conference")

        assert result["themes"][0] == "A"  # highest score

    def test_empty_description_raises_value_error(self):
        """Empty or whitespace-only input should raise ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            event_analyzer.analyze_event("   ")

    def test_pipeline_failure_raises_runtime_error(self):
        """If the pipeline throws, a RuntimeError is propagated."""
        mock_clf = MagicMock(side_effect=RuntimeError("CUDA out of memory"))

        with patch.object(event_analyzer, "_get_classifier", return_value=mock_clf):
            with pytest.raises(RuntimeError, match="pipeline failed"):
                event_analyzer.analyze_event("A valid event description")

    def test_scores_are_rounded(self):
        """Scores should be rounded to 4 decimal places."""
        labels = ["AI"]
        scores = [0.923456789]
        mock_clf = _make_mock_classifier(labels, scores)

        with patch.object(event_analyzer, "_get_classifier", return_value=mock_clf):
            result = event_analyzer.analyze_event("AI summit")

        assert result["scores"][0] == 0.9235

    def test_classifier_initialized_once(self):
        """The classifier should only be loaded once (singleton)."""
        labels = ["AI"]
        scores = [0.9]
        mock_clf = _make_mock_classifier(labels, scores)

        with patch("services.event_analyzer.pipeline", return_value=mock_clf) as mock_pipeline:
            event_analyzer._classifier = None  # force reload
            event_analyzer.analyze_event("AI summit")
            event_analyzer.analyze_event("Climate conference")

        # pipeline() should only be called once across both invocations
        assert mock_pipeline.call_count == 1
