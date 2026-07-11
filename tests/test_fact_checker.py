"""
tests/test_fact_checker.py - Unit tests for the Wikipedia fact checker service.

Wikipedia API calls are mocked using unittest.mock so tests are offline-safe.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from services import fact_checker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_wiki_client():
    """Reset the module-level Wikipedia client singleton before each test."""
    original = fact_checker._wiki
    fact_checker._wiki = None
    yield
    fact_checker._wiki = original


def _make_mock_page(
    exists: bool = True,
    title: str = "Test Page",
    summary: str = "This is the first sentence. This is the second sentence. This is the third.",
    fullurl: str = "https://en.wikipedia.org/wiki/Test_Page",
    categories: dict | None = None,
):
    """Create a mock Wikipedia page object."""
    page = MagicMock()
    page.exists.return_value = exists
    page.title = title
    page.summary = summary
    page.fullurl = fullurl
    page.categories = categories or {}
    return page


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFactCheckTopic:
    """Tests for fact_check_topic()."""

    def test_happy_path_returns_verified(self):
        """A found Wikipedia page returns verified=True with summary."""
        mock_page = _make_mock_page(
            title="Machine Learning",
            summary="Machine learning is a field of AI. It enables computers to learn from data. It is widely used.",
        )
        mock_wiki = MagicMock()
        mock_wiki.page.return_value = mock_page

        with patch.object(fact_checker, "_get_wiki", return_value=mock_wiki):
            result = fact_checker.fact_check_topic("Machine Learning")

        assert result["verified"] is True
        assert result["title"] == "Machine Learning"
        assert "Machine learning" in result["summary"]
        assert result["url"] is not None

    def test_not_found_page_returns_unverified(self):
        """Non-existent Wikipedia page returns verified=False."""
        mock_page = _make_mock_page(exists=False)
        mock_wiki = MagicMock()
        mock_wiki.page.return_value = mock_page

        with patch.object(fact_checker, "_get_wiki", return_value=mock_wiki):
            result = fact_checker.fact_check_topic("Nonexistent Topic XYZ123")

        assert result["verified"] is False
        assert "No reliable reference" in result["summary"]

    def test_disambiguation_page_returns_unverified(self):
        """Disambiguation pages are detected and return verified=False."""
        mock_page = _make_mock_page(
            categories={"Category:All disambiguation pages": None},
        )
        mock_wiki = MagicMock()
        mock_wiki.page.return_value = mock_page

        with patch.object(fact_checker, "_get_wiki", return_value=mock_wiki):
            result = fact_checker.fact_check_topic("Mercury")

        assert result["verified"] is False
        assert "disambiguation" in result["summary"].lower()

    def test_network_failure_returns_unverified(self):
        """Network exceptions are caught and return verified=False."""
        mock_wiki = MagicMock()
        mock_wiki.page.side_effect = ConnectionError("Network error")

        with patch.object(fact_checker, "_get_wiki", return_value=mock_wiki):
            result = fact_checker.fact_check_topic("Blockchain")

        assert result["verified"] is False
        assert "network error" in result["summary"].lower()

    def test_empty_topic_returns_unverified(self):
        """Empty or whitespace topic returns unverified without calling Wikipedia."""
        result = fact_checker.fact_check_topic("   ")
        assert result["verified"] is False

    def test_summary_truncated_to_three_sentences(self):
        """Summary should be capped at 3 sentences."""
        long_summary = ". ".join([f"Sentence {i}" for i in range(10)])
        mock_page = _make_mock_page(summary=long_summary)
        mock_wiki = MagicMock()
        mock_wiki.page.return_value = mock_page

        with patch.object(fact_checker, "_get_wiki", return_value=mock_wiki):
            result = fact_checker.fact_check_topic("AI")

        sentence_count = result["summary"].count(". ") + 1
        assert sentence_count <= 4  # 3 sentences + possible trailing period

    def test_original_topic_preserved_in_result(self):
        """The 'topic' field in the result matches the input query."""
        mock_page = _make_mock_page(title="Blockchain")
        mock_wiki = MagicMock()
        mock_wiki.page.return_value = mock_page

        with patch.object(fact_checker, "_get_wiki", return_value=mock_wiki):
            result = fact_checker.fact_check_topic("Blockchain in Healthcare")

        assert result["topic"] == "Blockchain in Healthcare"


class TestExtractSummary:
    """Tests for the _extract_summary helper."""

    def test_returns_empty_message_for_blank_input(self):
        result = fact_checker._extract_summary("")
        assert result == "No summary available."

    def test_truncates_to_max_sentences(self):
        text = "S1. S2. S3. S4. S5."
        result = fact_checker._extract_summary(text, max_sentences=2)
        assert "S3" not in result

    def test_adds_trailing_period(self):
        text = "Hello world"
        result = fact_checker._extract_summary(text)
        assert result.endswith(".")
