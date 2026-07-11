"""
tests/test_routes.py - Integration tests for FastAPI route handlers.

Uses FastAPI's TestClient with mocked service dependencies so tests
are fast, isolated, and do not require GPU or network access.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tmp_data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("route_data")
    (base / "history.json").write_text("[]")
    (base / "feedback.json").write_text("[]")
    return base


@pytest.fixture(scope="module")
def client(tmp_data: Path) -> TestClient:
    """TestClient with data paths redirected to a temp dir."""
    with (
        patch("backend.config.HISTORY_FILE", tmp_data / "history.json"),
        patch("backend.config.FEEDBACK_FILE", tmp_data / "feedback.json"),
        patch("services.history_logger.HISTORY_FILE", tmp_data / "history.json"),
        patch("services.feedback_logger.FEEDBACK_FILE", tmp_data / "feedback.json"),
    ):
        app = create_app()
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_ok(self, client: TestClient):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "version" in body


# ---------------------------------------------------------------------------
# Analyze Event
# ---------------------------------------------------------------------------


class TestAnalyzeEvent:
    def test_happy_path(self, client: TestClient):
        mock_result = {
            "themes": ["AI", "Climate"],
            "scores": [0.92, 0.85],
        }
        with patch("backend.routes.analyze_event", return_value=mock_result):
            resp = client.post(
                "/api/v1/analyze-event",
                json={"event_description": "AI for Sustainable Cities summit"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["themes"] == ["AI", "Climate"]
        assert data["scores"] == [0.92, 0.85]

    def test_short_description_returns_422(self, client: TestClient):
        resp = client.post(
            "/api/v1/analyze-event",
            json={"event_description": "x"},
        )
        assert resp.status_code == 422

    def test_missing_body_returns_422(self, client: TestClient):
        resp = client.post("/api/v1/analyze-event", json={})
        assert resp.status_code == 422

    def test_service_failure_returns_500(self, client: TestClient):
        with patch("backend.routes.analyze_event", side_effect=RuntimeError("model error")):
            resp = client.post(
                "/api/v1/analyze-event",
                json={"event_description": "AI Healthcare Summit 2024"},
            )
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# Generate Conversation
# ---------------------------------------------------------------------------


class TestGenerateConversation:
    def test_happy_path(self, client: TestClient):
        mock_analysis = {"themes": ["AI", "Urban Planning"], "scores": [0.9, 0.8]}
        mock_starters = [
            "How do you apply AI to urban design?",
            "What AI tools have you used for city planning?",
            "Have you seen AI reduce traffic congestion?",
        ]
        with (
            patch("backend.routes.analyze_event", return_value=mock_analysis),
            patch("backend.routes.generate_starters", return_value=mock_starters),
        ):
            resp = client.post(
                "/api/v1/generate-conversation",
                json={
                    "event_description": "AI for Sustainable Cities tech summit event",
                    "user_interests": ["climate change", "urban planning"],
                },
            )
        assert resp.status_code == 201
        data = resp.json()
        assert len(data["starters"]) == 3
        assert "conversation_id" in data
        assert "timestamp" in data

    def test_no_interests_still_works(self, client: TestClient):
        mock_analysis = {"themes": ["Finance"], "scores": [0.88]}
        mock_starters = ["What fintech innovations excite you?"] * 3
        with (
            patch("backend.routes.analyze_event", return_value=mock_analysis),
            patch("backend.routes.generate_starters", return_value=mock_starters),
        ):
            resp = client.post(
                "/api/v1/generate-conversation",
                json={"event_description": "Finance and Blockchain Innovation Summit"},
            )
        assert resp.status_code == 201

    def test_generation_failure_returns_500(self, client: TestClient):
        with (
            patch("backend.routes.analyze_event", return_value={"themes": ["AI"], "scores": [0.9]}),
            patch("backend.routes.generate_starters", side_effect=RuntimeError("GPT-2 error")),
        ):
            resp = client.post(
                "/api/v1/generate-conversation",
                json={"event_description": "Healthcare AI Innovation Summit 2024"},
            )
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# Fact Check
# ---------------------------------------------------------------------------


class TestFactCheck:
    def test_verified_topic(self, client: TestClient):
        mock_result = {
            "topic": "Machine Learning",
            "title": "Machine Learning",
            "summary": "ML is a subset of AI.",
            "verified": True,
            "url": "https://en.wikipedia.org/wiki/Machine_learning",
        }
        with patch("backend.routes.fact_check_topic", return_value=mock_result):
            resp = client.get("/api/v1/fact-check", params={"topic": "Machine Learning"})
        assert resp.status_code == 200
        assert resp.json()["verified"] is True

    def test_unverified_topic(self, client: TestClient):
        mock_result = {
            "topic": "Unknown XYZ",
            "title": "N/A",
            "summary": "No reliable reference available.",
            "verified": False,
            "url": None,
        }
        with patch("backend.routes.fact_check_topic", return_value=mock_result):
            resp = client.get("/api/v1/fact-check", params={"topic": "Unknown XYZ"})
        assert resp.status_code == 200
        assert resp.json()["verified"] is False

    def test_missing_topic_param_returns_422(self, client: TestClient):
        resp = client.get("/api/v1/fact-check")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


class TestHistory:
    def test_empty_history(self, client: TestClient):
        with patch("backend.routes.load_history", return_value=[]):
            resp = client.get("/api/v1/history")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_history_with_entries(self, client: TestClient):
        entries = [
            {
                "conversation_id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_description": "AI summit event",
                "themes": ["AI"],
                "scores": [0.9],
                "starters": ["How are you using AI?"],
                "user_interests": [],
                "favorite": False,
            }
        ]
        with patch("backend.routes.load_history", return_value=entries):
            resp = client.get("/api/v1/history")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

    def test_favorites_only_filter(self, client: TestClient):
        entries = [
            {
                "conversation_id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_description": "Fav event",
                "themes": ["AI"],
                "scores": [0.9],
                "starters": ["Question?"],
                "user_interests": [],
                "favorite": True,
            },
            {
                "conversation_id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_description": "Non-fav event",
                "themes": ["Climate"],
                "scores": [0.8],
                "starters": ["Another?"],
                "user_interests": [],
                "favorite": False,
            },
        ]
        with patch("backend.routes.load_history", return_value=entries):
            resp = client.get("/api/v1/history", params={"favorites_only": "true"})
        assert resp.status_code == 200
        assert resp.json()["total"] == 1


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------


class TestFeedback:
    def test_post_thumbs_up(self, client: TestClient):
        with patch("backend.routes.save_feedback"):
            resp = client.post(
                "/api/v1/feedback",
                json={
                    "conversation_id": str(uuid.uuid4()),
                    "thumbs_up": True,
                    "thumbs_down": False,
                    "comment": "Great starters!",
                },
            )
        assert resp.status_code == 201
        data = resp.json()
        assert data["thumbs_up"] is True
        assert "feedback_id" in data

    def test_post_thumbs_down(self, client: TestClient):
        with patch("backend.routes.save_feedback"):
            resp = client.post(
                "/api/v1/feedback",
                json={
                    "conversation_id": str(uuid.uuid4()),
                    "thumbs_up": False,
                    "thumbs_down": True,
                },
            )
        assert resp.status_code == 201

    def test_both_thumbs_returns_422(self, client: TestClient):
        resp = client.post(
            "/api/v1/feedback",
            json={
                "conversation_id": str(uuid.uuid4()),
                "thumbs_up": True,
                "thumbs_down": True,
            },
        )
        assert resp.status_code == 422

    def test_get_feedback_empty(self, client: TestClient):
        with patch("backend.routes.load_feedback", return_value=[]):
            resp = client.get("/api/v1/feedback")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_get_feedback_filter_by_conversation(self, client: TestClient):
        conv_id = str(uuid.uuid4())
        entries = [
            {
                "feedback_id": str(uuid.uuid4()),
                "conversation_id": conv_id,
                "thumbs_up": True,
                "thumbs_down": False,
                "comment": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ]
        with patch("backend.routes.load_feedback", return_value=entries):
            resp = client.get("/api/v1/feedback", params={"conversation_id": conv_id})
        assert resp.status_code == 200
        assert resp.json()["total"] == 1
