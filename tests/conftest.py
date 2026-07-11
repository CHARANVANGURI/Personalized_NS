"""
tests/conftest.py - Shared pytest fixtures and configuration.

Provides reusable fixtures including a configured TestClient for the
FastAPI application and a temporary data directory for isolation.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ─── Patch data files BEFORE importing the app ───────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def tmp_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create an isolated temporary data directory for the entire test session."""
    base = tmp_path_factory.mktemp("data")
    (base / "history.json").write_text("[]")
    (base / "feedback.json").write_text("[]")
    return base


@pytest.fixture(scope="session")
def app_client(tmp_data_dir: Path) -> Generator:
    """
    Create a FastAPI TestClient with patched data paths.

    The backend config is monkey-patched so all file I/O uses the
    temporary directory instead of the real data/ folder.
    """
    with (
        patch("backend.config.HISTORY_FILE", tmp_data_dir / "history.json"),
        patch("backend.config.FEEDBACK_FILE", tmp_data_dir / "feedback.json"),
        patch("services.history_logger.HISTORY_FILE", tmp_data_dir / "history.json"),
        patch("services.feedback_logger.FEEDBACK_FILE", tmp_data_dir / "feedback.json"),
    ):
        from backend.main import create_app
        test_app = create_app()
        with TestClient(test_app) as client:
            yield client
