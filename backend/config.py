"""
backend/config.py - Application Configuration

Centralizes all application settings using environment variables with
sensible defaults. Loaded once at startup to avoid repeated I/O.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Base Paths
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
HISTORY_FILE: Path = DATA_DIR / "history.json"
FEEDBACK_FILE: Path = DATA_DIR / "feedback.json"

# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
API_HOST: str = os.getenv("API_HOST", "127.0.0.1")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
API_TITLE: str = "Personalized Networking Assistant API"
API_VERSION: str = "1.0.0"
API_DESCRIPTION: str = (
    "AI-powered API for generating personalized networking conversation starters, "
    "analyzing event themes, verifying facts, and managing history/feedback."
)

# ---------------------------------------------------------------------------
# ML Models
# ---------------------------------------------------------------------------
ZERO_SHOT_MODEL: str = os.getenv(
    "ZERO_SHOT_MODEL", "cross-encoder/nli-distilroberta-base"
)
TEXT_GEN_MODEL: str = os.getenv("TEXT_GEN_MODEL", "gpt2")

# Candidate classification labels for event analysis
CANDIDATE_LABELS: list[str] = [
    "AI",
    "Sustainability",
    "Climate",
    "Healthcare",
    "Finance",
    "Robotics",
    "Data Science",
    "Urban Planning",
    "Blockchain",
    "Cybersecurity",
    "Startups",
    "Entrepreneurship",
]

# ---------------------------------------------------------------------------
# Generation Parameters
# ---------------------------------------------------------------------------
MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "80"))
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.8"))
TOP_P: float = float(os.getenv("TOP_P", "0.95"))
NUM_STARTERS: int = int(os.getenv("NUM_STARTERS", "3"))

# ---------------------------------------------------------------------------
# Wikipedia
# ---------------------------------------------------------------------------
WIKIPEDIA_LANGUAGE: str = os.getenv("WIKIPEDIA_LANGUAGE", "en")
WIKIPEDIA_USER_AGENT: str = os.getenv(
    "WIKIPEDIA_USER_AGENT", "PersonalizedNetworkingAssistant/1.0"
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
