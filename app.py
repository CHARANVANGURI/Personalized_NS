"""
app.py - Application Entry Point

Provides a unified entry point to launch either the FastAPI backend
or Streamlit frontend based on user arguments.

Usage:
    python app.py --mode backend   # Start FastAPI
    python app.py --mode frontend  # Start Streamlit
"""

import argparse
import subprocess
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def start_backend() -> None:
    """Launch the FastAPI backend via Uvicorn."""
    logger.info("Starting FastAPI backend on http://127.0.0.1:8000")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.main:app",
            "--reload",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        check=True,
    )


def start_frontend() -> None:
    """Launch the Streamlit frontend."""
    logger.info("Starting Streamlit frontend on http://localhost:8501")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "frontend/streamlit_ui.py",
            "--server.port",
            "8501",
        ],
        check=True,
    )


def main() -> None:
    """Parse arguments and start the appropriate service."""
    parser = argparse.ArgumentParser(
        description="Personalized Networking Assistant - Launcher"
    )
    parser.add_argument(
        "--mode",
        choices=["backend", "frontend"],
        default="backend",
        help="Service to start: 'backend' (FastAPI) or 'frontend' (Streamlit)",
    )
    args = parser.parse_args()

    if args.mode == "backend":
        start_backend()
    else:
        start_frontend()


if __name__ == "__main__":
    main()
