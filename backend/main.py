"""
backend/main.py - FastAPI Application Factory

Creates and configures the FastAPI application instance.
Registers routers, middleware, and startup/shutdown events.
"""

from __future__ import annotations

import logging
import logging.config

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import (
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    LOG_LEVEL,
)
from backend.routes import router

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application Factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS – allow Streamlit dev server and same-host requests
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register all routes under the /api/v1 prefix
    application.include_router(router, prefix="/api/v1")

    @application.on_event("startup")
    async def _startup() -> None:
        logger.info("🚀 Personalized Networking Assistant API v%s started.", API_VERSION)

    @application.on_event("shutdown")
    async def _shutdown() -> None:
        logger.info("🛑 API server shutting down.")

    return application


# ---------------------------------------------------------------------------
# Application Instance (used by Uvicorn)
# ---------------------------------------------------------------------------

app = create_app()
