"""FastAPI application factory for the medical AI assistant.

Start with: uvicorn src.api.app:app --reload --port 8000
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.infra.db import engine, Base
from src.shared.config import settings
from src.shared.logging import get_logger
from src.api.routes import conversations, kb

logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Create all database tables on startup
    Base.metadata.create_all(bind=engine)
    logger.info(f"Medical AI Assistant starting up — model: {settings.primary_model}")
    yield
    logger.info("Shutting down API server")


app = FastAPI(
    title="医疗设备知识助手",
    description="Enterprise-grade RAG assistant for medical device support",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Streamlit dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(conversations.router, prefix="/api", tags=["conversations"])
app.include_router(kb.router, prefix="/api", tags=["knowledge-base"])


@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "models_available": bool(settings.dashscope_api_key),
    }
