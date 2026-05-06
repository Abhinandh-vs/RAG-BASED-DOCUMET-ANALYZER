"""
FastAPI application entry point.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.agent.document_handler import DocumentStore, EMBEDDINGS_MODEL
from app.agent.chain import AgentChain
from app.api.routes import router as agent_router
from app.models.schemas import HealthResponse

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "info").upper())
logger = logging.getLogger(__name__)

APP_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Lifespan — initialise heavy objects once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading embeddings model …")
    store = DocumentStore()
    agent = AgentChain(document_store=store)
    app.state.store = store
    app.state.agent = agent
    logger.info("Ready. Model: %s", agent.model_id)
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Document Q&A Agent",
    description=(
        "Upload a document and chat with it. "
        "Answers are grounded in the document when relevant; "
        "the HuggingFace model answers from general knowledge otherwise."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# All agent routes live under /agent
app.include_router(agent_router, prefix="/agent", tags=["Agent"])


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
)
async def health():
    return HealthResponse(
        status="ok",
        model_id=os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct"),
        embeddings_model=EMBEDDINGS_MODEL,
        active_sessions=app.state.store.active_sessions,
        version=APP_VERSION,
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def ui():
    return (_STATIC_DIR / "index.html").read_text()
