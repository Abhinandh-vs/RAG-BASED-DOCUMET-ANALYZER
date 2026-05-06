"""
FastAPI routes — all agent endpoints are mounted under /agent.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Request, status

from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ClearSessionResponse,
    HealthResponse,
    SuggestRequest,
    SuggestResponse,
    UploadResponse,
)

router = APIRouter()

_ALLOWED_TYPES = {
    "application/pdf",
    "text/plain",
    "text/markdown",
    "text/csv",
    "application/octet-stream",  # some clients send this for .txt
}
_ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _agent(request: Request):
    """Dependency — retrieve AgentChain from app state."""
    return request.app.state.agent


def _store(request: Request):
    """Dependency — retrieve DocumentStore from app state."""
    return request.app.state.store


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload a document",
    description=(
        "Upload a PDF, TXT, MD, or CSV file. The server chunks and embeds it into a FAISS "
        "vector store. The returned `session_id` must be passed to `/agent/chat` so answers "
        "are grounded in this document."
    ),
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: UploadFile = File(..., description="PDF, TXT, MD or CSV document"),
    store=Depends(_store),
):
    import os

    ext = os.path.splitext(file.filename or "")[-1].lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(_ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    try:
        session_id, chunk_count = store.ingest(content, file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    return UploadResponse(
        session_id=session_id,
        filename=file.filename,
        chunks_loaded=chunk_count,
        message=f"Document indexed successfully. Use session_id '{session_id}' in /agent/chat.",
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat with the agent",
    description=(
        "Send a message. If `session_id` is provided and the document contains relevant context, "
        "the answer is grounded in the document (source='document'). "
        "Otherwise the HuggingFace model answers from general knowledge (source='model')."
    ),
)
async def chat(body: ChatRequest, agent=Depends(_agent)):
    try:
        result = agent.chat(message=body.message, session_id=body.session_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    return ChatResponse(
        reply=result["reply"],
        source=result["source"],
        session_id=body.session_id,
        chunks_used=result["chunks_used"],
    )


@router.delete(
    "/session/{session_id}",
    response_model=ClearSessionResponse,
    summary="Clear a document session",
    description="Remove the FAISS index for a session to free memory.",
)
async def clear_session(session_id: str, store=Depends(_store)):
    cleared = store.clear(session_id)
    if not cleared:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    return ClearSessionResponse(session_id=session_id, cleared=True)


@router.post(
    "/suggest",
    response_model=SuggestResponse,
    summary="Get suggested questions for a document",
    description=(
        "Analyze the uploaded document and return a brief summary plus "
        "suggested questions the user might want to ask."
    ),
)
async def suggest_questions(body: SuggestRequest, agent=Depends(_agent)):
    try:
        result = agent.suggest_questions(session_id=body.session_id)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    return SuggestResponse(
        session_id=body.session_id,
        questions=result["questions"],
        summary=result["summary"],
    )
