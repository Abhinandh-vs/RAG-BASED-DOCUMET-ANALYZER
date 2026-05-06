from pydantic import BaseModel, Field
from typing import Literal, Optional


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096, description="User question or message")
    session_id: Optional[str] = Field(None, description="Session ID returned from /agent/upload")

    model_config = {"json_schema_extra": {"examples": [
        {"message": "What is the main topic of this document?", "session_id": "abc-123"},
        {"message": "Who invented the internet?"},
    ]}}


class ChatResponse(BaseModel):
    reply: str = Field(..., description="Agent's answer")
    source: Literal["document", "model"] = Field(
        ..., description="'document' = answer from uploaded doc, 'model' = HuggingFace model fallback"
    )
    session_id: Optional[str] = Field(None, description="Echoed session ID if provided")
    chunks_used: int = Field(0, description="Number of document chunks used as context")


class SuggestRequest(BaseModel):
    session_id: str = Field(..., description="Session ID returned from /agent/upload")


class SuggestResponse(BaseModel):
    session_id: str
    questions: list[str] = Field(..., description="List of suggested questions based on the document")
    summary: str = Field("", description="Brief document summary")


class UploadResponse(BaseModel):
    session_id: str = Field(..., description="Use this ID in subsequent /agent/chat calls")
    filename: str
    chunks_loaded: int = Field(..., description="Number of text chunks indexed into FAISS")
    message: str


class ClearSessionResponse(BaseModel):
    session_id: str
    cleared: bool


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_id: str
    embeddings_model: str
    active_sessions: int
    version: str
