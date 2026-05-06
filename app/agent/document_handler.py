"""
Document ingestion, chunking, embedding, and FAISS vector store management.
Each uploaded document is isolated in its own session keyed by a UUID.
"""
from __future__ import annotations

import io
import uuid
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf import PdfReader


EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class DocumentStore:
    """In-memory per-session FAISS vector store."""

    def __init__(self) -> None:
        # Runs locally — no API token required
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
        self._sessions: dict[str, FAISS] = {}
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, content: bytes, filename: str) -> tuple[str, int]:
        """
        Extract text from *content*, chunk it, embed it, and store it.
        Returns (session_id, chunk_count).
        """
        raw_text = self._extract_text(content, filename)
        if not raw_text.strip():
            raise ValueError(f"No readable text found in '{filename}'.")

        chunks = self._splitter.split_text(raw_text)
        docs = [
            Document(page_content=chunk, metadata={"source": filename, "chunk_index": i})
            for i, chunk in enumerate(chunks)
        ]

        session_id = str(uuid.uuid4())
        self._sessions[session_id] = FAISS.from_documents(docs, self.embeddings)
        return session_id, len(chunks)

    def search(
        self,
        session_id: str,
        query: str,
        k: int = 5,
        score_threshold: float = 2.0,
    ) -> list[tuple[Document, float]]:
        """
        Return up to *k* chunks from the session's vector store.
        When a document is loaded, always return results (the user explicitly
        uploaded a document, so all chunks are relevant context).
        Lower L2 distance = higher similarity.
        Returns [] when the session does not exist.
        """
        store = self._sessions.get(session_id)
        if store is None:
            return []

        results = store.similarity_search_with_score(query, k=k)
        # Always return top-k results when a session exists — the user
        # explicitly uploaded this document and is asking about it.
        return results

    def clear(self, session_id: str) -> bool:
        """Delete a session. Returns True if it existed."""
        return self._sessions.pop(session_id, None) is not None

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_text(self, content: bytes, filename: str) -> str:
        lower = filename.lower()
        if lower.endswith(".pdf"):
            return self._extract_pdf(content)
        # .txt / .md / .csv — treated as plain UTF-8
        return content.decode("utf-8", errors="replace")

    @staticmethod
    def _extract_pdf(content: bytes) -> str:
        reader = PdfReader(io.BytesIO(content))
        pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n".join(pages)
