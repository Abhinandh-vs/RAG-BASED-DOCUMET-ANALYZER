"""
LangChain agent chain.

Flow:
  1. If session_id provided → search FAISS for relevant chunks.
  2. If relevant chunks found → RAG prompt → LLM (source = "document").
  3. Otherwise → plain model prompt → LLM (source = "model").
"""
from __future__ import annotations

import os
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from app.agent.document_handler import DocumentStore


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _rag_messages(context: str, question: str):
    return [
        SystemMessage(content=(
            "You are a helpful document analysis assistant. The user has uploaded a document "
            "and is asking questions about it. Answer the user's question using the document "
            "context provided below. Be specific, accurate, and quote relevant details directly "
            "from the context. Do not refuse to answer if the information is present in the context.\n\n"
            "--- DOCUMENT CONTEXT ---\n" + context + "\n--- END CONTEXT ---"
        )),
        HumanMessage(content=question),
    ]

def _fallback_messages(question: str):
    return [
        SystemMessage(content="You are a helpful assistant. Answer the question based on your general knowledge."),
        HumanMessage(content=question),
    ]


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _build_llm() -> ChatHuggingFace:
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    model_id = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
    if not token:
        raise RuntimeError(
            "HUGGINGFACEHUB_API_TOKEN is not set. "
            "Add it to your .env file (free token from https://huggingface.co/settings/tokens)."
        )
    endpoint = HuggingFaceEndpoint(
        repo_id=model_id,
        huggingfacehub_api_token=token,
        max_new_tokens=512,
        temperature=0.3,
        task="conversational",
    )
    return ChatHuggingFace(llm=endpoint)


# ---------------------------------------------------------------------------
# Agent chain
# ---------------------------------------------------------------------------

class AgentChain:
    """Orchestrates document retrieval + LLM calls."""

    def __init__(self, document_store: DocumentStore) -> None:
        self._store = document_store
        self._llm = _build_llm()
        self._threshold = float(os.getenv("RELEVANCE_THRESHOLD", "2.0"))

    # ------------------------------------------------------------------

    def chat(self, message: str, session_id: Optional[str] = None) -> dict:
        """
        Returns a dict with keys: reply, source, chunks_used.
        """
        if session_id:
            relevant = self._store.search(
                session_id, message, k=5, score_threshold=self._threshold
            )
            if relevant:
                return self._answer_from_document(message, relevant)

        return self._answer_from_model(message)

    # ------------------------------------------------------------------

    def _answer_from_document(self, question: str, chunks: list) -> dict:
        context = "\n\n---\n\n".join(doc.page_content for doc, _ in chunks)
        raw = self._llm.invoke(_rag_messages(context, question))
        return {
            "reply": self._clean(raw),
            "source": "document",
            "chunks_used": len(chunks),
        }

    def _answer_from_model(self, question: str) -> dict:
        raw = self._llm.invoke(_fallback_messages(question))
        return {
            "reply": self._clean(raw),
            "source": "model",
            "chunks_used": 0,
        }

    @staticmethod
    def _clean(msg) -> str:
        """Extract text from an AIMessage or plain string."""
        if hasattr(msg, "content"):
            return msg.content.strip()
        return str(msg).strip()

    @property
    def model_id(self) -> str:
        return os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")

    # ------------------------------------------------------------------

    def suggest_questions(self, session_id: str) -> dict:
        """
        Analyze document chunks and generate suggested questions + a brief summary.
        Returns dict with keys: questions (list[str]), summary (str).
        """
        store = self._store._sessions.get(session_id)
        if not store:
            return {"questions": [], "summary": ""}

        # Get all docs from the store (sample broadly)
        all_docs = store.similarity_search("document overview summary", k=5)
        context = "\n\n---\n\n".join(doc.page_content for doc in all_docs)

        messages = [
            SystemMessage(content=(
                "You are a document analysis assistant. Based on the document content below, do two things:\n"
                "1. Write a ONE sentence summary of what this document is about.\n"
                "2. Generate exactly 5 specific, useful questions that a user would likely want to ask about this document.\n\n"
                "Respond in this EXACT format (no extra text):\n"
                "SUMMARY: <one sentence summary>\n"
                "Q1: <question>\n"
                "Q2: <question>\n"
                "Q3: <question>\n"
                "Q4: <question>\n"
                "Q5: <question>\n\n"
                "--- DOCUMENT CONTENT ---\n" + context + "\n--- END ---"
            )),
            HumanMessage(content="Generate the summary and questions."),
        ]

        raw = self._llm.invoke(messages)
        text = self._clean(raw)

        # Parse the response
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        summary = ""
        questions = []

        for line in lines:
            if line.upper().startswith("SUMMARY:"):
                summary = line.split(":", 1)[1].strip()
            elif line[:2] in ("Q1", "Q2", "Q3", "Q4", "Q5") and ":" in line:
                questions.append(line.split(":", 1)[1].strip())

        # Fallback if parsing failed
        if not questions:
            questions = [
                "What is this document about?",
                "What are the key details mentioned?",
                "Who is mentioned in this document?",
                "What skills or qualifications are listed?",
                "What is the timeline or dates mentioned?",
            ]

        return {"questions": questions[:5], "summary": summary}
