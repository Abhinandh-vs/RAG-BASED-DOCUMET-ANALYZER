"""
Integration test suite — uses httpx against a live server.
Run with: pytest tests/ -v
"""
from __future__ import annotations

import io
import pytest
from httpx import AsyncClient, ASGITransport


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module")
async def client():
    """Spin up the app in-process without a real network socket."""
    import os
    os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", "hf_test_token")

    from app.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# TC-01 — Health check
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_health_ok(client):
    """Health endpoint must return 200 and status=ok."""
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "model_id" in body
    assert "embeddings_model" in body
    assert isinstance(body["active_sessions"], int)


# ---------------------------------------------------------------------------
# TC-02 — Upload valid TXT document
# ---------------------------------------------------------------------------

SAMPLE_DOC = b"""
Rugby World Cup 2027 will be held in Australia.
The All Blacks are the most successful team in Rugby World Cup history with three titles.
South Africa won in 2019 defeating England 32-12 in the final.
"""

@pytest.mark.anyio
async def test_upload_txt_document(client):
    """Valid TXT upload must return 201 with a session_id and chunk count."""
    resp = await client.post(
        "/agent/upload",
        files={"file": ("rugby_facts.txt", io.BytesIO(SAMPLE_DOC), "text/plain")},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert "session_id" in body
    assert body["chunks_loaded"] > 0
    assert body["filename"] == "rugby_facts.txt"
    return body["session_id"]


# ---------------------------------------------------------------------------
# TC-03 — Upload unsupported file type
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_upload_unsupported_type(client):
    """Uploading an unsupported file type must return 415."""
    resp = await client.post(
        "/agent/upload",
        files={"file": ("data.xlsx", io.BytesIO(b"fake excel"), "application/vnd.ms-excel")},
    )
    assert resp.status_code == 415


# ---------------------------------------------------------------------------
# TC-04 — Upload empty file
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_upload_empty_file(client):
    """Empty file must return 400."""
    resp = await client.post(
        "/agent/upload",
        files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# TC-05 — Chat without session (model fallback)
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_chat_no_session(client):
    """Chat without session_id must call the model and return source='model'."""
    resp = await client.post(
        "/agent/chat",
        json={"message": "What is the capital of France?"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "model"
    assert body["chunks_used"] == 0
    assert len(body["reply"]) > 0


# ---------------------------------------------------------------------------
# TC-06 — Chat WITH session, question IS in document
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_chat_with_session_relevant_question(client):
    """Question covered by the document must return source='document'."""
    # Re-upload to get a fresh session
    upload = await client.post(
        "/agent/upload",
        files={"file": ("rugby_facts.txt", io.BytesIO(SAMPLE_DOC), "text/plain")},
    )
    session_id = upload.json()["session_id"]

    resp = await client.post(
        "/agent/chat",
        json={
            "message": "Where is the Rugby World Cup 2027 being held?",
            "session_id": session_id,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    # The document has the answer — should use document
    assert body["source"] == "document"
    assert body["chunks_used"] > 0
    assert body["session_id"] == session_id


# ---------------------------------------------------------------------------
# TC-07 — Chat WITH session, question NOT in document → model fallback
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_chat_with_session_irrelevant_question(client):
    """Question not covered by doc must fall back to model (source='model')."""
    upload = await client.post(
        "/agent/upload",
        files={"file": ("rugby_facts.txt", io.BytesIO(SAMPLE_DOC), "text/plain")},
    )
    session_id = upload.json()["session_id"]

    resp = await client.post(
        "/agent/chat",
        json={
            "message": "How many planets are in the solar system?",
            "session_id": session_id,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    # Completely off-topic → should fall back to model
    assert body["source"] == "model"


# ---------------------------------------------------------------------------
# TC-08 — Chat with invalid session_id (treated as no document)
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_chat_unknown_session(client):
    """Unknown session_id must not crash — fall back to model."""
    resp = await client.post(
        "/agent/chat",
        json={"message": "Tell me something", "session_id": "non-existent-session-xyz"},
    )
    assert resp.status_code == 200
    assert resp.json()["source"] == "model"


# ---------------------------------------------------------------------------
# TC-09 — Delete session
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_delete_session(client):
    """Deleting a valid session must return cleared=True."""
    upload = await client.post(
        "/agent/upload",
        files={"file": ("rugby_facts.txt", io.BytesIO(SAMPLE_DOC), "text/plain")},
    )
    session_id = upload.json()["session_id"]

    del_resp = await client.delete(f"/agent/session/{session_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["cleared"] is True


# ---------------------------------------------------------------------------
# TC-10 — Delete non-existent session
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_delete_nonexistent_session(client):
    """Deleting a session that does not exist must return 404."""
    resp = await client.delete("/agent/session/does-not-exist-123")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# TC-11 — Empty message validation
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_chat_empty_message(client):
    """Empty message must be rejected by Pydantic (422)."""
    resp = await client.post("/agent/chat", json={"message": ""})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# TC-12 — Active sessions counter in health
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_active_sessions_reflected_in_health(client):
    """active_sessions must increment after upload and decrement after delete."""
    health_before = (await client.get("/health")).json()["active_sessions"]

    upload = await client.post(
        "/agent/upload",
        files={"file": ("rugby_facts.txt", io.BytesIO(SAMPLE_DOC), "text/plain")},
    )
    session_id = upload.json()["session_id"]
    health_after_upload = (await client.get("/health")).json()["active_sessions"]
    assert health_after_upload == health_before + 1

    await client.delete(f"/agent/session/{session_id}")
    health_after_delete = (await client.get("/health")).json()["active_sessions"]
    assert health_after_delete == health_before
