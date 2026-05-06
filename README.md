# Document Q&A Agent — FastAPI + LangChain + HuggingFace

> **Stack:** FastAPI · uv · LangChain · HuggingFace Inference API (free) · FAISS · sentence-transformers

---

## API Chart

```
┌──────────────────────────────────────────────────────────────────────┐
│                         CLIENT (curl / browser / test)               │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
          ┌─────────────────────▼─────────────────────────────────────┐
          │                  FastAPI App                               │
          │  GET  /health          → system status                     │
          │  POST /agent/upload    → ingest document                   │
          │  POST /agent/chat      → question / answer                 │
          │  DEL  /agent/session/{id} → clear FAISS session            │
          │  GET  /docs            → Swagger UI (auto-generated)       │
          │  GET  /redoc           → ReDoc UI (auto-generated)         │
          └─────┬───────────────────────────┬───────────────────────── ┘
                │                           │
     ┌──────────▼──────────┐    ┌───────────▼─────────────────────────┐
     │   DocumentStore      │    │          AgentChain                 │
     │  (document_handler)  │    │           (chain.py)                │
     │                      │    │                                     │
     │  ingest(bytes, name) │    │  chat(message, session_id?)         │
     │  → chunk (500 tok)   │    │  ┌──────────────────────────────┐   │
     │  → embed (local)     │    │  │ session_id given?            │   │
     │  → FAISS index       │    │  │  YES → search FAISS          │   │
     │  → return session_id │    │  │    relevant? (L2 < 1.2)      │   │
     │                      │    │  │      YES → RAG prompt → LLM  │   │
     │  search(id, query)   │    │  │      NO  → plain prompt→ LLM │   │
     │  → top-k chunks      │    │  │  NO  → plain prompt → LLM    │   │
     │    (L2 filtered)     │    │  └──────────────────────────────┘   │
     └──────────┬───────────┘    └───────────┬─────────────────────────┘
                │                            │
     ┌──────────▼───────────┐    ┌───────────▼───────────────────────┐
     │  FAISS Vector Store  │    │   HuggingFace Inference API        │
     │  (in-memory, FAISS)  │    │   model: Mistral-7B-Instruct-v0.2  │
     │  per-session isolated│    │   (free tier, HF token required)   │
     └──────────────────────┘    └───────────────────────────────────┘
```

### Decision Tree — `/agent/chat`

```
POST /agent/chat { message, session_id? }
        │
        ├─ session_id provided?
        │       │
        │      YES → FAISS.search(session_id, message)
        │               │
        │               ├─ L2 score < threshold (1.2)?
        │               │       │
        │               │      YES → RAG prompt (context + question) → LLM
        │               │               └─ reply + source="document" + chunks_used=N
        │               │
        │               └─ NO relevant chunks → fall through ↓
        │
        └─ Plain prompt (question only) → LLM
                └─ reply + source="model" + chunks_used=0
```

---

## Project Structure

```
fastapi-agent/
├── pyproject.toml          ← uv project manifest + dependencies
├── .env.example            ← copy to .env and fill in HF token
├── app/
│   ├── main.py             ← FastAPI app, lifespan, CORS, /health
│   ├── agent/
│   │   ├── chain.py        ← LangChain AgentChain (RAG + fallback)
│   │   └── document_handler.py  ← ingest, chunk, embed, FAISS
│   ├── api/
│   │   └── routes.py       ← /agent/upload, /agent/chat, /agent/session
│   └── models/
│       └── schemas.py      ← Pydantic request/response models
└── tests/
    └── test_agent.py       ← 12 pytest integration tests
```

---

## Quick Start

### 1 — Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2 — Install dependencies

```bash
cd fastapi-agent
uv sync
```

### 3 — Configure environment

```bash
cp .env.example .env
# Edit .env — add your free HuggingFace token
# Get one at: https://huggingface.co/settings/tokens
```

### 4 — Run the server

```bash
uv run uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

---

## API Reference

### `POST /agent/upload`

Upload a PDF, TXT, MD, or CSV file. Returns a `session_id`.

```bash
curl -X POST http://localhost:8000/agent/upload \
  -F "file=@/path/to/document.pdf"
```

**Response 201:**
```json
{
  "session_id": "a1b2c3d4-...",
  "filename": "document.pdf",
  "chunks_loaded": 42,
  "message": "Document indexed successfully. Use session_id 'a1b2c3d4-...' in /agent/chat."
}
```

---

### `POST /agent/chat`

Ask a question. Optionally pass `session_id` to ground the answer in your document.

```bash
# With document context
curl -X POST http://localhost:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the main topic?", "session_id": "a1b2c3d4-..."}'

# Without document (model answers from general knowledge)
curl -X POST http://localhost:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the capital of France?"}'
```

**Response 200:**
```json
{
  "reply": "The document primarily covers ...",
  "source": "document",
  "session_id": "a1b2c3d4-...",
  "chunks_used": 3
}
```

| `source` | Meaning |
|---|---|
| `"document"` | Answer grounded in uploaded document (FAISS hit) |
| `"model"` | Answer from HuggingFace model general knowledge |

---

### `DELETE /agent/session/{session_id}`

Free the in-memory FAISS index for a session.

```bash
curl -X DELETE http://localhost:8000/agent/session/a1b2c3d4-...
```

---

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
  "embeddings_model": "sentence-transformers/all-MiniLM-L6-v2",
  "active_sessions": 2,
  "version": "1.0.0"
}
```

---

## Test Plan

| # | Test Case | Input | Expected |
|---|-----------|-------|----------|
| TC-01 | Health check | `GET /health` | 200, `status=ok` |
| TC-02 | Upload valid TXT | POST `.txt` file | 201, `session_id` + `chunks_loaded > 0` |
| TC-03 | Upload unsupported type | POST `.xlsx` | 415 |
| TC-04 | Upload empty file | POST empty bytes | 400 |
| TC-05 | Chat — no session (model fallback) | message only | 200, `source=model`, `chunks_used=0` |
| TC-06 | Chat — question IN document | message + valid `session_id` | 200, `source=document`, `chunks_used > 0` |
| TC-07 | Chat — question NOT in document | off-topic message + `session_id` | 200, `source=model` (FAISS score above threshold) |
| TC-08 | Chat — unknown session_id | random UUID | 200, `source=model` (graceful fallback) |
| TC-09 | Delete valid session | `DELETE /agent/session/{id}` | 200, `cleared=true` |
| TC-10 | Delete non-existent session | `DELETE /agent/session/bad-id` | 404 |
| TC-11 | Empty message validation | `{"message": ""}` | 422 Pydantic error |
| TC-12 | Health reflects session count | upload → health → delete → health | count increments & decrements |

### Run tests

```bash
uv run pytest tests/ -v
```

### Manual end-to-end walkthrough

```bash
# 1. Start server
uv run uvicorn app.main:app --reload --port 8000

# 2. Upload a document
SESSION=$(curl -s -X POST http://localhost:8000/agent/upload \
  -F "file=@my_doc.pdf" | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")

echo "Session: $SESSION"

# 3. Ask something FROM the document
curl -s -X POST http://localhost:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Summarise the document\", \"session_id\": \"$SESSION\"}" | python3 -m json.tool

# 4. Ask something NOT in the document
curl -s -X POST http://localhost:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"What year was Python created?\", \"session_id\": \"$SESSION\"}" | python3 -m json.tool

# 5. Clear the session
curl -s -X DELETE http://localhost:8000/agent/session/$SESSION | python3 -m json.tool
```

---

## Free Models (HuggingFace Inference API)

All models below work with a free HF account token:

| Model ID | Quality | Notes |
|---|---|---|
| `mistralai/Mistral-7B-Instruct-v0.2` | ⭐⭐⭐⭐ | Best quality, recommended |
| `HuggingFaceH4/zephyr-7b-beta` | ⭐⭐⭐⭐ | Good instruction-following |
| `google/flan-t5-xxl` | ⭐⭐⭐ | Smaller, faster, weaker |

Change the model by setting `HF_MODEL_ID=...` in `.env`.
