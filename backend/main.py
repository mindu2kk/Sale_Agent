"""
FastAPI API Gateway — AI Sales Copilot

Endpoints:
  POST /api/chat/threads        — create new conversation thread
  POST /api/chat                — send message, stream SSE response
  GET  /api/chat/threads/{id}/messages — get conversation history
  GET  /health                  — simple liveness check (no auth)
  GET  /health/chat             — deep health check (no auth)
  GET  /metrics                 — runtime metrics (no auth)
"""

import hmac
import json
import logging
import os
import re
import time
import unicodedata
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","message":"%(message)s"}',
)
logger = logging.getLogger("backend.api")

# ── Config ────────────────────────────────────────────────────────────────────

API_TOKEN = os.getenv("API_BEARER_TOKEN", "dev-token-123")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

# UUID v4 regex for validation
_UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)

# ── In-memory metrics ─────────────────────────────────────────────────────────

_start_time = time.time()
_total_requests = 0
_active_streams = 0
_metrics_cache: Optional[dict] = None
_metrics_cache_time = 0.0

# ── Startup / Shutdown ────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Init DB only. AI workflow will load lazily on first request (~60s wait)."""
    from backend.database import init_db

    logger.info("Starting up API Gateway...")
    await init_db()
    logger.info("Database initialised ✓")

    # LAZY LOADING: Workflow sẽ load khi có request đầu tiên
    # Giúp backend khởi động ngay lập tức thay vì chờ 60s
    logger.info("API Gateway ready — listening on http://0.0.0.0:8000")
    logger.info("AI workflow will load on first request (expect ~60s delay)")
    
    yield
    logger.info("Shutting down...")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Sales Copilot API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — MUST be added before any routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        FRONTEND_URL,
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Thread-ID", "X-Correlation-ID"],
)

# ── Auth ──────────────────────────────────────────────────────────────────────

security = HTTPBearer(auto_error=False)


def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Dependency that validates Bearer token using constant-time comparison."""
    if credentials is None or not hmac.compare_digest(credentials.credentials, API_TOKEN):
        raise HTTPException(status_code=401, detail={"error": "unauthorized"})
    return credentials.credentials


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_uuid4(value: str) -> bool:
    return bool(_UUID4_RE.match(value.lower()))


def _sanitize_message(text: str) -> str:
    """
    Remove Unicode control chars (except tab, newline, CR) and
    escape HTML special characters.
    """
    # Remove control characters
    sanitized = "".join(
        ch for ch in text
        if unicodedata.category(ch) != "Cc" or ch in "\t\n\r"
    )
    # Escape HTML specials
    sanitized = (
        sanitized
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )
    return sanitized


def _log_request(request: Request, status: int, thread_id: str = "", latency_ms: float = 0):
    logger.info(
        json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": request.method,
            "path": request.url.path,
            "status_code": status,
            "latency_ms": round(latency_ms, 2),
            "thread_id": thread_id,
        })
    )


# ── Health endpoints (no auth) ────────────────────────────────────────────────

@app.get("/health", tags=["health"])
async def health():
    """Simple liveness — always 200 if process is alive. Used for Docker healthcheck."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/health/chat", tags=["health"])
async def health_chat():
    """Deep health check — verifies DB and AI workflow are reachable."""
    failed = []
    details = {}

    # Check DB
    try:
        from backend.database import engine
        from sqlalchemy import text as sql_text
        async with engine.connect() as conn:
            await conn.execute(sql_text("SELECT 1"))
        details["database"] = "healthy"
    except Exception as exc:
        details["database"] = f"unhealthy: {exc}"
        failed.append("database")

    # Check workflow module (don't trigger load, just check if already loaded)
    try:
        from backend.workflow_factory import _workflow
        if _workflow is not None:
            details["langgraph_backend"] = "healthy"
        else:
            details["langgraph_backend"] = "loading (wait ~60s)"
            # Don't add to failed — it's still starting up, not broken
    except Exception as exc:
        details["langgraph_backend"] = f"unhealthy: {exc}"
        failed.append("langgraph_backend")

    status_code = 503 if failed else 200
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "degraded" if failed else "healthy",
            "dependencies": details,
            "failed": failed,
        },
    )


@app.get("/metrics", tags=["observability"])
async def metrics():
    """Runtime metrics — cached for 5 seconds."""
    global _metrics_cache, _metrics_cache_time

    now = time.time()
    if _metrics_cache and (now - _metrics_cache_time) < 5:
        return _metrics_cache

    _metrics_cache = {
        "active_streams": _active_streams,
        "total_requests": _total_requests,
        "uptime_seconds": int(now - _start_time),
    }
    _metrics_cache_time = now
    return _metrics_cache


# ── Thread management ─────────────────────────────────────────────────────────

@app.post("/api/chat/threads", status_code=201, tags=["chat"])
async def create_thread(
    request: Request,
    token: str = Depends(verify_token),
):
    """Create a new conversation thread. Returns thread_id."""
    global _total_requests
    _total_requests += 1
    t0 = time.time()

    from backend.database import get_db
    from backend import crud

    async with __import__("backend.database", fromlist=["AsyncSessionLocal"]).AsyncSessionLocal() as db:
        thread = await crud.create_thread(db)

    _log_request(request, 201, thread_id=thread.id, latency_ms=(time.time() - t0) * 1000)
    return {
        "thread_id": thread.id,
        "created_at": thread.created_at.isoformat(),
        "status": thread.status,
    }


@app.get("/api/chat/threads/{thread_id}/messages", tags=["chat"])
async def get_messages(
    thread_id: str,
    request: Request,
    limit: int = Query(default=50, ge=1, le=100),
    before: Optional[str] = Query(default=None),
    token: str = Depends(verify_token),
):
    """Get conversation history for a thread with cursor pagination."""
    global _total_requests
    _total_requests += 1
    t0 = time.time()

    # Validate format before hitting DB
    if not _validate_uuid4(thread_id):
        raise HTTPException(status_code=400, detail={"error": "invalid_thread_id_format"})

    from backend.database import AsyncSessionLocal
    from backend import crud

    async with AsyncSessionLocal() as db:
        thread = await crud.get_thread(db, thread_id)
        if thread is None:
            raise HTTPException(status_code=404, detail={"error": "thread_not_found"})

        messages, has_more = await crud.get_messages(db, thread_id, limit=limit, before_id=before)

    result = {
        "thread_id": thread_id,
        "messages": [
            {
                "id": m.id,
                "thread_id": m.thread_id,
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat(),
            }
            for m in messages
        ],
        "has_more": has_more,
    }

    _log_request(request, 200, thread_id=thread_id, latency_ms=(time.time() - t0) * 1000)
    return result


# ── Chat SSE endpoint ─────────────────────────────────────────────────────────

@app.post("/api/chat", tags=["chat"])
async def chat(
    request: Request,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token),
):
    """
    Send a message and receive streaming SSE response.

    Body: {"message": "...", "thread_id": "..."}

    Returns text/event-stream with events:
      data: {"type": "token", "data": "..."}
      data: {"type": "stream_end", "thread_id": "...", "workflow_status": "..."}
      data: {"type": "error", "message": "..."}
    """
    global _total_requests, _active_streams
    _total_requests += 1
    t0 = time.time()

    # Parse body manually (can't use Pydantic model + StreamingResponse together cleanly)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=422, detail={"error": "validation_error", "details": "Invalid JSON body"})

    raw_message = body.get("message", "")
    thread_id = body.get("thread_id", "")

    # Validate thread_id format
    if not thread_id or not _validate_uuid4(thread_id):
        raise HTTPException(status_code=400, detail={"error": "invalid_thread_id_format"})

    # Sanitize and validate message
    message = _sanitize_message(raw_message)
    if not message.strip():
        raise HTTPException(status_code=422, detail={"error": "validation_error", "details": "message: cannot be empty"})
    if len(message) > 10_000:
        raise HTTPException(status_code=422, detail={"error": "validation_error", "details": "message: exceeds 10000 characters"})

    # Verify thread exists
    from backend.database import AsyncSessionLocal
    from backend import crud

    async with AsyncSessionLocal() as db:
        thread = await crud.get_thread(db, thread_id)
        if thread is None:
            raise HTTPException(status_code=404, detail={"error": "thread_not_found"})
        if thread.status == "expired":
            raise HTTPException(status_code=410, detail={"error": "thread_expired"})

    # Save user message immediately (before stream starts)
    async with AsyncSessionLocal() as db:
        await crud.save_message(db, thread_id=thread_id, role="user", content=message)

    # Collect assistant response for background save
    assistant_chunks: list[str] = []
    workflow_status_holder: list[str] = ["unknown"]

    from backend.stream_relay import stream_workflow

    async def generate():
        nonlocal assistant_chunks
        _active_streams_inc()
        try:
            async for chunk in stream_workflow(message, thread_id):
                yield chunk
                # Parse to collect assistant content and status
                try:
                    line = chunk.strip()
                    if line.startswith("data: "):
                        event = json.loads(line[6:])
                        if event.get("type") == "token":
                            assistant_chunks.append(event.get("data", ""))
                        elif event.get("type") == "stream_end":
                            workflow_status_holder[0] = event.get("workflow_status", "unknown")
                except Exception:
                    pass
        finally:
            _active_streams_dec()
            _log_request(request, 200, thread_id=thread_id, latency_ms=(time.time() - t0) * 1000)

    async def _save_assistant_message():
        """Background task: save the complete assistant response after stream ends."""
        full_response = "".join(assistant_chunks)
        if full_response:
            async with AsyncSessionLocal() as db:
                await crud.save_message(db, thread_id=thread_id, role="assistant", content=full_response)
                # Mark thread escalated if needed
                if workflow_status_holder[0] == "escalated":
                    await crud.update_thread_metadata(db, thread_id=thread_id, escalated=True)

    background_tasks.add_task(_save_assistant_message)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable Nginx buffering for SSE
            "Connection": "keep-alive",
        },
    )


# ── Metrics helpers ───────────────────────────────────────────────────────────

def _active_streams_inc():
    global _active_streams
    _active_streams += 1
    if _active_streams > 100:
        logger.warning("High active stream count: %d", _active_streams)


def _active_streams_dec():
    global _active_streams
    _active_streams = max(0, _active_streams - 1)
