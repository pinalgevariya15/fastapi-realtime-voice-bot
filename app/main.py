"""
FastAPI RealTime Voice Bot Application Entrypoint
"""

import os
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.config import settings
from app.core.session import session_manager
from app.websocket.audio_handler import router as ws_router

app = FastAPI(
    title="Real-Time Voice Bot API",
    description="End-to-End Real-Time Voice Pipeline: STT -> LLM (Intent + Response) -> TTS over WebSocket with Latency Instrumentation",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register WebSocket routes
app.include_router(ws_router)


@app.get("/health")
async def health_check():
    """Health check endpoint displaying system status and configured providers."""
    return {
        "status": "healthy",
        "stt_provider": settings.STT_PROVIDER,
        "stt_model": settings.STT_MODEL,
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": settings.LLM_MODEL,
        "tts_provider": settings.TTS_PROVIDER,
        "tts_voice": settings.TTS_VOICE,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/api/sessions")
async def list_sessions():
    """Returns all active session IDs in memory."""
    return {"sessions": session_manager.list_sessions()}


@app.get("/api/sessions/{session_id}")
async def get_session_details(session_id: str):
    """Retrieves session state, message turns, intent, and latency metrics."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session.session_id,
        "turn_count": session.turn_count,
        "last_intent": session.last_intent,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "intent": m.intent,
                "timestamp": m.timestamp.isoformat(),
                "latency_ms": m.latency_ms
            }
            for m in session.messages
        ]
    }


# Static files and Web UI client
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    """Serves the voice bot web client."""
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {
        "message": "Real-Time Voice Bot API is running. Connect to WebSocket at /ws/{session_id}",
        "docs": "/docs",
        "health": "/health"
    }
