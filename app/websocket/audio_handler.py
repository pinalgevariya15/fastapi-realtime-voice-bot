"""
FastAPI WebSocket Audio Handler
Manages full-duplex WebSocket connections for streaming voice in/out.
"""

import base64
import json
import logging
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.core.session import session_manager
from app.pipeline.orchestrator import pipeline_orchestrator

logger = logging.getLogger("voicebot.websocket")
router = APIRouter()


@router.websocket("/ws/{session_id}")
@router.websocket("/ws")
async def websocket_audio_endpoint(
    websocket: WebSocket,
    session_id: Optional[str] = None
):
    await websocket.accept()

    # If no session_id in path or default
    if not session_id or session_id == "new":
        session = session_manager.get_or_create_session()
        session_id = session.session_id
    else:
        session = session_manager.get_or_create_session(session_id)

    logger.info(f"WebSocket client connected. Session ID: {session_id}")

    # Send initial connection acknowledgment
    await websocket.send_json({
        "type": "connected",
        "session_id": session_id,
        "turn_count": session.turn_count,
        "last_intent": session.last_intent,
        "message": "Connected to RealTime Voice Bot pipeline."
    })

    # Audio accumulation buffer for the current turn
    audio_buffer = bytearray()
    audio_format = "audio/webm"

    try:
        while True:
            # Receive either text (JSON) or binary (raw audio bytes)
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                logger.info(f"Client disconnected cleanly for session: {session_id}")
                break

            if "bytes" in message and message["bytes"]:
                # Binary audio chunk
                audio_buffer.extend(message["bytes"])

            elif "text" in message and message["text"]:
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "message": "Invalid JSON format."})
                    continue

                msg_type = data.get("type")

                if msg_type == "audio_start":
                    audio_buffer.clear()
                    audio_format = data.get("format", "audio/webm")
                    await websocket.send_json({"type": "audio_started"})

                elif msg_type == "audio_chunk":
                    raw_b64 = data.get("data", "")
                    if raw_b64:
                        audio_buffer.extend(base64.b64decode(raw_b64))

                elif msg_type == "audio_end":
                    # User completed speaking turn, process audio buffer
                    if len(audio_buffer) == 0:
                        await websocket.send_json({"type": "warning", "message": "No audio received."})
                        continue

                    # Stream pipeline events back to client
                    await _execute_pipeline_turn(
                        websocket=websocket,
                        turn_coro=pipeline_orchestrator.process_audio_turn(
                            audio_bytes=bytes(audio_buffer),
                            session_id=session_id,
                            audio_format=audio_format
                        )
                    )
                    audio_buffer.clear()

                elif msg_type == "text_query":
                    # Direct text-based question
                    query = data.get("text", "").strip()
                    if query:
                        await _execute_pipeline_turn(
                            websocket=websocket,
                            turn_coro=pipeline_orchestrator.process_text_turn(
                                user_text=query,
                                session_id=session_id
                            )
                        )

                elif msg_type == "clear_session":
                    session.clear()
                    await websocket.send_json({
                        "type": "session_cleared",
                        "session_id": session_id,
                        "turn_count": 0
                    })

                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected for session: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}", exc_info=True)


async def _execute_pipeline_turn(websocket: WebSocket, turn_coro):
    """Iterates through pipeline generator events and sends them to WebSocket."""
    async for event in turn_coro:
        if event.event_type == "audio_chunk":
            # Send audio chunks both as binary bytes for Web Audio API playback
            # and notify client
            await websocket.send_bytes(event.data)
        else:
            await websocket.send_json({
                "type": event.event_type,
                **event.data
            })
