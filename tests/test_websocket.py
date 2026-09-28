"""Tests for FastAPI endpoints, WebSocket streaming, and pipeline orchestration."""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "stt_provider" in data
    assert "llm_provider" in data
    assert "tts_provider" in data


def test_websocket_ping_pong(client):
    with client.websocket_connect("/ws/session-ping") as ws:
        connected = ws.receive_json()
        assert connected["type"] == "connected"
        assert connected["session_id"] == "session-ping"

        ws.send_json({"type": "ping"})
        pong = ws.receive_json()
        assert pong["type"] == "pong"


def test_websocket_text_query_turn(client):
    with client.websocket_connect("/ws/session-turn-1") as ws:
        connected = ws.receive_json()
        assert connected["type"] == "connected"

        # Send a query
        ws.send_json({"type": "text_query", "text": "What is the capital of Italy?"})

        events = []
        audio_chunks_count = 0
        turn_complete = None

        while True:
            message = ws.receive()
            if "bytes" in message and message["bytes"]:
                audio_chunks_count += 1
            elif "text" in message and message["text"]:
                import json
                event = json.loads(message["text"])
                events.append(event)
                if event.get("type") == "turn_complete":
                    turn_complete = event
                    break

        # Verify pipeline events occurred
        event_types = [e.get("type") for e in events]
        assert "stt_transcript" in event_types
        assert "intent_detected" in event_types
        assert "llm_token" in event_types

        # Verify audio was streamed back
        assert audio_chunks_count > 0

        # Verify turn completion and latency reporting
        assert turn_complete is not None
        assert "latency" in turn_complete
        assert turn_complete["latency"]["e2e_first_audio_ms"] > 0
        assert turn_complete["intent"] in ["Question", "General"]

    # Verify session API has captured the dialogue
    res = client.get("/api/sessions/session-turn-1")
    assert res.status_code == 200
    session_data = res.json()
    assert session_data["turn_count"] == 1
    assert len(session_data["messages"]) == 2  # user + assistant
    assert session_data["messages"][0]["role"] == "user"
    assert session_data["messages"][1]["role"] == "assistant"
    assert session_data["messages"][1]["latency_ms"] is not None
