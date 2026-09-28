"""Unit tests for core session and latency tracking modules."""

import time
import pytest
from app.core.session import SessionManager, SessionState
from app.core.latency import LatencyTracker


def test_session_creation_and_state():
    manager = SessionManager()
    session = manager.get_or_create_session("test-session-1")
    assert session.session_id == "test-session-1"
    assert session.turn_count == 0
    assert len(session.messages) == 0

    # Add user message
    session.add_message(role="user", content="Hello, who are you?")
    assert len(session.messages) == 1
    assert session.messages[0].content == "Hello, who are you?"
    assert session.turn_count == 0

    # Add assistant message with intent and latency
    session.add_message(
        role="assistant",
        content="I am your real-time voice assistant.",
        intent="greeting",
        latency_ms={"e2e_first_audio_ms": 350.5}
    )
    assert len(session.messages) == 2
    assert session.turn_count == 1
    assert session.last_intent == "greeting"

    # Context history for LLM
    history = session.get_history_for_llm(max_turns=5)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"

    # Clear session
    session.clear()
    assert len(session.messages) == 0
    assert session.turn_count == 0
    assert session.last_intent == "initial"


def test_latency_tracker():
    tracker = LatencyTracker()
    tracker.mark_audio_received()
    time.sleep(0.01)

    tracker.mark_stt_start()
    time.sleep(0.02)
    tracker.mark_stt_end()

    tracker.mark_llm_start()
    time.sleep(0.015)
    tracker.mark_llm_first_token()
    time.sleep(0.02)
    tracker.mark_llm_end()

    tracker.mark_tts_start()
    time.sleep(0.01)
    tracker.mark_tts_first_chunk()
    time.sleep(0.01)
    tracker.mark_tts_end()

    metrics = tracker.get_metrics()
    assert metrics["stt_ms"] >= 15.0
    assert metrics["llm_ttft_ms"] >= 10.0
    assert metrics["llm_total_ms"] >= metrics["llm_ttft_ms"]
    assert metrics["tts_first_chunk_ms"] >= 8.0
    assert metrics["tts_total_ms"] >= metrics["tts_first_chunk_ms"]
    assert metrics["e2e_first_audio_ms"] > 0
    assert metrics["e2e_total_ms"] >= metrics["e2e_first_audio_ms"]

    summary = tracker.summary()
    assert "[Latency Report]" in summary
    assert "STT:" in summary
    assert "LLM TTFT:" in summary
