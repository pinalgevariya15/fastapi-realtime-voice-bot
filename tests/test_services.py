"""Tests for STT, LLM, and TTS services."""

import pytest
from app.pipeline.stt import STTService
from app.pipeline.llm import LLMService
from app.pipeline.tts import TTSService
from app.config import settings


@pytest.mark.asyncio
async def test_mock_stt():
    stt = STTService()
    stt.provider = "mock"
    text = await stt.transcribe(b"fake audio data" * 10)
    assert "simulated transcription" in text


@pytest.mark.asyncio
async def test_mock_llm_stream():
    llm = LLMService()
    llm.provider = "mock"
    chunks = []
    intents = []
    async for text_chunk, intent in llm.stream_response([], "Hello there"):
        if intent:
            intents.append(intent)
        if text_chunk:
            chunks.append(text_chunk)

    assert "Greeting" in intents
    full_text = "".join(chunks)
    assert "Hello there" in full_text


@pytest.mark.asyncio
async def test_edge_tts_synthesis():
    tts = TTSService()
    tts.provider = "edge-tts"
    audio_chunks = []
    async for chunk in tts.stream_audio("Testing real-time voice synthesis."):
        audio_chunks.append(chunk)

    assert len(audio_chunks) > 0
    total_bytes = sum(len(c) for c in audio_chunks)
    assert total_bytes > 500  # Synthesized MP3 audio received


@pytest.mark.asyncio
async def test_groq_llm_live():
    # Only run if GROQ_API_KEY is present
    if not settings.GROQ_API_KEY:
        pytest.skip("GROQ_API_KEY not set")

    llm = LLMService()
    chunks = []
    detected_intent = None

    async for text_chunk, intent in llm.stream_response([], "What is the highest mountain on Earth?"):
        if intent and not detected_intent:
            detected_intent = intent
        if text_chunk:
            chunks.append(text_chunk)

    full_response = "".join(chunks)
    print(f"\n[Live Groq LLM] Intent: {detected_intent} | Response: {full_response}")
    assert detected_intent is not None
    assert len(full_response) > 5
    assert "Everest" in full_response or "Mount" in full_response
