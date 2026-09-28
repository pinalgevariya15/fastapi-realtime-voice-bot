"""
Speech-to-Text (STT) Module
Transcribes user audio buffers into text using Groq Whisper API (or fallback).
"""

import io
import logging
from typing import Optional
from app.config import settings

logger = logging.getLogger("voicebot.stt")


class STTService:
    def __init__(self):
        self.provider = settings.STT_PROVIDER
        self.model = settings.STT_MODEL
        self._groq_client = None
        self._openai_client = None

    def _get_groq_client(self):
        if self._groq_client is None:
            from groq import AsyncGroq
            self._groq_client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        return self._groq_client

    def _get_openai_client(self):
        if self._openai_client is None:
            from openai import AsyncOpenAI
            self._openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        return self._openai_client

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        content_type: str = "audio/wav"
    ) -> str:
        """
        Transcribes audio bytes into text.
        audio_bytes can be WAV, WebM, MP3, etc.
        """
        if not audio_bytes or len(audio_bytes) < 100:
            logger.warning("Audio buffer is empty or too short.")
            return ""

        if self.provider == "groq":
            return await self._transcribe_groq(audio_bytes, filename)
        elif self.provider == "openai":
            return await self._transcribe_openai(audio_bytes, filename)
        elif self.provider == "mock":
            return "Hello, this is a simulated transcription for testing."
        else:
            raise ValueError(f"Unsupported STT provider: {self.provider}")

    async def _transcribe_groq(self, audio_bytes: bytes, filename: str) -> str:
        client = self._get_groq_client()
        file_tuple = (filename, audio_bytes)
        try:
            transcription = await client.audio.transcriptions.create(
                file=file_tuple,
                model=self.model,
                response_format="json",
                language="en",
                temperature=0.0
            )
            text = transcription.text.strip()
            logger.info(f"Groq STT transcription completed: '{text}'")
            return text
        except Exception as e:
            logger.error(f"Groq STT transcription error: {e}", exc_info=True)
            raise

    async def _transcribe_openai(self, audio_bytes: bytes, filename: str) -> str:
        client = self._get_openai_client()
        file_tuple = (filename, audio_bytes)
        try:
            transcription = await client.audio.transcriptions.create(
                file=file_tuple,
                model=self.model or "whisper-1",
                response_format="json"
            )
            text = transcription.text.strip()
            logger.info(f"OpenAI STT transcription completed: '{text}'")
            return text
        except Exception as e:
            logger.error(f"OpenAI STT transcription error: {e}", exc_info=True)
            raise


stt_service = STTService()
