"""
Text-to-Speech (TTS) Module
Converts textual responses into natural audio using Edge-TTS (or OpenAI TTS).
Produces clean, self-contained MP3 audio clips for gapless playback.
"""

import io
import logging
from typing import AsyncGenerator
from app.config import settings

logger = logging.getLogger("voicebot.tts")


class TTSService:
    def __init__(self):
        self.provider = settings.TTS_PROVIDER
        self.voice = settings.TTS_VOICE
        self._openai_client = None

    def _get_openai_client(self):
        if self._openai_client is None:
            from openai import AsyncOpenAI
            self._openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        return self._openai_client

    async def synthesize_clip(self, text: str) -> bytes:
        """
        Synthesizes text into a complete, valid MP3 audio clip.
        Guarantees complete MP3 headers and frames so audio does not stutter or break.
        """
        clean_text = text.strip()
        if not clean_text:
            return b""

        if self.provider == "edge-tts":
            return await self._synthesize_edge_tts(clean_text)
        elif self.provider == "openai":
            return await self._synthesize_openai(clean_text)
        elif self.provider == "mock":
            return b"\xFF\xFB\x90\x64" + b"\x00" * 512
        else:
            raise ValueError(f"Unsupported TTS provider: {self.provider}")

    async def stream_audio(self, text: str) -> AsyncGenerator[bytes, None]:
        """Backward-compatible generator yielding audio clip."""
        clip = await self.synthesize_clip(text)
        if clip:
            yield clip

    async def _synthesize_edge_tts(self, text: str) -> bytes:
        import edge_tts
        try:
            communicate = edge_tts.Communicate(text, self.voice)
            audio_buffer = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buffer.extend(chunk["data"])
            return bytes(audio_buffer)
        except Exception as e:
            logger.error(f"Edge-TTS synthesis error: {e}", exc_info=True)
            raise

    async def _synthesize_openai(self, text: str) -> bytes:
        client = self._get_openai_client()
        try:
            response = await client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text,
                response_format="mp3"
            )
            return await response.aread()
        except Exception as e:
            logger.error(f"OpenAI TTS synthesis error: {e}", exc_info=True)
            raise


tts_service = TTSService()
