"""
Pipeline Orchestrator
Connects STT -> LLM (Intent Detection + Response) -> TTS
with microsecond latency tracking, session state persistence, and sentence-level audio clipping.
"""

import re
import logging
from typing import AsyncGenerator, Dict, Any, Optional
from app.core.session import session_manager, SessionState
from app.core.latency import LatencyTracker
from app.pipeline.stt import stt_service
from app.pipeline.llm import llm_service
from app.pipeline.tts import tts_service

logger = logging.getLogger("voicebot.orchestrator")


class PipelineEvent:
    def __init__(self, event_type: str, data: Any):
        self.event_type = event_type
        self.data = data


class VoicePipelineOrchestrator:
    def __init__(self):
        self.stt = stt_service
        self.llm = llm_service
        self.tts = tts_service
        self.session_mgr = session_manager

    async def process_audio_turn(
        self,
        audio_bytes: bytes,
        session_id: str,
        audio_format: str = "audio/webm"
    ) -> AsyncGenerator[PipelineEvent, None]:
        """
        Executes the end-to-end voice pipeline:
        1. Latency mark: audio received
        2. STT: transcribe audio
        3. Session: record user message
        4. LLM: stream response & extract intent
        5. TTS: synthesize sentence by sentence for minimal TTFA (Time to First Audio)
        6. Latency mark: TTFT, TTFA, and Total E2E
        """
        tracker = LatencyTracker()
        tracker.mark_audio_received()

        session = self.session_mgr.get_or_create_session(session_id)

        # --- Stage 1: Speech-to-Text ---
        tracker.mark_stt_start()
        try:
            filename = f"audio.{'webm' if 'webm' in audio_format else 'wav'}"
            user_transcript = await self.stt.transcribe(audio_bytes, filename=filename)
        except Exception as e:
            logger.error(f"STT failed: {e}")
            yield PipelineEvent("error", {"stage": "stt", "message": str(e)})
            return
        finally:
            tracker.mark_stt_end()

        if not user_transcript or not user_transcript.strip():
            logger.warning("Empty STT transcript received.")
            yield PipelineEvent("stt_empty", {"message": "No speech detected"})
            return

        yield PipelineEvent("stt_transcript", {
            "text": user_transcript,
            "stt_ms": tracker.get_metrics()["stt_ms"]
        })

        # Save user message in session
        session.add_message(role="user", content=user_transcript)

        # --- Stage 2 & 3: LLM Intent + Streaming Response -> TTS Pipeline ---
        async for event in self._process_llm_and_tts(
            user_text=user_transcript,
            session=session,
            tracker=tracker
        ):
            yield event

    async def process_text_turn(
        self,
        user_text: str,
        session_id: str
    ) -> AsyncGenerator[PipelineEvent, None]:
        """Direct text-input turn (useful for testing or hybrid text/voice clients)."""
        tracker = LatencyTracker()
        tracker.mark_audio_received()
        session = self.session_mgr.get_or_create_session(session_id)
        session.add_message(role="user", content=user_text)

        yield PipelineEvent("stt_transcript", {
            "text": user_text,
            "stt_ms": 0.0
        })

        async for event in self._process_llm_and_tts(
            user_text=user_text,
            session=session,
            tracker=tracker
        ):
            yield event

    async def _process_llm_and_tts(
        self,
        user_text: str,
        session: SessionState,
        tracker: LatencyTracker
    ) -> AsyncGenerator[PipelineEvent, None]:
        history = session.get_history_for_llm(max_turns=6)
        if history and history[-1]["role"] == "user":
            history = history[:-1]

        tracker.mark_llm_start()
        tracker.mark_tts_start()

        full_response_text = ""
        detected_intent: Optional[str] = None
        sentence_buffer = ""
        # Sentence boundary: punctuation followed by space
        sentence_end_pattern = re.compile(r'([.?!]+(?:\s+|$))')

        try:
            async for token_chunk, intent in self.llm.stream_response(history, user_text):
                tracker.mark_llm_first_token()

                if intent and not detected_intent:
                    detected_intent = intent
                    session.last_intent = detected_intent
                    yield PipelineEvent("intent_detected", {"intent": detected_intent})

                if not token_chunk:
                    continue

                full_response_text += token_chunk
                sentence_buffer += token_chunk
                yield PipelineEvent("llm_token", {"token": token_chunk})

                # Split on sentence boundaries if buffer has substantial length
                if len(sentence_buffer) > 25:
                    parts = sentence_end_pattern.split(sentence_buffer)
                    if len(parts) > 2:
                        sentence_to_speak = "".join(parts[:-1]).strip()
                        sentence_buffer = parts[-1]

                        if sentence_to_speak:
                            clip = await self.tts.synthesize_clip(sentence_to_speak)
                            if clip:
                                tracker.mark_tts_first_chunk()
                                yield PipelineEvent("audio_chunk", clip)

            # End of LLM stream
            tracker.mark_llm_end()

            # Synthesize any remaining sentence buffer
            remaining_text = sentence_buffer.strip()
            if remaining_text:
                clip = await self.tts.synthesize_clip(remaining_text)
                if clip:
                    tracker.mark_tts_first_chunk()
                    yield PipelineEvent("audio_chunk", clip)

            tracker.mark_tts_end()

        except Exception as e:
            logger.error(f"LLM or TTS streaming error: {e}", exc_info=True)
            yield PipelineEvent("error", {"stage": "llm_tts", "message": str(e)})
            return

        # Compute full latency metrics
        latency_metrics = tracker.get_metrics()
        logger.info(tracker.summary())

        # Save assistant message into session
        session.add_message(
            role="assistant",
            content=full_response_text.strip(),
            intent=detected_intent or "General",
            latency_ms=latency_metrics
        )

        yield PipelineEvent("turn_complete", {
            "session_id": session.session_id,
            "full_response": full_response_text.strip(),
            "intent": detected_intent or "General",
            "turn_count": session.turn_count,
            "latency": latency_metrics
        })


pipeline_orchestrator = VoicePipelineOrchestrator()
