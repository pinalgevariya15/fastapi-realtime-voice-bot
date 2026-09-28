"""
Latency Profiling & Reporting Module
Tracks microsecond-precision timestamps across each stage:
STT -> LLM (TTFT + Total) -> TTS (First Chunk + Total) -> Total End-to-End
"""

import time
from typing import Dict, Optional


class LatencyTracker:
    def __init__(self):
        # High resolution monotonic timestamps
        self.t_audio_received: float = time.perf_counter()
        self.t_stt_start: Optional[float] = None
        self.t_stt_end: Optional[float] = None
        self.t_llm_start: Optional[float] = None
        self.t_llm_first_token: Optional[float] = None
        self.t_llm_end: Optional[float] = None
        self.t_tts_start: Optional[float] = None
        self.t_tts_first_chunk: Optional[float] = None
        self.t_tts_end: Optional[float] = None

    def mark_audio_received(self):
        self.t_audio_received = time.perf_counter()

    def mark_stt_start(self):
        self.t_stt_start = time.perf_counter()

    def mark_stt_end(self):
        self.t_stt_end = time.perf_counter()

    def mark_llm_start(self):
        self.t_llm_start = time.perf_counter()

    def mark_llm_first_token(self):
        if self.t_llm_first_token is None:
            self.t_llm_first_token = time.perf_counter()

    def mark_llm_end(self):
        self.t_llm_end = time.perf_counter()

    def mark_tts_start(self):
        self.t_tts_start = time.perf_counter()

    def mark_tts_first_chunk(self):
        if self.t_tts_first_chunk is None:
            self.t_tts_first_chunk = time.perf_counter()

    def mark_tts_end(self):
        self.t_tts_end = time.perf_counter()

    def get_metrics(self) -> Dict[str, float]:
        """Calculates stage durations in milliseconds."""
        now = time.perf_counter()

        def diff_ms(start: Optional[float], end: Optional[float]) -> float:
            if start is not None and end is not None:
                return round((end - start) * 1000.0, 2)
            return 0.0

        stt_ms = diff_ms(self.t_stt_start, self.t_stt_end)
        llm_ttft_ms = diff_ms(self.t_llm_start, self.t_llm_first_token)
        llm_total_ms = diff_ms(self.t_llm_start, self.t_llm_end)
        tts_first_chunk_ms = diff_ms(self.t_tts_start, self.t_tts_first_chunk)
        tts_total_ms = diff_ms(self.t_tts_start, self.t_tts_end)

        # End-to-end to first audible sound chunk
        first_audio_ts = self.t_tts_first_chunk or self.t_tts_end or now
        e2e_first_audio_ms = round((first_audio_ts - self.t_audio_received) * 1000.0, 2)

        # Full end-to-end completion
        final_ts = self.t_tts_end or now
        e2e_total_ms = round((final_ts - self.t_audio_received) * 1000.0, 2)

        return {
            "stt_ms": stt_ms,
            "llm_ttft_ms": llm_ttft_ms,
            "llm_total_ms": llm_total_ms,
            "tts_first_chunk_ms": tts_first_chunk_ms,
            "tts_total_ms": tts_total_ms,
            "e2e_first_audio_ms": e2e_first_audio_ms,
            "e2e_total_ms": e2e_total_ms,
        }

    def summary(self) -> str:
        m = self.get_metrics()
        return (
            f"[Latency Report] "
            f"STT: {m['stt_ms']}ms | "
            f"LLM TTFT: {m['llm_ttft_ms']}ms (Total: {m['llm_total_ms']}ms) | "
            f"TTS First: {m['tts_first_chunk_ms']}ms (Total: {m['tts_total_ms']}ms) | "
            f"E2E First Audio: {m['e2e_first_audio_ms']}ms | "
            f"E2E Total: {m['e2e_total_ms']}ms"
        )
