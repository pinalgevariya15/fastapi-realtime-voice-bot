
# Real-Time Voice Bot Pipeline

An end-to-end, low-latency conversational voice assistant pipeline built with **FastAPI**, **WebSockets**, **Groq** (Whisper STT + Llama/GPT-OSS LLM), and **Edge-TTS** streaming audio synthesis.

```
       [ Micro ] 
           │
           ▼
[ Frontend Web Client ] ──── WebSocket Duplex ──── [ FastAPI /ws/{session_id} ]
                                                             │
  ┌──────────────────────────────────────────────────────────┴──────────────────────────┐
  │ Pipeline Orchestration & Microsecond Latency Profiler                               │
  │                                                                                     │
  │  1. STT (Speech-to-Text)       ──> Groq Whisper (whisper-large-v3)                  │
  │  2. Session State Manager      ──> Multi-turn dialogue history & session memory     │
  │  3. LLM Engine                 ──> Single-pass Intent Detection + Token Streaming  │
  │  4. TTS (Text-to-Speech)       ──> Streaming Edge-TTS (en-US-JennyNeural)           │
  │  5. Latency Telemetry          ──> T_STT, T_TTFT, T_TTS, T_E2E (Perceived & Total)  │
  └──────────────────────────────────────────────────────────┬──────────────────────────┘
                                                             │
[ Speaker / Audio Playback ] ◄── Binary Audio Chunks ────────┘
```

---

## Key Features

1. **End-to-End Voice Pipeline**:
   - **Speech-to-Text (STT)**: Groq Whisper (`whisper-large-v3`) with sub-second audio transcription.
   - **LLM Engine**: High-speed conversational LLM (`openai/gpt-oss-20b` or `llama-3.3-70b-versatile` via Groq) performing simultaneous single-pass **intent detection** (e.g., `Greeting`, `Question`, `Command`, `ChitChat`, `Farewell`) and streaming conversational response generation.
   - **Text-to-Speech (TTS)**: Streaming **Edge-TTS** (`en-US-JennyNeural`) generating neural speech chunks in real time without requiring expensive cloud TTS API keys.
   - **Sentence-Level Pipelining**: Synthesizes the first sentence as soon as it is generated, yielding near-instant Time-to-First-Audio while the LLM continues generating subsequent sentences.

2. **Conversation State Management**:
   - Maintains multi-turn context, turn counters, dialogue history (`role`, `content`, `intent`, `latency_ms`), and sliding context windows across turns in `SessionManager`.

3. **Duplex WebSocket Audio Streaming**:
   - Exposed at `/ws/{session_id}` for bi-directional audio streaming.
   - Supports binary audio frames as well as structured JSON control frames (`audio_start`, `audio_chunk`, `audio_end`, `text_query`, `clear_session`).

4. **Microsecond Latency Profiler**:
   - Tracks:
     - $T_{\text{STT}}$: Audio received to text transcript.
     - $T_{\text{TTFT}}$: LLM Time-to-First-Token.
     - $T_{\text{LLM\_Total}}$: Full response generation duration.
     - $T_{\text{TTS\_First\_Audio}}$: First synthesized audio chunk latency.
     - $T_{\text{E2E\_First\_Audio}}$: Perceived speech-to-ear round-trip time.
     - $T_{\text{E2E\_Total}}$: Full pipeline cycle turnaround.

5. **Modern Web Client**:
   - Responsive dark-mode glassmorphism interface.
   - Web Audio API microphone recording with real-time waveform canvas visualizer.
   - Conversation history with color-coded intent badges.
   - Live telemetry dashboard updating latency cards after every turn.

---

## Directory Structure

```
RealTime_Voice_BOT/
├── app/
│   ├── main.py                  # FastAPI application, CORS, static routes, REST APIs
│   ├── config.py                # Pydantic environment & model configuration
│   ├── core/
│   │   ├── session.py           # Multi-turn conversation state & history manager
│   │   └── latency.py           # Latency profiler & stage timing instrumentation
│   ├── pipeline/
│   │   ├── stt.py               # Speech-to-Text service (Groq Whisper / OpenAI / Mock)
│   │   ├── llm.py               # LLM intent detection + streaming response generator
│   │   ├── tts.py               # Streaming Edge-TTS neural speech synthesis
│   │   └── orchestrator.py      # Pipeline coordinator & sentence streaming logic
│   ├── websocket/
│   │   └── audio_handler.py     # Duplex WebSocket streaming endpoint (/ws/{session_id})
│   └── static/
│       ├── index.html           # Web client UI
│       ├── css/style.css        # Dark-mode glassmorphic styling & animations
│       └── js/app.js            # Web Audio API mic capture, visualizer & audio queue
├── tests/
│   ├── test_core.py             # Unit tests for session & latency tracking
│   ├── test_services.py         # Tests for STT, LLM (live Groq), and TTS
│   └── test_websocket.py        # End-to-end WebSocket round-trip tests
├── .env.example                 # Example environment variables
├── requirements.txt             # Python dependencies
└── pytest.ini                   # Pytest configuration
```

---

## Quickstart & Setup

### 1. Environment Configuration
Ensure your Groq API key is present in `.env`:
```env
GROQ_API_KEY="your_groq_api_key_here"
STT_PROVIDER=groq
STT_MODEL=whisper-large-v3
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-20b
TTS_PROVIDER=edge-tts
TTS_VOICE=en-US-JennyNeural
```

### 2. Run Tests
```powershell
uv run pytest
```

### 3. Start Server
```powershell
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open in Browser
Visit **`http://localhost:8000`** to interact with the voice bot via microphone or text!

---

## WebSocket API Protocol

### Connection
- **URL**: `ws://localhost:8000/ws/{session_id}` (or `ws://localhost:8000/ws` for auto-generated session)

### Client -> Server Frames
| Event Type | Format | Description |
| :--- | :--- | :--- |
| `audio_start` | JSON `{"type": "audio_start", "format": "audio/webm"}` | Initiates a new user speech turn |
| Raw Audio | Binary `ArrayBuffer` | Audio bytes recorded from microphone |
| `audio_end` | JSON `{"type": "audio_end"}` | Finishes user speech turn and triggers pipeline |
| `text_query` | JSON `{"type": "text_query", "text": "..."}` | Direct text query for testing |
| `clear_session` | JSON `{"type": "clear_session"}` | Clears conversation state |

### Server -> Client Frames
| Event Type | Format | Description |
| :--- | :--- | :--- |
| `connected` | JSON | Connection confirmation with session ID |
| `stt_transcript` | JSON | Transcribed user speech text with $T_{\text{STT}}$ |
| `intent_detected` | JSON | Detected user intent tag (e.g. `Question`, `Greeting`) |
| `llm_token` | JSON | Live streaming conversational response tokens |
| Audio Packet | Binary `bytes` | MP3 audio stream for browser speaker playback |
| `turn_complete` | JSON | Complete turn summary with full latency metrics breakdown |
