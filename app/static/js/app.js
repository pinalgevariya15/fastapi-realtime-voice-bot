/**
 * Real-Time Voice Bot Frontend Client
 * Handles WebSocket duplex audio streaming, Web Audio API capture, gapless audio playback,
 * live canvas waveform visualizer, and latency metrics display.
 */

// --- Global State ---
let socket = null;
let sessionId = localStorage.getItem("voicebot_session_id") || null;
let mediaRecorder = null;
let audioStream = null;
let audioContext = null;
let micAnalyser = null;
let botAnalyser = null;
let isRecording = false;
let isBotSpeaking = false;
let animationFrameId = null;

// Audio Playback Engine (Gapless Web Audio API Scheduler)
let nextPlayTime = 0;
let activeSources = [];

// DOM Elements
const connectionBadge = document.getElementById("connectionBadge");
const connectionText = document.getElementById("connectionText");
const micBtn = document.getElementById("micBtn");
const micLabel = document.getElementById("micLabel");
const waveformCanvas = document.getElementById("waveformCanvas");
const canvasCtx = waveformCanvas.getContext("2d");
const messagesList = document.getElementById("messagesList");
const chatForm = document.getElementById("chatForm");
const textQuery = document.getElementById("textQuery");
const clearBtn = document.getElementById("clearBtn");

// Metrics DOM Elements
const valE2EFirst = document.getElementById("valE2EFirst");
const valSTT = document.getElementById("valSTT");
const valTTFT = document.getElementById("valTTFT");
const valTTSFirst = document.getElementById("valTTSFirst");
const valE2ETotal = document.getElementById("valE2ETotal");

const stepSTT = document.getElementById("stepSTT");
const stepLLM = document.getElementById("stepLLM");
const stepTTS = document.getElementById("stepTTS");
const timeSTT = document.getElementById("timeSTT");
const timeLLM = document.getElementById("timeLLM");
const timeTTS = document.getElementById("timeTTS");

const sessionValId = document.getElementById("sessionValId");
const sessionValTurns = document.getElementById("sessionValTurns");
const sessionValIntent = document.getElementById("sessionValIntent");

// Active Assistant Bubble for Streaming
let currentAssistantBubble = null;
let currentIntentTag = null;

// --- Initialize App ---
window.addEventListener("DOMContentLoaded", () => {
  initWebSocket();
  initVisualizer();
  bindEvents();
});

// Ensure AudioContext is initialized on user gesture
function getAudioContext() {
  if (!audioContext) {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    audioContext = new AudioCtx();
    botAnalyser = audioContext.createAnalyser();
    botAnalyser.fftSize = 256;
    botAnalyser.connect(audioContext.destination);
  }
  if (audioContext.state === "suspended") {
    audioContext.resume();
  }
  return audioContext;
}

// --- WebSocket Management ---
function initWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = window.location.host || "localhost:8000";
  const url = `${protocol}//${host}/ws/${sessionId || "new"}`;

  updateStatus("connecting", "Connecting...");

  socket = new WebSocket(url);
  socket.binaryType = "arraybuffer";

  socket.onopen = () => {
    updateStatus("connected", "Connected");
  };

  socket.onclose = () => {
    updateStatus("disconnected", "Disconnected. Reconnecting...");
    setTimeout(initWebSocket, 2500);
  };

  socket.onerror = (err) => {
    console.error("WebSocket error:", err);
  };

  socket.onmessage = async (event) => {
    if (typeof event.data === "string") {
      handleJsonEvent(JSON.parse(event.data));
    } else if (event.data instanceof ArrayBuffer) {
      // Inbound audio clip: decode and play gaplessly
      await queueAudioClip(event.data);
    }
  };
}

function updateStatus(state, text) {
  connectionBadge.className = `status-badge ${state}`;
  connectionText.textContent = text;
}

// --- WebSocket Event Handlers ---
function handleJsonEvent(data) {
  switch (data.type) {
    case "connected":
      sessionId = data.session_id;
      localStorage.setItem("voicebot_session_id", sessionId);
      sessionValId.textContent = sessionId.substring(0, 8) + "...";
      sessionValTurns.textContent = data.turn_count;
      sessionValIntent.textContent = data.last_intent || "None";
      break;

    case "stt_transcript":
      updateStatus("processing", "Processing...");
      setStepActive("stepSTT", true);
      timeSTT.textContent = `${data.stt_ms}ms`;
      valSTT.innerHTML = `${data.stt_ms}<span class="metric-unit">ms</span>`;
      appendUserMessage(data.text);
      createAssistantPlaceholder();
      setStepActive("stepLLM", true);
      break;

    case "intent_detected":
      sessionValIntent.textContent = data.intent;
      if (currentIntentTag) {
        currentIntentTag.textContent = data.intent;
        currentIntentTag.className = `intent-tag ${data.intent}`;
      }
      break;

    case "llm_token":
      if (currentAssistantBubble) {
        currentAssistantBubble.textContent += data.token;
        messagesList.scrollTop = messagesList.scrollHeight;
      }
      break;

    case "turn_complete":
      setStepActive("stepTTS", true);
      updateMetricsDashboard(data.latency);
      sessionValTurns.textContent = data.turn_count;
      resetStepHighlighting();
      currentAssistantBubble = null;
      currentIntentTag = null;
      if (!isRecording && !isBotSpeaking) {
        updateStatus("connected", "Ready");
      }
      break;

    case "session_cleared":
      messagesList.innerHTML = "";
      sessionValTurns.textContent = "0";
      sessionValIntent.textContent = "None";
      break;

    case "error":
      console.error("Pipeline Error:", data);
      alert(`Pipeline error: ${data.message}`);
      updateStatus("connected", "Ready");
      resetStepHighlighting();
      break;
  }
}

// --- Gapless Web Audio Playback ---
async function queueAudioClip(arrayBuffer) {
  if (!arrayBuffer || arrayBuffer.byteLength === 0) return;

  const ctx = getAudioContext();
  try {
    // Decode MP3 audio into AudioBuffer
    const audioBuffer = await ctx.decodeAudioData(arrayBuffer.slice(0));
    
    // Create source node
    const source = ctx.createBufferSource();
    source.buffer = audioBuffer;

    // Connect to botAnalyser (which feeds destination) for visualization
    source.connect(botAnalyser);

    const currentTime = ctx.currentTime;
    const startTime = Math.max(currentTime, nextPlayTime);
    source.start(startTime);
    nextPlayTime = startTime + audioBuffer.duration;

    isBotSpeaking = true;
    updateStatus("speaking", "Speaking...");
    activeSources.push(source);

    source.onended = () => {
      const idx = activeSources.indexOf(source);
      if (idx !== -1) activeSources.splice(idx, 1);

      if (activeSources.length === 0 && ctx.currentTime >= nextPlayTime) {
        isBotSpeaking = false;
        nextPlayTime = 0;
        if (!isRecording) {
          updateStatus("connected", "Ready");
        }
      }
    };
  } catch (err) {
    console.error("Audio decoding error:", err);
  }
}

function stopBotAudio() {
  activeSources.forEach((src) => {
    try { src.stop(); } catch (e) {}
  });
  activeSources = [];
  nextPlayTime = 0;
  isBotSpeaking = false;
}

// --- Microphone Audio Recording ---
async function toggleRecording() {
  getAudioContext(); // Unlock Web Audio context
  if (isRecording) {
    stopRecording();
  } else {
    // If bot is speaking when user hits mic, stop bot audio (barge-in interruption)
    stopBotAudio();
    await startRecording();
  }
}

async function startRecording() {
  try {
    const ctx = getAudioContext();

    audioStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        sampleRate: 16000
      }
    });

    const source = ctx.createMediaStreamSource(audioStream);
    micAnalyser = ctx.createAnalyser();
    micAnalyser.fftSize = 256;
    source.connect(micAnalyser);

    const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
      ? "audio/webm;codecs=opus"
      : "audio/webm";

    mediaRecorder = new MediaRecorder(audioStream, { mimeType });

    mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0 && socket && socket.readyState === WebSocket.OPEN) {
        socket.send(event.data);
      }
    };

    mediaRecorder.start(250); // Send 250ms chunks
    isRecording = true;

    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "audio_start", format: mimeType }));
    }

    micBtn.classList.add("active");
    micLabel.textContent = "Listening... Click when done speaking";
    updateStatus("listening", "Listening...");
    resetStepHighlighting();

  } catch (err) {
    console.error("Microphone access error:", err);
    alert("Could not access microphone: " + err.message);
  }
}

function stopRecording() {
  if (!isRecording) return;
  isRecording = false;

  micBtn.classList.remove("active");
  micLabel.textContent = "Transcribing voice...";
  updateStatus("processing", "Processing STT...");

  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }

  if (audioStream) {
    audioStream.getTracks().forEach((track) => track.stop());
  }

  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: "audio_end" }));
  }
}

// --- Text Form & Clear Session ---
function bindEvents() {
  micBtn.addEventListener("click", toggleRecording);

  // Resume AudioContext on any click
  document.body.addEventListener("click", () => {
    getAudioContext();
  }, { once: true });

  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    getAudioContext();
    stopBotAudio();

    const query = textQuery.value.trim();
    if (!query || !socket || socket.readyState !== WebSocket.OPEN) return;

    textQuery.value = "";
    resetStepHighlighting();
    setStepActive("stepLLM", true);

    socket.send(JSON.stringify({
      type: "text_query",
      text: query
    }));
  });

  clearBtn.addEventListener("click", () => {
    stopBotAudio();
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "clear_session" }));
    }
  });
}

// --- UI Transcript Management ---
function appendUserMessage(text) {
  const item = document.createElement("div");
  item.className = "message-item user";
  item.innerHTML = `
    <div class="message-meta">
      <span>You</span>
    </div>
    <div class="message-bubble">${escapeHtml(text)}</div>
  `;
  messagesList.appendChild(item);
  messagesList.scrollTop = messagesList.scrollHeight;
}

function createAssistantPlaceholder() {
  const item = document.createElement("div");
  item.className = "message-item assistant";
  
  const meta = document.createElement("div");
  meta.className = "message-meta";
  
  currentIntentTag = document.createElement("span");
  currentIntentTag.className = "intent-tag";
  currentIntentTag.textContent = "Detecting...";
  meta.appendChild(currentIntentTag);
  
  const label = document.createElement("span");
  label.textContent = "Bot";
  meta.appendChild(label);

  currentAssistantBubble = document.createElement("div");
  currentAssistantBubble.className = "message-bubble";
  currentAssistantBubble.textContent = "";

  item.appendChild(meta);
  item.appendChild(currentAssistantBubble);
  messagesList.appendChild(item);
  messagesList.scrollTop = messagesList.scrollHeight;
}

function updateMetricsDashboard(latency) {
  valE2EFirst.innerHTML = `${latency.e2e_first_audio_ms}<span class="metric-unit">ms</span>`;
  valSTT.innerHTML = `${latency.stt_ms}<span class="metric-unit">ms</span>`;
  valTTFT.innerHTML = `${latency.llm_ttft_ms}<span class="metric-unit">ms</span>`;
  valTTSFirst.innerHTML = `${latency.tts_first_chunk_ms}<span class="metric-unit">ms</span>`;
  valE2ETotal.innerHTML = `${latency.e2e_total_ms}<span class="metric-unit">ms</span>`;

  timeSTT.textContent = `${latency.stt_ms}ms`;
  timeLLM.textContent = `${latency.llm_total_ms}ms (TTFT: ${latency.llm_ttft_ms}ms)`;
  timeTTS.textContent = `${latency.tts_total_ms}ms (First: ${latency.tts_first_chunk_ms}ms)`;
}

function setStepActive(stepId, active) {
  const el = document.getElementById(stepId);
  if (el) {
    if (active) el.classList.add("active");
    else el.classList.remove("active");
  }
}

function resetStepHighlighting() {
  setStepActive("stepSTT", false);
  setStepActive("stepLLM", false);
  setStepActive("stepTTS", false);
}

// --- Canvas Audio Waveform Visualizer ---
function initVisualizer() {
  function draw() {
    animationFrameId = requestAnimationFrame(draw);
    const width = waveformCanvas.width;
    const height = waveformCanvas.height;

    canvasCtx.clearRect(0, 0, width, height);

    if (isRecording && micAnalyser) {
      // User speaking: live microphone frequency waveform (rose color)
      const bufferLength = micAnalyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      micAnalyser.getByteTimeDomainData(dataArray);

      canvasCtx.lineWidth = 2.5;
      canvasCtx.strokeStyle = "#f43f5e";
      canvasCtx.beginPath();

      const sliceWidth = width / bufferLength;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * height) / 2;

        if (i === 0) canvasCtx.moveTo(x, y);
        else canvasCtx.lineTo(x, y);
        x += sliceWidth;
      }
      canvasCtx.lineTo(width, height / 2);
      canvasCtx.stroke();

    } else if (isBotSpeaking && botAnalyser) {
      // Bot speaking: live synthesized voice audio waveform (cyan color)
      const bufferLength = botAnalyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      botAnalyser.getByteTimeDomainData(dataArray);

      canvasCtx.lineWidth = 2.5;
      canvasCtx.strokeStyle = "#06b6d4";
      canvasCtx.beginPath();

      const sliceWidth = width / bufferLength;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * height) / 2;

        if (i === 0) canvasCtx.moveTo(x, y);
        else canvasCtx.lineTo(x, y);
        x += sliceWidth;
      }
      canvasCtx.lineTo(width, height / 2);
      canvasCtx.stroke();

    } else {
      // Idle ambient breathing wave
      const time = Date.now() * 0.0025;
      canvasCtx.lineWidth = 1.5;
      canvasCtx.strokeStyle = "rgba(99, 102, 241, 0.35)";
      canvasCtx.beginPath();

      for (let x = 0; x < width; x += 6) {
        const y = height / 2 + Math.sin(x * 0.02 + time) * 5;
        if (x === 0) canvasCtx.moveTo(x, y);
        else canvasCtx.lineTo(x, y);
      }
      canvasCtx.stroke();
    }
  }

  draw();
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}
