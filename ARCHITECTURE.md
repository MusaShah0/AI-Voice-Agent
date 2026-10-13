# Voice Appointment Agent — Comprehensive Architecture & Technical Reference

This document provides a complete technical reference for the real-time, bidirectional Voice Appointment Agent. It details the end-to-end audio and data flow, WebSocket protocols, model configurations, database schema, individual file and function responsibilities, parameters, and server management commands.

---

## 1. High-Level Architecture & End-to-End Flow

The system coordinates a real-time conversational voice loop connecting a React frontend to an asynchronous Django ASGI backend over WebSockets.

```
 [User Browser (Microphone / Text)]
         │
         │  Downsampled PCM-16 Audio Chunks (16 kHz, Mono)
         │  or Control / Text JSON via WebSocket
         ▼
 ┌───────────────────────────────────────────────────────────────┐
 │ Daphne ASGI Server (Port 8001)                                │
 │ ws://127.0.0.1:8001/ws/voice/                                 │
 ├───────────────────────────────────────────────────────────────┤
 │ VoiceConsumer (channels.generic.websocket.AsyncWebsocketConsumer)
 │                                                               │
 │ 1. VAD Layer (Silero VAD on CPU)                             │
 │    • 512-sample (32ms) sliding evaluation                     │
 │    • 300ms pre-speech buffer                                  │
 │    • 500ms silence endpoint threshold                         │
 │    • User barge-in / interruption cancellation                │
 │                                                               │
 │ 2. STT Layer (Faster-Whisper "base.en" INT8 on CPU)           │
 │    • Direct 16kHz PCM-to-float32 NumPy input                  │
 │    • Transcribes user speech utterance to text                │
 │                                                               │
 │ 3. Dialog & Tool Layer (Groq: openai/gpt-oss-20b)             │
 │    • System instruction persona ("Janus" Receptionist)        │
 │    • Multi-turn history management                            │
 │    • Model configured by GROQ_MODEL                           │
 │    • Explicit bounded tool-calling loop:                       │
 │        - check_doctor_availability()                          │
 │        - book_appointment()                                   │
 │                                                               │
 │ 4. Scheduling & Persistence Layer                             │
 │    • Google Calendar API v3 (event creation)                  │
 │    • SQLite Database via Django ORM (Doctor, Appointment,     │
 │      Task, Transcript)                                        │
 │                                                               │
 │ 5. TTS Layer (local Piper "en_US-lessac-medium")              │
 │    • Groq final answer streams sentence by sentence           │
 │    • Piper generates local mono PCM; sent as ordered chunks   │
 └───────────────────────────────┬───────────────────────────────┘
                                 │
                                 │ JSON: audio_chunk PCM frames + agent_response caption
                                 ▼
                     [React Voice UI Playback]
              (Web Audio PCM queue + Real-time Captions)
```

### Detailed Sequence of a Voice Turn
0. **Connection Welcome**: After voice services initialize, the backend sends Janus’s welcome caption and synthesizes it with Piper. The browser displays the introduction and queues its PCM audio before handling user turns.
1. **Audio Capture**: The browser requests a 16kHz `AudioContext` so browser-managed resampling is used when supported. At 16kHz it uses 512-sample frames, converts float32 samples directly to 16-bit signed PCM, and streams binary frames over WebSocket. If the requested context rate is unavailable, it falls back to the default context and existing client-side conversion. Browser noise suppression is disabled for the current accuracy comparison; echo cancellation and automatic gain control remain enabled.
2. **Speech Detection (VAD)**: In `VoiceConsumer`, each chunk is evaluated by `Silero VAD`.
   - Per-connection buffering retains partial chunks until a complete 512-sample (1024-byte) window is available, avoiding discarded remainders without mixing audio between WebSocket sessions.
   - If silence precedes speech, a rolling 300ms pre-speech ring buffer retains starting consonants.
   - Once speech is detected, audio accumulates into `audio_buffer`.
   - If the user speaks while the agent is playing an earlier response, the active turn task is immediately cancelled, and `{"status": "audio_stopped"}` is dispatched to cut off frontend audio playback.
3. **Endpointing**: When `consecutive_silence_bytes` reaches 500ms (`SILENCE_THRESHOLD_BYTES = 16,000 bytes`), the utterance is marked complete and passed to `process_audio_pipeline()`.
4. **Transcription (STT)**: The accumulated 16kHz mono PCM bytes are converted directly to a normalized float32 NumPy array and passed to `Faster-Whisper` (`compute_type="int8"`, `beam_size=1`, `vad_filter=False`, `condition_on_previous_text=False`, and English language hint by default). No temporary WAV file is written.
5. **Dialog & Tool Execution (LLM)**: The transcribed text is sent to `LLMService.stream_dialog()`. Groq streams the final answer; any requested scheduling tools are executed before its spoken confirmation:
   - Queries `check_doctor_availability(start_time, end_time)` against SQLite only; appointment writes are synced to Google Calendar when booking.
   - If a conflict exists, it informs the caller and suggests alternative slots.
   - Once the user confirms, it executes `book_appointment(caller_name, start_time, end_time, purpose)`.
6. **Speech Synthesis (TTS)**: As streamed text reaches sentence boundaries, `VoiceConsumer` synthesizes each sentence locally with Piper on a worker thread. A bounded queue lets Groq continue generating while Piper synthesizes the previous sentence.
7. **Persistence**: The full turn (`User: ... | Agent: ...`) is saved to the `Transcript` table linked to the unique session ID.
8. **Delivery**: Each synthesized sentence is sent as ordered, Base64-encoded mono PCM chunks; the final `agent_response` carries the complete caption and tool status.
9. **Playback**: The frontend converts PCM samples into Web Audio buffers and schedules them consecutively as they arrive. Barge-in stops all queued audio for the interrupted turn.

### Local Piper Voice Setup

Install Python dependencies, then download the default voice model once:
```powershell
python -m piper.download_voices --download-dir models/piper en_US-lessac-medium
```
The application expects `models/piper/en_US-lessac-medium.onnx` by default. Set `PIPER_VOICE_PATH` to an alternate ONNX voice file if needed. Piper synthesis runs locally after the voice model is installed; it does not send response text to a hosted TTS service.

---

## 2. WebSocket Protocol & Parameter Specification

- **Connection URL**: `ws://127.0.0.1:8001/ws/voice/`
- **Protocol Handshake**: Standard WebSocket RFC 6455 over ASGI (`daphne`).

### A. Inbound Messages (Client to Server)

#### 1. Binary PCM Audio Stream
Sent continuously while microphone recording is active.
- **Data Type**: `bytes` (Binary WebSocket frame)
- **Sample Rate**: `16,000 Hz`
- **Bit Depth**: `16-bit Signed Integer (Little-Endian)`
- **Channels**: `1 (Mono)`
- **Chunk Size**: Dynamically emitted by the browser audio processor (~4096 input frames resampled to ~1,365 samples = ~2,730 bytes).
- **Alignment Requirement**: The server enforces 2-byte alignment (`aligned_len = len(bytes_data) - (len(bytes_data) % 2)`).

#### 2. Process Text Turn
Used when the user types in the input box instead of using speech.
```json
{
  "action": "process_text_turn",
  "text": "Is Dr. Smith available tomorrow at 2 PM?"
}
```
- `action` (string, required): Must be `"process_text_turn"`.
- `text` (string, required): The user's typed message.

#### 3. Stop Audio / Interruption Signal
Sent when the user clicks the Stop/Mute button or begins speaking during playback.
```json
{
  "action": "stop_audio"
}
```
- `action` (string, required): Must be `"stop_audio"`.

---

### B. Outbound Messages (Server to Client)

#### 1. Audio Chunk
Emitted as local Piper synthesis produces audio for each completed sentence.
```json
{
  "type": "audio_chunk",
  "turn_id": 1,
  "sequence": 0,
  "format": "pcm_s16le",
  "channels": 1,
  "sample_rate": 22050,
  "audio_payload": "base64-encoded PCM bytes"
}
```
- `turn_id` (integer): Turn identifier used to discard audio after interruption.
- `sequence` (integer): Zero-based order of PCM frames within the turn.
- `format` (string): Signed 16-bit little-endian PCM.
- `audio_payload` (string): Base64-encoded audio frame; frame size is at most 8192 bytes.

#### 2. Agent Response
Sent after the final audio chunk for the turn.
```json
{
  "type": "agent_response",
  "turn_id": 1,
  "user_transcription": "Is Dr. Smith available tomorrow at 2 PM?",
  "caption": "Dr. Melvin is available tomorrow at 2 PM.",
  "tool_status": "availability_checked"
}
```
- `caption` (string): The complete verbal response streamed from the configured Groq model.
- `tool_status` (string | null): Scheduling tool status as described below.

Scheduling tool status values:
  - `"availability_checked"`: Availability query executed, slot is open.
  - `"conflict_detected"`: Slot is occupied on Google Calendar or DB.
  - `"calendar_booked"`: Appointment confirmed and written to DB + Google Calendar.
  - `null`: Pure conversational turn (no tool invoked).
  - `"error"`: Tool execution failed.

#### 3. Audio Interruption / Cutoff Signal
Emitted when an active turn is cancelled due to user barge-in.
```json
{
  "status": "audio_stopped",
  "turn_id": 1
}
```

#### 4. Error Notification
Emitted if an unrecoverable failure occurs during processing.
```json
{
  "type": "error",
  "message": "A voice turn is already being processed."
}
```

---

## 3. Database Schema & Data Models ([models.py](file:///C:/Users/Musa%20Shah/Desktop/Voice%20Agent/core/models.py))

Located in [core/models.py](file:///C:/Users/Musa%20Shah/Desktop/Voice%20Agent/core/models.py). The database engine is SQLite (`db.sqlite3`).

### 1. `Doctor` Model
Enforces a singleton doctor record representing Dr. Smith.
```python
class Doctor(models.Model):
    SINGLETON_ID = 1
    name = models.CharField(max_length=255)
    email = models.EmailField(unique=True)
    timezone = models.CharField(max_length=50, default='Asia/Karachi')
```
- **Constraints**: Database-level check constraint `doctor_singleton_id` guarantees `id == 1`.
- **Default Record**: ID 1, `name="Dr. Melvin"`, `email="musa813705@gmail.com"`, `timezone="Asia/Karachi"`.

### 2. `Appointment` Model
Stores scheduled clinic appointments.
```python
class Appointment(models.Model):
    doctor = models.ForeignKey(Doctor, on_delete=models.CASCADE)
    caller_name = models.CharField(max_length=255)
    start_time = models.DateTimeField()
    end_time = models.DateTimeField()
    purpose = models.TextField()
    status = models.CharField(max_length=50, default='Scheduled')
```
- `doctor`: ForeignKey to `Doctor` (ID 1).
- `caller_name`: Patient or caller full name.
- `start_time` / `end_time`: Timezone-aware timestamps representing the clinic-local appointment window; Django stores instants in UTC.
- `purpose`: Reason for visit or symptoms described by patient.
- `status`: Default `"Scheduled"`. Can be `"Cancelled"`, `"Completed"`.

### 3. `Task` Model
Follow-up clinical tasks automatically created for the doctor upon booking.
```python
class Task(models.Model):
    doctor = models.ForeignKey(Doctor, on_delete=models.CASCADE)
    appointment = models.ForeignKey(Appointment, on_delete=models.SET_NULL, null=True)
    description = models.TextField()
    due_time = models.DateTimeField()
    status = models.CharField(max_length=20, default='Pending')
```
- Automatically scheduled by `DataRepository.create_task_sync()` for 1 hour prior to appointment start time (e.g. *"Review medical chart for appointment with Alice Johnson"*).

### 4. `Transcript` Model
Audit trail of dialogue turns for all sessions.
```python
class Transcript(models.Model):
    appointment = models.ForeignKey(Appointment, on_delete=models.SET_NULL, null=True, blank=True)
    session_id = models.CharField(max_length=100)
    full_transcript = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
```
- `session_id`: Unique identifier derived from the WebSocket channel name (`session_specific..inmemory!...`).
- `full_transcript`: Formatted turn string (`"User: ... | Agent: ..."`).

---

## 4. File-by-File Breakdown & Function Responsibilities

### `core/consumers.py`
**Responsibility**: WebSocket connection lifecycle, VAD buffering, speech endpointing, audio pipeline dispatch, barge-in interruption handling, and response delivery.

| Constant / Parameter | Value | Description |
| :--- | :--- | :--- |
| `AUDIO_SAMPLE_RATE` | `16000` | Input audio sample frequency in Hz. |
| `AUDIO_SAMPLE_WIDTH` | `2` | 16-bit audio = 2 bytes per sample. |
| `BYTES_PER_MS` | `32` | 1 millisecond = (16000 * 2) / 1000 = 32 bytes. |
| `SILENCE_THRESHOLD_MS` | `500` | Sustained silence duration required to trigger end of utterance. |
| `SILENCE_THRESHOLD_BYTES`| `16000` | 500 ms * 32 bytes/ms. |
| `PRE_SPEECH_BUFFER_BYTES`| `9600` | 300 ms lead-in ring buffer to prevent clipping starting consonants. |
| `MIN_SPEECH_BYTES` | `8000` | 250 ms minimum audio required to reject mic pops/coughs. |
| `MAX_SPEECH_BYTES` | `800000` | 25 seconds safety ceiling to force processing in noisy environments. |

#### Methods in `VoiceConsumer`:
- `connect(self)`: Accepts the WebSocket handshake, assigns `self.session_id`, initializes singletons (`vad`, `stt`, `tts`, `llm`), and allocates memory buffers.
- `disconnect(self, close_code)`: Cancels any active async turn task and logs disconnection.
- `receive(self, text_data=None, bytes_data=None)`:
  - If `bytes_data`: Aligns byte boundaries, calls `vad.is_speech(clean_chunk)`. Handles user interruption (cancelling active task and sending `audio_stopped`). Manages transitions between pre-speech buffer, active speech accumulation, and silence endpoint detection.
  - If `text_data`: Parses JSON actions (`"process_text_turn"` or `"stop_audio"`).
- `_reset_speech_state(self)`: Clears `audio_buffer`, `pre_speech_buffer`, resets `speech_detected = False` and `consecutive_silence_bytes = 0`.
- `_start_task(self, coroutine)`: Dispatches `coroutine` as an `asyncio.Task` with collision guard (`_send_busy()` if a task is already executing).
- `_run_safely(self, coroutine)`: Wraps coroutine execution to catch `asyncio.CancelledError` and log unhandled exceptions.
- `process_audio_pipeline(self, audio_data: bytes)`: Converts PCM-16 input directly to normalized float32 NumPy samples, calls `stt.transcribe()`, and triggers `run_dialog_and_respond()`.
- `run_dialog_and_respond(self, user_text: str)`: Executes `llm.process_dialog()`, calls `tts.synthesize()`, encodes audio as base64, persists turn to `Transcript` in the DB, and dispatches `agent_response` over WebSocket.

---

### `core/services/vad_service.py`
**Responsibility**: Voice Activity Detection on CPU using Silero VAD to detect speech presence in streaming audio chunks.

- `VADService.__init__(self)`: Loads Silero VAD PyTorch model on CPU from PyTorch Hub cache.
- `is_speech(self, audio_chunk: bytes, sampling_rate: int = 16000, threshold: float = 0.5) -> bool`:
  - **Input**: Raw PCM-16 `bytes` chunk.
  - **Processing**: Converts bytes to `np.int16`, normalizes to `float32` in range `[-1.0, 1.0]`, converts to `torch.Tensor`.
  - **Window Evaluation**: Slices tensor into 512-sample (32ms) windows. If any window exceeds `threshold` probability (default 0.5), returns `True`.
- `get_vad_service() -> VADService`: Module-level singleton provider.

---

### `core/services/stt_service.py`
**Responsibility**: English Speech-to-Text transcription on CPU using `faster-whisper`.

- `STTService.__init__(self, model_size="base.en", language="en")`: Loads Faster-Whisper on CPU using `compute_type="int8"`. The consumer accepts `STT_MODEL` and `STT_LANGUAGE` environment overrides.
- `transcribe(self, audio_input) -> str`:
  - **Input**: Float32 NumPy audio array (used by the live consumer), WAV file path, raw PCM-16 bytes, or numpy float32 array.
  - **Processing**: Calls `model.transcribe(beam_size=1, vad_filter=False, condition_on_previous_text=False, language="en")` by default. Local VAD has already detected speech, so Whisper's VAD filter is disabled.
  - **Output**: Joined text transcription string.
- `get_stt_service(model_size="base.en", language="en") -> STTService`: Module-level singleton cache keyed by model and language.

The English-only `base.en` model was compared locally with multilingual `base` on the supplied sample: mean transcription times were approximately 0.89 s vs. 0.91 s, and normalized transcripts were 98.7% similar. This single-sample measurement is not a general accuracy or latency guarantee. STT stays local by default; no audio is sent to Groq.

---

### `core/services/llm_service.py`
**Responsibility**: Conversational state management, Groq API interactions, and tool calling for scheduling actions.

- **Model**: Reads `GROQ_MODEL` from `.env`, defaulting to `openai/gpt-oss-20b`.
- `__init__(self, calendar_service=None)`: Initializes the Groq client with `GROQ_API_KEY`.
- `_get_system_instruction(self) -> str`: Dynamically embeds current date and time (`Monday, October 12, 2026 at...`) and instructs Janus on tool calling, concise voice responses, and medical receptionist persona.
- `_parse_datetime(self, dt_str: str) -> datetime.datetime`: Parses ISO and common datetime strings; naive values are clinic-local (`Asia/Karachi`), while offset-aware values are converted to the clinic timezone.
- `check_doctor_availability(self, start_time: str, end_time: str) -> Dict[str, Any]`:
  - Queries SQLite only for conflicting `Appointment` records where `start_time < slot_end AND end_time > slot_start`; does not call Google Calendar.
  - Formats conflicts and available times in the clinic timezone.
  - Returns `{"available": bool, "conflicts": list, "message": str}`.
  - Updates `self.last_tool_status` to `"availability_checked"` or `"conflict_detected"`.
- `book_appointment(self, caller_name: str, start_time: str, end_time: str, purpose: str) -> Dict[str, Any]`:
  - Inserts event to Google Calendar if authenticated, using an explicit clinic-local UTC offset.
  - Calls `DataRepository.create_appointment_sync()` to persist appointment to SQLite.
  - Calls `DataRepository.create_task_sync()` to create doctor chart review task.
  - Updates `self.last_tool_status` to `"calendar_booked"`.
- `_build_messages(self, conversation_history: list, user_input: str)`: Builds provider messages from prior turns and the current input.
- `process_dialog(self, conversation_history: list, user_input: str) -> Tuple[str, Optional[str]]`: Calls the configured Groq model and runs at most five tool-call rounds, returning tool results to the model before the final response.
  - Appends turns to `conversation_history`.
  - Returns `(response_text, tool_status)`.

---

### `core/services/tts_service.py`
**Responsibility**: Local sentence-by-sentence speech synthesis to mono PCM using Piper.

- `TTSService.__init__(self)`: Loads the `en_US-lessac-medium` ONNX voice from `PIPER_VOICE_PATH` or `models/piper/en_US-lessac-medium.onnx`.
- `synthesize(self, text: str) -> Tuple[bytes, int]`: Generates local signed 16-bit mono PCM and returns `(audio_bytes, sample_rate)`.
- `get_tts_service() -> TTSService`: Module-level singleton provider.

---

### `core/services/calendar_service.py`
**Responsibility**: Google Calendar API v3 integration, OAuth credential handling, and event queries.

- `__init__(self, credentials_file=None, token_file=None, base_dir=None)`: Discovers `client_secret_*.json` in root directory and defaults token storage to `token.json`.
- `is_authenticated(self) -> bool`: Checks if valid, unexpired credentials exist in `token.json`. Gracefully degrades to local DB scheduling if false.
- `authenticate_interactive(self, port=0) -> bool`: Launches interactive OAuth consent flow in browser and saves `token.json`. Can be run directly via CLI.
- `check_availability(self, start_time, end_time, calendar_id='primary') -> Dict[str, Any]`: Optional direct FreeBusy helper; the appointment availability tool does not call it and checks SQLite only.
- `create_event(self, summary, start_time, end_time, description="", calendar_id='primary', caller_name=None) -> Dict[str, Any]`: Inserts confirmed event via `service.events().insert()`.

---

### `core/repositories.py`
**Responsibility**: Synchronous database persistence helpers executed within `database_sync_to_async` thread pools.

- `save_transcript(session_id: str, full_transcript: str) -> Transcript`: Persists conversational turns to `Transcript`.
- `create_appointment_sync(caller_name: str, start_time, end_time, purpose: str) -> int`: Persists confirmed appointment linked to `Doctor` singleton ID 1 and returns the new appointment ID.
- `create_task_sync(appointment_id: int, description: str, due_time) -> int`: Persists preparation task linked to the appointment and returns task ID.

---

### `frontend/src/App.jsx`
**Responsibility**: Web Audio recording, real-time PCM downsampling, WebSocket transport, audio decoding/playback, captions, and interruption controls.

- `pcm16FromFloat(input: Float32Array, inputSampleRate: number) -> ArrayBuffer`: Resamples browser microphone audio (typically 44.1kHz or 48kHz) down to 16,000 Hz 16-bit signed Linear PCM integers.
- `connectSocket()`: Connects to `ws://localhost:8001/ws/voice/` with a 6-second timeout.
- `startMicrophone()`: Requests `getUserMedia` (mono, echoCancellation, noiseSuppression), creates `ScriptProcessor(4096, 1, 1)`, and streams binary PCM chunks over WebSocket.
- `stopMicrophone(notifyBackend = true)`: Disconnects audio nodes, stops mic tracks, closes `AudioContext`, and notifies backend with `{"action": "stop_audio"}`.
- `playReply(audioPayload: string)`: Decodes base64 string to a binary `Uint8Array`, creates a `Blob` of type `audio/mpeg`, and plays it via HTML5 `Audio`.
- `interrupt()`: Instantly cuts off local audio playback and sends `stop_audio` over WebSocket.
- `sendTextTurn(event)`: Submits typed text turns via `{"action": "process_text_turn", "text": "..."}`.

---

## 5. Server Startup & Verification Commands

All commands are executed from the project root (`C:\Users\Musa Shah\Desktop\Voice Agent`).

### 1. Python Environment Setup
```powershell
# Activate the existing virtual environment
.\venv\Scripts\Activate.ps1

# Verify all dependencies are installed
pip install -r requirements.txt
```

### 2. Database Migrations & Seeding
```powershell
# Run pending migrations
.\venv\Scripts\python.exe manage.py migrate

# Verify the Doctor singleton record (ID 1: Dr. Melvin) is seeded
.\venv\Scripts\python.exe -c "import django, os; os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'voice_agent_project.settings'); django.setup(); from core.models import Doctor; doc, _ = Doctor.objects.get_or_create(id=Doctor.SINGLETON_ID, defaults={'name': 'Dr. Melvin', 'email': 'musa813705@gmail.com', 'timezone': 'Asia/Karachi'}); print(f'Doctor seeded: {doc.name} (ID: {doc.id})')"
```

### 3. Optional: Google Calendar Interactive OAuth
```powershell
# Run browser OAuth flow to generate token.json (optional; falls back to DB if omitted)
.\venv\Scripts\python.exe -m core.services.calendar_service
```

### 4. Start the Daphne ASGI Backend Server
```powershell
# Launch Daphne on port 8001
.\venv\Scripts\daphne.exe -b 127.0.0.1 -p 8001 voice_agent_project.asgi:application
```

Pipeline INFO logs are written to the same terminal. Follow the `Pipeline` messages by session ID to trace connection and service initialization, VAD/endpointing, STT, dialog and tool execution, TTS, transcript persistence, response delivery, and disconnects. Audio and transcript contents are not included in these pipeline logs.

### 5. Start the Vite Frontend Development Server
In a separate terminal window:
```powershell
cd frontend
npm install
npm run dev -- --port 3000
```
Open **`http://localhost:3000`** in Google Chrome or Microsoft Edge.

### 6. Run the Automated Test Suite
```powershell
# Execute all 11 unit tests across VAD, STT, TTS, Calendar, LLM, and Consumers
.\venv\Scripts\python.exe manage.py test core
```

---

## 6. Verification Checklist

- [x] **Microphone Capture**: Captures 1-channel audio, converts float32 to 16kHz PCM-16.
- [x] **WebSocket Transport**: Binary PCM audio streams cleanly to `ws://127.0.0.1:8001/ws/voice/`.
- [x] **VAD Endpointing**: 512-sample evaluation on CPU with 500ms silence detection.
- [x] **Faster-Whisper STT**: Transcribes local English voice input directly from float32 PCM arrays with `base.en` and CPU INT8 inference.
- [x] **LLM Tool Calling**: Groq's configured model executes free/busy checks and appointment bookings.
- [x] **Database Persistence**: Appointments, tasks, and dialogue transcripts persist to SQLite.
- [x] **Streaming Speech Playback**: Groq sentence output is synthesized by local Piper, relayed as PCM chunks, and queued with Web Audio.
- [x] **Interruption / Barge-in**: User speaking mid-reply cuts off playback immediately.
