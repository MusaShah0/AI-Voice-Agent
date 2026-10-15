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
 │    • 700ms silence endpoint threshold                         │
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
 │        - list_upcoming_appointments()                         │
 │        - reschedule_appointment()                             │
 │        - cancel_appointment()                                 │
 │                                                               │
 │ 4. Scheduling & Persistence Layer                             │
 │    • Google Calendar API v3 (event creation)                  │
 │    • SQLite Database via Django ORM (Doctor, Appointment,     │
 │      Reminder, ReminderEvent, Transcript)                     │
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
   - In parallel, the frontend uses the browser's `SpeechRecognition` / `webkitSpeechRecognition` API with interim results to show live captions while the user speaks. This does not add STT work to the Django backend or delay its audio pipeline. Browser support, recognition latency, and recognition service/privacy behavior depend on the browser.
2. **Speech Detection (VAD)**: In `VoiceConsumer`, each chunk is evaluated by `Silero VAD`.
   - Per-connection buffering retains partial chunks until a complete 512-sample (1024-byte) window is available, avoiding discarded remainders without mixing audio between WebSocket sessions.
   - If silence precedes speech, a rolling 300ms pre-speech ring buffer retains starting consonants.
   - Once speech is detected, audio accumulates into `audio_buffer`.
   - Once a turn has endpointed and backend processing begins, microphone audio frames are ignored until the turn response has finished playing. This prevents VAD false positives or speaker echo from cancelling STT/LLM work; the explicit stop-reply control remains available during playback.
3. **Endpointing**: When `consecutive_silence_bytes` reaches 700ms (`SILENCE_THRESHOLD_BYTES = 22,400 bytes`), the utterance is marked complete and passed to `process_audio_pipeline()`.
4. **Transcription (STT)**: The accumulated 16kHz mono PCM bytes are converted directly to a normalized float32 NumPy array and passed to `Faster-Whisper` (`compute_type="int8"`, `beam_size=1`, `vad_filter=False`, `condition_on_previous_text=False`, and English language hint by default). No temporary WAV file is written.
   - If transcription returns no text, the backend sends a short spoken retry prompt and a final `agent_response` instead of ending silently; no empty user utterance is sent to the dialog model or saved as a transcript.
5. **Dialog & Tool Execution (LLM)**: The transcribed text is sent to `LLMService.stream_dialog()`. Groq streams the final answer; any requested scheduling tools are executed before its spoken confirmation:
   - Queries `check_doctor_availability(start_time, end_time)` against SQLite only; when only a start time is supplied the tool checks the default 30-minute window. Appointment writes are synced to Google Calendar when booking.
   - During a booking availability check, the model also passes any known `caller_name` and `purpose` as optional structured tool arguments. The LLM service retains these with the resolved time in a per-WebSocket booking-context JSON object and reuses it in later tool requests, so only missing fields need to be asked; this uses the existing model call rather than adding a separate extraction request.
   - If a conflict exists, it informs the caller and suggests alternative slots.
   - Once the user confirms, it executes `book_appointment(caller_name, start_time, end_time, purpose)`.
   - For an existing patient's schedule, `list_upcoming_appointments(caller_name)` returns up to ten future scheduled rows from SQLite, case-insensitively matched by name.
   - `reschedule_appointment(caller_name, existing_start_time, new_start_time, new_end_time=None)` requires an unambiguous matching future appointment and rejects conflicts. An omitted `new_end_time` preserves the original duration; a provided end time is used as requested if it is later than the new start. The linked Google Calendar event is patched when authenticated.
   - `cancel_appointment(caller_name, existing_start_time)` requires an unambiguous future appointment. It deletes the linked Google Calendar event when authenticated, marks the database appointment as cancelled, and refuses to change it when the required calendar event is missing or the calendar deletion fails.
   - New appointments default to 30 minutes when no end time is provided. Explicitly requested durations are honored.
6. **Speech Synthesis (TTS)**: As streamed text reaches sentence boundaries, `VoiceConsumer` synthesizes each sentence locally with Piper on a worker thread. A bounded queue lets Groq continue generating while Piper synthesizes the previous sentence.
7. **Persistence**: The full turn (`User: ... | Agent: ...`) is saved to the `Transcript` table linked to the unique session ID.
8. **Delivery**: Each synthesized sentence is sent as ordered, Base64-encoded mono PCM chunks; the final `agent_response` carries the complete caption and tool status.
9. **Playback and captions**: The frontend converts PCM samples into Web Audio buffers and schedules them consecutively as they arrive. Interim browser captions appear while the user speaks; the completed turn uses the backend Faster-Whisper transcript. Barge-in stops all queued audio for the interrupted turn.

### Doctor Dashboard and Appointment Reminders
- The React route `/doc` is a separate doctor-facing dashboard with **Upcoming** and **All appointments** tabs. The all-appointments view includes patient, start/end time, purpose, and status; the upcoming tab is limited to future `Scheduled` visits.
- `GET /api/doc/dashboard/` returns the schedule. `GET /api/doc/reminders/active/` updates doctor-dashboard presence and returns active or snoozed reminders; the dashboard polls it every 10 seconds.
- **Ask Janus** starts a separate general doctor assistant with a signed token from `POST /api/doc/assistant/call-token/` and `/ws/doc-assistant/`. It shares the existing voice transport, VAD/STT/TTS stack, but uses a doctor-specific prompt and appointment tools to search future appointments by patient, clinic-local date, or start time. Cancellation and rescheduling are proposed against a looked-up appointment, then require explicit confirmation in a later turn before `AppointmentService` applies the change.
- Chart-review preparation tasks are not currently part of booking or reminders. The legacy `Task` model/table is retained so existing rows are not deleted, but the dashboard and doctor call no longer read or modify those rows.
- The in-app reminder engine creates a reminder for each appointment booked at least two minutes ahead. It first activates 30 minutes before the appointment and repeats every five minutes, up to six attempts. Snoozing defaults to 15 minutes, but the doctor can ask for another whole-minute duration; the requested time must leave at least one minute before the appointment. Snoozing changes only the reminder's `next_fire_at`, never the appointment. Acknowledge, cancellation, rescheduling, and appointment start update or stop the reminder.
- Start the scheduler in a separate terminal with `manage.py run_reminder_scheduler`. Use `--once` for a single sweep. The old `send_appointment_reminders` command remains a compatibility alias for one in-app sweep and no longer sends email. Disable any old Windows Task Scheduler email command entries; configure at most one looping scheduler per database.
- Clicking a reminder's **Call** requests a short-lived signed token at `POST /api/doc/reminders/<id>/call-token/` and opens `/ws/doc-voice/`. The token identifies the reminder; the WebSocket consumer validates it and loads that reminder's linked appointment and doctor through `ReminderService.get_call_context()`. The initial appointment-specific greeting is built by the consumer and spoken with local Piper TTS; it is not an LLM-generated greeting. For each user turn, common actions such as acknowledge and snooze are handled directly in code. Other utterances are sent to the doctor LLM with a system prompt containing the bound appointment's patient name, start time, purpose, reminder status, current time, and clinic timezone. The reminder-specific assistant and general **Ask Janus** assistant are separate conversations, but share the same audio transport and speech services.
- The doctor dashboard, reminder APIs, and doctor voice calls are intentionally available only while Django `DEBUG=True`. They are unauthenticated local-development features and must not be exposed to a network or deployed with `DEBUG=True`; real doctor authentication is required first.

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
    google_event_id = models.CharField(max_length=255, null=True, blank=True)
```
- `doctor`: ForeignKey to `Doctor` (ID 1).
- `caller_name`: Patient or caller full name.
- `start_time` / `end_time`: Timezone-aware timestamps representing the clinic-local appointment window; Django stores instants in UTC.
- `purpose`: Reason for visit or symptoms described by patient.
- `status`: Default `"Scheduled"`. Can be `"Cancelled"`, `"Completed"`.
- `google_event_id`: Optional ID of the clinic calendar event created for this appointment; used to patch its time when rescheduling.

### 3. `Transcript` Model
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
| `SILENCE_THRESHOLD_MS` | `700` | Sustained silence duration required to trigger end of utterance. |
| `SILENCE_THRESHOLD_BYTES`| `22400` | 700 ms * 32 bytes/ms. |
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

The current default is the English-only `base.en` model. It was compared locally with multilingual `base` on one supplied sample: mean transcription times were approximately 0.89 s vs. 0.91 s, and normalized transcripts were 98.7% similar. This single-sample measurement is not a general accuracy or latency guarantee. STT stays local by default; no audio is sent to Groq.

---

### `core/services/llm_service.py`
**Responsibility**: Conversational state management, Groq API interactions, and tool calling for scheduling actions.

- **Model**: Reads `GROQ_MODEL` from `.env`, defaulting to `openai/gpt-oss-20b`.
- `__init__(self, calendar_service=None)`: Initializes the Groq client with `GROQ_API_KEY`.
- `_get_system_instruction(self) -> str`: Dynamically embeds current date and time (`Monday, October 12, 2026 at...`) and instructs Janus on tool calling, concise voice responses, and medical receptionist persona.
- `_parse_datetime(self, dt_str: str) -> datetime.datetime`: Parses ISO and common datetime strings; naive values are clinic-local (`Asia/Karachi`), while offset-aware values are converted to the clinic timezone.
- `check_doctor_availability(self, start_time: str, end_time: Optional[str] = None) -> Dict[str, Any]`:
  - Queries SQLite only for conflicting `Appointment` records where `start_time < slot_end AND end_time > slot_start`; does not call Google Calendar.
  - Defaults the checked window to 30 minutes from the requested start time.
  - Formats conflicts and available times in the clinic timezone.
  - Returns `{"available": bool, "conflicts": list, "message": str}`.
  - Updates `self.last_tool_status` to `"availability_checked"` or `"conflict_detected"`.
- `book_appointment(self, caller_name: str, start_time: str, purpose: str, end_time: Optional[str] = None) -> Dict[str, Any]`:
  - Inserts event to Google Calendar if authenticated, using an explicit clinic-local UTC offset.
  - Calls `DataRepository.create_appointment_sync()` to persist appointment and its Google event ID to SQLite.
  - Defaults to a 30-minute appointment unless an end time is explicitly requested.
  - Updates `self.last_tool_status` to `"calendar_booked"`.
- `list_upcoming_appointments(self, caller_name: str) -> Dict[str, Any]`:
  - Returns up to ten future `"Scheduled"` appointments with a case-insensitive exact patient-name match, ordered by start time.
  - Returns date/time values in clinic-local time and asks the model to clarify when multiple appointments match.
- `reschedule_appointment(self, caller_name: str, existing_start_time: str, new_start_time: str, new_end_time: Optional[str] = None) -> Dict[str, Any]`:
  - Matches a future appointment by patient name and existing start minute; rejects missing or ambiguous matches and conflicting new slots.
  - Preserves appointment duration unless a new end time is specified and patches the corresponding Google Calendar event if authenticated.
  - Updates the database only after the calendar succeeds; attempts to restore the old calendar time if the database update fails.
- `cancel_appointment(self, caller_name: str, existing_start_time: str) -> Dict[str, Any]`:
  - Matches one future `"Scheduled"` appointment by case-insensitive exact caller name and existing start minute; refuses missing or ambiguous matches.
  - Deletes the saved Google Calendar event before marking the appointment `"Cancelled"` in SQLite. If Calendar is authenticated but the appointment has no stored event ID, or Calendar deletion fails, it leaves the database unchanged and reports an error.
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
---

### `frontend/src/App.jsx`
**Responsibility**: Web Audio recording, real-time PCM downsampling, WebSocket transport, audio decoding/playback, captions, and interruption controls.

- `pcm16FromFloat(input: Float32Array, inputSampleRate: number) -> ArrayBuffer`: Resamples browser microphone audio (typically 44.1kHz or 48kHz) down to 16,000 Hz 16-bit signed Linear PCM integers.
- `connectSocket()`: Connects to `ws://localhost:8001/ws/voice/` with a 6-second timeout.
- `startMicrophone()`: Requests `getUserMedia` (mono, echoCancellation, noiseSuppression), creates `ScriptProcessor(4096, 1, 1)`, and streams binary PCM chunks over WebSocket.
- `startLiveCaptions()`: Starts browser speech recognition with interim results and updates the temporary live-caption display. If unavailable, the UI indicates that the final backend transcript will appear after processing.
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

Run a one-time reminder sweep to test the in-app reminder engine:
```powershell
.\venv\Scripts\python.exe manage.py run_reminder_scheduler --once
```

Run the patient-facing UI at `http://localhost:3000/`; the local doctor dashboard is at `http://localhost:3000/doc/`. The dashboard reads `http://localhost:8000/api/doc/dashboard/`. Set `VITE_API_BASE_URL` in the frontend environment if the Django API runs at a different host or port.

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
- [x] **VAD Endpointing**: 512-sample evaluation on CPU with 700ms silence detection.
- [x] **Faster-Whisper STT**: Transcribes local English voice input directly from float32 PCM arrays with `base.en` and CPU INT8 inference.
- [x] **LLM Tool Calling**: Groq's configured model executes free/busy checks and appointment bookings.
- [x] **Database Persistence**: Appointments, tasks, and dialogue transcripts persist to SQLite.
- [x] **Streaming Speech Playback**: Groq sentence output is synthesized by local Piper, relayed as PCM chunks, and queued with Web Audio.
- [x] **Interruption / Barge-in**: User speaking mid-reply cuts off playback immediately.
