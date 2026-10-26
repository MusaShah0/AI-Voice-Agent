# Voice Agent: Architecture and Project Guide

This document describes the application as implemented in the repository. It covers the browser clients, Django/ASGI services, the speech pipeline, appointment and reminder workflows, local configuration, and current operational limitations.

## 1. What the application does

The project is a local-development voice appointment assistant for a clinic. Patients use a browser chat/voice interface to ask about appointments and request bookings or changes. A separate doctor dashboard lists appointments and reminders, starts reminder calls, and offers a doctor-facing voice assistant.

The primary technologies are:

- **Frontend:** React 18 and Vite.
- **HTTP and WebSocket backend:** Django, Django Channels, and Daphne/ASGI.
- **Persistence:** SQLite through Django's ORM.
- **Speech recognition:** Faster-Whisper running locally on CPU.
- **Speech activity detection:** Silero VAD running locally through PyTorch.
- **Speech synthesis:** Piper running locally.
- **Dialog model:** Groq API, with `GROQ_MODEL` configurable.
- **Calendar integration:** Google Calendar API, when OAuth credentials are available.

```text
Browser
  ├─ Patient UI (/)
  │    ├─ microphone PCM / typed text ── WebSocket ── VoiceConsumer
  │    └─ HTTP API calls (dashboard only) ─────────── Django views
  └─ Doctor UI (/doc)
       ├─ dashboard and reminders ────────────────── Django views
       ├─ reminder call ───────────────────────────── DoctorVoiceConsumer
       └─ Ask Janus ───────────────────────────────── DoctorAssistantConsumer

Django/ASGI
  ├─ VAD → endpointing → Faster-Whisper → Groq dialog/tools → Piper → browser
  ├─ ORM/repository → SQLite (appointments, reminders, events, transcripts)
  └─ Google Calendar service (appointment event synchronization)
```

## 2. Repository map

| Location | Responsibility |
|---|---|
| `frontend/src/App.jsx` | Patient chat, microphone capture, WebSocket messages, playback, and route selection for `/doc`. |
| `frontend/src/DoctorDashboard.jsx` | Doctor schedule and reminder dashboard; starts signed-token voice sessions. |
| `frontend/src/DoctorCallPanel.jsx` | Doctor call UI, microphone/PCM transport, and returned audio playback. |
| `frontend/src/LiveOrb.jsx` | Shared animated microphone/orb visual. |
| `frontend/src/*.css` | Patient and doctor dashboard presentation and layout. |
| `voice_agent_project/settings.py` | Django settings, SQLite, Channels, time zone, reminder options, and logging. |
| `voice_agent_project/asgi.py` | ASGI HTTP/WebSocket setup and reminder scheduler startup. |
| `voice_agent_project/wsgi.py` | WSGI setup; also starts the reminder scheduler. |
| `core/urls.py`, `core/views.py` | HTTP routes and doctor dashboard/reminder token endpoints. |
| `core/routing.py` | WebSocket route table. |
| `core/consumers.py` | Patient voice connection, audio/VAD/STT/dialog/TTS pipeline, cancellation, and transcript save. |
| `core/doc_consumers.py` | Doctor reminder calls and the doctor assistant voice consumer. |
| `core/models.py` | Doctor, appointment, reminder, reminder-event, legacy task, and transcript models. |
| `core/repositories.py` | Transactional ORM operations and transcript persistence. |
| `core/services/` | VAD, STT, TTS, Groq dialog, Calendar, and appointment services. |
| `core/reminders/` | Reminder lifecycle policy, service, scheduler, and sweep. |
| `core/tests.py` | Django tests for voice/WebSocket, scheduling, reminders, and integrations. |
| `manage.py`, `requirements.txt`, `frontend/package.json` | Django entry point and Python/JavaScript dependencies. |

## 3. Browser pages and interfaces

### Patient experience

`App.jsx` renders the patient application at `/` and other paths, and renders `DoctorDashboard` when the pathname is `/doc`. This is a pathname switch, not a client-side router.

The patient UI supports:

- Voice turns from the microphone control.
- Typed turns sent over the same WebSocket.
- Interim live captions from the browser SpeechRecognition API when supported.
- User/assistant transcript display and queued assistant audio playback.
- Automatic voice interruption during assistant playback (details in section 4).

The patient WebSocket URL is constructed by the frontend as:

```text
ws[s]://<current-page-host>:8000/ws/voice/
```

The page protocol selects `ws` or `wss`; the port is currently hard-coded as `8000`. The frontend's Vite development proxy also forwards `/api` requests to `http://127.0.0.1:8000`. Configure the backend and page origin to match these assumptions, or update the frontend when changing ports.

### Doctor dashboard

At `/doc`, the dashboard requests appointment and active-reminder data from the Django API. The API base is same-origin by default; `VITE_API_BASE_URL` may override it. With Vite's default development server, `/api` is proxied to port 8000.

The dashboard refreshes its appointment data periodically and polls active reminders more frequently. A reminder call and **Ask Janus** each obtain a short-lived signed token over HTTP, then connect to the WebSocket URL returned by Django. `DoctorCallPanel.jsx` manages the call's microphone, audio chunks, and call controls.

### HTTP endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Basic backend-running response. |
| `GET` | `/api/doc/dashboard/` | Doctor profile, upcoming appointments, all appointments, and the near-term reminder window. |
| `GET` | `/api/doc/reminders/active/` | Active or due-snoozed reminders; also records dashboard presence. |
| `POST` | `/api/doc/reminders/<reminder_id>/call-token/` | Issue a signed token and WebSocket URL for an eligible reminder call. |
| `POST` | `/api/doc/assistant/call-token/` | Issue a signed token and WebSocket URL for Ask Janus. |

The doctor dashboard, reminder, and call-token endpoints explicitly return 404 when `DEBUG` is false. The project is not configured as a production-ready deployment.

### WebSocket endpoints

| Path | Consumer | Purpose |
|---|---|---|
| `/ws/voice/` | `VoiceConsumer` | Patient conversation. The current `authorize()` implementation permits the connection without an application login. |
| `/ws/doc-voice/` | `DoctorVoiceConsumer` | Signed-token doctor reminder call. |
| `/ws/doc-assistant/` | `DoctorAssistantConsumer` | Signed-token doctor scheduling assistant. |

The ASGI application wraps WebSockets in Django authentication middleware and `AllowedHostsOriginValidator`; this middleware does not replace the consumer-level authorization check. Doctor tokens are timestamp-signed and validated with a 120-second maximum age.

## 4. Patient voice request/response cycle

### 4.1 Capture and transport

1. The browser requests microphone access with mono capture, echo cancellation, noise suppression, and automatic gain control enabled.
2. It requests an `AudioContext` at 16 kHz. At that rate, audio is processed in 512-sample frames; if the browser cannot provide the requested rate, the frontend uses a larger processing frame and converts/resamples client-side.
3. Float audio is converted to signed 16-bit PCM. Binary PCM audio frames are sent over the WebSocket. Text turns and control events are JSON messages on that socket.
4. Live captions may be generated in parallel by browser SpeechRecognition. These captions are for display; final recognition is performed by the backend STT service.

### 4.2 VAD, speech confirmation, and endpointing

`VoiceConsumer.receive()` buffers incoming bytes until complete 512-sample, 1024-byte VAD windows are available. `VADService.speech_probabilities()` returns a probability for each 32 ms window at 16 kHz. These are contiguous, non-overlapping windows.

The consumer uses the probabilities and per-connection buffers to:

- Keep up to about 300 ms of pre-speech audio to preserve utterance onsets.
- Confirm ordinary speech after at least three positive VAD windows (probability at least 0.5); brief gaps are tolerated.
- End an utterance after 700 ms of detected silence.
- Discard very short utterances below the 250 ms minimum and cap exceptionally long input at 25 seconds.
- Send a completed utterance to STT after endpointing.

### 4.3 Automatic interruption (barge-in)

The patient microphone stays active while assistant audio is playing. The frontend sends a `playback_complete` control message only after queued audio has finished; this lets the backend distinguish actual playback from the time it takes to generate/send audio.

During playback, barge-in is confirmed only when **at least six of the latest eight VAD windows** have probability at least **0.75**. This is stricter than normal speech-start detection and is intended to suppress brief clicks and cough-like bursts. Once confirmed, the backend cancels the active response task, marks its audio stopped, and retains the user's audio as the start of a new utterance. The new utterance then follows the regular silence endpoint → STT → dialog → TTS path; no interruption button is required.

VAD identifies speech-like activity, not the meaning of audio. Sustained coughs or other speech-like sounds can still meet the threshold. Browser echo cancellation/noise suppression are additional mitigations, not guarantees. No threshold can perfectly distinguish every cough from speech; the tests cover controlled VAD outputs and signal duration, not a comprehensive real-world cough/noise dataset.

During the assistant's thinking/processing phase before audio playback starts, the backend suppresses microphone frames so incidental noise does not cancel model or STT work. The user can interrupt once reply playback is active.

### 4.4 Recognition, dialog, and tools

After endpointing:

1. The backend converts the 16 kHz PCM samples to normalized `float32`.
2. Faster-Whisper transcribes on CPU, normally with the `base.en` model and `int8` compute. The model and language are configurable through `STT_MODEL` and `STT_LANGUAGE`; the implementation uses beam size 1 and disables Whisper's internal VAD and previous-text conditioning.
3. Empty transcription produces a spoken retry prompt rather than an empty dialog turn.
4. `LLMService.stream_dialog()` sends the conversation to Groq, streams text, and processes tool calls through a bounded tool loop. Available patient operations include availability lookup, booking, listing upcoming appointments, rescheduling, and cancellation.
5. Appointment and calendar rules are handled by the relevant service/repository. In particular, database availability checks do not query Google Calendar FreeBusy.

### 4.5 Sentence streaming, TTS, and persistence

The answer is not waited on in full before synthesis. Groq text is accumulated to sentence boundaries; Piper synthesizes each sentence locally, then the backend sends its mono signed 16-bit PCM as ordered Base64 `audio_chunk` messages (up to 8192 bytes per chunk). The browser queues chunks in order and schedules them with Web Audio. A final `agent_response` message contains the transcript, complete caption, and tool status. Transcript rows are saved per completed turn using a session ID.

### 4.6 Patient WebSocket messages

Implemented client JSON actions include:

- `process_text_turn` — submit a typed user message.
- `playback_complete` — confirm that the corresponding assistant audio turn finished playback.
- `stop_audio` — explicit cancellation/control path; it is separate from VAD-triggered interruption.
- `end_call` — close the connection.

Assistant messages include welcome/processing events, `audio_chunk`, `agent_response`, errors, and `audio_stopped`. Incoming microphone audio itself is sent as binary PCM, not a JSON-wrapped audio object.

## 5. Doctor workflows

### Reminder call

The dashboard polls for reminders eligible for action. A reminder call token is only issued when the reminder remains open, the appointment is scheduled, and its start time is in the future. The signed token authorizes a short-lived `/ws/doc-voice/` connection.

The reminder consumer loads appointment context, records that the call started, and speaks an appointment-specific greeting. It supports acknowledgement and snooze actions, with policy limits and reminder-event records. Call completion and appointment state are handled by the reminder service.

### Ask Janus

The dashboard obtains a signed token for `/ws/doc-assistant/`. This assistant uses doctor-specific context and appointment tools for lookups by patient/date/time, and for proposed cancellations/rescheduling. Destructive schedule changes require a separate affirmative confirmation before they are applied.

The doctor voice consumers share speech services and PCM transport concepts with the patient path, but they are distinct consumers and workflows.

## 6. Scheduling, persistence, and calendar integration

### Data model

- **Doctor:** clinic name/email/time zone and dashboard presence. A database constraint makes the doctor a singleton with ID 1.
- **Appointment:** patient/caller, start/end time, purpose, status, and optional Google Calendar event ID.
- **Reminder:** one-to-one with an appointment; status, next-fire time, attempt/snooze/call counts, and acknowledgement metadata.
- **ReminderEvent:** reminder lifecycle/event record with JSON payload.
- **Transcript:** a row per saved conversation turn, linked by a session ID; appointment linkage is optional.
- **Task:** retained legacy model; it is not the primary current reminder mechanism.

`DataRepository` wraps appointment updates and state transitions in database transactions. Appointment writes create, reset, or cancel their related reminders through `ReminderService`.

### Google Calendar

`GoogleCalendarService` looks for a `client_secret*.json` (or `credentials.json`) under the project root and stores OAuth credentials in root-level `token.json` by default. It can refresh credentials and create, update, or delete events. Booking/rescheduling/cancellation may synchronize Calendar events when valid credentials are available; sync failures are logged and the database remains the application's local appointment store.

The patient availability tool currently checks the local database, not Google Calendar FreeBusy. Treat Calendar synchronization and availability checking as separate behaviors.

### Reminder scheduler

Reminder policy defaults are configured in `settings.py` (lead time, retry interval, snooze interval, attempt limit, and minimum reminder lead time). The scheduler runs an in-process sweep and is started when either the ASGI or WSGI application module loads. Running multiple application worker processes can therefore start multiple sweepers; production deployments should move this to a single dedicated worker or an external scheduler before scaling.

## 7. Configuration and local development

`settings.py` loads a root `.env`. Important settings/environment variables include:

| Variable | Purpose |
|---|---|
| `DEBUG` | Enables development-only doctor endpoints; defaults to `true`. |
| `GROQ_API_KEY` | Required for Groq dialog. |
| `GROQ_MODEL` | Patient dialog model; defaults in `LLMService`. |
| `DOCTOR_GROQ_MODEL` | Optional doctor assistant model override. |
| `STT_MODEL`, `STT_LANGUAGE` | Faster-Whisper model/language overrides. |
| `PIPER_VOICE_PATH` | Optional Piper ONNX voice model path. |
| `VITE_API_BASE_URL` | Optional dashboard API base URL; empty means same-origin. |
| `REMINDER_LEAD_MINUTES`, `REMINDER_REPEAT_MINUTES` | Reminder lead and retry intervals. |
| `SNOOZE_MINUTES`, `SNOOZE_MIN_BUFFER_MINUTES`, `REMINDER_MAX_ATTEMPTS`, `REMINDER_MIN_LEAD_TO_REMIND_MINUTES` | Reminder snooze and attempt policy. |
| `DOCTOR_PRESENCE_TTL_SECONDS`, `VOICE_INCLUDE_PURPOSE`, `CALL_AUTO_HANGUP_AFTER_TERMINAL` | Doctor presence and voice-call behavior. |

The default Piper model path is `models/piper/en_US-lessac-medium.onnx`; the corresponding voice can be downloaded with:

```powershell
python -m piper.download_voices --download-dir models/piper en_US-lessac-medium
```

Typical local startup from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python manage.py migrate
daphne -b 127.0.0.1 -p 8000 voice_agent_project.asgi:application
```

In another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL (normally `http://localhost:5173/`); use `/doc` for the doctor dashboard. Keep the backend on port 8000 unless the hard-coded patient WebSocket URL and Vite `/api` proxy are changed. Set local credentials in `.env`; do not commit API keys, OAuth client secrets, or token files.

Useful checks:

```powershell
python manage.py check
python manage.py test core.tests
cd frontend
npm run build
```

The backend's `CHANNEL_LAYERS` uses an in-memory channel layer and its database is SQLite. These are convenient for local development, but are not shared across processes and are not a production scaling configuration.

## 8. Operational constraints and known boundaries

- **Development posture:** `DEBUG` defaults to true, `ALLOWED_HOSTS` is wildcarded, and the settings file contains a development secret. Configure deployment security before exposing the service.
- **Patient authorization:** the patient consumer currently accepts connections without application-level authentication. ASGI middleware does not change that consumer policy.
- **Doctor API access:** doctor dashboard/reminder APIs are disabled outside debug mode; review and implement production authentication/authorization before enabling them.
- **Scaling:** SQLite, the in-memory channel layer, singleton service caches, and an in-process reminder scheduler are local/single-process assumptions.
- **Calendar:** valid OAuth is needed for Google synchronization; local database status remains the source of local appointment data. Availability does not query Calendar FreeBusy.
- **Speech interruption:** probability thresholds reduce accidental interruptions but are not semantic speech recognition and cannot guarantee rejection of every cough/noise.
- **Browser behavior:** microphone capture and SpeechRecognition support depend on browser permissions and implementation. Capture options are requests to the browser/device, not universal guarantees.

## 9. Source-of-truth files

When behavior changes, update this guide alongside the implementation. The main runtime paths are `frontend/src/App.jsx`, `frontend/src/DoctorDashboard.jsx`, `frontend/src/DoctorCallPanel.jsx`, `core/consumers.py`, `core/doc_consumers.py`, `core/services/`, `core/reminders/`, `core/models.py`, and `core/repositories.py`.
