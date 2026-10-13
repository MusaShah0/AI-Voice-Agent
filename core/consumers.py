import asyncio
import base64
import concurrent.futures
import json
import logging
import os
import re
import tempfile
import threading
import time
import wave
from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer
from .repositories import DataRepository
from .services.vad_service import VADService, get_vad_service
from .services.stt_service import STTService, get_stt_service
from .services.llm_service import LLMService
from .services.tts_service import TTSService, get_tts_service

logger = logging.getLogger(__name__)

AUDIO_SAMPLE_RATE = 16000
AUDIO_SAMPLE_WIDTH = 2
BYTES_PER_MS = (AUDIO_SAMPLE_RATE * AUDIO_SAMPLE_WIDTH) // 1000  # 32 bytes/ms

# VAD Endpointing & Buffering Parameters
SILENCE_THRESHOLD_MS = 700
SILENCE_THRESHOLD_BYTES = SILENCE_THRESHOLD_MS * BYTES_PER_MS  # ~22,400 bytes (~700ms silence)
PRE_SPEECH_BUFFER_BYTES = 300 * BYTES_PER_MS                    # ~9,600 bytes (~300ms lead-in)
MIN_SPEECH_BYTES = 250 * BYTES_PER_MS                           # ~8,000 bytes (~250ms min utterance)
MAX_SPEECH_BYTES = 25 * AUDIO_SAMPLE_RATE * AUDIO_SAMPLE_WIDTH  # ~800,000 bytes (~25s max safety ceiling)
AUDIO_CHUNK_BYTES = 8192
SENTENCE_BOUNDARY = re.compile(r'[.!?](?:["\')\]]*)\s+')
SENTENCE_ABBREVIATIONS = {"dr.", "mr.", "mrs.", "ms.", "e.g.", "i.e.", "etc."}


def _resolve_vad():
    if hasattr(VADService, 'return_value'):
        return VADService()
    return get_vad_service()


def _resolve_stt(model_size="base"):
    if hasattr(STTService, 'return_value'):
        return STTService(model_size)
    return get_stt_service(model_size)


def _resolve_tts():
    if hasattr(TTSService, 'return_value'):
        return TTSService()
    return get_tts_service()


class VoiceConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self.session_id = "session_" + str(self.channel_name)
        self.conversation_history = []
        self.active_task = None
        self.turn_counter = 0

        # Audio buffering & speech state
        self.audio_buffer = bytearray()
        self.pre_speech_buffer = bytearray()
        self.speech_detected = False
        self.consecutive_silence_bytes = 0
        self.speech_started_at = None
        self.speech_vad_elapsed_seconds = 0.0

        initialization_started = time.perf_counter()
        logger.info("Pipeline connect: initializing voice services session=%s", self.session_id)
        try:
            self.vad, self.stt, self.tts = await asyncio.gather(
                asyncio.to_thread(_resolve_vad),
                asyncio.to_thread(_resolve_stt, "base"),
                asyncio.to_thread(_resolve_tts),
            )
            logger.info(
                "Pipeline services ready session=%s elapsed_ms=%.2f",
                self.session_id,
                (time.perf_counter() - initialization_started) * 1000,
            )
            logger.info("Pipeline connect: initializing dialog service session=%s", self.session_id)
            self.llm = LLMService()
        except Exception:
            logger.exception(
                "Pipeline connect failed while initializing services session=%s elapsed_ms=%.2f",
                self.session_id,
                (time.perf_counter() - initialization_started) * 1000,
            )
            await self.close(code=4000)
            return

        logger.info("Pipeline ready: WebSocket connected session=%s", self.session_id)

    async def disconnect(self, close_code):
        if self.active_task and not self.active_task.done():
            logger.info("Pipeline disconnect: cancelling active turn session=%s", self.session_id)
            self.active_task.cancel()
        logger.info(
            "Pipeline disconnect: WebSocket closed session=%s close_code=%s",
            self.session_id,
            close_code,
        )

    async def receive(self, text_data=None, bytes_data=None):
        if bytes_data is not None:
            aligned_len = len(bytes_data) - (len(bytes_data) % AUDIO_SAMPLE_WIDTH)
            if aligned_len <= 0:
                return
            clean_chunk = bytes_data[:aligned_len]

            vad_started_at = time.perf_counter()
            try:
                has_speech = await asyncio.to_thread(self.vad.is_speech, clean_chunk)
            except Exception:
                logger.exception(
                    "Pipeline VAD failed session=%s chunk_bytes=%s",
                    self.session_id,
                    len(clean_chunk),
                )
                raise
            vad_elapsed = time.perf_counter() - vad_started_at
            if self.speech_detected or has_speech:
                self.speech_vad_elapsed_seconds += vad_elapsed

            if has_speech:
                # 1. Handle user interruption: cut off active TTS / processing task
                if self.active_task and not self.active_task.done():
                    logger.info("User interrupted active turn in session %s", self.session_id)
                    self.active_task.cancel()
                    self.active_task = None
                    await self.send(text_data=json.dumps({
                        "status": "audio_stopped",
                        "turn_id": self.turn_counter,
                    }))

                # 2. Transition into speech-detected state
                if not self.speech_detected:
                    self.speech_detected = True
                    self.speech_started_at = vad_started_at
                    self.audio_buffer.extend(self.pre_speech_buffer)
                    logger.info(
                        "Pipeline VAD: speech started session=%s pre_speech_bytes=%s",
                        self.session_id,
                        len(self.pre_speech_buffer),
                    )
                    self.pre_speech_buffer.clear()

                self.consecutive_silence_bytes = 0
                self.audio_buffer.extend(clean_chunk)

                # 3. Maximum speech duration safeguard
                if len(self.audio_buffer) >= MAX_SPEECH_BYTES:
                    audio_data = bytes(self.audio_buffer)
                    logger.info(
                        "Pipeline endpoint: maximum utterance reached session=%s audio_bytes=%s",
                        self.session_id,
                        len(audio_data),
                    )
                    self._log_voice_endpoint_timing()
                    self._reset_speech_state()
                    if not await self._start_task(
                        self.process_audio_pipeline(audio_data),
                        source="voice-max-duration",
                    ):
                        await self._send_busy()

            else:
                if self.speech_detected:
                    # User was speaking; accumulating silence pause
                    self.audio_buffer.extend(clean_chunk)
                    self.consecutive_silence_bytes += len(clean_chunk)

                    # 4. Silence endpoint detected (user finished speaking)
                    if self.consecutive_silence_bytes >= SILENCE_THRESHOLD_BYTES:
                        if len(self.audio_buffer) >= MIN_SPEECH_BYTES:
                            audio_data = bytes(self.audio_buffer)
                            logger.info(
                                "Pipeline endpoint: silence threshold reached session=%s audio_bytes=%s silence_ms=%s",
                                self.session_id,
                                len(audio_data),
                                SILENCE_THRESHOLD_MS,
                            )
                            self._log_voice_endpoint_timing()
                            self._reset_speech_state()
                            if not await self._start_task(
                                self.process_audio_pipeline(audio_data),
                                source="voice-silence-endpoint",
                            ):
                                await self._send_busy()
                        else:
                            # Transient noise pop; discard
                            logger.info(
                                "Pipeline endpoint: discarded short audio session=%s audio_bytes=%s",
                                self.session_id,
                                len(self.audio_buffer),
                            )
                            self._reset_speech_state()
                else:
                    # Ambient background before speech: maintain rolling pre-speech buffer
                    self.pre_speech_buffer.extend(clean_chunk)
                    if len(self.pre_speech_buffer) > PRE_SPEECH_BUFFER_BYTES:
                        del self.pre_speech_buffer[:-PRE_SPEECH_BUFFER_BYTES]

        elif text_data:
            try:
                data = json.loads(text_data)
            except json.JSONDecodeError:
                logger.warning("Pipeline receive: invalid JSON session=%s", self.session_id)
                await self.send(text_data=json.dumps({"type": "error", "message": "Invalid JSON"}))
                return

            action = data.get("action")
            logger.info("Pipeline receive: action=%s session=%s", action, self.session_id)

            if action == "stop_audio":
                stopped_turn_id = (
                    self.turn_counter
                    if self.active_task and not self.active_task.done()
                    else None
                )
                if self.active_task and not self.active_task.done():
                    logger.info("Pipeline stop: cancelling active turn session=%s", self.session_id)
                    self.active_task.cancel()
                self.active_task = None

                await self.send(text_data=json.dumps({
                    "status": "audio_stopped",
                    "turn_id": stopped_turn_id,
                }))

                # If user had speech pending when stopping, finalize and process
                if self.speech_detected and len(self.audio_buffer) >= MIN_SPEECH_BYTES:
                    audio_data = bytes(self.audio_buffer)
                    self._log_voice_endpoint_timing()
                    logger.info(
                        "Pipeline stop: processing buffered speech session=%s audio_bytes=%s",
                        self.session_id,
                        len(audio_data),
                    )
                    self._reset_speech_state()
                    await self._start_task(
                        self.process_audio_pipeline(audio_data),
                        source="voice-stop",
                    )
                else:
                    if self.speech_detected:
                        logger.info(
                            "Pipeline stop: discarded short buffered speech session=%s audio_bytes=%s",
                            self.session_id,
                            len(self.audio_buffer),
                        )
                    self._reset_speech_state()

                return

            if action == "process_text_turn":
                user_text = data.get("text", "")
                if user_text:
                    logger.info(
                        "Pipeline text input received session=%s characters=%s",
                        self.session_id,
                        len(user_text),
                    )
                    if not await self._start_task(
                        self.run_dialog_and_respond(user_text),
                        source="text",
                    ):
                        await self._send_busy()
                else:
                    logger.warning("Pipeline receive: empty text turn session=%s", self.session_id)
            elif action != "stop_audio":
                logger.warning("Pipeline receive: unknown action=%s session=%s", action, self.session_id)

    def _reset_speech_state(self):
        self.audio_buffer.clear()
        self.pre_speech_buffer.clear()
        self.speech_detected = False
        self.consecutive_silence_bytes = 0
        self.speech_started_at = None
        self.speech_vad_elapsed_seconds = 0.0

    def _log_voice_endpoint_timing(self):
        if self.speech_started_at is None:
            return

        logger.info(
            "Pipeline VAD: utterance endpointed session=%s speech_to_endpoint_ms=%.2f vad_compute_ms=%.2f",
            self.session_id,
            (time.perf_counter() - self.speech_started_at) * 1000,
            self.speech_vad_elapsed_seconds * 1000,
        )

    async def _start_task(self, coroutine, source="voice"):
        if self.active_task and not self.active_task.done():
            coroutine.close()
            return False

        await self.send(text_data=json.dumps({
            "type": "processing",
            "message": "Processing your request.",
        }))
        started_at = time.perf_counter()
        logger.info(
            "Pipeline turn: started session=%s source=%s",
            self.session_id,
            source,
        )
        self.active_task = asyncio.create_task(
            self._run_safely(coroutine, started_at, source)
        )
        self.active_task.add_done_callback(self._clear_active_task)
        return True

    async def _run_safely(self, coroutine, started_at, source):
        outcome = "completed"
        try:
            await coroutine
        except asyncio.CancelledError:
            outcome = "cancelled"
            logger.info("Pipeline turn: task cancelled session=%s", self.session_id)
            raise
        except Exception:
            outcome = "failed"
            logger.exception("Pipeline turn failed session=%s", self.session_id)
            await self.send(text_data=json.dumps({
                "type": "error",
                "message": "The voice turn could not be processed.",
            }))
        finally:
            logger.info(
                "Pipeline turn: total session=%s source=%s outcome=%s elapsed_ms=%.2f",
                self.session_id,
                source,
                outcome,
                (time.perf_counter() - started_at) * 1000,
            )

    def _clear_active_task(self, task):
        if self.active_task is task:
            self.active_task = None

    async def _send_busy(self):
        logger.warning("Pipeline turn rejected: another turn is active session=%s", self.session_id)
        await self.send(text_data=json.dumps({
            "type": "error",
            "message": "A voice turn is already being processed.",
        }))

    async def process_audio_pipeline(self, audio_data):
        audio_data = audio_data[:len(audio_data) - len(audio_data) % AUDIO_SAMPLE_WIDTH]
        if not audio_data:
            logger.warning("Pipeline STT skipped: empty audio session=%s", self.session_id)
            return

        logger.info(
            "Pipeline STT: preparing audio session=%s audio_bytes=%s duration_ms=%.0f",
            self.session_id,
            len(audio_data),
            len(audio_data) / BYTES_PER_MS,
        )
        audio_preparation_started = time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            temp_audio_path = temp_audio_file.name

        try:
            with wave.open(temp_audio_path, "wb") as audio_file:
                audio_file.setnchannels(1)
                audio_file.setsampwidth(AUDIO_SAMPLE_WIDTH)
                audio_file.setframerate(AUDIO_SAMPLE_RATE)
                audio_file.writeframes(audio_data)
            logger.info(
                "Pipeline STT: audio prepared session=%s elapsed_ms=%.2f",
                self.session_id,
                (time.perf_counter() - audio_preparation_started) * 1000,
            )

            stt_started = time.perf_counter()
            logger.info("Pipeline STT: transcription started session=%s", self.session_id)
            user_text = await asyncio.to_thread(self.stt.transcribe, temp_audio_path)
            logger.info(
                "Pipeline STT: transcription completed session=%s characters=%s elapsed_ms=%.2f",
                self.session_id,
                len(user_text),
                (time.perf_counter() - stt_started) * 1000,
            )
        except Exception:
            logger.exception("Pipeline STT failed session=%s", self.session_id)
            user_text = ""
        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                logger.debug("Pipeline STT: removed temporary WAV session=%s", self.session_id)

        if user_text.strip():
            await self.run_dialog_and_respond(user_text)
        else:
            logger.warning("Pipeline STT: no speech transcription produced session=%s", self.session_id)

    async def run_dialog_and_respond(self, user_text: str):
        dialog_started = time.perf_counter()
        self.turn_counter += 1
        turn_id = self.turn_counter
        logger.info(
            "Pipeline dialog: started session=%s user_characters=%s history_turns=%s",
            self.session_id,
            len(user_text),
            len(self.conversation_history) // 2,
        )
        logger.info(
            "Pipeline TTS: streaming synthesis started session=%s turn=%s",
            self.session_id,
            turn_id,
        )
        loop = asyncio.get_running_loop()
        stream_queue = asyncio.Queue(maxsize=16)
        stop_event = threading.Event()
        producer_task = asyncio.create_task(
            database_sync_to_async(
                self._publish_dialog_stream,
                thread_sensitive=True,
            )(
                stream_queue,
                loop,
                stop_event,
                user_text,
            )
        )
        response_parts = []
        sentence_buffer = ""
        sequence = 0

        async def synthesize_and_send(sentence: str):
            nonlocal sequence
            sentence = sentence.strip()
            if not sentence:
                return

            synthesis_started = time.perf_counter()
            audio_bytes, sample_rate = await asyncio.to_thread(
                self.tts.synthesize,
                sentence,
            )
            if not audio_bytes:
                raise RuntimeError("Piper returned no audio for a non-empty sentence")
            if len(audio_bytes) % AUDIO_SAMPLE_WIDTH:
                raise ValueError("Piper PCM output is not aligned to 16-bit samples")

            for offset in range(0, len(audio_bytes), AUDIO_CHUNK_BYTES):
                audio_chunk = audio_bytes[offset:offset + AUDIO_CHUNK_BYTES]
                if len(audio_chunk) % AUDIO_SAMPLE_WIDTH:
                    audio_chunk = audio_chunk[:-1]
                if not audio_chunk:
                    continue
                await self.send(text_data=json.dumps({
                    "type": "audio_chunk",
                    "turn_id": turn_id,
                    "sequence": sequence,
                    "format": "pcm_s16le",
                    "channels": 1,
                    "sample_rate": sample_rate,
                    "audio_payload": base64.b64encode(audio_chunk).decode("ascii"),
                }))
                sequence += 1

            logger.info(
                "Pipeline TTS: sentence synthesized session=%s turn=%s audio_bytes=%s elapsed_ms=%.2f",
                self.session_id,
                turn_id,
                len(audio_bytes),
                (time.perf_counter() - synthesis_started) * 1000,
            )

        try:
            stream_complete = False
            while not stream_complete:
                item = await stream_queue.get()
                if "error" in item:
                    raise RuntimeError("Groq response stream failed") from item["error"]
                if item.get("finished"):
                    stream_complete = True
                    continue

                fragment = item.get("fragment", "")
                response_parts.append(fragment)
                sentence_buffer += fragment
                while True:
                    boundary = SENTENCE_BOUNDARY.search(sentence_buffer)
                    if boundary is None:
                        break
                    sentence = sentence_buffer[:boundary.end()].strip()
                    preceding_word = re.search(r"(?:^|\s)([\w.]+)$", sentence)
                    if preceding_word and preceding_word.group(1).lower() in SENTENCE_ABBREVIATIONS:
                        next_boundary = SENTENCE_BOUNDARY.search(sentence_buffer, boundary.end())
                        if next_boundary is None:
                            break
                        boundary = next_boundary
                        sentence = sentence_buffer[:boundary.end()].strip()
                    sentence_buffer = sentence_buffer[boundary.end():]
                    await synthesize_and_send(sentence)

            if sentence_buffer.strip():
                await synthesize_and_send(sentence_buffer)
        finally:
            stop_event.set()
            if not producer_task.done():
                producer_task.cancel()
            try:
                await producer_task
            except asyncio.CancelledError:
                pass

        response_text = "".join(response_parts).strip()
        tool_status = self.llm.last_tool_status
        logger.info(
            "Pipeline dialog: completed session=%s response_characters=%s tool_status=%s elapsed_ms=%.2f",
            self.session_id,
            len(response_text),
            tool_status or "none",
            (time.perf_counter() - dialog_started) * 1000,
        )
        logger.info(
            "Pipeline TTS: streaming synthesis completed session=%s turn=%s audio_chunks=%s",
            self.session_id,
            turn_id,
            sequence,
        )

        # Transcript Persistence with try-except fallback
        try:
            persistence_started = time.perf_counter()
            logger.info("Pipeline persistence: saving transcript session=%s", self.session_id)
            await database_sync_to_async(
                DataRepository.save_transcript,
                thread_sensitive=True,
            )(
                session_id=self.session_id,
                full_transcript=f"User: {user_text} | Agent: {response_text}",
            )
            logger.info(
                "Pipeline persistence: transcript saved session=%s elapsed_ms=%.2f",
                self.session_id,
                (time.perf_counter() - persistence_started) * 1000,
            )
        except Exception:
            logger.exception("Pipeline persistence failed session=%s", self.session_id)

        # Send response to client with try-except fallback
        try:
            delivery_started = time.perf_counter()
            logger.info("Pipeline delivery: sending response session=%s", self.session_id)
            await self.send(text_data=json.dumps({
                "type": "agent_response",
                "turn_id": turn_id,
                "user_transcription": user_text,
                "caption": response_text,
                "tool_status": tool_status,
            }))
            logger.info(
                "Pipeline delivery: response sent session=%s elapsed_ms=%.2f",
                self.session_id,
                (time.perf_counter() - delivery_started) * 1000,
            )
        except Exception:
            logger.exception("Pipeline delivery failed session=%s", self.session_id)

    def _publish_dialog_stream(self, stream_queue, loop, stop_event, user_text):
        def publish(item):
            while not stop_event.is_set():
                try:
                    pending_put = asyncio.run_coroutine_threadsafe(
                        stream_queue.put(item),
                        loop,
                    )
                    pending_put.result(timeout=0.1)
                    return True
                except concurrent.futures.TimeoutError:
                    pending_put.cancel()
                except RuntimeError:
                    return False
            return False

        try:
            for fragment in self.llm.stream_dialog(self.conversation_history, user_text):
                if not publish({"fragment": fragment}):
                    break
        except Exception as stream_error:
            logger.exception("Pipeline Groq: streaming dialog failed session=%s", self.session_id)
            publish({"error": stream_error})
        finally:
            publish({"finished": True})