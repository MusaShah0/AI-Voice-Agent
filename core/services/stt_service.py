import logging
import os
from typing import Optional

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

_stt_singletons = {}


class STTService:
    def __init__(self, model_size="base.en", language="en"):
        logger.info(
            "Initializing Faster-Whisper (%s) on CPU language=%s...",
            model_size,
            language,
        )
        self.model_size = model_size
        self.language = language
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, audio_input) -> str:
        """
        Transcribes audio (file path, raw PCM-16 bytes, or float32 numpy array) into text using Faster-Whisper.
        """
        import numpy as np
        import wave

        audio_data = None
        if isinstance(audio_input, np.ndarray):
            audio_data = audio_input
        elif isinstance(audio_input, (bytes, bytearray)):
            audio_data = np.frombuffer(audio_input, dtype=np.int16).astype(np.float32) / 32768.0
        elif isinstance(audio_input, str):
            if not os.path.exists(audio_input):
                return ""
            try:
                with wave.open(audio_input, "rb") as wf:
                    frames = wf.readframes(wf.getnframes())
                    arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    sr = wf.getframerate()
                    if sr != 16000 and len(arr) > 0:
                        num_16k = int(len(arr) * 16000 / sr)
                        arr = np.interp(np.linspace(0, len(arr), num_16k, endpoint=False), np.arange(len(arr)), arr).astype(np.float32)
                    audio_data = arr
            except Exception:
                audio_data = audio_input
        else:
            return ""

        try:
            segments, _ = self.model.transcribe(
                audio_data,
                beam_size=1,
                vad_filter=False,
                condition_on_previous_text=False,
                language=self.language,
            )
            transcript = " ".join([segment.text for segment in segments])
            return transcript.strip()
        except Exception as e:
            logger.error("STT Transcription Error: %s", e)
            return ""


def get_stt_service(model_size: str = "base.en", language: str = "en") -> STTService:
    global _stt_singletons
    singleton_key = (model_size, language)
    if singleton_key not in _stt_singletons:
        _stt_singletons[singleton_key] = STTService(
            model_size=model_size,
            language=language,
        )
    return _stt_singletons[singleton_key]