import logging
import os
from pathlib import Path
from typing import Optional, Tuple

from piper import PiperVoice

logger = logging.getLogger(__name__)

_tts_singleton: Optional["TTSService"] = None


def clean_spoken_text(text: str) -> str:
    return text.replace("*", "").strip()


class TTSService:
    def __init__(self):
        default_voice_path = (
            Path(__file__).resolve().parents[2]
            / "models"
            / "piper"
            / "en_US-lessac-medium.onnx"
        )
        configured_voice_path = Path(
            os.environ.get("PIPER_VOICE_PATH", str(default_voice_path))
        )
        if not configured_voice_path.is_absolute():
            configured_voice_path = Path(__file__).resolve().parents[2] / configured_voice_path
        if not configured_voice_path.is_file():
            raise FileNotFoundError(
                f"Piper voice model not found at {configured_voice_path}. "
                "Download en_US-lessac-medium with "
                "`python -m piper.download_voices --download-dir models/piper en_US-lessac-medium`."
            )

        logger.info("Loading local Piper voice from %s", configured_voice_path)
        self.voice = PiperVoice.load(configured_voice_path)
        self.sample_rate = self.voice.config.sample_rate
        logger.info("Local Piper voice ready sample_rate=%s", self.sample_rate)

    def synthesize(self, text: str) -> Tuple[bytes, int]:
        """Synthesize text into mono signed 16-bit PCM and return audio plus sample rate."""
        text = clean_spoken_text(text)
        if not text:
            return b"", self.sample_rate

        audio_chunks = []
        for chunk in self.voice.synthesize(text):
            if chunk.sample_width != 2 or chunk.sample_channels != 1:
                raise ValueError(
                    "Piper voice must produce mono 16-bit PCM audio "
                    f"(got sample_width={chunk.sample_width}, channels={chunk.sample_channels})"
                )
            if chunk.sample_rate != self.sample_rate:
                raise ValueError(
                    f"Piper sample rate changed during synthesis: {chunk.sample_rate}"
                )
            audio_chunks.append(chunk.audio_int16_bytes)

        return b"".join(audio_chunks), self.sample_rate


def get_tts_service() -> TTSService:
    global _tts_singleton
    if _tts_singleton is None:
        _tts_singleton = TTSService()
    return _tts_singleton
