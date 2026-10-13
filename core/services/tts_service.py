import io
import logging
from typing import Optional
from gtts import gTTS

logger = logging.getLogger(__name__)

_tts_singleton: Optional["TTSService"] = None


class TTSService:
    def __init__(self, lang="en", slow=False):
        self.lang = lang
        self.slow = slow

    def synthesize(self, text: str) -> bytes:
        """
        Converts text response into raw audio bytes (MP3 format) 
        optimized for lightweight CPU execution.
        """
        if not text.strip():
            return b""

        try:
            fp = io.BytesIO()
            tts = gTTS(text=text, lang=self.lang, slow=self.slow)
            tts.write_to_fp(fp)
            fp.seek(0)
            return fp.read()
        except Exception as e:
            logger.error("TTS Synthesis Error: %s", e)
            return b""


def get_tts_service(lang: str = "en", slow: bool = False) -> TTSService:
    global _tts_singleton
    if _tts_singleton is None:
        _tts_singleton = TTSService(lang=lang, slow=slow)
    return _tts_singleton