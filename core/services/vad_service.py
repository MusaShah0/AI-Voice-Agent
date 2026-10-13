import logging
from typing import Optional
import numpy as np
import torch

logger = logging.getLogger(__name__)

_vad_singleton: Optional["VADService"] = None


class VADService:
    def __init__(self):
        logger.info("Initializing Silero VAD on CPU...")
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True,
        )
        (self.get_speech_timestamps, _, _, self.vad_iterator_cls, _) = utils

    def is_speech(self, audio_chunk: bytes, sampling_rate: int = 16000, threshold: float = 0.5) -> bool:
        """
        Determines whether the provided PCM-16 audio chunk contains active speech.
        Evaluates 512-sample (32ms at 16kHz) windows using Silero VAD.
        """
        if not audio_chunk or len(audio_chunk) < 2:
            return False

        # Convert raw PCM-16 bytes into normalized float32 numpy array cleanly
        audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
        if len(audio_int16) == 0:
            return False

        audio_float = audio_int16.astype(np.float32) / 32768.0
        tensor = torch.from_numpy(audio_float)

        window_size = 512 if sampling_rate == 16000 else 256
        if len(tensor) < window_size:
            padded = torch.zeros(window_size, dtype=torch.float32)
            padded[:len(tensor)] = tensor
            prob = self.model(padded, sampling_rate).item()
            return prob >= threshold

        # Check slices of window_size across the chunk
        for start_idx in range(0, len(tensor) - window_size + 1, window_size):
            slice_tensor = tensor[start_idx:start_idx + window_size]
            prob = self.model(slice_tensor, sampling_rate).item()
            if prob >= threshold:
                return True

        return False


def get_vad_service() -> VADService:
    global _vad_singleton
    if _vad_singleton is None:
        _vad_singleton = VADService()
    return _vad_singleton