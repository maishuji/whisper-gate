"""
Unit tests for record_and_transcribe.py helper functions.
"""

import io
import wave

import numpy as np

from record_and_transcribe import to_wav_bytes


class TestToWavBytes:
    def _silent_audio(self, seconds: float = 1.0) -> np.ndarray:
        """Return a silent mono int16 ndarray at 16 kHz."""
        return np.zeros((int(16000 * seconds), 1), dtype="int16")

    def test_sample_rate(self):
        """to_wav_bytes encodes audio at 16 kHz as required by whisper.cpp."""
        wav = to_wav_bytes(self._silent_audio())
        with wave.open(io.BytesIO(wav)) as wf:
            assert wf.getframerate() == 16000

    def test_mono(self):
        """to_wav_bytes produces a single-channel (mono) WAV file."""
        wav = to_wav_bytes(self._silent_audio())
        with wave.open(io.BytesIO(wav)) as wf:
            assert wf.getnchannels() == 1

    def test_int16_sample_width(self):
        """to_wav_bytes uses 2-byte sample width (int16)."""
        wav = to_wav_bytes(self._silent_audio())
        with wave.open(io.BytesIO(wav)) as wf:
            assert wf.getsampwidth() == 2

    def test_frame_count(self):
        """to_wav_bytes preserves the exact number of audio frames."""
        seconds = 2.0
        wav = to_wav_bytes(self._silent_audio(seconds))
        with wave.open(io.BytesIO(wav)) as wf:
            assert wf.getnframes() == int(16000 * seconds)
