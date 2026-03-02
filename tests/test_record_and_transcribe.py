"""
Unit tests for record_and_transcribe.py helper functions.
"""

import io
import wave
from unittest.mock import Mock, patch

import numpy as np
import pytest
import requests

from record_and_transcribe import normalize_base_url, to_wav_bytes, transcribe


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


class TestTranscribe:
    def test_normalize_base_url_adds_http_for_bare_host(self):
        """normalize_base_url prepends http:// when scheme is omitted."""
        assert normalize_base_url("100.110.122.117:8178") == "http://100.110.122.117:8178"

    def test_normalize_base_url_keeps_existing_scheme(self):
        """normalize_base_url keeps explicit http/https schemes unchanged."""
        assert normalize_base_url("http://localhost:8178") == "http://localhost:8178"
        assert normalize_base_url("https://example.com") == "https://example.com"

    def test_returns_text_on_success(self):
        """transcribe returns the API `text` field when the request succeeds."""
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"text": "hello"}

        with patch("record_and_transcribe.requests.post", return_value=response):
            text = transcribe(b"wav", "localhost:8178", "en")

        assert text == "hello"

    def test_uses_normalized_endpoint_when_scheme_is_missing(self):
        """transcribe posts to an http endpoint when URL has no scheme."""
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"text": "ok"}

        with patch("record_and_transcribe.requests.post", return_value=response) as post_mock:
            transcribe(b"wav", "100.110.122.117:8178", "en")

        assert post_mock.call_args.args[0] == "http://100.110.122.117:8178/transcribe"

    def test_connection_error_has_actionable_message(self):
        """transcribe raises a clear RuntimeError when the API is unreachable."""
        with (
            patch(
                "record_and_transcribe.requests.post",
                side_effect=requests.ConnectionError("refused"),
            ),
            pytest.raises(RuntimeError) as exc_info,
        ):
            transcribe(b"wav", "http://localhost:8178", "en")

        assert "Cannot reach whisper API" in str(exc_info.value)
        assert "make run" in str(exc_info.value)
