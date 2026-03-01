"""
Unit tests for voice_input_daemon.py helper functions.

Audio hardware (sounddevice) and keyboard injection (pynput) are not tested —
those require hardware and a display server respectively.
"""

import io
import wave

import numpy as np
import pytest
from pynput.keyboard import Key

from voice_input_daemon import parse_hotkey, to_wav_bytes, transcribe_stream


class TestParseHotkey:
    def test_modifier_plus_space(self):
        """parse_hotkey('ctrl+alt+space') must return the three expected Key members."""
        result = parse_hotkey("ctrl+alt+space")
        assert Key.ctrl in result
        assert Key.alt in result
        assert Key.space in result

    def test_single_modifier(self):
        """parse_hotkey with one key returns a frozenset with that key."""
        result = parse_hotkey("ctrl")
        assert Key.ctrl in result
        assert len(result) == 1

    def test_character_key(self):
        """parse_hotkey falls back to KeyCode for plain character keys."""
        from pynput.keyboard import KeyCode

        result = parse_hotkey("ctrl+r")
        assert Key.ctrl in result
        assert KeyCode.from_char("r") in result

    def test_unknown_key_returns_keycode(self):
        """parse_hotkey falls back to KeyCode for any string that is not a Key name.

        'foobar123' is not a valid Key enum member, so parse_hotkey silently
        wraps it in a KeyCode rather than raising — callers are responsible for
        passing sensible key names.
        """
        from pynput.keyboard import KeyCode

        result = parse_hotkey("ctrl+foobar123")
        assert Key.ctrl in result
        # The unknown part becomes a KeyCode, not a Key enum member
        key_codes = {k for k in result if isinstance(k, KeyCode)}
        assert len(key_codes) == 1


class TestToWavBytes:
    def _make_frames(self, seconds: float = 1.0) -> list[np.ndarray]:
        """Generate silent mono int16 frames at 16 kHz."""
        samples = int(16000 * seconds)
        return [np.zeros((samples, 1), dtype="int16")]

    def test_roundtrip_sample_rate(self):
        """to_wav_bytes produces a WAV with framerate == 16000."""
        wav = to_wav_bytes(self._make_frames())
        with wave.open(io.BytesIO(wav)) as wf:
            assert wf.getframerate() == 16000

    def test_roundtrip_mono(self):
        """to_wav_bytes produces a mono (1-channel) WAV."""
        wav = to_wav_bytes(self._make_frames())
        with wave.open(io.BytesIO(wav)) as wf:
            assert wf.getnchannels() == 1

    def test_roundtrip_int16(self):
        """to_wav_bytes produces a WAV with 2-byte (int16) sample width."""
        wav = to_wav_bytes(self._make_frames())
        with wave.open(io.BytesIO(wav)) as wf:
            assert wf.getsampwidth() == 2

    def test_roundtrip_frame_count(self):
        """to_wav_bytes encodes the correct number of frames."""
        seconds = 1.5
        wav = to_wav_bytes(self._make_frames(seconds))
        with wave.open(io.BytesIO(wav)) as wf:
            assert wf.getnframes() == int(16000 * seconds)


class _FakeResponse:
    def __init__(self, lines: list[str], status_code: int = 200):
        self._lines = lines
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError("bad status")

    def iter_lines(self, decode_unicode: bool = False):
        for line in self._lines:
            yield line if decode_unicode else line.encode("utf-8")


class TestTranscribeStream:
    def test_transcribe_stream_happy_path(self, monkeypatch):
        """transcribe_stream returns final text and calls on_partial for chunks."""
        lines = [
            "event: partial",
            'data: {"text": "hello"}',
            "",
            "event: partial",
            'data: {"text": "world"}',
            "",
            "event: done",
            'data: {"text": "hello world"}',
            "",
        ]
        response = _FakeResponse(lines)
        captured: list[str] = []

        def fake_post(*args, **kwargs):
            return response

        monkeypatch.setattr("requests.post", fake_post)

        result = transcribe_stream(b"wav", "http://localhost:8178", "en", captured.append)
        assert result == "hello world"
        assert captured == ["hello", "world"]

    def test_transcribe_stream_error_event(self, monkeypatch):
        """transcribe_stream raises on error event payload."""
        lines = ["event: error", 'data: {"error": "boom"}', ""]
        response = _FakeResponse(lines)

        def fake_post(*args, **kwargs):
            return response

        monkeypatch.setattr("requests.post", fake_post)

        with pytest.raises(RuntimeError, match="boom"):
            transcribe_stream(b"wav", "http://localhost:8178", "en")
