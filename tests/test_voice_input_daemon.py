"""
Unit tests for voice_input_daemon.py helper functions.

Audio hardware (sounddevice) and keyboard injection (pynput) are not tested —
those require hardware and a display server respectively.
"""

import io
import wave

import numpy as np
from pynput.keyboard import Key

from voice_input_daemon import parse_hotkey, to_wav_bytes


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
