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

from voice_input_daemon import PushToTalkDaemon, parse_hotkey, to_wav_bytes, transcribe_stream


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
        """transcribe_stream returns final text, calls on_partial for chunks, and
        calls on_received when the server acknowledges receipt."""
        lines = [
            "event: received",
            'data: {"status": "queued"}',
            "",
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
        captured_partial: list[str] = []
        captured_received: list[str] = []

        def fake_post(*args, **kwargs):
            return response

        monkeypatch.setattr("requests.post", fake_post)

        result = transcribe_stream(
            b"wav",
            "http://localhost:8178",
            "en",
            on_partial=captured_partial.append,
            on_received=captured_received.append,
        )
        assert result == "hello world"
        assert captured_partial == ["hello", "world"]
        assert captured_received == ["queued"]

    def test_transcribe_stream_received_ignored_when_no_callback(self, monkeypatch):
        """transcribe_stream silently ignores the received event when no on_received
        callback is supplied, returning the final text as normal."""
        lines = [
            "event: received",
            'data: {"status": "queued"}',
            "",
            "event: done",
            'data: {"text": "hi"}',
            "",
        ]
        response = _FakeResponse(lines)

        monkeypatch.setattr("requests.post", lambda *a, **kw: response)

        # No on_received supplied — must not raise
        result = transcribe_stream(b"wav", "http://localhost:8178", "en")
        assert result == "hi"

    def test_transcribe_stream_error_event(self, monkeypatch):
        """transcribe_stream raises on error event payload."""
        lines = ["event: error", 'data: {"error": "boom"}', ""]
        response = _FakeResponse(lines)

        def fake_post(*args, **kwargs):
            return response

        monkeypatch.setattr("requests.post", fake_post)

        with pytest.raises(RuntimeError, match="boom"):
            transcribe_stream(b"wav", "http://localhost:8178", "en")


class TestPushToTalkStateMachine:
    def test_rearm_requires_full_hotkey_release(self):
        """After stop, daemon must not restart until all hotkey keys are released."""
        daemon = PushToTalkDaemon(
            url="http://localhost:8178",
            lang="en",
            hotkey=parse_hotkey("ctrl+alt+space"),
        )

        starts = 0
        stops = 0

        def fake_start() -> None:
            nonlocal starts
            starts += 1
            daemon._recording = True

        def fake_stop() -> None:
            nonlocal stops
            stops += 1
            daemon._recording = False

        daemon._start_recording = fake_start
        daemon._stop_recording = fake_stop

        daemon._on_press(Key.ctrl)
        daemon._on_press(Key.alt)
        daemon._on_press(Key.space)
        assert starts == 1

        daemon._on_release(Key.space)
        assert stops == 1

        daemon._on_press(Key.space)
        assert starts == 1

        daemon._on_release(Key.ctrl)
        daemon._on_release(Key.alt)
        daemon._on_release(Key.space)

        daemon._on_press(Key.ctrl)
        daemon._on_press(Key.alt)
        daemon._on_press(Key.space)
        assert starts == 2

    def test_synthetic_space_during_typing_does_not_start_recording(self):
        """Synthetic Key.space from type_text must not trigger a ghost recording.

        Root cause of the "remote duplicate" bug: pynput's Controller.type()
        synthesises X11 events that the Listener also receives.  If the user
        holds ctrl+alt while waiting for a slow remote API, a synthetic
        Key.space press completes the hotkey combo and starts an unintended
        second recording.  The _is_typing guard must prevent this.
        """
        daemon = PushToTalkDaemon(
            url="http://localhost:8178",
            lang="en",
            hotkey=parse_hotkey("ctrl+alt+space"),
        )

        starts = 0

        def fake_start() -> None:
            nonlocal starts
            starts += 1

        daemon._start_recording = fake_start

        # User pre-stages ctrl+alt while waiting for the remote API to respond.
        daemon._on_press(Key.ctrl)
        daemon._on_press(Key.alt)

        # Daemon begins typing the previous result — _is_typing = True.
        daemon._is_typing = True

        # Synthetic Key.space from Controller.type(" ") must be ignored.
        daemon._on_press(Key.space)
        assert starts == 0, "ghost recording must not start while _is_typing is True"

        daemon._on_release(Key.space)

        # Typing finishes — _is_typing cleared.
        daemon._is_typing = False

        # Real user press of space should now start a new recording normally.
        daemon._on_press(Key.space)
        assert starts == 1, "recording must start again once _is_typing is False"
