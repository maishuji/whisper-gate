#!/usr/bin/env python3
"""
voice_input_daemon.py - Push-to-talk daemon that records while a hotkey is
held, transcribes via the whisper-gate API, then types the result.

Usage:
    python voice_input_daemon.py --url http://localhost:8178 --lang en --key "ctrl+alt+space"
"""

import argparse
import io
import json
import sys
import threading
import time
import wave
from collections.abc import Callable

import numpy as np
import requests
import sounddevice as sd
from pynput import keyboard
from pynput.keyboard import Controller, Key

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
CHANNELS = 1
MIN_DURATION = 0.5  # seconds - recordings shorter than this are skipped
DTYPE = "int16"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_hotkey(key_str: str) -> frozenset:
    """
    Convert a string like "ctrl+alt+space" into a frozenset of pynput keys,
    e.g. {Key.ctrl, Key.alt, Key.space}.
    """
    parts = [p.strip().lower() for p in key_str.split("+")]
    keys: set = set()
    for part in parts:
        # Try special keys first (ctrl, alt, shift, space, …)
        try:
            keys.add(Key[part])
        except KeyError:
            # Single character key
            keys.add(keyboard.KeyCode.from_char(part))
    return frozenset(keys)


def to_wav_bytes(frames: list[np.ndarray]) -> bytes:
    audio = np.concatenate(frames, axis=0)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # int16 → 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


def transcribe(wav_bytes: bytes, url: str, lang: str) -> str:
    resp = requests.post(
        f"{url.rstrip('/')}/transcribe",
        files={"audio": ("audio.wav", wav_bytes, "audio/wav")},
        data={"lang": lang},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("text", "").strip()


def transcribe_stream(
    wav_bytes: bytes,
    url: str,
    lang: str,
    on_partial: Callable[[str], None] | None = None,
    on_received: Callable[[str], None] | None = None,
) -> str:
    resp = requests.post(
        f"{url.rstrip('/')}/transcribe/stream",
        files={"audio": ("audio.wav", wav_bytes, "audio/wav")},
        data={"lang": lang},
        timeout=120,
        stream=True,
    )
    resp.raise_for_status()

    current_event: str | None = None
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("event:"):
            current_event = line.split(":", 1)[1].strip()
            continue
        if line.startswith("data:"):
            payload = line.split(":", 1)[1].strip()
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue

            if current_event == "received":
                status = str(data.get("status", "queued"))
                if on_received is not None:
                    on_received(status)
            elif current_event == "partial":
                text = str(data.get("text", ""))
                if on_partial is not None:
                    on_partial(text)
            elif current_event == "done":
                return str(data.get("text", "")).strip()
            elif current_event == "error":
                message = str(data.get("error", "unknown error"))
                raise RuntimeError(message)

    return ""


def type_text(text: str) -> None:
    """Type text at the current cursor position using pynput."""
    ctrl = Controller()
    # Small delay to let the user's key release propagate before typing
    time.sleep(0.05)
    ctrl.type(text)


# ---------------------------------------------------------------------------
# Daemon
# ---------------------------------------------------------------------------


class PushToTalkDaemon:
    def __init__(self, url: str, lang: str, hotkey: frozenset):
        self.url = url
        self.lang = lang
        self.hotkey = hotkey

        self._pressed: set = set()
        self._recording = False
        self._armed = True
        self._lock = threading.Lock()
        self._frames: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if self._recording:
            self._frames.append(indata.copy())

    def _start_recording(self) -> None:
        self._frames = []
        self._recording = True
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=self._audio_callback,
        )
        self._stream.start()
        print("\n🔴 Recording … (release hotkey to stop)", flush=True)

    def _stop_recording(self) -> None:
        self._recording = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        # Snapshot and clear immediately so any duplicate stop call or
        # concurrent _audio_callback append cannot affect this session.
        frames = list(self._frames)
        self._frames = []
        print("⏹  Stopped. Transcribing …", flush=True)
        threading.Thread(target=self._process, args=(frames,), daemon=True).start()

    def _process(self, frames: list[np.ndarray]) -> None:
        if not frames:
            print("  (no audio, skipped)", flush=True)
            return

        # Duration check
        total_samples = sum(f.shape[0] for f in frames)
        duration = total_samples / SAMPLE_RATE
        if duration < MIN_DURATION:
            print("  (too short, skipped)", flush=True)
            return

        wav_bytes = to_wav_bytes(frames)
        partials: list[str] = []

        def _on_partial(text: str) -> None:
            partials.append(text)
            sys.stdout.write(f"\r  ✍️  {text}")
            sys.stdout.flush()

        def _on_received(status: str) -> None:
            sys.stdout.write(f"\r  ⏳ {status}…")
            sys.stdout.flush()

        try:
            text = transcribe_stream(
                wav_bytes,
                self.url,
                self.lang,
                on_partial=_on_partial,
                on_received=_on_received,
            )
        except requests.HTTPError as exc:
            if partials:
                print("", flush=True)
            print(f"  ❌ API error: {exc.response.status_code}", flush=True)
            return
        except Exception as exc:
            if partials:
                print("", flush=True)
            print(f"  ❌ Error: {exc}", flush=True)
            return

        if partials:
            print("", flush=True)

        if not text:
            print("  (empty result, skipped)", flush=True)
            return

        print(f"  ✅ {text}", flush=True)
        type_text(text)

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def _normalise(self, key) -> object:
        """Return a canonical representation of a key for set membership."""
        # Map left/right modifiers to their generic counterpart
        _aliases = {
            Key.ctrl_l: Key.ctrl,
            Key.ctrl_r: Key.ctrl,
            Key.alt_l: Key.alt,
            Key.alt_r: Key.alt,
            Key.shift_l: Key.shift,
            Key.shift_r: Key.shift,
        }
        return _aliases.get(key, key)

    def _hotkey_active(self) -> bool:
        return all(self._normalise(k) in self._pressed for k in self.hotkey)

    def _any_hotkey_key_pressed(self) -> bool:
        return any(self._normalise(k) in self._pressed for k in self.hotkey)

    def _on_press(self, key) -> None:
        normalised = self._normalise(key)
        with self._lock:
            self._pressed.add(normalised)
            if self._hotkey_active() and self._armed and not self._recording:
                self._start_recording()

    def _on_release(self, key) -> None:
        normalised = self._normalise(key)
        with self._lock:
            was_active = self._hotkey_active()
            self._pressed.discard(normalised)
            if was_active and self._recording:
                self._stop_recording()
                self._armed = False
            if not self._any_hotkey_key_pressed():
                self._armed = True

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> None:
        hotkey_str = "+".join(str(k) for k in self.hotkey)
        print("🎙  Push-to-talk daemon ready.")
        print(f"    Hotkey : {hotkey_str}")
        print(f"    API    : {self.url}")
        print(f"    Lang   : {self.lang}")
        print("    Hold the hotkey and speak. Release to transcribe & type.")
        print("    Press Ctrl+C to quit.\n", flush=True)

        with keyboard.Listener(on_press=self._on_press, on_release=self._on_release) as listener:
            listener.join()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Push-to-talk transcription daemon")
    parser.add_argument("--url", default="http://localhost:8178", help="whisper-gate API base URL")
    parser.add_argument("--lang", default="en", help="Language code (e.g. en, fr, auto)")
    parser.add_argument("--key", default="ctrl+alt+space", help="Hotkey combo, e.g. ctrl+alt+space")
    args = parser.parse_args()

    hotkey = parse_hotkey(args.key)
    daemon = PushToTalkDaemon(url=args.url, lang=args.lang, hotkey=hotkey)
    try:
        daemon.run()
    except KeyboardInterrupt:
        print("\nBye.", flush=True)
        sys.exit(0)


if __name__ == "__main__":
    main()
