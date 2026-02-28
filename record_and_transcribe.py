#!/usr/bin/env python3
"""
record_and_transcribe.py - Record from the microphone and transcribe via the
whisper-gate API.

Usage:
    python record_and_transcribe.py --url http://localhost:8178 --duration 5 --lang en
    python record_and_transcribe.py --list-devices
"""

import argparse
import io
import sys
import wave

import numpy as np
import requests
import sounddevice as sd

SAMPLE_RATE = 16000  # whisper expects 16 kHz
CHANNELS = 1


def list_devices() -> None:
    print(sd.query_devices())


def record(duration: int, device: int | None) -> np.ndarray:
    print(f"🎙  Recording for {duration}s …", flush=True)
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        device=device,
    )
    sd.wait()
    print("⏹  Done recording.", flush=True)
    return audio


def to_wav_bytes(audio: np.ndarray) -> bytes:
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
    return resp.json().get("text", "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record and transcribe via whisper-gate API")
    parser.add_argument("--url", default="http://localhost:8178", help="Base URL of the API")
    parser.add_argument("--duration", type=int, default=5, help="Recording duration in seconds")
    parser.add_argument("--lang", default="en", help="Language code (e.g. en, fr, auto)")
    parser.add_argument(
        "--device", type=int, default=None, help="Input device index (see --list-devices)"
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="List available audio input devices and exit"
    )
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        sys.exit(0)

    audio = record(args.duration, args.device)
    wav_bytes = to_wav_bytes(audio)

    print("⏳  Transcribing …", flush=True)
    try:
        text = transcribe(wav_bytes, args.url, args.lang)
    except requests.HTTPError as exc:
        print(f"❌  API error: {exc.response.status_code} - {exc.response.text}", file=sys.stderr)
        sys.exit(1)

    print(f"\n📝  {text}\n")


if __name__ == "__main__":
    main()
