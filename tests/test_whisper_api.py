"""
Unit tests for whisper_api.py.

All subprocess and filesystem calls are mocked so no GPU, whisper-cli binary,
or model file is required to run these tests.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Patch os.path.exists globally so the app believes CLI and model are present
# by default.  Individual tests can override this when testing missing-file paths.
_FAKE_CLI = "/fake/whisper-cli"
_FAKE_MODEL = "/fake/ggml-base.bin"


@pytest.fixture(autouse=True)
def patch_paths(monkeypatch):
    """Make CLI and MODEL point to fake paths that os.path.exists returns True for."""
    monkeypatch.setenv("WHISPER_CLI", _FAKE_CLI)
    monkeypatch.setenv("WHISPER_MODEL", _FAKE_MODEL)


@pytest.fixture()
def client():
    """Return a TestClient with the CLI/model env vars already set."""
    # Re-import after env vars are set so module-level CLI/MODEL are updated.
    import importlib

    import whisper_api

    importlib.reload(whisper_api)
    return TestClient(whisper_api.APP)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_ok(self, client):
        """GET /health returns 200 when both CLI and model exist."""
        with patch("os.path.exists", return_value=True):
            resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert "cli" in body
        assert "model" in body

    def test_health_missing_cli(self, client):
        """GET /health returns 500 when whisper-cli binary is not found."""

        def fake_exists(path):
            return path != _FAKE_CLI

        with patch("os.path.exists", side_effect=fake_exists):
            resp = client.get("/health")
        assert resp.status_code == 500
        assert any("whisper-cli" in e for e in resp.json()["detail"]["errors"])

    def test_health_missing_model(self, client):
        """GET /health returns 500 when the model file is not found."""

        def fake_exists(path):
            return path != _FAKE_MODEL

        with patch("os.path.exists", side_effect=fake_exists):
            resp = client.get("/health")
        assert resp.status_code == 500
        assert any("model" in e for e in resp.json()["detail"]["errors"])


# ---------------------------------------------------------------------------
# /transcribe
# ---------------------------------------------------------------------------


def _make_wav_bytes() -> bytes:
    """Return a minimal valid PCM WAV (silence, 16 kHz, mono, int16)."""
    import io
    import wave

    import numpy as np

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.zeros(16000, dtype="int16").tobytes())
    return buf.getvalue()


class TestTranscribe:
    def test_transcribe_happy_path(self, client):
        """POST /transcribe returns the cleaned stdout text from whisper-cli."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "  Hello world.\n"
        mock_result.stderr = ""

        with (
            patch("os.path.exists", return_value=True),
            patch("subprocess.run", return_value=mock_result),
            patch("os.remove"),
        ):
            resp = client.post(
                "/transcribe",
                files={"audio": ("test.wav", _make_wav_bytes(), "audio/wav")},
                data={"lang": "en"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["text"] == "Hello world."

    def test_transcribe_subprocess_crash_returns_500(self, client):
        """POST /transcribe returns 500 when whisper-cli exits with non-zero code."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "CUDA error: out of memory"

        with (
            patch("os.path.exists", return_value=True),
            patch("subprocess.run", return_value=mock_result),
            patch("os.remove"),
        ):
            resp = client.post(
                "/transcribe",
                files={"audio": ("test.wav", _make_wav_bytes(), "audio/wav")},
                data={"lang": "en"},
            )

        assert resp.status_code == 500
        assert "whisper-cli exited" in resp.json()["detail"]["error"]

    def test_transcribe_missing_model_returns_500(self, client):
        """POST /transcribe returns 500 immediately when the model file is absent."""

        def fake_exists(path):
            return path != _FAKE_MODEL

        with patch("os.path.exists", side_effect=fake_exists):
            resp = client.post(
                "/transcribe",
                files={"audio": ("test.wav", _make_wav_bytes(), "audio/wav")},
                data={"lang": "en"},
            )

        assert resp.status_code == 500

    def test_transcribe_missing_cli_returns_500(self, client):
        """POST /transcribe returns 500 immediately when whisper-cli is absent."""

        def fake_exists(path):
            return path != _FAKE_CLI

        with patch("os.path.exists", side_effect=fake_exists):
            resp = client.post(
                "/transcribe",
                files={"audio": ("test.wav", _make_wav_bytes(), "audio/wav")},
                data={"lang": "en"},
            )

        assert resp.status_code == 500

    def test_transcribe_no_flash_attn_flag(self, client):
        """POST /transcribe with no_flash_attn=true passes -nfa to whisper-cli."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "ok\n"
        mock_result.stderr = ""

        with (
            patch("os.path.exists", return_value=True),
            patch("subprocess.run", return_value=mock_result) as mock_run,
            patch("os.remove"),
        ):
            client.post(
                "/transcribe",
                files={"audio": ("test.wav", _make_wav_bytes(), "audio/wav")},
                data={"lang": "en", "no_flash_attn": "true"},
            )

        cmd = mock_run.call_args[0][0]
        assert "-nfa" in cmd
