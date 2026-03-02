import asyncio
import contextlib
import json
import os
import subprocess
import tempfile
import threading
from collections.abc import Generator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

APP = FastAPI(
    title="Whisper REST API",
    description="A thin REST wrapper around whisper-cli for audio transcription.",
    version="1.0.0",
)

CLI = os.path.expanduser(
    os.environ.get("WHISPER_CLI", "~/Workplace/whisper.cpp/build/bin/whisper-cli")
)
# Default: multilingual base model.  Override with WHISPER_MODEL env var.
# Use ggml-base.en.bin for English-only (faster, but ignores --lang).
MODEL = os.path.expanduser(
    os.environ.get("WHISPER_MODEL", "~/Workplace/whisper.cpp/models/ggml-base.bin")
)

# Serialise concurrent whisper-cli invocations so multiple requests cannot OOM
# the GPU simultaneously.  Remote clients queued behind the semaphore receive an
# immediate SSE "received" acknowledgement so they are not left in silence.
_WHISPER_SEM = threading.BoundedSemaphore(1)


@APP.get("/health")
async def health() -> dict[str, str | bool]:
    """Basic health check. Also verifies that the CLI and model are reachable."""
    missing = []
    if not os.path.exists(CLI):
        missing.append(f"whisper-cli not found: {CLI}")
    if not os.path.exists(MODEL):
        missing.append(f"model not found: {MODEL}")
    if missing:
        raise HTTPException(status_code=500, detail={"errors": missing})
    return {"ok": True, "cli": CLI, "model": MODEL}


def _build_cmd(
    audio_path: str,
    lang: str,
    device: int,
    no_flash_attn: bool,
) -> list[str]:
    cmd = [CLI, "-m", MODEL, "-f", audio_path, "-l", lang, "-dev", str(device), "-nt"]
    if no_flash_attn:
        cmd.append("-nfa")
    return cmd


def _sse_event(event: str, data: dict[str, str]) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


@APP.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(..., description="WAV audio file to transcribe"),
    lang: str = Form("en", description="Language code, e.g. 'en', 'fr', 'auto'"),
    device: int = Form(0, description="GPU/CPU device index passed to whisper-cli via -dev"),
    no_flash_attn: bool = Form(False, description="Disable flash attention (-nfa flag)"),
) -> dict[str, str]:
    """
    Transcribe an audio file using whisper-cli.

    - **audio**: WAV file (other formats may work if ffmpeg is installed)
    - **lang**: language code (default: en)
    - **device**: compute device index (default: 0)
    - **no_flash_attn**: set to true to disable flash attention
    """
    if not os.path.exists(CLI):
        raise HTTPException(status_code=500, detail=f"whisper-cli not found: {CLI}")
    if not os.path.exists(MODEL):
        raise HTTPException(status_code=500, detail=f"model not found: {MODEL}")

    # Write the uploaded audio to a temp file
    suffix = os.path.splitext(audio.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        tmp = f.name
        f.write(await audio.read())

    cmd = _build_cmd(tmp, lang, device, no_flash_attn)

    def _run() -> subprocess.CompletedProcess[str]:
        with _WHISPER_SEM:
            return subprocess.run(cmd, capture_output=True, text=True)

    try:
        result = await asyncio.to_thread(_run)
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail={"error": "whisper-cli exited with an error", "raw": result.stderr},
            )
        out = result.stdout
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": "failed to run whisper-cli", "raw": str(exc)},
        ) from exc
    finally:
        with contextlib.suppress(OSError):
            os.remove(tmp)

    # whisper-cli emits timestamps + text; strip leading/trailing whitespace
    text = out.strip()
    return {"text": text, "raw": out}


@APP.post("/transcribe/stream")
async def transcribe_stream(
    audio: UploadFile = File(..., description="WAV audio file to transcribe"),
    lang: str = Form("en", description="Language code, e.g. 'en', 'fr', 'auto'"),
    device: int = Form(0, description="GPU/CPU device index passed to whisper-cli via -dev"),
    no_flash_attn: bool = Form(False, description="Disable flash attention (-nfa flag)"),
) -> StreamingResponse:
    """
    Stream transcription chunks via Server-Sent Events (SSE).

    Events:
    - partial: {"text": "<line>"}
    - done: {"text": "<full transcript>"}
    - error: {"error": "<message>"}
    """
    if not os.path.exists(CLI):
        raise HTTPException(status_code=500, detail=f"whisper-cli not found: {CLI}")
    if not os.path.exists(MODEL):
        raise HTTPException(status_code=500, detail=f"model not found: {MODEL}")

    suffix = os.path.splitext(audio.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        tmp = f.name
        f.write(await audio.read())

    cmd = _build_cmd(tmp, lang, device, no_flash_attn)

    def event_stream() -> Generator[str, None, None]:
        process: subprocess.Popen[str] | None = None
        collected: list[str] = []
        try:
            # Acknowledge receipt immediately so the remote client is not left
            # in silence while the semaphore blocks or the model loads.
            yield _sse_event("received", {"status": "queued"})
            with _WHISPER_SEM:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
                if process.stdout is None:
                    yield _sse_event("error", {"error": "failed to capture stdout"})
                    return

                for line in process.stdout:
                    chunk = line.rstrip("\n")
                    if not chunk:
                        continue
                    collected.append(chunk)
                    yield _sse_event("partial", {"text": chunk})

                return_code = process.wait()
                if return_code != 0:
                    yield _sse_event("error", {"error": "whisper-cli exited with an error"})
                    return

                full_text = "\n".join(collected).strip()
                yield _sse_event("done", {"text": full_text})
        except Exception:
            yield _sse_event("error", {"error": "failed to run whisper-cli"})
        finally:
            if process is not None and process.poll() is None:
                with contextlib.suppress(Exception):
                    process.terminate()
                with contextlib.suppress(Exception):
                    process.wait(timeout=1)
            with contextlib.suppress(OSError):
                os.remove(tmp)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
