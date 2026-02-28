import contextlib
import os
import subprocess
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

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


@APP.get("/health")
def health():
    """Basic health check. Also verifies that the CLI and model are reachable."""
    missing = []
    if not os.path.exists(CLI):
        missing.append(f"whisper-cli not found: {CLI}")
    if not os.path.exists(MODEL):
        missing.append(f"model not found: {MODEL}")
    if missing:
        raise HTTPException(status_code=500, detail={"errors": missing})
    return {"ok": True, "cli": CLI, "model": MODEL}


@APP.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(..., description="WAV audio file to transcribe"),
    lang: str = Form("en", description="Language code, e.g. 'en', 'fr', 'auto'"),
    device: int = Form(0, description="GPU/CPU device index passed to whisper-cli via -dev"),
    no_flash_attn: bool = Form(False, description="Disable flash attention (-nfa flag)"),
):
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

    cmd = [CLI, "-m", MODEL, "-f", tmp, "-l", lang, "-dev", str(device), "-nt"]
    if no_flash_attn:
        cmd.append("-nfa")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
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
