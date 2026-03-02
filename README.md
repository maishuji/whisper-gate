# whisper-gate

A lightweight local gateway that wraps [`whisper-cli`](https://github.com/ggerganov/whisper.cpp) (from whisper.cpp) and exposes it over HTTP. Comes with two client utilities:

- **`record_and_transcribe.py`** – record a fixed-duration clip from your microphone and print the transcription.
- **`voice_input_daemon.py`** – a push-to-talk daemon that listens for a hotkey, records while the hotkey is held, then types the transcription at your cursor.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Models](#models)
5. [Running the API server](#running-the-api-server)
6. [API Reference](#api-reference)
7. [Client utilities](#client-utilities)
   - [record_and_transcribe.py](#record_and_transcribepy)
   - [voice_input_daemon.py (push-to-talk)](#voice_input_daemonpy-push-to-talk)
8. [Makefile reference](#makefile-reference)
9. [Configuration & environment variables](#configuration--environment-variables)
10. [Troubleshooting](#troubleshooting)

---

## Architecture

```
┌─────────────────────────┐        HTTP POST /transcribe        ┌─────────────────────────┐
│  voice_input_daemon.py  │ ──────────────────────────────────► │    whisper_api.py        │
│  record_and_transcribe  │ ◄────────────────────────────────── │    (FastAPI + uvicorn)   │
└─────────────────────────┘           JSON { text }             └────────────┬────────────┘
                                                                             │ subprocess
                                                                             ▼
                                                                  ┌─────────────────────────┐
                                                                  │  whisper-cli (C++ binary)│
                                                                  │  + ggml model file       │
                                                                  └─────────────────────────┘

┌─────────────────────────┐      HTTP POST /transcribe/stream     ┌─────────────────────────┐
│  voice_input_daemon.py  │ ──────────────────────────────────►   │    whisper_api.py        │
│  (streaming mode)       │ ◄──────────────────────────────────   │    (SSE stream)          │
└─────────────────────────┘    SSE events: partial, done, error    └────────────┬────────────┘
                                                                             │ subprocess
                                                                             ▼
                                                                  ┌─────────────────────────┐
                                                                  │  whisper-cli (C++ binary)│
                                                                  │  + ggml model file       │
                                                                  └─────────────────────────┘
```

The API server is a thin wrapper: it writes the uploaded audio to a temp file, shells out to `whisper-cli`, captures only stdout (clean transcription), and returns it as JSON. All model loading noise goes to stderr and is discarded.

---

## Prerequisites

| Dependency | Notes |
|---|---|
| **Python ≥ 3.10** | Required for `str | None` union syntax |
| **[uv](https://github.com/astral-sh/uv)** | Fast Python package manager. Installed automatically by `make install` / `setup.sh` if missing. |
| **[whisper.cpp](https://github.com/ggerganov/whisper.cpp)** | Must be compiled separately. The `whisper-cli` binary is expected at `~/Workplace/whisper.cpp/build/bin/whisper-cli` (overridable via `WHISPER_CLI`). |
| **A ggml model file** | See [Models](#models) below. |
| **PortAudio** | Required by `sounddevice` for microphone access. Install with `sudo apt install portaudio19-dev` (Debian/Ubuntu) or `sudo pacman -S portaudio` (Arch). |
| **X11 / Wayland + `xdotool`-equivalent** | `pynput` needs a display server to inject keystrokes. Works out of the box on X11; on Wayland you may need `ydotool` or a Wayland-compatible pynput build. |
| **NVIDIA GPU (optional)** | CUDA acceleration is detected automatically by whisper.cpp when compiled with CUDA support. Falls back to CPU otherwise. |

---

## Installation

```bash
# Clone and enter the repo
git clone <repo-url> whisper-gate
cd whisper-gate

# Install uv (if not present) and sync all Python dependencies
make install
# or manually:
bash setup.sh
```

This creates a `.venv/` directory managed by uv and installs all packages listed in `pyproject.toml`.

---

## Models

Models are ggml-format binary files downloaded from Hugging Face. They live in `~/Workplace/whisper.cpp/models/` by default.

| Make target | Model file | Size | Notes |
|---|---|---|---|
| `make download-model` | `ggml-base.bin` | ~142 MB | Multilingual, good balance of speed/accuracy |
| `make download-model-en` | `ggml-base.en.bin` | ~142 MB | English-only, slightly faster than `base` |
| `make download-model-large` | `ggml-large-v3.bin` | ~3.1 GB | Best accuracy, requires ~6 GB VRAM for GPU inference |

```bash
# Download the default base model
make download-model

# Download the large-v3 model (best quality)
make download-model-large
```

To use a different model, set the `WHISPER_MODEL` variable:

```bash
make run WHISPER_MODEL=~/Workplace/whisper.cpp/models/ggml-large-v3.bin
```

---

## Running the API server

```bash
# Start the server (kills any existing process on the port first)
make run

# Start with auto-reload (development)
make dev

# Or using the shell script directly
bash run.sh
```

The server listens on `0.0.0.0:8178` by default. Swagger UI is available at [http://localhost:8178/docs](http://localhost:8178/docs).

---

## API Reference

### `GET /health`

Verifies the server is running and that both `whisper-cli` and the model file are accessible.

**Response (200 OK):**
```json
{
  "ok": true,
  "cli": "/home/user/Workplace/whisper.cpp/build/bin/whisper-cli",
  "model": "/home/user/Workplace/whisper.cpp/models/ggml-base.bin"
}
```

**Response (500):**
```json
{
  "detail": {
    "errors": ["whisper-cli not found: /path/to/whisper-cli"]
  }
}
```

---

### `POST /transcribe`

Transcribe an audio file.

**Request** – `multipart/form-data`:

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `audio` | file | ✅ | — | WAV audio file (16 kHz mono recommended) |
| `lang` | string | | `en` | Language code: `en`, `fr`, `de`, `ja`, `auto`, etc. |
| `device` | integer | | `0` | GPU/CPU device index passed to `whisper-cli` via `-dev` |
| `no_flash_attn` | boolean | | `false` | Disable flash attention (`-nfa` flag) |

**Response (200 OK):**
```json
{
  "text": "Hello, world.",
  "raw": "Hello, world.\n"
}
```

**Response (500):**
```json
{
  "detail": {
    "error": "whisper-cli exited with an error",
    "raw": "<stderr output>"
  }
}
```

**curl example:**
```bash
curl -s -X POST http://localhost:8178/transcribe \
  -F "audio=@recording.wav" \
  -F "lang=en" \
  | python3 -m json.tool
```

---

### `POST /transcribe/stream`

Stream partial transcription text via Server-Sent Events (SSE). This is useful for long clips: the client can show partial lines as soon as the model emits them, then a final `done` event with the full transcript.

**Request** – `multipart/form-data`:

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `audio` | file | ✅ | — | WAV audio file (16 kHz mono recommended) |
| `lang` | string | | `en` | Language code: `en`, `fr`, `de`, `ja`, `auto`, etc. |
| `device` | integer | | `0` | GPU/CPU device index passed to `whisper-cli` via `-dev` |
| `no_flash_attn` | boolean | | `false` | Disable flash attention (`-nfa` flag) |

**SSE events:**

- `partial` → `{"text": "<line>"}`
- `done` → `{"text": "<full transcript>"}`
- `error` → `{"error": "<message>"}`

**curl example:**

```bash
curl -N -X POST http://localhost:8178/transcribe/stream \
  -F "audio=@recording.wav" \
  -F "lang=en"
```

---

## Client utilities

### `record_and_transcribe.py`

Records a fixed-duration audio clip from the microphone, sends it to the API, and prints the transcription.

```
usage: record_and_transcribe.py [-h] [--url URL] [--duration DURATION]
                                 [--lang LANG] [--device DEVICE]
                                 [--list-devices]
```

| Argument | Default | Description |
|---|---|---|
| `--url` | `http://localhost:8178` | API base URL |
| `--duration` | `5` | Recording length in seconds |
| `--lang` | `en` | Language code |
| `--device` | system default | Microphone device index |
| `--list-devices` | — | Print available audio devices and exit |

**Examples:**

```bash
# List available microphone devices
uv run record_and_transcribe.py --list-devices

# Record 10 seconds in French
uv run record_and_transcribe.py --duration 10 --lang fr

# Use a specific microphone by index
uv run record_and_transcribe.py --device 2 --duration 5
```

---

### `voice_input_daemon.py` (push-to-talk)

A background daemon that watches for a configurable hotkey. While the hotkey is **held**, audio is recorded from the default microphone. On **release**, the audio is sent to the API and the transcription is **typed at the current cursor position** using `pynput`.

The daemon uses the streaming endpoint by default, so you will see partial lines appear in the terminal while the server processes the audio. Final typing still happens after the `done` event.

```
usage: voice_input_daemon.py [-h] [--url URL] [--lang LANG] [--key KEY]
```

| Argument | Default | Description |
|---|---|---|
| `--url` | `http://localhost:8178` | API base URL |
| `--lang` | `en` | Language code |
| `--key` | `ctrl+alt+space` | Hotkey combo (see below) |

**Hotkey syntax:**

Combine modifier names and key names with `+`. Modifier aliases: `ctrl`, `alt`, `shift`. Key names follow [pynput's `Key` enum](https://pynput.readthedocs.io/en/latest/keyboard.html#pynput.keyboard.Key).

```
ctrl+alt+space      # default
ctrl+shift+r
alt+f9
```

Left and right variants (`ctrl_l`/`ctrl_r`, `alt_l`/`alt_r`, `shift_l`/`shift_r`) are normalised to their generic forms, so either side of the keyboard triggers the hotkey.

**Examples:**

```bash
# Default hotkey (Ctrl+Alt+Space)
uv run voice_input_daemon.py

# Custom hotkey, French language
uv run voice_input_daemon.py --key "ctrl+shift+f" --lang fr

# Point at a remote server
uv run voice_input_daemon.py --url http://192.168.1.10:8178
```

**Workflow:**

1. Start the API server: `make run`
2. Start the daemon in another terminal: `make ptt`
3. Focus any text input (terminal, browser, editor, etc.)
4. **Hold** the hotkey → 🔴 Recording
5. **Speak**
6. **Release** hotkey → transcription is typed at the cursor

Recordings shorter than 0.5 seconds are silently dropped to avoid spurious API calls.

---

## Makefile reference

| Target | Description |
|---|---|
| `make install` | Install uv (if missing) and sync all Python dependencies |
| `make run` | Kill any process on `PORT`, then start the API server |
| `make dev` | Start the API server with `--reload` (auto-restart on file changes) |
| `make health` | Hit `GET /health` and pretty-print the response |
| `make test AUDIO=file.wav` | Transcribe a specific WAV file via the API |
| `make record` | Record `DURATION` seconds and transcribe |
| `make list-devices` | List available audio input devices |
| `make ptt` | Start the push-to-talk daemon |
| `make download-model` | Download `ggml-base.bin` |
| `make download-model-en` | Download `ggml-base.en.bin` |
| `make download-model-large` | Download `ggml-large-v3.bin` (~3.1 GB) |
| `make tailscale-install` | Install Tailscale on Linux |
| `make tailscale-install-windows` | Install Tailscale on Windows via winget |
| `make tailscale-ip` | Print this machine's Tailscale IPv4 |
| `make tailscale-health` | Hit `/health` via this machine's Tailscale IP |
| `make clean` | Delete `.venv/`, `__pycache__/`, and `.pyc` files |
| `make help` | Show all targets and configurable variables |

**Overridable variables:**

```bash
make run  PORT=9000
make run  WHISPER_MODEL=~/Workplace/whisper.cpp/models/ggml-large-v3.bin
make ptt  PTT_KEY="ctrl+shift+r"  WHISPER_LANG=fr
make record  DURATION=10  WHISPER_LANG=de
make record  URL=http://100.110.122.117:8178
make ptt  URL=http://100.110.122.117:8178  WHISPER_LANG=en
```

---

## Configuration & environment variables

| Variable | Default | Description |
|---|---|---|
| `WHISPER_CLI` | `~/Workplace/whisper.cpp/build/bin/whisper-cli` | Path to the compiled `whisper-cli` binary |
| `WHISPER_MODEL` | `~/Workplace/whisper.cpp/models/ggml-base.bin` | Path to the ggml model file |
| `WHISPER_HOST` | `0.0.0.0` | Bind address (used by `run.sh`) |
| `WHISPER_PORT` | `8178` | Bind port (used by `run.sh`) |

These can be set in your shell environment or prefixed inline:

```bash
WHISPER_MODEL=~/Workplace/whisper.cpp/models/ggml-large-v3.bin make run
```

---

## Troubleshooting

### `address already in use` on `make run`

The `run` target calls `fuser -k <PORT>/tcp` before starting, which kills any existing process on the port. If you see this error it means `fuser` is not installed — install it with `sudo apt install psmisc`.

### Transcription types garbage / escape sequences

This was caused by whisper.cpp writing its model-loading and timing logs to stdout. This is fixed: the API now captures stdout and stderr separately and only returns stdout (the clean transcription text).

### `pynput` fails to inject keystrokes on Wayland

`pynput` relies on X11 APIs. Under Wayland, set `GDK_BACKEND=x11` or run inside an XWayland session. Alternatively, replace `type_text()` in `voice_input_daemon.py` with a `ydotool type` subprocess call.

### Push-to-talk hotkey doesn't fire

- Make sure the daemon has permission to read global keyboard events. On some distros this requires adding your user to the `input` group: `sudo usermod -aG input $USER` (then log out and back in).
- Double-check the key names: run `python3 -c "from pynput.keyboard import Key; print(list(Key))"` to list all valid special key names.

### GPU not used / CUDA errors

- Confirm whisper.cpp was compiled with CUDA: the build output should mention `CUDA : ARCHS = <arch>`.
- Check available VRAM — `ggml-large-v3.bin` requires ~3.1 GB of VRAM at fp16.
- Disable flash attention if you see numerical errors: pass `no_flash_attn=true` to `/transcribe` or add `-nfa` to the CLI manually.

### Model not found

Ensure the model was downloaded to the expected path:

```bash
ls ~/Workplace/whisper.cpp/models/
make download-model        # base multilingual
make download-model-large  # large-v3
```
