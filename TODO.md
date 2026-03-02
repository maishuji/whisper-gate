# TODO – whisper-rest improvements

Items are grouped by area and roughly ordered by priority within each section.

---

## API (`whisper_api.py`)

- [ ] **Persistent model loading** – currently `whisper-cli` is spawned fresh for every request, reloading the model each time (~2–3 s on large-v3). Explore keeping a long-running `whisper-cli` process or switching to the `whispercpp` Python binding to hold the model in memory between calls.
- [x] **Request queue / concurrency limit** – add a semaphore so concurrent transcription calls are serialised rather than spawning multiple whisper-cli processes at once and OOM-ing the GPU.
- [x] **Streaming response** – stream partial transcription tokens back via Server-Sent Events so the caller can display results as they appear.
- [ ] **Audio validation** – reject uploads that are not WAV, or not 16 kHz / mono, with a clear 422 error before invoking whisper-cli.
- [ ] **File size limit** – enforce a max upload size (e.g. 25 MB) to prevent accidental large uploads from hanging the server.
- [ ] **Authentication** – add an optional API key header (`X-API-Key`) so the server can be safely exposed over a network.
- [ ] **`/transcribe` returns `raw` field** – `raw` currently duplicates `text`; either drop it or make it genuinely useful (e.g. include timestamps when `-nt` is not passed).
- [ ] **Configurable beam/processor count** – expose `--beam-size` and `--processors` as optional form fields.

---

## Push-to-talk daemon (`voice_input_daemon.py`)

- [ ] **Wayland support** – replace `pynput` keyboard injection with `ydotool` subprocess call when `$WAYLAND_DISPLAY` is set, so the daemon works without XWayland.
- [ ] **Audio device selection** – add a `--device` CLI flag (same as `record_and_transcribe.py`) so the user can pick a specific microphone.
- [ ] **Visual/audio feedback** – play a short beep or show a system notification (via `notify-send`) on recording start/stop, so it's obvious when the mic is live.
- [ ] **Clipboard mode** – add a `--mode clipboard` flag that copies the transcription to the clipboard instead of typing it, useful in terminals where `pynput` injection is unreliable.
- [ ] **Configurable minimum duration** – expose `MIN_DURATION` as a `--min-duration` CLI argument.
- [ ] **Graceful reconnect** – if the API is unreachable, retry with exponential back-off instead of printing an error and dropping the recording.
- [ ] **systemd unit file** – provide a `voice_input_daemon.service` template so the daemon can be managed as a user service (`systemctl --user start voice_input_daemon`).

---

## CLI recorder (`record_and_transcribe.py`)

- [ ] **Output modes** – add `--output {text,json,clipboard}` so the result can be piped, formatted, or pasted directly.
- [ ] **VAD (voice activity detection)** – auto-stop recording when silence is detected rather than requiring a fixed duration.

---

## Infrastructure / tooling

- [x] **`ruff` linting & formatting** – add `ruff` as a dev dependency and a `make lint` / `make fmt` target; enforce via pre-commit hook.
- [ ] **`mypy` type checking** – add `mypy` strict mode and a `make typecheck` target.
- [ ] **Pre-commit hooks** – add a `.pre-commit-config.yaml` running `ruff`, `mypy`, and conventional-commit message validation (`commitlint` or `pre-commit-msg-linter`).
- [ ] **Tests** – add `pytest` with at minimum: API unit tests (mock subprocess), `parse_hotkey` tests, `to_wav_bytes` roundtrip test.
- [ ] **CI (GitHub Actions)** – add a workflow that runs lint + tests on every push/PR.
- [ ] **`make URL=` variable** – allow overriding the API URL from the Makefile for remote-server workflows (`make ptt URL=http://192.168.1.50:8178`).
- [ ] **`pyproject.toml` dev dependencies** – move `ruff`, `mypy`, `pytest` etc. into `[dependency-groups] dev` so they are not installed in production.

---

## Documentation

- [ ] **Remote GPU setup section in README** – document the SSH-tunnel and direct-network workflows with copy-paste commands.
- [ ] **`CONTRIBUTING.md`** – branch naming, PR checklist, and how to run tests locally.
- [ ] **Changelog** – start a `CHANGELOG.md` following [Keep a Changelog](https://keepachangelog.com/) format.
