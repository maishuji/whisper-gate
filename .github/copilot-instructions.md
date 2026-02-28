# Copilot Instructions – whisper-rest

## Project overview

This is a local speech-to-text tool composed of:

- **`whisper_api.py`** – a FastAPI server that wraps `whisper-cli` (from whisper.cpp) and exposes a REST API.
- **`voice_input_daemon.py`** – a push-to-talk daemon that records audio on hotkey hold and types the transcription at the cursor.
- **`record_and_transcribe.py`** – a CLI tool that records a fixed-duration clip and prints the transcription.

Runtime is managed by **uv**. Never suggest `pip install`; always use `uv add` or `uv sync`.

---

## Commit messages

All commit messages **must** follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.

### Format

```
<type>(<scope>): <short summary>

[optional body]

[optional footer(s)]
```

### Types

| Type | When to use |
|---|---|
| `feat` | A new feature or capability |
| `fix` | A bug fix |
| `docs` | Documentation changes only |
| `style` | Formatting, whitespace — no logic change |
| `refactor` | Code restructuring with no behaviour change |
| `perf` | Performance improvements |
| `test` | Adding or updating tests |
| `chore` | Tooling, dependencies, build scripts, CI |
| `revert` | Reverting a previous commit |

### Scopes (optional but encouraged)

| Scope | Covers |
|---|---|
| `api` | `whisper_api.py` |
| `daemon` | `voice_input_daemon.py` |
| `record` | `record_and_transcribe.py` |
| `makefile` | `Makefile` |
| `deps` | `pyproject.toml`, `uv.lock` |
| `docs` | `README.md`, `copilot-instructions.md` |

### Examples

```
feat(daemon): add configurable minimum recording duration
fix(api): separate stderr from stdout to avoid leaking whisper logs
docs: add extensive README with API reference and troubleshooting
chore(deps): add pynput and sounddevice to pyproject.toml
refactor(api): extract temp file cleanup into a context manager
perf(daemon): skip transcription for recordings under 0.5s
```

### Rules

- Use the **imperative mood** in the summary: *"add"* not *"added"* or *"adds"*.
- Keep the summary line under **72 characters**.
- Do **not** end the summary with a period.
- Add a blank line between the summary and the body.
- Reference issues or PRs in the footer: `Closes #12`, `Fixes #34`.
- Mark breaking changes with `!` after the type/scope and a `BREAKING CHANGE:` footer:
  ```
  feat(api)!: rename /transcribe response field raw → stdout

  BREAKING CHANGE: clients reading the `raw` field must update to `stdout`.
  ```

---

## Code style

- **Python ≥ 3.10**. Use modern union syntax (`X | Y`) and built-in generics (`list[str]`, `dict[str, int]`).
- Follow **PEP 8**. Max line length: **100 characters**.
- Use **type annotations** on all function signatures.
- Prefer `subprocess.run(...)` over `subprocess.check_output(...)`. Always capture stdout and stderr separately (`stdout=subprocess.PIPE, stderr=subprocess.PIPE`).
- Keep FastAPI route functions `async`. Use `Form(...)` and `File(...)` for multipart fields.
- Do not print or log whisper-cli stderr output to the user — it is internal noise.

## Audio conventions

- Always record/send audio at **16 kHz, mono, int16** — that is what whisper.cpp expects.
- Encode microphone frames as a standard PCM WAV before posting to `/transcribe`.

## Dependency management

- All runtime dependencies go in `pyproject.toml` under `[project] dependencies`.
- Use `uv add <package>` to add a new dependency; commit both `pyproject.toml` and `uv.lock`.
- Do not vendor or bundle whisper.cpp — it is expected to be compiled separately at `~/Workplace/whisper.cpp`.
