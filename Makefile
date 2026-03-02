.PHONY: install run dev health test test-unit record list-devices ptt download-model download-model-en download-model-large lint fmt clean help tailscale-install tailscale-install-windows tailscale-ip tailscale-health

HOST          ?= 0.0.0.0
PORT          ?= 8178
URL           ?= http://localhost:$(PORT)
DURATION      ?= 5
WHISPER_LANG  ?= en
WHISPER_MODEL ?= $(HOME)/Workplace/whisper.cpp/models/ggml-base.bin
MODELS_DIR    ?= $(HOME)/Workplace/whisper.cpp/models
PTT_KEY       ?= ctrl+alt+space

## Install uv (if missing) and sync all dependencies
install:
	@if ! command -v uv >/dev/null 2>&1; then \
		echo ">>> Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		export PATH="$$HOME/.local/bin:$$PATH"; \
	fi
	uv sync

## Start the API server
run:
	@fuser -k $(PORT)/tcp 2>/dev/null || true
	WHISPER_MODEL=$(WHISPER_MODEL) uv run uvicorn whisper_api:APP --host $(HOST) --port $(PORT)

## Start with auto-reload (development)
dev:
	WHISPER_MODEL=$(WHISPER_MODEL) uv run uvicorn whisper_api:APP --host $(HOST) --port $(PORT) --reload

## Check the /health endpoint
health:
	curl -s http://localhost:$(PORT)/health | python3 -m json.tool

## Quick transcription smoke-test (set AUDIO=path/to/file.wav)
test:
	@if [ -z "$(AUDIO)" ]; then \
		echo "Usage: make test AUDIO=path/to/file.wav"; exit 1; \
	fi
	curl -s -X POST http://localhost:$(PORT)/transcribe \
		-F "audio=@$(AUDIO)" \
		-F "lang=en" | python3 -m json.tool

## Record from microphone and transcribe (DURATION=seconds WHISPER_LANG=code URL=http://...)
record:
	uv run record_and_transcribe.py --url $(URL) --duration $(DURATION) --lang $(WHISPER_LANG)

## List available audio input devices
list-devices:
	uv run record_and_transcribe.py --list-devices

## Start push-to-talk daemon (hold PTT_KEY to record, release to type)
ptt:
	uv run voice_input_daemon.py --url $(URL) --lang $(WHISPER_LANG) --key "$(PTT_KEY)"

## Download the multilingual base model (required for non-English languages)
download-model:
	$(HOME)/Workplace/whisper.cpp/models/download-ggml-model.sh base

## Download the English-only base model (faster, English only)
download-model-en:
	$(HOME)/Workplace/whisper.cpp/models/download-ggml-model.sh base.en

## Download large-v3 multilingual model (~3.1 GB, best quality, needs ~6 GB VRAM)
download-model-large:
	$(HOME)/Workplace/whisper.cpp/models/download-ggml-model.sh large-v3

## Run unit tests
test-unit:
	uv run pytest tests/ -v

## Lint all Python files with ruff
lint:
	uv run ruff check .

## Format all Python files with ruff
fmt:
	uv run ruff format .

## Install Tailscale (Linux only)
tailscale-install:
	curl -fsSL https://tailscale.com/install.sh | sh
	@echo ">>> Run: sudo tailscale up"
	@echo ">>> Then run: make tailscale-ip"

## Install Tailscale (Windows, via winget)
tailscale-install-windows:
	winget install --id Tailscale.Tailscale --exact --source winget
	@echo ">>> Open an elevated PowerShell and run: tailscale up"
	@echo ">>> Then run: make tailscale-ip"

## Show this machine's Tailscale IP
tailscale-ip:
	@tailscale ip -4 2>/dev/null || echo "Tailscale is not running – run: sudo tailscale up"

## Check /health via Tailscale IP (use from another machine: make tailscale-health)
tailscale-health:
	$(eval TS_IP := $(shell tailscale ip -4 2>/dev/null))
	@if [ -z "$(TS_IP)" ]; then echo "Tailscale is not running – run: sudo tailscale up"; exit 1; fi
	@echo ">>> Checking health at http://$(TS_IP):$(PORT)/health"
	curl -s http://$(TS_IP):$(PORT)/health | python3 -m json.tool

## Remove uv-managed venv and build artifacts
clean:
	rm -rf .venv __pycache__ *.pyc

## Show available targets
help:
	@grep -E '^##' Makefile | sed 's/## /  /'
	@echo ""
	@echo "Variables (override with make <target> VAR=value):"
	@echo "  HOST  (default: $(HOST))"
	@echo "  PORT         (default: $(PORT))"
	@echo "  URL          (default: $(URL))"
	@echo "  DURATION     (default: $(DURATION))"
	@echo "  WHISPER_LANG (default: $(WHISPER_LANG))"
	@echo "  AUDIO        (path to .wav for 'make test')"
	@echo ""
	@echo "Tailscale targets (cross-network access):"
	@echo "  tailscale-install           install Tailscale on Linux"
	@echo "  tailscale-install-windows   install Tailscale on Windows (winget)"
	@echo "  tailscale-ip        print this machine's Tailscale IP"
	@echo "  tailscale-health    hit /health via Tailscale IP"
