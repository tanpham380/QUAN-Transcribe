# QUAN-Transcribe — agent guide

Vietnamese speech recognition: an inference library plus an OpenAI-compatible
serving layer. This file is the shared entry point for coding agents (VS Code
Copilot, Google Antigravity, Codex, and any other `AGENTS.md`-aware tool).

## What this repository is

Two packages that install together from one `pyproject.toml`:

| Package | Role | Entry point |
|---|---|---|
| `quan_transcribe/` | Inference library: offline, streaming and live-microphone decoding, optional KenLM beam search | `quan-transcribe` |
| `app/` | Serving layer: OpenAI-compatible HTTP / SSE / WebSocket API, browser playground, gRPC | `quan-asr-server` |

The model is **CTC, not autoregressive**: one forward pass per audio piece, no KV
cache, no incremental state across calls. Anything that assumes token-by-token
decoding is wrong for this codebase.

## Non-negotiable invariants

These are load-bearing. Breaking one is a bug even if tests pass.

1. **Exactly one thread touches the model.** All model access goes through
   `app/core.py` (`EngineCore`), a dedicated model thread with a priority
   scheduler. Never call the model directly from a request thread, and never
   reintroduce a lock held across a whole decode — that is the head-of-line
   blocking `core.py` exists to remove.
2. **Backpressure is a feature.** When the bulk queue is full, `submit_bulk`
   raises `EngineBusy` and the HTTP layer returns `503` with `Retry-After`.
   Do not "fix" this by making the queue unbounded.
3. **VAD runs on request threads, never the model thread.** `app/vad.py` is
   lazy-loaded and optional; a missing `silero-vad` must degrade to energy-based
   splitting, not fail the request.
4. **`silero_vad` import mutates global torch state.** It calls
   `torch.set_num_threads(1)` at import. `app/vad.py` captures and restores the
   thread count around the import — keep that.
5. **`app/audio.py` split logic mirrors `quan_transcribe/audio.py`.** The
   `split_with_offsets` cut points must stay identical to `split_quiet` so
   reported segment timestamps line up with what the model actually decoded.
   Change one, change the other.
6. **The tests never import `torch` or `quan_transcribe`.** `tests/test_app.py`
   monkeypatches a scripted fake engine into `app.server`. Keep it that way so
   the suite runs on a plain Python image with no GPU and no model download.

## Layout

```
quan_transcribe/          inference library (upstream; see lint note below)
  model.py                QuanCTC, ModelConfig, greedy_decode, load_encoder
  features.py             batched log-mel, numerically matching the HF extractor
  audio.py                load_audio, to_16k_mono, split_quiet
  transcriber.py          QuanTranscribe.from_pretrained, transcribe, _forward
  lm.py                   KenLM CTC prefix beam search (shallow fusion)
  live.py                 LiveSession — incremental microphone decoding
  cli.py                  quan-transcribe CLI
app/                      serving layer
  core.py                 EngineCore — the one model-owning scheduler thread
  engine.py               TranscriptionEngine — load once, serve many
  vad.py                  Silero VAD wrapper (lazy, optional)
  audio.py                decode_audio_bytes, split_with_offsets, Segment
  postprocess.py          Vietnamese punctuation / capitalization / ITN
  formats.py              srt, vtt, verbose_json rendering
  schemas.py              OpenAI-compatible form parsing + APIError envelope
  config.py               Settings.from_env() — env-driven, flags win
  server.py               FastAPI app, routes, error handlers
  grpc_server.py          Google Speech v2 compatible gRPC service
  ui.py                   single-file Material 3 playground
  proto/                  generated protobuf — never hand-edit
tests/test_app.py         the whole suite; no torch, no model
scripts/                  dev_server.py, smoke.sh, probe_audio.py, repro_local.py
deploy/                   GPUStack backend manifests
```

## Commands

```bash
pytest -q                          # full suite, no GPU or model needed
pytest -q tests/test_app.py -k vad # focused run
ruff check app tests scripts
ruff format app tests scripts
```

`ruff` **excludes `quan_transcribe/`** on purpose: it is the upstream inference
library and is not held to the server's lint rules. Do not add it to the lint
targets, and do not reformat it to satisfy `ruff`.

Line length is 88, target is `py310`. `E501` is ignored (the formatter owns
wrapping).

## Conventions

- **Style.** `from __future__ import annotations` at the top of every module.
  Modern typing (`list[str]`, `X | None`), not `List`/`Optional`.
- **Docstrings explain *why*, not *what*.** The existing modules open with a
  rationale section (`app/core.py`, `app/vad.py`, `app/audio.py` are the model
  to follow). A docstring that restates the signature adds nothing.
- **Comment only what needs clarifying.** Do not narrate obvious code.
- **Errors carry meaning.** `APIError` renders the OpenAI error envelope;
  `AudioDecodeError` means "not decodable audio"; `EngineBusy` means "retry".
  Pick the existing type rather than raising bare `ValueError`.
- **Config is env-driven.** Add a field to `Settings` in `app/config.py` and a
  matching flag in `app/cli.py`; real environment variables win over flag
  defaults.
- **Optional dependencies stay optional.** `kenlm`, `silero-vad`/`onnxruntime`
  and `grpcio` are extras. Import them lazily inside the function that needs
  them and degrade gracefully when absent.

## Gotchas

- `app/proto/*_pb2*.py` is generated by `protoc`. Regenerate, never hand-edit.
- `app/server.py` has `B008` ignored in `pyproject.toml` because FastAPI's
  dependency injection is built on calls in argument defaults.
- `app/ui.py` is one large HTML string. It is not a template — edit the string.
- Two checkpoint layouts are supported (`safetensors` and a single `*.pt`).
  `head_cosine` in the config selects a cosine output head instead of a dot
  product. Both paths share everything else.
- `samples/*.mp3` are real recordings used by `scripts/probe_audio.py` and
  `scripts/repro_local.py` to answer "is this badly transcribed, or just mostly
  silence?".

## Where to look first

| Task | Start here |
|---|---|
| Latency, throughput, batching, 503s | `app/core.py` |
| Model loading, warmup, readiness | `app/engine.py` |
| Silence handling, cut points, timestamps | `app/vad.py`, `app/audio.py` |
| Transcript text quality (punctuation, numbers) | `app/postprocess.py` |
| API shape, error envelopes, SSE framing | `app/server.py`, `app/schemas.py` |
| Decoding accuracy, beam search | `quan_transcribe/lm.py`, `quan_transcribe/transcriber.py` |
| Streaming / microphone | `quan_transcribe/live.py` |
