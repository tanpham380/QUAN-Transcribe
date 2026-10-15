# QUAN-ASR

An OpenAI-compatible speech-to-text server for
[`qnguyen3/QUAN-Transcribe-1.0`](https://huggingface.co/qnguyen3/QUAN-Transcribe-1.0),
packaged as a container that runs on [GPUStack](https://gpustack.ai) as a custom
backend.

Drop-in replacement for the OpenAI transcription endpoint: point any existing
client at this server and it keeps working. On top of that it adds real
timestamps, Server-Sent Events streaming, a WebSocket for live microphone audio,
optional KenLM beam-search decoding, and a small browser playground.

The model is Vietnamese-only and single-pass CTC, so it is fast — roughly
3,700× realtime on an H100 — and its output is lowercase Vietnamese without
punctuation, with numbers written as words.

## Contents

- [Quick start](#quick-start)
- [API](#api)
- [Streaming](#streaming)
- [Optional n-gram language model](#optional-n-gram-language-model)
- [Configuration](#configuration)
- [Performance](#performance)
- [Docker](#docker)
- [Deploying on GPUStack](#deploying-on-gpustack)
- [GitLab CI](#gitlab-ci)
- [Development](#development)

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install "quan-transcribe @ git+https://github.com/qnguyen3/QUAN-Transcribe"
pip install .
quan-asr --serve --port 8000
```

Then, from another shell:

```bash
curl -s -F "file=@sample.wav" http://127.0.0.1:8000/v1/audio/transcriptions
# {"text": "xin chào tôi tên là nam"}
```

Open <http://127.0.0.1:8000/> for the playground: upload a file to watch the
streamed transcript, or hold the record button to transcribe from the mic.

### Requirements

- **PyTorch matching your CUDA driver.** The Dockerfile installs the cu126
  build; for a manual install pick the right index, e.g.
  `pip install torch --index-url https://download.pytorch.org/whl/cu124`.
- **An `HF_TOKEN` with access to [`google/embeddinggemma-2`].** The audio
  encoder is a gated Gemma model, so you must accept its terms on Hugging Face
  and then `hf auth login` (or export `HF_TOKEN`) before the first run.
- **ffmpeg** if you want to transcribe containers libsndfile cannot read
  (m4a, webm/opus). WAV, FLAC, OGG and MP3 work without it.

[`google/embeddinggemma-2`]: https://huggingface.co/google/embeddinggemma-2

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/audio/transcriptions` | Transcribe an uploaded file (OpenAI-compatible) |
| `WS` | `/v1/audio/transcriptions/stream` | Live transcription of raw PCM |
| `GET` | `/v1/models` | List the served model |
| `GET` | `/v1/models/{id}` | Fetch one model |
| `GET` | `/health` | Liveness plus real model state |
| `GET` | `/metrics` | Request counters and engine info |
| `GET` | `/` | Browser playground |
| `GET` | `/docs` | Interactive OpenAPI documentation |

### `POST /v1/audio/transcriptions`

`multipart/form-data`. All documented OpenAI parameters are accepted:

| Parameter | Default | Notes |
| --- | --- | --- |
| `file` | *required* | Audio to transcribe |
| `model` | served name | Validated against `/v1/models` |
| `language` | `vi` | `vi`/`vietnamese` only; anything else is `unsupported_language` |
| `response_format` | `json` | `json`, `text`, `srt`, `verbose_json`, `vtt` |
| `prompt` | – | Accepted for compatibility; a CTC model has no prompt channel |
| `temperature` | `0.0` | Accepted for compatibility; decoding is deterministic |

Extensions this server adds:

| Parameter | Default | Notes |
| --- | --- | --- |
| `stream` | `false` | Return Server-Sent Events instead of a buffered body |
| `stream_partials` | `true` | Emit in-progress text while streaming |
| `use_lm` | server setting | Force beam search on/off for this request |
| `beam_width` | `32` | Beams kept by the beam search |
| `left_chunks` | server setting | Past chunks the reader may attend to |
| `chunk_seconds` | `30` | Cut long audio into pieces of at most this many seconds |

```bash
# Plain text
curl -s -F "file=@sample.wav" -F "response_format=text" \
  http://127.0.0.1:8000/v1/audio/transcriptions

# Subtitles
curl -s -F "file=@sample.wav" -F "response_format=srt" \
  http://127.0.0.1:8000/v1/audio/transcriptions
```

`verbose_json` returns OpenAI-shaped segments with `start`/`end` in seconds,
derived from the quiet points where the audio was actually cut, so the
timestamps line up with what the model decoded. Segments that were split
mid-sentence are merged back together when there is no terminal punctuation and
the gap is under 0.5 s.

## Streaming

### Server-Sent Events

Set `stream=true` and the response becomes a stream of events:

```
data: {"type": "transcript.text.delta", "text": "xin", "start": 0.0, "end": 0.48}
data: {"type": "transcript.text.segment", "text": "xin chào", "start": 0.0, "end": 1.0}
data: {"type": "transcript.text.completed", "text": "", "start": 0.0, "end": 2.0}
data: {"type": "transcript.text.done", "text": "xin chào tôi tên là nam", "duration": 2.0}
```

- `transcript.text.delta` — unstable text, expected to change (`stream_partials=false` removes these)
- `transcript.text.segment` — a finalized segment with its timestamps
- `transcript.text.completed` — end-of-audio marker
- `transcript.text.done` — the final transcript plus `duration`, `model`, `language`, `elapsed`

Disconnecting the client stops the decode thread at the next chunk boundary.

### WebSocket

`/v1/audio/transcriptions/stream` accepts raw audio and pushes JSON events back.
Send a JSON handshake first, then binary frames:

```json
{"type": "config", "sample_rate": 16000, "encoding": "pcm16", "left_chunks": null}
```

`encoding` is `pcm16` or `f32`, `sample_rate` is the rate you are sending (the
server resamples to 16 kHz), and `left_chunks` overrides the server default for
this session. Send `{"type": "flush"}` to force the current utterance out, and
`{"type": "stop"}` to finish. Audio is segmented on ~0.8 s of silence
(`ENDPOINT_SECONDS`), and an utterance is force-cut at a quiet point after
`MAX_UTTERANCE_SECONDS`.

### How streaming stays honest

The encoder attention is chunked with left context only, the reader's
convolution is causal, and the reader attends to the current 0.48 s chunk plus
earlier ones. Nothing ever looks more than one chunk ahead, so the partial
output is the same text the model would produce if the audio simply stopped
there. `left_chunks` caps how far back the reader looks: `None` (the default)
uses all history for best accuracy, `0` gives the lowest latency.

## Optional n-gram language model

By default the server decodes greedily. Turning on the language model swaps in
CTC prefix beam search against a 4-gram KenLM, which improves accuracy on
Vietnamese text:

| | WER (FLEURS valid) |
| --- | --- |
| Greedy | 6.71% |
| + KenLM, beam 32 | **4.79%** |

```bash
# Bundled LM: the model repo's lm/ folder (vi_4gram.bin, ~91 MB) downloads on
# first use.
USE_LM=true quan-asr --serve

# Your own KenLM model: a .bin/.arpa file, or a directory containing
# vi_4gram.bin and optionally unigrams.txt / lm_config.json.
LM_PATH=/models/vi_4gram.bin quan-asr --serve

# Tune the search.
USE_LM=true BEAM_WIDTH=16 LM_ALPHA=0.5 LM_BETA=1.5 quan-asr --serve
```

Or with flags: `--lm`, `--lm-path`, `--beam-width`, `--lm-alpha`, `--lm-beta`.

- **`LM_ALPHA`** (LM weight, default **0.3**) and **`LM_BETA`** (word-insertion
  bonus, default **1.0**) are the values the model was tuned with. Leaving them
  unset keeps whatever the LM's own `lm_config.json` specifies, which is the
  right choice unless you are tuning for your own domain.
- **`BEAM_WIDTH`** defaults to 32. Lower it to trade a little accuracy for
  latency.
- The LM expects **lowercase Vietnamese with numbers spelled out**, matching the
  model's own output format. Feed it text in another casing or script and the
  scores will be meaningless.
- **Beam search runs on the CPU**, one utterance at a time, and becomes the
  bottleneck on a fast GPU — roughly 15 ms per utterance at beam 32 on a single
  core. If you need maximum throughput, leave the LM off; if you need maximum
  accuracy, turn it on and give the worker spare CPU.
- The bundled LM was trained on about **29M words** of Vietnamese news
  (Leipzig corpora) and Wikipedia.
- A request can override the server default with `use_lm=false` (force greedy)
  or `use_lm=true`. Asking for an LM that is not loaded is a `503`, not a silent
  downgrade.

Installing it needs a C++ compiler because `kenlm` has no prebuilt wheel:

```bash
pip install ".[lm]"
```

In Docker this is gated by `--build-arg INSTALL_LM=false` if you would rather
skip the compile entirely and ship a greedy-only image.

## Configuration

Every setting is an environment variable, so the same image works for
GPUStack, `docker run`, and the CLI. Command-line flags win over the
environment.

### Model

| Variable | Default | Notes |
| --- | --- | --- |
| `MODEL_PATH` | `qnguyen3/QUAN-Transcribe-1.0` | HF repo id or local directory |
| `SERVED_MODEL_NAME` | `QUAN-Transcribe-1.0` | Name clients pass as `model` |
| `DEVICE` | auto | `cuda`, `cuda:1`, `mps`, `cpu` |
| `HF_TOKEN` | – | For the gated audio encoder |
| `PRELOAD` | `true` | Load at startup rather than on the first request |

### Decoding

| Variable | Default | Notes |
| --- | --- | --- |
| `USE_LM` | `false` | Beam search with the bundled KenLM |
| `LM_PATH` | – | Your own KenLM model; implies `USE_LM` |
| `LM_ALPHA` | `lm_config.json` | LM weight (shipped value 0.3) |
| `LM_BETA` | `lm_config.json` | Word-insertion bonus (shipped value 1.0) |
| `BEAM_WIDTH` | `32` | Beams kept |
| `LEFT_CHUNKS` | all | Streaming lookback; `0` = lowest latency |
| `CHUNK_SECONDS` | `30.0` | Max seconds per decoded piece |
| `BATCH_SECONDS` | `600.0` | Padding budget per forward pass; lower it if VRAM is tight |

### Throughput

These trade latency for throughput and are the knobs that matter when serving
more than one client at a time. See [Performance](#performance).

| Variable | Default | Notes |
| --- | --- | --- |
| `MICROBATCH_SECONDS` | `0.15` | Overlap required before concurrent requests share a forward pass; `0` disables cross-request batching |
| `MICROBATCH_MAX_WAIT_MS` | `20.0` | How long a request waits for companions before its batch launches |
| `MICROBATCH_MAX_REQUESTS` | `8` | Cap on requests packed into one batch |
| `TF32` | `true` | Allow TF32 matmuls on Ampere+ (CUDA only); `false` for exact FP32 |
| `CPU_THREADS` | auto | torch thread count for the CPU profile; set it when the container has a cgroup quota |

### Streaming

| Variable | Default | Notes |
| --- | --- | --- |
| `ENDPOINT_SECONDS` | `0.8` | Silence that ends an utterance |
| `MAX_UTTERANCE_SECONDS` | `30.0` | Hard cap per utterance |
| `STREAM_STEP_SECONDS` | `0.48` | One encoder chunk; leave it alone |
| `MAX_STREAM_SECONDS` | `3600.0` | Reject longer streams |

### HTTP

| Variable | Default | Notes |
| --- | --- | --- |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | |
| `API_KEY` | – | When set, every request needs `Authorization: Bearer <key>` |
| `CORS_ORIGINS` | `*` | Comma-separated |
| `MAX_UPLOAD_MB` | `200` | Rejects larger uploads |
| `LOG_LEVEL` | `info` | |

`/health` returns `200` with `"status": "loading"` while the weights download,
and only `503` after a hard failure. That keeps the GPUStack probe from
flapping during a multi-minute first start.

## Performance

QUAN-Transcribe is a CTC model: one forward pass per piece of audio, no
autoregressive loop. That makes the useful optimisation different from an LLM
server's, and there are four levers, in rough order of impact.

### 1. Cross-request batching (the default)

A forward pass pads every piece of audio in it to the longest piece, so
batching *k* pieces of up to *n* samples costs `k * n`. Upstream the batching is
within one request: a 10-minute file becomes ~20 pieces that share passes. Two
requests arriving at the same time, though, each launch their own passes.

This server widens that window across requests (`app/batch.py`). A collector
thread gathers pieces from concurrent requests for `MICROBATCH_MAX_WAIT_MS`,
groups them by decode signature, sorts ascending by length, and greedily packs
them under the same padded-cost budget the library uses. Encoding is where
almost all the compute goes, and it is a pure GEMM pipeline, so packing more
work into one launch is close to free throughput.

The cost is up to `MICROBATCH_MAX_WAIT_MS` of added latency, and only on the
first piece of a request. Set `MICROBATCH_SECONDS=0` to turn it off entirely and
get the plain per-request path; set `MICROBATCH_MAX_WAIT_MS=0` to batch only
what happens to already be queued (no added latency at all, still helps under
load). Pieces with different decode signatures — streaming vs not, different
`left_chunks`, greedy vs beam search — are never mixed, because they need
different model behaviour.

Try `MICROBATCH_SECONDS=0` first if you serve one client at a time.

### 2. TF32 on Ampere and later

`TF32=true` (the default) lets FP32 matmuls run on the tensor cores. For a CTC
model the precision loss is immaterial next to the speedup, and it is a no-op
before Ampere and on CPU. Set `TF32=false` if you need bit-exact FP32.

The model already autocasts the encoder to bf16 on CUDA, so there is no bf16
knob to turn: it is on whenever the device is CUDA.

### 3. The language model is a CPU bottleneck

With `USE_LM=true` the beam search runs on the CPU, one utterance at a time,
*after* the GPU forward pass. At beam 32 it is roughly 15 ms per utterance on a
single core, which on a fast GPU becomes the wall-clock bottleneck — the GPU
finishes and waits. If throughput matters more than the last two WER points,
leave the LM off; if accuracy matters more, turn it on and give the worker spare
CPU. Raising or lowering `BEAM_WIDTH` moves this cost linearly.

### 4. CPU is a different machine

On CPU there is no fast device to hide behind, so the model is a plain FP32
GEMM pipeline that scales with cores and with batch width:

- `DEVICE=cpu` pins the profile, so a host that happens to expose a GPU does not
  silently switch paths.
- `CPU_THREADS` sets the torch thread count. Leave it unset to let torch size
  itself to the visible cores; set it to the container's quota when it is
  cgroup-limited, or the threads will oversubscribe and thrash.
- Widening the batch window helps more than it does on GPU
  (`MICROBATCH_SECONDS=0.25`, `MICROBATCH_MAX_WAIT_MS=50` are the shipped CPU
  values).
- bf16 autocast does not apply on CPU, so the CUDA speedups above do not carry
  over. Expect realtime-factor numbers in the single digits, not thousands.

`GET /metrics` reports `realtime_factor` and, under `batcher`, how many batches
and pieces have run — the direct way to see whether batching is actually
coalescing your traffic.

## Docker

```bash
./scripts/build.sh                          # -> quan-asr:1.0.0
TAG=registry.example.com/asr:1.0.0 ./scripts/build.sh
INSTALL_LM=false ./scripts/build.sh         # skip the kenlm compile
CUDA=12.4.1 TORCH_INDEX=https://download.pytorch.org/whl/cu124 ./scripts/build.sh
```

Run it, keeping the Hugging Face cache on a volume so the weights survive a
restart:

```bash
docker run --gpus all -p 8000:8000 \
  -v hf-cache:/cache/huggingface \
  -e HF_TOKEN=hf_... \
  -e USE_LM=true \
  quan-asr:1.0.0
```

`docker/docker-compose.yaml` ships two profiles built from the same image:

```bash
docker compose --profile gpu up    # GPU: port 8000, no device pin, TF32 on
docker compose --profile cpu up    # CPU: port 8001, DEVICE=cpu, wider batching
```

## Deploying on GPUStack

1. Build and push the image to a registry the workers can reach.
2. On the **Inference Backends** page choose **Add Backend → Custom** and paste
   `deploy/gpustack-backend.yaml` (or apply
   `deploy/gpustack-backend-source.yaml` as an `InferenceBackendSource`).
   The `backend_name` must end in `-custom` — GPUStack enforces this.
3. Point `image_name` at your pushed image, then deploy from that backend and
   pick the version that matches the worker.

The manifest declares **two versions of the same backend**, which is how
GPUStack expresses "CPU or GPU": scheduling is filtered by the version's
framework, and `cpu` matches any worker while `cuda` matches only workers with
an NVIDIA GPU.

| Version | `custom_framework` | Env | Schedulable on |
| --- | --- | --- | --- |
| `1.0.0` | `cuda` | TF32 on, GPU batch window | GPU workers |
| `1.0.0-cpu` | `cpu` | `DEVICE=cpu`, wider batch window | any worker, no GPU needed |

Both point at the same image tag. The environment is the only difference, which
is deliberate: the CPU path is the same code with the same single-pass model, so
the image does not need to be built twice. GPUStack applies the version's `env`
on top of `default_env`, and version-level values win.

The manifest also tells GPUStack to:

- run the image's entrypoint with
  `--model-path {{model_path}} --served-model-name {{model_name}} --host {{worker_ip}} --port {{port}}`,
  which the server accepts as ordinary flags;
- poll `health_check_path: /health` to decide when the model is ready.

Because GPUStack downloads and mounts the model itself and passes the local
directory through `{{model_path}}`, the server never re-downloads it. Custom
backends are single-worker deployments.

## GitLab CI

`.gitlab-ci.yml` builds on tags matching `v-*-release` using the shared runner,
so nothing is built for ordinary commits:

```bash
git tag v-1.0.0-release
git push origin v-1.0.0-release
```

That runs `ruff` and `pytest`, builds the image with BuildKit, and pushes
`$CI_REGISTRY_IMAGE/asr-engine:$CI_COMMIT_TAG` plus `:latest` to the project
registry, then opens a release with the changelog. Use the pushed image in
`deploy/gpustack-backend.yaml`:

```yaml
image_name: gitlab.huongnghiepaau.com:5050/dev/asr-engine/asr-engine:v-1.0.0-release
```

One tag serves both backend versions: point the `1.0.0-cpu` version at the same
image and let its `env` do the work.

There is no deploy stage: GPUStack pulls the image itself when you deploy.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"

pytest -q                     # ~88 tests, no GPU or model download needed
ruff check app tests scripts
ruff format app tests scripts
```

The tests monkeypatch a scripted fake engine into `app.server`, so they exercise
the HTTP, SSE and WebSocket contracts without loading the model.

To drive the real endpoints without downloading 356M parameters:

```bash
python scripts/dev_server.py --port 8123   # stub engine, real ASGI stack
./scripts/smoke.sh http://127.0.0.1:8123   # curl through every response format
```

`scripts/make_sample_audio.py` writes a small tone/quiet WAV if you need a
fixture.

### Layout

```
app/config.py     settings, all env-driven
app/audio.py      decode, resample, quiet-point splitting, segments
app/formats.py    json/text/srt/vtt/verbose_json rendering
app/batch.py      cross-request micro-batching of decode pieces
app/engine.py     thread-safe model wrapper, streaming and live sessions
app/schemas.py    request validation, OpenAI-shaped errors
app/server.py     FastAPI routes, SSE, WebSocket
app/ui.py         browser playground
app/cli.py        `quan-asr` entrypoint
```

Model calls are serialized behind a lock, and loading happens behind its own
lock so concurrent first requests cannot race. A load failure does not kill the
container — it surfaces through `/health` as `"error"`.

## Credits

Model and reference implementation: [QUAN-Transcribe](https://github.com/qnguyen3/QUAN-Transcribe)
by [qnguyen3](https://huggingface.co/qnguyen3).
