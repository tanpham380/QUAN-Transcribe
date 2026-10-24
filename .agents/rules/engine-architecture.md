---
trigger: model_decision
description: "Apply when changing the engine core, request scheduling, batching, backpressure, concurrency, latency, or anything that touches how the model is called. Covers the single-model-thread invariant and the EngineBusy/503 contract."
---

# Engine architecture rules

## The one-thread invariant

`app/core.py` (`EngineCore`) owns the model. It is a dedicated thread running a
scheduler loop; **only that thread ever touches the model**. Callers from any
thread hand it work through `submit` (one callable, run to completion) or
`submit_bulk` (audio pieces the core may coalesce).

Why: the model is not re-entrant. A forward pass mutates shared buffers, and on
CUDA concurrent kernels would fight for the same memory.

**Never** call the model directly from a request thread. **Never** reintroduce a
global lock held for the duration of a decode — that is exactly the
head-of-line blocking this design removes. A ten-minute file is ~20 pieces; a
microphone step wanting to decode one 0.48 s chunk must not wait for all of them.

## Priority classes

Two classes, strict priority order:

- **interactive** — one callable, run immediately, never delayed to batch. A
  streaming step (`LiveSession.accept`) or a flush.
- **bulk** — audio pieces from `transcribe()`. The core waits `window_ms` for
  near-simultaneous pieces, groups them by *decode signature*, sorts by length
  and greedily packs them into sub-batches under the same padded-cost budget the
  library uses.

Interactive work preempts bulk at **sub-batch granularity**: the core checks for
interactive jobs before each sub-batch and pushes the rest back to the head of
the queue. At least one sub-batch always runs per iteration, so gathered work
can never be starved into livelock. Preserve that property.

## Backpressure is a feature, not a bug

The bulk queue is bounded (`max_queued_pieces`). When full, `submit_bulk` raises
`EngineBusy` and the HTTP layer returns `503` with `Retry-After`.

This converts unbounded latency growth into a fast, honest refusal — the single
most important property for keeping a saturated server usable. Do not "fix" a
503 by making the queue unbounded.

## What this is not

vLLM's engine core shares the *shape* (one owner of the model, a frontend that
submits and awaits) but not the strategy. vLLM schedules *tokens* across steps
with a KV cache so an autoregressive decode can interleave many sequences.

QUAN-Transcribe is **CTC**: one forward pass per piece, no cache, no incremental
state across calls. There is no continuous batching here, no paged attention, no
step-level iteration. The only reuse available is putting several whole pieces
into one padded forward pass — which is what the bulk path does. Do not propose
token-level scheduling.

## Decode signature

Pieces are only coalesced into one forward pass when their signatures are equal,
because they would otherwise need different model behaviour. If you add a
parameter that changes *how* a piece is decoded, it belongs in the signature.

## Validation

`app/core.py` and `app/engine.py` are the highest-risk files in the repo. After
changing either, run the full suite, not a filtered subset:

```bash
pytest -q
```
