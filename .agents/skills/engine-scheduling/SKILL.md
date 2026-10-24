---
name: engine-scheduling
description: Diagnose and change request scheduling, batching, priority, backpressure and latency in the QUAN-Transcribe serving layer. Use when a request is slow, a 503 appears, live audio stutters behind a bulk upload, or throughput needs to improve.
---

# Engine scheduling

`app/core.py` (`EngineCore`) is a dedicated thread that owns the model. Every
caller hands it work; only that thread touches the model.

## Decision tree

**Is a request slow?**
1. Is it a *bulk* request (file upload) waiting behind another bulk request?
   → Check `microbatch_seconds` / `microbatch_max_wait_ms`. The core waits a
   short window to gather companions; too long a window adds latency to every
   request, too short a window loses batching.
2. Is it an *interactive* request (streaming step) waiting behind bulk?
   → It should not be. Interactive preempts bulk at sub-batch granularity. If it
   is not, check `subbatch_seconds`: a large value makes one big file a single
   uninterruptible pass. `0` means no cap.
3. Is the queue full? → `EngineBusy` → HTTP `503`. That is backpressure working.
   Raise `max_queue` only if you have measured headroom.

**Is throughput low?**
- Are pieces being coalesced? Coalescing requires an equal *decode signature*.
  A parameter that changes how a piece is decoded splits the batch.
- Is the padded-cost budget (`batch_seconds`) the limit? The core packs by
  `longest * (len(batch) + 1) <= batch_seconds`.

**Is live audio stuttering?**
- `max_interactive` caps in-flight streaming jobs. Hitting it is a real signal.
- VAD runs on request threads, never the model thread — verify that is still
  true if you touched `app/vad.py`.

## Invariants

- One thread owns the model. Never call it from a request thread.
- Never hold a lock across a whole decode.
- At least one sub-batch runs per scheduler iteration (no livelock).
- `EngineBusy` → `503` + `Retry-After` is intentional. Do not make the queue
  unbounded.

## Not applicable here

This is CTC, not autoregressive. There is no KV cache, no continuous batching,
no paged attention, no token-level scheduling. Do not propose them.

## Verify

```bash
pytest -q
```

`app/core.py` and `app/engine.py` are the highest-risk files. Run the full
suite, not a filtered subset.
