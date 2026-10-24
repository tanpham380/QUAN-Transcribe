---
name: engine-review
description: Review a diff to the QUAN-Transcribe serving layer for concurrency, scheduling, backpressure and audio-pipeline correctness. Use when reviewing changes to app/core.py, app/engine.py, app/vad.py or app/audio.py, or when asked to review a PR that touches the engine.
---

# Engine review

Read-only review of changes to the serving layer. Report findings; do not edit.

Work through these in order. Report only findings you can point at a specific
line for.

## 1. The one-thread invariant

`app/core.py` (`EngineCore`) owns the model. Only its dedicated thread may touch
the model.

- Does any new code call the model from a request thread?
- Is a lock held across a whole decode? That is the head-of-line blocking
  `core.py` exists to remove.
- Does a new decode parameter change *how* a piece is decoded without being
  added to the decode signature? Pieces with different signatures must not be
  coalesced.

## 2. Backpressure

`EngineBusy` → HTTP `503` + `Retry-After` is intentional. Flag any change that
makes the bulk queue unbounded, swallows `EngineBusy`, or retries it internally
without a bound.

## 3. Scheduler liveness

At least one sub-batch must run per scheduler iteration, or gathered work can be
starved into livelock. Check that interactive preemption still pushes the
remainder back to the head of the queue.

## 4. VAD

- VAD must run on request threads, never the model thread.
- A missing `silero-vad` must degrade to energy-based splitting, not fail the
  request.
- The `torch.set_num_threads` capture/restore around the `silero_vad` import
  must survive. Without it, CPU decoding is silently capped process-wide.

## 5. Cut points

`app/audio.py::split_with_offsets` must mirror
`quan_transcribe/audio.py::split_quiet` cut-for-cut. If one changed and the other
did not, segment timestamps are subtly wrong — call it out as a bug.

## 6. Tests

`tests/test_app.py` must never import `torch` or `quan_transcribe`. It
monkeypatches a scripted fake engine into `app.server`. Flag any change that
breaks that property, because it is what lets the suite run on a plain Python
image with no GPU and no model download.

## Output

For each finding: file, line, what is wrong, and why it matters. Rank by
severity. If you find nothing, say so plainly — do not invent issues.

Run `pytest -q` and report the result. Do not run a filtered subset for changes
to `app/core.py` or `app/engine.py`.
