---
name: diagnose-latency
description: Investigate a slow request, a 503, or stuttering live audio in the QUAN-Transcribe serving layer. Use when latency is high, requests are refused, throughput is low, or streaming audio stutters behind a bulk upload.
---

# Diagnose latency

Investigate a performance or saturation problem in the serving layer. Report
findings; do not change code unless asked.

## What to check, in order

1. **Which path?** Bulk (file upload) or interactive (streaming step)? They have
different priority and different failure modes.

2. **Is it backpressure working correctly?** A `503` with `Retry-After` means the
bulk queue hit `max_queued_pieces`. That is the designed behaviour, not a bug.
Confirm the queue bound and whether the client retries.

3. **Is interactive being preempted by bulk?** It should not be. Check
subbatch_seconds — a large value makes one big file a single
uninterruptible forward pass. `0` means no cap.

4. **Is batching happening?** Pieces only coalesce when their *decode signature*
is equal. A parameter that changes how a piece is decoded splits the batch.
Check microbatch_seconds and microbatch_max_wait_ms.

5. **Is the padded-cost budget the limit?** The core packs by
longest * (len(batch) + 1) <= batch_seconds.

6. **Is VAD on the model thread?** It must not be. VAD runs on request threads.

7. **Is `max_interactive` being hit?** That caps in-flight streaming jobs.

## Report

- The path you traced, with file:line references.
- The specific bound or parameter that is the bottleneck.
- Whether the behaviour is by design or a bug.
- If it is a bug, the smallest change that would fix it — but do not apply it.

Remember: this is CTC, not autoregressive. There is no KV cache, no continuous
batching, no paged attention. Do not propose them.
