---
trigger: glob
globs: "app/audio.py, app/vad.py, quan_transcribe/audio.py, quan_transcribe/features.py"
description: "Audio decoding, resampling, VAD and segmentation rules. Apply when changing cut points, timestamps, silence handling or the feature extractor."
---

# Audio and segmentation rules

## Cut points must stay in lockstep

`app/audio.py::split_with_offsets` mirrors `quan_transcribe/audio.py::split_quiet`
cut-for-cut. The API's reported segment timestamps must line up with the pieces
the model actually decoded.

**If you change one, change the other.** A divergence produces timestamps that
are subtly wrong rather than obviously broken, which is the worst failure mode.

The rule itself: cut at the lowest-energy 50 ms window in the last `search_s`
seconds before the `chunk_s` limit. `search` is clamped to `span - win` so a
small span cannot search back before the range start.

## VAD is optional and lazy

`app/vad.py` imports `silero_vad` on first use. The process must still start
instantly when the VAD is off, and the package must not be required at all.

A missing `silero-vad` degrades to energy-based splitting. It must never fail
the request.

## The torch thread-count trap

`silero_vad.model` calls `torch.set_num_threads(1)` **at import time**. Left
alone that silently cripples CPU decoding for the whole process. `app/vad.py`
captures the thread count before the import and restores it afterwards. Keep
that guard.

## VAD runs on request threads

Never on the model thread — it must never block a decode. Silero keeps state
between calls, so the model call is serialised with a lock. ONNX inference on a
full recording costs ~1% of realtime, so the serialisation is not a bottleneck.

Prefer the ONNX backend (~2.5x faster than TorchScript on CPU, no GPU needed)
and fall back when `onnxruntime` is absent.

## Speech padding

`speech_pad_ms` defaults to **250 ms**, not Silero's 30 ms. Silero's default is
too tight for this encoder: the first syllable of a region gets clipped
("quý khách" → "cách"). Do not lower it without measuring.

## `trim_silence`

- `True` (default): silence-only pieces are dropped instead of decoded. The
  model returns `""` for them, so this is a pure compute saving, not a quality
  trade-off.
- `False`: silence is kept and VAD only supplies cut points, which is
  bit-for-bit the VAD-less transcript.

Both modes must stay correct. `keep_silence=True` returns the whole waveform cut
at every speech boundary.

## Feature extractor

`quan_transcribe/features.py` computes the log-mel features in torch so they can
run batched on the GPU. It must stay **numerically identical** to the HF
extractor, which is a per-utterance numpy loop. The window and mel filters are
copied from the extractor; the asserts at the top of `LogMel.__init__` guard the
extractor settings the implementation assumes. Do not relax them.

## Sample rate

Everything is 16 kHz mono. `to_16k_mono` down-mixes and resamples with `soxr`.
Silero only supports 8 kHz and 16 kHz, and the engine works at 16 kHz.
