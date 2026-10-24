---
name: 'Inference library'
description: 'Rules for the upstream quan_transcribe/ inference library: decoding, streaming, LM beam search and the feature extractor.'
applyTo: 'quan_transcribe/**/*.py'
---

# Inference library rules

`quan_transcribe/` is the upstream inference library. It is **excluded from
`ruff`** on purpose (`pyproject.toml` → `[tool.ruff] exclude`). Do not add it to
the lint targets and do not reformat it to satisfy the server's style rules.

## Two checkpoint layouts

`from_pretrained` accepts either:

| layout | files | example |
|---|---|---|
| safetensors | `config.json` + `units.json` + `model.safetensors` | `qnguyen3/QUAN-Transcribe-1.0` |
| torch | one `*.pt` holding `model`, `model_config`, `gemma_ids` | `5CD-AI/VieGem-ASR` |

The format is detected from the files present, so no flag is needed. Both paths
share everything except what the loader reads: `head_cosine` in the config
selects a cosine-similarity output head (with a learned `logit_scale`) instead of
a dot product, and the number of unit rows follows `vocab_size`.

Each family keeps the beam-search weights it was tuned with (α 0.3 / β 1.0 for
QUAN-Transcribe-1.0, α 0.35 / β 1.5 for VieGem-ASR); `lm_config.json` overrides
them when present.

A checkpoint using an architecture feature this package does not implement (for
example a LoRA-adapted encoder) must fail with a `checkpoint does not match the
model` message naming the unexpected keys — never silently produce garbage.

## The model is CTC

One forward pass per piece. No autoregressive decoding, no KV cache, no
incremental state across calls. `greedy_decode` collapses repeats and blanks.

## Feature extractor

`features.py` computes log-mel in torch so it can run batched on the GPU. It must
stay **numerically identical** to the HF extractor, which is a per-utterance
numpy loop. The window and mel filters are copied from the extractor; the asserts
in `LogMel.__init__` guard the extractor settings the implementation assumes. Do
not relax them.

## Cut points

`audio.py::split_quiet` is the reference. `app/audio.py::split_with_offsets`
mirrors it cut-for-cut so API timestamps line up with what the model decoded.
Change one, change the other.

## Streaming

`live.py::LiveSession` re-decodes the current utterance in streaming mode each
time a new 0.48 s chunk arrives. Streaming never looks more than one chunk ahead,
so text from finished chunks stays put and only the newest words can change. An
utterance ends after a pause with no new tokens, then is decoded once more with
full context.

## Optional dependency

`kenlm` is an extra (`pip install '.[lm]'`), built from source and needing a C++
compiler. Import it lazily inside `KenLMDecoder.__init__` and inside `decode`.
