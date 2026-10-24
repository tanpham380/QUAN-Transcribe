---
name: transcript-tuning
description: Improve Vietnamese transcript text quality — punctuation, capitalization, inverse text normalization and segment timestamps. Use when output text is unpunctuated, lowercase, or has numbers spelled out as words.
---

# Transcript tuning

Raw QUAN-Transcribe output is **lowercase Vietnamese without punctuation**, with
numbers spelled out as words. `app/postprocess.py` fixes that. `app/formats.py`
renders it.

## Decision tree

**Text is lowercase / unpunctuated?**
→ `app/postprocess.py`. Check `PostProcessor` flags first: `enable_punctuation`,
`enable_capitalization`, `enable_itn`, `norm_mode`. A caller may have disabled a
stage.

**Numbers are spelled out (`"năm mươi"` instead of `"50"`)?**
→ `inverse_text_normalization`. It runs six regex passes in a fixed order:
dates → years → percentages → currency → decimals → compound numbers. Order
matters; a later pass can re-match an earlier pass's output.

**A word is capitalized that should not be, or vice versa?**
→ `capitalize_vietnamese` uses a hardcoded dict (`_PROPER_NOUNS`, `_ACRONYMS`).
Anything outside the dict stays lowercase. This is a known limit.

**Timestamps are wrong or segments are split mid-sentence?**
→ Two separate causes:
- Cut points: `app/audio.py::split_with_offsets` must mirror
  `quan_transcribe/audio.py::split_quiet` cut-for-cut. If they diverge,
  timestamps are subtly wrong.
- Rendering: `merge_segments_by_sentence` joins adjacent pieces whose boundary
  is not a real pause. A boundary is real when the previous text ends in
  `.?!…:;` **or** the gap is ≥ 0.5 s.

**Streaming partials get a trailing `.` that then disappears?**
→ `punctuate_vietnamese(text, is_final=...)` must only punctuate when
`is_final` is true. `process_segments` passes `is_final=True` only for the last
segment.

## Known limits (candidates for improvement)

| Stage | Current approach | Limit |
|---|---|---|
| Capitalization | ~30 place names + ~23 acronyms in a dict, plus a sentence-start regex | Anything outside the dict stays lowercase |
| Punctuation | Regex question-word detection (`phải không`, `ai`, `gì`, `ở đâu`, ...), appends `.` or `?` | No commas or colons; only end-of-segment |
| ITN | Six sequential regex passes over number words | Passes interact; the year heuristic (`1900 <= val <= 2100`) is a disambiguation hack |

## Rules

- **Keep the public surface.** `PostProcessor` flags (`enable_punctuation`,
  `enable_itn`, `enable_capitalization`, `enable_g2p`, `norm_mode`) and the
  module-level function signatures are used by `app/server.py` and
  `app/grpc_server.py`. Do not break them.
- **`is_final` matters.** Streaming partials must not get a trailing `.` that
  then disappears. Only the last segment is punctuated.
- **Add a test for every behaviour change** in `tests/test_app.py`. The suite
  never imports `torch` or `quan_transcribe` — keep it that way.
- **Do not add a heavy dependency** to the core `dependencies` list for a text
  improvement. If a model is genuinely needed, it belongs in an extra and must
  degrade gracefully when absent.
- **Measure before claiming an improvement.** State the input, the old output
  and the new output.

## Verify

```bash
pytest -q tests/test_app.py -k "postprocess or format or segment"
ruff check app tests
```
