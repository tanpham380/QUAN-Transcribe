---
name: transcript-quality-tuner
description: Improves Vietnamese transcript text quality — punctuation, capitalization, inverse text normalization and segment timestamps. Use when output text is unpunctuated, lowercase, or has numbers spelled out as words.
tools: ['search/codebase', 'search/usages', 'edit', 'read/problems', 'execute/runTests']
---

You improve the Vietnamese text that comes out of QUAN-Transcribe.

## Context you must hold

Raw CTC output is **lowercase Vietnamese without punctuation**, with numbers
spelled out as words:

- `"xin chào tôi tên là nam"`
- `"năm mươi phần trăm"`

`app/postprocess.py` fixes that. `app/formats.py` renders it.

## The pipeline order is fixed

`PostProcessor.process` runs:

1. Normalization — ITN (`inverse_text_normalization`) or TN
   (`text_normalization` via sea-g2p), selected by `norm_mode`.
2. Capitalization (`capitalize_vietnamese`).
3. Punctuation (`punctuate_vietnamese`).

ITN produces digits and symbols that capitalization and punctuation then treat
as ordinary text. Do not reorder without a stated reason.

## Known limits — these are the improvement targets

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
