---
trigger: glob
globs: "*.py"
description: "Python style, typing, docstring and error-handling conventions for this repository."
---

# Python conventions

## Module shape

- Start every module with `from __future__ import annotations`.
- Use modern typing: `list[str]`, `dict[str, int]`, `X | None`. Not `List`,
  `Dict`, `Optional`.
- Group imports: stdlib, third-party, then local (`from .audio import ...`).
  `ruff` enforces this with the `I` rule.

## Docstrings explain *why*

The existing modules open with a rationale section, not a description of the
signature. `app/core.py`, `app/vad.py` and `app/audio.py` are the model to
follow — they state the problem, the alternatives considered, and why this
design won.

A docstring that restates the function name adds nothing. A docstring that
records a measurement ("a 90 s voicemail had 78 s of silence and 7 s of speech")
is worth ten lines of prose.

## Comments

Comment only what needs clarifying. Do not narrate obvious code. When a line
exists because of a non-obvious external constraint, say so — for example:

```python
# ``silero_vad.model`` calls ``torch.set_num_threads(1)`` at import time,
# which would cap CPU decoding for the whole process. Capture and restore it.
```

## Errors

Use the existing error types; they carry meaning the HTTP layer depends on:

| Type | Meaning |
|---|---|
| `APIError` | Renders the OpenAI error envelope. Carries `status_code`, `code`, `param`, `error_type`. |
| `AudioDecodeError` | The bytes are not decodable audio. |
| `EngineBusy` | Saturated; the caller should retry. Maps to `503` + `Retry-After`. |
| `EngineNotReady` | The model cannot be loaded. |

Do not raise bare `ValueError` where one of these fits.

## Configuration

Settings are environment-driven so the same image works as a GPUStack custom
backend or a plain `docker run`. To add one:

1. Add a field to `Settings` in `app/config.py` with a comment explaining the
   trade-off.
2. Add a matching flag in `app/cli.py` and wire it into the `overrides` dict.
3. Document it in the README configuration table.

Real environment variables win over flag defaults — that is deliberate, so
options only need passing when they differ from the image's configuration.

## Optional dependencies

`kenlm`, `silero-vad`/`onnxruntime` and `grpcio` are extras, not core
dependencies. Import them lazily inside the function that needs them, and
degrade gracefully when absent. A missing VAD must fall back to energy-based
splitting, not fail the request.

## Formatting

Line length 88, target `py310`. `E501` is ignored because the formatter owns
wrapping. Run:

```bash
ruff check app tests scripts
ruff format app tests scripts
```

`ruff` **excludes `quan_transcribe/`** on purpose — it is the upstream inference
library and is not held to the server's lint rules. Do not add it to the lint
targets and do not reformat it.
