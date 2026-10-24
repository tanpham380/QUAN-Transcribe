---
description: 'Run the full validation gate: tests, lint and format check.'
agent: 'agent'
tools: ['execute/runInTerminal', 'read/problems']
---

Run the repository's validation gate and report the result. Do not fix anything
unless I ask.

Run these in order and stop at the first failure:

```bash
pytest -q
ruff check app tests scripts
ruff format --check app tests scripts
```

Then report:

1. **Tests** — pass/fail counts. If anything failed, show the failing test names
   and the assertion output.
2. **Lint** — the rule codes and file:line for each finding.
3. **Format** — which files would be reformatted.

Notes:

- `ruff` deliberately excludes `quan_transcribe/`. Do not add it to the targets
  and do not report its style as a finding.
- The suite needs no GPU and no model download. If it tries to download
  something, that is a bug in the test — report it.
- If `pytest` is not installed, say so and stop; do not install it without
  asking.
