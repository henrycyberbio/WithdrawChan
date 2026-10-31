---
name: withdrawchan-dev
description: Use when building, testing or changing the WithdrawChan recommendation rule in src/. Skip for lab/ experiments.
---

# Build and test the recommendation rule

`python/` and `js/` implement the same rule. Change both together and keep both suites passing.

## Python (`python/`)

- `uv sync` installs the package and the dev group (pytest, fire).
- Test with `uv run python -m pytest -q`. On Windows, `uv run pytest` can fail with "uv trampoline failed to canonicalize script path"; call pytest through `python -m`.

## TypeScript (`js/`)

- `npm ci`, then `npm test`. The test runs `index.test.ts` directly with `node --test`, so Node must strip TypeScript types natively (tested on Node 24).
- `npm run build` compiles `index.ts` to `dist/` with `tsc`. `dist/` is not tracked.

## Shared fixtures

- `vectors.json` is generated: never edit it by hand. After changing the ladder, the walk or how it calls `rng`, regenerate it from `python/` with `uv run python tests/vectors.py write`, review the diff, then run both suites. Only the Python suite re-enumerates the vectors; the TypeScript suite only reads them.
- `examples.json` holds hand-computed USDT examples. Edit it by hand; both suites check every row. Keep the README's example table consistent with it.
- `units.json` holds per-asset rounding units in whole tokens; both implementations read it at runtime — never add a hard-coded copy.
- `python/tests/test_withdrawchan.py` and `js/index.test.ts` test the same behaviour under matching test names, except tests that only make sense in one language (vector regeneration in Python, fraction reduction in TypeScript). A new behaviour needs a test in both files.
