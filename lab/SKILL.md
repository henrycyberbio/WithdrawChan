---
name: withdrawchan-lab-dev
description: Use when testing or changing the lab/ simulation, observers or reports. Skip for the recommendation rule in src/.
---

# Test and change the lab

## Setup and tests

- `uv sync` installs the `lab` package, its `lab` command and the dev group (pytest).
- Test with `uv run python -m pytest -q`. On Windows, `uv run pytest` can fail with "uv trampoline failed to canonicalize script path". Call pytest through `python -m`.

## Running

- Use the `lab` command (`uv run lab …`). `python -m lab.cli` does nothing because the module has no `__main__` guard.
- `group` is slow. On the machine used for these experiments, it took about 4 hours on USDT and 36 hours on USDC with 12 workers. Leave it out of a configuration unless you need the four-observer results.
- Simulated amounts are reproducible per path for a fixed snapshot and configuration. Observer scores are only reproducible at the level of the conclusions, because `group` resampling diverges on last-bit differences.
