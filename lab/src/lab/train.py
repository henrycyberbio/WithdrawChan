"""Fit the trained-logit observer's coefficients on simulated paths.

The inner layer fits one conditional logit over the groups of every simulated training path; the
outer layer owns the config, the snapshot and the output file.
"""

from __future__ import annotations

import json
import random
from fractions import Fraction
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

from lab import run, sim, study
from lab.behaviour import Theta
from lab.observers import logit
from lab.sim import DRAIN, KEEP, NON

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lab.study import StudyInputs

__all__ = ["train", "train_logit"]


# ==== inner layer ====


def train_logit(inputs: StudyInputs, *, paths: int, tag: str, pairs: Sequence[tuple[str, str]],
                rhos: Sequence[Fraction], adopters: Sequence[str]) -> dict:
    """Fit the trained-logit observer's coefficients over `paths` simulated training paths.

    Each path draws its historical `pi`, `s`, `w`, `g`, its rule and maximum model, `rho` and adopter
    kind from `tag`-seeded draws, simulates once and contributes one conditional-logit group per
    event.  The coefficients are fitted at `p = 0`, whatever the config's `p` is.

    Args:
        inputs: The study inputs the paths simulate over.
        paths: The number of training paths to simulate.
        tag: The seed label; distinct tags give distinct draws.
        pairs: The `(rule, m_model)` choices.
        rhos: The adoption ratios to draw from.
        adopters: The adopter kinds to draw from.

    Returns:
        The fitted record: the `beta` coefficients, the `feature_names`, the `whole_unit` (the `d`
        the features divide by) and `logit_units` that pin the axis, and the `n_groups`/`n_paths`
        counts.
    """
    config = inputs.config
    spec = config.logit_train
    pi_lo, pi_hi = float(Fraction(spec.pi_lo)), float(Fraction(spec.pi_hi))
    s_hi, w_hi, l2 = float(Fraction(spec.s_hi)), float(Fraction(spec.w_hi)), float(Fraction(spec.l2))
    groups: list[logit.Group] = []
    # Exact memory reduction: `logit.features` is a pure function of the public amount and the
    # ordered balances, and `config.logit_units` and `inputs.d` are fixed for the whole call, so
    # one `(matrix, eligible)` pair per distinct `y` is shared across the call's events.
    features_by_y: dict[int, tuple] = {}
    for t in range(paths):
        n = len(inputs.balances)
        # The "ml-train" seed labels are part of the fitted coefficients' identity: renaming them refits beta.
        rng = random.Random(sim.derive_seed("ml-train", tag, t))
        pi = rng.uniform(pi_lo, pi_hi)
        s = rng.uniform(0.0, s_hi)
        w = rng.uniform(0.0, w_hi)
        g = rng.choice(("hi", "lo"))
        rho = rng.choice(rhos)
        rule, m_model = rng.choice(pairs)
        kind = KEEP if rng.choice(adopters) == "keep" else DRAIN
        # The trained-logit coefficients are fitted at p = 0, whatever config.p is.
        behaviour = study.behaviour(inputs, Theta(pi, s, w, g), rule, m_model, p=Fraction(0))
        admitted = sim.adoption_membership(n, rho, sim.derive_seed("ml-train-adoption", tag, t))
        kinds = tuple(kind if is_adopter else NON for is_adopter in admitted)
        events = sim.simulate(behaviour, inputs.balances, kinds, sim.derive_seed("ml-train-path", tag, t))
        for event in events:
            pair = features_by_y.get(event.y)
            if pair is None:
                pair = features_by_y[event.y] = logit.features(event.y, inputs.balances, config.logit_units,
                                                               inputs.d)
            matrix, eligible = pair
            groups.append((matrix, eligible, event.source))
    beta = logit.fit(groups, l2=l2)
    return {
        "beta": [float(coef) for coef in beta],
        "feature_names": list(logit.feature_names(config.logit_units)),
        "whole_unit": inputs.d,
        "logit_units": list(config.logit_units),
        "n_groups": len(groups),
        "n_paths": paths,
    }


# ==== outer layer ====


def train(config_path: str, *, data: str, out: str) -> dict:
    """Fit the trained-logit observer on the config's own paths and write `<out>/logit_beta.json`.

    Loads and pins the config, manifest and snapshot like :func:`lab.run.run`, fits one beta over
    `config.logit_train.paths` paths, and records the code commit and the snapshot sha256 alongside
    the fit.

    Args:
        config_path: The experiment config JSON file.
        data: The data directory holding `<manifest stem>/snapshot.json`.
        out: The output directory for `logit_beta.json`.

    Returns:
        The output directory and the fitted path and group counts.

    Raises:
        lab.schema.PinError: The snapshot pins do not match the config.
    """
    inputs = study.load(Path(config_path), Path(data))
    config = inputs.config
    result = train_logit(inputs, paths=config.logit_train.paths, tag=config.seed_root,
                         pairs=tuple(product(config.rules, config.m_models)),
                         rhos=tuple(Fraction(r) for r in config.rhos), adopters=config.adopters)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {**result, "commit": run.commit(Path(__file__).parent), "snapshot_sha256": inputs.snapshot_sha256}
    (out_dir / "logit_beta.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8",
                                          newline="\n")
    return {"out": str(out_dir), "n_paths": result["n_paths"], "n_groups": result["n_groups"]}
