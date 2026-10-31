"""The two-layer experiment runner: one path (inner) and the grid over paths (outer).

The inner layer takes the participants as data; the outer layer owns the config, the snapshot, the
grid and the output files.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import random
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from lab import sim, study
from lab.observers.combinatorial import CombinatorialObserver
from lab.observers.historical import HistoricalObserver
from lab.observers.logit import TrainedLogitObserver
from lab.schema import ExperimentConfig, PinError
from lab.score import event_row, write_rows
from lab.sim import DRAIN, KEEP, NON

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lab.study import StudyInputs

__all__ = ["PathSpec", "commit", "grid", "run", "run_path"]


# ==== path spec ====


@dataclass(frozen=True)
class PathSpec:
    """One path of the grid: a replicate, `rho`, a rule, a maximum model and the adopters' kind."""

    replicate: int
    rho: Fraction
    rule: str
    m_model: str
    adopters: str

    @property
    def key(self) -> str:
        """The path's directory name, unique across the grid."""
        rho = f"{self.rho.numerator}-{self.rho.denominator}"
        return f"r{self.replicate:03d}_rho{rho}_{self.rule}_{self.m_model}_{self.adopters}"


# ==== inner layer ====

#: The `group` observer's fixed particle count; deliberately not a config axis.
GROUP_STATES = 1000


def run_path(spec: PathSpec, inputs: StudyInputs, beta: tuple[float, ...] | None) -> tuple[dict, list[dict]]:
    """Simulate one path and score every observer on it.

    The observers scored are the config's `observers`, in that order.  `group`, when listed, is the
    rule-aware comparison: it is built only then, from the path's own kind's behaviour and the
    candidate tool-user counts the config's `rhos` imply, and is scored from the public amount alone
    like the others.

    Args:
        spec: The path's replicate, `rho`, rule, maximum model and adopter kind.
        inputs: The study inputs the path simulates over.
        beta: The trained-logit observer's coefficients, or `None` when that observer is absent.

    Returns:
        The path summary and the per-event rows, observer order then event order.
    """
    config = inputs.config
    balances = inputs.balances
    n = len(balances)
    admitted = sim.adoption_membership(n, spec.rho, sim.derive_seed("adoption", config.seed_root, n, spec.replicate))
    kind = KEEP if spec.adopters == "keep" else DRAIN
    kinds = tuple(kind if is_adopter else NON for is_adopter in admitted)
    k = sum(admitted)
    behaviour = study.behaviour(inputs, study.theta_of(config.theta), spec.rule, spec.m_model,
                                p=Fraction(config.p))
    events = sim.simulate(behaviour, balances, kinds, sim.derive_seed("path", config.seed_root, n, spec.replicate))
    historical = behaviour[NON]
    # `inputs.d` is the rounding unit; the combinatorial observer pools on the adopter need grid
    # `config.quantum`, not on `d`.
    builders = {
        "historical": lambda: HistoricalObserver(historical, balances),
        "trained-logit": lambda: TrainedLogitObserver(np.asarray(beta), balances, config.logit_units, inputs.d),
        "combinatorial": lambda: CombinatorialObserver(balances, [e.y for e in events],
                                                       Fraction(config.combinatorial_tolerance), config.quantum),
        "group": lambda: _group_observer(historical, behaviour[kind], balances, config, spec, n),
    }
    rows: list[dict] = []
    summary: dict[str, dict[str, float | int]] = {}
    for name in config.observers:
        observer = builders[name]()
        started = time.perf_counter()
        for event in events:
            rows.append(event_row(observer.posterior(event.y), event, spec.key, name))
            observer.update(event.y)
        summary[name] = {"seconds": time.perf_counter() - started, "fallback_count": observer.fallback_count}
    content = {
        "spec": {"replicate": spec.replicate, "rho": f"{spec.rho.numerator}/{spec.rho.denominator}",
                 "rule": spec.rule, "m_model": spec.m_model, "adopters": spec.adopters},
        "K": k,
        "N": n,
        "n_events": len(events),
        "observers": summary,
    }
    if "group" in config.observers:
        content["group"] = {"K_candidates": list(_group_k_candidates(config, n)), "states": GROUP_STATES}
    return content, rows


def _group_k_candidates(config: ExperimentConfig, n: int) -> tuple[int, ...]:
    """The tool-user counts the `group` observer must consider: one per `rhos` entry, duplicates kept.

    Each entry maps to `K = floor(rho*N + 1/2)`; the observer weights one entry's `rho` uniformly, so
    a repeated `rho` carries its weight twice rather than being merged.
    """
    return tuple(math.floor(Fraction(rho) * n + Fraction(1, 2)) for rho in config.rhos)


def _group_observer(historical: object, adopter: object, balances: Sequence[int], config: ExperimentConfig,
                    spec: PathSpec, n: int) -> object:
    """Build the `group` observer for one path, importing its module only when a config scores it.

    `adopter` is the path's own kind's behaviour, which carries the rule and stopping policy but not
    who adopted; `historical` is the no-tool behaviour.  The observer draws from its own seeded
    `random.Random`, so it cannot shift any other observer's stream.
    """
    from lab.observers.group import GroupObserver

    rng = random.Random(sim.derive_seed("group", config.seed_root, n, spec.replicate, str(spec.rho), spec.rule))
    return GroupObserver(historical, adopter, balances, _group_k_candidates(config, n), GROUP_STATES, rng)


# ==== outer layer ====


def grid(config: ExperimentConfig) -> tuple[PathSpec, ...]:
    """Every path the config defines, in replicate → rho → rule → maximum model → adopter order.

    Args:
        config: The experiment config.

    Returns:
        The path specs, in that order.
    """
    specs: list[PathSpec] = []
    for replicate in range(config.replicates):
        for ratio in config.rhos:
            rho = Fraction(ratio)
            if rho == 0:
                specs.append(PathSpec(replicate, rho, "need", "double", "keep"))
                continue
            specs.extend(PathSpec(replicate, rho, rule, m_model, adopters)
                         for rule in config.rules
                         for m_model in (config.m_models[:1] if rule == "need" else config.m_models)
                         for adopters in config.adopters)
    return tuple(specs)


def run(config_path: str, *, data: str, out: str, workers: int = 1,
        logit_beta_path: str | None = None) -> dict:
    """Run every path of an experiment config into `out`, skipping a path that finished before.

    A path whose directory already holds a `path.json` is skipped, so a rerun resumes where it left
    off.  Each path's amounts use the config's `p`, the probability of adding the random decimal.

    Args:
        config_path: The experiment config JSON file.
        data: The data directory holding `<manifest stem>/snapshot.json`.
        out: The output directory for the per-path results and `run.json`.
        workers: The number of spawned worker processes; the results do not depend on it.
        logit_beta_path: Path to the trained-logit JSON artifact, or `None`.  Its `beta` holds the
            coefficients, its `whole_unit` and `logit_units` pin the rounding unit and tail axis
            they were fitted on, and any `snapshot_sha256` pins the snapshot.

    Returns:
        The output directory, the path and written counts and the participant count.

    Raises:
        lab.schema.PinError: The snapshot or the logit artifact does not match the config.
        ValueError: The config lists `trained-logit` but no artifact was given, or `out` already
            holds a `run.json` of a different config, snapshot or manifest.
    """
    started = time.perf_counter()
    inputs = study.load(Path(config_path), Path(data))
    config = inputs.config
    specs = grid(config)
    if logit_beta_path:
        payload = json.loads(Path(logit_beta_path).read_text(encoding="utf-8"))
        beta = _read_beta(payload, inputs)
    else:
        beta = None
    if beta is None and "trained-logit" in config.observers:
        msg = "config lists the trained-logit observer, so --logit-beta must name its artifact"
        raise ValueError(msg)
    out_dir = Path(out)
    manifest_sha256 = _file_sha256(Path(config_path).parent / config.manifest)
    _check_provenance(out_dir, config, inputs.snapshot_sha256, manifest_sha256)
    pending = [spec for spec in specs if not (out_dir / spec.key / "path.json").exists()]
    if workers > 1:
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(name, "1")
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            compute = partial(run_path, inputs=inputs, beta=beta)
            futures = {pool.submit(compute, spec): spec for spec in pending}
            for future in as_completed(futures):
                content, rows = future.result()
                _write_path(out_dir, futures[future], content, rows)
    else:
        for spec in pending:
            content, rows = run_path(spec, inputs, beta)
            _write_path(out_dir, spec, content, rows)
    _write_run(out_dir, config, len(inputs.balances), inputs.snapshot_sha256, manifest_sha256, specs,
               time.perf_counter() - started)
    return {"out": str(out_dir), "paths": len(specs), "written": len(pending), "N": len(inputs.balances)}


def _exact_positive_int(value: object) -> int | None:
    """The integer `value` is, or `None` when it is not an exact positive integer.

    Accepts a JSON integer or an `Amount`-style decimal string; booleans and floats are rejected so
    a fractional or boolean unit cannot be truncated into a match.
    """
    if isinstance(value, bool):
        return None
    if type(value) is int:
        number = value
    elif type(value) is str and value.isascii() and value.isdigit():
        number = int(value)
    else:
        return None
    return number if number > 0 else None


def _read_beta(payload: object, inputs: StudyInputs) -> tuple[float, ...]:
    """The coefficients of a trained-logit artifact, checked against this run's provenance.

    The artifact must pin `whole_unit` (the `d` the features divide by), `logit_units` (the tail
    axis, in order) and, when present, the `snapshot_sha256` it was fitted on.  Coefficients fitted
    on another rounding unit, tail axis or snapshot, or carrying no such record, are rejected rather
    than read blindly; every malformed field raises :class:`PinError`.
    """
    units = inputs.config.logit_units
    required = ("beta", "whole_unit", "logit_units")
    if not isinstance(payload, dict):
        raise PinError("logit_beta", type(payload).__name__, "object with beta, whole_unit, logit_units")
    missing = [key for key in required if key not in payload]
    if missing:
        raise PinError("logit_beta", sorted(payload), list(required))
    if _exact_positive_int(payload["whole_unit"]) != inputs.d:
        raise PinError("whole_unit", payload["whole_unit"], inputs.d)
    axis = payload["logit_units"]
    if not isinstance(axis, list):
        raise PinError("logit_units", axis, list(units))
    if [_exact_positive_int(unit) for unit in axis] != list(units):
        raise PinError("logit_units", axis, list(units))
    if "snapshot_sha256" in payload and payload["snapshot_sha256"] != inputs.snapshot_sha256:
        raise PinError("snapshot_sha256", payload["snapshot_sha256"], inputs.snapshot_sha256)
    beta = payload["beta"]
    width = 4 + len(units)
    if not isinstance(beta, list):
        raise PinError("beta", beta, f"{width} coefficients")
    if len(beta) != width:
        raise PinError("beta", f"{len(beta)} coefficients", f"{width} coefficients")
    coefficients: list[float] = []
    for coefficient in beta:
        if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)) \
                or not math.isfinite(coefficient):
            raise PinError("beta", coefficient, "a finite number")
        coefficients.append(float(coefficient))
    return tuple(coefficients)


def _file_sha256(path: Path) -> str:
    """The hex SHA-256 of `path`'s bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_provenance(out_dir: Path, config: ExperimentConfig, snapshot_sha256: str,
                      manifest_sha256: str) -> None:
    """Refuse an output directory whose `run.json` records different inputs than this run's.

    A directory without a `run.json` is an interrupted run and resumes unchecked.  The message names
    every field that differs, so the caller knows which of the config, the snapshot or the manifest
    changed.
    """
    path = out_dir / "run.json"
    if not path.exists():
        return
    recorded = json.loads(path.read_text(encoding="utf-8"))
    current = {"config": config.to_json(), "snapshot_sha256": snapshot_sha256,
               "manifest_sha256": manifest_sha256}
    changed: list[str] = []
    for name, value in current.items():
        if recorded.get(name) == value:
            continue
        if name == "config" and isinstance(recorded.get(name), dict):
            changed.extend(f"config.{key}" for key in sorted(value) if recorded[name].get(key) != value[key])
        else:
            changed.append(f"{name} {recorded.get(name)} -> {value}")
    if changed:
        msg = f"{path} is from a different run: {'; '.join(changed)}"
        raise ValueError(msg)


def _write_path(out_dir: Path, spec: PathSpec, content: dict, rows: list[dict]) -> None:
    """Write one path's rows, then its summary last: the summary's presence marks it finished."""
    folder = out_dir / spec.key
    folder.mkdir(parents=True, exist_ok=True)
    write_rows(folder / "rows.csv.gz", rows)
    (folder / "path.json").write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8", newline="\n")


def _write_run(out_dir: Path, config: ExperimentConfig, n: int, digest: str, manifest_digest: str,
               specs: Sequence[PathSpec], seconds: float) -> None:
    """Write `run.json`: what produced this output, over which paths, and the wall time."""
    report = {"config": config.to_json(), "N": n, "snapshot_sha256": digest,
              "manifest_sha256": manifest_digest, "commit": commit(Path(__file__).parent),
              "paths": [spec.key for spec in specs], "seconds": seconds}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")


def commit(package_dir: Path) -> str:
    """`git rev-parse HEAD` in `package_dir`, marked `-dirty` when the tree has changes.

    Args:
        package_dir: The directory whose git tree is inspected.

    Returns:
        The HEAD hash, with `-dirty` appended when the tree has uncommitted changes.
    """
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=package_dir, check=True, capture_output=True,
                          text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=package_dir, check=True, capture_output=True,
                            text=True).stdout
    return head + ("-dirty" if status else "")
