"""Study inputs: pinned snapshot loading, split, demand fitting, participant selection, behaviour.

The train/holdout split is a block number: withdrawals strictly below `split_block` train the
ordinary-user demand model, those at or above it are the holdout.  Participants are every positive
deposit before `split_block`, each used once.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from lab.behaviour import Demand, GridHistorical, Historical, Theta
from lab.rules import DecimalLadder, Need, PopularLadder, model, popular_counts
from lab.schema import ExperimentConfig, Manifest, PinError, Snapshot, sha256

if TYPE_CHECKING:
    from pathlib import Path

    from lab.schema import ThetaSpec

__all__ = ["HISTORICAL", "RULE_LABELS", "StudyInputs", "behaviour", "inputs_from", "load", "rule_object", "theta_of"]

#: Historical model builders, by the `historical` label of a spec.
HISTORICAL: dict[str, object] = {"plain": Historical, "grid": GridHistorical}

#: Rule labels of the published grids: rung source and climb probability.
RULE_LABELS: dict[str, tuple[str, Fraction | None]] = {
    "need": ("need", None),
    "decstep": ("decimal", Fraction(1)),
    "decsmooth": ("decimal", Fraction(1, 2)),
    "popstep": ("popular", Fraction(1)),
    "popsmooth": ("popular", Fraction(1, 2)),
}


@dataclass(frozen=True)
class StudyInputs:
    """Everything one experiment reads: its config and what the pinned snapshot determines.

    `d` is the manifest's rounding unit, the single source of the rounding unit.  `balances`
    are the participants' balances before the split, in `(block, log_index)` order; `demands` maps
    the version `hi`/`lo` to its demand; `train_gross` are the training withdrawals' grosses.
    """

    config: ExperimentConfig
    snapshot_sha256: str
    d: int
    balances: tuple[int, ...]
    demands: dict[str, Demand]
    train_gross: tuple[int, ...]


def load(config_path: Path, data: Path) -> StudyInputs:
    """Load a config, its manifest and the pinned snapshot under `data` into the study inputs.

    The manifest path is relative to the config file; `data` holds `<manifest stem>/snapshot.json`.

    Args:
        config_path: The experiment config JSON file.
        data: The data directory holding `<manifest stem>/snapshot.json`.

    Returns:
        The study inputs of the pinned snapshot.

    Raises:
        lab.schema.PinError: The snapshot or the manifest hash does not match the config's pin.
    """
    config = ExperimentConfig.from_json(json.loads(config_path.read_text(encoding="utf-8")))
    manifest_path = config_path.parent / config.manifest
    manifest = Manifest.from_json(json.loads(manifest_path.read_text(encoding="utf-8")))
    snapshot = Snapshot.from_json(json.loads((data / manifest_path.stem / "snapshot.json")
                                             .read_text(encoding="utf-8")))
    digest = sha256(snapshot)
    if digest != config.snapshot_sha256:
        raise PinError("snapshot_sha256", digest, config.snapshot_sha256)
    if manifest.snapshot_sha256 != config.snapshot_sha256:
        raise PinError("manifest_snapshot_sha256", manifest.snapshot_sha256, config.snapshot_sha256)
    return inputs_from(config, manifest, snapshot, digest)


def inputs_from(config: ExperimentConfig, manifest: Manifest, snapshot: Snapshot, digest: str) -> StudyInputs:
    """The study inputs of a pinned snapshot: participants, demands and training withdrawals.

    Training withdrawals are the positive-gross ones strictly below `config.split_block`; the demand
    counts the deposits before the last of them (for the `lo` removal and `max_deposit`).
    Participants are every positive deposit before the split, each used once, in `(block, log_index)`
    order.

    Args:
        config: The experiment config.
        manifest: The manifest whose rounding unit `d` the inputs carry.
        snapshot: The pinned snapshot.
        digest: The snapshot's sha256, stored on the inputs.

    Returns:
        The study inputs, with `d` taken from `manifest.d`.
    """
    positive = sorted((w for w in snapshot.withdrawals if int(w.gross) > 0),
                      key=lambda w: (w.block, w.log_index))
    train = [w for w in positive if w.block < config.split_block]
    cutoff = (train[-1].block, train[-1].log_index)
    prior = [int(d.value) for d in snapshot.deposits if (d.block, d.log_index) < cutoff]
    train_gross = [int(w.gross) for w in train]
    demands = {g: Demand.build(train_gross, prior, g, config.step, max(prior)) for g in ("hi", "lo")}
    participants = sorted((d for d in snapshot.deposits if int(d.value) > 0 and d.block < config.split_block),
                          key=lambda d: (d.block, d.log_index))
    return StudyInputs(config=config, snapshot_sha256=digest, d=manifest.d,
                       balances=tuple(int(d.value) for d in participants), demands=demands,
                       train_gross=tuple(train_gross))


def theta_of(spec: ThetaSpec) -> Theta:
    """The behaviour model `Theta` for a spec, with each fraction string read exactly.

    Args:
        spec: The theta spec; its `pi`, `s` and `w` are exact fraction strings.

    Returns:
        The `Theta`, with the three shares as floats of the exact fractions and `g` unchanged.
    """
    return Theta(float(Fraction(spec.pi)), float(Fraction(spec.s)), float(Fraction(spec.w)), spec.g)


def rule_object(label: str, d: int, train_gross: tuple[int, ...],
                p: Fraction = Fraction(0)) -> Need | DecimalLadder | PopularLadder:
    """The tool rule named by `label`.

    Args:
        label: A key of :data:`RULE_LABELS`, e.g. `need` or `decsmooth`.
        d: The rounding unit (`manifest.d`); the ladder's rungs are multiples of it.
        train_gross: The training withdrawals' grosses, the popular ladder's counts.
        p: The probability of adding the random decimal below `d`; the need rule ignores it.

    Returns:
        A `Need`, `DecimalLadder` or `PopularLadder` per the label's rung source.
    """
    source, q = RULE_LABELS[label]
    if source == "need":
        return Need()
    if source == "decimal":
        return DecimalLadder(d, q, p)
    return PopularLadder(*popular_counts(train_gross), d, q, p)


def behaviour(inputs: StudyInputs, theta: Theta, rule: str, m_model: str, *, p: Fraction) -> tuple:
    """The `(historical, keep, drain)` behaviour tuple of one path of the study.

    Args:
        inputs: The study inputs.
        theta: The historical model's parameters.
        rule: The tool rule label.
        m_model: The maximum model, e.g. `double` or `balance`.
        p: The probability of adding the random decimal; passed to the tool rule.

    Returns:
        The `(historical, keep, drain)` tuple the simulator walks.
    """
    config = inputs.config
    historical = HISTORICAL[config.historical](theta, inputs.demands[theta.g], config.theta.unit)
    tool = rule_object(rule, inputs.d, inputs.train_gross, p)
    return model(historical, tool, config.threshold, config.quantum, inputs.d, m_model)
