"""The data contract: frozen dataclasses with one generic JSON codec.

JSON conventions: `Amount` fields are decimal strings in the asset's smallest unit (safe for
`BigInt`), other ints are numbers, nested records are objects and tuples are arrays.  Decoding checks
every field's type and rejects unknown keys; a field with a dataclass default may be omitted, and
`to_json` omits it when its value equals that default.
"""

from __future__ import annotations

import hashlib
import json
import types
from dataclasses import MISSING, dataclass, fields, is_dataclass
from fractions import Fraction
from typing import Annotated, Any, Self, Union, get_args, get_origin, get_type_hints

#: An integer amount in the asset's smallest unit, written as a decimal string. Amount fields
#: such as `value`, `d`, `threshold`, `quantum`, `step` and `logit_units` are counted in this unit.
Amount = Annotated[int, "decimal string"]


class Ratio(str):
    """An exact fraction kept as its JSON text, such as "422/740" or "0.001".

    Decoding validates the text once and keeps it verbatim, so a non-string, unparsable or
    zero-divisor value is rejected at the boundary and the surviving text is never reduced.  `value`
    reparses the text with `Fraction` on each access rather than caching the result.
    """

    @property
    def value(self) -> Fraction:
        """The exact rational the text names, reparsed from the text on each access."""
        return Fraction(self)


# ==== codec ====


class SchemaError(TypeError):
    """A JSON value does not have the shape its field declares (for a record: its key set)."""

    def __init__(self, expected: str, value: object) -> None:
        super().__init__(f"expected {expected}, got {value!r}")


class PinError(ValueError):
    """Fetched or built data differs from what the manifest pins."""

    def __init__(self, what: str, got: object, pinned: object) -> None:
        super().__init__(f"{what} is {got}, manifest pins {pinned}")


#: The label :class:`SchemaError` reports for a field whose text is not an exact fraction.
_FRACTION = "fraction string"


def _decode(tp: object, value: object) -> object:
    if tp is Ratio:
        if type(value) is not str:
            raise SchemaError(_FRACTION, value)
        try:
            Fraction(value)
        except (ValueError, ZeroDivisionError) as error:
            raise SchemaError(_FRACTION, value) from error
        return Ratio(value)
    origin = get_origin(tp)
    if origin is Annotated:
        if type(value) is not str or not value.isdigit():
            raise SchemaError("Amount", value)
        return int(value)
    if origin is tuple:
        return tuple(_decode(get_args(tp)[0], item) for item in value)
    if origin in (Union, types.UnionType):
        options = get_args(tp)
        if value is None and type(None) in options:
            return None
        return _decode(next(o for o in options if o is not type(None)), value)
    if is_dataclass(tp):
        return tp.from_json(value)
    if type(value) is not tp:
        raise SchemaError(tp.__name__, value)
    return value


def _encode(value: object, tp: object) -> object:
    if isinstance(value, Record):
        return value.to_json()
    if get_origin(tp) is Annotated:
        return str(value)
    if isinstance(value, tuple):
        return [_encode(item, get_args(tp)[0]) for item in value]
    return value


@dataclass(frozen=True)
class Record:
    """Base of every schema type: `from_json`/`to_json` driven by the field annotations."""

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Self:
        """Decode a JSON object into `cls`, checking every field's shape.

        Args:
            data: The JSON object, one key per field.

        Returns:
            The decoded record.

        Raises:
            SchemaError: If `data`'s key set differs from `cls`'s fields (an unknown key or a
                missing required field), or a value has the wrong type.
        """
        hints = get_type_hints(cls, include_extras=True)
        all_fields = fields(cls)
        names = {f.name for f in all_fields}
        required = {f.name for f in all_fields if f.default is MISSING and f.default_factory is MISSING}
        if set(data) - names or required - set(data):
            raise SchemaError(cls.__name__, sorted(data))
        return cls(**{name: _decode(hints[name], value) for name, value in data.items()})

    def to_json(self) -> dict[str, Any]:
        """Encode the record, omitting fields left at their dataclass default.

        Returns:
            The JSON object, one key per non-default field.
        """
        hints = get_type_hints(type(self), include_extras=True)
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.default is not MISSING and value == f.default:
                continue
            out[f.name] = _encode(value, hints[f.name])
        return out


def canonical(record: Record) -> bytes:
    """The byte form `sha256` hashes: the record's JSON with sorted keys and no whitespace.

    Args:
        record: The record to serialize.

    Returns:
        The UTF-8 encoded canonical JSON.
    """
    return json.dumps(record.to_json(), sort_keys=True, separators=(",", ":")).encode()


def sha256(record: Record) -> str:
    """The hex digest of the canonical bytes, the value a manifest pins as `snapshot_sha256`.

    Args:
        record: The record to hash.

    Returns:
        The SHA-256 hex digest of `canonical`(record).
    """
    return hashlib.sha256(canonical(record)).hexdigest()


# ==== chain data ====


@dataclass(frozen=True)
class Manifest(Record):
    """One pool at one pinned block: the definition of a dataset.

    `chain_id` identifies the chain; `entrypoint` and `pool` name the contracts to query and `asset`
    the token; `decimals` is that token's decimal count; `d` is the rounding unit in the asset's
    smallest unit; `start_block`/`end_block` bound the extraction and `end_block_hash` pins the chain
    history up to `end_block`; `snapshot_sha256` pins the extracted content and is `None` until the
    first build.
    """

    chain_id: int
    entrypoint: str
    pool: str
    asset: str
    decimals: int
    d: Amount
    start_block: int
    end_block: int
    end_block_hash: str
    snapshot_sha256: str | None


@dataclass(frozen=True)
class Deposit(Record):
    """A `Deposited` log: `depositor` deposited `value` at `(block, log_index)` in transaction `tx`.

    `commitment`, `label` and `precommitment_hash` are the pool's fields for the deposit, each a big
    integer rendered as decimal text.
    """

    block: int
    log_index: int
    tx: str
    depositor: str
    commitment: str
    label: str
    value: Amount
    precommitment_hash: str


@dataclass(frozen=True)
class Withdrawal(Record):
    """`gross` is the protocol amount the observer sees; `fee` is the relay fee taken out of it
    (`0` and `relayer` `None` for a direct withdrawal)."""

    block: int
    log_index: int
    tx: str
    gross: Amount
    fee: Amount
    recipient: str
    relayer: str | None
    spent_nullifier: str
    new_commitment: str


@dataclass(frozen=True)
class Ragequit(Record):
    """A `Ragequit` log: `ragequitter` exited with `value` at `(block, log_index)` in transaction `tx`.

    `commitment` and `label` identify the exited commitment.
    """

    block: int
    log_index: int
    tx: str
    ragequitter: str
    commitment: str
    label: str
    value: Amount


@dataclass(frozen=True)
class RootUpdate(Record):
    """An ASP root and the IPFS CID of its label list."""

    block: int
    log_index: int
    tx: str
    root: str
    cid: str


@dataclass(frozen=True)
class Snapshot(Record):
    """Every pool event in the manifest's block range, each list ordered by `(block, log_index)`."""

    deposits: tuple[Deposit, ...]
    withdrawals: tuple[Withdrawal, ...]
    ragequits: tuple[Ragequit, ...]
    roots: tuple[RootUpdate, ...]


@dataclass(frozen=True)
class ThetaSpec(Record):
    """The historical model: `pi`, `s`, `w` as :class:`Ratio` fraction strings, the demand version
    `g`, the historical tail `unit` of `s`, and the `source` of the numbers (see :mod:`lab`)."""

    pi: Ratio
    s: Ratio
    w: Ratio
    g: str
    unit: Amount
    source: str


@dataclass(frozen=True)
class LogitTrainSpec(Record):
    """The `trained-logit` observer's training: `paths` simulated paths, the L2 penalty `l2`, and the
    ranges each path draws its historical model from: `pi` uniform in `[pi_lo, pi_hi]`, `s` in
    `[0, s_hi]`, `w` in `[0, w_hi]`; `source` says where the ranges come from."""

    paths: int
    l2: Ratio
    pi_lo: Ratio
    pi_hi: Ratio
    s_hi: Ratio
    w_hi: Ratio
    source: str


@dataclass(frozen=True)
class ExperimentConfig(Record):
    """One reproducible experiment run: the snapshot it reads and every axis it sweeps.

    `manifest` is a path to the manifest (relative to the config file) that pins the snapshot's
    sha256, which `snapshot_sha256` records.  `theta` is the historical model.  `split_block` divides
    the withdrawals into the training set (strictly below) and the holdout set (at or above).
    `replicates` is the number of independent draws of adoption and path per grid cell.  `rhos` are
    adoption ratios written as :class:`Ratio` fractions ("1/10"); the number of tool users is
    `K = floor(rho*N + 1/2)`, where `N` is the participant count the snapshot determines.  `rules`
    are the rounding rules swept, `m_models` the maximum models (`double`, `balance`), `adopters` the
    adopter kinds, and `observers` the observers scored.  `threshold` is the tool-use threshold;
    `quantum` is the adopter need grid; `step` is the demand grid; `historical` names the no-tool
    behaviour model; `logit_units` are the logit tail-feature units of the `trained-logit` observer;
    `logit_train` is the spec that trains its coefficients.  `combinatorial_tolerance` is the
    :class:`Ratio` fraction of a deposit's balance the `combinatorial` observer may leave unlinked.
    `seed_root` prefixes every derived seed, so two runs with different roots share no random draws.
    `p` is the probability of adding the random decimal, written as a :class:`Ratio` and passed to
    every rounding rule (`need` is unaffected).
    """

    manifest: str
    snapshot_sha256: str
    split_block: int
    theta: ThetaSpec
    replicates: int
    rhos: tuple[Ratio, ...]
    rules: tuple[str, ...]
    m_models: tuple[str, ...]
    adopters: tuple[str, ...]
    observers: tuple[str, ...]
    combinatorial_tolerance: Ratio
    threshold: Amount
    quantum: Amount
    step: Amount
    historical: str
    logit_units: tuple[Amount, ...]
    logit_train: LogitTrainSpec
    seed_root: str
    p: Ratio = Ratio("0")
