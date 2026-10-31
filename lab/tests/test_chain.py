from __future__ import annotations

import json

import pytest

from lab import chain
from lab.schema import Deposit, Manifest, Snapshot, Withdrawal

POOL = "0x" + "1a" * 20
ENTRYPOINT = "0x" + "2b" * 20
ASSET = "0x" + "3c" * 20
MANIFEST = Manifest(1, ENTRYPOINT, POOL, ASSET, 6, 10**6, 1, 9, "0x" + "00" * 32, None)


def _topic(address: str) -> str:
    return "0x" + "00" * 12 + address[2:]


def _log(address: str, topics: list[str], words: list[int], block: int, index: int, tx: str) -> dict:
    return {"address": address, "topics": topics, "data": "0x" + "".join(f"{w:064x}" for w in words),
            "blockNumber": hex(block), "logIndex": hex(index), "transactionHash": tx}


def _checksummed(address: str) -> str:
    """A mixed-case spelling of a lower-case address, the way a checksummed address is written."""
    return "0x" + "".join(character.upper() if index % 2 else character
                          for index, character in enumerate(address[2:]))


class _Reply:
    """The subset of `http.client.HTTPResponse` that `urllib.request.urlopen` is used for here."""

    def __init__(self, payload: object) -> None:
        self._body = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _Reply:
        return self

    def __exit__(self, *info: object) -> None:
        return None


def _answer_with(monkeypatch: pytest.MonkeyPatch, payload: object) -> None:
    monkeypatch.setattr(chain.urllib.request, "urlopen", lambda request, timeout: _Reply(payload))


# ==== JSON-RPC ====


@pytest.mark.parametrize("payload", [
    {"jsonrpc": "2.0", "id": 1, "result": None},
    {"jsonrpc": "2.0", "id": 1},
])
def test_a_reply_without_a_usable_result_raises_the_rpc_error(monkeypatch: pytest.MonkeyPatch,
                                                              payload: object) -> None:
    # GIVEN: A node answering a null result, as it does for an unknown block, or no result at all.
    _answer_with(monkeypatch, payload)
    # WHEN: Calling it.
    # THEN: The readable RPC error is raised instead of a TypeError or a KeyError.
    with pytest.raises(chain.RpcError):
        chain.call("http://rpc.invalid", "eth_getBlockByNumber", ["0x1", False])


def test_fetch_reports_an_unknown_end_block_as_an_rpc_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # GIVEN: A node that knows no block at the manifest's end block.
    _answer_with(monkeypatch, {"jsonrpc": "2.0", "id": 1, "result": None})
    # WHEN: Fetching the manifest's logs.
    # THEN: The readable RPC error is raised, not a TypeError about a None block.
    with pytest.raises(chain.RpcError):
        chain.fetch("http://rpc.invalid", MANIFEST)


def test_fetch_queries_the_lower_cased_contract_addresses(monkeypatch: pytest.MonkeyPatch) -> None:
    # GIVEN: A mixed-case manifest and a recording node.
    mixed = Manifest(1, _checksummed(ENTRYPOINT), _checksummed(POOL), _checksummed(ASSET), 6, 10**6, 1, 9,
                     "0x" + "00" * 32, None)
    filters: list[dict] = []

    def record(rpc: str, method: str, params: list) -> object:
        if method == "eth_getBlockByNumber":
            return {"hash": mixed.end_block_hash}
        filters.append(params[0])
        return []

    monkeypatch.setattr(chain, "call", record)
    # WHEN: Fetching the manifest's logs.
    chain.fetch("http://rpc.invalid", mixed)
    # THEN: The filter names the contracts in lower case, the form the node matches.
    assert filters[0]["address"] == [POOL, ENTRYPOINT]


# ==== schema ====


def test_withdrawal_round_trips_with_amounts_as_decimal_strings() -> None:
    # GIVEN: A relayed withdrawal.
    w = Withdrawal(5, 0, "0xab", 420_000_000, 7_686_000, POOL, ENTRYPOINT, "1", "2")
    # WHEN: Encoding and decoding it.
    data = w.to_json()
    # THEN: Amounts travel as strings, safe for BigInt in JavaScript, and decode back unchanged.
    assert (data["gross"], Withdrawal.from_json(data)) == ("420000000", w)


@pytest.mark.parametrize(("key", "value"), [("gross", 420), ("gross", "4.2"), ("extra", 1)])
def test_malformed_json_is_rejected_at_load(key: str, value: object) -> None:
    # GIVEN: A withdrawal with a numeric or fractional amount, or an unknown field.
    data = {**Withdrawal(5, 0, "0xab", 1, 0, POOL, None, "1", "2").to_json(), key: value}
    # WHEN: Loading it.
    # THEN: The boundary check refuses it, so later code never sees bad data.
    with pytest.raises(TypeError):
        Withdrawal.from_json(data)


# ==== decoding ====


def test_relayed_withdrawal_takes_fee_and_recipient_from_its_relay_event() -> None:
    # GIVEN: One direct and one relayed withdrawal, the relay event emitted after its Withdrawn log,
    # plus a relay of another asset in the same transaction.
    user, relayer, other = "0x" + "44" * 20, "0x" + "55" * 20, "0x" + "66" * 20
    logs = [
        _log(POOL, [chain.WITHDRAWN, _topic(user)], [100, 1, 2], 3, 0, "0xa"),
        _log(POOL, [chain.WITHDRAWN, _topic(ENTRYPOINT)], [200, 3, 4], 4, 0, "0xb"),
        _log(ENTRYPOINT, [chain.WITHDRAWAL_RELAYED, _topic(relayer), _topic(other), _topic(user)], [9, 9], 4, 1, "0xb"),
        _log(ENTRYPOINT, [chain.WITHDRAWAL_RELAYED, _topic(relayer), _topic(user), _topic(ASSET)], [2, 7], 4, 2, "0xb"),
    ]
    # WHEN: Building the snapshot.
    snap = chain.build(MANIFEST, logs)
    # THEN: The direct one has no fee, and the relayed one pairs with this asset's relay only.
    assert snap == Snapshot((), (
        Withdrawal(3, 0, "0xa", 100, 0, user, None, "1", "2"),
        Withdrawal(4, 0, "0xb", 200, 7, user, relayer, "3", "4"),
    ), (), ())


def test_build_matches_a_mixed_case_manifest_against_lower_case_logs() -> None:
    # GIVEN: A manifest naming its pool, entrypoint and asset in mixed case, and logs carrying them in
    # lower case, as nodes return them.
    mixed = Manifest(1, _checksummed(ENTRYPOINT), _checksummed(POOL), _checksummed(ASSET), 6, 10**6, 1, 9,
                     "0x" + "00" * 32, None)
    user, relayer = "0x" + "44" * 20, "0x" + "55" * 20
    logs = [
        _log(POOL, [chain.DEPOSITED, _topic(user)], [11, 12, 500, 13], 2, 0, "0xa"),
        _log(POOL, [chain.WITHDRAWN, _topic(ENTRYPOINT)], [200, 3, 4], 3, 0, "0xb"),
        _log(ENTRYPOINT, [chain.WITHDRAWAL_RELAYED, _topic(relayer), _topic(user), _topic(ASSET)],
             [1, 7], 3, 1, "0xb"),
    ]
    # WHEN: Building the snapshot.
    snap = chain.build(mixed, logs)
    # THEN: Nothing is dropped: the deposit and the relayed withdrawal with its fee and asset are kept.
    assert snap.deposits == (Deposit(2, 0, "0xa", user, "11", "12", 500, "13"),)
    assert snap.withdrawals == (Withdrawal(3, 0, "0xb", 200, 7, user, relayer, "3", "4"),)
