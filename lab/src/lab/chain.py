"""Manifest -> raw logs (JSON-RPC) -> snapshot.

Privacy Pools v1 events: the pool emits `Deposited`, `Withdrawn` and `Ragequit`; the
Entrypoint, shared by every pool, emits `RootUpdated` and `WithdrawalRelayed`.  A relayed
`Withdrawn` names the Entrypoint in its `processooor` field; within one transaction the relayed
`Withdrawn` logs and this asset's `WithdrawalRelayed` logs appear in the same order.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

from lab.schema import Deposit, Manifest, PinError, Ragequit, RootUpdate, Snapshot, Withdrawal

DEPOSITED = "0xe3b53cd1a44fbf11535e145d80b8ef1ed6d57a73bf5daa7e939b6b01657d6549"
WITHDRAWN = "0x75e161b3e824b114fc1a33274bd7091918dd4e639cede50b78b15a4eea956a21"
RAGEQUIT = "0xd2b3e868ae101106371f2bd93abc8d5a4de488b9fe47ed122c23625aa7172f13"
ROOT_UPDATED = "0xf43641373d5bc20fddfffcd0496fd39bde41e1aa0ce326d8899b7e658261ac02"
WITHDRAWAL_RELAYED = "0xe9b67844a7bb6e6ac95e8a0de02e4448dbb0c9460be9194348e4bbac6d13c2cf"

# ==== JSON-RPC ====


class RpcError(RuntimeError):
    """A JSON-RPC error object returned by the node."""


type Json = dict[str, Json] | list[Json] | str | int | float | bool | None


def call(rpc: str, method: str, params: list[Json]) -> Json:
    """One JSON-RPC request to `rpc`, returning its reply's `result`.

    Args:
        rpc: The JSON-RPC endpoint URL.
        method: The RPC method, e.g. `eth_getLogs`.
        params: The method's positional parameters.

    Returns:
        The reply's `result`, which is never `None`: a null result is an error.

    Raises:
        RpcError: The node answered with an `error` object, or with no result at all, as it does for
            an unknown block.
    """
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    request = urllib.request.Request(rpc, data=body, headers={"Content-Type": "application/json", "User-Agent": "lab"})
    with urllib.request.urlopen(request, timeout=120) as response:
        reply = json.loads(response.read())
    if "error" in reply:
        raise RpcError(reply["error"])
    if reply.get("result") is None:
        msg = f"{method} returned no result"
        raise RpcError(msg)
    return reply["result"]


def _logs(rpc: str, flt: dict[str, Any], lo: int, hi: int) -> list[dict[str, Any]]:
    """Every log in `[lo, hi]`; a failing range is split in half until one block still fails."""
    try:
        return call(rpc, "eth_getLogs", [{**flt, "fromBlock": hex(lo), "toBlock": hex(hi)}])
    except (RpcError, OSError):
        if lo == hi:
            raise
    mid = (lo + hi) // 2
    return _logs(rpc, flt, lo, mid) + _logs(rpc, flt, mid + 1, hi)


def fetch(rpc: str, manifest: Manifest) -> list[dict[str, Any]]:
    """Raw logs of the five events over the manifest's range, after checking the end block hash.

    Args:
        rpc: The JSON-RPC endpoint URL.
        manifest: The manifest naming the pool, entrypoint, asset and block range.

    Returns:
        Every matching log, unordered; `build` orders and decodes them.

    Raises:
        lab.schema.PinError: The end block's hash does not match the manifest.
    """
    head = call(rpc, "eth_getBlockByNumber", [hex(manifest.end_block), False])
    if head["hash"] != manifest.end_block_hash:
        raise PinError("end_block_hash", head["hash"], manifest.end_block_hash)
    pool, entrypoint = manifest.pool.lower(), manifest.entrypoint.lower()
    flt = {
        "address": [pool, entrypoint],
        "topics": [[DEPOSITED, WITHDRAWN, RAGEQUIT, ROOT_UPDATED, WITHDRAWAL_RELAYED]],
    }
    return _logs(rpc, flt, manifest.start_block, manifest.end_block)


# ==== decoding ====


def _words(data: str) -> list[int]:
    return [int(data[i : i + 64], 16) for i in range(2, len(data), 64)]


def _address(topic: str) -> str:
    return "0x" + topic[-40:]


def _cid(data: str) -> str:
    body = bytes.fromhex(data[2:])
    offset = int.from_bytes(body[32:64], "big")
    length = int.from_bytes(body[offset : offset + 32], "big")
    return body[offset + 32 : offset + 32 + length].decode()


def build(manifest: Manifest, logs: list[dict[str, Any]]) -> Snapshot:
    """Decode raw logs into a snapshot, joining relayed withdrawals with their relay fees.

    Args:
        manifest: The manifest naming the pool, entrypoint and asset to match logs against.
        logs: Raw logs, in any order.

    Returns:
        The snapshot of deposits, withdrawals, ragequits and root updates; a relayed withdrawal
        takes its fee and recipient from this asset's relay event in the same transaction.
    """
    logs = sorted(logs, key=lambda log: (int(log["blockNumber"], 16), int(log["logIndex"], 16)))
    # Manifest addresses may be written checksummed; the decoded logs are lower case.
    pool, entrypoint, asset = manifest.pool.lower(), manifest.entrypoint.lower(), manifest.asset.lower()
    deposits, withdrawals, ragequits, roots = [], [], [], []
    relays: dict[str, list[dict[str, Any]]] = {}
    pending: list[tuple[dict[str, Any], list[int]]] = []
    for log in logs:
        topics = [t.lower() for t in log["topics"]]
        source = log["address"].lower()
        words = _words(log["data"])
        at = {"block": int(log["blockNumber"], 16), "log_index": int(log["logIndex"], 16),
              "tx": log["transactionHash"].lower()}
        if source == pool and topics[0] == DEPOSITED:
            deposits.append(Deposit(**at, depositor=_address(topics[1]), commitment=str(words[0]), label=str(words[1]),
                                    value=words[2], precommitment_hash=str(words[3])))
        elif source == pool and topics[0] == RAGEQUIT:
            ragequits.append(Ragequit(**at, ragequitter=_address(topics[1]), commitment=str(words[0]),
                                      label=str(words[1]), value=words[2]))
        elif source == pool and topics[0] == WITHDRAWN:
            pending.append(({**at, "processooor": _address(topics[1])}, words))
        elif source == entrypoint and topics[0] == ROOT_UPDATED:
            roots.append(RootUpdate(**at, root=str(words[0]), cid=_cid(log["data"])))
        elif (source == entrypoint and topics[0] == WITHDRAWAL_RELAYED
              and _address(topics[3]) == asset):
            relays.setdefault(at["tx"], []).append(
                {"relayer": _address(topics[1]), "recipient": _address(topics[2]), "fee": words[1]})
    for at, words in pending:
        processooor = at.pop("processooor")
        relay = relays[at["tx"]].pop(0) if processooor == entrypoint else None
        withdrawals.append(Withdrawal(
            **at, gross=words[0], fee=relay["fee"] if relay else 0,
            recipient=relay["recipient"] if relay else processooor, relayer=relay["relayer"] if relay else None,
            spent_nullifier=str(words[1]), new_commitment=str(words[2]),
        ))
    return Snapshot(tuple(deposits), tuple(withdrawals), tuple(ragequits), tuple(roots))
