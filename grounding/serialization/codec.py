"""Indexed relation blocks and canonical spelling for serializer version 2 (A2.1d, D69).

The layout is the one approved in the indexed audit (Second_Eyes_A2_1d_Indexed_Audit_Brief.md, sections 3-4), copied
here rather than imported from analysis/. Cells are keyed by index tuples into the ordered objects array:
objects (t,), ordered_pairs (target, anchor), unordered_pairs (i, j) with i < j, target_anchor_pairs (t, a, b) with
a < b. Decoding is strict: besides cell shapes it enforces the canonical forms, so a uniform domain written as full
rows, an empty domain written with anything but [] and false, or a uniform conditional written as a string is an error.
"""
from __future__ import annotations

import json
import math

DOMAINS = ("objects", "unordered_pairs", "ordered_pairs", "target_anchor_pairs")
TRUTH = "TFU"
BITS = "01"


class CodecError(ValueError):
    """A block that is malformed or not in canonical form."""


# --------------------------------------------------------------------------------------------- canonical spelling
def number(x) -> str:
    """Integers in decimal; integral finite floats as the equal integer (both zeros as 0); other floats by repr."""
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if not isinstance(x, float) or not math.isfinite(x):
        raise ValueError(f"{x!r} is not a finite number")
    if x == 0:
        return "0"
    if x.is_integer():
        return str(int(x))
    return repr(x)


def enc(value) -> str:
    """Compact JSON, keys in insertion order, UTF-8 text (ensure_ascii=False), section 5 number spelling."""
    if value is None:
        return "null"
    if isinstance(value, (bool, int, float)):
        return number(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(enc(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(json.dumps(k, ensure_ascii=False) + ":" + enc(v) for k, v in value.items()) + "}"
    raise TypeError(f"can't spell {type(value).__name__}")


# ------------------------------------------------------------------------------------------------------ domains
def legal_cells(domain: str, n: int) -> list:
    """Every legal index tuple of a domain, in layout order."""
    if domain == "objects":
        return [(t,) for t in range(n)]
    if domain == "unordered_pairs":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    if domain == "ordered_pairs":
        return [(i, j) for i in range(n) for j in range(n) if i != j]
    if domain == "target_anchor_pairs":
        return [(t, a, b) for a in range(n) for b in range(a + 1, n) for t in range(n) if t not in (a, b)]
    raise CodecError(f"unknown domain {domain!r}")


def legal_count(domain: str, n: int) -> int:
    return {"objects": n, "unordered_pairs": n * (n - 1) // 2, "ordered_pairs": n * (n - 1),
            "target_anchor_pairs": n * (n - 1) * (n - 2) // 2}[domain]


# ------------------------------------------------------------------------------------------------------ encoding
def _layout(domain: str, n: int, symbol: dict):
    """symbol: {index tuple: one character} over the whole domain -> a uniform character or the full rows."""
    cells = legal_cells(domain, n)
    first = symbol[cells[0]]
    if all(symbol[c] == first for c in cells):
        return first
    if domain == "objects":
        return "".join(symbol[(t,)] for t in range(n))
    if domain == "ordered_pairs":
        return ["".join("-" if i == j else symbol[(i, j)] for j in range(n)) for i in range(n)]
    if domain == "unordered_pairs":
        return ["".join(symbol[(i, j)] for j in range(i + 1, n)) for i in range(n - 1)]
    return [[a, b, "".join("-" if t in (a, b) else symbol[(t, a, b)] for t in range(n))]
            for a in range(n) for b in range(a + 1, n)]


def _membership(domain: str, n: int, flags: dict):
    if not flags:
        return False
    bits = _layout(domain, n, {c: "1" if flags[c] else "0" for c in flags})
    return (bits == "1") if bits in ("0", "1") else bits


def encode_block(relation: str, frame, domain: str, n: int, cells: dict) -> dict:
    """cells: {index tuple: (state 'T'/'F'/'U', conditional bool)} covering the domain exactly."""
    if set(cells) != set(legal_cells(domain, n)):
        raise CodecError(f"{relation}: cells don't cover exactly the {domain} domain of {n} objects")
    if any(s not in TRUTH or not isinstance(c, bool) for s, c in cells.values()):
        raise CodecError(f"{relation}: a cell isn't (T/F/U, bool)")
    if not cells:
        return {"relation": relation, "frame": frame, "domain": domain, "states": [], "conditional": False}
    states = _layout(domain, n, {k: s for k, (s, _) in cells.items()})
    return {"relation": relation, "frame": frame, "domain": domain, "states": states,
            "conditional": _membership(domain, n, {k: c for k, (_, c) in cells.items()})}


def encode_distances(n: int, cells: dict) -> dict:
    """cells: {(i, j): (finite number or None, conditional bool)} over every unordered pair."""
    if set(cells) != set(legal_cells("unordered_pairs", n)):
        raise CodecError("distances don't cover every unordered pair")
    for value, _ in cells.values():
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value)):
            raise CodecError(f"distance {value!r} is not a finite number or null")
    values = [[cells[(i, j)][0] for j in range(i + 1, n)] for i in range(n - 1)]
    return {"measure": "center_distance_m", "domain": "unordered_pairs", "values": values,
            "conditional": _membership("unordered_pairs", n, {k: c for k, (_, c) in cells.items()})}


# ------------------------------------------------------------------------------------------------------ decoding
def _rows(domain: str, n: int, value, alphabet: str, what: str) -> dict:
    """A mixed layout -> {index tuple: symbol}, validating every cell and every exclusion mark."""
    cells = {}
    if domain == "objects":
        if not isinstance(value, str) or len(value) != n:
            raise CodecError(f"{what}: objects layout must be one string of {n} symbols")
        for t, ch in enumerate(value):
            if ch not in alphabet:
                raise CodecError(f"{what}: symbol {ch!r} for target {t}")
            cells[(t,)] = ch
        return cells
    if not isinstance(value, list):
        raise CodecError(f"{what}: a mixed layout must be a list of rows")
    if domain == "ordered_pairs":
        if len(value) != n:
            raise CodecError(f"{what}: {len(value)} rows where {n} are due")
        for i, row in enumerate(value):
            if not isinstance(row, str) or len(row) != n:
                raise CodecError(f"{what}: row {i} must be a string of {n} symbols")
            for j, ch in enumerate(row):
                if i == j:
                    if ch != "-":
                        raise CodecError(f"{what}: diagonal cell ({i},{i}) is {ch!r}, not '-'")
                elif ch not in alphabet:
                    raise CodecError(f"{what}: symbol {ch!r} at target {i}, anchor {j}")
                else:
                    cells[(i, j)] = ch
        return cells
    if domain == "unordered_pairs":
        if len(value) != n - 1:
            raise CodecError(f"{what}: {len(value)} rows where {n - 1} are due")
        for i, row in enumerate(value):
            if not isinstance(row, str) or len(row) != n - 1 - i:
                raise CodecError(f"{what}: row {i} must hold {n - 1 - i} symbols")
            for k, ch in enumerate(row):
                if ch not in alphabet:
                    raise CodecError(f"{what}: symbol {ch!r} at row {i}, position {k}")
                cells[(i, i + 1 + k)] = ch
        return cells
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    if len(value) != len(pairs):
        raise CodecError(f"{what}: {len(value)} anchor rows where {len(pairs)} are due")
    for (ea, eb), row in zip(pairs, value):
        if not isinstance(row, list) or len(row) != 3:
            raise CodecError(f"{what}: anchor row {row!r} must be [a,b,row]")
        a, b, s = row
        if not all(isinstance(x, int) and not isinstance(x, bool) for x in (a, b)):
            raise CodecError(f"{what}: anchor indices {a!r}, {b!r} must be integers")
        if not 0 <= a < b < n:
            raise CodecError(f"{what}: anchor indices ({a},{b}) out of range or out of order")
        if (a, b) != (ea, eb):
            raise CodecError(f"{what}: anchor row ({a},{b}) where ({ea},{eb}) is due (duplicate or missing row)")
        if not isinstance(s, str) or len(s) != n:
            raise CodecError(f"{what}: anchor row ({a},{b}) must hold {n} symbols")
        for t, ch in enumerate(s):
            if t in (a, b):
                if ch != "-":
                    raise CodecError(f"{what}: anchor position {t} of row ({a},{b}) is {ch!r}, not '-'")
            elif ch not in alphabet:
                raise CodecError(f"{what}: symbol {ch!r} for target {t} in row ({a},{b})")
            else:
                cells[(t, a, b)] = ch
    return cells


def _decode_states(domain: str, n: int, value, what: str) -> dict:
    if legal_count(domain, n) == 0:
        if value != [] or not isinstance(value, list):
            raise CodecError(f"{what}: an empty domain's states must be []")
        return {}
    if isinstance(value, str) and len(value) == 1:
        if value not in TRUTH:
            raise CodecError(f"{what}: uniform state {value!r} is not T, F or U")
        return {c: value for c in legal_cells(domain, n)}
    cells = _rows(domain, n, value, TRUTH, what)
    if len(set(cells.values())) == 1:
        raise CodecError(f"{what}: full rows for a uniform domain are not canonical; use the single symbol")
    return cells


def _decode_membership(domain: str, n: int, value, what: str) -> dict:
    if legal_count(domain, n) == 0:
        if value is not False:
            raise CodecError(f"{what}: an empty domain's conditional must be false")
        return {}
    if isinstance(value, bool):
        return {c: value for c in legal_cells(domain, n)}
    if isinstance(value, str) and len(value) == 1 and domain != "objects":
        raise CodecError(f"{what}: a uniform conditional must be a JSON Boolean")
    cells = _rows(domain, n, value, BITS, what)
    if len(set(cells.values())) == 1:
        raise CodecError(f"{what}: a uniform conditional must be a JSON Boolean, not full rows")
    return {k: v == "1" for k, v in cells.items()}


def decode_block(block: dict, n: int) -> dict:
    """A relation block -> {index tuple: (state, conditional)}; raises CodecError unless valid and canonical."""
    if not isinstance(block, dict) or list(block) != ["relation", "frame", "domain", "states", "conditional"]:
        raise CodecError(f"relation block fields {list(block) if isinstance(block, dict) else block!r}")
    domain, what = block["domain"], f"{block['relation']}/{block['frame']}"
    if domain not in DOMAINS:
        raise CodecError(f"{what}: unknown domain {domain!r}")
    states = _decode_states(domain, n, block["states"], what)
    membership = _decode_membership(domain, n, block["conditional"], what)
    return {c: (states[c], membership[c]) for c in legal_cells(domain, n)}


def decode_distances(block: dict, n: int) -> dict:
    """A distance block -> {(i, j): (number or None, conditional)}; raises CodecError unless valid and canonical."""
    if not isinstance(block, dict) or list(block) != ["measure", "domain", "values", "conditional"] \
            or block["measure"] != "center_distance_m" or block["domain"] != "unordered_pairs":
        raise CodecError("distance block fields or names")
    values = block["values"]
    if not isinstance(values, list) or len(values) != max(n - 1, 0):
        raise CodecError(f"distances: {max(n - 1, 0)} rows are due")
    out = {}
    for i, row in enumerate(values):
        if not isinstance(row, list) or len(row) != n - 1 - i:
            raise CodecError(f"distances: row {i} must hold {n - 1 - i} values")
        for k, d in enumerate(row):
            if d is not None and (isinstance(d, bool) or not isinstance(d, (int, float)) or not math.isfinite(d)):
                raise CodecError(f"distances: {d!r} at row {i}, position {k} is not a finite number or null")
            out[(i, i + 1 + k)] = d
    membership = _decode_membership("unordered_pairs", n, block["conditional"], "distances")
    return {c: (out[c], membership[c]) for c in legal_cells("unordered_pairs", n)}


def mirror(decoded: dict) -> dict:
    """left from right, behind from in_front_of: T and F exchanged, U kept, membership kept, legal cells only."""
    return {c: ({"T": "F", "F": "T", "U": "U"}[s], m) for c, (s, m) in decoded.items()}
