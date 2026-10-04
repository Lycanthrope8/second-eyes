#!/usr/bin/env python3
"""A2.1d indexed relation encoding: one bounded token audit (Second_Eyes_A2_1d_Indexed_Audit_Brief.md).

    python analysis/A2.1d_indexed_token_sizing/indexed_token_audit.py --out OUT_DIR
        [--runtime-lib LIB --runtime-gguf GGUF]   # llama.cpp's tokenizer through the app's se_tokenize
        [--hf-reference DIR]                      # A1's Hugging Face download (a second Hugging Face copy)
        [--compare-with RESULTS_JSON]             # check this run reproduces a delivered results.json

Run from the repository root. It reads the eight saved inputs of the v0.2 audit (analysis/A2.1d_token_sizing/output),
which it never changes, and re-encodes the same facts: every relation block's truth states and conditional
memberships, and every distance, in the indexed rows the brief defines. Nothing is recomputed from geometry. Before
any token is counted, every original block is validated and expanded over its complete domain, the new block is
decoded by separately written layout logic and compared with it exactly, and the brief's acceptance checks run; any
failure makes the audit unsuccessful. Tokens are then measured with the original audit's tokenizer, special-token
options and chat wrapper (imported from its script, read-only).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ORIGINAL = REPO / "analysis" / "A2.1d_token_sizing"
_spec = importlib.util.spec_from_file_location("token_sizing_audit", ORIGINAL / "token_sizing_audit.py")
V02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V02)  # the original audit, reused for its tokenizer, wrapper and number spelling
sys.path.insert(0, str(HERE))
import expected_cases as X  # noqa: E402  (expectations written before this codec)

enc = V02.enc
COVERAGE_V2 = (
    "Indices are zero-based positions in objects. Every declared domain is complete; T=true, F=false, U=unknown. A "
    "single T/F/U broadcasts over legal tuples; [] means an empty domain. objects uses one character per target. "
    "ordered_pairs uses target rows and anchor columns. unordered_pairs uses upper-triangle rows: row i lists j=i+1 "
    "onward. target_anchor_pairs uses [a,b,row] for every a<b in index order; character t describes target t. '-' "
    "marks excluded repeated-object cells, never unknown or false. Conditional membership is a Boolean broadcast or "
    "matching rows of 0/1; 1 means consulted assumptions. Distances use numeric upper-triangle rows, with null "
    "unknown. Undeclared or excluded tuples have no truth value. left=opposite(right) and behind=opposite(in_front_of), "
    "exchanging T/F and retaining U on legal tuples.")
CASES = ["a17.annotated", "a17.restricted", "dirh10.annotated", "dirh10.restricted"]
OLD_FORMATS = {"coordinates_v1": "coordinates_v2", "coordinates_relations_v1": "coordinates_relations_v2"}
STATIC = [("near", None, "unordered_pairs", "near"), ("above", None, "ordered_pairs", "above"),
          ("below", None, "ordered_pairs", "below"), ("on", None, "ordered_pairs", "on"),
          ("inside", None, "ordered_pairs", "inside"), ("between", None, "target_anchor_pairs", "between"),
          ("right", "object_intrinsic", "ordered_pairs", "intrinsic.right"),
          ("in_front_of", "object_intrinsic", "ordered_pairs", "intrinsic.in_front_of")]
DYNAMIC = [("right", "user_heading", "objects", "user_heading.right"),
           ("in_front_of", "user_heading", "objects", "user_heading.in_front_of"),
           ("right", "user_to_anchor", "ordered_pairs", "user_to_anchor.right"),
           ("in_front_of", "user_to_anchor", "ordered_pairs", "user_to_anchor.in_front_of")]


class CodecError(ValueError):
    pass


class NoValue(LookupError):
    """An excluded, undeclared or out-of-domain tuple: it has no truth value."""


# ---------------------------------------------------------------------- the domains (shared definition of legality)
def legal_tuples(domain, O):
    n = len(O)
    if domain == "objects":
        return [(O[i],) for i in range(n)]
    if domain == "unordered_pairs":
        return [(O[i], O[j]) for i in range(n) for j in range(i + 1, n)]
    if domain == "ordered_pairs":
        return [(O[i], O[j]) for i in range(n) for j in range(n) if i != j]
    if domain == "target_anchor_pairs":
        return [(O[t], O[a], O[b]) for a in range(n) for b in range(a + 1, n) for t in range(n) if t not in (a, b)]
    raise CodecError(f"unknown domain {domain!r}")


# ------------------------------------------------------------------- the ORIGINAL v0.2 blocks: validate and expand
def expand_original(block, O):
    """A v0.2 frequency-default block -> {tuple: (state, conditional)} over its complete domain."""
    if list(block) != ["relation", "frame", "domain", "default", "exceptions", "conditional"]:
        raise CodecError(f"original block keys {list(block)}")
    legal = legal_tuples(block["domain"], O)
    legal_set = set(legal)
    default = block["default"]
    if default not in ("T", "F", "U") or list(block["exceptions"]) != [s for s in "TFU" if s != default]:
        raise CodecError(f"original default {default!r} with exception keys {list(block['exceptions'])}")
    states = {}
    for s, tuples in block["exceptions"].items():
        for t in tuples:
            t = tuple(t)
            if t not in legal_set or t in states:
                raise CodecError(f"original exception {t} is outside the domain or repeated")
            states[t] = s
    cond = block["conditional"]
    if list(cond) != ["default", "exceptions"] or not isinstance(cond["default"], bool):
        raise CodecError("original conditional map malformed")
    flipped = set()
    for t in cond["exceptions"]:
        t = tuple(t)
        if t not in legal_set or t in flipped:
            raise CodecError(f"original conditional exception {t} is outside the domain or repeated")
        flipped.add(t)
    return {t: (states.get(t, default), (not cond["default"]) if t in flipped else cond["default"]) for t in legal}


def expand_original_distances(block, O):
    if list(block) != ["measure", "domain", "values", "conditional"] or block["measure"] != "center_distance_m" \
            or block["domain"] != "unordered_pairs":
        raise CodecError("original distance block malformed")
    legal = legal_tuples("unordered_pairs", O)
    values = {}
    for row in block["values"]:
        if len(row) != 3:
            raise CodecError(f"original distance row {row}")
        t, d = (row[0], row[1]), row[2]
        if t not in set(legal) or t in values or not (d is None or (isinstance(d, (int, float))
                                                                      and not isinstance(d, bool) and math.isfinite(d))):
            raise CodecError(f"original distance {row}")
        values[t] = d
    if set(values) != set(legal):
        raise CodecError("original distances don't cover every pair")
    cond = block["conditional"]
    flipped = {tuple(t) for t in cond["exceptions"]}
    if not flipped <= set(legal) or len(flipped) != len(cond["exceptions"]):
        raise CodecError("original distance conditional map malformed")
    return {t: (values[t], (not cond["default"]) if t in flipped else cond["default"]) for t in legal}


# -------------------------------------------------------------------------------------------- ENCODER (canonical)
def _uniform_or_rows(domain, O, legal, cell):
    """cell: {tuple: symbol} over the legal tuples. A uniform symbol, or the complete row layout."""
    first = cell[legal[0]]
    if all(cell[t] == first for t in legal):
        return first
    n = len(O)
    if domain == "objects":
        return "".join(cell[(O[i],)] for i in range(n))
    if domain == "ordered_pairs":
        return ["".join("-" if i == j else cell[(O[i], O[j])] for j in range(n)) for i in range(n)]
    if domain == "unordered_pairs":
        return ["".join(cell[(O[i], O[j])] for j in range(i + 1, n)) for i in range(n - 1)]
    return [[a, b, "".join("-" if t in (a, b) else cell[(O[t], O[a], O[b])] for t in range(n))]
            for a in range(n) for b in range(a + 1, n)]


def encode_block(relation, frame, domain, O, mapping):
    legal = legal_tuples(domain, O)
    if set(mapping) != set(legal):
        raise CodecError("the mapping doesn't cover exactly the legal domain")
    if not legal:
        states, cond = [], False
    else:
        states = _uniform_or_rows(domain, O, legal, {t: mapping[t][0] for t in legal})
        bits = _uniform_or_rows(domain, O, legal, {t: "1" if mapping[t][1] else "0" for t in legal})
        cond = (bits == "1") if bits in ("0", "1") else bits
    return {"relation": relation, "frame": frame, "domain": domain, "states": states, "conditional": cond}


def encode_distances(O, mapping):
    n = len(O)
    legal = legal_tuples("unordered_pairs", O)
    if set(mapping) != set(legal):
        raise CodecError("the distance mapping doesn't cover every pair")
    values = [[mapping[(O[i], O[j])][0] for j in range(i + 1, n)] for i in range(n - 1)]
    if not legal:
        cond = False
    else:
        bits = _uniform_or_rows("unordered_pairs", O, legal, {t: "1" if mapping[t][1] else "0" for t in legal})
        cond = (bits == "1") if bits in ("0", "1") else bits
    return {"measure": "center_distance_m", "domain": "unordered_pairs", "values": values, "conditional": cond}


# ------------------------------------------------------------- DECODER (written separately from the encoder above)
def _count_legal(domain, n):
    return {"objects": n, "unordered_pairs": n * (n - 1) // 2, "ordered_pairs": n * (n - 1),
            "target_anchor_pairs": n * (n - 1) * (n - 2) // 2}[domain]


def _read_grid(domain, n, value, alphabet, what):
    """One field's layout -> {index tuple: symbol}, or ('uniform', symbol). Validates every cell."""
    if isinstance(value, str) and len(value) == 1 and value in alphabet:
        return ("uniform", value)  # for objects with n == 1 the one-symbol row means the same
    cells = {}
    if domain == "objects":
        if not isinstance(value, str) or len(value) != n:
            raise CodecError(f"{what}: objects row must be a string of {n} symbols")
        for i, ch in enumerate(value):
            if ch not in alphabet:
                raise CodecError(f"{what}: symbol {ch!r} at target {i}")
            cells[(i,)] = ch
        return cells
    if not isinstance(value, list):
        raise CodecError(f"{what}: rows must be a list")
    if domain == "ordered_pairs":
        if len(value) != n:
            raise CodecError(f"{what}: {len(value)} rows, expected {n}")
        for i, row in enumerate(value):
            if not isinstance(row, str) or len(row) != n:
                raise CodecError(f"{what}: row {i} must be a string of {n} symbols")
            for j, ch in enumerate(row):
                if i == j:
                    if ch != "-":
                        raise CodecError(f"{what}: diagonal cell {i} is {ch!r}, not '-'")
                elif ch not in alphabet:
                    raise CodecError(f"{what}: symbol {ch!r} at row {i}, column {j}")
                else:
                    cells[(i, j)] = ch
        return cells
    if domain == "unordered_pairs":
        if len(value) != max(n - 1, 0):
            raise CodecError(f"{what}: {len(value)} rows, expected {max(n - 1, 0)}")
        for i, row in enumerate(value):
            if not isinstance(row, str) or len(row) != n - 1 - i:
                raise CodecError(f"{what}: row {i} must hold {n - 1 - i} symbols")
            for k, ch in enumerate(row):
                if ch not in alphabet:
                    raise CodecError(f"{what}: symbol {ch!r} at row {i}, position {k}")
                cells[(i, i + 1 + k)] = ch
        return cells
    if domain == "target_anchor_pairs":
        expected = [(a, b) for a in range(n) for b in range(a + 1, n)]
        if len(value) != len(expected):
            raise CodecError(f"{what}: {len(value)} anchor rows, expected {len(expected)}")
        for (ea, eb), row in zip(expected, value):
            if not (isinstance(row, list) and len(row) == 3):
                raise CodecError(f"{what}: anchor row {row!r} must be [a,b,row]")
            a, b, s = row
            if not all(isinstance(x, int) and not isinstance(x, bool) for x in (a, b)):
                raise CodecError(f"{what}: anchor indices {a!r}, {b!r} must be integers")
            if not (0 <= a < b < n):
                raise CodecError(f"{what}: anchor indices {a}, {b} out of range or out of order")
            if (a, b) != (ea, eb):
                raise CodecError(f"{what}: anchor row ({a},{b}) where ({ea},{eb}) is due (duplicate or missing)")
            if not isinstance(s, str) or len(s) != n:
                raise CodecError(f"{what}: anchor row ({a},{b}) must hold {n} symbols")
            for t, ch in enumerate(s):
                if t in (a, b):
                    if ch != "-":
                        raise CodecError(f"{what}: anchor position {t} in row ({a},{b}) is {ch!r}, not '-'")
                elif ch not in alphabet:
                    raise CodecError(f"{what}: symbol {ch!r} for target {t} in row ({a},{b})")
                else:
                    cells[(t, a, b)] = ch
        return cells
    raise CodecError(f"{what}: unknown domain {domain!r}")


def _broadcast(domain, n):
    """Every legal index tuple of a domain, enumerated independently of the encoder."""
    if domain == "objects":
        return [(i,) for i in range(n)]
    if domain == "ordered_pairs":
        return [(i, j) for i in range(n) for j in range(n) if j != i]
    if domain == "unordered_pairs":
        return [(i, j) for i in range(n) for j in range(n) if i < j]
    return [(t, a, b) for a in range(n) for b in range(n) if a < b for t in range(n) if t != a and t != b]


def _read_field(domain, n, value, kind, what):
    empty = _count_legal(domain, n) == 0
    if kind == "conditional" and isinstance(value, bool):
        if empty and value:
            raise CodecError(f"{what}: an empty domain's conditional must be false")
        return {} if empty else {k: value for k in _broadcast(domain, n)}
    if value == []:
        if not empty:
            raise CodecError(f"{what}: [] for a nonempty domain")
        return {}
    if empty:
        raise CodecError(f"{what}: an empty domain needs [] / false")
    alphabet = "TFU" if kind == "states" else "01"
    got = _read_grid(domain, n, value, alphabet, what)
    if isinstance(got, tuple):
        if kind != "states":
            raise CodecError(f"{what}: a uniform conditional must be a JSON Boolean")
        return {k: got[1] for k in _broadcast(domain, n)}
    if kind == "conditional":
        return {k: v == "1" for k, v in got.items()}
    return got


def decode_block(block, O):
    if list(block) != ["relation", "frame", "domain", "states", "conditional"]:
        raise CodecError(f"block fields {list(block)}")
    domain, n = block["domain"], len(O)
    if domain not in ("objects", "unordered_pairs", "ordered_pairs", "target_anchor_pairs"):
        raise CodecError(f"unknown domain {domain!r}")
    states = _read_field(domain, n, block["states"], "states", block["relation"])
    cond = _read_field(domain, n, block["conditional"], "conditional", block["relation"])
    if set(states) != set(cond) or len(states) != _count_legal(domain, n):
        raise CodecError("states and conditional don't cover the same complete domain")
    return {tuple(O[i] for i in k): (states[k], cond[k]) for k in states}


def decode_distances(block, O):
    if list(block) != ["measure", "domain", "values", "conditional"] or block["measure"] != "center_distance_m" \
            or block["domain"] != "unordered_pairs":
        raise CodecError(f"distance block fields {list(block)}")
    n, values = len(O), block["values"]
    if not isinstance(values, list) or len(values) != max(n - 1, 0):
        raise CodecError(f"distances: {len(values) if isinstance(values, list) else '?'} rows, expected {max(n - 1, 0)}")
    out = {}
    for i, row in enumerate(values):
        if not isinstance(row, list) or len(row) != n - 1 - i:
            raise CodecError(f"distances: row {i} must hold {n - 1 - i} values")
        for k, d in enumerate(row):
            if not (d is None or (isinstance(d, (int, float)) and not isinstance(d, bool) and math.isfinite(d))):
                raise CodecError(f"distances: value {d!r} at row {i}, position {k}")
            out[(i, i + 1 + k)] = d
    cond = _read_field("unordered_pairs", n, block["conditional"], "conditional", "distances")
    if set(cond) != set(out):
        raise CodecError("distance conditional doesn't match the values' shape")
    return {(O[i], O[j]): (out[(i, j)], cond[(i, j)]) for (i, j) in out}


def lookup(decoded, domain, ids, O):
    """A tuple's (state, conditional), retrieving unordered pairs and between's anchors in either order."""
    if any(x not in O for x in ids):
        raise NoValue(f"{ids} has an ID that isn't in objects")
    if domain == "unordered_pairs" and len(set(ids)) == 2:
        key = tuple(sorted(ids, key=O.index))
    elif domain == "target_anchor_pairs" and len(set(ids)) == 3:
        key = (ids[0],) + tuple(sorted(ids[1:], key=O.index))
    else:
        key = tuple(ids)
    if key not in decoded:
        raise NoValue(f"{ids} has no truth value in this {domain} block")
    return decoded[key]


def mirror(decoded):
    """left from right, behind from in_front_of: legal tuples only; T<->F, U stays U, membership kept."""
    return {t: ({"T": "F", "F": "T", "U": "U"}[s], c) for t, (s, c) in decoded.items()}


# ------------------------------------------------------------------------------------------------ acceptance checks
class Checks:
    def __init__(self):
        self.rows = []

    def add(self, name, ok, detail=""):
        self.rows.append({"check": name, "ok": bool(ok), "detail": detail})

    def raises(self, name, fn, exc=CodecError):
        try:
            fn()
        except exc as e:
            self.add(name, True, str(e)[:120])
            return
        except Exception as e:  # noqa: BLE001
            self.add(name, False, f"raised {type(e).__name__}, not {exc.__name__}: {e}")
            return
        self.add(name, False, "accepted")

    @property
    def ok(self):
        return all(r["ok"] for r in self.rows)


def unit_checks(ck: Checks):
    # 8.1: the literal section 4 examples and their stated mappings
    for text, domain, want in X.LITERAL_BLOCKS:
        block = json.loads(text)
        got = decode_block(block, X.O_ABC)
        again = enc(encode_block(block["relation"], block["frame"], domain, X.O_ABC, got))
        ck.add(f"8.1 literal {domain} example decodes to the brief's mapping and re-encodes to its exact text",
               got == want and again == text, "" if got == want else str(got))
    text, want = X.LITERAL_DISTANCES
    got = decode_distances(json.loads(text), X.O_ABC)
    ck.add("8.1/8.4 literal distances: 0.1006 and 0.1508 exact, B/C null, re-encoded to the exact text",
           got == want and enc(encode_distances(X.O_ABC, got)) == text and got[("B", "C")][0] is None)
    zero_vs_null = {("A", "B"): (0.0, False), ("A", "C"): (None, False), ("B", "C"): (0.1006, True)}
    back = decode_distances(json.loads(enc(encode_distances(X.O_ABC, zero_vs_null))), X.O_ABC)
    ck.add("8.4 a zero distance and a null distance stay distinct", back == zero_vs_null
           and back[("A", "B")][0] == 0 and back[("A", "C")][0] is None)
    # 8.2: every domain at n = 0, 1, 2, 3, 6, 10, uniform and mixed truth, uniform and mixed membership
    count = 0
    for domain in ("objects", "unordered_pairs", "ordered_pairs", "target_anchor_pairs"):
        for n in (0, 1, 2, 3, 6, 10):
            O = [f"obj_{i + 1:03d}" for i in range(n)]
            legal = legal_tuples(domain, O)
            patterns = {"uniform T": lambda k, t: "T", "uniform F": lambda k, t: "F", "uniform U": lambda k, t: "U",
                        "mixed": lambda k, t: "TFU"[(k * 7 + len(t)) % 3]}
            memberships = {"false": lambda k: False, "true": lambda k: True, "mixed": lambda k: k % 2 == 0}
            for pname, pf in patterns.items():
                for mname, mf in memberships.items():
                    mapping = {t: (pf(k, t), mf(k)) for k, t in enumerate(legal)}
                    block = encode_block("r", None, domain, O, mapping)
                    text = enc(block)
                    decoded = decode_block(json.loads(text), O)
                    states = {s for s, _ in mapping.values()}
                    members = {c for _, c in mapping.values()}
                    if not legal:
                        form_ok = block["states"] == [] and block["conditional"] is False
                    else:
                        form_ok = ((block["states"] in ("T", "F", "U")) == (len(states) == 1)
                                   and isinstance(block["conditional"], bool) == (len(members) == 1))
                    again = enc(encode_block("r", None, domain, O, decoded))
                    count += 1
                    if not (decoded == mapping and form_ok and again == text):
                        ck.add(f"8.2 unit case {domain} n={n} {pname} membership {mname}", False, text[:120])
        for n in (0, 1, 2, 3, 6, 10):
            O = [f"obj_{i + 1:03d}" for i in range(n)]
            mapping = {t: (None if k % 5 == 3 else 0.1006 + 0.0502 * k, k % 3 == 0)
                       for k, t in enumerate(legal_tuples("unordered_pairs", O))}
            text = enc(encode_distances(O, mapping))
            count += 1
            if decode_distances(json.loads(text), O) != mapping or enc(encode_distances(O, mapping)) != text:
                ck.add(f"8.2 distance unit case n={n}", False, text[:120])
    ck.add(f"8.2 {count} unit cases over every domain at n = 0, 1, 2, 3, 6, 10 round-trip in canonical form",
           not any(r["check"].startswith("8.2 unit") or r["check"].startswith("8.2 distance") for r in ck.rows))
    # 8.5: reverse retrieval
    near = decode_block(json.loads(X.LITERAL_BLOCKS[2][0]), X.O_ABC)
    btw = decode_block(json.loads(X.LITERAL_BLOCKS[3][0]), X.O_ABC)
    ordered = decode_block(json.loads(X.LITERAL_BLOCKS[1][0]), X.O_ABC)
    dist = decode_distances(json.loads(X.LITERAL_DISTANCES[0]), X.O_ABC)
    ok = (all(lookup(near, "unordered_pairs", k, X.O_ABC)[0] == v for k, v in X.REVERSE["near"].items())
          and all(lookup(btw, "target_anchor_pairs", k, X.O_ABC)[0] == v for k, v in X.REVERSE["between"].items())
          and all(lookup(ordered, "ordered_pairs", k, X.O_ABC)[0] == v for k, v in X.REVERSE["ordered"].items())
          and lookup(dist, "unordered_pairs", ("B", "A"), X.O_ABC)[0] == 0.1006)
    ck.add("8.5 near, distances and between's anchors read the same in either order; ordered pairs are not "
           "symmetrized (right(A,B)=T, right(B,A)=F)", ok)
    # 8.6: excluded cells and empty domains never take a value, by broadcast or by mirroring
    uniform = decode_block({"relation": "right", "frame": "user_to_anchor", "domain": "ordered_pairs", "states": "T",
                            "conditional": True}, X.O_ABC)
    empty = decode_block({"relation": "near", "frame": None, "domain": "unordered_pairs", "states": [],
                          "conditional": False}, ["A"])
    for name, fn in (("8.6 a uniform T doesn't reach the diagonal (A,A)",
                      lambda: lookup(uniform, "ordered_pairs", ("A", "A"), X.O_ABC)),
                     ("8.6 mirroring doesn't create the diagonal",
                      lambda: lookup(mirror(uniform), "ordered_pairs", ("B", "B"), X.O_ABC)),
                     ("8.6 between with a repeated object has no value",
                      lambda: lookup(btw, "target_anchor_pairs", ("A", "A", "B"), X.O_ABC)),
                     ("8.6 an empty domain has no tuples", lambda: lookup(empty, "unordered_pairs", ("A", "A"), ["A"])),
                     ("8.6 an ID outside objects has no value",
                      lambda: lookup(ordered, "ordered_pairs", ("A", "D"), X.O_ABC))):
        ck.raises(name, fn, NoValue)
    m = mirror(ordered)
    ck.add("8.6 mirroring maps T<->F, keeps U and membership, and only over legal tuples",
           set(m) == set(ordered) and all(m[t][0] == {"T": "F", "F": "T", "U": "U"}[ordered[t][0]]
                                          and m[t][1] == ordered[t][1] for t in ordered))
    # 8.7: the literal directed example's orientation
    ck.add("8.7 literal directed example: row = target, column = anchor (right(A,B)=T conditional, right(C,A)=U "
           "conditional)", ordered[("A", "B")] == ("T", True) and ordered[("C", "A")] == ("U", True)
           and ordered[("B", "A")] == ("F", False))
    # 8.8: invalid rows fail clearly
    base = {"relation": "right", "frame": "user_to_anchor", "domain": "ordered_pairs"}
    bad = {
        "wrong row length": dict(base, states=["-T", "F-T", "UF-"], conditional=False),
        "wrong diagonal": dict(base, states=["TTU", "F-T", "UF-"], conditional=False),
        "'-' off the diagonal": dict(base, states=["--U", "F-T", "UF-"], conditional=False),
        "illegal truth symbol": dict(base, states=["-TX", "F-T", "UF-"], conditional=False),
        "missing row": dict(base, states=["-TU", "F-T"], conditional=False),
        "illegal conditional symbol": dict(base, states="T", conditional=["-1X", "0-0", "10-"]),
        "conditional diagonal not '-'": dict(base, states="T", conditional=["110", "0-0", "10-"]),
        "uniform conditional as a string": dict(base, states="T", conditional="1"),
        "objects row too short": {"relation": "right", "frame": "user_heading", "domain": "objects",
                                  "states": "TF", "conditional": False},
        "triangle row too long": {"relation": "near", "frame": None, "domain": "unordered_pairs",
                                  "states": ["TUF", "F"], "conditional": False},
        "[] for a nonempty domain": dict(base, states=[], conditional=False),
        "unknown field": dict(base, states="T", conditional=False, extra=1),
        "unknown domain": {"relation": "r", "frame": None, "domain": "pairs", "states": "T", "conditional": False},
    }
    btw_base = {"relation": "between", "frame": None, "domain": "target_anchor_pairs", "conditional": False}
    bad.update({
        "duplicate between row": dict(btw_base, states=[[0, 1, "--T"], [0, 1, "--T"], [1, 2, "F--"]]),
        "missing between row": dict(btw_base, states=[[0, 1, "--T"], [1, 2, "F--"]]),
        "between rows out of order": dict(btw_base, states=[[0, 2, "-U-"], [0, 1, "--T"], [1, 2, "F--"]]),
        "between index out of range": dict(btw_base, states=[[0, 1, "--T"], [0, 3, "-U-"], [1, 2, "F--"]]),
        "between anchors reversed": dict(btw_base, states=[[1, 0, "--T"], [0, 2, "-U-"], [1, 2, "F--"]]),
        "between anchor not '-'": dict(btw_base, states=[[0, 1, "T-T"], [0, 2, "-U-"], [1, 2, "F--"]]),
        "between target as '-'": dict(btw_base, states=[[0, 1, "---"], [0, 2, "-U-"], [1, 2, "F--"]]),
        "between index not an integer": dict(btw_base, states=[[0, "1", "--T"], [0, 2, "-U-"], [1, 2, "F--"]]),
        "between index a Boolean": dict(btw_base, states=[[False, 1, "--T"], [0, 2, "-U-"], [1, 2, "F--"]]),
    })
    for name, block in bad.items():
        ck.raises(f"8.8 rejects: {name}", lambda b=block: decode_block(b, X.O_ABC))
    ck.raises("8.8 rejects: true conditional for an empty domain (one object, no pairs)",
              lambda: decode_block({"relation": "near", "frame": None, "domain": "unordered_pairs", "states": [],
                                    "conditional": True}, ["A"]))
    dbase = {"measure": "center_distance_m", "domain": "unordered_pairs", "conditional": False}
    for name, values in (("distance row too short", [[0.1006], [None]]), ("distance as a string", [["0.1", 0.2], [None]]),
                         ("distance as a Boolean", [[True, 0.2], [None]]), ("missing distance row", [[0.1, 0.2]])):
        ck.raises(f"8.8 rejects: {name}", lambda v=values: decode_distances(dict(dbase, values=v), X.O_ABC))
    # 8.11: indices map through O, not through suffixes
    sb = decode_block(json.loads(X.SUFFIX_BLOCK), X.O_SUFFIX)
    right = all(lookup(sb, "objects", (k,), X.O_SUFFIX)[0] == v for k, v in X.SUFFIX_EXPECTED.items())
    states = json.loads(X.SUFFIX_BLOCK)["states"]
    by_suffix = [k for k, i in X.SUFFIX_WRONG["suffix_minus_one"].items() if i < len(states)]
    numeric = sorted(X.O_SUFFIX, key=lambda s: int(s.split("_")[1]))
    wrong_read = states[numeric.index(X.SUFFIX_WRONG["numeric_order_index_1"])]
    ck.add("8.11 O=[obj_001,obj_1000,obj_999] reads T,F,U through O; suffix indices 999/998 don't exist, and numeric "
           "suffix order would read obj_999 as F instead of U", right and not by_suffix and wrong_read == "F"
           and X.SUFFIX_EXPECTED["obj_999"] == "U")


# ---------------------------------------------------------------------------------------- converting saved inputs
def convert(case, fmt_old, ck: Checks):
    path = ORIGINAL / "output" / "inputs" / f"{case}.{fmt_old}.jsonl"
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    if lines[-1] != "":
        raise CodecError(f"{path.name} doesn't end in LF")
    lines = lines[:-1]
    parsed = [json.loads(x) for x in lines]
    if list(parsed[0]) != ["header"] or list(parsed[1]) != ["semantics"] or list(parsed[2]) != ["objects"] \
            or list(parsed[-1]) != ["command"]:
        raise CodecError(f"{path.name}: unexpected line order")
    O = [row[0] for row in parsed[2]["objects"]]
    if len(set(O)) != len(O):
        raise CodecError(f"{path.name}: repeated object IDs")
    pose_at = next(i for i, p in enumerate(parsed) if list(p) == ["pose"])
    header = dict(parsed[0]["header"])
    if enc({"header": header}) != lines[0]:
        raise CodecError(f"{path.name}: header doesn't re-encode to itself")
    header.update(serializer_version=2, format=OLD_FORMATS[header["format"]], coverage=COVERAGE_V2)
    new = [enc({"header": header}), lines[1], lines[2]]
    names = ["header", "semantics", "objects"]
    facts = {}
    augmented = fmt_old == "coordinates_relations_v1"
    static_lines, dynamic_lines = lines[3:pose_at], lines[pose_at + 1:-1]
    if not augmented and (static_lines or dynamic_lines):
        raise CodecError(f"{path.name}: derived blocks in a coordinates input")
    if augmented and len(static_lines) != len(STATIC) + 1 or augmented and len(dynamic_lines) != len(DYNAMIC):
        raise CodecError(f"{path.name}: missing or extra derived blocks")

    def convert_relation(line, spec):
        relation, frame, domain, name = spec
        block = json.loads(line)
        if (block.get("relation"), block.get("frame"), block.get("domain")) != (relation, frame, domain):
            raise CodecError(f"{path.name}: expected {name}, found {block.get('relation')}/{block.get('frame')}")
        original = expand_original(block, O)
        out = enc(encode_block(relation, frame, domain, O, original))
        decoded = decode_block(json.loads(out), O)
        again = enc(encode_block(relation, frame, domain, O, decoded))
        facts[name] = {"tuples": len(original), "equal": decoded == original, "canonical": again == out,
                       "unknown": sum(1 for s, _ in original.values() if s == "U"),
                       "conditional": sum(1 for _, c in original.values() if c)}
        new.append(out)
        names.append(name)
        return decoded

    decoded_blocks = {}
    if augmented:
        for line, spec in zip(static_lines[:-1], STATIC):
            decoded_blocks[spec[3]] = convert_relation(line, spec)
        original = expand_original_distances(json.loads(static_lines[-1]), O)
        out = enc(encode_distances(O, original))
        decoded = decode_distances(json.loads(out), O)
        exact = all(decoded[t][0] == v and type(decoded[t][0]) is type(v) and decoded[t][1] == c
                    for t, (v, c) in original.items())
        facts["center_distance_m"] = {"tuples": len(original), "equal": decoded == original and exact,
                                      "canonical": enc(encode_distances(O, decoded)) == out,
                                      "null": sum(1 for v, _ in original.values() if v is None),
                                      "conditional": sum(1 for _, c in original.values() if c)}
        new.append(out)
        names.append("center_distance_m")
    new.append(lines[pose_at])
    names.append("pose")
    if augmented:
        for line, spec in zip(dynamic_lines, DYNAMIC):
            decoded_blocks[spec[3]] = convert_relation(line, spec)
    new.append(lines[-1])
    names.append("command")
    n_static = names.index("pose")
    return {"O": O, "old_lines": lines, "lines": new, "names": names, "n_static": n_static, "old_pose_at": pose_at,
            "facts": facts, "decoded": decoded_blocks}


# ------------------------------------------------------------------------------------------------------- measuring
def measure(primary, desc, conv, rate):
    lines, names, n_static = conv["lines"], conv["names"], conv["n_static"]
    doc = "".join(x + "\n" for x in lines)
    static = "".join(x + "\n" for x in lines[:n_static])
    prompt = V02.wrap(desc, doc)
    head = prompt[:prompt.index(doc)]
    tail = prompt[len(head) + len(doc):]
    ids = primary.ids(prompt)
    marginal, cum = {"wrapper_head": len(primary.ids(head))}, head
    prev = marginal["wrapper_head"]
    for name, line in zip(names, lines):
        cum += line + "\n"
        now = len(primary.ids(cum))
        marginal[name] = now - prev
        prev = now
    marginal["wrapper_tail"] = len(ids) - prev
    prefix_ids = primary.ids(head + static)
    common = V02.common_prefix(prefix_ids, ids)
    stripped = []
    for line in lines:
        v = json.loads(line)
        if isinstance(v, dict) and "conditional" in v and ("relation" in v or "measure" in v):
            v = {k: x for k, x in v.items() if k != "conditional"}
        stripped.append(enc(v))
    no_cond = len(primary.ids(V02.wrap(desc, "".join(x + "\n" for x in stripped))))
    return doc, prompt, head, static, ids, {
        "bytes": {"wrapped": len(prompt.encode()), "document": len(doc.encode()), "static_prefix": len(static.encode()),
                  "dynamic_suffix": len(doc.encode()) - len(static.encode()), "wrapper_head": len(head.encode()),
                  "wrapper_tail": len(tail.encode()), "by_block": {n: len((x + "\n").encode())
                                                                   for n, x in zip(names, lines)}},
        "tokens": {"wrapped": len(ids), "document_alone": len(primary.ids(doc)),
                   "static_lines": sum(marginal[x] for x in names[:n_static]),
                   "dynamic_lines": sum(marginal[x] for x in names[n_static:]), "marginal": marginal,
                   "marginal_sum_equals_total": sum(marginal.values()) == len(ids)},
        "warm_prefix": {"candidate_prefix_tokens": len(prefix_ids), "common_prefix_tokens": common,
                        "tokens_after_reusable_prefix": len(ids) - common,
                        "wrapper_head_tokens": marginal["wrapper_head"]},
        "conditional_diagnostic_overlapping": {"block_conditional_tokens": len(ids) - no_cond,
                                               "note": "overlaps the relation blocks' tokens; not additive"},
        "seconds_linear_extrapolation_illustration": {"cold": len(ids) / rate,
                                                      "warm_best_case": (len(ids) - common) / rate},
        "text_sha256": {"wrapped": V02.sha256_bytes(prompt.encode()), "document": V02.sha256_bytes(doc.encode())},
        "ids_sha256": {"wrapped": V02.ids_hash(ids)}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--runtime-lib")
    ap.add_argument("--runtime-gguf")
    ap.add_argument("--hf-reference")
    ap.add_argument("--compare-with")
    args = ap.parse_args(argv)
    out = Path(args.out)
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    original_results = json.loads((ORIGINAL / "output" / "results.json").read_text(encoding="utf-8"))
    ck = Checks()

    # 1. the codec on its own, against expectations written before it
    unit_checks(ck)

    # 2. the saved inputs: the delivered v0.2 documents, re-encoded and verified
    conversions = {}
    for case in CASES:
        for fmt in ("coordinates_v1", "coordinates_relations_v1"):
            old_key = f"{case}.{fmt}"
            saved = (ORIGINAL / "output" / "inputs" / f"{old_key}.jsonl").read_bytes()
            ck.add(f"saved input {old_key} is the delivered one (hash in the original results.json)",
                   V02.sha256_bytes(saved) == original_results["inputs"][old_key]["text_sha256"]["document"])
            conversions[old_key] = convert(case, fmt, ck)
    for case in CASES:
        coord, aug = conversions[f"{case}.coordinates_v1"], conversions[f"{case}.coordinates_relations_v1"]
        bad = [n for n, f in aug["facts"].items() if not (f["equal"] and f["canonical"])]
        ck.add(f"8.3/8.4 {case}: all {len(aug['facts'])} derived blocks decode to exactly the original truth, "
               f"conditional and distance maps, in canonical form", not bad, ", ".join(bad))
        same_shared = (coord["lines"][1:3] == coord["old_lines"][1:3] and coord["lines"][-2:] == coord["old_lines"][-2:]
                       and aug["lines"][1:3] == aug["old_lines"][1:3] and aug["lines"][-1] == aug["old_lines"][-1]
                       and aug["lines"][aug["n_static"]] == aug["old_lines"][aug["old_pose_at"]])
        hc, ha = json.loads(coord["lines"][0])["header"], json.loads(aug["lines"][0])["header"]
        headers = ({k: v for k, v in hc.items() if k != "format"} == {k: v for k, v in ha.items() if k != "format"}
                   and hc["serializer_version"] == 2 and hc["coverage"] == COVERAGE_V2
                   and list(hc) == list(json.loads(coord["old_lines"][0])["header"]))
        common = coord["lines"][1:3] == aug["lines"][1:3] and coord["lines"][-1] == aug["lines"][-1]
        ck.add(f"8.9 {case}: semantics, objects, pose and command are byte-identical to the originals; the two new "
               f"headers differ only in format; both formats share the same evidence", same_shared and headers and common)
        boundary = (aug["n_static"] == aug["old_pose_at"] and coord["n_static"] == coord["old_pose_at"]
                    and aug["names"][:aug["n_static"]][-1] == "center_distance_m"
                    and all(n.startswith(("user_heading", "user_to_anchor")) for n in aug["names"][aug["n_static"] + 1:-1]))
        ck.add(f"8.10 {case}: the static/dynamic boundary is where it was; viewer-relative blocks stay after pose",
               boundary)
    a17 = conversions["a17.annotated.coordinates_relations_v1"]["decoded"]["user_to_anchor.right"]
    O17 = conversions["a17.annotated.coordinates_relations_v1"]["O"]
    got = {k: lookup(a17, "ordered_pairs", k, O17)[0] for k in X.A17_ORIENTATION["cells"]}
    ck.add("8.7 a17 user-to-anchor right: row obj_005 / column obj_001 is T and the transposed cell F, as worked by "
           "hand (D07/D10 themselves are not among the four audited inputs, which hold one pose each)",
           got == X.A17_ORIENTATION["cells"], str(got))
    for r in ck.rows:
        print(("PASS  " if r["ok"] else "FAIL  ") + r["check"] + (f": {r['detail']}" if r["detail"] and not r["ok"]
                                                                  else ""))
    if not ck.ok:
        print("Verification failed: the audit is unsuccessful and no tokens are reported.")
        (out / "results.json").write_text(json.dumps({"verification": ck.rows, "successful": False}, indent=1) + "\n",
                                          encoding="utf-8")
        return 1

    # 3. tokens, through the original audit's tokenizer, special-token options and wrapper
    primary = V02.HFTokenizer(V02.TOKENIZER_DIR)
    tokenizers = [primary]
    skipped = {}
    if args.runtime_lib or args.runtime_gguf:
        if all(p and Path(p).exists() for p in (args.runtime_lib, args.runtime_gguf)):
            tokenizers.append(V02.RuntimeTokenizer(args.runtime_lib, args.runtime_gguf))
        else:
            skipped["runtime tokenizer"] = "not found"
    if args.hf_reference:
        if Path(args.hf_reference).exists():
            tokenizers.append(V02.HFTokenizer(args.hf_reference))
        else:
            skipped["second Hugging Face copy"] = f"not found: {args.hf_reference}"
    desc = json.loads(V02.MODEL_DESC.read_text(encoding="utf-8"))
    rate = original_results["extrapolation"]["tokens_per_second"]
    results = {"successful": True, "verification": ck.rows, "tokenizers": [t.name for t in tokenizers],
               "calibration": V02.calibration(desc, tokenizers, []), "inputs": {}, "comparison_old_new": {},
               "skipped": skipped, "extrapolation": original_results["extrapolation"]}
    token_ids, parity_texts = {}, {}
    for old_key, conv in conversions.items():
        case, fmt_old = old_key.split(".", 2)[0] + "." + old_key.split(".", 2)[1], old_key.split(".", 2)[2]
        key = f"{case}.{OLD_FORMATS[fmt_old]}"
        doc, prompt, head, static, ids, m = measure(primary, desc, conv, rate)
        (out / "inputs" / f"{key}.jsonl").write_bytes(doc.encode("utf-8"))
        (out / "inputs" / f"{key}.prompt.txt").write_bytes(prompt.encode("utf-8"))
        m["facts"] = conv["facts"]
        results["inputs"][key] = m
        token_ids[key] = {"wrapped": ids}
        parity_texts.update({f"{key}.prompt": prompt, f"{key}.document": doc, f"{key}.static_prefix_input": head + static})
        old = original_results["inputs"][old_key]
        results["comparison_old_new"][key] = {
            "old_input": old_key, "old_tokens": old["tokens"]["wrapped"], "new_tokens": m["tokens"]["wrapped"],
            "change": m["tokens"]["wrapped"] - old["tokens"]["wrapped"],
            "old_after_prefix": old["warm_prefix"]["tokens_after_reusable_prefix"],
            "new_after_prefix": m["warm_prefix"]["tokens_after_reusable_prefix"],
            "old_bytes": old["bytes"]["wrapped"], "new_bytes": m["bytes"]["wrapped"],
            "blocks": {b: {"old": old["tokens"]["marginal"].get(b), "new": m["tokens"]["marginal"].get(b)}
                       for b in m["tokens"]["marginal"]}}
    for t in tokenizers[1:]:
        diffs = [name for name, text in parity_texts.items() if primary.ids(text) != t.ids(text)]
        results.setdefault("parity", {})[t.name] = {"texts": len(parity_texts), "identical": len(parity_texts) - len(diffs),
                                                    "differences": diffs}
    if args.compare_with and Path(args.compare_with).exists():
        old = json.loads(Path(args.compare_with).read_text(encoding="utf-8"))
        results["reproduction"] = {k: {"same_text": old["inputs"].get(k, {}).get("text_sha256") == v["text_sha256"],
                                       "same_token_ids": old["inputs"].get(k, {}).get("ids_sha256") == v["ids_sha256"]}
                                   for k, v in results["inputs"].items()}
    (out / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "token_ids.json").write_text(json.dumps(token_ids) + "\n", encoding="utf-8")
    (out / "report.md").write_text(report(results), encoding="utf-8")
    print(report(results))
    return 0


def report(r) -> str:
    out = ["# A2.1d indexed encoding: token audit results", "",
           f"Verification: {sum(c['ok'] for c in r['verification'])} of {len(r['verification'])} checks passed. "
           f"Tokenizers: {'; '.join(r['tokenizers'])}.", "", "## Calibration against A1's records", "",
           "| A1 prompt | Recorded | Measured (each tokenizer) | Scene recorded | Scene measured |", "|---|---|---|---|---|"]
    for pid, row in r["calibration"].items():
        out.append(f"| {pid} | {row['recorded_tokens']} | " + ", ".join(str(m["tokens"]) for m in row["measured"].values())
                   + f" | {row['recorded_scene_tokens']} | " + ", ".join(str(m["scene_tokens"]) for m in
                                                                        row["measured"].values()) + " |")
    out += ["", "## Old against new", "",
           "| New input | Old input | Old tokens | New tokens | Change | Old after prefix | New after prefix | "
           "Old bytes | New bytes |", "|---|---|---|---|---|---|---|---|---|"]
    for k, c in r["comparison_old_new"].items():  # noqa: B007
        out.append(f"| {k} | {c['old_input']} | {c['old_tokens']} | {c['new_tokens']} | {c['change']:+d} | "
                   f"{c['old_after_prefix']} | {c['new_after_prefix']} | {c['old_bytes']} | {c['new_bytes']} |")
    out += ["", "## Per-block marginal tokens, old -> new", ""]
    keys = list(r["comparison_old_new"])
    blocks = []
    for k in keys:
        blocks += [b for b in r["comparison_old_new"][k]["blocks"] if b not in blocks]
    out.append("| Block | " + " | ".join(keys) + " |")
    out.append("|---" * (len(keys) + 1) + "|")
    for b in blocks:
        cells = []
        for k in keys:
            v = r["comparison_old_new"][k]["blocks"].get(b)
            cells.append("" if v is None else f"{v['old']} -> {v['new']}")
        out.append(f"| {b} | " + " | ".join(cells) + " |")
    out += ["", "## New inputs", "", "| Input | Bytes | Tokens | Document alone | Static lines | Dynamic lines | "
            "After reusable prefix | Block conditional maps (overlapping) |", "|---|---|---|---|---|---|---|---|"]
    for k, v in r["inputs"].items():
        out.append(f"| {k} | {v['bytes']['wrapped']} | {v['tokens']['wrapped']} | {v['tokens']['document_alone']} | "
                   f"{v['tokens']['static_lines']} | {v['tokens']['dynamic_lines']} | "
                   f"{v['warm_prefix']['tokens_after_reusable_prefix']} | "
                   f"{v['conditional_diagnostic_overlapping']['block_conditional_tokens']} |")
    for name, p in r.get("parity", {}).items():
        out += ["", f"Parity with {name}: {p['identical']} of {p['texts']} texts identical"
                + (f"; differences: {p['differences']}" if p["differences"] else "") + "."]
    if "reproduction" in r:
        out += ["", "Against the delivered results: " + "; ".join(
            f"{k} {'same' if v['same_text'] and v['same_token_ids'] else 'DIFFERENT'}"
            for k, v in r["reproduction"].items()) + "."]
    out += ["", f"Skipped: {r['skipped'] or 'nothing'}.", ""]
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
