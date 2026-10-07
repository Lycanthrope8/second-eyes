"""The crossed cyclic design (A2.3c, D94): pure functions, no model and no files except the versioned policy.

For a base request with objects O[0..n-1] (scene order) and the first n letters C[0..n-1] of A..J:

    code assigned to object O[i]           = C[(i + a) mod n]
    object at choices-list position j      = O[(j + p) mod n]
    choices[j]                             = [C[(j + p + a) mod n], O[(j + p) mod n]],  then [K, ASK] last

for every a, p in 0..n-1 (positions zero-based; shown to readers as 1..n). (0, 0) is the original request. Across the
n * n cells every object meets every (letter, list position) pair exactly once. The scene document never changes.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from ...evaluation.iref_vla.protocol import EvaluationInputError, issue, strict_json

POLICY_PATH = Path(__file__).resolve().parent / "ordering-policy.v1.json"
POLICY_ID = "iref.order.crossed_cyclic.v1"
LETTERS = tuple("ABCDEFGHIJ")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schemas" / "iref-ordering.v1.json"
_VALIDATORS = {}


def _int(x) -> bool:
    return type(x) is int


def cells(n: int) -> list:
    """Every (a, p) of an n x n grid in canonical order: a ascending, then p ascending."""
    if not _int(n) or n < 1:
        raise ValueError("a grid needs a plain positive object count")
    return [(a, p) for a in range(n) for p in range(n)]


def transform(object_ids, a: int, p: int, letters=LETTERS) -> list:
    """The cell's choices list without the final [K, ASK]: [[code, object ID], ...] in list order."""
    n = len(object_ids)
    if n < 1 or n > len(letters):
        raise ValueError(f"{n} objects; the interface offers 1 to {len(letters)}")
    if not (_int(a) and _int(p) and 0 <= a < n and 0 <= p < n):
        raise ValueError("a and p must be plain integers in 0..n-1")
    return [[letters[(j + p + a) % n], object_ids[(j + p) % n]] for j in range(n)]


def full_mapping(object_ids, a: int, p: int, policy_proto) -> list:
    """The complete ordered mapping [[code, target], ...] with [K, ASK] last, as the prompt shows it."""
    return transform(object_ids, a, p, tuple(policy_proto["object_codes"])) + \
        [[policy_proto["ask_code"], policy_proto["ask_target"]]]


def variant_id(base_request_id: str, a: int, p: int) -> str:
    return f"{base_request_id}.a{a:02d}.p{p:02d}"


def balance_problems(object_ids, mappings: dict) -> list:
    """mappings: {(a, p): choices without K}. Each object meets every (code, list position) exactly once."""
    n, bad = len(object_ids), []
    if sorted(mappings) != cells(n):
        return [f"the grid has {len(mappings)} cells, not the {n * n} of a complete {n} x {n} grid"]
    seen = {o: set() for o in object_ids}
    for (a, p), m in mappings.items():
        if len(m) != n or sorted(o for _, o in m) != sorted(object_ids) or len({c for c, _ in m}) != n:
            bad.append(f"cell ({a}, {p}) is not a bijection between {n} codes and the objects")
            continue
        for j, (code, obj) in enumerate(m):
            if (code, j) in seen[obj]:
                bad.append(f"{obj} meets code {code} at list position {j + 1} twice")
            seen[obj].add((code, j))
    want = {(c, j) for c in LETTERS[:n] for j in range(n)}
    for o, s in seen.items():
        if s != want:
            bad.append(f"{o} does not meet every (code, list position) pair exactly once")
    return bad


def schedule_order(variants, salt: str) -> list:
    """Variant IDs in execution order: identity cells first in base order, then the rest by salted SHA-256."""
    ident = [v["variant_id"] for v in variants if v["assignment_shift"] == 0 and v["order_shift"] == 0]
    rest = [v["variant_id"] for v in variants if (v["assignment_shift"], v["order_shift"]) != (0, 0)]
    rest.sort(key=lambda vid: (hashlib.sha256((salt + vid).encode("utf-8")).hexdigest(), vid))
    return ident + rest


def schedule_sha256(order) -> str:
    return hashlib.sha256("".join(v + "\n" for v in order).encode("utf-8")).hexdigest()


def validator(name: str):
    if name not in _VALIDATORS:
        import jsonschema
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        _VALIDATORS[name] = jsonschema.Draft202012Validator({"$ref": f"#/$defs/{name}", "$defs": schema["$defs"]})
    return _VALIDATORS[name]


_IDENT_INTS = ("format_version", "variant_index", "base_request_index", "selection_rank", "object_count",
               "assignment_shift", "order_shift")
PLAIN_INTS = {"request_row": _IDENT_INTS + ("prompt_bytes", "input_tokens", "schedule_position"),
              "result_row": _IDENT_INTS + ("execution_index", "input_tokens", "choice_list_position", "choice_scene_position"),
              "score_row": _IDENT_INTS + ("source_target_list_position", "source_target_scene_position",
                                          "choice_list_position", "choice_scene_position")}


def plain_int_problems(name: str, record) -> list:
    """JSON Schema accepts 3.0 as an integer; this increment's versions, counters and positions must be plain integers."""
    if not isinstance(record, dict):
        return []
    bad = [f"{k}: {record[k]!r} is not a plain integer" for k in PLAIN_INTS.get(name, ())
           if k in record and record[k] is not None and type(record[k]) is not int]
    m = record.get("mapping")
    if isinstance(m, list) and any(not isinstance(x, list) or len(x) != 3 or type(x[2]) is not int for x in m):
        bad.append("mapping: every token ID must be a plain integer")
    return bad


def schema_errors(name: str, record) -> list:
    return [f"{'/'.join(str(x) for x in e.absolute_path) or '(record)'}: {e.message}"
            for e in validator(name).iter_errors(record)] + plain_int_problems(name, record)


def load_policy(path=None) -> dict:
    p = Path(path) if path is not None else POLICY_PATH
    try:
        data = p.read_bytes()
    except OSError as e:
        raise EvaluationInputError([issue(str(p), "E_ORDER_POLICY", f"cannot read the policy: {e}")]) from None
    pol = strict_json("ordering policy", data, "E_ORDER_POLICY")
    bad = schema_errors("policy", pol) if isinstance(pol, dict) else ["the policy must be a JSON object"]
    if not bad and pol["policy_id"] != POLICY_ID:
        bad.append(f"policy_id must be {POLICY_ID}")
    if bad:
        raise EvaluationInputError([issue(str(p), "E_ORDER_POLICY", m) for m in bad])
    return pol


def policy_sha256(path=None) -> str:
    return hashlib.sha256(Path(path if path is not None else POLICY_PATH).read_bytes()).hexdigest()
