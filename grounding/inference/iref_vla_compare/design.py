"""A2.3d (D95) design: answer-blind selection and one frozen code and list order per parent and view. Pure functions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ...evaluation.iref_vla.protocol import EvaluationInputError, issue, strict_json

POLICY_PATH = Path(__file__).resolve().parent / "compare-policy.v1.json"
POLICY_ID = "iref.compare.zero_shot.v1"
VIEWS = ("full_inventory", "source_known_nyu")
FORMATS = ("coordinates_v2", "coordinates_relations_v2")
LETTERS = tuple("ABCDEFGHIJ")


def h(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def list_sha256(ids) -> str:
    return hashlib.sha256("".join(i + "\n" for i in ids).encode("utf-8")).hexdigest()


def select(remaining, salt: str, count: int) -> list:
    """The first `count` IDs sorted by SHA-256(salt + LF + ID), ties by ID."""
    return sorted(remaining, key=lambda p: (h(salt + "\n" + p), p))[:count]


def _order(salt: str, parent: str, view: str, objects) -> list:
    return sorted(objects, key=lambda o: (h(salt + "\n" + parent + "\n" + view + "\n" + o), o))


def code_assignment(policy, parent: str, view: str, objects) -> dict:
    """{object ID: letter}: the k-th object in the codes stream's order gets the k-th letter."""
    if len(objects) > len(LETTERS) or len(set(objects)) != len(objects):
        raise ValueError("between 0 and 10 distinct objects are required")
    return {o: LETTERS[k] for k, o in enumerate(_order(policy["codes"]["salt"], parent, view, objects))}


def list_order(policy, parent: str, view: str, objects) -> list:
    return _order(policy["list_order"]["salt"], parent, view, objects)


def mapping_for(policy, proto, parent: str, view: str, objects) -> list:
    """The ordered choices [[letter, object ID], ...] with [K, ASK] last; no model, format or annotation enters."""
    codes = code_assignment(policy, parent, view, objects)
    return [[codes[o], o] for o in list_order(policy, parent, view, objects)] + [[proto["ask_code"], proto["ask_target"]]]


def request_id(k: int) -> str:
    return f"r{k:04d}"


def load_policy(path=None) -> dict:
    p = Path(path) if path is not None else POLICY_PATH
    try:
        pol = strict_json("compare policy", p.read_bytes(), "E_COMPARE_POLICY")
    except OSError as e:
        raise EvaluationInputError([issue(str(p), "E_COMPARE_POLICY", f"cannot read: {e}")]) from None
    bad = []
    if not isinstance(pol, dict) or pol.get("record_type") != "iref_compare_policy" or pol.get("policy_id") != POLICY_ID:
        bad.append(f"not the {POLICY_ID} policy")
    else:
        for k in ("population", "selection", "codes", "list_order", "models", "context_limit_tokens", "expected"):
            if k not in pol:
                bad.append(f"missing {k}")
        sel = pol.get("selection", {})
        if type(sel.get("count")) is not int or sel.get("count", 0) < 1 or not isinstance(sel.get("salt"), str):
            bad.append("selection needs a plain integer count and a salt")
        if len({pol.get("selection", {}).get("salt"), pol.get("codes", {}).get("salt"), pol.get("list_order", {}).get("salt")}) != 3:
            bad.append("the selection, code and list-order streams need three different salts")
    if bad:
        raise EvaluationInputError([issue(str(p), "E_COMPARE_POLICY", m) for m in bad])
    return pol


def policy_sha256(path=None) -> str:
    return hashlib.sha256(Path(path if path is not None else POLICY_PATH).read_bytes()).hexdigest()


def plain_int(x) -> bool:
    return type(x) is int


def json_line(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
