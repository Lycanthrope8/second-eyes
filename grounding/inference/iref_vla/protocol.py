"""The A2.3a pilot protocol (D81): one versioned file, loaded strictly."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ...evaluation.iref_vla.protocol import EvaluationInputError, issue, strict_json

PROTOCOL_PATH = Path(__file__).resolve().parent / "pilot-protocol.v1.json"
PROTOCOL_ID = "iref.zero_shot.direct_selection.pilot.v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _int(x) -> bool:
    return type(x) is int


def load_protocol(path=None) -> dict:
    p = Path(path) if path is not None else PROTOCOL_PATH
    try:
        data = p.read_bytes()
    except OSError as e:
        raise EvaluationInputError([issue(str(p), "E_PILOT_PROTOCOL", f"cannot read the protocol: {e}")]) from None
    proto = strict_json("pilot protocol", data, "E_PILOT_PROTOCOL")
    bad = []
    if not isinstance(proto, dict):
        bad.append("the protocol must be a JSON object")
    else:
        if proto.get("record_type") != "iref_zero_shot_pilot_protocol" or not _int(proto.get("format_version")) \
                or proto["format_version"] != 1:
            bad.append("not a version-1 pilot protocol (record_type, plain integer format_version 1)")
        if proto.get("protocol_id") != PROTOCOL_ID:
            bad.append(f"protocol_id must be {PROTOCOL_ID}")
        if proto.get("object_codes") != list("ABCDEFGHIJ") or proto.get("ask_code") != "K" or proto.get("ask_target") != "ASK":
            bad.append("codes must be A..J for objects and K for ASK")
        ids = proto.get("code_token_ids")
        if not isinstance(ids, dict) or sorted(ids) != list("ABCDEFGHIJK") \
                or not all(_int(v) and v >= 0 for v in ids.values()) or len(set(ids.values())) != 11:
            bad.append("code_token_ids must give eleven distinct plain non-negative integers for A..K")
        sel = proto.get("selection") if isinstance(proto.get("selection"), dict) else {}
        if not (isinstance(sel.get("salt"), str) and _int(sel.get("count")) and sel["count"] > 0
                and _int(sel.get("expected_eligible")) and isinstance(sel.get("expected_selected_sha256"), str)
                and HEX64.match(sel["expected_selected_sha256"])):
            bad.append("selection needs a salt, a plain positive count, a plain expected_eligible and a 64-hex "
                       "expected_selected_sha256")
        if not (_int(proto.get("context_limit_tokens")) and proto["context_limit_tokens"] > 1
                and _int(proto.get("continuation_tokens")) and proto["continuation_tokens"] == 1):
            bad.append("context_limit_tokens must be a plain integer above 1 and continuation_tokens the integer 1")
        if not (isinstance(proto.get("views"), list) and isinstance(proto.get("formats"), list)
                and proto.get("request_order") == [[v, f] for v in proto["views"] for f in proto["formats"]]):
            bad.append("request_order must list each view with each format, views outer")
        w = proto.get("wrapper") if isinstance(proto.get("wrapper"), dict) else {}
        if not all(isinstance(w.get(k), str) for k in ("before_system", "between", "after_user")) \
                or not isinstance(proto.get("system_message"), str):
            bad.append("wrapper and system_message must be strings")
        c = proto.get("canary") if isinstance(proto.get("canary"), dict) else {}
        if not all(type(c.get(k)) is float and c[k] > 0 for k in ("atol", "rtol")):
            bad.append("canary atol and rtol must be positive numbers")
        ceil = proto.get("input_only_ceilings")
        if not (isinstance(ceil, list) and ceil and all(_int(x) and x > 0 for x in ceil)):
            bad.append("input_only_ceilings must be plain positive integers")
    if bad:
        raise EvaluationInputError([issue(str(p), "E_PILOT_PROTOCOL", m) for m in bad])
    return proto


def protocol_sha256(path=None) -> str:
    return hashlib.sha256(Path(path if path is not None else PROTOCOL_PATH).read_bytes()).hexdigest()
