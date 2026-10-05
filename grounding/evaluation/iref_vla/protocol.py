"""The A2.2b evaluation protocol, record checks and encodings, shared by prediction and scoring (D75).

The tracked protocol configuration (protocol.v1.json) fixes the colours, the relation aliases, the fixed action, the
two inventory views, the resolver's identity and allowances, the expected library identities and the semantic-hash
rule. The evaluation's own records are checked against schemas/iref-evaluation.v1.json; their version, count and
limit fields must also be plain integers, so a whole-valued float such as 1.0 is rejected, never coerced.
"""
from __future__ import annotations

import copy
import hashlib
import json
import platform
from importlib import metadata
from pathlib import Path

import jsonschema

from ...contract import validate as contract
from ...resolution import validate as RV

REPO = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = Path(__file__).resolve().parent / "protocol.v1.json"
SCHEMA_PATH = REPO / "schemas" / "iref-evaluation.v1.json"
RESULT_SCHEMA_PATH = REPO / "schemas" / "grounding-result.v1.json"
ANNOTATION_SCHEMA_PATH = REPO / "schemas" / "iref-annotations.v1.json"
EVALUATION_VERSION = "iref_vla_evaluation.v1"
VIEWS = ("full_inventory", "source_known_nyu")
BINS = ("resolved", "ambiguous", "no_match", "insufficient_information", "unsupported", "invalid_input", "budget_exceeded")
FEATURES = ("coreference", "plural_reference", "size_comparison")
LIMITS = ("max_interpretations", "max_nodes", "max_constraints", "max_candidate_checks", "max_binding_attempts",
          "max_predicate_calls", "max_rank_subset_evaluations", "max_trace_events", "max_trace_bytes")
QUERY_REASON = {"unrecognized_text": "unsupported_composition", "ambiguous_parse": "unsupported_composition",
                "plural_reference": "unsupported_plural", "coreference": "unsupported_coreference",
                "size_comparison": "unsupported_size_comparison"}
INT_FIELDS = {"protocol": [("format_version",)] + [("resolver", "limits", n) for n in LIMITS],
              "parse": [("format_version",), ("analysis_count",)],
              "prediction": [("format_version",)],
              "score": [("format_version",), ("annotation_count",)]}
CODE_ROOTS = ("grounding/evaluation", "grounding/resolution", "grounding/relations", "grounding/contract")


class EvaluationInputError(ValueError):
    """Invalid batch input, reference data or prediction output, or an existing destination (CLI exit 2)."""

    def __init__(self, issues):
        self.issues = [dict(i) for i in issues]
        super().__init__("; ".join(f"{i['path']}: {i['code']}: {i['message']}" for i in self.issues))


class EvaluationOutputError(OSError):
    """An output could not be written; nothing was published (CLI exit 3)."""

    def __init__(self, issues):
        self.issues = [dict(i) for i in issues]
        super().__init__("; ".join(f"{i['path']}: {i['code']}: {i['message']}" for i in self.issues))


def issue(path, code, message):
    return {"path": path, "code": code, "message": message}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode_json(record) -> bytes:
    return (json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def encode_line(record) -> str:
    return json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def encode_jsonl(records) -> bytes:
    return "".join(encode_line(r) + "\n" for r in records).encode("utf-8")


def ratio(numerator: int, denominator: int) -> dict:
    """A ratio kept with its parts; null, never 0, when the denominator is 0."""
    return {"numerator": numerator, "denominator": denominator,
            "value": None if denominator == 0 else numerator / denominator}


_VALIDATORS = {}


def _validator(name):
    if name not in _VALIDATORS:
        if name == "resolution":
            _VALIDATORS[name] = jsonschema.Draft202012Validator(json.loads(RESULT_SCHEMA_PATH.read_text(encoding="utf-8")))
        elif name == "annotations":
            _VALIDATORS[name] = jsonschema.Draft202012Validator(
                json.loads(ANNOTATION_SCHEMA_PATH.read_text(encoding="utf-8")))
        else:
            schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
            _VALIDATORS[name] = jsonschema.Draft202012Validator(
                {"$schema": schema["$schema"], "$defs": schema["$defs"], "$ref": f"#/$defs/{name}"})
    return _VALIDATORS[name]


def _path(parts) -> str:
    return "$" + "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in parts)


def schema_issues(name, record, where, code) -> list:
    """Schema findings for one record, then the plain-integer rule for its version, count and limit fields."""
    errors = sorted(_validator(name).iter_errors(record), key=lambda e: [str(p) for p in e.absolute_path])
    out = [issue(f"{where} {_path(e.absolute_path)}", code, e.message[:300]) for e in errors]
    if out or name not in INT_FIELDS:
        return out
    for path in INT_FIELDS[name]:
        value = record
        for key in path:
            value = value[key]
        if type(value) is not int:
            out.append(issue(f"{where} $.{'.'.join(path)}", code, f"{value!r} is not a plain integer"))
    return out


def strict_json(label, data: bytes, code):
    """Parse JSON as the contract does: UTF-8 without a byte-order mark, no repeated keys, no NaN or Infinity."""
    value, found = contract.parse(label, data)
    if found:
        raise EvaluationInputError([issue(f"{label} {i.path}", code, f"{i.code}: {i.message}") for i in found])
    return value


def load_protocol(path=PROTOCOL_PATH) -> dict:
    """The protocol configuration, checked; an EvaluationInputError (E_EVAL_PROTOCOL) if anything is off."""
    try:
        data = Path(path).read_bytes()
    except OSError as e:
        raise EvaluationInputError([issue(str(path), "E_EVAL_PROTOCOL", f"cannot read: {e}")]) from e
    value = strict_json("protocol", data, "E_EVAL_PROTOCOL")
    out = schema_issues("protocol", value, "protocol", "E_EVAL_PROTOCOL")
    if not out:
        views = value["views"]
        if [v["view_id"] for v in views] != list(VIEWS) or [v["command_suffix"] for v in views] != [".fi", ".kn"] \
                or [v["exclusion_reason"] for v in views] != [None, "source_unknown_category"]:
            out.append(issue("protocol $.views", "E_EVAL_PROTOCOL", "expected full_inventory (.fi) then "
                                                                    "source_known_nyu (.kn, source_unknown_category)"))
        phrases = [a["phrase"] for a in value["relation_aliases"]]
        if len(set(phrases)) != len(phrases):
            out.append(issue("protocol $.relation_aliases", "E_EVAL_PROTOCOL", "a phrase is listed twice"))
        for i, a in enumerate(value["relation_aliases"]):
            if (a["relation"] in ("closest", "farthest")) != (a["k"] is not None) or type(a["k"]) not in (int, type(None)):
                out.append(issue(f"protocol $.relation_aliases[{i}]", "E_EVAL_PROTOCOL",
                                 "ranking aliases need an integer k; other aliases need null"))
        if value["colour_labels"] != sorted(value["colour_labels"]):
            out.append(issue("protocol $.colour_labels", "E_EVAL_PROTOCOL", "colour labels must be sorted"))
    if out:
        raise EvaluationInputError(out)
    return value


def interpretation_issues(interpretation, where, code) -> list:
    """Check one interpretation with the accepted query validator, inside a placeholder query envelope."""
    envelope = {"schema_version": 1, "record_type": "grounding_query", "query_id": "parse.check",
                "scene_id": "parse.check", "scene_revision": 0, "evidence_profile": "annotated",
                "command_id": "parse.check", "interpretations": [interpretation]}
    found, _ = RV.check_query(envelope)
    return [issue(f"{where} {i['path']}", code, f"{i['code']}: {i['message']}") for i in found]


def parse_record_issues(record, where="parse") -> list:
    """The closed parse record's schema, its interpretation, and the section 3.4 consistency rules."""
    out = schema_issues("parse", record, where, "E_EVAL_PARSE")
    if out:
        return out
    interp = record["interpretation"]
    out = interpretation_issues(interp, f"{where} $.interpretation", "E_EVAL_PARSE")
    if out:
        return out
    reason, status, n = record["parse_reason"], record["parse_status"], record["analysis_count"]
    want_kind = "query" if status == "parsed" else "unsupported"
    if (status == "parsed") != (reason is None) or interp["kind"] != want_kind \
            or (status == "unsupported" and interp.get("reason") != QUERY_REASON.get(reason)) \
            or interp["interpretation_id"] != "main" or interp["action"] != "INSPECT" \
            or record["features"] != sorted(record["features"]) or (status == "parsed" and record["features"]) \
            or not ((n == 0) if reason == "unrecognized_text" else (n >= 2) if reason == "ambiguous_parse" else (n == 1)):
        out.append(issue(where, "E_EVAL_PARSE", "status, reason, features, analysis count and interpretation disagree"))
    return out


def validate_parse_record(record) -> dict:
    found = parse_record_issues(record)
    if found:
        raise EvaluationInputError(found)
    return record


def semantic_hash(predictions) -> str:
    """SHA-256 of the canonical JSONL stream with exactly resolution.diagnostics.timings_s removed from each copy."""
    projected = []
    for record in predictions:
        r = copy.deepcopy(record)
        diagnostics = r.get("resolution", {}).get("diagnostics")
        if isinstance(diagnostics, dict):
            diagnostics.pop("timings_s", None)
        projected.append(r)
    return sha256(encode_jsonl(projected))


def code_hashes() -> dict:
    """SHA-256 of the evaluation, resolver, relation and contract code, keyed by repository-relative path."""
    out = {}
    for root in CODE_ROOTS:
        for p in sorted((REPO / root).rglob("*")):
            if p.is_file() and p.suffix in (".py", ".json") and "__pycache__" not in p.parts:
                out[p.relative_to(REPO).as_posix()] = sha256(p.read_bytes())
    for name in ("iref-evaluation.v1.json", "grounding-query.v1.json", "grounding-result.v1.json",
                 "resolver-config.v1.json", "iref-annotations.v1.json"):
        out[f"schemas/{name}"] = sha256((REPO / "schemas" / name).read_bytes())
    return out


def runtime() -> dict:
    return {"python": platform.python_version(), "jsonschema": metadata.version("jsonschema")}
