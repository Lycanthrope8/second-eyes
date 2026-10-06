"""The A2.3b scoring policy (D82) and the helpers its two operations share.

The policy file (scoring-policy.v1.json) fixes the views, formats, outcome names, the derived-score tolerance, the
diagnostic alias, the exit codes and the four error codes. Records of this increment are checked against
schemas/iref-pilot-scoring.v1.json; their versions, counts and counters must also be plain integers, so a
whole-valued float such as 32.0 is rejected, never coerced.
"""
from __future__ import annotations

import copy
import json
import os
import platform
import shutil
import tempfile
from functools import lru_cache
from importlib import metadata
from pathlib import Path

import jsonschema

from ..iref_vla import output
from ..iref_vla.protocol import (EvaluationInputError, EvaluationOutputError, encode_json, encode_jsonl, issue, ratio,
                                 sha256, strict_json)

REPO = Path(__file__).resolve().parents[3]
POLICY_PATH = Path(__file__).resolve().parent / "scoring-policy.v1.json"
SCHEMA_PATH = REPO / "schemas" / "iref-pilot-scoring.v1.json"
POLICY_ID = "iref.pilot.scoring.v1"
VIEWS = ("full_inventory", "source_known_nyu")
FORMATS = ("coordinates_v2", "coordinates_relations_v2")
OUTCOMES = ("resolved", "ambiguous", "no_match", "insufficient_information", "unsupported")
TECHNICAL = ("invalid_input", "budget_exceeded")
RULES_BINS = OUTCOMES + TECHNICAL
CODES = tuple("ABCDEFGHIJK")
INPUT, INTEGRITY, REFERENCE, INTERNAL = ("E_SCORING_INPUT", "E_SCORING_INTEGRITY", "E_SCORING_REFERENCE",
                                         "E_SCORING_INTERNAL")
MODES = ("pinned_sample", "fixture")
CODE_ROOTS = ("grounding/evaluation", "grounding/inference/iref_vla", "grounding/preparation/iref_vla",
              "grounding/resolution", "grounding/relations", "grounding/contract")
CODE_FILES = ("grounding/adapters/iref_vla/pinned.py", "grounding/adapters/iref_vla/convert.py")
SCHEMAS = ("iref-pilot-scoring.v1.json", "iref-zero-shot-pilot.v1.json", "iref-evaluation.v1.json",
           "iref-annotations.v1.json", "iref-model-input-audit.v1.json", "grounding-query.v1.json",
           "grounding-result.v1.json", "resolver-config.v1.json")


class ScoringInternalError(RuntimeError):
    """An output failed its own readback before publication: a defect here, not bad input (exit 3)."""

    code = INTERNAL


def fail(code, problems):
    """Raise the CLI's exit-2 error with (path, message) problems under one code."""
    raise EvaluationInputError([issue(p, code, m) for p, m in problems])


def kind(value) -> str:
    if value is None:
        return "null"
    return {bool: "a boolean", int: "an integer", float: "a number", str: "a string", list: "an array",
            dict: "an object"}.get(type(value), type(value).__name__)


def plain_int(x) -> bool:
    return type(x) is int


def read_bytes(path, label, code=INPUT) -> bytes:
    try:
        return Path(path).read_bytes()
    except OSError as e:
        fail(code, [(str(path), f"cannot read the {label}: {e}")])


def read_json(path, label, code=INPUT):
    """Strict JSON: UTF-8 without a byte-order mark, no repeated keys, no NaN or Infinity."""
    return strict_json(label, read_bytes(path, label, code), code)


def read_jsonl(path, label, code=INPUT) -> list:
    data = read_bytes(path, label, code)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        fail(code, [(label, f"not UTF-8: {e}")])
    if text and not text.endswith("\n"):
        fail(code, [(label, "truncated: the last line has no LF")])
    return [strict_json(f"{label} line {k}", line.encode("utf-8"), code)
            for k, line in enumerate(text.split("\n")[:-1] if text else [], 1)]


@lru_cache(maxsize=None)
def _schema():
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def validator(name):
    schema = _schema()
    sub = {"$schema": schema["$schema"], "$defs": schema["$defs"], "$ref": f"#/$defs/{name}"}
    return jsonschema.Draft202012Validator(sub)


def _plain_int_paths(value, path=()):
    """Every number at a counting path (see schema descriptions) that is a float rather than a plain integer."""
    out = []
    if isinstance(value, dict):
        for k, v in value.items():
            out += _plain_int_paths(v, path + (k,))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            out += _plain_int_paths(v, path + (i,))
    elif isinstance(value, float) and value.is_integer() and path and path[-1] in INT_KEYS:
        out.append(path)
    return out


# Keys whose values are counts, counters, ranks, indices or versions: a float there is rejected, never coerced.
INT_KEYS = {"format_version", "selection_rank", "request_index", "input_tokens", "object_count", "annotation_count",
            "N", "C", "R", "completed", "object_choices", "ask", "planned_exclusions", "technical_failures",
            "numerator", "denominator", "parents", "parent_views", "requests", "annotation_records", "both",
            "coordinates_only", "augmented_only", "neither", "unscored", "same_choice", "different_choice",
            "not_comparable", "model_only", "rules_only", "same_object", "different_object", "offered",
            "source_is_alias", "unknown_category_choices", "records", "rules_records", "parse_records", "parsed",
            "distinct_texts", "score_rows", "comparison_rows", "selected_parents", "records_total",
            "records_selected", "model_planned_exclusions", "model_technical_failures", "rules_technical_failures"}


def schema_issues(name, record, where, code=INPUT) -> list:
    """Schema findings for one record, then the plain-integer rule for counting fields."""
    errors = sorted(validator(name).iter_errors(record), key=lambda e: [str(p) for p in e.absolute_path])
    out = [issue(f"{where} $" + "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in e.absolute_path), code,
                 e.message[:300]) for e in errors]
    if not out:
        out += [issue(f"{where} $." + ".".join(map(str, p)), code, "a whole-valued float where a plain integer is "
                                                                   "required") for p in _plain_int_paths(record)]
    return out


def require(name, record, where, code=INPUT):
    found = schema_issues(name, record, where, code)
    if found:
        raise EvaluationInputError(found)
    return record


def load_policy(path=None) -> dict:
    p = Path(path) if path is not None else POLICY_PATH
    data = read_bytes(p, "scoring policy")
    value = strict_json("scoring policy", data, INPUT)
    require("policy", value, "scoring policy")
    bad = []
    if value["policy_id"] != POLICY_ID or tuple(value["views"]) != VIEWS or tuple(value["formats"]) != FORMATS:
        bad.append("the policy's identity, views or formats are not this version's")
    if tuple(value["rules_outcomes"]) != OUTCOMES or tuple(value["rules_technical_failures"]) != TECHNICAL:
        bad.append("the policy's resolver outcome names are not this version's")
    if value["diagnostic_alias"] not in CODES[:-1]:
        bad.append("the diagnostic alias must be one of the object codes A..J")
    if bad:
        fail(INPUT, [(str(p), m) for m in bad])
    return value


def policy_sha256(path=None) -> str:
    return sha256(read_bytes(Path(path) if path is not None else POLICY_PATH, "scoring policy"))


def semantic_rules_hash(records) -> str:
    """SHA-256 of the canonical JSONL stream with exactly resolution.diagnostics.timings_s removed from each copy."""
    projected = []
    for record in records:
        r = copy.deepcopy(record)
        d = r.get("resolution", {}).get("diagnostics") if isinstance(r, dict) else None
        if isinstance(d, dict):
            d.pop("timings_s", None)
        projected.append(r)
    return sha256(encode_jsonl(projected))


def code_hashes() -> dict:
    out = {}
    for root in CODE_ROOTS:
        for p in sorted((REPO / root).rglob("*")):
            if p.is_file() and p.suffix in (".py", ".json") and "__pycache__" not in p.parts:
                out[p.relative_to(REPO).as_posix()] = sha256(p.read_bytes())
    for rel in CODE_FILES:
        out[rel] = sha256((REPO / rel).read_bytes())
    for name in SCHEMAS:
        out[f"schemas/{name}"] = sha256((REPO / "schemas" / name).read_bytes())
    return dict(sorted(out.items()))


def runtime() -> dict:
    return {"python": platform.python_version(), "jsonschema": metadata.version("jsonschema")}


def check_overlap(out: Path, inputs) -> None:
    o = Path(out).resolve()
    for p in inputs:
        q = Path(p).resolve()
        if o == q or o in q.parents or q in o.parents:
            fail(INPUT, [(str(out), f"the output folder overlaps the input {p}")])


def publish_verified(out: Path, files: dict, verify) -> None:
    """Write every file into a new sibling folder, read it back with verify(), then rename it into place.

    A readback problem is an internal failure (nothing published); a write failure is an EvaluationOutputError."""
    out = Path(out)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=str(out.parent)))
    except OSError as e:
        raise EvaluationOutputError([issue(str(out), "E_EVAL_OUTPUT_IO", f"{type(e).__name__}: {e}")]) from e
    try:
        for rel, data in files.items():
            target = staging / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            output._write_file(target, data)
        problems = verify(staging)
        if problems:
            raise ScoringInternalError(f"{INTERNAL}: the output failed its own readback: " + "; ".join(problems[:5]))
        os.rename(staging, out)
    except OSError as e:
        shutil.rmtree(staging, ignore_errors=True)
        raise EvaluationOutputError([issue(str(out), "E_EVAL_OUTPUT_IO",
                                           f"{type(e).__name__}: {e}; nothing was published")]) from e
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def file_hashes(paths: dict) -> dict:
    """SHA-256 of named input files, read now; used to show the inputs did not change during a run."""
    return {name: sha256(read_bytes(p, name, INTEGRITY)) for name, p in sorted(paths.items())}


__all__ = ["CODES", "FORMATS", "INPUT", "INTEGRITY", "INTERNAL", "MODES", "OUTCOMES", "POLICY_ID", "POLICY_PATH",
           "REFERENCE", "RULES_BINS", "SCHEMA_PATH", "TECHNICAL", "VIEWS", "EvaluationInputError",
           "EvaluationOutputError", "ScoringInternalError", "check_overlap", "code_hashes", "encode_json",
           "encode_jsonl", "fail", "file_hashes", "kind", "load_policy", "plain_int", "policy_sha256",
           "publish_verified", "ratio", "read_bytes", "read_json", "read_jsonl", "require", "runtime",
           "schema_issues", "semantic_rules_hash", "sha256", "validator"]
