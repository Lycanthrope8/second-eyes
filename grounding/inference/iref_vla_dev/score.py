"""A2.6c scoring: the established outcomes, per request for the models and per parent and view for the rules.

The targets (`reference_only/targets.jsonl`) are read here, and nowhere else. Each model run is first read back against
the frozen requests.

**Models** (A2.3d's `outcome_of`, unchanged):

- `context_budget_exceeded` or `execution_failed` when the request did not complete;
- `ask` when the model chose K. An ASK on a uniquely annotated command is not correct, and is never a technical failure;
- otherwise `correct` (the chosen object is the published target) or `wrong_object`.

**Rules:** `correct` when the resolver resolved to the published target. Otherwise its own outcome:

- `resolved_other_object`;
- the resolver's bins (`ambiguous`, `insufficient_information`, `unsupported` and so on);
- a technical failure.

The rules give one result per parent and view, shared by both formats, and are scored once per parent and view.

Every sampled parent keeps its row whatever its outcome: a missing or failed evaluation stays in the denominator.
`target_offered` records whether the target is in that view's selection, as a diagnostic.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from ...evaluation.iref_vla.protocol import EvaluationInputError, encode_json, encode_jsonl, issue, runtime
from ..iref_vla_compare.score import outcome_of
from . import preflight as PF
from . import runs as RN


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _jl(p) -> list:
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def _fail(where, message):
    raise EvaluationInputError([issue(str(where), "E_A26C_SCORE", message)])


def rules_outcome(rule, target) -> str:
    if rule["technical_failure"]:
        return f"technical:{rule['outcome']}"
    if rule["outcome"] == "resolved":
        return "correct" if rule["target_id"] == target else "resolved_other_object"
    return rule["outcome"]


def score(*, prep, rules, runs: dict, out, frozen=PF.FROZEN) -> dict:
    """runs: model key -> that model's run folder. Writes scores.jsonl (one row per request) and rules-scores.jsonl (one
    row per parent and view)."""
    prep, out = Path(prep), Path(out)
    if out.exists():
        _fail(out, "the output folder exists")
    pman = json.loads((prep / "manifest.json").read_text(encoding="utf-8"))
    req = prep / "requests.jsonl"   # absent from a review archive: the manifest's recorded hash stands in for it
    req_sha = _sha(req.read_bytes()) if req.is_file() else pman["hashes"]["requests"]
    if req_sha != frozen["requests_sha256"] or pman["hashes"]["requests"] != frozen["requests_sha256"]:
        _fail(prep, "not the frozen request file")
    for name in ("requests-index.jsonl", "reference_only/targets.jsonl"):
        if _sha((prep / name).read_bytes()) != pman["files"][name]:
            _fail(prep / name, "changed since the preparation")
    index = _jl(prep / "requests-index.jsonl")
    targets = {t["parent_command_id"]: t for t in _jl(prep / "reference_only" / "targets.jsonl")}
    results = {}
    for key, folder in runs.items():
        bad = RN.verify_results(folder, prep)
        if bad:
            _fail(folder, f"{key}: the run does not read back: {bad[:3]}")
        results[key] = {r["request_id"]: r for r in _jl(Path(folder) / "results.jsonl")}
    rman = json.loads((Path(rules) / "manifest.json").read_text(encoding="utf-8"))
    for name, h in rman["outputs"].items():
        if _sha((Path(rules) / name).read_bytes()) != h:
            _fail(rules, f"{name} changed")
    rules_by = {(r["parent_command_id"], r["view_id"]): r for r in _jl(Path(rules) / "rules.jsonl")}
    rows, rule_rows = [], []
    for x in index:
        t = targets[x["parent_command_id"]]
        target = t["target_object_id"]
        row = {k: x[k] for k in ("request_id", "partition", "group", "scene", "stratum", "parent_command_id", "view", "format")}
        row["target_offered"] = t["target_offered"][x["view"]]
        row["models"] = {k: {"technical_status": results[k][x["request_id"]]["technical_status"],
                             "choice_code": results[k][x["request_id"]]["choice_code"],
                             "outcome": outcome_of(results[k][x["request_id"]], target)} for k in results}
        rows.append(row)
    seen = set()
    for x in index:
        pv = (x["parent_command_id"], x["view"])
        if pv in seen:
            continue
        seen.add(pv)
        if pv not in rules_by:
            _fail(rules, f"no rules result for {pv}")
        r = rules_by[pv]
        rule_rows.append({k: x[k] for k in ("partition", "group", "scene", "stratum", "parent_command_id", "view")}
                         | {"outcome": rules_outcome(r, targets[x["parent_command_id"]]["target_object_id"]),
                            "target_offered": targets[x["parent_command_id"]]["target_offered"][x["view"]], "observations": 1})
    counts = {k: dict(Counter(r["models"][k]["outcome"] for r in rows)) for k in results}
    counts["rules"] = dict(Counter(r["outcome"] for r in rule_rows))
    files = {"scores.jsonl": encode_jsonl(rows), "rules-scores.jsonl": encode_jsonl(rule_rows)}
    manifest = {"format_version": 1, "record_type": "a26c_scores_manifest", "run_id": PF.RUN_ID, "frozen": dict(frozen),
                "inputs": {"requests_index_sha256": _sha((prep / "requests-index.jsonl").read_bytes()),
                           "targets_sha256": _sha((prep / "reference_only" / "targets.jsonl").read_bytes()),
                           "rules_manifest_sha256": _sha((Path(rules) / "manifest.json").read_bytes()),
                           "runs": {k: _sha((Path(f) / "manifest.json").read_bytes()) for k, f in runs.items()}},
                "counts": counts, "outputs": {k: _sha(v) for k, v in files.items()},
                "code": {p.name: _sha(p.read_bytes()) for p in sorted(Path(__file__).parent.glob("*.py"))},
                "runtime": runtime()}
    out.mkdir(parents=True)
    for k, v in files.items():
        (out / k).write_bytes(v)
    (out / "manifest.json").write_bytes(encode_json(manifest))
    return manifest
