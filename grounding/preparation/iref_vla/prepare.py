"""A2.2d: materialize A2.2c's fitting selections and measure both accepted formats (D78).

Inputs are exactly the four A2.2a originals, the accepted A2.2c selection audit, the two library configurations, the pinned
tokenizer files and the model description. The audit is trusted only after its artifact hashes, its recorded input hashes,
the D77 parent-scene hash and a full recomputation from the originals all agree with it; any forgery fails before a render
or a token is counted. Each fitting command/view pair becomes a derived scene (one per selection ID, shared) and a derived
command, rendered by the accepted serializer in both formats without its token checker, wrapped in the fixed sizing wrapper
and measured. No annotation, statement, graph, prediction or score is read, and no language model is run. Rendered texts
are sizing diagnostics, not accepted inference requests. Token ceilings are declared analysis scenarios, not budgets.
"""
from __future__ import annotations

import copy
import json
import math
import os
import shutil
import tempfile
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import jsonschema

from ...contract import validate as contract
from ...evaluation.iref_vla import output
from ...evaluation.iref_vla.protocol import (EvaluationInputError, EvaluationOutputError, encode_json, encode_jsonl,
                                             issue, load_protocol, PROTOCOL_PATH, ratio, runtime, sha256, strict_json)
from ...relations import directions as D
from ...relations import predicates as Pr
from ...resolution.validate import canonical_sha256
from ...serialization import serialize
from ...serialization.constants import work_slots
from ...subscenes.iref_vla.audit import POLICY_ID as SOURCE_POLICY
from ...subscenes.iref_vla.audit import _read_commands, parent_scene_sha256
from ...subscenes.iref_vla.audit import _validator as audit_validator
from ...subscenes.iref_vla.audit import audit as recompute_audit
from . import tokens as T

REPO = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO / "schemas" / "iref-model-input-audit.v1.json"
POLICY = "iref_model_inputs.v1"
BASELINE = "c5b7704"
VIEWS = ("full_inventory", "source_known_nyu")
FORMATS = T.FORMATS
CEILINGS = T.CEILINGS
OBJECT_LIMIT = 10
WORK_CAP = work_slots(OBJECT_LIMIT)  # 1,190: the accepted formula at n = 10, an offline construction bound
LIBRARIES = {"relation_config": "relations.v1@d922fe902669", "direction_config": "directions.v1@73590e939d4f"}
AUDIT_FILES = ("selections.jsonl", "summary.json", "report.md", "manifest.json")
TEXT_FILES = {"document": "document.jsonl", "static_prefix": "static_prefix.jsonl", "dynamic_suffix": "dynamic_suffix.jsonl",
              "prompt": "prompt.txt"}
LIMITATIONS = [
    "One inspected development room; distinct commands and selections are not independent rooms.",
    "Oracle annotated geometry (the annotated evidence profile only); no perception error.",
    "Selections are parser-conditioned and category-conditioned (A2.2c), not spatial or visibility crops.",
    "Unassessed and over-limit commands stay in the population but produce no model input.",
    "No model was run: no accuracy, no output contract and no output or scoring reserve yet.",
    "No natural-command validation; the sample's commands are the dataset's own phrasings.",
    "Token counts are exact for the pinned Hugging Face tokenizer and this sizing wrapper; parity with the deployed "
    "Q8_0 GGUF tokenizer was not executed.",
    "No Quest latency, peak-memory or KV-cache measurement; input-only ceilings are analysis scenarios, not budgets.",
    "This sample carries no viewer pose; its unknown-pose behaviour does not transfer automatically to perspective "
    "commands.",
]


# ---------------------------------------------------------------------------------------------------- helpers
@lru_cache(maxsize=None)
def _validator(name):
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    sub = {"$schema": schema["$schema"], "$defs": schema["$defs"], "$ref": f"#/$defs/{name}"}
    return jsonschema.validators.validator_for(sub)(sub)


def _same(a, b) -> bool:
    """Type-aware deep equality: 1.0 is not 1, True is not 1, and list order counts."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _plain(x):
    return json.loads(json.dumps(x, allow_nan=False))


def _jsonl(label, data: bytes, code) -> list:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise EvaluationInputError([issue(label, code, f"not UTF-8: {e}")]) from None
    if text and not text.endswith("\n"):
        raise EvaluationInputError([issue(label, code, "truncated: the last line has no LF")])
    return [strict_json(f"{label} line {k}", line.encode("utf-8"), code)
            for k, line in enumerate(text.split("\n")[:-1] if text else [], 1)]


def _stats(values) -> dict:
    v = sorted(values)
    n = len(v)
    if not n:
        return {"n": 0, "min": None, "median": None, "p95": None, "max": None}
    if n % 2:
        median = v[n // 2]
    else:
        s = v[n // 2 - 1] + v[n // 2]
        median = s // 2 if s % 2 == 0 else s / 2
    return {"n": n, "min": v[0], "median": median, "p95": v[math.ceil(0.95 * n) - 1], "max": v[-1]}


def _aggregate(pairs) -> str:
    """SHA-256 of the sorted lines '<relative path> <sha256>\\n'."""
    return sha256("".join(sorted(f"{p} {h}\n" for p, h in pairs)).encode("utf-8"))


# ------------------------------------------------------------------------------------------------- inputs
def _read_bytes(path, label):
    try:
        return Path(path).read_bytes()
    except OSError as e:
        raise EvaluationInputError([issue(str(path), "E_EVAL_INPUT", f"cannot read the {label}: {e}")]) from None


def _read_originals(scene, commands, category_map, inventory_views, protocol, sample_only) -> dict:
    data = {"scene": _read_bytes(scene, "scene file"), "category_map": _read_bytes(category_map, "category map"),
            "inventory_views": _read_bytes(inventory_views, "inventory views")}
    rec = {k: strict_json(k, v, "E_EVAL_INPUT") for k, v in data.items()}
    for k in ("scene", "category_map"):
        if not isinstance(rec[k], dict):
            raise EvaluationInputError([issue(k, "E_EVAL_INPUT", f"the {k.replace('_', ' ')} file must hold one JSON object")])
    cmds, lines = _read_commands(Path(commands))
    if sample_only and (rec["scene"].get("scene_id") != protocol["sample"]["scene_id"]
                        or rec["category_map"].get("map_id") != protocol["sample"]["category_map_id"]):
        raise EvaluationInputError([issue("scene", "E_EVAL_INPUT", f"this command line prepares only the A2.2a sample "
                                                                   f"{protocol['sample']['scene_id']!r}")])
    return {"scene": rec["scene"], "category_map": rec["category_map"], "views": rec["inventory_views"], "commands": cmds,
            "hashes": {k: sha256(v) for k, v in data.items()},
            "commands_sha256": sha256("".join(sorted(lines)).encode("utf-8"))}


def _verify_audit(folder: Path, orig: dict, protocol) -> dict:
    """Trust the A2.2c audit only when its hashes, its recorded inputs and a recomputation from the originals agree."""
    def fail(messages):
        return EvaluationInputError([issue(str(folder), "E_PREP_AUDIT", m) for m in messages[:20]])
    if not folder.is_dir():
        raise fail(["the selection audit is not a folder"])
    names = sorted(p.name for p in folder.iterdir())
    if names != sorted(AUDIT_FILES):
        raise fail([f"the selection audit must hold exactly {list(AUDIT_FILES)}; found {names}"])
    raw = {n: (folder / n).read_bytes() for n in AUDIT_FILES}
    manifest = strict_json("selection audit manifest.json", raw["manifest.json"], "E_PREP_AUDIT")
    summary = strict_json("selection audit summary.json", raw["summary.json"], "E_PREP_AUDIT")
    rows = _jsonl("selection audit selections.jsonl", raw["selections.jsonl"], "E_PREP_AUDIT")
    problems = [f"manifest: {e.message}" for e in audit_validator("manifest").iter_errors(manifest)]
    problems += [f"summary: {e.message}" for e in audit_validator("summary").iter_errors(summary)]
    for k, r in enumerate(rows, 1):
        problems += [f"selections line {k}: {e.message}" for e in audit_validator("selection").iter_errors(r)]
    if problems:
        raise fail(problems)
    for n in ("selections.jsonl", "summary.json", "report.md"):
        if manifest["outputs"].get(n) != sha256(raw[n]):
            problems.append(f"the audit manifest's hash of {n} does not match the file")
    want = {"scene_sha256": orig["hashes"]["scene"], "category_map_sha256": orig["hashes"]["category_map"],
            "inventory_views_sha256": orig["hashes"]["inventory_views"], "commands_sha256": orig["commands_sha256"],
            "commands": len(orig["commands"])}
    for k, v in want.items():
        if not _same(manifest["inputs"].get(k), v):
            problems.append(f"the audit's recorded input {k} does not match the supplied originals")
    if manifest["parent_scene"].get("canonical_sha256") != parent_scene_sha256(orig["scene"]):
        problems.append("the audit's parent-scene hash does not match the supplied scene (D77 sorted-object rule)")
    if problems:
        raise fail(problems)
    again = recompute_audit(orig["scene"], orig["commands"], orig["category_map"], orig["views"], protocol=protocol)
    if not _same(rows, _plain(again.rows)):
        bad = next((k for k, (a, b) in enumerate(zip(rows, _plain(again.rows)), 1) if not _same(a, b)), len(rows) + 1)
        raise fail([f"the selection rows differ from a recomputation from the supplied originals (first at row {bad})"])
    if not _same(summary, _plain(again.summary)):
        raise fail(["the audit summary differs from a recomputation from the supplied originals"])
    return {"rows": rows, "hashes": {n: sha256(raw[n]) for n in AUDIT_FILES}}


def _check_libraries(relation_config, direction_config) -> dict:
    out = {}
    for label, path, loader in (("relation_config", relation_config, Pr.load_config),
                                ("direction_config", direction_config, D.load_direction_config)):
        b = _read_bytes(path, label.replace("_", " "))
        try:
            cfg = loader(Path(path))
        except (OSError, ValueError, jsonschema.exceptions.ValidationError) as e:
            raise EvaluationInputError([issue(str(path), "E_PREP_LIBRARY", f"{type(e).__name__}: {e}")]) from None
        identity = cfg.get("identity") if isinstance(cfg, dict) else getattr(cfg, "identity", None)
        if identity != LIBRARIES[label]:
            raise EvaluationInputError([issue(str(path), "E_PREP_LIBRARY", f"identity {identity!r} is not the accepted "
                                                                           f"{LIBRARIES[label]}")])
        out[label] = {"identity": identity, "sha256": sha256(b)}
    return out


def _check_overlap(out: Path, inputs) -> None:
    o = out.resolve()
    for p in inputs:
        q = Path(p).resolve()
        if o == q or o in q.parents or q in o.parents:
            raise EvaluationInputError([issue(str(out), "E_EVAL_INPUT", f"the output folder overlaps the input {p}")])


# ------------------------------------------------------------------------------------------ materialization
def derived_command_id(parent: dict, selection_id: str) -> str:
    payload = {"preparation_policy": POLICY, "parent_command_id": parent["command_id"],
               "parent_command_sha256": canonical_sha256(parent), "selection_id": selection_id}
    return "iref.input." + canonical_sha256(payload)


def materialize_scene(scene: dict, retained_ids, selection_id: str) -> dict:
    by_id = {o["object_id"]: o for o in scene["objects"]}
    s = {k: copy.deepcopy(v) for k, v in scene.items() if k != "objects"}
    s["scene_id"], s["scene_revision"] = selection_id, 0
    s["objects"] = [copy.deepcopy(by_id[i]) for i in sorted(retained_ids)]
    return s


def materialize_command(parent: dict, selection_id: str) -> dict:
    c = copy.deepcopy(parent)
    c["command_id"] = derived_command_id(parent, selection_id)
    c["scene_id"], c["scene_revision"] = selection_id, 0
    if isinstance(c.get("user_pose"), dict) and "scene_revision" in c["user_pose"]:
        c["user_pose"]["scene_revision"] = 0
    return c


def _check_preservation(parent_scene, scene, retained, parent_cmd, cmd) -> None:
    """Independent structural checks: only the declared fields may differ from the parents."""
    rest = lambda x, drop: {k: v for k, v in x.items() if k not in drop}  # noqa: E731
    by_id = {o["object_id"]: o for o in parent_scene["objects"]}
    if rest(scene, ("scene_id", "scene_revision", "objects")) != rest(parent_scene, ("scene_id", "scene_revision", "objects")):
        raise RuntimeError("a materialized scene differs from its parent outside its ID, revision and objects")
    if [o["object_id"] for o in scene["objects"]] != sorted(retained) or any(o != by_id[o["object_id"]] for o in scene["objects"]):
        raise RuntimeError("a materialized scene's objects are not exactly its planned original records")
    if cmd is not None:
        keep = ("command_id", "scene_id", "scene_revision", "user_pose")
        pose = lambda x: rest(x.get("user_pose") or {}, ("scene_revision",))  # noqa: E731
        if rest(cmd, keep) != rest(parent_cmd, keep) or pose(cmd) != pose(parent_cmd) or cmd["text"] != parent_cmd["text"]:
            raise RuntimeError("a derived command differs from its parent outside the four rebound fields")


# --------------------------------------------------------------------------------------------- measurement
def line_names(document: str) -> list:
    """Block names from each JSONL line's own identity: relation[@frame], measure, or the line's leading key."""
    names = []
    for line in document.split("\n")[:-1]:
        d = json.loads(line)
        if "relation" in d:
            names.append(d["relation"] if d.get("frame") is None else f"{d['frame']}.{d['relation']}")
        elif "measure" in d:
            names.append(d["measure"])
        else:
            names.append(next(iter(d)))
    if len(set(names)) != len(names):
        raise RuntimeError(f"a rendered document repeats a line identity: {names}")
    return names


def _encode(tok, text):
    ids = tok.encode(text)
    if not isinstance(ids, list) or any(type(i) is not int for i in ids):
        raise RuntimeError("the tokenizer returned something other than a list of integer IDs")
    return ids


def measure(tok, document: str, static: str, dynamic: str) -> dict:
    """Token measurements of one rendered document; the complete input is counted in one call."""
    prompt = T.wrap(document)
    ids = _encode(tok, prompt)
    prefix_ids = _encode(tok, T.HEAD + static)
    compatible = T.common_prefix_length(prefix_ids, ids)
    lines = document.split("\n")[:-1]
    text, cumulative = T.HEAD, [len(_encode(tok, T.HEAD))]
    for line in lines:
        text += line + "\n"
        cumulative.append(len(_encode(tok, text)))
    if text + T.TAIL != prompt:
        raise RuntimeError("block attribution did not reproduce the complete input")
    names = line_names(document)
    blocks = ([{"name": "wrapper_head", "marginal_tokens": cumulative[0]}]
              + [{"name": n, "marginal_tokens": cumulative[k + 1] - cumulative[k]} for k, n in enumerate(names)]
              + [{"name": "wrapper_tail", "marginal_tokens": len(ids) - cumulative[-1]}])
    stat = lambda s: {"bytes": len(s.encode("utf-8")), "sha256": sha256(s.encode("utf-8"))}  # noqa: E731
    return {"prompt": prompt, "prefix_ids": prefix_ids, "blocks": blocks, "non_nfc": not unicodedata.is_normalized("NFC", prompt),
            "tokens": {"input_tokens": len(ids), "input_bytes": len(prompt.encode("utf-8")),
                       "input_sha256": sha256(prompt.encode("utf-8")),
                       "token_ids_sha256": sha256(",".join(str(i) for i in ids).encode("ascii")),
                       "document_tokens": len(_encode(tok, document)), "document": stat(document),
                       "static_prefix": stat(static), "dynamic_suffix": stat(dynamic), "prefix_tokens": len(prefix_ids),
                       "prefix_compatible_tokens": compatible, "prefix_fully_compatible": compatible == len(prefix_ids),
                       "tokens_after_compatible_prefix": len(ids) - compatible}}


def _work(md) -> dict:
    w = md.get("work") or {}
    d = w.get("distances") or {}
    return {"planned_slots": w.get("planned_slots", 0), "cap": w.get("cap"), "status": w.get("status", "within_cap"),
            "actual_total": w.get("actual_total", 0),
            "distances": {"slots": d.get("slots", 0), "performed": d.get("performed", 0), "missing": d.get("missing", 0)}}


def _row(index_row, fmt, status, reason) -> dict:
    return {"format_version": 1, "record_type": "iref_model_input_measurement", "policy_id": POLICY,
            "parent_command_id": index_row["parent_command_id"], "view_id": index_row["view_id"], "format": fmt,
            "selection_id": index_row["selection_id"], "derived_scene_id": index_row["derived_scene_id"],
            "derived_command_id": index_row["derived_command_id"], "status": status, "reason": reason,
            "object_count": index_row["required_object_count"], "measurement_kind": None, "tokenizer_identity": None,
            "wrapper_identity": None, "serializer_token_status": None, "paths": None, "tokens": None, "work": None,
            "states_by_block": None, "blocks": None, "ceilings": None, "non_nfc": None}


# ---------------------------------------------------------------------------------------------- summarizing
def summarize(index_rows, meas, kind, cap) -> dict:
    views, paired = {}, {}
    for v in VIEWS:
        idx = [r for r in index_rows if r["view_id"] == v]
        n_all = len(idx)
        fit = sum(r["preparation_status"] == "materialized" for r in idx)
        views[v] = {}
        for f in FORMATS:
            ms = [m for m in meas if m["view_id"] == v and m["format"] == f]
            done = [m for m in ms if m["status"] == "rendered"]
            ceil = {}
            for c in CEILINGS:
                within = sum(m["ceilings"][str(c)] == "within_input_ceiling" for m in done)
                ceil[str(c)] = {"within": within, "exceeds": len(done) - within, "not_measured": n_all - len(done),
                                "within_of_all_commands": ratio(within, n_all), "within_of_fitting_pairs": ratio(within, fit),
                                "within_of_measured_inputs": ratio(within, len(done))}
            per_sel, groups, by_n, blocks = defaultdict(int), {}, defaultdict(list), defaultdict(list)
            for m in done:
                t = m["tokens"]
                per_sel[m["selection_id"]] = max(per_sel[m["selection_id"]], t["input_tokens"])
                groups[m["selection_id"]] = t["group_size"]
                by_n[m["object_count"]].append(t["input_tokens"])
                for b in m["blocks"]:
                    blocks[b["name"]].append(b["marginal_tokens"])
            views[v][f] = {
                "parent_commands": n_all, "fitting_pairs": fit,
                "over_object_budget": sum(r["preparation_status"] == "over_object_budget" for r in idx),
                "unassessed_parse": sum(r["preparation_status"] == "unassessed_parse" for r in idx),
                "rendered": len(done), "work_budget_exceeded": sum(m["status"] == "work_budget_exceeded" for m in ms),
                "measured_inputs": len(done), "ceilings": ceil,
                "input_tokens": _stats(m["tokens"]["input_tokens"] for m in done),
                "document_tokens": _stats(m["tokens"]["document_tokens"] for m in done),
                "tokens_after_compatible_prefix": _stats(m["tokens"]["tokens_after_compatible_prefix"] for m in done),
                "tokens_after_group_prefix": _stats(m["tokens"]["tokens_after_group_prefix"] for m in done),
                "group_reusable_prefix_tokens": _stats(m["tokens"]["group_reusable_prefix_tokens"] for m in done),
                "selection_level": {"distinct_selections": len(per_sel), "max_input_tokens_per_selection": _stats(per_sel.values())},
                "group_sizes": {str(k): n for k, n in sorted(Counter(groups.values()).items())},
                "by_object_count": {str(k): _stats(by_n[k]) for k in sorted(by_n)},
                "blocks": {name: _stats(vals) for name, vals in sorted(blocks.items())},
                "non_nfc_inputs": sum(bool(m["non_nfc"]) for m in done),
                "planned_work_slots": _stats(m["work"]["planned_slots"] for m in done)}
        pair = {}
        coords = {(m["parent_command_id"]): m for m in meas if m["view_id"] == v and m["format"] == FORMATS[0]}
        augs = {(m["parent_command_id"]): m for m in meas if m["view_id"] == v and m["format"] == FORMATS[1]}
        fitting = [r["parent_command_id"] for r in idx if r["preparation_status"] == "materialized"]
        for c in CEILINGS:
            k = {"both": 0, "only_coordinates": 0, "only_augmented": 0, "neither": 0, "work_failure": 0}
            for pid in fitting:
                a, b = coords[pid], augs[pid]
                if "work_budget_exceeded" in (a["status"], b["status"]):
                    k["work_failure"] += 1
                    continue
                wa, wb = (x["ceilings"][str(c)] == "within_input_ceiling" for x in (a, b))
                k["both" if wa and wb else "only_coordinates" if wa else "only_augmented" if wb else "neither"] += 1
            pair[str(c)] = k
        paired[v] = {"fitting_pairs": len(fitting), "ceilings": pair}
    scenes = {r["derived_scene_id"] for r in index_rows if r["derived_scene_id"]}
    population = {"parent_commands": len({r["parent_command_id"] for r in index_rows}), "pairs": len(index_rows),
                  "fitting_pairs": sum(r["preparation_status"] == "materialized" for r in index_rows),
                  "over_object_budget": sum(r["preparation_status"] == "over_object_budget" for r in index_rows),
                  "unassessed_parse": sum(r["preparation_status"] == "unassessed_parse" for r in index_rows),
                  "distinct_fitting_selections": {v: len({r["derived_scene_id"] for r in index_rows if r["view_id"] == v
                                                          and r["derived_scene_id"]}) for v in VIEWS},
                  "scene_files": len(scenes), "command_files": sum(bool(r["derived_command_id"]) for r in index_rows),
                  "rendered_documents": sum(m["status"] == "rendered" for m in meas), "measurement_rows": len(meas)}
    return {"format_version": 1, "record_type": "iref_model_input_summary", "policy_id": POLICY,
            "measurement_label": T.MEASUREMENT_LABEL if kind == "exact" else "test double: fixture measurement, not "
                                                                             "real-token evidence",
            "measurement_kind": kind, "ceilings": list(CEILINGS), "object_limit": OBJECT_LIMIT, "work_cap": cap,
            "population": population, "views": views, "paired": paired, "limitations": LIMITATIONS}


def _r(x):
    return "n/a" if x["value"] is None else f"{x['numerator']}/{x['denominator']} ({100 * x['value']:.1f}%)"


def render_report(s) -> str:
    p = s["population"]
    lines = ["# IRef-VLA model-input preparation and token measurement (A2.2d)", "",
             f"Measurement: {s['measurement_label']}.", "",
             "Rendered texts are sizing diagnostics under `diagnostic_inputs/`, not accepted inference requests. The "
             "input-only ceilings (1,024, 2,048 and 4,096 tokens, equality passing) are declared analysis scenarios, not "
             "the app's context allocation or latency budget. No format is eliminated and no winner is chosen on length.", "",
             "## Population", "",
             f"{p['parent_commands']} parent commands, {p['pairs']} command/view pairs: {p['fitting_pairs']} fitting, "
             f"{p['over_object_budget']} over the {s['object_limit']}-object limit, {p['unassessed_parse']} unassessed. "
             f"{p['scene_files']} scenes, {p['command_files']} derived commands, {p['rendered_documents']} rendered "
             f"documents, {p['measurement_rows']} measurement rows. Relation work cap: {s['work_cap']} slots.", ""]
    for v in VIEWS:
        lines += [f"## {v}", "", "| Format | Fitting | Rendered | Work failures | Median tokens | p95 | Max |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for f in FORMATS:
            x = s["views"][v][f]
            t = x["input_tokens"]
            lines.append(f"| {f} | {x['fitting_pairs']} | {x['rendered']} | {x['work_budget_exceeded']} | {t['median']} | "
                         f"{t['p95']} | {t['max']} |")
        lines += ["", "| Ceiling | Format | Within | Exceeds | Not measured | Of all commands | Of fitting | Of measured |",
                  "|---:|---|---:|---:|---:|---:|---:|---:|"]
        for c in CEILINGS:
            for f in FORMATS:
                x = s["views"][v][f]["ceilings"][str(c)]
                lines.append(f"| {c} | {f} | {x['within']} | {x['exceeds']} | {x['not_measured']} | "
                             f"{_r(x['within_of_all_commands'])} | {_r(x['within_of_fitting_pairs'])} | "
                             f"{_r(x['within_of_measured_inputs'])} |")
        lines += ["", "| Ceiling | Both formats within | Only coordinates | Only augmented | Neither | Work failure |",
                  "|---:|---:|---:|---:|---:|---:|"]
        for c in CEILINGS:
            k = s["paired"][v]["ceilings"][str(c)]
            lines.append(f"| {c} | {k['both']} | {k['only_coordinates']} | {k['only_augmented']} | {k['neither']} | "
                         f"{k['work_failure']} |")
        aug = s["views"][v][FORMATS[1]]["blocks"]
        top = sorted(((st["median"] or 0, name) for name, st in aug.items()), reverse=True)[:5]
        lines += ["", "Largest augmented blocks by median marginal tokens (order-dependent attribution): "
                  + ", ".join(f"{name} {m}" for m, name in top) + ".", ""]
    lines += ["## Cache interpretation", "",
              "Reusable-prefix figures are potential reuse, conditional on that selection's prefix already being cached. "
              "Moving to another selection can change the scene prefix, so no cross-selection cache hit is assumed and no "
              "hit rate is derived from the artificial command order. A one-command group shows no observed reuse.", "",
              "## Limitations", ""] + [f"- {x}" for x in s["limitations"]] + [
              "", "The earlier Linux/Windows prediction semantic-hash difference (A2.2b) remains unexplained and "
                  "nonblocking; this increment does not reopen it.", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------- publishing
def _relpath_ok(rel: str) -> bool:
    parts = rel.split("/")
    return bool(rel) and not rel.startswith("/") and "\\" not in rel and all(p not in ("", ".", "..") for p in parts)


def _put(staging: Path, rel: str, data: bytes, hashes: dict) -> None:
    if not _relpath_ok(rel):
        raise RuntimeError(f"refusing an unsafe output path {rel!r}")
    target = staging / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    output._write_file(target, data)
    hashes[rel] = sha256(data)


def _code_hashes() -> dict:
    files = sorted(p for d in ("grounding/preparation", "grounding/serialization", "grounding/subscenes/iref_vla")
                   for p in (REPO / d).rglob("*.py"))
    files += [REPO / "grounding/scene.py", REPO / "grounding/evaluation/iref_vla/parse.py",
              REPO / "grounding/evaluation/iref_vla/protocol.py", REPO / "grounding/evaluation/iref_vla/output.py", SCHEMA_PATH]
    return {p.relative_to(REPO).as_posix(): sha256(p.read_bytes()) for p in files}


def run_preparation(*, scene, commands, category_map, inventory_views, selection_audit, relation_config,
                    direction_config, tokenizer_dir, model_description, out, tokenizer=None, sample_only=True,
                    work_cap=None) -> dict:
    """The command line's operation; returns the summary. `tokenizer` and `work_cap` exist for fixture tests only."""
    out = output.refuse_existing(out)
    _check_overlap(out, (scene, commands, category_map, inventory_views, selection_audit, relation_config,
                         direction_config, tokenizer_dir, model_description))
    protocol = load_protocol()
    desc_bytes = _read_bytes(model_description, "model description")
    desc = strict_json("model description", desc_bytes, "E_PREP_MODEL")
    T.check_model_description(desc)
    orig = _read_originals(scene, commands, category_map, inventory_views, protocol, sample_only)
    audit = _verify_audit(Path(selection_audit), orig, protocol)
    libraries = _check_libraries(relation_config, direction_config)
    cap = WORK_CAP if work_cap is None else work_cap
    if tokenizer is None:
        tok = T.load_pinned_tokenizer(tokenizer_dir)
        cal = T.calibrate(tok, desc)
        T.check_calibration(cal)
        cal = dict(cal, status="passed before rendering")
    else:
        tok, cal = tokenizer, {"status": "not run: fixture test double"}
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=str(out.parent)))
    try:
        summary = _prepare(staging, orig, audit, tok, cap, Path(relation_config), Path(direction_config), {
            "libraries": libraries, "calibration": cal, "protocol": protocol, "desc": desc, "desc_sha256": sha256(desc_bytes)})
        problems = verify_bundle(staging)
        if problems:
            raise RuntimeError("the bundle failed readback before publication: " + "; ".join(problems[:5]))
        os.rename(staging, out)
    except OSError as e:
        shutil.rmtree(staging, ignore_errors=True)
        raise EvaluationOutputError([issue(str(out), "E_EVAL_OUTPUT_IO", f"{type(e).__name__}: {e}")]) from e
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def _prepare(staging, orig, audit, tok, cap, rel_cfg, dir_cfg, info) -> dict:
    scene, cmap = orig["scene"], orig["category_map"]
    parents = {c["command_id"]: c for c in orig["commands"]}
    hashes, index_rows, scenes, commands, members = {}, [], {}, {}, defaultdict(list)
    _put(staging, "model_records/category-map.json", encode_json(cmap), hashes)
    for r in audit["rows"]:
        row = {"format_version": 1, "record_type": "iref_model_input_index", "policy_id": POLICY,
               "parent_command_id": r["parent_command_id"], "view_id": r["view_id"], "selection_id": r["selection_id"],
               "source_status": r["status"], "source_reason": r["parse_reason"],
               "required_object_count": r["required_object_count"], "planned_object_ids": r["retained_object_ids"],
               "preparation_status": {"fits": "materialized", "over_budget": "over_object_budget"}.get(r["status"], r["status"]),
               "derived_scene_id": None, "derived_command_id": None, "scene_path": None, "command_path": None,
               "scene_sha256": None, "command_sha256": None}
        if r["status"] == "fits":
            sel, parent = r["selection_id"], parents[r["parent_command_id"]]
            s = materialize_scene(scene, r["retained_object_ids"], sel)
            if sel in scenes and scenes[sel] != s:
                raise RuntimeError(f"selection {sel} implies two different scenes")
            scenes[sel] = s
            c = materialize_command(parent, sel)
            _check_preservation(scene, s, r["retained_object_ids"], parent, c)
            commands[c["command_id"]] = c
            members[sel].append(c["command_id"])
            row.update(derived_scene_id=sel, derived_command_id=c["command_id"], scene_path=f"model_records/scenes/{sel}.json",
                       command_path=f"model_records/commands/{c['command_id']}.json", scene_sha256=canonical_sha256(s),
                       command_sha256=canonical_sha256(c))
        index_rows.append(row)
    for sel, s in scenes.items():  # validate each scene with its map and every command bound to it
        found = contract.validate([("scene", s), ("map", cmap)] + [("command", commands[c]) for c in members[sel]])
        if found:
            raise RuntimeError(f"materialized selection {sel} failed the contract: " + "; ".join(i.line() for i in found[:3]))
        _put(staging, f"model_records/scenes/{sel}.json", encode_json(s), hashes)
    for cid, c in commands.items():
        _put(staging, f"model_records/commands/{cid}.json", encode_json(c), hashes)
    meas, groups = [], defaultdict(list)
    for row in index_rows:
        if row["preparation_status"] != "materialized":
            reason = row["source_reason"] if row["preparation_status"] == "unassessed_parse" else "required_object_count_above_limit"
            meas += [_row(row, f, row["preparation_status"], reason) for f in FORMATS]
            continue
        s, c = scenes[row["derived_scene_id"]], commands[row["derived_command_id"]]
        for f in FORMATS:
            res = serialize(s, c, format=f, relation_config_path=rel_cfg, direction_config_path=dir_cfg,
                            max_relation_work_units=None if f == FORMATS[0] else cap, category_maps=[cmap])
            m = _row(row, f, None, None)
            if res.status == "work_budget_exceeded":
                m.update(status="work_budget_exceeded", reason="relation_work_above_cap", work=_work(res.metadata))
                meas.append(m)
                continue
            if res.status != "ok" or res.static_prefix + res.dynamic_suffix != res.document:
                raise RuntimeError(f"unexpected serializer outcome {res.status!r} for {row['derived_command_id']} {f}")
            got = measure(tok, res.document, res.static_prefix, res.dynamic_suffix)
            base = f"diagnostic_inputs/{row['derived_command_id']}/{f}/"
            paths = {k: base + name for k, name in TEXT_FILES.items()}
            for k, text in (("document", res.document), ("static_prefix", res.static_prefix),
                            ("dynamic_suffix", res.dynamic_suffix), ("prompt", got["prompt"])):
                _put(staging, paths[k], text.encode("utf-8"), hashes)
            md = res.metadata
            m.update(status="rendered", measurement_kind=tok.kind, tokenizer_identity=tok.identity,
                     wrapper_identity=T.wrapper_identity(), serializer_token_status=md["token"]["token_budget_status"],
                     paths=paths, tokens=got["tokens"], work=_work(md),
                     states_by_block={k: {"T": v["T"], "F": v["F"], "U": v["U"]} for k, v in md["states_by_block"].items()},
                     blocks=got["blocks"], ceilings=T.ceiling_results(got["tokens"]["input_tokens"]), non_nfc=got["non_nfc"])
            groups[(row["derived_scene_id"], f)].append((len(meas), sha256((T.HEAD + res.static_prefix).encode("utf-8")),
                                                         got["prefix_ids"]))
            meas.append(m)
            del res, md, got  # the per-tuple evidence is not kept or written
    for members_ in groups.values():  # potential reuse within one (selection, format) group only
        if len({h for _, h, _ in members_}) != 1 or len({tuple(p) for _, _, p in members_}) != 1:
            raise RuntimeError("one selection's wrapped static prefix differs between its commands")
        least = min(meas[k]["tokens"]["prefix_compatible_tokens"] for k, _, _ in members_)
        for k, _, _ in members_:
            t = meas[k]["tokens"]
            t.update(group_size=len(members_), group_reusable_prefix_tokens=least,
                     tokens_after_group_prefix=t["input_tokens"] - least)
    for m in meas:  # fixed key order in every rendered row
        if m["tokens"] is not None:
            t = m["tokens"]
            m["tokens"] = {k: t[k] for k in ("input_tokens", "input_bytes", "input_sha256", "token_ids_sha256",
                                             "document_tokens", "document", "static_prefix", "dynamic_suffix",
                                             "prefix_tokens", "prefix_compatible_tokens", "prefix_fully_compatible",
                                             "tokens_after_compatible_prefix", "group_size",
                                             "group_reusable_prefix_tokens", "tokens_after_group_prefix")}
    summary = summarize(index_rows, meas, tok.kind, cap)
    for name, data in (("preparation-index.jsonl", encode_jsonl(index_rows)), ("measurements.jsonl", encode_jsonl(meas)),
                       ("summary.json", encode_json(summary)), ("report.md", render_report(summary).encode("utf-8"))):
        _put_top(staging, name, data, hashes)
    agg = {d: _aggregate((p, h) for p, h in hashes.items() if p.startswith(d + "/"))
           for d in ("model_records/scenes", "model_records/commands", "diagnostic_inputs")}
    top = {p: h for p, h in hashes.items() if "/" not in p or p == "model_records/category-map.json"}
    manifest = {
        "format_version": 1, "record_type": "iref_model_input_manifest", "policy_id": POLICY,
        "source_selection_policy": SOURCE_POLICY,
        "baseline": {"claimed": BASELINE, "verified": False, "note": "the checkout is the user's; this tool does not query Git"},
        "inputs": {"hash_kind": "file_bytes", "scene_sha256": orig["hashes"]["scene"],
                   "category_map_sha256": orig["hashes"]["category_map"],
                   "inventory_views_sha256": orig["hashes"]["inventory_views"], "commands_sha256": orig["commands_sha256"],
                   "commands": len(orig["commands"]), "commands_hash_rule": "A2.2c's: SHA-256 of the sorted lines "
                                                                            "'<command_id> <file SHA-256>\\n'",
                   "parent_scene_canonical_sha256": parent_scene_sha256(scene)},
        "selection_audit": {"manifest_sha256": audit["hashes"]["manifest.json"],
                            "selections_sha256": audit["hashes"]["selections.jsonl"],
                            "summary_sha256": audit["hashes"]["summary.json"], "report_sha256": audit["hashes"]["report.md"],
                            "verification": "artifact hashes, recorded input hashes, the D77 parent-scene hash and a "
                                            "full recomputation from the originals all agreed"},
        "libraries": info["libraries"],
        "model_description": {"sha256": info["desc_sha256"], "hf_repo": info["desc"]["hf_repo"],
                              "hf_revision": info["desc"]["hf_revision"]},
        "tokenizer": {"repo": T.MODEL_REPO, "revision": T.REVISION, "files": T.PINNED_FILES, "identity": tok.identity,
                      "kind": tok.kind, "class": getattr(tok, "class_name", None), "versions": getattr(tok, "versions", None),
                      "staging": "only the three verified files, copied alone into a temporary folder"},
        "wrapper": {"identity": T.wrapper_identity(), "template_sha256": sha256(T.WRAPPER_TEMPLATE.encode("utf-8")),
                    "system_sha256": sha256(T.SYSTEM_MESSAGE.encode("utf-8")), "measurement_label": summary["measurement_label"]},
        "calibration": info["calibration"],
        "parser": {"protocol_id": info["protocol"]["protocol_id"],
                   "protocol_sha256": sha256(Path(PROTOCOL_PATH).read_bytes())},
        "object_limit": OBJECT_LIMIT, "work_cap": cap, "ceilings": list(CEILINGS),
        "materialized": {"scenes": len(scenes), "commands": len(commands),
                         "scenes_aggregate_sha256": agg["model_records/scenes"],
                         "commands_aggregate_sha256": agg["model_records/commands"]},
        "outputs": {"files": top, "aggregates": agg, "aggregate_rule": "SHA-256 of the sorted lines "
                                                                      "'<relative path> <file SHA-256>\\n'"},
        "code": _code_hashes(), "runtime": dict(runtime(), **(getattr(tok, "versions", None) or {})),
        "notes": ["Rendered texts are sizing diagnostics, not accepted inference requests.",
                  "Token ceilings are declared analysis scenarios, not deployment budgets.",
                  "Per-tuple serializer evidence was inspected and discarded after extracting compact fields; this is an "
                  "audit-output choice, not a reduction of the serializer's peak memory or Quest cost."]}
    _put_top(staging, "manifest.json", encode_json(manifest), {})
    return summary


def _put_top(staging, name, data, hashes) -> None:
    output._write_file(staging / name, data)
    hashes[name] = sha256(data)


# ------------------------------------------------------------------------------------------------- readback
def verify_bundle(folder) -> list:
    """Read a published (or staged) bundle back: every file, hash, link, count, pairing and concatenation."""
    f, bad = Path(folder), []
    try:
        manifest = json.loads((f / "manifest.json").read_text(encoding="utf-8"))
        index = [json.loads(x) for x in (f / "preparation-index.jsonl").read_text(encoding="utf-8").splitlines()]
        meas = [json.loads(x) for x in (f / "measurements.jsonl").read_text(encoding="utf-8").splitlines()]
        summary = json.loads((f / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [f"unreadable bundle: {type(e).__name__}: {e}"]
    bad += [f"manifest: {e.message}" for e in _validator("manifest").iter_errors(manifest)]
    bad += [f"summary: {e.message}" for e in _validator("summary").iter_errors(summary)]
    for rel, want in manifest.get("outputs", {}).get("files", {}).items():
        p = f / rel
        if not p.is_file() or sha256(p.read_bytes()) != want:
            bad.append(f"{rel}: missing or changed")
    found = {}
    for d in ("model_records/scenes", "model_records/commands", "diagnostic_inputs"):
        files = [p for p in (f / d).rglob("*") if p.is_file()] if (f / d).is_dir() else []
        pairs = [(p.relative_to(f).as_posix(), sha256(p.read_bytes())) for p in files]
        found.update(pairs)
        if _aggregate(pairs) != manifest.get("outputs", {}).get("aggregates", {}).get(d):
            bad.append(f"{d}: contents differ from the manifest's aggregate")
    if len(meas) != 2 * len(index):
        bad.append(f"{len(meas)} measurement rows for {len(index)} index rows")
    for k, r in enumerate(index):
        bad += [f"index row {k + 1}: {e.message}" for e in _validator("index_row").iter_errors(r)]
        pair = meas[2 * k:2 * k + 2]
        if [(m.get("parent_command_id"), m.get("view_id"), m.get("format")) for m in pair] != [
                (r["parent_command_id"], r["view_id"], fm) for fm in FORMATS]:
            bad.append(f"index row {k + 1}: measurement rows are not its two formats in order")
            continue
        if r["preparation_status"] == "materialized":
            try:
                s = json.loads((f / r["scene_path"]).read_text(encoding="utf-8"))
                c = json.loads((f / r["command_path"]).read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                bad.append(f"index row {k + 1}: a linked record is missing or unreadable")
                continue
            if (s.get("scene_id") != r["derived_scene_id"] or canonical_sha256(s) != r["scene_sha256"]
                    or [o["object_id"] for o in s.get("objects", [])] != r["planned_object_ids"]
                    or c.get("command_id") != r["derived_command_id"] or c.get("scene_id") != r["derived_scene_id"]
                    or canonical_sha256(c) != r["command_sha256"]):
                bad.append(f"index row {k + 1}: stale or inconsistent scene/command link")
        for m in pair:
            bad += [f"measurement {m['parent_command_id']} {m['format']}: {e.message}"
                    for e in _validator("measurement_row").iter_errors(m)]
            if m.get("status") != "rendered":
                if m.get("paths") is not None or m.get("tokens") is not None:
                    bad.append(f"measurement {m['parent_command_id']} {m['format']}: a non-rendered row carries texts")
                continue
            try:
                t = {k2: (f / p).read_bytes().decode("utf-8") for k2, p in m["paths"].items()}
            except (OSError, UnicodeDecodeError):
                bad.append(f"measurement {m['parent_command_id']} {m['format']}: a text file is missing")
                continue
            tk = m["tokens"]
            if (t["document"] != t["static_prefix"] + t["dynamic_suffix"] or t["prompt"] != T.wrap(t["document"])
                    or sha256(t["prompt"].encode("utf-8")) != tk["input_sha256"]
                    or any(sha256(t[x].encode("utf-8")) != tk[x]["sha256"] for x in ("document", "static_prefix", "dynamic_suffix"))
                    or sum(b["marginal_tokens"] for b in m["blocks"]) != tk["input_tokens"]
                    or m["ceilings"] != T.ceiling_results(tk["input_tokens"])
                    or not 0 <= tk["tokens_after_group_prefix"] <= tk["input_tokens"]
                    or tk["group_reusable_prefix_tokens"] > tk["prefix_compatible_tokens"]):
                bad.append(f"measurement {m['parent_command_id']} {m['view_id']} {m['format']}: texts, hashes or "
                           f"token arithmetic inconsistent")
    kind = next((m["measurement_kind"] for m in meas if m.get("measurement_kind")), summary.get("measurement_kind"))
    if not bad and not _same(summary, _plain(summarize(index, meas, kind, summary.get("work_cap")))):
        bad.append("summary.json differs from a recomputation from the rows")
    return bad
