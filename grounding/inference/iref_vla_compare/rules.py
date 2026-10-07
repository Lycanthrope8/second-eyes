"""A2.3d rules (D95): the unchanged A2.2b parser and A2.1e resolver on exactly the models' subscenes, on the laptop.

One parse per exact text and one resolver call per selected parent and inventory view, as in A2.3b. Each call receives
the bundle's derived scene and command for that parent and view: the same objects, the same evidence and the same
candidate set the two models are offered (the models' letters and list order never reach the rules). The entries are
built and checked by A2.3b's own join, given the request identities with the scene-order object list. No annotation is
read; targets stay in scoring.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from ...evaluation.iref_vla import output
from ...evaluation.iref_vla.protocol import EvaluationInputError, encode_json, encode_jsonl, issue, runtime, sha256, strict_json
from ...evaluation.iref_vla_pilot import baseline as BL
from ..iref_vla.prepare import rows_of
from . import design as D
from .prepare import code_hashes, verify_compare_requests

FILES = ("manifest.json", "parses.jsonl", "rules.jsonl", "summary.json")


def _fail(code, problems):
    raise EvaluationInputError([issue(w, code, m) for w, m in problems])


def _relabel(record: dict, kind: str) -> dict:
    record = dict(record)
    record["record_type"], record["policy_id"] = kind, D.POLICY_ID
    return record


def run_compare_rules(*, requests, bundle, relation_config, direction_config, out, sample=True, limits=None) -> dict:
    """The `rules` command; returns the summary. `sample=False` and `limits` serve labelled fixtures only."""
    out = output.refuse_existing(out)
    req_dir = Path(requests)
    for p in (req_dir, Path(bundle), Path(relation_config), Path(direction_config)):
        o, q = out.resolve(), p.resolve()
        if o == q or q in o.parents or o in q.parents:
            _fail("E_COMPARE_PATH", [(str(out), f"the output overlaps the input {p}")])
    bad = verify_compare_requests(req_dir)
    if bad:
        _fail("E_COMPARE_REQUESTS", [(str(req_dir), m) for m in bad[:20]])
    rman_bytes = (req_dir / "manifest.json").read_bytes()
    rman = json.loads(rman_bytes)
    rows = rows_of("request-index.jsonl", (req_dir / "request-index.jsonl").read_bytes(), "E_COMPARE_REQUESTS")
    selection = strict_json("selection.json", (req_dir / "selection.json").read_bytes(), "E_COMPARE_REQUESTS")
    src = BL.IN.load_bundle(bundle)
    if src["manifest_sha256"] != rman["source_bundle"]["manifest_sha256"] \
            or src["index_sha256"] != rman["source_bundle"]["preparation_index_sha256"]:
        _fail("E_COMPARE_BUNDLE", [(str(bundle), "not the bundle the requests were prepared from")])
    mode = BL.MODES[0] if sample else BL.MODES[1]
    rules_protocol = BL.IN.relabel(BL.INPUT, BL.A2PROTOCOL.load_protocol)
    proto_sha = sha256(BL.A2PROTOCOL.PROTOCOL_PATH.read_bytes())
    parser_rec = src["manifest"].get("parser", {})
    if parser_rec.get("protocol_id") != rules_protocol["protocol_id"] or parser_rec.get("protocol_sha256") != proto_sha:
        _fail("E_COMPARE_BUNDLE", [("rules protocol", "the A2.2b protocol differs from the one the bundle recorded")])
    rel_path, dir_path = Path(relation_config), Path(direction_config)
    identities = BL._identities(rules_protocol, rel_path, dir_path, src)
    by_key = {}
    for r in rows:
        by_key[(r["parent_command_id"], r["view_id"], r["format"])] = {
            "request_id": r["request_id"], "derived_scene_id": r["derived_scene_id"],
            "derived_command_id": r["derived_command_id"], "object_ids": list(r["object_ids"]),
            "mapping": [["", o, 0] for o in r["object_ids"]] + [["K", "ASK", 0]],
            "source_document_path": r["source_document_path"], "source_document_sha256": r["source_document_sha256"]}
    entries = BL.IN.join({"selected": list(selection["selected_parent_ids"]), "by_key": by_key}, src, sample=sample)
    cmap = src["category_map"]
    config = BL.PREDICT.resolver_config(rules_protocol, cmap, limits)
    found = BL.RV.check_config(config)
    if found:
        _fail("E_COMPARE_RULES", [(f"resolver config {p['path']}", f"{p['code']}: {p['message']}") for p in found])
    vocabulary, colours = cmap["model_vocabulary"], rules_protocol["colour_labels"]
    by_text, parses = {}, []
    for e in entries:
        if e["view"] != D.VIEWS[0]:
            continue
        if e["text"] not in by_text:
            by_text[e["text"]] = BL.PARSE.parse_record(e["parent"], e["text"], categories=vocabulary, colours=colours)
        p = dict(copy.deepcopy(by_text[e["text"]]), parent_command_id=e["parent"])
        bad = BL.A2PROTOCOL.parse_record_issues(p, f"parse {e['parent']}")
        if bad:
            _fail("E_COMPARE_RULES", [(i["path"], f"{i['code']}: {i['message']}") for i in bad])
        if p["parse_status"] != "parsed":
            _fail("E_COMPARE_RULES", [(f"parse {e['parent']}", f"a selected parent no longer parses ({p['parse_reason']})")])
        parses.append(p)
    parse_of = {p["parent_command_id"]: p for p in parses}
    rules = []
    for e in entries:
        scene, command = e["scene"], e["command"]
        query = {"schema_version": 1, "record_type": "grounding_query",
                 "query_id": command["command_id"] + rules_protocol["query_suffix"], "scene_id": scene["scene_id"],
                 "scene_revision": scene["scene_revision"], "evidence_profile": scene["evidence_profile"],
                 "command_id": command["command_id"],
                 "interpretations": [copy.deepcopy(parse_of[e["parent"]]["interpretation"])]}
        found, _ = BL.RV.check_query(query)
        if found:
            _fail("E_COMPARE_RULES", [(f"query {e['parent']} {e['view']} {p['path']}", f"{p['code']}: {p['message']}")
                                      for p in found])
        resolution = BL.RESOLVE.resolve(scene, command, query, resolver_config=config, relation_config_path=rel_path,
                                        direction_config_path=dir_path, category_maps=[cmap], trace=False)
        rules.append(_relabel(BL.rules_record(e, query, resolution), "iref_compare_rules"))
    summary = _relabel(BL.summarize_rules(parses, rules, mode), "iref_compare_rules_summary")
    files = {"parses.jsonl": encode_jsonl(parses), "rules.jsonl": encode_jsonl(rules), "summary.json": encode_json(summary)}
    manifest = {"format_version": 1, "record_type": "iref_compare_rules_manifest", "policy_id": D.POLICY_ID, "mode": mode,
                "rules_protocol": {"protocol_id": rules_protocol["protocol_id"], "file_sha256": proto_sha},
                "resolver": {"config_id": config["config_id"], "vocabulary_id": config["vocabulary_id"],
                             "limits": dict(config["limits"]), "trace": False,
                             "config_sha256": BL.RV.canonical_sha256(config)},
                "library_identities": identities,
                "library_files": {"relation_config_sha256": sha256(rel_path.read_bytes()),
                                  "direction_config_sha256": sha256(dir_path.read_bytes())},
                "inputs": {"requests_manifest_sha256": sha256(rman_bytes), "selection_sha256": selection["selected_sha256"],
                           "bundle_manifest_sha256": src["manifest_sha256"], "preparation_index_sha256": src["index_sha256"],
                           "category_map_sha256": sha256(src["raw"]["model_records/category-map.json"])},
                "counts": dict(summary["counts"]), "outputs": {k: sha256(v) for k, v in files.items()},
                "code": code_hashes(), "runtime": runtime(),
                "notes": ["One resolver call per selected parent and view, on the subscene the models are offered; both "
                          "formats and both models share it.", "No annotation was read."]}
    files["manifest.json"] = encode_json(manifest)
    BL.publish_verified(out, files, verify_compare_rules)
    return summary


def verify_compare_rules(folder, requests=None) -> list:
    f = Path(folder)
    present = sorted(p.name for p in f.iterdir() if p.is_file()) if f.is_dir() else []
    if present != sorted(FILES):
        return [f"expected exactly {sorted(FILES)}; found {present}"]
    try:
        manifest = strict_json("manifest.json", (f / "manifest.json").read_bytes(), "E_COMPARE_RULES")
        parses = rows_of("parses.jsonl", (f / "parses.jsonl").read_bytes(), "E_COMPARE_RULES")
        rules = rows_of("rules.jsonl", (f / "rules.jsonl").read_bytes(), "E_COMPARE_RULES")
        summary = strict_json("summary.json", (f / "summary.json").read_bytes(), "E_COMPARE_RULES")
    except (OSError, EvaluationInputError) as e:
        return [f"unreadable rules folder: {e}"]
    bad = [f"{n}: changed" for n, want in manifest.get("outputs", {}).items() if sha256((f / n).read_bytes()) != want]
    if bad:
        return bad
    if any(r.get("record_type") != "iref_compare_rules" for r in rules) or len({(r["parent_command_id"], r["view_id"]) for r in rules}) != len(rules):
        bad.append("rules records are not one compare record per parent and view")
    if _relabel(BL.summarize_rules(parses, rules, manifest.get("mode")), "iref_compare_rules_summary") != summary:
        bad.append("summary.json differs from a recomputation")
    if requests is not None:
        rq = rows_of("request-index.jsonl", (Path(requests) / "request-index.jsonl").read_bytes(), "E_COMPARE_REQUESTS")
        want = {(r["parent_command_id"], r["view_id"]): sorted(x["request_id"] for x in rq if (x["parent_command_id"], x["view_id"]) == (r["parent_command_id"], r["view_id"])) for r in rq}
        got = {(r["parent_command_id"], r["view_id"]): sorted(r["request_ids"]) for r in rules}
        if got != want:
            bad.append("the rules records do not cover exactly the request bundle's parents and views")
        if sha256((Path(requests) / "manifest.json").read_bytes()) != manifest["inputs"]["requests_manifest_sha256"]:
            bad.append("the rules were run on another request bundle")
    return bad
