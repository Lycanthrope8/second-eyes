"""The A2.3b matched rules baseline (D82): the accepted parser and resolver on the pilot's exact subscenes.

For each selected parent and inventory view, the frozen A2.2d derived scene and command go unchanged into the
accepted A2.2b parser (once per exact text), an A2.2b-style query bound to that derived scene and command, and the
accepted resolver with A2.2b's configuration and limits: 64 resolver calls for the pilot, not 128, because the two
serialization formats share their rules inputs. It reads no annotation and no model output, and its command line
accepts neither. The parser and resolver are reached through their modules' attributes, never through local copies.
"""
from __future__ import annotations

import copy
import importlib
from collections import Counter
from pathlib import Path

from ...relations import directions as D
from ...relations import predicates as Pr
from ...resolution import validate as RV
from ..iref_vla import output
from ..iref_vla import protocol as A2PROTOCOL
from . import inputs as IN
from .policy import (FORMATS, INPUT, INTEGRITY, MODES, POLICY_ID, RULES_BINS, TECHNICAL, VIEWS, check_overlap, code_hashes,
                     encode_json, encode_jsonl, fail, file_hashes, load_policy, policy_sha256,
                     publish_verified, read_bytes, read_json, read_jsonl, runtime, schema_issues,
                     semantic_rules_hash, sha256)

PARSE = importlib.import_module("grounding.evaluation.iref_vla.parse")
PREDICT = importlib.import_module("grounding.evaluation.iref_vla.predict")
RESOLVE = importlib.import_module("grounding.resolution.resolve")
FILES = ("manifest.json", "parses.jsonl", "rules.jsonl", "summary.json")
NOTES = ["One resolver call per selected parent and inventory view; both serialization formats share it.",
         "No annotation, statement, graph, model output or score was read.",
         "INSPECT is the evaluation protocol's declared action, not a dataset annotation or a learned action."]


def outcome_of(resolution) -> str:
    """The resolver's semantic status when it completed, otherwise its technical processing status."""
    return resolution["result"]["status"] if resolution["processing_status"] == "completed" \
        else resolution["processing_status"]


def rules_record(entry, query, resolution) -> dict:
    res = resolution["result"]
    outcome = outcome_of(resolution)
    return {"format_version": 1, "record_type": "iref_pilot_rules", "policy_id": POLICY_ID, "selection_rank": entry["rank"],
            "parent_command_id": entry["parent"], "view_id": entry["view"],
            "request_ids": [entry["requests"][f]["request_id"] for f in FORMATS],
            "derived_scene_id": entry["scene"]["scene_id"], "derived_command_id": entry["command"]["command_id"],
            "scene_sha256": entry["index_row"]["scene_sha256"], "command_sha256": entry["index_row"]["command_sha256"],
            "object_ids": list(entry["object_ids"]), "query": query, "resolution": resolution, "outcome": outcome,
            "technical_failure": outcome in TECHNICAL,
            "target_id": res["target_id"] if outcome == "resolved" else None,
            "candidate_ids": list(res["candidates"]["target_ids"]) if res else [],
            "reason_code": res["reason_code"] if res else None, "reason_codes": list(res["reason_codes"]) if res else []}


def summarize_rules(parses, rules, mode) -> dict:
    views = {}
    for v in VIEWS:
        rs = [r for r in rules if r["view_id"] == v]
        work = [r["resolution"]["diagnostics"]["work"] for r in rs]
        keys = sorted({k for w in work for k, x in w.items() if type(x) is int})
        bins = {b: sum(r["outcome"] == b for r in rs) for b in RULES_BINS}
        views[v] = {"records": len(rs), "outcomes": {b: bins[b] for b in RULES_BINS if b not in TECHNICAL},
                    "technical_failures": {b: bins[b] for b in TECHNICAL}, "R": bins["resolved"],
                    "reason_codes_nonexclusive": dict(sorted(Counter(c for r in rs for c in r["reason_codes"]).items())),
                    "work_totals": {k: sum(w.get(k, 0) for w in work) for k in keys},
                    "work_maxima": {k: max((w.get(k, 0) for w in work), default=0) for k in keys}}
    return {"format_version": 1, "record_type": "iref_pilot_rules_summary", "policy_id": POLICY_ID, "mode": mode,
            "counts": {"parents": len(parses), "rules_records": len(rules), "parse_records": len(parses)},
            "parser": {"parsed": sum(p["parse_status"] == "parsed" for p in parses),
                       "distinct_texts": len({p["text"] for p in parses})},
            "views": views, "technical_failures": sum(r["technical_failure"] for r in rules),
            "semantic_rules_hash": semantic_rules_hash(rules)}


def _identities(rules_protocol, rel_path, dir_path, bundle) -> dict:
    try:
        got = {"relation_config": Pr.load_config(rel_path).identity,
               "direction_config": D.load_direction_config(dir_path).identity}
    except Exception as e:  # noqa: BLE001 - a configuration that will not load is invalid input
        fail(INPUT, [("library configuration", f"{type(e).__name__}: {e}")])
    libs = bundle["manifest"].get("libraries", {})
    prep = {k: (libs.get(k) or {}).get("identity") for k in got}
    if got != rules_protocol["library_identities"] or got != prep:
        fail(INTEGRITY, [("library configuration", f"identities {got} differ from the rules protocol's "
                                                   f"{rules_protocol['library_identities']} or the preparation record's "
                                                   f"{prep}: stop and report, never tune")])
    return got


def run_baseline(*, requests, bundle, relation_config, direction_config, out, sample=True, limits=None) -> dict:
    """The `baseline` command; returns the rules summary. `sample=False` and `limits` serve labelled fixtures only."""
    out = output.refuse_existing(out)
    check_overlap(out, (requests, bundle, relation_config, direction_config))
    mode = MODES[0] if sample else MODES[1]
    policy = load_policy()
    rules_protocol = IN.relabel(INPUT, A2PROTOCOL.load_protocol)
    proto_sha = sha256(read_bytes(A2PROTOCOL.PROTOCOL_PATH, "rules protocol"))
    req = IN.load_requests(requests, sample=sample)
    src = IN.load_bundle(bundle)
    IN.check_request_bundle(req, src)
    parser_rec = src["manifest"].get("parser", {})
    if parser_rec.get("protocol_id") != rules_protocol["protocol_id"] or parser_rec.get("protocol_sha256") != proto_sha:
        fail(INTEGRITY, [("rules protocol", "the A2.2b protocol differs from the one the preparation bundle recorded")])
    rel_path, dir_path = Path(relation_config), Path(direction_config)
    identities = _identities(rules_protocol, rel_path, dir_path, src)
    watched = {"requests manifest": req["dir"] / "manifest.json", "request index": req["dir"] / "request-index.jsonl",
               "bundle manifest": src["dir"] / "manifest.json", "preparation index": src["dir"] / "preparation-index.jsonl",
               "category map": src["dir"] / "model_records" / "category-map.json"}
    before = file_hashes(watched)
    entries = IN.join(req, src, sample=sample)
    cmap = src["category_map"]
    config = PREDICT.resolver_config(rules_protocol, cmap, limits)
    problems = RV.check_config(config)
    if problems:
        fail(INPUT, [(f"resolver config {p['path']}", f"{p['code']}: {p['message']}") for p in problems])
    vocabulary, colours = cmap["model_vocabulary"], rules_protocol["colour_labels"]
    by_text, parses = {}, []
    for e in entries:
        if e["view"] != VIEWS[0]:
            continue
        if e["text"] not in by_text:  # one parse per exact text, rebound to each parent
            by_text[e["text"]] = PARSE.parse_record(e["parent"], e["text"], categories=vocabulary, colours=colours)
        p = dict(copy.deepcopy(by_text[e["text"]]), parent_command_id=e["parent"])
        bad = A2PROTOCOL.parse_record_issues(p, f"parse {e['parent']}")
        if bad:
            fail(INTEGRITY, [(i["path"], f"{i['code']}: {i['message']}") for i in bad])
        if p["parse_status"] != "parsed":
            fail(INTEGRITY, [(f"parse {e['parent']}", f"a selected parent no longer parses ({p['parse_reason']}): the "
                                                      "selection assumed parser support, so the inputs or the protocol "
                                                      "are inconsistent; the grammar is not extended")])
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
        found, _ = RV.check_query(query)
        if found:
            fail(INTEGRITY, [(f"query for {e['parent']} {e['view']} {p['path']}", f"{p['code']}: {p['message']}")
                             for p in found])
        resolution = RESOLVE.resolve(scene, command, query, resolver_config=config, relation_config_path=rel_path,
                                     direction_config_path=dir_path, category_maps=[cmap], trace=False)
        rules.append(rules_record(e, query, resolution))
    if file_hashes(watched) != before:
        fail(INTEGRITY, [("inputs", "an input file changed while the baseline ran")])
    summary = summarize_rules(parses, rules, mode)
    files = {"parses.jsonl": encode_jsonl(parses), "rules.jsonl": encode_jsonl(rules), "summary.json": encode_json(summary)}
    manifest = {
        "format_version": 1, "record_type": "iref_pilot_rules_manifest", "policy_id": POLICY_ID,
        "policy_sha256": policy_sha256(), "mode": mode, "policy_decision": policy["decision"],
        "rules_protocol": {"protocol_id": rules_protocol["protocol_id"], "file_sha256": proto_sha},
        "action": rules_protocol["action"], "action_origin": rules_protocol["action_origin"],
        "resolver": {"config_id": config["config_id"], "vocabulary_id": config["vocabulary_id"],
                     "limits": dict(config["limits"]), "trace": False, "config_sha256": RV.canonical_sha256(config)},
        "library_identities": identities,
        "library_files": {"relation_config_sha256": sha256(read_bytes(rel_path, "relation configuration")),
                          "direction_config_sha256": sha256(read_bytes(dir_path, "direction configuration"))},
        "vocabulary": {"category_map_id": cmap["map_id"], "category_labels": len(config["category_labels"]),
                       "colour_labels": list(colours)},
        "inputs": {"requests_manifest_sha256": req["manifest_sha256"], "request_index_sha256": req["index_sha256"],
                   "request_protocol_sha256": req["protocol_sha256"],
                   "selected_sha256": req["manifest"]["selection"]["selected_sha256"],
                   "selected_parent_ids": list(req["selected"]), "bundle_manifest_sha256": src["manifest_sha256"],
                   "preparation_index_sha256": src["index_sha256"],
                   "category_map_sha256": sha256(src["raw"]["model_records/category-map.json"])},
        "semantic_hash_rule": policy["rules_semantic_hash_rule"], "semantic_rules_hash": summary["semantic_rules_hash"],
        "counts": dict(summary["counts"]), "outputs": {k: sha256(v) for k, v in files.items()},
        "code": code_hashes(), "runtime": runtime(), "notes": list(NOTES)}
    files["manifest.json"] = encode_json(manifest)
    publish_verified(out, files, verify_rules_dir)
    return summary


# ------------------------------------------------------------------------------------------------- readback
def verify_rules_dir(folder) -> list:
    """Every file, hash, record, order, link and recomputed count of a rules folder; [] when it may be used."""
    f, bad = Path(folder), []
    present = sorted(p.relative_to(f).as_posix() for p in f.rglob("*") if p.is_file()) if f.is_dir() else []
    if present != sorted(FILES):
        return [f"expected exactly the files {sorted(FILES)}; found {present[:8]}"]
    try:
        manifest = read_json(f / "manifest.json", "rules manifest")
        summary = read_json(f / "summary.json", "rules summary")
        parses = read_jsonl(f / "parses.jsonl", "parses")
        rules = read_jsonl(f / "rules.jsonl", "rules")
    except Exception as e:  # noqa: BLE001 - reported as a finding
        return [f"unreadable rules output: {e}"]
    bad += [f"{i['path']}: {i['message']}" for i in schema_issues("rules_manifest", manifest, "manifest.json")]
    bad += [f"{i['path']}: {i['message']}" for i in schema_issues("rules_summary", summary, "summary.json")]
    if bad:
        return bad
    for name, want in manifest["outputs"].items():
        if sha256((f / name).read_bytes()) != want:
            bad.append(f"{name}: changed since the manifest was written")
    for k, p in enumerate(parses, 1):
        bad += [f"{i['path']}: {i['message']}" for i in A2PROTOCOL.parse_record_issues(p, f"parses.jsonl line {k}")]
    for k, r in enumerate(rules, 1):
        bad += [f"{i['path']}: {i['message']}" for i in schema_issues("rules_record", r, f"rules.jsonl line {k}")]
    if bad:
        return bad
    parse_ids = [p["parent_command_id"] for p in parses]
    if parse_ids != manifest["inputs"]["selected_parent_ids"]:
        bad.append("parses.jsonl is not one parse per selected parent, in selection order")
    keys = [(r["selection_rank"], r["parent_command_id"], r["view_id"]) for r in rules]
    want = [(k, p, v) for k, p in enumerate(manifest["inputs"]["selected_parent_ids"], 1) for v in VIEWS]
    if keys != want:
        bad.append("rules.jsonl is not one record per selected parent and view, in selection order with full first")
    parse_of = {p["parent_command_id"]: p for p in parses}
    for r in rules:
        where = f"rules {r['selection_rank']} {r['view_id']}"
        q, res = r["query"], r["resolution"]
        found, _ = RV.check_query(q)
        bad += [f"{where} query {i['path']}: {i['message']}" for i in found]
        bad += [f"{where} resolution: {e.message[:200]}"
                for e in A2PROTOCOL._validator("resolution").iter_errors(res)]
        if found or bad:
            continue
        p = parse_of.get(r["parent_command_id"])
        want_q = {"query_id": r["derived_command_id"] + ".q", "scene_id": r["derived_scene_id"],
                  "command_id": r["derived_command_id"], "scene_revision": 0, "evidence_profile": "annotated"}
        if any(q[k] != v for k, v in want_q.items()) or p is None or q["interpretations"] != [p["interpretation"]]:
            bad.append(f"{where}: the query is not bound to its derived scene and command, or not its parent's parse")
        outcome = outcome_of(res)
        result = res["result"]
        if (r["outcome"], r["technical_failure"]) != (outcome, outcome in TECHNICAL) \
                or r["target_id"] != (result["target_id"] if outcome == "resolved" else None) \
                or r["reason_code"] != (result["reason_code"] if result else None) \
                or r["reason_codes"] != (list(result["reason_codes"]) if result else []) \
                or r["candidate_ids"] != (list(result["candidates"]["target_ids"]) if result else []):
            bad.append(f"{where}: the outcome fields disagree with the saved resolution")
        ctx, ids = res["diagnostics"]["context"], res["diagnostics"]["identities"]
        want_ctx = dict(want_q, query_id=q["query_id"])
        if ctx not in (None, want_ctx) or (result is not None and any(result[k] != want_ctx[k] for k in want_ctx)):
            bad.append(f"{where}: the resolution names another query, scene or command")
        if ids is not None and (ids["scene_sha256"] != r["scene_sha256"] or ids["command_sha256"] != r["command_sha256"]
                                or ids["query_sha256"] != RV.canonical_sha256(q)
                                or ids["resolver_config_sha256"] != manifest["resolver"]["config_sha256"]
                                or ids["relation_config_identity"] != manifest["library_identities"]["relation_config"]
                                or ids["direction_config_identity"] != manifest["library_identities"]["direction_config"]):
            bad.append(f"{where}: the resolution's input identities disagree with its record or the manifest")
    if bad:
        return bad
    if summarize_rules(parses, rules, manifest["mode"]) != summary:
        bad.append("summary.json differs from a recomputation from the records")
    if manifest["semantic_rules_hash"] != semantic_rules_hash(rules) or manifest["counts"] != summary["counts"]:
        bad.append("the manifest's semantic hash or counts disagree with the records")
    return bad


def load_rules_dir(folder) -> dict:
    f = Path(folder)
    if not f.is_dir():
        fail(INPUT, [(str(f), "the rules folder does not exist")])
    for name in ("manifest.json", "summary.json"):
        read_json(f / name, f"rules {name}")
    for name in ("parses.jsonl", "rules.jsonl"):
        rows = read_jsonl(f / name, f"rules {name}")
        if any(not isinstance(r, dict) for r in rows):
            fail(INPUT, [(str(f / name), "every row must be a JSON object")])
    manifest = read_json(f / "manifest.json", "rules manifest")
    for name, rec in (("manifest.json", manifest), ("summary.json", read_json(f / "summary.json", "rules summary"))):
        found = schema_issues("rules_manifest" if name == "manifest.json" else "rules_summary", rec, f"rules {name}")
        if found:
            fail(INPUT, [(i["path"], i["message"]) for i in found[:10]])
    for k, r in enumerate(read_jsonl(f / "rules.jsonl", "rules"), 1):
        found = schema_issues("rules_record", r, f"rules.jsonl line {k}")
        if found:
            fail(INPUT, [(i["path"], i["message"]) for i in found[:10]])
    problems = verify_rules_dir(f)
    if problems:
        fail(INTEGRITY, [(str(f), m) for m in problems[:20]])
    return {"dir": f, "manifest": manifest, "parses": read_jsonl(f / "parses.jsonl", "parses"),
            "rules": read_jsonl(f / "rules.jsonl", "rules"),
            "files": {n: sha256(read_bytes(f / n, n)) for n in FILES}}


def check_rules_links(rules_dir, requests, bundle, entries) -> dict:
    """The saved rules were computed from exactly these requests and bundle, for exactly these parents and views."""
    m, bad = rules_dir["manifest"], []
    inp = m["inputs"]
    if inp["requests_manifest_sha256"] != requests["manifest_sha256"] or inp["selected_parent_ids"] != requests["selected"]:
        bad.append("the rules were computed for other requests or another selection")
    if inp["bundle_manifest_sha256"] != bundle["manifest_sha256"] or inp["preparation_index_sha256"] != bundle["index_sha256"]:
        bad.append("the rules were computed from another preparation bundle")
    by_key = {(r["parent_command_id"], r["view_id"]): r for r in rules_dir["rules"]}
    for e in entries:
        r = by_key.get((e["parent"], e["view"]))
        if r is None:
            bad.append(f"no rules record for {e['parent']} {e['view']}")
            continue
        if (r["selection_rank"], r["derived_scene_id"], r["derived_command_id"], r["scene_sha256"], r["command_sha256"],
                r["object_ids"], r["request_ids"]) != (e["rank"], e["scene"]["scene_id"], e["command"]["command_id"],
                                                       e["index_row"]["scene_sha256"], e["index_row"]["command_sha256"],
                                                       e["object_ids"], [e["requests"][f]["request_id"] for f in FORMATS]):
            bad.append(f"rules record {e['parent']} {e['view']}: its context IDs disagree with the joined request")
    texts = {p["parent_command_id"]: p["text"] for p in rules_dir["parses"]}
    if any(texts.get(e["parent"]) != e["text"] for e in entries):
        bad.append("a saved parse's text is not its parent's exact command text")
    if len(by_key) != len(rules_dir["rules"]) or len(rules_dir["rules"]) != len(entries):
        bad.append("rules records are repeated, missing or extra")
    if bad:
        fail(INTEGRITY, [("rules", x) for x in bad[:20]])
    return by_key


__all__ = ["FILES", "check_rules_links", "load_rules_dir", "outcome_of", "rules_record", "run_baseline",
           "summarize_rules", "verify_rules_dir"]
