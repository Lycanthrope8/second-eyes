"""Prediction for the A2.2b development evaluation (D75).

predict() turns a validated scene, its category map, its commands and their inventory-view manifest into parse
records, two derived inventory views, derived command contexts, grounding queries and resolver results. It takes no
answers and reads nothing it isn't given; run_predict() reads exactly the paths on its command line plus the tracked
protocol. Scoring is a separate operation (score.py) that never reruns or changes a prediction.
"""
from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ...contract import validate as contract
from ...relations import directions as D
from ...relations import predicates as P
from ...resolution import resolve
from ...resolution import validate as RV
from . import output
from .parse import parse_record
from .protocol import (BINS, EVALUATION_VERSION, LIMITS, VIEWS, EvaluationInputError, _validator, code_hashes, encode_json,
                       encode_jsonl, issue, load_protocol, parse_record_issues, ratio, runtime, schema_issues,
                       semantic_hash, sha256, strict_json)

FILES = ("manifest.json", "category-map.json", "resolver-config.json", "scenes/full_inventory.json",
         "scenes/source_known_nyu.json", "parses.jsonl", "command-contexts.jsonl", "predictions.jsonl",
         "prediction-summary.json")
MANIFEST_KEYS = {"format_version", "record_type", "evaluation_version", "protocol_id", "protocol_sha256", "source_sample",
                 "parent", "action", "action_origin", "views", "vocabulary", "resolver", "library_identities", "inputs",
                 "code", "runtime", "semantic_hash_rule", "semantic_prediction_hash", "counts"}


@dataclass
class Prediction:
    manifest: dict
    category_map: dict
    resolver_config: dict
    scenes: dict
    parses: list
    contexts: list
    predictions: list
    summary: dict


def _number(object_id):
    return int(object_id[4:])


def _kind(value) -> str:
    if value is None:
        return "null"
    return {bool: "a boolean", int: "an integer", float: "a number", str: "a string", list: "an array",
            dict: "an object"}.get(type(value), type(value).__name__)


def _is_version_1(value) -> bool:
    """A new evaluation record's version: exactly the integer 1, never 1.0, true, "1" or another integer."""
    return type(value) is int and value == 1


def _view_shape(views) -> list:
    """The inventory-view list's shape, checked before any view ID or object ID is sorted or used as a key."""
    if not isinstance(views, list):
        return [("inventory views $.views", f"expected an array of views, not {_kind(views)}")]
    out = []
    for i, v in enumerate(views):
        where = f"inventory views $.views[{i}]"
        if not isinstance(v, dict) or set(v) != {"view_id", "policy", "included_object_ids", "excluded"}:
            out.append((where, "expected an object with exactly view_id, policy, included_object_ids and excluded"))
            continue
        for key in ("view_id", "policy"):
            if not isinstance(v[key], str):
                out.append((f"{where}.{key}", f"expected a string, not {_kind(v[key])}"))
        if not isinstance(v["included_object_ids"], list):
            out.append((f"{where}.included_object_ids", f"expected an array, not {_kind(v['included_object_ids'])}"))
        else:
            out += [(f"{where}.included_object_ids[{j}]", f"expected an object ID string, not {_kind(x)}")
                    for j, x in enumerate(v["included_object_ids"]) if not isinstance(x, str)]
        if not isinstance(v["excluded"], list):
            out.append((f"{where}.excluded", f"expected an array, not {_kind(v['excluded'])}"))
            continue
        for j, e in enumerate(v["excluded"]):
            if not isinstance(e, dict) or set(e) != {"object_id", "source_object_id", "reason"}:
                out.append((f"{where}.excluded[{j}]", "expected an object with exactly object_id, source_object_id and reason"))
                continue
            out += [(f"{where}.excluded[{j}].{k}", f"expected a string, not {_kind(e[k])}")
                    for k in ("object_id", "source_object_id", "reason") if not isinstance(e[k], str)]
    return out


def _fail(code, problems):
    raise EvaluationInputError([issue(p, code, m) for p, m in problems])


def _check_inputs(scene, commands, category_map):
    if not isinstance(scene, dict) or not isinstance(category_map, dict) or not isinstance(commands, list) \
            or not commands or not all(isinstance(c, dict) for c in commands):
        _fail("E_EVAL_INPUT", [("inputs", "expected a scene object, a category map object and a nonempty list of "
                                          "command objects")])
    roles = [("scene", scene, "scene"), ("category map", category_map, "category_map")] + [
        (f"command {c.get('command_id')!r}", c, "command_context") for c in commands]
    wrong = [(label, f"holds a {r.get('record_type')!r} record, not {want}") for label, r, want in roles
             if r.get("record_type") != want]
    if wrong:
        _fail("E_EVAL_INPUT", wrong)
    found = contract.validate([(label, r) for label, r, _ in roles])
    if found:  # a warning means a record could not be cross-checked against its partner: input failure too
        _fail("E_EVAL_INPUT", [(f"{i.file} {i.path}", f"{i.code}: {i.message}") for i in found])
    if scene["evidence_profile"] != "annotated":
        _fail("E_EVAL_INPUT", [("scene $.evidence_profile", "this protocol evaluates annotated scenes only")])


def _check_libraries(protocol, relation_config_path, direction_config_path):
    try:
        got = {"relation_config": P.load_config(relation_config_path).identity,
               "direction_config": D.load_direction_config(direction_config_path).identity}
    except Exception as e:  # noqa: BLE001 - a configuration that won't load is invalid input
        _fail("E_EVAL_INPUT", [("library configuration", f"{type(e).__name__}: {e}")])
    if got != protocol["library_identities"]:
        _fail("E_EVAL_LIBRARY", [("library configuration", f"identities {got} differ from the accepted "
                                                           f"{protocol['library_identities']}: stop and report, "
                                                           "never tune")])
    return got


def _membership(scene, category_map, protocol):
    """The recomputed policy: which objects each view keeps, and why the others are excluded."""
    vocabulary = set(category_map["model_vocabulary"])
    reason = protocol["views"][1]["exclusion_reason"]
    known = [o["object_id"] for o in scene["objects"]
             if o["category"]["state"] == "known" and o["category"]["value"]["model"] in vocabulary]
    others = {o["object_id"]: (o["source_ref"]["source_object_id"], reason) for o in scene["objects"]
              if o["object_id"] not in set(known)}
    return {"full_inventory": ({o["object_id"] for o in scene["objects"]}, {}),
            "source_known_nyu": (set(known), others)}


def _check_views(manifest, scene, category_map, protocol):
    """The A2.2a inventory-view manifest against the recomputed policy; failures name what is wrong."""
    bad = []
    keys = {"format_version", "record_type", "parent_scene_id", "parent_scene_revision", "views"}
    if not isinstance(manifest, dict) or set(manifest) != keys:
        _fail("E_EVAL_VIEWS", [("inventory views", f"expected exactly the keys {sorted(keys)}")])
    if type(manifest["format_version"]) is not int or manifest["format_version"] != 1 \
            or manifest["record_type"] != "iref_inventory_views":
        bad.append(("inventory views", "expected format_version 1 and record_type iref_inventory_views"))
    if manifest["parent_scene_id"] != scene["scene_id"] or type(manifest["parent_scene_revision"]) is not int \
            or manifest["parent_scene_revision"] != scene["scene_revision"]:
        bad.append(("inventory views $.parent_scene_id", f"parent {manifest['parent_scene_id']!r} revision "
                                                         f"{manifest['parent_scene_revision']!r} is not the scene "
                                                         f"{scene['scene_id']!r} revision {scene['scene_revision']}"))
    views = manifest["views"]
    shape = _view_shape(views)
    if shape:
        _fail("E_EVAL_VIEWS", bad + shape)
    if sorted(v["view_id"] for v in views) != sorted(VIEWS) or len(views) != 2:
        _fail("E_EVAL_VIEWS", bad + [("inventory views $.views", f"expected exactly the views {list(VIEWS)}")])
    all_ids = [o["object_id"] for o in scene["objects"]]
    policy = _membership(scene, category_map, protocol)
    for v in views:
        vid = v["view_id"]
        where = f"inventory views {vid}"
        if set(v) != {"view_id", "policy", "included_object_ids", "excluded"} or not isinstance(v["policy"], str) \
                or not isinstance(v["included_object_ids"], list) or not isinstance(v["excluded"], list):
            bad.append((where, "expected view_id, policy, included_object_ids and excluded"))
            continue
        inc = v["included_object_ids"]
        exc = [e for e in v["excluded"] if isinstance(e, dict) and set(e) == {"object_id", "source_object_id", "reason"}]
        if len(exc) != len(v["excluded"]):
            bad.append((where, "an excluded entry is not {object_id, source_object_id, reason}"))
            continue
        exc_ids = [e["object_id"] for e in exc]
        dup = sorted({x for x in inc + exc_ids if (inc + exc_ids).count(x) > 1})
        if dup:
            bad.append((where, f"objects listed more than once: {dup}"))
        extra = sorted(set(inc + exc_ids) - set(all_ids))
        missing = sorted(set(all_ids) - set(inc + exc_ids))
        if extra or missing:
            bad.append((where, f"not a partition of the scene: extra {extra}, missing {missing}"))
        want_inc, want_exc = policy[vid]
        if set(inc) != want_inc:
            bad.append((where, f"membership differs from the recomputed policy: unexpected "
                               f"{sorted(set(inc) - want_inc)}, absent {sorted(want_inc - set(inc))}"))
        for e in exc:
            want = want_exc.get(e["object_id"])
            if want is None or (e["source_object_id"], e["reason"]) != want:
                bad.append((where, f"exclusion of {e['object_id']} ({e['source_object_id']!r}, {e['reason']!r}) differs "
                                   f"from the recomputed {want}"))
    if bad:
        _fail("E_EVAL_VIEWS", bad)
    return {vid: policy[vid][0] for vid in VIEWS}


def _view_scene_id(scene, view, protocol):
    """The sample's derived IDs come from the protocol table; a fixture's are its own ID plus the suffix."""
    if scene["scene_id"] == protocol["sample"]["scene_id"]:
        return view["sample_scene_id"]
    return scene["scene_id"] + view["command_suffix"]


def resolver_config(protocol, category_map, limits=None) -> dict:
    expanded = dict(protocol["resolver"]["limits"])
    for name, value in (limits or {}).items():
        if name not in expanded:
            _fail("E_EVAL_INPUT", [("limits", f"{name!r} is not a resolver limit")])
        expanded[name] = value
    return {"schema_version": 1, "record_type": "resolver_config", "config_id": protocol["resolver"]["config_id"],
            "vocabulary_id": protocol["resolver"]["vocabulary_id"],
            "category_labels": sorted(category_map["model_vocabulary"]), "colour_labels": list(protocol["colour_labels"]),
            "limits": {n: expanded[n] for n in LIMITS}}


def bin_of(record) -> str:
    res = record["resolution"]
    return res["result"]["status"] if res["processing_status"] == "completed" else res["processing_status"]


def summarize(parses, predictions, semantic) -> dict:
    """Totals, parser coverage and per-view outcomes. No reference answer is used or reported."""
    n = len(parses)
    parsed = sum(p["parse_status"] == "parsed" for p in parses)
    views = {}
    for vid in VIEWS:
        recs = [r for r in predictions if r["view_id"] == vid]
        bins = {b: 0 for b in BINS}
        for r in recs:
            bins[bin_of(r)] += 1
        work = [r["resolution"]["diagnostics"]["work"] for r in recs]
        keys = sorted({k for w in work for k, v in w.items() if type(v) is int})
        views[vid] = {"records": len(recs),
                      "processing_status": dict(sorted(Counter(r["resolution"]["processing_status"] for r in recs).items())),
                      "bins": bins,
                      "budget_limits": dict(sorted(Counter(r["resolution"]["budget"]["name"] for r in recs
                                                           if r["resolution"]["budget"]).items())),
                      "reason_codes_nonexclusive": dict(sorted(Counter(
                          c for r in recs if r["resolution"]["result"] for c in r["resolution"]["result"]["reason_codes"]
                      ).items())),
                      "work_totals": {k: sum(w.get(k, 0) for w in work) for k in keys},
                      "work_maxima": {k: max((w.get(k, 0) for w in work), default=0) for k in keys}}
    return {"format_version": 1, "record_type": "iref_prediction_summary",
            "totals": {"parent_commands": n, "views": len(VIEWS), "prediction_records": len(predictions)},
            "parser": {"parsed": parsed, "coverage": ratio(parsed, n),
                       "unsupported_reasons": dict(sorted(Counter(p["parse_reason"] for p in parses
                                                                  if p["parse_status"] == "unsupported").items())),
                       "features_nonexclusive": dict(sorted(Counter(f for p in parses for f in p["features"]).items()))},
            "views": views,
            "technical_failures": sum(views[v]["bins"]["invalid_input"] + views[v]["bins"]["budget_exceeded"] for v in VIEWS),
            "semantic_prediction_hash": semantic}


def finalize(prediction: Prediction) -> Prediction:
    """Recompute the semantic hash, the summary and the manifest's counts from the records."""
    h = semantic_hash(prediction.predictions)
    prediction.summary = summarize(prediction.parses, prediction.predictions, h)
    prediction.manifest["semantic_prediction_hash"] = h
    prediction.manifest["counts"] = {"parent_commands": len(prediction.parses),
                                     "prediction_records": len(prediction.predictions)}
    return prediction


def _canonical_inputs(scene, commands, category_map, views_manifest, relation_config_path, direction_config_path):
    lines = "".join(f"{c['command_id']} {RV.canonical_sha256(c)}\n" for c in sorted(commands, key=lambda c: c["command_id"]))
    return {"hash_kind": "canonical_json", "scene_sha256": RV.canonical_sha256(scene),
            "category_map_sha256": RV.canonical_sha256(category_map),
            "inventory_views_sha256": RV.canonical_sha256(views_manifest),
            "relation_config_sha256": sha256(Path(relation_config_path).read_bytes()),
            "direction_config_sha256": sha256(Path(direction_config_path).read_bytes()),
            "commands_sha256": sha256(lines.encode("utf-8")), "commands": len(commands)}


def predict(scene, commands, category_map, views_manifest, *, relation_config_path, direction_config_path,
            limits=None, view_order=None, protocol=None, input_hashes=None) -> Prediction:
    """Queries and resolver results for every command in both inventory views; see the module docstring."""
    protocol = protocol or load_protocol()
    _check_inputs(scene, commands, category_map)
    identities = _check_libraries(protocol, relation_config_path, direction_config_path)
    membership = _check_views(views_manifest, scene, category_map, protocol)
    order = list(view_order) if view_order is not None else list(VIEWS)
    if sorted(order) != sorted(VIEWS) or len(order) != len(VIEWS):
        _fail("E_EVAL_INPUT", [("view_order", f"expected an order of {list(VIEWS)}")])
    by_view = {v["view_id"]: v for v in protocol["views"]}
    scenes = {}
    for vid in VIEWS:
        s = copy.deepcopy(scene)
        s["scene_id"] = _view_scene_id(scene, by_view[vid], protocol)
        s["objects"] = sorted((copy.deepcopy(o) for o in scene["objects"] if o["object_id"] in membership[vid]),
                              key=lambda o: _number(o["object_id"]))
        scenes[vid] = s
    parents = sorted(commands, key=lambda c: c["command_id"])
    contexts = {}
    for c in parents:
        for vid in VIEWS:
            ctx = copy.deepcopy(c)
            ctx["command_id"] = c["command_id"] + by_view[vid]["command_suffix"]
            ctx["scene_id"] = scenes[vid]["scene_id"]
            contexts[(c["command_id"], vid)] = ctx
    found = contract.validate([(f"derived scene {vid}", scenes[vid]) for vid in VIEWS] + [("category map", category_map)]
                              + [(ctx["command_id"], ctx) for ctx in contexts.values()])
    if found:
        _fail("E_EVAL_IDENTITY", [(f"{i.file} {i.path}", f"{i.code}: {i.message}") for i in found])
    vocabulary, colours = category_map["model_vocabulary"], protocol["colour_labels"]
    by_text, parses = {}, []
    for c in parents:
        if c["text"] not in by_text:  # one parse per distinct text, rebound to each parent
            by_text[c["text"]] = parse_record(c["command_id"], c["text"], categories=vocabulary, colours=colours)
        parses.append(dict(copy.deepcopy(by_text[c["text"]]), parent_command_id=c["command_id"]))
    bad = [p for rec in parses for p in parse_record_issues(rec, f"parse {rec['parent_command_id']}")]
    if bad:
        raise EvaluationInputError(bad)
    parse_of = {p["parent_command_id"]: p for p in parses}
    queries = {}
    for (cid, vid), ctx in contexts.items():
        s = scenes[vid]
        q = {"schema_version": 1, "record_type": "grounding_query", "query_id": ctx["command_id"] + protocol["query_suffix"],
             "scene_id": s["scene_id"], "scene_revision": s["scene_revision"], "evidence_profile": s["evidence_profile"],
             "command_id": ctx["command_id"], "interpretations": [copy.deepcopy(parse_of[cid]["interpretation"])]}
        problems, _ = RV.check_query(q)
        if problems:
            _fail("E_EVAL_IDENTITY", [(f"query for {ctx['command_id']} {p['path']}", f"{p['code']}: {p['message']}")
                                      for p in problems])
        queries[(cid, vid)] = q
    config = resolver_config(protocol, category_map, limits)
    problems = RV.check_config(config)
    if problems:
        _fail("E_EVAL_INPUT", [(f"resolver config {p['path']}", f"{p['code']}: {p['message']}") for p in problems])
    results = {}
    for vid in order:
        for c in parents:
            key = (c["command_id"], vid)
            results[key] = resolve(scenes[vid], contexts[key], queries[key], resolver_config=config,
                                   relation_config_path=Path(relation_config_path),
                                   direction_config_path=Path(direction_config_path), category_maps=[category_map],
                                   trace=False)
    records = [{"format_version": 1, "record_type": "iref_prediction", "parent_command_id": c["command_id"],
                "view_id": vid, "command_id": contexts[(c["command_id"], vid)]["command_id"],
                "scene_id": scenes[vid]["scene_id"], "query": queries[(c["command_id"], vid)],
                "resolution": results[(c["command_id"], vid)]} for c in parents for vid in VIEWS]
    manifest = {
        "format_version": 1, "record_type": "iref_prediction_manifest", "evaluation_version": EVALUATION_VERSION,
        "protocol_id": protocol["protocol_id"], "protocol_sha256": RV.canonical_sha256(protocol),
        "source_sample": {"status": "declared", "scene_id": scene["scene_id"], "scene_revision": scene["scene_revision"],
                          "category_map_id": category_map["map_id"], "scene_sources": copy.deepcopy(scene["sources"]),
                          "command_source_releases": sorted({s["release"] for c in commands for s in c["sources"]})},
        "parent": {"scene_id": scene["scene_id"], "scene_revision": scene["scene_revision"],
                   "evidence_profile": scene["evidence_profile"]},
        "action": protocol["action"], "action_origin": protocol["action_origin"],
        "views": [{"view_id": vid, "scene_id": scenes[vid]["scene_id"], "command_suffix": by_view[vid]["command_suffix"],
                   "policy": by_view[vid]["policy"], "objects": len(scenes[vid]["objects"]),
                   "excluded_object_ids": sorted(set(o["object_id"] for o in scene["objects"]) - membership[vid],
                                                 key=_number)} for vid in VIEWS],
        "vocabulary": {"category_map_id": category_map["map_id"], "category_labels": len(config["category_labels"]),
                       "colour_labels": list(colours), "size_words": list(protocol["size_words"])},
        "resolver": {"config_id": config["config_id"], "vocabulary_id": config["vocabulary_id"],
                     "limits": dict(config["limits"]), "trace": False, "config_sha256": RV.canonical_sha256(config)},
        "library_identities": identities,
        "inputs": input_hashes or _canonical_inputs(scene, commands, category_map, views_manifest, relation_config_path,
                                                   direction_config_path),
        "code": code_hashes(), "runtime": runtime(), "semantic_hash_rule": protocol["semantic_hash_rule"],
        "semantic_prediction_hash": None, "counts": None}
    prediction = Prediction(manifest=manifest, category_map=copy.deepcopy(category_map), resolver_config=config,
                            scenes=scenes, parses=parses,
                            contexts=[contexts[(c["command_id"], vid)] for c in parents for vid in VIEWS],
                            predictions=records, summary={})
    return finalize(prediction)


def prediction_files(prediction: Prediction) -> dict:
    return {"manifest.json": encode_json(prediction.manifest), "category-map.json": encode_json(prediction.category_map),
            "resolver-config.json": encode_json(prediction.resolver_config),
            "scenes/full_inventory.json": encode_json(prediction.scenes["full_inventory"]),
            "scenes/source_known_nyu.json": encode_json(prediction.scenes["source_known_nyu"]),
            "parses.jsonl": encode_jsonl(prediction.parses), "command-contexts.jsonl": encode_jsonl(prediction.contexts),
            "predictions.jsonl": encode_jsonl(prediction.predictions),
            "prediction-summary.json": encode_json(prediction.summary)}


def publish_prediction(prediction: Prediction, out) -> None:
    output.publish(out, prediction_files(prediction))


def run_predict(*, scene, commands, category_map, inventory_views, relation_config, direction_config, out,
                sample_only=True, protocol_path=None) -> dict:
    """The command line's operation: read exactly these inputs, predict, publish; returns the prediction summary."""
    out = output.refuse_existing(out)
    protocol = load_protocol(protocol_path) if protocol_path else load_protocol()
    paths = {"scene": Path(scene), "category_map": Path(category_map), "inventory_views": Path(inventory_views),
             "relation_config": Path(relation_config), "direction_config": Path(direction_config)}
    data = {}
    for name, p in paths.items():
        try:
            data[name] = p.read_bytes()
        except OSError as e:
            _fail("E_EVAL_INPUT", [(str(p), f"cannot read the {name} file: {e}")])
    records = {name: strict_json(name, data[name], "E_EVAL_INPUT") for name in ("scene", "category_map", "inventory_views")}
    for name in ("scene", "category_map"):
        if not isinstance(records[name], dict):
            _fail("E_EVAL_INPUT", [(str(paths[name]), f"the {name.replace('_', ' ')} file must hold one JSON object, not "
                                                      f"{_kind(records[name])}")])
    cdir = Path(commands)
    if not cdir.is_dir():
        _fail("E_EVAL_INPUT", [(str(cdir), "the commands path is not a folder")])
    entries = sorted(cdir.iterdir(), key=lambda p: p.name)
    strays = [p.name for p in entries if not (p.is_file() and p.suffix == ".json")]
    if strays:
        _fail("E_EVAL_INPUT", [(str(cdir), f"only command .json files are allowed; found {strays[:5]}")])
    cmds, lines = [], []
    for p in entries:
        b = p.read_bytes()
        c = strict_json(p.name, b, "E_EVAL_INPUT")
        if not isinstance(c, dict) or c.get("command_id") != p.stem:
            _fail("E_EVAL_INPUT", [(p.name, "a command file must hold one command named by its command_id")])
        cmds.append(c)
        lines.append(f"{c['command_id']} {sha256(b)}\n")
    if sample_only and (records["scene"].get("scene_id") != protocol["sample"]["scene_id"]
                        or records["category_map"].get("map_id") != protocol["sample"]["category_map_id"]):
        _fail("E_EVAL_INPUT", [("scene", f"this command line evaluates only the A2.2a sample "
                                         f"{protocol['sample']['scene_id']!r} with map "
                                         f"{protocol['sample']['category_map_id']!r}")])
    hashes = {"hash_kind": "file_bytes", "scene_sha256": sha256(data["scene"]),
              "category_map_sha256": sha256(data["category_map"]),
              "inventory_views_sha256": sha256(data["inventory_views"]),
              "relation_config_sha256": sha256(data["relation_config"]),
              "direction_config_sha256": sha256(data["direction_config"]),
              "commands_sha256": sha256("".join(sorted(lines)).encode("utf-8")), "commands": len(cmds)}
    prediction = predict(records["scene"], cmds, records["category_map"], records["inventory_views"],
                         relation_config_path=paths["relation_config"], direction_config_path=paths["direction_config"],
                         protocol=protocol, input_hashes=hashes)
    publish_prediction(prediction, out)
    return prediction.summary


# --------------------------------------------------------------------------------------------- reading back
def _jsonl(name, data: bytes):
    text = data.decode("utf-8") if data.endswith(b"\n") else None
    if text is None:
        _fail("E_EVAL_PREDICTIONS", [(name, "a JSONL file must end with LF")])
    return [strict_json(f"{name} line {i + 1}", line.encode("utf-8"), "E_EVAL_PREDICTIONS")
            for i, line in enumerate(text[:-1].split("\n"))] if text != "" else []


def _readback_shape(prediction) -> list:
    """Shapes check_prediction and scoring rely on, checked before any of their fields is read (D76)."""
    out = []

    def bad(path, message):
        out.append(issue(path, "E_EVAL_PREDICTIONS", message))
    m = prediction.manifest
    if not isinstance(m, dict):
        bad("manifest.json", f"expected one JSON object, not {_kind(m)}")
    elif set(m) != MANIFEST_KEYS:
        bad("manifest.json", f"expected exactly the keys {sorted(MANIFEST_KEYS)}")
    else:
        if m["record_type"] != "iref_prediction_manifest":
            bad("manifest.json $.record_type", f"{m['record_type']!r} is not iref_prediction_manifest")
        if not _is_version_1(m["format_version"]):
            bad("manifest.json $.format_version", f"{m['format_version']!r} is not the supported version, the integer 1")
        views = m["views"]
        if not isinstance(views, list) or len(views) != len(VIEWS) or not all(
                isinstance(v, dict) and all(isinstance(v.get(k), str) for k in ("view_id", "scene_id", "command_suffix"))
                for v in views) or sorted(v["view_id"] for v in views) != sorted(VIEWS):
            bad("manifest.json $.views", f"expected one object per view {list(VIEWS)}, each with string view_id, "
                                         "scene_id and command_suffix")
        r = m["resolver"]
        if not (isinstance(r, dict) and isinstance(r.get("limits"), dict) and isinstance(r.get("config_sha256"), str)):
            bad("manifest.json $.resolver", "expected an object with a limits object and a config_sha256 string")
        ids = m["library_identities"]
        if not (isinstance(ids, dict) and isinstance(ids.get("relation_config"), str)
                and isinstance(ids.get("direction_config"), str)):
            bad("manifest.json $.library_identities", "expected an object with relation_config and direction_config strings")
        if not (isinstance(m["parent"], dict) and isinstance(m["parent"].get("scene_id"), str)):
            bad("manifest.json $.parent", "expected an object with a scene_id string")
        for key in ("protocol_id", "semantic_prediction_hash"):
            if not isinstance(m[key], str):
                bad(f"manifest.json $.{key}", f"expected a string, not {_kind(m[key])}")
    s = prediction.summary
    if not isinstance(s, dict):
        bad("prediction-summary.json", f"expected one JSON object, not {_kind(s)}")
    elif not _is_version_1(s.get("format_version")):
        bad("prediction-summary.json $.format_version", f"{s.get('format_version')!r} is not the supported version, "
                                                        "the integer 1")
    sc = prediction.scenes
    if not isinstance(sc, dict) or set(sc) != set(VIEWS) or not all(isinstance(sc[v], dict) for v in VIEWS):
        bad("scenes", f"expected one scene object for each view {list(VIEWS)}")
    for name, value in (("category-map.json", prediction.category_map), ("resolver-config.json", prediction.resolver_config)):
        if not isinstance(value, dict):
            bad(name, f"expected one JSON object, not {_kind(value)}")
    for name, records in (("parses.jsonl", prediction.parses), ("command-contexts.jsonl", prediction.contexts),
                          ("predictions.jsonl", prediction.predictions)):
        if not isinstance(records, list):
            bad(name, f"expected a list of records, not {_kind(records)}")
            continue
        for i, rec in enumerate(records):
            if not isinstance(rec, dict):
                bad(f"{name} record {i + 1}", f"expected a record object, not {_kind(rec)}")
    return out


def check_prediction(prediction: Prediction) -> list:
    """Every invariant of a prediction output; an empty list means it may be scored."""
    out = []

    def bad(path, message):
        out.append(issue(path, "E_EVAL_PREDICTIONS", message))
    out.extend(_readback_shape(prediction))
    if out:
        return out
    m = prediction.manifest
    if not isinstance(m, dict) or set(m) != MANIFEST_KEYS or m.get("record_type") != "iref_prediction_manifest" \
            or type(m.get("format_version")) is not int:
        bad("manifest.json", f"expected exactly the keys {sorted(MANIFEST_KEYS)}")
        return out
    scenes, cmap, config = prediction.scenes, prediction.category_map, prediction.resolver_config
    if set(scenes) != set(VIEWS):
        bad("scenes", f"expected the views {list(VIEWS)}")
        return out
    found = contract.validate([(f"scenes/{v}.json", scenes[v]) for v in VIEWS] + [("category-map.json", cmap)]
                              + [(f"command-contexts.jsonl {c.get('command_id')!r}", c) for c in prediction.contexts])
    for i in found:
        bad(f"{i.file} {i.path}", f"{i.code}: {i.message}")
    for i in RV.check_config(config):
        bad(f"resolver-config.json {i['path']}", f"{i['code']}: {i['message']}")
    if out:
        return out
    if config["limits"] != m["resolver"]["limits"] or RV.canonical_sha256(config) != m["resolver"]["config_sha256"] \
            or config["category_labels"] != sorted(cmap["model_vocabulary"]):
        bad("resolver-config.json", "the resolver configuration disagrees with the manifest or the category map")
    view_ids = {v["view_id"]: v for v in m["views"]}
    for vid in VIEWS:
        if view_ids.get(vid, {}).get("scene_id") != scenes[vid]["scene_id"]:
            bad(f"scenes/{vid}.json", "the scene ID disagrees with the manifest")
    parses = prediction.parses
    parents = [p.get("parent_command_id") if isinstance(p, dict) else None for p in parses]
    for p, pid in zip(parses, parents):
        out.extend(parse_record_issues(p, f"parses.jsonl {pid!r}"))
    if any(not isinstance(x, str) for x in parents) or parents != sorted(set(parents)):
        bad("parses.jsonl", "parent commands must be unique and ordered")
    if out:
        return out
    parse_of = {p["parent_command_id"]: p for p in parses}
    ctx_of = {}
    for c in prediction.contexts:
        if c["command_id"] in ctx_of:
            bad(f"command-contexts.jsonl {c['command_id']!r}", "repeated command context")
        ctx_of[c["command_id"]] = c
    suffix = {v["view_id"]: v["command_suffix"] for v in m["views"]}
    required = {parent + suffix[vid]: (parent, vid) for parent in parse_of for vid in VIEWS}
    for cid in sorted(set(ctx_of) - set(required)):
        bad(f"command-contexts.jsonl {cid!r}", f"context {cid} belongs to no parent command and view: an extra context")
    for cid, (parent, vid) in sorted(required.items()):
        if cid not in ctx_of:
            bad(f"command-contexts.jsonl {cid!r}", f"parent {parent} has no {vid} context ({cid})")
        elif ctx_of[cid]["text"] != parse_of[parent]["text"]:  # exact: no trimming, case folding or normalization
            bad(f"command-contexts.jsonl {cid!r} $.text", f"parent {parent}, view {vid}: the context text "
                f"{ctx_of[cid]['text']!r} is not the parse text {parse_of[parent]['text']!r}")
    seen, pairs = set(), {}
    sha = {vid: RV.canonical_sha256(scenes[vid]) for vid in VIEWS}
    config_sha = RV.canonical_sha256(config)
    for rec in prediction.predictions:
        where = f"predictions.jsonl {rec.get('parent_command_id')!r} {rec.get('view_id')!r}"
        found = schema_issues("prediction", rec, "predictions.jsonl", "E_EVAL_PREDICTIONS")
        if found:
            out.extend(found)
            continue
        cid, vid = rec["parent_command_id"], rec["view_id"]
        if (cid, vid) in seen:
            bad(where, f"duplicate prediction for {cid} in {vid}")
            continue
        seen.add((cid, vid))
        pairs.setdefault(cid, []).append(vid)
        suffix = view_ids[vid]["command_suffix"]
        ctx, parse, q, res = ctx_of.get(rec["command_id"]), parse_of.get(cid), rec["query"], rec["resolution"]
        if parse is None:
            bad(where, f"prediction for {cid}, which has no parse record")
            continue
        if ctx is None or rec["command_id"] != cid + suffix or ctx["scene_id"] != scenes[vid]["scene_id"] \
                or rec["scene_id"] != scenes[vid]["scene_id"]:
            bad(where, f"identities of {cid} disagree with its context or view scene")
            continue
        for i in RV.check_query(q)[0]:
            bad(f"{where} query {i['path']}", f"{i['code']}: {i['message']}")
        res_errors = sorted(_validator("resolution").iter_errors(res), key=lambda e: list(map(str, e.absolute_path)))
        for e in res_errors:
            bad(f"{where} resolution", e.message[:200])
        if out:
            continue
        want_q = {"query_id": rec["command_id"] + ".q", "scene_id": scenes[vid]["scene_id"],
                  "scene_revision": scenes[vid]["scene_revision"], "evidence_profile": scenes[vid]["evidence_profile"],
                  "command_id": rec["command_id"]}
        if any(q[k] != v for k, v in want_q.items()) or q["interpretations"] != [parse["interpretation"]]:
            bad(where, f"the query of {cid} disagrees with its context or parse")
        d = res["diagnostics"]
        ids = d["identities"]
        if d["context"] not in (None, want_q) or (res["result"] is not None and any(
                res["result"][k] != want_q[k] for k in want_q)):
            bad(where, f"the resolution of {cid} names another query, scene or command")
        if ids is not None and (ids["scene_sha256"] != sha[vid] or ids["command_sha256"] != RV.canonical_sha256(ctx)
                                or ids["query_sha256"] != RV.canonical_sha256(q) or ids["resolver_config_sha256"] != config_sha
                                or ids["relation_config_identity"] != m["library_identities"]["relation_config"]
                                or ids["direction_config_identity"] != m["library_identities"]["direction_config"]):
            bad(where, f"the resolution identities of {cid} disagree with its inputs")
    for cid in sorted(set(pairs) | set(parse_of)):
        if sorted(pairs.get(cid, [])) != sorted(VIEWS):
            bad(f"predictions.jsonl {cid!r}", f"{cid} has predictions for {sorted(pairs.get(cid, []))}, not both views")
    if out:
        return out
    order = [(r.get("parent_command_id"), VIEWS.index(r["view_id"]) if r.get("view_id") in VIEWS else -1)
             for r in prediction.predictions]
    if order != sorted(order):
        bad("predictions.jsonl", "records must be ordered by parent command, then full before source-known")
    if out:
        return out
    h = semantic_hash(prediction.predictions)
    if h != m["semantic_prediction_hash"] or h != prediction.summary.get("semantic_prediction_hash"):
        bad("predictions.jsonl", "the semantic hash disagrees with the manifest or summary: records were changed")
    if summarize(parses, prediction.predictions, h) != prediction.summary:
        bad("prediction-summary.json", "the summary's counts disagree with the records: recomputed, not trusted")
    if m["counts"] != {"parent_commands": len(parses), "prediction_records": len(prediction.predictions)}:
        bad("manifest.json $.counts", "the manifest's counts disagree with the records")
    return out


def load_prediction(directory) -> Prediction:
    """A published prediction folder, read back and checked in full; EvaluationInputError if anything disagrees."""
    d = Path(directory)
    present = sorted(str(p.relative_to(d)).replace("\\", "/") for p in d.rglob("*") if p.is_file()) if d.is_dir() else []
    if present != sorted(FILES):
        _fail("E_EVAL_PREDICTIONS", [(str(d), f"expected exactly the files {sorted(FILES)}; found {present[:12]}")])
    raw = {f: (d / f).read_bytes() for f in FILES}
    js = {f: strict_json(f, raw[f], "E_EVAL_PREDICTIONS") for f in FILES if f.endswith(".json")}
    prediction = Prediction(manifest=js["manifest.json"], category_map=js["category-map.json"],
                            resolver_config=js["resolver-config.json"],
                            scenes={v: js[f"scenes/{v}.json"] for v in VIEWS}, parses=_jsonl("parses.jsonl", raw["parses.jsonl"]),
                            contexts=_jsonl("command-contexts.jsonl", raw["command-contexts.jsonl"]),
                            predictions=_jsonl("predictions.jsonl", raw["predictions.jsonl"]),
                            summary=js["prediction-summary.json"])
    found = check_prediction(prediction)
    if found:
        raise EvaluationInputError(found)
    prediction.raw_predictions_sha256 = sha256(raw["predictions.jsonl"])  # actual-file provenance for scoring
    return prediction


__all__ = ["Prediction", "bin_of", "check_prediction", "finalize", "load_prediction", "predict", "prediction_files",
           "publish_prediction", "resolver_config", "run_predict", "summarize"]
