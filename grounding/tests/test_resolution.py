"""Tests of the A2.1e structured resolver (D71).

    python grounding/tests/test_resolution.py      (or: python -m grounding.tests.test_resolution)

Needs only Python 3.9+ and jsonschema. Every expectation comes from fixtures/resolution/cases.json and dispatch.json,
both fixed before the resolver existed (their hashes are in notes/phases/A2.1e_resolution.md). Dispatch values come
from the reviewed direction and relation case tables, never from the code under test. Each R-fixture case gets its own
scene ID, so the relation library's geometry cache never sees two snapshots under one key. Prints one line per check
and exits 1 if any fails.
"""
from __future__ import annotations

import contextlib
import importlib
import copy
import io
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))  # the repository root only, so the file also runs directly
from grounding.contract import validate as contract  # noqa: E402

FX = REPO / "grounding" / "tests" / "fixtures"
RES = FX / "resolution"
REL_CFG = REPO / "grounding" / "relations" / "relations.v1.json"
DIR_CFG = REPO / "grounding" / "relations" / "directions.v1.json"
FAILED, COUNT = [], [0]


def check(name, ok, detail=""):
    COUNT[0] += 1
    print(("PASS  " if ok else "FAIL  ") + name + (f": {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


DOC = load(RES / "cases.json")
CASES = {c["id"]: c for c in DOC["cases"]}
BASE_SCENE, BASE_COMMAND = load(RES / "scene.r.json"), load(RES / "command.r.json")
BASE_CONFIG, MAP = load(RES / "resolver-config.fixture.json"), load(FX / "contract/valid/category-map.fx.json")
UNKNOWN = {"state": "unknown", "reason": "not_in_source", "source": "fx"}


def known(value, evidence=None):
    return {"state": "known", "value": value,
            "evidence": evidence or {"kind": "annotated", "source": "fx", "assumptions": []}}


# ---------------------------------------------------------------------------------------- building case records
def interpretations_of(case):
    if "interpretations" in case:
        return case["interpretations"]
    return CASES[case["base_case"]]["interpretations"]


def build(case):
    """(scene, command, query, maps, resolver_config, relation_config_path) for one fixed case."""
    v = case.get("variant", {})
    if "records" in case:
        scene, command = load(FX / case["records"]["scene"]), load(FX / case["records"]["command"])
        if "keep" in case["records"]:
            scene["objects"] = [o for o in scene["objects"] if o["object_id"] in case["records"]["keep"]]
        maps = [MAP] if scene["category_map"] == MAP["map_id"] else []
    else:
        cid = case["id"].lower()
        scene, command = copy.deepcopy(BASE_SCENE), copy.deepcopy(BASE_COMMAND)
        scene["scene_id"] = command["scene_id"] = f"fixture.resolver.{cid}"
        command["command_id"], command["text"] = f"fx.{cid}.command", f"Resolver unit case {case['id']}"
        if "keep" in v:
            scene["objects"] = [o for o in scene["objects"] if o["object_id"] in v["keep"]]
        scene["assumptions"] += v.get("scene_assumptions", [])
        for oid, change in v.get("objects", {}).items():
            o = next(o for o in scene["objects"] if o["object_id"] == oid)
            for field, value in change.items():
                if field == "colours":
                    o["attributes"]["colours"] = dict(UNKNOWN) if value == "unknown" else known(value)
                elif field == "colours_evidence":
                    o["attributes"]["colours"]["evidence"] = value
                elif field == "category":
                    o["category"] = dict(UNKNOWN)
                elif field == "front":
                    o["semantic_front"] = dict(UNKNOWN) if value == "unknown" else known(value)
                elif field == "centre":
                    o["geometry"]["center_m"] = dict(UNKNOWN) if value == "unknown" else known(value)
                else:
                    raise KeyError(field)
        if v.get("reverse_objects"):
            scene["objects"].reverse()
        for field, value in v.get("command", {}).items():
            if field == "heading":
                command["user_pose"]["heading_xy"] = dict(UNKNOWN)
            elif field == "position":
                command["user_pose"]["position_m"] = known(value)
            elif field == "text":
                command["text"] = value
            elif field == "scene_id":
                command["scene_id"] = value
            elif field == "frame_id":
                command["user_pose"]["frame_id"] = value
            else:
                raise KeyError(field)
        other = dict(copy.deepcopy(MAP), map_id="map.other")
        unmapped = copy.deepcopy(MAP)
        unmapped["entries"] = [e for e in unmapped["entries"] if e["standard"] != "table"]
        maps = {"default": [MAP], "omit": [], "other_only": [other], "incompatible": [unmapped],
                "matching_plus_other": [MAP, other], "duplicate": [MAP, copy.deepcopy(MAP)]}[
            v.get("category_maps", "default")]
    config = copy.deepcopy(BASE_CONFIG)
    config["limits"].update(v.get("limits", {}))
    for key, value in v.get("resolver_config", {}).items():
        if key.startswith("limits."):
            config["limits"][key[7:]] = value
        else:
            config[key] = value
    rel = REL_CFG if v.get("relation_config") != "missing" else RES / "no-such-relations.json"
    query = {"schema_version": 1, "record_type": "grounding_query",
             "query_id": f"fx.{case['id'].lower()}.query" if "records" not in case else f"fx.{case['id'].lower()}.query",
             "scene_id": scene["scene_id"], "scene_revision": scene["scene_revision"],
             "evidence_profile": scene["evidence_profile"], "command_id": command["command_id"],
             "interpretations": copy.deepcopy(interpretations_of(case))}
    query.update(case.get("envelope_patch", {}))
    return scene, command, query, maps, config, rel


def resolve_case(case, trace=False, **overrides):
    from grounding.resolution import resolve
    scene, command, query, maps, config, rel = build(case)
    config = overrides.get("config", config)
    return resolve(scene, command, query, resolver_config=config, relation_config_path=rel,
                   direction_config_path=DIR_CFG, category_maps=maps, trace=trace)


# ------------------------------------------------------------------------------------------ comparing outcomes
SEMANTIC = ("status", "action", "target_id", "reason_code", "reason_codes", "candidates", "assumptions", "conditional")


def semantic(record):
    out = {"processing_status": record["processing_status"], "budget": record["budget"],
           "issues": [i["code"] for i in record["issues"]], "work": record["diagnostics"]["work"],
           "outcome_counts": record["diagnostics"]["outcome_counts"]}
    if record["result"] is not None:
        out.update({k: record["result"][k] for k in SEMANTIC})
    return out


def compare(case, record):
    e, problems = case["expect"], []
    result, diag = record["result"], record["diagnostics"]

    def want(cond, text):
        if not cond:
            problems.append(text)

    want(record["processing_status"] == e["processing_status"],
         f"processing_status {record['processing_status']} (issues {[i['code'] for i in record['issues']]})")
    if record["processing_status"] != e["processing_status"]:
        return problems
    if e["processing_status"] == "invalid_input":
        codes = {i["code"] for i in record["issues"]}
        want(set(e["issue_codes_include"]) <= codes, f"issue codes {sorted(codes)}")
        return problems
    if e["processing_status"] == "budget_exceeded":
        want(record["budget"] == e["budget"], f"budget {record['budget']}")
    for key, value in e.get("work", {}).items():
        want(diag["work"][key] == value, f"work.{key} = {diag['work'][key]}, expected {value}")
    for key, value in e.get("outcome_counts", {}).items():
        want(diag["outcome_counts"][key] == value, f"outcome_counts.{key} = {diag['outcome_counts'][key]}")
    if "warning_codes" in e:
        want([w["code"] for w in diag["warnings"]] == e["warning_codes"], f"warnings {diag['warnings']}")
    if result is None:
        return problems
    for key in ("status", "target_id", "action", "coverage", "conditional"):
        if key in e:
            got = result["candidates"]["coverage"] if key == "coverage" else result[key]
            want(got == e[key], f"{key} = {got!r}, expected {e[key]!r}")
    for key in ("target_ids", "uncertain_match_ids", "rank_definite_eligible_ids", "rank_possible_eligible_ids"):
        if key in e:
            want(result["candidates"][key] == e[key], f"{key} = {result['candidates'][key]}, expected {e[key]}")
    if "reason_codes_include" in e:
        want(set(e["reason_codes_include"]) <= set(result["reason_codes"]), f"reason_codes {result['reason_codes']}")
    got = [[a["record_type"], a["record_id"], a["assumption_id"]] for a in result["assumptions"]]
    if "assumptions" in e:
        want(got == e["assumptions"], f"assumptions {got}")
    for a in e.get("assumptions_include", []):
        want(a in got, f"assumption {a} missing from {got}")
    return problems


# --------------------------------------------------------------------------------------------------- the checks
def check_fixtures():
    print("-- fixtures (accepted contract validator only)")
    bad = []
    for case in DOC["cases"]:
        scene, command, _, maps, _, _ = build(case)
        issues = contract.validate([("scene", scene), ("command", command)]
                                   + [(f"category_map[{i}]", m) for i, m in enumerate(maps)])
        errors = sorted({i.code for i in issues if i.is_error})
        expected = {"V33": ["E_CATEGORY_UNMAPPED"], "V35": ["E_DUPLICATE_RECORD"], "V37": ["E_FRAME_MISMATCH"]}.get(
            case["id"], [])
        if errors != expected:
            bad.append(f"{case['id']}: {errors}")
    check(f"every case's scene, command and maps are contract-valid, except the 3 cases built to fail it "
          f"({len(DOC['cases'])} cases)", not bad, "; ".join(bad[:5]))


def check_cases(prefixes, title):
    print(f"-- {title}")
    for case in DOC["cases"]:
        if not case["id"].startswith(prefixes) or "same_as" in case["expect"] or "trace_invariance" in case["expect"]:
            continue
        try:
            problems = compare(case, resolve_case(case))
        except Exception as e:  # report, don't crash: one broken case must not hide the others
            problems = [f"raised {type(e).__name__}: {str(e)[:160]}"]
        check(f"{case['id']}: {case.get('note') or 'brief section 12.2'} -> {case['expect'].get('status') or case['expect']['processing_status']}",
              not problems, "; ".join(problems))


def check_metamorphic():
    print("-- metamorphic: order, text and caches")
    for case in DOC["cases"]:
        if "same_as" not in case["expect"]:
            continue
        base = CASES[case["expect"]["same_as"]]
        a, b = resolve_case(base, trace=case["expect"].get("same_trace", False)), \
            resolve_case(case, trace=case["expect"].get("same_trace", False))
        same = semantic(a) == semantic(b)
        if case["expect"].get("same_trace"):
            same = same and a["diagnostics"]["trace"] == b["diagnostics"]["trace"]
        check(f"{case['id']}: {case['note']} -> the same semantic result, counters"
              + (" and trace" if case["expect"].get("same_trace") else ""), same)
    first = resolve_case(CASES["R08"])
    second = resolve_case(CASES["R08"])
    check("caches belong to one call: a second R08 resolution makes the same 4 calls and 0 hits",
          second["diagnostics"]["work"]["predicate_calls"] == 4
          and second["diagnostics"]["work"]["predicate_cache_hits"] == 0 and semantic(first) == semantic(second))


def check_trace():
    print("-- trace")
    R = importlib.import_module("grounding.resolution.resolve")  # the module; the package's resolve is the function
    off, on = resolve_case(CASES["R08"]), resolve_case(CASES["R08"], trace=True)
    check("trace on and off give identical semantic results and work counters",
          semantic(off) == semantic(on) and on["diagnostics"]["trace"] and not on["diagnostics"]["trace_truncated"])
    check("with trace off: no events, not truncated, 0 bytes", off["diagnostics"]["trace"] == []
          and off["diagnostics"]["trace_truncated"] is False and off["diagnostics"]["trace_bytes"] == 0)
    kinds = {e["kind"] for e in on["diagnostics"]["trace"]}
    check("R08's trace has node, binding, predicate and terminal events", {"node", "binding", "predicate",
                                                                          "terminal"} <= kinds, str(sorted(kinds)))
    events = [e for e in on["diagnostics"]["trace"] if e["kind"] == "predicate"]
    check("predicate events record the 4 actual calls in target/anchor order with the library's reasons",
          len(events) == 4 and events[0]["object_ids"] == ["obj_003", "obj_001"] and events[0]["outcome"] == "TRUE"
          and events[1]["object_ids"] == ["obj_004", "obj_001"] and events[1]["reasons"] == ["failed:right"],
          json.dumps(events[:2]))
    size = sum(len(json.dumps(e, separators=(",", ":"), ensure_ascii=False).encode()) for e in on["diagnostics"]["trace"])
    check("trace_bytes is the sum of the events' compact UTF-8 JSON lengths", size == on["diagnostics"]["trace_bytes"])
    for name, limits in (("2 events", {"max_trace_events": 2}), ("300 bytes", {"max_trace_bytes": 300})):
        case = dict(CASES["R08"], variant={"limits": limits})
        small = resolve_case(case, trace=True)
        d = small["diagnostics"]
        check(f"a {name} trace cap truncates the trace only: same semantic result and counters",
              d["trace_truncated"] and semantic(small) == semantic(on) and len(d["trace"]) <= limits.get(
                  "max_trace_events", 10 ** 9) and d["trace_bytes"] <= limits.get("max_trace_bytes", 10 ** 9),
              f"{len(d['trace'])} events, {d['trace_bytes']} bytes")
    real = R._trace_event

    def refuse(*a, **k):
        raise AssertionError("a trace event was built with tracing off")

    R._trace_event = refuse
    try:
        quiet = resolve_case(CASES["R08"])
        ok = quiet["result"]["status"] == "resolved"
    except AssertionError:
        ok = False
    finally:
        R._trace_event = real
    check("the default path builds no trace data (event construction made to raise)", ok)


def check_budget_instrumentation():
    print("-- work limits before work")
    from grounding.relations import directions as D
    from grounding.relations import predicates as P
    saved = (P.Relations.rank,)

    def refuse(*a, **k):
        raise AssertionError("rank() was called")

    P.Relations.rank = refuse
    try:
        record = resolve_case(CASES["B04b"])
        ok = compare(CASES["B04b"], record) == []
    except AssertionError:
        ok = False
    finally:
        (P.Relations.rank,) = saved
    check("R16 with 3 rank subsets: budget_exceeded before rank() is ever called (rank made to raise)", ok)
    saved = (P.Relations.__init__, D.DirectionalRelations.__init__)

    def boom(*a, **k):
        raise AssertionError("a relation evaluator was built")

    P.Relations.__init__ = D.DirectionalRelations.__init__ = boom
    try:
        results = [compare(CASES[c], resolve_case(CASES[c])) == [] for c in ("B07b", "B07c", "B07d")]
        ok = all(results)
    except AssertionError:
        ok = False
    finally:
        P.Relations.__init__, D.DirectionalRelations.__init__ = saved
    check("static-limit failures build no relation evaluator and make no relation call", ok)


def check_dispatch():
    print("-- dispatch to the accepted libraries (expected values from the reviewed tables)")
    from grounding.resolution import resolve
    disp = load(RES / "dispatch.json")
    dtab, rtab = load(FX / "directions/cases.json"), load(FX / "relations/cases.json")
    table = {"true": "TRUE", "false": "FALSE", "unknown": "UNKNOWN"}
    mismatches, reached = [], 0

    def run(scene, command, keep, root_cat, relation, frame, anchors, cats):
        sub = copy.deepcopy(scene)
        sub["objects"] = [o for o in sub["objects"] if o["object_id"] in keep]
        nodes = [{"node_id": f"anchor{i}", "category": cats[a], "colours_all": [], "constraints": [], "rank": None}
                 for i, a in enumerate(anchors)]
        nodes.append({"node_id": "target", "category": root_cat, "colours_all": [], "rank": None,
                      "constraints": [{"relation": relation, "frame": frame,
                                       "anchors": [f"anchor{i}" for i in range(len(anchors))]}]})
        query = {"schema_version": 1, "record_type": "grounding_query", "query_id": "fx.dispatch.query",
                 "scene_id": sub["scene_id"], "scene_revision": sub["scene_revision"],
                 "evidence_profile": sub["evidence_profile"], "command_id": command["command_id"],
                 "interpretations": [{"interpretation_id": "i0", "kind": "query", "action": "INSPECT",
                                      "root": "target", "nodes": nodes}]}
        maps = [MAP] if sub["category_map"] == MAP["map_id"] else []
        return resolve(sub, command, query, resolver_config=BASE_CONFIG, relation_config_path=REL_CFG,
                       direction_config_path=DIR_CFG, category_maps=maps, trace=True)

    def category(scene, oid):
        o = next(o for o in scene["objects"] if o["object_id"] == oid)
        return o["category"]["value"]["model"] if o["category"]["state"] == "known" else None

    rows = {f"{row['id']}.{i + 1}": call for row in dtab["rows"] for i, call in enumerate(row["calls"])}
    for cid in disp["direction_calls"]:
        call = rows[cid]
        scene = load(FX / "directions" / dtab["scenes"][call["scene"]])
        cmd_key = call["command"] or disp["direction_default_commands"][call["scene"]]
        command = load(FX / "directions" / dtab["commands"][cmd_key])
        anchors = [call["anchor"]] if call["anchor"] else []
        cats = {oid: category(scene, oid) for oid in [call["target"]] + anchors}
        rec = run(scene, command, {call["target"], *anchors}, cats[call["target"]], call["relation"], call["frame"],
                  anchors, cats)
        want = [call["target"]] + anchors
        ev = [e for e in rec["diagnostics"]["trace"] if e["kind"] == "predicate" and e["object_ids"] == want
              and e["relation"] == call["relation"] and e["frame"] == call["frame"]]
        reached += bool(ev)
        if not ev or ev[0]["outcome"] != table[call["expect"]["value"]]:
            mismatches.append(f"{cid}: {ev[0]['outcome'] if ev else 'not reached'} vs {call['expect']['value']}")
    check(f"all {len(disp['direction_calls'])} reviewed directional calls reached through queries give the table's "
          "value", not mismatches and reached == len(disp["direction_calls"]), "; ".join(mismatches[:6]))
    mismatches, reached = [], 0
    rcases = {c["id"]: c for c in rtab["cases"]}
    base_cmd = load(FX / "contract/valid/command.a17.c001.json")
    for cid in disp["relation_cases"]:
        c = rcases[cid]
        scene = load(FX / "relations" / rtab["scenes"][c["scene"]])
        command = copy.deepcopy(base_cmd)
        command["scene_id"], command["scene_revision"] = scene["scene_id"], scene["scene_revision"]
        command["user_pose"]["scene_revision"] = scene["scene_revision"]
        t, *anchors = c["call"]["args"]
        cats = {oid: category(scene, oid) for oid in c["call"]["args"]}
        rec = run(scene, command, set(c["call"]["args"]), cats[t], c["call"]["relation"], None, anchors, cats)
        # near is symmetric and memoized canonically (brief section 8): of near(a, b) and near(b, a) only the first
        # call is traced, so its arguments are compared as a set; between's anchors likewise
        if c["call"]["relation"] == "near":
            match = lambda ids: sorted(ids) == sorted([t] + anchors)  # noqa: E731
        else:
            match = lambda ids: ids[0] == t and sorted(ids[1:]) == sorted(anchors)  # noqa: E731
        ev = [e for e in rec["diagnostics"]["trace"] if e["kind"] == "predicate"
              and e["relation"] == c["call"]["relation"] and match(e["object_ids"])]
        reached += bool(ev)
        if not ev or ev[0]["outcome"] != table[c["expect"]["value"]]:
            mismatches.append(f"case {cid}: {ev[0]['outcome'] if ev else 'not reached'} vs {c['expect']['value']}")
    check(f"all {len(disp['relation_cases'])} reviewed relation cases reached through queries give the table's value",
          not mismatches and reached == len(disp["relation_cases"]), "; ".join(mismatches[:6]))


def check_schemas_and_invariants():
    print("-- schemas, invariants and inputs")
    import jsonschema
    schemas = {n: load(REPO / "schemas" / f"{n}.v1.json") for n in ("grounding-query", "resolver-config",
                                                                     "grounding-result")}
    for name, schema in schemas.items():
        try:
            jsonschema.Draft202012Validator.check_schema(schema)
            ok = True
        except jsonschema.SchemaError as e:
            ok = False
            print("   ", e)
        check(f"schema {name}.v1.json is valid JSON Schema 2020-12", ok)
    scene_schema = load(REPO / "schemas/scene.v1.json")
    q = schemas["grounding-query"]
    check("the query schema reuses the contract's ID and label patterns and profile enum, unwidened",
          q["$defs"]["id"]["pattern"] == scene_schema["$defs"]["id"]["pattern"]
          and q["$defs"]["label"]["pattern"] == scene_schema["$defs"]["label"]["pattern"]
          and q["properties"]["evidence_profile"]["enum"] == scene_schema["properties"]["evidence_profile"]["enum"])
    validator = jsonschema.Draft202012Validator(schemas["grounding-result"])
    records, problems = [], []
    for case in DOC["cases"]:
        try:
            records.append((case["id"], resolve_case(case)))
        except Exception as e:  # report, don't crash
            problems.append(f"{case['id']}: raised {type(e).__name__}: {str(e)[:120]}")
    reason_of = {"resolved": "unique_target_consensus", "ambiguous": "nonunique_target_or_interpretation",
                 "no_match": "no_matching_reference", "insufficient_information": "required_information_unavailable",
                 "unsupported": "unsupported_interpretation"}
    for cid, rec in records:
        errs = list(validator.iter_errors(rec))
        if errs:
            problems.append(f"{cid}: schema {errs[0].message[:80]}")
        r, st = rec["result"], rec["processing_status"]
        if st == "completed" and not (r is not None and rec["issues"] == [] and rec["budget"] is None):
            problems.append(f"{cid}: completed envelope")
        if st == "invalid_input" and not (r is None and rec["issues"] and rec["budget"] is None):
            problems.append(f"{cid}: invalid_input envelope")
        if st == "budget_exceeded" and not (r is None and rec["issues"] == [] and rec["budget"] is not None):
            problems.append(f"{cid}: budget envelope")
        if r is not None:
            if r["reason_code"] != reason_of[r["status"]]:
                problems.append(f"{cid}: reason_code")
            if (r["status"] == "resolved") != (r["target_id"] is not None) or (r["status"] == "resolved" and not r["action"]):
                problems.append(f"{cid}: target/action invariant")
    check(f"all {len(records)} case records validate against grounding-result.v1.json and keep the envelope, "
          "target and reason-code invariants", not problems, "; ".join(problems[:4]))
    case = CASES["R08"]
    scene, command, query, maps, config, rel = build(case)
    before = [json.dumps(x, sort_keys=True) for x in (scene, command, query, maps, config)]
    from grounding.resolution import resolve
    resolve(scene, command, query, resolver_config=config, relation_config_path=rel, direction_config_path=DIR_CFG,
            category_maps=maps)
    check("no input record is mutated", before == [json.dumps(x, sort_keys=True) for x in (scene, command, query, maps,
                                                                                            config)])
    rec = resolve_case(CASES["R08"])
    ids = rec["diagnostics"]["identities"]
    check("identities: four 64-hex snapshot hashes and the two accepted configuration identities",
          all(re.fullmatch(r"[0-9a-f]{64}", ids[k]) for k in ("scene_sha256", "command_sha256", "query_sha256",
                                                              "resolver_config_sha256"))
          and ids["relation_config_identity"].startswith("relations.v1@")
          and ids["direction_config_identity"].startswith("directions.v1@"))
    check("timings are finite and nonnegative", all(isinstance(t, float) and 0 <= t < 60
                                                    for t in rec["diagnostics"]["timings_s"].values()))


def check_imports():
    print("-- imports and interfaces")
    probe = ("import sys, json; sys.path.insert(0, sys.argv[1])\n"
             "import importlib, grounding.resolution\n"
             "R = importlib.import_module('grounding.resolution.resolve')\n"
             "import grounding.relations.predicates as P, grounding.relations.directions as D\n"
             "bad = [m for m in sys.modules if m.startswith(('analysis', 'grounding.tests', 'grounding.serialization')) "
             "or m in ('predicates', 'directions', 'geometry')]\n"
             "print(json.dumps({'one_truth': D.P is P and R.P is P, 'bad': bad}))\n")
    out = subprocess.run([sys.executable, "-c", probe, str(REPO)], capture_output=True, text=True, cwd=str(REPO))
    rep = json.loads(out.stdout) if out.returncode == 0 else {"error": out.stderr[-300:]}
    check("a fresh interpreter imports grounding.resolution with one predicates.Truth and nothing from analysis/, "
          "tests or the serializer", rep.get("one_truth") is True and rep.get("bad") == [], str(rep))
    imports = [ln for f in (REPO / "grounding" / "resolution").glob("*.py")
               for ln in f.read_text(encoding="utf-8").splitlines() if re.match(r"\s*(from|import)\s", ln)]
    check("production code imports nothing from analysis/, tests or the serializer",
          not any(re.search(r"analysis|tests|serializ", ln) for ln in imports))
    import inspect
    from grounding.relations import predicates as P
    sig = list(inspect.signature(P.Relations.rank).parameters)
    check("the accepted rank interface is unchanged", sig == ["self", "relation", "anchor", "candidates", "possible",
                                                              "k"], str(sig))


def check_cli():
    print("-- command line")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        def files(case, folder, query_text=None):
            scene, command, query, maps, config, rel = build(case)
            folder.mkdir(parents=True, exist_ok=True)
            paths = {}
            for name, rec in (("scene", scene), ("command", command), ("query", query), ("config", config)):
                paths[name] = folder / f"{name}.json"
                paths[name].write_text(json.dumps(rec), encoding="utf-8")
            if query_text is not None:
                paths["query"].write_text(query_text, encoding="utf-8")
            for i, m in enumerate(maps):
                paths[f"map{i}"] = folder / f"map{i}.json"
                paths[f"map{i}"].write_text(json.dumps(m), encoding="utf-8")
            args = ["--scene", str(paths["scene"]), "--command", str(paths["command"]), "--query", str(paths["query"]),
                    "--resolver-config", str(paths["config"]), "--relation-config", str(rel), "--direction-config",
                    str(DIR_CFG)]
            if maps:
                args += ["--category-map"] + [str(paths[f"map{i}"]) for i in range(len(maps))]
            return args

        def cli(args, out):
            return subprocess.run([sys.executable, "-m", "grounding.resolution"] + args + ["--out", str(out)],
                                  capture_output=True, text=True, cwd=str(REPO))

        args = files(CASES["R08"], tmp / "r08")
        r = cli(args, tmp / "out_r08")
        doc = load(tmp / "out_r08" / "resolution.json") if (tmp / "out_r08" / "resolution.json").exists() else None
        api = resolve_case(CASES["R08"])
        text = (tmp / "out_r08" / "resolution.json").read_bytes() if doc else b""
        check("CLI R08: exit 0, one resolution.json ending in LF, semantically equal to the API's record",
              r.returncode == 0 and doc is not None and semantic(doc) == semantic(api) and text.endswith(b"\n"),
              r.stderr[-200:])
        again = cli(args, tmp / "out_r08")
        check("CLI refuses to overwrite resolution.json (exit 2) and leaves it as it was",
              again.returncode == 2 and (tmp / "out_r08" / "resolution.json").read_bytes() == text)
        amb = cli(files(CASES["R25"], tmp / "r25"), tmp / "out_r25")
        check("CLI completed but ambiguous (R25) still exits 0", amb.returncode == 0
              and load(tmp / "out_r25" / "resolution.json")["result"]["status"] == "ambiguous")
        bud = cli(files(CASES["B01b"], tmp / "b01b"), tmp / "out_b01b")
        check("CLI budget_exceeded exits 1 and writes the technical record", bud.returncode == 1
              and load(tmp / "out_b01b" / "resolution.json")["processing_status"] == "budget_exceeded")
        dup = cli(files(CASES["R01"], tmp / "dup", query_text='{"record_type": "a", "record_type": "b"}'),
                  tmp / "out_dup")
        rec = load(tmp / "out_dup" / "resolution.json") if (tmp / "out_dup" / "resolution.json").exists() else {}
        check("CLI strict parsing: a repeated key exits 2 with an invalid_input record naming E_PARSE_DUPLICATE_KEY",
              dup.returncode == 2 and rec.get("processing_status") == "invalid_input"
              and "E_PARSE_DUPLICATE_KEY" in [i["code"] for i in rec.get("issues", [])])
        graph = cli(files(CASES["V07"], tmp / "v07"), tmp / "out_v07")
        rec = load(tmp / "out_v07" / "resolution.json") if (tmp / "out_v07" / "resolution.json").exists() else {}
        check("CLI invalid query graph exits 2 with E_QUERY_GRAPH", graph.returncode == 2
              and "E_QUERY_GRAPH" in [i["code"] for i in rec.get("issues", [])])
        import importlib
        main = importlib.import_module("grounding.resolution.__main__")
        real, err = main.resolve, io.StringIO()

        def controlled_failure(*a, **k):
            raise RuntimeError("controlled test exception")

        main.resolve = controlled_failure  # test-only substitution; the CLI has no fault-injection option
        try:
            with contextlib.redirect_stderr(err):
                code = main.main(files(CASES["R08"], tmp / "boom") + ["--out", str(tmp / "out_boom")])
        finally:
            main.resolve = real
        check("CLI: an unexpected exception prints its traceback and exits 3 with no resolution.json",
              code == 3 and "Traceback" in err.getvalue() and not (tmp / "out_boom" / "resolution.json").exists(),
              f"exit {code}")


def check_after_first_run():
    """Hand-derived checks added after the first run, to pin brief rules no fixed case isolated. They are not part
    of the before-code record (notes/phases/A2.1e_resolution.md lists them)."""
    print("-- added after the first run: hand-derived pins of brief rules")
    from grounding.resolution import resolve
    rec = resolve_case(CASES["B02b"])
    check("binding cap 1 (R08): the earlier partial resolved outcome existed (1 terminal outcome, 14 checks), yet no "
          "result is returned", rec["result"] is None and rec["diagnostics"]["work"]["terminal_outcomes"] == 1
          and rec["diagnostics"]["work"]["candidate_checks"] == 14, str(rec["diagnostics"]["work"]))
    # R11's scene (obj_002 front unknown). Constraints run in canonical-JSON order, so the relation name decides:
    # "in_front_of" sorts before "left". By hand, user (-2,-2) facing +y: left score -(x+2) is -3 for obj_003 and -1
    # for obj_004, FALSE for both. Under obj_001 (front (0,1)) both boxes score 2 in front, TRUE, then the left clause
    # makes them FALSE. Under obj_002 in_front_of is UNKNOWN (missing front) and the later FALSE must still exclude
    # both: no_match, with 6 actual calls and 2 cache hits (the user-heading clause repeats). Stopping at an UNKNOWN
    # would leave both boxes uncertain under obj_002 and give insufficient_information.
    case = dict(CASES["R11"], id="X01", interpretations=[{"interpretation_id": "i0", "kind": "query",
                "action": "INSPECT", "root": "target", "nodes": [
                    {"node_id": "target", "category": "box", "colours_all": [], "rank": None, "constraints": [
                        {"relation": "in_front_of", "frame": "object_intrinsic", "anchors": ["table_ref"]},
                        {"relation": "left", "frame": "user_heading", "anchors": []}]},
                    {"node_id": "table_ref", "category": "table", "colours_all": [], "constraints": [], "rank": None}]}])
    rec = resolve_case(case)
    w = rec["diagnostics"]["work"]
    check("an UNKNOWN clause never short-circuits a later FALSE (R11 scene: intrinsic UNKNOWN first, then user-heading "
          "FALSE): no_match, 6 calls, 2 cache hits", rec["result"]["status"] == "no_match"
          and rec["result"]["candidates"]["uncertain_match_ids"] == [] and w["predicate_calls"] == 6
          and w["predicate_cache_hits"] == 2, f"{rec['result']['status']}, {w['predicate_calls']} calls, "
          f"{w['predicate_cache_hits']} hits")
    # Restricted a17 without the lamp. behind(user_heading) sorts before near; by hand, front axis (0.9976,0.0698):
    # obj_002 front score 1.523 and obj_003 2.720, so behind is FALSE for both and near (assumed sizes) is never read.
    case = dict(CASES["P02"], id="X02", interpretations=[{"interpretation_id": "i0", "kind": "query",
                "action": "INSPECT", "root": "target", "nodes": [
                    {"node_id": "table_ref", "category": "table", "colours_all": [], "constraints": [], "rank": None},
                    {"node_id": "target", "category": "box", "colours_all": [], "rank": None, "constraints": [
                        {"relation": "near", "frame": None, "anchors": ["table_ref"]},
                        {"relation": "behind", "frame": "user_heading", "anchors": []}]}]}])
    rec = resolve_case(case)
    r = rec["result"]
    check("a FALSE clause ends the conjunction before an unused clause's assumptions are read (no class_size_prior)",
          r["status"] == "no_match" and r["conditional"] is False and r["assumptions"] == []
          and rec["diagnostics"]["work"]["predicate_calls"] == 2, f"{r['status']}, {r['assumptions']}")
    for cid, bound, subsets in (("R16", 4, 4), ("R17", 2, 2), ("R18", 2, 0)):
        w = resolve_case(CASES[cid])["diagnostics"]["work"]
        check(f"{cid}: max_rank_preflight_subsets is the admitted call's exact bound {bound}; actual subsets {subsets}",
              w["max_rank_preflight_subsets"] == bound and w["rank_subsets"] == subsets, str(w))
    r = resolve_case(CASES["R11"])["result"]
    check("an intrinsic UNKNOWN never adds missing_pose (R11)", "missing_pose" not in r["reason_codes"]
          and "missing_semantic_front" in r["reason_codes"], str(r["reason_codes"]))
    scene, command, query, maps, config, rel = build(CASES["R01"])
    rec = resolve(command, scene, query, resolver_config=config, relation_config_path=rel,
                  direction_config_path=DIR_CFG, category_maps=maps)
    check("records in the wrong argument slots are invalid_input E_RECORD_ROLE", rec["processing_status"] ==
          "invalid_input" and "E_RECORD_ROLE" in [i["code"] for i in rec["issues"]])
    for name, kwargs in (("trace 'yes'", {"trace": "yes"}), ("category_maps not a list", {"category_maps": MAP}),
                         ("resolver_config not an object", {"resolver_config": [config]})):
        args = dict(resolver_config=config, relation_config_path=rel, direction_config_path=DIR_CFG,
                    category_maps=maps, trace=False)
        args.update(kwargs)
        rec = resolve(scene, command, query, **args)
        check(f"malformed option ({name}) is invalid_input E_RESOLVER_CONFIG", rec["processing_status"] ==
              "invalid_input" and [i["code"] for i in rec["issues"]] == ["E_RESOLVER_CONFIG"])
    near = {"interpretation_id": "i0", "kind": "query", "action": "INSPECT", "root": "t", "nodes": [
        {"node_id": "a", "category": "table", "colours_all": [], "constraints": [], "rank": None},
        {"node_id": "t", "category": "box", "colours_all": [], "rank": None,
         "constraints": [{"relation": "near", "frame": None, "anchors": ["a"]}]}]}
    first, _, q3, _, _, _ = build(dict(CASES["R01"], id="X03", interpretations=[near]))
    resolve(first, dict(command, scene_id=first["scene_id"]), dict(q3, command_id=command["command_id"]),
            resolver_config=config, relation_config_path=rel, direction_config_path=DIR_CFG, category_maps=maps)
    changed = copy.deepcopy(first)  # the same scene key, different geometry: the accepted cache refuses it
    changed["objects"][0]["geometry"]["center_m"]["value"] = [9.0, 9.0, 0.0]
    try:
        resolve(changed, dict(command, scene_id=first["scene_id"]), dict(q3, command_id=command["command_id"]),
                resolver_config=config, relation_config_path=rel, direction_config_path=DIR_CFG, category_maps=maps)
        raised = None
    except ValueError as e:
        raised = str(e)
    check("a geometry-cache conflict propagates from the API as an exception, never a semantic result",
          raised is not None and "reused for different geometry" in raised, str(raised))


def check_integer_representation():
    """The A2.1e integer-validation correction (D72), written and run against the delivered resolver before the fix.

    JSON Schema's integer type admits whole-valued floats such as 10000.0 and 1.0. The resolver must still reject them
    for its nine limits and every rank k, without conversion, rounding or coercion, before any configuration loader,
    evaluator or rank call. Inputs are the frozen fixtures and query R15, changing only the field under test; the
    ranked-anchor and later-interpretation queries below are hand-derived. cases.json and dispatch.json are untouched.
    """
    print("-- A2.1e correction (D72): integer representation of resolver limits and rank k")
    import jsonschema
    from grounding.relations import directions as D
    from grounding.relations import predicates as P
    from grounding.resolution import resolve
    results = jsonschema.Draft202012Validator(load(REPO / "schemas" / "grounding-result.v1.json"))
    r15 = CASES["R15"]

    def run(limits=None, edit=None, case=r15, config=None):
        scene, command, query, maps, built, rel = build(case)
        config = built if config is None else config
        config["limits"].update(limits or {})
        if edit:
            edit(query)
        return resolve(scene, command, query, resolver_config=config, relation_config_path=rel,
                       direction_config_path=DIR_CFG, category_maps=maps)

    def rejected(thunk, code, label, path):
        """A validated invalid_input envelope naming exactly this one representation issue."""
        try:
            rec = thunk()
        except Exception as e:  # a crash is this check's failure, never its expected result
            return False, f"raised {type(e).__name__}: {e}"
        schema = [e.message[:80] for e in results.iter_errors(rec)]
        issues = [(i["code"], i["label"], i["path"]) for i in rec["issues"]]
        message = rec["issues"][0]["message"] if rec["issues"] else ""
        ok = (not schema and rec["processing_status"] == "invalid_input" and rec["result"] is None
              and rec["budget"] is None and issues == [(code, label, path)] and "integer" in message
              and "float" in message)
        return ok, f"{rec['processing_status']}, issues {issues}" + (f"; schema: {schema[0]}" if schema else "")

    floats = []  # every float input below, replayed by the instrumentation check
    for name, value in BASE_CONFIG["limits"].items():
        def limit(name=name, value=value):
            return run(limits={name: float(value)})
        floats.append(limit)
        ok, detail = rejected(limit, "E_RESOLVER_CONFIG", "resolver_config", f"$.limits.{name}")
        crash = ", the reported crash" if name == "max_rank_subset_evaluations" else ""
        check(f"R15 with {name} = {float(value)!r}{crash}: invalid_input, E_RESOLVER_CONFIG at $.limits.{name}",
              ok, detail)

    def zero():
        return run(limits={"max_binding_attempts": 0.0})
    floats.append(zero)
    ok, detail = rejected(zero, "E_RESOLVER_CONFIG", "resolver_config", "$.limits.max_binding_attempts")
    check("R15 with max_binding_attempts = 0.0, zero as a float: invalid_input, E_RESOLVER_CONFIG at "
          "$.limits.max_binding_attempts", ok, detail)
    rec = run(limits={"max_binding_attempts": 0})
    check("control: the integer 0 is still a valid allowance (R15 needs one binding: budget_exceeded, limit 0, "
          "used 0, next 1)", rec["processing_status"] == "budget_exceeded" and rec["budget"] == {
              "name": "max_binding_attempts", "limit": 0, "used": 0, "next_required": 1}, str(rec["budget"]))
    text = json.dumps(BASE_CONFIG).replace('"max_trace_bytes": 2000000', '"max_trace_bytes": 2e6')
    parsed, parse_issues = contract.parse("resolver_config", text.encode("utf-8"))

    def scientific():
        return run(config=copy.deepcopy(parsed))
    floats.append(scientific)
    ok, detail = rejected(scientific, "E_RESOLVER_CONFIG", "resolver_config", "$.limits.max_trace_bytes")
    check("a limit written 2e6 decodes as a float and is rejected the same way (E_RESOLVER_CONFIG at "
          "$.limits.max_trace_bytes)", not parse_issues and type(parsed["limits"]["max_trace_bytes"]) is float
          and ok, detail)

    def set_k(i, j, k):
        def edit(query):
            query["interpretations"][i]["nodes"][j]["rank"]["k"] = k
        return edit

    def k_root():
        return run(edit=set_k(0, 1, 1.0))
    floats.append(k_root)
    ok, detail = rejected(k_root, "E_QUERY_SCHEMA", "query", "$.interpretations[0].nodes[1].rank.k")
    check("R15 with rank k = 1.0, the reported crash: invalid_input, E_QUERY_SCHEMA at "
          "$.interpretations[0].nodes[1].rank.k", ok, detail)
    # Hand-derived on R15's scene, with the rank on an anchor node: "anchor" binds the green table obj_001; the chair
    # closest to it is obj_005 (2 m; obj_006 and obj_007 are 4 and 6 m); the root is the red box left of that chair,
    # user_to_anchor. User (-2,-2), anchor (2,0): v=(4,2)/sqrt20, right=(0.4472,-0.8944); obj_003's offset (-1,2)
    # scores right -2.236, so left is TRUE; obj_004 is blue. With k=1 the query resolves obj_003.
    anchor_case = dict(r15, interpretations=[{"interpretation_id": "i0", "kind": "query", "action": "INSPECT",
                                              "root": "target", "nodes": [
        {"node_id": "target", "category": "box", "colours_all": ["red"], "rank": None,
         "constraints": [{"relation": "left", "frame": "user_to_anchor", "anchors": ["chair_ref"]}]},
        {"node_id": "anchor", "category": "table", "colours_all": ["green"], "constraints": [], "rank": None},
        {"node_id": "chair_ref", "category": "chair", "colours_all": [], "constraints": [],
         "rank": {"relation": "closest", "anchor": "anchor", "k": 1}}]}])

    def k_anchor():
        return run(edit=set_k(0, 2, 1.0), case=anchor_case)
    floats.append(k_anchor)
    ok, detail = rejected(k_anchor, "E_QUERY_SCHEMA", "query", "$.interpretations[0].nodes[2].rank.k")
    check("rank k = 1.0 on an anchor node, not the root: E_QUERY_SCHEMA at $.interpretations[0].nodes[2].rank.k",
          ok, detail)
    rec = run(case=anchor_case)
    r = rec["result"] or {}
    check("control: that ranked-anchor query with k = 1 resolves obj_003 (hand-derived above)",
          r.get("status") == "resolved" and r.get("target_id") == "obj_003", f"{rec['processing_status']}, "
          f"{r.get('status')} {r.get('target_id')}")
    # R15's reading, interpretation i0, placed second: first by ID once normalized, so a path taken from the sorted
    # order would say interpretations[0]. The path must name the original input, interpretations[1].
    later_case = dict(r15, interpretations=[{"interpretation_id": "i1", "kind": "query", "action": "INSPECT",
                                             "root": "target", "nodes": [
        {"node_id": "target", "category": "box", "colours_all": ["red"], "constraints": [], "rank": None}]},
        copy.deepcopy(r15["interpretations"][0])])

    def k_later():
        return run(edit=set_k(1, 1, 1.0), case=later_case)
    floats.append(k_later)
    ok, detail = rejected(k_later, "E_QUERY_SCHEMA", "query", "$.interpretations[1].nodes[1].rank.k")
    check("rank k = 1.0 in a QUERY that is not the first interpretation: E_QUERY_SCHEMA at the original input path "
          "$.interpretations[1].nodes[1].rank.k", ok, detail)
    rec = run(case=later_case)
    r = rec["result"] or {}
    check("control: with k = 1 that request completes (red box obj_003 against R15's obj_006: ambiguous)",
          r.get("status") == "ambiguous" and r.get("candidates", {}).get("target_ids") == ["obj_003", "obj_006"],
          f"{rec['processing_status']}, {r.get('status')}")

    saved = (P.load_config, D.load_direction_config, P.Relations.__init__, D.DirectionalRelations.__init__,
             P.Relations.rank)

    def refuse(*a, **k):
        raise AssertionError("reached after an invalid numeric representation")

    P.load_config = D.load_direction_config = refuse
    P.Relations.__init__ = D.DirectionalRelations.__init__ = P.Relations.rank = refuse
    statuses = []
    try:
        for thunk in floats:
            try:
                statuses.append(thunk()["processing_status"])
            except AssertionError:
                statuses.append("a loader, evaluator or rank was called")
            except Exception as e:  # noqa: BLE001 - reported, never expected
                statuses.append(f"raised {type(e).__name__}")
    finally:
        (P.load_config, D.load_direction_config, P.Relations.__init__, D.DirectionalRelations.__init__,
         P.Relations.rank) = saved
    check(f"none of the {len(floats)} float inputs reaches a configuration loader, evaluator construction or rank "
          "(all made to raise)", statuses == ["invalid_input"] * len(floats), str(sorted(set(statuses))))

    found = [(value, run(limits={"max_interpretations": value})) for value in (1.5, "8", None)]
    check("fractional, string and null limits stay invalid_input E_RESOLVER_CONFIG (Boolean and negative: V39, V40)",
          all(rec["processing_status"] == "invalid_input" and {i["code"] for i in rec["issues"]} == {
              "E_RESOLVER_CONFIG"} for _, rec in found), str([(v, rec["processing_status"]) for v, rec in found]))
    found = [(value, run(edit=set_k(0, 1, value))) for value in (1.5, "1", None, 0, 4)]
    check("fractional, string, null and out-of-range k (1.5, '1', null, 0, 4) stay invalid_input E_QUERY_SCHEMA "
          "(Boolean: V15)", all(rec["processing_status"] == "invalid_input" and {i["code"] for i in rec["issues"]}
                                == {"E_QUERY_SCHEMA"} for _, rec in found),
          str([(v, rec["processing_status"]) for v, rec in found]))

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        scene, command, query, maps, config, rel = build(r15)
        bad_config, bad_query = copy.deepcopy(config), copy.deepcopy(query)
        bad_config["limits"]["max_rank_subset_evaluations"] = 10000.0
        bad_query["interpretations"][0]["nodes"][1]["rank"]["k"] = 1.0
        paths = {}
        for name, rec in (("scene", scene), ("command", command), ("query", query), ("config", config),
                          ("map", maps[0]), ("config_float", bad_config), ("query_float", bad_query)):
            paths[name] = tmp / f"{name}.json"
            paths[name].write_text(json.dumps(rec), encoding="utf-8")

        def cli(query_path, config_path, out):
            return subprocess.run([sys.executable, "-m", "grounding.resolution", "--scene", str(paths["scene"]),
                                   "--command", str(paths["command"]), "--query", str(query_path),
                                   "--resolver-config", str(config_path), "--relation-config", str(rel),
                                   "--direction-config", str(DIR_CFG), "--category-map", str(paths["map"]),
                                   "--out", str(out)], capture_output=True, text=True, cwd=str(REPO))

        for label, query_path, config_path, holder, literal, code, path in (
                ("max_rank_subset_evaluations 10000.0", paths["query"], paths["config_float"], paths["config_float"],
                 '"max_rank_subset_evaluations": 10000.0', "E_RESOLVER_CONFIG",
                 "$.limits.max_rank_subset_evaluations"),
                ("rank k 1.0", paths["query_float"], paths["config"], paths["query_float"], '"k": 1.0',
                 "E_QUERY_SCHEMA", "$.interpretations[0].nodes[1].rank.k")):
            out = tmp / ("out_" + label.split()[0])
            r = cli(query_path, config_path, out)
            doc = load(out / "resolution.json") if (out / "resolution.json").exists() else {}
            check(f"CLI, a real file holding {label}: exit 2, the normal invalid_input resolution.json "
                  f"({code} at {path}), result null, no traceback",
                  literal in holder.read_text(encoding="utf-8") and r.returncode == 2
                  and doc.get("processing_status") == "invalid_input" and doc.get("result", "absent") is None
                  and [(i["code"], i["path"]) for i in doc.get("issues", [])] == [(code, path)]
                  and "Traceback" not in r.stderr, f"exit {r.returncode}; {r.stderr.strip()[-160:]}")
        out = tmp / "out_max_rank_subset_evaluations"
        before = (out / "resolution.json").read_bytes() if (out / "resolution.json").exists() else None
        again = cli(paths["query"], paths["config_float"], out)
        check("CLI keeps its no-overwrite rule: the same run again exits 2 and leaves resolution.json as it was",
              before is not None and again.returncode == 2 and "refusing" in again.stderr
              and (out / "resolution.json").read_bytes() == before, f"exit {again.returncode}")
        ctl = cli(paths["query"], paths["config"], tmp / "out_control")
        doc = load(tmp / "out_control" / "resolution.json") if (tmp / "out_control" / "resolution.json").exists() \
            else {}
        res = doc.get("result") or {}
        check("CLI control: the valid R15 files still resolve obj_006 (exit 0)", ctl.returncode == 0
              and res.get("status") == "resolved" and res.get("target_id") == "obj_006", f"exit {ctl.returncode}")


def main() -> int:
    check_fixtures()
    try:
        importlib.import_module("grounding.resolution")
    except ImportError as e:
        check("grounding.resolution imports", False, str(e))
        print(f"{COUNT[0]} checks; FAILED: {', '.join(FAILED)}")
        return 1
    check_cases(("R",), "12.2 hand-established cases")
    check_cases(("V",), "12.3 validation, maps and configuration")
    check_cases(("B",), "12.3 work limits")
    check_budget_instrumentation()
    check_cases(("M02", "M04", "M05b"), "12.3 caching, aggregation and pose")
    check_metamorphic()
    check_trace()
    check_cases(("P",), "12.3 provenance")
    check_dispatch()
    check_schemas_and_invariants()
    check_imports()
    check_cli()
    check_after_first_run()
    check_integer_representation()
    print(f"{COUNT[0]} checks; {'FAILED: ' + ', '.join(FAILED) if FAILED else 'all checks passed'}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
