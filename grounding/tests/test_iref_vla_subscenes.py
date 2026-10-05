"""Tests of the A2.2c category-complete selection audit (D77).

    python grounding/tests/test_iref_vla_subscenes.py --sample DIR    full acceptance: fixtures, then the pinned sample
    python grounding/tests/test_iref_vla_subscenes.py --unit-only     fixtures only; NOT sample acceptance
    (or: python -m grounding.tests.test_iref_vla_subscenes ...)

DIR holds A2.2a's five pinned IRef-VLA files; SECOND_EYES_IREF_VLA_SAMPLE may name it instead. The sample section
re-imports them through A2.2a's pin-verified adapter and audits the import once: parsing and counting only. Without the
files and without --unit-only, the sample check is one FAIL, never a skip. Expectations come from
fixtures/iref_vla_subscenes/expectations.json, written by hand before the selector existed. Prints one line per check
and exits 1 if any fails.
"""
from __future__ import annotations

import contextlib
import copy
import importlib
import inspect
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

FX = REPO / "grounding" / "tests" / "fixtures" / "iref_vla_subscenes"
EV = REPO / "grounding" / "tests" / "fixtures" / "iref_vla_evaluation"
EXP = json.loads((FX / "expectations.json").read_text(encoding="utf-8"))
REL_CFG = REPO / "grounding" / "relations" / "relations.v1.json"
DIR_CFG = REPO / "grounding" / "relations" / "directions.v1.json"
VIEWS = ("full_inventory", "source_known_nyu")
FAILED, COUNT = [], [0]


def check(name, ok, detail=""):
    COUNT[0] += 1
    print(("PASS  " if ok else "FAIL  ") + name + (f": {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def ev(name):
    return json.loads((EV / name).read_text(encoding="utf-8"))


def commands_for(texts, scene_id="fixture.iref_eval.f1"):
    template = ev("commands.f1.json")[0]
    return [dict(copy.deepcopy(template), command_id=f"fixture.iref_sub.c{i:02d}", text=t, scene_id=scene_id)
            for i, t in enumerate(texts, 1)]


def outcome(E, fn):
    try:
        return ("ok", fn())
    except E.EvaluationInputError as e:
        return ("rejected", sorted({i["code"] for i in e.issues}))
    except Exception as e:  # noqa: BLE001 - reported as the check's failure
        return ("crashed", f"{type(e).__name__}: {e}"[:140])


def write_inputs(folder, scene, commands, cmap, views, *, reverse=False):
    """The four permitted inputs in A2.2a's layout; returns the CLI paths."""
    mi, ro = folder / "model_inputs", folder / "reference_only"
    (mi / "commands").mkdir(parents=True)
    ro.mkdir()
    (mi / "scene.annotated.json").write_text(json.dumps(scene), encoding="utf-8")
    (mi / "category-map.json").write_text(json.dumps(cmap), encoding="utf-8")
    (ro / "inventory-views.json").write_text(json.dumps(views), encoding="utf-8")
    for c in (reversed(commands) if reverse else commands):
        (mi / "commands" / f"{c['command_id']}.json").write_text(json.dumps(c), encoding="utf-8")
    return {"scene": mi / "scene.annotated.json", "commands": mi / "commands", "category_map": mi / "category-map.json",
            "inventory_views": ro / "inventory-views.json"}


# ---------------------------------------------------------------------------------------------- the selector
def check_selector(S):
    print("-- the core selector: hand-worked cases (section 7.1)")
    proj = [S.ProjectedObject(i, c) for i, c in EXP["projection"].items()]
    for case in EXP["selector_cases"]:
        for v in VIEWS:
            got = list(S.select(tuple(p for p in proj if p.object_id in EXP["views"][v]), frozenset(case["categories"])))
            check(f"{case['id']}, {v}", got == case[v], str(got))
    for case in EXP["cap_cases"]:
        objs = [S.ProjectedObject(f"obj_{i:03d}", "chair") for i in range(1, case["chairs"] + 1)]
        objs += [S.ProjectedObject(f"obj_{100 + i:03d}", "table") for i in range(case["tables"])]
        unknown = [S.ProjectedObject(f"obj_{200 + i:03d}", None) for i in range(case["unknown"])]
        got = {}
        for v, members in (("full_inventory", objs + unknown), ("source_known_nyu", objs)):
            kept = S.select(tuple(members), frozenset({"chair", "table"}))
            got[v] = [len(kept), S.primary_status(len(kept))]
        check(f"{case['id']}: the whole required set is kept, its status set by the count",
              got == {v: case[v] for v in VIEWS}, str(got))
    oc = EXP["ordering_case"]
    got = list(S.select(tuple(S.ProjectedObject(i, "chair") for i in oc["ids"]), frozenset({"chair"})))
    check("obj_999 and obj_1000 follow explicit lexicographic order, no suffix arithmetic", got == oc["expected"], str(got))
    for name, args in (("a full scene", (ev("scene.f1.json"), frozenset({"chair"}))),
                       ("a list of categories", (tuple(proj), ["chair"])),
                       ("raw dictionaries", (({"object_id": "obj_001", "model_category": "chair"},), frozenset({"chair"})))):
        try:
            S.select(*args)
            got = "accepted"
        except TypeError:
            got = "TypeError"
        check(f"the selector refuses {name}: it takes only a validated projection and a category set", got == "TypeError")
    params = list(inspect.signature(S.select).parameters)
    check("the selector's only parameters are the projection and the category set", params == ["objects", "categories"],
          str(params))


# ---------------------------------------------------------------------------------------------- the pipeline
def check_pipeline(S, E):
    print("-- the audit on fixture F1, through the accepted parser")
    cmap, views, scene = ev("category-map.fixture.json"), ev("views.f1.json"), ev("scene.f1.json")
    cases = EXP["pipeline_cases"]
    res = S.audit(scene, commands_for([c["text"] for c in cases]), cmap, views)
    rows = {(r["parent_command_id"], r["view_id"]): r for r in res.rows}
    nulls = ("required_categories", "retained_object_ids", "retained_unknown_category_ids", "required_object_count",
             "selection_id")
    for n, case in enumerate(cases, 1):
        cid, e = f"fixture.iref_sub.c{n:02d}", case["expected"]
        rs = [rows.get((cid, v)) for v in VIEWS]
        if None in rs:
            check(f"{case['id']}", False, "missing row")
            continue
        if "unassessed" in e:
            ok = all(r["status"] == "unassessed_parse" and r["parse_status"] == "unsupported"
                     and r["parse_reason"] == e["unassessed"] and all(r[k] is None for k in nulls)
                     and r["base_object_count"] == len(EXP["views"][v]) for r, v in zip(rs, VIEWS))
            check(f"{case['id']} {case['text']!r}: unassessed ({e['unassessed']}), every selection field null", ok,
                  json.dumps(rs[0])[:160])
        else:
            ok = all(r["required_categories"] == e["required_categories"] and r["retained_object_ids"] == e[v]["retained"]
                     and r["retained_unknown_category_ids"] == e[v]["unknown"] and r["required_object_count"] == e[v]["count"]
                     and r["status"] == e[v]["status"] and r["selection_id"].startswith("iref.subset.")
                     and r["primary_object_limit"] == 10 and r["base_object_count"] == len(EXP["views"][v])
                     for r, v in zip(rs, VIEWS))
            check(f"{case['id']} {case['text']!r}: {e['required_categories']}", ok,
                  json.dumps({v: r["retained_object_ids"] for r, v in zip(rs, VIEWS)}))
    ids = {v: {rows[(f"fixture.iref_sub.c{[c['id'] for c in cases].index(t) + 1:02d}", v)]["selection_id"]
               for t in EXP["same_selection_ids"]} for v in VIEWS}
    check("closest, farthest, k = 3 and a colour modifier share one membership and one selection ID per view",
          all(len(s) == 1 for s in ids.values()), str({v: len(s) for v, s in ids.items()}))
    table, known_cats = EXP["independent_category_table"], set(EXP["independent_category_table"]) - {"unknown"}
    bad = []
    for r in res.rows:
        if r["status"] == "unassessed_parse":
            continue
        base = set(EXP["views"][r["view_id"]])
        want = set(table["unknown"]) | {i for c in r["required_categories"] if c in known_cats for i in table[c]}
        want &= base
        excluded = base - set(r["retained_object_ids"])
        outside = all(any(x in table[c] for c in known_cats - set(r["required_categories"])) for x in excluded)
        if set(r["retained_object_ids"]) != want or not outside:
            bad.append((r["parent_command_id"], r["view_id"]))
    check("category completeness, from the independent table: every unknown or requested object kept, every excluded "
          "object of a known unrequested category", not bad, str(bad[:3]))
    mg = EXP["missing_geometry_case"]
    s2 = copy.deepcopy(scene)
    o3 = next(o for o in s2["objects"] if o["object_id"] == mg["object"])
    for field in ("center_m", "rotation_xyzw"):
        o3["geometry"][field] = {"state": "unknown", "reason": "not_in_source", "source": "fx"}
    s2["scene_id"] = "fixture.iref_sub.f1_no_geometry"
    v2 = dict(copy.deepcopy(views), parent_scene_id=s2["scene_id"])
    r2 = outcome(E, lambda: S.audit(s2, commands_for([mg["text"]], s2["scene_id"]), cmap, v2))
    got = {r["view_id"]: r["retained_object_ids"] for r in r2[1].rows} if r2[0] == "ok" else r2
    check("a chair with unknown centre and rotation stays in a chair selection",
          got == {v: mg["expected"][v]["retained"] for v in VIEWS}, str(got))


# ---------------------------------------------------------------------------------------------- input failures
def check_inputs(S, E):
    print("-- input failures, through the accepted validation")
    cmap, views, scene = ev("category-map.fixture.json"), ev("views.f1.json"), ev("scene.f1.json")
    cmds = commands_for(["the chair that is closest to the table", "the box that is on the table"])

    def swap(v):
        v = copy.deepcopy(v)
        v["views"][1]["included_object_ids"][0] = "obj_008"
        v["views"][1]["excluded"][0].update(object_id="obj_001", source_object_id="1")
        return v
    other = dict(copy.deepcopy(cmds[1]), scene_id="fixture.iref_eval.f0")
    cases = {"a null scene record": ((None, cmds, cmap, views), "E_EVAL_INPUT"),
             "duplicate command IDs": ((scene, cmds + [copy.deepcopy(cmds[0])], cmap, views), "E_EVAL_INPUT"),
             "a category map that isn't the scene's": ((scene, cmds, dict(copy.deepcopy(cmap), map_id="fixture.other"), views),
                                                     "E_EVAL_INPUT"),
             "a command for another scene": ((scene, [cmds[0], other], cmap, views), "E_EVAL_INPUT"),
             "an inventory membership swap": ((scene, cmds, cmap, swap(views)), "E_EVAL_VIEWS"),
             "an inventory format_version of 1.0": ((scene, cmds, cmap, dict(copy.deepcopy(views), format_version=1.0)),
                                                    "E_EVAL_VIEWS"),
             "the map given in the scene's role": ((cmap, cmds, cmap, views), "E_EVAL_INPUT")}
    for name, (args, code) in cases.items():
        got = outcome(E, lambda: S.audit(*args))
        check(f"{name}: rejected with {code}", got[0] == "rejected" and code in got[1], str(got)[:140])
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        p = write_inputs(tmp / "in", scene, cmds, cmap, views)
        cli = [sys.executable, "-m", "grounding.subscenes.iref_vla"]
        args = lambda p, out: ["--scene", str(p["scene"]), "--commands", str(p["commands"]), "--category-map",  # noqa: E731
                               str(p["category_map"]), "--inventory-views", str(p["inventory_views"]), "--out", str(out)]
        bad_json = tmp / "bad.json"
        bad_json.write_bytes(b"{\"scene\": ")
        (p["commands"] / "notes.txt").write_text("stray", encoding="utf-8")
        for name, override in (("malformed JSON", {"scene": bad_json}), ("a stray file in the commands folder", {})):
            r = subprocess.run(cli + args(dict(p, **override), tmp / f"out-{len(name)}"), capture_output=True, text=True,
                               cwd=str(REPO))
            check(f"the CLI with {name}: exit 2, E_EVAL_INPUT, nothing published",
                  r.returncode == 2 and "E_EVAL_INPUT" in r.stdout and not (tmp / f"out-{len(name)}").exists(),
                  f"exit {r.returncode}")


# ---------------------------------------------------------------------------------------------- outputs
def check_outputs(S, E):
    print("-- outputs: four closed artifacts, consistent rows, summary arithmetic")
    import jsonschema
    schema = json.loads((REPO / "schemas" / "iref-subscene-audit.v1.json").read_text(encoding="utf-8"))
    row_schema = jsonschema.Draft202012Validator({"$schema": schema["$schema"], "$defs": schema["$defs"],
                                                  "$ref": "#/$defs/selection"})
    cmap, views, scene = ev("category-map.fixture.json"), ev("views.f1.json"), ev("scene.f1.json")
    cmds = commands_for([c["text"] for c in EXP["pipeline_cases"]])
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        p = write_inputs(tmp / "in", scene, cmds, cmap, views)
        S.run_audit(**p, out=tmp / "out", sample_only=False)
        names = sorted(f.name for f in (tmp / "out").iterdir())
        check("exactly four artifacts", names == ["manifest.json", "report.md", "selections.jsonl", "summary.json"], str(names))
        raw = (tmp / "out" / "selections.jsonl").read_bytes()
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
        check("selections.jsonl: compact sorted-key lines, LF only", raw.endswith(b"\n") and b"\r" not in raw and all(
            line == json.dumps(json.loads(line), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            for line in raw.decode("utf-8").splitlines()))
        check("every row validates against the closed selection definition",
              all(not list(row_schema.iter_errors(r)) for r in rows))
        extra = dict(rows[0], note="extra")
        check("an extra field is rejected by the closed definition", bool(list(row_schema.iter_errors(extra))))
        pairs = [(r["parent_command_id"], r["view_id"]) for r in rows]
        want = [(c["command_id"], v) for c in sorted(cmds, key=lambda c: c["command_id"]) for v in VIEWS]
        check("rows: one per command and view, ordered by command ID then the fixed view order", pairs == want)
        consistent = all(r["status"] == "unassessed_parse" or (
            r["required_object_count"] == len(r["retained_object_ids"])
            and r["status"] == ("fits" if r["required_object_count"] <= 10 else "over_budget")
            and r["retained_object_ids"] == sorted(r["retained_object_ids"])
            and set(r["retained_unknown_category_ids"]) == {i for i in r["retained_object_ids"]
                                                            if EXP["projection"].get(i, "x") is None}
            and set(r["retained_object_ids"]) <= set(EXP["views"][r["view_id"]])) for r in rows)
        check("counts, statuses, order, unknown subsets and base membership agree in every row", consistent)
        s = json.loads((tmp / "out" / "summary.json").read_text(encoding="utf-8"))
        arith = []
        for v in VIEWS:
            vr = [r for r in rows if r["view_id"] == v]
            assessed = [r for r in vr if r["status"] != "unassessed_parse"]
            x = s["views"][v]
            hist = {}
            for r in assessed:
                hist[str(r["required_object_count"])] = hist.get(str(r["required_object_count"]), 0) + 1
            arith.append(x["N"] == len(vr) and x["P"] == len(assessed) and x["N"] == x["P"] + x["U"]
                         and x["fits"] + x["over_budget"] == x["P"] and x["required_count_histogram"] == hist
                         and x["fit_coverage_all"]["denominator"] == x["N"]
                         and x["fit_coverage_parsed"]["denominator"] == x["P"]
                         and x["distinct_assessed_selections"] == len({r["selection_id"] for r in assessed}))
        check("summary arithmetic, recomputed from the rows in this test", all(arith) and s == S.summarize(rows))
        m = json.loads((tmp / "out" / "manifest.json").read_text(encoding="utf-8"))
        import hashlib
        check("the manifest records the hashes of the three other artifacts", all(
            m["outputs"][n] == hashlib.sha256((tmp / "out" / n).read_bytes()).hexdigest()
            for n in ("selections.jsonl", "summary.json", "report.md")))
        report = (tmp / "out" / "report.md").read_text(encoding="utf-8")
        check("the report carries the five required statements", all(st in report for st in EXP["report_statements"]))
        check("no timestamp or output path in rows or summary", str(tmp) not in raw.decode("utf-8")
              and str(tmp) not in json.dumps(s))


# ---------------------------------------------------------------------------------------------- determinism
def check_determinism(S):
    print("-- determinism and what membership ignores")
    cmap, views, scene = ev("category-map.fixture.json"), ev("views.f1.json"), ev("scene.f1.json")
    cmds = commands_for([c["text"] for c in EXP["pipeline_cases"]])
    base = S.audit(scene, cmds, cmap, views)
    shuffled = copy.deepcopy(scene)
    shuffled["objects"].reverse()
    perm = S.audit(shuffled, list(reversed(cmds)), cmap, views)
    check("reversed object and command order: identical rows and summary",
          perm.rows == base.rows and perm.summary == base.summary)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, rev in (("a", False), ("b", True)):
            S.run_audit(**write_inputs(tmp / f"in-{name}", scene, cmds, cmap, views, reverse=rev), out=tmp / name,
                        sample_only=False)
        same = all((tmp / "a" / f).read_bytes() == (tmp / "b" / f).read_bytes()
                   for f in ("selections.jsonl", "summary.json", "report.md"))
        check("command files written in another order: byte-identical rows, summary and report", same)
    changed = copy.deepcopy(scene)
    for o in changed["objects"]:
        o["geometry"]["center_m"]["value"] = [x + 1.5 for x in o["geometry"]["center_m"]["value"]]
        o["attributes"]["colours"]["value"] = ["purple"]
    changed["scene_id"] = "fixture.iref_sub.f1_moved"
    moved = S.audit(changed, commands_for([c["text"] for c in EXP["pipeline_cases"]], changed["scene_id"]), cmap,
                    dict(copy.deepcopy(views), parent_scene_id=changed["scene_id"]))
    member = lambda rs: [(r["parent_command_id"], r["view_id"], r["retained_object_ids"]) for r in rs]  # noqa: E731
    check("other geometry and colours: identical membership; selection IDs may change with the parent scene",
          member(moved.rows) == member(base.rows) and moved.parent_scene_sha256 != base.parent_scene_sha256)


# ---------------------------------------------------------------------------------------------- no answer access
def check_no_answers(S):
    print("-- no answer access, and no relation, resolver or serializer execution")
    cmap, views, scene = ev("category-map.fixture.json"), ev("views.f1.json"), ev("scene.f1.json")
    cmds = commands_for([c["text"] for c in EXP["pipeline_cases"]])
    forbidden_names = ("iref-annotations", "scene_graph", "referential_statements", "predictions.jsonl", "scores.jsonl",
                       "prediction-summary", "import-report", "source-manifest")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        clean = write_inputs(tmp / "clean", scene, cmds, cmap, views)
        dirty = write_inputs(tmp / "dirty", scene, cmds, cmap, views)
        for n in ("iref-annotations.json", "scene0010_01_scene_graph.json", "import-report.json"):
            (dirty["inventory_views"].parent / n).write_bytes(b"\x00 not json")
        (dirty["scene"].parent.parent / "predictions.jsonl").write_bytes(b"garbage")
        opened = []
        real_open = io.open

        def watching_open(file, *a, **k):
            opened.append(str(file))
            return real_open(file, *a, **k)
        mods = {"grounding.resolution": "resolve", "grounding.resolution.resolve": "resolve",
                "grounding.evaluation.iref_vla.predict": "resolve", "grounding.serialization": "serialize",
                "grounding.relations.predicates": "load_config", "grounding.relations.directions": "load_direction_config"}
        saved = {(m, a): getattr(importlib.import_module(m), a) for m, a in mods.items()}
        calls = []

        def sentinel(*a, **k):
            calls.append("called")
            raise AssertionError("relation, resolver or serializer work during the audit")
        try:
            io.open = watching_open
            import builtins
            builtins_open = builtins.open
            builtins.open = watching_open
            for (m, a) in saved:
                setattr(importlib.import_module(m), a, sentinel)
            S.run_audit(**dirty, out=tmp / "dirty-out", sample_only=False)
        finally:
            io.open = real_open
            builtins.open = builtins_open
            for (m, a), f in saved.items():
                setattr(importlib.import_module(m), a, f)
        bad = [p for p in opened if any(n in p for n in forbidden_names)]
        inputs = {str(v) for k, v in dirty.items() if k != "commands"}
        runtime_roots = tuple({sys.prefix, sys.base_prefix, sys.exec_prefix})  # the interpreter and package metadata
        stray = [p for p in opened if not (p in inputs or p.startswith(str(dirty["commands"])) or p.startswith(str(REPO))
                                           or ".partial-" in p or p.startswith(runtime_roots) or "site-packages" in p
                                           or "dist-packages" in p)]
        check("the audit opens no annotation, graph, statement, prediction or scoring file", not bad, str(bad[:3]))
        check("it reads only the four permitted inputs and tracked repository files", not stray, str(stray[:3]))
        check("no relation, resolver or serializer function runs during the audit", not calls, f"{len(calls)} calls")
        S.run_audit(**clean, out=tmp / "clean-out", sample_only=False)
        same = all((tmp / "clean-out" / f).read_bytes() == (tmp / "dirty-out" / f).read_bytes()
                   for f in ("selections.jsonl", "summary.json", "report.md"))
        check("unusable annotation and prediction siblings change no row, summary or report", same)
    params = set(inspect.signature(S.audit).parameters) | set(inspect.signature(S.run_audit).parameters)
    check("no production parameter accepts annotations, statements, graphs, targets, predictions or scores",
          not any(w in p for p in params for w in ("annot", "statement", "graph", "target", "predict", "score")),
          str(sorted(params)))


# ---------------------------------------------------------------------------------------------- resolver check
def check_resolver(S):
    print("-- resolver integration on F0/F1 only: the full base view against its selected view")
    from grounding.evaluation.iref_vla.parse import parse
    from grounding.evaluation.iref_vla.predict import resolver_config
    from grounding.evaluation.iref_vla.protocol import load_protocol
    from grounding.resolution import resolve
    protocol, cmap = load_protocol(), ev("category-map.fixture.json")
    config = resolver_config(protocol, cmap)
    cases = [("f0", "full_inventory", "the chair that is closest to the table"),
             ("f0", "full_inventory", "the chair that is near the table"),
             ("f0", "full_inventory", "the box that is on the table"),
             ("f0", "full_inventory", "the chair that is between the table and the black chair"),
             ("f1", "full_inventory", "the chair that is closest to the table"),
             ("f1", "full_inventory", "the chair that is farthest from the black box"),
             ("f1", "source_known_nyu", "the chair that is farthest from the black box"),
             ("f0", "full_inventory", "the crib that is near the table")]
    for n, (key, view, text) in enumerate(cases):
        scene, views = ev(f"scene.{key}.json"), ev(f"views.{key}.json")
        audit = S.audit(scene, commands_for([text], scene["scene_id"]), cmap, views)
        row = next(r for r in audit.rows if r["view_id"] == view)
        members = next(v["included_object_ids"] for v in views["views"] if v["view_id"] == view)
        outcomes = []
        for tag, keep in (("base", set(members)), ("sel", set(row["retained_object_ids"]))):
            s = copy.deepcopy(scene)
            s["objects"] = [o for o in s["objects"] if o["object_id"] in keep]
            s["scene_id"] = f"fixture.iref_sub.it{n}.{tag}.{len(keep)}"
            c = dict(commands_for([text], s["scene_id"])[0], command_id=f"fixture.iref_sub.it{n}.{tag}")
            q = {"schema_version": 1, "record_type": "grounding_query", "query_id": c["command_id"] + ".q",
                 "scene_id": s["scene_id"], "scene_revision": 0, "evidence_profile": "annotated", "command_id": c["command_id"],
                 "interpretations": [parse(text, categories=cmap["model_vocabulary"],
                                           colours=protocol["colour_labels"])["interpretation"]]}
            rec = resolve(s, c, q, resolver_config=config, relation_config_path=REL_CFG, direction_config_path=DIR_CFG,
                          category_maps=[cmap])
            r = rec["result"] or {}
            outcomes.append((rec["processing_status"], r.get("status"), r.get("target_id"), r.get("action"),
                             sorted((r.get("candidates") or {}).get("target_ids", [])),
                             sorted((r.get("candidates") or {}).get("uncertain_match_ids", []))))
        a, b = outcomes
        check(f"{key} {view} {text!r}: base and selected views agree ({a[1]} {a[2] or ''})".rstrip(),
              a[0] == b[0] == "completed" and a == b and len(row["retained_object_ids"]) < len(members), f"{a} vs {b}")


# ---------------------------------------------------------------------------------------------- the CLI
def check_cli(S, E):
    print("-- the command line: success, refusal and failure exits")
    am = importlib.import_module("grounding.subscenes.iref_vla.audit")
    mm = importlib.import_module("grounding.subscenes.iref_vla.__main__")
    om = importlib.import_module("grounding.evaluation.iref_vla.output")
    cmap, views, scene = ev("category-map.fixture.json"), ev("views.f1.json"), ev("scene.f1.json")
    cmds = commands_for([c["text"] for c in EXP["pipeline_cases"]])
    real_protocol = am.load_protocol

    def fixture_protocol(*a, **k):  # test-only: lets the sample-limited CLI accept fixture F1
        p = copy.deepcopy(real_protocol(*a, **k))
        p["sample"].update(scene_id=scene["scene_id"], category_map_id=cmap["map_id"])
        return p
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        p = write_inputs(tmp / "in", scene, cmds, cmap, views)
        argv = lambda out: ["--scene", str(p["scene"]), "--commands", str(p["commands"]), "--category-map",  # noqa: E731
                            str(p["category_map"]), "--inventory-views", str(p["inventory_views"]), "--out", str(out)]

        def main(out, patch=None):
            saved = []
            am.load_protocol = fixture_protocol
            for mod, name, fn in patch or []:
                saved.append((mod, name, getattr(mod, name)))
                setattr(mod, name, fn)
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    return mm.main(argv(out))
            finally:
                am.load_protocol = real_protocol
                for mod, name, fn in saved:
                    setattr(mod, name, fn)
        check("a successful audit with unassessed rows exits 0", main(tmp / "ok") == 0 and (tmp / "ok/summary.json").exists())
        check("an existing output folder is refused with exit 2", main(tmp / "ok") == 2)

        def boom(*a, **k):
            raise RuntimeError("controlled test exception")
        check("a controlled internal failure exits 3 and publishes nothing",
              main(tmp / "boom", [(am, "summarize", boom)]) == 3 and not (tmp / "boom").exists())
        calls = [0]
        real_write = om._write_file

        def failing(path, data):
            calls[0] += 1
            if calls[0] == 2:
                raise OSError("simulated disk failure")
            real_write(path, data)
        code = main(tmp / "broken", [(om, "_write_file", failing)])
        check("a write failure exits 3 and leaves no partial output",
              code == 3 and sorted(x.name for x in tmp.iterdir()) == ["in", "ok"], f"exit {code}")
        r = subprocess.run([sys.executable, "-m", "grounding.subscenes.iref_vla"] + argv(tmp / "real"), capture_output=True,
                           text=True, cwd=str(REPO))
        check("the real CLI limits itself to the A2.2a sample: a fixture scene exits 2", r.returncode == 2
              and "E_EVAL_INPUT" in r.stdout and not (tmp / "real").exists(), f"exit {r.returncode}")


# ---------------------------------------------------------------------------------------------- pinned sample
def sample_dir():
    for i, a in enumerate(sys.argv):
        if a == "--sample" and i + 1 < len(sys.argv):
            return Path(sys.argv[i + 1])
    env = os.environ.get("SECOND_EYES_IREF_VLA_SAMPLE")
    return Path(env) if env else None


def check_sample(S):
    print("-- the pinned sample (acceptance): import, then one audit")
    d = sample_dir()
    A = importlib.import_module("grounding.adapters.iref_vla")
    files = A.PINNED["files"]
    if d is None or not all((d / f["name"]).is_file() for f in files.values()):
        check("pinned sample available (pass --sample DIR, or run --unit-only for fixtures only)", False,
              f"not found in {d}" if d else "no location given")
        return
    s = EXP["sample"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        imp = tmp / "import"
        A.run_import(*(d / files[r]["name"] for r in ("objects", "regions", "vocabulary")), imp,
                     statements=d / files["statements"]["name"], graph=d / files["graph"]["name"])
        mi, ro = imp / "model_inputs", imp / "reference_only"
        r = subprocess.run([sys.executable, "-m", "grounding.subscenes.iref_vla", "--scene", str(mi / "scene.annotated.json"),
                            "--commands", str(mi / "commands"), "--category-map", str(mi / "category-map.json"),
                            "--inventory-views", str(ro / "inventory-views.json"), "--out", str(tmp / "audit")],
                           capture_output=True, text=True, cwd=str(REPO))
        check("the sample audit completes with exit 0", r.returncode == 0, (r.stdout + r.stderr)[-200:])
        if r.returncode != 0:
            return
        rows = [json.loads(x) for x in (tmp / "audit/selections.jsonl").read_text(encoding="utf-8").splitlines()]
        sm = json.loads((tmp / "audit/summary.json").read_text(encoding="utf-8"))
        check(f"{s['rows']} rows, two per command", len(rows) == s["rows"]
              and len({r["parent_command_id"] for r in rows}) == s["N"])
        v = sm["views"]
        check("N 1,936 = P 1,179 + U 757 in each view, with the parser's reasons",
              all((v[x]["N"], v[x]["P"], v[x]["U"]) == (s["N"], s["P"], s["U"])
                  and v[x]["unsupported_reasons"] == s["unsupported_reasons"] for x in VIEWS))
        check("the required-size histograms, both views", all(v[x]["required_count_histogram"] == s["histogram"][x]
                                                              for x in VIEWS), str({x: v[x]["required_count_histogram"] for x in VIEWS}))
        check("at ten objects: fits, over budget and unassessed in each view",
              all({"fits": v[x]["fits"], "over_budget": v[x]["over_budget"], "unassessed_parse": v[x]["U"]} == s["at_10"][x]
                  for x in VIEWS))
        paired = {f"{a}|{b}": n for a, row in sm["paired"]["counts"].items() for b, n in row.items() if n}
        check("paired statuses: 1,004 fit both, 91 only source-known, 84 neither, 757 unassessed, none only full",
              {k: paired.get(k, 0) for k in s["paired"]} == s["paired"], str(paired))
        check("coverage at every limit 3 to 10, with both denominators", all(
            v[x]["by_limit"][k]["count"] == n and v[x]["by_limit"][k]["of_all"]["denominator"] == s["N"]
            and v[x]["by_limit"][k]["of_parsed"]["denominator"] == s["P"]
            for x in VIEWS for k, n in s["at_most_limit"][x].items()))
        check("123 distinct assessed memberships per view", all(v[x]["distinct_assessed_selections"]
                                                                == s["distinct_assessed_memberships"] for x in VIEWS))
        by = {}
        for r in rows:
            by.setdefault(r["parent_command_id"], {})[r["view_id"]] = r
        added = all(sorted(set(p["full_inventory"]["retained_object_ids"]) - set(p["source_known_nyu"]["retained_object_ids"]))
                    == s["unknown_added_in_full"] for p in by.values() if p["full_inventory"]["status"] != "unassessed_parse")
        check("the full inventory adds exactly obj_055 and obj_058 to every assessed selection", added)
        report = (tmp / "audit/report.md").read_text(encoding="utf-8")
        check("the sample report carries the five required statements",
              all(st in report for st in EXP["report_statements"]))
        print(f"   distinct fitting selections (measured): full {v['full_inventory']['distinct_fitting_selections']}, "
              f"source-known {v['source_known_nyu']['distinct_fitting_selections']}")


def main() -> int:
    S = importlib.import_module("grounding.subscenes.iref_vla")
    E = importlib.import_module("grounding.evaluation.iref_vla")
    unit_only = "--unit-only" in sys.argv
    check_selector(S)
    check_pipeline(S, E)
    check_inputs(S, E)
    check_outputs(S, E)
    check_determinism(S)
    check_no_answers(S)
    check_resolver(S)
    check_cli(S, E)
    if unit_only:
        print("UNIT-ONLY MODE: the pinned-sample acceptance did not run; this is not sample acceptance")
    else:
        check_sample(S)
    tail = " (UNIT-ONLY: not sample acceptance)" if unit_only else ""
    print(f"{COUNT[0]} checks; {'FAILED: ' + ', '.join(FAILED) if FAILED else 'all checks passed'}{tail}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
