"""Tests of the A2.2b text-only rules baseline on the pinned IRef-VLA sample (D75).

    python grounding/tests/test_iref_vla_evaluation.py --sample DIR    full acceptance: fixtures, then the real sample
    python grounding/tests/test_iref_vla_evaluation.py --unit-only     fixtures only; NOT sample acceptance
    (or: python -m grounding.tests.test_iref_vla_evaluation ...)

DIR holds A2.2a's five pinned IRef-VLA files; SECOND_EYES_IREF_VLA_SAMPLE may name it instead. The sample section
re-imports them through A2.2a's pin-verified adapter, then predicts and scores the whole population: a few minutes.
Without the files and without --unit-only, the sample check is one FAIL, never a skip. Expectations come from
fixtures/iref_vla_evaluation/expectations.json, fixed before the implementation existed. Prints one line per check
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
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from grounding.contract import validate as contract  # noqa: E402

FX = REPO / "grounding" / "tests" / "fixtures" / "iref_vla_evaluation"
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


def load(name):
    return json.loads((FX / name).read_text(encoding="utf-8"))


def codes(err):
    return sorted({i["code"] for i in getattr(err, "issues", [])}) or [f"{type(err).__name__}: {err}"]


def text_of(err):
    return " ".join(f"{i.get('path', '')} {i['message']}" for i in getattr(err, "issues", []))


def run_fixture(E, key, *, commands=None, limits=None, view_order=None, scene=None, views=None):
    return E.predict(scene or load(f"scene.{key}.json"), commands if commands is not None else load(f"commands.{key}.json"),
                     load("category-map.fixture.json"), views or load(f"views.{key}.json"),
                     relation_config_path=REL_CFG, direction_config_path=DIR_CFG, limits=limits, view_order=view_order)


def record(pred, parent, view):
    found = [p for p in pred.predictions if p["parent_command_id"] == parent and p["view_id"] == view]
    return found[0] if len(found) == 1 else None


# ------------------------------------------------------------------------------------------------------ parser
def check_parser(E):
    print("-- parser: the reviewed case table (section 10)")
    colours = EXP["colours"]
    for case in EXP["parser_cases"]:
        got = E.parse(case["text"], categories=case["categories"], colours=colours)
        check(f"{case['id']} {case['text']!r}", got == case["expected"],
              "" if got == case["expected"] else json.dumps(got)[:240])
    print("-- parser: every surface alias, under both wrappers")
    bad = [f"{c['wrapper']} {c['phrase']}" for c in EXP["alias_cases"]
           if E.parse(c["text"], categories=EXP["alias_lexicon"], colours=colours) != c["expected"]]
    check(f"all {len(EXP['alias_cases'])} alias cases map to their declared relation and k", not bad, str(bad[:5]))
    from grounding.resolution import validate as RV
    parsed = [c for c in EXP["parser_cases"] if c["expected"]["parse_status"] == "parsed"] + EXP["alias_cases"]
    rejected = []
    for c in parsed:
        interp = E.parse(c["text"], categories=c.get("categories", EXP["alias_lexicon"]), colours=colours)["interpretation"]
        q = {"schema_version": 1, "record_type": "grounding_query", "query_id": "test.q", "scene_id": "test.scene",
             "scene_revision": 0, "evidence_profile": "annotated", "command_id": "test.command", "interpretations": [interp]}
        if RV.check_query(q)[0]:
            rejected.append(c["text"])
    check("every accepted parsed query passes the existing query validator", not rejected, str(rejected[:3]))
    misuse = EXP["api_misuse"]
    for value in (b"the box that is on the table", None, 42):
        try:
            E.parse(value, categories=["box", "table"], colours=colours)
            got = "no error"
        except Exception as e:  # noqa: BLE001 - the type is the check
            got = type(e).__name__
        check(f"wrong text type {type(value).__name__} is rejected as {misuse['non_str_text']}, never str()-converted",
              got == misuse["non_str_text"], got)
    try:
        E.parse("", categories=["box"], colours=colours)
        got = "no error"
    except Exception as e:  # noqa: BLE001
        got = type(e).__name__
    check(f"empty text is rejected as {misuse['empty_text']}", got == misuse["empty_text"], got)
    cmd = copy.deepcopy(load("commands.f0.json")[0])
    cmd["text"] = ""
    found = [i.code for i in contract.validate([("command", cmd)]) if i.is_error]
    check("the command contract rejects empty text", found == [misuse["empty_command_contract_code"]], str(found))
    params = list(inspect.signature(E.parse).parameters)
    check("the parser takes text and vocabularies only: no scene, IDs or annotations", params == ["text", "categories", "colours"],
          str(params))
    rec = E.parse_record("fixture.iref_eval.f0.r01", "the box that is on the table", categories=["box", "table"],
                         colours=colours)
    check("a parse record is the closed iref_parse record of section 3.4",
          set(rec) == {"format_version", "record_type", "parent_command_id", "text", "parse_status", "parse_reason",
                       "features", "analysis_count", "interpretation"} and rec["record_type"] == "iref_parse"
          and type(rec["format_version"]) is int and type(rec["analysis_count"]) is int)


# ------------------------------------------------------------------------------------------- resolver cases
def check_resolver_cases(E):
    print("-- resolver integration: the reviewed cases (section 11), end to end")
    cache = {}
    for case in EXP["resolver_cases"]:
        key = (case["scene"], json.dumps(case.get("limits"), sort_keys=True))
        if key not in cache:
            cmds = load(f"commands.{case['scene']}.json")
            if case.get("limits"):
                cmds = [c for c in cmds if c["command_id"] == case["command"]]
            cache[key] = run_fixture(E, case["scene"], commands=cmds, limits=case.get("limits"))
        pred = cache[key]
        for view in case["views"]:
            rec = record(pred, case["command"], view)
            if rec is None:
                check(f"{case['id']} {view}", False, "no prediction record")
                continue
            res = rec["resolution"]
            if "processing_status" in case:
                b = res["budget"] or {}
                ok = (res["processing_status"] == case["processing_status"] and res["result"] is None
                      and b.get("name") == case["budget_name"] and b.get("limit") == case["budget_limit"]
                      and ("budget_used" not in case or b.get("used") == case["budget_used"])
                      and ("budget_next_required" not in case or b.get("next_required") == case["budget_next_required"])
                      and ("candidate_checks" not in case
                           or res["diagnostics"]["work"]["candidate_checks"] == case["candidate_checks"]))
                check(f"{case['id']} {view}: {case['processing_status']} at {case['budget_name']}", ok,
                      f"{res['processing_status']} {b}")
                continue
            r = res["result"] or {}
            ok = (res["processing_status"] == "completed" and r.get("status") == case["status"]
                  and r.get("target_id") == case["target_id"]
                  and all(c in r.get("reason_codes", []) for c in case.get("reason_codes_include", [])))
            check(f"{case['id']} {view}: {case['status']} {case['target_id'] or ''}".rstrip(), ok,
                  f"{res['processing_status']} {r.get('status')} {r.get('target_id')} {r.get('reason_codes')}")


# ------------------------------------------------------------------------------------------- views, identity
def check_views(E):
    print("-- inventory views and identities (section 4)")
    pred = run_fixture(E, "f1")
    scene, full, known = load("scene.f1.json"), pred.scenes["full_inventory"], pred.scenes["source_known_nyu"]
    enc = lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False)  # noqa: E731
    check("full view keeps all 8 objects; source-known keeps 7, without obj_008",
          [o["object_id"] for o in full["objects"]] == [o["object_id"] for o in scene["objects"]]
          and [o["object_id"] for o in known["objects"]] == [o["object_id"] for o in scene["objects"]][:7])
    same = all(enc(o) == enc(c) for v in (full, known) for o in v["objects"] for c in scene["objects"]
               if o["object_id"] == c["object_id"])
    check("retained object records are canonically byte-equivalent to the source's", same)
    rest = lambda s: {k: v for k, v in s.items() if k not in ("scene_id", "objects")}  # noqa: E731
    check("only scene_id and membership change: frame, sources, revision, profile and map kept",
          rest(full) == rest(scene) == rest(known) and full["scene_id"] == "fixture.iref_eval.f1.fi"
          and known["scene_id"] == "fixture.iref_eval.f1.kn")
    ctx = {c["command_id"]: c for c in pred.contexts}
    parent = load("commands.f1.json")[0]
    fi = ctx.get(parent["command_id"] + ".fi")
    check("a derived context: suffix on the ID, derived scene_id, everything else the original's",
          fi is not None and {k: v for k, v in fi.items() if k not in ("command_id", "scene_id")}
          == {k: v for k, v in parent.items() if k not in ("command_id", "scene_id")} and fi["scene_id"] == full["scene_id"])
    q = record(pred, parent["command_id"], "full_inventory")["query"]
    check("query_id is the derived command ID plus .q", q["query_id"] == parent["command_id"] + ".fi.q"
          and q["command_id"] == parent["command_id"] + ".fi" and q["scene_id"] == full["scene_id"])
    pm = importlib.import_module("grounding.evaluation.iref_vla.predict")
    base = load("views.f1.json")

    def tampered(fn):
        v = copy.deepcopy(base)
        fn(v)
        return v
    ex = lambda v: v["views"][1]  # noqa: E731
    cases = {
        "a same-count swap": tampered(lambda v: (ex(v)["included_object_ids"].__setitem__(0, "obj_008"),
                                                ex(v)["excluded"][0].update(object_id="obj_001", source_object_id="1"))),
        "a duplicate member": tampered(lambda v: ex(v)["included_object_ids"].__setitem__(1, "obj_001")),
        "a missing member": tampered(lambda v: ex(v)["included_object_ids"].pop()),
        "an extra member": tampered(lambda v: ex(v)["included_object_ids"].append("obj_009")),
        "a wrong exclusion reason": tampered(lambda v: ex(v)["excluded"][0].__setitem__("reason", "too_ambiguous")),
        "a wrong parent identity": tampered(lambda v: v.__setitem__("parent_scene_id", "fixture.iref_eval.f0")),
        "a third view": tampered(lambda v: v["views"].append(copy.deepcopy(v["views"][0]))),
    }
    real, calls = pm.resolve, [0]

    def counting(*a, **k):
        calls[0] += 1
        return real(*a, **k)
    pm.resolve = counting
    try:
        for name, views in cases.items():
            calls[0] = 0
            try:
                run_fixture(E, "f1", views=views)
                got = ["no error"]
            except E.EvaluationInputError as e:
                got = codes(e)
            check(f"a view manifest with {name} fails before any resolver call", calls[0] == 0 and got != ["no error"]
                  and all(c.startswith("E_EVAL_") for c in got), f"{got}, {calls[0]} calls")
    finally:
        pm.resolve = real
    long = copy.deepcopy(load("commands.f0.json")[:1])
    long[0]["command_id"] = "fixture.iref_eval.f0." + "x" * 104
    try:
        run_fixture(E, "f0", commands=long)
        got = ["no error"]
    except E.EvaluationInputError as e:
        got = codes(e)
    check("a derived identity over 128 characters is rejected, never truncated", got != ["no error"], str(got))
    forward = E.semantic_hash(run_fixture(E, "f1").predictions)
    script = ("import json, sys; sys.path.insert(0, %r)\nfrom pathlib import Path\nfrom grounding.evaluation.iref_vla import "
              "predict, semantic_hash\nfx = Path(%r)\nl = lambda n: json.loads((fx / n).read_text(encoding='utf-8'))\n"
              "p = predict(l('scene.f1.json'), l('commands.f1.json'), l('category-map.fixture.json'), l('views.f1.json'), "
              "relation_config_path=Path(%r), direction_config_path=Path(%r), view_order=['source_known_nyu', "
              "'full_inventory'])\nprint(semantic_hash(p.predictions))\n") % (str(REPO), str(FX), str(REL_CFG), str(DIR_CFG))
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=str(REPO))
    check("views run in reverse order in a fresh process: the same semantic hash",
          out.returncode == 0 and out.stdout.strip() == forward, out.stderr[-200:])
    moved = copy.deepcopy(load("scene.f0.json"))
    moved["scene_id"] = "fixture.iref_eval.f0_moved"
    moved["objects"][1]["geometry"]["center_m"]["value"] = [2.03, 0.0, 0.0]
    cmds = [dict(copy.deepcopy(c), scene_id=moved["scene_id"]) for c in load("commands.f0.json")]
    views = copy.deepcopy(load("views.f0.json"))
    views["parent_scene_id"] = moved["scene_id"]
    a, b = run_fixture(E, "f0"), run_fixture(E, "f0", scene=moved, commands=cmds, views=views)
    r_a, r_b = (record(p, "fixture.iref_eval.f0.r01", "full_inventory")["resolution"]["result"]["status"] for p in (a, b))
    check("another geometry with the same texts: identical parses, its own resolver result",
          [enc(x) for x in a.parses] == [enc(x) for x in b.parses] and (r_a, r_b) == ("resolved", "ambiguous"),
          f"{r_a} {r_b}")
    shuffled = copy.deepcopy(load("scene.f1.json"))
    shuffled["objects"].reverse()
    c = run_fixture(E, "f1", scene=shuffled, commands=list(reversed(load("commands.f1.json"))))
    check("reordered scene objects and commands give the same semantic predictions and parses",
          E.semantic_hash(c.predictions) == forward and [enc(x) for x in c.parses]
          == [enc(x) for x in run_fixture(E, "f1").parses])


# --------------------------------------------------------------------------------------- prediction outputs
def check_outputs(E):
    print("-- prediction outputs, determinism and the semantic hash (section 6)")
    pred = run_fixture(E, "f1")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "pred"
        E.publish_prediction(pred, out)
        names = sorted(str(p.relative_to(out)).replace(os.sep, "/") for p in out.rglob("*") if p.is_file())
        check("prediction layout as section 6", names == ["category-map.json", "command-contexts.jsonl", "manifest.json",
                                                          "parses.jsonl", "prediction-summary.json", "predictions.jsonl",
                                                          "resolver-config.json", "scenes/full_inventory.json",
                                                          "scenes/source_known_nyu.json"], str(names))
        jsonl = (out / "predictions.jsonl").read_bytes()
        lines = jsonl.decode("utf-8").split("\n")
        check("JSONL: one compact sorted-key record per line, LF only",
              jsonl.endswith(b"\n") and b"\r" not in jsonl and lines[-1] == ""
              and all(l == json.dumps(json.loads(l), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                      for l in lines[:-1]))
        man = (out / "manifest.json").read_bytes()
        check("JSON: sorted keys, final LF; no absolute machine path in the manifest",
              man.endswith(b"\n") and str(Path(tmp)).encode() not in man and str(REPO).encode() not in man)
        m = json.loads(man)
        prot = EXP["protocol"]
        check("manifest: action origin, exact allowances, library identities and the semantic-hash rule",
              m["action_origin"] == prot["action_origin"] and m["resolver"]["limits"] == prot["limits"]
              and m["library_identities"] == prot["library_identities"] and "timings_s" in m["semantic_hash_rule"])
        cfg = json.loads((out / "resolver-config.json").read_text(encoding="utf-8"))
        check("resolver-config.json: the protocol's IDs and limits, sorted map categories, the ten colours",
              cfg["config_id"] == prot["resolver_config_id"] and cfg["vocabulary_id"] == prot["vocabulary_id"]
              and cfg["limits"] == prot["limits"] and cfg["colour_labels"] == EXP["colours"]
              and cfg["category_labels"] == sorted(load("category-map.fixture.json")["model_vocabulary"]))
        check("category-map.json is the input map unchanged",
              json.loads((out / "category-map.json").read_text(encoding="utf-8")) == load("category-map.fixture.json"))
        order = [(p["parent_command_id"], p["view_id"]) for p in pred.predictions]
        check("predictions ordered by parent command, then full before source-known",
              order == sorted(order, key=lambda x: (x[0], VIEWS.index(x[1]))) and len(order) == 6)
        summary = json.loads((out / "prediction-summary.json").read_text(encoding="utf-8"))
        flat = json.dumps(summary)
        check("prediction summary: no correctness or annotation-derived field",
              not any(w in flat for w in ("correct", "agreement", "annotation", "source_target")))
    again = run_fixture(E, "f1")
    enc = lambda xs: [json.dumps(x, sort_keys=True) for x in xs]  # noqa: E731
    check("a second run: parses, contexts, scenes, config and semantic hash identical",
          enc(again.parses) == enc(pred.parses) and enc(again.contexts) == enc(pred.contexts)
          and enc(again.scenes.values()) == enc(pred.scenes.values()) and again.resolver_config == pred.resolver_config
          and E.semantic_hash(again.predictions) == E.semantic_hash(pred.predictions))
    recs = pred.predictions
    before = copy.deepcopy(recs)
    h = E.semantic_hash(recs)
    check("the semantic projection does not mutate its input", recs == before)

    def variant(fn):
        v = copy.deepcopy(recs)
        fn(v)
        return E.semantic_hash(v)
    t = lambda v: v[0]["resolution"]["diagnostics"]["timings_s"].update(evaluation=99.0)  # noqa: E731
    check("changing only timings_s leaves the semantic hash unchanged", variant(t) == h)
    res = lambda v: v[0]["resolution"]  # noqa: E731
    changes = {
        "a target": lambda v: res(v)["result"].__setitem__("target_id", "obj_007"),
        "a reason code": lambda v: res(v)["result"].__setitem__("reason_code", "no_matching_reference"),
        "a work count": lambda v: res(v)["diagnostics"]["work"].__setitem__("candidate_checks", 12345),
        "the query": lambda v: v[0]["query"]["interpretations"][0].__setitem__("action", "GO_TO"),
        "a configuration identity": lambda v: res(v)["diagnostics"]["identities"].__setitem__(
            "relation_config_identity", "relations.v1@000000000000"),
    }
    for name, fn in changes.items():
        check(f"changing {name} changes the semantic hash", variant(fn) != h)
    params = set(inspect.signature(E.predict).parameters) | set(inspect.signature(E.run_predict).parameters)
    check("prediction takes no annotation argument", not any("annot" in p for p in params), str(sorted(params)))
    acts = {it["action"] for p in pred.predictions for it in p["query"]["interpretations"]}
    check("the fixed protocol action INSPECT is in every query; command text unchanged",
          acts == {"INSPECT"} and all(c["text"] == next(o["text"] for o in load("commands.f1.json")
                                                         if c["command_id"].startswith(o["command_id"] + "."))
                                      for c in pred.contexts))
    names = {n for mod in ("parse", "predict", "score", "protocol", "output") for n in
             dir(importlib.import_module(f"grounding.evaluation.iref_vla.{mod}"))}
    check("no execute or action-inference path exists", not any("execute" in n or "infer_action" in n for n in names))


# -------------------------------------------------------------------------------------------------- scoring
def scoring_fixture(E, tmp, *, outcomes=None, annotations=None):
    """Real predictions for the score scene, with the reviewed outcome pairs patched in as synthetic results."""
    pred = run_fixture(E, "score")
    outcomes = outcomes or EXP["scoring"]["outcomes"]
    for p in pred.predictions:
        short = p["parent_command_id"].rsplit(".", 1)[1]
        status, target = outcomes[p["view_id"]][short]
        res = p["resolution"]
        counts = {k: 0 for k in ("resolved", "ambiguous", "no_match", "insufficient_information", "unsupported")}
        if status == "budget_exceeded":
            res.update(processing_status="budget_exceeded", result=None,
                       budget={"name": "max_predicate_calls", "limit": 0, "used": 0, "next_required": 1})
            res["diagnostics"]["outcome_counts"] = counts
            continue
        if status == "unsupported":
            continue  # the real parser-driven result
        ids = target if isinstance(target, list) else [target]
        counts[status] = 1
        res.update(processing_status="completed", budget=None)
        res["result"].update(status=status, target_id=target if status == "resolved" else None, action="INSPECT",
                             reason_code="unique_target_consensus" if status == "resolved"
                             else "nonunique_target_or_interpretation",
                             reason_codes=[] if status == "resolved" else ["multiple_definite_matches"],
                             conditional=False)
        res["result"]["candidates"].update(target_ids=ids, uncertain_match_ids=[], coverage="evaluated_branch_union")
        res["diagnostics"]["outcome_counts"] = counts
    E.finalize(pred)
    out = Path(tmp) / "pred"
    E.publish_prediction(pred, out)
    ann = annotations or load("annotations.score.json")
    return E.load_prediction(out), ann


def check_scoring(E):
    print("-- scoring: the reviewed outcome pairs (section 12)")
    want = EXP["scoring"]["expected"]
    with tempfile.TemporaryDirectory() as tmp:
        pred, ann = scoring_fixture(E, tmp)
        res = E.score(pred, ann)
        s = res.summary
        check("population: N 5, P 4, 7 annotation records, 1 repeated-expression group",
              (s["views"]["full_inventory"]["N"], s["views"]["full_inventory"]["P"], s["population"]["annotation_records"],
               s["population"]["repeated_expression_groups"]) == (want["N"], want["P"], want["annotation_records"],
                                                                 want["repeated_groups"]))
        for view in VIEWS:
            v, w = s["views"][view], want[view]
            ratios = {k: [v[k]["numerator"], v[k]["denominator"]] for k in ("parser_coverage", "resolved_fraction",
                                                                             "all_command_agreement", "resolved_agreement")}
            check(f"{view}: R {w['R']}, C {w['C']}, and each ratio with its numerator and denominator",
                  (v["R"], v["C"]) == (w["R"], w["C"]) and all(ratios[k] == w[k] for k in ratios), str(ratios))
            check(f"{view}: the seven outcome bins sum to N", sum(v["bins"].values()) == want["N"] and len(v["bins"]) == 7)
        scores = {r["parent_command_id"].rsplit(".", 1)[1]: r for r in res.scores}
        for view in VIEWS:
            got = sorted(c for c, r in scores.items() if r["views"][view]["correct"])
            check(f"{view}: correct selections are exactly {want['correct'][view]}", got == want["correct"][view], str(got))
        c3 = scores["c3"]["views"]["full_inventory"]
        check("an ambiguous result listing the source target is not a correct selection",
              c3["bin"] == "ambiguous" and c3["correct"] is False)
        check("the paired transition table sums to N", s["transitions"]["total"] == want["transitions_total"]
              and sum(n for row in s["transitions"]["counts"].values() for n in row.values()) == want["transitions_total"])
        more = copy.deepcopy(ann)
        extra = [e for e in more["entries"] if e["command_id"].endswith(".c1")][0]
        for i in (3, 4):
            more["entries"].append(dict(copy.deepcopy(extra), annotation_id=f"{extra['command_id']}.a{i:03d}",
                                        source_annotation_index=i))
        s2 = E.score(pred, more).summary
        same = all(s2["views"][v][k] == s["views"][v][k] for v in VIEWS
                   for k in ("parser_coverage", "resolved_fraction", "all_command_agreement", "resolved_agreement"))
        check("repeating c1's annotations changes none of the ratios", same and s2["population"]["annotation_records"] == 9)
        alt = copy.deepcopy(ann)
        for e in alt["entries"]:
            if e["command_id"].endswith(".c2"):
                e["mapped_references"]["target"] = "obj_005"
        s3 = E.score(pred, alt).summary
        check("a valid alternate reference target may change the scores",
              s3["views"]["full_inventory"]["C"] == 2 and s3["views"]["source_known_nyu"]["C"] == 1)
    with tempfile.TemporaryDirectory() as tmp:
        none = {v: {c: ["unsupported", None] if c == "c4" else ["ambiguous", ["obj_002", "obj_004"]]
                    for c in ("c1", "c2", "c3", "c4", "c5")} for v in VIEWS}
        pred, ann = scoring_fixture(E, tmp, outcomes=none)
        v = E.score(pred, ann).summary["views"]["full_inventory"]["resolved_agreement"]
        check("no resolved command: resolved-only agreement is null over a zero denominator", v["numerator"] == 0
              and v["denominator"] == 0 and v["value"] is None, str(v))
    print("-- scoring rejects inconsistent predictions or references, naming the identity")
    with tempfile.TemporaryDirectory() as tmp:
        pred, ann = scoring_fixture(E, tmp)
        c1 = "fixture.iref_eval.score.c1"

        def ann_case(fn):
            a = copy.deepcopy(ann)
            fn(a)
            return pred, a

        def pred_case(fn):
            p = copy.deepcopy(pred)
            fn(p)
            return p, ann
        first = lambda a, c: [e for e in a["entries"] if e["command_id"] == c]  # noqa: E731
        cases = {
            "a duplicate prediction": (pred_case(lambda p: p.predictions.append(copy.deepcopy(p.predictions[0]))), c1),
            "a missing view": (pred_case(lambda p: p.predictions.pop(1)), c1),
            "an extra command": (ann_case(lambda a: a["entries"].pop(next(i for i, e in enumerate(a["entries"])
                                                                          if e["command_id"].endswith(".c5")))),
                                 "fixture.iref_eval.score.c5"),
            "a nonexistent target": (ann_case(lambda a: first(a, c1)[0]["mapped_references"].__setitem__(
                "target", "obj_099")), "obj_099"),
            "contradictory targets in one command": (ann_case(lambda a: first(a, c1)[1]["mapped_references"].__setitem__(
                "target", "obj_003")), c1),
            "conflicting source relation labels": (ann_case(lambda a: first(a, c1)[2]["source_payload"].__setitem__(
                "relation", "near")), c1),
            "a malformed reference": (ann_case(lambda a: first(a, c1)[0].pop("mapped_references")), c1),
        }
        for name, ((p, a), ident) in cases.items():
            try:
                E.score(p, a)
                got, txt = ["no error"], ""
            except E.EvaluationInputError as e:
                got, txt = codes(e), text_of(e)
            check(f"{name} fails scoring and names {ident}", got != ["no error"] and ident in txt, f"{got} {txt[:120]}")
        out = Path(tmp) / "pred"
        lines = (out / "predictions.jsonl").read_text(encoding="utf-8").split("\n")
        rec = json.loads(lines[0])
        rec["resolution"]["result"]["target_id"] = "obj_007" if rec["resolution"]["result"] else None
        tamper = Path(tmp) / "tampered"
        tamper.mkdir()
        for f in out.rglob("*"):
            if f.is_file():
                dst = tamper / f.relative_to(out)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(f.read_bytes())
        (tamper / "predictions.jsonl").write_text("\n".join([json.dumps(rec, sort_keys=True, ensure_ascii=False,
                                                                        separators=(",", ":"))] + lines[1:]),
                                                  encoding="utf-8")
        try:
            E.load_prediction(tamper)
            got = ["no error"]
        except E.EvaluationInputError as e:
            got = codes(e)
        check("an edited prediction is caught by the semantic hash and recomputed counts", got != ["no error"], str(got))


# ------------------------------------------------------------------------------------------ safety and types
def check_safety(E):
    print("-- publication safety, technical failures and types")
    fx = {"scene": FX / "scene.f1.json", "commands": None, "category_map": FX / "category-map.fixture.json",
          "inventory_views": FX / "views.f1.json", "relation_config": REL_CFG, "direction_config": DIR_CFG}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cdir = tmp / "commands"
        cdir.mkdir()
        for c in load("commands.f1.json"):
            (cdir / f"{c['command_id']}.json").write_text(json.dumps(c), encoding="utf-8")
        fx["commands"] = cdir
        E.run_predict(**fx, out=tmp / "a", sample_only=False)
        try:
            E.run_predict(**fx, out=tmp / "a", sample_only=False)
            got = ["no error"]
        except E.EvaluationInputError as e:
            got = codes(e)
        check("an existing prediction folder is refused", got == ["E_EVAL_OUTPUT_EXISTS"], str(got))
        om = importlib.import_module("grounding.evaluation.iref_vla.output")
        real, calls = om._write_file, [0]

        def failing(path, data):
            calls[0] += 1
            if calls[0] == 3:
                raise OSError("simulated disk failure")
            real(path, data)
        om._write_file = failing
        try:
            E.run_predict(**fx, out=tmp / "broken", sample_only=False)
            got = ["no error"]
        except E.EvaluationOutputError as e:
            got = codes(e)
        finally:
            om._write_file = real
        check("a write failure publishes nothing and leaves no partial folder",
              got == ["E_EVAL_OUTPUT_IO"] and sorted(p.name for p in tmp.iterdir()) == ["a", "commands"], str(got))
        pm = importlib.import_module("grounding.evaluation.iref_vla.predict")
        realr = pm.resolve

        def boom(*a, **k):
            raise RuntimeError("controlled test exception")
        pm.resolve = boom
        try:
            E.run_predict(**fx, out=tmp / "boom", sample_only=False)
            got = "no error"
        except RuntimeError:
            got = "RuntimeError"
        finally:
            pm.resolve = realr
        check("an unexpected exception propagates: it never becomes a semantic result, and nothing is published",
              got == "RuntimeError" and not (tmp / "boom").exists())
        try:
            E.run_predict(**fx, out=tmp / "sample_only")
            got = ["no error"]
        except E.EvaluationInputError as e:
            got = codes(e)
        check("the command-line path is limited to the A2.2a sample", got != ["no error"], str(got))
        zero = run_fixture(E, "f1", limits={"max_candidate_checks": 0})
        tech = [p for p in zero.predictions if p["resolution"]["processing_status"] == "budget_exceeded"]
        check("a zero work limit is a recorded technical failure, never retried", len(tech) > 0
              and zero.summary["technical_failures"] == len(tech)
              and all(p["resolution"]["budget"]["limit"] == 0 for p in tech))
    prot = json.loads(Path(E.PROTOCOL_PATH).read_text(encoding="utf-8"))
    for path, value in ((("resolver", "limits", "max_candidate_checks"), 10000.0), (("format_version",), 1.0)):
        bad = copy.deepcopy(prot)
        node = bad
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = value
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "protocol.json"
            f.write_text(json.dumps(bad), encoding="utf-8")
            try:
                E.load_protocol(f)
                got = ["no error"]
            except E.EvaluationInputError as e:
                got = codes(e)
        check(f"a whole-valued float at {'.'.join(path)} is rejected without coercion", got != ["no error"], str(got))
    rec = E.parse_record("a.b", "the box that is on the table", categories=["box", "table"], colours=EXP["colours"])
    for field, value in (("analysis_count", 1.0), ("format_version", 1.0)):
        bad = dict(rec, **{field: value})
        try:
            E.validate_parse_record(bad)
            got = ["no error"]
        except E.EvaluationInputError as e:
            got = codes(e)
        check(f"a parse record with {field} {value!r} is rejected", got != ["no error"], str(got))


def check_imports(E):
    print("-- imports and information boundaries")
    bare = [n for n in ("predicates", "directions", "resolve", "parse", "predict", "score", "protocol", "output")
            if n in sys.modules]
    rr = importlib.import_module("grounding.resolution.resolve")
    preds = importlib.import_module("grounding.relations.predicates")
    check("no bare-module fallback imports; one Truth enum shared by the resolver", not bare and rr.P is preds, str(bare))
    pkg = REPO / "grounding" / "evaluation"
    texts = {p.name: p.read_text(encoding="utf-8") for p in pkg.rglob("*.py")}
    check("no sys.path manipulation in the evaluation modules", not any("sys.path" in t for t in texts.values()))
    pm = importlib.import_module("grounding.evaluation.iref_vla.predict")
    src = inspect.getsource(pm)
    check("the prediction module never names the annotation bundle or reference_only data",
          "iref-annotations" not in src and "iref_annotations" not in src and "reference_only" not in src)


# ---------------------------------------------------------------------------------- the D76 correction's cases
def attempt(E, fn):
    """("ok", value), ("rejected", codes, text) or ("crashed", text): an accidental exception is the defect tested."""
    try:
        return ("ok", fn())
    except E.EvaluationInputError as e:
        return ("rejected", codes(e), text_of(e))
    except Exception as e:  # noqa: BLE001 - recorded, not raised, so every case is reported
        return ("crashed", f"{type(e).__name__}: {e}"[:160])


def rejected_with(result, code, *names):
    return result[0] == "rejected" and code in result[1] and all(n in result[2] for n in names)


def check_correction(E):
    print("-- validation correction (D76): text and context consistency, input shapes")
    pm = importlib.import_module("grounding.evaluation.iref_vla.predict")
    from grounding.resolution import validate as RV
    cli = [sys.executable, "-m", "grounding.evaluation.iref_vla"]
    ann_path = FX / "annotations.score.json"
    c1 = "fixture.iref_eval.score.c1"
    farther = "the chair that is farthest from the table"

    def run(args):
        return subprocess.run(cli + args, capture_output=True, text=True, cwd=str(REPO))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pred, ann = scoring_fixture(E, tmp)
        src = tmp / "pred"
        r = run(["score", "--predictions", str(src), "--annotations", str(ann_path), "--out", str(tmp / "s0")])
        s0 = json.loads((tmp / "s0/summary.json").read_text(encoding="utf-8")) if (tmp / "s0/summary.json").exists() else {}
        check("the unchanged five-command fixture scores through the CLI: exit 1 for its two supplied budget outcomes, "
              "the reviewed C counts", r.returncode == 1 and s0.get("technical_failures") == 2
              and [s0["views"][v]["C"] for v in VIEWS] == [1, 2], f"exit {r.returncode}")
        alt = tmp / "pred_text"
        shutil.copytree(src, alt)
        lines = (alt / "parses.jsonl").read_bytes().decode("utf-8").split("\n")
        first = json.loads(lines[0])
        first["text"] = farther if first["text"] == "the chair that is closest to the table" else first["text"]
        lines[0] = json.dumps(first, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        (alt / "parses.jsonl").write_bytes("\n".join(lines).encode("utf-8"))
        r = run(["score", "--predictions", str(alt), "--annotations", str(ann_path), "--out", str(tmp / "s1")])
        check("parse text changed closest -> farthest, nothing else: E_EVAL_PREDICTIONS, CLI exit 2, no score output",
              first["text"] == farther and r.returncode == 2 and "E_EVAL_PREDICTIONS" in r.stdout and c1 in r.stdout
              and not (tmp / "s1").exists(), f"exit {r.returncode} {r.stdout[-160:]}")

        def ctx(p, cid):
            return next(i for i, c in enumerate(p.contexts) if c["command_id"] == cid)
        p = copy.deepcopy(pred)
        p.contexts[ctx(p, c1 + ".fi")]["text"] = farther
        check("one context's text differs from its parse: rejected, never repaired",
              rejected_with(attempt(E, lambda: E.score(p, ann)), "E_EVAL_PREDICTIONS", c1))
        p = copy.deepcopy(pred)
        i = ctx(p, c1 + ".fi")
        p.contexts[i]["text"] = farther
        rec = next(x for x in p.predictions if x["command_id"] == c1 + ".fi")
        rec["resolution"]["diagnostics"]["identities"]["command_sha256"] = RV.canonical_sha256(p.contexts[i])
        E.finalize(p)
        r = attempt(E, lambda: E.score(p, ann))
        check("the same, with its resolver digest and hashes rewritten to match: the text rule still rejects it, naming "
              "parent and view", rejected_with(r, "E_EVAL_PREDICTIONS", c1, "full_inventory"), str(r)[:160])
        p = copy.deepcopy(pred)
        extra = dict(copy.deepcopy(p.contexts[0]), command_id=p.contexts[0]["command_id"] + ".extra")
        p.contexts.append(extra)
        r = attempt(E, lambda: E.score(p, ann))
        check("an extra, otherwise valid context used by no prediction is rejected with its identity",
              rejected_with(r, "E_EVAL_PREDICTIONS", extra["command_id"]), str(r)[:160])
        p = copy.deepcopy(pred)
        p.contexts.pop(ctx(p, c1 + ".kn"))
        check("a missing context is rejected with its identity",
              rejected_with(attempt(E, lambda: E.score(p, ann)), "E_EVAL_PREDICTIONS", c1))
        p = copy.deepcopy(pred)
        p.contexts.append(copy.deepcopy(p.contexts[0]))
        check("a duplicate context is rejected with its identity",
              rejected_with(attempt(E, lambda: E.score(p, ann)), "E_EVAL_PREDICTIONS", p.contexts[0]["command_id"]))
        for value, valid in ((99, False), (1.0, False), (True, False), ("1", False), (1, True)):
            p = copy.deepcopy(pred)
            p.manifest["format_version"] = value
            r = attempt(E, lambda: E.score(p, ann))
            check(f"prediction manifest format_version {value!r}: {'accepted' if valid else 'rejected'}",
                  r[0] == "ok" if valid else rejected_with(r, "E_EVAL_PREDICTIONS"), str(r)[:120])
        for coll in ("parses", "contexts", "predictions"):
            for bad in (None, 7, []):
                p = copy.deepcopy(pred)
                getattr(p, coll).append(bad)
                r = attempt(E, lambda: E.score(p, ann))
                check(f"a {json.dumps(bad)} element in {coll}: a structured rejection, never an exception",
                      rejected_with(r, "E_EVAL_PREDICTIONS"), str(r)[:120])
        nul = tmp / "pred_null"
        shutil.copytree(src, nul)
        with open(nul / "predictions.jsonl", "ab") as f:
            f.write(b"null\n")
        r = attempt(E, lambda: E.load_prediction(nul))
        check("a null line in predictions.jsonl, read back from disk: a structured rejection naming the file",
              rejected_with(r, "E_EVAL_PREDICTIONS", "predictions.jsonl"), str(r)[:120])
        r = run(["score", "--predictions", str(nul), "--annotations", str(ann_path), "--out", str(tmp / "s2")])
        check("the same folder through the score CLI: exit 2, no output, no traceback",
              r.returncode == 2 and not (tmp / "s2").exists() and "Traceback" not in r.stderr, f"exit {r.returncode}")
        roots = {"a null manifest": lambda p: setattr(p, "manifest", None),
                 "a null summary": lambda p: setattr(p, "summary", None),
                 "a scalar summary": lambda p: setattr(p, "summary", 7),
                 "a null manifest resolver": lambda p: p.manifest.__setitem__("resolver", None),
                 "a null manifest view list": lambda p: p.manifest.__setitem__("views", None),
                 "a scalar manifest view entry": lambda p: p.manifest["views"].__setitem__(0, 5),
                 "an array of library identities": lambda p: p.manifest.__setitem__("library_identities", [])}
        for name, fn in roots.items():
            p = copy.deepcopy(pred)
            fn(p)
            r = attempt(E, lambda: E.score(p, ann))
            check(f"{name}: a structured rejection before field access", rejected_with(r, "E_EVAL_PREDICTIONS"),
                  str(r)[:120])
    with tempfile.TemporaryDirectory() as t2:
        quiet = {v: {"c1": ["resolved", "obj_002"], "c2": ["resolved", "obj_006"], "c3": ["resolved", "obj_002"],
                     "c4": ["unsupported", None], "c5": ["resolved", "obj_004"]} for v in VIEWS}
        pq, ann = scoring_fixture(E, t2, outcomes=quiet)
        p = copy.deepcopy(pq)
        p.parses[0]["text"] = farther
        r = attempt(E, lambda: E.score(p, ann))
        check("the same text mismatch is rejected when no outcome is a technical failure",
              pq.summary["technical_failures"] == 0 and rejected_with(r, "E_EVAL_PREDICTIONS", c1), str(r)[:160])
    realr, realp, calls = pm.resolve, pm.parse_record, [0]

    def sentinel(*a, **k):
        calls[0] += 1
        raise AssertionError("computation reached despite invalid input")
    pm.resolve = pm.parse_record = sentinel
    try:
        base = load("views.f1.json")
        cases = {"a null view_id": lambda v: v["views"][1].__setitem__("view_id", None),
                 "an array view_id": lambda v: v["views"][1].__setitem__("view_id", ["source_known_nyu"]),
                 "an array included object ID": lambda v: v["views"][1]["included_object_ids"].__setitem__(0, ["obj_001"]),
                 "a null included object ID": lambda v: v["views"][1]["included_object_ids"].__setitem__(0, None),
                 "an array excluded object ID": lambda v: v["views"][1]["excluded"][0].__setitem__("object_id", ["obj_008"])}
        for name, fn in cases.items():
            v = copy.deepcopy(base)
            fn(v)
            calls[0] = 0
            r = attempt(E, lambda: run_fixture(E, "f1", views=v))
            check(f"inventory views with {name}: E_EVAL_VIEWS before any parser or resolver call",
                  rejected_with(r, "E_EVAL_VIEWS") and calls[0] == 0, f"{str(r)[:120]}, {calls[0]} calls")
        calls[0] = 0
        r = attempt(E, lambda: E.predict(None, load("commands.f1.json"), load("category-map.fixture.json"),
                                         load("views.f1.json"), relation_config_path=REL_CFG, direction_config_path=DIR_CFG))
        check("a null scene through the API: E_EVAL_INPUT before any parser or resolver call",
              rejected_with(r, "E_EVAL_INPUT") and calls[0] == 0, str(r)[:120])
    finally:
        pm.resolve, pm.parse_record = realr, realp
    with tempfile.TemporaryDirectory() as t3:
        t3 = Path(t3)
        cdir = t3 / "commands"
        cdir.mkdir()
        for c in load("commands.f1.json"):
            (cdir / f"{c['command_id']}.json").write_text(json.dumps(c), encoding="utf-8")
        files = (("a null scene file", "null", None), ("a list scene file", "[]", None),
                 ("a scalar map file beside a sample-named scene", json.dumps({"scene_id": "iref.scannet.scene0010_01.full"}),
                  "7"))
        for n, (name, scene_text, map_text) in enumerate(files):
            sp, mp, out = t3 / f"scene{n}.json", t3 / f"map{n}.json", t3 / f"out{n}"
            sp.write_text(scene_text, encoding="utf-8")
            mp.write_text(map_text, encoding="utf-8") if map_text else mp.write_bytes((FX / "category-map.fixture.json").read_bytes())
            r = run(["predict", "--scene", str(sp), "--commands", str(cdir), "--category-map", str(mp), "--inventory-views",
                     str(FX / "views.f1.json"), "--relation-config", str(REL_CFG), "--direction-config", str(DIR_CFG),
                     "--out", str(out)])
            check(f"{name} through the predict CLI: E_EVAL_INPUT, exit 2, no prediction output",
                  r.returncode == 2 and "E_EVAL_INPUT" in r.stdout and not out.exists() and "Traceback" not in r.stderr,
                  f"exit {r.returncode} {(r.stdout + r.stderr)[-120:]}")
    sm = importlib.import_module("grounding.evaluation.iref_vla.score")
    mm = importlib.import_module("grounding.evaluation.iref_vla.__main__")
    with tempfile.TemporaryDirectory() as t4:
        t4 = Path(t4)
        pv, ann = scoring_fixture(E, t4)
        real_s = sm._summary

        def boom(*a, **k):
            raise RuntimeError("controlled test exception")
        sm._summary, err = boom, io.StringIO()
        try:
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                code = mm.main(["score", "--predictions", str(t4 / "pred"), "--annotations", str(ann_path), "--out",
                                str(t4 / "boom")])
        finally:
            sm._summary = real_s
        check("an injected exception in a valid scoring run still exits 3 and publishes nothing",
              code == 3 and "internal error" in err.getvalue() and not (t4 / "boom").exists(), f"exit {code}")
        rp = importlib.import_module("grounding.evaluation.iref_vla.parse")
        rr = importlib.import_module("grounding.resolution.resolve")
        saved = (pm.resolve, pm.parse_record, rp.parse, rp.parse_record, rr.resolve)

        def never(*a, **k):
            raise AssertionError("scoring must not rerun inference")
        pm.resolve = pm.parse_record = rp.parse = rp.parse_record = rr.resolve = never
        try:
            r = attempt(E, lambda: E.score(pv, ann))
        finally:
            pm.resolve, pm.parse_record, rp.parse, rp.parse_record, rr.resolve = saved
        check("with the parser and resolver replaced by fail-on-call sentinels, valid stored predictions still score",
              r[0] == "ok", str(r)[:160])


# ------------------------------------------------------------------------------------------- pinned sample
def sample_dir():
    for i, a in enumerate(sys.argv):
        if a == "--sample" and i + 1 < len(sys.argv):
            return Path(sys.argv[i + 1])
    env = os.environ.get("SECOND_EYES_IREF_VLA_SAMPLE")
    return Path(env) if env else None


def check_sample(E):
    print("-- the real pinned sample (acceptance): import, predict, score")
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
        cli = [sys.executable, "-m", "grounding.evaluation.iref_vla"]
        p = subprocess.run(cli + ["predict", "--scene", str(mi / "scene.annotated.json"), "--commands",
                                  str(mi / "commands"), "--category-map", str(mi / "category-map.json"),
                                  "--inventory-views", str(ro / "inventory-views.json"), "--relation-config",
                                  str(REL_CFG), "--direction-config", str(DIR_CFG), "--out", str(tmp / "pred")],
                           capture_output=True, text=True, cwd=str(REPO))
        check("predict on the sample completes (exit 0, or 1 with technical failures)", p.returncode in (0, 1),
              (p.stdout + p.stderr)[-300:])
        if p.returncode not in (0, 1):
            return
        pd = tmp / "pred"
        lines = lambda f: (pd / f).read_text(encoding="utf-8").splitlines()  # noqa: E731
        summ = json.loads((pd / "prediction-summary.json").read_text(encoding="utf-8"))
        check(f"exactly {s['commands']} parses and {s['prediction_records']} prediction records",
              len(lines("parses.jsonl")) == s["commands"] and len(lines("predictions.jsonl")) == s["prediction_records"]
              and len(lines("command-contexts.jsonl")) == s["prediction_records"])
        scenes = {v: json.loads((pd / "scenes" / f"{v}.json").read_text(encoding="utf-8")) for v in VIEWS}
        ids = {v: [o["object_id"] for o in scenes[v]["objects"]] for v in VIEWS}
        check("views: 61 and 59 objects; source-known excludes exactly obj_055 and obj_058",
              len(ids["full_inventory"]) == s["objects"] and len(ids["source_known_nyu"]) == s["known_objects"]
              and sorted(set(ids["full_inventory"]) - set(ids["source_known_nyu"])) == s["excluded"])
        check("view scene IDs as the protocol table", [scenes[v]["scene_id"] for v in VIEWS]
              == [x[1] for x in EXP["protocol"]["views"]])
        cfg = json.loads((pd / "resolver-config.json").read_text(encoding="utf-8"))
        check("resolver configuration: 893 sorted category labels and the protocol's allowances",
              len(cfg["category_labels"]) == s["category_labels"] and cfg["category_labels"] == sorted(cfg["category_labels"])
              and cfg["limits"] == EXP["protocol"]["limits"])
        check("predict's exit code agrees with its technical-failure count",
              (p.returncode == 1) == (summ["technical_failures"] > 0), f"exit {p.returncode}, {summ['technical_failures']}")
        q = subprocess.run(cli + ["score", "--predictions", str(pd), "--annotations", str(ro / "iref-annotations.json"),
                                  "--out", str(tmp / "score")], capture_output=True, text=True, cwd=str(REPO))
        check("score on the sample completes with the same exit class", q.returncode == p.returncode,
              (q.stdout + q.stderr)[-300:])
        if q.returncode not in (0, 1):
            return
        sd = tmp / "score"
        ss = json.loads((sd / "summary.json").read_text(encoding="utf-8"))
        scores = (sd / "scores.jsonl").read_text(encoding="utf-8").splitlines()
        check(f"all {s['annotations']} annotations used, each command scored once, {s['repeated_groups']} repeated groups",
              ss["population"]["annotation_records"] == s["annotations"] and len(scores) == s["commands"]
              and ss["population"]["repeated_expression_groups"] == s["repeated_groups"]
              and all(ss["views"][v]["N"] == s["commands"] for v in VIEWS))
        check("the seven outcome bins and the transition table each sum to 1,936",
              all(sum(ss["views"][v]["bins"].values()) == s["commands"] for v in VIEWS)
              and ss["transitions"]["total"] == s["commands"])
        report = (sd / "report.md").read_text(encoding="utf-8")
        first = report.split("\n\n")[1] if report.startswith("#") else report.split("\n\n")[0]
        check("the report opens with the development-room caveat",
              "one inspected development room" in first and "annotated geometry" in first
              and "rules parser" in first and "provisional relation semantics" in first, first[:200])
        sm = json.loads((sd / "manifest.json").read_text(encoding="utf-8"))
        check("the scoring manifest records the semantic and the actual prediction hashes",
              sm["predictions"]["semantic_hash"] == summ["semantic_prediction_hash"]
              and len(sm["predictions"]["predictions_jsonl_sha256"]) == 64)
        print("   development measurements (not expectations):")
        for v in VIEWS:
            x = ss["views"][v]
            print(f"   {v}: P {x['P']}/{x['N']}, R {x['R']}, C {x['C']}; bins {x['bins']}")


def main() -> int:
    E = importlib.import_module("grounding.evaluation.iref_vla")
    unit_only = "--unit-only" in sys.argv
    check_parser(E)
    check_resolver_cases(E)
    check_views(E)
    check_outputs(E)
    check_scoring(E)
    check_safety(E)
    check_imports(E)
    check_correction(E)
    if unit_only:
        print("UNIT-ONLY MODE: the pinned-sample acceptance did not run; this is not sample acceptance")
    else:
        check_sample(E)
    tail = " (UNIT-ONLY: not sample acceptance)" if unit_only else ""
    print(f"{COUNT[0]} checks; {'FAILED: ' + ', '.join(FAILED) if FAILED else 'all checks passed'}{tail}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
