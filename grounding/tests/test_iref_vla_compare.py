"""A2.3d (D95) checks for preparation and rules: answer-blind selection, frozen mappings, both tokenizers, integrity.

Run directly (python grounding/tests/test_iref_vla_compare.py) or as a module. Fixtures are A2.3b's labelled fixture
chain with labelled tokenizer doubles; expectations are recomputed independently with hashlib.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import sys
from collections import Counter
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
FAILS, PASSES = [], []
REL_CFG = REPO / "grounding" / "relations" / "relations.v1.json"
DIR_CFG = REPO / "grounding" / "relations" / "directions.v1.json"
KEYS = ("qwen2.5-0.5b-instruct", "qwen2.5-7b-instruct")


def check(name, ok, detail=""):
    (PASSES if ok else FAILS).append(name)
    print(("  ok    " if ok else "  FAIL  ") + name + ("" if ok or not detail else f"  [{detail}]"))


def outcome(fn):
    EI = importlib.import_module("grounding.evaluation.iref_vla.protocol").EvaluationInputError
    try:
        return ("ok", fn())
    except EI as e:
        return ("input_error", sorted({i["code"] for i in e.issues}))
    except (Exception, KeyboardInterrupt) as e:  # noqa: BLE001  (a simulated interruption is an outcome here)
        return ("crashed", f"{type(e).__name__}: {e}")


def hs(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def jsonl(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines()]


def write_jsonl(p, rows):
    Path(p).write_text("".join(json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n" for r in rows),
                       encoding="utf-8")


def rehash(folder, rels):
    m = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    for rel in rels:
        m["files"][rel] = hashlib.sha256((folder / rel).read_bytes()).hexdigest()
    (folder / "manifest.json").write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")


def check_design(D):
    print("-- design: selection and permutations (pure)")
    pol = D.load_policy()
    ids = [f"p{i:02d}" for i in range(20)]
    want = sorted(ids, key=lambda p: (hs("second-eyes/a23d/compare/v1\n" + p), p))[:5]
    check("selection: the first k by SHA-256(salt + LF + ID), recomputed independently", D.select(ids, pol["selection"]["salt"], 5) == want)
    objs = ["obj_003", "obj_001", "obj_010", "obj_002"]
    codes = D.code_assignment(pol, "parent.a", "full_inventory", objs)
    want_codes = {o: "ABCD"[k] for k, o in enumerate(sorted(objs, key=lambda o: (hs("second-eyes/a23d/codes/v1\nparent.a\nfull_inventory\n" + o), o)))}
    check("letters: the k-th object of the codes stream gets the k-th letter (independent recomputation)", codes == want_codes)
    order = D.list_order(pol, "parent.a", "full_inventory", objs)
    want_order = sorted(objs, key=lambda o: (hs("second-eyes/a23d/list/v1\nparent.a\nfull_inventory\n" + o), o))
    check("list order: a separate hash stream (independent recomputation)", order == want_order)
    m = D.mapping_for(pol, {"ask_code": "K", "ask_target": "ASK"}, "parent.a", "full_inventory", objs)
    check("the mapping lists objects in list order with their letters, every letter once, K last",
          [x[1] for x in m[:-1]] == want_order and sorted(x[0] for x in m[:-1]) == list("ABCD") and m[-1] == ["K", "ASK"])
    check("the mapping depends on the view (separate stream inputs)", m != D.mapping_for(pol, {"ask_code": "K", "ask_target": "ASK"},
                                                                                          "parent.a", "source_known_nyu", objs))
    check("the inputs exclude model, format and annotations (the functions take only parent, view and objects)",
          D.mapping_for.__code__.co_varnames[:5] == ("policy", "proto", "parent", "view", "objects"))
    try:
        D.code_assignment(pol, "p", "v", [f"o{i}" for i in range(11)])
        check("eleven objects are refused", False)
    except ValueError:
        check("eleven objects are refused", True)


def check_chain(D, P, R):
    print("-- preparation and rules on A2.3b's fixture chain (labelled tokenizer doubles)")
    SB = importlib.import_module("grounding.tests.test_iref_vla_pilot_scoring")
    A = SB.pilot_helpers()
    PI = importlib.import_module("grounding.inference.iref_vla.prepare")
    C = importlib.import_module("grounding.inference.iref_vla.choices")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle = SB.scoring_bundle(tmp, A)
        ann = SB.annotation_bundle(tmp / "annotations.json", SB.TARGETS)
        src = PI.read_bundle(bundle)
        eligible = PI.eligible_parents(src)
        pilot = sorted(eligible, key=lambda p: (hs("fixture-pilot\n" + p), p))[:1]
        pol = json.loads(D.POLICY_PATH.read_text(encoding="utf-8"))
        pol["population"] = {"rule": "fixture", "expected_eligible": len(eligible), "expected_remaining": len(eligible) - 1,
                             "excluded": {"what": "fixture", "salt": "fixture-pilot\n", "count": 1,
                                          "selected_sha256": hashlib.sha256((pilot[0] + "\n").encode()).hexdigest()}}
        pol["selection"]["count"] = len(eligible) - 1
        pol["expected"] = {"parents": len(eligible) - 1, "parent_views": 2 * (len(eligible) - 1), "requests": 4 * (len(eligible) - 1)}
        pp = tmp / "policy.json"
        pp.write_text(json.dumps(pol), encoding="utf-8")
        toks = {k: A.OffsetCharTokenizer() for k in KEYS}
        with SB.Watch([ann]) as w:
            r = outcome(lambda: P.prepare_compare(bundle=bundle, tokenizer_small=None, tokenizer_large=None, out=tmp / "req",
                                                  policy=pp, tokenizers=toks))
        check(f"preparation completes on {len(eligible)} eligible fixture parents with the annotations inaccessible",
              r[0] == "ok" and not w.denied, str(r)[:200])
        if r[0] != "ok":
            return
        rows = jsonl(tmp / "req" / "request-index.jsonl")
        sel = json.loads((tmp / "req" / "selection.json").read_text())
        check("the pilot parent is excluded and every other eligible parent selected",
              sel["excluded_pilot"]["parent_ids"] == pilot and set(sel["selected_parent_ids"]) == set(eligible) - set(pilot))
        check("four requests per parent in the canonical order, readback clean",
              len(rows) == 4 * len(sel["selected_parent_ids"]) and P.verify_compare_requests(tmp / "req") == [])
        pairs = {}
        for x in rows:
            pairs.setdefault((x["parent_command_id"], x["view_id"]), []).append(x["mapping"])
        check("both formats of each parent and view share one mapping", all(len(v) == 2 and v[0] == v[1] for v in pairs.values()))
        check("at least one mapping departs from scene order (letters or list order)",
              any([m[1] for m in x["mapping"][:-1]] != x["object_ids"] or [m[0] for m in x["mapping"][:-1]] != list("ABCDEFGHIJ"[:len(x["object_ids"])])
                  for x in rows))
        check("both models' token files hold one line per request", all(len((tmp / "req" / "tokens" / f"{k}.jsonl").read_text().splitlines()) == len(rows) for k in KEYS))
        cases = []
        for label, edit in (("a changed letter", lambda x: x["mapping"][0].__setitem__(0, "J")),
                            ("a reordered choices list", lambda x: x["list_order"].reverse()),
                            ("a whole-valued float counter", lambda x: x.__setitem__("object_count", float(x["object_count"])))):
            d = tmp / f"t{len(cases)}"
            shutil.copytree(tmp / "req", d)
            rr = jsonl(d / "request-index.jsonl"); edit(rr[1]); write_jsonl(d / "request-index.jsonl", rr)
            rehash(d, ["request-index.jsonl"])
            cases.append((label, P.verify_compare_requests(d)))
        d = tmp / "tt"
        shutil.copytree(tmp / "req", d)
        tl = jsonl(d / "tokens" / f"{KEYS[1]}.jsonl"); tl[0]["token_ids"][0] += 1; write_jsonl(d / "tokens" / f"{KEYS[1]}.jsonl", tl)
        rehash(d, [f"tokens/{KEYS[1]}.jsonl"])
        cases.append(("an edited token ID of one model", P.verify_compare_requests(d)))
        for label, probs in cases:
            check(f"the verifier rejects {label}", bool(probs))
        bad = dict(pol); bad["population"] = dict(pol["population"], excluded=dict(pol["population"]["excluded"], selected_sha256="0" * 64))
        bp = tmp / "bad-policy.json"; bp.write_text(json.dumps(bad), encoding="utf-8")
        r2 = outcome(lambda: P.prepare_compare(bundle=bundle, tokenizer_small=None, tokenizer_large=None, out=tmp / "x",
                                               policy=bp, tokenizers=toks))
        check("a pilot list that does not match its pinned hash stops preparation", r2 == ("input_error", ["E_COMPARE_POPULATION"]), str(r2))
        r3 = outcome(lambda: P.prepare_compare(bundle=bundle, tokenizer_small=None, tokenizer_large=None, out=tmp / "req",
                                               policy=pp, tokenizers=toks))
        check("an existing destination is refused", r3 == ("input_error", ["E_EVAL_OUTPUT_EXISTS"]), str(r3))
        with SB.Watch([ann]) as w:
            r4 = outcome(lambda: R.run_compare_rules(requests=tmp / "req", bundle=bundle, relation_config=REL_CFG,
                                                     direction_config=DIR_CFG, out=tmp / "rules", sample=False))
        check("rules complete on the same subscenes with the annotations inaccessible", r4[0] == "ok" and not w.denied, str(r4)[:200])
        if r4[0] == "ok":
            rules = jsonl(tmp / "rules" / "rules.jsonl")
            by = {(x["parent_command_id"], x["view_id"]): x for x in rules}
            check("one rules record per parent and view, naming both formats' requests, readback clean",
                  len(rules) == len(pairs) and R.verify_compare_rules(tmp / "rules", tmp / "req") == [])
            check("each rules record saw exactly the objects the models are offered",
                  all(by[(x["parent_command_id"], x["view_id"])]["object_ids"] == x["object_ids"]
                      and sorted(m[1] for m in x["mapping"][:-1]) == sorted(x["object_ids"]) for x in rows))
    counts = P.request_counts([{"parent_command_id": "a", "view_id": "v", "object_count": 3,
                                "models": {KEYS[0]: {"context_status": "within_context_limit"},
                                           KEYS[1]: {"context_status": "context_budget_exceeded"}}}])
    check("context exclusions are counted per model", counts["context_budget_exceeded"] == {KEYS[0]: 0, KEYS[1]: 1})


class Interrupting:
    """Labelled double: the fake model, interrupted (as by Ctrl+C) after a number of forwards, or NaN on one prompt."""

    def __init__(self, base, stop_after=None, nan_ids=None, dtype=None):
        self.base, self.stop_after, self.nan_ids, self.dtype, self.calls = base, stop_after, nan_ids, dtype, 0

    def info(self):
        i = self.base.info()
        if self.dtype:
            i = dict(i, dtype=self.dtype)
        return i

    def forward_last(self, ids):
        self.calls += 1
        if self.stop_after is not None and self.calls > self.stop_after:
            raise KeyboardInterrupt("simulated interruption")
        row = self.base.forward_last(ids)
        if self.nan_ids is not None and list(ids) == list(self.nan_ids):
            row = [float("nan")] * len(row)
        return row

    def independent_last(self, ids):
        return self.base.independent_last(ids)


def check_runs(D, P, RUN):
    print("-- smoke check and resumable runs (labelled fake model, injected checkpoint tie)")
    SB = importlib.import_module("grounding.tests.test_iref_vla_pilot_scoring")
    A = SB.pilot_helpers()
    PI = importlib.import_module("grounding.inference.iref_vla.prepare")
    ev = lambda *a: {"revision": "fixture", "tie": "labelled fixture", "files": {}}  # noqa: E731
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle = SB.scoring_bundle(tmp, A)
        src = PI.read_bundle(bundle)
        eligible = PI.eligible_parents(src)
        pilot = sorted(eligible, key=lambda p: (hs("fixture-pilot\n" + p), p))[:1]
        pol = json.loads(D.POLICY_PATH.read_text(encoding="utf-8"))
        pol["population"] = {"rule": "fixture", "expected_eligible": len(eligible), "expected_remaining": len(eligible) - 1,
                             "excluded": {"what": "fixture", "salt": "fixture-pilot\n", "count": 1,
                                          "selected_sha256": hashlib.sha256((pilot[0] + "\n").encode()).hexdigest()}}
        pol["selection"]["count"] = len(eligible) - 1
        pol["expected"] = {"parents": len(eligible) - 1, "parent_views": 2 * (len(eligible) - 1), "requests": 4 * (len(eligible) - 1)}
        pp = tmp / "policy.json"
        pp.write_text(json.dumps(pol), encoding="utf-8")
        P.prepare_compare(bundle=bundle, tokenizer_small=None, tokenizer_large=None, out=tmp / "req", policy=pp,
                          tokenizers={k: A.OffsetCharTokenizer() for k in KEYS})
        rows = jsonl(tmp / "req" / "request-index.jsonl")
        n = len(rows)
        common = dict(requests=tmp / "req", model_dir=tmp, tokenizer_dir=None, device="cpu", evidence_fn=ev)
        for key in KEYS:
            fake = A.FakeModel()
            r = outcome(lambda: RUN.smoke(**common, model_key=key, out=tmp / f"smoke-{key}.json",
                                          model_loader=A.loader_for(fake), tokenizer=A.OffsetCharTokenizer()))
            check(f"{key}: the smoke check passes its canaries and writes an accepted record", r[0] == "ok" and r[1]["accepted"], str(r)[:200])
            fake = A.FakeModel()
            r = outcome(lambda: RUN.run_compare(**common, model_key=key, smoke_record=tmp / f"smoke-{key}.json",
                                                out=tmp / f"run-{key}", model_loader=A.loader_for(fake),
                                                tokenizer=A.OffsetCharTokenizer(), progress=lambda m: None))
            ok = r[0] == "ok" and r[1]["counts"]["completed"] == n
            check(f"{key}: the run completes all {n} requests and reads back against the bundle",
                  ok and RUN.verify_compare_results(tmp / f"run-{key}", tmp / "req") == []
                  and not (tmp / f"run-{key}.partial").exists(), str(r)[:200])
        check("every forward is fresh: canaries plus one forward per request, nothing reused",
              fake.forward_calls == n + len(jsonl(tmp / f"run-{KEYS[1]}" / "sessions.jsonl")[0]["canaries"]))
        r = outcome(lambda: RUN.run_compare(**common, model_key=KEYS[0], smoke_record=tmp / f"smoke-{KEYS[1]}.json",
                                            out=tmp / "x1", model_loader=A.loader_for(A.FakeModel()),
                                            tokenizer=A.OffsetCharTokenizer(), progress=lambda m: None))
        check("a smoke record of the other model is refused before loading", r == ("input_error", ["E_COMPARE_SMOKE"]), str(r))
        stop = Interrupting(A.FakeModel(), stop_after=5)
        r = outcome(lambda: RUN.run_compare(**common, model_key=KEYS[0], smoke_record=tmp / f"smoke-{KEYS[0]}.json",
                                            out=tmp / "res", model_loader=A.loader_for(stop),
                                            tokenizer=A.OffsetCharTokenizer(), progress=lambda m: None))
        kept = jsonl(tmp / "res.partial" / "rows.jsonl") if (tmp / "res.partial" / "rows.jsonl").exists() else []
        check("an interruption keeps the rows already written and publishes nothing", r[0] == "crashed" and "KeyboardInterrupt" in r[1]
              and len(kept) == 5 - 2 and not (tmp / "res").exists(), f"{r} {len(kept)}")
        with open(tmp / "res.partial" / "rows.jsonl", "ab") as fh:
            fh.write(b'{"request_id": "r00')
        r = outcome(lambda: RUN.run_compare(**common, model_key=KEYS[0], smoke_record=tmp / f"smoke-{KEYS[0]}.json",
                                            out=tmp / "res", model_loader=A.loader_for(A.FakeModel()),
                                            tokenizer=A.OffsetCharTokenizer(), progress=lambda m: None))
        check("without --resume an existing partial run is refused", r == ("input_error", ["E_COMPARE_RESUME"]), str(r))
        other = tmp / "smoke-again.json"
        RUN.smoke(**common, model_key=KEYS[0], out=other, model_loader=A.loader_for(A.FakeModel()), tokenizer=A.OffsetCharTokenizer())
        r = outcome(lambda: RUN.run_compare(**common, model_key=KEYS[0], smoke_record=other, out=tmp / "res", resume=True,
                                            model_loader=A.loader_for(A.FakeModel()), tokenizer=A.OffsetCharTokenizer(),
                                            progress=lambda m: None))
        check("resuming with a different frozen configuration (another smoke record) is refused",
              r == ("input_error", ["E_COMPARE_RESUME"]), str(r))
        r = outcome(lambda: RUN.run_compare(**common, model_key=KEYS[0], smoke_record=tmp / f"smoke-{KEYS[0]}.json",
                                            out=tmp / "res", resume=True, model_loader=A.loader_for(A.FakeModel()),
                                            tokenizer=A.OffsetCharTokenizer(), progress=lambda m: None))
        a, b = jsonl(tmp / "res" / "results.jsonl") if r[0] == "ok" else [], jsonl(tmp / f"run-{KEYS[0]}" / "results.jsonl")
        check("--resume drops the torn last line, finishes, and gives the uninterrupted run's choices in two sessions",
              r[0] == "ok" and [x["choice_code"] for x in a] == [x["choice_code"] for x in b] and r[1]["counts"]["sessions"] == 2,
              str(r)[:200])
        nan_ids = jsonl(tmp / "req" / "tokens" / f"{KEYS[0]}.jsonl")[2]["token_ids"]
        r = outcome(lambda: RUN.run_compare(**common, model_key=KEYS[0], smoke_record=tmp / f"smoke-{KEYS[0]}.json",
                                            out=tmp / "nan", model_loader=A.loader_for(Interrupting(A.FakeModel(), nan_ids=nan_ids)),
                                            tokenizer=A.OffsetCharTokenizer(), progress=lambda m: None))
        res = jsonl(tmp / "nan" / "results.jsonl") if r[0] == "ok" else []
        check("a non-finite output is kept as one execution_failed row; the others complete",
              r[0] == "ok" and r[1]["counts"]["execution_failed"] == 1 and res[2]["technical_status"] == "execution_failed"
              and res[2]["choice_code"] is None and RUN.verify_compare_results(tmp / "nan", tmp / "req") == [], str(r)[:200])
        r = outcome(lambda: RUN.run_compare(**common, model_key=KEYS[0], smoke_record=tmp / f"smoke-{KEYS[0]}.json",
                                            out=tmp / "dt", model_loader=A.loader_for(Interrupting(A.FakeModel(), dtype="torch.bfloat16")),
                                            tokenizer=A.OffsetCharTokenizer(), progress=lambda m: None))
        check("a model loaded with other settings than the smoke record's stops the run", r[0] == "crashed" and "settings" in r[1], str(r))
        r = outcome(lambda: RUN.run_compare(**dict(common, device="cpu"), model_key=KEYS[0], smoke_record=tmp / f"smoke-{KEYS[0]}.json",
                                            out=tmp / "cpu", tokenizer=A.OffsetCharTokenizer(), progress=lambda m: None))
        check("a CPU run without an injected test loader is refused", r == ("input_error", ["E_COMPARE_DEVICE"]), str(r))


def check_scoring(D, P, R, RUN, S):
    print("-- scoring against fixture annotations (labelled fake runs)")
    SB = importlib.import_module("grounding.tests.test_iref_vla_pilot_scoring")
    A = SB.pilot_helpers()
    PI = importlib.import_module("grounding.inference.iref_vla.prepare")
    ev = lambda *a: {"revision": "fixture", "tie": "labelled fixture", "files": {}}  # noqa: E731
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle = SB.scoring_bundle(tmp, A)
        ann = SB.annotation_bundle(tmp / "annotations.json", SB.TARGETS)
        src = PI.read_bundle(bundle)
        eligible = PI.eligible_parents(src)
        pilot = sorted(eligible, key=lambda p: (hs("fixture-pilot\n" + p), p))[:1]
        pol = json.loads(D.POLICY_PATH.read_text(encoding="utf-8"))
        pol["population"] = {"rule": "fixture", "expected_eligible": len(eligible), "expected_remaining": len(eligible) - 1,
                             "excluded": {"what": "fixture", "salt": "fixture-pilot\n", "count": 1,
                                          "selected_sha256": hashlib.sha256((pilot[0] + "\n").encode()).hexdigest()}}
        pol["selection"]["count"] = len(eligible) - 1
        pol["expected"] = {"parents": len(eligible) - 1, "parent_views": 2 * (len(eligible) - 1), "requests": 4 * (len(eligible) - 1)}
        pp = tmp / "policy.json"
        pp.write_text(json.dumps(pol), encoding="utf-8")
        P.prepare_compare(bundle=bundle, tokenizer_small=None, tokenizer_large=None, out=tmp / "req", policy=pp,
                          tokenizers={k: A.OffsetCharTokenizer() for k in KEYS})
        R.run_compare_rules(requests=tmp / "req", bundle=bundle, relation_config=REL_CFG, direction_config=DIR_CFG,
                            out=tmp / "rules", sample=False)
        common = dict(requests=tmp / "req", model_dir=tmp, tokenizer_dir=None, device="cpu", evidence_fn=ev)
        for key in KEYS:
            RUN.smoke(**common, model_key=key, out=tmp / f"smoke-{key}.json", model_loader=A.loader_for(A.FakeModel()),
                      tokenizer=A.OffsetCharTokenizer())
            RUN.run_compare(**common, model_key=key, smoke_record=tmp / f"smoke-{key}.json", out=tmp / f"run-{key}",
                            model_loader=A.loader_for(A.FakeModel()), tokenizer=A.OffsetCharTokenizer(), progress=lambda m: None)
        args = dict(requests=tmp / "req", rules=tmp / "rules", small_run=tmp / f"run-{KEYS[0]}", large_run=tmp / f"run-{KEYS[1]}",
                    bundle=bundle, annotations=ann, sample=False, fixture_scene_id="fixture.iref_eval.f1")
        r = outcome(lambda: S.score_compare(**args, out=tmp / "scores"))
        check("scoring completes on the fixture and reads back", r[0] == "ok" and S.verify_compare_scores(tmp / "scores") == [], str(r)[:300])
        if r[0] != "ok":
            return
        rows = jsonl(tmp / "scores" / "scores.jsonl")
        req = {x["request_id"]: x for x in jsonl(tmp / "req" / "request-index.jsonl")}
        check("one score row per request, targets inherited from the annotations", len(rows) == len(req)
              and all(x["source_target_id"] == SB.TARGETS[x["parent_command_id"]] for x in rows))
        ok_ref = all(x["reference_always_B"] == (dict((m[1], m[0]) for m in req[x["request_id"]]["mapping"][:-1])[x["source_target_id"]] == "B")
                     and x["reference_second_position"] == (req[x["request_id"]]["mapping"][1][1] == x["source_target_id"]) for x in rows)
        check("the always-B and second-position references follow each request's own mapping (independent recomputation)", ok_ref)
        res = {k: {x["request_id"]: x for x in jsonl(tmp / f"run-{k}" / "results.jsonl")} for k in KEYS}
        check("each model's outcome is correct exactly when its chosen object is the target; ASK is never correct",
              all((x["models"][k]["outcome"] == "correct") == (res[k][x["request_id"]]["choice_object_id"] == x["source_target_id"])
                  and (res[k][x["request_id"]]["model_choice"] != "model_choice_ask" or x["models"][k]["outcome"] == "ask")
                  for x in rows for k in KEYS))
        s = r[1]
        pr = s["by_view"]["full_inventory"]["by_format"]["coordinates_v2"]["pairs"]["small_vs_large"]
        check("paired counts add up to the common requests", pr["both"] + pr["first_only"] + pr["second_only"] + pr["neither"] == pr["common"])
        check("the failure sample holds at most six per model and view, all wrong or ASK",
              all(f["outcome"] in ("wrong_object", "ask") for f in s["failure_sample"])
              and max(Counter((f["model_key"], f["view_id"]) for f in s["failure_sample"]).values(), default=0) <= 6)
        r2 = outcome(lambda: S.score_compare(**dict(args, small_run=tmp / f"run-{KEYS[1]}", large_run=tmp / f"run-{KEYS[0]}"), out=tmp / "x"))
        check("swapped model runs are refused", r2 == ("input_error", ["E_COMPARE_INPUT"]), str(r2))
        bad = tmp / "scores-copy"
        shutil.copytree(tmp / "scores", bad)
        rr = jsonl(bad / "scores.jsonl"); rr[0]["models"][KEYS[0]]["outcome"] = "correct" if rr[0]["models"][KEYS[0]]["outcome"] != "correct" else "ask"
        write_jsonl(bad / "scores.jsonl", rr)
        m = json.loads((bad / "manifest.json").read_text()); m["outputs"]["scores.jsonl"] = hashlib.sha256((bad / "scores.jsonl").read_bytes()).hexdigest()
        (bad / "manifest.json").write_text(json.dumps(m))
        check("an edited outcome is caught by readback even when re-hashed", S.verify_compare_scores(bad) != [])


def check_audit(D, P, AU):
    print("-- the A2.3e input audit (labelled tokenizer doubles)")
    SB = importlib.import_module("grounding.tests.test_iref_vla_pilot_scoring")
    A = SB.pilot_helpers()
    PI = importlib.import_module("grounding.inference.iref_vla.prepare")
    RUNM = importlib.import_module("grounding.inference.iref_vla_compare.run")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle = SB.scoring_bundle(tmp, A)
        ann = SB.annotation_bundle(tmp / "annotations.json", SB.TARGETS)
        src = PI.read_bundle(bundle)
        eligible = PI.eligible_parents(src)
        pilot = sorted(eligible, key=lambda p: (hs("fixture-pilot\n" + p), p))[:1]
        pol = json.loads(D.POLICY_PATH.read_text(encoding="utf-8"))
        pol["population"] = {"rule": "fixture", "expected_eligible": len(eligible), "expected_remaining": len(eligible) - 1,
                             "excluded": {"what": "fixture", "salt": "fixture-pilot\n", "count": 1,
                                          "selected_sha256": hashlib.sha256((pilot[0] + "\n").encode()).hexdigest()}}
        pol["selection"]["count"] = len(eligible) - 1
        pol["expected"] = {"parents": len(eligible) - 1, "parent_views": 2 * (len(eligible) - 1), "requests": 4 * (len(eligible) - 1)}
        pp = tmp / "policy.json"
        pp.write_text(json.dumps(pol), encoding="utf-8")
        toks = {k: A.OffsetCharTokenizer() for k in KEYS}
        P.prepare_compare(bundle=bundle, tokenizer_small=None, tokenizer_large=None, out=tmp / "req", policy=pp, tokenizers=toks)
        rows = jsonl(tmp / "req" / "request-index.jsonl")
        with SB.Watch([ann]) as w:
            r = outcome(lambda: AU.run_audit(requests=tmp / "req", bundle=bundle, tokenizer_small=None, tokenizer_large=None,
                                             out=tmp / "input-audit", tokenizers=toks))
        check("the audit completes with every check passing and the annotations inaccessible",
              r[0] == "ok" and r[1]["passed"] and not w.denied, str(r)[:300])
        if r[0] != "ok":
            return
        longest = {k: max(rows, key=lambda x: (x["models"][k]["input_tokens"], x["request_index"]))["request_id"] for k in KEYS}
        want = [rows[0]["parent_command_id"]] + [p for p in dict.fromkeys(x["parent_command_id"] for x in rows if x["request_id"] in longest.values())
                                                 if p != rows[0]["parent_command_id"]]
        check("the audited parents are those of the runs' canaries (first request; longest, ties to the later index)",
              r[1]["parents"] == want and len(r[1]["requests"]) == 4 * len(want))
        a = json.loads((tmp / "input-audit" / "requests" / r[1]["requests"][0] / "audit.json").read_text())
        prompt = (tmp / "input-audit" / "requests" / r[1]["requests"][0] / "prompt.txt").read_bytes().decode("utf-8")
        check("the exported prompt and spans rebuild the turns: system message, choices line, document, open assistant turn",
              prompt[a["char_spans"]["system_message"][0]:a["char_spans"]["system_message"][1]] in prompt
              and prompt[a["char_spans"]["document"][0]:].endswith("<|im_end|>\n<|im_start|>assistant\n")
              and prompt[a["char_spans"]["choices_line"][0]:a["char_spans"]["choices_line"][1]].startswith('{"choices":'))
        leak = AU.audit_request(rows[0], prompt.replace('"choices"', '"choices","target_id":"x"', 1),
                                {k: [] for k in KEYS}, toks, importlib.import_module("grounding.inference.iref_vla.protocol").load_protocol(tmp / "req" / "protocol.json"),
                                (bundle / rows[0]["source_document_path"]).read_bytes(), {"text": "x"},
                                importlib.import_module("grounding.inference.iref_vla_compare.design").load_policy(tmp / "req" / "policy.json"))
        failed = {c["check"] for c in leak["checks"] if not c["passed"]}
        check("a prompt carrying a target field fails the rebuild and leak checks",
              "no annotation ID, target field or rules outcome appears in the prompt" in failed
              and "the prompt is exactly the protocol's system turn, the user turn and an open assistant turn" in failed)


def main() -> int:
    D = importlib.import_module("grounding.inference.iref_vla_compare.design")
    P = importlib.import_module("grounding.inference.iref_vla_compare.prepare")
    R = importlib.import_module("grounding.inference.iref_vla_compare.rules")
    RUN = importlib.import_module("grounding.inference.iref_vla_compare.run")
    S = importlib.import_module("grounding.inference.iref_vla_compare.score")
    AU = importlib.import_module("grounding.inference.iref_vla_compare.audit")
    check_design(D)
    check_chain(D, P, R)
    check_runs(D, P, RUN)
    check_scoring(D, P, R, RUN, S)
    check_audit(D, P, AU)
    print(f"\n{len(PASSES)} passed, {len(FAILS)} failed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
