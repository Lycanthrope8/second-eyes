"""A2.6c: baseline execution and the frozen analysis. Run directly:
python grounding/tests/test_iref_vla_baseline.py --samples DIR --official DIR

The analysis checks need no files; they are hand-calculable. The rest builds a small real A2.6b preparation from the
pinned samples (test_iref_vla_dev's fixture release: the real chain and the pinned tokenizer). The models run through
A2.3d's accepted runner with its labelled doubles (A2.3a/b's fake model, an injected checkpoint tie), and the desktop
build is a labelled fake native. The frozen hashes are the fixture's own, labelled; the command line enforces the real
ones. Without the samples or the lists, the fixture checks fail rather than skip.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import json
import math
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
FAILS, PASSES = [], []


def check(name, ok, detail=""):
    (PASSES if ok else FAILS).append(name)
    print(("  ok    " if ok else "  FAIL  ") + name + ("" if ok or not detail else f"  [{detail}]"))


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def analysis_checks(AN):
    print("-- the frozen analysis (hand-calculable fixtures)")
    obs = {("e1", "a"): ["correct", "wrong_object"], ("e1", "b"): ["correct"], ("e2", "a"): ["ask"]}
    strata = {("e1", "a"): (4, 2), ("e1", "b"): (2, 1), ("e2", "a"): (1, 1)}
    agg = AN.aggregates(obs, strata)
    e = AN.estimate(agg)
    check("command-weighted 4/7 (N/n per stratum), environment-weighted 1/3 (2/3 and 0), unweighted 1/2",
          math.isclose(e["command_weighted"], 4 / 7) and math.isclose(e["environment_weighted"], 1 / 3) and e["unweighted"] == 0.5, str(e))
    r = AN.estimate(agg, {"e1": 2})
    check("an environment drawn twice counts twice: 8/12 command-weighted, 2/3 environment-weighted, 4/6 unweighted",
          all(math.isclose(v, 2 / 3) for v in r.values()), str(r))
    big = AN.aggregates({("A", "a"): ["correct", "correct"], ("B", "a"): ["wrong_object"]}, {("A", "a"): (10, 2), ("B", "a"): (1, 1)})
    u = AN.estimate(big)
    check("unequal environment sizes: command-weighted 10/11 follows the commands, environment-weighted 1/2 the environments",
          math.isclose(u["command_weighted"], 10 / 11) and math.isclose(u["environment_weighted"], 0.5), str(u))
    f = AN.aggregates({("A", "a"): ["correct", "execution_failed"], ("A", "b"): ["context_budget_exceeded"]},
                      {("A", "a"): (2, 2), ("A", "b"): (1, 1)})
    rates = AN.categories(f)
    check("a failure stays in the denominator (1/3 correct) and every outcome keeps its own rate, summing to 1",
          math.isclose(AN.estimate(f)["command_weighted"], 1 / 3) and math.isclose(sum(rates.values()), 1.0)
          and math.isclose(rates["execution_failed"], 1 / 3) and math.isclose(rates["context_budget_exceeded"], 1 / 3))
    d1, d2 = list(AN.draws(["x", "y", "z"], b=50, seed=AN.SEED)), list(AN.draws(["z", "x", "y"], b=50, seed=AN.SEED))
    check("the draws are fixed by the seed over the sorted environments; each replicate draws E with multiplicity",
          d1 == d2 and all(sum(c.values()) == 3 for c in d1) and any(max(c.values()) > 1 for c in d1))
    check("percentile intervals interpolate linearly between order statistics (Hyndman-Fan type 7)",
          AN.quantile([1, 2, 3, 4], 0.5) == 2.5 and math.isclose(AN.quantile([1, 2, 3, 4], 0.025), 1.075))
    reps_a, reps_b = [], []
    agg2 = AN.aggregates({("A", "a"): ["correct", "wrong_object"], ("B", "a"): ["correct"], ("C", "a"): ["ask", "correct"]},
                         {("A", "a"): (5, 2), ("B", "a"): (3, 1), ("C", "a"): (2, 2)})
    for cnt in AN.draws(["A", "B", "C"], b=500, seed=AN.SEED):
        reps_a.append(AN.estimate(agg2, cnt)["command_weighted"])
        reps_b.append(AN.estimate(agg2, cnt)["command_weighted"])
    ci = AN.interval([x - y for x, y in zip(reps_a, reps_b)])
    spread = AN.interval(reps_a)
    check("paired: two identical systems on the same draws differ by exactly zero in every replicate, though each varies",
          ci["low"] == 0 and ci["high"] == 0 and spread["high"] > spread["low"])


def prepared_fixture(tmp: Path, samples: Path, official: Path):
    """A small real A2.6b preparation: the dev suite's fixture release, inventory, freeze and prepare."""
    TD = importlib.import_module("grounding.tests.test_iref_vla_dev")
    R = importlib.import_module("grounding.adapters.iref_vla.release")
    D = importlib.import_module("grounding.preparation.iref_vla_dev.dev")
    TD.build_release(samples, tmp / "release")
    vocab = samples / "data" / "NYU_Object_Classes.csv"
    R.write_inventory(R.build_inventory([("Scannet", tmp / "release" / "Scannet")], R.read_vocabulary(vocab.read_bytes())), tmp / "inventory")
    with zipfile.ZipFile(tmp / "Scannet.zip", "w") as z:
        for f in sorted((tmp / "release").rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(tmp / "release").as_posix())
    spec = D.freeze(inventory_dir=tmp / "inventory", official_dir=official, out=tmp / "spec")
    D.prepare(spec=Path(spec["folder"]) / "spec.json", zip_path=tmp / "Scannet.zip", inventory_dir=tmp / "inventory",
              vocabulary=vocab, out=tmp / "prep", work=tmp / "work", progress=lambda m: None)
    prep = tmp / "prep"
    frozen = {"spec_sha256": sha((prep / "spec.json").read_bytes()), "requests_sha256": sha((prep / "requests.jsonl").read_bytes())}
    return prep, tmp / "work", frozen


def tampered(prep: Path, dst: Path, change) -> dict:
    """A copy of the preparation with one request changed, the manifest made consistent with it, and its frozen hashes."""
    shutil.copytree(prep, dst)
    rows = [json.loads(x) for x in (dst / "requests.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    change(rows[0])
    body = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    (dst / "requests.jsonl").write_bytes(body)
    idx = [json.loads(x) for x in (dst / "requests-index.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    idx[0] = {k: rows[0].get(k) for k in idx[0]}
    (dst / "requests-index.jsonl").write_bytes("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in idx).encode())
    man = json.loads((dst / "manifest.json").read_text(encoding="utf-8"))
    man["files"]["requests.jsonl"] = man["hashes"]["requests"] = sha(body)
    man["files"]["requests-index.jsonl"] = sha((dst / "requests-index.jsonl").read_bytes())
    (dst / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    return {"spec_sha256": sha((dst / "spec.json").read_bytes()), "requests_sha256": sha(body)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=os.environ.get("SECOND_EYES_IREF_RELEASE_SAMPLES"))
    ap.add_argument("--official", default=os.environ.get("SECOND_EYES_IREF_OFFICIAL"))
    a = ap.parse_args()
    AN = importlib.import_module("grounding.inference.iref_vla_dev.analysis")
    analysis_checks(AN)
    R = importlib.import_module("grounding.adapters.iref_vla.release")
    have = bool(a.samples and a.official) and not R.verify_samples(a.samples)
    check("the pinned samples and the pinned ScanNet official lists are available", have, "pass --samples DIR and --official DIR")
    if not have:
        print(f"\n{len(PASSES)} passed, {len(FAILS)} failed")
        return 1
    PF = importlib.import_module("grounding.inference.iref_vla_dev.preflight")
    RN = importlib.import_module("grounding.inference.iref_vla_dev.runs")
    DK = importlib.import_module("grounding.inference.iref_vla_dev.desktop")
    SC = importlib.import_module("grounding.inference.iref_vla_dev.score")
    T = importlib.import_module("grounding.preparation.iref_vla.tokens")
    A = importlib.import_module("grounding.tests.test_iref_vla_pilot_scoring").pilot_helpers()
    TC = importlib.import_module("grounding.tests.test_iref_vla_compare")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        prep, work, frozen = prepared_fixture(tmp, Path(a.samples), Path(a.official))
        tok = T.load_pinned_tokenizer(REPO / "quest-app" / "Assets" / "SecondEyes" / "Models" / "qwen2.5-0.5b-instruct")
        toks = {k: tok for k in PF.MODEL_KEYS}   # labelled: the 0.5B's pinned tokenizer stands in for the 7B's here
        quiet = lambda m: None  # noqa: E731
        print("-- the preflight (read-only)")
        pre = PF.preflight(prep=prep, work=work, tokenizers=toks, frozen=frozen, progress=quiet)
        rc = pre["receipt"]
        check("the frozen requests pass: four per parent, every model's tokens and boundaries, the receipt complete",
              rc["passed"] and rc["counts"]["requests"] == 4 * rc["counts"]["parents"]
              and all(v["same_token_ids_as_prepared"] == rc["counts"]["requests"] for v in rc["models"].values()), str(rc["counts"]))
        try:
            PF.preflight(prep=prep, work=work, tokenizers=toks, progress=quiet)
            ok = False
        except PF.PreflightError as e:
            ok = "frozen" in str(e)
        check("the real frozen hashes are enforced: this fixture's request file is refused", ok)
        for label, change, want in (
                ("a document changed (bundle and hash checks)", lambda r: r.update(document=r["document"] + " "), "document"),
                ("a token changed (token hash and tokenizer checks)", lambda r: r.update(token_ids=r["token_ids"][:-1] + [r["token_ids"][-1] + 1]), "tokens"),
                ("a code swapped (the recomputed D104 mapping)", lambda r: r.update(codes=[r["codes"][1], r["codes"][0]] + r["codes"][2:]), "mapping")):
            dst = tmp / f"tampered-{want}"
            fz = tampered(prep, dst, change)
            try:
                PF.preflight(prep=dst, work=work, tokenizers=toks, frozen=fz, progress=quiet)
                ok, msg = False, "accepted"
            except PF.PreflightError as e:
                ok, msg = any(i["message"].startswith(want) for i in e.issues) and hasattr(e, "receipt"), str(e)[:200]
            check(f"refused with diagnostics, the manifest made consistent: {label}", ok, msg)
        pdir = PF.write_preflight(pre, tmp / "preflight")
        tokdir = REPO / "quest-app" / "Assets" / "SecondEyes" / "Models" / "qwen2.5-0.5b-instruct"
        par = PF.preflight(prep=prep, work=work, tokenizer_dirs={PF.MODEL_KEYS[0]: tokdir}, workers=2, frozen=frozen, progress=quiet)
        check("the preflight in two spawned worker processes gives the same token rows and bundle bindings as in one",
              par["tokens"][PF.MODEL_KEYS[0]] == pre["tokens"][PF.MODEL_KEYS[0]]
              and par["receipt"]["binding"] == pre["receipt"]["binding"] and par["receipt"]["workers"] == 2)

        class Other:
            """Labelled double: a tokenizer that claims another identity, or this identity with altered token IDs."""
            def __init__(self, base, identity=None, shift=False):
                self.base, self.identity, self.shift = base, identity or base.identity, shift
            def encode(self, text):
                ids = list(self.base.encode(text))
                return ids + [0] if self.shift else ids
        refused = []
        bent = tmp / "preflight-other-code"
        shutil.copytree(pdir, bent)
        rj = json.loads((bent / "receipt.json").read_text(encoding="utf-8"))
        rj["binding"]["code"]["preflight.py"] = "0" * 64
        (bent / "receipt.json").write_text(json.dumps(rj), encoding="utf-8")
        for label, pd, tk in (("code", bent, tok), ("identity", pdir, Other(tok, identity="another tokenizer")),
                              ("tokens", pdir, Other(tok, shift=True))):
            try:
                PF.bound_context(preflight_dir=pd, prep=prep, work=work, model_key=PF.MODEL_KEYS[0], model_dir=tmp,
                                 tokenizer=tk, frozen=frozen, evidence_fn=lambda *x: {"revision": "fixture", "tie": "labelled", "files": {}})
            except PF.PreflightError as e:
                refused.append((label, "E_A26C_BINDING" in str(e)))
        check("a step refuses a receipt from other code, a tokenizer of another identity, and token IDs that differ from "
              "the verified ones", refused == [("code", True), ("identity", True), ("tokens", True)], str(refused))
        noref = tmp / "prep-without-answers"
        shutil.copytree(prep, noref)
        shutil.rmtree(noref / "reference_only")
        ev = lambda *x: {"revision": "fixture", "tie": "labelled fixture", "files": {}}  # noqa: E731
        print("-- the models through A2.3d's accepted runner (labelled fake model)")
        out = tmp / "runs"
        out.mkdir()
        common = dict(prep=noref, work=work, preflight_dir=pdir, model_dir=out, device="cpu", tokenizers=toks, evidence_fn=ev,
                      frozen=frozen, progress=quiet)
        sm = RN.smoke(**common, model_key=PF.MODEL_KEYS[0], out=out / "smoke-small.json", model_loader=A.loader_for(A.FakeModel()))
        check("without reference_only, a step bound to the preflight runs A2.3d's smoke: canaries accepted, settings recorded, "
              "the binding record beside it", sm["accepted"] and sm["canaries"] and (out / "smoke-small.json.binding.json").is_file())
        stop = TC.Interrupting(A.FakeModel(), stop_after=5)
        try:
            RN.run(**common, model_key=PF.MODEL_KEYS[0], smoke_record=out / "smoke-small.json", out=out / "small",
                   model_loader=A.loader_for(stop))
            interrupted = False
        except KeyboardInterrupt:
            interrupted = (out / "small.partial" / "rows.jsonl").is_file()
        s = RN.run(**common, model_key=PF.MODEL_KEYS[0], smoke_record=out / "smoke-small.json", out=out / "small", resume=True,
                   model_loader=A.loader_for(A.FakeModel()))
        check("an interrupted run keeps its partial rows and resumes under the same lock to exact coverage",
              interrupted and s["counts"]["completed"] == s["counts"]["requests"] == rc["counts"]["requests"]
              and RN.verify_results(out / "small", prep) == [])
        allreq = [json.loads(x) for x in (prep / "requests.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        longest = max(allreq, key=lambda r: (len(r["token_ids"]), -allreq.index(r)))
        first = next(r for r in allreq[1:] if r["request_id"] != longest["request_id"])   # neither canary (first, longest)
        RN.smoke(**common, model_key=PF.MODEL_KEYS[1], out=out / "smoke-large.json", model_loader=A.loader_for(A.FakeModel()))
        s7 = RN.run(**common, model_key=PF.MODEL_KEYS[1], smoke_record=out / "smoke-large.json", out=out / "large",
                    model_loader=A.loader_for(TC.Interrupting(A.FakeModel(), nan_ids=first["token_ids"])))
        rows7 = [json.loads(x) for x in (out / "large" / "results.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        failed = [r for r in rows7 if r["technical_status"] == "execution_failed"]
        check("a forward that fails stays visible as execution_failed, in exact coverage, and the run says so",
              len(rows7) == rc["counts"]["requests"] and len(failed) == 1 and failed[0]["request_id"] == first["request_id"]
              and s7["counts"]["execution_failed"] == 1)
        print("-- the rules, the desktop comparison, scoring and the analysis")
        rs = RN.run_rules(prep=noref, work=work, out=tmp / "rules", frozen=frozen)
        rr = [json.loads(x) for x in (tmp / "rules" / "rules.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        check("rules without reference_only: one result per parent and view, both formats' request IDs, one observation each",
              len(rr) == 2 * rc["counts"]["parents"] and len({(r["parent_command_id"], r["view_id"]) for r in rr}) == len(rr)
              and all(set(r["request_ids"]) == set(PF.FORMATS) and r["observations"] == 1 for r in rr))
        DK.export(prep=prep, out=tmp / "desk-export", frozen=frozen)
        recs = [json.loads(x) for x in (tmp / "desk-export" / "desktop-requests.jsonl").read_text(encoding="utf-8").splitlines() if x]
        f32 = {json.loads(x)["request_id"]: json.loads(x) for x in (out / "small" / "results.jsonl").read_text(encoding="utf-8").splitlines() if x}

        class FakeDesktopNative:
            """Labelled double: the exported prompts' frozen token IDs; the 0.5B float32 run's offered logits."""
            def __init__(self):
                self.tok = {base64.b64decode(r["prompt_b64"]): r["token_ids"] for r in recs}
                self.log = {tuple(r["token_ids"]): {x["token_id"]: x["logit"] for x in f32[r["request_id"]]["scores"]} for r in recs}
                self.cache = []
            def version(self): return "fake"
            def system_info(self): return "fake"
            def last_error(self): return ""
            def set_verbose(self, on): pass
            def memory_kb(self, peak): return 1000
            def load(self, path, n_ctx, threads, n_seq, flags): return True
            def info(self): return [8192, 151936, 16, 0]
            def tokenize(self, data): return list(self.tok[data])
            def eval(self, ids, keep):
                self.cache = self.cache[:keep] + list(ids)
                return 0
            def n_cached(self): return len(self.cache)
            def logprob(self, t): return -1.0
            def logits(self, n):
                row = [-30.0] * n
                for t, x in self.log[tuple(self.cache)].items():
                    row[t] = x
                return row
            def free(self): pass
        model = tmp / "fake.gguf"
        model.write_bytes(b"labelled fake model")
        done = DK.run(export_dir=tmp / "desk-export", out=tmp / "desk-run", model=model, native=FakeDesktopNative(),
                      accepted_sha256=sha(model.read_bytes()), progress=quiet)
        cmp_ = DK.compare(desktop=tmp / "desk-run", run05=out / "small", out=tmp / "desk-cmp")
        check("desktop: the frozen subset exported, path U only, each input checked with the native tokenizer, compared and labelled",
              done["completed"] == len(recs) and not done["failed"] and cmp_["same_choice"] == cmp_["compared"] == len(recs)
              and cmp_["max_abs_logit_difference"] == 0.0 and "not Quest agreement" in cmp_["label"])
        m = SC.score(prep=prep, rules=tmp / "rules", runs={PF.MODEL_KEYS[0]: out / "small", PF.MODEL_KEYS[1]: out / "large"},
                     out=tmp / "scores", frozen=frozen)
        srows = [json.loads(x) for x in (tmp / "scores" / "scores.jsonl").read_text(encoding="utf-8").splitlines() if x]
        check("scoring: one row per request; ASK separate from failures; the failed request kept as execution_failed",
              len(srows) == rc["counts"]["requests"] and m["counts"][PF.MODEL_KEYS[1]].get("execution_failed") == 1
              and all(r["models"][k]["technical_status"] == "completed" for r in srows for k in PF.MODEL_KEYS
                      if r["models"][k]["outcome"] == "ask")
              and next(r for r in srows if r["request_id"] == first["request_id"])["models"][PF.MODEL_KEYS[1]]["outcome"] == "execution_failed"
              and set(m["counts"]["rules"]) <= {"correct", "resolved_other_object", "ambiguous", "insufficient_information",
                                                 "unsupported", "unsatisfiable", "no_candidate"} | {k for k in m["counts"]["rules"] if k.startswith("technical:")})
        res = AN.analyze(prep=prep, scores=tmp / "scores", out=tmp / "analysis", b=200, desktop=tmp / "desk-cmp")
        tg = [json.loads(x) for x in (prep / "reference_only" / "targets.jsonl").read_text(encoding="utf-8").splitlines() if x]
        flagged = sum(1 for x in tg if x["partition"] == "development" and x["parsing_label"] == "disagreeing")
        st = [json.loads(x) for x in (prep / "strata.jsonl").read_text(encoding="utf-8").splitlines() if x]
        dz = sorted({s["group"] for s in st if s["N"] == 0 and s["partition"] == "development"})
        d0 = res["development"]
        check("reporting corrections: every score carries its parsing-label flag; the analysis states the contributing, "
              "assigned and zero-eligible development environments and the flagged selected parents, kept for scoring",
              all("parsing_label" in r for r in srows) and d0["contributing_environments"] == len(d0["environments"])
              and d0["assigned_environments"] == len({s["group"] for s in st if s["partition"] == "development"})
              and d0["zero_eligible_environments"] == dz and d0["parsing_label_disagreeing_parents"]["count"] == flagged
              and "kept for unique-target" in d0["parsing_label_disagreeing_parents"]["treatment"],
              str((d0["contributing_environments"], d0["assigned_environments"], dz, flagged)))
        dev = res["development"]
        rel = {(x["system"], x["relation"]) for x in dev["by_relation_command_weighted"]}
        check("the analysis: ten system, view and format results, twenty paired differences, legacy separate, coverage, "
              "'above' and 'in' unavailable, the scope stated",
              len(dev["systems"]) == 10 and len(dev["paired_differences"]) == 20
              and res["legacy-development"].get("note", "").startswith("one environment")
              and ("all", "above") in rel and ("all", "in") in rel and res["coverage"] and len(res["scope"]) >= 5
              and (tmp / "analysis" / "report.md").is_file())
        rules_sys = [k for k in dev["systems"] if k.startswith("rules")]
        check("the rules are reported per view only, never per format", rules_sys == ["rules | full_inventory", "rules | source_known_nyu"]
              or sorted(rules_sys) == ["rules | full_inventory", "rules | source_known_nyu"])
    print(f"\n{len(PASSES)} passed, {len(FAILS)} failed")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
