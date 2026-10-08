"""A2.5 delivery 1, step 3 (D101, D103): replay-compare and the desktop driver.

Run directly (python grounding/tests/test_quest_replay_compare.py) or as a module. A fixture bundle is built on A2.3b's
fixture chain (labelled doubles). A labelled fake native library replays it through the desktop driver, giving each
path the float32 reference's logits plus an optional perturbation, and replay-compare checks the result. Expectations
are worked out by hand from the D101 rule.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
FAILS, PASSES = [], []
KV_LINE = "llama_kv_cache: size =   96.00 MiB (  8192 cells,  24 layers, 16/1 seqs), K (f16):   48.00 MiB, V (f16):   48.00 MiB"


def check(name, ok, detail=""):
    (PASSES if ok else FAILS).append(name)
    print(("  ok    " if ok else "  FAIL  ") + name + ("" if ok or not detail else f"  [{detail}]"))


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


class FakeNative:
    """Labelled test double for the wrapper: tokenizes known prompts to their frozen IDs, keeps a cache, and returns the
    float32 reference's logits for the cached request, plus a perturbation chosen per path (U, R or P)."""

    def __init__(self, bundle, n_vocab, jitter=None, kv=True):
        recs = [json.loads(x) for x in (bundle / "headset" / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
        fixes = [json.loads(x) for x in (bundle / "headset" / "fixtures-written.jsonl").read_text(encoding="utf-8").splitlines()]
        import base64
        self.tok = {base64.b64decode(r["prompt_b64"]): r["token_ids"] for r in recs + fixes}
        refs = [json.loads(x) for x in (bundle / "reference" / "references.jsonl").read_text(encoding="utf-8").splitlines()]
        self.logit = {tuple(r["token_ids"]): {s["token_id"]: s["logit"] for s in ref["reference"]["scores"]}
                      for r, ref in zip(recs, refs)}
        self.n_vocab, self.jitter, self.kv = n_vocab, jitter or {}, kv
        self.cache, self.last = [], None

    def version(self): return "b11277 (eae11d22)"
    def system_info(self): return "fake"
    def last_error(self): return ""
    def set_verbose(self, on): pass
    def memory_kb(self, peak): return 1000

    def load(self, path, n_ctx, threads, n_seq, flags):
        lines = ["print_info: file type   = Q8_0", "llama_context: n_ctx                 = 8192",
                 "llama_context: n_ctx_seq             = 8192", "llama_context: flash_attn            = auto"]
        if self.kv:
            lines += ["llama_kv_cache:        CPU KV buffer size =    96.00 MiB", KV_LINE]
        os.write(2, ("\n".join(lines) + "\n").encode())   # the runtime's own output goes to stderr
        return True

    def info(self): return [8192, self.n_vocab, 16, 0]
    def tokenize(self, data): return list(self.tok[data])

    def eval(self, ids, keep):
        self.cache = self.cache[:keep] + list(ids)
        self.last = "R" if len(ids) == 1 else ("P" if keep > 0 else "U")
        return 0

    def n_cached(self): return len(self.cache)
    def logprob(self, token): return -1.0

    def logits(self, n):
        """The reference's logits; on a perturbed path the lowest offered code is lifted `add` above the top one, so
        that path's best candidate always changes."""
        row = [0.0] * n
        ref = dict(self.logit.get(tuple(self.cache), {}))
        add = self.jitter.get(self.last)
        if add is not None and ref:
            low = min(ref, key=lambda tid: (ref[tid], tid))
            ref[low] = max(ref.values()) + add
        for tid, x in ref.items():
            row[tid] = f32(x)
        return row

    def free(self): pass


def build(tmp):
    TQ = importlib.import_module("grounding.tests.test_quest_replay_bundle")
    D = importlib.import_module("grounding.inference.iref_vla_compare.design")
    P = importlib.import_module("grounding.inference.iref_vla_compare.prepare")
    RUN = importlib.import_module("grounding.inference.iref_vla_compare.run")
    RB = importlib.import_module("grounding.quest.replay_bundle")
    A, _, _ = TQ.fixture_chain(tmp, D, P, RUN)
    req, run, run7 = tmp / "req", tmp / f"run-{TQ.KEYS[0]}", tmp / f"run-{TQ.KEYS[1]}"
    sc = TQ.scores_folder(tmp / "scores", req, run, run7)
    pol = TQ.replay_policy(RB, tmp, "pol", req, run, sc, TQ.jsonl(req / "request-index.jsonl"), **{"headset.vocab_size": 300000})
    RB.build_replay_bundle(requests=req, small_run=run, scores=tmp / "scores", tokenizer_dir=None, out=tmp / "replay",
                           policy=pol, tokenizer=A.OffsetCharTokenizer())
    return tmp / "replay"


def unit(RC):
    print("-- the D101 rule on hand-written logits")
    c = RC.compare(["B", "A", "K"], [2.0, 1.0, 0.0], [2.0, 1.0, 0.0], "K", True)
    check("identical paths pass with TVD 0", c["pass"] and c["tvd"] == 0.0)
    c = RC.compare(["B", "A", "K"], [2.0, 1.0, 0.0], [1.0, 2.0, 0.0], "K", True)
    e2, e1, e0 = math.exp(2), math.exp(1), 1.0
    tvd = (e2 - e1) / (e2 + e1 + e0)   # half of twice the swapped difference
    check("swapped top two: best, ranking (B/A) and TVD (worked by hand) fail",
          c["failed"] == ["best candidate", "plausible ranking", "TVD"] and c["changed_pairs"] == ["B/A"]
          and abs(c["tvd"] - tvd) < 1e-15, str(c))
    c = RC.compare(["B", "A", "K"], [2.0, 1.0, 0.0], [1.0, 2.0, 0.0], "K", False)
    check("against float32 for U the TVD is reported, not judged", "TVD" not in c["failed"] and c["tvd"] > 0.05)
    c = RC.compare(["B", "A", "K"], [3.0, 3.0, 0.0], [3.0, 3.0, 0.0], "K", True)
    check("an exact tie at the top is K on both paths", c["choice_a"] == "K" and c["choice_b"] == "K" and c["pass"])
    # A at 0.5% on one path and 1.5% on the other is in the union; its order against K flips
    a = [5.0, math.log(0.005 * (math.exp(5) + 1) / (1 - 0.005)), 0.0]
    b = [5.0, math.log(0.015 * (math.exp(5) + 1) / (1 - 0.015)), 0.0]
    c = RC.compare(["B", "A", "K"], a, b, "K", True)
    check("a code at 1% or more on either path joins the ranked union", "A" in c["union"], str(c["union"]))


def main() -> int:
    RC = importlib.import_module("grounding.quest.replay_compare")
    DK = importlib.import_module("grounding.quest.replay_desktop")
    unit(RC)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle = build(tmp)
        model = tmp / "model.gguf"
        model.write_bytes(b"labelled fake model")
        msha = hashlib.sha256(model.read_bytes()).hexdigest()
        common = dict(bundle=bundle, model=model, library=tmp / "none", accepted_sha256=msha, progress=lambda m: None)
        print("-- the desktop driver and the comparison (labelled fake native)")
        d = DK.run_desktop(out=tmp / "d0", native=FakeNative(bundle, 300000), **common)
        check("the desktop replay writes every request and fixture, self-checks passing",
              d["written"] == d["requests"] and d["fixtures_as_expected"] == 5 and d["self_checks_passed"], str(d))
        ident = json.loads((tmp / "d0" / "identity.json").read_text(encoding="utf-8"))
        here = REPO / "grounding" / "quest" / "replay_desktop.py"
        check("the desktop identity records the SHA-256 of the code that ran, replay_desktop.py included",
              ident["code"].get("grounding/quest/replay_desktop.py") == hashlib.sha256(here.read_bytes()).hexdigest()
              and "grounding/quest/publish.py" in ident["code"])
        log = (tmp / "d0" / "startup-log.txt").read_text(encoding="utf-8")
        check("the runtime's stderr during the load lands in startup-log.txt", KV_LINE in log)
        s = RC.compare_replay(bundle=bundle, results=tmp / "d0", out=tmp / "c0")
        n = s["per_comparison"]["U vs float32"]["compared"]
        check(f"paths equal to float32: acceptance passes, 5 comparisons for each of {n} requests, no problem",
              s["verdict"]["replay_acceptance"] == "pass" and s["verdict"]["comparisons"] == 5 * n and not s["problems"],
              json.dumps(s["verdict"]) + str(s["problems"][:2]))
        check("K and V types and sizes come from the log line, at its precision",
              s["startup_log"]["kv"]["k_type"] == "f16" and s["startup_log"]["kv"]["v_size"] == "48.00 MiB")
        d = DK.run_desktop(out=tmp / "d1", native=FakeNative(bundle, 300000, jitter={"R": 10.0}), **common)
        s = RC.compare_replay(bundle=bundle, results=tmp / "d1", out=tmp / "c1")
        pc = s["per_comparison"]
        check("R perturbed (its lowest code lifted 10 above the top): R against U and against float32 fail for every request; U and P pass",
              pc["R vs U"]["fail"] == n and pc["R vs float32"]["fail"] == n and pc["U vs float32"]["fail"] == 0
              and pc["P vs U"]["fail"] == 0 and pc["P vs float32"]["fail"] == 0 and s["verdict"]["replay_acceptance"] == "fail",
              json.dumps({k: v["fail"] for k, v in pc.items()}))
        check("every failure is listed with its request and failed criteria",
              len(s["failures"]) == 2 * n and all(f["failed"] for f in s["failures"]))
        d = DK.run_desktop(out=tmp / "d2", native=FakeNative(bundle, 300000, kv=False), **common)
        s = RC.compare_replay(bundle=bundle, results=tmp / "d2", out=tmp / "c2")
        check("without the KV line, K/V reporting is incomplete and the field is listed missing",
              s["verdict"]["k_v_reporting"] == "incomplete" and "kv_cache" in s["startup_log"]["missing"])

        print("-- structural enforcement on tampered results")
        def tampered(name, edit):
            t = tmp / f"t-{name}"
            shutil.copytree(tmp / "d0", t)
            edit(t)
            return RC.compare_replay(bundle=bundle, results=t, out=tmp / f"ct-{name}")

        def edit_line(t, file, fn, which=0):
            lines = (t / file).read_text(encoding="utf-8").splitlines()
            rows = [json.loads(x) for x in lines]
            fn(rows[which])
            (t / file).write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")

        cases = [("an evaluation offset moved", lambda t: edit_line(t, "results.jsonl",
                                                                    lambda r: r["paths"]["R"]["evals"][0].update(offset=0)), "D103 schedule"),
                 ("the last result line dropped", lambda t: (t / "results.jsonl").write_text(
                     "".join(x + "\n" for x in (t / "results.jsonl").read_text(encoding="utf-8").splitlines()[:-1]), encoding="utf-8"),
                  "coverage"),
                 ("a recorded choice changed", lambda t: edit_line(t, "results.jsonl",
                                                                   lambda r: r["paths"]["U"].update(choice_code="Z")), "recorded decision"),
                 ("a fixture that was evaluated", lambda t: edit_line(t, "fixtures.jsonl", lambda r: r.update(evaluated=True)), "fixtures"),
                 ("other bundle files named in the identity", lambda t: (t / "identity.json").write_text(json.dumps(dict(
                     json.loads((t / "identity.json").read_text(encoding="utf-8")), bundle={"manifest_sha256": "0" * 64})),
                     encoding="utf-8"), "inputs"),
                 ("a tokenization difference", lambda t: edit_line(t, "results.jsonl",
                                                                   lambda r: r["input"].update(first_token_difference=3)), "input checks")]
        for label, edit, why in cases:
            s = tampered(label.replace(" ", "_"), edit)
            check(f"reported: {label}", s["verdict"]["replay_acceptance"] == "fail" and any(why in p for p in s["problems"]),
                  str(s["problems"][:2]))
        root = hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest()
        hs = {n_: hashlib.sha256((bundle / "headset" / n_).read_bytes()).hexdigest()
              for n_ in ("replay-manifest.json", "requests.jsonl", "fixtures-written.jsonl")}
        good = tmp / "push.json"
        good.write_text(json.dumps({"bundle_manifest_sha256": root, "files": [{"name": k, "sha256": v} for k, v in hs.items()]}))
        s = RC.compare_replay(bundle=bundle, results=tmp / "d0", out=tmp / "c3", push_receipt=good)
        check("a push receipt naming the root manifest and the pushed files links them",
              s["inputs"]["push_receipt"]["links_root_to_pushed_files"] and s["verdict"]["replay_acceptance"] == "pass")
        bad = tmp / "push-bad.json"
        bad.write_text(json.dumps({"bundle_manifest_sha256": "0" * 64, "files": []}))
        s = RC.compare_replay(bundle=bundle, results=tmp / "d0", out=tmp / "c4", push_receipt=bad)
        check("a push receipt for another bundle is reported", any("push receipt" in p for p in s["problems"]))

        print("-- refusals")
        EI = importlib.import_module("grounding.evaluation.iref_vla.protocol").EvaluationInputError
        def refused(fn):
            try:
                fn()
            except EI:
                return True
            return False
        check("the desktop driver refuses another model file",
              refused(lambda: DK.run_desktop(out=tmp / "x1", native=FakeNative(bundle, 300000),
                                             **dict(common, accepted_sha256="0" * 64))))
        check("and a missing host build when no native is given",
              refused(lambda: DK.run_desktop(out=tmp / "x2", **common)))
        check("the comparison refuses an existing destination",
              refused(lambda: RC.compare_replay(bundle=bundle, results=tmp / "d0", out=tmp / "c0")))
    print("-- publishing a folder (a simulated Windows file lock)")
    PB = importlib.import_module("grounding.quest.publish")
    real = os.rename
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "s").mkdir()
        calls = []
        def flaky(a, b):
            calls.append(1)
            if len(calls) <= 2:
                raise PermissionError(13, "Access is denied")
            real(a, b)
        os.rename = flaky
        try:
            k = PB.publish(tmp / "s", tmp / "o", sleep=lambda s: None)
        finally:
            os.rename = real
        check("two refused renames, then the folder is published on the third try", k == 3 and (tmp / "o").is_dir())
        (tmp / "s2").mkdir()
        os.rename = lambda a, b: (_ for _ in ()).throw(PermissionError(13, "Access is denied"))
        try:
            PB.publish(tmp / "s2", tmp / "o2", sleep=lambda s: None)
            raised = False
        except PermissionError:
            raised = True
        finally:
            os.rename = real
        check("a lock that never clears still fails, after the last try, and nothing is published",
              raised and (tmp / "s2").is_dir() and not (tmp / "o2").exists())
    print(f"\n{len(PASSES)} passed, {len(FAILS)} failed")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
