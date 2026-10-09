"""A2.5 delivery 1: the repeatability and context-history diagnostic (ChatGPT's proposal).

Run directly (python grounding/tests/test_quest_replay_repeat.py) or as a module. A fixture bundle is built on A2.3b's
fixture chain, and labelled fake natives (from test_quest_replay_compare) stand in for the wrapper:

- a deterministic one, where R is lifted;
- one whose later evaluations in a context drift;
- one that changes from one fresh context to the next;
- one whose tokenizer fails in the second context.

Expectations are worked out by hand.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
FAILS, PASSES = [], []


def check(name, ok, detail=""):
    (PASSES if ok else FAILS).append(name)
    print(("  ok    " if ok else "  FAIL  ") + name + ("" if ok or not detail else f"  [{detail}]"))


def main() -> int:
    T = importlib.import_module("grounding.tests.test_quest_replay_compare")
    RR = importlib.import_module("grounding.quest.replay_repeat")
    EI = importlib.import_module("grounding.evaluation.iref_vla.protocol").EvaluationInputError

    class Drifting(T.FakeNative):
        """Labelled double: from the fifth evaluation after a load (that is, U'), every value is 0.25 higher."""
        def load(self, *a):
            self.count = 0
            return super().load(*a)

        def eval(self, ids, keep):
            self.count += 1
            return super().eval(ids, keep)

        def logits(self, n):
            row = super().logits(n)
            return [x + 0.25 for x in row] if self.count >= 5 else row

    class Unrepeatable(T.FakeNative):
        """Labelled double: each fresh context adds its own offset (0.125 times the number of loads so far)."""
        loads = 0

        def load(self, *a):
            Unrepeatable.loads += 1
            return super().load(*a)

        def logits(self, n):
            return [x + 0.125 * Unrepeatable.loads for x in super().logits(n)]

    class Breaking(T.FakeNative):
        """Labelled double: the tokenizer returns one wrong token from the second load on."""
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.loads = 0

        def load(self, *a):
            self.loads += 1
            return super().load(*a)

        def tokenize(self, data):
            ids = super().tokenize(data)
            return ids[:-1] + [ids[-1] + 1] if self.loads >= 2 else ids

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle = T.build(tmp)
        ids = [json.loads(x)["request_id"] for x in (bundle / "headset" / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
        sel = (ids[0], ids[3], ids[-1])
        model = tmp / "model.gguf"
        model.write_bytes(b"labelled fake model")
        common = dict(bundle=bundle, model=model, library=tmp / "none", accepted_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                      progress=lambda m: None, requests=sel)

        print("-- a deterministic native (R lifted above the top on purpose)")
        d = RR.run_repeat(out=tmp / "r1", native=T.FakeNative(bundle, 300000, jitter={"R": 10.0}), **common)
        lines = [json.loads(x) for x in (tmp / "r1" / "results.jsonl").read_text(encoding="utf-8").splitlines()]
        check("six sequences, in order: each request twice, each in a fresh context",
              d["sequences"] == 6 and [(x["request_id"], x["repeat"]) for x in lines] == [(r, k) for r in sel for k in (1, 2)])
        n0 = len(json.loads((bundle / "headset" / "requests.jsonl").read_text(encoding="utf-8").splitlines()[0])["token_ids"])
        u = lines[0]["paths"]
        check("each sequence runs U, R, P, then U' with keep 0 over all tokens, every cache count as expected",
              lines[0]["path_order"] == ["U", "R", "P", "Uprime"] and [(e["offset"], e["count"], e["keep"]) for e in u["Uprime"]["evals"]] == [(0, n0, 0)]
              and all(e["n_cached"] == e["expected_n_cached"] for p in u.values() for e in p["evals"]))
        ident = json.loads((tmp / "r1" / "identity.json").read_text(encoding="utf-8"))
        check("every load is recorded with its own startup log, captured from stderr",
              len(ident["loads"]) == 6 and all((tmp / "r1" / x["capture_file"]).is_file() for x in ident["loads"])
              and "llama_kv_cache" in (tmp / "r1" / ident["loads"][0]["capture_file"]).read_text(encoding="utf-8"))
        check("every scored path keeps its full row's SHA-256 and its raw offered logits",
              all(p["row_sha256"] and p["offered"] for x in lines for p in x["paths"].values()))
        check("the identity says it is ChatGPT's proposal and that threads are requested, not read back",
              "ChatGPT's proposal" in ident["scope"] and "not read back" in ident["threads"])
        s = RR.compare_repeat(bundle=bundle, results=tmp / "r1", out=tmp / "c1", requests=sel)
        f = s["facts"]
        check("no problem; every path repeats bit for bit (12 of 12); U' equals U (6 of 6); path differences repeat (6 of 6)",
              not s["problems"] and (f["repeat_pairs"], f["repeat_pairs_full_row_identical"]) == (12, 12)
              and (f["uprime_pairs"], f["uprime_full_row_identical"]) == (6, 6)
              and (f["path_difference_pairs"], f["path_differences_identical"]) == (6, 6), str(f))
        check("D101 on each sequence: R against U and against float32 fail, 12 of 30 (2 comparisons x 2 contexts x 3 requests)",
              (f["d101_comparisons"], f["d101_failures"]) == (30, 12))
        st = " ".join(s["statements"])
        check("the readings: repeatability supported, no sign of history dependence, stable path differences, no defect claimed",
              "supports repeatability" in st and "no sign" in st and "supports stable differences" in st
              and "None of these readings alone establishes" in st)

        print("-- a native whose later evaluations drift (labelled)")
        RR.run_repeat(out=tmp / "r2", native=Drifting(bundle, 300000), **common)
        s = RR.compare_repeat(bundle=bundle, results=tmp / "r2", out=tmp / "c2", requests=sel)
        check("U' differs from U in all 6 contexts, read as context-history dependence",
              s["facts"]["uprime_full_row_identical"] == 0 and "context-history dependence" in " ".join(s["statements"]))
        print("-- a native that changes between fresh contexts (labelled)")
        RR.run_repeat(out=tmp / "r3", native=Unrepeatable(bundle, 300000), **common)
        s = RR.compare_repeat(bundle=bundle, results=tmp / "r3", out=tmp / "c3", requests=sel)
        check("no path repeats across contexts, read as not supporting repeatability",
              s["facts"]["repeat_pairs_full_row_identical"] == 0 and "does not support repeatability" in " ".join(s["statements"]))

        print("-- stopping and refusals")
        try:
            RR.run_repeat(out=tmp / "r4", native=Breaking(bundle, 300000), **common)
            stopped = False
        except RuntimeError:
            stopped = True
        kept = [json.loads(x) for x in (tmp / "r4.failed" / "results.jsonl").read_text(encoding="utf-8").splitlines()] \
            if (tmp / "r4.failed").is_dir() else []
        check("a tokenization mismatch in the second context stops the run, with no retry",
              stopped and not (tmp / "r4").exists())
        check("what was written is kept as .failed: the first sequence, and the second context's stopped input checks",
              len(kept) == 2 and kept[0]["paths"] and kept[1]["input"]["outcome"] == "tokenization_mismatch" and "paths" not in kept[1])
        dll = tmp / "other.dll"
        dll.write_bytes(b"another build")
        try:
            RR.run_repeat(out=tmp / "r5", **dict(common, library=dll))
            refused = False
        except EI:
            refused = True
        check("a host DLL other than r010's is refused before anything runs", refused and not (tmp / "r5").exists())
        try:
            RR.run_repeat(out=tmp / "r6", native=T.FakeNative(bundle, 300000), **dict(common, requests=(sel[1], sel[0])))
            refused = False
        except EI:
            refused = True
        check("requests out of bundle order are refused", refused)
        bad = tmp / "r1b"
        import shutil
        shutil.copytree(tmp / "r1", bad)
        (bad / "done.json").unlink()
        s = RR.compare_repeat(bundle=bundle, results=bad, out=tmp / "c5", requests=sel)
        check("the analysis reports a missing completion record and offers no reading",
              any("completion" in p for p in s["problems"]) and "no reading is offered" in s["statements"][0])
    print(f"\n{len(PASSES)} passed, {len(FAILS)} failed")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
