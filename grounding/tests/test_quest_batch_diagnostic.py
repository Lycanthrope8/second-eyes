"""A2.5 delivery 1: the bounded batch-arithmetic diagnostic (labelled fake natives; the replay-compare suite's fixture
bundle). Run directly (python grounding/tests/test_quest_batch_diagnostic.py) or as a module.

The main double follows the dispatch model: the final position's row matches U's bit for bit only when the batch it was
computed in has U's size class (1 token; 2 to Q - 1; Q or more, where Q stands in for the 64-query threshold).
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
    BD = importlib.import_module("grounding.quest.batch_diagnostic")
    EI = importlib.import_module("grounding.evaluation.iref_vla.protocol").EvaluationInputError

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle = T.build(tmp)
        recs = [json.loads(x) for x in (bundle / "headset" / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
        last = lambda r: len(r["token_ids"]) - 512 * ((len(r["token_ids"]) - 1) // 512)  # noqa: E731  U's last batch
        by_last = sorted(recs, key=last)
        short1, short2, long_ = by_last[0], by_last[1], by_last[-1]
        Q = last(long_)   # the long request's U last batch is >= Q (the "tiled" class); the short ones' are below it
        assert last(short2) < Q - 1, "the fixture needs two last batches smaller than the largest by 2 tokens"

        class Dispatch(T.FakeNative):
            """Labelled double: the dispatch model. Rows differ (by 0.001) unless the final batch has U's size class."""
            def size_class(self, b):
                return "one" if b == 1 else ("chunk" if b < Q else "tiled")

            def eval(self, ids, keep):
                rc = super().eval(ids, keep)
                total = keep + len(ids)
                b = total - 512 * ((total - 1) // 512) if keep == 0 else len(ids)
                self.cls = self.size_class(b)
                if keep == 0:
                    self.u_cls = self.cls
                return rc

            def logits(self, n):
                row = super().logits(n)
                return row if self.cls == self.u_cls else [x + 0.001 for x in row]

        class Always(Dispatch):
            """Labelled double: every recomputation differs (batch dependence within a class)."""
            def logits(self, n):
                row = T.FakeNative.logits(self, n)
                return [x + 0.001 for x in row] if self.last != "U" else row

        class Drifting(Dispatch):
            """Labelled double: U' differs from U (a history effect)."""
            def load(self, *a):
                self.count = 0
                return super().load(*a)

            def eval(self, ids, keep):
                self.count += 1
                return super().eval(ids, keep)

            def logits(self, n):
                row = super().logits(n)
                return [x + 0.5 for x in row] if self.count >= 3 else row

        pairs = ((short1["request_id"], 2, "equal"), (short2["request_id"], 2, "equal"), (long_["request_id"], Q, "equal"),
                 (long_["request_id"], Q - 1, "differ"))
        sel = tuple(dict.fromkeys(r["request_id"] for r in recs if r["request_id"] in {p[0] for p in pairs}))
        model = tmp / "model.gguf"
        model.write_bytes(b"labelled fake model")
        common = dict(bundle=bundle, model=model, library=tmp / "none", accepted_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                      progress=lambda m: None)
        RR.run_repeat(out=tmp / "r013", native=Dispatch(bundle, 300000), requests=sel, **common)

        print("-- the run (labelled dispatch-model native)")
        d = BD.run_batch_diagnostic(out=tmp / "b1", native=Dispatch(bundle, 300000), pairs=pairs, **common)
        lines = [json.loads(x) for x in (tmp / "b1" / "results.jsonl").read_text(encoding="utf-8").splitlines()]
        ident = json.loads((tmp / "b1" / "identity.json").read_text(encoding="utf-8"))
        check("four contexts, twelve path evaluations, one fresh load per pair (m = Q - 1 never runs after m = Q in one context)",
              d["contexts"] == 4 and d["path_evaluations"] == 12 and len(ident["loads"]) == 4
              and [(x["request_id"], x["m"]) for x in lines] == [(p[0], p[1]) for p in pairs])
        ok_plan = all(x["path_order"] == ["U", "Rm", "Uprime"]
                      and [(e["offset"], e["count"], e["keep"]) for e in x["paths"]["Rm"]["evals"]] == [(x["n"] - x["m"], x["m"], x["n"] - x["m"])]
                      and [(e["offset"], e["count"], e["keep"]) for e in x["paths"]["U"]["evals"]] == [(0, x["n"], 0)]
                      and [(e["offset"], e["count"], e["keep"]) for e in x["paths"]["Uprime"]["evals"]] == [(0, x["n"], 0)]
                      for x in lines)
        check("each context runs U (keep 0), recompute-m (keep n - m, the last m tokens), then U' (keep 0)", ok_plan)
        check("every cache count is verified, and every path keeps its full-row hash and offered scores",
              all(e["n_cached"] == e["expected_n_cached"] for x in lines for p in x["paths"].values() for e in p["evals"])
              and all(p["row_sha256"] and p["offered"] for x in lines for p in x["paths"].values()))
        s = BD.compare_batch_diagnostic(results=tmp / "b1", r013=tmp / "r013", out=tmp / "c1")
        check("controls hold: U' = U, U = r013's U, cache counts, inputs; then all four predictions hold",
              s["controls_hold"] and all(q["held"] for q in s["pairs"]) and [q["observed"] for q in s["pairs"]] == ["equal", "equal", "equal", "differ"])
        check("the reading is support on these cases, explicitly not proof", "support the batching hypothesis" in s["reading"][0]
              and "not proof" in s["reading"][0] and len(s["reading"]) == 1)
        print("-- other outcomes")
        BD.run_batch_diagnostic(out=tmp / "b2", native=Always(bundle, 300000), pairs=pairs, **common)
        s2 = BD.compare_batch_diagnostic(results=tmp / "b2", r013=tmp / "r013", out=tmp / "c2")
        check("batch dependence within a predicted class: the dispatch model is reported incomplete, naming the pairs",
              s2["controls_hold"] and any("dispatch model is incomplete" in r for r in s2["reading"]) and short1["request_id"] in s2["reading"][0])
        BD.run_batch_diagnostic(out=tmp / "b3", native=T.FakeNative(bundle, 300000), pairs=pairs, **common)
        RR.run_repeat(out=tmp / "r013b", native=T.FakeNative(bundle, 300000), requests=sel, **common)
        s3 = BD.compare_batch_diagnostic(results=tmp / "b3", r013=tmp / "r013b", out=tmp / "c3")
        check("m = Q - 1 equal to U: reported as not establishing the one-token path as the sole cause",
              s3["controls_hold"] and any("does not establish that the one-token path is the sole cause" in r for r in s3["reading"]))
        BD.run_batch_diagnostic(out=tmp / "b4", native=Drifting(bundle, 300000), pairs=pairs, **common)
        s4 = BD.compare_batch_diagnostic(results=tmp / "b4", r013=tmp / "r013", out=tmp / "c4")
        check("U' differing from U: the controls fail and no interpretation is drawn",
              not s4["controls_hold"] and s4["reading"][0].startswith("The controls did not all hold") and "uprime_equals_u" in s4["reading"][0])
        class Offset(Dispatch):
            """Labelled double: every row 0.25 higher (another runtime than r013's)."""
            def logits(self, n):
                return [x + 0.25 for x in super().logits(n)]

        RR.run_repeat(out=tmp / "r013c", native=Offset(bundle, 300000), requests=sel, **common)
        s5 = BD.compare_batch_diagnostic(results=tmp / "b1", r013=tmp / "r013c", out=tmp / "c5")
        check("U differing from r013's U: the controls fail and no interpretation is drawn",
              not s5["controls_hold"] and "u_equals_r013" in s5["reading"][0])
        print("-- refusals and the pinned specification")
        for label, kw in (("a model that is not the accepted one", dict(common, accepted_sha256="0" * 64)),
                          ("an m that does not fit the request", dict(common)),
                          ("no host DLL with the pinned hash", dict(common))):
            try:
                if label.startswith("an m"):
                    BD.run_batch_diagnostic(out=tmp / "x1", native=Dispatch(bundle, 300000), pairs=((short1["request_id"], 10 ** 6, "equal"),), **kw)
                elif label.startswith("no host"):
                    BD.run_batch_diagnostic(out=tmp / "x2", pairs=pairs, **kw)
                else:
                    BD.run_batch_diagnostic(out=tmp / "x3", native=Dispatch(bundle, 300000), pairs=pairs, **kw)
                refused = False
            except EI:
                refused = True
            check(f"refused: {label}", refused)
        check("the command line's pairs are the approved ones, with their predictions",
              BD.PAIRS == (("r0004", 2, "equal"), ("r0329", 2, "equal"), ("r0001", 64, "equal"), ("r0001", 63, "differ")))
    print(f"\n{len(PASSES)} passed, {len(FAILS)} failed")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
