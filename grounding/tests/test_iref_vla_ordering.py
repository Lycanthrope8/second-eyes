"""A2.3c (D94) focused checks: the crossed cyclic design, its metrics, the file chain, the gates and the boundaries.

Run directly (python grounding/tests/test_iref_vla_ordering.py) or as a module
(python -m grounding.tests.test_iref_vla_ordering). Expectations below were derived by hand before the scoring code
was written; fixtures are small and labelled, and no real-model accuracy or stability is assumed.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

FAILS, PASSES = [], []
REL_CFG = REPO / "grounding" / "relations" / "relations.v1.json"
DIR_CFG = REPO / "grounding" / "relations" / "directions.v1.json"
PROTO = {"object_codes": list("ABCDEFGHIJ"), "ask_code": "K", "ask_target": "ASK",
         "code_token_ids": {c: 32 + i for i, c in enumerate("ABCDEFGHIJK")}}

# Hand-derived, independently of the implementation: code assignment a gives obj_010, obj_020, obj_030 the letters
# (A,B,C), (B,C,A), (C,A,B) for a = 0, 1, 2; list order p shows them as (010,020,030), (020,030,010), (030,010,020).
EXPECTED_N3 = {
    (0, 0): [["A", "obj_010"], ["B", "obj_020"], ["C", "obj_030"]],
    (0, 1): [["B", "obj_020"], ["C", "obj_030"], ["A", "obj_010"]],
    (0, 2): [["C", "obj_030"], ["A", "obj_010"], ["B", "obj_020"]],
    (1, 0): [["B", "obj_010"], ["C", "obj_020"], ["A", "obj_030"]],
    (1, 1): [["C", "obj_020"], ["A", "obj_030"], ["B", "obj_010"]],
    (1, 2): [["A", "obj_030"], ["B", "obj_010"], ["C", "obj_020"]],
    (2, 0): [["C", "obj_010"], ["A", "obj_020"], ["B", "obj_030"]],
    (2, 1): [["A", "obj_020"], ["B", "obj_030"], ["C", "obj_010"]],
    (2, 2): [["B", "obj_030"], ["C", "obj_010"], ["A", "obj_020"]],
}


def check(name, ok, detail=""):
    (PASSES if ok else FAILS).append(name)
    print(("  ok    " if ok else "  FAIL  ") + name + ("" if ok or not detail else f"  [{detail}]"))


def outcome(fn):
    EI = importlib.import_module("grounding.evaluation.iref_vla.protocol").EvaluationInputError
    try:
        return ("ok", fn())
    except EI as e:
        return ("input_error", sorted({i["code"] for i in e.issues}), [i["message"] for i in e.issues][:2])
    except Exception as e:  # noqa: BLE001
        return ("crashed", f"{type(e).__name__}: {e}")


def mods():
    return (importlib.import_module("grounding.inference.iref_vla_ordering.design"),
            importlib.import_module("grounding.inference.iref_vla_ordering.prepare"),
            importlib.import_module("grounding.inference.iref_vla_ordering.run"),
            importlib.import_module("grounding.inference.iref_vla_ordering.score"))


def close(x, y) -> bool:
    return x is not None and abs(x - float(y)) < 1e-12


# ------------------------------------------------------------------------------------------------- the design
def check_design(D):
    print("-- the crossed cyclic design: literal mappings, balance, cardinality, schedule")
    objs = ["obj_010", "obj_020", "obj_030"]
    got = {c: D.transform(objs, *c) for c in D.cells(3)}
    check("all nine n = 3 cells equal the hand-derived mappings", got == EXPECTED_N3,
          str([c for c in EXPECTED_N3 if got.get(c) != EXPECTED_N3[c]]))
    b = [dict(got[c])["B"] for c in ((0, 0), (1, 0), (0, 1), (1, 1))]
    check("a B choice decodes to obj_020, obj_010, obj_020, obj_010 in the brief's four examples",
          b == ["obj_020", "obj_010", "obj_020", "obj_010"], str(b))
    check("the brief's (a=0,p=1) example is not sorted back into alphabetical order",
          [m[0] for m in got[(0, 1)]] == ["B", "C", "A"])
    pairs = {o: sorted((m[0], j) for c in D.cells(3) for j, m in enumerate(got[c]) if m[1] == o) for o in objs}
    check("each object meets every (letter, list position) pair exactly once across the nine cells",
          all(v == sorted((c, j) for c in "ABC" for j in range(3)) for v in pairs.values()))
    check("the balance check accepts the n = 3 grid", D.balance_problems(objs, got) == [])
    for n in (1, 2, 10):
        o = [f"obj_{i:03d}" for i in range(n)]
        g = {c: D.transform(o, *c) for c in D.cells(n)}
        check(f"n = {n}: {n * n} cells in canonical order, balanced", D.cells(n) == [(a, p) for a in range(n) for p in range(n)]
              and len(g) == n * n and D.balance_problems(o, g) == [])
    bad = dict(got)
    bad[(1, 1)] = got[(0, 0)]
    check("the balance check rejects a grid with one cell repeated", D.balance_problems(objs, bad) != [])
    for label, fn in (("a = n", lambda: D.transform(objs, 3, 0)), ("p negative", lambda: D.transform(objs, 0, -1)),
                      ("zero objects", lambda: D.transform([], 0, 0)), ("eleven objects", lambda: D.transform(list("abcdefghijk"), 0, 0)),
                      ("a float shift", lambda: D.transform(objs, 1.0, 0)), ("a zero grid", lambda: D.cells(0))):
        try:
            fn()
            check(f"{label} is refused", False)
        except ValueError:
            check(f"{label} is refused", True)
    m = D.full_mapping(objs, 2, 1, PROTO)
    check("K stays last and is never assigned to an object", m[-1] == ["K", "ASK"] and all(x[0] != "K" for x in m[:-1]))
    rows = [{"variant_id": D.variant_id(b_, a, p), "assignment_shift": a, "order_shift": p}
            for b_ in ("q001", "q002") for a, p in D.cells(3)]
    order = D.schedule_order(rows, "second-eyes/a23c/order/v1\n")
    rest = sorted((r["variant_id"] for r in rows if (r["assignment_shift"], r["order_shift"]) != (0, 0)),
                  key=lambda v: (hashlib.sha256(("second-eyes/a23c/order/v1\n" + v).encode()).hexdigest(), v))
    check("the schedule runs the identity cells first in base order, then the salted-hash order",
          order == ["q001.a00.p00", "q002.a00.p00"] + rest)
    pol = D.load_policy()
    check("the shipped policy's salt has exactly one LF, at the end", pol["schedule"]["salt"] == "second-eyes/a23c/order/v1\n")
    e = pol["expected"]
    check("the policy's expected population is the brief's table (128 bases, 4,536 cells, 3,004 + 1,532)",
          e["bases"] == 128 and e["cells"] == 4536 and e["identity_cells"] == 128 and e["nonidentity_cells"] == 4408
          and e["cells_by_view"] == {"full_inventory": 3004, "source_known_nyu": 1532}
          and sum(int(n) ** 2 * k for n, k in e["bases_by_object_count"].items()) == 4536
          and sum(e["bases_by_object_count"].values()) == 128)


# ------------------------------------------------------------------------------------------------- metrics
def synthetic(D, bid, n, target, responder, *, index0=0, parent=None, view="full_inventory", fmt="coordinates_v2", rank=1):
    """One base request's grid: request rows, result rows and its source. responder(mapping, a, p) -> code."""
    objs = [f"obj_{10 * (i + 1):03d}" for i in range(n)]
    parent = parent or f"parent.{bid}"
    reqs, ress = [], []
    for k, (a, p) in enumerate(D.cells(n), 1):
        mapping = [[c, t, PROTO["code_token_ids"][c]] for c, t in D.full_mapping(objs, a, p, PROTO)]
        q = {"variant_index": index0 + k, "variant_id": D.variant_id(bid, a, p), "base_request_index": int(bid[1:]),
             "base_request_id": bid, "selection_rank": rank, "parent_command_id": parent, "view_id": view, "format": fmt,
             "derived_scene_id": f"scene.{bid}", "derived_command_id": f"command.{bid}", "object_count": n,
             "assignment_shift": a, "order_shift": p, "object_ids": objs, "mapping": mapping}
        code = responder(mapping, a, p)
        codes = [m[0] for m in mapping]
        obj = None if code == "K" else dict((m[0], m[1]) for m in mapping)[code]
        ress.append({"base_request_id": bid, "assignment_shift": a, "order_shift": p, "choice_code": code,
                     "choice_object_id": obj, "model_choice": "model_choice_ask" if obj is None else "model_choice_object",
                     "choice_list_position": None if obj is None else codes.index(code) + 1,
                     "choice_scene_position": None if obj is None else objs.index(obj) + 1})
        reqs.append(q)
    src = {(parent, view, fmt): {"target": objs[target], "relation": "near", "rules_outcome": "resolved",
                                 "rules_target": objs[target], "request_id": bid}}
    return reqs, ress, src


def code_of(obj):
    return lambda m, a, p: next(x[0] for x in m if x[1] == obj)


ALWAYS_B = lambda m, a, p: "B"  # noqa: E731
SECOND = lambda m, a, p: m[1][0]  # noqa: E731
ALWAYS_K = lambda m, a, p: "K"  # noqa: E731
FIRST = lambda m, a, p: m[0][0]  # noqa: E731


def metrics(D, S, n, target, responder, **kw):
    reqs, ress, src = synthetic(D, "q001", n, target, responder, **kw)
    rows = S.score_rows(reqs, ress, src, "test")
    return rows, S.command_metrics(rows)


def check_metrics(D, S):
    print("-- metrics: section 8 responders, contrasts, null denominators, macro against pooled")
    table = {"always B": (ALWAYS_B, 3, 9), "always second list position": (SECOND, 3, 9),
             "always obj_020": (code_of("obj_020"), 9, 9), "always obj_010": (code_of("obj_010"), 0, 9),
             "always K": (ALWAYS_K, 0, 9)}
    got = {}
    for name, (fn, c, planned) in table.items():
        rows, m = metrics(D, S, 3, 1, fn)
        got[name] = m
        a = m["subsets"]["all"]
        check(f"n = 3, target obj_020, {name}: {c}/{planned} source matches across the nine cells",
              a["source_target_selections"] == c and a["planned"] == planned and m["target_agreement"]["numerator"] == c,
              f"{a['source_target_selections']}/{a['planned']}")
    k = got["always K"]["subsets"]["all"]
    check("always K: zero object choices, C/object choices is null, list positions null",
          k["object_choices"] == 0 and k["c_over_object_choices"] is None and k["chosen_list_positions"] == {}
          and got["always K"]["stability"]["list_position"] is None)
    B, P2, O20, O10 = (got[x]["contrasts"] for x in ("always B", "always second list position", "always obj_020", "always obj_010"))
    check("always B changes object on every code shift at fixed p (6 of 6) and never on a list shift (0 of 6)",
          (B["code_assignment"]["object_or_ask_changed"], B["code_assignment"]["pairs"], B["list_order"]["object_or_ask_changed"])
          == (6, 6, 0) and B["code_assignment"]["same_letter"] == 6)
    check("always-second-position does the opposite: 0 of 6 code-shift changes, 6 of 6 list-shift changes, same position 6/6",
          (P2["code_assignment"]["object_or_ask_changed"], P2["list_order"]["object_or_ask_changed"],
           P2["list_order"]["same_position"], P2["list_order"]["same_position_pairs"]) == (0, 6, 6, 6))
    check("a fixed-object responder changes under neither factor (right or wrong object)",
          all(x[k2]["object_or_ask_changed"] == 0 for x in (O20, O10) for k2 in ("code_assignment", "list_order")))
    for n in (5, 4):
        _, m = metrics(D, S, n, 2, ALWAYS_B)
        c = m["contrasts"]
        check(f"n = {n}, always B: {n}/{n * n} matches; code-shift changes {n * (n - 1)}/{n * (n - 1)}, list-shift 0",
              m["target_agreement"]["numerator"] == n and m["target_agreement"]["denominator"] == n * n
              and c["code_assignment"]["object_or_ask_changed"] == c["code_assignment"]["pairs"] == n * (n - 1)
              and c["list_order"]["object_or_ask_changed"] == 0)
        _, m2 = metrics(D, S, n, 2, SECOND)
        c2 = m2["contrasts"]
        check(f"n = {n}, always second position: {n}/{n * n} matches; list-shift changes all {n * (n - 1)}, code-shift 0",
              m2["target_agreement"]["numerator"] == n and c2["list_order"]["object_or_ask_changed"] == n * (n - 1)
              and c2["code_assignment"]["object_or_ask_changed"] == 0)
    _, m1 = metrics(D, S, 1, 0, FIRST)
    check("n = 1: no nonidentity contrast, so both contrast rates are null (not a division by zero)",
          m1["contrasts"]["code_assignment"]["change_rate"] is None and m1["contrasts"]["list_order"]["change_rate"] is None
          and m1["reference_policies"]["always_B"] is None)
    # perturb one factor only: a responder that follows list position except in one code-shifted cell
    def one_off(m, a, p):
        return m[2][0] if (a, p) == (1, 0) else m[1][0]
    # Hand derivation (amended: the first version expected 6 list-shift changes). Second position holds
    # O[(1 + p) mod 3]: p = 0, 1, 2 give O1, O2, O0 in every row a; the perturbed cell (1, 0) picks position 3, O2.
    # Code shifts: only (0, 0) = O1 against (1, 0) = O2 changes, so 1 of 6. List shifts: rows a = 0 and a = 2 change
    # twice each; in row a = 1 the reference (1, 0) is now O2, equal to (1, 1) = O2 and unlike (1, 2) = O0, so 1.
    # Total 5 of 6.
    _, mp = metrics(D, S, 3, 1, one_off)
    c = mp["contrasts"]
    check("a single changed cell (1, 0): one code-shift change (1 of 6); as its row's reference it leaves 5 of 6 list-shift changes",
          c["code_assignment"]["object_or_ask_changed"] == 1 and c["list_order"]["object_or_ask_changed"] == 5,
          f"{c['code_assignment']['object_or_ask_changed']} {c['list_order']['object_or_ask_changed']}")
    # macro against pooled
    r3, s3, src3 = synthetic(D, "q001", 3, 1, ALWAYS_B, rank=1, parent="p1")
    r5, s5, src5 = synthetic(D, "q002", 5, 1, ALWAYS_B, index0=9, rank=2, parent="p2")
    rows = S.score_rows(r3 + r5, s3 + s5, {**src3, **src5}, "test")
    cmds = [S.command_metrics([x for x in rows if x["base_request_id"] == b]) for b in ("q001", "q002")]
    agg = S.aggregate(cmds)["full_inventory/coordinates_v2"]
    t = agg["target_agreement"]
    check("macro full-grid agreement is (1/3 + 1/5) / 2 = 4/15; the pooled cell-weighted value is 8/34 = 4/17",
          close(t["macro_mean"], Fraction(4, 15)) and t["pooled_cell_weighted"]["numerator"] == 8
          and t["pooled_cell_weighted"]["denominator"] == 34 and not close(t["macro_mean"], Fraction(4, 17)),
          f"{t['macro_mean']} {t['pooled_cell_weighted']}")
    check("the always-B and second-position references average 1/n per command: 4/15",
          close(agg["reference_policies"]["always_B_macro"], Fraction(4, 15))
          and close(agg["reference_policies"]["always_second_list_position_macro"], Fraction(4, 15)))
    check("contrast rates are averaged per command (both 1.0), with raw counts 6 + 20 = 26 code-shift pairs",
          close(agg["contrasts"]["code_assignment"]["macro_change_rate"], 1) and agg["contrasts"]["code_assignment"]["pairs"] == 26)
    # targets differ across views but are grouped by view; rules are counted once per parent and view
    ra, sa, srca = synthetic(D, "q001", 3, 0, ALWAYS_B, parent="p1", view="full_inventory", fmt="coordinates_v2")
    rb, sb, srcb = synthetic(D, "q002", 3, 0, ALWAYS_B, index0=9, parent="p1", view="full_inventory", fmt="coordinates_relations_v2")
    rows = S.score_rows(ra + rb, sa + sb, {**srca, **srcb}, "test")
    cmds = [S.command_metrics([x for x in rows if x["base_request_id"] == b]) for b in ("q001", "q002")]
    summ = S.summarize(rows, cmds, "test")
    check("rules outcomes are inherited once per parent and view, not once per format or cell",
          summ["rules_by_view"] == {"full_inventory": {"resolved": 1}})
    ident = rows[0]
    check("the identity cell agrees with itself; the target's letter and list position follow each cell's mapping",
          ident["same_object_as_identity"] and ident["source_target_code"] == "A" and rows[1]["source_target_list_position"] == 3)


# -------------------------------------------------------------------------------------------- the file chain
class Perturbed:
    """Wraps the labelled fake model: identity cells' offered logits are changed on both model paths (canaries agree)."""

    def __init__(self, base, ident_ids, *, swap=False, shift=0.0):
        self.base, self.ident, self.swap, self.shift = base, ident_ids, swap, shift
        self.forward_calls, self.seen = 0, []

    def info(self):
        return self.base.info()

    def _fix(self, ids, row):
        if tuple(ids) not in self.ident:
            return row
        row = list(row)
        offered = list(range(32, 43))
        if self.swap:
            top = max(offered, key=lambda i: row[i])
            other = min(offered, key=lambda i: row[i])
            row[top], row[other] = row[other], row[top]
        for i in offered:
            row[i] += self.shift
        return row

    def forward_last(self, ids):
        self.forward_calls += 1
        self.seen.append(tuple(ids))
        return self._fix(ids, self.base._row(ids))

    def independent_last(self, ids):
        return self._fix(ids, self.base._row(ids))


def rehash(folder: Path, changed):
    m = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    for rel in changed:
        m["files"][rel] = hashlib.sha256((folder / rel).read_bytes()).hexdigest()
    (folder / "manifest.json").write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")


def jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines()]


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r, separators=(",", ":"), ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def tree(root: Path) -> dict:
    root = Path(root)
    if root.is_file():
        return {root.name: hashlib.sha256(root.read_bytes()).hexdigest()}
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def build_chain(tmp: Path):
    """The accepted fixture chain from A2.3b's tests: bundle, base requests, fake-model pilot, rules, annotations, scores."""
    SB = importlib.import_module("grounding.tests.test_iref_vla_pilot_scoring")
    A = SB.pilot_helpers()
    P = importlib.import_module("grounding.inference.iref_vla")
    BL = importlib.import_module("grounding.evaluation.iref_vla_pilot.baseline")
    SC = importlib.import_module("grounding.evaluation.iref_vla_pilot.score")
    bundle = SB.scoring_bundle(tmp, A)
    proto_path, selected = A.fixture_protocol(tmp, bundle, P)
    A.prepare(P, bundle, proto_path, tmp / "req")
    model_dir = A.fake_checkpoint(tmp / "hf", P.load_protocol())
    A.run(P, tmp / "req", tmp / "pilot", model_dir)
    ann = SB.annotation_bundle(tmp / "annotations.json", SB.TARGETS)
    BL.run_baseline(requests=tmp / "req", bundle=bundle, relation_config=REL_CFG, direction_config=DIR_CFG,
                    out=tmp / "rules", sample=False)
    SC.run_score(pilot=tmp / "pilot", requests=tmp / "req", bundle=bundle, rules=tmp / "rules", annotations=ann,
                 out=tmp / "scores", sample=False, fixture_scene_id="fixture.iref_eval.f1")
    return SB, A, P, bundle, model_dir, ann, selected


def fixture_policy(D, tmp: Path, name="policy.json", **over):
    pol = json.loads(D.POLICY_PATH.read_text(encoding="utf-8"))
    rows = jsonl(tmp / "req" / "request-index.jsonl")
    man = json.loads((tmp / "req" / "manifest.json").read_text(encoding="utf-8"))
    by_n = {}
    for r in rows:
        by_n[str(len(r["object_ids"]))] = by_n.get(str(len(r["object_ids"])), 0) + 1
    views = {}
    for r in rows:
        views[r["view_id"]] = views.get(r["view_id"], 0) + len(r["object_ids"]) ** 2
    cells = sum(len(r["object_ids"]) ** 2 for r in rows)
    pol["expected"] = {"bases": len(rows), "cells": cells, "identity_cells": len(rows), "nonidentity_cells": cells - len(rows),
                       "bases_by_object_count": dict(sorted(by_n.items(), key=lambda x: int(x[0]))),
                       "cells_by_view": dict(sorted(views.items()))}
    pol["pins"] = {"selected_parents_sha256": man["selection"]["selected_sha256"],
                   "base_results_sha256": hashlib.sha256((tmp / "pilot" / "results.jsonl").read_bytes()).hexdigest(),
                   "base_scores_summary_sha256": hashlib.sha256((tmp / "scores" / "summary.json").read_bytes()).hexdigest(),
                   "base_scores_report_sha256": hashlib.sha256((tmp / "scores" / "report.md").read_bytes()).hexdigest()}
    pol["context_limit_tokens"] = json.loads((tmp / "req" / "protocol.json").read_text(encoding="utf-8"))["context_limit_tokens"]
    for k, v in over.items():
        pol[k] = v
    path = tmp / name
    path.write_text(json.dumps(pol, indent=1) + "\n", encoding="utf-8")
    return path


def check_file_chain(D, PRE, RUN, S):
    print("-- the file chain: A2.3b's fixture chain, preparation, the gated run and scoring")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        SB, A, P, bundle, model_dir, ann, selected = build_chain(tmp)
        pol = fixture_policy(D, tmp)
        base_rows = jsonl(tmp / "req" / "request-index.jsonl")
        before = {n: tree(p) for n, p in (("req", tmp / "req"), ("bundle", bundle), ("pilot", tmp / "pilot"),
                                          ("scores", tmp / "scores"), ("ann", ann))}
        common = dict(base_requests=tmp / "req", bundle=bundle, tokenizer_dir=A.TOK_DIR, model_description=A.MODEL_DESC,
                      policy=pol)
        with SB.Watch([ann, tmp / "scores", tmp / "pilot", tmp / "rules"]) as w:
            res = outcome(lambda: PRE.prepare_ordering(**common, out=tmp / "oreq", tokenizer=A.OffsetCharTokenizer()))
        check("preparation succeeds with the annotations, A2.3b scores, rules and pilot outputs made inaccessible",
              res[0] == "ok" and not w.denied, f"{res[:2]} denied {w.denied[:1]}")
        if res[0] != "ok":
            return
        m = res[1]
        rows = jsonl(tmp / "oreq" / "request-index.jsonl")
        texts = PRE.read_variants(tmp / "oreq", rows)
        cells = sum(len(r["object_ids"]) ** 2 for r in base_rows)
        check(f"the bundle holds every cell: {cells} for {len(base_rows)} base requests, readback clean",
              m["counts"]["cells"] == cells == len(rows) and PRE.verify_order_requests(tmp / "oreq") == [])
        same = all((tmp / "req" / b["prompt_path"]).read_bytes() == texts[f"{b['request_id']}.a00.p00"][0].encode("utf-8")
                   and json.loads((tmp / "req" / b["token_ids_path"]).read_bytes()) == texts[f"{b['request_id']}.a00.p00"][1]
                   for b in base_rows)
        check("every (0, 0) cell reproduces its original prompt bytes, token IDs and mapping exactly", same)
        w_ = json.loads((tmp / "req" / "protocol.json").read_text(encoding="utf-8"))["wrapper"]["after_user"]
        docs_ok = True
        for b in base_rows:
            heads = [texts[r["variant_id"]][0] for r in rows if r["base_request_id"] == b["request_id"]]
            tails = {h[h.index("\n", h.index('{"choices"')) + 1:] for h in heads}
            docs_ok &= len(tails) == 1 and next(iter(tails)).endswith(w_)
        check("removing the choices line leaves one byte-identical document per base request", docs_ok)
        # tampering: each change is rehashed so only the semantic check can catch it
        cases = []

        def tamper(label, edit):
            d = tmp / f"t{len(cases)}"
            shutil.copytree(tmp / "oreq", d)
            changed = edit(d)
            rehash(d, changed)
            cases.append((label, PRE.verify_order_requests(d)))
        def dup(d):
            r = jsonl(d / "request-index.jsonl"); r.insert(3, r[2]); write_jsonl(d / "request-index.jsonl", r)
            return ["request-index.jsonl"]
        def omit(d):
            r = jsonl(d / "request-index.jsonl"); gone = r.pop(4); write_jsonl(d / "request-index.jsonl", r)
            v = [x for x in jsonl(d / gone["variants_file"]) if x["variant_id"] != gone["variant_id"]]
            write_jsonl(d / gone["variants_file"], v)
            return ["request-index.jsonl", gone["variants_file"]]
        def factor(d):
            r = jsonl(d / "request-index.jsonl"); r[5]["order_shift"] = (r[5]["order_shift"] + 1) % r[5]["object_count"]
            write_jsonl(d / "request-index.jsonl", r)
            return ["request-index.jsonl"]
        def swap_ids(d):
            r = jsonl(d / "request-index.jsonl"); mp = r[6]["mapping"]; mp[0][2], mp[1][2] = mp[1][2], mp[0][2]
            write_jsonl(d / "request-index.jsonl", r)
            return ["request-index.jsonl"]
        def k_first(d):
            r = jsonl(d / "request-index.jsonl"); r[7]["mapping"] = [r[7]["mapping"][-1]] + r[7]["mapping"][:-1]
            write_jsonl(d / "request-index.jsonl", r)
            return ["request-index.jsonl"]
        def doc(d):
            r = jsonl(d / "request-index.jsonl"); x = r[8]
            v = jsonl(d / x["variants_file"])
            for line in v:
                if line["variant_id"] == x["variant_id"]:
                    line["prompt"] = line["prompt"].replace("obj_", "obj-", 1)
                    x["prompt_sha256"] = hashlib.sha256(line["prompt"].encode()).hexdigest()
            write_jsonl(d / x["variants_file"], v)
            write_jsonl(d / "request-index.jsonl", r)
            return ["request-index.jsonl", x["variants_file"]]
        def floaty(d):
            r = jsonl(d / "request-index.jsonl"); r[1]["object_count"] = float(r[1]["object_count"])
            write_jsonl(d / "request-index.jsonl", r)
            return ["request-index.jsonl"]
        def float_token(d):
            r = jsonl(d / "request-index.jsonl"); x = r[9]
            v = jsonl(d / x["variants_file"])
            for line in v:
                if line["variant_id"] == x["variant_id"]:
                    line["token_ids"][0] = float(line["token_ids"][0])
            write_jsonl(d / x["variants_file"], v)
            return [x["variants_file"]]
        for label, fn in (("a repeated cell", dup), ("an omitted cell", omit), ("a wrong factor value", factor),
                          ("swapped code-token IDs", swap_ids), ("K moved from last", k_first),
                          ("a changed document in one cell", doc), ("a whole-valued float counter", floaty),
                          ("a whole-valued float token ID in a variant line", float_token)):
            tamper(label, fn)
        for label, probs in cases:
            check(f"the bundle verifier rejects {label}", bool(probs), "accepted")
        stale = tmp / "stale"
        shutil.copytree(tmp / "oreq", stale)
        (stale / "schedule.json").write_bytes((stale / "schedule.json").read_bytes() + b" ")
        check("a stale file hash is reported", any("changed" in p for p in PRE.verify_order_requests(stale)))
        check("an existing destination is refused", outcome(lambda: PRE.prepare_ordering(**common, out=tmp / "oreq",
                                                                                          tokenizer=A.OffsetCharTokenizer()))[1]
              == ["E_EVAL_OUTPUT_EXISTS"])
        # the run
        fake = A.FakeModel()
        run_common = dict(requests=tmp / "oreq", base_pilot=tmp / "pilot", model_dir=model_dir, device="cpu",
                          tokenizer_dir=A.TOK_DIR, model_description=A.MODEL_DESC, progress=lambda m_: None)
        with SB.Watch([ann, tmp / "scores", tmp / "rules"]) as w:
            res = outcome(lambda: RUN.run_ordering(**run_common, out=tmp / "ores", model_loader=A.loader_for(fake),
                                                   tokenizer=A.OffsetCharTokenizer()))
        check("the run completes with the annotations and A2.3b scores inaccessible", res[0] == "ok" and not w.denied,
              f"{res[:2]} {w.denied[:1]}")
        if res[0] != "ok":
            return
        order = json.loads((tmp / "oreq" / "schedule.json").read_text())["order"]
        canary_ids = [f"{rows[0]['base_request_id']}.a00.p00", f"{rows[0]['base_request_id']}.a01.p00",
                      f"{rows[0]['base_request_id']}.a00.p01"]
        seq = [texts[v][1] for v in canary_ids] + [texts[v][1] for v in order]
        check("three canaries, then the identity cells first, then the rest in schedule order; each cell forwarded once",
              [list(x) for x in fake.seen] == seq and fake.forward_calls == 3 + len(rows) and fake.independent_calls == 3,
              f"{fake.forward_calls} forwards for {len(rows)} cells")
        out = jsonl(tmp / "ores" / "results.jsonl")
        ctl = jsonl(tmp / "ores" / "controls.jsonl")
        check("the identity controls are the grid's (0, 0) cells and all pass against the pilot",
              len(ctl) == len(base_rows) and all(c["passed"] for c in ctl)
              and [r["variant_id"] for r in out if r["repeat_control"]] == [f"{b['request_id']}.a00.p00" for b in base_rows])
        pilot = {r["request_id"]: r for r in jsonl(tmp / "pilot" / "results.jsonl")}
        check("decoded choices follow each cell's own mapping, never alphabetical position",
              all(r["choice_object_id"] == (None if r["choice_code"] == "K" else dict((m_[0], m_[1]) for m_ in r["mapping"])[r["choice_code"]])
                  for r in out)
              and any(r["choice_code"] != "K" and r["choice_object_id"] != r["mapping"]["ABCDEFGHIJ".index(r["choice_code"])][1]
                      for r in out))
        check("identity cells reproduce the pilot's choices", all(
            r["choice_code"] == pilot[r["base_request_id"]]["choice_code"] for r in out if r["repeat_control"]))
        check("the results read back against the bundle", RUN.verify_order_results(tmp / "ores", tmp / "oreq") == [])
        ident = {tuple(texts[f"{b['request_id']}.a00.p00"][1]) for b in base_rows}
        for label, kw in (("a changed identity choice", {"swap": True}), ("a logit drift beyond tolerance, same choice",
                                                                          {"shift": 1e-3})):
            pm = Perturbed(A.FakeModel(), ident, **kw)
            r2 = outcome(lambda: RUN.run_ordering(**run_common, out=tmp / f"bad-{len(label)}", model_loader=A.loader_for(pm),
                                                  tokenizer=A.OffsetCharTokenizer()))
            diag = list(tmp.glob(f"bad-{len(label)}.failed-*"))
            check(f"{label}: the run stops after the controls, before any other cell, and publishes nothing",
                  r2[0] == "crashed" and "OrderingFailure" in r2[1] and pm.forward_calls == 3 + len(base_rows)
                  and not (tmp / f"bad-{len(label)}").exists() and len(diag) == 1, f"{r2} calls {pm.forward_calls}")
        r3 = outcome(lambda: RUN.run_ordering(**run_common, out=tmp / "bad-canary",
                                              model_loader=A.loader_for(A.FakeModel(independent_offset=0.01)),
                                              tokenizer=A.OffsetCharTokenizer()))
        check("a canary disagreement stops the run before any control", r3[0] == "crashed" and "canary" in r3[1]
              and not (tmp / "bad-canary").exists())
        check("a CPU run without an injected test loader is refused", outcome(lambda: RUN.run_ordering(
            **{**run_common, "out": tmp / "cpu"}, tokenizer=A.OffsetCharTokenizer()))[1] == ["E_ORDER_DEVICE"])
        moved = tmp / "pilot-moved"
        shutil.copytree(tmp / "pilot", moved)
        rr = jsonl(moved / "results.jsonl")
        rr[0]["timing_ms"]["verify"] = rr[0]["timing_ms"]["verify"] + 1.0
        write_jsonl(moved / "results.jsonl", rr)
        SB.rehash_pilot(moved)
        r4 = outcome(lambda: RUN.run_ordering(**{**run_common, "base_pilot": moved, "out": tmp / "ex1"},
                                              model_loader=A.refusing_loader, tokenizer=A.OffsetCharTokenizer()))
        check("a pilot other than the pinned one is refused before the model loads (E_ORDER_PIN)",
              r4[0] == "input_error" and r4[1] == ["E_ORDER_PIN"], str(r4[:3]))
        r5 = outcome(lambda: RUN.run_ordering(**{**run_common, "out": tmp / "ex2"}, model_loader=A.refusing_loader,
                                              tokenizer=A.MergingTokenizer()))
        check("a tokenizer that re-tokenizes differently is refused before the model loads", r5[0] == "input_error", str(r5[:2]))
        # scoring
        with SB.FailOnCall() as f:
            probe = outcome(lambda: importlib.import_module("grounding.resolution.resolve").resolve({}, {}, {}))
            f.calls.clear()
            res = outcome(lambda: S.score_ordering(requests=tmp / "oreq", results=tmp / "ores", base_pilot=tmp / "pilot",
                                                   base_scores=tmp / "scores", out=tmp / "osc"))
        check("scoring completes with parser, resolver, relations, serializer and inference made to fail on call",
              probe[0] == "crashed" and res[0] == "ok" and not f.calls, f"{res[:2]} {f.calls[:2]}")
        if res[0] == "ok":
            sc = jsonl(tmp / "osc" / "scores.jsonl")
            cm = jsonl(tmp / "osc" / "commands.jsonl")
            targets = {s["parent_command_id"]: s["source_target_id"] for s in sc}
            check("source targets are inherited from A2.3b unchanged", targets == SB.TARGETS)
            check("one score row per cell and one command row per base request; the folder reads back",
                  len(sc) == len(rows) and len(cm) == len(base_rows) and S.verify_order_scores(tmp / "osc") == [])
            check("every rows' source-target selection decodes through its own mapping",
                  all(s["source_target_selected"] == (s["choice_object_id"] == s["source_target_id"]) for s in sc))
            bad = tmp / "osc-copy"
            shutil.copytree(tmp / "osc", bad)
            r = jsonl(bad / "scores.jsonl")
            r[0]["source_target_selected"] = not r[0]["source_target_selected"]
            write_jsonl(bad / "scores.jsonl", r)
            check("an edited score row is caught by readback", S.verify_order_scores(bad) != [])
        after = {n: tree(p) for n, p in (("req", tmp / "req"), ("bundle", bundle), ("pilot", tmp / "pilot"),
                                         ("scores", tmp / "scores"), ("ann", ann))}
        check("every input byte is unchanged by preparation, the runs and scoring", before == after)


def check_cli():
    print("-- command line: exit codes")
    r = subprocess.run([sys.executable, "-m", "grounding.inference.iref_vla_ordering", "run", "--requests", "x",
                        "--base-pilot", "x", "--model-dir", "x", "--tokenizer-dir", "x", "--model-description", "x",
                        "--out", "y", "--device", "cpu"], cwd=REPO, capture_output=True, text=True)
    check("--device cpu is not offered (exit 2)", r.returncode == 2)
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([sys.executable, "-m", "grounding.inference.iref_vla_ordering", "score", "--requests", tmp,
                            "--results", tmp, "--base-pilot", tmp, "--base-scores", tmp, "--out", tmp],
                           cwd=REPO, capture_output=True, text=True)
        check("an existing destination exits 2", r.returncode == 2 and "E_EVAL_OUTPUT_EXISTS" in r.stderr, r.stderr[-200:])


def check_real_tokenizer(PRE):
    print("-- the pinned tokenizer on a non-alphabetical choices line (skipped if unavailable)")
    try:
        T = importlib.import_module("grounding.preparation.iref_vla.tokens")
        tok = T.load_pinned_tokenizer(REPO / "quest-app" / "Assets" / "SecondEyes" / "Models" / "qwen2.5-0.5b-instruct")
    except Exception as e:  # noqa: BLE001
        print(f"  skipped: {type(e).__name__}")
        return
    C = importlib.import_module("grounding.inference.iref_vla.choices")
    proto = importlib.import_module("grounding.inference.iref_vla.protocol").load_protocol()
    D = importlib.import_module("grounding.inference.iref_vla_ordering.design")
    objs = [f"obj_{i:03d}" for i in range(1, 11)]
    ok = True
    for a, p in ((0, 0), (3, 7), (9, 1), (5, 5)):
        m = D.full_mapping(objs, a, p, proto)
        prompt = C.build_prompt(proto, m, '{"objects":[]}\n')
        out = C.check_boundary(tok, prompt, m, proto)
        ok &= [x[2] for x in out] == [proto["code_token_ids"][x[0]] for x in m]
    check("every code is one pinned token continuing prompts whose choices are rotated", ok)


def main() -> int:
    D, PRE, RUN, S = mods()
    check_design(D)
    check_metrics(D, S)
    check_file_chain(D, PRE, RUN, S)
    check_cli()
    check_real_tokenizer(PRE)
    print(f"\n{len(PASSES)} passed, {len(FAILS)} failed")
    for f in FAILS:
        print("  FAILED:", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
