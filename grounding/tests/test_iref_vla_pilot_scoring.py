"""Tests of A2.3b: saved-pilot scoring and the matched rules baseline (D82).

    python grounding/tests/test_iref_vla_pilot_scoring.py --unit-only
        hand-written fixtures and labelled doubles only (fake tokenizers, a fake model); not real-data acceptance
    python grounding/tests/test_iref_vla_pilot_scoring.py --rules DIR --scores DIR [--pilot DIR]
        also reads back a real run's published rules and score folders, without recomputing rules or inference
    (or: python -m grounding.tests.test_iref_vla_pilot_scoring ...)

Expectations: fixtures/iref_vla_pilot_scoring/expectations.json, written by hand before the scoring layer existed.
"""
from __future__ import annotations

import builtins
import copy
import hashlib
import importlib
import io
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

FX = REPO / "grounding" / "tests" / "fixtures" / "iref_vla_pilot_scoring"
EV = REPO / "grounding" / "tests" / "fixtures" / "iref_vla_evaluation"
EXP = json.loads((FX / "expectations.json").read_text(encoding="utf-8"))
REL_CFG = REPO / "grounding" / "relations" / "relations.v1.json"
DIR_CFG = REPO / "grounding" / "relations" / "directions.v1.json"
FORMATS = ("coordinates_v2", "coordinates_relations_v2")
FAILED, COUNT = [], [0]


def check(name, ok, detail=""):
    COUNT[0] += 1
    print(("PASS  " if ok else "FAIL  ") + name + (f": {detail}" if detail and not ok else ""))
    if not ok:
        FAILED.append(name)


def outcome(fn):
    E = importlib.import_module("grounding.evaluation.iref_vla")
    S = importlib.import_module("grounding.evaluation.iref_vla_pilot")
    try:
        return ("ok", fn())
    except E.EvaluationInputError as e:
        return ("rejected", sorted({i["code"] for i in e.issues}))
    except E.EvaluationOutputError as e:
        return ("write_failed", sorted({i["code"] for i in e.issues}))
    except S.ScoringInternalError as e:
        return ("internal", str(e)[:160])
    except Exception as e:  # noqa: BLE001
        return ("crashed", f"{type(e).__name__}: {e}"[:200])


def r_eq(got, want) -> bool:
    """A ratio record against [numerator, denominator]; a zero denominator must give null."""
    n, d = want
    return got == {"numerator": n, "denominator": d, "value": None if d == 0 else n / d}


def nonzero(d) -> dict:
    return {k: v for k, v in d.items() if v}


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_hashes(root) -> dict:
    root = Path(root)
    if root.is_file():
        return {root.name: sha(root)}
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


def jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines()]


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows), encoding="utf-8")


# ---------------------------------------------------------------------------------------- in-memory groups
def build_group(view_maps, cases, target, *, unknown=()):
    """Joined entries, saved results, rules records and sources for labelled fixture cases (never the pilot)."""
    entries, results, rules, sources, k = [], {}, {}, {}, 0
    for rank, case in enumerate(cases, 1):
        parent = f"fixture.scoring.{case['id'].lower()}"
        n = case.get("annotations", 1)
        sources[parent] = {"target_id": target, "relation": case["relation"],
                           "annotation_ids": [f"{parent}.a{i:03d}" for i in range(n)], "annotation_count": n}
        for view, mapping in view_maps.items():
            ids = [o for c, o in mapping if c != "K"]
            entries.append({"rank": rank, "parent": parent, "view": view, "object_ids": ids, "unknown_ids": list(unknown),
                            "text": f"fixture text {case['id']}", "derived_scene_id": f"fixture.scene.{rank}.{view}",
                            "derived_command_id": f"fixture.command.{rank}.{view}"})
            full = [[c, o, 32 + "ABCDEFGHIJK".index(c)] for c, o in mapping]
            for fmt in FORMATS:
                k += 1
                choice = case["choices"][view][fmt]
                row = {"request_index": k, "request_id": f"q{k:03d}", "parent_command_id": parent, "view_id": view,
                       "format": fmt, "derived_scene_id": f"fixture.scene.{rank}.{view}",
                       "derived_command_id": f"fixture.command.{rank}.{view}", "source_document_sha256": "0" * 64,
                       "prompt_sha256": "1" * 64, "token_ids_sha256": "2" * 64, "input_tokens": 100, "mapping": full}
                if choice == "EXCLUDED":
                    row.update(technical_status="context_budget_exceeded", model_choice=None, choice_code=None,
                               choice_object_id=None, selection_reason=None)
                else:
                    row.update(technical_status="completed", choice_code=choice, selection_reason="max_offered_logit",
                               model_choice="model_choice_ask" if choice == "K" else "model_choice_object",
                               choice_object_id=None if choice == "K" else dict(mapping)[choice])
                results[(parent, view, fmt)] = row
            out, tid = case["rules"][view]
            rules[(parent, view)] = {"outcome": out, "technical_failure": out in ("invalid_input", "budget_exceeded"),
                                     "target_id": tid if out == "resolved" else None, "candidate_ids": [],
                                     "reason_code": None, "reason_codes": []}
    return entries, results, rules, sources


def one_view(g):
    """Cases of an expectation group in one view, as build_group wants them."""
    v = g["view_id"]
    return [dict(c, choices={v: {f: c[f] for f in FORMATS}}, rules={v: c["rules"]}) for c in g["cases"]], {v: g["mapping"]}


def summarize_group(S, entries, results, rules, sources):
    scores, comps = S.build_rows(entries, results, rules, sources)
    return scores, comps, S.summarize(scores, comps, "fixture")


# ------------------------------------------------------------------------------------------- unit checks
def check_policy_and_metric_fixture(S):
    print("-- the policy and the brief's six-command metric fixture (labelled fixture, not the pilot)")
    pol = S.load_policy()
    check("the versioned policy loads strictly, with the four error codes and tolerance 1e-10",
          sorted(pol["error_codes"]) == sorted(EXP["error_codes"]) and pol["score_check_absolute_tolerance"] == EXP["tolerance"])
    g = EXP["metric_fixture"]
    cases, maps = one_view(g)
    scores, comps, s = summarize_group(S, *build_group(maps, cases, g["target"]))
    e, v = g["expected"], g["view_id"]
    for f in FORMATS:
        m, w = s["models"][f"{v}/{f}"], e["models"][f]
        check(f"{f}: N, completed, objects, ASK and C as the brief derives",
              all(m[k] == w[k] for k in ("N", "completed", "object_choices", "ask", "planned_exclusions", "technical_failures", "C")), str(m))
        check(f"{f}: C/N 2/6, C/completed 2/6, C/object choices 2/5",
              all(r_eq(m[k], w[k]) for k in ("C_over_N", "C_over_completed", "C_over_object_choices")))
    r, w = s["rules"][v], e["rules"]
    check("rules: resolved 3, ambiguous 1, insufficient 1, no match 1; C 2, C/N 2/6, C/R 2/3",
          r["outcomes"] == w["outcomes"] and r["technical_failures"] == w["technical_failures"] and r["R"] == w["R"]
          and r["C"] == w["C"] and r_eq(r["C_over_N"], w["C_over_N"]) and r_eq(r["C_over_R"], w["C_over_R"]), str(r)[:300])
    p, w = s["format_pairs"][v], e["format_pairs"]
    check("paired formats: both 1, coordinates only 1, augmented only 1, neither 3; same 2, different 4",
          all(p[k] == w[k] for k in w if k != "difference_pp") and p["difference_pp"]["value"] == w["difference_pp"], str(p)[:300])
    for f in FORMATS:
        x, w = s["model_vs_rules"][f"{v}/{f}"], e["model_vs_rules"][f]
        check(f"{f} against rules: {w}", all(x[k] == w[k] for k in w), str(x))
        check(f"{f} against rules: not-source rules outcomes kept by status",
              x["rules_not_source_by_outcome"] == e["rules_not_source_by_outcome"], str(x["rules_not_source_by_outcome"]))
        ct = s["policy_crosstab"][f"{v}/{f}"]
        check(f"{f}: model outcome against rules outcome", {k: nonzero(c) for k, c in ct["counts"].items() if nonzero(c)}
              == e["crosstab"][f], str(ct["counts"]))
        check(f"{f}: within rules-resolved cases, same/different/ASK", ct["within_rules_resolved"] == e["within_rules_resolved"][f])
        c = s["concentration"][f"{v}/{f}"]
        check(f"{f}: chosen aliases {e['chosen_alias'][f]}, source aliases all B", c["chosen_alias"] == e["chosen_alias"][f]
              and c["source_alias"] == e["source_alias"], str(c["chosen_alias"]))
        check(f"{f}: source alias against chosen alias", c["source_by_chosen"] == e["source_by_chosen"][f], str(c["source_by_chosen"]))
        a = c["always_alias"]
        check(f"{f}: always-B diagnostic 6/6 with B offered in all six", a["alias"] == "B" and a["offered"] == 6
              and a["source_is_alias"] == 6 and r_eq(a["agreement"], e["always_alias"]["agreement"]))
        check(f"{f}: chosen objects", c["chosen_objects"] == e["chosen_objects"][f], str(c["chosen_objects"]))
    br = s["breakdowns"]["source_relation"]
    ok = True
    for label, w in e["by_relation"].items():
        x = br[label]
        ok &= x["N"] == w["N"] and all(x["models"][f"{v}/{f}"][k] == w[f][k] for f in FORMATS for k in w[f]) \
            and x["rules"][v]["R"] == w["rules"]["R"] and x["rules"][v]["C"] == w["rules"]["C"]
    check("breakdown by source relation, N per label (closest 3, near 3)", ok, json.dumps(br)[:300])
    check("breakdown by object count: all six at three objects", list(s["breakdowns"]["object_count"][v]) == ["3"]
          and s["breakdowns"]["object_count"][v]["3"]["N"] == 6)
    pop = s["population"]
    check("population: 6 parents, 6 parent/view pairs, 12 requests, 6 annotations",
          all(pop[k] == w for k, w in e["population"].items()), str(pop))
    cases2 = copy.deepcopy(cases)
    cases2[1]["annotations"] = 2
    _, _, s2 = summarize_group(S, *build_group(maps, cases2, g["target"]))
    w = e["with_F2_annotated_twice"]
    check("F2 annotated twice: annotations 7, multiplicity {1: 5, 2: 1}; N and C unchanged (no multiplicity weighting)",
          s2["population"]["annotation_records"] == w["annotation_records"] and s2["population"]["annotation_multiplicity"]
          == w["multiplicity"] and all(s2["models"][f"{v}/{f}"]["N"] == w["N"] and s2["models"][f"{v}/{f}"]["C"] == w["C"]
                                       for f in FORMATS))
    rows_ok = all(x["source_target_selected"] == (x["model_choice"] == "model_choice_object" and x["choice_object_id"] == g["target"])
                  for x in scores)
    check("each score row: the source target is selected exactly when the chosen object is the mapped target; ASK is false",
          rows_ok and all(x["source_target_selected"] is False for x in scores if x["choice_code"] == "K"))


def check_extra_groups(S):
    print("-- separate fixture groups: absent B, zero object choices, a planned exclusion, a resolver technical "
          "failure, two views with different aliases")
    G = EXP["extra_groups"]
    g = G["absent_b"]
    cases, maps = one_view(g)
    _, _, s = summarize_group(S, *build_group(maps, cases, g["target"]))
    v = g["view_id"]
    ok = all(s["concentration"][f"{v}/{f}"]["always_alias"]["offered"] == 0
             and s["concentration"][f"{v}/{f}"]["always_alias"]["source_is_alias"] == 0
             and r_eq(s["concentration"][f"{v}/{f}"]["always_alias"]["agreement"], g["expected"]["always_alias"]["agreement"])
             and s["concentration"][f"{v}/{f}"]["source_alias"] == g["expected"]["source_alias"]
             and s["models"][f"{v}/{f}"]["C"] == g["expected"]["models_C"][f] for f in FORMATS)
    check("absent B alias: offered 0, source-is-B 0, agreement 0/1; the source alias is A", ok)
    g = G["zero_object_choices"]
    cases, maps = one_view(g)
    _, _, s = summarize_group(S, *build_group(maps, cases, g["target"]))
    ok = True
    for f in FORMATS:
        m, w = s["models"][f"{v}/{f}"], g["expected"]["models"][f]
        ok &= all(m[k] == w[k] for k in ("N", "completed", "object_choices", "ask", "C")) \
            and all(r_eq(m[k], w[k]) for k in ("C_over_N", "C_over_completed", "C_over_object_choices"))
    check("zero object choices: C/object choices is 0/0 with a null value, never 0 or NaN; C/N stays 0/2", ok,
          str(s["models"][f"{v}/{FORMATS[0]}"]))
    check("zero object choices: ASK against no_match counted 2 per format", all(
        s["policy_crosstab"][f"{v}/{f}"]["counts"]["model_choice_ask"]["no_match"] == 2 for f in FORMATS))
    g = G["planned_exclusion"]
    cases, maps = one_view(g)
    scores, comps, s = summarize_group(S, *build_group(maps, cases, g["target"]))
    w = g["expected"]
    ok = True
    for f in FORMATS:
        m = s["models"][f"{v}/{f}"]
        ok &= all(m[k] == x for k, x in w["models"][f].items() if not isinstance(x, list)) \
            and all(r_eq(m[k], x) for k, x in w["models"][f].items() if isinstance(x, list))
    check("planned exclusion: in N, separately counted, unscored, and never an ASK", ok
          and s["models"][f"{v}/{FORMATS[1]}"]["ask"] == 0, str(s["models"][f"{v}/{FORMATS[1]}"]))
    ex = [x for x in scores if x["technical_status"] == "context_budget_exceeded"]
    check("the excluded row is unscored with a null agreement and no choice", len(ex) == 1
          and ex[0]["score_status"] == w["excluded_row"]["score_status"] and ex[0]["source_target_selected"] is None
          and ex[0]["choice_code"] is None and ex[0]["model_choice"] is None)
    p = s["format_pairs"][v]
    check("planned exclusion: the pair is unscored and not comparable; the difference is -50.0 points",
          all(p[k] == x for k, x in w["format_pairs"].items() if k != "difference_pp")
          and p["difference_pp"]["value"] == w["format_pairs"]["difference_pp"], str(p)[:200])
    x = s["model_vs_rules"][f"{v}/{FORMATS[1]}"]
    check("planned exclusion against rules: both 1, unscored 1", all(x[k] == n for k, n in w["augmented_vs_rules"].items()))
    c = s["concentration"][f"{v}/{FORMATS[1]}"]
    check("planned exclusion: source aliases count every planned request; chosen aliases only scored ones",
          c["source_alias"] == w["augmented_source_alias"] and c["chosen_alias"] == w["augmented_chosen_alias"]
          and c["source_by_chosen"] == {"B": {"B": 1, "unscored": 1}}, str(c["source_by_chosen"]))
    g = G["rules_technical_failure"]
    cases, maps = one_view(g)
    _, comps, s = summarize_group(S, *build_group(maps, cases, g["target"]))
    r, w = s["rules"][v], g["expected"]
    check("resolver technical failure: kept in N, counted apart, never a semantic outcome; R 1, C 1, C/N 1/2, C/R 1/1",
          r["N"] == 2 and r["R"] == 1 and r["C"] == 1 and r_eq(r["C_over_N"], w["rules"]["C_over_N"])
          and r_eq(r["C_over_R"], w["rules"]["C_over_R"]) and r["technical_failures"]["budget_exceeded"] == 1
          and sum(r["outcomes"].values()) == 1, str(r)[:300])
    check("resolver technical failure: the pair against each format is unscored", all(
        s["model_vs_rules"][f"{v}/{f}"]["both"] == 1 and s["model_vs_rules"][f"{v}/{f}"]["unscored"] == 1 for f in FORMATS))
    check("resolver technical failure: model object against budget_exceeded in the cross-tab", all(
        nonzero(s["policy_crosstab"][f"{v}/{f}"]["counts"]["model_choice_object"]) == w["crosstab"]["model_choice_object"]
        for f in FORMATS))
    check("resolver technical failure: its comparison row is flagged and its agreement is null",
          comps[0]["rules"]["source_target_selected"] is None and "rules_technical_failure" in comps[0]["flags"])
    g = G["two_views"]
    cases = [{"id": "V1", "relation": "near", "choices": g["choices"], "rules": g["rules"]}]
    scores, comps, s = summarize_group(S, *build_group(g["views"], cases, g["target"]))
    w = g["expected"]
    check("two views: 1 parent, 2 parent/view pairs, 4 requests; rules counted once per view",
          all(s["population"][k] == n for k, n in w["population"].items())
          and all(s["rules"][vv]["N"] == n for vv, n in w["rules_N"].items()))
    check("two views: C per view and format follows each view's own alias mapping",
          all(s["models"][k]["C"] == n for k, n in w["C"].items()), str({k: s["models"][k]["C"] for k in s["models"]}))
    check("two views: the target's alias is B in the full view and A in the source-known view",
          all(s["concentration"][f"{vv}/{f}"]["source_alias"] == w["source_alias"][vv] for vv in g["views"] for f in FORMATS)
          and all(r_eq(s["concentration"][f"{vv}/{f}"]["always_alias"]["agreement"], w["always_alias"][vv])
                  for vv in g["views"] for f in FORMATS))
    check("two views: format pairs per view", all(all(s["format_pairs"][vv][k] == n for k, n in x.items())
                                                 for vv, x in w["format_pairs"].items()))
    e, res, rl, src = build_group({"full_inventory": [["A", "obj_007"], ["B", "obj_008"], ["K", "ASK"]]},
                                  [{"id": "U1", "relation": "near", "choices": {"full_inventory": {f: "A" for f in FORMATS}},
                                    "rules": {"full_inventory": ["resolved", "obj_008"]}}], "obj_008", unknown=["obj_007"])
    _, comps, s = summarize_group(S, e, res, rl, src)
    e2, res2, rl2, src2 = build_group({"full_inventory": [["A", "obj_007"], ["B", "obj_008"], ["K", "ASK"]]},
                                      [{"id": "U2", "relation": "near", "choices": {"full_inventory": {f: "A" for f in FORMATS}},
                                        "rules": {"full_inventory": ["resolved", "obj_008"]}}], "obj_008")
    _, _, s2 = summarize_group(S, e2, res2, rl2, src2)
    check("the unknown-category diagnostic follows the scene's field state, not the object ID",
          all(s["concentration"][f"full_inventory/{f}"]["unknown_category_choices"] == 1 for f in FORMATS)
          and all(s2["concentration"][f"full_inventory/{f}"]["unknown_category_choices"] == 0 for f in FORMATS)
          and any(x.startswith("unknown_category_choice:") for x in comps[0]["flags"]))


def check_score_rechecks(S):
    print("-- saved decisions rechecked against their own offered scores")
    proto = importlib.import_module("grounding.inference.iref_vla").load_protocol()
    CH = importlib.import_module("grounding.inference.iref_vla.choices")

    def row(logits, code, reason, tied=()):
        mapping = [[c, "ASK" if c == "K" else f"obj_{i:03d}", 32 + "ABCDEFGHIJK".index(c)] for i, c in enumerate(logits)]
        offered = list(logits.values())
        shares = CH.restricted_shares(offered)
        lse = 30.0
        return {"technical_status": "completed", "mapping": mapping, "choice_code": code, "selection_reason": reason,
                "tied_codes": list(tied), "model_choice": "model_choice_ask" if code == "K" else "model_choice_object",
                "choice_object_id": None if code == "K" else dict((m[0], m[1]) for m in mapping)[code],
                "scores": [{"code": m[0], "target": m[1], "token_id": m[2], "logit": x, "log_prob": x - lse,
                            "restricted_share": s} for m, x, s in zip(mapping, offered, shares)],
                "top_restricted_share": max(shares), "logit_margin": sorted(offered, reverse=True)[0] - sorted(offered, reverse=True)[1]}
    t = EXP["score_checks"]["tie_counterexample"]
    good = row(t["logits"], t["expected_code"], t["expected_reason"], t["expected_tied"])
    check("tie counterexample: K with exact_score_tie is accepted although K is not tied",
          S.score_problems(good, proto, EXP["tolerance"]) == [], str(S.score_problems(good, proto, EXP["tolerance"])))
    bad = row(t["logits"], "A", "max_offered_logit")
    check("tie counterexample: a stored A with max_offered_logit is rejected", S.score_problems(bad, proto, EXP["tolerance"]) != [])
    c = EXP["score_checks"]["contradiction"]
    bad = row(c["logits"], c["stored_code"], "max_offered_logit")
    check("a stored choice that contradicts its offered scores is rejected", S.score_problems(bad, proto, EXP["tolerance"]) != [])
    fine = row(c["logits"], "B", "max_offered_logit")
    check("the consistent decision for the same scores is accepted", S.score_problems(fine, proto, EXP["tolerance"]) == [])
    tampered = copy.deepcopy(fine)
    tampered["scores"][0]["restricted_share"] += 1e-6
    check("a restricted share changed by 1e-6 is rejected (tolerance 1e-10)", S.score_problems(tampered, proto, EXP["tolerance"]) != [])
    tampered = copy.deepcopy(fine)
    tampered["logit_margin"] += 1e-6
    check("a changed logit margin is rejected", S.score_problems(tampered, proto, EXP["tolerance"]) != [])
    excl = {"technical_status": "context_budget_exceeded", "mapping": fine["mapping"], "model_choice": "model_choice_ask",
            "choice_code": "K", "choice_object_id": None, "selection_reason": None, "scores": None,
            "top_restricted_share": None, "logit_margin": None}
    check("a context exclusion carrying an ASK is rejected: exclusions never become ASK",
          S.score_problems(excl, proto, EXP["tolerance"]) != [])


# --------------------------------------------------------------------------------------- the file chain
def pilot_helpers():
    return importlib.import_module("grounding.tests.test_iref_vla_pilot")


def scoring_bundle(tmp: Path, A):
    """The accepted A2.2c audit and A2.2d preparation (token double) over fixture scene F1 and A2.2b's reviewed commands."""
    sub = importlib.import_module("grounding.subscenes.iref_vla.audit")
    prep = importlib.import_module("grounding.preparation.iref_vla")
    mi, ro = tmp / "in" / "model_inputs", tmp / "in" / "reference_only"
    (mi / "commands").mkdir(parents=True)
    ro.mkdir(parents=True)
    (mi / "scene.annotated.json").write_text(json.dumps(A.ev("scene.f1.json")), encoding="utf-8")
    (mi / "category-map.json").write_text(json.dumps(A.ev("category-map.fixture.json")), encoding="utf-8")
    (ro / "inventory-views.json").write_text(json.dumps(A.ev("views.f1.json")), encoding="utf-8")
    for c in A.ev("commands.f1.json"):
        (mi / "commands" / f"{c['command_id']}.json").write_text(json.dumps(c), encoding="utf-8")
    paths = {"scene": mi / "scene.annotated.json", "commands": mi / "commands", "category_map": mi / "category-map.json",
             "inventory_views": ro / "inventory-views.json"}
    sub.run_audit(**paths, out=tmp / "audit", sample_only=False)

    class Char:
        identity, kind = "test_double.char", "test_double"

        def encode(self, text):
            return [ord(ch) for ch in text]
    prep.run_preparation(**paths, selection_audit=tmp / "audit", relation_config=REL_CFG, direction_config=DIR_CFG,
                         tokenizer_dir=A.TOK_DIR, model_description=A.MODEL_DESC, out=tmp / "bundle", tokenizer=Char(),
                         sample_only=False)
    return tmp / "bundle"


def annotation_bundle(path, targets, *, extra=True, scene_id="fixture.iref_eval.f1"):
    entries = []
    rel = {"fixture.iref_eval.f1.r14": "closest", "fixture.iref_eval.f1.r16": "closest", "fixture.iref_eval.f1.r17": "farthest"}
    for cid, t in targets.items():
        ts = t if isinstance(t, list) else [t]
        for i, tt in enumerate(ts):
            entries.append({"annotation_id": f"{cid}.a{i:03d}", "command_id": cid, "source_region_id": "0",
                            "source_annotation_index": i, "source_payload": {"relation": rel.get(cid, "near"), "target_size_used": "",
                                                                             "anchors": {"anchor_1": {"size_used": ""}}},
                            "mapped_references": {"target": tt, "anchors": {"anchor_1": "obj_001"}, "distractors": []}})
    if extra:
        entries.append({"annotation_id": "fixture.iref_eval.f1.r99.a000", "command_id": "fixture.iref_eval.f1.r99",
                        "source_region_id": "0", "source_annotation_index": 0,
                        "source_payload": {"relation": "near", "target_size_used": "", "anchors": {}},
                        "mapped_references": {"target": "obj_007", "anchors": {"anchor_1": "obj_001"}, "distractors": []}})
    Path(path).write_text(json.dumps({"schema_version": 1, "record_type": "iref_annotations", "scene_id": scene_id,
                                      "source_id": "fixture_source", "source_commit": "0" * 40, "statement_sha256": "0" * 64,
                                      "entries": entries}, indent=2), encoding="utf-8")
    return Path(path)


TARGETS = {"fixture.iref_eval.f1.r14": "obj_002", "fixture.iref_eval.f1.r16": "obj_002", "fixture.iref_eval.f1.r17": "obj_004"}


class Watch:
    """Controlled access: opening a forbidden path raises PermissionError; every opened path is recorded."""

    def __init__(self, forbidden):
        self.forbidden = [Path(p).resolve() for p in forbidden]
        self.opened, self.denied = [], []

    def __enter__(self):
        self.real_open, self.real_io = builtins.open, io.open

        def watch(file, *a, **k):
            p = Path(file).resolve() if isinstance(file, (str, Path)) else None
            if p is not None:
                self.opened.append(str(p))
                if any(p == f or f in p.parents for f in self.forbidden):
                    self.denied.append(str(p))
                    raise PermissionError(f"access denied by the test: {p}")
            return self.real_io(file, *a, **k)
        io.open = builtins.open = watch
        return self

    def __exit__(self, *exc):
        io.open, builtins.open = self.real_io, self.real_open


class FailOnCall:
    """Every binding of the parser, resolver, relation, serializer and inference entry points raises when called."""
    TARGETS = [("grounding.evaluation.iref_vla.parse", "parse"), ("grounding.evaluation.iref_vla.parse", "parse_record"),
               ("grounding.evaluation.iref_vla.predict", "predict"), ("grounding.evaluation.iref_vla.predict", "run_predict"),
               ("grounding.resolution.resolve", "resolve"), ("grounding.serialization.serializer", "serialize"),
               ("grounding.relations.predicates", "load_config"), ("grounding.relations.directions", "load_direction_config"),
               ("grounding.inference.iref_vla.run", "run_pilot"), ("grounding.inference.iref_vla.prepare", "prepare_requests"),
               ("grounding.inference.iref_vla.model", "checkpoint_evidence"),
               ("grounding.preparation.iref_vla.tokens", "load_pinned_tokenizer"),
               ("grounding.preparation.iref_vla.prepare", "run_preparation")]
    METHODS = [("grounding.relations.predicates", "Relations", "__init__"),
               ("grounding.relations.directions", "DirectionalRelations", "__init__"),
               ("grounding.inference.iref_vla.model", "TorchModel", "load")]

    def __init__(self):
        self.calls, self.saved = [], []

    def __enter__(self):
        for mod, name in self.TARGETS:
            real = getattr(importlib.import_module(mod), name)

            def boom(*a, _n=f"{mod}.{name}", **k):
                self.calls.append(_n)
                raise AssertionError(f"forbidden call during scoring: {_n}")
            for m in [x for n, x in list(sys.modules.items()) if n.startswith("grounding") and x is not None]:
                for attr, val in list(vars(m).items()):
                    if val is real:
                        self.saved.append((m, attr, val))
                        setattr(m, attr, boom)
        for mod, cls, meth in self.METHODS:
            c = getattr(importlib.import_module(mod), cls)
            real = c.__dict__[meth]

            def boom2(*a, _n=f"{mod}.{cls}.{meth}", **k):
                self.calls.append(_n)
                raise AssertionError(f"forbidden call during scoring: {_n}")
            self.saved.append((c, meth, real))
            setattr(c, meth, boom2)
        return self

    def __exit__(self, *exc):
        for obj, attr, val in reversed(self.saved):
            setattr(obj, attr, val)


def rehash_pilot(folder: Path):
    """Make tampered pilot results internally consistent again, so that only the new cross-checks can catch them."""
    RUN = importlib.import_module("grounding.inference.iref_vla.run")
    m = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    rows = jsonl(folder / "results.jsonl")
    s = RUN.summarize(rows, m["model_info"], m["canary"], {"parents": m["counts"]["parents"], "requests": m["counts"]["requests"]})
    E = importlib.import_module("grounding.evaluation.iref_vla.protocol")
    files = {"results.jsonl": E.encode_jsonl(rows), "summary.json": E.encode_json(s),
             "report.md": RUN.render_report(s, m).encode("utf-8")}
    for k, v in files.items():
        (folder / k).write_bytes(v)
    m["outputs"] = {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
    (folder / "manifest.json").write_bytes(E.encode_json(m))


def rehash_outputs(folder: Path):
    m = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    m["outputs"] = {k: sha(folder / k) for k in m["outputs"]}
    E = importlib.import_module("grounding.evaluation.iref_vla.protocol")
    (folder / "manifest.json").write_bytes(E.encode_json(m))


def check_file_chain(S):
    print("-- the file chain: fixture bundle, frozen requests, a labelled fake model, the baseline and scoring")
    A = pilot_helpers()
    P = importlib.import_module("grounding.inference.iref_vla")
    BL = importlib.import_module("grounding.evaluation.iref_vla_pilot.baseline")
    SC = importlib.import_module("grounding.evaluation.iref_vla_pilot.score")
    RES = importlib.import_module("grounding.resolution.resolve")
    RV = importlib.import_module("grounding.resolution.validate")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle = scoring_bundle(tmp, A)
        proto_path, selected = A.fixture_protocol(tmp, bundle, P)
        check("the fixture selects the three reviewed commands", sorted(selected) == sorted(TARGETS), str(selected))
        A.prepare(P, bundle, proto_path, tmp / "req")
        model_dir = A.fake_checkpoint(tmp / "hf", P.load_protocol())
        A.run(P, tmp / "req", tmp / "pilot", model_dir)
        ann = annotation_bundle(tmp / "annotations.json", TARGETS)
        common = dict(requests=tmp / "req", bundle=bundle)
        real, calls = RES.resolve, []

        def counting(scene, command, query, **kw):
            calls.append((RV.canonical_sha256(scene), RV.canonical_sha256(command), scene["scene_id"], command["command_id"]))
            return real(scene, command, query, **kw)
        RES.resolve = counting
        try:
            with Watch([ann, tmp / "pilot"]) as w:
                probe = outcome(lambda: open(ann, "rb").read())
                probe2 = outcome(lambda: (tmp / "pilot" / "results.jsonl").read_bytes())
                w.denied.clear()
                res = outcome(lambda: BL.run_baseline(**common, relation_config=REL_CFG, direction_config=DIR_CFG,
                                                      out=tmp / "rules", sample=False))
        finally:
            RES.resolve = real
        check("the access instrumentation is live: the annotations and pilot results cannot be opened inside it",
              all(x[0] == "crashed" and "PermissionError" in x[1] for x in (probe, probe2)), f"{probe} {probe2}")
        check("the baseline completes with the annotations and the model outputs made inaccessible", res[0] == "ok"
              and not w.denied, f"{res} denied {w.denied[:2]}")
        index = {(r["parent_command_id"], r["view_id"]): r for r in jsonl(bundle / "preparation-index.jsonl")}
        rules = jsonl(tmp / "rules" / "rules.jsonl") if res[0] == "ok" else []
        want = [(index[(p, v)]["scene_sha256"], index[(p, v)]["command_sha256"]) for p in selected
                for v in ("full_inventory", "source_known_nyu")]
        check("the resolver ran once per parent and view (6 calls for 3 parents), never per format", len(calls) == 6
              and len(rules) == 6, f"{len(calls)} calls")
        check("each call received exactly the derived scene and command of the preparation index, unchanged",
              [(a, b) for a, b, _, _ in calls] == want)
        rc = EXP["reviewed_resolver_cases"]["commands"]
        got = {(r["parent_command_id"], r["view_id"]): (r["outcome"], r["target_id"]) for r in rules}
        for kind_, keys in EXP["reviewed_resolver_cases"]["kinds"].items():
            for key in keys:
                cid, view = key.split("/")
                check(f"reviewed {kind_.replace('_', '-')} case {cid.split('.')[-1]} ({view}): {rc[cid][view]}",
                      list(got.get((cid, view), (None, None))) == rc[cid][view], str(got.get((cid, view))))
        check("reviewed case r16 (full view): resolved obj_002",
              list(got.get(("fixture.iref_eval.f1.r16", "full_inventory"), (None, None))) == rc["fixture.iref_eval.f1.r16"]["full_inventory"])
        check("the rules folder reads back cleanly", res[0] == "ok" and BL.verify_rules_dir(tmp / "rules") == [],
              str(BL.verify_rules_dir(tmp / "rules")[:2]))
        before = {n: tree_hashes(p) for n, p in (("pilot", tmp / "pilot"), ("req", tmp / "req"), ("bundle", bundle),
                                                 ("rules", tmp / "rules"), ("ann", ann))}
        args = dict(pilot=tmp / "pilot", requests=tmp / "req", bundle=bundle, rules=tmp / "rules", annotations=ann,
                    sample=False, fixture_scene_id="fixture.iref_eval.f1")
        with FailOnCall() as f:
            probe = outcome(lambda: importlib.import_module("grounding.resolution.resolve").resolve({}, {}, {}))
            probe2 = outcome(lambda: importlib.import_module("grounding.evaluation.iref_vla.parse").parse_record("p", "t", categories=[], colours=[]))
            probe3 = outcome(lambda: importlib.import_module("grounding.relations.predicates").Relations(None))
            f.calls.clear()
            res = outcome(lambda: SC.run_score(**args, out=tmp / "scores"))
        check("the fail-on-call instrumentation is live: resolver, parser and relation probes raise inside it",
              all(x[0] == "crashed" and "forbidden call" in x[1] for x in (probe, probe2, probe3)), f"{probe} {probe2} {probe3}")
        check("scoring completes with the parser, resolver, relations, serializer and inference made to fail on call",
              res[0] == "ok" and not f.calls, f"{res} calls {f.calls[:3]}")
        after = {n: tree_hashes(p) for n, p in (("pilot", tmp / "pilot"), ("req", tmp / "req"), ("bundle", bundle),
                                                ("rules", tmp / "rules"), ("ann", ann))}
        check("every input byte is unchanged by the baseline and scoring", before == after)
        check("the score folder reads back: hashes, schemas, cardinalities, summary arithmetic and report",
              res[0] == "ok" and SC.verify_score_dir(tmp / "scores") == [], str(SC.verify_score_dir(tmp / "scores")[:2]))
        scores = jsonl(tmp / "scores" / "scores.jsonl") if res[0] == "ok" else []
        comps = jsonl(tmp / "scores" / "comparisons.jsonl") if res[0] == "ok" else []
        results = {r["request_id"]: r for r in jsonl(tmp / "pilot" / "results.jsonl")}
        check("12 score rows in request order and 6 comparison rows", len(scores) == 12 and len(comps) == 6
              and [s["request_index"] for s in scores] == list(range(1, 13)))
        check("every score row reproduces its saved choice", all(s["choice_code"] == results[s["request_id"]]["choice_code"]
                                                                and s["choice_object_id"] == results[s["request_id"]]["choice_object_id"]
                                                                for s in scores))
        check("every score row's agreement follows its own target and mapping",
              all(s["source_target_selected"] == (s["model_choice"] == "model_choice_object" and s["choice_object_id"]
                                                   == TARGETS[s["parent_command_id"]]) for s in scores)
              and all(s["source_alias"] == next(m[0] for m in results[s["request_id"]]["mapping"] if m[1] == s["source_target_id"])
                      for s in scores))
        rmap = {(r["parent_command_id"], r["view_id"]): r for r in rules}
        check("rules agreement only for a resolved source target, and one rules record per parent and view in both formats' rows",
              all(c["rules"]["source_target_selected"] == (rmap[(c["parent_command_id"], c["view_id"])]["outcome"] == "resolved"
                                                           and rmap[(c["parent_command_id"], c["view_id"])]["target_id"]
                                                           == c["source"]["target_id"]) for c in comps)
              and all(s["rules_record"] == {"selection_rank": next(c["selection_rank"] for c in comps
                                                                   if c["parent_command_id"] == s["parent_command_id"]),
                                            "view_id": s["view_id"]} for s in scores))
        unknown = [s for s in scores if s["choice_category_state"] == "unknown"]
        check("unknown-category choices are read from the scene records (obj_008 is unknown in F1)",
              all(s["choice_object_id"] == "obj_008" for s in unknown)
              and all(s["choice_category_state"] == "known" for s in scores if s["choice_object_id"] not in (None, "obj_008")))
        summ = json.loads((tmp / "scores" / "summary.json").read_text(encoding="utf-8")) if res[0] == "ok" else {}
        check("the fixture mode is labelled in the summary and the report", summ.get("mode") == "fixture"
              and "Mode: fixture" in (tmp / "scores" / "report.md").read_text(encoding="utf-8"))
        res = outcome(lambda: SC.run_score(**args, out=tmp / "scores"))
        check("an existing score folder is refused", res == ("rejected", ["E_EVAL_OUTPUT_EXISTS"]), str(res))
        res = outcome(lambda: BL.run_baseline(**common, relation_config=REL_CFG, direction_config=DIR_CFG,
                                              out=tmp / "rules", sample=False))
        check("an existing rules folder is refused", res == ("rejected", ["E_EVAL_OUTPUT_EXISTS"]), str(res))
        res = outcome(lambda: SC.run_score(**args, out=tmp / "pilot" / "inside"))
        check("an output inside an input is refused", res[0] == "rejected", str(res))
        OUT = importlib.import_module("grounding.evaluation.iref_vla.output")
        real_write, n = OUT._write_file, [0]

        def flaky(path, data):
            n[0] += 1
            if n[0] == 2:
                raise OSError("disk full (test double)")
            return real_write(path, data)
        OUT._write_file = flaky
        try:
            res = outcome(lambda: SC.run_score(**args, out=tmp / "scores-io"))
        finally:
            OUT._write_file = real_write
        check("a failed write publishes nothing and leaves no partial folder", res[0] == "write_failed"
              and not (tmp / "scores-io").exists() and not list(tmp.glob(".scores-io.partial-*")), str(res))
        ann2 = annotation_bundle(tmp / "annotations-2.json", dict(TARGETS, **{"fixture.iref_eval.f1.r16": "obj_003"}))
        res = outcome(lambda: SC.run_score(**dict(args, annotations=ann2), out=tmp / "scores-2"))
        s2 = jsonl(tmp / "scores-2" / "scores.jsonl") if res[0] == "ok" else []
        model_keys = ("request_id", "choice_code", "choice_object_id", "model_choice", "score_status", "rules_outcome",
                      "rules_target_id")
        check("changing one annotation changes only that command's source fields, never the saved model or rules results",
              res[0] == "ok" and [{k: s[k] for k in model_keys} for s in s2] == [{k: s[k] for k in model_keys} for s in scores]
              and all((s["source_target_id"] == "obj_003") == (s["parent_command_id"] == "fixture.iref_eval.f1.r16") for s in s2)
              and tree_hashes(tmp / "pilot") == before["pilot"] and tree_hashes(tmp / "rules") == before["rules"], str(res))
        for name, targets, code in (
                ("a selected command without annotations", {k: v for k, v in TARGETS.items() if not k.endswith("r17")}, "E_SCORING_REFERENCE"),
                ("contradictory targets for one command", dict(TARGETS, **{"fixture.iref_eval.f1.r14": ["obj_002", "obj_003"]}), "E_SCORING_REFERENCE"),
                ("a target absent from a selected subscene", dict(TARGETS, **{"fixture.iref_eval.f1.r14": "obj_007"}), "E_SCORING_REFERENCE"),
                ("consistent repeated annotations", dict(TARGETS, **{"fixture.iref_eval.f1.r14": ["obj_002", "obj_002"]}), None)):
            p = annotation_bundle(tmp / f"ann-{len(list(tmp.iterdir()))}.json", targets)
            out_ = tmp / f"s-{len(list(tmp.iterdir()))}"
            res = outcome(lambda: SC.run_score(**dict(args, annotations=p), out=out_))
            if code is None:
                c2 = jsonl(out_ / "comparisons.jsonl") if res[0] == "ok" else []
                check(f"{name} pass and count once (annotation count 2, one comparison per view)", res[0] == "ok" and
                      [c["source"]["annotation_count"] for c in c2 if c["parent_command_id"].endswith("r14")] == [2, 2], str(res))
            else:
                check(f"{name} is rejected ({code}) and nothing is published", res == ("rejected", [code]) and not out_.exists(), str(res))
        check("unrelated valid annotations in the full bundle are accepted (r99 is not selected)",
              "fixture.iref_eval.f1.r99" in (tmp / "annotations.json").read_text(encoding="utf-8"))
        p = tmp / "ann-bad-schema.json"
        d = json.loads(ann.read_text(encoding="utf-8"))
        d["entries"][0]["source_annotation_index"] = 0.0
        p.write_text(json.dumps(d), encoding="utf-8")
        res = outcome(lambda: SC.run_score(**dict(args, annotations=p), out=tmp / "s-float"))
        check("a whole-valued float annotation index is rejected (E_SCORING_REFERENCE)", res == ("rejected", ["E_SCORING_REFERENCE"]), str(res))
        d = json.loads(ann.read_text(encoding="utf-8"))
        d["entries"].append(copy.deepcopy(d["entries"][0]))
        p.write_text(json.dumps(d), encoding="utf-8")
        res = outcome(lambda: SC.run_score(**dict(args, annotations=p), out=tmp / "s-dup"))
        check("a repeated annotation ID is rejected (E_SCORING_REFERENCE)", res == ("rejected", ["E_SCORING_REFERENCE"]), str(res))
        res = outcome(lambda: SC.run_score(**dict(args, fixture_scene_id="another.scene"), out=tmp / "s-scene"))
        check("annotations of another scene are rejected (E_SCORING_REFERENCE)", res == ("rejected", ["E_SCORING_REFERENCE"]), str(res))
        tamper_cases(S, SC, BL, tmp, args)
        cli_cases(tmp, args, common)


def tamper_cases(S, SC, BL, tmp, args):
    print("-- tampered, malformed and mismatched inputs fail before anything is published")

    def case(name, edit, code, rehash=None, key="pilot"):
        src = args[key]
        t = tmp / f"t{len(list(tmp.iterdir()))}"
        shutil.copytree(src, t)
        edit(t)
        if rehash:
            rehash(t)
        out_ = tmp / f"o{len(list(tmp.iterdir()))}"
        res = outcome(lambda: SC.run_score(**dict(args, **{key: t}), out=out_))
        check(f"{name}: rejected ({code}), nothing published", res == ("rejected", [code]) and not out_.exists(), str(res))

    def lines(t, name, fn):
        rows = (t / name).read_text(encoding="utf-8").splitlines(keepends=True)
        (t / name).write_text("".join(fn(rows)), encoding="utf-8")

    def rows_edit(fn):
        def edit(t):
            rows = jsonl(t / "results.jsonl")
            fn(rows)
            write_jsonl(t / "results.jsonl", rows)
        return edit
    case("a repeated result row", lambda t: lines(t, "results.jsonl", lambda r: r + r[-1:]), "E_SCORING_INTEGRITY")
    case("a missing result row", lambda t: lines(t, "results.jsonl", lambda r: r[:-1]), "E_SCORING_INTEGRITY")

    def extra(rows):
        x = copy.deepcopy(rows[-1])
        x.update(request_index=len(rows) + 1, request_id=f"q{len(rows) + 1:03d}")
        rows.append(x)
    case("an extra result row (re-hashed)", rows_edit(extra), "E_SCORING_INTEGRITY", rehash_pilot)

    def swap_format(rows):
        rows[0]["format"], rows[1]["format"] = rows[1]["format"], rows[0]["format"]
    case("a mismatched parent/view/format key (re-hashed)", rows_edit(swap_format), "E_SCORING_INTEGRITY", rehash_pilot)

    def swap_alias(rows):
        r = next(x for x in rows if x["technical_status"] == "completed" and len(x["mapping"]) >= 3)
        (a, b) = r["mapping"][0][1], r["mapping"][1][1]
        r["mapping"][0][1], r["mapping"][1][1] = b, a
        r["scores"][0]["target"], r["scores"][1]["target"] = b, a
        if r["choice_object_id"] in (a, b):
            r["choice_object_id"] = b if r["choice_object_id"] == a else a
    case("a swapped alias-object association (re-hashed)", rows_edit(swap_alias), "E_SCORING_INTEGRITY", rehash_pilot)

    def contradict(rows):
        r = next(x for x in rows if x["technical_status"] == "completed" and x["choice_code"] != "K")
        other = next(m for m in r["mapping"] if m[0] not in (r["choice_code"], "K"))
        r.update(choice_code=other[0], choice_object_id=other[1])
    case("a stored choice contradicting its offered scores (re-hashed)", rows_edit(contradict), "E_SCORING_INTEGRITY", rehash_pilot)

    def to_exclusion(rows):
        rows[0].update(technical_status="context_budget_exceeded", scores=None, top_restricted_share=None, logit_margin=None)
    case("a context exclusion that keeps a choice (re-hashed)", rows_edit(to_exclusion), "E_SCORING_INTEGRITY", rehash_pilot)

    def stale(t):
        m = json.loads((t / "manifest.json").read_text(encoding="utf-8"))
        m["requests_manifest_sha256"] = "f" * 64
        (t / "manifest.json").write_text(json.dumps(m, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    case("a stale request-manifest hash in the pilot manifest", stale, "E_SCORING_INTEGRITY")
    case("a non-finite score", lambda t: lines(t, "results.jsonl", lambda r: [r[0].replace('"logit":', '"logit":NaN,"x":', 1)] + r[1:]),
         "E_SCORING_INPUT")
    case("a whole-valued float request index", lambda t: lines(t, "results.jsonl",
                                                               lambda r: [r[0].replace('"request_index":1,', '"request_index":1.0,', 1)] + r[1:]),
         "E_SCORING_INPUT")
    case("a result line that is not an object", lambda t: lines(t, "results.jsonl", lambda r: ["[]\n"] + r[1:]), "E_SCORING_INPUT")

    def text(t):
        p = next((t / "model_records" / "commands").iterdir())
        d = json.loads(p.read_text(encoding="utf-8"))
        d["text"] = d["text"] + " "
        p.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    case("an altered command text in the bundle", text, "E_SCORING_INTEGRITY", key="bundle")

    def scene(t):
        p = next((t / "model_records" / "scenes").iterdir())
        d = json.loads(p.read_text(encoding="utf-8"))
        d["objects"][0]["geometry"]["center_m"]["value"][0] += 0.5
        p.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    case("an altered scene in the bundle", scene, "E_SCORING_INTEGRITY", key="bundle")

    def mapping(t):
        rows = jsonl(t / "request-index.jsonl")
        rows[0]["mapping"][0][1], rows[0]["mapping"][1][1] = rows[0]["mapping"][1][1], rows[0]["mapping"][0][1]
        write_jsonl(t / "request-index.jsonl", rows)
    case("an altered mapping in the request index", mapping, "E_SCORING_INTEGRITY", key="requests")

    def rules_float(t):
        rows = jsonl(t / "rules.jsonl")
        rows[0]["selection_rank"] = 1.0
        write_jsonl(t / "rules.jsonl", rows)
    case("a whole-valued float in a rules record (re-hashed)", rules_float, "E_SCORING_INPUT", rehash_outputs, key="rules")

    def rules_version(t):
        m = json.loads((t / "manifest.json").read_text(encoding="utf-8"))
        m["format_version"] = 2
        (t / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    case("an unsupported rules manifest version", rules_version, "E_SCORING_INPUT", key="rules")

    def rules_shape(t):
        rows = (t / "rules.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)
        (t / "rules.jsonl").write_text("[]\n" + "".join(rows[1:]), encoding="utf-8")
    case("a rules line that is not an object (re-hashed)", rules_shape, "E_SCORING_INPUT", rehash_outputs, key="rules")

    def rules_outcome(t):
        rows = jsonl(t / "rules.jsonl")
        rows[0]["outcome"] = "resolved" if rows[0]["outcome"] != "resolved" else "ambiguous"
        write_jsonl(t / "rules.jsonl", rows)
    case("a rules outcome that disagrees with its saved resolution (re-hashed)", rules_outcome, "E_SCORING_INTEGRITY",
         rehash_outputs, key="rules")
    t = tmp / "readback"
    shutil.copytree(tmp / "scores", t)
    s = json.loads((t / "summary.json").read_text(encoding="utf-8"))
    s["models"][next(iter(s["models"]))]["C"] += 1
    (t / "summary.json").write_text(json.dumps(s, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rehash_outputs(t)
    check("readback: a summary count changed and re-hashed still fails its recomputation", SC.verify_score_dir(t) != [])
    t2 = tmp / "readback-2"
    shutil.copytree(tmp / "scores", t2)
    (t2 / "report.md").write_text((t2 / "report.md").read_text(encoding="utf-8") + "edited\n", encoding="utf-8")
    check("readback: an edited report fails its hash", SC.verify_score_dir(t2) != [])


def cli_cases(tmp, args, common):
    print("-- the command line (exit codes)")
    M = importlib.import_module("grounding.evaluation.iref_vla_pilot.__main__")
    out = io.StringIO()
    real = sys.stdout
    sys.stdout = out
    try:
        code = M.main(["score", "--pilot", str(args["pilot"]), "--requests", str(args["requests"]), "--bundle",
                       str(args["bundle"]), "--rules", str(args["rules"]), "--annotations", str(args["annotations"]),
                       "--out", str(tmp / "cli-score")])
        code_b = M.main(["baseline", "--requests", str(common["requests"]), "--bundle", str(common["bundle"]),
                         "--relation-config", str(REL_CFG), "--direction-config", str(DIR_CFG), "--out", str(tmp / "cli-rules")])
    finally:
        sys.stdout = real
    check("the command line serves only the pinned sample: fixture inputs exit 2 with nothing published",
          code == 2 and code_b == 2 and not (tmp / "cli-score").exists() and not (tmp / "cli-rules").exists(), out.getvalue()[-200:])
    SC = importlib.import_module("grounding.evaluation.iref_vla_pilot.score")
    real_run = SC.run_score
    for tech, want in (({"model_planned_exclusions": 0, "model_technical_failures": 0, "rules_technical_failures": 0}, 0),
                       ({"model_planned_exclusions": 1, "model_technical_failures": 0, "rules_technical_failures": 0}, 1),
                       ({"model_planned_exclusions": 0, "model_technical_failures": 0, "rules_technical_failures": 1}, 1)):
        SC.run_score = lambda **k: {"population": {"parents": 1, "requests": 4}, "models": {}, "rules": {}, "technical": tech}
        sys.stdout = io.StringIO()
        try:
            code = M.main(["score", "--pilot", "p", "--requests", "r", "--bundle", "b", "--rules", "x", "--annotations", "a",
                           "--out", "o"])
        finally:
            sys.stdout, SC.run_score = real, real_run
        check(f"exit {want} for technical counts {nonzero(tech) or 'none'} (low agreement is never an exit code)", code == want)


# ------------------------------------------------------------------------------------------- real readback
def arg(name):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv and sys.argv.index(name) + 1 < len(sys.argv) else None


def check_real(S):
    rules, scores, pilot = arg("--rules"), arg("--scores"), arg("--pilot")
    if not (rules and scores):
        return
    print("-- readback acceptance of a real run (no rules or inference recomputed)")
    BL = importlib.import_module("grounding.evaluation.iref_vla_pilot.baseline")
    SC = importlib.import_module("grounding.evaluation.iref_vla_pilot.score")
    R = EXP["real_run"]
    check("the rules folder verifies", BL.verify_rules_dir(Path(rules)) == [], str(BL.verify_rules_dir(Path(rules))[:2]))
    check("the score folder verifies", SC.verify_score_dir(Path(scores)) == [], str(SC.verify_score_dir(Path(scores))[:2]))
    rr, pr = jsonl(Path(rules) / "rules.jsonl"), jsonl(Path(rules) / "parses.jsonl")
    sr, cr = jsonl(Path(scores) / "scores.jsonl"), jsonl(Path(scores) / "comparisons.jsonl")
    sm = json.loads((Path(scores) / "summary.json").read_text(encoding="utf-8"))
    mf = json.loads((Path(scores) / "manifest.json").read_text(encoding="utf-8"))
    check(f"{R['rules_records']} rules records and {R['parse_records']} parses", len(rr) == R["rules_records"] and len(pr) == R["parse_records"])
    check(f"{R['score_rows']} score rows and {R['comparison_rows']} comparison rows", len(sr) == R["score_rows"] and len(cr) == R["comparison_rows"])
    check("pinned-sample mode", sm["mode"] == "pinned_sample" and mf["mode"] == "pinned_sample")
    check("the inputs are the accepted artifacts: pilot outputs, selected list and Windows preparation index",
          mf["inputs"]["pilot"]["files"] == R["pilot_outputs"] and mf["inputs"]["selected_sha256"] == R["selected_sha256"]
          and mf["inputs"]["preparation_index_sha256"] == R["preparation_index_sha256"], json.dumps(mf["inputs"])[:300])
    scored = [s for s in sr if s["score_status"] == "scored"]
    check("128 completed requests, 128 object choices, no ASK, no exclusions",
          len(scored) == R["completed"] and sum(s["model_choice"] == "model_choice_object" for s in scored) == R["object_choices"]
          and sum(s["model_choice"] == "model_choice_ask" for s in scored) == R["ask_choices"])
    check("choice codes B 118 and C 10", dict(Counter(s["choice_code"] for s in scored)) == R["choice_codes"],
          str(Counter(s["choice_code"] for s in scored)))
    unk = [s for s in scored if s["choice_category_state"] == "unknown"]
    check("four unknown-category choices, found from the scene records", len(unk) == R["unknown_category_choices"]
          and sum(c["unknown_category_choices"] for c in sm["concentration"].values()) == R["unknown_category_choices"],
          str(sorted({s["choice_object_id"] for s in unk})))
    check("format agreement: same choice 30/32 (full) and 28/32 (source-known)",
          all([sm["format_pairs"][v]["same_choice"], sm["format_pairs"][v]["N"]] == R["format_agreement"][v] for v in R["format_agreement"]))
    if pilot:
        res = {r["request_id"]: r for r in jsonl(Path(pilot) / "results.jsonl")}
        check("the 128 saved choices are reproduced exactly", len(res) == 128 and all(
            (s["choice_code"], s["choice_object_id"], s["model_choice"]) == (res[s["request_id"]]["choice_code"],
                                                                             res[s["request_id"]]["choice_object_id"],
                                                                             res[s["request_id"]]["model_choice"]) for s in sr))
        check("the pilot's own outputs still have their accepted hashes",
              {n: sha(Path(pilot) / n) for n in R["pilot_outputs"]} == R["pilot_outputs"])


def main() -> int:
    S = importlib.import_module("grounding.evaluation.iref_vla_pilot")
    check_policy_and_metric_fixture(S)
    check_extra_groups(S)
    check_score_rechecks(S)
    check_file_chain(S)
    check_real(S)
    real = bool(arg("--rules") and arg("--scores"))
    print(f"{COUNT[0]} checks; {'FAILED: ' + ', '.join(FAILED) if FAILED else 'all checks passed'}"
          + ("" if real else " (UNIT-ONLY: fixtures and labelled doubles; not real-data acceptance)"))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
