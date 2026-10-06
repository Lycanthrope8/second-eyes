"""A2.3b scoring (D82): the saved pilot choices and the saved rules results against the reference annotations.

score never parses, resolves, evaluates relations, serializes or runs a model: it reads saved outputs, checks every
link and saved decision, then compares. The summary and the report are derived only from the published score and
comparison rows, so a readback recomputes them exactly. Each command counts once per view and format; the rules
result counts once per view; nothing here is a significance test, a format winner or a Gate B decision.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from ..iref_vla import output
from ..iref_vla import protocol as A2PROTOCOL
from . import baseline as BL
from . import inputs as IN
from ..iref_vla.protocol import strict_json
from .policy import (CODES, FORMATS, INPUT, INTEGRITY, MODES, POLICY_ID, RULES_BINS, TECHNICAL, VIEWS, check_overlap,
                     code_hashes, encode_json, encode_jsonl, fail, file_hashes, load_policy, policy_sha256,
                     publish_verified, ratio, read_bytes, read_json, read_jsonl, runtime, schema_issues, sha256)

FILES = ("comparisons.jsonl", "manifest.json", "report.md", "scores.jsonl", "summary.json")
ALIAS = "B"
LIMITATIONS = [
    "One previously inspected IRef-VLA development room (ScanNet scene0010_01) with annotated geometry: not held-out "
    "data, and the inventory views are not the restricted evidence profile.",
    "Commands were selected through the existing rules parser (A2.2b) and the category-complete subscene policy "
    "(A2.2c): 32 paired commands, not 128 independent examples, and not a representative sample.",
    "The alias order is fixed (A..J by object ID, K for ASK); choice concentration is described, not tested for bias.",
    "Source agreement is a development diagnostic, not held-out natural-language grounding accuracy.",
    "Conservative unknown-category handling and the dataset's relation definitions may differ from the source labels.",
    "No viewpoint-dependent language, actual perception noise or broad ASK behaviour is tested.",
    "Restricted shares are not calibrated correctness probabilities; no threshold is tuned here.",
    "No inference rerun, training decision, deployment claim, Paper 1 claim change or Gate B conclusion follows.",
]
INTERPRETATION = (
    "This is one previously inspected IRef-VLA development room with annotated geometry. Its commands were selected "
    "through the existing rules parser and the category-complete subscene policy, and the model's choice codes follow "
    "a fixed alias order. Source agreement is a development diagnostic, not held-out natural-language grounding "
    "accuracy. Conservative unknown handling and the dataset's relation definitions may differ. The sample tests no "
    "viewpoint-dependent language, actual perception noise or broad ASK behaviour. Agreement among object choices can "
    "rise by abstaining, so C/N stays alongside it. No significance test, format winner or Gate B decision is made "
    "here.")


# ------------------------------------------------------------------------------------------------ the rows
def _pair(a, b):
    if a is None or b is None:
        return "unscored"
    return {(True, True): "both", (True, False): "coordinates_only", (False, True): "augmented_only",
            (False, False): "neither"}[(a, b)]


def build_rows(entries, results_by_key, rules_by_key, sources, ask_code="K"):
    """Score rows in request order and one comparison row per parent and view, from verified joined inputs."""
    scores, comparisons = [], []
    for e in entries:
        p, v = e["parent"], e["view"]
        src = sources[p]
        target = src["target_id"]
        unknown = set(e["unknown_ids"])
        rr = rules_by_key[(p, v)]
        rules_sel = None if rr["technical_failure"] else (rr["outcome"] == "resolved" and rr["target_id"] == target)
        models = {}
        for f in FORMATS:
            res = results_by_key[(p, v, f)]
            st = res["technical_status"]
            status = "scored" if st == "completed" else "unscored_planned_exclusion" \
                if st in ("context_budget_exceeded", "empty_scene_bypass") else "unscored_technical_failure"
            scored = status == "scored"
            obj = res["choice_object_id"] if scored else None
            mapping = res["mapping"]
            alias = next(m[0] for m in mapping if m[1] == target and m[0] != ask_code)
            row = {"format_version": 1, "record_type": "iref_pilot_score", "policy_id": POLICY_ID,
                   "request_index": res["request_index"], "request_id": res["request_id"], "selection_rank": e["rank"],
                   "parent_command_id": p, "view_id": v, "format": f, "derived_scene_id": res["derived_scene_id"],
                   "derived_command_id": res["derived_command_id"], "source_document_sha256": res["source_document_sha256"],
                   "prompt_sha256": res["prompt_sha256"], "token_ids_sha256": res["token_ids_sha256"],
                   "input_tokens": res["input_tokens"], "object_count": len(e["object_ids"]),
                   "offered_codes": [m[0] for m in mapping], "source_target_id": target, "source_alias": alias,
                   "source_relation": src["relation"], "annotation_ids": list(src["annotation_ids"]),
                   "annotation_count": src["annotation_count"], "technical_status": st, "score_status": status,
                   "model_choice": res["model_choice"] if scored else None,
                   "choice_code": res["choice_code"] if scored else None, "choice_object_id": obj,
                   "choice_category_state": None if obj is None else ("unknown" if obj in unknown else "known"),
                   "selection_reason": res["selection_reason"] if scored else None,
                   "source_target_selected": None if not scored else (res["model_choice"] == "model_choice_object"
                                                                      and obj == target),
                   "rules_record": {"selection_rank": e["rank"], "view_id": v}, "rules_outcome": rr["outcome"],
                   "rules_target_id": rr["target_id"], "rules_source_target_selected": rules_sel}
            scores.append(row)
            models[f] = {k: row[k] for k in ("request_id", "score_status", "model_choice", "choice_code", "choice_object_id",
                                              "choice_category_state", "source_target_selected")}
        a, b = (models[f] for f in FORMATS)
        same = None if any(m["score_status"] != "scored" for m in (a, b)) else a["choice_code"] == b["choice_code"]
        flags = (["format_disagreement"] if same is False else []) + [
            f"unknown_category_choice:{f}" for f in FORMATS if models[f]["choice_category_state"] == "unknown"] + [
            f"unscored:{f}" for f in FORMATS if models[f]["score_status"] != "scored"] + (
            ["rules_technical_failure"] if rr["technical_failure"] else [])
        comparisons.append({
            "format_version": 1, "record_type": "iref_pilot_comparison", "policy_id": POLICY_ID, "selection_rank": e["rank"],
            "parent_command_id": p, "view_id": v, "text": e["text"], "derived_scene_id": e["derived_scene_id"],
            "derived_command_id": e["derived_command_id"], "object_count": len(e["object_ids"]),
            "unknown_object_ids": sorted(unknown),
            "source": {"target_id": target, "alias": scores[-1]["source_alias"], "relation": src["relation"],
                       "annotation_ids": list(src["annotation_ids"]), "annotation_count": src["annotation_count"]},
            "models": models,
            "rules": {"outcome": rr["outcome"], "technical_failure": rr["technical_failure"], "target_id": rr["target_id"],
                      "candidate_ids": list(rr["candidate_ids"]), "reason_code": rr["reason_code"],
                      "reason_codes": list(rr["reason_codes"]), "source_target_selected": rules_sel},
            "pair": {"source": _pair(a["source_target_selected"], b["source_target_selected"]), "same_choice": same},
            "flags": flags})
    return scores, comparisons


# --------------------------------------------------------------------------------------------- the summary
def _nonzero(counter, order=None) -> dict:
    keys = [k for k in (order or sorted(counter)) if counter.get(k)]
    return {k: counter[k] for k in keys}


def _model_block(rows) -> dict:
    scored = [s for s in rows if s["score_status"] == "scored"]
    obj = [s for s in scored if s["model_choice"] == "model_choice_object"]
    c = sum(s["source_target_selected"] is True for s in rows)
    return {"N": len(rows), "completed": len(scored), "object_choices": len(obj), "ask": len(scored) - len(obj),
            "planned_exclusions": sum(s["score_status"] == "unscored_planned_exclusion" for s in rows),
            "technical_failures": sum(s["score_status"] == "unscored_technical_failure" for s in rows), "C": c,
            "C_over_N": ratio(c, len(rows)), "C_over_completed": ratio(c, len(scored)),
            "C_over_object_choices": ratio(c, len(obj))}


def _rules_block(comps) -> dict:
    outcomes = Counter(c["rules"]["outcome"] for c in comps)
    r_ = outcomes["resolved"]
    c_ = sum(c["rules"]["source_target_selected"] is True for c in comps)
    return {"N": len(comps), "outcomes": {o: outcomes[o] for o in RULES_BINS if o not in TECHNICAL},
            "technical_failures": {t: outcomes[t] for t in TECHNICAL}, "R": r_, "C": c_,
            "C_over_N": ratio(c_, len(comps)), "C_over_R": ratio(c_, r_),
            "reason_codes_nonexclusive": dict(sorted(Counter(x for c in comps for x in c["rules"]["reason_codes"]).items()))}


def _small(rows) -> dict:
    scored = [s for s in rows if s["score_status"] == "scored"]
    return {"C": sum(s["source_target_selected"] is True for s in rows),
            "object_choices": sum(s["model_choice"] == "model_choice_object" for s in scored),
            "ask": sum(s["model_choice"] == "model_choice_ask" for s in scored),
            "unscored": len(rows) - len(scored)}


def summarize(scores, comparisons, mode) -> dict:
    """Every count from the published rows alone; ratios keep their parts and a zero denominator gives null."""
    views = [v for v in VIEWS if any(c["view_id"] == v for c in comparisons)]
    first = {}
    for c in comparisons:
        first.setdefault(c["parent_command_id"], c)
    mult = Counter(c["source"]["annotation_count"] for c in first.values())
    population = {"parents": len(first), "parent_views": len(comparisons), "requests": len(scores), "views": views,
                  "formats": list(FORMATS),
                  "annotation_records": sum(c["source"]["annotation_count"] for c in first.values()),
                  "annotation_multiplicity": {str(k): mult[k] for k in sorted(mult)},
                  "note": "each command counts once per view and format; the views are paired, not independent"}
    models, rules, pairs, versus, crosstab, concentration = {}, {}, {}, {}, {}, {}
    for v in views:
        comps = [c for c in comparisons if c["view_id"] == v]
        rules[v] = _rules_block(comps)
        n = len(comps)
        cnt = Counter(c["pair"]["source"] for c in comps)
        same = Counter(c["pair"]["same_choice"] for c in comps)
        for f in FORMATS:
            key = f"{v}/{f}"
            rows = [s for s in scores if s["view_id"] == v and s["format"] == f]
            models[key] = _model_block(rows)
            vc, by_outcome = Counter(), Counter()
            grid = {r: {o: 0 for o in RULES_BINS} for r in ("model_choice_object", "model_choice_ask", "unscored")}
            within = {"same_object": 0, "different_object": 0, "ask": 0, "unscored": 0}
            for c in comps:
                m, r = c["models"][f], c["rules"]
                ms, rs = m["source_target_selected"], r["source_target_selected"]
                if ms is None or rs is None:
                    vc["unscored"] += 1
                else:
                    vc[{(True, True): "both", (True, False): "model_only", (False, True): "rules_only",
                        (False, False): "neither"}[(ms, rs)]] += 1
                    if rs is False:
                        by_outcome[r["outcome"]] += 1
                row = m["model_choice"] if m["score_status"] == "scored" else "unscored"
                grid[row][r["outcome"]] += 1
                if r["outcome"] == "resolved":
                    within["unscored" if row == "unscored" else "ask" if row == "model_choice_ask" else
                           "same_object" if m["choice_object_id"] == r["target_id"] else "different_object"] += 1
            versus[key] = {"both": vc["both"], "model_only": vc["model_only"], "rules_only": vc["rules_only"],
                           "neither": vc["neither"], "unscored": vc["unscored"],
                           "rules_not_source_by_outcome": dict(sorted(by_outcome.items()))}
            crosstab[key] = {"rows": "model outcome", "columns": "rules outcome", "counts": grid,
                             "within_rules_resolved": within}
            scored = [s for s in rows if s["score_status"] == "scored"]
            src_by = {}
            for s in rows:
                col = s["choice_code"] if s["score_status"] == "scored" else "unscored"
                src_by.setdefault(s["source_alias"], Counter())[col] += 1
            concentration[key] = {
                "chosen_alias": _nonzero(Counter(s["choice_code"] for s in scored), CODES),
                "source_alias": _nonzero(Counter(s["source_alias"] for s in rows), CODES),
                "source_by_chosen": {a: _nonzero(src_by[a], CODES + ("unscored",)) for a in CODES if a in src_by},
                "always_alias": {"alias": ALIAS, "offered": sum(ALIAS in s["offered_codes"] for s in rows),
                                 "source_is_alias": sum(s["source_alias"] == ALIAS for s in rows),
                                 "agreement": ratio(sum(s["source_alias"] == ALIAS for s in rows), len(rows)),
                                 "label": "post-observation diagnostic on this sample, not a competitive baseline"},
                "chosen_objects": dict(sorted(Counter(s["choice_object_id"] for s in scored
                                                      if s["choice_object_id"] is not None).items())),
                "unknown_category_choices": sum(s["choice_category_state"] == "unknown" for s in scored)}
        ca, cc = models[f"{v}/{FORMATS[1]}"]["C"], models[f"{v}/{FORMATS[0]}"]["C"]
        pairs[v] = {"N": n, "both": cnt["both"], "coordinates_only": cnt["coordinates_only"],
                    "augmented_only": cnt["augmented_only"], "neither": cnt["neither"], "unscored": cnt["unscored"],
                    "same_choice": same[True], "different_choice": same[False], "not_comparable": same[None],
                    "C_over_N": {f: models[f"{v}/{f}"]["C_over_N"] for f in FORMATS},
                    "difference_pp": {"value": None if n == 0 else 100 * (ca - cc) / n,
                                      "rule": "100 * (C of coordinates_relations_v2 - C of coordinates_v2) / N"}}
    relations = {}
    for label in sorted({c["source"]["relation"] for c in first.values()}):
        parents = {p for p, c in first.items() if c["source"]["relation"] == label}
        relations[label] = {"N": len(parents),
                            "models": {f"{v}/{f}": _small([s for s in scores if s["parent_command_id"] in parents
                                                           and s["view_id"] == v and s["format"] == f])
                                       for v in views for f in FORMATS},
                            "rules": {v: {"R": sum(c["rules"]["outcome"] == "resolved" for c in comparisons
                                                   if c["parent_command_id"] in parents and c["view_id"] == v),
                                          "C": sum(c["rules"]["source_target_selected"] is True for c in comparisons
                                                   if c["parent_command_id"] in parents and c["view_id"] == v),
                                          "technical": sum(c["rules"]["technical_failure"] for c in comparisons
                                                           if c["parent_command_id"] in parents and c["view_id"] == v)}
                                      for v in views}}
    counts = {}
    for v in views:
        comps = [c for c in comparisons if c["view_id"] == v]
        counts[v] = {}
        for n in sorted({c["object_count"] for c in comps}):
            sub = [c for c in comps if c["object_count"] == n]
            keys = {(c["parent_command_id"], v) for c in sub}
            counts[v][str(n)] = {"N": len(sub),
                                 "models": {f: _small([s for s in scores if (s["parent_command_id"], s["view_id"]) in keys
                                                       and s["format"] == f]) for f in FORMATS},
                                 "rules": {"R": sum(c["rules"]["outcome"] == "resolved" for c in sub),
                                           "C": sum(c["rules"]["source_target_selected"] is True for c in sub),
                                           "technical": sum(c["rules"]["technical_failure"] for c in sub)}}
    return {"format_version": 1, "record_type": "iref_pilot_score_summary", "policy_id": POLICY_ID, "mode": mode,
            "population": population, "models": models, "rules": rules, "format_pairs": pairs,
            "model_vs_rules": versus, "policy_crosstab": crosstab,
            "breakdowns": {"source_relation": relations, "object_count": counts}, "concentration": concentration,
            "technical": {"model_planned_exclusions": sum(s["score_status"] == "unscored_planned_exclusion" for s in scores),
                          "model_technical_failures": sum(s["score_status"] == "unscored_technical_failure" for s in scores),
                          "rules_technical_failures": sum(c["rules"]["technical_failure"] for c in comparisons)},
            "limitations": list(LIMITATIONS)}


# ---------------------------------------------------------------------------------------------- the report
def _r(x) -> str:
    return f"{x['numerator']}/{x['denominator']}" + (" (null)" if x["value"] is None else f" ({x['value']:.3f})")


def _mark(sel) -> str:
    return "unscored" if sel is None else ("✓" if sel else "✗")


def _cell_model(m) -> str:
    if m["score_status"] != "scored":
        return f"{m['score_status']} ({m['request_id']})"
    what = "ASK" if m["model_choice"] == "model_choice_ask" else m["choice_object_id"]
    unk = " [unknown category]" if m["choice_category_state"] == "unknown" else ""
    return f"{m['choice_code']}={what}{unk} {_mark(m['source_target_selected'])}"


def _cell_rules(r) -> str:
    what = r["target_id"] if r["outcome"] == "resolved" else (
        "[" + ", ".join(r["candidate_ids"]) + "]" if r["candidate_ids"] else "")
    codes = ", ".join(r["reason_codes"][:3]) + (" …" if len(r["reason_codes"]) > 3 else "")
    why = f"({r['reason_code']}; {codes})" if r["reason_code"] and codes else f"({r['reason_code']})" if r["reason_code"] else ""
    return " ".join(x for x in (r["outcome"], what, _mark(r["source_target_selected"]), why) if x)


def _esc(text) -> str:
    return text.replace("|", "\\|")


def render_report(s, comparisons) -> str:
    """Markdown from the summary and comparison rows only, in fixed orders, so a readback renders the same bytes."""
    views = s["population"]["views"]
    keys = [f"{v}/{f}" for v in views for f in FORMATS]
    L = ["# IRef-VLA zero-shot pilot scored against source annotations, with a matched rules baseline (A2.3b)", "",
         INTERPRETATION, "",
         f"Mode: {s['mode']}. Parents {s['population']['parents']}, parent/view pairs {s['population']['parent_views']}, "
         f"model requests {s['population']['requests']}; annotation records {s['population']['annotation_records']} "
         f"(annotations per command: " + ", ".join(f"{k}: {n}" for k, n in s["population"]["annotation_multiplicity"].items())
         + ").", "",
         "## Model choices (each row is one view and one format)", "",
         "| View / format | N | Completed | Objects | ASK | Excluded | Technical | Source target C | C/N | C/completed | "
         "C/object choices |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in keys:
        m = s["models"][k]
        L.append(f"| {k} | {m['N']} | {m['completed']} | {m['object_choices']} | {m['ask']} | {m['planned_exclusions']} | "
                 f"{m['technical_failures']} | {m['C']} | {_r(m['C_over_N'])} | {_r(m['C_over_completed'])} | "
                 f"{_r(m['C_over_object_choices'])} |")
    L += ["", "C/object choices can rise by abstaining; C/N keeps every planned command in the denominator.", "",
          "## Rules baseline (once per view; formats do not change its inputs)", "",
          "| View | N | " + " | ".join(o for o in RULES_BINS) + " | R | C | C/N | C/R |",
          "|---|---:|" + "---:|" * len(RULES_BINS) + "---:|---:|---:|---:|"]
    for v in views:
        r = s["rules"][v]
        cells = [str(r["outcomes"].get(o, r["technical_failures"].get(o, 0))) for o in RULES_BINS]
        L.append(f"| {v} | {r['N']} | " + " | ".join(cells) + f" | {r['R']} | {r['C']} | {_r(r['C_over_N'])} | "
                 f"{_r(r['C_over_R'])} |")
    L += ["", "A rules result counts as selecting the source target only when it resolved to it; ambiguous, insufficient "
              "and no-match results may be justified under our policy even though the source names one target.", "",
          "## Formats compared within each view (paired by command)", "",
          "| View | N | Both match | Coordinates only | Augmented only | Neither | Unscored | Same choice | Different | "
          "Not comparable | C/N coordinates | C/N augmented | Difference (pp) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for v in views:
        p = s["format_pairs"][v]
        d = p["difference_pp"]["value"]
        L.append(f"| {v} | {p['N']} | {p['both']} | {p['coordinates_only']} | {p['augmented_only']} | {p['neither']} | "
                 f"{p['unscored']} | {p['same_choice']} | {p['different_choice']} | {p['not_comparable']} | "
                 f"{_r(p['C_over_N'][FORMATS[0]])} | {_r(p['C_over_N'][FORMATS[1]])} | "
                 f"{'null' if d is None else f'{d:+.2f}'} |")
    L += ["", "Difference: augmented (coordinates_relations_v2) minus coordinates, in percentage points of C/N. No "
              "winner is chosen.", "", "## Each format against the rules baseline (source-target agreement, paired)", "",
          "| View / format | Both | Model only | Rules only | Neither | Unscored | Rules not selecting the source, by outcome |",
          "|---|---:|---:|---:|---:|---:|---|"]
    for k in keys:
        x = s["model_vs_rules"][k]
        by = ", ".join(f"{o} {n}" for o, n in x["rules_not_source_by_outcome"].items()) or "none"
        L.append(f"| {k} | {x['both']} | {x['model_only']} | {x['rules_only']} | {x['neither']} | {x['unscored']} | {by} |")
    L += ["", "## Model outcome against rules outcome", "",
          "ASK against a non-resolved rules result is not labelled a correct clarification.", ""]
    for k in keys:
        x = s["policy_crosstab"][k]
        L += [f"**{k}**", "", "| Model \\ rules | " + " | ".join(RULES_BINS) + " |", "|---|" + "---:|" * len(RULES_BINS)]
        for row in ("model_choice_object", "model_choice_ask", "unscored"):
            cols = x["counts"][row]
            L.append(f"| {row} | " + " | ".join(str(cols[o]) for o in RULES_BINS) + " |")
        w = x["within_rules_resolved"]
        L += ["", f"Where the rules resolved: same object {w['same_object']}, different object {w['different_object']}, "
                  f"ASK {w['ask']}, unscored {w['unscored']}.", ""]
    L += ["## Choice concentration (descriptive; no bias conclusion without a controlled test)", "",
          "| View / format | Chosen aliases | Source-target aliases | Always-B agreement (B offered) | Unknown-category "
          "choices |", "|---|---|---|---|---:|"]
    for k in keys:
        c = s["concentration"][k]
        a = c["always_alias"]
        L.append(f"| {k} | " + ", ".join(f"{x} {c['chosen_alias'][x]}" for x in CODES if x in c["chosen_alias"]) + " | "
                 + ", ".join(f"{x} {c['source_alias'][x]}" for x in CODES if x in c["source_alias"])
                 + f" | {_r(a['agreement'])} (offered in {a['offered']}) | {c['unknown_category_choices']} |")
    L += ["", "Source alias against chosen alias (rows: the source target's alias; columns: the choice):", ""]
    for k in keys:
        sb = s["concentration"][k]["source_by_chosen"]
        L.append(f"- {k}: " + "; ".join(f"{src} → " + ", ".join(f"{x} {sb[src][x]}" for x in CODES + ("unscored",)
                                                                 if x in sb[src]) for src in CODES if src in sb))
    L += ["", "Most chosen objects:", ""]
    for k in keys:
        c = s["concentration"][k]
        top = sorted(c["chosen_objects"].items(), key=lambda kv: (-kv[1], kv[0]))[:6]
        L.append(f"- {k}: " + (", ".join(f"{o} {n}" for o, n in top) or "none"))
    L += ["", "The always-B row is a post-observation diagnostic on this sample, not a competitive baseline or a trained "
              "rule. Restricted shares are not calibrated correctness probabilities.", "",
          "## By source relation label (N = commands)", "",
          "| Relation | N | " + " | ".join(f"{v}/{f}: C" for v in views for f in FORMATS) + " | "
          + " | ".join(f"rules {v}: R, C" for v in views) + " |",
          "|---|---:|" + "---:|" * (len(views) * len(FORMATS)) + "---|" * len(views)]
    for label in sorted(s["breakdowns"]["source_relation"]):
        x = s["breakdowns"]["source_relation"][label]
        L.append(f"| {label} | {x['N']} | " + " | ".join(str(x["models"][f"{v}/{f}"]["C"]) for v in views for f in FORMATS)
                 + " | " + " | ".join(f"{x['rules'][v]['R']}, {x['rules'][v]['C']}" for v in views) + " |")
    L += ["", "## By subscene object count (N = parent/view pairs)", "",
          "| View | Objects | N | " + " | ".join(f"{f}: C" for f in FORMATS) + " | Rules R, C |",
          "|---|---:|---:|" + "---:|" * len(FORMATS) + "---|"]
    for v in views:
        byn = s["breakdowns"]["object_count"][v]
        for n in sorted(byn, key=int):
            x = byn[n]
            L.append(f"| {v} | {n} | {x['N']} | " + " | ".join(str(x["models"][f]["C"]) for f in FORMATS)
                     + f" | {x['rules']['R']}, {x['rules']['C']} |")
    t = s["technical"]
    L += ["", f"Technical: model planned exclusions {t['model_planned_exclusions']}, model technical failures "
              f"{t['model_technical_failures']}, rules technical failures {t['rules_technical_failures']}.", "",
          "## Review table (every parent and view; complete evidence in comparisons.jsonl and scores.jsonl)", "",
          "| # | View | Command | Source (relation → target, alias) | n | coordinates_v2 | coordinates_relations_v2 | "
          "Rules | Records | Flags |", "|---:|---|---|---|---:|---|---|---|---|---|"]
    for c in comparisons:
        recs = " ".join(c["models"][f]["request_id"] for f in FORMATS) + f"; rules {c['selection_rank']}/{c['view_id']}"
        L.append(f"| {c['selection_rank']} | {c['view_id']} | {_esc(c['text'])} | {c['source']['relation']} → "
                 f"{c['source']['target_id']} ({c['source']['alias']}) | {c['object_count']} | "
                 f"{_cell_model(c['models'][FORMATS[0]])} | {_cell_model(c['models'][FORMATS[1]])} | "
                 f"{_esc(_cell_rules(c['rules']))} | {recs} | {', '.join(c['flags']) or '—'} |")
    L += ["", "## Limitations", ""] + [f"- {x}" for x in s["limitations"]] + [""]
    return "\n".join(L)


# ------------------------------------------------------------------------------------------- the command
def run_score(*, pilot, requests, bundle, rules, annotations, out, sample=True, fixture_scene_id=None) -> dict:
    """The `score` command; returns the summary. `sample=False` with a fixture scene serves labelled fixtures only."""
    out = output.refuse_existing(out)
    check_overlap(out, (pilot, requests, bundle, rules, annotations))
    mode = MODES[0] if sample else MODES[1]
    policy = load_policy()
    tol = policy["score_check_absolute_tolerance"]
    rules_protocol = IN.relabel(INPUT, A2PROTOCOL.load_protocol)
    req = IN.load_requests(requests, sample=sample)
    src = IN.load_bundle(bundle)
    IN.check_request_bundle(req, src)
    pil = IN.load_pilot(pilot)
    IN.check_pilot_links(pil, req)
    results_by_key = IN.check_results(pil["results"], req)
    IN.check_scores(pil["results"], req["protocol"], tol)
    watched = {"pilot manifest": Path(pilot) / "manifest.json", "pilot results": Path(pilot) / "results.jsonl",
               "requests manifest": req["dir"] / "manifest.json", "bundle manifest": src["dir"] / "manifest.json",
               "preparation index": src["dir"] / "preparation-index.jsonl", "rules": Path(rules) / "rules.jsonl",
               "rules manifest": Path(rules) / "manifest.json", "annotations": Path(annotations)}
    before = file_hashes(watched)
    entries = IN.join(req, src, sample=sample)
    rd = BL.load_rules_dir(rules)
    if rd["manifest"]["mode"] != mode:
        fail(INTEGRITY, [("rules manifest $.mode", f"rules computed in {rd['manifest']['mode']} mode cannot be scored in "
                                                   f"{mode} mode")])
    rules_by_key = BL.check_rules_links(rd, req, src, entries)
    sources, ann = IN.load_annotations(annotations, entries, req, rules_protocol, sample=sample,
                                       fixture_scene_id=fixture_scene_id)
    scores, comparisons = build_rows(entries, results_by_key, rules_by_key, sources, req["protocol"]["ask_code"])
    summary = summarize(scores, comparisons, mode)
    if file_hashes(watched) != before:
        fail(INTEGRITY, [("inputs", "an input file changed while scoring ran")])
    canonical = strict_json("summary", encode_json(summary), INPUT)  # render exactly what a readback will load
    files = {"scores.jsonl": encode_jsonl(scores), "comparisons.jsonl": encode_jsonl(comparisons),
             "summary.json": encode_json(summary), "report.md": render_report(canonical, comparisons).encode("utf-8")}
    manifest = {
        "format_version": 1, "record_type": "iref_pilot_score_manifest", "policy_id": POLICY_ID,
        "policy_sha256": policy_sha256(), "policy_decision": policy["decision"], "mode": mode,
        "inputs": {"pilot": {"manifest_sha256": pil["manifest_sha256"], "files": pil["files"],
                             "protocol_id": pil["manifest"]["protocol_id"]},
                   "requests_manifest_sha256": req["manifest_sha256"], "request_protocol_sha256": req["protocol_sha256"],
                   "selected_sha256": req["manifest"]["selection"]["selected_sha256"],
                   "bundle_manifest_sha256": src["manifest_sha256"], "preparation_index_sha256": src["index_sha256"],
                   "rules": {"manifest_sha256": sha256(read_bytes(Path(rules) / "manifest.json", "rules manifest")),
                             "files": {k: v for k, v in rd["files"].items() if k != "manifest.json"},
                             "semantic_rules_hash": rd["manifest"]["semantic_rules_hash"]},
                   "annotations": {k: v for k, v in ann.items() if k != "checks"}},
        "reference_checks": ann["checks"], "score_check_absolute_tolerance": tol,
        "counts": {"parents": summary["population"]["parents"], "score_rows": len(scores),
                   "comparison_rows": len(comparisons)},
        "outputs": {k: sha256(v) for k, v in sorted(files.items())}, "code": code_hashes(), "runtime": runtime(),
        "notes": ["No parser, resolver, relation evaluation, serializer or model ran during scoring.",
                  "Annotations were read only here, after the saved model and rules results were verified."]}
    files["manifest.json"] = encode_json(manifest)
    publish_verified(out, files, verify_score_dir)
    return summary


# ------------------------------------------------------------------------------------------------ readback
def verify_score_dir(folder) -> list:
    """Every file, hash, record, cardinality and recomputed summary and report of a score folder."""
    f, bad = Path(folder), []
    present = sorted(p.relative_to(f).as_posix() for p in f.rglob("*") if p.is_file()) if f.is_dir() else []
    if present != sorted(FILES):
        return [f"expected exactly the files {sorted(FILES)}; found {present[:8]}"]
    try:
        manifest = read_json(f / "manifest.json", "score manifest")
        summary = read_json(f / "summary.json", "score summary")
        scores = read_jsonl(f / "scores.jsonl", "scores")
        comparisons = read_jsonl(f / "comparisons.jsonl", "comparisons")
        report = (f / "report.md").read_bytes()
    except Exception as e:  # noqa: BLE001 - reported as a finding
        return [f"unreadable score output: {e}"]
    for name, rec in (("score_manifest", manifest), ("score_summary", summary)):
        bad += [f"{i['path']}: {i['message']}" for i in schema_issues(name, rec, name)]
    bad += [f"{i['path']}: {i['message']}" for k, r in enumerate(scores, 1)
            for i in schema_issues("score_row", r, f"scores.jsonl line {k}")]
    bad += [f"{i['path']}: {i['message']}" for k, r in enumerate(comparisons, 1)
            for i in schema_issues("comparison_row", r, f"comparisons.jsonl line {k}")]
    if bad:
        return bad
    for name, want in manifest["outputs"].items():
        if sha256((f / name).read_bytes()) != want:
            bad.append(f"{name}: changed since the manifest was written")
    if len(scores) != len(FORMATS) * len(comparisons) or [s["request_index"] for s in scores] != list(range(1, len(scores) + 1)):
        bad.append("scores.jsonl is not two rows per comparison, in request order")
    order = [(c["selection_rank"], VIEWS.index(c["view_id"])) for c in comparisons]
    if order != sorted(order) or len(set(order)) != len(order):
        bad.append("comparisons.jsonl is not one row per parent and view, in selection order with full first")
    by_req = {s["request_id"]: s for s in scores}
    for c in comparisons:
        for fm in FORMATS:
            m, s = c["models"][fm], by_req.get(c["models"][fm]["request_id"])
            if s is None or (s["parent_command_id"], s["view_id"], s["format"]) != (c["parent_command_id"], c["view_id"], fm) \
                    or any(s[k] != m[k] for k in m) or s["source_target_id"] != c["source"]["target_id"] \
                    or s["source_alias"] != c["source"]["alias"] or s["rules_outcome"] != c["rules"]["outcome"] \
                    or s["rules_source_target_selected"] != c["rules"]["source_target_selected"]:
                bad.append(f"comparison {c['selection_rank']} {c['view_id']}: its {fm} entry disagrees with the score row")
    if bad:
        return bad
    again = summarize(scores, comparisons, manifest["mode"])
    if again != summary:
        bad.append("summary.json differs from a recomputation from the rows")
    if render_report(summary, comparisons).encode("utf-8") != report:
        bad.append("report.md differs from a rendering of the summary and rows")
    if manifest["counts"] != {"parents": summary["population"]["parents"], "score_rows": len(scores),
                              "comparison_rows": len(comparisons)}:
        bad.append("the manifest's counts disagree with the rows")
    return bad


__all__ = ["FILES", "build_rows", "render_report", "run_score", "summarize", "verify_score_dir"]
