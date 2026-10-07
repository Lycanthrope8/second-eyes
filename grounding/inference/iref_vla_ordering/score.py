"""A2.3c scoring (D94): the crossed grid's choices against the accepted A2.3b source targets, on the laptop.

Reads the frozen request bundle, the run's results, run r003 and the accepted A2.3b score folder, all verified with
their own checks and pins. It never parses, resolves, serializes or runs a model, and never infers a target from a
letter: each cell's choice is decoded through that cell's own mapping, and the target's letter and list position are
looked up in the same mapping. Per command, then per view and format with every command weighted equally:

- subsets of the same grid: identity (0, 0), assignment-only (a > 0, p = 0), list-order-only (a = 0, p > 0), crossed
  (a > 0, p > 0) and all cells; counts, source-target selections C, C/planned and C/object-choices;
- one-factor contrasts: at each p, (0, p) against (a, p) for a > 0 (code assignment changed, list order fixed); at
  each a, (a, 0) against (a, p) for p > 0 (list order changed, codes fixed); rates per command first, then macro means;
- the always-B and always-second-list-position reference policies, which select the source in exactly 1/n of a
  complete grid, and the source targets' positions in the scene document, where a fixed-scene-position policy could
  still appear to succeed.
Descriptive only: no significance test, no format winner, no Gate B conclusion.
"""
from __future__ import annotations

import json
import os
import shutil
import statistics
import tempfile
from collections import Counter

from pathlib import Path

from ...evaluation.iref_vla import output
from ...evaluation.iref_vla.protocol import (EvaluationInputError, EvaluationOutputError, encode_json, encode_jsonl,
                                             issue, runtime, sha256, strict_json)
from ...evaluation.iref_vla_pilot.score import verify_score_dir
from ..iref_vla.prepare import rows_of
from ..iref_vla.run import verify_results
from . import design as D
from .prepare import code_hashes, verify_order_requests
from .run import verify_order_results

OUTPUTS = ("commands.jsonl", "report.md", "scores.jsonl", "summary.json")
SUBSETS = ("identity", "assignment_only", "list_order_only", "crossed")
VIEWS = ("full_inventory", "source_known_nyu")
FORMATS = ("coordinates_v2", "coordinates_relations_v2")
INTERPRETATION = [
    "One previously inspected IRef-VLA development room, annotated geometry, commands selected through the rules parser "
    "and the category-complete subscene policy; the 32 parents are the research sample, not 4,536 independent examples.",
    "The scene document, including its object order and IDs, is byte identical in every cell: the design varies only the "
    "letter assigned to each object and the order of the choices list.",
    "Persistence of one letter while its object changes is compatible with a code preference; following a choices-list "
    "position while its letter and object change is compatible with position sensitivity; following one object across "
    "the grid shows invariance to these two transformations, not correctness, and may reflect its fixed place or ID in "
    "the document. Mixed responses may reflect interacting prompt sensitivities. No mechanism is claimed.",
    "K is a model abstention, not evidence of correct ambiguity handling. Rules outcomes are inherited once per parent "
    "and view from A2.3b. Restricted shares are not calibrated probabilities.",
    "No significance test, generalization interval, format winner, Gate B conclusion or Paper 1 claim follows.",
]


def subset_of(a: int, p: int) -> str:
    return "identity" if (a, p) == (0, 0) else "assignment_only" if p == 0 else "list_order_only" if a == 0 else "crossed"


def frac(num: int, den: int):
    return None if den == 0 else {"numerator": num, "denominator": den, "value": num / den}


def mean_or_none(values):
    v = [x for x in values if x is not None]
    return None if not v else sum(v) / len(v)


# ------------------------------------------------------------------------------------------------ pure scoring
def score_rows(requests, results, sources, policy_id: str) -> list:
    """One score row per cell. sources: {(parent, view, format): {target, relation, rules_outcome, rules_target}}."""
    by_base = {}
    for r in results:
        by_base.setdefault(r["base_request_id"], {})[(r["assignment_shift"], r["order_shift"])] = r
    out = []
    for q, r in zip(requests, results):
        src = sources[(q["parent_command_id"], q["view_id"], q["format"])]
        target = src["target"]
        codes = [m[0] for m in q["mapping"]]
        objs = [m[1] for m in q["mapping"]]
        if target not in q["object_ids"] or target not in objs:
            raise EvaluationInputError([issue(q["variant_id"], "E_ORDER_REFERENCE", f"source target {target} is not offered")])
        ident = by_base[q["base_request_id"]][(0, 0)]
        obj = r["choice_object_id"]
        out.append({"format_version": 1, "record_type": "iref_order_score", "policy_id": policy_id,
                    **{k: q[k] for k in ("variant_index", "variant_id", "base_request_index", "base_request_id",
                                         "selection_rank", "parent_command_id", "view_id", "format", "derived_scene_id",
                                         "derived_command_id", "object_count", "assignment_shift", "order_shift")},
                    "subset": subset_of(q["assignment_shift"], q["order_shift"]),
                    "source_target_id": target, "source_target_code": codes[objs.index(target)],
                    "source_target_list_position": objs.index(target) + 1,
                    "source_target_scene_position": q["object_ids"].index(target) + 1,
                    "source_relation": src["relation"], "rules_outcome": src["rules_outcome"],
                    "rules_target_id": src["rules_target"], "model_choice": r["model_choice"],
                    "choice_code": r["choice_code"], "choice_object_id": obj,
                    "choice_list_position": r["choice_list_position"], "choice_scene_position": r["choice_scene_position"],
                    "source_target_selected": obj is not None and obj == target,
                    "same_object_as_identity": obj == ident["choice_object_id"],
                    "same_code_as_identity": r["choice_code"] == ident["choice_code"],
                    "same_list_position_as_identity": None if obj is None or ident["choice_object_id"] is None
                    else r["choice_list_position"] == ident["choice_list_position"]})
    return out


def _subset_block(rows) -> dict:
    c = sum(r["source_target_selected"] for r in rows)
    objs = [r for r in rows if r["model_choice"] == "model_choice_object"]
    comparable = [r for r in rows if r["same_list_position_as_identity"] is not None]
    return {"planned": len(rows), "completed": len(rows), "object_choices": len(objs), "ask": len(rows) - len(objs),
            "unscored": 0, "source_target_selections": c, "c_over_planned": frac(c, len(rows)),
            "c_over_object_choices": frac(c, len(objs)),
            "chosen_codes": dict(sorted(Counter(r["choice_code"] for r in rows).items())),
            "chosen_list_positions": dict(sorted(Counter(str(r["choice_list_position"]) for r in objs).items())),
            "chosen_objects": dict(sorted(Counter(r["choice_object_id"] or "ASK" for r in rows).items())),
            "chosen_scene_positions": dict(sorted(Counter(str(r["choice_scene_position"]) for r in objs).items())),
            "source_target_codes": dict(sorted(Counter(r["source_target_code"] for r in rows).items())),
            "source_target_list_positions": dict(sorted(Counter(str(r["source_target_list_position"]) for r in rows).items())),
            "agreement_with_identity": {
                "same_object_or_ask": frac(sum(r["same_object_as_identity"] for r in rows), len(rows)),
                "same_code": frac(sum(r["same_code_as_identity"] for r in rows), len(rows)),
                "same_list_position": frac(sum(r["same_list_position_as_identity"] for r in comparable), len(comparable))}}


def contrasts(grid: dict, n: int) -> dict:
    """grid: {(a, p): score row}. Code contrast: (0, p) vs (a, p), a > 0. Order contrast: (a, 0) vs (a, p), p > 0."""
    code = {"pairs": 0, "object_or_ask_changed": 0, "same_letter": 0}
    order = {"pairs": 0, "object_or_ask_changed": 0, "same_position_pairs": 0, "same_position": 0}
    for p in range(n):
        ref = grid[(0, p)]
        for a in range(1, n):
            x = grid[(a, p)]
            code["pairs"] += 1
            code["object_or_ask_changed"] += x["choice_object_id"] != ref["choice_object_id"]
            code["same_letter"] += x["choice_code"] == ref["choice_code"]
    for a in range(n):
        ref = grid[(a, 0)]
        for p in range(1, n):
            x = grid[(a, p)]
            order["pairs"] += 1
            order["object_or_ask_changed"] += x["choice_object_id"] != ref["choice_object_id"]
            if x["choice_object_id"] is not None and ref["choice_object_id"] is not None:
                order["same_position_pairs"] += 1
                order["same_position"] += x["choice_list_position"] == ref["choice_list_position"]
    code["change_rate"] = frac(code["object_or_ask_changed"], code["pairs"])
    code["same_letter_rate"] = frac(code["same_letter"], code["pairs"])
    order["change_rate"] = frac(order["object_or_ask_changed"], order["pairs"])
    order["same_position_rate"] = frac(order["same_position"], order["same_position_pairs"])
    return {"code_assignment": code, "list_order": order}


def _share(counter: Counter, total: int):
    if not total:
        return None
    k, v = counter.most_common(1)[0]
    return {"value": k, "share": v / total, "count": v}


def command_metrics(rows) -> dict:
    """Everything for one base request's complete grid (rows: its n * n score rows)."""
    first = rows[0]
    n = first["object_count"]
    grid = {(r["assignment_shift"], r["order_shift"]): r for r in rows}
    if sorted(grid) != D.cells(n):
        raise EvaluationInputError([issue(first["base_request_id"], "E_ORDER_INTERNAL", "incomplete grid")])
    subsets = {s: _subset_block([r for r in rows if r["subset"] == s]) for s in SUBSETS}
    subsets["all"] = _subset_block(rows)
    ident = grid[(0, 0)]
    objs = [r for r in rows if r["model_choice"] == "model_choice_object"]
    c_all = subsets["all"]["source_target_selections"]
    return {"base_request_index": first["base_request_index"], "base_request_id": first["base_request_id"],
            "selection_rank": first["selection_rank"], "parent_command_id": first["parent_command_id"],
            "view_id": first["view_id"], "format": first["format"], "object_count": n,
            "source_target_id": first["source_target_id"], "source_relation": first["source_relation"],
            "source_target_scene_position": first["source_target_scene_position"],
            "rules_outcome": first["rules_outcome"], "rules_target_id": first["rules_target_id"],
            "identity_choice": {"code": ident["choice_code"], "object_id": ident["choice_object_id"],
                                "list_position": ident["choice_list_position"], "selected_source": ident["source_target_selected"]},
            "subsets": subsets, "contrasts": contrasts(grid, n),
            "target_agreement": frac(c_all, n * n),
            "stability": {"object_or_ask": _share(Counter(r["choice_object_id"] or "ASK" for r in rows), len(rows)),
                          "code": _share(Counter(r["choice_code"] for r in rows), len(rows)),
                          "list_position": _share(Counter(r["choice_list_position"] for r in objs), len(objs)),
                          "scene_position": _share(Counter(r["choice_scene_position"] for r in objs), len(objs))},
            "reference_policies": {"always_B": frac(n, n * n) if n >= 2 else None,
                                   "always_second_list_position": frac(n, n * n) if n >= 2 else None}}


def aggregate(commands) -> dict:
    """Per view and format: every command weighted equally (macro means), with raw counts and pooled diagnostics."""
    out = {}
    for v in VIEWS:
        for fm in FORMATS:
            cs = [c for c in commands if c["view_id"] == v and c["format"] == fm]
            if not cs:
                continue
            block = {"commands": len(cs), "cells": sum(c["object_count"] ** 2 for c in cs), "subsets": {}}
            for s in SUBSETS + ("all",):
                bl = [c["subsets"][s] for c in cs]
                block["subsets"][s] = {
                    "planned": sum(b["planned"] for b in bl), "object_choices": sum(b["object_choices"] for b in bl),
                    "ask": sum(b["ask"] for b in bl), "unscored": 0,
                    "source_target_selections": sum(b["source_target_selections"] for b in bl),
                    "macro_c_over_planned": mean_or_none([b["c_over_planned"]["value"] if b["c_over_planned"] else None for b in bl]),
                    "macro_c_over_object_choices": mean_or_none([b["c_over_object_choices"]["value"]
                                                                 if b["c_over_object_choices"] else None for b in bl]),
                    "commands_with_object_choices": sum(1 for b in bl if b["c_over_object_choices"]),
                    "pooled_c_over_planned_cell_weighted": frac(sum(b["source_target_selections"] for b in bl),
                                                                sum(b["planned"] for b in bl)),
                    "macro_same_object_as_identity": mean_or_none([b["agreement_with_identity"]["same_object_or_ask"]["value"]
                                                                   if b["agreement_with_identity"]["same_object_or_ask"] else None
                                                                   for b in bl]),
                    "chosen_codes": dict(sorted(sum((Counter(b["chosen_codes"]) for b in bl), Counter()).items())),
                    "chosen_list_positions": dict(sorted(sum((Counter(b["chosen_list_positions"]) for b in bl), Counter()).items(),
                                                         key=lambda x: int(x[0]))),
                    "chosen_scene_positions": dict(sorted(sum((Counter(b["chosen_scene_positions"]) for b in bl), Counter()).items(),
                                                          key=lambda x: int(x[0])))}
            con = {}
            for kind, rate, extra in (("code_assignment", "change_rate", "same_letter_rate"),
                                      ("list_order", "change_rate", "same_position_rate")):
                xs = [c["contrasts"][kind] for c in cs]
                con[kind] = {"macro_change_rate": mean_or_none([x[rate]["value"] if x[rate] else None for x in xs]),
                             f"macro_{extra}": mean_or_none([x[extra]["value"] if x[extra] else None for x in xs]),
                             "changed": sum(x["object_or_ask_changed"] for x in xs), "pairs": sum(x["pairs"] for x in xs),
                             "commands_with_pairs": sum(1 for x in xs if x["pairs"])}
            ta = [c["target_agreement"]["value"] for c in cs]
            block["contrasts"] = con
            block["target_agreement"] = {"macro_mean": sum(ta) / len(ta), "min": min(ta), "max": max(ta),
                                         "pooled_cell_weighted": frac(sum(c["subsets"]["all"]["source_target_selections"] for c in cs),
                                                                      block["cells"])}
            block["reference_policies"] = {
                "always_B_macro": mean_or_none([c["reference_policies"]["always_B"]["value"]
                                                if c["reference_policies"]["always_B"] else None for c in cs]),
                "always_second_list_position_macro": mean_or_none([c["reference_policies"]["always_second_list_position"]["value"]
                                                                   if c["reference_policies"]["always_second_list_position"] else None
                                                                   for c in cs])}
            block["source_target_scene_positions"] = dict(sorted(Counter(str(c["source_target_scene_position"]) for c in cs).items(),
                                                                 key=lambda x: int(x[0])))
            block["identity_source_selections"] = sum(c["identity_choice"]["selected_source"] for c in cs)
            out[f"{v}/{fm}"] = block
    return out


def summarize(scores, commands, policy_id) -> dict:
    rules = {}
    seen = set()
    for c in commands:
        key = (c["parent_command_id"], c["view_id"])
        if key in seen:
            continue
        seen.add(key)
        rules.setdefault(c["view_id"], Counter())[c["rules_outcome"]] += 1
    return {"format_version": 1, "record_type": "iref_order_score_summary", "policy_id": policy_id,
            "counts": {"cells": len(scores), "commands": len(commands),
                       "parents": len({c["parent_command_id"] for c in commands}),
                       "subsets": dict(sorted(Counter(s["subset"] for s in scores).items())),
                       "source_target_selections": sum(s["source_target_selected"] for s in scores)},
            "by_view_format": aggregate(commands),
            "rules_by_view": {v: dict(sorted(c.items())) for v, c in sorted(rules.items())},
            "reference_policies": {"always_B": "selects the source target in exactly n of n * n cells (1/n) of every "
                                               "complete grid, because B is offered in every cell when n >= 2",
                                   "always_second_list_position": "likewise 1/n: the second list position holds each "
                                                                  "object in exactly n cells",
                                   "fixed_scene_object": "a policy that always picks one scene position keeps its result "
                                                         "across the grid; see source_target_scene_positions"},
            "interpretation": INTERPRETATION}


def _pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def _fr(f):
    return "-" if f is None else f"{f['numerator']}/{f['denominator']}"


def _keys(summary) -> list:
    """View/format keys in a fixed order: JSON round trips sort keys, so rendering never relies on dict order."""
    return [f"{v}/{fm}" for v in VIEWS for fm in FORMATS if f"{v}/{fm}" in summary["by_view_format"]]


def _by_int(d) -> list:
    return sorted(d.items(), key=lambda x: int(x[0]))


def render_report(summary, commands) -> str:
    L = ["# A2.3c crossed code-assignment and choices-order diagnostic", "",
         "Descriptive development diagnostic on the A2.3a pilot (D94). " + INTERPRETATION[0], "",
         "## By view and format (every command weighted equally)", "",
         "| View / format | Commands | Cells | Identity C | Full-grid agreement, macro (min-max) | Pooled, cell-weighted "
         "| Always-B / second-position reference, macro | Code-shift change rate, macro | List-shift change rate, macro |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k in _keys(summary):
        b = summary["by_view_format"][k]
        t = b["target_agreement"]
        L.append(f"| {k} | {b['commands']} | {b['cells']} | {b['identity_source_selections']}/{b['commands']} | "
                 f"{_pct(t['macro_mean'])} ({_pct(t['min'])}-{_pct(t['max'])}) | {_fr(t['pooled_cell_weighted'])} | "
                 f"{_pct(b['reference_policies']['always_B_macro'])} / {_pct(b['reference_policies']['always_second_list_position_macro'])} | "
                 f"{_pct(b['contrasts']['code_assignment']['macro_change_rate'])} | "
                 f"{_pct(b['contrasts']['list_order']['macro_change_rate'])} |")
    L += ["", "Code-shift: at each fixed list order p, (0, p) against (a, p) for a > 0. List-shift: at each fixed code "
          "assignment a, (a, 0) against (a, p) for p > 0. Rates are computed per command, then averaged.", "",
          "## Subsets of the same grid", "",
          "| View / format | Subset | Planned | Object choices | ASK | C | C/planned, macro | C/object choices, macro "
          "| Same object as identity, macro | Chosen codes |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k in _keys(summary):
        b = summary["by_view_format"][k]
        for s in SUBSETS + ("all",):
            x = b["subsets"][s]
            L.append(f"| {k} | {s} | {x['planned']} | {x['object_choices']} | {x['ask']} | {x['source_target_selections']} | "
                     f"{_pct(x['macro_c_over_planned'])} | {_pct(x['macro_c_over_object_choices'])} | "
                     f"{_pct(x['macro_same_object_as_identity'])} | "
                     + ", ".join(f"{c}: {n}" for c, n in sorted(x["chosen_codes"].items())) + " |")
    L += ["", "## Same-letter and same-position rates", "",
          "| View / format | Same letter under a code shift, macro | Same list position under a list shift, macro |",
          "|---|---|---|"]
    for k in _keys(summary):
        b = summary["by_view_format"][k]
        L.append(f"| {k} | {_pct(b['contrasts']['code_assignment']['macro_same_letter_rate'])} | "
                 f"{_pct(b['contrasts']['list_order']['macro_same_position_rate'])} |")
    L += ["", "## Source targets' positions in the scene document (balance check)", ""]
    for k in _keys(summary):
        b = summary["by_view_format"][k]
        L.append(f"- {k}: " + ", ".join(f"position {p}: {n}" for p, n in _by_int(b["source_target_scene_positions"])))
    L += ["", "## Rules outcomes (inherited once per parent and view from A2.3b)", ""]
    for v, c in sorted(summary["rules_by_view"].items()):
        L.append(f"- {v}: " + ", ".join(f"{o}: {n}" for o, n in c.items()))
    L += ["", "## Every command", "",
          "| Rank | View | Format | n | Source (scene pos.) | Identity choice | Grid agreement | Modal object | Modal code "
          "| Modal list pos. | Code-shift change | List-shift change | Rules |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in commands:
        st = c["stability"]
        ic = c["identity_choice"]
        flag = " *" if ic["object_id"] and c["subsets"]["all"]["agreement_with_identity"]["same_object_or_ask"]["value"] < 1 else ""
        L.append(f"| {c['selection_rank']} | {c['view_id']} | {c['format']} | {c['object_count']} | "
                 f"{c['source_target_id']} ({c['source_target_scene_position']}) | {ic['code']} {ic['object_id'] or 'ASK'}{flag} | "
                 f"{_fr(c['target_agreement'])} | {st['object_or_ask']['value']} {_pct(st['object_or_ask']['share'])} | "
                 f"{st['code']['value']} {_pct(st['code']['share'])} | "
                 f"{'-' if st['list_position'] is None else st['list_position']['value']} "
                 f"{'' if st['list_position'] is None else _pct(st['list_position']['share'])} | "
                 f"{_fr(c['contrasts']['code_assignment']['change_rate'])} | {_fr(c['contrasts']['list_order']['change_rate'])} | "
                 f"{c['rules_outcome']} |")
    L += ["", "`*`: the identity cell's object is not chosen in every cell of its grid. Full evidence: commands.jsonl and "
          "scores.jsonl.", "", "## Interpretation", ""] + [f"- {x}" for x in INTERPRETATION]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------------------------------- the command
def _load_sources(base_scores: Path) -> dict:
    rows = rows_of("scores.jsonl", (base_scores / "scores.jsonl").read_bytes(), "E_ORDER_BASE_SCORES")
    out = {}
    for r in rows:
        key = (r["parent_command_id"], r["view_id"], r["format"])
        if key in out:
            raise EvaluationInputError([issue(str(base_scores), "E_ORDER_BASE_SCORES", f"{key} appears twice")])
        out[key] = {"target": r["source_target_id"], "relation": r["source_relation"], "rules_outcome": r["rules_outcome"],
                    "rules_target": r["rules_target_id"], "request_id": r["request_id"]}
    return out


def score_ordering(*, requests, results, base_pilot, base_scores, out, policy=None) -> dict:
    """The `score` command; returns the summary. `policy` is injectable for fixture tests only."""
    out = output.refuse_existing(out)
    req, res, bp, bs = Path(requests), Path(results), Path(base_pilot), Path(base_scores)
    for p in (req, res, bp, bs):
        o, q = out.resolve(), p.resolve()
        if o == q or q in o.parents or o in q.parents:
            raise EvaluationInputError([issue(str(out), "E_ORDER_PATH", f"the output overlaps the input {p}")])
    problems = verify_order_requests(req)
    if problems:
        raise EvaluationInputError([issue(str(req), "E_ORDER_REQUESTS", m) for m in problems[:20]])
    problems = verify_order_results(res, req)
    if problems:
        raise EvaluationInputError([issue(str(res), "E_ORDER_RESULTS", m) for m in problems[:20]])
    pol = D.load_policy(req / "policy.json")
    if policy is not None and D.policy_sha256(policy) != D.policy_sha256(req / "policy.json"):
        raise EvaluationInputError([issue(str(policy), "E_ORDER_POLICY", "not the policy of the request bundle")])
    rman_bytes, resman_bytes = (req / "manifest.json").read_bytes(), (res / "manifest.json").read_bytes()
    rman, resman = json.loads(rman_bytes), json.loads(resman_bytes)
    bad = verify_results(bp)
    if bad:
        raise EvaluationInputError([issue(str(bp), "E_ORDER_BASE_PILOT", m) for m in bad[:20]])
    bres = sha256((bp / "results.jsonl").read_bytes())
    bman = json.loads((bp / "manifest.json").read_bytes())
    if bres != pol["pins"]["base_results_sha256"] or resman["base_pilot"]["results_sha256"] != bres \
            or bman["requests_manifest_sha256"] != rman["base_requests"]["manifest_sha256"]:
        raise EvaluationInputError([issue(str(bp), "E_ORDER_PIN", "run r003 is not the pinned pilot of these requests and results")])
    bad = verify_score_dir(bs)
    if bad:
        raise EvaluationInputError([issue(str(bs), "E_ORDER_BASE_SCORES", m) for m in bad[:20]])
    sman = strict_json("A2.3b score manifest", (bs / "manifest.json").read_bytes(), "E_ORDER_BASE_SCORES")
    ssum, srep = sha256((bs / "summary.json").read_bytes()), sha256((bs / "report.md").read_bytes())
    if ssum != pol["pins"]["base_scores_summary_sha256"] or srep != pol["pins"]["base_scores_report_sha256"]:
        raise EvaluationInputError([issue(str(bs), "E_ORDER_PIN", f"summary {ssum} or report {srep} is not the pinned A2.3b result")])
    if sman["inputs"]["pilot"]["files"]["results.jsonl"] != bres \
            or sman["inputs"]["requests_manifest_sha256"] != rman["base_requests"]["manifest_sha256"]:
        raise EvaluationInputError([issue(str(bs), "E_ORDER_BASE_SCORES", "the A2.3b scores were not made from r003 and these base requests")])
    sources = _load_sources(bs)
    rq = rows_of("request-index.jsonl", (req / "request-index.jsonl").read_bytes(), "E_ORDER_REQUESTS")
    rs = rows_of("results.jsonl", (res / "results.jsonl").read_bytes(), "E_ORDER_RESULTS")
    for q in rq:
        src = sources.get((q["parent_command_id"], q["view_id"], q["format"]))
        if src is None or src["request_id"] != q["base_request_id"]:
            raise EvaluationInputError([issue(q["variant_id"], "E_ORDER_REFERENCE", "no A2.3b source target for this base request")])
    scores = score_rows(rq, rs, sources, pol["policy_id"])
    commands = []
    for bid in dict.fromkeys(s["base_request_id"] for s in scores):
        commands.append(command_metrics([s for s in scores if s["base_request_id"] == bid]))
    summary = summarize(scores, commands, pol["policy_id"])
    files = {"scores.jsonl": encode_jsonl(scores), "commands.jsonl": encode_jsonl(commands),
             "summary.json": encode_json(summary), "report.md": render_report(summary, commands).encode("utf-8")}
    manifest = {"format_version": 1, "record_type": "iref_order_score_manifest", "policy_id": pol["policy_id"],
                "policy_sha256": rman["policy_sha256"],
                "inputs": {"requests_manifest_sha256": sha256(rman_bytes), "results_manifest_sha256": sha256(resman_bytes),
                           "base_pilot_results_sha256": bres, "base_pilot_manifest_sha256": sha256((bp / "manifest.json").read_bytes()),
                           "base_scores_manifest_sha256": sha256((bs / "manifest.json").read_bytes()),
                           "base_scores_summary_sha256": ssum, "base_scores_report_sha256": srep,
                           "base_requests_manifest_sha256": rman["base_requests"]["manifest_sha256"]},
                "counts": summary["counts"], "code": code_hashes(), "runtime": runtime(),
                "outputs": {k: sha256(v) for k, v in files.items()},
                "notes": ["Targets come only from the accepted A2.3b scores; no parser, resolver, serializer or model ran.",
                          "Each choice is decoded through its own cell's mapping."]}
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=str(out.parent)))
    try:
        for k, v in files.items():
            output._write_file(staging / k, v)
        output._write_file(staging / "manifest.json", encode_json(manifest))
        bad = verify_order_scores(staging)
        if bad:
            raise RuntimeError("the scores failed readback: " + "; ".join(bad[:5]))
        os.rename(staging, out)
    except OSError as e:
        shutil.rmtree(staging, ignore_errors=True)
        raise EvaluationOutputError([issue(str(out), "E_EVAL_OUTPUT_IO", f"{type(e).__name__}: {e}")]) from e
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def verify_order_scores(folder) -> list:
    """Files, hashes, schemas, cardinalities, and the commands, summary and report recomputed from scores.jsonl."""
    f, bad = Path(folder), []
    present = sorted(p.relative_to(f).as_posix() for p in f.rglob("*") if p.is_file()) if f.is_dir() else []
    if present != sorted(OUTPUTS + ("manifest.json",)):
        return [f"expected exactly {sorted(OUTPUTS + ('manifest.json',))}; found {present[:8]}"]
    try:
        manifest = strict_json("manifest.json", (f / "manifest.json").read_bytes(), "E_ORDER_SCORES")
        scores = rows_of("scores.jsonl", (f / "scores.jsonl").read_bytes(), "E_ORDER_SCORES")
        commands = rows_of("commands.jsonl", (f / "commands.jsonl").read_bytes(), "E_ORDER_SCORES")
        summary = strict_json("summary.json", (f / "summary.json").read_bytes(), "E_ORDER_SCORES")
    except (OSError, EvaluationInputError) as e:
        return [f"unreadable scores: {e}"]
    bad += [f"manifest: {m}" for m in D.schema_errors("score_manifest", manifest)]
    bad += [f"summary: {m}" for m in D.schema_errors("score_summary", summary)]
    for k, s in enumerate(scores, 1):
        errs = D.schema_errors("score_row", s)
        if errs:
            return bad + [f"scores line {k}: {errs[0]}"]
        if s["variant_index"] != k:
            return bad + [f"scores line {k}: not in canonical order"]
    for c in commands:
        bad += [f"command {c.get('base_request_id')}: {m}" for m in D.schema_errors("command_row", c)]
    for name, want in (manifest.get("outputs") or {}).items():
        if sha256((f / name).read_bytes()) != want:
            bad.append(f"{name}: changed since the manifest was written")
    if bad:
        return bad
    again = [command_metrics([s for s in scores if s["base_request_id"] == bid])
             for bid in dict.fromkeys(s["base_request_id"] for s in scores)]
    if again != commands:
        bad.append("commands.jsonl differs from a recomputation from scores.jsonl")
    elif summarize(scores, commands, manifest["policy_id"]) != summary:
        bad.append("summary.json differs from a recomputation")
    elif render_report(summary, commands).encode("utf-8") != (f / "report.md").read_bytes():
        bad.append("report.md differs from a rendering of the summary")
    return bad



