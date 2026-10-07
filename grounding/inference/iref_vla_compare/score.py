"""A2.3d scoring (D95): rules, the 0.5B and the 7B model against the published targets, on the laptop.

Inputs, each verified by its own readback before use: the frozen request bundle, the rules folder, the two model runs,
the A2.2d bundle and the pinned IRef-VLA annotations (A2.3b's loader: one mapped target and one relation label per
command, present in both views' subscenes and offered in every request). Annotations are read here and nowhere else.
No parser, resolver or model runs.

Per request and system the outcome is one of: correct (the chosen object is the published target), wrong object, ASK
(an ASK on a uniquely annotated command is not correct), over the context ceiling, or execution failure. Every rate keeps
its numerator and denominator. The full inventory is the primary comparison, paired by parent, model and format; the
source-known view is reported separately. Agreement with the published target is a development diagnostic: a mismatch
does not by itself show the model wrong under every reading of the adapted geometry, and the resolver's answerability
judgement is reported apart from source agreement. Restricted shares are never treated as confidence; no threshold is
applied and no ambiguity credit is given. A small failure sample is drawn by a salted hash for inspection only.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from ...evaluation.iref_vla import output
from ...evaluation.iref_vla.protocol import (EvaluationInputError, EvaluationOutputError, encode_json, encode_jsonl,
                                             issue, runtime, sha256, strict_json)
from ...evaluation.iref_vla_pilot import baseline as BL
from ..iref_vla.prepare import rows_of
from . import design as D
from .prepare import MODEL_KEYS, code_hashes, verify_compare_requests
from .rules import verify_compare_rules
from .run import verify_compare_results

SCORING_PATH = Path(__file__).resolve().parent / "compare-scoring.v1.json"
OUTPUTS = ("report.md", "scores.jsonl", "summary.json")
SYSTEMS = MODEL_KEYS
OUTCOMES = ("correct", "wrong_object", "ask", "context_budget_exceeded", "execution_failed")


def _fail(code, problems):
    raise EvaluationInputError([issue(w, code, m) for w, m in problems])


def frac(n: int, d: int):
    return None if d == 0 else {"numerator": n, "denominator": d, "value": n / d}


def load_scoring(path=None) -> dict:
    p = Path(path) if path is not None else SCORING_PATH
    s = strict_json("scoring policy", p.read_bytes(), "E_COMPARE_SCORING")
    if not isinstance(s, dict) or s.get("record_type") != "iref_compare_scoring" or type(s.get("failures_per_cell")) is not int:
        _fail("E_COMPARE_SCORING", [(str(p), "not a version-1 compare scoring policy")])
    return s


def outcome_of(res, target) -> str:
    if res["technical_status"] != "completed":
        return res["technical_status"]
    if res["model_choice"] == "model_choice_ask":
        return "ask"
    return "correct" if res["choice_object_id"] == target else "wrong_object"


def score_rows(requests, results, rules_by_pv, sources) -> list:
    out = []
    for q in requests:
        src = sources[q["parent_command_id"]]
        target = src["target_id"]
        codes = [m[0] for m in q["mapping"][:-1]]
        objs = [m[1] for m in q["mapping"][:-1]]
        rl = rules_by_pv[(q["parent_command_id"], q["view_id"])]
        row = {"format_version": 1, "record_type": "iref_compare_score", "policy_id": D.POLICY_ID,
               **{k: q[k] for k in ("request_index", "request_id", "selection_rank", "parent_command_id", "view_id", "format",
                                    "object_count")},
               "source_target_id": target, "source_relation": src["relation"],
               "target_letter": codes[objs.index(target)], "target_list_position": objs.index(target) + 1,
               "target_scene_position": q["object_ids"].index(target) + 1,
               "reference_always_B": codes[objs.index(target)] == "B",
               "reference_second_position": objs.index(target) == 1,
               "rules": {"outcome": rl["outcome"], "target_id": rl["target_id"],
                         "agrees_with_source": rl["outcome"] == "resolved" and rl["target_id"] == target,
                         "technical_failure": rl["technical_failure"]}, "models": {}}
        for key in SYSTEMS:
            r = results[key][q["request_index"] - 1]
            if r["request_id"] != q["request_id"]:
                _fail("E_COMPARE_RESULTS", [(key, f"{q['request_id']}: results out of order")])
            row["models"][key] = {"technical_status": r["technical_status"], "choice_code": r["choice_code"],
                                  "choice_object_id": r["choice_object_id"], "choice_list_position": r["choice_list_position"],
                                  "outcome": outcome_of(r, target)}
        out.append(row)
    return out


def _block(rows, get) -> dict:
    c = Counter(get(r) for r in rows)
    planned = len(rows)
    objects = c["correct"] + c["wrong_object"]
    return {"planned": planned, **{o: c[o] for o in OUTCOMES}, "correct_over_planned": frac(c["correct"], planned),
            "correct_over_object_choices": frac(c["correct"], objects)}


def _pairs(rows, a, b) -> dict:
    """Paired 2 x 2 over rows where both sides completed; a and b map a row to (completed, correct)."""
    n = {"both": 0, "first_only": 0, "second_only": 0, "neither": 0, "not_comparable": 0}
    for r in rows:
        (ca, ka), (cb, kb) = a(r), b(r)
        if not (ca and cb):
            n["not_comparable"] += 1
        else:
            n["both" if ka and kb else "first_only" if ka else "second_only" if kb else "neither"] += 1
    common = len(rows) - n["not_comparable"]
    n["common"] = common
    n["difference_first_minus_second"] = frac(n["first_only"] - n["second_only"], common)
    return n


def _model(key):
    return lambda r: (r["models"][key]["technical_status"] == "completed", r["models"][key]["outcome"] == "correct")


def _rules(r):
    return (not r["rules"]["technical_failure"], r["rules"]["agrees_with_source"])


def summarize(rows, scoring) -> dict:
    by = {}
    for v in D.VIEWS:
        vb = {}
        for fm in D.FORMATS:
            rs = [r for r in rows if r["view_id"] == v and r["format"] == fm]
            sys = {key: _block(rs, lambda r, k=key: r["models"][k]["outcome"]) for key in SYSTEMS}
            sys["rules"] = {"planned": len(rs), "agrees_with_source": sum(r["rules"]["agrees_with_source"] for r in rs),
                            "correct_over_planned": frac(sum(r["rules"]["agrees_with_source"] for r in rs), len(rs)),
                            "outcomes": dict(sorted(Counter(r["rules"]["outcome"] for r in rs).items()))}
            refs = {"always_B": frac(sum(r["reference_always_B"] for r in rs), len(rs)),
                    "always_second_list_position": frac(sum(r["reference_second_position"] for r in rs), len(rs))}
            pairs = {"small_vs_large": _pairs(rs, _model(SYSTEMS[0]), _model(SYSTEMS[1])),
                     **{f"{key}_vs_rules": _pairs(rs, _model(key), _rules) for key in SYSTEMS}}
            rel = {}
            for lab in sorted({r["source_relation"] for r in rs}):
                sub = [r for r in rs if r["source_relation"] == lab]
                rel[lab] = {"planned": len(sub), **{key: sum(r["models"][key]["outcome"] == "correct" for r in sub) for key in SYSTEMS},
                            "rules": sum(r["rules"]["agrees_with_source"] for r in sub)}
            cnt = {}
            for n in sorted({r["object_count"] for r in rs}):
                sub = [r for r in rs if r["object_count"] == n]
                cnt[str(n)] = {"planned": len(sub), **{key: sum(r["models"][key]["outcome"] == "correct" for r in sub) for key in SYSTEMS},
                               "rules": sum(r["rules"]["agrees_with_source"] for r in sub),
                               "always_B": sum(r["reference_always_B"] for r in sub)}
            vb[fm] = {"systems": sys, "references": refs, "pairs": pairs, "by_relation": rel, "by_candidate_count": cnt}
        fmt_pairs = {}
        for key in SYSTEMS:
            ra = {r["parent_command_id"]: r for r in rows if r["view_id"] == v and r["format"] == D.FORMATS[0]}
            rb = {r["parent_command_id"]: r for r in rows if r["view_id"] == v and r["format"] == D.FORMATS[1]}
            joined = [(ra[p], rb[p]) for p in ra]
            fmt_pairs[key] = _pairs(joined, lambda x, k=key: _model(k)(x[0]), lambda x, k=key: _model(k)(x[1]))
        pv = [r for r in rows if r["view_id"] == v and r["format"] == D.FORMATS[0]]
        disagree = Counter("resolved_to_another_object" if r["rules"]["outcome"] == "resolved" and not r["rules"]["agrees_with_source"]
                           else r["rules"]["outcome"] for r in pv if not r["rules"]["agrees_with_source"])
        by[v] = {"by_format": vb, "format_pairs": fmt_pairs,
                 "rules_disagreements_with_source": dict(sorted(disagree.items())), "parent_views": len(pv)}
    return {"format_version": 1, "record_type": "iref_compare_score_summary", "policy_id": D.POLICY_ID,
            "primary_view": D.VIEWS[0], "by_view": by, "failure_sample": failure_sample(rows, scoring),
            "counts": {"requests": len(rows), "parents": len({r["parent_command_id"] for r in rows})},
            "interpretation": scoring["interpretation"]}


def failure_sample(rows, scoring) -> list:
    """Per model and view, the first k non-correct completed requests by SHA-256(salt + LF + model + LF + request ID)."""
    out = []
    for key in SYSTEMS:
        for v in D.VIEWS:
            pool = [r for r in rows if r["view_id"] == v and r["models"][key]["outcome"] in ("wrong_object", "ask")]
            pool.sort(key=lambda r: (hashlib.sha256((scoring["failure_salt"] + "\n" + key + "\n" + r["request_id"]).encode()).hexdigest(),
                                     r["request_id"]))
            for r in pool[:scoring["failures_per_cell"]]:
                m = r["models"][key]
                out.append({"model_key": key, "view_id": v, "request_id": r["request_id"], "parent_command_id": r["parent_command_id"],
                            "format": r["format"], "object_count": r["object_count"], "source_relation": r["source_relation"],
                            "source_target_id": r["source_target_id"], "target_letter": r["target_letter"],
                            "outcome": m["outcome"], "choice_code": m["choice_code"], "choice_object_id": m["choice_object_id"],
                            "rules_outcome": r["rules"]["outcome"], "rules_target_id": r["rules"]["target_id"]})
    return out


def _p(f):
    return "-" if f is None else f"{f['numerator']}/{f['denominator']} ({100 * f['value']:.1f}%)"


def render_report(s) -> str:
    L = ["# A2.3d comparison: rules, Qwen2.5-0.5B-Instruct and Qwen2.5-7B-Instruct", "",
         "Development screen on 256 parents of one inspected room, scored against the published (annotated) targets. "
         "Each view shows both formats; the full inventory is the primary comparison.", ""]
    for v in D.VIEWS:
        b = s["by_view"][v]
        L += [f"## {v}" + (" (primary)" if v == s["primary_view"] else " (secondary)"), ""]
        for fm in D.FORMATS:
            x = b["by_format"][fm]
            L += [f"### {fm}", "", "| System | Correct / planned | Correct / object choices | Wrong object | ASK | Over ceiling | Failed |",
                  "|---|---|---|---|---|---|---|"]
            for key in SYSTEMS:
                y = x["systems"][key]
                L.append(f"| {key} | {_p(y['correct_over_planned'])} | {_p(y['correct_over_object_choices'])} | {y['wrong_object']} | "
                         f"{y['ask']} | {y['context_budget_exceeded']} | {y['execution_failed']} |")
            rl = x["systems"]["rules"]
            L.append(f"| rules (resolved to the target) | {_p(rl['correct_over_planned'])} | - | - | - | - | - |")
            L.append(f"| always B (reference) | {_p(x['references']['always_B'])} | | | | | |")
            L.append(f"| always second list position (reference) | {_p(x['references']['always_second_list_position'])} | | | | | |")
            pr = x["pairs"]
            L += ["", "Paired on common completed requests (both / first only / second only / neither):", "",
                  f"- 0.5B vs 7B: {pr['small_vs_large']['both']} / {pr['small_vs_large']['first_only']} / "
                  f"{pr['small_vs_large']['second_only']} / {pr['small_vs_large']['neither']} of {pr['small_vs_large']['common']}"]
            for key in SYSTEMS:
                q = pr[f"{key}_vs_rules"]
                L.append(f"- {key} vs rules: {q['both']} / {q['first_only']} / {q['second_only']} / {q['neither']} of {q['common']}")
            L += ["", "| Relation | Planned | 0.5B | 7B | Rules |", "|---|---|---|---|---|"]
            for lab, y in sorted(x["by_relation"].items()):
                L.append(f"| {lab} | {y['planned']} | {y[SYSTEMS[0]]} | {y[SYSTEMS[1]]} | {y['rules']} |")
            L += ["", "| Candidates | Planned | 0.5B | 7B | Rules | Always B |", "|---|---|---|---|---|---|"]
            for n, y in sorted(x["by_candidate_count"].items(), key=lambda z: int(z[0])):
                L.append(f"| {n} | {y['planned']} | {y[SYSTEMS[0]]} | {y[SYSTEMS[1]]} | {y['rules']} | {y['always_B']} |")
            L.append("")
        L += ["Format pairs per parent (coordinates vs augmented; both / coordinates only / augmented only / neither, on parents "
              "where both formats completed):", ""]
        for key in SYSTEMS:
            q = b["format_pairs"][key]
            L.append(f"- {key}: {q['both']} / {q['first_only']} / {q['second_only']} / {q['neither']} of {q['common']}; "
                     f"difference {_p(q['difference_first_minus_second'])}")
        L += ["", f"Rules outcomes that do not agree with the published target ({b['parent_views']} parent-views): "
              + ", ".join(f"{k}: {n}" for k, n in sorted(b["rules_disagreements_with_source"].items())), ""]
    L += ["## Failure sample for inspection", "", "Drawn by a salted hash among wrong or ASK outcomes; for reading only, never "
          "for retuning.", "", "| Model | View | Request | Format | n | Relation | Target (letter) | Outcome | Choice | Rules |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for f in s["failure_sample"]:
        L.append(f"| {f['model_key']} | {f['view_id']} | {f['request_id']} | {f['format']} | {f['object_count']} | "
                 f"{f['source_relation']} | {f['source_target_id']} ({f['target_letter']}) | {f['outcome']} | "
                 f"{f['choice_code']} {f['choice_object_id'] or ''} | {f['rules_outcome']} {f['rules_target_id'] or ''} |")
    L += ["", "## Reading", ""] + [f"- {x}" for x in s["interpretation"]]
    return "\n".join(L) + "\n"


def score_compare(*, requests, rules, small_run, large_run, bundle, annotations, out, sample=True, fixture_scene_id=None,
                  scoring=None) -> dict:
    """The `score` command; returns the summary. `sample=False`, `fixture_scene_id` and `scoring` serve fixtures only."""
    out = output.refuse_existing(out)
    req = Path(requests)
    for p in (req, Path(rules), Path(small_run), Path(large_run), Path(bundle), Path(annotations)):
        o, q = out.resolve(), p.resolve()
        if o == q or q in o.parents or o in q.parents:
            _fail("E_COMPARE_PATH", [(str(out), f"the output overlaps the input {p}")])
    sc_path = Path(scoring) if scoring is not None else SCORING_PATH
    sc = load_scoring(sc_path)
    for label, bad in (("requests", verify_compare_requests(req)), ("rules", verify_compare_rules(rules, req)),
                       ("0.5B run", verify_compare_results(small_run, req)), ("7B run", verify_compare_results(large_run, req))):
        if bad:
            _fail("E_COMPARE_INPUT", [(label, m) for m in bad[:10]])
    runs = {}
    for key, folder in zip(SYSTEMS, (small_run, large_run)):
        man = json.loads((Path(folder) / "manifest.json").read_bytes())
        if man["model_key"] != key:
            _fail("E_COMPARE_INPUT", [(str(folder), f"this run is of {man['model_key']}, not {key}")])
        runs[key] = rows_of("results.jsonl", (Path(folder) / "results.jsonl").read_bytes(), "E_COMPARE_RESULTS")
    if sample and sha256(Path(annotations).read_bytes()) != sc["annotations_sha256"]:
        _fail("E_COMPARE_REFERENCE", [(str(annotations), "not the pinned IRef-VLA annotation file")])
    rq = rows_of("request-index.jsonl", (req / "request-index.jsonl").read_bytes(), "E_COMPARE_REQUESTS")
    selection = strict_json("selection.json", (req / "selection.json").read_bytes(), "E_COMPARE_REQUESTS")
    src = BL.IN.load_bundle(bundle)
    rman = json.loads((req / "manifest.json").read_bytes())
    if src["manifest_sha256"] != rman["source_bundle"]["manifest_sha256"]:
        _fail("E_COMPARE_BUNDLE", [(str(bundle), "not the bundle the requests were prepared from")])
    by_key = {(r["parent_command_id"], r["view_id"], r["format"]): {
        "request_id": r["request_id"], "derived_scene_id": r["derived_scene_id"], "derived_command_id": r["derived_command_id"],
        "object_ids": list(r["object_ids"]), "mapping": [["", o, 0] for o in r["object_ids"]] + [["K", "ASK", 0]],
        "source_document_path": r["source_document_path"], "source_document_sha256": r["source_document_sha256"]} for r in rq}
    req_like = {"selected": list(selection["selected_parent_ids"]), "by_key": by_key}
    entries = BL.IN.join(req_like, src, sample=sample)
    rules_protocol = BL.IN.relabel(BL.INPUT, BL.A2PROTOCOL.load_protocol)
    sources, ann_info = BL.IN.load_annotations(annotations, entries, req_like, rules_protocol, sample=sample,
                                               fixture_scene_id=fixture_scene_id)
    rrows = rows_of("rules.jsonl", (Path(rules) / "rules.jsonl").read_bytes(), "E_COMPARE_RULES")
    rules_by_pv = {(r["parent_command_id"], r["view_id"]): r for r in rrows}
    scores = score_rows(rq, runs, rules_by_pv, sources)
    summary = summarize(scores, sc)
    files = {"scores.jsonl": encode_jsonl(scores), "summary.json": encode_json(summary),
             "report.md": render_report(summary).encode("utf-8")}
    manifest = {"format_version": 1, "record_type": "iref_compare_score_manifest", "policy_id": D.POLICY_ID,
                "scoring_sha256": sha256(sc_path.read_bytes()), "mode": "pinned_sample" if sample else "fixture",
                "inputs": {"requests_manifest_sha256": sha256((req / "manifest.json").read_bytes()),
                           "rules_manifest_sha256": sha256((Path(rules) / "manifest.json").read_bytes()),
                           "small_run_manifest_sha256": sha256((Path(small_run) / "manifest.json").read_bytes()),
                           "large_run_manifest_sha256": sha256((Path(large_run) / "manifest.json").read_bytes()),
                           "bundle_manifest_sha256": src["manifest_sha256"], "annotations": ann_info},
                "counts": summary["counts"], "outputs": {k: sha256(v) for k, v in files.items()}, "code": code_hashes(),
                "runtime": runtime()}
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=str(out.parent)))
    try:
        for k, v in files.items():
            output._write_file(staging / k, v)
        output._write_file(staging / "manifest.json", encode_json(manifest))
        bad = verify_compare_scores(staging, sc_path)
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


def verify_compare_scores(folder, scoring=None) -> list:
    f = Path(folder)
    present = sorted(p.name for p in f.iterdir() if p.is_file()) if f.is_dir() else []
    if present != sorted(OUTPUTS + ("manifest.json",)):
        return [f"expected exactly {sorted(OUTPUTS + ('manifest.json',))}; found {present}"]
    try:
        manifest = strict_json("manifest.json", (f / "manifest.json").read_bytes(), "E_COMPARE_SCORES")
        scores = rows_of("scores.jsonl", (f / "scores.jsonl").read_bytes(), "E_COMPARE_SCORES")
        summary = strict_json("summary.json", (f / "summary.json").read_bytes(), "E_COMPARE_SCORES")
    except (OSError, EvaluationInputError) as e:
        return [f"unreadable scores: {e}"]
    bad = [f"{n}: changed" for n, want in manifest.get("outputs", {}).items() if sha256((f / n).read_bytes()) != want]
    if bad:
        return bad
    sc = load_scoring(scoring)
    if manifest.get("scoring_sha256") != sha256((Path(scoring) if scoring else SCORING_PATH).read_bytes()):
        return ["scored under another scoring policy"]
    if [r.get("request_index") for r in scores] != list(range(1, len(scores) + 1)):
        return ["scores.jsonl is not one row per request in order"]
    for r in scores:
        for key in SYSTEMS:
            m = r["models"][key]
            if m["outcome"] not in OUTCOMES or (m["outcome"] == "correct") != (m["choice_object_id"] == r["source_target_id"]
                                                                              and m["technical_status"] == "completed"):
                return [f"{r['request_id']}: {key}'s outcome is inconsistent with its choice"]
    if summarize(scores, sc) != summary:
        return ["summary.json differs from a recomputation"]
    if render_report(summary).encode("utf-8") != (f / "report.md").read_bytes():
        return ["report.md differs from a rendering of the summary"]
    return []
