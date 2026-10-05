"""Scoring for the A2.2b development evaluation (D75): completed predictions against the separate source annotations.

score() first checks the whole prediction output again (records, identities, semantic hash, recomputed summary) and
the annotation bundle (schema, one target and one relation label per command, targets present in both views, exactly
the predicted command population). It never reruns or changes a prediction. A correct selection is only a completed,
resolved result whose target is the mapped source target; an ambiguous result listing that target is not one. Each
command counts once; ratios keep their numerator and denominator, and a zero denominator gives null.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ...resolution import validate as RV
from . import output
from .predict import bin_of, check_prediction, load_prediction
from .protocol import (BINS, EVALUATION_VERSION, VIEWS, EvaluationInputError, _path, _validator, code_hashes, encode_json,
                       encode_jsonl, issue, ratio, runtime, schema_issues, sha256, strict_json)


@dataclass
class ScoreResult:
    scores: list
    summary: dict
    report: str
    manifest: dict


def _has_size(payload) -> bool:
    return bool(payload["target_size_used"]) or any(a["size_used"] for a in payload["anchors"].values())


def _annotation_issues(ann, prediction):
    out = []

    def bad(path, message):
        out.append(issue(path, "E_EVAL_REFERENCE", message))
    if not isinstance(ann, dict):
        bad("annotations", "expected an annotation bundle object")
        return out, None
    entries_raw = ann.get("entries") if isinstance(ann.get("entries"), list) else []
    for e in sorted(_validator("annotations").iter_errors(ann), key=lambda e: [str(p) for p in e.absolute_path]):
        path, who = list(e.absolute_path), ""
        if len(path) >= 2 and path[0] == "entries" and isinstance(path[1], int) and path[1] < len(entries_raw) \
                and isinstance(entries_raw[path[1]], dict):
            entry = entries_raw[path[1]]
            who = f" (annotation {entry.get('annotation_id')!r} of command {entry.get('command_id')!r})"
        bad(f"annotations {_path(path)}", e.message[:200] + who)
    if out:
        return out, None
    entries = ann["entries"]
    if type(ann["schema_version"]) is not int:
        bad("annotations $.schema_version", "not a plain integer")
    if ann["scene_id"] != prediction.manifest["parent"]["scene_id"]:
        bad("annotations $.scene_id", f"{ann['scene_id']!r} is not the predicted scene "
                                      f"{prediction.manifest['parent']['scene_id']!r}")
    counts = Counter(e["annotation_id"] for e in entries)
    for aid in sorted(a for a, n in counts.items() if n > 1):
        bad("annotations", f"annotation {aid!r} is listed {counts[aid]} times")
    groups = {}
    for e in entries:
        groups.setdefault(e["command_id"], []).append(e)
    full = {o["object_id"] for o in prediction.scenes["full_inventory"]["objects"]}
    known = {o["object_id"] for o in prediction.scenes["source_known_nyu"]["objects"]}
    for cid in sorted(groups):
        es = groups[cid]
        where = f"annotations command {cid!r}"
        indices = [e["source_annotation_index"] for e in es]
        if any(type(i) is not int for i in indices) or len(set(indices)) != len(indices):
            bad(where, f"command {cid} has repeated or non-integer annotation indices {indices}")
        for e in es:
            p = e["source_payload"]
            if not e["annotation_id"].startswith(cid + ".a"):
                bad(where, f"annotation {e['annotation_id']!r} is not named for command {cid}")
            if not (isinstance(p.get("relation"), str) and p["relation"] and isinstance(p.get("target_size_used"), str)
                    and isinstance(p.get("anchors"), dict) and p["anchors"]
                    and all(isinstance(a, dict) and isinstance(a.get("size_used"), str) for a in p["anchors"].values())):
                bad(where, f"annotation {e['annotation_id']!r} of command {cid} lacks a relation label, target size "
                           "word or anchor size words in the pinned form")
        if out:
            continue
        targets = sorted({e["mapped_references"]["target"] for e in es})
        if len(targets) > 1:
            bad(where, f"command {cid} has contradictory targets {targets} across its annotations")
        relations = sorted({e["source_payload"]["relation"] for e in es})
        if len(relations) > 1:
            bad(where, f"command {cid} has conflicting source relation labels {relations}")
        for t in targets:
            if t not in full:
                bad(where, f"target {t} of command {cid} is not in the canonical full inventory")
            elif t not in known:
                bad(where, f"target {t} of command {cid} is not in the source-known view; this sample's protocol "
                           "assumes it is")
    parents = {p["parent_command_id"] for p in prediction.parses}
    for cid in sorted(parents - set(groups)):
        bad("annotations", f"command {cid} was predicted but has no source annotation")
    for cid in sorted(set(groups) - parents):
        bad("annotations", f"command {cid} has annotations but no prediction")
    return out, groups


def _metrics(rows, vid):
    n = len(rows)
    p = sum(r["parse_status"] == "parsed" for r in rows)
    r_ = sum(r["views"][vid]["bin"] == "resolved" for r in rows)
    c = sum(r["views"][vid]["correct"] for r in rows)
    bins = {b: 0 for b in BINS}
    for r in rows:
        bins[r["views"][vid]["bin"]] += 1
    return {"N": n, "P": p, "R": r_, "C": c, "parser_coverage": ratio(p, n), "resolved_fraction": ratio(r_, n),
            "all_command_agreement": ratio(c, n), "resolved_agreement": ratio(c, r_), "bins": bins}


def _by(rows, key):
    out = {}
    for value in sorted({key(r) for r in rows}):
        subset = [r for r in rows if key(r) == value]
        out[value] = {vid: _metrics(subset, vid) for vid in VIEWS}
    return out


def _summary(rows, prediction, entries):
    views = {}
    for vid in VIEWS:
        recs = [r for r in prediction.predictions if r["view_id"] == vid]
        views[vid] = dict(_metrics(rows, vid),
                          budget_limits=dict(sorted(Counter(r["resolution"]["budget"]["name"] for r in recs
                                                            if r["resolution"]["budget"]).items())),
                          reason_codes_nonexclusive=dict(sorted(Counter(c for r in rows
                                                                        for c in r["views"][vid]["reason_codes"]).items())))
    counts = {a: {b: 0 for b in BINS} for a in BINS}
    for r in rows:
        counts[r["views"]["full_inventory"]["bin"]][r["views"]["source_known_nyu"]["bin"]] += 1
    full_r = {r["parent_command_id"]: r["views"]["full_inventory"] for r in rows}
    known_r = {r["parent_command_id"]: r["views"]["source_known_nyu"] for r in rows}
    both = [c for c in full_r if full_r[c]["bin"] == "resolved" and known_r[c]["bin"] == "resolved"]
    multiplicity = Counter(r["annotation_count"] for r in rows)
    parser_size = {r["parent_command_id"] for r in rows if "size_comparison" in r["features"]}
    source_size = {r["parent_command_id"] for r in rows if r["source_size_any"]}
    return {
        "format_version": 1, "record_type": "iref_score_summary", "protocol_id": prediction.manifest["protocol_id"],
        "population": {"commands": len(rows), "annotation_records": len(entries),
                       "repeated_expression_groups": sum(n > 1 for n in (r["annotation_count"] for r in rows)),
                       "multiplicity": {str(k): v for k, v in sorted(multiplicity.items())}},
        "parser": {"parsed": sum(r["parse_status"] == "parsed" for r in rows),
                   "unsupported_reasons": dict(sorted(Counter(r["parse_reason"] for r in rows
                                                              if r["parse_status"] == "unsupported").items())),
                   "features_nonexclusive": dict(sorted(Counter(f for r in rows for f in r["features"]).items()))},
        "views": views,
        "transitions": {"rows": "full_inventory", "columns": "source_known_nyu", "counts": counts, "total": len(rows)},
        "selection_changes": {"both_resolved_same_target": sum(full_r[c]["target_id"] == known_r[c]["target_id"] for c in both),
                              "both_resolved_different_target": sum(full_r[c]["target_id"] != known_r[c]["target_id"]
                                                                    for c in both),
                              "resolved_in_full_only": sum(full_r[c]["bin"] == "resolved" and known_r[c]["bin"] != "resolved"
                                                           for c in full_r),
                              "resolved_in_source_known_only": sum(full_r[c]["bin"] != "resolved"
                                                                   and known_r[c]["bin"] == "resolved" for c in full_r)},
        "breakdowns": {"source_relation": _by(rows, lambda r: r["source_relation"]),
                       "source_size": _by(rows, lambda r: "with_size" if r["source_size_any"] else "without_size"),
                       "parser": _by(rows, lambda r: r["parse_reason"] or "parsed")},
        "size": {"parser_detected": len(parser_size), "source_labelled": len(source_size),
                 "both": len(parser_size & source_size), "parser_only": len(parser_size - source_size),
                 "source_only": len(source_size - parser_size),
                 "within_command_disagreement": sum(r["source_size_disagreement"] for r in rows)},
        "technical_failures": sum(views[v]["bins"]["invalid_input"] + views[v]["bins"]["budget_exceeded"] for v in VIEWS)}


def _fmt(x):
    if x["value"] is None:
        return f"{x['numerator']}/{x['denominator']} (null)"
    return f"{x['numerator']}/{x['denominator']} ({x['value']:.3f})"


def render_report(s) -> str:
    """The Markdown report, rendered only from the machine-readable summary."""
    v = s["views"]
    lines = ["# IRef-VLA development evaluation: text-only rules baseline (A2.2b)", "",
             "This is one inspected development room with annotated geometry, a sample-specific rules parser and "
             "provisional relation semantics. Its counts describe that room under this protocol: they are not held-out "
             "grounding accuracy, and agreement with source labels does not show that our relation semantics match the "
             "dataset's. Each command counts once; nothing below averages the two views.", "",
             "## Population", "",
             f"Commands: {s['population']['commands']}. Annotation records: {s['population']['annotation_records']}, in "
             f"{s['population']['repeated_expression_groups']} repeated-expression groups. Multiplicity (annotations per "
             f"command: commands): " + ", ".join(f"{k}: {n}" for k, n in s["population"]["multiplicity"].items()) + ".", "",
             "## Parser", "", "| Parse | Commands |", "|---|---:|", f"| parsed | {s['parser']['parsed']} |"]
    lines += [f"| unsupported: {k} | {n} |" for k, n in s["parser"]["unsupported_reasons"].items()]
    lines += ["", "Recognized features (nonexclusive): " + (", ".join(f"{k} {n}" for k, n in
                                                                       s["parser"]["features_nonexclusive"].items()) or "none") + ".",
              "", "## Outcomes by inventory view", "", "| Measure | full_inventory | source_known_nyu |", "|---|---:|---:|"]
    for label, key in (("Commands (N)", "N"), ("Parsed (P)", "P"), ("Resolved (R)", "R"), ("Source target selected (C)", "C")):
        lines.append(f"| {label} | {v['full_inventory'][key]} | {v['source_known_nyu'][key]} |")
    for label, key in (("Parser coverage P/N", "parser_coverage"), ("Resolved fraction R/N", "resolved_fraction"),
                       ("All-command target agreement C/N", "all_command_agreement"),
                       ("Agreement among resolved C/R", "resolved_agreement")):
        lines.append(f"| {label} | {_fmt(v['full_inventory'][key])} | {_fmt(v['source_known_nyu'][key])} |")
    lines += ["", "## Outcome bins (mutually exclusive; each column sums to N)", "",
              "| Outcome | full_inventory | source_known_nyu |", "|---|---:|---:|"]
    lines += [f"| {b} | {v['full_inventory']['bins'][b]} | {v['source_known_nyu']['bins'][b]} |" for b in BINS]
    t = s["transitions"]
    lines += ["", f"## Paired transitions (rows {t['rows']}, columns {t['columns']}; total {t['total']})", "",
              "| full \\ source-known | " + " | ".join(BINS) + " |", "|---|" + "---:|" * len(BINS)]
    lines += [f"| {a} | " + " | ".join(str(t["counts"][a][b]) for b in BINS) + " |" for a in BINS]
    sc = s["selection_changes"]
    lines += ["", f"Target selections: resolved in both views with the same target {sc['both_resolved_same_target']}, "
                  f"with different targets {sc['both_resolved_different_target']}; resolved in the full inventory only "
                  f"{sc['resolved_in_full_only']}; in the source-known view only {sc['resolved_in_source_known_only']}. "
                  "Neither view is deployment truth.", ""]
    for title, key in (("source relation label", "source_relation"), ("source size words", "source_size"),
                       ("parser status and reason", "parser")):
        lines += [f"## By {title}", "", "| Value | N | full: R | full: C | source-known: R | source-known: C |",
                  "|---|---:|---:|---:|---:|---:|"]
        for value, per in s["breakdowns"][key].items():
            lines.append(f"| {value} | {per['full_inventory']['N']} | {per['full_inventory']['R']} | "
                         f"{per['full_inventory']['C']} | {per['source_known_nyu']['R']} | {per['source_known_nyu']['C']} |")
        lines.append("")
    z = s["size"]
    lines += ["## Size words", "", f"Parser-detected size modifiers: {z['parser_detected']} commands; source-labelled: "
                                   f"{z['source_labelled']}; both {z['both']}, parser only {z['parser_only']}, source "
                                   f"only {z['source_only']}; commands whose annotations disagree on size: "
                                   f"{z['within_command_disagreement']}.", "",
              "## Technical failures, budgets and reason codes (nonexclusive)", "",
              f"Technical failures (invalid_input or budget_exceeded, both views): {s['technical_failures']}."]
    for vid in VIEWS:
        lines.append(f"- {vid}: budget limits {v[vid]['budget_limits'] or 'none'}; reason codes "
                     f"{v[vid]['reason_codes_nonexclusive'] or 'none'}.")
    return "\n".join(lines) + "\n"


def score(prediction, annotations, *, predictions_file_sha256=None, annotations_file_sha256=None) -> ScoreResult:
    """Per-command scores, the summary, the report and the scoring manifest. Changes nothing it is given."""
    found = check_prediction(prediction)
    if found:
        raise EvaluationInputError(found)
    found, groups = _annotation_issues(annotations, prediction)
    if found:
        raise EvaluationInputError(found)
    parse_of = {p["parent_command_id"]: p for p in prediction.parses}
    rec_of = {(r["parent_command_id"], r["view_id"]): r for r in prediction.predictions}
    rows = []
    for cid in sorted(parse_of):
        es = sorted(groups[cid], key=lambda e: e["source_annotation_index"])
        target, sizes = es[0]["mapped_references"]["target"], [_has_size(e["source_payload"]) for e in es]
        views = {}
        for vid in VIEWS:
            rec = rec_of[(cid, vid)]
            b, res = bin_of(rec), rec["resolution"]["result"]
            tid = res["target_id"] if b == "resolved" else None
            views[vid] = {"bin": b, "target_id": tid, "correct": b == "resolved" and tid == target,
                          "reason_codes": sorted(res["reason_codes"]) if res else []}
        p = parse_of[cid]
        rows.append({"format_version": 1, "record_type": "iref_score", "parent_command_id": cid, "text": p["text"],
                     "source_target_id": target, "annotation_ids": [e["annotation_id"] for e in es],
                     "annotation_count": len(es), "source_relation": es[0]["source_payload"]["relation"],
                     "source_size_any": any(sizes), "source_size_disagreement": len(set(sizes)) > 1,
                     "parse_status": p["parse_status"], "parse_reason": p["parse_reason"], "features": list(p["features"]),
                     "views": views})
    for r in rows:
        bad = schema_issues("score", r, f"score {r['parent_command_id']}", "E_EVAL_INTERNAL")
        if bad:
            raise RuntimeError(f"internal error: a score record is invalid: {bad[:2]}")
    summary = _summary(rows, prediction, annotations["entries"])
    manifest = {"format_version": 1, "record_type": "iref_score_manifest", "evaluation_version": EVALUATION_VERSION,
                "protocol_id": prediction.manifest["protocol_id"],
                "predictions": {"semantic_hash": prediction.manifest["semantic_prediction_hash"],
                                "predictions_jsonl_sha256": predictions_file_sha256
                                or sha256(encode_jsonl(prediction.predictions)),
                                "hash_kind": "file_bytes" if predictions_file_sha256 else "encoded_records",
                                "manifest_sha256": sha256(encode_json(prediction.manifest))},
                "annotations": {"sha256": annotations_file_sha256 or RV.canonical_sha256(annotations),
                                "hash_kind": "file_bytes" if annotations_file_sha256 else "canonical_json",
                                "scene_id": annotations["scene_id"], "source_commit": annotations["source_commit"],
                                "statement_sha256": annotations["statement_sha256"],
                                "records": len(annotations["entries"])},
                "code": code_hashes(), "runtime": runtime()}
    return ScoreResult(scores=rows, summary=summary, report=render_report(summary), manifest=manifest)


def run_score(*, predictions, annotations, out) -> dict:
    """The command line's operation: load and check the predictions, read the annotations, score, publish."""
    out = output.refuse_existing(out)
    prediction = load_prediction(predictions)
    try:
        data = Path(annotations).read_bytes()
    except OSError as e:
        raise EvaluationInputError([issue(str(annotations), "E_EVAL_REFERENCE", f"cannot read: {e}")]) from e
    result = score(prediction, strict_json("annotations", data, "E_EVAL_REFERENCE"),
                   predictions_file_sha256=prediction.raw_predictions_sha256, annotations_file_sha256=sha256(data))
    output.publish(out, {"scores.jsonl": encode_jsonl(result.scores), "summary.json": encode_json(result.summary),
                         "report.md": result.report.encode("utf-8"), "manifest.json": encode_json(result.manifest)})
    return result.summary
