"""The A2.2c selection audit (D77): validation, parsing, projection, selection rows, summary, report and provenance.

For each distinct parent command, the accepted A2.2b parser runs once on its text. A parsed command's required
categories are the sorted unique labels on every node of its one interpretation, both between anchors included. In
each accepted inventory view the selector keeps every object whose category is unknown or among them. An unsupported
command gets an unassessed row per view; nothing is guessed from its words.

Input, role, cross-record and inventory-policy validation is the accepted A2.2b code, called unchanged. Nothing here
reads annotations, statements, graphs, predictions or scores, or runs relations, the resolver or the serializer.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import jsonschema

from ...evaluation.iref_vla import output
from ...evaluation.iref_vla.predict import _check_inputs as accepted_inputs
from ...evaluation.iref_vla.predict import _check_views as accepted_views
from ...evaluation.iref_vla.parse import parse_record
from ...evaluation.iref_vla.protocol import (VIEWS, EvaluationInputError, encode_json, encode_jsonl, issue,
                                             load_protocol, parse_record_issues, ratio, runtime, sha256, strict_json)
from ...resolution.validate import canonical_sha256
from .selector import POLICY_ID, PRIMARY_OBJECT_LIMIT, SENSITIVITY_LIMITS, ProjectedObject, primary_status, select

REPO = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO / "schemas" / "iref-subscene-audit.v1.json"
STATUSES = ("fits", "over_budget", "unassessed_parse")
NULL_WHEN_UNASSESSED = ("required_categories", "retained_object_ids", "retained_unknown_category_ids",
                        "required_object_count", "selection_id")
HASH_RULE = ("the accepted canonical JSON hash (grounding.resolution.validate.canonical_sha256) of the scene with its "
             "objects in lexicographic object-ID order, so the object order of the file cannot change an identity")
CODE = ("grounding/subscenes", "grounding/evaluation", "grounding/contract", "grounding/resolution/validate.py")
SCHEMAS = ("iref-subscene-audit.v1.json", "iref-evaluation.v1.json", "grounding-query.v1.json", "scene.v1.json",
           "scene.v2.json", "command-context.v1.json", "category-map.v1.json")
_VALIDATORS = {}


@dataclass
class Audit:
    rows: list
    summary: dict
    parent_scene_sha256: str


def _validator(name):
    if name not in _VALIDATORS:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        _VALIDATORS[name] = jsonschema.Draft202012Validator(
            {"$schema": schema["$schema"], "$defs": schema["$defs"], "$ref": f"#/$defs/{name}"})
    return _VALIDATORS[name]


def parent_scene_sha256(scene) -> str:
    return canonical_sha256(dict(scene, objects=sorted(scene["objects"], key=lambda o: o["object_id"])))


def selection_id(parent_sha, view_id, retained) -> str:
    """iref.subset. + SHA-256 of the canonical payload: policy, parent-scene hash, view and ordered retained IDs."""
    payload = {"policy_id": POLICY_ID, "parent_scene_sha256": parent_sha, "view_id": view_id,
               "retained_object_ids": list(retained)}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return "iref.subset." + hashlib.sha256(text.encode("utf-8")).hexdigest()


def project(scene, members) -> tuple:
    """The selector's view of a validated scene: IDs and model categories (None if unknown), base view only."""
    return tuple(ProjectedObject(o["object_id"], o["category"]["value"]["model"] if o["category"]["state"] == "known"
                                 else None) for o in scene["objects"] if o["object_id"] in members)


def required_categories(parse, vocabulary) -> frozenset:
    """Every node's category from a parsed command's one interpretation; anything else is an invariant failure."""
    interp = parse["interpretation"]
    if interp.get("kind") != "query" or not isinstance(interp.get("nodes"), list) or not interp["nodes"]:
        raise RuntimeError(f"internal invariant: parsed command {parse['parent_command_id']} has no query nodes")
    found = set()
    for node in interp["nodes"]:
        c = node.get("category") if isinstance(node, dict) else None
        if not isinstance(c, str) or c not in vocabulary:
            raise RuntimeError(f"internal invariant: a node of {parse['parent_command_id']} has category {c!r}, not a "
                               "vocabulary label")
        found.add(c)
    return frozenset(found)


def _row(parse, view_id, categories, projection, parent):
    row = {"format_version": 1, "record_type": "iref_subscene_selection", "policy_id": POLICY_ID,
           "parent_command_id": parse["parent_command_id"], "view_id": view_id, "parse_status": parse["parse_status"],
           "parse_reason": parse["parse_reason"], "base_object_count": len(projection),
           "primary_object_limit": PRIMARY_OBJECT_LIMIT}
    if categories is None:
        row.update(status="unassessed_parse", **{k: None for k in NULL_WHEN_UNASSESSED})
        return row
    retained = select(projection, categories)
    unknown = {o.object_id for o in projection if o.model_category is None}
    row.update(status=primary_status(len(retained)), required_categories=sorted(categories),
               retained_object_ids=list(retained), retained_unknown_category_ids=[i for i in retained if i in unknown],
               required_object_count=len(retained), selection_id=selection_id(parent, view_id, retained))
    return row


def check_rows(rows, projections, parent) -> None:
    """Every row's schema and cross-field rules, and the exact command-view pairing. Raises on the audit's own fault."""
    bad = []
    for r in rows:
        errs = list(_validator("selection").iter_errors(r))
        if errs:
            bad.append(f"{r.get('parent_command_id')!r} {r.get('view_id')!r}: {errs[0].message[:120]}")
            continue
        where = f"{r['parent_command_id']} {r['view_id']}"
        ints = [r["format_version"], r["base_object_count"], r["primary_object_limit"]] + (
            [r["required_object_count"]] if r["required_object_count"] is not None else [])
        if any(type(x) is not int for x in ints):
            bad.append(f"{where}: a count is not a plain integer")
        proj = projections[r["view_id"]]
        if r["base_object_count"] != len(proj):
            bad.append(f"{where}: base count disagrees with the view")
        if r["status"] == "unassessed_parse":
            if r["parse_status"] != "unsupported" or r["parse_reason"] is None or any(r[k] is not None
                                                                                     for k in NULL_WHEN_UNASSESSED):
                bad.append(f"{where}: an unassessed row needs an unsupported parse and null selection fields")
            continue
        ids, unknown = r["retained_object_ids"], {o.object_id for o in proj if o.model_category is None}
        if r["parse_status"] != "parsed" or r["parse_reason"] is not None or any(r[k] is None for k in NULL_WHEN_UNASSESSED) \
                or r["required_object_count"] != len(ids) or r["status"] != primary_status(len(ids)) \
                or ids != sorted(set(ids)) or not set(ids) <= {o.object_id for o in proj} \
                or r["retained_unknown_category_ids"] != [i for i in ids if i in unknown] \
                or r["required_categories"] != sorted(set(r["required_categories"])) \
                or r["selection_id"] != selection_id(parent, r["view_id"], ids):
            bad.append(f"{where}: count, status, order, membership, unknown subset or selection ID disagree")
    pairs = [(r.get("parent_command_id"), r.get("view_id")) for r in rows]
    parents = sorted({p for p, _ in pairs if isinstance(p, str)})
    if pairs != [(p, v) for p in parents for v in VIEWS]:
        bad.append("rows are not exactly one per command and view, ordered by command ID then view")
    if bad:
        raise RuntimeError("internal error: the audit produced inconsistent rows: " + "; ".join(bad[:5]))


def summarize(rows) -> dict:
    """Counts and distributions derived from the rows alone."""
    views = {}
    for v in VIEWS:
        vr = [r for r in rows if r["view_id"] == v]
        assessed = [r for r in vr if r["status"] != "unassessed_parse"]
        n, p = len(vr), len(assessed)
        fits = sum(r["status"] == "fits" for r in assessed)
        hist = Counter(r["required_object_count"] for r in assessed)
        sel = Counter(r["selection_id"] for r in assessed)
        views[v] = {
            "N": n, "P": p, "U": n - p,
            "unsupported_reasons": dict(sorted(Counter(r["parse_reason"] for r in vr
                                                       if r["status"] == "unassessed_parse").items())),
            "required_count_histogram": {str(k): hist[k] for k in sorted(hist)},
            "small_selections": {str(k): hist.get(k, 0) for k in (0, 1, 2)},
            "fits": fits, "over_budget": p - fits,
            "fit_coverage_all": ratio(fits, n), "fit_coverage_parsed": ratio(fits, p),
            "by_limit": {str(lim): {"count": c, "of_all": ratio(c, n), "of_parsed": ratio(c, p)}
                         for lim in SENSITIVITY_LIMITS for c in [sum(r["required_object_count"] <= lim for r in assessed)]},
            "distinct_assessed_selections": len(sel),
            "distinct_fitting_selections": len({r["selection_id"] for r in assessed if r["status"] == "fits"}),
            "commands_per_selection": {str(k): m for k, m in sorted(Counter(sel.values()).items())},
            "retained_unknown_histogram": {str(k): m for k, m in sorted(
                Counter(len(r["retained_unknown_category_ids"]) for r in assessed).items())},
            "base_object_count": vr[0]["base_object_count"] if vr else 0}
    status = {(r["parent_command_id"], r["view_id"]): r["status"] for r in rows}
    parents = sorted({r["parent_command_id"] for r in rows})
    counts = {a: {b: 0 for b in STATUSES} for a in STATUSES}
    for c in parents:
        counts[status[(c, "full_inventory")]][status[(c, "source_known_nyu")]] += 1
    return {"format_version": 1, "record_type": "iref_subscene_summary", "policy_id": POLICY_ID,
            "primary_object_limit": PRIMARY_OBJECT_LIMIT, "sensitivity_limits": list(SENSITIVITY_LIMITS),
            "population": {"parent_commands": len(parents), "rows": len(rows)}, "views": views,
            "paired": {"rows": "full_inventory", "columns": "source_known_nyu", "counts": counts, "total": len(parents)}}


def audit(scene, commands, category_map, views_manifest, *, protocol=None) -> Audit:
    """Selection rows and their summary for every parent command in both inventory views."""
    protocol = protocol or load_protocol()
    accepted_inputs(scene, commands, category_map)
    membership = accepted_views(views_manifest, scene, category_map, protocol)
    parent = parent_scene_sha256(scene)
    vocabulary, colours = category_map["model_vocabulary"], protocol["colour_labels"]
    labels = frozenset(vocabulary)
    projections = {v: project(scene, membership[v]) for v in VIEWS}
    by_text, rows = {}, []
    for c in sorted(commands, key=lambda c: c["command_id"]):
        if c["text"] not in by_text:  # one parse per distinct text, rebound to each parent
            by_text[c["text"]] = parse_record(c["command_id"], c["text"], categories=vocabulary, colours=colours)
        p = dict(copy.deepcopy(by_text[c["text"]]), parent_command_id=c["command_id"])
        found = parse_record_issues(p, f"parse {c['command_id']}")
        if found:
            raise RuntimeError(f"internal invariant: the accepted parser produced an invalid record: {found[:2]}")
        cats = required_categories(p, labels) if p["parse_status"] == "parsed" else None
        rows.extend(_row(p, v, cats, projections[v], parent) for v in VIEWS)
    check_rows(rows, projections, parent)
    summary = summarize(rows)
    errs = list(_validator("summary").iter_errors(summary))
    if errs:
        raise RuntimeError(f"internal error: the summary is invalid: {errs[0].message[:160]}")
    return Audit(rows=rows, summary=summary, parent_scene_sha256=parent)


def _fmt(r):
    return f"{r['numerator']}/{r['denominator']}" + (" (null)" if r["value"] is None else f" ({r['value']:.3f})")


def render_report(s) -> str:
    """The Markdown report, rendered only from the summary."""
    v = s["views"]
    lines = ["# IRef-VLA category-complete selection audit (A2.2c)", "",
             "- This is command-conditioned category-complete selection under the accepted parser, not a spatial or "
             "visibility crop.",
             "- Fitting within ten objects is a count constraint, not a Quest latency, memory or context-window result.",
             "- Unsupported commands remain in the total population and were not assessed for selection.",
             "- The full and source-known inventories express different assumptions about unknown-category objects.",
             "- All material comes from one development room. Distinct commands and subsets are not independent rooms.",
             "",
             f"Policy `{s['policy_id']}`: for each parsed command, every object of the base inventory view whose model "
             "category the parse names on any node, plus every object whose category is unknown. Colour, size, "
             "geometry, pose and relations play no part. A set above the provisional budget of "
             f"{s['primary_object_limit']} objects is recorded whole as `over_budget`, never truncated.", "",
             f"Population: {s['population']['parent_commands']} commands, {s['population']['rows']} rows.", "",
             "## By inventory view", "", "| Measure | full_inventory | source_known_nyu |", "|---|---:|---:|"]
    for label, key in (("Commands (N)", "N"), ("Parsed and assessed (P)", "P"), ("Unassessed (U)", "U"),
                       ("Fits", "fits"), ("Over budget", "over_budget"), ("Base objects", "base_object_count"),
                       ("Distinct assessed selections", "distinct_assessed_selections"),
                       ("Distinct fitting selections", "distinct_fitting_selections")):
        lines.append(f"| {label} | {v['full_inventory'][key]} | {v['source_known_nyu'][key]} |")
    for label, key in (("Fits / N", "fit_coverage_all"), ("Fits / P", "fit_coverage_parsed")):
        lines.append(f"| {label} | {_fmt(v['full_inventory'][key])} | {_fmt(v['source_known_nyu'][key])} |")
    lines += ["", "## Unsupported commands (not assessed)", "", "| Parser reason | full_inventory | source_known_nyu |",
              "|---|---:|---:|"]
    for reason in sorted(set(v["full_inventory"]["unsupported_reasons"]) | set(v["source_known_nyu"]["unsupported_reasons"])):
        lines.append(f"| {reason} | {v['full_inventory']['unsupported_reasons'].get(reason, 0)} | "
                     f"{v['source_known_nyu']['unsupported_reasons'].get(reason, 0)} |")
    lines += ["", "## Required object counts (assessed rows)", "", "| Objects | full_inventory | source_known_nyu |",
              "|---:|---:|---:|"]
    sizes = sorted({int(k) for x in v.values() for k in x["required_count_histogram"]})
    lines += [f"| {k} | {v['full_inventory']['required_count_histogram'].get(str(k), 0)} | "
              f"{v['source_known_nyu']['required_count_histogram'].get(str(k), 0)} |" for k in sizes]
    lines += ["", "Selections of zero, one or two objects: full " + ", ".join(f"{k}: {n}" for k, n in
                                                                          v["full_inventory"]["small_selections"].items())
              + "; source-known " + ", ".join(f"{k}: {n}" for k, n in v["source_known_nyu"]["small_selections"].items())
              + ".", "", "## Commands needing at most each limit", "",
              "| Limit | full: of N | full: of P | source-known: of N | source-known: of P |", "|---:|---:|---:|---:|---:|"]
    for lim in s["sensitivity_limits"]:
        a, b = v["full_inventory"]["by_limit"][str(lim)], v["source_known_nyu"]["by_limit"][str(lim)]
        lines.append(f"| {lim} | {_fmt(a['of_all'])} | {_fmt(a['of_parsed'])} | {_fmt(b['of_all'])} | {_fmt(b['of_parsed'])} |")
    c = s["paired"]["counts"]
    lines += ["", f"## Paired primary statuses (rows full_inventory, columns source_known_nyu; total {s['paired']['total']})",
              "", "| full \\ source-known | " + " | ".join(STATUSES) + " |", "|---|" + "---:|" * len(STATUSES)]
    lines += [f"| {a} | " + " | ".join(str(c[a][b]) for b in STATUSES) + " |" for a in STATUSES]
    lines += ["", "## Distributions", ""]
    for vid in VIEWS:
        lines.append(f"- {vid}: retained unknown-category objects per assessed row "
                     f"{v[vid]['retained_unknown_histogram'] or 'none'}; commands per distinct selection "
                     f"{v[vid]['commands_per_selection'] or 'none'}.")
    lines += ["", "## Notes", "",
              "- Repeated commands share a selection; distinct selections are not independent scenes.",
              "- A future comparison of prompt formats would have to give both formats the identical preselected "
              "inventory. A parser-conditioned subset is an additional pipeline assumption, not a neutral end-to-end "
              "benchmark, and this audit doesn't choose it.",
              "- A per-command subset can change the cached scene prefix, so A1's warm-cache timings don't transfer to "
              "these selections."]
    return "\n".join(lines) + "\n"


def code_hashes() -> dict:
    out = {}
    for root in CODE:
        path = REPO / root
        for p in ([path] if path.is_file() else sorted(path.rglob("*"))):
            if p.is_file() and p.suffix in (".py", ".json") and "__pycache__" not in p.parts:
                out[p.relative_to(REPO).as_posix()] = sha256(p.read_bytes())
    for name in SCHEMAS:
        out[f"schemas/{name}"] = sha256((REPO / "schemas" / name).read_bytes())
    return out


def _read_commands(cdir: Path):
    """The declared folder's direct .json files only, each named by its command_id (A2.2b's convention)."""
    if not cdir.is_dir():
        raise EvaluationInputError([issue(str(cdir), "E_EVAL_INPUT", "the commands path is not a folder")])
    entries = sorted(cdir.iterdir(), key=lambda p: p.name)
    strays = [p.name for p in entries if not (p.is_file() and p.suffix == ".json")]
    if strays:
        raise EvaluationInputError([issue(str(cdir), "E_EVAL_INPUT", f"only command .json files are allowed; found "
                                                                     f"{strays[:5]}")])
    cmds, lines = [], []
    for p in entries:
        b = p.read_bytes()
        c = strict_json(p.name, b, "E_EVAL_INPUT")
        if not isinstance(c, dict) or c.get("command_id") != p.stem:
            raise EvaluationInputError([issue(p.name, "E_EVAL_INPUT", "a command file must hold one command named by "
                                                                       "its command_id")])
        cmds.append(c)
        lines.append(f"{c['command_id']} {sha256(b)}\n")
    return cmds, lines


def run_audit(*, scene, commands, category_map, inventory_views, out, sample_only=True) -> dict:
    """The command line's operation: read exactly the four inputs, audit, publish four artifacts; returns the summary."""
    out = output.refuse_existing(out)
    protocol = load_protocol()
    paths = {"scene": Path(scene), "category_map": Path(category_map), "inventory_views": Path(inventory_views)}
    data = {}
    for name, p in paths.items():
        try:
            data[name] = p.read_bytes()
        except OSError as e:
            raise EvaluationInputError([issue(str(p), "E_EVAL_INPUT", f"cannot read the {name} file: {e}")]) from e
    records = {name: strict_json(name, data[name], "E_EVAL_INPUT") for name in paths}
    for name in ("scene", "category_map"):
        if not isinstance(records[name], dict):
            raise EvaluationInputError([issue(str(paths[name]), "E_EVAL_INPUT",
                                              f"the {name.replace('_', ' ')} file must hold one JSON object")])
    cmds, lines = _read_commands(Path(commands))
    if sample_only and (records["scene"].get("scene_id") != protocol["sample"]["scene_id"]
                        or records["category_map"].get("map_id") != protocol["sample"]["category_map_id"]):
        raise EvaluationInputError([issue("scene", "E_EVAL_INPUT", f"this command line audits only the A2.2a sample "
                                                                   f"{protocol['sample']['scene_id']!r} with map "
                                                                   f"{protocol['sample']['category_map_id']!r}")])
    result = audit(records["scene"], cmds, records["category_map"], records["inventory_views"], protocol=protocol)
    files = {"selections.jsonl": encode_jsonl(result.rows), "summary.json": encode_json(result.summary),
             "report.md": render_report(result.summary).encode("utf-8")}
    manifest = {"format_version": 1, "record_type": "iref_subscene_manifest", "policy_id": POLICY_ID,
                "primary_object_limit": PRIMARY_OBJECT_LIMIT, "sensitivity_limits": list(SENSITIVITY_LIMITS),
                "parent_scene": {"scene_id": records["scene"]["scene_id"],
                                 "scene_revision": records["scene"]["scene_revision"],
                                 "canonical_sha256": result.parent_scene_sha256, "hash_rule": HASH_RULE},
                "inputs": {"hash_kind": "file_bytes", "scene_sha256": sha256(data["scene"]),
                           "category_map_sha256": sha256(data["category_map"]),
                           "inventory_views_sha256": sha256(data["inventory_views"]),
                           "commands_sha256": sha256("".join(sorted(lines)).encode("utf-8")), "commands": len(cmds)},
                "parser": {"protocol_id": protocol["protocol_id"], "protocol_sha256": canonical_sha256(protocol),
                           "module": "grounding.evaluation.iref_vla.parse (A2.2b, unchanged)"},
                "code": code_hashes(), "runtime": runtime(),
                "outputs": {name: sha256(b) for name, b in files.items()}}
    errs = list(_validator("manifest").iter_errors(manifest))
    if errs:
        raise RuntimeError(f"internal error: the manifest is invalid: {errs[0].message[:160]}")
    files["manifest.json"] = encode_json(manifest)
    output.publish(out, files)
    return result.summary
