"""A2.4's training labels and their checks (D109: a positive-reference pilot).

**The label builder** maps a parent's IRef-VLA annotation to a v1 semantic structure by a written table
(`RELATION_MAP`): `closest`, `second_closest`, `third_closest` and the `farthest` forms to rank k; `near`, `below`, `on`
and `between` unchanged; anchors in the annotation's order (`anchor_1` first, which is the written order in the
sample's non-plural statements); a colour only when the annotation says the text used it; any size word, any relation
outside v1 (`above`, `in`), any label outside the vocabulary or any colour outside the ten makes the parent unsupported.
Every entry of a parent must give the same structure. Nothing is repaired.

**The checks**, none of them the builder itself:

1. the annotation's own consistency (A2.6b's correction 1: one target, one parsing label over all entries);
2. the accepted query contract (`resolution.validate.check_query`) on the converted query;
3. the accepted rule parser (A2.2b), as a diagnostic: a different structure quarantines the parent. The rule parser is
   not truth; quarantine is an eligibility restriction of this pilot, not a judgement that either side is right;
4. per inventory view: the target is among the view's objects, and the accepted resolver, given the label's query on
   that view's derived scene, resolves to exactly the source target.

A parent joins the common cohort when 1 to 3 pass and its source-known view is supported. Its direct-training rows are
the views that pass 4 on their own; its parser example is one per parent. Every exclusion keeps its reason.
"""
from __future__ import annotations

import copy
import importlib
from pathlib import Path

from . import language as L

RELATION_MAP = {"closest": ("closest", 1), "second_closest": ("closest", 2), "third_closest": ("closest", 3),
                "farthest": ("farthest", 1), "second_farthest": ("farthest", 2), "third_farthest": ("farthest", 3),
                "near": ("near", None), "below": ("below", None), "on": ("on", None), "between": ("between", None)}
ANCHORS = {"binary": 1, "ternary": 2}
VIEWS = ("full_inventory", "source_known_nyu")
REPO = Path(__file__).resolve().parents[2]
REL = REPO / "grounding" / "relations" / "relations.v1.json"
DIRS = REPO / "grounding" / "relations" / "directions.v1.json"


class Unsupported(ValueError):
    """A parent the v1 label cannot express, with a reason code."""


def _colour(value, spec, where):
    if value in ("", None):
        return None
    if value not in spec["colours"]:
        raise Unsupported(f"unsupported_semantics:colour_{where}")
    return value


def structure(payload, spec) -> dict:
    """One annotation entry's v1 structure, or Unsupported(reason)."""
    rel = payload["relation"]
    if rel not in RELATION_MAP:
        raise Unsupported(f"unsupported_semantics:relation_{rel}")
    name, k = RELATION_MAP[rel]
    anchors = payload["anchors"]
    want = 2 if name == "between" else 1
    if ANCHORS.get(payload["relation_type"]) != want or len(anchors) != want:
        raise Unsupported("mapping_disagreement:relation_type")
    labels = set(spec["categories"]["labels"])
    if payload.get("target_size_used"):
        raise Unsupported("unsupported_semantics:size_word")
    if payload["target_class"] not in labels:
        raise Unsupported("unsupported_semantics:target_class")
    objs = []
    for i in range(1, want + 1):
        a = anchors.get(f"anchor_{i}")
        if a is None:
            raise Unsupported("mapping_disagreement:anchor_keys")
        if a.get("size_used"):
            raise Unsupported("unsupported_semantics:size_word")
        if a["class"] not in labels:
            raise Unsupported("unsupported_semantics:anchor_class")
        objs.append({"category": a["class"], "colour": _colour(a.get("color_used"), spec, "anchor")})
    return {"target": {"category": payload["target_class"], "colour": _colour(payload.get("target_color_used"), spec,
                                                                                "target")},
            "relation": name, "k": k, "anchors": objs}


def build(entries, spec) -> dict:
    """{'structure', 'string'} for a parent's entries, or Unsupported. Every entry must give the same structure."""
    found = []
    for e in entries:
        found.append(structure(e["source_payload"], spec))
    if not found:
        raise Unsupported("annotation_disagreement:no_entry")
    if any(f != found[0] for f in found):
        raise Unsupported("annotation_disagreement:label_fields")
    return {"structure": found[0], "string": L.render(found[0])}


def compare(label, rule_interp) -> list:
    """The fields where the rule parser's structure differs from the label's (order-sensitive anchors)."""
    try:
        r = L.from_interpretation(rule_interp)
    except L.LanguageError:
        return ["rule_parse_outside_v1"]
    out = []
    if r["target"] != label["target"]:
        out.append("target")
    if (r["relation"], r["k"]) != (label["relation"], label["k"]):
        out.append("relation")
    if r["anchors"] != label["anchors"]:
        out.append("anchors" if sorted(map(str, r["anchors"])) != sorted(map(str, label["anchors"])) else "anchor_order")
    return out


class Checker:
    """The accepted parser, validator and resolver, configured as A2.3b's rules baseline configures them."""

    def __init__(self, spec):
        self.spec = spec
        self.PARSE = importlib.import_module("grounding.evaluation.iref_vla.parse")
        self.PREDICT = importlib.import_module("grounding.evaluation.iref_vla.predict")
        self.RESOLVE = importlib.import_module("grounding.resolution.resolve")
        self.RV = importlib.import_module("grounding.resolution.validate")
        self.proto = importlib.import_module("grounding.evaluation.iref_vla.protocol").load_protocol()
        self._configs = {}

    def config(self, cmap):
        key = cmap["map_id"]
        if key not in self._configs:
            cfg = self.PREDICT.resolver_config(self.proto, cmap)
            bad = self.RV.check_config(cfg)
            if bad:
                raise ValueError(f"the resolver configuration does not validate: {bad[:2]}")
            self._configs[key] = cfg
        return self._configs[key]

    def rule_parse(self, text, cmap) -> dict:
        return self.PARSE.parse(text, categories=cmap["model_vocabulary"], colours=self.proto["colour_labels"])

    def resolve(self, interp, scene, command, cmap) -> dict:
        q = L.query(copy.deepcopy(interp), scene, command, self.proto["query_suffix"])
        found, _ = self.RV.check_query(q)
        if found:
            return {"outcome": "invalid_label_query", "target_id": None, "candidate_ids": [],
                    "issues": [f"{p['code']}: {p['path']}" for p in found[:3]]}
        res = self.RESOLVE.resolve(scene, command, q, resolver_config=self.config(cmap), relation_config_path=REL,
                                   direction_config_path=DIRS, category_maps=[cmap], trace=False)
        done = res["processing_status"] == "completed"
        outcome = res["result"]["status"] if done else res["processing_status"]
        r = res["result"] or {}
        return {"outcome": outcome, "target_id": r.get("target_id") if outcome == "resolved" else None,
                "candidate_ids": list((r.get("candidates") or {}).get("target_ids") or []),
                "reason_code": r.get("reason_code")}


def label_row(elig, entries, text, views, cmap, checker: Checker) -> dict:
    """One parent's label and checks. elig: A2.6b's eligibility row; views: view -> (scene, command) records of the
    derived subscene (None when not materialized)."""
    spec = checker.spec
    row = {"parent_command_id": elig["parent_command_id"], "scene": elig["scene"], "group": elig["group"],
           "partition": elig["partition"], "text": text, "eligible": elig["eligible"],
           "eligibility_reasons": list(elig["reasons"]), "target_object_id": elig["target_object_id"],
           "stratum": elig["stratum"], "parsing_label": elig["parsing_label"], "label": None, "rule_parse": None,
           "rule_disagreement": [], "views": {}, "parent_reasons": [], "retained": False, "direct_views": []}
    if not elig["eligible"]:
        row["parent_reasons"].append("not_eligible")
        return row
    if elig["parsing_label"] != "consistent":
        row["parent_reasons"].append("annotation_disagreement:parsing_label")
        return row
    try:
        lab = build(entries, spec)
    except Unsupported as e:
        row["parent_reasons"].append(str(e))
        return row
    row["label"] = lab["string"]
    interp = L.interpretation(lab["structure"])
    rp = checker.rule_parse(text, cmap)
    row["rule_parse"] = {"status": rp["parse_status"], "reason": rp["parse_reason"],
                         "string": L.render(L.from_interpretation(rp["interpretation"]))
                         if rp["parse_status"] == "parsed" else None}
    diff = compare(lab["structure"], rp["interpretation"]) if rp["parse_status"] == "parsed" else ["rule_parse_failed"]
    row["rule_disagreement"] = diff
    for v in VIEWS:
        vv = {"target_offered": elig["views"][v]["target_offered"], "objects": elig["views"][v]["objects"]}
        if views.get(v) is None:
            vv.update(outcome="not_materialized", supported=False)
        else:
            res = checker.resolve(interp, views[v][0], views[v][1], cmap)
            vv.update(res)
            vv["supported"] = bool(vv["target_offered"]) and res["outcome"] == "resolved" \
                and res["target_id"] == elig["target_object_id"]
            if not vv["target_offered"]:
                vv["reason"] = "target_absent"
            elif res["outcome"] == "invalid_label_query":
                vv["reason"] = "invalid_label_query"
            elif res["outcome"] == "resolved" and not vv["supported"]:
                vv["reason"] = "resolver:resolved_other_object"
            elif not vv["supported"]:
                vv["reason"] = f"resolver:{res['outcome']}"
        row["views"][v] = vv
    if any(row["views"][v].get("outcome") == "invalid_label_query" for v in VIEWS):
        row["parent_reasons"].append("invalid_label_query")
    if diff:
        row["parent_reasons"].append("rule_parser_disagreement:" + "+".join(diff))
    sk = row["views"]["source_known_nyu"]
    if not sk["supported"]:
        row["parent_reasons"].append("source_known:" + sk.get("reason", "unsupported"))
    row["retained"] = not row["parent_reasons"]
    row["direct_views"] = [v for v in VIEWS if row["retained"] and row["views"][v]["supported"]]
    return row
