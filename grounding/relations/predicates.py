"""The relation library, v1 (A2.1b, D67): closest and farthest (ranks 1-3), near, above, below, on, inside and
between, over a scene record that has passed the contract validator (grounding/contract/validate.py).

    from grounding.relations.predicates import Relations   # through the package, from the repository root
    rel = Relations(scene_record)                 # loads relations.v1.json unless a Config is given
    rel.near("obj_005", "obj_001").value          # Truth.TRUE, Truth.FALSE or Truth.UNKNOWN
    rel.rank("closest", "obj_001", ["obj_004", "obj_005"], possible=["obj_006"], k=1).status

Results are three-valued. Each condition of a relation is TRUE, FALSE or UNKNOWN; conditions combine by AND and OR
(FALSE wins an AND, TRUE wins an OR, otherwise UNKNOWN), so a result is FALSE only when a condition that could be
evaluated is FALSE, and missing geometry never becomes FALSE. Gates (a support or container category, a support's
tilt, between's anchor separation) are checked first and give UNKNOWN when they fail. Every result lists the source
fields it consulted, with their evidence and assumptions, and the identity of the configuration used. Directions,
the resolver and serializers are not part of this module. Definitions: docs/relations.md.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import jsonschema

from . import geometry as g

REPO = Path(__file__).resolve().parents[2]
CONFIG_FILE = Path(__file__).resolve().parent / "relations.v1.json"
CONFIG_SCHEMA = REPO / "schemas" / "relation-config.v1.json"


class Truth(Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


TRUE, FALSE, UNKNOWN = Truth.TRUE, Truth.FALSE, Truth.UNKNOWN


def AND(*values):
    if FALSE in values:
        return FALSE
    return TRUE if all(v is TRUE for v in values) else UNKNOWN


def OR(*values):
    if TRUE in values:
        return TRUE
    return FALSE if all(v is FALSE for v in values) else UNKNOWN


def le(x, t, b):
    """x <= t with band b: TRUE if x <= t - b, FALSE if x > t + b, otherwise UNKNOWN."""
    if x <= t - b:
        return TRUE
    if x > t + b:
        return FALSE
    return UNKNOWN


def ge(x, t, b):
    """x >= t with band b: TRUE if x >= t + b, FALSE if x < t - b, otherwise UNKNOWN."""
    if x >= t + b:
        return TRUE
    if x < t - b:
        return FALSE
    return UNKNOWN


# ---------------------------------------------------------------------------------------------- configuration

@dataclass(frozen=True)
class Config:
    config_id: str
    status: str
    identity: str        # config_id plus a hash of the parsed content, the same on every platform
    thresholds: dict
    bands: dict
    categories: dict


def _strict_json(data: bytes):
    if data.startswith(b"\xef\xbb\xbf"):
        raise ValueError("the configuration starts with a byte-order mark")

    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError(f"key {k!r} appears twice")
            out[k] = v
        return out

    def no_constant(name):
        raise ValueError(f"{name} is not a JSON number")

    def finite(text):
        v = float(text)
        if not math.isfinite(v):
            raise ValueError(f"{text} is not finite")
        return v

    return json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=no_constant, parse_float=finite)


def load_config(path=CONFIG_FILE) -> Config:
    """Load and validate a relation configuration against schemas/relation-config.v1.json."""
    raw = _strict_json(Path(path).read_bytes())
    schema = json.loads(CONFIG_SCHEMA.read_text(encoding="utf-8"))
    jsonschema.validators.validator_for(schema)(schema).validate(raw)
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
    identity = f"{raw['config_id']}@{hashlib.sha256(canonical).hexdigest()[:12]}"
    return Config(raw["config_id"], raw["status"], identity, dict(raw["thresholds"]), dict(raw["bands"]),
                  {k: frozenset(v) for k, v in raw["categories"].items()})


# ------------------------------------------------------------------------------------- geometry and provenance

@dataclass(frozen=True)
class FieldRef:
    """One source field a result consulted."""
    object_id: str
    field: str            # e.g. "geometry.center_m"
    state: str            # "known" or "unknown"
    kind: object          # evidence kind of a known value, else None
    source: str
    assumptions: tuple    # assumption IDs of a known value
    reason: object        # reason of an unknown value, else None


@dataclass(frozen=True)
class ObjectGeometry:
    object_id: str
    category: object      # model-visible label, or None if unknown
    centre: object        # (x, y, z), or None
    box: object           # geometry.Box, or None unless centre, size and rotation are all known
    missing: tuple        # reasons the box is incomplete, e.g. "missing_rotation:obj_003"
    refs: dict            # field name -> FieldRef


_FIELDS = {"category": ("category",), "center_m": ("geometry", "center_m"), "size_m": ("geometry", "size_m"),
           "rotation_xyzw": ("geometry", "rotation_xyzw")}
_MISSING = {"center_m": "missing_centre", "size_m": "missing_size", "rotation_xyzw": "missing_rotation"}

# Per-object geometry, keyed by (scene_id, scene_revision, evidence_profile, object_id). The configuration is not
# part of the key because geometry doesn't depend on it; a cache of relation results would have to add
# Config.identity. A key reused for different geometry raises instead of returning stale geometry.
_GEOMETRY_CACHE = {}


def clear_geometry_cache() -> None:
    _GEOMETRY_CACHE.clear()


def _value(obj, path):
    w = obj
    for k in path:
        w = w[k]
    return w


def _ref(object_id, name, w) -> FieldRef:
    field = ".".join(_FIELDS[name])
    if w["state"] == "known":
        ev = w["evidence"]
        return FieldRef(object_id, field, "known", ev["kind"], ev["source"], tuple(ev["assumptions"]), None)
    return FieldRef(object_id, field, "unknown", None, w["source"], (), w["reason"])


def _build(obj) -> ObjectGeometry:
    oid = obj["object_id"]
    values = {name: _value(obj, path) for name, path in _FIELDS.items()}
    refs = {name: _ref(oid, name, w) for name, w in values.items()}
    known = {name: w["value"] for name, w in values.items() if w["state"] == "known"}
    missing = tuple(f"{_MISSING[n]}:{oid}" for n in ("center_m", "size_m", "rotation_xyzw") if n not in known)
    centre = tuple(known["center_m"]) if "center_m" in known else None
    box = None if missing else g.Box(known["center_m"], known["size_m"], known["rotation_xyzw"])
    category = known["category"]["model"] if "category" in known else None
    return ObjectGeometry(oid, category, centre, box, missing, refs)


class Scene:
    def __init__(self, record: dict):
        self.scene_id, self.revision = record["scene_id"], record["scene_revision"]
        self.profile = record["evidence_profile"]
        self._objects = {o["object_id"]: o for o in record["objects"]}

    def key(self, object_id):
        return (self.scene_id, self.revision, self.profile, object_id)

    def geometry(self, object_id) -> ObjectGeometry:
        if object_id not in self._objects:
            raise ValueError(f"{object_id} is not an object of scene {self.scene_id}")
        obj = self._objects[object_id]
        fingerprint = json.dumps({k: _value(obj, p) for k, p in _FIELDS.items()}, sort_keys=True)
        key = self.key(object_id)
        hit = _GEOMETRY_CACHE.get(key)
        if hit is not None:
            if hit[0] != fingerprint:
                raise ValueError(f"geometry cache key {key} was reused for different geometry")
            return hit[1]
        built = _build(obj)
        _GEOMETRY_CACHE[key] = (fingerprint, built)
        return built


# ---------------------------------------------------------------------------------------------------- results

@dataclass(frozen=True)
class RelationResult:
    relation: str
    object_ids: tuple
    value: Truth
    reasons: tuple       # why FALSE or UNKNOWN; empty when TRUE
    basis: object        # on: top_face_support; inside: outer_box_containment; below: its branch; else None
    measures: dict       # computed quantities, e.g. {"distance_m": 0.3}
    inputs: tuple        # FieldRef of every source field consulted
    assumptions: frozenset
    evidence: frozenset
    config: str


@dataclass(frozen=True)
class RankResult:
    relation: str
    anchor: str
    rank: int
    status: str          # resolved, ambiguous, no_match or insufficient
    object_ids: tuple    # one if resolved; the tie group if ambiguous; empty otherwise
    reasons: tuple
    distances_m: dict    # centre distance to the anchor, for every candidate whose centre is known
    possible_count: int  # number of candidates of unknown eligibility
    combinations_checked: int   # 2 ** possible_count when evaluated; exhaustive, an offline reference
    inputs: tuple
    assumptions: frozenset
    evidence: frozenset
    config: str


def _why(conditions):
    out = []
    for name, value in conditions:
        if value is FALSE:
            out.append(f"failed:{name}")
        elif value is UNKNOWN:
            out.append(f"boundary:{name}")
    return out


def _provenance(refs):
    known = [r for r in refs if r.state == "known"]
    return frozenset(a for r in known for a in r.assumptions), frozenset(r.kind for r in known)


class Relations:
    """The relations of one validated scene record, under one configuration."""

    def __init__(self, record: dict, config: Config = None):
        self.scene = Scene(record)
        self.cfg = config or load_config()
        self.th, self.bd = self.cfg.thresholds, self.cfg.bands

    # -- helpers
    def _geo(self, oid):
        return self.scene.geometry(oid)

    def _box_refs(self, *oids):
        return [self._geo(o).refs[n] for o in oids for n in ("center_m", "size_m", "rotation_xyzw")]

    def _result(self, relation, ids, value, reasons, basis, measures, refs):
        assumptions, evidence = _provenance(refs)
        reasons = () if value is TRUE else tuple(dict.fromkeys(reasons))
        return RelationResult(relation, tuple(ids), value, reasons, basis if value is TRUE else None,
                              dict(measures), tuple(refs), assumptions, evidence, self.cfg.identity)

    def _overlap(self, t, a):
        ratio = g.footprint_overlap(self._geo(t).box, self._geo(a).box)
        if ratio is None:
            return None, UNKNOWN, ["degenerate_footprint"]
        c = ge(ratio, self.th["footprint_overlap_min"], self.bd["overlap_band"])
        return ratio, c, _why([("overlap", c)])

    def _not_distinct(self, relation, ids):
        return self._result(relation, ids, FALSE, ["not_distinct"], None, {}, [])

    # -- near
    def near(self, t, a):
        ids = (t, a)
        if t == a:
            return self._not_distinct("near", ids)
        gt, ga = self._geo(t), self._geo(a)
        refs = self._box_refs(t, a)
        if gt.box is None or ga.box is None:
            return self._result("near", ids, UNKNOWN, gt.missing + ga.missing, None, {}, refs)
        d = g.box_distance(gt.box, ga.box)
        c = le(d, self.th["near_max_m"], self.bd["distance_band_m"])
        return self._result("near", ids, c, _why([("distance", c)]), None, {"distance_m": d}, refs)

    # -- above
    def above(self, t, a):
        ids = (t, a)
        if t == a:
            return self._not_distinct("above", ids)
        gt, ga = self._geo(t), self._geo(a)
        refs = self._box_refs(t, a)
        if gt.box is None or ga.box is None:
            return self._result("above", ids, UNKNOWN, gt.missing + ga.missing, None, {}, refs)
        gap = gt.box.z_min - ga.box.z_max
        c_gap = ge(gap, self.th["above_gap_min_m"], self.bd["vertical_band_m"])
        ratio, c_ov, ov_why = self._overlap(t, a)
        return self._result("above", ids, AND(c_gap, c_ov), _why([("vertical_gap", c_gap)]) + ov_why, None,
                            {"gap_m": gap, "overlap": ratio}, refs)

    # -- on
    def on(self, t, a):
        ids = (t, a)
        if t == a:
            return self._not_distinct("on", ids)
        gt, ga = self._geo(t), self._geo(a)
        cat_refs = [ga.refs["category"]]
        if ga.category is None:
            return self._result("on", ids, UNKNOWN, [f"category_unknown:{a}"], None, {}, cat_refs)
        if ga.category not in self.cfg.categories["support"]:
            return self._result("on", ids, UNKNOWN, [f"support_not_represented:{a}"], None, {}, cat_refs)
        refs = cat_refs + self._box_refs(t, a)
        if gt.box is None or ga.box is None:
            return self._result("on", ids, UNKNOWN, gt.missing + ga.missing, None, {}, refs)
        tilt = ga.box.tilt_deg
        if tilt > self.th["on_support_tilt_max_deg"]:
            return self._result("on", ids, UNKNOWN, [f"support_tilted:{a}"], None, {"tilt_deg": tilt}, refs)
        gap = gt.box.z_min - ga.box.z_max
        vb = self.bd["vertical_band_m"]
        c_gap = le(gap, self.th["on_gap_max_m"], vb)
        c_pen = ge(gap, -self.th["on_penetration_max_m"], vb)
        ratio, c_ov, ov_why = self._overlap(t, a)
        return self._result("on", ids, AND(c_gap, c_pen, c_ov), _why([("gap", c_gap), ("penetration", c_pen)]) + ov_why,
                            "top_face_support", {"gap_m": gap, "overlap": ratio, "tilt_deg": tilt}, refs)

    # -- below
    def below(self, t, a):
        ids = (t, a)
        if t == a:
            return self._not_distinct("below", ids)
        gt, ga = self._geo(t), self._geo(a)
        refs = self._box_refs(t, a) + [ga.refs["category"]]
        if gt.box is None or ga.box is None:
            extra = [] if ga.category is not None else [f"category_unknown:{a}"]
            return self._result("below", ids, UNKNOWN, list(gt.missing + ga.missing) + extra, None, {}, refs)
        vb = self.bd["vertical_band_m"]
        ratio, c_ov, ov_why = self._overlap(t, a)
        separated_gap = gt.box.z_max - ga.box.z_min
        c_sep = le(separated_gap, -self.th["below_separated_gap_min_m"], vb)
        separated = AND(c_sep, c_ov)
        offset = abs(gt.box.z_min - ga.box.z_min)
        margin = gt.box.z_max - ga.box.z_max
        if ga.category is None:
            furniture, f_why = UNKNOWN, [f"category_unknown:{a}"]
        elif ga.category not in self.cfg.categories["furniture"]:
            furniture, f_why = FALSE, ["failed:furniture_category"]
        else:
            c_off = le(offset, self.th["below_furniture_bottom_max_m"], self.bd["distance_band_m"])
            c_mar = le(margin, -self.th["below_furniture_top_margin_m"], vb)
            furniture = AND(c_off, c_mar, c_ov)
            f_why = _why([("bottom_offset", c_off), ("top_margin", c_mar)]) + ov_why
        value = OR(separated, furniture)
        basis = "+".join(n for n, v in (("separated", separated), ("beneath_furniture", furniture)) if v is TRUE)
        reasons = _why([("separated_gap", c_sep)]) + ov_why + f_why
        return self._result("below", ids, value, reasons, basis or None,
                            {"separated_gap_m": separated_gap, "bottom_offset_m": offset, "top_margin_m": margin,
                             "overlap": ratio}, refs)

    # -- inside
    def inside(self, t, a):
        ids = (t, a)
        if t == a:
            return self._not_distinct("inside", ids)
        gt, ga = self._geo(t), self._geo(a)
        cat_refs = [ga.refs["category"]]
        if ga.category is None:
            return self._result("inside", ids, UNKNOWN, [f"category_unknown:{a}"], None, {}, cat_refs)
        if ga.category not in self.cfg.categories["container"]:
            return self._result("inside", ids, UNKNOWN, [f"container_not_represented:{a}"], None, {}, cat_refs)
        refs = cat_refs + self._box_refs(t, a)
        if gt.box is None or ga.box is None:
            return self._result("inside", ids, UNKNOWN, gt.missing + ga.missing, None, {}, refs)
        excess = g.containment_excess(gt.box, ga.box)
        c = le(excess, self.th["inside_tolerance_m"], self.bd["vertical_band_m"])
        return self._result("inside", ids, c, _why([("containment", c)]), "outer_box_containment",
                            {"excess_m": excess}, refs)

    # -- between
    def between(self, t, a, b):
        ids = (t, a, b)
        if len(set(ids)) < 3:
            return self._not_distinct("between", ids)
        gt, ga, gb = self._geo(t), self._geo(a), self._geo(b)
        centre_refs = [x.refs["center_m"] for x in (gt, ga, gb)]
        if gt.centre is None or ga.centre is None or gb.centre is None:
            missing = [f"missing_centre:{x.object_id}" for x in (gt, ga, gb) if x.centre is None]
            return self._result("between", ids, UNKNOWN, missing, None, {}, centre_refs)
        db = self.bd["distance_band_m"]
        ab = g.sub(gb.centre, ga.centre)
        separation = g.norm(ab)
        if ge(separation, self.th["between_min_separation_m"], db) is not TRUE:
            return self._result("between", ids, UNKNOWN, ["degenerate_anchors"], None,
                                {"separation_m": separation}, centre_refs)
        along = g.dot(g.sub(gt.centre, ga.centre), ab) / separation
        foot = g.add(ga.centre, g.scale(ab, along / separation))
        lateral = g.norm(g.sub(gt.centre, foot))
        limit = max(self.th["between_lateral_min_m"], self.th["between_lateral_ratio"] * separation)
        c_a = ge(along, 0.0, db)
        c_b = le(along, separation, db)
        c_lat = le(lateral, limit, db)
        refs = centre_refs + [x.refs[n] for x in (gt, ga, gb) for n in ("size_m", "rotation_xyzw")]
        measures = {"separation_m": separation, "along_m": along, "lateral_m": lateral, "lateral_limit_m": limit}
        conditions, reasons = [c_a, c_b, c_lat], _why([("endpoint_a", c_a), ("endpoint_b", c_b), ("lateral", c_lat)])
        for name, gx in (("overlap_a", ga), ("overlap_b", gb)):
            if gt.box is None or gx.box is None:
                conditions.append(UNKNOWN)
                reasons += list(gt.missing + gx.missing)
                measures[name] = None
                continue
            if g.vertical_overlap(gt.box, gx.box) > 0.0:
                ratio = g.footprint_overlap(gt.box, gx.box)
            else:
                ratio = 0.0
            if ratio is None:
                conditions.append(UNKNOWN)
                reasons.append("degenerate_footprint")
            else:
                c = le(ratio, self.th["between_overlap_max"], self.bd["overlap_band"])
                conditions.append(c)
                reasons += _why([(name, c)])
            measures[name] = ratio
        return self._result("between", ids, AND(*conditions), reasons, None, measures, refs)

    # -- closest and farthest
    def rank(self, relation, anchor, candidates, possible=(), k=1):
        """Rank k (1-3) of the closest or farthest candidates to an anchor, by centre distance (S17).

        candidates are eligible for certain; possible are candidates whose eligibility is unknown. The result is
        the outcome shared by every combination of the possible candidates being eligible or not, and
        insufficient if any two combinations differ. All 2 ** len(possible) combinations are evaluated: an exact
        offline reference for small fixtures, exponential in the number of possible candidates, with no cap. A
        bounded or faster method would need its own verification before any use on the headset.
        """
        if relation not in ("closest", "farthest"):
            raise ValueError(f"unknown ranking relation {relation!r}")
        if k not in (1, 2, 3):
            raise ValueError("ranks 1, 2 and 3 are supported")
        definite = [c for c in dict.fromkeys(candidates) if c != anchor]
        maybe = [p for p in dict.fromkeys(possible) if p != anchor]
        if set(definite) & set(maybe):
            raise ValueError("an object can't be both a candidate and a possible candidate")
        everyone = [anchor] + definite + maybe
        geos = {x: self._geo(x) for x in everyone}
        refs = [geos[x].refs["center_m"] for x in everyone]
        assumptions, evidence = _provenance(refs)
        anchor_centre = geos[anchor].centre
        distances = {} if anchor_centre is None else {
            x: g.norm(g.sub(geos[x].centre, anchor_centre)) for x in definite + maybe if geos[x].centre is not None}

        def result(status, object_ids, reasons, combinations):
            return RankResult(relation, anchor, k, status, tuple(object_ids), tuple(reasons), dict(distances),
                              len(maybe), combinations, tuple(refs), assumptions, evidence, self.cfg.identity)

        missing = [f"missing_centre:{x}" for x in everyone if geos[x].centre is None]
        if missing:
            return result("insufficient", (), missing, 0)
        outcomes = set()
        combinations = 0
        for r in range(len(maybe) + 1):
            for subset in itertools.combinations(maybe, r):
                combinations += 1
                outcomes.add(self._rank_outcome(definite + list(subset), distances, relation, k))
        if len(outcomes) > 1:
            return result("insufficient", (), ["possible_competitors:" + ",".join(maybe)], combinations)
        status, object_ids = outcomes.pop()
        reasons = {"resolved": (), "ambiguous": ("tie",), "no_match": ("too_few_candidates",)}[status]
        return result(status, object_ids, reasons, combinations)

    def _rank_outcome(self, members, distances, relation, k):
        sign = 1.0 if relation == "closest" else -1.0
        order = sorted(members, key=lambda x: (sign * distances[x], x))
        if len(order) < k:
            return ("no_match", ())
        tie = self.bd["rank_tie_m"]
        linked = [abs(distances[order[i]] - distances[order[i + 1]]) <= tie for i in range(len(order) - 1)]
        lo = hi = k - 1
        while lo > 0 and linked[lo - 1]:
            lo -= 1
        while hi < len(order) - 1 and linked[hi]:
            hi += 1
        if lo == hi:
            return ("resolved", (order[k - 1],))
        return ("ambiguous", tuple(sorted(order[lo:hi + 1])))
