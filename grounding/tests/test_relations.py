"""Tests of the relation library (A2.1b, D67).

    python grounding/tests/test_relations.py

Needs only Python 3.9+ and jsonschema. The reviewed cases in fixtures/relations/cases.json carry expected values
calculated by hand before the library existed; the library must reproduce them. Further checks: the configuration,
the comparison rules, hand-calculated geometry of a tilted box, invariance under moving and turning the scene about
+z, swapped between-anchors, the restricted profile, field-level provenance, the geometry cache, and the cost of
joint eligibility. The sampling and grid checks at the end are approximate diagnostics, not expected values. Prints
one line per check and exits 1 if any fails.
"""
from __future__ import annotations

import copy
import itertools
import json
import math
import random
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))  # the repository root only, so the file also runs directly
from grounding.contract import validate  # noqa: E402
from grounding.relations import geometry as g  # noqa: E402
from grounding.relations import predicates as P  # noqa: E402

FIXTURES = REPO / "grounding" / "tests" / "fixtures" / "relations"
MAP = REPO / "grounding" / "tests" / "fixtures" / "contract" / "valid" / "category-map.fx.json"
T, F, U = P.TRUE, P.FALSE, P.UNKNOWN
FAILED = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("PASS  " if ok else "FAIL  ") + name + (f": {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_case(rel, case):
    call = case["call"]
    if "anchor" in call:
        return rel.rank(call["relation"], call["anchor"], call["candidates"], call["possible"], call["k"])
    return getattr(rel, call["relation"])(*call["args"])


def compare(case, result, tol):
    """Differences between a result and a case's hand-calculated expectation; empty if they agree."""
    e, out = case["expect"], []
    if "status" in e:
        if (result.status, list(result.object_ids)) != (e["status"], e["object_ids"]):
            out.append(f"got {result.status} {list(result.object_ids)}")
        for k, v in e.get("distances_m", {}).items():
            if k not in result.distances_m or abs(result.distances_m[k] - v) > tol:
                out.append(f"distance {k} = {result.distances_m.get(k)}, expected {v}")
        if "possible_count" in e and (result.possible_count, result.combinations_checked) != (
                e["possible_count"], e["combinations_checked"]):
            out.append(f"counts {result.possible_count}, {result.combinations_checked}")
    else:
        if result.value.value != e["value"]:
            out.append(f"got {result.value.value}")
        for k, v in e.get("measures", {}).items():
            got = result.measures.get(k)
            if got is None or abs(got - v) > tol:
                out.append(f"{k} = {got}, expected {v}")
        if "basis" in e and result.basis != e["basis"]:
            out.append(f"basis {result.basis}")
    missing = [r for r in e.get("reasons_include", []) if r not in result.reasons]
    if missing:
        out.append(f"reasons {list(result.reasons)} lack {missing}")
    return out


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz]


def moved(record, theta, dx, dy):
    """The scene turned by theta about +z and shifted horizontally, under a new scene ID (a new cache key)."""
    rec = copy.deepcopy(record)
    rec["scene_id"] += ".moved"
    c, s = math.cos(theta), math.sin(theta)
    yaw = [0.0, 0.0, math.sin(theta / 2), math.cos(theta / 2)]
    for obj in rec["objects"]:
        geo = obj["geometry"]
        if geo["center_m"]["state"] == "known":
            x, y, z = geo["center_m"]["value"]
            geo["center_m"]["value"] = [c * x - s * y + dx, s * x + c * y + dy, z]
        if geo["rotation_xyzw"]["state"] == "known":
            geo["rotation_xyzw"]["value"] = qmul(yaw, geo["rotation_xyzw"]["value"])
            if geo["rotation_xyzw"]["support"] == "axis_aligned":
                geo["rotation_xyzw"]["support"] = "yaw_only"
        if obj["semantic_front"]["state"] == "known":
            x, y, z = obj["semantic_front"]["value"]
            obj["semantic_front"]["value"] = [c * x - s * y, s * x + c * y, z]
    return rec


def same_result(a, b, tol=1e-9):
    if isinstance(a, P.RankResult):
        return (a.status, a.object_ids) == (b.status, b.object_ids) and all(
            abs(a.distances_m[k] - b.distances_m[k]) <= tol for k in a.distances_m)
    if a.value is not b.value:
        return False
    for k, v in a.measures.items():
        w = b.measures.get(k)
        if (v is None) != (w is None) or (v is not None and abs(v - w) > tol):
            return False
    return True


def main() -> int:
    cfg = P.load_config()
    doc = load_json(FIXTURES / "cases.json")
    tol = doc["tolerance"]
    records = {k: load_json(FIXTURES / v) for k, v in doc["scenes"].items()}

    # 1. Configuration
    check("relations.v1.json loads, validates and is marked provisional",
          cfg.status == "provisional" and cfg.identity.startswith("relations.v1@"), cfg.identity)
    raw = load_json(P.CONFIG_FILE)
    bad = copy.deepcopy(raw)
    bad["thresholds"]["footprint_overlap_min"] = 1.5
    rejected = []
    with tempfile.TemporaryDirectory() as folder:
        tmp = Path(folder) / "config.json"
        for name, text in (("ratio above 1", json.dumps(bad)), ("duplicate key", '{"a": 1, "a": 2}'),
                           ("NaN", json.dumps(raw).replace("0.05", "NaN", 1))):
            tmp.write_text(text, encoding="utf-8")
            try:
                P.load_config(tmp)
                rejected.append(f"{name} accepted")
            except Exception:  # noqa: BLE001  (any rejection will do)
                pass
    check("a configuration with an out-of-range ratio, a repeated key or NaN is rejected", not rejected,
          "; ".join(rejected))

    # 2. Fixtures are valid contract records
    files = sorted(FIXTURES.glob("scene.*.json")) + [MAP]
    issues = validate.validate_paths(files)
    check("the relation fixture scenes pass the contract validator", not issues,
          "; ".join(i.line() for i in issues[:3]))

    # 3. Comparisons and three-valued logic, with exact binary inputs
    le_ok = [P.le(x, 0.5, 0.25) for x in (0.25, 0.5, 0.75, 0.875)] == [T, U, U, F]
    ge_ok = [P.ge(x, 0.5, 0.25) for x in (0.75, 0.5, 0.25, 0.125)] == [T, U, U, F]
    zero_ok = P.le(0.5, 0.5, 0.0) is T and P.ge(0.5, 0.5, 0.0) is T
    check("le and ge decide outside the band, give unknown inside it, and treat equality as true without a band",
          le_ok and ge_ok and zero_ok)
    logic_ok = (P.AND(T, T) is T and P.AND(T, U) is U and P.AND(F, U) is F and P.OR(F, F) is F
                and P.OR(F, U) is U and P.OR(T, U) is T)
    check("AND and OR follow three-valued logic: false wins an AND, true wins an OR", logic_ok)
    tie = P.Config(cfg.config_id, cfg.status, cfg.identity, cfg.thresholds, dict(cfg.bands, rank_tie_m=0.0625),
                   cfg.categories)
    exact = copy.deepcopy(records["rank"])
    exact["scene_id"] = "fixture.rank.exact"
    exact["objects"][2]["geometry"]["center_m"]["value"] = [1.0625, 0.0, 0.0]
    r = P.Relations(exact, tie).rank("closest", "obj_001", ["obj_002", "obj_003"], k=1)
    check("a ranking gap exactly equal to the tie tolerance links the two candidates", r.status == "ambiguous",
          r.status)

    # 4. Reviewed cases
    rels = {k: P.Relations(rec, cfg) for k, rec in records.items()}
    results = {}
    for case in doc["cases"]:
        result = run_case(rels[case["scene"]], case)
        results[case["id"]] = result
        differences = compare(case, result, tol)
        check(f"case {case['id']}: {case['call']['relation']} -> "
              f"{case['expect'].get('value', case['expect'].get('status'))}", not differences,
              "; ".join(differences))
    consistent = all((r.value is T) == (r.reasons == ()) and (r.basis is None or r.value is T)
                     for r in results.values() if isinstance(r, P.RelationResult))
    check("true results have no reasons, and only true results have a basis", consistent)

    # 5. Hand-calculated geometry
    for item in doc["geometry"]:
        geo = rels[item["scene"]].scene.geometry(item["object_id"]).box
        got = {"z_min": geo.z_min, "z_max": geo.z_max, "footprint_area": geo.footprint_area, "tilt_deg": geo.tilt_deg}
        wrong = {k: got[k] for k, v in item["expect"].items() if abs(got[k] - v) > tol}
        check(f"geometry of {item['scene']}/{item['object_id']} matches the hand calculation", not wrong, str(wrong))

    # 6. Invariance under turning the scene about +z and shifting it horizontally
    for key in ("rel", "rank", "tilt"):
        turned = moved(records[key], math.radians(37.0), 0.37, -1.21)
        found = [i for i in validate.validate([("moved scene", turned)]) if i.is_error]
        rel_moved = P.Relations(turned, cfg)
        differ = [c["id"] for c in doc["cases"] if c["scene"] == key
                  and not same_result(results[c["id"]], run_case(rel_moved, c))]
        check(f"{key}: turning the scene 37 degrees about +z and shifting it changes no result", not found and not differ,
              f"differ: {differ}" if differ else "; ".join(i.line() for i in found[:2]))

    # 7. Swapping between's anchors: same value, same lateral distance, along measured from the other end
    wrong = []
    for c in (c for c in doc["cases"] if c["call"]["relation"] == "between"):
        t, a, b = c["call"]["args"]
        r, s = results[c["id"]], rels[c["scene"]].between(t, b, a)
        if r.value is not s.value:
            wrong.append(f"{c['id']}: {r.value.value} vs {s.value.value}")
        elif "along_m" in r.measures and (abs(r.measures["lateral_m"] - s.measures["lateral_m"]) > 1e-9 or abs(
                r.measures["along_m"] - (r.measures["separation_m"] - s.measures["along_m"])) > 1e-9):
            wrong.append(f"{c['id']}: measures")
    check("swapping between's anchors leaves every value and the lateral distance unchanged", not wrong,
          "; ".join(wrong))

    # 8. Restricted profile: box relations unknown for every pair; between never true; ranking unaffected
    restricted = rels["a17r"]
    ids = [o["object_id"] for o in records["a17r"]["objects"]]
    not_unknown = [(name, t, a) for name in ("near", "above", "below", "on", "inside")
                   for t, a in itertools.permutations(ids, 2) if getattr(restricted, name)(t, a).value is not U]
    check("restricted a17: near, above, below, on and inside are unknown for every pair of objects", not not_unknown,
          str(not_unknown[:3]))
    true_between = [(t, a, b) for t, a, b in itertools.permutations(ids, 3) if restricted.between(t, a, b).value is T]
    check("restricted a17: between is never true", not true_between, str(true_between[:3]))

    # 9. Field-level provenance
    r = results["4r"]
    refs = {(x.object_id, x.field): x for x in r.inputs}
    rot = refs[("obj_002", "geometry.rotation_xyzw")]
    size = refs[("obj_002", "geometry.size_m")]
    centre = refs[("obj_002", "geometry.center_m")]
    ok = (rot.state == "unknown" and rot.reason == "withheld_by_profile" and rot.source == "profile.restricted"
          and size.state == "known" and size.kind == "assumed" and size.source == "prior.fx"
          and size.assumptions == ("class_size_prior",) and centre.kind == "annotated" and centre.source == "fx"
          and r.assumptions == frozenset({"class_size_prior"}) and r.evidence == frozenset({"annotated", "assumed"})
          and r.config == cfg.identity and len(r.inputs) == 6)
    check("a restricted near result lists every consulted field with its source, evidence and assumptions", ok,
          f"{len(r.inputs)} inputs, assumptions {set(r.assumptions)}, evidence {set(r.evidence)}")

    # 10. Geometry cache
    P.clear_geometry_cache()
    first = P.Relations(records["rel"], cfg).scene.geometry("obj_001")
    again = P.Relations(records["rel"], cfg).scene.geometry("obj_001")
    key_ok = P.Scene(records["rel"]).key("obj_001") == ("fixture.rel", 0, "annotated", "obj_001")
    altered = copy.deepcopy(records["rel"])
    altered["objects"][0]["geometry"]["center_m"]["value"] = [0.0, 0.0, 0.4]
    try:
        P.Relations(altered, cfg).scene.geometry("obj_001")
        raised = False
    except ValueError:
        raised = True
    check("geometry is cached by scene ID, revision, profile and object ID, and a reused key with other geometry "
          "raises", first is again and key_ok and raised)
    P.clear_geometry_cache()

    # 11. Joint eligibility is exhaustive: 2 ** m combinations for m possible candidates
    rank_rel = P.Relations(records["rank"], cfg)
    pool = ["obj_003", "obj_004", "obj_005", "obj_006"]
    counts = [rank_rel.rank("closest", "obj_001", ["obj_002"], pool[:m], 1).combinations_checked for m in range(5)]
    check("joint eligibility checks 2 ** m combinations for m possible candidates", counts == [1, 2, 4, 8, 16],
          str(counts))

    # 12. Approximate diagnostics (sampling and grid counting; not expected values)
    rng = random.Random(20261002)

    def random_box():
        axis = [rng.uniform(-1, 1) for _ in range(3)]
        n = math.sqrt(sum(a * a for a in axis)) or 1.0
        angle = rng.uniform(0, math.pi)
        q = [a / n * math.sin(angle / 2) for a in axis] + [math.cos(angle / 2)]
        return g.Box([rng.uniform(-1.0, 1.0) for _ in range(3)], [rng.uniform(0.2, 1.0) for _ in range(3)], q)

    bad, touching = [], 0
    for _ in range(25):
        a, b = random_box(), random_box()
        exact = g.box_distance(a, b)
        touching += exact == 0.0
        n = 11
        sampled, delta = math.inf, 0.0
        for axis in range(3):
            for side in (-1.0, 1.0):
                u, v = [i for i in range(3) if i != axis]
                du, dv = 2 * a.half[u] / (n - 1), 2 * a.half[v] / (n - 1)
                delta = max(delta, math.hypot(du, dv) / 2)
                for i in range(n):
                    for j in range(n):
                        loc = [0.0, 0.0, 0.0]
                        loc[axis] = side * a.half[axis]
                        loc[u] = -a.half[u] + i * du
                        loc[v] = -a.half[v] + j * dv
                        p = g.add(a.centre, g.add(g.add(g.scale(a.axes[0], loc[0]), g.scale(a.axes[1], loc[1])),
                                                  g.scale(a.axes[2], loc[2])))
                        sampled = min(sampled, g.point_box_distance(p, b))
        if not exact - 1e-9 <= sampled <= exact + delta + 1e-9:
            bad.append(f"exact {exact:.4f}, sampled {sampled:.4f}, delta {delta:.4f}")
    check("diagnostic: box distance agrees with surface sampling within the sampling resolution (25 random pairs)",
          not bad, "; ".join(bad[:3]) or f"{touching} intersecting, {25 - touching} apart")

    def inside(poly, x, y):
        return all((poly[(i + 1) % len(poly)][0] - poly[i][0]) * (y - poly[i][1])
                   - (poly[(i + 1) % len(poly)][1] - poly[i][1]) * (x - poly[i][0]) >= 0 for i in range(len(poly)))

    def perimeter(poly):
        return sum(math.dist(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly)))

    bad = []
    for _ in range(8):
        a, b = random_box(), random_box()
        b = g.Box([a.centre[0] + rng.uniform(-0.3, 0.3), a.centre[1] + rng.uniform(-0.3, 0.3), b.centre[2]],
                  [2 * h for h in b.half], [0.0, 0.0, 0.0, 1.0])
        exact = g.intersection_area(a.footprint, b.footprint)
        h = 0.01
        x0 = max(min(p[0] for p in a.footprint), min(p[0] for p in b.footprint))
        x1 = min(max(p[0] for p in a.footprint), max(p[0] for p in b.footprint))
        y0 = max(min(p[1] for p in a.footprint), min(p[1] for p in b.footprint))
        y1 = min(max(p[1] for p in a.footprint), max(p[1] for p in b.footprint))
        count = sum(1 for i in range(max(0, int((x1 - x0) / h) + 1)) for j in range(max(0, int((y1 - y0) / h) + 1))
                    if inside(a.footprint, x0 + (i + 0.5) * h, y0 + (j + 0.5) * h)
                    and inside(b.footprint, x0 + (i + 0.5) * h, y0 + (j + 0.5) * h))
        bound = 2 * (perimeter(a.footprint) + perimeter(b.footprint)) * h
        if abs(count * h * h - exact) > bound:
            bad.append(f"exact {exact:.4f}, grid {count * h * h:.4f}, bound {bound:.4f}")
    check("diagnostic: footprint intersection area agrees with grid counting within the grid's resolution "
          "(8 random pairs)", not bad, "; ".join(bad[:3]))

    print(f"{'FAILED: ' + ', '.join(FAILED) if FAILED else 'all checks passed'}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
