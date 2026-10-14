"""A2.7 delivery 2 (D115): golden cases for the C# port of the resolution path, from the Python reference.

    python -m grounding.a27 goldens --scenes WORK_OR_FOLDER --out FILE.json [--seed N] [--pairs N]

Stage 1, the relation library: every scene record found under the folder (A2.6 work bundles hold them) gives cases on
its objects' boxes, pairs of objects under each binary relation, triples under between, and closest and farthest
ranks with possible candidates; labelled synthetic boxes add full rotations. Every float is written with repr, so the
C# compares bit for bit. The C# checks: tools/csharp-tests/Resolution.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

from ..relations import geometry as G
from ..relations import predicates as P

BINARY = ("near", "above", "on", "below", "inside")


def _f(x):
    return None if x is None else repr(float(x))


def _box(b: G.Box) -> dict:
    return {"corners": [[_f(v) for v in c] for c in b.corners], "z_min": _f(b.z_min), "z_max": _f(b.z_max),
            "footprint": [[_f(v) for v in p] for p in b.footprint], "area": _f(b.footprint_area),
            "tilt_deg": _f(b.tilt_deg)}


def _conditional(refs) -> bool:
    """The resolver's flag: a consulted known field was assumed or carried assumptions."""
    return any(r.state == "known" and (r.kind == "assumed" or bool(r.assumptions)) for r in refs)


def _rel(r) -> dict:
    return {"value": r.value.name, "reasons": list(r.reasons), "basis": r.basis, "conditional": _conditional(r.inputs),
            "measures": {k: _f(v) for k, v in sorted(r.measures.items())}}


def scene_records(folder: Path):
    seen, out = set(), []
    for p in sorted(Path(folder).rglob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:   # noqa: BLE001
            continue
        if isinstance(d, dict) and d.get("record_type") == "scene" and d.get("objects"):
            key = (d["scene_id"], str(d["scene_revision"]), d["evidence_profile"])
            if key not in seen:
                seen.add(key)
                out.append(d)
    return out


def relation_cases(scenes, seed=7, pairs=60, triples=40, ranks=40) -> dict:
    rng = random.Random(seed)
    cfg = P.load_config()
    out = {"format_version": 1, "record_type": "a27_relation_goldens", "seed": seed, "scenes": [], "boxes": [],
           "pairs": [], "triples": [], "ranks": [], "synthetic": []}
    for si, scene in enumerate(scenes):
        P.clear_geometry_cache()
        rel = P.Relations(scene, cfg)
        out["scenes"].append(scene)
        ids = sorted(o["object_id"] for o in scene["objects"])
        for oid in ids:
            geo = rel.scene.geometry(oid)
            if geo.box is not None:
                out["boxes"].append({"scene": si, "object_id": oid, **_box(geo.box)})
        all_pairs = [(t, a) for t in ids for a in ids]
        for t, a in rng.sample(all_pairs, min(pairs, len(all_pairs))):
            case = {"scene": si, "t": t, "a": a, "relations": {r: _rel(getattr(rel, r)(t, a)) for r in BINARY}}
            bt, ba = rel.scene.geometry(t).box, rel.scene.geometry(a).box
            if bt is not None and ba is not None:
                case["geometry"] = {"distance": _f(G.box_distance(bt, ba)), "intersect": G.boxes_intersect(bt, ba),
                                    "footprint_overlap": _f(G.footprint_overlap(bt, ba)),
                                    "vertical_overlap": _f(G.vertical_overlap(bt, ba)),
                                    "containment_excess": _f(G.containment_excess(bt, ba))}
            out["pairs"].append(case)
        if len(ids) >= 3:
            for _ in range(triples):
                t, a, b = rng.sample(ids, 3) if rng.random() > 0.1 else (ids[0], ids[0], ids[1])
                out["triples"].append({"scene": si, "t": t, "a": a, "b": b, "result": _rel(rel.between(t, a, b))})
        for _ in range(ranks):
            anchor = rng.choice(ids)
            others = [x for x in ids if x != anchor]
            rng.shuffle(others)
            n_def = rng.randint(0, min(6, len(others)))
            n_pos = rng.randint(0, min(4, len(others) - n_def))
            definite, possible = others[:n_def], others[n_def:n_def + n_pos]
            relation, k = rng.choice(("closest", "farthest")), rng.randint(1, 3)
            r = rel.rank(relation, anchor, definite, possible=possible, k=k)
            out["ranks"].append({"scene": si, "relation": relation, "anchor": anchor, "candidates": definite,
                                 "possible": possible, "k": k, "status": r.status, "object_ids": list(r.object_ids),
                                 "reasons": list(r.reasons), "combinations": r.combinations_checked,
                                 "conditional": _conditional(r.inputs),
                                 "distances": {x: _f(d) for x, d in sorted(r.distances_m.items())}})
    for _ in range(200):   # labelled synthetic boxes: full rotations, near and overlapping placements
        def rnd_box():
            q = [rng.gauss(0, 1) for _ in range(4)]
            if rng.random() < 0.3:
                q = [0.0, 0.0, math.sin(rng.uniform(0, math.pi)), math.cos(rng.uniform(0, math.pi))]
            c = [rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0, 2)]
            s = [rng.uniform(0.05, 1.2) for _ in range(3)]
            return {"centre": [_f(v) for v in c], "size": [_f(v) for v in s], "q": [_f(v) for v in q]}, G.Box(c, s, q)
        ja, a = rnd_box()
        jb, b = rnd_box()
        out["synthetic"].append({"a": ja, "b": jb, "box_a": _box(a), "box_b": _box(b),
                                 "distance": _f(G.box_distance(a, b)), "intersect": G.boxes_intersect(a, b),
                                 "footprint_overlap": _f(G.footprint_overlap(a, b)),
                                 "vertical_overlap": _f(G.vertical_overlap(a, b)),
                                 "containment_excess": _f(G.containment_excess(a, b))})
    return out


def _known(value, kind="annotated", assumptions=()):
    return {"state": "known", "value": value,
            "evidence": {"kind": kind, "source": "a27_golden", "assumptions": list(assumptions)}}


def _unknown():
    return {"state": "unknown", "source": "a27_golden", "reason": "not_observed"}


def variants(scenes, seed=11):
    """Labelled variants of real scenes: about a fifth of the fields unknown, about a fifth of the known ones assumed or
    carrying an assumption, so missing information, UNKNOWN results and the conditional flag are all exercised."""
    import copy
    rng = random.Random(seed)
    out = []
    for s in scenes:
        v = copy.deepcopy(s)
        v["scene_id"] = s["scene_id"] + "__variant"
        for o in v["objects"]:
            slots = [(o, "category"), (o["geometry"], "center_m"), (o["geometry"], "size_m"),
                     (o["geometry"], "rotation_xyzw"), (o["attributes"], "colours")]
            for holder, key in slots:
                u = rng.random()
                if u < 0.2:
                    holder[key] = _unknown()
                elif u < 0.3 and holder[key]["state"] == "known":
                    holder[key]["evidence"]["kind"] = "assumed"
                elif u < 0.4 and holder[key]["state"] == "known":
                    holder[key]["evidence"]["assumptions"] = ["asm_a27_golden"]
        out.append(v)
    return out


def _obj(oid, cat, centre, size, quat=(0.0, 0.0, 0.0, 1.0)):
    return {"object_id": oid, "category": _known({"model": cat, "standard": cat}),
            "geometry": {"center_m": _known([float(c) for c in centre]), "size_m": _known([float(x) for x in size]),
                         "rotation_xyzw": _known([float(q) for q in quat])},
            "attributes": {"colours": _known([])}}


def boundary_scenes():
    """Synthetic scenes whose measures sit at both band edges of every threshold, nudged by 0, +-1e-12, +-1e-9 and
    +-1e-6: decisions on both sides of each boundary, and the rank tie."""
    cfg = P.load_config()
    th, bd, cats = cfg.thresholds, cfg.bands, cfg.categories
    support, furniture, container = sorted(cats["support"])[0], sorted(cats["furniture"])[0], sorted(cats["container"])[0]
    deltas = (0.0, 1e-12, -1e-12, 1e-9, -1e-9, 1e-6, -1e-6)
    scenes, cases = [], []

    def edges(t, b):
        return [t + sign * b + d for sign in (-1.0, 1.0) for d in deltas]

    def add(objs, checks):
        sid = f"a27_boundary_{len(scenes):04d}"
        scenes.append({"scene_id": sid, "scene_revision": 1, "evidence_profile": "annotated", "objects": objs})
        for c in checks:
            cases.append(dict(c, scene=len(scenes) - 1))

    unit = (1.0, 1.0, 1.0)
    for g in edges(th["near_max_m"], bd["distance_band_m"]):
        add([_obj("a", "box", (0, 0, 0.5), unit), _obj("t", "box", (1.0 + g, 0, 0.5), unit)],
            [{"kind": "binary", "relation": "near", "t": "t", "a": "a"}])
    for g in edges(th["above_gap_min_m"], bd["vertical_band_m"]):
        add([_obj("a", "box", (0, 0, 0.5), unit), _obj("t", "box", (0, 0, 1.5 + g), unit)],
            [{"kind": "binary", "relation": "above", "t": "t", "a": "a"}])
    for r in edges(th["footprint_overlap_min"], bd["overlap_band"]):
        add([_obj("a", "box", (0, 0, 0.5), unit), _obj("t", "box", (1.0 - r, 0, 2.0), unit)],
            [{"kind": "binary", "relation": "above", "t": "t", "a": "a"}])
    for g in edges(th["on_gap_max_m"], bd["vertical_band_m"]) + edges(-th["on_penetration_max_m"], bd["vertical_band_m"]):
        add([_obj("a", support, (0, 0, 0.5), unit), _obj("t", "box", (0, 0, 1.25 + g), (0.5, 0.5, 0.5))],
            [{"kind": "binary", "relation": "on", "t": "t", "a": "a"}])
    for dd in (0.0, 1e-9, -1e-9, 1e-6, -1e-6, 1e-3, -1e-3):
        ang = math.radians(th["on_support_tilt_max_deg"] + dd)
        q = (math.sin(ang / 2), 0.0, 0.0, math.cos(ang / 2))
        add([_obj("a", support, (0, 0, 0.5), unit, q), _obj("t", "box", (0, 0, 1.3), (0.5, 0.5, 0.5))],
            [{"kind": "binary", "relation": "on", "t": "t", "a": "a"}])
    for s_ in edges(-th["below_separated_gap_min_m"], bd["vertical_band_m"]):
        add([_obj("a", "box", (0, 0, 2.5), unit), _obj("t", "box", (0, 0, 1.5 + s_), unit)],
            [{"kind": "binary", "relation": "below", "t": "t", "a": "a"}])
    for off in edges(th["below_furniture_bottom_max_m"], bd["distance_band_m"]):
        add([_obj("a", furniture, (0, 0, 0.5), unit), _obj("t", "box", (0, 0, off + 0.2), (0.4, 0.4, 0.4))],
            [{"kind": "binary", "relation": "below", "t": "t", "a": "a"}])
    for m in edges(-th["below_furniture_top_margin_m"], bd["vertical_band_m"]):
        add([_obj("a", furniture, (0, 0, 0.5), unit), _obj("t", "box", (0, 0, (1.0 + m) / 2), (0.4, 0.4, 1.0 + m))],
            [{"kind": "binary", "relation": "below", "t": "t", "a": "a"}])
    for e in edges(th["inside_tolerance_m"], bd["vertical_band_m"]):
        add([_obj("a", container, (0, 0, 0.5), unit), _obj("t", "box", (0.25 + e, 0, 0.5), (0.5, 0.5, 0.5))],
            [{"kind": "binary", "relation": "inside", "t": "t", "a": "a"}])
    db, L = bd["distance_band_m"], 2.0
    small = (0.1, 0.1, 0.1)
    for along in edges(0.0, db) + edges(L, db):
        add([_obj("a", "box", (0, 0, 0.5), small), _obj("b", "box", (L, 0, 0.5), small), _obj("t", "box", (along, 0.05, 0.5), small)],
            [{"kind": "between", "t": "t", "a": "a", "b": "b"}])
    limit = max(th["between_lateral_min_m"], th["between_lateral_ratio"] * L)
    for lat in edges(limit, db):
        add([_obj("a", "box", (0, 0, 0.5), small), _obj("b", "box", (L, 0, 0.5), small), _obj("t", "box", (1.0, lat, 0.5), small)],
            [{"kind": "between", "t": "t", "a": "a", "b": "b"}])
    for sep in edges(th["between_min_separation_m"], db):
        add([_obj("a", "box", (0, 0, 0.5), small), _obj("b", "box", (sep, 0, 0.5), small), _obj("t", "box", (sep / 2, 0, 0.5), small)],
            [{"kind": "between", "t": "t", "a": "a", "b": "b"}])
    for r in edges(th["between_overlap_max"], bd["overlap_band"]):
        add([_obj("a", "box", (0, 0, 0.5), unit), _obj("b", "box", (4.0, 0, 0.5), unit), _obj("t", "box", (1.0 - r, 0, 0.5), unit)],
            [{"kind": "between", "t": "t", "a": "a", "b": "b"}])
    for d in (0.0, 1e-12, -1e-12, 1e-9, -1e-9, 1e-6, -1e-6):
        gap = bd["rank_tie_m"] + d
        objs = [_obj("anchor", "box", (0, 0, 0.5), small), _obj("c1", "box", (1.0, 0, 0.5), small),
                _obj("c2", "box", (0, 1.0 + gap, 0.5), small), _obj("c3", "box", (-3.0, 0, 0.5), small)]
        add(objs, [{"kind": "rank", "relation": rel, "anchor": "anchor", "candidates": ["c1", "c2", "c3"], "possible": [],
                    "k": k} for rel in ("closest", "farthest") for k in (1, 2)])
    return scenes, cases


def boundary_cases() -> dict:
    scenes, checks = boundary_scenes()
    cfg = P.load_config()
    rels = []
    for sc in scenes:
        P.clear_geometry_cache()
        rels.append(P.Relations(sc, cfg))
    out = []
    for c in checks:
        rel = rels[c["scene"]]
        if c["kind"] == "binary":
            out.append(dict(c, result=_rel(getattr(rel, c["relation"])(c["t"], c["a"]))))
        elif c["kind"] == "between":
            out.append(dict(c, result=_rel(rel.between(c["t"], c["a"], c["b"]))))
        else:
            r = rel.rank(c["relation"], c["anchor"], c["candidates"], possible=c["possible"], k=c["k"])
            out.append(dict(c, status=r.status, object_ids=list(r.object_ids), reasons=list(r.reasons),
                            conditional=_conditional(r.inputs)))
    return {"scenes": scenes, "cases": out}


def write(scenes_folder, out_file, seed=7, pairs=60) -> dict:
    scenes = scene_records(Path(scenes_folder))
    if not scenes:
        raise SystemExit(f"no scene records under {scenes_folder}")
    g = relation_cases(scenes + variants(scenes), seed=seed, pairs=pairs)
    g["boundary"] = boundary_cases()
    Path(out_file).write_text(json.dumps(g, separators=(",", ":")), encoding="utf-8")
    counts = {k: len(v) for k, v in g.items() if isinstance(v, list)}
    counts["boundary"] = len(g["boundary"]["cases"])
    return counts


# ------------------------------------------------------------------------------------- stage 2: schemas, fuzzed
SCHEMA_FILES = ("scene.v1.json", "scene.v2.json", "command-context.v1.json", "category-map.v1.json",
                "grounding-query.v1.json", "resolver-config.v1.json", "relation-config.v1.json",
                "direction-config.v1.json")


def _jpath(parts) -> str:
    return "$" + "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in parts)


def valid_instances(work: Path) -> dict:
    """Valid records of every schema the resolution path uses, from A2.6's work bundles and the accepted code."""
    import importlib
    from ..a24 import gbnf_check as GC
    from ..a24 import language as L
    from ..contract import validate as CV
    PREDICT = importlib.import_module("grounding.evaluation.iref_vla.predict")
    proto = importlib.import_module("grounding.evaluation.iref_vla.protocol").load_protocol()
    out = {f: [] for f in SCHEMA_FILES}
    scenes, commands, maps = [], [], []
    for p in sorted(Path(work).rglob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:   # noqa: BLE001
            continue
        if not isinstance(d, dict):
            continue
        rt = d.get("record_type")
        if rt == "scene":
            scenes.append(d)
        elif rt == "command_context":
            commands.append(d)
        elif rt == "category_map":
            maps.append(d)
    for rec in scenes + commands + maps:
        out[CV.SCHEMAS[(rec["record_type"], rec["schema_version"])]].append(rec)
    spec = L.load_spec()
    texts = [s for s, ok in GC.corpus(spec) if ok]
    by_scene = {}
    for c in commands:
        by_scene.setdefault(c["scene_id"], c)
    pairs = [(s, by_scene[s["scene_id"]]) for s in scenes if s["scene_id"] in by_scene][:20]
    for i, text in enumerate(texts):
        scene, cmd = pairs[i % len(pairs)]
        out["grounding-query.v1.json"].append(L.query(L.interpretation(L.parse_output(text, spec)), scene, cmd,
                                                      proto["query_suffix"]))
    for m in maps[:3]:
        out["resolver-config.v1.json"].append(PREDICT.resolver_config(proto, m))
    out["relation-config.v1.json"].append(json.loads((Path(P.__file__).parent / "relations.v1.json").read_text()))
    out["direction-config.v1.json"].append(json.loads((Path(P.__file__).parent / "directions.v1.json").read_text()))
    return out


def _paths(x, path=()):
    yield path
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _paths(v, path + (k,))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _paths(v, path + (i,))


def _get(x, path):
    for p in path:
        x = x[p]
    return x


STRINGS = ("", "A", " x", "x ", "x\n", "\u001cx", "x\u00a0", "obj_12", "obj_007", "OBJ", "a b", "known", "unknown",
           "assumed", "yaw_only", "none", "\n", "a\u2028b")
NUMBERS = (-1, 0, -0.0, 1, 2, 3.0, 0.5, -0.5, 1e308, 10000.0, 64, 2.5, 1e-7, 1.000001)


def mutate(x, rng):
    import copy
    y = copy.deepcopy(x)
    for _ in range(rng.choice((1, 1, 2))):
        paths = list(_paths(y))
        path = rng.choice(paths)
        parent_path, key = (path[:-1], path[-1]) if path else (None, None)
        node = _get(y, path)
        op = rng.random()
        if path and op < 0.15:   # delete
            parent = _get(y, parent_path)
            if isinstance(parent, dict):
                del parent[key]
            elif len(parent) > 0:
                parent.pop(key)
        elif op < 0.25 and isinstance(node, dict):   # an unknown key
            node[rng.choice(("extra", "value", "evidence", "reason", "note"))] = rng.choice(STRINGS + NUMBERS)
        elif op < 0.45 and path:   # another value of another or the same type
            new = rng.choice((rng.choice(STRINGS), rng.choice(NUMBERS), True, False, None, [], {},
                              (node + rng.choice(("\n", " ", "x", "_1"))) if isinstance(node, str) else 0))
            _get(y, parent_path)[key] = new
        elif op < 0.65 and isinstance(node, list):   # array edits
            r = rng.random()
            if r < 0.3 and node:
                node.append(copy.deepcopy(node[0]))
            elif r < 0.5:
                node.clear()
            elif r < 0.75 and node:
                node.pop(rng.randrange(len(node)))
            else:
                node.extend(copy.deepcopy(node[:1]) * 5 if node else [1, 2, 3, 4, 5])
        elif op < 0.8 and isinstance(node, (int, float)) and not isinstance(node, bool) and path:
            _get(y, parent_path)[key] = rng.choice(NUMBERS)
        elif op < 0.9 and isinstance(node, str) and path:
            _get(y, parent_path)[key] = rng.choice(STRINGS)
        elif isinstance(node, dict) and "state" in node and path:   # flip a known/unknown wrapper's branch
            node["state"] = "unknown" if node["state"] == "known" else "known"
    return y


def schema_cases(work: Path, seed=13, per_family=500) -> dict:
    import jsonschema
    rng = random.Random(seed)
    schemas_dir = Path(P.__file__).resolve().parents[2] / "schemas"
    validators = {f: jsonschema.Draft202012Validator(json.loads((schemas_dir / f).read_text())) for f in SCHEMA_FILES}
    cases = []
    for f, items in valid_instances(work).items():
        if not items:
            continue
        for x in items[:20]:
            cases.append((f, x))
        for _ in range(per_family):
            cases.append((f, mutate(rng.choice(items), rng)))
    out = []
    for f, x in cases:
        try:
            json.dumps(x, allow_nan=False)
        except ValueError:
            continue
        errs = sorted((_jpath(e.absolute_path), e.validator) for e in validators[f].iter_errors(x))
        out.append({"schema": f, "instance": x, "errors": [list(e) for e in errs]})
    return {"format_version": 1, "record_type": "a27_schema_goldens", "seed": seed, "cases": out}


# ------------------------------------------------------------------------- stage 2: validation, targeted and fuzzed
def _issues(found) -> list:
    out = []
    for i in found:
        if isinstance(i, dict):
            out.append([i["label"], i["path"], i["code"]])
        else:
            out.append([i.file, i.path, i.code])
    return sorted(out)


def _semantic_mutations(scene, command, cmap):
    """(name, records) cases, each breaking one semantic or cross-record rule, or sitting just inside a tolerance."""
    import copy
    C = copy.deepcopy
    out = []

    def obj0(s):
        return s["objects"][0]

    s = C(scene); s["objects"].append(C(obj0(s))); out.append(("duplicate object", [("scene", s)]))
    s = C(scene); obj0(s)["category"]["evidence"]["source"] = "src_missing"; out.append(("unresolved source", [("scene", s)]))
    s = C(scene); obj0(s)["geometry"]["center_m"]["evidence"]["assumptions"] = ["asm_missing"]
    out.append(("unresolved assumption", [("scene", s)]))
    s = C(scene); obj0(s)["geometry"]["size_m"]["evidence"]["kind"] = "assumed"; out.append(("assumed from a dataset", [("scene", s)]))
    s = C(scene); obj0(s)["geometry"]["size_m"] = {"state": "unknown", "source": obj0(s)["geometry"]["size_m"]["evidence"]["source"],
                                                  "reason": "withheld_by_profile"}
    out.append(("withheld from a dataset", [("scene", s)]))
    s = C(scene); s["sources"].append(C(s["sources"][0])); out.append(("duplicate source", [("scene", s)]))
    s = C(scene); s.setdefault("assumptions", []).extend([{"assumption_id": "asm_x", "description": "x"}] * 2)
    out.append(("duplicate assumption", [("scene", s)]))
    for f, name in ((1 + 2e-6, "norm outside"), (1 + 0.5e-6, "norm inside")):
        s = C(scene); w = obj0(s)["geometry"]["rotation_xyzw"]
        if w["state"] == "known":
            w["value"] = [v * f for v in w["value"]]
            out.append((f"quaternion {name}", [("scene", s)]))
    for tilt, name in ((2e-6, "tilt outside"), (0.5e-6, "tilt inside")):
        s = C(scene); w = obj0(s)["geometry"]["rotation_xyzw"]
        if w["state"] == "known":
            w["value"] = [tilt] + w["value"][1:]
            out.append((f"yaw {name}", [("scene", s)]))
    s = C(scene); w = obj0(s)["geometry"]["rotation_xyzw"]
    if w["state"] == "known":
        w["support"] = "axis_aligned"; out.append(("axis aligned, not identity", [("scene", s)]))
        s2 = C(s); obj0(s2)["geometry"]["rotation_xyzw"]["value"] = [0.0, 0.0, 0.0, 1.0]
        out.append(("axis aligned identity", [("scene", s2)]))
    s = C(scene); s["evidence_profile"] = "restricted"; out.append(("restricted without withholding", [("scene", s)]))
    if s["schema_version"] == 2:
        s = C(scene); s["coordinate_frame"]["conversion"]["source_id"] = "src_missing"; out.append(("frame source", [("scene", s)]))
        s = C(scene); s["coordinate_frame"]["conversion"]["source_frame_id"] = "other_frame"; out.append(("frame conversion", [("scene", s)]))
    c = C(command); c["user_pose"]["scene_revision"] = (c["user_pose"]["scene_revision"] or 0) + 1 if isinstance(c["user_pose"]["scene_revision"], int) else 7
    out.append(("pose revision", [("command", c)]))
    for kind in ("quest_head", "none", "authored", "dataset_camera"):
        c = C(command); c["user_pose"]["pose_kind"] = kind; out.append((f"pose kind {kind}", [("command", c)]))
    c = C(command); h = c["user_pose"]["heading_xy"]
    if h["state"] == "known":
        h["value"] = [v * (1 + 2e-6) for v in h["value"]]; out.append(("heading length outside", [("command", c)]))
        c2 = C(command); c2["user_pose"]["heading_xy"]["heading_source"] = "authored"; out.append(("heading source", [("command", c2)]))
    m = C(cmap); m["entries"][0]["model"] = "not_a_model_label"; out.append(("map vocabulary", [("category_map", m)]))
    m = C(cmap); m["entries"].append(C(m["entries"][0])); out.append(("map duplicate", [("category_map", m)]))
    out.append(("valid set", [("scene", scene), ("command", command), ("category_map", cmap)]))
    out.append(("duplicate scene", [("scene", scene), ("scene2", C(scene)), ("category_map", cmap)]))
    c = C(command); c["scene_revision"] = 99; c["user_pose"]["scene_revision"] = 99
    out.append(("scene mismatch", [("scene", scene), ("command", c), ("category_map", cmap)]))
    c = C(command); c["user_pose"]["frame_id"] = "another_frame"; out.append(("frame mismatch", [("scene", scene), ("command", c)]))
    s = C(scene)
    for o in s["objects"]:
        if o["category"]["state"] == "known":
            o["category"]["value"]["standard"] = "not_a_standard_label"
            break
    out.append(("category unmapped", [("scene", s), ("category_map", cmap)]))
    out.append(("command alone", [("command", command)]))
    out.append(("scene alone", [("scene", scene)]))
    src = command["sources"][0]["source_id"]
    ev = {"kind": "annotated", "source": src, "assumptions": []}
    kp = C(command)
    up = kp["user_pose"]
    up["pose_kind"] = "dataset_camera"
    up["position_m"] = {"state": "known", "value": [1.0, 2.0, 1.5], "evidence": dict(ev)}
    up["rotation_xyzw"] = {"state": "known", "value": [0.0, 0.0, 0.6, 0.8], "support": "yaw_only", "evidence": dict(ev)}
    up["heading_xy"] = {"state": "known", "value": [0.6, 0.8], "heading_source": "camera_yaw", "evidence": dict(ev)}
    out.append(("known pose", [("command", kp)]))
    c = C(kp); c["user_pose"]["pose_kind"] = "none"; out.append(("none with known components", [("command", c)]))
    c = C(kp); c["user_pose"]["position_m"]["evidence"]["kind"] = "inferred"; out.append(("pose evidence kind", [("command", c)]))
    c = C(kp); c["user_pose"]["pose_kind"] = "authored"; out.append(("authored from a dataset", [("command", c)]))
    c = C(kp); c["user_pose"]["heading_xy"]["heading_source"] = "authored"; out.append(("heading source", [("command", c)]))
    for f, name in ((1 + 2e-6, "outside"), (1 + 0.5e-6, "inside")):
        c = C(kp); c["user_pose"]["heading_xy"]["value"] = [v * f for v in c["user_pose"]["heading_xy"]["value"]]
        out.append((f"heading length {name}", [("command", c)]))
        c = C(kp); c["user_pose"]["rotation_xyzw"]["value"] = [v * f for v in c["user_pose"]["rotation_xyzw"]["value"]]
        out.append((f"pose rotation norm {name}", [("command", c)]))
        sf = C(scene); sf["objects"][0]["semantic_front"] = {"state": "known", "value": [f, 0.0, 0.0],
                                                             "evidence": {"kind": "annotated", "source": scene["sources"][0]["source_id"], "assumptions": []}}
        out.append((f"front length {name}", [("scene", sf)]))
    return out


def validation_cases(work: Path, schema_goldens: dict) -> dict:
    import copy
    import importlib
    from ..contract import validate as CV
    from ..resolution import validate as RV
    PREDICT = importlib.import_module("grounding.evaluation.iref_vla.predict")
    proto = importlib.import_module("grounding.evaluation.iref_vla.protocol").load_protocol()
    inst = valid_instances(work)
    scenes, commands, maps = inst["scene.v2.json"] + inst["scene.v1.json"], inst["command-context.v1.json"], inst["category-map.v1.json"]
    by_id = {}
    for c in commands:
        by_id.setdefault(c["scene_id"], c)
    trios = [(s, by_id[s["scene_id"]], next(m for m in maps if m["map_id"] == s["category_map"]))
             for s in scenes if s["scene_id"] in by_id and any(m["map_id"] == s["category_map"] for m in maps)]
    contract_cases = []
    for k, (s, c, m) in enumerate(trios[:12]):
        for name, recs in _semantic_mutations(s, c, m):
            contract_cases.append({"name": f"{k}:{name}", "records": [[n, r] for n, r in recs],
                                   "issues": _issues(CV.validate([(n, r) for n, r in recs]))})
    family = {"scene.v1.json", "scene.v2.json", "command-context.v1.json", "category-map.v1.json"}
    for case in schema_goldens["cases"]:
        if case["schema"] in family:
            contract_cases.append({"name": "fuzz:" + case["schema"], "records": [["record", case["instance"]]],
                                   "issues": _issues(CV.validate([("record", case["instance"])]))})
    query_cases = []
    rng = random.Random(17)
    for q in inst["grounding-query.v1.json"]:
        query_cases.append(q)
    graph = []
    base = next(q for q in inst["grounding-query.v1.json"]
                if any(n["constraints"] for n in q["interpretations"][0]["nodes"]))
    ranked = next(q for q in inst["grounding-query.v1.json"]
                  if any(n["rank"] is not None for n in q["interpretations"][0]["nodes"]))

    def variant(fn, start=None):
        q = copy.deepcopy(start or base)
        fn(q["interpretations"][0], q)
        graph.append(q)
    variant(lambda i, q: i["nodes"].append(copy.deepcopy(i["nodes"][0])))                                   # repeated node
    variant(lambda i, q: i["nodes"][1].__setitem__("node_id", "obj_001"))                                   # reserved
    variant(lambda i, q: i.__setitem__("root", "nowhere"))                                                  # root missing
    variant(lambda i, q: i["nodes"].append({"node_id": "spare", "category": "chair", "colours_all": [],
                                            "constraints": [], "rank": None}))                              # unreachable
    def cyc(i, q):
        for n in i["nodes"]:
            if n["node_id"] != i["root"]:
                n["constraints"] = [{"relation": "near", "frame": None, "anchors": [i["root"]]}]
    variant(cyc)                                                                                            # cycle
    def selfdep(i, q):
        n = i["nodes"][0]
        n["constraints"] = [{"relation": "near", "frame": None, "anchors": [n["node_id"]]}]
    variant(selfdep)
    def unknown_anchor(i, q):
        i["nodes"][0]["constraints"] = [{"relation": "near", "frame": None, "anchors": ["ghost"]}]
    variant(unknown_anchor)
    def repeat_cons(i, q):
        n = next(n for n in i["nodes"] if n["constraints"])
        n["constraints"].append(copy.deepcopy(n["constraints"][0]))
    variant(repeat_cons)
    def float_k(i, q):
        for n in i["nodes"]:
            if n["rank"] is not None:
                n["rank"]["k"] = float(n["rank"]["k"])
    variant(float_k, ranked)
    variant(lambda i, q: q["interpretations"].append(copy.deepcopy(i)))                                     # repeated ID
    variant(lambda i, q: i["nodes"][0].__setitem__("category", (i["nodes"][0]["category"] or "chair") + "\n"))
    variant(lambda i, q: q.__setitem__("query_id", q["query_id"] + "\n"))
    variant(lambda i, q: i["nodes"][0].__setitem__("colours_all", ["red", "blue", "green"]))
    query_cases += graph
    query_cases += [c["instance"] for c in schema_goldens["cases"] if c["schema"] == "grounding-query.v1.json"]
    queries = []
    for q in query_cases:
        found, norm = RV.check_query(q)
        queries.append({"query": q, "issues": _issues(found),
                        "normalized": None if norm is None else json.dumps(norm, sort_keys=True, separators=(",", ":"))})
    configs = []
    for cfg in inst["resolver-config.v1.json"]:
        for mut in (None, "float_limit", "newline_id", "space_label", "fuzz"):
            c = copy.deepcopy(cfg)
            if mut == "float_limit":
                k = sorted(c["limits"])[0]; c["limits"][k] = float(c["limits"][k])
            elif mut == "newline_id":
                c["config_id"] = c["config_id"] + "\n"
            elif mut == "space_label":
                c["category_labels"] = c["category_labels"] + [c["category_labels"][0] + " "]
            elif mut == "fuzz":
                c = mutate(c, rng)
            configs.append({"config": c, "issues": _issues(RV.check_config(c))})
    records = []
    for s, c, m in trios[:12]:
        other = next((x for x in maps if x["map_id"] != m["map_id"]), None)
        for label, sc, cm, ms in (("valid", s, c, [m]), ("no map", s, c, []), ("other map", s, c, [other] if other else []),
                                  ("role", c, c, [m]), ("other scene", s, dict(copy.deepcopy(c), scene_id="another_scene"), [m])):
            warnings_out = []
            errors, warnings = RV.check_records(sc, cm, ms)
            records.append({"label": label, "scene": sc, "command": cm, "maps": ms, "errors": _issues(errors),
                            "warnings": _issues(warnings)})
    contexts = []
    for s, c, m in trios[:6]:
        q = next(q for q in inst["grounding-query.v1.json"])
        q = copy.deepcopy(q)
        for key, value in (("scene_id", s["scene_id"]), ("scene_revision", s["scene_revision"]),
                           ("evidence_profile", s["evidence_profile"]), ("command_id", c["command_id"])):
            q[key] = value
        contexts.append({"query": q, "scene": s, "command": c, "issues": _issues(RV.check_context(q, s, c))})
        for key, value in (("scene_id", "other"), ("scene_revision", 99), ("evidence_profile", "restricted"), ("command_id", "other")):
            q2 = dict(q, **{key: value})
            contexts.append({"query": q2, "scene": s, "command": c, "issues": _issues(RV.check_context(q2, s, c))})
    return {"format_version": 1, "record_type": "a27_validation_goldens", "contract": contract_cases,
            "queries": queries, "configs": configs, "records": records, "contexts": contexts}


# ------------------------------------------------------------------------------- stage 3: resolution, end to end
def resolver_protocol() -> dict:
    """The resolver protocol the headset loads (resolver-protocol.json): what predict.resolver_config takes from it."""
    import importlib
    PREDICT = importlib.import_module("grounding.evaluation.iref_vla.predict")
    proto = importlib.import_module("grounding.evaluation.iref_vla.protocol").load_protocol()
    return {"format_version": 1, "record_type": "a27_resolver_protocol", "config_id": proto["resolver"]["config_id"],
            "vocabulary_id": proto["resolver"]["vocabulary_id"],
            "limits": {n: proto["resolver"]["limits"][n] for n in PREDICT.LIMITS},
            "colour_labels": list(proto["colour_labels"]), "query_suffix": proto["query_suffix"]}


def _run(res) -> dict:
    r = res.get("result")
    out = {"processing_status": res["processing_status"], "issues": _issues(res["issues"]),
           "warnings": _issues(res["diagnostics"]["warnings"]), "budget": res["budget"],
           "work": res["diagnostics"]["work"] if res["processing_status"] != "invalid_input" else None, "result": None}
    if r is not None:
        out["result"] = {k: r[k] for k in ("status", "action", "target_id", "reason_code", "reason_codes", "conditional")}
        out["result"]["candidates"] = r["candidates"]
    return out


def resolution_cases(work: Path, seed=23, per_scene=40) -> dict:
    import copy
    import importlib
    from ..a24 import language as L
    from ..contract import validate as CV
    RS = importlib.import_module("grounding.resolution.resolve")
    from ..resolution import validate as RV
    from ..relations import directions as DIR
    PREDICT = importlib.import_module("grounding.evaluation.iref_vla.predict")
    proto = importlib.import_module("grounding.evaluation.iref_vla.protocol").load_protocol()
    rng = random.Random(seed)
    spec = L.load_spec()
    vocab = sorted(L._sets(spec)[0])   # the language's 893 category labels
    rel_path, dir_path = P.CONFIG_FILE, Path(DIR.__file__).resolve().parent / "directions.v1.json"
    inst = valid_instances(work)
    scenes, commands, maps = inst["scene.v2.json"] + inst["scene.v1.json"], inst["command-context.v1.json"], inst["category-map.v1.json"]
    by_id = {}
    for c in commands:
        by_id.setdefault(c["scene_id"], c)
    trios = [(s, by_id[s["scene_id"]], next(m for m in maps if m["map_id"] == s["category_map"]))
             for s in scenes if s["scene_id"] in by_id and any(m["map_id"] == s["category_map"] for m in maps)]
    cases, table, index = [], [], {}

    def ref(record):
        key = json.dumps(record, sort_keys=True)
        if key not in index:
            index[key] = len(table)
            table.append(record)
        return index[key]

    def resolve_case(name, interp, scene, command, cmap, config=None):
        q = L.query(copy.deepcopy(interp), scene, command, proto["query_suffix"])
        if config is None:
            found, _ = RV.check_query(q)
            if found:
                cases.append({"kind": "checker", "name": name, "interp": interp, "scene": ref(scene), "command": ref(command),
                              "map": ref(cmap), "outcome": "invalid_label_query", "query_issues": _issues(found), "run": None})
                return
            cfg = PREDICT.resolver_config(proto, cmap)
        else:
            cfg = config
        P.clear_geometry_cache()   # variants reuse their scene's IDs with other geometry; the cache only speeds up
        res = RS.resolve(scene, command, q, resolver_config=cfg, relation_config_path=rel_path,
                         direction_config_path=dir_path, category_maps=[cmap], trace=False)
        done = res["processing_status"] == "completed"
        outcome = res["result"]["status"] if done else res["processing_status"]
        case = {"kind": "checker" if config is None else "resolve", "name": name, "interp": interp, "scene": ref(scene),
                "command": ref(command), "map": ref(cmap), "outcome": outcome, "run": _run(res)}
        if config is not None:
            case["config"] = cfg
            case["query"] = q
        cases.append(case)

    def random_parsed(scene_cats, colours):
        form = rng.choice(L.relation_forms())
        parts = form.split()
        rel, k = parts[0], (int(parts[1]) if len(parts) > 1 else None)

        def o():
            cat = rng.choice(scene_cats) if scene_cats and rng.random() < 0.75 else rng.choice(vocab)
            return {"category": cat, "colour": rng.choice(colours) if rng.random() < 0.25 else None}
        return {"target": o(), "relation": rel, "k": k, "anchors": [o(), o()] if rel == L.BETWEEN else [o()]}

    def interp_of(parsed):
        return L.interpretation(L.parse_output(L.render(parsed), spec))

    def unknown_variant(scene):
        v = copy.deepcopy(scene)
        src = v["sources"][0]["source_id"]
        v.setdefault("assumptions", []).append({"assumption_id": "asm_a27_golden", "description": "a labelled golden assumption"})
        for o in v["objects"]:
            for holder, key in ((o, "category"), (o["geometry"], "center_m"), (o["geometry"], "size_m"),
                                (o["geometry"], "rotation_xyzw"), (o["attributes"], "colours")):
                u = rng.random()
                if u < 0.15:
                    holder[key] = {"state": "unknown", "source": src, "reason": "not_in_source"}
                elif u < 0.30 and holder[key]["state"] == "known":
                    holder[key]["evidence"]["assumptions"] = ["asm_a27_golden"]
        v["scene_id"] = scene["scene_id"]
        return v

    colours = list(L.COLOURS)
    for si, (scene, command, cmap) in enumerate(trios):
        for variant, sc in (("real", scene), ("unknown", unknown_variant(scene))):
            if variant == "unknown" and any(i.code.startswith("E_") for i in CV.validate([("s", sc)])):
                continue
            cats = sorted({o["category"]["value"]["model"] for o in sc["objects"] if o["category"]["state"] == "known"})
            for k in range(per_scene if variant == "real" else per_scene // 2):
                resolve_case(f"{si}:{variant}:{k}", interp_of(random_parsed(cats, colours)), sc, command, cmap)
    # boundary scenes through the whole resolver: three objects of the scene, placed at the band edges
    cfg = P.load_config()
    th, bd = cfg.thresholds, cfg.bands
    deltas = (0.0, 1e-12, -1e-12, 1e-9, -1e-9, 1e-6, -1e-6)
    base_scene, base_cmd, base_map = next((s, c, m) for s, c, m in trios if len(s["objects"]) >= 3)

    def placed(positions, sizes):
        s = copy.deepcopy(base_scene)
        s["objects"] = s["objects"][:len(positions)]
        for o, pos, size in zip(s["objects"], positions, sizes):
            o["geometry"]["center_m"]["value"] = [float(x) for x in pos]
            o["geometry"]["size_m"]["value"] = [float(x) for x in size]
            o["geometry"]["rotation_xyzw"]["value"] = [0.0, 0.0, 0.0, 1.0]
            o["geometry"]["rotation_xyzw"]["support"] = "yaw_only"
        return s

    def cat_of(s, i):
        return s["objects"][i]["category"]["value"]["model"]
    unit, small = (1.0, 1.0, 1.0), (0.1, 0.1, 0.1)
    for d in deltas:
        for edge in (-1.0, 1.0):
            g = th["near_max_m"] + edge * bd["distance_band_m"] + d
            s = placed([(0, 0, 0.5), (1.0 + g, 0, 0.5)], [unit, unit])
            resolve_case(f"boundary near {edge} {d}", interp_of({"target": {"category": cat_of(s, 1), "colour": None}, "relation": "near",
                                                                 "k": None, "anchors": [{"category": cat_of(s, 0), "colour": None}]}),
                         s, base_cmd, base_map)
            along = edge * bd["distance_band_m"] + d
            s = placed([(0, 0, 0.5), (2.0, 0, 0.5), (along, 0.05, 0.5)], [small, small, small])
            resolve_case(f"boundary between {edge} {d}", interp_of({"target": {"category": cat_of(s, 2), "colour": None}, "relation": "between",
                                                                    "k": None, "anchors": [{"category": cat_of(s, 0), "colour": None},
                                                                                           {"category": cat_of(s, 1), "colour": None}]}),
                         s, base_cmd, base_map)
            sep = -th["below_separated_gap_min_m"] + edge * bd["vertical_band_m"] + d
            s = placed([(0, 0, 2.5), (0, 0, 1.5 + sep)], [unit, unit])
            resolve_case(f"boundary below {edge} {d}", interp_of({"target": {"category": cat_of(s, 1), "colour": None}, "relation": "below",
                                                                  "k": None, "anchors": [{"category": cat_of(s, 0), "colour": None}]}),
                         s, base_cmd, base_map)
        gap = bd["rank_tie_m"] + d
        s = placed([(0, 0, 0.5), (1.0, 0, 0.5), (0, 1.0 + gap, 0.5)], [small, small, small])
        for rel in ("closest", "farthest"):
            resolve_case(f"boundary rank {rel} {d}", interp_of({"target": {"category": cat_of(s, 1), "colour": None}, "relation": rel,
                                                                "k": 1, "anchors": [{"category": cat_of(s, 0), "colour": None}]}),
                         s, base_cmd, base_map)
    # deliberate failure paths
    scene, command, cmap = trios[0]
    good = interp_of(random_parsed(sorted({o["category"]["value"]["model"] for o in scene["objects"]
                                            if o["category"]["state"] == "known"}), colours))
    bad = copy.deepcopy(good)
    bad["nodes"][0]["category"] = (bad["nodes"][0]["category"] or "chair") + "\n"
    resolve_case("invalid label", bad, scene, command, cmap)
    present = {o["category"]["value"]["model"] for o in scene["objects"] if o["category"]["state"] == "known"}
    absent = next(c for c in sorted(cmap["model_vocabulary"]) if c not in present)
    small_map = copy.deepcopy(cmap)
    small_map["model_vocabulary"] = [c for c in small_map["model_vocabulary"] if c != absent]
    small_map["entries"] = [e for e in small_map["entries"] if e["model"] != absent]
    small_map["map_id"] = cmap["map_id"] + ".reduced"   # a map's ID is its identity: the checker caches configurations by it
    small_scene = copy.deepcopy(scene)
    small_scene["category_map"] = small_map["map_id"]
    resolve_case("unsupported label", interp_of({"target": {"category": absent, "colour": None}, "relation": "near", "k": None,
                                                 "anchors": [{"category": sorted(present)[0], "colour": None}]}), small_scene, command, small_map)
    def forced(cats, form, k=None):
        o = {"category": cats[0], "colour": None}
        return interp_of({"target": dict(o), "relation": form, "k": k, "anchors": [dict(o), dict(o)] if form == L.BETWEEN else [dict(o)]})
    for limits, form, k in (({"max_candidate_checks": 5}, "near", None), ({"max_nodes": 2}, L.BETWEEN, None),
                            ({"max_rank_subset_evaluations": 1}, "closest", 1), ({"max_predicate_calls": 3}, "near", None),
                            ({"max_binding_attempts": 1}, "near", None)):
        for i in range(4):
            s, c, m = trios[i % len(trios)]
            uv = unknown_variant(s)
            if any(x.code.startswith("E_") for x in CV.validate([("s", uv)])):
                uv = s
            cats = sorted({o["category"]["value"]["model"] for o in uv["objects"] if o["category"]["state"] == "known"})
            resolve_case(f"budget {limits} {i}", forced(cats, form, k), uv, c, m,
                         config=PREDICT.resolver_config(proto, m, limits=limits))
    broken = copy.deepcopy(scene); broken["objects"].append(copy.deepcopy(broken["objects"][0]))
    resolve_case("invalid scene", good, broken, command, cmap, config=PREDICT.resolver_config(proto, cmap))
    other = next((c for c in commands if c["scene_id"] != scene["scene_id"]), None)
    if other is not None:
        resolve_case("other scene's command", good, scene, other, cmap, config=PREDICT.resolver_config(proto, cmap))
    float_cfg = PREDICT.resolver_config(proto, cmap); float_cfg["limits"]["max_nodes"] = 3.0
    resolve_case("float limit", good, scene, command, cmap, config=float_cfg)
    return {"format_version": 1, "record_type": "a27_resolution_goldens", "protocol": resolver_protocol(), "records": table,
            "cases": cases}


# ------------------------------------------------------------------------- stage 4: reader, pipeline, native adapter
def _canon(x) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"))


def reader_cases(work: Path, seed=29, renders=600) -> dict:
    import copy
    from ..a24 import gbnf_check as GC
    from ..a24 import language as L
    rng = random.Random(seed)
    spec = L.load_spec()
    cats = sorted(L._sets(spec)[0])
    inst = valid_instances(work)
    scene, command = inst["scene.v2.json"][0], next(c for c in inst["command-context.v1.json"] if c["scene_id"] == inst["scene.v2.json"][0]["scene_id"])
    texts = [s for s, _ in GC.corpus(spec)]
    for _ in range(renders):
        form = rng.choice(L.relation_forms()).split()
        rel, k = form[0], (int(form[1]) if len(form) > 1 else None)
        o = lambda: {"category": rng.choice(cats), "colour": rng.choice(L.COLOURS) if rng.random() < 0.3 else None}   # noqa: E731
        texts.append(L.render({"target": o(), "relation": rel, "k": k, "anchors": [o(), o()] if rel == L.BETWEEN else [o()]}))
    broken = []
    for t in texts[:300]:
        for f in (lambda s: s.replace(" | ", "|", 1), lambda s: s + " ", lambda s: " " + s, lambda s: s.replace("closest", "nearest"),
                  lambda s: s + " | chair", lambda s: s.replace(" [", "[", 1), lambda s: s.replace("]", "", 1),
                  lambda s: s.replace(" | ", " | between | ", 1), lambda s: s.replace(" 1", " 4"), lambda s: s.upper(), lambda s: ""):
            broken.append(f(t))
    out = []
    for t in texts + broken:
        try:
            parsed = L.parse_output(t, spec)
        except L.LanguageError:
            out.append({"text": t, "valid": False})
            continue
        interp = L.interpretation(parsed)
        out.append({"text": t, "valid": True, "parsed": parsed, "render": L.render(parsed), "interpretation": _canon(interp),
                    "query": _canon(L.query(copy.deepcopy(interp), scene, command, ".q"))})
    return {"format_version": 1, "record_type": "a27_reader_goldens", "scene": scene, "command": command, "cases": out}


def pipeline_cases(work: Path, seed=31, per_scene=12) -> dict:
    """End to end through the pipeline with a fake native side: prompts tokenized by the pinned tokenizer, decoded
    outputs supplied as token IDs, and the checker's expected outcome from the Python reference."""
    import copy
    import importlib
    from ..a24 import decoder as DC
    from ..a24 import language as L
    from ..a24 import prep as PR
    from ..preparation.iref_vla import tokens as T
    PREDICT = importlib.import_module("grounding.evaluation.iref_vla.predict")
    proto = importlib.import_module("grounding.evaluation.iref_vla.protocol").load_protocol()
    RS = importlib.import_module("grounding.resolution.resolve")
    from ..resolution import validate as RV
    from ..relations import directions as DIR
    rng = random.Random(seed)
    spec, settings = L.load_spec(), PR.load_settings()
    tok = T.load_pinned_tokenizer(PR.DEV.TOKENIZER)
    table = DC.TokenTable(tok._backend)
    eos = table.eos_id
    inst = valid_instances(work)
    scenes, commands, maps = inst["scene.v2.json"], inst["command-context.v1.json"], inst["category-map.v1.json"]
    by_id = {}
    for c in commands:
        by_id.setdefault(c["scene_id"], c)
    trios = [(s, by_id[s["scene_id"]], next(m for m in maps if m["map_id"] == s["category_map"]))
             for s in scenes if s["scene_id"] in by_id and any(m["map_id"] == s["category_map"] for m in maps)]
    cats_all = sorted(L._sets(spec)[0])
    records, index, cases, used = [], {}, [], set()

    def ref(x):
        k = json.dumps(x, sort_keys=True)
        if k not in index:
            index[k] = len(records)
            records.append(x)
        return index[k]

    def expected(text, scene, command, cmap):
        try:
            parsed = L.parse_output(text, spec)
        except L.LanguageError:
            return {"status": "failed", "outcome": "invalid_output"}
        q = L.query(L.interpretation(parsed), scene, command, proto["query_suffix"])
        found, _ = RV.check_query(q)
        if found:
            return {"status": "invalid", "outcome": "invalid_label_query"}
        P.clear_geometry_cache()
        res = RS.resolve(scene, command, q, resolver_config=PREDICT.resolver_config(proto, cmap), relation_config_path=P.CONFIG_FILE,
                         direction_config_path=Path(DIR.__file__).resolve().parent / "directions.v1.json", category_maps=[cmap], trace=False)
        done = res["processing_status"] == "completed"
        outcome = res["result"]["status"] if done else res["processing_status"]
        r = res["result"] or {}
        kind = {"resolved": "selected", "ambiguous": "ask", "no_match": "ask", "insufficient_information": "ask",
                "unsupported": "unsupported", "invalid_input": "invalid", "budget_exceeded": "budget"}[outcome]
        return {"status": kind, "outcome": outcome, "target_id": r.get("target_id") if outcome == "resolved" else None,
                "candidate_ids": list((r.get("candidates") or {}).get("target_ids") or []), "reason_code": r.get("reason_code")}

    def add(name, scene, command, cmap, out_tokens, status, exp, scene_text=None, command_text=None, map_text=None):
        prompt = PR.parser_prompt(command["text"], settings) if isinstance(command, dict) and isinstance(command.get("text"), str) else ""
        ids = tok.encode(prompt) if prompt else []
        used.update(out_tokens)
        cases.append({"name": name, "scene_text": scene_text if scene_text is not None else json.dumps(scene),
                      "command_text": command_text if command_text is not None else json.dumps(command),
                      "map_text": map_text if map_text is not None else json.dumps(cmap),
                      "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest() if prompt else None,
                      "prompt_ids": ids, "output_tokens": out_tokens, "decode_status": status, "expected": exp})

    for si, (scene, command, cmap) in enumerate(trios):
        present = sorted({o["category"]["value"]["model"] for o in scene["objects"] if o["category"]["state"] == "known"})
        for k in range(per_scene):
            form = rng.choice(L.relation_forms()).split()
            rel, kk = form[0], (int(form[1]) if len(form) > 1 else None)
            o = lambda: {"category": rng.choice(present) if rng.random() < 0.8 else rng.choice(cats_all),   # noqa: E731
                         "colour": rng.choice(L.COLOURS) if rng.random() < 0.2 else None}
            text = L.render({"target": o(), "relation": rel, "k": kk, "anchors": [o(), o()] if rel == L.BETWEEN else [o()]})
            add(f"{si}:{k}", scene, command, cmap, list(tok.encode(text)) + [eos], 0, expected(text, scene, command, cmap))
    scene, command, cmap = trios[0]
    good = tok.encode("chair | near | table")
    add("length exhausted", scene, command, cmap, list(good), 1, {"status": "failed", "outcome": "decode:length_exhausted"})
    add("empty allowed set", scene, command, cmap, list(good[:2]), 2, {"status": "failed", "outcome": "decode:empty_allowed_set"})
    add("technical decode", scene, command, cmap, [], -1, {"status": "failed", "outcome": "technical_failure"})
    add("outside the language", scene, command, cmap, list(tok.encode("chair | nearby | table")) + [eos], 0,
        {"status": "failed", "outcome": "invalid_output"})
    add("byte-order mark", scene, command, cmap, [], 0, {"status": "invalid", "outcome": "invalid_input"}, scene_text="\ufeff" + json.dumps(scene))
    add("malformed scene", scene, command, cmap, [], 0, {"status": "invalid", "outcome": "invalid_input"}, scene_text=json.dumps(scene)[:-1])
    add("repeated key", scene, command, cmap, [], 0, {"status": "invalid", "outcome": "invalid_input"},
        command_text=json.dumps(command)[:-1] + ', "command_id": "x"}')
    nt = copy.deepcopy(command); del nt["text"]
    add("no command text", scene, nt, cmap, [], 0, {"status": "invalid", "outcome": "invalid_input"})
    bad_scene = copy.deepcopy(scene); bad_scene["objects"].append(copy.deepcopy(bad_scene["objects"][0]))
    text = "chair | near | table"
    add("invalid scene record", bad_scene, command, cmap, list(tok.encode(text)) + [eos], 0, expected(text, bad_scene, command, cmap))
    return {"format_version": 1, "record_type": "a27_pipeline_goldens", "protocol": resolver_protocol(), "end_token": eos,
            "token_bytes": {str(t): table.bytes_of[t].hex() for t in sorted(used) if t != eos}, "cases": cases}


def native_cases(lib, gguf, seed=37) -> dict:
    """What Python gets from the native library through ctypes, for the C# adapter to reproduce through P/Invoke."""
    import ctypes as C
    import importlib
    from ..a24 import decoder as DC
    from ..a24 import language as L
    from ..a24 import prep as PR
    from ..preparation.iref_vla import tokens as T
    from .native import Native
    spec, settings = L.load_spec(), PR.load_settings()
    tok = T.load_pinned_tokenizer(PR.DEV.TOKENIZER)
    con = DC.Constraint(spec, DC.TokenTable(tok._backend))
    nat = Native(lib, gguf, L.gbnf(spec), con.T.eos_id, n_ctx=1024, n_threads=2)
    nat.lib.se_tokenize.restype, nat.lib.se_tokenize.argtypes = C.c_int32, [C.c_void_p, C.c_char_p, C.POINTER(C.c_int32), C.c_int32]
    prompts, decodes, masks = [], [], []
    for text in ("the chair near the table", "the second closest lamp to the bed", "the box between the sofa and the door",
                 "a white mug on the desk"):
        prompt = PR.parser_prompt(text, settings)
        buf = (C.c_int32 * 4096)()
        n = nat.lib.se_tokenize(nat.m, prompt.encode("utf-8"), buf, 4096)
        ids = list(buf[:n])
        prompts.append({"prompt": prompt, "native_ids": ids, "pinned_ids": tok.encode(prompt)})
        decodes.append({"prompt_ids": ids, **nat.decode(ids, spec["max_new_tokens"])})
    for p in ("", "chair | ", "chair | closest ", "lamp [white] | between | sofa | "):
        prefix = tok.encode(p)
        allowed = nat.allowed_after(prefix)
        masks.append({"prefix": prefix, "count": len(allowed),
                      "ids_sha256": hashlib.sha256(",".join(map(str, sorted(allowed))).encode("ascii")).hexdigest()})
    nat.close()
    return {"format_version": 1, "record_type": "a27_native_goldens", "end_token": con.T.eos_id, "max_new_tokens": spec["max_new_tokens"],
            "gbnf": L.gbnf(spec),
            "prompts": prompts, "decodes": decodes, "masks": masks}


# ------------------------------------------------------------------------------------------- everything at once
def write_all(*, samples, official, out, lib=None, gguf=None) -> dict:
    """A2.4's fixture from the release samples, then every golden file the C# tests compare with (stages 1-4)."""
    import importlib
    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} already exists")
    out.mkdir(parents=True)
    TB = importlib.import_module("grounding.tests.test_iref_vla_baseline")
    TP = importlib.import_module("grounding.tests.test_a24_prep")
    _, work, _ = TP.locked(lambda k: TB.prepared_fixture(out / f"fixture-{k}", Path(samples), Path(official)), fresh=True)
    work = Path(work)
    counts = {}

    def save(name, g):
        (out / name).write_text(json.dumps(g, separators=(",", ":")), encoding="utf-8")
        counts[name] = (out / name).stat().st_size
    scenes = scene_records(work)
    rel = relation_cases(scenes + variants(scenes))
    rel["boundary"] = boundary_cases()
    save("relations.json", rel)
    sg = schema_cases(work)
    save("schemas.json", sg)
    save("validation.json", validation_cases(work, sg))
    save("resolution.json", resolution_cases(work))
    save("reader.json", reader_cases(work))
    save("pipeline.json", pipeline_cases(work))
    if lib and gguf:
        save("native.json", native_cases(lib, gguf))
    return counts
