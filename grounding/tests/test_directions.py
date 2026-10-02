"""Tests of the directional relations (A2.1c, D68).

    python grounding/tests/test_directions.py

Needs only Python 3.9+ and jsonschema. fixtures/directions/cases.json holds the brief's 43 fixed rows, transcribed
before the directional code existed; expected scores are the brief's arithmetic expressions, evaluated here by a
small arithmetic reader, never by the library. The cross-cutting checks follow Section 8 of the brief, numbered as
there. Prints one line per check and a count of rows, calls and checks, and exits 1 if any check fails.
"""
from __future__ import annotations

import ast
import copy
import dataclasses
import json
import math
import operator
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "grounding" / "relations"))
sys.path.insert(0, str(REPO / "grounding" / "contract"))
import directions as D  # noqa: E402
import geometry as g  # noqa: E402
import predicates as P  # noqa: E402
import validate  # noqa: E402

FIXTURES = REPO / "grounding" / "tests" / "fixtures" / "directions"
MAP = REPO / "grounding" / "tests" / "fixtures" / "contract" / "valid" / "category-map.fx.json"
T, F, U = P.TRUE, P.FALSE, P.UNKNOWN
VALUES = {"true": T, "false": F, "unknown": U}
OPPOSITE = {"right": "left", "left": "right", "in_front_of": "behind", "behind": "in_front_of"}
FAILED = []
COUNT = {"checks": 0}


def check(name: str, ok: bool, detail: str = "") -> None:
    COUNT["checks"] += 1
    print(("PASS  " if ok else "FAIL  ") + name + (f": {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def expr(text):
    """Value of one of the brief's arithmetic expressions: numbers, + - * /, unary minus, brackets and sqrt."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return -ev(node.operand) if isinstance(node.op, ast.USub) else ev(node.operand)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "sqrt" and len(node.args) == 1:
            return math.sqrt(ev(node.args[0]))
        raise ValueError(f"not plain arithmetic: {text}")
    return ev(ast.parse(text, mode="eval"))


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(evaluators, commands, call, relation=None):
    return evaluators[call["scene"]].evaluate(relation or call["relation"], call["target"], frame=call["frame"],
                                              anchor_id=call["anchor"],
                                              command=commands[call["command"]] if call["command"] else None)


def differences(call, result, tol):
    e, out = call["expect"], []
    if result.value is not VALUES[e["value"]]:
        out.append(f"got {result.value.value}")
    if e["score"] is None:
        if result.requested_score_m is not None:
            out.append(f"score {result.requested_score_m}, expected none")
    elif result.requested_score_m is None or abs(result.requested_score_m - expr(e["score"])) > tol:
        out.append(f"score {result.requested_score_m}, expected {e['score']}")
    if set(result.reasons) != set(e["reasons"]):
        out.append(f"reasons {list(result.reasons)}")
    for k, v in e.get("measures", {}).items():
        if k not in result.measures or abs(result.measures[k] - expr(v)) > tol:
            out.append(f"{k} = {result.measures.get(k)}, expected {v}")
    return out


# ---------------------------------------------------------------------------------------- record transforms

def transform_records(scene, commands, point, vector, quaternion=None):
    """Copies of a scene and its commands with every position mapped by point(), every heading and semantic front
    by vector(), and, if given, every known rotation pre-multiplied by quaternion (a yaw)."""
    scene = copy.deepcopy(scene)
    for o in scene["objects"]:
        c = o["geometry"]["center_m"]
        if c["state"] == "known":
            c["value"] = point(c["value"])
        f = o["semantic_front"]
        if f["state"] == "known":
            f["value"] = vector(f["value"])
        r = o["geometry"]["rotation_xyzw"]
        if quaternion and r["state"] == "known":
            r["value"] = qmul(quaternion, r["value"])
            if r["support"] == "axis_aligned":
                r["support"] = "yaw_only"
    out = {}
    for key, cmd in commands.items():
        cmd = copy.deepcopy(cmd)
        pose = cmd["user_pose"]
        if pose["position_m"]["state"] == "known":
            pose["position_m"]["value"] = point(pose["position_m"]["value"])
        if pose["heading_xy"]["state"] == "known":
            pose["heading_xy"]["value"] = vector(pose["heading_xy"]["value"] + [0.0])[:2]
        if quaternion and pose["rotation_xyzw"]["state"] == "known":
            pose["rotation_xyzw"]["value"] = qmul(quaternion, pose["rotation_xyzw"]["value"])
        out[key] = cmd
    return scene, out


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz]


def turn(theta):
    c, s = math.cos(theta), math.sin(theta)
    return (lambda p: [c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]],
            lambda v: [c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]],
            [0.0, 0.0, math.sin(theta / 2), math.cos(theta / 2)])


def contract_errors(records):
    return [i for i in validate.validate(records) if i.is_error]


def main() -> int:
    doc = load_json(FIXTURES / "cases.json")
    tol = doc["tolerance_m"]
    scenes = {k: load_json(FIXTURES / v) for k, v in doc["scenes"].items()}
    commands = {k: load_json(FIXTURES / v) for k, v in doc["commands"].items()}
    cfg = D.load_direction_config()
    ev = {k: D.DirectionalRelations(rec, cfg) for k, rec in scenes.items()}
    rows = doc["rows"]
    calls = [c for r in rows for c in r["calls"]]
    by_id = {r["id"]: r for r in rows}

    # ---- Fixed cases (Section 7)
    print("-- Section 7: fixed rows")
    results = {}
    for r in rows:
        for n, call in enumerate(r["calls"]):
            result = run(ev, commands, call)
            results[(r["id"], n)] = result
            diff = differences(call, result, tol)
            check(f"{r['id']}.{n + 1} {call['frame']} {call['relation']}({call['target']}"
                  f"{', ' + call['anchor'] if call['anchor'] else ''}) -> {call['expect']['value']}", not diff,
                  "; ".join(diff))
    cases_checks = COUNT["checks"]

    print("-- Section 8: cross-cutting checks")
    # 1. Compatible records
    files = sorted(FIXTURES.glob("scene.*.json")) + sorted(FIXTURES.glob("command.*.json")) + [MAP]
    issues = validate.validate_paths(files)
    no_pose = all(c["command"] is None for c in calls if c["frame"] == "object_intrinsic")
    check("8.1 every direction fixture passes the contract validator with its map, and intrinsic calls take no pose",
          not issues and no_pose, f"{len(files)} files" if not issues else "; ".join(i.line() for i in issues[:3]))

    # 2. Input errors
    h, v, i = ev["h"], ev["v"], ev["i"]
    hb, vb = commands["h.base"], commands["v.base"]
    other_scene = dict(vb, scene_id="fixture.other")
    other_revision = dict(vb, scene_revision=1)
    other_frame = copy.deepcopy(vb)
    other_frame["user_pose"]["frame_id"] = "elsewhere"
    misuse = {
        "unsupported frame": lambda: h.evaluate("right", "obj_001", frame="world", command=hb),
        "unsupported relation": lambda: h.evaluate("above", "obj_001", frame="user_heading", command=hb),
        "user_heading without a command": lambda: h.evaluate("right", "obj_001", frame="user_heading"),
        "user_heading with an anchor": lambda: h.evaluate("right", "obj_001", frame="user_heading",
                                                          anchor_id="obj_002", command=hb),
        "user_to_anchor without an anchor": lambda: v.evaluate("right", "obj_002", frame="user_to_anchor",
                                                               command=vb),
        "user_to_anchor without a command": lambda: v.evaluate("right", "obj_002", frame="user_to_anchor",
                                                               anchor_id="obj_001"),
        "object_intrinsic without an anchor": lambda: i.evaluate("right", "obj_002", frame="object_intrinsic"),
        "absent target": lambda: v.evaluate("right", "obj_999", frame="user_to_anchor", anchor_id="obj_001",
                                            command=vb),
        "absent anchor": lambda: v.evaluate("right", "obj_002", frame="user_to_anchor", anchor_id="obj_999",
                                            command=vb),
        "repeated absent ID": lambda: i.evaluate("right", "obj_999", frame="object_intrinsic", anchor_id="obj_999"),
        "command for another scene": lambda: v.evaluate("right", "obj_002", frame="user_to_anchor",
                                                        anchor_id="obj_001", command=other_scene),
        "command for another revision": lambda: v.evaluate("right", "obj_002", frame="user_to_anchor",
                                                           anchor_id="obj_001", command=other_revision),
        "pose in another frame": lambda: v.evaluate("right", "obj_002", frame="user_to_anchor", anchor_id="obj_001",
                                                    command=other_frame),
        "incompatible command for intrinsic": lambda: i.evaluate("right", "obj_003", frame="object_intrinsic",
                                                                 anchor_id="obj_001", command=vb),
        "a scene passed as a command": lambda: v.evaluate("right", "obj_002", frame="user_to_anchor",
                                                          anchor_id="obj_001", command=scenes["v"]),
        "a command passed as a scene": lambda: D.DirectionalRelations(vb, cfg),
    }
    accepted = []
    for name, fn in misuse.items():
        try:
            fn()
            accepted.append(name)
        except ValueError:
            pass
    check(f"8.2 each of {len(misuse)} kinds of API misuse raises ValueError", not accepted, f"accepted: {accepted}")

    # 3. Normalization
    r = h.evaluate("right", "obj_001", frame="user_heading", command=commands["h.drift"])
    check("8.3 heading (0, 1.0000005) is normalized: right of T=(1,0) scores 1",
          r.value is T and abs(r.requested_score_m - 1.0) <= tol, f"score {r.requested_score_m!r}")

    # 4-6. Translation, rotation and height
    def compare_variant(label, ids, scene_map, command_map, filt=None):
        evs = {k: D.DirectionalRelations(s, cfg) for k, s in scene_map.items()}
        bad, n = [], 0
        for rid in ids:
            for k, call in enumerate(by_id[rid]["calls"]):
                if filt and not filt(call):
                    continue
                n += 1
                base, moved = results[(rid, k)], run(evs, command_map, call)
                same_score = (base.requested_score_m is None) == (moved.requested_score_m is None) and (
                    base.requested_score_m is None or abs(base.requested_score_m - moved.requested_score_m) <= tol)
                if moved.value is not base.value or not same_score:
                    bad.append(f"{rid}.{k + 1}: {moved.value.value} {moved.requested_score_m}")
        records = [(f"{k} moved", s) for k, s in scene_map.items()] + [(f"{k} moved", c) for k, c in
                                                                      command_map.items()]
        errors = contract_errors(records + [("map", load_json(MAP))])
        check(label, not bad and not errors and n > 0,
              f"{n} calls" if not bad and not errors else "; ".join(bad[:3] + [e.line() for e in errors[:2]]))

    def variant(point, vector, quaternion=None):
        scene_map, command_map = {}, {}
        for key in ("h", "v", "i"):
            cmds = {k: c for k, c in commands.items() if k.startswith(key + ".")}
            s, cs = transform_records(scenes[key], cmds, point, vector, quaternion)
            scene_map[key] = s
            command_map.update(cs)
        return scene_map, command_map

    moved = ["D01", "D08", "D13", "D17", "D18"]
    compare_variant("8.4 shifting every position by (10, -7, 3) changes no value or score", moved,
                    *variant(lambda p: [p[0] + 10, p[1] - 7, p[2] + 3], lambda w: w))
    p90, v90, q90 = turn(math.pi / 2)
    exact90 = lambda p: [-p[1], p[0], p[2]]  # noqa: E731  (x, y) -> (-y, x), exactly
    compare_variant("8.5a turning everything +90 degrees about +Z, (x, y) -> (-y, x), changes no value or score",
                    moved, *variant(exact90, exact90, q90))
    p37, v37, q37 = turn(math.radians(37))
    away = lambda call: call["expect"]["value"] != "unknown"  # noqa: E731  (well away from the band)
    compare_variant("8.5b turning everything 37 degrees changes no value or score away from the band", moved,
                    *variant(p37, v37, q37), filt=away)
    lift = {"obj_001": 0.7, "obj_002": -0.4, "obj_003": 1.3, "obj_005": 2.0, "obj_006": -1.1}

    def heights():
        scene_map, command_map = {}, {}
        for key in ("h", "v", "i"):
            s = copy.deepcopy(scenes[key])
            for o in s["objects"]:
                if o["geometry"]["center_m"]["state"] == "known" and o["object_id"] in lift:
                    o["geometry"]["center_m"]["value"][2] += lift[o["object_id"]]
            scene_map[key] = s
        for k, c in commands.items():
            c = copy.deepcopy(c)
            if c["user_pose"]["position_m"]["state"] == "known":
                c["user_pose"]["position_m"]["value"][2] = 0.25
            command_map[k] = c
        return scene_map, command_map

    compare_variant("8.6 changing only heights of targets, anchors and the user changes no value or score",
                    ["D01", "D03", "D07", "D08", "D12", "D13", "D17", "D18"], *heights())

    # 7. Frame independence
    vb_all = [v.evaluate(rel, "obj_002", frame="user_to_anchor", anchor_id="obj_001", command=c).value
              for c in (vb, commands["v.east"]) for rel in OPPOSITE]
    i_turned = copy.deepcopy(scenes["i"])
    i_turned["objects"][0]["geometry"]["rotation_xyzw"]["value"] = [0.0, 0.0, 0.7071067811865476,
                                                                    0.7071067811865476]
    i_turned["objects"][0]["geometry"]["rotation_xyzw"]["support"] = "yaw_only"
    i2 = D.DirectionalRelations(i_turned, cfg)
    rot_same = all(i.evaluate(rel, t, frame="object_intrinsic", anchor_id="obj_001").value
                   is i2.evaluate(rel, t, frame="object_intrinsic", anchor_id="obj_001").value
                   for rel in OPPOSITE for t in ("obj_002", "obj_003"))
    with_pose = i.evaluate("in_front_of", "obj_002", frame="object_intrinsic", anchor_id="obj_001",
                           command=commands["i.nopose"])
    fronts = {}
    for key in ("h", "v"):
        s = copy.deepcopy(scenes[key])
        for o in s["objects"]:
            o["semantic_front"] = {"state": "known", "value": [0.0, -1.0, 0.0],
                                   "evidence": {"kind": "annotated", "source": "fx", "assumptions": []}}
        fronts[key] = D.DirectionalRelations(s, cfg)
    hv_same = all(fronts[c["scene"]].evaluate(c["relation"], c["target"], frame=c["frame"], anchor_id=c["anchor"],
                                              command=commands[c["command"]]).value is results[(rid, n)].value
                  for rid in ("D01", "D05", "D07", "D08", "D17") for n, c in enumerate(by_id[rid]["calls"]))
    check("8.7 heading doesn't affect V, box rotation doesn't affect I, a pose-less command doesn't block I, and "
          "semantic fronts affect I (D12 against D14) but not H or V",
          vb_all[:4] == vb_all[4:] and rot_same and with_pose.value is T and with_pose.command_id is None
          and all(x.record_type == "scene" for x in with_pose.inputs) and hv_same
          and results[("D12", 0)].value is T and results[("D14", 0)].value is T
          and i.evaluate("in_front_of", "obj_002", frame="object_intrinsic", anchor_id="obj_004").value is F)

    # 8. Opposites and diagonals
    bad, n = [], 0
    for (rid, k), res in results.items():
        if res.requested_score_m is None:
            continue
        call = by_id[rid]["calls"][k]
        opp = run(ev, commands, call, OPPOSITE[call["relation"]])
        n += 1
        if abs(res.requested_score_m) > cfg.direction_band_m:
            if {res.value, opp.value} != {T, F}:
                bad.append(f"{rid}.{k + 1}")
        elif res.value is not U or opp.value is not U:
            bad.append(f"{rid}.{k + 1}")
    diagonal = results[("D05", 0)].value is T and results[("D05", 1)].value is T
    check("8.8 opposites are opposite outside the band and both unknown inside it; a diagonal target is right and "
          "in front", not bad and diagonal, f"{n} pairs" if not bad else str(bad))

    # 9. Command freshness on one evaluator
    seq = [(c, rel, v.evaluate(rel, "obj_003", frame="user_to_anchor", anchor_id="obj_001", command=commands[c]).value)
           for c in ("v.base", "v.north", "v.base") for rel in ("behind", "in_front_of")]
    check("8.9 one evaluator follows each command snapshot: behind goes TRUE to FALSE and front FALSE to TRUE, and back",
          [s[2] for s in seq] == [T, F, F, T, T, F], str([(a, b, c.value) for a, b, c in seq]))

    # 10. Evidence
    r_h, r_v, r_i = results[("D01", 0)], results[("D07", 0)], results[("D12", 0)]
    fields = lambda r: [(x.record_type, x.object_id, x.field) for x in r.inputs]  # noqa: E731
    consulted = (fields(r_h) == [("scene", "obj_001", "geometry.center_m"), ("command_context", None,
                                                                            "user_pose.position_m"),
                                 ("command_context", None, "user_pose.heading_xy")]
                 and fields(r_v) == [("scene", "obj_002", "geometry.center_m"), ("scene", "obj_001",
                                                                                 "geometry.center_m"),
                                     ("command_context", None, "user_pose.position_m")]
                 and fields(r_i) == [("scene", "obj_002", "geometry.center_m"), ("scene", "obj_001",
                                                                                 "geometry.center_m"),
                                     ("scene", "obj_001", "semantic_front")])
    heading_ref = r_h.inputs[2]
    scope = D.DirectionalRelations(scenes["scope"], cfg).evaluate("right", "obj_001", frame="user_heading",
                                                                  command=commands["scope.c001"])
    scoped = scope.assumptions == frozenset({("scene", ("fixture.dir.scope", 0, "annotated"), "local_note"),
                                             ("command_context", ("fx.dir.scope.c001",), "local_note")})
    sources = {(x.record_type, x.record_id, x.source) for x in scope.inputs}
    r35, r37, r28 = results[("D35", 0)], results[("D37", 0)], results[("D28", 0)]
    front37 = [x for x in r37.inputs if x.field == "semantic_front"][0]
    centre28 = [x for x in r28.inputs if x.object_id == "obj_019"][0]
    ok = (consulted and heading_ref.heading_source == "authored" and r_h.command_id == "fx.dir.h.base"
          and r_i.command_id is None and scope.value is T and scoped and len(sources) == 2
          and front37.reason == "withheld_by_profile" and front37.source == "profile.restricted"
          and centre28.reason == "not_in_source" and r35.assumptions == frozenset() and r35.evidence == {"annotated"}
          and all(r.config == cfg.identity for r in results.values()))
    check("8.10 only consulted fields are listed, with heading source, record-scoped sources and assumptions, "
          "original unknown reasons, and no size prior", ok)

    # 11. Configuration
    raw = load_json(D.CONFIG_FILE)
    rejected = []
    with tempfile.TemporaryDirectory() as folder:
        tmp = Path(folder) / "config.json"

        def loads(text):
            tmp.write_bytes(text.encode("utf-8"))  # bytes, so line endings stay as written (Python 3.9)
            return D.load_direction_config(tmp)

        def mutated(fn):
            r2 = copy.deepcopy(raw)
            fn(r2)
            return json.dumps(r2)

        bad_texts = {
            "missing key": mutated(lambda r: r["thresholds"].pop("viewer_anchor_min_horizontal_m")),
            "extra key": mutated(lambda r: r["bands"].update(extra_band_m=0.1)),
            "negative metres": mutated(lambda r: r["bands"].update(direction_band_m=-0.01)),
            "norm cutoff of 1": mutated(lambda r: r["thresholds"].update(semantic_front_min_horizontal_norm=1)),
            "string number": mutated(lambda r: r["bands"].update(direction_band_m="0.02")),
            "NaN": json.dumps(raw).replace("0.02", "NaN", 1),
            "huge number": json.dumps(raw).replace("0.02", "1" + "0" * 400, 1),
            "repeated key": json.dumps(raw)[:-1] + ', "status": "provisional"}',
        }
        for name, text in bad_texts.items():
            try:
                loads(text)
                rejected.append(f"{name} accepted")
            except Exception:  # noqa: BLE001  (any rejection will do)
                pass
        spaced = loads(json.dumps(raw, indent=4).replace("\n", "\r\n"))
        wider = loads(mutated(lambda r: r["bands"].update(direction_band_m=0.04)))
    try:
        cfg.direction_band_m = 0.5
        frozen = False
    except dataclasses.FrozenInstanceError:
        frozen = True
    s03 = lambda c: D.DirectionalRelations(scenes["h"], c).evaluate(  # noqa: E731
        "right", "obj_006", frame="user_heading", command=hb)
    relations_cfg = P.load_config()
    check("8.11 bad configurations fail; the hash ignores layout; values can't be reassigned; a band of .04 is a "
          "new identity making .03 UNKNOWN while .02 still makes it TRUE; relations.v1 is accepted unchanged",
          not rejected and spaced.identity == cfg.identity and wider.identity != cfg.identity and frozen
          and cfg.status == "provisional" and s03(wider).value is U and s03(cfg).value is T
          and s03(wider).config == wider.identity and relations_cfg.identity == "relations.v1@d922fe902669",
          "; ".join(rejected) or f"{cfg.identity}; {wider.identity}; {relations_cfg.identity}")

    # 12. No box work
    saved = (g.Box.__init__, P.Scene.geometry, P._build)

    def refuse(*args, **kwargs):
        raise AssertionError("box geometry was built on the directional path")

    g.Box.__init__, P.Scene.geometry, P._build = refuse, refuse, refuse
    try:
        sample = [run(ev, commands, by_id[rid]["calls"][0]) for rid in ("D01", "D07", "D12", "D18", "D34")]
        no_box, detail = all(s.value is T for s in sample), "D01, D07, D12, D18 and D34 ran"
    except AssertionError as e:
        no_box, detail = False, str(e)
    finally:
        g.Box.__init__, P.Scene.geometry, P._build = saved
    check("8.12 with box construction made to raise, representative H, V and I calls still succeed", no_box, detail)

    # 13. Regression: the accepted suites, unchanged
    for script in ("test_relations.py", "test_contract.py"):
        out = subprocess.run([sys.executable, str(REPO / "grounding" / "tests" / script)], capture_output=True,
                             text=True)
        passed = out.stdout.count("\nPASS") + out.stdout.startswith("PASS")
        check(f"8.13 {script} still passes unchanged", out.returncode == 0 and "all checks passed" in out.stdout,
              f"exit {out.returncode}; {passed} checks passed")

    cross = COUNT["checks"] - cases_checks
    print(f"{len(rows)} fixed rows, {len(calls)} predicate calls in them ({cases_checks} checks); {cross} "
          f"cross-cutting checks; {COUNT['checks']} checks in all")
    print(f"{'FAILED: ' + ', '.join(FAILED) if FAILED else 'all checks passed'}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
