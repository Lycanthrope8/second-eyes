"""Tests of the A2.1d offline serializer (D69), by the implementation brief's acceptance checklist.

    python grounding/tests/test_serialization.py      (or: python -m grounding.tests.test_serialization)

Needs only Python 3.9+ and jsonschema. The goldens in fixtures/serialization/ were fixed before the serializer was
written: eight copied byte for byte from the accepted indexed audit, two literal from the brief's Appendix B. Checklist
item 16 (the contract, relation and direction suites) runs separately; see docs/serialization.md. Variants that change
geometry get their own scene ID, so the relation library's geometry cache is never asked to forget anything. Prints one
line per check and exits 1 if any fails.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))  # the repository root only, so the file also runs directly
from grounding.relations import directions as D  # noqa: E402
from grounding.relations import geometry as g  # noqa: E402
from grounding.relations import predicates as P  # noqa: E402
from grounding.serialization import (SerializationInputError, TokenCheck, TokenCheckError, codec,  # noqa: E402
                                     serialize, serializer, work_slots)

FX = REPO / "grounding" / "tests" / "fixtures"
SER = FX / "serialization"
REL = REPO / "grounding" / "relations" / "relations.v1.json"
DIR = REPO / "grounding" / "relations" / "directions.v1.json"
MAP = json.loads((FX / "contract/valid/category-map.fx.json").read_text(encoding="utf-8"))
FAILED, COUNT = [], [0]
STATE = {P.TRUE: "T", P.FALSE: "F", P.UNKNOWN: "U"}


def check(name, ok, detail=""):
    COUNT[0] += 1
    print(("PASS  " if ok else "FAIL  ") + name + (f": {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def load(rel):
    return json.loads((FX / rel).read_text(encoding="utf-8"))


def render(scene, command, fmt="coordinates_relations_v2", cap="exact", **kw):
    if cap == "exact":
        cap = work_slots(len(scene["objects"])) if fmt == "coordinates_relations_v2" else None
    kw.setdefault("relation_config_path", REL)
    kw.setdefault("direction_config_path", DIR)
    return serialize(scene, command, format=fmt, max_relation_work_units=cap, **kw)


def raises(fn, exc):
    try:
        fn()
    except exc as e:
        return e
    except Exception as e:  # noqa: BLE001
        return f"wrong exception {type(e).__name__}: {e}"
    return None


def codes(err):
    return sorted({i["code"] for i in err.issues}) if isinstance(err, SerializationInputError) else [str(err)]


def first(scene, n):
    keep = sorted(o["object_id"] for o in scene["objects"])[:n]
    out = copy.deepcopy(scene)
    out["objects"] = [o for o in out["objects"] if o["object_id"] in keep]
    return out


def renamed(scene, command, scene_id):
    """A copy under its own scene ID (a new geometry-cache identity), with a matching command."""
    s, c = copy.deepcopy(scene), copy.deepcopy(command)
    s["scene_id"] = c["scene_id"] = scene_id
    return s, c


def blocks(document):
    """The derived blocks of a rendered document, decoded by the production codec: {name: (domain, cells)}."""
    lines = [json.loads(x) for x in document.split("\n")[:-1]]
    ids = [row[0] for row in lines[2]["objects"]]
    out = {}
    for v in lines[3:-1]:
        if "relation" in v:
            name = v["relation"] if v["frame"] is None else f"{v['frame']}.{v['relation']}"
            out[name] = (v["domain"], codec.decode_block(v, len(ids)))
        elif "measure" in v:
            out["center_distance_m"] = ("unordered_pairs", codec.decode_distances(v, len(ids)))
    return ids, out


_DIRECT = {}


def direct(scene, command, name, ids, cell):
    """The accepted libraries' own answer for one cell, outside the serializer."""
    if id(scene) not in _DIRECT:
        _DIRECT[id(scene)] = (scene, P.Relations(scene), D.DirectionalRelations(scene))
    _, rel, drel = _DIRECT[id(scene)]
    frame, _, relation = name.rpartition(".")
    args = [ids[i] for i in cell]
    if not frame:
        return STATE[getattr(rel, relation)(*args).value]
    if frame == "object_intrinsic":
        return STATE[drel.evaluate(relation, args[0], frame=frame, anchor_id=args[1]).value]
    if frame == "user_heading":
        return STATE[drel.evaluate(relation, args[0], frame=frame, command=command).value]
    return STATE[drel.evaluate(relation, args[0], frame=frame, anchor_id=args[1], command=command).value]


def main() -> int:
    a17, a17r = load("contract/valid/scene.a17.annotated.json"), load("contract/valid/scene.a17.restricted.json")
    c001, c002 = load("contract/valid/command.a17.c001.json"), load("contract/valid/command.a17.c002.json")
    h, hr, hbase = load("directions/scene.h.annotated.json"), load("directions/scene.h.restricted.json"), \
        load("directions/command.h.base.json")
    expected = json.loads((SER / "expected.json").read_text(encoding="utf-8"))
    inputs = {"a17.annotated": (a17, c001), "a17.restricted": (a17r, c001), "dirh10.annotated": (first(h, 10), hbase),
              "dirh10.restricted": (first(hr, 10), hbase), "one_object.a17": (first(a17, 1), c001)}

    # 1. Goldens
    print("-- 1. goldens")
    for e in expected["goldens"]:
        data = (SER / e["golden"]).read_bytes()
        check(f"golden {e['golden']} is unchanged since it was fixed", hashlib.sha256(data).hexdigest() == e["sha256"])
    results = {}
    for case, (scene, command) in inputs.items():
        for fmt in ("coordinates_v2", "coordinates_relations_v2"):
            r = render(scene, command, fmt, category_maps=[MAP])
            want = (SER / "golden" / f"{case}.{fmt}.jsonl").read_bytes()  # bytes: no newline translation
            results[(case, fmt)] = r
            ok = (r.status == "ok" and r.document.encode("utf-8") == want
                  and r.static_prefix + r.dynamic_suffix == r.document
                  and r.document.endswith("\n") and r.dynamic_suffix.startswith('{"pose":'))
            check(f"{case} {fmt}: exact bytes of the golden; prefix + suffix = document; split before pose", ok)
    r1 = results[("one_object.a17", "coordinates_relations_v2")]
    check("Appendix B: augmented work is exactly two calls (user-heading right and front)",
          r1.metadata["work"]["planned_slots"] == 2 and r1.metadata["work"]["actual_total"] == 2
          and r1.metadata["work"]["actual_calls_by_block"]["user_heading.right"] == 1)
    mismatches = compared = 0
    for case in ("a17.annotated", "a17.restricted", "dirh10.annotated", "dirh10.restricted"):
        scene, command = inputs[case]
        ids, decoded = blocks((SER / "golden" / f"{case}.coordinates_relations_v2.jsonl").read_text(encoding="utf-8"))
        for name, (domain, cells) in decoded.items():
            if name == "center_distance_m":
                cen = {o["object_id"]: o["geometry"]["center_m"] for o in scene["objects"]}
                for (i, j), (value, _) in cells.items():
                    wa, wb = cen[ids[i]], cen[ids[j]]
                    want = g.norm(g.sub(wb["value"], wa["value"])) if wa["state"] == wb["state"] == "known" else None
                    mismatches += value != want
                    compared += 1
                continue
            for cell, (state, _) in cells.items():
                mismatches += state != direct(scene, command, name, ids, cell)
                compared += 1
    check("every decoded cell of the 4 augmented goldens (52 blocks) equals the accepted libraries' direct answer",
          mismatches == 0 and compared == 3064, f"{compared} cells compared, {mismatches} mismatches")

    # 2. Rejections
    print("-- 2. rejections")
    bad_scene = copy.deepcopy(a17)
    bad_scene["objects"][0]["geometry"]["center_m"]["value"] = [1.0, 2.0]
    answer = copy.deepcopy(c001)
    answer["target"] = "obj_003"
    other_id, other_rev, other_frame = copy.deepcopy(c001), copy.deepcopy(c001), copy.deepcopy(c001)
    other_id["scene_id"] = "fixture.other"
    other_rev["scene_revision"] = other_rev["user_pose"]["scene_revision"] = 1
    other_frame["user_pose"]["frame_id"] = "elsewhere"
    unmapped = copy.deepcopy(MAP)
    unmapped["entries"] = [e for e in unmapped["entries"] if e.get("standard") != "table"]
    other_map = copy.deepcopy(MAP)
    other_map["map_id"] = "map.other"
    with tempfile.TemporaryDirectory() as tmp:
        broken = Path(tmp) / "relations.json"
        cfg = json.loads(REL.read_text(encoding="utf-8"))
        cfg["thresholds"]["near_max_m"] = -1
        broken.write_text(json.dumps(cfg), encoding="utf-8")
        cases = [
            ("a malformed record", lambda: render(bad_scene, c001), "E_SCHEMA_LENGTH"),
            ("an answer field in the command", lambda: render(a17, answer), "E_SCHEMA_ADDITIONAL"),
            ("a command for another scene ID", lambda: render(a17, other_id, "coordinates_v2"), "E_SCENE_MISMATCH"),
            ("a command for another revision", lambda: render(a17, other_rev, "coordinates_v2"), "E_SCENE_MISMATCH"),
            ("a pose in another frame", lambda: render(a17, other_frame, "coordinates_v2"), "E_FRAME_MISMATCH"),
            ("an incompatible category map", lambda: render(a17, c001, category_maps=[unmapped]), "E_CATEGORY_UNMAPPED"),
            ("a command passed as the scene", lambda: render(c001, a17, "coordinates_v2"), "E_RECORD_ROLE"),
            ("a missing configuration file", lambda: render(a17, c001, relation_config_path=Path(tmp) / "no.json"),
             "E_CONFIG"),
            ("an invalid configuration", lambda: render(a17, c001, relation_config_path=broken), "E_CONFIG"),
            ("an unsupported format", lambda: render(a17, c001, "coordinates_v1", cap=None), "E_OPTION_FORMAT"),
            ("augmented without a cap", lambda: render(a17, c001, cap=None), "E_OPTION_WORK_CAP"),
            ("a Boolean cap", lambda: render(a17, c001, cap=True), "E_OPTION_WORK_CAP"),
            ("a negative cap", lambda: render(a17, c001, cap=-1), "E_OPTION_WORK_CAP"),
            ("a float cap", lambda: render(a17, c001, cap=342.0), "E_OPTION_WORK_CAP"),
            ("a string cap for coordinates", lambda: render(a17, c001, "coordinates_v2", cap="10"),
             "E_OPTION_WORK_CAP"),
            ("category maps not a list", lambda: render(a17, c001, category_maps=MAP), "E_OPTION_CATEGORY_MAPS"),
            ("a malformed token check", lambda: render(a17, c001, token_check=TokenCheck(len, len, -1, "x", "y",
                                                                                         "guess")),
             "E_OPTION_TOKEN_CHECK"),
        ]
        for name, fn, code in cases:
            err = raises(fn, SerializationInputError)
            check(f"rejects {name} with {code}, and no result", isinstance(err, SerializationInputError)
                  and code in codes(err), str(codes(err)) if err is not None else "accepted")
    absent = render(a17, c001)
    elsewhere = render(a17, c001, category_maps=[other_map])
    check("an absent category map renders with the existing W_NOT_CROSS_CHECKED warning kept in metadata",
          absent.status == "ok" and [w["code"] for w in absent.metadata["validation"]["warnings"]] == [
              "W_NOT_CROSS_CHECKED"])
    check("current behaviour, flagged: a map with another map ID counts as absent (warning), as the validator decides",
          elsewhere.status == "ok" and "W_NOT_CROSS_CHECKED" in [w["code"] for w in
                                                                 elsewhere.metadata["validation"]["warnings"]])

    # 3. Field projection
    print("-- 3. field projection")
    rows = {r[0]: r for r in json.loads(results[("a17.annotated", "coordinates_v2")].document.split("\n")[2])["objects"]}
    check("colours: known ['brown'], known-empty [] and unknown null stay distinct",
          rows["obj_001"][2] == ["brown"] and rows["obj_003"][2] == [] and rows["obj_005"][2] is None)
    check("an unknown category keeps its object, as null, and the raw label 'floor lamp' appears nowhere",
          rows["obj_006"][1] is None and "floor lamp" not in results[("a17.annotated", "coordinates_relations_v2")].document)
    v, vbase = load("directions/scene.v.annotated.json"), load("directions/command.v.base.json")
    rv = render(v, vbase, "coordinates_v2")
    vrows = {r[0]: r for r in json.loads(rv.document.split("\n")[2])["objects"]}
    check("a missing centre stays null in the object row", vrows["obj_007"][3] is None and vrows["obj_015"][3] is None)
    inf_scene, inf_cmd = renamed(a17, c001, "fixture.a17.rotation_assumed")
    inf_scene["assumptions"] = [{"assumption_id": "rotation_from_footprint",
                                 "description": "The table's rotation is inferred from its footprint (test variant)."}]
    inf_scene["objects"][0]["geometry"]["rotation_xyzw"]["evidence"] = {"kind": "inferred", "source": "fx",
                                                                        "assumptions": ["rotation_from_footprint"]}
    ri = render(inf_scene, inf_cmd)
    row = json.loads(ri.document.split("\n")[2])["objects"][0]
    ev = {(e["block"], tuple(e["objects"])): e for e in ri.metadata["tuple_evidence"]}
    near = ev[("near", ("obj_001", "obj_002"))]
    check("an inferred rotation with an assumption: rotation_xyzw is conditional, its support shown from the same "
          "wrapper, and box relations through it carry the scoped assumption",
          row[8] == ["rotation_xyzw"] and row[6] == "axis_aligned" and near["conditional"]
          and ["scene", ["fixture.a17.rotation_assumed", 0, "annotated"], "rotation_from_footprint"] in near["assumptions"],
          f"{row[8]}, {near['assumptions']}")
    check("...while centre-only directions and distances through the same table are not conditional",
          not ev[("user_heading.right", ("obj_001",))]["conditional"]
          and not ev[("center_distance_m", ("obj_001", "obj_002"))]["conditional"])

    # 4. Numbers and text
    print("-- 4. numbers and text")
    num_scene, num_cmd = renamed(first(a17, 3), c001, "fixture.a17.numbers")
    num_scene["objects"][0]["geometry"]["center_m"]["value"] = [1.0, -0.0, 0.5]
    num_scene["objects"][0]["geometry"]["size_m"]["value"] = [1, 2.0, 0.25]
    rn = render(num_scene, num_cmd, "coordinates_v2")
    line = rn.document.split("\n")[2]
    check("1.0 spells 1, -0.0 spells 0, and the values round-trip", '[1,0,0.5]' in line and '[1,2,0.25]' in line
          and json.loads(line)["objects"][0][3] == [1.0, -0.0, 0.5])
    d_scene, d_cmd = renamed(first(a17, 3), c001, "fixture.a17.distances")
    for o, x in zip(sorted(d_scene["objects"], key=lambda o: o["object_id"]), (0.0, 0.1006, 0.1508)):
        o["geometry"]["center_m"]["value"] = [x, 0.0, 0.0]
    rd = render(d_scene, d_cmd)
    _, dec = blocks(rd.document)
    dist = dec["center_distance_m"][1]
    check("distances 0.1006 and 0.1508 are kept exactly, not rounded to 0.101 and 0.151",
          dist[(0, 1)][0] == 0.1006 and dist[(0, 2)][0] == 0.1508 and "0.1006" in rd.document
          and "0.1508" in rd.document and "0.101," not in rd.document)
    ids10, dec10 = blocks(results[("dirh10.annotated", "coordinates_relations_v2")].document)
    uh = dec10["user_heading.right"][1]
    check("band edges: user-heading right is T at s = .03, U at .02, .01, 0 and -.01",
          [uh[(i,)][0] for i in range(5, 10)] == ["T", "U", "U", "U", "U"])
    nan_scene = copy.deepcopy(a17)
    nan_scene["objects"][0]["geometry"]["center_m"]["value"] = [float("nan"), 1.4, 0.37]
    err = raises(lambda: render(nan_scene, c001), SerializationInputError)
    check("NaN is rejected by the contract (E_NONFINITE)", isinstance(err, SerializationInputError)
          and "E_NONFINITE" in codes(err), str(codes(err)) if err else "accepted")
    check("the codec refuses to spell infinity", isinstance(raises(lambda: codec.number(float("inf")), ValueError),
                                                            ValueError))
    text = 'Say "hi"\nthen the caf\u00e9 \u2014 cafe\u0301 \u6771\u4eac \U0001F600  '
    t_cmd = copy.deepcopy(c001)
    t_cmd["text"] = text
    rt = render(a17, t_cmd, "coordinates_v2")
    back = json.loads(rt.document.split("\n")[-2])["command"]
    check("command text round-trips exactly: quotes, a real newline, non-ASCII, decomposed Unicode, trailing spaces",
          back == text and "cafe\u0301" in back and "\\n" in rt.document.split("\n")[-2])

    # 5. Determinism and no mutation
    print("-- 5. determinism")
    shuffled = copy.deepcopy(a17)
    shuffled["objects"].reverse()
    for o in shuffled["objects"]:
        if o["attributes"]["colours"]["state"] == "known":
            o["attributes"]["colours"]["value"].reverse()
    check("reversed object order and colour order give identical bytes",
          render(shuffled, c001).document == results[("a17.annotated", "coordinates_relations_v2")].document)
    o_scene, o_cmd = renamed(first(a17, 3), c001, "fixture.a17.ids")
    for o, new in zip(sorted(o_scene["objects"], key=lambda o: o["object_id"]), ("obj_001", "obj_999", "obj_1000")):
        o["object_id"] = new
    ro = render(o_scene, o_cmd)
    ids_o, dec_o = blocks(ro.document)
    hdec = dec_o["user_heading.right"][1]
    check("obj_001, obj_1000, obj_999 follow lexicographic array order, and indices read through that order",
          ids_o == ["obj_001", "obj_1000", "obj_999"]
          and all(hdec[(i,)][0] == direct(o_scene, o_cmd, "user_heading.right", ids_o, (i,)) for i in range(3)))
    meta_scene, meta_cmd = copy.deepcopy(a17), copy.deepcopy(c001)
    meta_scene["sources"][0]["description"] = "a different description"
    meta_scene["sources"][0]["release"] = "fixture-v9"
    meta_scene["objects"][0]["source_ref"]["source_label"] = "dining table"
    meta_scene["objects"][0]["source_ref"]["source_object_id"] = "other_id"
    meta_cmd["command_id"] = "fx.a17.renamed"
    meta_cmd["sources"][0]["description"] = "another description"
    check("metadata-only changes (source descriptions, releases, source labels and IDs, command ID) leave the model "
          "bytes unchanged", render(meta_scene, meta_cmd).document
          == results[("a17.annotated", "coordinates_relations_v2")].document)
    before = [json.dumps(x, sort_keys=True) for x in (a17, c001, MAP)]
    render(a17, c001, category_maps=[MAP])
    check("no input record is mutated", before == [json.dumps(x, sort_keys=True) for x in (a17, c001, MAP)])

    # 6. Profile isolation, in both rendering orders, each in a fresh interpreter
    print("-- 6. profile isolation")
    probe = (
        "import json,sys; sys.path.insert(0, sys.argv[1]); from pathlib import Path\n"
        "from grounding.serialization import serialize, work_slots\n"
        "FX = Path(sys.argv[1]) / 'grounding/tests/fixtures'\n"
        "G = FX / 'serialization/golden'\n"
        "def load(p): return json.loads((FX / p).read_text(encoding='utf-8'))\n"
        "def first(s, n):\n"
        "    keep = sorted(o['object_id'] for o in s['objects'])[:n]; s['objects'] = [o for o in s['objects'] "
        "if o['object_id'] in keep]; return s\n"
        "cases = {'a17.annotated': (load('contract/valid/scene.a17.annotated.json'), "
        "load('contract/valid/command.a17.c001.json')), 'a17.restricted': (load('contract/valid/scene.a17.restricted"
        ".json'), load('contract/valid/command.a17.c001.json')), 'dirh10.annotated': (first(load('directions/scene.h"
        ".annotated.json'), 10), load('directions/command.h.base.json')), 'dirh10.restricted': (first(load('directions"
        "/scene.h.restricted.json'), 10), load('directions/command.h.base.json'))}\n"
        "ok = []\n"
        "for case in sys.argv[2].split(','):\n"
        "    s, c = cases[case]\n"
        "    r = serialize(s, c, format='coordinates_relations_v2', relation_config_path=Path(sys.argv[1]) / "
        "'grounding/relations/relations.v1.json', direction_config_path=Path(sys.argv[1]) / "
        "'grounding/relations/directions.v1.json', max_relation_work_units=work_slots(len(s['objects'])))\n"
        "    ok.append(r.document.encode('utf-8') == (G / (case + '.coordinates_relations_v2.jsonl')).read_bytes())\n"
        "print(json.dumps(ok))\n")
    for order in ("a17.annotated,a17.restricted,dirh10.annotated,dirh10.restricted",
                  "dirh10.restricted,dirh10.annotated,a17.restricted,a17.annotated"):
        out = subprocess.run([sys.executable, "-c", probe, str(REPO), order], capture_output=True, text=True,
                             cwd=str(REPO))
        got = json.loads(out.stdout) if out.returncode == 0 else out.stderr[-200:]
        check(f"profile isolation, order {order.split(',')[0]} first: all four augmented goldens reproduced",
              got == [True] * 4, str(got))
    _, dec_r = blocks(results[("a17.restricted", "coordinates_relations_v2")].document)
    btw = {s for s, _ in dec_r["between"][1].values()}
    check("restricted predicates are not all UNKNOWN: decisive centre conditions make between FALSE", "F" in btw,
          str(btw))

    # 7. Configuration consistency
    print("-- 7. configuration")
    with tempfile.TemporaryDirectory() as tmp:
        cfg = json.loads(REL.read_text(encoding="utf-8"))
        cfg["thresholds"]["near_max_m"] = 0.9
        cfg["categories"]["container"] = ["box", "chair"]
        changed = Path(tmp) / "relations.json"
        changed.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        dcfg = json.loads(DIR.read_text(encoding="utf-8"))
        dcfg["bands"]["direction_band_m"] = 0.5
        dchanged = Path(tmp) / "directions.json"
        dchanged.write_text(json.dumps(dcfg), encoding="utf-8")
        rc = render(a17, c001, relation_config_path=changed, direction_config_path=dchanged)
        rel_cfg, dir_cfg = P.load_config(changed), D.load_direction_config(dchanged)
        sem = json.loads(rc.document.split("\n")[1])["semantics"]
        ids_c, dec_c = blocks(rc.document)
        rel, drel = P.Relations(a17, rel_cfg), D.DirectionalRelations(a17, dir_cfg)
        near_ok = all(s == STATE[rel.near(ids_c[i], ids_c[j]).value] for (i, j), (s, _) in dec_c["near"][1].items())
        uh_ok = all(s == STATE[drel.evaluate("right", ids_c[t], frame="user_heading", command=c001).value]
                    for (t,), (s, _) in dec_c["user_heading.right"][1].items())
        check("changed configuration values appear in the semantics line", sem["parameters"]["thresholds"]["near_max_m"]
              == 0.9 and sem["categories"]["container"] == ["box", "chair"]
              and sem["parameters"]["directions"]["direction_band_m"] == 0.5)
        check("...and in the computation (near and user-heading right equal the libraries under the same files)",
              near_ok and uh_ok and rc.document != results[("a17.annotated", "coordinates_relations_v2")].document)
        check("...and in metadata, with the new identities rather than stale ones",
              rc.metadata["configs"]["relation_config"]["identity"] == rel_cfg.identity != P.load_config(REL).identity
              and rc.metadata["configs"]["direction_config"]["identity"] == dir_cfg.identity)
    for case in ("a17.annotated", "dirh10.restricted"):
        a = results[(case, "coordinates_v2")].document.split("\n")
        b = results[(case, "coordinates_relations_v2")].document.split("\n")
        ha, hb = json.loads(a[0])["header"], json.loads(b[0])["header"]
        pose_b = next(x for x in b if x.startswith('{"pose":'))
        check(f"{case}: both formats share header (but format), semantics, objects, pose and command",
              {k: x for k, x in ha.items() if k != "format"} == {k: x for k, x in hb.items() if k != "format"}
              and a[1:3] == b[1:3] and a[3] == pose_b and a[-2] == b[-2])

    # 8. Domains, work counts and the codec
    print("-- 8. domains and codec")
    for n in (0, 1, 2, 3, 6, 10):
        sc = first(h, n)
        r = render(sc, hbase)
        ids_n, dec_n = blocks(r.document)
        counts_ok = all(len(cells) == codec.legal_count(domain, n) for domain, cells in dec_n.values())
        empty_ok = all(json.loads(x).get("states", []) == [] and json.loads(x)["conditional"] is False
                       for x in r.document.split("\n")[3:-1] if '"domain":' in x
                       and codec.legal_count(json.loads(x)["domain"], n) == 0)
        check(f"n = {n}: planned = actual = {work_slots(n)} work slots, every domain complete, empty domains [] / false",
              r.metadata["work"]["planned_slots"] == r.metadata["work"]["actual_total"] == work_slots(n) and counts_ok
              and empty_ok and len(dec_n) == 13)
    check("work slots at n = 0, 1, 2, 3, 6, 10 are 0, 2, 22, 63, 342, 1190",
          [work_slots(n) for n in (0, 1, 2, 3, 6, 10)] == [0, 2, 22, 63, 342, 1190])
    cc = json.loads((SER / "codec_cases.json").read_text(encoding="utf-8"))
    for case in cc["blocks"]:
        block = json.loads(case["text"])
        want = {tuple(c[0]): (c[1], c[2]) for c in case["cells"]}
        got = codec.decode_block(block, case["n"])
        again = codec.enc(codec.encode_block(block["relation"], block["frame"], block["domain"], case["n"], got))
        check(f"literal {block['domain']} example decodes to the stated mapping and re-encodes to its text",
              got == want and again == case["text"])
    want = {tuple(c[0]): (c[1], c[2]) for c in cc["distances"]["cells"]}
    got = codec.decode_distances(json.loads(cc["distances"]["text"]), 3)
    check("literal distances: 0.1006, 0.1508 and null decode exactly and re-encode to the text",
          got == want and codec.enc(codec.encode_distances(3, got)) == cc["distances"]["text"])
    unit_bad = unit_count = 0
    for domain in codec.DOMAINS:
        for n in (0, 1, 2, 3, 6, 10):
            cells = codec.legal_cells(domain, n)
            for pattern in ("T", "F", "U", "mixed"):
                for member in ("false", "true", "mixed"):
                    m = {c: ("TFU"[(k * 7 + len(c)) % 3] if pattern == "mixed" else pattern,
                             (k % 2 == 0) if member == "mixed" else member == "true") for k, c in enumerate(cells)}
                    text = codec.enc(codec.encode_block("r", None, domain, n, m))
                    unit_bad += codec.decode_block(json.loads(text), n) != m
                    unit_bad += codec.enc(codec.encode_block("r", None, domain, n, codec.decode_block(json.loads(text),
                                                                                                      n))) != text
                    unit_count += 1
    for n in (0, 1, 2, 3, 6, 10):
        for member in ("false", "true", "mixed"):
            m = {c: (None if k % 5 == 3 else 0.1006 + 0.0502 * k, (k % 2 == 0) if member == "mixed" else member == "true")
                 for k, c in enumerate(codec.legal_cells("unordered_pairs", n))}
            text = codec.enc(codec.encode_distances(n, m))
            unit_bad += codec.decode_distances(json.loads(text), n) != m
            unit_bad += codec.enc(codec.encode_distances(n, codec.decode_distances(json.loads(text), n))) != text
            unit_count += 1
    check("codec unit cases: every domain and the distances at n = 0, 1, 2, 3, 6, 10, uniform and mixed truth and "
          "membership, round-trip canonically", unit_bad == 0 and unit_count == 306,
          f"{unit_count} cases, {unit_bad} failures")
    def blk(relation, frame, domain, states, conditional, **extra):
        return dict({"relation": relation, "frame": frame, "domain": domain, "states": states,
                     "conditional": conditional}, **extra)

    def btw(rows):
        return blk("between", None, "target_anchor_pairs", rows, False)

    ua = ("right", "user_to_anchor", "ordered_pairs")
    rejected = {  # name: (block, n, the reason the error must give)
        "full rows for a uniform domain": (blk(*ua, ["-TT", "T-T", "TT-"], False), 3, "not canonical"),
        "full 0/1 rows for a uniform membership": (blk(*ua, "T", ["-11", "1-1", "11-"]), 3, "not full rows"),
        "[] as an empty domain's conditional": (blk("near", None, "unordered_pairs", [], []), 1,
                                                "conditional must be false"),
        "true as an empty domain's conditional": (blk("near", None, "unordered_pairs", [], True), 1,
                                                  "conditional must be false"),
        "a uniform state for an empty domain": (blk("near", None, "unordered_pairs", "U", False), 1,
                                                "states must be []"),
        "a uniform conditional as a string": (blk(*ua, "T", "1"), 3, "must be a JSON Boolean"),
        "a one-object objects membership as a string": (blk("right", "user_heading", "objects", "T", "1"), 1,
                                                        "must be a JSON Boolean"),
        "wrong row length": (blk(*ua, ["-T", "F-T", "UF-"], False), 3, "row 0 must be a string of 3"),
        "wrong diagonal": (blk(*ua, ["TTU", "F-T", "UF-"], False), 3, "diagonal cell (0,0)"),
        "illegal symbol": (blk(*ua, ["-TX", "F-T", "UF-"], False), 3, "symbol 'X'"),
        "duplicate between row": (btw([[0, 1, "--T"], [0, 1, "--T"], [1, 2, "F--"]]), 3, "duplicate or missing"),
        "missing between row": (btw([[0, 1, "--T"], [1, 2, "F--"]]), 3, "2 anchor rows where 3 are due"),
        "between rows out of order": (btw([[0, 2, "-U-"], [0, 1, "--T"], [1, 2, "F--"]]), 3, "duplicate or missing"),
        "a Boolean between index": (btw([[False, 1, "--T"], [0, 2, "-U-"], [1, 2, "F--"]]), 3, "must be integers"),
        "a between index out of range": (btw([[0, 1, "--T"], [0, 3, "-U-"], [1, 2, "F--"]]), 3, "out of range"),
        "an anchor position not '-'": (btw([[0, 1, "T-T"], [0, 2, "-U-"], [1, 2, "F--"]]), 3, "not '-'"),
        "a target marked '-'": (btw([[0, 1, "---"], [0, 2, "-U-"], [1, 2, "F--"]]), 3, "symbol '-'"),
        "an unknown field": (blk(*ua, "T", False, extra=1), 3, "relation block fields"),
    }
    for name, (block, n, reason) in rejected.items():
        err = raises(lambda b=block, k=n: codec.decode_block(b, k), codec.CodecError)
        check(f"codec rejects {name}, for that reason", isinstance(err, codec.CodecError) and reason in str(err),
              str(err))
    for name, values, reason in (("a short distance row", [[0.1006], [None]], "row 0 must hold 2"),
                                 ("a Boolean distance", [[True, 0.2], [None]], "not a finite number")):
        err = raises(lambda v=values: codec.decode_distances({"measure": "center_distance_m",
                                                               "domain": "unordered_pairs", "values": v,
                                                               "conditional": False}, 3), codec.CodecError)
        check(f"codec rejects {name}, for that reason", isinstance(err, codec.CodecError) and reason in str(err),
              str(err))
    lit = codec.decode_block(json.loads(cc["blocks"][1]["text"]), 3)
    check("ordered pairs are not symmetrized and the diagonal has no cell: right(A,B)=T, right(B,A)=F, (A,A) absent",
          lit[(0, 1)][0] == "T" and lit[(1, 0)][0] == "F" and (0, 0) not in lit)

    # 9. Consulted assumptions
    print("-- 9. assumptions")
    lines = [json.loads(x) for x in results[("a17.restricted", "coordinates_relations_v2")].document.split("\n")[:-1]]
    cond = {(x.get("frame"), x.get("relation", x.get("measure"))): x["conditional"] for x in lines
            if "conditional" in x and ("relation" in x or "measure" in x)}
    check("restricted a17: near, above and below consult the assumed sizes (conditional true)",
          cond[(None, "near")] is True and cond[(None, "above")] is True and cond[(None, "below")] is True)
    check("...while directions and distances don't inherit the unused size assumption",
          all(cond[k] is False for k in cond if k[0] in ("user_heading", "user_to_anchor", "object_intrinsic")
              or k[1] == "center_distance_m"))
    sc_scope, cmd_scope = load("directions/scene.scope.annotated.json"), load("directions/command.scope.c001.json")
    rs = render(sc_scope, cmd_scope)
    e = [x for x in rs.metadata["tuple_evidence"] if x["block"] == "user_heading.right"][0]
    check("record-scoped assumptions: the scene's and the command's local_note stay distinct",
          e["conditional"] and ["scene", ["fixture.dir.scope", 0, "annotated"], "local_note"] in e["assumptions"]
          and ["command_context", ["fx.dir.scope.c001"], "local_note"] in e["assumptions"])

    # 10. Frames and mirrors
    print("-- 10. frames and mirrors")
    vnorth, vat = load("directions/command.v.north.json"), load("directions/command.v.at_anchor.json")
    i_scene, i_nopose = load("directions/scene.i.annotated.json"), load("directions/command.i.nopose.json")
    renders = {"v.base": (v, vbase), "v.north": (v, vnorth), "v.at_anchor": (v, vat), "i.nopose": (i_scene, i_nopose)}
    dec_all = {}
    for key, (scene, command) in renders.items():
        r = render(scene, command)
        dec_all[key] = blocks(r.document)
    ids_v, dv = dec_all["v.base"]
    ix = {oid: k for k, oid in enumerate(ids_v)}

    def cell(key, name, t, a=None):
        ids_k, dk = dec_all[key]
        k = {oid: n for n, oid in enumerate(ids_k)}
        return dk[name][1][(k[t],) if a is None else (k[t], k[a])][0]

    m = codec.mirror
    _, dn = dec_all["v.north"]
    facts = {
        "D07 right(obj_002, obj_001) T, so left F": cell("v.base", "user_to_anchor.right", "obj_002", "obj_001") == "T"
        and m(dv["user_to_anchor.right"][1])[(ix["obj_002"], ix["obj_001"])][0] == "F",
        "D10 from the other side: right F, so left T": cell("v.north", "user_to_anchor.right", "obj_002", "obj_001") == "F"
        and m(dn["user_to_anchor.right"][1])[(ix["obj_002"], ix["obj_001"])][0] == "T",
        "D08 in_front_of(obj_003, obj_001) F, so behind T": cell("v.base", "user_to_anchor.in_front_of", "obj_003",
                                                                 "obj_001") == "F",
        "D09 from the other side: in_front_of T": cell("v.north", "user_to_anchor.in_front_of", "obj_003",
                                                       "obj_001") == "T",
        "D26 coincident target: U": cell("v.base", "user_to_anchor.right", "obj_004", "obj_001") == "U",
        "D31 anchor with no centre: U": cell("v.base", "user_to_anchor.right", "obj_002", "obj_007") == "U",
        "D39 viewer-anchor 0.02 m (degenerate): U": cell("v.base", "user_to_anchor.right", "obj_009", "obj_008") == "U",
        "D40 viewer-anchor 0.03 m: T": cell("v.base", "user_to_anchor.right", "obj_011", "obj_010") == "T",
        "D38 user above the anchor: U": cell("v.at_anchor", "user_to_anchor.right", "obj_002", "obj_001") == "U",
        "D12 intrinsic in_front_of(obj_002, obj_001) T": cell("i.nopose", "object_intrinsic.in_front_of", "obj_002",
                                                             "obj_001") == "T",
        "D14 intrinsic in_front_of(obj_002, obj_004) F (behind T)": cell("i.nopose", "object_intrinsic.in_front_of",
                                                                         "obj_002", "obj_004") == "F",
        "D33 missing front: U": cell("i.nopose", "object_intrinsic.in_front_of", "obj_002", "obj_012") == "U",
        "D41 vertical front: U": cell("i.nopose", "object_intrinsic.in_front_of", "obj_008", "obj_007") == "U",
    }
    for name, ok in facts.items():
        check(f"frame case {name}", ok)
    _, di = dec_all["i.nopose"]
    check("with no pose, every user-heading and user-to-anchor cell is a covered UNKNOWN",
          all(s == "U" for name in ("user_heading.right", "user_heading.in_front_of", "user_to_anchor.right",
                                    "user_to_anchor.in_front_of") for s, _ in di[name][1].values()))
    bad = 0
    for key, (scene, command) in renders.items():
        ids_k, dk = dec_all[key]
        drel = D.DirectionalRelations(scene)
        for frame in ("object_intrinsic", "user_heading", "user_to_anchor"):
            for stored, other in (("right", "left"), ("in_front_of", "behind")):
                for c, (s, _) in m(dk[f"{frame}.{stored}"][1]).items():
                    kw = {"frame": frame}
                    if len(c) == 2:
                        kw["anchor_id"] = ids_k[c[1]]
                    if frame != "object_intrinsic":
                        kw["command"] = command
                    bad += s != STATE[drel.evaluate(other, ids_k[c[0]], **kw).value]
    check("mirrored right/front equal the libraries' own left/behind in every legal cell of four renders "
          "(verification calls only)", bad == 0, f"{bad} differences")

    # 11. Static and dynamic parts
    print("-- 11. static and dynamic")
    base_r = results[("a17.annotated", "coordinates_relations_v2")]
    t_only = copy.deepcopy(c001)
    t_only["text"] = "Inspect the chair on my left."
    moved = copy.deepcopy(c001)
    moved["user_pose"]["position_m"]["value"] = [3.5, 1.0, 1.62]
    r_t, r_m, r_c2 = render(a17, t_only), render(a17, moved), render(a17, c002)
    check("changing the command text changes only the command line", r_t.static_prefix == base_r.static_prefix
          and r_t.dynamic_suffix.split("\n")[:-2] == base_r.dynamic_suffix.split("\n")[:-2])
    check("changing the pose leaves the static prefix and changes the viewer-relative tables",
          r_m.static_prefix == base_r.static_prefix and r_m.dynamic_suffix != base_r.dynamic_suffix
          and r_m.dynamic_suffix.split("\n")[3] != base_r.dynamic_suffix.split("\n")[3])
    check("a pose-less command (c002) keeps the static prefix", r_c2.static_prefix == base_r.static_prefix)
    g_scene, g_cmd = renamed(a17, c001, "fixture.a17.moved")
    g_scene["objects"][0]["geometry"]["center_m"]["value"] = [2.0, 1.4, 0.37]
    check("a model-visible scene change updates the static prefix",
          render(g_scene, g_cmd).static_prefix != base_r.static_prefix)
    check("command ID, scene ID and timings never enter the model text",
          "fx.a17.c001" not in base_r.document and "fixture.a17" not in base_r.document and "_s\"" not in base_r.document)

    # 12. No ranking; distances; between's domain
    print("-- 12. ranking and distances")
    orig_rank = P.Relations.rank

    def refuse(*a, **k):
        raise AssertionError("rank was called")

    P.Relations.rank = refuse
    try:
        ok = render(a17, c001).status == "ok"
    except AssertionError:
        ok = False
    finally:
        P.Relations.rank = orig_rank
    check("ranking is never called while rendering", ok)
    clash = copy.deepcopy(a17)  # same scene key as the cached a17 renders, different geometry
    clash["objects"][0]["geometry"]["center_m"]["value"] = [2.0, 1.4, 0.37]
    err = raises(lambda: render(clash, c001), ValueError)
    check("a conflicting snapshot under a cached scene key fails with the library's own error: not swallowed, not an "
          "input error, the cache not cleared", isinstance(err, ValueError)
          and not isinstance(err, SerializationInputError) and "was reused for different" in str(err)
          and render(a17, c001).document == base_r.document, str(err)[:120])
    dd = dv["center_distance_m"][1]
    check("distances: a true zero (coincident centres) spells 0 and stays distinct from null (missing centres)",
          dd[tuple(sorted((ix["obj_001"], ix["obj_004"])))][0] == 0
          and dd[tuple(sorted((ix["obj_002"], ix["obj_007"])))][0] is None)
    vdist = render(v, vbase).metadata["work"]["distances"]
    check("distance slots count nulls; calculations count only known pairs", vdist["slots"] == 105
          and vdist["performed"] + vdist["missing"] == 105 and vdist["missing"] == 27, str(vdist))
    check("between covers its complete target/unordered-anchor domain",
          len(dv["between"][1]) == codec.legal_count("target_anchor_pairs", len(ids_v)))

    # 13. Work cap and zero work for coordinates
    print("-- 13. work cap")
    saved = (P.Relations.__init__, D.DirectionalRelations.__init__, g.Box.__init__, P.Scene.geometry,
             serializer.codec.legal_cells)

    def boom(*a, **k):
        raise AssertionError("relation work before the cap check")

    P.Relations.__init__ = D.DirectionalRelations.__init__ = g.Box.__init__ = P.Scene.geometry = boom
    serializer.codec.legal_cells = boom
    try:
        over = render(a17, c001, cap=341)
        coords = render(a17, c001, "coordinates_v2")
        guarded = over.status == "work_budget_exceeded" and over.document is None and over.static_prefix is None \
            and over.metadata["work"]["planned_slots"] == 342 and over.metadata["work"]["cap"] == 341
        coord_ok = coords.status == "ok" and coords.metadata["work"]["planned_slots"] == 0
    except AssertionError as e:
        guarded = coord_ok = False
        print("   ", e)
    finally:
        (P.Relations.__init__, D.DirectionalRelations.__init__, g.Box.__init__, P.Scene.geometry,
         serializer.codec.legal_cells) = saved
    check("a cap one below the work fails before any evaluator, box, geometry lookup or tuple enumeration", guarded)
    check("coordinates_v2 does no relation or distance work at all (fail-on-call instrumentation)", coord_ok)
    check("a cap equal to the work succeeds", render(a17, c001, cap=342).status == "ok")

    # 14. Token check
    print("-- 14. token check")
    seen = []

    def wrap(doc):
        return "<|start|>" + doc + "<|end|>"

    def whitespace_counter_test_double(text):
        seen.append(text)
        return len(text.split())

    need = whitespace_counter_test_double(wrap(base_r.document))
    seen.clear()

    def tc(limit, w=wrap, cnt=whitespace_counter_test_double, kind="test_double"):
        return TokenCheck(w, cnt, limit, "whitespace counter (test double)", "test wrapper v1", kind)

    none_r = base_r
    at = render(a17, c001, token_check=tc(need))
    once = len(seen) == 1 and seen[0] == wrap(at.document)
    above = render(a17, c001, token_check=tc(need - 1))
    check("no token check: token_count null, status not_checked", none_r.metadata["token"] == {
        "token_count": None, "token_budget_status": "not_checked"})
    check("a limit equal to the count passes, counting the complete wrapped input in one call",
          at.status == "ok" and at.metadata["token"]["token_count"] == need and once)
    check("one token over: token_budget_exceeded with the count and limit, and no document, prefix or suffix",
          above.status == "token_budget_exceeded" and above.document is None and above.static_prefix is None
          and above.metadata["token"]["token_count"] == need and above.metadata["token"]["max_input_tokens"] == need - 1)
    check("a test double is labelled as one", at.metadata["token"]["measurement_kind"] == "test_double"
          and "test double" in at.metadata["token"]["note"])
    for name, t in (("wrap returning bytes", tc(10 ** 6, w=lambda d: d.encode())),
                    ("a Boolean count", tc(10 ** 6, cnt=lambda s: True)),
                    ("a negative count", tc(10 ** 6, cnt=lambda s: -1)),
                    ("a float count", tc(10 ** 6, cnt=lambda s: 1.5)),
                    ("a failing counter", tc(10 ** 6, cnt=lambda s: 1 // 0))):
        err = raises(lambda t=t: render(a17, c001, token_check=t), TokenCheckError)
        check(f"token check fails clearly on {name}", isinstance(err, TokenCheckError), str(err))

    # 15. Imports and interfaces
    print("-- 15. imports")
    probe = ("import sys, json; sys.path.insert(0, sys.argv[1])\n"
             "import grounding.serialization as S\n"
             "from grounding.serialization import serializer\n"
             "import grounding.relations.predicates as P, grounding.relations.directions as D\n"
             "bad = [m for m in sys.modules if m.startswith(('analysis', 'grounding.tests')) or m in ('predicates', "
             "'directions', 'geometry', 'test_serialization')]\n"
             "print(json.dumps({'same_P': serializer.P is P and D.P is P, 'same_truth': serializer.P.TRUE is P.TRUE, "
             "'bad': bad, 'relations_dir_on_path': any(p.rstrip('/\\\\').endswith('relations') for p in sys.path)}))\n")
    out = subprocess.run([sys.executable, "-c", probe, str(REPO)], capture_output=True, text=True, cwd=str(REPO))
    rep = json.loads(out.stdout) if out.returncode == 0 else {"error": out.stderr[-300:]}
    check("a fresh interpreter imports grounding.serialization with one predicates module and one Truth, nothing from "
          "analysis/ or tests, and no relations folder on the path", rep.get("same_P") and rep.get("same_truth")
          and rep.get("bad") == [] and rep.get("relations_dir_on_path") is False, str(rep))
    import re
    imports = [ln for f in (REPO / "grounding" / "serialization").glob("*.py")
               for ln in f.read_text(encoding="utf-8").splitlines() if re.match(r"\s*(from|import)\s", ln)]
    check("production code imports nothing from analysis/ or tests",
          not any(re.search(r"analysis|tests|test_", ln) for ln in imports), "; ".join(imports[:3]))
    sig = (list(inspect.signature(D.DirectionalRelations.evaluate).parameters),
           list(inspect.signature(P.Relations.between).parameters))
    check("the library interfaces the serializer calls are unchanged", sig == (
        ["self", "relation", "target_id", "frame", "anchor_id", "command"], ["self", "t", "a", "b"]), str(sig))

    # CLI
    print("-- CLI")
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "out"
        args = [sys.executable, "-m", "grounding.serialization", "--scene",
                str(FX / "contract/valid/scene.a17.annotated.json"), "--command",
                str(FX / "contract/valid/command.a17.c001.json"), "--format", "coordinates_relations_v2",
                "--relation-config", str(REL), "--direction-config", str(DIR), "--category-map",
                str(FX / "contract/valid/category-map.fx.json"), "--out", str(out_dir)]
        ok = subprocess.run(args + ["--max-relation-work-units", "342"], capture_output=True, text=True, cwd=str(REPO))
        files = sorted(p.name for p in out_dir.iterdir()) if out_dir.exists() else []
        golden = (SER / "golden/a17.annotated.coordinates_relations_v2.jsonl").read_bytes()
        check("CLI success: exit 0, four files, document = golden, metadata not_checked", ok.returncode == 0
              and files == ["document.jsonl", "dynamic_suffix.jsonl", "metadata.json", "static_prefix.jsonl"]
              and (out_dir / "document.jsonl").read_bytes() == golden
              and json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))["token"]["token_budget_status"]
              == "not_checked", ok.stderr[-200:])
        again = subprocess.run(args + ["--max-relation-work-units", "342"], capture_output=True, text=True,
                               cwd=str(REPO))
        check("CLI refuses to overwrite existing outputs (exit 2), leaving them as they were",
              again.returncode == 2 and (out_dir / "document.jsonl").read_bytes() == golden)
        small = Path(tmp) / "small"
        r_small = subprocess.run(args[:-1] + [str(small), "--max-relation-work-units", "10"], capture_output=True,
                                 text=True, cwd=str(REPO))
        check("CLI work-budget failure: exit 1 and only metadata.json", r_small.returncode == 1
              and sorted(p.name for p in small.iterdir()) == ["metadata.json"])
        dup = Path(tmp) / "dup.json"
        dup.write_text('{"record_type": "scene", "record_type": "scene"}', encoding="utf-8")
        bad_dir = Path(tmp) / "bad"
        r_bad = subprocess.run([args[0], "-m", "grounding.serialization", "--scene", str(dup)] + args[5:-1]
                               + [str(bad_dir), "--max-relation-work-units", "342"], capture_output=True, text=True,
                               cwd=str(REPO))
        check("CLI strict parsing: a repeated key exits 2 with its code and writes no files", r_bad.returncode == 2
              and "E_PARSE_DUPLICATE_KEY" in r_bad.stderr and not bad_dir.exists())
        nocap = Path(tmp) / "nocap"
        r_nocap = subprocess.run(args[:-1] + [str(nocap)], capture_output=True, text=True, cwd=str(REPO))
        check("CLI augmented without a cap exits 2 and writes no files", r_nocap.returncode == 2 and not nocap.exists()
              and "E_OPTION_WORK_CAP" in r_nocap.stderr)

    print(f"{COUNT[0]} checks; {'FAILED: ' + ', '.join(FAILED) if FAILED else 'all checks passed'}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
