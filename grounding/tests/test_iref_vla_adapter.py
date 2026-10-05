"""Tests of the A2.2a IRef-VLA metadata adapter (D74).

    python grounding/tests/test_iref_vla_adapter.py --sample DIR      (or: python -m grounding.tests.test_iref_vla_adapter)

DIR holds the five pinned IRef-VLA files under their own names (see docs/iref-vla-adapter.md). The environment variable
SECOND_EYES_IREF_VLA_SAMPLE may name it instead. Without them the pinned-sample checks are reported as one FAIL, never
skipped: acceptance can't pass without its data. Everything else uses the hand-written source-format fixtures in
fixtures/iref_vla/. Expectations come from fixtures/iref_vla/expectations.json, fixed before the adapter existed.
Needs only Python 3.9+ and jsonschema. Prints one line per check and exits 1 if any fails.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from grounding.contract import validate as contract  # noqa: E402

FX = REPO / "grounding" / "tests" / "fixtures" / "iref_vla"
EXP = json.loads((FX / "expectations.json").read_text(encoding="utf-8"))
REL_CFG = REPO / "grounding" / "relations" / "relations.v1.json"
DIR_CFG = REPO / "grounding" / "relations" / "directions.v1.json"
ANNOTATION_KEYS = {"target_index", "target_class", "target_position", "target_colors", "target_size", "target_color_used",
                   "target_size_used", "distractor_ids", "relation", "relation_type", "anchors", "false_statements",
                   "index", "class", "position", "color", "size", "color_used", "size_used", "false_target_color",
                   "false_target_class", "false_anchors", "false_anchor_color", "false_anchor_class"}
FAILED, COUNT = [], [0]


def check(name, ok, detail=""):
    COUNT[0] += 1
    print(("PASS  " if ok else "FAIL  ") + name + (f": {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def mini(name):
    return (FX / f"mini_{name}").read_bytes()


def keys_of(x):
    out = set()
    if isinstance(x, dict):
        for k, v in x.items():
            out.add(k)
            out |= keys_of(v)
    elif isinstance(x, list):
        for v in x:
            out |= keys_of(v)
    return out


def codes(err):
    return sorted({i["code"] for i in getattr(err, "issues", [])}) or [f"{type(err).__name__}: {err}"]


def rotate(q, v):
    """Test-owned: rotate v by the unit quaternion q = [x, y, z, w] through its rotation matrix."""
    x, y, z, w = q
    m = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    return [sum(m[i][j] * v[j] for j in range(3)) for i in range(3)]


def corners(obj):
    c, s, q = (obj["geometry"][k]["value"] for k in ("center_m", "size_m", "rotation_xyzw"))
    return [[c[i] + r for i, r in enumerate(rotate(q, [sx * s[0] / 2, sy * s[1] / 2, sz * s[2] / 2]))]
            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]


def same_corners(a, b, tol):
    """Unordered 8-point sets agree within tol: a one-to-one nearest match both ways."""
    used = set()
    for p in a:
        j = min((k for k in range(len(b)) if k not in used), key=lambda k: math.dist(p, b[k]), default=None)
        if j is None or math.dist(p, b[j]) > tol:
            return False
        used.add(j)
    return len(used) == len(b) == len(a)


def by_id(scene):
    return {o["object_id"]: o for o in scene["objects"]}


# ------------------------------------------------------------------------------------------------- mini scene
def check_mini(A):
    print("-- hand-written source-format fixtures: scene stage")
    m = EXP["mini"]
    stage = A.convert_scene(mini("objects.csv"), mini("regions.csv"), mini("vocabulary.csv"))
    scene, cmap, views = stage.scene, stage.category_map, stage.inventory_views
    objs = by_id(scene)
    ident = EXP["identity"]
    check("scene v2 identity: version, scene ID, revision, profile, frame, map and dataset source as approved",
          scene["schema_version"] == 2 and scene["scene_id"] == ident["scene_id"] and scene["scene_revision"] == 0
          and scene["evidence_profile"] == "annotated" and scene["category_map"] == ident["category_map"]
          and scene["coordinate_frame"]["frame_id"] == ident["frame_id"]
          and scene["sources"] == [s for s in scene["sources"] if s["source_id"] == ident["source_id"]]
          and scene["sources"][0]["kind"] == "dataset")
    conv = scene["coordinate_frame"]["conversion"]
    check("conversion: dataset_identity from the dataset source, source frame = scene frame, identity matrix",
          conv == {"kind": "dataset_identity", "source_id": ident["source_id"], "source_frame_id": ident["frame_id"],
                   "source_to_scene": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}, str(conv))
    check("objects in numeric source-ID order, obj_ + at least three digits, all kept",
          [o["object_id"] for o in scene["objects"]] == m["object_ids"]
          and [o["source_ref"]["source_object_id"] for o in scene["objects"]] == ["0", "1", "2", "3"])
    unknown = [k for k, o in objs.items() if o["category"]["state"] == "unknown"]
    check("the NYU 'unknown' object stays, as an unknown category (unmapped_label) with its raw label intact",
          unknown == m["unknown_category"] and objs["obj_002"]["category"]["reason"] == "unmapped_label"
          and objs["obj_002"]["source_ref"]["source_label"] == "object")
    check("known categories use the NYU label as both standard and model label, annotated by the dataset",
          all(o["category"]["value"] == {"standard": o["category"]["value"]["model"], "model": o["category"]["value"]["model"]}
              and o["category"]["evidence"] == {"kind": "annotated", "source": "iref_scannet", "assumptions": []}
              for k, o in objs.items() if k not in unknown))
    check("category map: one entry per known raw label, none for 'object'; the vocabulary minus 'unknown', sorted",
          sorted([e["source_label"], e["standard"], e["model"]] for e in cmap["entries"]) == m["map_entries"]
          and cmap["model_vocabulary"] == m["model_vocabulary"] and cmap["map_id"] == ident["category_map"])
    tol = m["tolerance"]
    rot = {k: objs[k]["geometry"]["rotation_xyzw"] for k in objs}
    check("heading 0 gives the identity quaternion; heading pi/2 gives [0, 0, sin(pi/4), cos(pi/4)]; support yaw_only",
          all(max(abs(a - b) for a, b in zip(rot[k]["value"], v)) <= tol for k, v in m["rotations"].items())
          and all(r["support"] == "yaw_only" for r in rot.values()), str(rot["obj_001"]["value"]))
    turned = rotate(rot["obj_001"]["value"], [1.0, 0.0, 0.0])
    check("heading pi/2 rotates the box's local +x onto scene +y (test-owned rotation matrix)",
          max(abs(a - b) for a, b in zip(turned, m["obj_001_local_x_in_scene"])) <= tol, str(turned))
    front = objs["obj_001"]["semantic_front"]
    check("front -pi/2 points along -y, independently of the box's yaw (pi/2)",
          front["state"] == "known" and max(abs(a - b) for a, b in zip(front["value"], m["obj_001_front"])) <= tol,
          str(front.get("value")))
    check("a '_' front is unknown not_in_source; the yaw never stands in for it",
          all(objs[k]["semantic_front"] == {"state": "unknown", "reason": "not_in_source", "source": "iref_scannet"}
              for k in m["fronts_unknown"]))
    got = {k: (o["attributes"]["colours"]["value"] if o["attributes"]["colours"]["state"] == "known" else None)
           for k, o in objs.items()}
    check("colours: slots in order, omitted groups absent, repeats removed keeping the first, all-absent unknown (not [])",
          got == m["colours"] and objs["obj_002"]["attributes"]["colours"]["reason"] == "not_in_source", str(got))
    check("short rows (complete trailing groups omitted) are accepted and counted",
          stage.short_rows == m["short_rows"], str(stage.short_rows))
    v = {x["view_id"]: x for x in views["views"]}
    check("inventory views: full 4; source_known_nyu 3, excluding obj_002 for source_unknown_category",
          len(v["full_inventory"]["included_object_ids"]) == m["views"]["full_inventory"]
          and v["full_inventory"]["excluded"] == []
          and len(v["source_known_nyu"]["included_object_ids"]) == m["views"]["source_known_nyu"]
          and [[e["object_id"], e["reason"]] for e in v["source_known_nyu"]["excluded"]] == m["views"]["excluded"])
    found = contract.validate([("scene", scene), ("category_map", cmap)])
    check("the shared validator accepts the v2 scene and its map together, with no issue at all",
          found == [], "; ".join(i.line() for i in found[:3]))
    return stage


def check_mini_text(A, stage):
    print("-- hand-written fixtures: commands and the annotation bundle")
    m, ident = EXP["mini"], EXP["identity"]
    statements = mini("statements.json")
    commands = A.convert_commands(statements, stage)
    texts = sorted(json.loads(statements)["regions"]["0"])
    want_ids = sorted(ident["command_id_prefix"] + hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts)
    check("one command per distinct expression, ID = prefix + SHA-256 of its UTF-8 bytes, text unchanged",
          len(commands) == m["commands"] and [c["command_id"] for c in commands] == want_ids
          and sorted(c["text"] for c in commands) == texts)
    pose = commands[0]["user_pose"]
    check("commands: v1 records for the full scene, pose kind none, every pose component unknown not_in_source",
          all(c["schema_version"] == 1 and c["record_type"] == "command_context" and c["scene_id"] == ident["scene_id"]
              and c["scene_revision"] == 0 and c["user_pose"]["pose_kind"] == "none"
              and c["user_pose"]["frame_id"] == ident["frame_id"] for c in commands)
          and all(pose[k] == {"state": "unknown", "reason": "not_in_source", "source": "iref_scannet"}
                  for k in ("position_m", "rotation_xyzw", "heading_xy", "snapshot_time")), json.dumps(pose)[:120])
    bundle = A.convert_annotations(statements, stage)
    entries = bundle["entries"]
    source = json.loads(statements)["regions"]["0"]
    check("annotation bundle: every record kept as an entry; IDs = command ID + .a + 3-digit index; sorted",
          len(entries) == m["annotations"]
          and [e["annotation_id"] for e in entries] == sorted(e["annotation_id"] for e in entries)
          and all(e["annotation_id"] == f"{e['command_id']}.a{e['source_annotation_index']:03d}" for e in entries))
    check("the source payload is the original annotation object, unchanged",
          all(e["source_payload"] == source[next(t for t in source if ident["command_id_prefix"]
                                                   + hashlib.sha256(t.encode()).hexdigest() == e["command_id"])]
              [e["source_annotation_index"]] for e in entries))
    check("mapped references: target obj_000, anchor_1 obj_001, distractor obj_003",
          all(e["mapped_references"] == {"target": "obj_000", "anchors": {"anchor_1": "obj_001"},
                                         "distractors": ["obj_003"]} for e in entries))
    per = {}
    for e in entries:
        per.setdefault(e["command_id"], []).append(e["mapped_references"]["target"])
    repeated = [t for t in per.values() if len(t) > 1]
    check("a repeated expression keeps both annotations, with one target, not relabelled as ambiguity",
          len(repeated) == m["repeated_expression_groups"] and all(len(set(t)) == 1 for t in repeated)
          and len(per) == m["commands"], str(per))
    report = A.crosscheck_graph(mini("graph.json"), stage)
    check("graph cross-check: 3 graph objects agree with the CSV; obj_002 recorded as omitted from the graph",
          report["graph_objects"] == m["graph_objects"] and report["omitted_from_graph"] == m["graph_omitted"]
          and report["mismatches"] == 0, json.dumps(report)[:160])
    together = contract.validate([("scene", stage.scene), ("category_map", stage.category_map)]
                                 + [(c["command_id"], c) for c in commands])
    check("scene, map and commands validate together with no issue (no unresolved-source or map warning)",
          together == [], "; ".join(i.line() for i in together[:3]))
    return commands, bundle


def check_mini_errors(A):
    print("-- malformed sources fail explicitly (codes fixed in expectations.json)")
    want = EXP["errors"]
    objects, regions, vocab, statements, graph = (mini(n) for n in ("objects.csv", "regions.csv", "vocabulary.csv",
                                                                   "statements.json", "graph.json"))
    lines = objects.decode().split("\r\n")

    def row(i, col, value):
        out = list(lines)
        cells = out[i].split(",")
        cells[col] = value
        out[i] = ",".join(cells)
        return "\r\n".join(out).encode()

    def scene_case(objs=objects, regs=regions, voc=vocab):
        return lambda: A.convert_scene(objs, regs, voc)

    def text_case(mutate=None, graph_bytes=None):
        def run():
            st = A.convert_scene(objects, regions, vocab)
            if graph_bytes is not None:
                return A.crosscheck_graph(graph_bytes, st)
            data = json.loads(statements)
            mutate(data)
            return A.convert_annotations(json.dumps(data).encode(), st)
        return run

    def set_first(data, key, value):
        rec = next(iter(data["regions"]["0"].values()))[0]
        rec[key] = value

    g = json.loads(graph)
    g["regions"]["0"]["objects"][0]["center"][0] += 0.5
    cases = {
        "duplicate_object_id": scene_case(row(2, 0, "0")),
        "noncanonical_object_id": scene_case(row(1, 0, "00")),
        "empty_field": scene_case(row(1, 2, "")),
        "nyu_id_label_mismatch": scene_case(row(1, 5, "table")),
        "unrecognized_nyu_label": scene_case(row(1, 3, "999")),
        "raw_label_conflict": scene_case(row(4, 3, "19").replace(b",chair,chair,1.5", b",table,table,1.5")),
        "zero_size": scene_case(row(1, 10, "0")),
        "nonfinite_heading": scene_case(row(1, 13, "nan")),
        "invalid_front": scene_case(row(1, 14, "abc")),
        "nonfinite_front": scene_case(row(2, 14, "inf")),
        "label_with_trailing_space": scene_case(row(1, 2, "chair ")),
        "colour_group_inconsistent": scene_case(row(3, 21, "12")),
        "partial_colour_group": scene_case("\r\n".join(lines[:2] + [lines[2] + ",1,2"] + lines[3:]).encode()),
        "extra_column": scene_case("\r\n".join(lines[:3] + [lines[3] + ",x"] + lines[4:]).encode()),
        "renamed_header": scene_case(objects.replace(b"object_bbox_heading", b"object_bbox_yaw")),
        "duplicate_header": scene_case(objects.replace(b"nyu40_label", b"nyu_label", 1)),
        "byte_order_mark": scene_case(b"\xef\xbb\xbf" + objects),
        "dangling_region": scene_case(row(1, 1, "1")),
        "second_region": scene_case(regs=regions + b"1,Kitchen,0,0,0,1,1,1,0\r\n"),
        "duplicate_json_key": lambda: A.convert_annotations(statements.replace(b'"scene_name": "scene0010_01",',
                                                                               b'"scene_name": "scene0010_01", "scene_name": "x",', 1),
                                                            A.convert_scene(objects, regions, vocab)),
        "unexpected_annotation_key": text_case(lambda d: set_first(d, "comment", "extra")),
        "dangling_annotation_target": text_case(lambda d: set_first(d, "target_index", "9")),
        "other_scene_name": text_case(lambda d: d.__setitem__("scene_name", "scene0011_00")),
        "annotation_position_mismatch": text_case(lambda d: set_first(d, "target_position", [9.0, 9.0, 9.0])),
        "graph_centre_mismatch": text_case(graph_bytes=json.dumps(g).encode()),
    }
    for name, fn in cases.items():
        try:
            fn()
            got = ["no error"]
        except A.AdapterInputError as e:
            got = codes(e)
        except Exception as e:  # noqa: BLE001 - an unexpected exception is this check's failure
            got = [f"raised {type(e).__name__}: {str(e)[:80]}"]
        check(f"{name} fails with exactly {want[name]}", got == [want[name]], str(got))


def check_frames(A, stage):
    print("-- scene v2 frame rules and the version boundary (shared validator)")
    want = EXP["frames"]
    base = stage.scene

    def variant(fn):
        s = copy.deepcopy(base)
        fn(s)
        return s
    conv = ("coordinate_frame", "conversion")
    cases = {
        "nonidentity_matrix": variant(lambda s: s[conv[0]][conv[1]]["source_to_scene"][0].__setitem__(3, 0.5)),
        "wrong_source_kind": variant(lambda s: s["sources"][0].__setitem__("kind", "fixture")),
        "unresolved_source": variant(lambda s: s[conv[0]][conv[1]].__setitem__("source_id", "nope")),
        "inconsistent_frame_ids": variant(lambda s: s[conv[0]][conv[1]].__setitem__("source_frame_id", "other.frame")),
        "v1_with_dataset_conversion": variant(lambda s: s.__setitem__("schema_version", 1)),
        "unsupported_version_3": variant(lambda s: s.__setitem__("schema_version", 3)),
    }
    for name, scene in cases.items():
        found = [(i.code, i.path) for i in contract.validate([("scene", scene)]) if i.is_error]
        check(f"{name}: exactly {want[name][0]} at {want[name][1]}", found == [tuple(want[name])], str(found))
    valid = sorted((REPO / "grounding/tests/fixtures/contract/valid").glob("*.json"))
    bad = [p.name for p in valid if any(i.is_error for i in contract.validate_paths([p]))]
    check(f"all {len(valid)} accepted contract fixtures stay valid", not bad, str(bad))


def check_interfaces(A, stage, commands):
    print("-- existing interfaces accept the v2 scene unchanged")
    from grounding.serialization import serialize
    from grounding.resolution import resolve
    res = serialize(stage.scene, commands[0], format="coordinates_v2", relation_config_path=REL_CFG,
                    direction_config_path=DIR_CFG, category_maps=[stage.category_map])
    check("the coordinates serializer renders the v2 scene with a no-pose command, with zero relation work",
          res.status == "ok" and res.document and res.metadata["format"] == "coordinates_v2", res.status)
    cfg = json.loads((REPO / "grounding/tests/fixtures/resolution/resolver-config.fixture.json").read_text())
    cfg["category_labels"] = stage.category_map["model_vocabulary"]
    cfg["colour_labels"] = ["black", "brown", "gray", "white"]
    command = dict(copy.deepcopy(commands[0]), command_id="test.iref.v2.command", text="Test-authored command, not a "
                   "dataset expression")
    query = {"schema_version": 1, "record_type": "grounding_query", "query_id": "test.iref.v2.query",
             "scene_id": stage.scene["scene_id"], "scene_revision": 0, "evidence_profile": "annotated",
             "command_id": command["command_id"], "interpretations": [{"interpretation_id": "i0", "kind": "query",
             "action": "INSPECT", "root": "t", "nodes": [{"node_id": "t", "category": "chair", "colours_all": [],
                                                          "constraints": [], "rank": None}]}]}
    rec = resolve(stage.scene, command, query, resolver_config=cfg, relation_config_path=REL_CFG,
                  direction_config_path=DIR_CFG, category_maps=[stage.category_map])
    r = rec["result"] or {}
    check("the resolver runs a test-authored query on the v2 scene: two chairs and the unknown object -> ambiguous",
          rec["processing_status"] == "completed" and r.get("status") == "ambiguous"
          and r["candidates"]["target_ids"] == ["obj_000", "obj_003"]
          and r["candidates"]["uncertain_match_ids"] == ["obj_002"], f"{rec['processing_status']} {r.get('status')}")
    bundle = A.convert_annotations(mini("statements.json"), stage)
    found = [i.code for i in contract.validate([("scene", bundle)])]
    check("an annotation bundle passed as a scene fails the shared validator's record-type check",
          found == ["E_RECORD_TYPE"], str(found))
    leaked = keys_of(stage.scene) | keys_of(stage.category_map) | set().union(*(keys_of(c) for c in commands))
    check("no annotation field name occurs as a key in the model-input scene, map or commands",
          not (leaked & ANNOTATION_KEYS), str(sorted(leaked & ANNOTATION_KEYS)))


def check_outputs(A):
    print("-- outputs: layout, determinism, separation, refusal and failure safety (mini, pins off)")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = {n: FX / f"mini_{n}" for n in ("objects.csv", "regions.csv", "vocabulary.csv", "statements.json",
                                            "graph.json")}
        before = {n: p.read_bytes() for n, p in src.items()}
        kw = dict(statements=src["statements.json"], graph=src["graph.json"], pins=None)
        A.run_import(src["objects.csv"], src["regions.csv"], src["vocabulary.csv"], tmp / "a", **kw)
        A.run_import(src["objects.csv"], src["regions.csv"], src["vocabulary.csv"], tmp / "b", **kw)
        A.run_import(src["objects.csv"], src["regions.csv"], src["vocabulary.csv"], tmp / "meta", pins=None)
        files = sorted(str(p.relative_to(tmp / "a")) for p in (tmp / "a").rglob("*") if p.is_file())
        want = {"model_inputs/scene.annotated.json", "model_inputs/category-map.json", "reference_only/iref-annotations.json",
                "reference_only/source-manifest.json", "reference_only/inventory-views.json",
                "reference_only/import-report.json"}
        check("layout: scene, map and commands under model_inputs; annotations and manifests under reference_only",
              want <= {f.replace(os.sep, "/") for f in files}
              and sum(f.replace(os.sep, "/").startswith("model_inputs/commands/") for f in files) == 2, str(files))
        same = all((tmp / "a" / f).read_bytes() == (tmp / "b" / f).read_bytes() for f in files)
        check("two runs into new folders write identical bytes", same)
        meta_same = all((tmp / "a" / f).read_bytes() == (tmp / "meta" / f).read_bytes()
                        for f in ("model_inputs/scene.annotated.json", "model_inputs/category-map.json",
                                  "reference_only/inventory-views.json"))
        report = json.loads((tmp / "meta" / "reference_only/import-report.json").read_text())
        check("without statements: same scene, map and view bytes; no commands or bundle; report says not_supplied",
              meta_same and not (tmp / "meta/model_inputs/commands").exists()
              and not (tmp / "meta/reference_only/iref-annotations.json").exists()
              and report["statements"] == "not_supplied")
        text = (tmp / "a/model_inputs/scene.annotated.json").read_bytes()
        check("output JSON: UTF-8, sorted keys, two-space indentation, one final LF",
              text.endswith(b"}\n") and not text.endswith(b"\n\n") and b"\r" not in text
              and text == (json.dumps(json.loads(text), ensure_ascii=False, allow_nan=False, sort_keys=True,
                                      indent=2) + "\n").encode("utf-8"))
        try:
            A.run_import(src["objects.csv"], src["regions.csv"], src["vocabulary.csv"], tmp / "a", **kw)
            got = ["no error"]
        except A.AdapterInputError as e:
            got = codes(e)
        check("an existing output folder is refused (E_IREF_OUTPUT_EXISTS) and left as it was",
              got == [EXP["errors"]["output_exists"]] and all((tmp / "a" / f).read_bytes() == (tmp / "b" / f).read_bytes()
                                                              for f in files), str(got))
        out_mod = importlib.import_module("grounding.adapters.iref_vla.output")
        real, calls = out_mod._write_file, [0]

        def failing(path, data):
            calls[0] += 1
            if calls[0] == 3:
                raise OSError("simulated disk failure")
            real(path, data)

        out_mod._write_file = failing
        try:
            A.run_import(src["objects.csv"], src["regions.csv"], src["vocabulary.csv"], tmp / "broken", **kw)
            got = ["no error"]
        except A.AdapterOutputError as e:
            got = codes(e)
        finally:
            out_mod._write_file = real
        left = sorted(p.name for p in tmp.iterdir())
        check("a write failure is E_IREF_OUTPUT_IO and leaves no bundle and no partial folder",
              got == [EXP["errors"]["output_write_failure"]] and left == ["a", "b", "meta"], f"{got} {left}")
        check("the source files are unchanged", all(p.read_bytes() == before[n] for n, p in src.items()))
        # Added after the first run: mutation testing showed that writing straight into the final folder passed the
        # failure check above (clean-up removed it), though a killed process would leave it half-written.
        seen = []

        def watching(path, data):
            seen.append((tmp / "late").exists())
            real(path, data)

        out_mod._write_file = watching
        try:
            A.run_import(src["objects.csv"], src["regions.csv"], src["vocabulary.csv"], tmp / "late", **kw)
        finally:
            out_mod._write_file = real
        check("(added after the first run) the final folder appears only once every file is written: one rename at the end",
              seen and not any(seen) and (tmp / "late/reference_only/import-report.json").is_file(), f"{len(seen)} writes")


# ---------------------------------------------------------------------------------------------- pinned sample
def sample_dir():
    for i, a in enumerate(sys.argv):
        if a == "--sample" and i + 1 < len(sys.argv):
            return Path(sys.argv[i + 1])
    env = os.environ.get("SECOND_EYES_IREF_VLA_SAMPLE")
    return Path(env) if env else None


def check_pinned(A):
    print("-- the pinned IRef-VLA sample (acceptance)")
    d = sample_dir()
    pins = EXP["pinned"]["files"]
    if d is None or not all((d / f["name"]).is_file() for f in pins.values()):
        check("pinned sample available (pass --sample DIR or set SECOND_EYES_IREF_VLA_SAMPLE)", False,
              f"not found in {d}" if d else "no location given")
        return
    bad = [r for r, f in pins.items() if len((d / f["name"]).read_bytes()) != f["bytes"]
           or hashlib.sha256((d / f["name"]).read_bytes()).hexdigest() != f["sha256"]]
    check("all five pinned files match Appendix A's bytes and SHA-256", not bad, str(bad))
    if bad:
        return
    p = EXP["pinned_sample"]
    path = {r: d / f["name"] for r, f in pins.items()}
    raw = {r: path[r].read_bytes() for r in path}
    stage = A.convert_scene(raw["objects"], raw["regions"], raw["vocabulary"])
    scene, cmap, objs = stage.scene, stage.category_map, by_id(stage.scene)
    check("61 objects, obj_000 to obj_060 in source order", [o["object_id"] for o in scene["objects"]] == p["object_ids"])
    unk = {k: o["source_ref"]["source_object_id"] for k, o in objs.items() if o["category"]["state"] == "unknown"}
    check("unknown categories are exactly 55 and 58, unmapped_label, raw label 'object'",
          unk == p["unknown_category"] and all(objs[k]["category"]["reason"] == p["unknown_reason"]
                                               and objs[k]["source_ref"]["source_label"] == p["unknown_source_label"]
                                               for k in unk))
    entries = {e["source_label"]: e["model"] for e in cmap["entries"]}
    check("59 known categories; 24 map entries; 893 model labels, sorted, without 'unknown'",
          len(objs) - len(unk) == p["known_categories"] and len(entries) == p["map_entries"]
          and len(cmap["model_vocabulary"]) == p["model_vocabulary"] and "unknown" not in cmap["model_vocabulary"]
          and cmap["model_vocabulary"] == sorted(cmap["model_vocabulary"])
          and all(entries.get(k) == v for k, v in p["map_examples"].items()) and "object" not in entries)
    known = lambda f: sum(o["geometry"][f]["state"] == "known" for o in objs.values())  # noqa: E731
    fronts = sum(o["semantic_front"]["state"] == "known" for o in objs.values())
    check("61 centres, sizes and yaw rotations; 14 fronts known, 47 unknown not_in_source",
          (known("center_m"), known("size_m"), known("rotation_xyzw")) == (61, 61, 61)
          and all(o["geometry"]["rotation_xyzw"]["support"] == "yaw_only" for o in objs.values())
          and (fronts, len(objs) - fronts) == (p["fronts_known"], p["fronts_unknown"])
          and all(o["semantic_front"].get("reason", "not_in_source") == "not_in_source" for o in objs.values()))
    lists = [o["attributes"]["colours"] for o in objs.values()]
    check("61 known colour lists, no placeholder inside one; short rows exactly 35, 42 and 43",
          sum(c["state"] == "known" for c in lists) == p["colour_lists_known"]
          and not any(x in ("_", "N/A", "") for c in lists if c["state"] == "known" for x in c["value"])
          and stage.short_rows == p["short_rows"], str(stage.short_rows))
    c = p["concrete"]
    o2 = objs[c["object_id"]]
    check("object 2: front and quaternion as hand-computed, within 1e-12",
          max(abs(a - b) for a, b in zip(o2["semantic_front"]["value"], c["front"])) <= c["tolerance"]
          and max(abs(a - b) for a, b in zip(o2["geometry"]["rotation_xyzw"]["value"], c["rotation"])) <= c["tolerance"])
    graph = json.loads(raw["graph"])["regions"]["0"]["objects"]
    worst = [o["object_id"] for o in graph if not same_corners(corners(objs[f"obj_{int(o['object_id']):03d}"]),
                                                               o["bbox"], p["corner_tolerance_m"])]
    check(f"all {len(graph)} graph boxes rebuilt from the OUTPUT quaternion, centre and size match their source "
          "corners within 1e-9 m", len(graph) == p["graph_objects"] and not worst, str(worst[:5]))
    v = {x["view_id"]: x for x in stage.inventory_views["views"]}
    check("views: full 61; known-NYU 59 excluding 55 and 58 for source_unknown_category",
          len(v["full_inventory"]["included_object_ids"]) == 61 and v["full_inventory"]["excluded"] == []
          and len(v["source_known_nyu"]["included_object_ids"]) == 59
          and [[e["object_id"], e["reason"]] for e in v["source_known_nyu"]["excluded"]]
          == p["views"]["source_known_nyu"]["excluded"])
    commands = A.convert_commands(raw["statements"], stage)
    bundle = A.convert_annotations(raw["statements"], stage)
    entries = bundle["entries"]
    check("1,936 commands and 1,951 annotation entries", (len(commands), len(entries)) == (p["commands"], p["annotations"]))
    groups = {}
    for e in entries:
        groups.setdefault(e["command_id"], []).append(e["mapped_references"]["target"])
    repeated = [g for g in groups.values() if len(g) > 1]
    check("12 repeated-expression groups, each with one target", len(repeated) == p["repeated_expression_groups"]
          and all(len(set(g)) == 1 for g in repeated))
    rel = {}
    for e in entries:
        rel[e["source_payload"]["relation"]] = rel.get(e["source_payload"]["relation"], 0) + 1
    check("the source relation count vector is unchanged", rel == p["relations"], str(rel))
    sizes, roles, hit = {}, {"anchor": 0, "target": 0}, 0
    for e in entries:
        sp, any_ = e["source_payload"], False
        if sp["target_size_used"]:
            sizes[sp["target_size_used"]] = sizes.get(sp["target_size_used"], 0) + 1
            roles["target"] += 1
            any_ = True
        for a in sp["anchors"].values():
            if a["size_used"]:
                sizes[a["size_used"]] = sizes.get(a["size_used"], 0) + 1
                roles["anchor"] += 1
                any_ = True
        hit += any_
    s = p["size"]
    check("size words kept: 703 entries, 706 occurrences (584 big, 122 small; 703 anchor, 3 target)",
          (hit, sum(sizes.values()), sizes.get("big"), sizes.get("small"), roles["anchor"], roles["target"])
          == (s["entries"], s["occurrences"], s["big"], s["small"], s["anchor_occurrences"], s["target_occurrences"]))
    anchors = sum(len(e["mapped_references"]["anchors"]) for e in entries)
    check("every target, anchor and distractor maps (2,089 anchor occurrences)",
          anchors == p["anchor_occurrences"] and all(e["mapped_references"]["target"] in objs for e in entries))
    report = A.crosscheck_graph(raw["graph"], stage)
    check("graph cross-check: 59 objects agree; 55 and 58 recorded as omitted",
          report["graph_objects"] == 59 and report["omitted_from_graph"] == p["graph_omitted"]
          and report["mismatches"] == 0)
    together = contract.validate([("scene", scene), ("category_map", cmap)] + [(x["command_id"], x) for x in commands])
    check("the shared validator accepts the scene, map and all 1,936 commands together, with no issue at all",
          together == [], "; ".join(i.line() for i in together[:3]))
    import jsonschema
    schema = json.loads((REPO / "schemas/iref-annotations.v1.json").read_text(encoding="utf-8"))
    errs = list(jsonschema.Draft202012Validator(schema).iter_errors(bundle))
    check("the annotation bundle validates against iref-annotations.v1.json", not errs, errs[0].message[:120] if errs else "")
    leaked = keys_of(scene) | keys_of(cmap) | set().union(*(keys_of(x) for x in commands))
    check("no annotation field name occurs in the model-input records", not (leaked & ANNOTATION_KEYS))
    mutated = json.loads(raw["statements"])
    for lst in mutated["regions"]["0"].values():
        for a in lst:
            a["target_position"] = [0.0, 0.0, 0.0]
            a["target_index"] = "0" if a["target_index"] != "0" else "1"
            a["distractor_ids"] = list(reversed(a["distractor_ids"]))
            for an in a["anchors"].values():
                an["index"], an["position"] = "1", [9.0, 9.0, 9.0]
    mutated_bytes = json.dumps(mutated).encode("utf-8")
    again = A.convert_scene(raw["objects"], raw["regions"], raw["vocabulary"])
    check("leakage: answers can't reach the scene stage (it takes only CSVs and vocabulary); its bytes are identical",
          all(A.encode(getattr(again, k)) == A.encode(getattr(stage, k)) for k in ("scene", "category_map",
                                                                                    "inventory_views")))
    check("leakage: with targets, anchors, distractors and positions mutated, command IDs and text are identical",
          [A.encode(x) for x in A.convert_commands(mutated_bytes, again)] == [A.encode(x) for x in commands])
    try:
        A.convert_annotations(mutated_bytes, again)
        got = ["no error"]
    except A.AdapterInputError as e:
        got = codes(e)
    check("leakage: the mutated answers are then rejected by the annotation cross-check, not repaired",
          got and all(c in ("E_IREF_SOURCE_MISMATCH", "E_IREF_REFERENCE") for c in got), str(got))
    check_pinned_cli(A, d, path, raw, stage, commands)


def check_pinned_cli(A, d, path, raw, stage, commands):
    print("-- the command line on the pinned sample")
    from grounding.serialization import serialize
    res = serialize(stage.scene, commands[0], format="coordinates_v2", relation_config_path=REL_CFG,
                    direction_config_path=DIR_CFG, category_maps=[stage.category_map])
    check("the coordinates serializer renders the full 61-object v2 scene with a no-pose command",
          res.status == "ok" and bool(res.document), res.status)

    def cli(out, *extra):
        args = [sys.executable, "-m", "grounding.adapters.iref_vla", "--objects", str(path["objects"]), "--regions",
                str(path["regions"]), "--vocabulary", str(path["vocabulary"]), *extra, "--out", str(out)]
        return subprocess.run(args, capture_output=True, text=True, cwd=str(REPO))
    full = ("--statements", str(path["statements"]), "--graph", str(path["graph"]))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        one, two, meta = cli(tmp / "one", *full), cli(tmp / "two", *full), cli(tmp / "meta")
        check("CLI: a full import exits 0", one.returncode == 0, one.stderr[-200:])
        files = sorted(p.relative_to(tmp / "one") for p in (tmp / "one").rglob("*") if p.is_file())
        check("CLI: two full imports into new folders are byte-identical",
              two.returncode == 0 and files and all((tmp / "one" / f).read_bytes() == (tmp / "two" / f).read_bytes()
                                                    for f in files), f"{len(files)} files")
        same = meta.returncode == 0 and all((tmp / "one" / f).read_bytes() == (tmp / "meta" / f).read_bytes()
                                            for f in ("model_inputs/scene.annotated.json",
                                                      "model_inputs/category-map.json",
                                                      "reference_only/inventory-views.json"))
        check("CLI: a metadata-only import writes the same scene, map and view bytes", same, meta.stderr[-200:])
        cmds = sorted((tmp / "one/model_inputs/commands").glob("*.json"))
        check("CLI: 1,936 command files, each named by its command ID", len(cmds) == 1936
              and all(json.loads(c.read_text(encoding="utf-8"))["command_id"] == c.stem for c in cmds[:50]))
        again = cli(tmp / "one", *full)
        check("CLI: an existing output exits 2 with E_IREF_OUTPUT_EXISTS", again.returncode == 2
              and "E_IREF_OUTPUT_EXISTS" in again.stdout + again.stderr, again.stdout[-200:])
        changed = tmp / "regions.changed.csv"
        changed.write_bytes(raw["regions"] + b" ")
        bad = subprocess.run([sys.executable, "-m", "grounding.adapters.iref_vla", "--objects", str(path["objects"]),
                              "--regions", str(changed), "--vocabulary", str(path["vocabulary"]), "--out",
                              str(tmp / "bad")], capture_output=True, text=True, cwd=str(REPO))
        check("CLI: an input that differs from its pin exits 2 with E_IREF_SOURCE_MISMATCH and writes nothing",
              bad.returncode == 2 and "E_IREF_SOURCE_MISMATCH" in bad.stdout + bad.stderr
              and not (tmp / "bad").exists() and "Traceback" not in bad.stderr, bad.stdout[-200:])
        main = importlib.import_module("grounding.adapters.iref_vla.__main__")
        real, err = main.run_import, io.StringIO()

        def boom(*a, **k):
            raise RuntimeError("controlled test exception")
        main.run_import = boom
        try:
            with contextlib.redirect_stderr(err):
                code = main.main(["--objects", str(path["objects"]), "--regions", str(path["regions"]),
                                  "--vocabulary", str(path["vocabulary"]), "--out", str(tmp / "boom")])
        finally:
            main.run_import = real
        check("CLI: an unexpected exception exits 3 as an internal error, with no output",
              code == 3 and "internal error" in err.getvalue() and not (tmp / "boom").exists(), f"exit {code}")
    unchanged = all(hashlib.sha256(path[r].read_bytes()).hexdigest() == f["sha256"]
                    for r, f in EXP["pinned"]["files"].items())
    check("the pinned source files are unchanged after every run", unchanged)


def main() -> int:
    A = importlib.import_module("grounding.adapters.iref_vla")
    stage = check_mini(A)
    commands, _ = check_mini_text(A, stage)
    check_mini_errors(A)
    check_frames(A, stage)
    check_interfaces(A, stage, commands)
    check_outputs(A)
    check_pinned(A)
    print(f"{COUNT[0]} checks; {'FAILED: ' + ', '.join(FAILED) if FAILED else 'all checks passed'}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
