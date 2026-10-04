"""Tests of the scene contract's offline validator (A2.1a, D66).

    python grounding/tests/test_contract.py

Needs only Python 3.9+ and jsonschema (grounding/requirements.txt or tools/requirements.txt). Checks: the schemas are
valid JSON Schema 2020-12 and share identical common definitions; every valid fixture passes alone and all pass
together with no warning; every invalid fixture fails with exactly the code in its name, together with its partner
for cross-record rules; every error code has a fixture; a few cases built in memory pass; malformed record types
fail cleanly (the A2.1a correction); the validator never changes a file; the command line exits 0 and 1. Prints one
line per check and exits 1 if any fails.
"""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "grounding" / "contract"))
import jsonschema  # noqa: E402

import checks  # noqa: E402
import validate  # noqa: E402

FIXTURES = REPO / "grounding" / "tests" / "fixtures" / "contract"
VALID, INVALID = FIXTURES / "valid", FIXTURES / "invalid"
PARTNERS = {"E_SCENE_MISMATCH": "scene.a17.annotated.json", "E_FRAME_MISMATCH": "scene.a17.annotated.json",
            "E_DUPLICATE_RECORD": "scene.a17.annotated.json", "E_CATEGORY_UNMAPPED": "category-map.fx.json"}
NO_FIXTURE = {"E_SCHEMA_OTHER", "W_NOT_CROSS_CHECKED"}   # no rule maps to the first; the second is a warning
FAILED = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("PASS  " if ok else "FAIL  ") + name + (f": {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def errors(issues):
    return [i for i in issues if i.is_error]


def load(name: str):
    return json.loads((VALID / name).read_text(encoding="utf-8"))


def digest(paths) -> dict:
    return {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def last_line(text: str) -> str:
    lines = text.strip().splitlines()
    return lines[-1] if lines else ""


# 9. O23 (D73): IDs, labels and object IDs match as whole strings. Written and run against the unchanged schemas first.
OLD = {"id": r"^[a-z0-9][a-z0-9_.-]{0,127}$", "label": r"^\S(.*\S)?$", "object_id": r"^obj_[0-9]{3,}$"}
NEW = {"id": r"^[a-z0-9][a-z0-9_.-]{0,127}(?![\s\S])", "label": r"^\S(.*\S)?(?![\s\S])",
       "object_id": r"^obj_[0-9]{3,}(?![\s\S])"}
OCCURRENCES = (  # schema file, path to the definition, kind, a valid value
    ("scene.v1.json", ("$defs", "id"), "id", "fixture.a17"),
    ("scene.v1.json", ("$defs", "label"), "label", "office chair"),
    ("scene.v1.json", ("$defs", "object", "properties", "object_id"), "object_id", "obj_001"),
    ("command-context.v1.json", ("$defs", "id"), "id", "fx.a17.c001"),
    ("command-context.v1.json", ("$defs", "label"), "label", "chair"),
    ("category-map.v1.json", ("$defs", "id"), "id", "map.fx"),
    ("category-map.v1.json", ("$defs", "label"), "label", "floor lamp"),
    ("grounding-query.v1.json", ("$defs", "id"), "id", "fx.r08.query"),
    ("grounding-query.v1.json", ("$defs", "label"), "label", "box"),
)


def check_o23(fixture_paths, before) -> None:
    """O23 (D73): a final LF no longer passes an ID, label or object-ID pattern.

    Python's $ also matches before a final newline, and jsonschema applies patterns with re.search. The inputs are
    small mutations of the valid fixtures, in memory or in a temporary file; no fixture file changes.
    """
    import itertools
    import re
    import tempfile

    def node(doc, path):
        for part in path:
            doc = doc[part]
        return doc

    schemas = {}
    for name, path, kind, value in OCCURRENCES:
        schemas.setdefault(name, json.loads((REPO / "schemas" / name).read_text(encoding="utf-8")))
        definition = node(schemas[name], path)
        v = jsonschema.Draft202012Validator(definition)
        check(f"O23: {name} {'.'.join(path)} is the corrected pattern, accepts {value!r} and rejects it with a final LF",
              definition.get("pattern") == NEW[kind] and v.is_valid(value) and not v.is_valid(value + "\n"),
              f"pattern {definition.get('pattern')!r}; {value!r} + LF valid: {v.is_valid(value + chr(10))}")
    scene_defs = schemas["scene.v1.json"]["$defs"]
    same = all(schemas[n]["$defs"][k] == scene_defs[k] for n in ("command-context.v1.json", "category-map.v1.json")
               for k in ("id", "label"))
    same = same and all(schemas["grounding-query.v1.json"]["$defs"][k]["pattern"] == scene_defs[k]["pattern"]
                        for k in ("id", "label"))
    check("O23: the id and label definitions stay identical in the scene, command and map schemas, and the query's "
          "copies carry the same patterns", same)

    current = {"id": scene_defs["id"]["pattern"], "label": scene_defs["label"]["pattern"],
               "object_id": scene_defs["object"]["properties"]["object_id"]["pattern"]}
    alphabets = {"id": ("a", "z", "0", "_", ".", "-", "A", " ", "\t", "\n", "\r", "\u00e9", "\u00a0"),
                 "label": ("a", "Z", "0", "_", "/", " ", "\t", "\n", "\r", "\u00e9", "\u00a0", "\u2028", "\u6905"),
                 "object_id": ("0", "9", "a", "_", " ", "\t", "\n", "\r", "\u00a0")}
    for kind in ("id", "label", "object_id"):
        prefix, longest = ("obj_", 4) if kind == "object_id" else ("", 3)
        strings = [prefix + "".join(t) for n in range(longest + 1) for t in itertools.product(alphabets[kind], repeat=n)]
        body = OLD[kind][1:-1]  # the old pattern without its anchors, matched as a whole string
        differ = [s for s in strings if bool(re.search(current[kind], s)) != bool(re.fullmatch(body, s))]
        check(f"O23: the {kind} pattern accepts exactly the old pattern's whole-string matches, over {len(strings)} "
              f"short strings", not differ, f"{len(differ)} differ, e.g. {differ[:3]!r}")

    scene, restricted = load("scene.a17.annotated.json"), load("scene.a17.restricted.json")
    command, cmap = load("command.a17.c001.json"), load("category-map.fx.json")
    originals = copy.deepcopy([scene, restricted, command, cmap])
    designated = (  # validated one record at a time: a missing partner would only add warnings
        ("scene", scene, ["scene_id"]), ("command", command, ["command_id"]), ("category_map", cmap, ["map_id"]),
        ("scene", scene, ["objects", 0, "object_id"]),
        ("scene", scene, ["objects", 0, "source_ref", "source_id"]),
        ("scene", scene, ["coordinate_frame", "frame_id"]), ("command", command, ["user_pose", "frame_id"]),
        ("scene", restricted, ["objects", 0, "geometry", "size_m", "evidence", "assumptions", 0]),
        ("scene", scene, ["objects", 0, "category", "value", "model"]),
        ("scene", scene, ["objects", 0, "attributes", "colours", "value", 0]),
        ("category_map", cmap, ["entries", 0, "source_label"]), ("category_map", cmap, ["entries", 0, "model"]),
    )
    for label, base, path in designated:
        rec = copy.deepcopy(base)
        node(rec, path[:-1])[path[-1]] += "\n"
        where = validate._json_path(path)
        found = [(i.code, i.path) for i in validate.validate([(label, rec)])]
        check(f"O23: a {label} with a final LF at {where} fails with exactly E_SCHEMA_PATTERN there",
              found == [("E_SCHEMA_PATTERN", where)], f"got {found}")

    rec, changed = copy.deepcopy(scene), []

    def relabel(x, path):  # the source registry ID and every reference to it, all changed alike
        for key, value in (x.items() if isinstance(x, dict) else enumerate(x)):
            if value == "fx":
                x[key] = "fx\n"
                changed.append(validate._json_path(path + [key]))
            elif isinstance(value, (dict, list)):
                relabel(value, path + [key])
    relabel(rec, [])
    found = validate.validate([("scene", rec)])
    check(f"O23: source ID 'fx' and all {len(changed) - 1} references changed alike to 'fx' + LF still fail, with "
          f"E_SCHEMA_PATTERN at each of the {len(changed)} strings",
          "$.sources[0].source_id" in changed and sorted((i.code, i.path) for i in found)
          == sorted(("E_SCHEMA_PATTERN", c) for c in changed), f"{len(found)} issues: {[i.code for i in found][:3]}")

    controls = {"id": "fixture.a17", "label": "chair", "object_id": "obj_001"}
    for kind, value in controls.items():
        v = jsonschema.Draft202012Validator({"type": "string", "pattern": current[kind]})
        cases = [(value + s, s == "\n") for s in ("\n", "\r\n", "\r", "\t", " ")] + \
                [(s + value, False) for s in (" ", "\t", "\n")]
        cases += {"id": [("Fixture.a17", False), ("fixture a17", False), ("fixture/a17", False)],
                  "object_id": [("OBJ_001", False), ("obj_01", False)]}.get(kind, [])
        wrong = [f"old pattern {'accepted' if not new else 'rejected'} {s!r}" for s, new in cases
                 if bool(re.search(OLD[kind], s)) != new]
        wrong += [f"{s!r} is accepted" for s, _ in cases if v.is_valid(s)]
        check(f"O23: {kind} with an LF, CRLF, CR, tab or space after it, or whitespace before it, or bad characters, is "
              f"rejected; the old pattern rejected all but the final LF", not wrong, "; ".join(wrong[:4]))

    v_id = jsonschema.Draft202012Validator({"type": "string", "pattern": current["id"]})
    v_obj = jsonschema.Draft202012Validator({"type": "string", "pattern": current["object_id"]})
    v_label = jsonschema.Draft202012Validator({"type": "string", "pattern": current["label"]})
    ids = ["fixture.a17", "map.fx", "fx", "a", "0", "x" * 128]
    check("O23: ordinary IDs and a 128-character ID stay valid, as before; 129 characters stay invalid",
          all(v_id.is_valid(s) and re.search(OLD["id"], s) for s in ids) and not v_id.is_valid("x" * 129)
          and not re.search(OLD["id"], "x" * 129))
    objs = ["obj_000", "obj_999", "obj_1000", "obj_12345"]
    check("O23: obj_999, obj_1000 and longer object IDs stay valid, as before; obj_01 stays invalid",
          all(v_obj.is_valid(s) and re.search(OLD["object_id"], s) for s in objs) and not v_obj.is_valid("obj_01"))
    labels = ["a", "office chair", "caf\u00e9", "\u6905\u5b50", "chair (blue)", "lamp/2", "K\u00fchlschrank"]
    check("O23: one-character, multiword, Unicode and punctuated labels stay valid, as before",
          all(v_label.is_valid(s) and re.search(OLD["label"], s) for s in labels),
          str([s for s in labels if not v_label.is_valid(s)]))

    free = copy.deepcopy(command)
    free["text"] = "Inspect the box\nnext to the table\n"
    snapshot = copy.deepcopy(free)
    found = validate.validate([("scene", scene), ("command", free), ("category_map", cmap)])
    check("O23: command text is free text, so an embedded and a final newline stay valid (no issue with its scene and "
          "map), and validation leaves the record unchanged", not found and free == snapshot, f"got {found[:2]}")

    script = REPO / "grounding" / "contract" / "validate.py"
    with tempfile.TemporaryDirectory() as tmp:
        bad, good = Path(tmp) / "scene.final_lf.json", Path(tmp) / "scene.json"
        rec = copy.deepcopy(scene)
        rec["scene_id"] += "\n"
        bad.write_text(json.dumps(rec), encoding="utf-8")
        good.write_text(json.dumps(scene), encoding="utf-8")
        escaped = '"scene_id": "fixture.a17\\n"' in bad.read_text(encoding="utf-8")
        bad_run = subprocess.run([sys.executable, str(script), str(bad)], capture_output=True, text=True)
        good_run = subprocess.run([sys.executable, str(script), str(good)], capture_output=True, text=True)
    check("O23: the command line on a well-formed file whose scene_id holds an escaped final newline exits 1 with "
          "E_SCHEMA_PATTERN at $.scene_id, not a parse error",
          escaped and bad_run.returncode == 1 and f"{bad}: $.scene_id: E_SCHEMA_PATTERN" in bad_run.stdout
          and "E_PARSE" not in bad_run.stdout, f"exit {bad_run.returncode}; {last_line(bad_run.stdout)}")
    check("O23: the same file without the newline still exits 0 (with only its missing-map warning)",
          good_run.returncode == 0 and "W_NOT_CROSS_CHECKED" in good_run.stdout,
          f"exit {good_run.returncode}; {last_line(good_run.stdout)}")
    check("O23: no record and no fixture file was changed by any of these validations",
          [scene, restricted, command, cmap] == originals and digest(fixture_paths) == before)


def main() -> int:
    valid = sorted(VALID.glob("*.json"))
    invalid = sorted(INVALID.glob("*.json"))
    before = digest(valid + invalid)

    # 1. The schemas
    schemas = {name: json.loads((REPO / "schemas" / name).read_text(encoding="utf-8"))
               for name in validate.SCHEMAS.values()}
    for name, schema in schemas.items():
        try:
            jsonschema.Draft202012Validator.check_schema(schema)
            check(f"schema {name} is valid JSON Schema 2020-12", True)
        except jsonschema.SchemaError as e:
            check(f"schema {name} is valid JSON Schema 2020-12", False, e.message)
    scene_defs = schemas["scene.v1.json"]["$defs"]
    command_defs = schemas["command-context.v1.json"]["$defs"]
    shared = sorted(set(scene_defs) & set(command_defs))
    differ = [d for d in shared if scene_defs[d] != command_defs[d]]
    check("scene and command-context schemas share identical common definitions", not differ,
          f"{len(shared)} shared" + (f"; different: {differ}" if differ else ""))

    # 2. Valid fixtures, alone and together
    for path in valid:
        found = errors(validate.validate_paths([path]))
        check(f"valid/{path.name} passes alone", not found, "; ".join(i.line() for i in found[:3]))
    together = validate.validate_paths(valid)
    check("all valid fixtures pass together with no warning", not together,
          "; ".join(i.line() for i in together[:3]))

    # 3. Invalid fixtures, one rule each
    covered = set()
    for path in invalid:
        code = path.name.split("__")[0]
        covered.add(code)
        partners = [VALID / PARTNERS[code]] if code in PARTNERS else []
        found = errors(validate.validate_paths(partners + [path]))
        codes = sorted({i.code for i in found})
        only_this_file = all(i.file == str(path) for i in found)
        check(f"invalid/{path.name} fails with exactly {code}", codes == [code] and only_this_file,
              f"got {codes}" + ("" if only_this_file else " (also in a partner file)"))
    missing = sorted(set(checks.CODES) - NO_FIXTURE - covered)
    unknown = sorted(covered - set(checks.CODES))
    check("every error code has an invalid fixture, and every fixture's code exists", not missing and not unknown,
          f"missing {missing}, unknown {unknown}" if missing or unknown else f"{len(covered)} codes covered")

    # 4. Errors carry file, path, code and message; warnings when a partner is absent
    sample = validate.validate_paths([INVALID / "E_QUAT_NORM__scaled.json"])
    check("an error names its file, JSON path, code and message",
          bool(sample) and all(i.file and i.path.startswith("$") and i.code in checks.CODES and i.message
                               for i in sample), sample[0].line() if sample else "no issue")
    alone = validate.validate_paths([VALID / "command.a17.c001.json"])
    check("a command validated without its scene passes with a not-cross-checked warning",
          not errors(alone) and [i.code for i in alone] == ["W_NOT_CROSS_CHECKED"],
          "; ".join(i.line() for i in alone))

    # 5. Cases built in memory: dataset poses and full rotations, which no valid fixture uses
    scene = load("scene.a17.annotated.json")
    cmd = load("command.a17.c001.json")
    dataset_pose = copy.deepcopy(cmd)
    dataset_pose["sources"].append({"source_id": "ds.example", "kind": "dataset",
                                    "description": "In-memory test case only", "release": "test"})
    pose = dataset_pose["user_pose"]
    pose["pose_kind"] = "dataset_viewpoint"
    pose["heading_xy"]["heading_source"] = "camera_yaw"
    for k in ("position_m", "heading_xy"):
        pose[k]["evidence"]["source"] = "ds.example"
    pose["rotation_xyzw"] = {"state": "known", "value": [0.0, 0.0, 0.0348995, 0.9993908], "support": "full",
                             "evidence": {"kind": "annotated", "source": "ds.example", "assumptions": []}}
    found = errors(validate.validate([("scene", scene), ("dataset-pose command", dataset_pose)]))
    check("a dataset_viewpoint pose annotated by a dataset source, with a full rotation, passes", not found,
          "; ".join(i.line() for i in found[:3]))
    inferred = copy.deepcopy(scene)
    inferred["objects"][0]["geometry"]["center_m"]["evidence"]["kind"] = "inferred"
    found = errors(validate.validate([("inferred centre", inferred)]))
    check("an inferred value from a fixture source is allowed in a scene", not found,
          "; ".join(i.line() for i in found[:3]))
    bad = copy.deepcopy(scene)
    bad["objects"][0]["geometry"]["center_m"]["value"] = [float("inf"), 1.4, 0.37]
    codes = sorted({i.code for i in errors(validate.validate([("in-memory infinity", bad)]))})
    check("an infinite number built in memory fails with E_NONFINITE", codes == ["E_NONFINITE"], f"got {codes}")

    # 6. Malformed record types (A2.1a correction): each fails with E_RECORD_TYPE and nothing raises
    shapes = [None, [], 1, "scene", True, {}, {"schema_version": 1}, {"record_type": "scene"},
              {"record_type": [], "schema_version": 1}, {"record_type": {}, "schema_version": 1},
              {"record_type": "scene", "schema_version": True}, {"record_type": "scene", "schema_version": 1.0},
              {"record_type": "scene", "schema_version": [1]}]
    wrong = []
    for shape in shapes:
        try:
            codes = sorted({i.code for i in validate.validate([("shape", shape)])})
        except Exception as e:  # noqa: BLE001  (a crash is exactly what this check looks for)
            codes = [f"raised {type(e).__name__}"]
        if codes != ["E_RECORD_TYPE"]:
            wrong.append(f"{shape!r} gave {codes}")
    check(f"{len(shapes)} malformed record types fail with E_RECORD_TYPE without raising", not wrong, "; ".join(wrong))

    # 7. Read-only
    check("validation changed no fixture file", digest(valid + invalid) == before)

    # 8. Command line (details show only the validator's summary line)
    script = REPO / "grounding" / "contract" / "validate.py"

    def run(*paths):
        return subprocess.run([sys.executable, str(script)] + [str(p) for p in paths], capture_output=True, text=True)

    ok_run = run(*valid)
    bad_run = run(INVALID / "E_QUAT_NORM__scaled.json")
    null_run = run(INVALID / "E_RECORD_TYPE__null_file.json")
    check("the command line exits 0 on the valid fixtures", ok_run.returncode == 0,
          f"exit {ok_run.returncode}; {last_line(ok_run.stdout)}")
    check("the command line exits 1 and names the code on an invalid fixture",
          bad_run.returncode == 1 and "E_QUAT_NORM" in bad_run.stdout,
          f"exit {bad_run.returncode}; {last_line(bad_run.stdout)}")
    check("the command line exits 1 on a file holding only null (A2.1a correction)",
          null_run.returncode == 1 and "E_RECORD_TYPE" in null_run.stdout,
          f"exit {null_run.returncode}; {last_line(null_run.stdout)}")

    # 9. O23 (D73)
    check_o23(valid + invalid, before)

    print(f"{'FAILED: ' + ', '.join(FAILED) if FAILED else 'all checks passed'}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
