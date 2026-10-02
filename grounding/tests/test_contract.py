"""Tests of the scene contract's offline validator (A2.1a, D66).

    python grounding/tests/test_contract.py

Needs only Python 3.9+ and jsonschema (grounding/requirements.txt or tools/requirements.txt). Checks: the schemas are
valid JSON Schema 2020-12 and share identical common definitions; every valid fixture passes alone and all pass
together with no warning; every invalid fixture fails with exactly the code in its name, together with its partner
for cross-record rules; every error code has a fixture; a few cases built in memory pass; the validator never
changes a file; the command line exits 0 and 1. Prints one line per check and exits 1 if any fails.
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

    # 6. Read-only
    check("validation changed no fixture file", digest(valid + invalid) == before)

    # 7. Command line
    script = REPO / "grounding" / "contract" / "validate.py"
    ok_run = subprocess.run([sys.executable, str(script)] + [str(p) for p in valid], capture_output=True, text=True)
    bad_run = subprocess.run([sys.executable, str(script), str(INVALID / "E_QUAT_NORM__scaled.json")],
                             capture_output=True, text=True)
    check("the command line exits 0 on the valid fixtures", ok_run.returncode == 0, ok_run.stdout.strip()[-120:])
    check("the command line exits 1 and names the code on an invalid fixture",
          bad_run.returncode == 1 and "E_QUAT_NORM" in bad_run.stdout, bad_run.stdout.strip()[-120:])

    print(f"{'FAILED: ' + ', '.join(FAILED) if FAILED else 'all checks passed'}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
