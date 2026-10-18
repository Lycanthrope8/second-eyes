"""Scene format v3 (A4 Delivery 1): the perceived profile and the association rules, branch by branch, and v1 and v2
unchanged. The one-rule fixtures in fixtures/contract/ cover each code once; these cases cover the rest of each rule.
Every expected code is written from docs/scene-contract.md, never read from the code under test.

    python grounding/tests/test_contract_v3.py
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "grounding" / "contract"))
import validate  # noqa: E402

FIXTURES = REPO / "grounding" / "tests" / "fixtures" / "contract" / "valid"
FAILED, PASSED = [], []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASSED if ok else FAILED).append(name)
    print(("  ok    " if ok else "  FAIL  ") + name + ("" if ok or not detail else f"  [{detail}]"))


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def found(record: dict) -> list:
    """(code, path) of every error, sorted."""
    return sorted((i.code, i.path) for i in validate.validate([("scene", record)]) if i.is_error)


def main() -> int:
    base = load("scene.bedroom.perceived_v3.json")
    C = copy.deepcopy

    print("The valid scene")
    check("the bedroom scene passes alone", found(base) == [], str(found(base)))

    print("Association: views and status")
    s = C(base); s["objects"][0]["observation"]["views"] = {"keyframes": 2, "positions": 3}
    check("more positions than keyframes", found(s) == [("E_ASSOCIATION", "$.objects[0].observation.views")], str(found(s)))
    s = C(base); s["objects"][4]["observation"]["association"]["status"] = "multi_position"
    check("multi_position from one position",
          found(s) == [("E_ASSOCIATION", "$.objects[4].observation.association.status")], str(found(s)))
    s = C(base); s["objects"][0]["observation"]["association"]["may_duplicate"] = ["obj_004"]
    s["objects"][3]["observation"]["association"]["may_duplicate"] = ["obj_001"]
    check("a multi_position object (and its partner) can't name possible duplicates",
          found(s) == [("E_ASSOCIATION", "$.objects[0].observation.association.may_duplicate"),
                       ("E_ASSOCIATION", "$.objects[3].observation.association.may_duplicate")], str(found(s)))
    s = C(base); s["objects"][1]["observation"]["association"]["may_duplicate"] = []
    check("a possible duplicate with no partner (and its partner, now not listed back)",
          found(s) == [("E_ASSOCIATION", "$.objects[1].observation.association.may_duplicate"),
                       ("E_ASSOCIATION", "$.objects[2].observation.association.may_duplicate[0]")], str(found(s)))

    print("Association: the partners")
    s = C(base); s["objects"][1]["observation"]["association"]["may_duplicate"] = ["obj_003", "obj_002"]
    check("an object can't duplicate itself",
          found(s) == [("E_ASSOCIATION", "$.objects[1].observation.association.may_duplicate[1]")], str(found(s)))
    s = C(base); s["objects"][1]["observation"]["association"]["may_duplicate"] = ["obj_003", "obj_009"]
    check("a partner must be an object of the scene",
          found(s) == [("E_ASSOCIATION", "$.objects[1].observation.association.may_duplicate[1]")], str(found(s)))

    print("The perceived profile")
    s = C(base); s["objects"][3]["attributes"]["colours"]["evidence"] = {"kind": "inferred", "source": "sizes.fx", "assumptions": []}
    check("colours inferred from the size table, not the scan",
          found(s) == [("E_PROFILE_PERCEIVED", "$.objects[3].attributes.colours")], str(found(s)))
    s = C(base); s["objects"][0]["observation"]["capture_time"]["evidence"]["kind"] = "inferred"
    check("a capture time is measured, not inferred",
          found(s) == [("E_PROFILE_PERCEIVED", "$.objects[0].observation.capture_time")], str(found(s)))
    s = C(base); s["objects"][0]["geometry"]["size_m"] = {"state": "unknown", "reason": "not_estimated", "source": "scan.fx"}
    check("an unknown size gives no_permitted_prior, not not_estimated",
          found(s) == [("E_PROFILE_PERCEIVED", "$.objects[0].geometry.size_m")], str(found(s)))
    s = C(base)
    s["objects"][0]["geometry"]["size_m"] = {"state": "known", "value": [0.55, 0.45, 0.88], "evidence": {"kind": "inferred", "source": "scan.fx", "assumptions": []}}
    s["objects"][0]["geometry"]["rotation_xyzw"] = {"state": "known", "value": [0, 0, 0.3826834324, 0.9238795325], "support": "yaw_only",
                                                    "evidence": {"kind": "inferred", "source": "scan.fx", "assumptions": []}}
    check("a 3D box from the scan: its size and turn inferred from the sensor capture", found(s) == [], str(found(s)))
    s = C(base); s["objects"][0]["geometry"]["size_m"]["evidence"] = {"kind": "measured", "source": "scan.fx", "assumptions": []}
    check("a box's size is inferred from the scan, not measured", found(s) == [("E_PROFILE_PERCEIVED", "$.objects[0].geometry.size_m")], str(found(s)))
    s = C(base); s["objects"][0]["geometry"]["center_m"]["evidence"]["kind"] = "measured"
    check("a centre is inferred from the scan, not measured",
          found(s) == [("E_PROFILE_PERCEIVED", "$.objects[0].geometry.center_m")], str(found(s)))
    s = C(base); s["objects"][0]["geometry"]["center_m"]["evidence"]["kind"] = "annotated"
    check("nothing in a perceived scene is annotated: no source of v3 can stand behind it",
          found(s) == [("E_EVIDENCE_SOURCE", "$.objects[0].geometry.center_m.evidence"),
                       ("E_PROFILE_PERCEIVED", "$.objects[0].geometry.center_m")], str(found(s)))
    s = C(base); s["objects"][4]["observation"]["views"] = {"keyframes": 1, "positions": 1}
    s["objects"][4]["geometry"]["center_m"] = {"state": "known", "value": [3.0, 0.5, 0.8],
                                              "evidence": {"kind": "inferred", "source": "scan.fx", "assumptions": []}}
    check("a single-position centre without the size prior's assumption",
          found(s) == [("E_PROFILE_PERCEIVED", "$.objects[4].geometry.center_m.evidence.assumptions")], str(found(s)))

    print("Formats 1 and 2 unchanged")
    v1 = load("scene.a17.annotated.json")
    s = C(v1); s["objects"][0]["geometry"]["center_m"]["evidence"]["kind"] = "measured"
    msgs = [i.message for i in validate.validate([("scene", s)]) if i.code == "E_EVIDENCE_SOURCE"]
    check("v1 still has no source kind for measured values", msgs == ["v1 has no source kind for measured values"], str(msgs))
    s = C(v1); k = len(s["sources"])
    s["sources"].append({"source_id": "scan.fx", "kind": "sensor_capture", "description": "a scan", "release": "x"})
    check("v1 doesn't take a sensor_capture source", found(s) == [("E_SCHEMA_ENUM", f"$.sources[{k}].kind")], str(found(s)))
    s = C(v1); s["evidence_profile"] = "perceived"
    check("v1 doesn't take the perceived profile", ("E_SCHEMA_ENUM", "$.evidence_profile") in found(s), str(found(s)))
    v2 = load("scene.a17.dataset_v2.json")
    check("the v2 fixture still passes", found(v2) == [], str(found(v2)))
    s = C(base); s["schema_version"] = 2
    check("a perceived scene labelled v2 fails v2's schema", any(c.startswith("E_SCHEMA_") for c, _ in found(s)), str(found(s)[:3]))

    print("The v3 schema is v2's except where docs/scene-contract.md says")
    s2 = json.loads((REPO / "schemas" / "scene.v2.json").read_text(encoding="utf-8"))
    s3 = json.loads((REPO / "schemas" / "scene.v3.json").read_text(encoding="utf-8"))
    for s in (s2, s3):   # the documented differences: identity, version, profile, source kinds, frame, views and association
        for k in ("$id", "title", "description"):
            s.pop(k)
        s["properties"].pop("schema_version"); s["properties"].pop("evidence_profile")
        s["$defs"]["source"]["properties"].pop("kind"); s["$defs"].pop("coordinateFrame")
        obs = s["$defs"]["object"]["properties"]["observation"]
        obs["properties"].pop("views", None); obs["properties"].pop("association", None)
        obs["required"] = [r for r in obs["required"] if r not in ("views", "association")]
    check("every other part of scene.v3.json equals scene.v2.json", s2 == s3)

    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
