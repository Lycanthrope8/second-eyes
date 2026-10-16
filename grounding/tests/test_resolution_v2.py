"""Format 2 of the grounding query and result (A4 Delivery 1) and the association guard.

Format 2 differs from format 1 only where docs/resolution.md says; format 1 stays the format of every non-perceived
scene. The resolver runs end to end on the perceived bedroom fixture, and the guard turns a ranking that picks one of a
possible duplicate pair into the ambiguity outcome. Expected answers come from fixtures/association/expected.json,
worked out by hand from the chairs' centres, never from the code under test.

    python grounding/tests/test_resolution_v2.py
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import jsonschema  # noqa: E402
from grounding.a24 import language as L  # noqa: E402
from grounding.resolution import association, resolve  # noqa: E402
from grounding.resolution import validate as RV  # noqa: E402

FX = REPO / "grounding" / "tests" / "fixtures"
AS = FX / "association"
FAILED, PASSED = [], []


def check(name, ok, detail=""):
    (PASSED if ok else FAILED).append(name)
    print(("  ok    " if ok else "  FAIL  ") + name + ("" if ok or not detail else f"  [{detail}]"))


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def schema(name):
    s = load(REPO / "schemas" / name)
    return jsonschema.validators.validator_for(s)(s)


def strip(s, result):
    s = copy.deepcopy(s)
    for k in ("$id", "title", "description"):
        s.pop(k)
    s["properties"].pop("schema_version")

    def walk(n):
        if isinstance(n, dict):
            for k, v in n.items():
                if k == "enum" and v in (["annotated", "restricted"], ["annotated", "restricted", "perceived"]):
                    n[k] = ["annotated", "restricted"]
                else:
                    walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)
    walk(s)
    if result:
        s["$defs"]["detail_code"]["enum"] = [c for c in s["$defs"]["detail_code"]["enum"] if c != "uncertain_association"]
    return s


def main() -> int:
    print("Format 2 is format 1 except where documented")
    for name, result in (("grounding-query", False), ("grounding-result", True)):
        v1, v2 = load(REPO / "schemas" / f"{name}.v1.json"), load(REPO / "schemas" / f"{name}.v2.json")
        check(f"{name}.v2.json equals v1 apart from its version, the perceived profile"
              + (" and the uncertain_association code" if result else ""), strip(v1, result) == strip(v2, result))

    scene = load(FX / "contract" / "valid" / "scene.bedroom.perceived_v3.json")
    cmap = load(FX / "contract" / "valid" / "category-map.bedroom.json")
    command = load(AS / "command.bedroom.json")
    a17 = load(FX / "contract" / "valid" / "scene.a17.annotated.json")
    a17_command = load(FX / "contract" / "valid" / "command.a17.c002.json")
    spec = load(REPO / "grounding" / "a24" / "language.v1.json")
    interp = L.interpretation(L.parse_output(" | ".join(["box", "closest 1", "table"]), spec))

    print("Which format a query gets")
    check("an annotated scene still gets a format-1 query", L.query(interp, a17, a17_command)["schema_version"] == 1)
    check("a perceived scene gets a format-2 query", L.query(interp, scene, command)["schema_version"] == 2)
    q2 = load(AS / "query.farthest.json")
    issues, _ = RV.check_query(q2)
    check("a format-2 perceived query passes check_query", issues == [], str([i["code"] for i in issues]))
    issues, _ = RV.check_query(dict(q2, schema_version=1))
    check("the same query labelled format 1 fails: format 1 doesn't take the perceived profile",
          [i["code"] for i in issues] == ["E_QUERY_SCHEMA"], str([i["code"] for i in issues]))
    issues, _ = RV.check_query(dict(q2, schema_version=2.0))
    check("a non-integer 2 is read as format 1, as before", {i["code"] for i in issues} == {"E_QUERY_SCHEMA"},
          str([i["code"] for i in issues]))

    print("The resolver end to end on the bedroom, then the guard")
    config = load(FX / "resolution" / "resolver-config.fixture.json")
    rel, dirs = REPO / "grounding" / "relations" / "relations.v1.json", REPO / "grounding" / "relations" / "directions.v1.json"
    result_schema = schema("grounding-result.v2.json")
    for case in load(AS / "expected.json")["cases"]:
        query = load(AS / case["query"])
        run = resolve(scene, command, query, resolver_config=config, relation_config_path=rel,
                      direction_config_path=dirs, category_maps=[cmap])
        name = case["query"].removeprefix("query.").removesuffix(".json")
        got = (run["processing_status"], (run["result"] or {}).get("status"), (run["result"] or {}).get("target_id"))
        want = ("completed", case["resolver"]["status"], case["resolver"]["target_id"])
        check(f"{name}: the resolver answers {want[1]} {want[2]}", got == want,
              f"{got}; issues {[i['code'] for i in run['issues']][:3]}")
        errors = sorted(e.message[:80] for e in result_schema.iter_errors(run))
        check(f"{name}: the run is a valid format-2 record", run["schema_version"] == 2 and not errors, str(errors[:2]))
        guarded = association.guard(scene, query, run)
        r, g = guarded["result"], case["guarded"]
        got = (r["status"], r["target_id"], r["reason_code"], r["candidates"]["target_ids"], "uncertain_association" in r["reason_codes"])
        want = (g["status"], g["target_id"], g["reason_code"], g["target_ids"], g["uncertain_association"])
        check(f"{name}: after the guard, {g['status']} {g['target_ids']}", got == want, str(got))
        errors = sorted(e.message[:80] for e in result_schema.iter_errors(guarded))
        check(f"{name}: the guarded run is a valid format-2 record", not errors, str(errors[:2]))
        check(f"{name}: the guard leaves the caller's run unchanged", run["result"]["status"] == case["resolver"]["status"])

    print("The guard leaves everything else alone")
    run = resolve(scene, command, load(AS / "query.farthest.json"), resolver_config=config, relation_config_path=rel,
                  direction_config_path=dirs, category_maps=[cmap])
    check("an annotated scene is never changed", association.guard(dict(scene, evidence_profile="annotated"), q2, run) is run)
    amb = copy.deepcopy(run); amb["result"]["status"] = "ambiguous"
    check("an answer that isn't resolved is never changed", association.guard(scene, q2, amb) is amb)
    bad = dict(run, processing_status="invalid_input", result=None)
    check("a run that didn't complete is never changed", association.guard(scene, q2, bad) is bad)

    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
