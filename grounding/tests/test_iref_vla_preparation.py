"""Tests of the A2.2d model-input preparation (D78).

    python grounding/tests/test_iref_vla_preparation.py --import DIR --selection-audit DIR   full acceptance (real tokenizer)
    python grounding/tests/test_iref_vla_preparation.py --bundle DIR                         full acceptance of a bundle the
                                                                                             CLI already published (readback)
    python grounding/tests/test_iref_vla_preparation.py --unit-only                          fixtures; NOT real-token acceptance
    (or: python -m grounding.tests.test_iref_vla_preparation ...)

--import is A2.2a's import root (model_inputs/ and reference_only/); --selection-audit is the accepted A2.2c output for
it. The full mode also needs the pinned measurement environment (transformers 4.57.6, tokenizers 0.22.2): it runs the
real-token calibration, then the full preparation once, then reads its bundle back. Without those inputs and without
--unit-only, the full check is one FAIL, never a skip. Fixture checks use labelled test doubles, never real-token
evidence. Expectations: fixtures/iref_vla_preparation/expectations.json, written before the package existed.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

FX = REPO / "grounding" / "tests" / "fixtures" / "iref_vla_preparation"
EV = REPO / "grounding" / "tests" / "fixtures" / "iref_vla_evaluation"
EXP = json.loads((FX / "expectations.json").read_text(encoding="utf-8"))
TOK_DIR = REPO / "quest-app" / "Assets" / "SecondEyes" / "Models" / "qwen2.5-0.5b-instruct"
MODEL_DESC = REPO / "grounding" / "models" / "qwen2.5-0.5b-instruct.json"
REL_CFG = REPO / "grounding" / "relations" / "relations.v1.json"
DIR_CFG = REPO / "grounding" / "relations" / "directions.v1.json"
VIEWS, FORMATS = ("full_inventory", "source_known_nyu"), ("coordinates_v2", "coordinates_relations_v2")
FAILED, COUNT = [], [0]


def check(name, ok, detail=""):
    COUNT[0] += 1
    print(("PASS  " if ok else "FAIL  ") + name + (f": {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


class CharTokenizer:
    """Test double: one token per character."""
    identity, kind = "test_double.char", "test_double"

    def __init__(self):
        self.calls = 0

    def encode(self, text):
        self.calls += 1
        return [ord(c) for c in text]


class MergingTokenizer(CharTokenizer):
    """Test double: LF followed by '{' is one token, so a prefix ending in LF loses its last token in the full text."""
    identity = "test_double.merge"

    def encode(self, text):
        self.calls += 1
        out, i = [], 0
        while i < len(text):
            if text.startswith("\n{", i):
                out.append(-1)
                i += 2
            else:
                out.append(ord(text[i]))
                i += 1
        return out


class BigTokenizer(CharTokenizer):
    """Test double: every text costs five thousand tokens plus one per character."""
    identity = "test_double.big"

    def encode(self, text):
        self.calls += 1
        return [7] * 5000 + [ord(c) for c in text]


class FailingTokenizer(CharTokenizer):
    identity = "test_double.failing"

    def encode(self, text):
        self.calls += 1
        if self.calls == 5:
            raise RuntimeError("controlled tokenizer failure")
        return [ord(c) for c in text]


def ev(name):
    return json.loads((EV / name).read_text(encoding="utf-8"))


def write_originals(folder, scene, commands, cmap, views, *, reverse=False):
    mi, ro = folder / "model_inputs", folder / "reference_only"
    (mi / "commands").mkdir(parents=True)
    ro.mkdir()
    (mi / "scene.annotated.json").write_text(json.dumps(scene), encoding="utf-8")
    (mi / "category-map.json").write_text(json.dumps(cmap), encoding="utf-8")
    (ro / "inventory-views.json").write_text(json.dumps(views), encoding="utf-8")
    for c in (list(reversed(commands)) if reverse else commands):
        (mi / "commands" / f"{c['command_id']}.json").write_text(json.dumps(c), encoding="utf-8")
    return {"scene": mi / "scene.annotated.json", "commands": mi / "commands", "category_map": mi / "category-map.json",
            "inventory_views": ro / "inventory-views.json"}


def fixture_commands(texts=None):
    template = ev("commands.f1.json")[0]
    texts = texts or EXP["fixture_commands"]
    return [dict(copy.deepcopy(template), command_id=f"fixture.iref_prep.{k}", text=t) for k, t in texts.items()]


def fixture_run(P, tmp, *, tokenizer=None, scene=None, commands=None, reverse=False, name="run", work_cap=None, audit=None):
    """Originals, the accepted A2.2c audit of them, then the preparation; returns (paths, out)."""
    tmp = Path(tmp)
    sub = importlib.import_module("grounding.subscenes.iref_vla.audit")
    paths = write_originals(tmp / f"{name}-in", scene or ev("scene.f1.json"), commands or fixture_commands(),
                            ev("category-map.fixture.json"), ev("views.f1.json"), reverse=reverse)
    audit_dir = audit or tmp / f"{name}-audit"
    if audit is None:
        sub.run_audit(**paths, out=audit_dir, sample_only=False)
    out = tmp / f"{name}-out"
    kw = {} if work_cap is None else {"work_cap": work_cap}
    P.run_preparation(**paths, selection_audit=audit_dir, relation_config=REL_CFG, direction_config=DIR_CFG,
                      tokenizer_dir=TOK_DIR, model_description=MODEL_DESC, out=out,
                      tokenizer=tokenizer or CharTokenizer(), sample_only=False, **kw)
    return paths, audit_dir, out


def jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines()]


def outcome(E, fn):
    try:
        return ("ok", fn())
    except E.EvaluationInputError as e:
        return ("rejected", sorted({i["code"] for i in e.issues}))
    except Exception as e:  # noqa: BLE001
        return ("crashed", f"{type(e).__name__}: {e}"[:140])


# ------------------------------------------------------------------------------------------------ constants
def check_constants(P):
    print("-- constants, the work formula, the wrapper and the ceilings")
    from grounding.serialization.constants import work_slots
    got = {k: work_slots(int(k)) for k in EXP["work_slots"]}
    check("the accepted work formula: n = 0, 1, 2, 3, 6, 10 give 0, 2, 22, 63, 342, 1,190", got == EXP["work_slots"], str(got))
    check("the preparation's work cap is the accepted helper's value at n = 10: 1,190", P.WORK_CAP == 1190 == work_slots(10))
    check("ceilings 1,024, 2,048 and 4,096; formats fixed", list(P.CEILINGS) == EXP["ceilings"] and tuple(P.FORMATS) == FORMATS)
    check("the system message is the accepted sizing audit's, unchanged", P.SYSTEM_MESSAGE == EXP["system_message"])
    w = EXP["wrapper"]
    lit = w[0] + EXP["system_message"] + w[2] + "DOC\n" + w[4]
    check("the wrapper is the literal sizing wrapper; the document's final LF sits before <|im_end|>",
          P.wrap("DOC\n") == lit and P.wrap("DOC\n").count("<|im_start|>assistant\n") == 1)
    res = {c: (P.ceiling_results(c - 1)[str(c)], P.ceiling_results(c)[str(c)], P.ceiling_results(c + 1)[str(c)])
           for c in EXP["ceilings"]}
    check("each ceiling: below and equal are within, above exceeds",
          all(v == ("within_input_ceiling", "within_input_ceiling", "exceeds_input_ceiling") for v in res.values()), str(res))
    desc = json.loads(MODEL_DESC.read_text(encoding="utf-8"))
    check("the model description's template, through grounding.scene.formatted, equals the literal wrapper",
          P.check_model_description(desc) is None)
    E = importlib.import_module("grounding.evaluation.iref_vla")
    bad = dict(copy.deepcopy(desc), hf_revision="0" * 40)
    check("a model description with another revision is refused",
          outcome(E, lambda: P.check_model_description(bad))[0] == "rejected")
    a = [5, 6, 7, 8]
    check("common token-ID prefix: actual IDs, not subtracted counts",
          (P.common_prefix_length(a, [5, 6, 9, 8]), P.common_prefix_length(a, a + [1]), P.common_prefix_length([], a)) == (2, 4, 0))
    grp = P.prefix_reuse([([1, 2, 3], [1, 2, 3, 4]), ([1, 2, 3], [1, 2, 9, 4])])
    check("a group's reusable prefix is its minimum compatible length", grp["group_reusable_prefix_tokens"] == 2
          and grp["compatible"] == [3, 2], str(grp))


# ------------------------------------------------------------------------------------------- materialization
def check_materialization(P):
    print("-- materialization on fixture F1 (test-double tokens)")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        scene_before, cmds = ev("scene.f1.json"), fixture_commands()
        cmds_before = copy.deepcopy(cmds)
        paths, audit_dir, out = fixture_run(P, tmp, commands=cmds)
        index, meas = jsonl(out / "preparation-index.jsonl"), jsonl(out / "measurements.jsonl")
        want = [(f["parent_command_id"], f["view_id"]) for f in EXP["fixture"]]
        check("one index row per command and view, in order", [(r["parent_command_id"], r["view_id"]) for r in index] == want)
        check("two measurement rows per index row, in the fixed format order",
              [(m["parent_command_id"], m["view_id"], m["format"]) for m in meas] == [(p, v, f) for p, v in want for f in FORMATS])
        rows = {(r["parent_command_id"], r["view_id"]): r for r in index}
        ok_scene, ok_cmd = True, True
        originals = {o["object_id"]: o for o in scene_before["objects"]}
        for f in EXP["fixture"]:
            r = rows[(f["parent_command_id"], f["view_id"])]
            if f["source_status"] != "fits":
                ok_scene &= r["preparation_status"] == "unassessed_parse" and r["derived_scene_id"] is None
                continue
            s = json.loads((out / r["scene_path"]).read_text(encoding="utf-8"))
            ok_scene &= (s["scene_id"] == f["selection_id"] and s["scene_revision"] == 0 and type(s["scene_revision"]) is int
                         and [o["object_id"] for o in s["objects"]] == f["objects"]
                         and all(o == originals[o["object_id"]] for o in s["objects"])
                         and {k: v for k, v in s.items() if k not in ("scene_id", "scene_revision", "objects")}
                         == {k: v for k, v in scene_before.items() if k not in ("scene_id", "scene_revision", "objects")})
            c = json.loads((out / r["command_path"]).read_text(encoding="utf-8"))
            parent = next(x for x in cmds_before if x["command_id"] == f["parent_command_id"])
            rest = lambda x: {k: v for k, v in x.items() if k not in ("command_id", "scene_id", "scene_revision", "user_pose")}  # noqa: E731
            pose = lambda x: {k: v for k, v in x["user_pose"].items() if k != "scene_revision"}  # noqa: E731
            ok_cmd &= (c["command_id"] == f["derived_command_id"] == r["derived_command_id"] and c["scene_id"] == f["selection_id"]
                       and c["scene_revision"] == 0 and c["user_pose"]["scene_revision"] == 0 and c["text"] == parent["text"]
                       and rest(c) == rest(parent) and pose(c) == pose(parent))
        check("scenes: ID = selection ID, revision 0, exactly the planned objects, deep-equal records, all else unchanged",
              ok_scene)
        check("commands: the independently hashed derived ID, rebound scene and revisions, text and all else unchanged", ok_cmd)
        empty = rows[("fixture.iref_prep.c03", "source_known_nyu")]
        es = json.loads((out / empty["scene_path"]).read_text(encoding="utf-8")) if empty["scene_path"] else {}
        check("an absent-category empty selection is materialized with zero objects, unlike an unassessed command",
              empty["preparation_status"] == "materialized" and es.get("objects") == []
              and rows[("fixture.iref_prep.c04", "full_inventory")]["preparation_status"] == "unassessed_parse")
        a, b = rows[("fixture.iref_prep.c01", "full_inventory")], rows[("fixture.iref_prep.c02", "full_inventory")]
        check("two commands with one membership share a scene, with distinct command identities",
              a["scene_path"] == b["scene_path"] and a["derived_command_id"] != b["derived_command_id"])
        per = {}
        for m in meas:
            if m["status"] == "rendered":
                per.setdefault((m["parent_command_id"], m["view_id"]), set()).add(m["derived_command_id"])
        check("each pair's two formats share one derived command ID", all(len(v) == 1 for v in per.values()) and per)
        scenes = sorted(p.name for p in (out / "model_records" / "scenes").iterdir())
        check("one scene file per distinct fitting selection", scenes == sorted({f["selection_id"] + ".json" for f in EXP["fixture"]
                                                                                  if f["source_status"] == "fits"}))
        check("the category map is copied unchanged", json.loads((out / "model_records/category-map.json").read_text(
            encoding="utf-8")) == ev("category-map.fixture.json"))
        check("the source scene and commands are not mutated, in memory or on disk",
              cmds == cmds_before and json.loads(paths["scene"].read_text(encoding="utf-8")) == scene_before)
        texts_ok = True
        for m in meas:
            if m["status"] != "rendered":
                texts_ok &= m["paths"] is None and m["tokens"] is None
                continue
            d = {k: (out / v).read_bytes().decode("utf-8") for k, v in m["paths"].items()}
            texts_ok &= d["document"] == d["static_prefix"] + d["dynamic_suffix"] and d["prompt"] == P.wrap(d["document"])
        check("rendered: document = static prefix + dynamic suffix, prompt = wrap(document); non-rendered rows carry no "
              "paths or measurements", texts_ok)
        from grounding.serialization import serialize
        r = rows[("fixture.iref_prep.c01", "full_inventory")]
        s, c = (json.loads((out / r[k]).read_text(encoding="utf-8")) for k in ("scene_path", "command_path"))
        direct = {f: serialize(s, c, format=f, relation_config_path=REL_CFG, direction_config_path=DIR_CFG,
                               max_relation_work_units=None if f == "coordinates_v2" else 1190,
                               category_maps=[ev("category-map.fixture.json")]).document for f in FORMATS}
        stored = {m["format"]: (out / m["paths"]["document"]).read_text(encoding="utf-8") for m in meas
                  if m["derived_command_id"] == r["derived_command_id"] and m["view_id"] == "full_inventory"}
        check("documents equal direct calls of the accepted serializer, both formats", stored == direct)
        coords = [m for m in meas if m["format"] == "coordinates_v2" and m["status"] == "rendered"]
        check("coordinates rows record no relation work", coords and all(m["work"]["planned_slots"] == 0
                                                                          and m["work"]["actual_total"] == 0 for m in coords))
        tele = all(sum(b["marginal_tokens"] for b in m["blocks"]) == m["tokens"]["input_tokens"] for m in meas
                   if m["status"] == "rendered")
        bounded = all(0 <= m["tokens"]["tokens_after_group_prefix"] <= m["tokens"]["input_tokens"]
                      and m["tokens"]["group_reusable_prefix_tokens"] <= m["tokens"]["prefix_compatible_tokens"]
                      for m in meas if m["status"] == "rendered")
        check("block marginals telescope to the full count; reuse figures never exceed it", tele and bounded)
        check("test-double measurements are labelled test_double", all(m["measurement_kind"] == "test_double" for m in meas
                                                                        if m["status"] == "rendered"))
        check("the published bundle passes readback", P.verify_bundle(out) == [], str(P.verify_bundle(out)[:2]))
        rev_paths, _, rev_out = fixture_run(P, tmp, name="rev", reverse=True,
                                            scene=dict(scene_before, objects=list(reversed(scene_before["objects"]))))
        same = all((out / f).read_bytes() == (rev_out / f).read_bytes() for f in ("preparation-index.jsonl", "measurements.jsonl"))
        same &= sorted(p.name for p in (out / "model_records/scenes").iterdir()) == sorted(
            p.name for p in (rev_out / "model_records/scenes").iterdir()) and all(
            (out / "model_records/scenes" / p.name).read_bytes() == p.read_bytes()
            for p in (rev_out / "model_records/scenes").iterdir())
        check("reordered objects and command files: identical selection IDs, scene bytes, index and measurements", same)


def check_budgets(P):
    print("-- object and work budgets, ceilings, the prefix seam and failures")
    E = importlib.import_module("grounding.evaluation.iref_vla")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        big = ev("scene.f1.json")
        chair = next(o for o in big["objects"] if o["object_id"] == "obj_002")
        for i in range(9):
            o = copy.deepcopy(chair)
            o["object_id"], o["source_ref"]["source_object_id"] = f"obj_{101 + i}", str(101 + i)
            big["objects"].append(o)
        views = ev("views.f1.json")
        views["views"][0]["included_object_ids"] += [f"obj_{101 + i}" for i in range(9)]
        views["views"][1]["included_object_ids"] += [f"obj_{101 + i}" for i in range(9)]
        sub = importlib.import_module("grounding.subscenes.iref_vla.audit")
        cmds = fixture_commands({"c01": "the chair that is closest to the table"})
        paths = write_originals(tmp / "big-in", big, cmds, ev("category-map.fixture.json"), views)
        sub.run_audit(**paths, out=tmp / "big-audit", sample_only=False)
        P.run_preparation(**paths, selection_audit=tmp / "big-audit", relation_config=REL_CFG, direction_config=DIR_CFG,
                          tokenizer_dir=TOK_DIR, model_description=MODEL_DESC, out=tmp / "big-out", tokenizer=CharTokenizer(),
                          sample_only=False)
        idx = jsonl(tmp / "big-out/preparation-index.jsonl")
        check("thirteen required objects: over_object_budget, count and selection ID kept, no records or texts",
              all(r["preparation_status"] == "over_object_budget" and r["required_object_count"] >= 13
                  and r["selection_id"] and r["derived_command_id"] is None for r in idx)
              and not (tmp / "big-out/model_records/scenes").exists() or not any((tmp / "big-out/model_records/scenes").iterdir()))
        _, _, low = fixture_run(P, tmp, name="lowcap", work_cap=10)
        m = jsonl(low / "measurements.jsonl")
        aug = [x for x in m if x["format"] == "coordinates_relations_v2" and x["view_id"] == "full_inventory"
               and x["parent_command_id"].endswith("c01")]
        check("a lower test cap gives work_budget_exceeded with no model text", aug and aug[0]["status"] == "work_budget_exceeded"
              and aug[0]["paths"] is None)
        _, _, bigtok = fixture_run(P, tmp, name="bigtok", tokenizer=BigTokenizer())
        m = [x for x in jsonl(bigtok / "measurements.jsonl") if x["status"] == "rendered"]
        check("an input over every ceiling stays under diagnostic_inputs, every ceiling marked exceeded", m and all(
            set(x["ceilings"].values()) == {"exceeds_input_ceiling"} and (bigtok / x["paths"]["prompt"]).exists() for x in m))
        _, _, merged = fixture_run(P, tmp, name="merge", tokenizer=MergingTokenizer())
        m = [x for x in jsonl(merged / "measurements.jsonl") if x["status"] == "rendered"]
        check("a token merging across the static/dynamic seam: compatible prefix one short of the prefix's own tokens",
              m and all(x["tokens"]["prefix_compatible_tokens"] == x["tokens"]["prefix_tokens"] - 1
                        and x["tokens"]["prefix_fully_compatible"] is False for x in m))
        failing = FailingTokenizer()
        r = outcome(E, lambda: fixture_run(P, tmp, name="fail", tokenizer=failing))
        check("an unexpected tokenizer error aborts the run and publishes nothing", r[0] == "crashed"
              and not (tmp / "fail-out").exists(), str(r))
        om = importlib.import_module("grounding.evaluation.iref_vla.output")
        real, calls = om._write_file, [0]

        def broken(path, data):
            calls[0] += 1
            if calls[0] == 7:
                raise OSError("simulated disk failure")
            real(path, data)
        om._write_file = broken
        try:
            r = outcome(E, lambda: fixture_run(P, tmp, name="wfail"))
        finally:
            om._write_file = real
        check("a write failure publishes nothing", r[0] == "crashed" and not (tmp / "wfail-out").exists()
              and not any(".partial-" in p.name for p in tmp.iterdir()), str(r))
        paths, audit_dir, out = fixture_run(P, tmp, name="exists")
        r = outcome(E, lambda: P.run_preparation(**paths, selection_audit=audit_dir, relation_config=REL_CFG,
                                                 direction_config=DIR_CFG, tokenizer_dir=TOK_DIR, model_description=MODEL_DESC,
                                                 out=out, tokenizer=CharTokenizer(), sample_only=False))
        check("an existing output folder is refused", r == ("rejected", ["E_EVAL_OUTPUT_EXISTS"]), str(r))


def forge(audit_dir, mutate_rows=None, mutate_manifest=None, raw=None):
    """Tamper with an A2.2c audit, then recompute its manifest's artifact hashes as an attacker would."""
    sel = audit_dir / "selections.jsonl"
    if raw is not None:
        sel.write_bytes(raw)
    elif mutate_rows:
        rows = jsonl(sel)
        mutate_rows(rows)
        sel.write_bytes("".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows).encode("utf-8"))
    m = json.loads((audit_dir / "manifest.json").read_text(encoding="utf-8"))
    for n in ("selections.jsonl", "summary.json", "report.md"):
        m["outputs"][n] = hashlib.sha256((audit_dir / n).read_bytes()).hexdigest()
    if mutate_manifest:
        mutate_manifest(m)
    (audit_dir / "manifest.json").write_text(json.dumps(m, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def check_forgeries(P):
    print("-- a forged or malformed selection audit fails before any render or token call")
    E = importlib.import_module("grounding.evaluation.iref_vla")
    import grounding.serialization as S
    pm = importlib.import_module("grounding.preparation.iref_vla.prepare")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        paths, audit_dir, _ = fixture_run(P, tmp, name="base")
        fits = lambda r: r["status"] == "fits" and r["retained_object_ids"]  # noqa: E731
        cases = {
            "a forged retained ID": dict(mutate_rows=lambda rs: next(r for r in rs if fits(r))["retained_object_ids"].__setitem__(0, "obj_007")),
            "a dropped member": dict(mutate_rows=lambda rs: next(r for r in rs if fits(r))["retained_object_ids"].pop()),
            "a changed status": dict(mutate_rows=lambda rs: next(r for r in rs if fits(r)).__setitem__("status", "over_budget")),
            "a changed selection hash": dict(mutate_rows=lambda rs: next(r for r in rs if fits(r)).__setitem__("selection_id", "iref.subset." + "0" * 64)),
            "a missing row": dict(mutate_rows=lambda rs: rs.pop()),
            "an extra row": dict(mutate_rows=lambda rs: rs.append(copy.deepcopy(rs[0]))),
            "a version of 1.0": dict(mutate_rows=lambda rs: rs[0].__setitem__("format_version", 1.0)),
            "a version of true": dict(mutate_rows=lambda rs: rs[0].__setitem__("format_version", True)),
            "truncated JSONL": dict(raw=b'{"format_version":1,"parent_command'),
            "a non-object record": dict(raw=b"7\n"),
            "a wrong manifest role": dict(mutate_manifest=lambda m: m.__setitem__("record_type", "iref_subscene_summary")),
        }
        renders, real_serialize = [0], S.serialize

        def counting(*a, **k):
            renders[0] += 1
            return real_serialize(*a, **k)
        for name, kw in cases.items():
            forged = tmp / f"forged-{len(name)}-{name[:6].replace(' ', '_')}"
            shutil.copytree(audit_dir, forged)
            forge(forged, **kw)
            tok, renders[0] = CharTokenizer(), 0
            saved = pm.serialize
            pm.serialize = counting
            try:
                r = outcome(E, lambda: P.run_preparation(**paths, selection_audit=forged, relation_config=REL_CFG,
                                                         direction_config=DIR_CFG, tokenizer_dir=TOK_DIR,
                                                         model_description=MODEL_DESC, out=forged.with_suffix(".out"),
                                                         tokenizer=tok, sample_only=False))
            finally:
                pm.serialize = saved
            check(f"{name}: an input error before any render or token call", r[0] == "rejected" and renders[0] == 0
                  and tok.calls == 0 and not forged.with_suffix(".out").exists(), f"{r} renders={renders[0]} tokens={tok.calls}")
        dup = tmp / "dupkeys"
        shutil.copytree(audit_dir, dup)
        (dup / "summary.json").write_bytes(b'{"format_version": 1, "format_version": 1}')
        forge(dup)
        r = outcome(E, lambda: P.run_preparation(**paths, selection_audit=dup, relation_config=REL_CFG, direction_config=DIR_CFG,
                                                 tokenizer_dir=TOK_DIR, model_description=MODEL_DESC, out=tmp / "dup.out",
                                                 tokenizer=CharTokenizer(), sample_only=False))
        check("duplicate JSON keys in the audit summary: an input error", r[0] == "rejected", str(r))
        changed = tmp / "changed-in"
        shutil.copytree(paths["scene"].parent.parent, changed)
        cfile = next((changed / "model_inputs/commands").iterdir())
        c = json.loads(cfile.read_text(encoding="utf-8"))
        c["text"] = c["text"] + " now"
        cfile.write_text(json.dumps(c), encoding="utf-8")
        p2 = {"scene": changed / "model_inputs/scene.annotated.json", "commands": changed / "model_inputs/commands",
              "category_map": changed / "model_inputs/category-map.json",
              "inventory_views": changed / "reference_only/inventory-views.json"}
        r = outcome(E, lambda: P.run_preparation(**p2, selection_audit=audit_dir, relation_config=REL_CFG, direction_config=DIR_CFG,
                                                 tokenizer_dir=TOK_DIR, model_description=MODEL_DESC, out=tmp / "changed.out",
                                                 tokenizer=CharTokenizer(), sample_only=False))
        check("a changed command text that the audit's input hashes don't cover: an input error", r[0] == "rejected", str(r))


def check_readback(P):
    print("-- readback catches tampering; no answer, graph, prediction or scoring access; no model")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _, _, out = fixture_run(P, tmp)
        meas = [m for m in jsonl(out / "measurements.jsonl") if m["status"] == "rendered"]
        index = jsonl(out / "preparation-index.jsonl")

        def tampered(fn):
            t = tmp / f"t{len(list(tmp.iterdir()))}"
            shutil.copytree(out, t)
            fn(t)
            return P.verify_bundle(t)
        cases = {"a changed prompt": lambda t: (t / meas[0]["paths"]["prompt"]).write_text("changed", encoding="utf-8"),
                 "a missing command record": lambda t: (t / next(r for r in index if r["command_path"])["command_path"]).unlink(),
                 "a stale scene link": lambda t: (t / "preparation-index.jsonl").write_text("".join(
                     json.dumps(dict(r, scene_path=next(x["scene_path"] for x in index if x["scene_path"] and x["scene_path"] != r["scene_path"]))
                                if r["scene_path"] else r, sort_keys=True, separators=(",", ":")) + "\n" for r in index), encoding="utf-8"),
                 "an altered summary count": lambda t: (t / "summary.json").write_text(
                     (t / "summary.json").read_text(encoding="utf-8").replace('"pairs": 8', '"pairs": 9'), encoding="utf-8")}
        for name, fn in cases.items():
            check(f"readback detects {name}", bool(tampered(fn)))
        import builtins
        opened, real_open, real_io = [], builtins.open, io.open

        def watch(file, *a, **k):
            opened.append(str(file))
            return real_io(file, *a, **k)
        io.open = builtins.open = watch
        try:
            fixture_run(P, tmp, name="watched")
        finally:
            io.open, builtins.open = real_io, real_open
        bad = [p for p in opened if any(n in p for n in ("iref-annotations", "scene_graph", "referential_statements",
                                                         "predictions.jsonl", "scores.jsonl", "prediction-summary"))]
        check("the preparation opens no answer, graph, statement, prediction or scoring file", not bad, str(bad[:3]))
        src = "".join(p.read_text(encoding="utf-8") for p in (REPO / "grounding/preparation").rglob("*.py"))
        check("the package loads no language model: no AutoModel, generate() or llama.cpp", not any(
            w in src for w in ("AutoModel", ".generate(", "llama_cpp", "from_pretrained(model")))


# ------------------------------------------------------------------------------------------- real tokenizer
def arg(name):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv and sys.argv.index(name) + 1 < len(sys.argv) else None


def check_real(P):
    print("-- real-token calibration and the full sample preparation (acceptance)")
    imp, audit = arg("--import"), arg("--selection-audit")
    try:
        tok = P.load_pinned_tokenizer(TOK_DIR)
    except Exception as e:  # noqa: BLE001
        check("the pinned measurement environment loads (transformers 4.57.6, tokenizers 0.22.2, the three pinned files)",
              False, f"{type(e).__name__}: {e}"[:200])
        return
    check("the pinned tokenizer: Qwen2TokenizerFast, exact versions, exact kind", tok.kind == "exact"
          and tok.versions == {"transformers": EXP["pins"]["transformers"], "tokenizers": EXP["pins"]["tokenizers"]}
          and tok.class_name == EXP["pins"]["tokenizer_class"], str(getattr(tok, "versions", None)))
    cal = P.calibrate(tok)
    check("A1 calibration: 230, 232, 230, 227 and a 191-token scene prefix", cal["a1_prompts"] == EXP["calibration"]["a1_prompts"]
          and set(cal["a1_scene_prefix"].values()) == {EXP["calibration"]["a1_scene_prefix"]}, json.dumps(cal["a1_prompts"]))
    check("the eight goldens under the sizing wrapper", cal["goldens"] == EXP["calibration"]["goldens"], json.dumps(cal["goldens"]))
    with tempfile.TemporaryDirectory() as tmp:
        extra = Path(tmp) / "tok"
        shutil.copytree(TOK_DIR, extra)
        (extra / "tokenizer.json").write_text("{}", encoding="utf-8")
        t2 = P.load_pinned_tokenizer(extra)
        check("an extra tokenizer.json beside the pinned files changes nothing: only the three are staged",
              t2.encode("table chair\n") == tok.encode("table chair\n"))
        (extra / "merges.txt").write_bytes((extra / "merges.txt").read_bytes() + b"\n")
        E = importlib.import_module("grounding.evaluation.iref_vla")
        check("a changed pinned file is refused", outcome(E, lambda: P.load_pinned_tokenizer(extra))[0] == "rejected")
    if arg("--bundle"):
        sample_checks(P, Path(arg("--bundle")))
        return
    if not imp or not audit:
        check("sample inputs given (--import DIR --selection-audit DIR, --bundle DIR, or run --unit-only)", False, "missing")
        return
    base = Path(imp)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "prep"
        r = subprocess.run([sys.executable, "-m", "grounding.preparation.iref_vla", "--scene", str(base / "model_inputs/scene.annotated.json"),
                            "--commands", str(base / "model_inputs/commands"), "--category-map", str(base / "model_inputs/category-map.json"),
                            "--inventory-views", str(base / "reference_only/inventory-views.json"), "--selection-audit", audit,
                            "--relation-config", str(REL_CFG), "--direction-config", str(DIR_CFG), "--tokenizer-dir", str(TOK_DIR),
                            "--model-description", str(MODEL_DESC), "--out", str(out)], capture_output=True, text=True, cwd=str(REPO))
        check("the full preparation completes with exit 0", r.returncode == 0, (r.stdout + r.stderr)[-300:])
        if r.returncode == 0:
            sample_checks(P, out)


def sample_checks(P, out):
    """Sample acceptance on a published bundle: readback, section 3's counts, exact planned memberships."""
    if True:
        check("the published bundle passes readback", P.verify_bundle(out) == [], str(P.verify_bundle(out)[:2]))
        s, idx, meas = EXP["sample"], jsonl(out / "preparation-index.jsonl"), jsonl(out / "measurements.jsonl")
        n = lambda d: sum(1 for p in (out / d).iterdir())  # noqa: E731
        check("199 scenes, 2,099 commands, 4,198 rendered documents, 3,872 index rows, 7,744 measurement rows",
              (n("model_records/scenes"), n("model_records/commands"), sum(m["status"] == "rendered" for m in meas), len(idx), len(meas))
              == (s["scene_files"], s["command_files"], s["rendered_documents"], s["index_rows"], s["measurement_rows"]))
        for v in VIEWS:
            st = [r["preparation_status"] for r in idx if r["view_id"] == v]
            check(f"{v}: fits {s['fits'][v]}, over {s['over'][v]}, unassessed {s['unassessed'][v]}",
                  (st.count("materialized"), st.count("over_object_budget"), st.count("unassessed_parse"))
                  == (s["fits"][v], s["over"][v], s["unassessed"][v]))
        check("no work failure; both formats rendered for every fitting pair", all(m["status"] != "work_budget_exceeded" for m in meas)
              and sum(m["status"] == "rendered" for m in meas) == 2 * (s["fits"]["full_inventory"] + s["fits"]["source_known_nyu"]))
        planned = {r["derived_scene_id"]: r for r in idx if r["derived_scene_id"]}
        exact = all([o["object_id"] for o in json.loads((out / r["scene_path"]).read_text(encoding="utf-8"))["objects"]]
                    == r["planned_object_ids"] for r in planned.values())
        check("every materialized scene holds exactly its planned objects", exact)
        summ = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        print("   measured (not expectations):", json.dumps({f"{v}/{f}": summ["views"][v][f]["ceilings"]
                                                             for v in VIEWS for f in FORMATS})[:600])


def main() -> int:
    P = importlib.import_module("grounding.preparation.iref_vla")
    check_constants(P)
    check_materialization(P)
    check_budgets(P)
    check_forgeries(P)
    check_readback(P)
    if "--unit-only" in sys.argv:
        print("UNIT-ONLY MODE: real-token calibration and the sample preparation did not run; not real-token acceptance")
    else:
        check_real(P)
    tail = " (UNIT-ONLY: not real-token or sample acceptance)" if "--unit-only" in sys.argv else ""
    print(f"{COUNT[0]} checks; {'FAILED: ' + ', '.join(FAILED) if FAILED else 'all checks passed'}{tail}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
