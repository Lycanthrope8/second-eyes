"""Tests of A2.4's labels and training preparation (D109).

    python grounding/tests/test_a24_prep.py --samples DIR --official DIR

The fixture release is the pinned ScanNet sample renamed to real list names: six training environments (one with two
scans), a development scene, a validation-list scene and the legacy scene, so the test can show that only the drawn
training scenes are ever read. Without the samples and lists the real-chain checks fail rather than skip.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
FAILS, PASSES = [], []
SCENES = {"scene0191_00": 0, "scene0191_01": 1, "scene0119_00": 2, "scene0230_00": 3, "scene0528_00": 4,
          "scene0705_00": 5, "scene0415_00": 6, "scene0081_00": 7, "scene0011_00": 8, "scene0010_01": 9}
DISAGREE_IN = "scene0415_00"   # a fitting scan (by the salted environment order)
TRAINING = {"scene0191_00", "scene0191_01", "scene0119_00", "scene0230_00", "scene0528_00", "scene0705_00", "scene0415_00"}
KEEP = 24



def _safe_console():
    """Never crash on printing: a Windows console piped to a file may not encode every character (cp1252)."""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass

def check(name, ok, detail=""):
    (PASSES if ok else FAILS).append(name)
    print(("  ok    " if ok else "  FAIL  ") + name + ("" if ok or not detail else f"  [{detail}]"))


RETRIED = []


def locked(fn, *, fresh=False, attempts=3, pauses=(2.0, 6.0)):
    """Runs fn, retrying only a transient Windows file lock by A2.6b's own rule (dev._transient: a PermissionError,
    possibly wrapped): an accepted publisher that fails this way publishes nothing. With fresh=True, attempt k calls
    fn(k), which writes a new folder. Every retry is listed at the end of the run."""
    D = importlib.import_module("grounding.preparation.iref_vla_dev.dev")
    for k in range(attempts):
        try:
            return fn(k) if fresh else fn()
        except Exception as e:   # noqa: BLE001 - only a transient lock is retried
            if k == attempts - 1 or not D._transient(e):
                raise
            RETRIED.append(f"{type(e).__name__}: {str(e)[:160]}")
            time.sleep(pauses[min(k, len(pauses) - 1)])


def scans_ok(P, out: Path, label: str, check_fn=None) -> bool:
    """The check that every drawn scan was prepared; a scan that was not is printed with its error and retries.
    check_fn: the calling suite's own check, so the result counts in that suite."""
    check_fn = check_fn or check
    scenes = [json.loads(x) for x in (out / "scenes.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    bad = [x for x in P.scene_problems(scenes) if x["status"] != "prepared"]
    s = P.summary(json.loads((out / "manifest.json").read_text(encoding="utf-8")), scenes)
    check_fn(f"every drawn scan was prepared ({label}: {len(scenes)} scans), and the summary says so",
          not bad and not s["scans_not_prepared"] and s["scans"] == {"prepared": len(scenes)},
          "; ".join(f"{x['scene']} {x['status']}: {x['error']} retries {json.dumps(x['retries'])[:400]}" for x in bad))
    for x in P.scene_problems(scenes):
        if x["retries"]:
            print(f"  note  {label}: {x['scene']} needed {len(x['retries'])} lock retries: {json.dumps(x['retries'])[:300]}")
    return not bad


def reuse_checks(P, R, D, tmp, a26_path, S, zp, vocab, sp, M):
    print("-- verified reuse of an earlier attempt's work folder")
    MR = locked(lambda: P.prepare(spec=Path(S["folder"]) / "spec.json", a26_spec=a26_path, zip_path=zp,
                                  inventory_dir=tmp / "inventory", vocabulary=vocab, out=tmp / "prep-reuse",
                                  work=tmp / "work", reuse=True, workers=2, settings_path=sp, progress=lambda m: None))
    out = Path(MR["folder"])
    scenes = [json.loads(x) for x in (out / "scenes.jsonl").read_text(encoding="utf-8").splitlines()]
    check("every scan is reused only after all its checks pass (complete; the import reproduced byte for byte; inputs, "
          "code, libraries, model description, tokenizer and runtime unchanged; the bundle verified by the accepted "
          "reader)", scenes and all(r["reused"] and r["status"] == "prepared" and all(r["reuse_checks"].values())
                                    and len(r["reuse_checks"]) == 12 for r in scenes)
          and MR["reuse"]["scans_reused"] == len(scenes), json.dumps([r.get("problems") for r in scenes])[:300])
    keys = [k for k in M["files"] if k.startswith(("training_data/", "cohort.json", "strata.jsonl", "rounds.json",
                                                   "reference_only/", "audit/", "example-failures.jsonl"))]
    check("a verified reuse gives exactly the fresh preparation's labels, cohorts, strata, rounds, examples and audit",
          all(M["files"][k] == MR["files"].get(k) for k in keys), str([k for k in keys if M["files"][k] != MR["files"].get(k)]))
    check("the reuse is recorded: the work folder, scans reused, and the specification draft says so",
          MR["reuse"]["work"] == str(tmp / "work") and "Reused from an earlier attempt's work folder after verification"
          in (out / "pilot-spec.md").read_text(encoding="utf-8"))
    inv_bytes = (tmp / "inventory" / "inventory.json").read_bytes()
    expected = {r["scene"]: r["input_sha256"] for r in json.loads(inv_bytes)["scenes"] if r["retained"]}
    auth = R.TrainingAuthorization(a26_spec_bytes=a26_path.read_bytes(),
                                   a24_spec_bytes=(Path(S["folder"]) / "spec.json").read_bytes(), inventory_bytes=inv_bytes)
    g = S["environments"]["fitting"][0]
    scan = S["environments"]["scenes"][g][0]
    found, _, _ = R.read_training_source(zp, auth, [scan])

    def verdict(mutate, label):
        w = tmp / f"tamper-{label}"
        shutil.copytree(tmp / "work" / scan, w / scan)
        mutate(w / scan)
        job = (scan, g, "fitting", found[scan], expected[scan], vocab.read_bytes(), w / scan, str(D.TOKENIZER),
               str(D.MODEL_DESCRIPTION), auth, True)
        return P._verify_reuse(job)

    def append(rel):
        def f(d):
            p = d / rel if not callable(rel) else rel(d)
            p.write_bytes(p.read_bytes() + b" ")
        return f
    def recode(d):
        m = json.loads((d / "bundle" / "manifest.json").read_text(encoding="utf-8"))
        k = sorted(m["code"])[0]
        m["code"][k] = "0" * 64
        (d / "bundle" / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    first_scene = lambda d: sorted((d / "bundle" / "model_records" / "scenes").rglob("*.json"))[0]   # noqa: E731
    cases = {"untouched": (lambda d: None, None),
             "derived scene changed": (append(first_scene), "bundle_verified_by_the_accepted_reader"),
             "audit output changed": (append("audit/selections.jsonl"), "audit_outputs_unchanged"),
             "import changed": (append("import/model_inputs/category-map.json"), "import_reproduced_byte_for_byte"),
             "recorded code differs from today's": (recode, "bundle_code_unchanged"),
             "chain unfinished (no bundle)": (lambda d: shutil.rmtree(d / "bundle"), "complete")}
    got = {}
    for label, (mutate, want) in cases.items():
        r = verdict(mutate, label.replace(" ", "-").replace("(", "").replace(")", "").replace("'", ""))
        got[label] = (r["status"], want in r["problems"] if want else not r["problems"])
    check("reuse refuses a scan whose derived scene, audit output or import changed, whose recorded code differs from "
          "today's, or whose chain did not finish; an untouched scan passes", got["untouched"] == ("prepared", True)
          and all(v == ("reuse_refused", True) for k, v in got.items() if k != "untouched"), json.dumps(got))
    try:
        shutil.copytree(tmp / "work", tmp / "work-bad")
        append(first_scene)(tmp / "work-bad" / scan)
        P.prepare(spec=Path(S["folder"]) / "spec.json", a26_spec=a26_path, zip_path=zp, inventory_dir=tmp / "inventory",
                  vocabulary=vocab, out=tmp / "prep-bad", work=tmp / "work-bad", reuse=True, workers=1, settings_path=sp,
                  progress=lambda m: None)
        stopped = False
    except P.PrepError as e:
        stopped = scan in str(e) and not (tmp / "prep-bad").exists()
    check("one refused scan stops the whole reuse before anything is published, naming the scan", stopped)


def audit_export_checks(P, out, tmp, audit):
    print("-- the audit re-exported from the saved records (fix 3)")
    before = {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob("*") if p.is_file()}
    AE = P.audit_export(prep=out, work=tmp / "work", out=tmp / "audit-export")
    d = Path(AE["folder"])
    items = [json.loads(x) for x in (d / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
    check("the same audit examples, in the same order, with their cells, labels and checks",
          [(i["cell"], i["parent_command_id"], i["label"]) for i in items] == [(i["cell"], i["parent_command_id"], i["label"]) for i in audit]
          and all(i["views"][v].get("check") == a["views"][v].get("check") for i, a in zip(items, audit)
                  for v in i["views"] if i["views"][v].get("materialized")))
    exact, linked = True, True
    idx_cache = {}
    for i in items:
        b = tmp / "work" / i["scene"] / "bundle"
        idx = idx_cache.setdefault(i["scene"], {x["command_path"]: x for x in
                                                (json.loads(y) for y in (b / "preparation-index.jsonl").read_text(encoding="utf-8").splitlines())
                                                if x.get("command_path")})
        for v, vv in i["views"].items():
            if not vv.get("materialized"):
                continue
            sb = (d / "evidence" / "scenes" / f"{vv['derived_scene']['sha256']}.json").read_bytes()
            cb = (d / "evidence" / "commands" / f"{vv['derived_command']['sha256']}.json").read_bytes()
            linked &= (hashlib.sha256(sb).hexdigest() == vv["derived_scene"]["sha256"] and sb == (b / vv["derived_scene"]["path"]).read_bytes()
                       and cb == (b / vv["derived_command"]["path"]).read_bytes()
                       and idx[vv["derived_command"]["path"]]["scene_sha256"] == vv["derived_scene"]["canonical_sha256"]
                       and idx[vv["derived_command"]["path"]]["command_sha256"] == vv["derived_command"]["canonical_sha256"])
            geo = {o["object_id"]: o["geometry"] for o in json.loads(sb)["objects"]}
            exact &= all(o[k] == geo[o["object_id"]][k] and o[k]["state"] == "known" for o in vv["objects"]
                         for k in ("center_m", "size_m"))
        a = d / "evidence" / "annotations" / f"{i['source_annotation_entries']['sha256']}.json"
        linked &= a.is_file() and len(json.loads(a.read_bytes())) == i["source_annotation_entries"]["entries"] > 0
    check("every visible object's center_m and size_m equal its original derived scene's fields exactly (known, full "
          "precision); no geometry is null", exact and AE["counts"]["center_m_known"] == AE["counts"]["objects"] > 0)
    check("each view links by SHA-256 to its original derived scene and command, byte for byte, matching the bundle's "
          "index; each parent to its source annotation entries; every record stored once", linked
          and len(list((d / "evidence").rglob("*.json"))) == AE["counts"]["derived_scenes"] + AE["counts"]["derived_commands"] + AE["counts"]["annotation_sets"])
    after = {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob("*") if p.is_file()}
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    check("the manifest records the preparation's identity, the superseded audit's hashes, and the specification, "
          "cohorts, labels and six training-data files unchanged; the preparation is untouched",
          before == after and m["preparation_manifest_sha256"] == hashlib.sha256((out / "manifest.json").read_bytes()).hexdigest()
          and m["supersedes"] == {"audit/audit.jsonl": before["audit/audit.jsonl"], "audit/audit.md": before["audit/audit.md"]}
          and len(m["unchanged"]) == 9 and all(m["unchanged"][k]["sha256"] == before[k] for k in m["unchanged"]))
    refused = []
    w2 = tmp / "work-tampered"
    shutil.copytree(tmp / "work", w2)
    first = next(i for i in items if any(vv.get("materialized") for vv in i["views"].values()))
    vv = next(vv for vv in first["views"].values() if vv.get("materialized"))
    p_ = w2 / first["scene"] / "bundle" / vv["derived_scene"]["path"]
    j = json.loads(p_.read_bytes())
    j["objects"][0]["geometry"]["center_m"]["value"][0] += 0.5
    p_.write_text(json.dumps(j), encoding="utf-8")
    for kw in ({"work": w2, "out": tmp / "ae-tampered"}, {"work": tmp / "work", "out": tmp / "audit-export"}):
        try:
            P.audit_export(prep=out, **kw)
            refused.append(False)
        except P.PrepError:
            refused.append(True)
    check("the export refuses a derived scene that differs from its bundle's index, and an existing output folder",
          refused == [True, True] and not (tmp / "ae-tampered").exists())


def report_retries():
    print(f"  note  transient Windows file locks retried: {len(RETRIED)}" + "".join(f"\n        {r}" for r in RETRIED))


def build_release(samples: Path, folder: Path):
    """As A2.6b's test builds its release: the pinned scene renamed, a slice of texts each, one unused object lengthened
    per scene (CSV and graph alike) so each scan has its own fingerprint."""
    src = samples / "sample_data" / "Scannet" / "scene0010_01"
    base = json.loads((src / "scene0010_01_referential_statements.json").read_text(encoding="utf-8"))
    texts = sorted(base["regions"]["0"])
    step = len(texts) // len(SCENES)
    for scene, k in SCENES.items():
        d = folder / "Scannet" / scene
        d.mkdir(parents=True)
        keep = texts[k * step:k * step + KEEP]
        region = {t: base["regions"]["0"][t] for t in keep}
        if scene == DISAGREE_IN:   # one target under two relations: a disagreeing parsing label (A2.6b's correction 1)
            for t in keep[:6]:
                tb = base["regions"]["0"][t][0]
                same = next((x for x in texts if x != t and base["regions"]["0"][x][0]["target_index"] == tb["target_index"]
                             and base["regions"]["0"][x][0]["relation"] != tb["relation"]), None)
                if same:
                    region[t] = [tb, base["regions"]["0"][same][0]]
        used = {e["target_index"] for es in region.values() for e in es}
        used |= {a["index"] for es in region.values() for e in es for a in e["anchors"].values()}
        used |= {x for es in region.values() for e in es for x in e["distractor_ids"]}
        rows = (src / "scene0010_01_object_result.csv").read_text(encoding="utf-8").split("\n")
        free = min((r.split(",")[0] for r in rows[1:] if r and r.split(",")[0] not in used), key=int)
        delta = 0.01 * (k + 1)
        volume = None
        for i, r in enumerate(rows):
            cells = r.split(",")
            if i and r and cells[0] == free:
                cells[10] = repr(float(cells[10]) + delta)
                volume = float(cells[10]) * float(cells[11]) * float(cells[12])
                rows[i] = ",".join(cells)
        (d / f"{scene}_object_result.csv").write_text("\n".join(rows), encoding="utf-8")
        (d / f"{scene}_region_result.csv").write_bytes((src / "scene0010_01_region_result.csv").read_bytes())
        g = json.loads((src / "scene0010_01_scene_graph.json").read_text(encoding="utf-8"))
        g["scene_name"] = scene
        for o in g["regions"]["0"]["objects"]:
            if o["object_id"] == free:
                o["size"][0] = o["size"][0] + delta
                o["volume"] = volume
        (d / f"{scene}_scene_graph.json").write_text(json.dumps(g), encoding="utf-8")
        (d / f"{scene}_referential_statements.json").write_text(
            json.dumps({"scene_name": scene, "regions": {"0": region}}), encoding="utf-8")


def unit_checks():
    L = importlib.import_module("grounding.a24.language")
    LB = importlib.import_module("grounding.a24.labels")
    P = importlib.import_module("grounding.a24.prep")
    spec = L.load_spec()

    def E(rel, typ="binary", anchors=(("table", "", ""),), tc="chair", tcol="", tsize=""):
        return {"source_payload": {"relation": rel, "relation_type": typ, "target_class": tc, "target_color_used": tcol,
                                   "target_size_used": tsize, "anchors": {f"anchor_{i + 1}": {"index": str(i), "class": c,
                                                                                              "color_used": col, "size_used": sz}
                                                                         for i, (c, col, sz) in enumerate(anchors)}}}
    print("-- the label mapping (synthetic entries)")
    want = {"closest": "chair | closest 1 | table", "second_closest": "chair | closest 2 | table",
            "third_closest": "chair | closest 3 | table", "farthest": "chair | farthest 1 | table",
            "second_farthest": "chair | farthest 2 | table", "third_farthest": "chair | farthest 3 | table",
            "near": "chair | near | table", "below": "chair | below | table", "on": "chair | on | table"}
    got = {r: LB.build([E(r)], spec)["string"] for r in want}
    check("every IRef-VLA relation in v1 maps by the written table", got == want, str(got))
    b = LB.build([E("between", "ternary", (("door", "black", ""), ("window", "", "")), tcol="white")], spec)
    check("between keeps the annotation's anchor order and the colours the text used",
          b["string"] == "chair [white] | between | door [black] | window")
    cases = {"unsupported_semantics:relation_above": E("above"), "unsupported_semantics:relation_in": E("in"),
             "unsupported_semantics:size_word": E("near", tsize="big"),
             "unsupported_semantics:colour_target": E("near", tcol="grey"),
             "mapping_disagreement:relation_type": E("between", "binary"),
             "unsupported_semantics:target_class": E("near", tc="unknown")}
    reasons = {}
    for want_reason, e in cases.items():
        try:
            LB.build([e], spec)
            reasons[want_reason] = None
        except LB.Unsupported as x:
            reasons[want_reason] = str(x)
    check("above, in, size words, other colours, a wrong relation type and an unknown class are refused with reasons",
          all(k == v for k, v in reasons.items()), str(reasons))
    try:
        LB.build([E("near"), E("near", tcol="white")], spec)
        dis = None
    except LB.Unsupported as x:
        dis = str(x)
    check("entries of one parent giving different labels are an annotation disagreement, never resolved by order",
          dis == "annotation_disagreement:label_fields")
    lab = L.parse_output("chair [white] | between | door | window", spec)
    rule = L.interpretation(L.parse_output("chair [white] | between | window | door", spec))
    rule2 = L.interpretation(L.parse_output("chair | closest 1 | door", spec))
    check("the rule-parser comparison names the differing fields, anchor order included",
          LB.compare(lab, rule) == ["anchor_order"] and LB.compare(lab, rule2) == ["target", "relation", "anchors"])
    print("-- the tests' lock retry (A2.6b's transient rule)")
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("[WinError 5] Access is denied (labelled double)")
        return "done"
    before = len(RETRIED)
    got = locked(flaky, pauses=(0, 0))
    check("a transient lock is retried and then succeeds; each retry is listed", got == "done" and calls["n"] == 3
          and len(RETRIED) - before == 2)
    calls["n"] = 0

    def data_error():
        calls["n"] += 1
        raise ValueError("a data error (labelled double)")
    try:
        locked(data_error, pauses=(0, 0))
        raised = False
    except ValueError:
        raised = calls["n"] == 1
    calls["n"] = 0

    def always_locked():
        calls["n"] += 1
        raise PermissionError("[WinError 5] Access is denied (labelled double)")
    try:
        locked(always_locked, pauses=(0, 0))
        final = False
    except PermissionError:
        final = calls["n"] == 3
    del RETRIED[before:]
    check("a data error is never retried; a lock lasting past the third attempt fails the run", raised and final)
    print("-- failed scans are reported plainly")
    fake_scenes = [{"scene": "scene0001_00", "group": "g1", "cohort": "fitting", "status": "prepared"},
                   {"scene": "scene0002_00", "group": "g2", "cohort": "fitting", "status": "failed",
                    "error": "AdapterOutputError: PermissionError: [WinError 5] Access is denied (labelled double)",
                    "retries": [{"stage": "import", "attempt": 1}, {"stage": "import", "attempt": 2}]}]
    fake_man = {"elapsed_s": 1.0, "counts": {"scenes": {"prepared": 1, "failed": 1}, "parents": 2, "eligible_parents": 1,
                                             "retained_parents": 1, "fitting": {"parents": 1}, "validation": {"parents": 0}}}
    sm = P.summary(fake_man, fake_scenes)
    check("the prepare summary names a scan that was not prepared, with its error, and lists lock retries",
          [x["scene"] for x in sm["scans_not_prepared"]] == ["scene0002_00"] and "WinError 5" in sm["scans_not_prepared"][0]["error"]
          and sm["scans_with_lock_retries"] == ["scene0002_00"] and sm["scans"] == {"prepared": 1, "failed": 1})
    print("-- audit geometry: the scene contract's fields, full precision, state kept (fix 3)")
    ev = {"assumptions": [], "kind": "annotated", "source": "iref_scannet"}
    known_c, known_s = [-1.8319513596543817, 1.6253435304181612, 1.1020223498344421], [0.123456789012345, 2.0, 0.3333333333333333]
    scene = {"objects": [
        {"object_id": "obj_001", "category": {"state": "known", "value": {"model": "chair", "standard": "chair"}},
         "attributes": {"colours": {"state": "known", "value": ["brown"]}},
         "geometry": {"center_m": {"evidence": ev, "state": "known", "value": known_c},
                      "size_m": {"evidence": ev, "state": "known", "value": known_s},
                      "rotation_xyzw": {"evidence": ev, "state": "known", "support": "yaw_only", "value": [0.0, 0.0, 0.0, 1.0]}}},
        {"object_id": "obj_002", "category": {"state": "known", "value": {"model": "lamp", "standard": "lamp"}},
         "geometry": {"center_m": {"state": "unknown", "reason": "not annotated"}, "size_m": {"state": "unknown"}}}]}
    vis = P._visible(scene, [("A", "obj_001"), ("B", "obj_002")])
    md = P.audit_markdown([{"cell": ["retained", "near"], "text": "the chair", "parent_command_id": "p1", "scene": "s",
                            "label": "chair | near | lamp", "rule_parse": None, "rule_disagreement": [],
                            "target_object_id": "obj_001", "retained": True, "parent_reasons": [],
                            "views": {"source_known_nyu": {"materialized": True, "check": None, "objects": vis}}}])
    check("known center_m and size_m survive at full precision with their state; unknown stays unknown with no value; "
          "a missing field is absent; only the display rounds",
          vis[0]["center_m"]["value"] == known_c and vis[0]["size_m"]["value"] == known_s
          and vis[0]["center_m"]["state"] == vis[0]["size_m"]["state"] == "known"
          and vis[1]["center_m"] == {"state": "unknown", "reason": "not annotated"} and "value" not in vis[1]["size_m"]
          and vis[1]["rotation_xyzw"] == {"state": "absent"} and vis[0]["letter"] == "A"
          and "c=(-1.83, 1.63, 1.10) s=(0.12, 2.00, 0.33)" in md and "c=unknown (unknown) s=unknown (unknown)" in md,
          json.dumps(vis)[:300])
    print("-- sampling with a salted stratum order")
    rows = [{"group": f"g{g}", "stratum": f"s{s}", "retained": True, "parent_command_id": f"g{g}s{s}p{p}"}
            for g in range(60) for s in range(10) for p in range(2)]
    strata, sel = P.sample(rows, [f"g{g}" for g in range(60)], cap=9)
    dropped = Counter(s["stratum"] for s in strata if s["n"] == 0)
    check("a cap below the number of strata drops a different stratum by environment (salted order), 9 each",
          len(dropped) >= 8 and all(sum(s["n"] for s in strata if s["group"] == g) == 9 for g in {r["group"] for r in rows}),
          str(dropped))
    check("the same inputs give the same selection", P.sample(rows, [f"g{g}" for g in range(60)], cap=9)[1] == sel)
    check("zero-selected strata are reported with their counts (N > 0, n = 0), never dropped from the table",
          all(sum(1 for s in strata if s["group"] == g) == 10 for g in {r["group"] for r in rows})
          and sum(s["n"] == 0 and s["N"] == 2 for s in strata) == 60)
    rows2 = rows + [{"group": "g0", "stratum": "s99", "retained": False, "eligible": True, "parent_command_id": "q"}]
    st2, _ = P.sample(rows2, [f"g{g}" for g in range(60)], cap=9)
    check("a stratum with eligible but no retained parent is reported with N = 0 and its eligible count",
          {"group": "g0", "stratum": "s99", "N": 0, "n": 0, "N_eligible": 1} in st2)


def _original_release():
    """release.py as accepted at 0395a43, loaded from git into the package, for byte comparisons."""
    src = subprocess.run(["git", "-C", str(REPO), "show", "0395a43:grounding/adapters/iref_vla/release.py"],
                         capture_output=True, check=True).stdout
    path = Path(tempfile.mkdtemp()) / "release_0395a43.py"
    path.write_bytes(src)
    spec = importlib.util.spec_from_file_location("grounding.adapters.iref_vla._release_0395a43", path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "grounding.adapters.iref_vla"
    spec.loader.exec_module(mod)
    return mod


def _tree(folder: Path) -> dict:
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()}


def importer_checks(R, P, tmp, a26_path, a24_path, zp, vocab, S):
    print("-- the importer's training mode (release.TrainingAuthorization)")
    inv_bytes = (tmp / "inventory" / "inventory.json").read_bytes()
    inv = json.loads(inv_bytes)
    expected = {r["scene"]: r["input_sha256"] for r in inv["scenes"] if r["retained"]}
    fit_scene = S["environments"]["scenes"][S["environments"]["fitting"][0]][0]
    res_scene = S["environments"]["scenes"][S["environments"]["reserve"][0]][0]
    found, _, _ = R.read_source(zp, "Scannet", only={fit_scene, "scene0081_00", "scene0010_01", "scene0011_00"})
    vb = vocab.read_bytes()

    def refused(fn):
        try:
            fn()
            return False
        except R.ReleaseError:
            return True
    check("by default a training scene is refused, as before",
          refused(lambda: R.import_release_scene("Scannet", fit_scene, found[fit_scene], vb, tmp / "i0", partition="training",
                                                 expected_sha256=expected[fit_scene])) and not (tmp / "i0").exists())
    auth = R.TrainingAuthorization(a26_spec_bytes=a26_path.read_bytes(), a24_spec_bytes=a24_path.read_bytes(),
                                   inventory_bytes=inv_bytes)
    locked(lambda: R.import_release_scene("Scannet", fit_scene, found[fit_scene], vb, tmp / "i1", partition="training",
                                          expected_sha256=expected[fit_scene], training=auth))
    desc = json.loads((tmp / "i1" / "model_inputs" / "scene.annotated.json").read_text(encoding="utf-8"))["sources"][0]["description"]
    check("under the authorization a fitting scan imports, described as A2.4 training material, never as development",
          "A2.4 training material (the fitting cohort" in desc and "Development material" not in desc, desc[-160:])
    check("closed and unknown partitions are refused even in training mode, and so is a closed scene labelled training",
          refused(lambda: R.import_release_scene("Scannet", "scene0011_00", found["scene0011_00"], vb, tmp / "i2",
                                                 partition="calibration", expected_sha256=expected["scene0011_00"], training=auth))
          and refused(lambda: R.import_release_scene("Scannet", "scene0011_00", found["scene0011_00"], vb, tmp / "i3",
                                                     partition="training", expected_sha256=expected["scene0011_00"], training=auth))
          and refused(lambda: R.release_identity("Scannet", "scene0011_00", found["scene0011_00"], "test", auth))
          and refused(lambda: auth.cohort_of("scene0999_00")) and not (tmp / "i2").exists() and not (tmp / "i3").exists())
    check("a development scene labelled training, or given an authorization as development, is refused",
          refused(lambda: auth.cohort_of("scene0081_00"))
          and refused(lambda: R.import_release_scene("Scannet", "scene0081_00", found["scene0081_00"], vb, tmp / "i4",
                                                     partition="development", expected_sha256=expected["scene0081_00"], training=auth)))
    check("an unactivated reserve environment is refused; it is permitted once activated in the frozen order",
          refused(lambda: auth.cohort_of(res_scene)) and auth.activate([S["environments"]["reserve"][0]]).cohort_of(res_scene) == "reserve"
          and refused(lambda: auth.activate(S["environments"]["reserve"][1:2] or ["scannet:none"])))
    other = json.loads(a24_path.read_text(encoding="utf-8"))
    other["environments"]["scenes"][S["environments"]["fitting"][0]] = []
    check("a scan the specification does not list for its environment is refused",
          refused(lambda: R.TrainingAuthorization(a26_spec_bytes=a26_path.read_bytes(), a24_spec_bytes=json.dumps(other).encode(),
                                                  inventory_bytes=inv_bytes).cohort_of(fit_scene)))
    check("an authorization built from another inventory or another A2.6 specification is refused",
          refused(lambda: R.TrainingAuthorization(a26_spec_bytes=a26_path.read_bytes(), a24_spec_bytes=a24_path.read_bytes(),
                                                  inventory_bytes=inv_bytes + b" "))
          and refused(lambda: R.TrainingAuthorization(a26_spec_bytes=a26_path.read_bytes() + b" ",
                                                      a24_spec_bytes=a24_path.read_bytes(), inventory_bytes=inv_bytes)))
    read, real = [], zipfile.ZipFile.read

    def spy(self, name, *x, **k):
        read.append(name)
        return real(self, name, *x, **k)
    zipfile.ZipFile.read = spy
    try:
        r1 = refused(lambda: R.read_training_source(zp, auth, [fit_scene, "scene0011_00"]))
    finally:
        zipfile.ZipFile.read = real
    check("read_training_source checks every scene before opening any member: nothing read when one is refused",
          r1 and not read)
    try:
        O = _original_release()
        same = {}
        for scene, part in (("scene0081_00", "development"), ("scene0010_01", "legacy-development")):
            locked(lambda: O.import_release_scene("Scannet", scene, found[scene], vb, tmp / f"old-{scene}", partition=part,
                                                  expected_sha256=expected[scene]))
            locked(lambda: R.import_release_scene("Scannet", scene, found[scene], vb, tmp / f"new-{scene}", partition=part,
                                                  expected_sha256=expected[scene]))
            same[scene] = _tree(tmp / f"old-{scene}") == _tree(tmp / f"new-{scene}")
        check("development and legacy imports are byte-identical to the accepted 0395a43 importer's", all(same.values()),
              str(same))
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        check("the accepted 0395a43 importer is available from git for the byte comparison", False, str(e)[:200])


def main() -> int:
    _safe_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=os.environ.get("SECOND_EYES_IREF_RELEASE_SAMPLES"))
    ap.add_argument("--official", default=os.environ.get("SECOND_EYES_IREF_OFFICIAL"))
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    unit_checks()
    R = importlib.import_module("grounding.adapters.iref_vla.release")
    D = importlib.import_module("grounding.preparation.iref_vla_dev.dev")
    P = importlib.import_module("grounding.a24.prep")
    L = importlib.import_module("grounding.a24.language")
    DC = importlib.import_module("grounding.a24.decoder")
    T = importlib.import_module("grounding.preparation.iref_vla.tokens")
    have = bool(a.samples and a.official) and not R.verify_samples(a.samples)
    check("the pinned samples and official lists are available", have, "pass --samples DIR and --official DIR")
    if not have:
        print(f"\n{len(PASSES)} passed, {len(FAILS)} failed")
        return 1
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        build_release(Path(a.samples), tmp / "release")
        vocab = Path(a.samples) / "data" / "NYU_Object_Classes.csv"
        inv = R.build_inventory([("Scannet", tmp / "release" / "Scannet")], R.read_vocabulary(vocab.read_bytes()))
        locked(lambda: R.write_inventory(inv, tmp / "inventory"))
        zp = tmp / "Scannet.zip"
        with zipfile.ZipFile(zp, "w") as z:
            for f in sorted((tmp / "release").rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(tmp / "release").as_posix())
        a26 = locked(lambda: D.freeze(inventory_dir=tmp / "inventory", official_dir=a.official, out=tmp / "a26spec"))
        a26_path = Path(a26["folder"]) / "spec.json"
        frozen = {"a26_spec_sha256": hashlib.sha256(a26_path.read_bytes()).hexdigest(),
                  "inventory_sha256": a26["inventory"]["inventory_sha256"]}
        st = json.loads(P.SETTINGS_PATH.read_text(encoding="utf-8"))
        st["cohort"] = {"fitting_environments": 3, "validation_environments": 1, "fitting_cap_per_environment": 4,
                        "validation_cap_per_environment": 3, "fitting_minimum": 13, "validation_minimum": 2,
                        "reserve_batch_environments": 1, "reserve_batches_at_most": 2}
        sp = tmp / "settings.json"
        sp.write_text(json.dumps(st), encoding="utf-8")
        print("-- freeze")
        try:
            P.freeze(a26_spec=a26_path, inventory_dir=tmp / "inventory", out=tmp / "nope", settings_path=sp)
            refused = False
        except P.PrepError:
            refused = not (tmp / "nope").exists()
        check("freeze refuses any A2.6 specification but the frozen one (here the real hash)", refused)
        S = P.freeze(a26_spec=a26_path, inventory_dir=tmp / "inventory", out=tmp / "spec", frozen=frozen, settings_path=sp)
        envs = S["environments"]
        training_groups = {f"scannet:{s.split('_')[0]}" for s in TRAINING}
        drawn = envs["fitting"] + envs["validation"] + envs["reserve"]
        check("only training environments are drawn, in salted-hash order: 3 fitting, 1 validation, the rest reserve",
              set(drawn) == training_groups and len(envs["fitting"]) == 3 and len(envs["validation"]) == 1
              and drawn == sorted(drawn, key=lambda g: P._key(P.ENV_SALT, g)) and "scannet:scene0010" not in drawn,
              str(envs))
        check("rescans stay with their environment", envs["scenes"]["scannet:scene0191"] == ["scene0191_00", "scene0191_01"])
        check("the specification binds the inventory by its bytes, and the language, settings and code by hash",
              S["inputs"]["inventory_sha256"] == frozen["inventory_sha256"] and S["inputs"]["language"]["sha256"] == L.spec_sha256()
              and len(S["inputs"]["code"]) >= 6)
        importer_checks(R, P, tmp, a26_path, Path(S["folder"]) / "spec.json", zp, vocab, S)
        print("-- prepare (the accepted chain, the pinned tokenizer)")
        read, real = [], zipfile.ZipFile.read

        def spy(self, name, *x, **k):
            read.append(name if isinstance(name, str) else name.filename)
            return real(self, name, *x, **k)
        zipfile.ZipFile.read = spy
        try:
            M = P.prepare(spec=Path(S["folder"]) / "spec.json", a26_spec=a26_path, zip_path=zp, inventory_dir=tmp / "inventory", vocabulary=vocab,
                          out=tmp / "prep", work=tmp / "work", workers=a.workers, settings_path=sp, progress=lambda m: None)
        finally:
            zipfile.ZipFile.read = real
        scenes_read = {Path(n).parts[1] for n in read if len(Path(n).parts) > 2}
        allowed = {s for g in drawn for s in envs["scenes"][g]}
        check("no development, validation-list or legacy scene is read from the zip", scenes_read <= allowed
              and not scenes_read & {"scene0081_00", "scene0011_00", "scene0010_01"}, str(sorted(scenes_read)))
        out = Path(M["folder"])
        scans_ok(P, out, "spawned workers")
        man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        hashes_ok = all(hashlib.sha256((out / n).read_bytes()).hexdigest() == h for n, h in man["files"].items())
        check("every published file reads back against the manifest; start and finish are recorded in UTC",
              hashes_ok and man["started_utc"].endswith("+00:00") and man["finished_utc"].endswith("+00:00"))
        rounds = json.loads((out / "rounds.json").read_text(encoding="utf-8"))
        check("a fitting cohort below its minimum takes the next reserve environment, recorded as a round",
              len(rounds) >= 2 and rounds[1]["environments"] == [envs["reserve"][0]] and rounds[0]["short"], str(rounds))
        rows = [json.loads(x) for x in (out / "reference_only" / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
        spec = L.load_spec()
        ret = [r for r in rows if r["retained"]]
        check("some parents are retained and some quarantined, each with reasons",
              ret and any(r["parent_reasons"] for r in rows if r["eligible"]), str(M["counts"]["quarantine_reasons_by_parent"]))
        check("every retained parent: a v1 label, the rule parser agreeing, its source-known view supported",
              all(L.parse_output(r["label"], spec) and not r["rule_disagreement"] and r["views"]["source_known_nyu"]["supported"]
                  and "source_known_nyu" in r["direct_views"] for r in ret))
        check("every quarantined eligible parent has a reason from the declared families",
              all(x.split(":")[0] in ("annotation_disagreement", "unsupported_semantics", "mapping_disagreement",
                                      "rule_parser_disagreement", "source_known", "invalid_label_query")
                  for r in rows if r["eligible"] for x in r["parent_reasons"]))
        ex = {}
        for c in P.COHORTS:
            for f in P.FORMATS:
                ex[(c, f)] = [json.loads(x) for x in (out / "training_data" / f"{c}-direct-{f}.jsonl").read_text(encoding="utf-8").splitlines()]
            ex[(c, "parser")] = [json.loads(x) for x in (out / "training_data" / f"{c}-parser.jsonl").read_text(encoding="utf-8").splitlines()]
        same_rows = all([e["key"] for e in ex[(c, P.FORMATS[0])]] == [e["key"] for e in ex[(c, P.FORMATS[1])]]
                        and [e["answer_code"] for e in ex[(c, P.FORMATS[0])]] == [e["answer_code"] for e in ex[(c, P.FORMATS[1])]]
                        for c in P.COHORTS)
        check("both direct formats hold identical parent and view rows, in the same order, with the same answers", same_rows)
        by_parent = {r["parent_command_id"]: r for r in rows}
        letters_ok = all(e["choice_object_ids"][e["codes"].index(e["answer_code"])] == e["target_object_id"]
                         and e["answer_token_id"] == e["code_token_ids"][e["codes"].index(e["answer_code"])]
                         and e["view"] in by_parent[e["parent_command_id"]]["direct_views"]
                         for c in P.COHORTS for f in P.FORMATS for e in ex[(c, f)])
        check("every direct answer is the target's D104 letter and token, and only on a view whose check passed", letters_ok)
        tok = T.load_pinned_tokenizer(D.TOKENIZER)
        Cn = DC.Constraint(spec, DC.TokenTable(tok._backend))
        pz = all(tok.encode(P.parser_prompt(e["text"], P.load_settings(sp)) + e["label"]) == e["prompt_ids"] + e["output_ids"][:-1]
                 and Cn.accepts_ids(e["output_ids"]) == (True, len(e["output_ids"]) - 1)
                 and e["output_ids"][-1] == Cn.T.eos_id and tok._backend.decode(e["output_ids"][:-1]) == e["label"]
                 for c in P.COHORTS for e in ex[(c, "parser")])
        check("every parser example: prompt and output tokenize apart, the output is a path the grammar allows, then "
              "the end token", pz)
        check("the parser prompt holds the command text and the fixed instructions only: no label, target or object",
              all(e["label"] not in P.parser_prompt(e["text"], P.load_settings(sp)) for c in P.COHORTS for e in ex[(c, "parser")]))
        cohort = json.loads((out / "cohort.json").read_text(encoding="utf-8"))
        fit_env = {x["group"] for x in cohort["fitting"]}
        val_env = {x["group"] for x in cohort["validation"]}
        check("the fitting and inner-validation cohorts share no environment", not fit_env & val_env and fit_env and val_env)
        audit = [json.loads(x) for x in (out / "audit" / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
        cells = {tuple(i["cell"][:1]) for i in audit}
        check("the audit covers retained constructions and quarantine reasons, with visible objects and letters per view",
              ("retained",) in cells and ("quarantined",) in cells
              and all(v.get("objects") is not None for i in audit for v in i["views"].values() if v.get("materialized"))
              and any(o["letter"] for i in audit for v in i["views"].values() if v.get("materialized") for o in v["objects"]))
        spec_md = (out / "pilot-spec.md").read_text(encoding="utf-8")
        check("the pilot specification draft carries the cohorts, exclusions, exposure, examples, matrix and settings",
              all(h in spec_md for h in ("## Cohorts", "## Scans", "Every drawn scan was prepared.",
                                         "## Supervision coverage and exclusions", "## Exposure",
                                         "## Input and output examples", "## The comparison matrix", "## Training settings",
                                         "## Decoding constraints")))
        early = [r for r in rows if r["eligible"] and "annotation_disagreement:parsing_label" in r["parent_reasons"]]
        check("eligible commands quarantined before their view checks (a disagreeing parsing label) keep their reason "
              "and have no view results", early and all(not r["views"] and r["label"] is None for r in early),
              str(len(early)))
        early_items = [i for i in audit if i["cell"] == ["quarantined", "annotation_disagreement:parsing_label"]]
        audit_md = (out / "audit" / "audit.md").read_text(encoding="utf-8")
        check("the audit shows them with their visible objects, each view marked not checked, and its markdown renders",
              early_items and all(v["check"] is None and v["objects"] for i in early_items for v in i["views"].values()
                                  if v.get("materialized"))
              and "not checked (the parent was quarantined before its view checks" in audit_md
              and "Label: none (quarantined before labelling)" in audit_md)
        strata = [json.loads(x) for x in (out / "strata.jsonl").read_text(encoding="utf-8").splitlines()]
        pairs = {(r["group"], r["stratum"]) for r in rows if r["eligible"] and r.get("cohort")}
        check("per environment and stratum: eligible (A2.6b), retained and selected counts for every pair with an "
              "eligible parent", all({"N", "n", "N_eligible", "cohort"} <= set(s) for s in strata)
              and pairs <= {(s["group"], s["stratum"]) for s in strata}
              and all(s["N_eligible"] >= s["N"] >= s["n"] for s in strata), str(strata[:3]))
        sums = {}
        for s_ in strata:
            a_ = sums.setdefault((s_["cohort"], s_["group"]), [0, 0])
            a_[0] += s_["N"]
            a_[1] += s_["N_eligible"]
        want_ret = {c: sorted(g for (cc, g), (n, _) in sums.items() if cc == c and n == 0) for c in P.COHORTS}
        want_el = {c: sorted(g for (cc, g), (_, e) in sums.items() if cc == c and e == 0) for c in P.COHORTS}
        contrib = {c: {s_["group"] for s_ in strata if s_["cohort"] == c and s_["n"]} for c in P.COHORTS}
        check("environments contributing nothing are those whose retained parents sum to zero, split into no eligible "
              "parent and all quarantined; never a contributing environment",
              all(M["counts"][c]["environments_without_retained_parents"] == want_ret[c]
                  and M["counts"][c]["environments_without_eligible_parents"] == want_el[c]
                  and not set(want_ret[c]) & contrib[c]
                  and M["counts"][c]["environments"] == len(contrib[c]) + len(want_ret[c]) for c in P.COHORTS),
              json.dumps({c: [M["counts"][c]["environments_without_retained_parents"], want_ret[c]] for c in P.COHORTS})[:300])
        before = {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob("*") if p.is_file()}
        RP = P.report(prep=out, out=tmp / "prep-report", settings_path=sp)
        after = {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob("*") if p.is_file()}
        rdir = Path(RP["folder"])
        check("report regenerates the counts and pilot-spec.md from the saved preparation, byte for byte equal to this "
              "code's own, and records what it supersedes; the preparation is untouched",
              RP["counts"] == M["counts"] and (rdir / "pilot-spec.md").read_bytes() == (out / "pilot-spec.md").read_bytes()
              and RP["supersedes"]["pilot-spec.md"] == before["pilot-spec.md"] and before == after
              and json.loads((rdir / "manifest.json").read_text(encoding="utf-8"))["preparation_manifest_sha256"]
              == hashlib.sha256((out / "manifest.json").read_bytes()).hexdigest())
        audit_export_checks(P, out, tmp, audit)
        bad = tmp / "prep-altered"
        shutil.copytree(out, bad)
        (bad / "strata.jsonl").write_bytes((bad / "strata.jsonl").read_bytes() + b"\n")
        refused = []
        for kw in ({"prep": bad, "out": tmp / "report-bad"}, {"prep": out, "out": tmp / "prep-report"}):
            try:
                P.report(settings_path=sp, **kw)
                refused.append(False)
            except P.PrepError:
                refused.append(True)
        check("report refuses a preparation whose files differ from its manifest, and an existing output folder",
              refused == [True, True] and not (tmp / "report-bad").exists())
        c = M["counts"]
        check("counts by parent and by view, before and after filtering, are recorded",
              c["parents"] >= c["eligible_parents"] >= c["retained_parents"] > 0 and c["view_outcomes_among_eligible"]
              and c["fitting"]["target_letters"])
        if a.workers > 1:
            M1 = P.prepare(spec=Path(S["folder"]) / "spec.json", a26_spec=a26_path, zip_path=zp, inventory_dir=tmp / "inventory", vocabulary=vocab,
                           out=tmp / "prep1", work=tmp / "work1", workers=1, settings_path=sp, progress=lambda m: None)
            scans_ok(P, Path(M1["folder"]), "one process")
            keys = [k for k in M["files"] if k.startswith(("training_data/", "cohort.json", "strata.jsonl",
                                                            "reference_only/labels.jsonl"))]
            check("spawned workers and one process give identical cohorts, labels and training data",
                  all(M["files"][k] == M1["files"][k] for k in keys), str([k for k in keys if M["files"][k] != M1["files"][k]]))
        reuse_checks(P, R, D, tmp, a26_path, S, zp, vocab, sp, M)
        try:
            P.prepare(spec=Path(S["folder"]) / "spec.json", a26_spec=a26_path, zip_path=zp, inventory_dir=tmp / "inventory",
                      vocabulary=vocab, out=tmp / "prep", work=tmp / "work2", settings_path=sp, progress=lambda m: None)
            refused = False
        except (P.PrepError, D.DevPrepError):
            refused = True
        check("an existing output folder is never overwritten", refused)
        print("   counts:", json.dumps({k: c[k] for k in ("parents", "eligible_parents", "retained_parents",
                                                         "quarantine_reasons_by_parent")}))
    report_retries()
    print(f"\n{len(PASSES)} passed, {len(FAILS)} failed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
