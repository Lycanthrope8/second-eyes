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
        strata = [json.loads(x) for x in (out / "strata.jsonl").read_text(encoding="utf-8").splitlines()]
        pairs = {(r["group"], r["stratum"]) for r in rows if r["eligible"] and r.get("cohort")}
        check("per environment and stratum: eligible (A2.6b), retained and selected counts for every pair with an "
              "eligible parent", all({"N", "n", "N_eligible", "cohort"} <= set(s) for s in strata)
              and pairs <= {(s["group"], s["stratum"]) for s in strata}
              and all(s["N_eligible"] >= s["N"] >= s["n"] for s in strata), str(strata[:3]))
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
