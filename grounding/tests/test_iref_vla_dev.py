"""A2.6b: the development preparation. Run directly:
python grounding/tests/test_iref_vla_dev.py --samples DIR --official DIR

- --samples: the pinned sample files (the release module's fetch-samples writes them).
- --official: the pinned ScanNet official lists (its fetch-official-splits writes them).

The fixture is a small release built from the pinned ScanNet scene, renamed to real ScanNet list names: development
(scene0081_00 and _01, one environment; scene0063_00), calibration (scene0011_00), training (scene0191_00) and legacy
(scene0010_01). Each scene keeps a few statement texts. scene0063_00 carries correction 1's cases, built from existing,
consistent entries: one text whose entries name two targets, and one whose entries name one target under two relations.
The real chain runs (import, audit, preparation) with the pinned tokenizer. Without the samples or the lists, the
checks fail rather than skip.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
FAILS, PASSES = [], []


def check(name, ok, detail=""):
    (PASSES if ok else FAILS).append(name)
    print(("  ok    " if ok else "  FAIL  ") + name + ("" if ok or not detail else f"  [{detail}]"))


SCENES = {"scene0081_00": 0, "scene0081_01": 1, "scene0063_00": 2, "scene0011_00": 3, "scene0191_00": 4, "scene0010_01": 5,
          "scene0081_02": 6}   # scene0081_02: a third scan of one environment, with no statements
KEEP = 9   # statement texts kept per scene


def build_release(samples: Path, folder: Path):
    """The fixture release folder (Scannet/<scene>/...), and the two injected texts of scene0063_00. Each scene is the
    pinned scene renamed, keeping a few texts; one object no kept statement uses is lengthened by 0.01 m x (index + 1),
    identically in the CSV and the scene graph, so each scene has its own object fingerprint, as distinct scans do."""
    R = importlib.import_module("grounding.adapters.iref_vla.release")
    src = samples / "sample_data" / "Scannet" / "scene0010_01"
    base = json.loads((src / "scene0010_01_referential_statements.json").read_text(encoding="utf-8"))
    texts = sorted(base["regions"]["0"])
    injected = {}
    for scene, k in SCENES.items():
        d = folder / "Scannet" / scene
        d.mkdir(parents=True)
        keep = texts[k * KEEP:(k + 1) * KEEP] if scene != "scene0081_02" else []
        region = {t: base["regions"]["0"][t] for t in keep}
        if scene == "scene0063_00":
            a, b2 = keep[0], keep[1]
            other = next(t for t in texts if base["regions"]["0"][t][0]["target_index"] !=
                         base["regions"]["0"][a][0]["target_index"])
            region[a] = [base["regions"]["0"][a][0], base["regions"]["0"][other][0]]   # two targets
            tb = base["regions"]["0"][b2][0]
            same = next(t for t in texts if t != b2 and base["regions"]["0"][t][0]["target_index"] == tb["target_index"]
                        and base["regions"]["0"][t][0]["relation"] != tb["relation"])
            region[b2] = [tb, base["regions"]["0"][same][0]]   # one target, two relations
            injected = {"conflict": a, "disagreement": b2}
        used = {e["target_index"] for es in region.values() for e in es}
        used |= {a["index"] for es in region.values() for e in es for a in e["anchors"].values()}
        used |= {x for es in region.values() for e in es for x in e["distractor_ids"]}
        rows = (src / "scene0010_01_object_result.csv").read_text(encoding="utf-8").split("\n")
        free = min((r.split(",")[0] for r in rows[1:] if r and r.split(",")[0] not in used), key=int)
        delta = 0.01 * (k + 1)
        for i, r in enumerate(rows):
            cells = r.split(",")
            if i and r and cells[0] == free:
                cells[10] = repr(float(cells[10]) + delta)   # object_bbox_xlength
                volume = float(cells[10]) * float(cells[11]) * float(cells[12])   # as the converter computes it
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
        st = {"scene_name": scene, "regions": {"0": region}}
        (d / f"{scene}_referential_statements.json").write_text(json.dumps(st), encoding="utf-8")
    return injected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=os.environ.get("SECOND_EYES_IREF_RELEASE_SAMPLES"))
    ap.add_argument("--official", default=os.environ.get("SECOND_EYES_IREF_OFFICIAL"))
    a = ap.parse_args()
    R = importlib.import_module("grounding.adapters.iref_vla.release")
    D = importlib.import_module("grounding.preparation.iref_vla_dev.dev")
    have = bool(a.samples and a.official) and not R.verify_samples(a.samples)
    if have:
        try:
            R.load_official(a.official, ["Scannet"])
        except R.ReleaseError:
            have = False
    check("the pinned samples and the pinned ScanNet official lists are available", have,
          "pass --samples DIR and --official DIR")
    print("-- sampling rules (synthetic rows)")
    rows = [{"group": "g1", "stratum": s, "eligible": True, "parent_command_id": f"{s}{i}", "partition": "development"}
            for s, n in (("a", 5), ("b", 1), ("c", 3)) for i in range(n)]
    rows += [{"group": "g1", "stratum": "a", "eligible": False, "parent_command_id": "x", "partition": "development"}]
    strata, sel = D.sample(rows, ["g1", "g2"], cap=6, salt="t")
    n = {s["stratum"]: s["n"] for s in strata if s["group"] == "g1"}
    check("round-robin over nonempty strata in name order: cap 6 over strata of 5, 1 and 3 gives 3, 1 and 2",
          n == {"a": 3, "b": 1, "c": 2} and len(sel) == 6, str(n))
    check("N counts eligible parents only; a zero-eligible environment is reported with N = 0, never replaced",
          {s["stratum"]: s["N"] for s in strata if s["group"] == "g1"} == {"a": 5, "b": 1, "c": 3}
          and [s for s in strata if s["group"] == "g2"] == [{"group": "g2", "stratum": None, "N": 0, "n": 0}])
    order = [r["parent_command_id"] for r in sel if r["stratum"] == "a"]
    check("within a stratum, parents follow the salted hash, and the same inputs give the same selection",
          order == sorted(order, key=lambda p: D._key("t", "g1", "a", p))[:3] and D.sample(rows, ["g1", "g2"], cap=6, salt="t")[1] == sel)
    E = lambda t, rel, anchors: {"annotation_id": f"{t}-{rel}", "source_payload": {  # noqa: E731
        "target_index": t, "relation": rel, "relation_type": "binary", "anchors": {f"anchor_{i}": {"index": x} for i, x in enumerate(anchors)}}}
    r1 = D.resolve([E("5", "near", ["1"]), E("5", "near", ["1"])])
    r2 = D.resolve([E("5", "near", ["1"]), E("7", "near", ["1"])])
    r3 = D.resolve([E("5", "near", ["1"]), E("5", "closest", ["1"])])
    r4 = D.resolve([E("5", "near", ["1"]), E("5", "near", ["2"])])
    check("correction 1: repeated entries naming one target are one unique target, counted once, with provenance",
          r1["target_status"] == "unique_target" and r1["target_object_id"] == "obj_005" and r1["entries"] == 2
          and r1["parsing_label"] == "consistent" and r1["stratum"] == "near")
    check("correction 1: two targets are conflicting_targets; one target under two relations or two anchor sets is a "
          "disagreeing parsing label, its stratum explicit, never entry order",
          r2["target_status"] == "conflicting_targets" and r2["target_object_id"] is None
          and r3["target_status"] == "unique_target" and r3["parsing_label"] == "disagreeing" and r3["stratum"] == "relation_disagreement"
          and r4["parsing_label"] == "disagreeing" and r4["stratum"] == "near"
          and D.resolve(list(reversed([E("5", "near", ["1"]), E("5", "closest", ["1"])]))) == r3)
    print("-- transient file locks (stub stages)")
    slept, att = [], []
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("[WinError 5] Access is denied")
        return "done"
    with tempfile.TemporaryDirectory() as td:
        ok1 = D._stage(flaky, Path(td) / "never", att, "import", sleep=slept.append) == "done"
        check("a transient lock is retried (twice here), each retry recorded with its pause, then the stage succeeds",
              ok1 and len(att) == 2 and slept == [2.0, 6.0] and all(a["stage"] == "import" for a in att))

        def data_error():
            raise ValueError("E_IREF_SOURCE_SHAPE")
        try:
            D._stage(data_error, Path(td) / "never", [], "import", sleep=slept.append)
            ok2 = False
        except ValueError:
            ok2 = True
        (Path(td) / "published").mkdir()

        def locked():
            raise PermissionError("[WinError 5] Access is denied")
        try:
            D._stage(locked, Path(td) / "published", [], "import", sleep=lambda s: None)
            ok3 = False
        except PermissionError:
            ok3 = True
        always = []
        try:
            D._stage(locked, Path(td) / "never", always, "import", sleep=lambda s: None)
            ok4 = False
        except PermissionError:
            ok4 = len(always) == D.ATTEMPTS - 1
        check("a data error is never retried; nothing is retried once something was published; the third failure is final",
              ok2 and ok3 and ok4)
    if not have:
        print(f"\n{len(PASSES)} passed, {len(FAILS)} failed")
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        injected = build_release(Path(a.samples), tmp / "release")
        vocab = Path(a.samples) / "data" / "NYU_Object_Classes.csv"
        inv = R.build_inventory([("Scannet", tmp / "release" / "Scannet")], R.read_vocabulary(vocab.read_bytes()))
        R.write_inventory(inv, tmp / "inventory")
        zp = tmp / "Scannet.zip"
        with zipfile.ZipFile(zp, "w") as z:
            for f in sorted((tmp / "release").rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(tmp / "release").as_posix())
        print("-- the frozen specification")
        spec = D.freeze(inventory_dir=tmp / "inventory", official_dir=a.official, out=tmp / "spec")
        asg = spec["partition"]["assignment"]
        check("freeze: official-first assignment of the fixture's real names; development, legacy, calibration, training",
              spec["development_groups"] == ["scannet:scene0063", "scannet:scene0081"] and spec["legacy_groups"] == ["scannet:scene0010"]
              and asg["scannet:scene0011"] == "calibration" and asg["scannet:scene0191"] == "training"
              and spec["scenes"]["scannet:scene0081"] == ["scene0081_00", "scene0081_01", "scene0081_02"], str(asg))
        check("freeze: the cap, salts, eligibility, desktop rule and the bootstrap's literal seed are in the hashed specification",
              spec["sampling"]["cap_per_environment"] == 64 and spec["analysis"]["bootstrap"]["seed"] == 20261010
              and spec["desktop_subset"]["parents"] == 32 and len(spec["eligibility"]) == 3
              and hashlib.sha256((Path(spec["folder"]) / "spec.json").read_bytes()).hexdigest() == spec["spec_sha256"])
        bad_inv = tmp / "inventory-altered"
        shutil.copytree(tmp / "inventory", bad_inv)
        (bad_inv / "inventory.json").write_bytes((tmp / "inventory" / "inventory.json").read_bytes() + b" ")
        try:
            D.prepare(spec=Path(spec["folder"]) / "spec.json", zip_path=zp, inventory_dir=bad_inv, vocabulary=vocab,
                      out=tmp / "x", work=tmp / "xw", progress=lambda m: None)
            refused = False
        except D.DevPrepError:
            refused = not (tmp / "xw").exists()
        check("prepare refuses an inventory other than the frozen one, before any work", refused)
        print("-- the preparation (the real chain, pinned tokenizer)")
        read = []
        real = zipfile.ZipFile.read

        def spy(self, name, *x, **k):
            read.append(name if isinstance(name, str) else name.filename)
            return real(self, name, *x, **k)
        zipfile.ZipFile.read = spy
        real_import = D.R.import_release_scene
        locks = {"scene0063_00": 1}   # its first import meets a simulated Windows file lock

        def locking_import(source, scene, *x, **k):
            if locks.get(scene):
                locks[scene] -= 1
                raise PermissionError("[WinError 5] Access is denied (simulated)")
            return real_import(source, scene, *x, **k)
        D.R.import_release_scene = locking_import
        try:
            m = D.prepare(spec=Path(spec["folder"]) / "spec.json", zip_path=zp, inventory_dir=tmp / "inventory",
                          vocabulary=vocab, out=tmp / "out", work=tmp / "work", progress=lambda msg: None,
                          review={"receipt.json": vocab})
        finally:
            zipfile.ZipFile.read = real
            D.R.import_release_scene = real_import
        out = Path(m["folder"])
        c = m["counts"]
        check("only development and legacy scenes are read from the zip; calibration and training are never read",
              read and not any(("scene0011_00" in n or "scene0191_00" in n) for n in read)
              and {n.split("/")[1] for n in read} == {"scene0081_00", "scene0081_01", "scene0081_02", "scene0063_00", "scene0010_01"})
        check("every assigned scene is accounted for: four prepared, and the scan without statements recorded as such",
              c["scenes"] == {"prepared": 4, "no_statements": 1}, str(c["scenes"]))
        scenes = {json.loads(x)["scene"]: json.loads(x) for x in (out / "scenes.jsonl").read_text(encoding="utf-8").splitlines() if x}
        sim = scenes["scene0063_00"].get("retries") or []
        others = {s: v for s, v in scenes.items() if v.get("retries") and s != "scene0063_00"}
        lock = lambda r: "PermissionError" in r["error"] or "WinError" in r["error"]  # noqa: E731
        check("a scene whose first import met a (simulated) file lock was retried, prepared, and reported with that retry; "
              "any other retry (a real lock on this machine) is recorded as a transient lock, and its scene prepared too",
              scenes["scene0063_00"]["status"] == "prepared" and sim and sim[0]["stage"] == "import"
              and "simulated" in sim[0]["error"] and all(lock(r) for r in sim)
              and all(v["status"] == "prepared" and all(lock(r) for r in v["retries"]) for v in others.values())
              and sorted(c["scenes_retried"]) == sorted({"scene0063_00", *others}),
              json.dumps({"retried": c["scenes_retried"], "scene0063_00": sim,
                          "others": {s: v["retries"] for s, v in others.items()}})[:900])
        elig = [json.loads(x) for x in (out / "reference_only" / "eligibility.jsonl").read_text(encoding="utf-8").splitlines() if x]
        texts = json.loads((tmp / "release" / "Scannet" / "scene0063_00" / "scene0063_00_referential_statements.json").read_text())
        conv = importlib.import_module("grounding.adapters.iref_vla.convert")
        idfor = lambda s, t: f"iref.scannet.{s}.r0.e." + hashlib.sha256(t.encode("utf-8")).hexdigest()  # noqa: E731
        conflict = next(r for r in elig if r["parent_command_id"] == idfor("scene0063_00", injected["conflict"]))
        disagree = next(r for r in elig if r["parent_command_id"] == idfor("scene0063_00", injected["disagreement"]))
        check("correction 1 on real records: the two-target text is quarantined; the one-target, two-relation text is a "
              "unique target with a disagreeing label in the explicit stratum",
              conflict["target_status"] == "conflicting_targets" and not conflict["eligible"] and "conflicting_targets" in conflict["reasons"]
              and conflict["entries"] == 2 and disagree["target_status"] == "unique_target" and disagree["parsing_label"] == "disagreeing"
              and disagree["stratum"] == "relation_disagreement", str((conflict["reasons"], disagree["stratum"])))
        ok_rule = all(r["eligible"] == (not r["reasons"]) for r in elig) and all(
            all(3 <= r["views"][v]["objects"] <= 10 and r["views"][v]["status"] == "materialized" for v in D.VIEWS)
            for r in elig if r["eligible"])
        check("eligible exactly when no reason applies; every eligible parent has 3 to 10 materialized objects in both views",
              ok_rule and c["eligible_parents"].get("development", 0) > 0, str(c["eligible_parents"]))
        strata = [json.loads(x) for x in (out / "strata.jsonl").read_text(encoding="utf-8").splitlines() if x]
        cnt = {}
        for r in elig:
            if r["eligible"]:
                cnt[(r["group"], r["stratum"])] = cnt.get((r["group"], r["stratum"]), 0) + 1
        check("the strata's N equal the eligible parents per environment and stratum; n never exceeds N or the cap",
              all(s["N"] == cnt.get((s["group"], s["stratum"]), 0) for s in strata if s["stratum"])
              and all(s["n"] <= s["N"] for s in strata)
              and all(sum(s["n"] for s in strata if s["group"] == g) <= 64 for g in {s["group"] for s in strata}))
        reqs = [json.loads(x) for x in (out / "requests.jsonl").read_text(encoding="utf-8").splitlines() if x]
        per = {}
        for r in reqs:
            per.setdefault(r["parent_command_id"], set()).add((r["view"], r["format"]))
        full = {(v, f) for v in D.VIEWS for f in D.FORMATS}
        check("four matched requests per built parent (both views, both formats), unique IDs, operational and uncached",
              reqs and all(s == full for s in per.values()) and len({r["request_id"] for r in reqs}) == len(reqs)
              and all(r["purpose"] == "operational" and r["cache_mode"] == "Off" for r in reqs)
              and len(per) == c["parents_with_all_four_requests"])
        PG = importlib.import_module("grounding.quest.prompt_goldens")
        proto = importlib.import_module("grounding.inference.iref_vla.protocol").load_protocol()
        tok = importlib.import_module("grounding.preparation.iref_vla.tokens").load_pinned_tokenizer(D.TOKENIZER)
        r0 = next(r for r in reqs if r["format"] == "coordinates_v2")
        b = tmp / "work" / r0["scene"] / "bundle"
        row = next(e for e in elig if e["parent_command_id"] == r0["parent_command_id"])["views"][r0["view"]]
        scene, command = json.loads((b / row["scene_path"]).read_text()), json.loads((b / row["command_path"]).read_text())
        cmap = json.loads((b / "model_records" / "category-map.json").read_text())
        gold, _ = PG._golden(1, "dataset_command", "s", len(scene["objects"]), command["command_id"],
                             (b / row["scene_path"]).read_bytes(), (b / row["command_path"]).read_bytes(), scene, command, cmap, proto, tok)
        same = all(r0[k] == gold[k] for k in ("document", "prompt_sha256", "token_ids", "codes", "code_token_ids", "mapping_sha256",
                                                 "keep_before_last_token", "keep_before_command_line")) and r0["choice_object_ids"] == gold["targets"]
        check("a coordinates_v2 request equals A2.5's D104 golden construction exactly (document, prompt, tokens, mapping, boundaries)", same)
        desk = json.loads((out / "desktop.json").read_text())["request_ids"]
        byid = {r["request_id"]: r for r in reqs}
        dp = {byid[i]["parent_command_id"] for i in desk}
        check("the desktop subset: coordinates_v2, both views per chosen development parent, at most 32 parents",
              desk and all(byid[i]["format"] == "coordinates_v2" and byid[i]["partition"] == "development" for i in desk)
              and len(desk) == 2 * len(dp) and len(dp) <= 32)
        text = (out / "requests.jsonl").read_text(encoding="utf-8")
        check("answers stay out of the requests: no target field in requests; targets only under reference_only",
              "target_object_id" not in text and "target_offered" not in text
              and (out / "reference_only" / "targets.jsonl").is_file())
        man = json.loads((out / "manifest.json").read_text())
        check("the manifest hashes every file, including the request manifest, sampling and desktop subset, and the review copies",
              all(hashlib.sha256((out / n).read_bytes()).hexdigest() == h for n, h in man["files"].items())
              and man["hashes"]["requests"] == man["files"]["requests.jsonl"] and "review/receipt.json" in man["files"]
              and man["status"].startswith("prepared"))
        legacy = [r for r in reqs if r["partition"] == "legacy-development"]
        check("the legacy environment is prepared beside development, labelled, and never in the desktop subset",
              all(r["group"] == "scannet:scene0010" for r in legacy) and not (set(desk) & {r["request_id"] for r in legacy}))
    print(f"\n{len(PASSES)} passed, {len(FAILS)} failed")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
