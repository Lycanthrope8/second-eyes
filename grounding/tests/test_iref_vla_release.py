"""A2.6a: the versioned release reader (iref_vla_release.v1). Run directly with the pinned sample files:
python grounding/tests/test_iref_vla_release.py --samples DIR, or with SECOND_EYES_IREF_RELEASE_SAMPLES set. DIR holds
sample_data/ and data/NYU_Object_Classes.csv at the pinned commit (the release module's fetch-samples writes them).
Without the pinned samples, the sample checks fail rather than skip (as A2.2a's adapter tests do). Expectations are
written by hand from the pinned files.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import io
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


EXPECTED = {"3RScan": (331, 334, 1), "ARKitScenes": (88, 88, 1), "HM3D": (12748, 12781, 23), "Matterport": (5475, 5490, 19),
            "Scannet": (1936, 1951, 1), "Unity": (2496, 2511, 3)}   # parent texts, entries, regions


def sources(root):
    return [(s, root / "sample_data" / s) for s in EXPECTED]


def scene_files(root, source):
    d = root / "sample_data" / source / __import__("grounding.adapters.iref_vla.release", fromlist=["x"]).SAMPLE_SCENES[source]
    return {p.name[len(d.name) + 1:]: p.read_bytes() for p in d.iterdir()   # the kind follows "<scene>_"
            if not p.name.endswith((".ply", ".npy", ".txt"))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=os.environ.get("SECOND_EYES_IREF_RELEASE_SAMPLES"))
    root = ap.parse_args().samples
    R = importlib.import_module("grounding.adapters.iref_vla.release")
    C = importlib.import_module("grounding.adapters.iref_vla.convert")
    K = importlib.import_module("grounding.adapters.iref_vla.pinned")
    print("-- access (fake responses; nothing is fetched)")
    import hashlib

    class Head(io.BytesIO):
        def __init__(self, b, headers):
            super().__init__(b)
            self.headers = headers

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    body = b"PK" + bytes(range(256)) * 1000
    heads = {"Content-Length": str(len(body)), "ETag": '"abc"', "Last-Modified": "Sat, 05 Oct 2024 10:00:00 GMT"}
    seen = []

    def head_opener(req, timeout=None):
        seen.append((req.get_method(), req.full_url))
        return Head(b"", heads)
    pr = R.probe(opener=head_opener)
    check("probe asks each of the six source zips for its headers only, recording size, ETag and date",
          len(seen) == 6 and all(m == "HEAD" for m, _ in seen) and seen[0][1].startswith(R.DOWNLOAD_ENDPOINT + "/iref-vla/")
          and pr["sources"]["Scannet"] == {"bytes": len(body), "etag": "abc", "last_modified": heads["Last-Modified"]})
    with tempfile.TemporaryDirectory() as dd:
        r = R.download("Scannet", dd, opener=lambda u, timeout=None: Head(body, heads), chunk=4096, progress=lambda m: None)
        check("download keeps the zip only when complete, hashed as it arrived, with a provenance receipt",
              r["sha256"] == hashlib.sha256(body).hexdigest() and (Path(dd) / "Scannet.zip").read_bytes() == body
              and json.loads((Path(dd) / "Scannet.zip.receipt.json").read_text(encoding="utf-8"))["release_commit"] == R.COMMIT)
        try:
            R.download("Scannet", dd, opener=lambda u, timeout=None: Head(body, heads))
            ow = False
        except R.ReleaseError:
            ow = True
        try:
            R.download("Unity", dd, opener=lambda u, timeout=None: Head(body, {"Content-Length": str(len(body) + 5)}), progress=lambda m: None)
            short = False
        except R.ReleaseError:
            short = not (Path(dd) / "Unity.zip").exists() and not (Path(dd) / "Unity.zip.partial").exists()
        check("a kept copy is never overwritten, and a short transfer is refused leaving nothing behind", ow and short)
    print("-- the pinned samples")
    have = root is not None and Path(root).is_dir() and not R.verify_samples(root)
    check("pinned samples available and exact (25 files: six scenes' four text files, and the vocabulary)", have,
          "pass --samples DIR (fetch-samples writes it)" if root is None else "; ".join(R.verify_samples(root)[:3]) if Path(root).is_dir() else root)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # fetch-samples with a fake opener: verified before kept
        blobs = {}
        if have:
            blobs = {p: (Path(root) / p).read_bytes() for p, _, _ in R._pinned_files()}

        class Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        if have:
            opener = lambda url: Resp(blobs[url[len(R.URL_PREFIX):]])  # noqa: E731
            res = R.fetch_samples(tmp / "fetched", opener=opener, progress=lambda m: None)
            again = R.fetch_samples(tmp / "fetched", opener=opener, progress=lambda m: None)
            check("fetch-samples verifies each file before keeping it, and keeps verified copies on a second run",
                  res["fetched"] == 25 and again == {"fetched": 0, "already_present": 25, "root": str(tmp / "fetched")}
                  and not R.verify_samples(tmp / "fetched"))
            bad = dict(blobs)
            first = sorted(R.SAMPLE_PINS)[0]
            bad[first] = blobs[first] + b"x"
            try:
                R.fetch_samples(tmp / "bad", opener=lambda url: Resp(bad[url[len(R.URL_PREFIX):]]), progress=lambda m: None)
                ok = False
            except R.ReleaseError:
                ok = not (tmp / "bad" / first).exists()
            check("fetch-samples refuses a download that differs from its pin, and keeps nothing of it", ok)
        if not have:
            print(f"\n{len(PASSES)} passed, {len(FAILS)} failed")
            return 1
        root = Path(root)
        vocab = R.read_vocabulary((root / "data" / "NYU_Object_Classes.csv").read_bytes())
        print("-- the six sources")
        inv = R.build_inventory(sources(root), vocab)
        by = inv["sources"]
        check("all six sample scenes are retained, with no statement excluded",
              all(by[s]["retained"] == 1 and by[s]["rejected"] == 0 and not by[s]["entry_exclusions"] for s in EXPECTED), str(by)[:200])
        check("parent texts, entries and regions per source equal the pinned files'",
              all((by[s]["texts"], by[s]["entries"]) == EXPECTED[s][:2] for s in EXPECTED)
              and all(r["regions"] == EXPECTED[r["source"]][2] for r in inv["scenes"]))
        unity = next(r for r in inv["scenes"] if r["source"] == "Unity")
        check("Unity's no-region objects (region -1) are counted, and are exactly the objects outside every region's count",
              unity["objects_without_region"] == 14
              and sum(v["objects"] for v in unity["per_region"].values()) == unity["objects"] - 14 and inv["official_lists"] == {})
        own = tmp / "UnityLists"
        (own / "loft").mkdir(parents=True)
        for k in R.KINDS:   # the four pinned files only, whatever else the samples folder holds
            (own / "loft" / f"loft_{k}").write_bytes((root / "sample_data" / "Unity" / "loft" / f"loft_{k}").read_bytes())
        (own / "loft" / "object_list.txt").write_text("loft\n", encoding="utf-8")   # looks like a scene list, but sits in the scene
        (own / "unity.txt").write_text("loft\n", encoding="utf-8")
        linv = R.build_inventory([("Unity", own)], vocab)
        check("a text file in a scene's own folder is never a split list, even when its lines are scene names; a top-level one is",
              linv["official_lists"] == {"Unity/unity.txt": ["loft"]} and linv["other_text_files"] == ["Unity/loft/object_list.txt"],
              str((linv["official_lists"], linv["other_text_files"])))
        g = inv["groups"]
        check("groups: the ScanNet scan series is the legacy group; Matterport, HM3D and Unity by name; 3RScan and ARKitScenes unverified",
              g["scannet:scene0010"]["legacy"] and g["scannet:scene0010"]["ancestry"] == "name"
              and all(g[k]["ancestry"] == "name" for k in ("matterport:5q7pvUzZiYa", "hm3d:00238-j6fHrce9pHR", "unity:loft"))
              and g["3rscan:02b33e01-be2b-2d54-93fb-4145a709cec5"]["ancestry"] == "unverified"
              and g["arkitscenes:47204840"]["ancestry"] == "unverified" and len(g) == 6)
        print("-- regression against the accepted one-scene adapter (v1)")
        sn = R.inspect_scene("Scannet", "scene0010_01", scene_files(root, "Scannet"), vocab, keep_entries=True)
        f = lambda k: (root / K.FILES[k]["path"]).read_bytes()  # noqa: E731
        stage = C.convert_scene(f("objects"), f("regions"), f("vocabulary"))
        cmds = C.convert_commands(f("statements"), stage)
        ann = C.convert_annotations(f("statements"), stage)["entries"]
        v1 = sorted((a["command_id"], a["source_payload"]["target_index"]) for a in ann)
        v2 = sorted((C.command_id(text), t) for _, text, t in sn["entry_list"])
        check("the pinned ScanNet scene reads as v1 converts it: 1,936 commands, 1,951 entries, and every entry's command and target",
              len(cmds) == sn["texts"] == 1936 and len(ann) == sn["entries"] == 1951 and v1 == v2)
        check("its object count equals v1's", sn["objects"] == len(stage.object_ids))
        objs = {}
        import csv
        for row in csv.DictReader(io.StringIO(f("objects").decode("utf-8"), newline="")):
            objs[row["object_id"]] = row["nyu_label"]
        st = json.loads(f("statements"))
        fits = 0
        for text, entries in st["regions"]["0"].items():
            e = entries[0]
            cats = {objs[e["target_index"]]} | {objs[a["index"]] for a in e["anchors"].values()}
            n = sum(1 for o, lab in objs.items() if lab in cats or lab in ("", "unknown"))
            fits += 3 <= n <= 10
        check("the compatibility estimate equals an independent count of A2.2c's selection rule (1,530 of 1,936 fit 3 to 10)",
              sn["compatibility_estimate"]["fits"] == fits == 1530, f"{sn['compatibility_estimate']} vs {fits}")
        print("-- reading a release zip")
        z = tmp / "Scannet.zip"
        read = []
        with zipfile.ZipFile(z, "w") as zf:
            d = root / "sample_data" / "Scannet" / "scene0010_01"
            for p in sorted(d.iterdir()):
                zf.write(p, f"Scannet/scene0010_01/{p.name}")
            zf.writestr("Scannet/train.txt", "scene0010_01\n")
            zf.writestr("Scannet/notes.txt", "these are notes\n")
        real = zipfile.ZipFile.read

        def spy(self, name, *a, **k):
            read.append(name if isinstance(name, str) else name.filename)
            return real(self, name, *a, **k)
        zipfile.ZipFile.read = spy
        try:
            zinv = R.build_inventory([("Scannet", z)], vocab)
        finally:
            zipfile.ZipFile.read = real
        zs = zinv["scenes"][0]
        check("a source zip reads as its folder does (texts, entries, fingerprint, input hashes)",
              (zs["texts"], zs["entries"], zs["fingerprint"], zs["input_sha256"]) == (sn["texts"], sn["entries"], sn["fingerprint"], sn["input_sha256"]))
        check("only the four text kinds and .txt files are read from a zip, never the .ply or .npy geometry",
              read and not any(n.endswith((".ply", ".npy")) for n in read))
        check("a top-level scene list is an official list; a text file of other lines is only named",
              zinv["official_lists"] == {"Scannet/train.txt": ["scene0010_01"]} and zinv["other_text_files"] == ["Scannet/Scannet/notes.txt"],
              str((zinv["official_lists"], zinv["other_text_files"])))
        print("-- refusals and exclusions (altered copies of the pinned ScanNet scene)")
        base = scene_files(root, "Scannet")

        def alt(kind, fn):
            fs = dict(base)
            fs[kind] = fn(fs[kind])
            return R.inspect_scene("Scannet", "scene0010_01", fs, vocab)
        lines = base["object_result.csv"].decode("utf-8").split("\n")
        r1 = alt("object_result.csv", lambda b: ("\n".join(lines[:2] + [lines[1]] + lines[2:])).encode("utf-8"))
        check("a repeated object ID rejects the scene", not r1["retained"] and r1["rejections"][0]["code"] == "E_RELEASE_OBJECTS")
        cells = lines[1].split(",")
        cells[7] = "nan"
        r2 = alt("object_result.csv", lambda b: ("\n".join([lines[0], ",".join(cells)] + lines[2:])).encode("utf-8"))
        check("a non-finite box rejects the scene", not r2["retained"] and r2["rejections"][0]["code"] == "E_RELEASE_GEOMETRY")
        r3 = alt("object_result.csv", lambda b: b.replace(b"object_id,", b"objectid,", 1))
        check("a changed header rejects the scene (v1's strict reader)", not r3["retained"] and r3["rejections"][0]["code"] == "E_RELEASE_SOURCE_SHAPE")
        doc = json.loads(base["referential_statements.json"])
        d4 = copy.deepcopy(doc)
        first = next(iter(d4["regions"]["0"]))
        d4["regions"]["0"][first][0]["target_index"] = "9999"
        r4 = alt("referential_statements.json", lambda b: json.dumps(d4).encode("utf-8"))
        check("a statement whose target is not in the scene is excluded and counted; the scene is kept",
              r4["retained"] and r4["entry_exclusions"] == {"E_RELEASE_TARGET": 1} and r4["texts"] == 1935)
        d5 = copy.deepcopy(doc)
        d5["regions"]["0"][first][0]["target_class"] = "teapot"
        r5 = alt("referential_statements.json", lambda b: json.dumps(d5).encode("utf-8"))
        check("a target class that is not the object's nyu_label is excluded and counted", r5["entry_exclusions"] == {"E_RELEASE_TARGET_CLASS": 1})
        d6 = copy.deepcopy(doc)
        d6["scene_name"] = "scene0011_00"
        r6 = alt("referential_statements.json", lambda b: json.dumps(d6).encode("utf-8"))
        check("statements naming another scene reject the scene", not r6["retained"] and r6["rejections"][0]["code"] == "E_RELEASE_SCENE_NAME")
        r7 = R.inspect_scene("Matterport", "scene0010_01", base, vocab)
        check("a scene name that is not the source's pattern is refused", not r7["retained"] and r7["rejections"][0]["code"] == "E_RELEASE_SCENE_NAME")
        print("-- grouping")
        dup = tmp / "UnityDup"
        shutil.copytree(root / "sample_data" / "Unity", dup)
        (dup / "loft").rename(dup / "loft2")
        for p in sorted((dup / "loft2").iterdir()):
            p.rename(p.with_name(p.name.replace("loft_", "loft2_", 1)))
        for k in ("scene_graph.json", "referential_statements.json"):
            q = dup / "loft2" / f"loft2_{k}"
            j = json.loads(q.read_text(encoding="utf-8"))
            j["scene_name"] = "loft2"
            q.write_text(json.dumps(j), encoding="utf-8")
        ginv = R.build_inventory([("Unity", root / "sample_data" / "Unity"), ("Unity", dup)], vocab)
        check("two scenes with identical object fingerprints are merged into one group and flagged",
              list(ginv["groups"]) == ["unity:loft"] and ginv["merged_by_fingerprint"][0]["groups"] == ["unity:loft", "unity:loft2"]
              and len(ginv["groups"]["unity:loft"]["scenes"]) == 2)
        ainv = R.build_inventory([("3RScan", root / "sample_data" / "3RScan")], vocab,
                                 {"3RScan": {"02b33e01-be2b-2d54-93fb-4145a709cec5": "reference-a"}})
        check("supplied metadata gives a 3RScan scan its reference group, marked as from metadata",
              list(ainv["groups"]) == ["3rscan:reference-a"] and ainv["groups"]["3rscan:reference-a"]["ancestry"] == "metadata")
        print("-- the partition proposal")
        prop = R.propose_partitions(inv)
        a = prop["assignment"]
        check("the legacy group goes to legacy-development, unverified groups are unassigned, never frozen",
              a["scannet:scene0010"]["partition"] == "legacy-development"
              and a["3rscan:02b33e01-be2b-2d54-93fb-4145a709cec5"]["partition"] == "unassigned"
              and a["arkitscenes:47204840"]["partition"] == "unassigned" and prop["status"] == "proposed, not frozen")
        check("the hashed assignment is deterministic, and the policy's hash is recorded",
              R.propose_partitions(inv) == prop and len(prop["policy_sha256"]) == 64)
        oinv = copy.deepcopy(inv)
        oinv["official_lists"] = {"Matterport/train.txt": ["5q7pvUzZiYa"], "HM3D/val.txt": ["00238-j6fHrce9pHR"],
                                  "Unity/train.txt": ["loft"], "Unity/test.txt": ["loft"]}
        op = R.propose_partitions(oinv)["assignment"]
        check("official lists are preserved: a training list gives training, a held-out list never training, both is a conflict",
              op["matterport:5q7pvUzZiYa"] == {"partition": "training", "basis": "an official training list"}
              and op["hm3d:00238-j6fHrce9pHR"]["partition"] in ("development", "calibration", "test")
              and op["unity:loft"]["partition"] == "unassigned" and "listed as both" in op["unity:loft"]["basis"])
        out = R.write_proposal(prop, tmp / "proposal")
        try:
            R.write_proposal(prop, tmp / "proposal")
            again = False
        except R.ReleaseError:
            again = True
        check("outputs are written to a new folder once, never over an earlier one",
              (out / "proposal.json").is_file() and (out / "report.md").is_file() and again)
    print(f"\n{len(PASSES)} passed, {len(FAILS)} failed")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
