"""A2.6a: the versioned multi-scene IRef-VLA release reader (iref_vla_release.v1).

The A2.2a adapter stays a one-scene adapter: `pinned.py` and `convert.py` are unchanged, with their pin checks. This
module reads the same source shape for any scene of the published release, at the same pinned commit (`741cf02`).

**It validates, counts and groups.** It converts nothing into scene or command records; that is the preparation's work
in A2.6b, through its own versioned step.

**Shared with v1, not copied:**

- the object and region headers;
- the accepted row widths (whole colour groups may be omitted);
- the statement record keys;
- the strict CSV and JSON readers, with their issue records.

**What it adds:**

- **The six sample scenes**, one per source, pinned by size and SHA-256 and fetchable at the pinned commit.
- **Reading a source** from a folder or straight from its release zip. Only the four text files of each scene are read,
  never the `.ply` or `.npy` geometry.
- **Validation of each scene:**
  - its name;
  - unique decimal object IDs;
  - finite boxes;
  - regions;
  - the scene graph's and the statements' scene names;
  - each statement entry's correspondence to the scene: the target, distractors and anchors exist; the target lies in
    the statement's region; and the target class equals the object's `nyu_label`.
- **Grouping by physical environment and ancestry,** recording how each group is known:
  - **by name:** ScanNet rescans share `sceneNNNN`; Matterport, HM3D and Unity scenes are single places, their regions
    kept together;
  - **from supplied metadata;**
  - **unverified:** 3RScan rescans and ARKitScenes visits need their sources' metadata.
  - Scenes with identical object fingerprints are merged into one group, and flagged.
- **A compatibility estimate** with the target-independent selection: A2.2c's own `select`, applied to the statement's
  annotated categories (the target's and the anchors' `nyu_label`), bucketed by the 3-to-10 range. The accepted parser's
  exact categories come with the preparation.
- **A partition proposal** by salted group hash. The legacy group goes to `legacy-development`; groups with unverified
  ancestry are left unassigned; official lists are preserved. A proposal is never frozen here.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from . import convert as V1C
from . import output as V1O
from . import pinned as V1
from .sources import Issues, read_csv, read_json
from ...subscenes.iref_vla import selector as SEL

RELEASE_VERSION = "iref_vla_release.v1"
COMMIT = V1.COMMIT
URL_PREFIX = V1.URL_PREFIX
KINDS = ("object_result.csv", "region_result.csv", "scene_graph.json", "referential_statements.json")
VOCABULARY = V1.FILES["vocabulary"]
SAMPLE_SCENES = {"Scannet": "scene0010_01", "3RScan": "02b33e01-be2b-2d54-93fb-4145a709cec5", "ARKitScenes": "47204840",
                 "Matterport": "5q7pvUzZiYa", "HM3D": "00238-j6fHrce9pHR", "Unity": "loft"}
SAMPLE_PINS = {
    "sample_data/3RScan/02b33e01-be2b-2d54-93fb-4145a709cec5/02b33e01-be2b-2d54-93fb-4145a709cec5_object_result.csv": (8417, "1fdc1433a698919dbae6ed8b9e50425989c86f2faa5729f5e1ac129e9750e1dd"),
    "sample_data/3RScan/02b33e01-be2b-2d54-93fb-4145a709cec5/02b33e01-be2b-2d54-93fb-4145a709cec5_referential_statements.json": (760469, "fd4d4df18b009a85dfd49f075436cdd9db770735937827e5dadb8a84bc32894b"),
    "sample_data/3RScan/02b33e01-be2b-2d54-93fb-4145a709cec5/02b33e01-be2b-2d54-93fb-4145a709cec5_region_result.csv": (272, "16bd9415b03d1661b7ecc0eabecd8b9eac301a772fecde2678e11b3f907c5939"),
    "sample_data/3RScan/02b33e01-be2b-2d54-93fb-4145a709cec5/02b33e01-be2b-2d54-93fb-4145a709cec5_scene_graph.json": (111899, "397aa305f9d9bae61fc3ace052bf07dceba63e2bcc989f485de0e6b82b166956"),
    "sample_data/ARKitScenes/47204840/47204840_object_result.csv": (2699, "b6cdf948d98e42201593ce489008eabe601589f9b676554e011300dfdf69ab4c"),
    "sample_data/ARKitScenes/47204840/47204840_referential_statements.json": (211396, "e4af8e9e0663aacf712c46fc34e072c53fa06ab0c9982ce22a55141f0e692a5a"),
    "sample_data/ARKitScenes/47204840/47204840_region_result.csv": (275, "8427007f81b75d4ce3c6c370380709c475fbf8941ae94d30d3ad9b2277e56d2c"),
    "sample_data/ARKitScenes/47204840/47204840_scene_graph.json": (29786, "7a63819865eb9614d2a6d8ad4b5a232718d9846cc5219a356b724a6b0bf932af"),
    "sample_data/HM3D/00238-j6fHrce9pHR/00238-j6fHrce9pHR_object_result.csv": (253215, "1b3f764cea7581b33b49f55593da372004a44f7ab99cb46c0648352902b59ff8"),
    "sample_data/HM3D/00238-j6fHrce9pHR/00238-j6fHrce9pHR_referential_statements.json": (30113020, "ad1b1c793a8b855ba72b96e897abeebf72d13ea735b83c6ec5378d1813ede569"),
    "sample_data/HM3D/00238-j6fHrce9pHR/00238-j6fHrce9pHR_region_result.csv": (3066, "c6925e0c2c77421fc9c5c956bfaaa0ffd1935e5478cd1d041b0b4a4e636fa8cc"),
    "sample_data/HM3D/00238-j6fHrce9pHR/00238-j6fHrce9pHR_scene_graph.json": (4058843, "d076b4e71972bab9d1441ba1535a93e06b8d31bc5645e9d1bbba7fc00adf922e"),
    "sample_data/Matterport/5q7pvUzZiYa/5q7pvUzZiYa_object_result.csv": (112158, "0eccf484ac5ae794f437c4b5859736fa650b6379e432a2e27ed7bbbb84ed8b87"),
    "sample_data/Matterport/5q7pvUzZiYa/5q7pvUzZiYa_referential_statements.json": (12786677, "e834c3c72672e7db9bb9d51f1e64a7c78e226434bacfd2f3bcabcd2dc6658168"),
    "sample_data/Matterport/5q7pvUzZiYa/5q7pvUzZiYa_region_result.csv": (1954, "7d47a40990e756eef526ca94f1fa5a8e637de844a5020439b2dd85ca763d3fbd"),
    "sample_data/Matterport/5q7pvUzZiYa/5q7pvUzZiYa_scene_graph.json": (1589999, "0116bb5b30c1adc833920767f0e08afb0bdceba40c1bdef835dc6bb0a0ce22fd"),
    "sample_data/Scannet/scene0010_01/scene0010_01_object_result.csv": (18654, "3bd5f282abf5cf5c6abe5d106a2fd87ad6090ec810d50b9e66e4d1e17aaf2996"),
    "sample_data/Scannet/scene0010_01/scene0010_01_referential_statements.json": (4716682, "0c077963bcdb0e7f19aa0885ddb547689dfe106da622d8d4b58d4e894d143d08"),
    "sample_data/Scannet/scene0010_01/scene0010_01_region_result.csv": (224, "1b77150f1e65bb5c377457bdfdb944575864bd04ab88fb94a24cc40510ee0091"),
    "sample_data/Scannet/scene0010_01/scene0010_01_scene_graph.json": (317449, "59eea627704d46b3028989c7eef418804f987955695697fa62c474dd53f27817"),
    "sample_data/Unity/loft/loft_object_result.csv": (28751, "bb693e0793204dacb62827b1d37adbf3cdef8803f7869b3c0607fa7a2c202aa3"),
    "sample_data/Unity/loft/loft_referential_statements.json": (5877139, "7c567d494684ca2c94e2f581e2f8136ac97995ad86a76af8cab9e7f232b2f10c"),
    "sample_data/Unity/loft/loft_region_result.csv": (441, "4ca8d79fbb7d590643dd2d2a26ca8cd65b069d17f75cd4f07cbfedcaaf944dca"),
    "sample_data/Unity/loft/loft_scene_graph.json": (486944, "2fd34ece6d33a3c2638d28001d0c84fecd6af629ff025a912915e322f1d62085"),
}

# release folder name -> (source ID, group prefix, scene-name pattern, grouping rule, what the rule rests on)
SOURCES = {
    "Scannet": ("iref_scannet", "scannet", r"scene(\d{4})_(\d{2})", "scan_series", "rescans of one space share sceneNNNN"),
    "3RScan": ("iref_3rscan", "3rscan", r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "metadata",
               "a reference scan and its rescans need 3RScan's metadata"),
    "ARKitScenes": ("iref_arkitscenes", "arkitscenes", r"\d{8}", "metadata", "videos of one venue need ARKitScenes' visit metadata"),
    "Matterport": ("iref_matterport", "matterport", r"[A-Za-z0-9]{11}", "scene", "one building; its regions stay together"),
    "HM3D": ("iref_hm3d", "hm3d", r"\d{5}-[A-Za-z0-9]{11}", "scene", "one building; its regions stay together"),
    "Unity": ("iref_unity", "unity", r"[A-Za-z0-9_]+", "scene", "one synthetic scene; its regions stay together"),
}
# the release's own distribution (data/download_dataset.py at the pinned commit): anonymous, one zip per source
DOWNLOAD_ENDPOINT = "https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/AUTH_ac8533a83cff4d48bc8c608ad222d330"
BUCKET = "iref-vla"
SOURCE_ZIPS = {"Matterport": "Matterport.zip", "Scannet": "Scannet.zip", "HM3D": "HM3D.zip", "Unity": "Unity.zip",
               "ARKitScenes": "ARKitScenes.zip", "3RScan": "3RScan.zip"}
# official split lists that a source publishes outside the release, pinned (repository, commit, size, SHA-256)
OFFICIAL_SPLITS = {
    "Scannet": {"repository": "https://github.com/ScanNet/ScanNet", "commit": "280d42f3c585a69fa52122c5706683461e1f8b50",
                "path": "Tasks/Benchmark", "files": {
                    "scannetv2_train.txt": (15613, "96acca299b7855f02824c496b19077904d80996e7ced1bb9f0dac98f7dd4d0c8"),
                    "scannetv2_val.txt": (4056, "d75d4971c3fa7128c643695840e279042c212ef904fe933bd00cf9918c61b083"),
                    "scannetv2_test.txt": (1300, "0214c6a3b1ee516ad653393b0321e7c0394c7662a4b3702eac1ddd7fbc00f7e0")}},
}
LEGACY_GROUPS = frozenset({"scannet:scene0010"})
NO_REGION = "-1"   # Unity marks objects outside every region with region -1; they belong to no region's selection
FINGERPRINT_MIN_OBJECTS = 3
BOX = ("object_bbox_cx", "object_bbox_cy", "object_bbox_cz", "object_bbox_xlength", "object_bbox_ylength", "object_bbox_zlength")
POLICY = {"policy_id": "a26.partition.v1", "salt": "second-eyes/a26/partition/v1",
          "shares": [["training", 70], ["development", 10], ["calibration", 10], ["test", 10]],
          "legacy_partition": "legacy-development", "unverified_partition": "unassigned"}


class ReleaseError(ValueError):
    pass


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# --------------------------------------------------------------------------------------------- the pinned samples
def _pinned_files():
    yield VOCABULARY["path"], VOCABULARY["bytes"], VOCABULARY["sha256"]
    for p, (n, h) in sorted(SAMPLE_PINS.items()):
        yield p, n, h


def verify_samples(root) -> list:
    """Problems with a local copy of the pinned sample files (relative paths as in the repository); [] when exact."""
    root, bad = Path(root), []
    for p, n, h in _pinned_files():
        f = root / p
        if not f.is_file():
            bad.append(f"{p}: missing")
            continue
        b = f.read_bytes()
        if len(b) != n or _sha(b) != h:
            bad.append(f"{p}: {len(b)} bytes, SHA-256 {_sha(b)[:12]}..., expected {n} bytes, {h[:12]}...")
    return bad


def fetch_samples(out, *, opener=urllib.request.urlopen, progress=print) -> dict:
    """Downloads the pinned sample files at the pinned commit, verifying each before it is kept."""
    out = Path(out)
    got, kept = 0, 0
    for p, n, h in _pinned_files():
        f = out / p
        if f.is_file() and f.stat().st_size == n and _sha(f.read_bytes()) == h:
            kept += 1
            continue
        progress(f"fetching {p}")
        with opener(URL_PREFIX + p) as r:
            b = r.read()
        if len(b) != n or _sha(b) != h:
            raise ReleaseError(f"{p}: the download is {len(b)} bytes, SHA-256 {_sha(b)}; the pin is {n} bytes, {h}")
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_name(f.name + ".partial")
        tmp.write_bytes(b)
        tmp.replace(f)
        got += 1
    return {"fetched": got, "already_present": kept, "root": str(out)}


# --------------------------------------------------------------------------------------------- official lists
def _official_url(source, name):
    o = OFFICIAL_SPLITS[source]
    return o["repository"].replace("https://github.com/", "https://raw.githubusercontent.com/") + f"/{o['commit']}/{o['path']}/{name}"


def fetch_official_splits(out, *, opener=urllib.request.urlopen, progress=print) -> dict:
    """Downloads each source's pinned official split lists into out/<source>/, verified before they are kept."""
    out, got = Path(out), []
    for source, o in OFFICIAL_SPLITS.items():
        for name, (n, h) in o["files"].items():
            f = out / source / name
            if f.is_file() and f.stat().st_size == n and _sha(f.read_bytes()) == h:
                continue
            progress(f"fetching {source}/{name}")
            with opener(_official_url(source, name)) as r:
                b = r.read()
            if len(b) != n or _sha(b) != h:
                raise ReleaseError(f"{source}/{name}: the download is {len(b)} bytes, SHA-256 {_sha(b)}; the pin is {n} bytes, {h}")
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_name(f.name + ".partial")
            tmp.write_bytes(b)
            tmp.replace(f)
            got.append(f"{source}/{name}")
    return {"fetched": got, "root": str(out)}


def load_official(folder, sources) -> dict:
    """The pinned official lists of the given sources, from folder/<source>/, each verified against its pin."""
    lists = {}
    for source in sources:
        if source not in OFFICIAL_SPLITS:
            continue
        for name, (n, h) in OFFICIAL_SPLITS[source]["files"].items():
            f = Path(folder) / source / name
            if not f.is_file():
                raise ReleaseError(f"{f} is missing; run fetch-official-splits")
            b = f.read_bytes()
            if len(b) != n or _sha(b) != h:
                raise ReleaseError(f"{f} is not the pinned {source} list ({len(b)} bytes, SHA-256 {_sha(b)[:12]}...)")
            lists[f"{source}/{name}"] = _lists(name, b)
    return lists


# --------------------------------------------------------------------------------------------- access
def zip_url(source) -> str:
    return f"{DOWNLOAD_ENDPOINT}/{BUCKET}/{SOURCE_ZIPS[source]}"


def probe(*, opener=urllib.request.urlopen) -> dict:
    """Each source zip's size, ETag and modification date, by a header-only request: nothing is downloaded."""
    out = {}
    for source in SOURCE_ZIPS:
        try:
            with opener(urllib.request.Request(zip_url(source), method="HEAD"), timeout=60) as r:
                h = r.headers
                out[source] = {"bytes": int(h.get("Content-Length")) if h.get("Content-Length") else None,
                               "etag": (h.get("ETag") or "").strip('"') or None, "last_modified": h.get("Last-Modified")}
        except Exception as e:   # recorded per source: an unreachable host is a finding, not a crash
            out[source] = {"error": f"{type(e).__name__}: {e}"}
    return {"format_version": 1, "record_type": "a26_release_probe", "endpoint": DOWNLOAD_ENDPOINT, "bucket": BUCKET,
            "release_commit": COMMIT, "sources": out}


def download(source, out_dir, *, opener=urllib.request.urlopen, chunk=1 << 20, progress=print) -> dict:
    """One source zip, streamed under a temporary name and hashed as it arrives; renamed only when complete, with a
    provenance receipt beside it. An existing complete copy is never overwritten."""
    if source not in SOURCE_ZIPS:
        raise ReleaseError(f"{source!r} is not a release source; one of {', '.join(SOURCE_ZIPS)}")
    out_dir = Path(out_dir)
    final = out_dir / SOURCE_ZIPS[source]
    if final.exists():
        raise ReleaseError(f"{final} exists; downloads never overwrite a kept copy")
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(final.name + ".partial")
    h, n = hashlib.sha256(), 0
    try:
        with opener(zip_url(source), timeout=120) as r, open(tmp, "wb") as f:
            expected = r.headers.get("Content-Length")
            etag, modified = (r.headers.get("ETag") or "").strip('"') or None, r.headers.get("Last-Modified")
            while True:
                b = r.read(chunk)
                if not b:
                    break
                f.write(b)
                h.update(b)
                n += len(b)
                if n % (256 * chunk) < len(b):
                    progress(f"  {n / 1e9:.2f} GB")
        if expected is not None and int(expected) != n:
            raise ReleaseError(f"{source}: received {n} bytes of {expected}")
        tmp.replace(final)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    import datetime
    receipt = {"format_version": 1, "record_type": "a26_release_download", "source": source, "url": zip_url(source),
               "bytes": n, "sha256": h.hexdigest(), "etag": etag, "last_modified": modified, "release_commit": COMMIT,
               "downloaded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (out_dir / (final.name + ".receipt.json")).write_text(_dumps(receipt), encoding="utf-8")
    return receipt


# --------------------------------------------------------------------------------------------- reading a release
def _scene_member(name):
    """(scene, kind) for a release member path .../<scene>/<scene>_<kind>, else None."""
    parts = name.replace("\\", "/").split("/")
    if len(parts) < 2:
        return None
    scene, base = parts[-2], parts[-1]
    for k in KINDS:
        if base == f"{scene}_{k}":
            return scene, k
    return None


def _lists(name, data):
    return [line.strip() for line in data.decode("utf-8").splitlines() if line.strip()]




def read_source(path, source, only=None):
    """(scenes, official lists, other text files) of one source. scenes maps a scene name to {kind: bytes}. A .txt file
    counts as an official list only outside the scenes' own folders (the scenes found in this source), and only when
    every line is a scene name of this source; other text files are named, not read further. A folder is walked; a zip
    is read member by member, the four kinds and .txt files only. only, a set of scene names, restricts reading to those
    scenes: every other scene's files are never read."""
    if source not in SOURCES:
        raise ReleaseError(f"{source!r} is not a release source; one of {', '.join(SOURCES)}")
    path = Path(path)
    scenes, texts = {}, []
    if path.is_file() and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                hit = _scene_member(info.filename)
                if hit and only is not None and hit[0] not in only:
                    continue
                if hit:
                    scenes.setdefault(hit[0], {})[hit[1]] = z.read(info)
                elif info.filename.lower().endswith(".txt"):
                    texts.append((info.filename.replace("\\", "/"), _lists(info.filename, z.read(info))))
    elif path.is_dir():
        for f in sorted(path.rglob("*")):
            if not f.is_file():
                continue
            hit = _scene_member(f.relative_to(path).as_posix())
            if hit and only is not None and hit[0] not in only:
                continue
            if hit:
                scenes.setdefault(hit[0], {})[hit[1]] = f.read_bytes()
            elif f.suffix.lower() == ".txt":
                texts.append((f.relative_to(path).as_posix(), _lists(f.name, f.read_bytes())))
    else:
        raise ReleaseError(f"{path} is neither a folder nor a zip")
    lists, other = {}, []
    for rel, lines in texts:   # classified after the scan, against the scenes actually found
        parts = rel.split("/")
        in_scene = len(parts) >= 2 and parts[-2] in scenes
        if not in_scene and lines and all(re.fullmatch(SOURCES[source][2], x) for x in lines):
            lists[parts[-1]] = lines
        else:
            other.append(rel)
    return scenes, lists, sorted(other)


def read_vocabulary(data: bytes) -> set:
    iss = Issues()
    rows = read_csv(VOCABULARY["name"], data, V1C.VOCABULARY_HEADER, {len(V1C.VOCABULARY_HEADER)}, iss)
    if rows is None:
        raise ReleaseError("the vocabulary does not read: " + "; ".join(i["message"] for i in iss.items[:3]))
    return {r[1] for r in rows}


# --------------------------------------------------------------------------------------------- one scene
def _finite(text):
    try:
        v = float(text)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def inspect_scene(source, scene, files, vocabulary, keep_entries=False) -> dict:
    """One scene's record: retained or rejected with its reasons, its counts, and its statement-level exclusions.
    keep_entries adds each kept entry as (region, text, target index), for regression checks; never written out."""
    sid, prefix, pattern, rule, basis = SOURCES[source]
    rec = {"source": source, "scene": scene, "retained": False, "rejections": [], "entry_exclusions": {}}

    def reject(code, message):
        rec["rejections"].append({"code": code, "message": message})
        return rec

    if not re.fullmatch(pattern, scene):
        return reject("E_RELEASE_SCENE_NAME", f"{scene!r} is not a {source} scene name")
    missing = [k for k in KINDS if k not in files]
    if missing:
        return reject("E_RELEASE_FILES", "missing " + ", ".join(missing))
    iss = Issues()
    orows = read_csv(f"{scene}_object_result.csv", files["object_result.csv"], V1C.OBJECT_HEADER, V1C.OBJECT_WIDTHS, iss)
    rrows = read_csv(f"{scene}_region_result.csv", files["region_result.csv"], V1C.REGION_HEADER, {len(V1C.REGION_HEADER)}, iss)
    graph = read_json(f"{scene}_scene_graph.json", files["scene_graph.json"], iss)
    stmts = read_json(f"{scene}_referential_statements.json", files["referential_statements.json"], iss)
    if iss.items:
        for i in iss.items[:5]:
            reject("E_RELEASE_SOURCE_SHAPE", f"{i['file']} {i['path']}: {i['message']}")
        return rec
    regions = [r[0] for r in rrows]
    if len(set(regions)) != len(regions):
        return reject("E_RELEASE_REGIONS", "a region ID is repeated")
    objects = {}
    for n, row in enumerate(orows, start=2):
        d = dict(zip(V1C.OBJECT_HEADER, row))
        oid = d["object_id"]
        if not re.fullmatch(r"\d+", oid):
            return reject("E_RELEASE_OBJECTS", f"row {n}: object ID {oid!r} is not decimal")
        if oid in objects:
            return reject("E_RELEASE_OBJECTS", f"row {n}: object ID {oid} is repeated")
        if d["region_id"] not in regions and d["region_id"] != NO_REGION:
            return reject("E_RELEASE_OBJECTS", f"row {n}: region {d['region_id']!r} is not in the region file")
        box = [_finite(d[k]) for k in BOX]
        if any(v is None for v in box) or any(v < 0 for v in box[3:]):
            return reject("E_RELEASE_GEOMETRY", f"row {n}: the box is not finite with nonnegative lengths")
        objects[oid] = {"region": d["region_id"], "label": d["nyu_label"], "box": box}
    for name, doc in (("scene graph", graph), ("statements", stmts)):
        if not isinstance(doc, dict) or doc.get("scene_name") != scene or "regions" not in doc:
            return reject("E_RELEASE_SCENE_NAME", f"the {name} does not name {scene!r} with its regions")
    if not isinstance(stmts["regions"], dict) or not set(stmts["regions"]) <= set(regions):
        return reject("E_RELEASE_STATEMENTS", "the statements' regions are not the region file's")

    excluded = Counter()
    texts = entries = multi = 0
    fit = Counter()
    kept_entries = []
    rel_types, relations, anchors_hist = Counter(), Counter(), Counter()
    distractors_hist, attributes, false_texts = Counter(), Counter(), 0
    per_region = {}
    projected = {}
    for rid in regions:
        projected[rid] = tuple(SEL.ProjectedObject(o, None if v["label"] in ("", V1.UNKNOWN_SENTINEL) else v["label"])
                               for o, v in sorted(objects.items()) if v["region"] == rid)
    for rid, block in sorted(stmts["regions"].items()):
        if not isinstance(block, dict):
            excluded["E_RELEASE_STATEMENT_SHAPE"] += 1
            continue
        kept_texts = 0
        for text, lst in block.items():
            if not text or not isinstance(lst, list) or not lst:
                excluded["E_RELEASE_STATEMENT_SHAPE"] += 1
                continue
            good = []
            for e in lst:
                code = None
                if not isinstance(e, dict) or set(e) != V1C.RECORD_KEYS:
                    code = "E_RELEASE_STATEMENT_SHAPE"
                else:
                    t = e.get("target_index")
                    anchors = e.get("anchors") if isinstance(e.get("anchors"), dict) else None
                    if t not in objects or objects[t]["region"] != rid:
                        code = "E_RELEASE_TARGET"
                    elif e.get("target_class") != objects[t]["label"]:
                        code = "E_RELEASE_TARGET_CLASS"
                    elif not isinstance(e.get("distractor_ids"), list) or any(x not in objects for x in e["distractor_ids"]):
                        code = "E_RELEASE_DISTRACTOR"
                    elif anchors is None or any(not isinstance(a, dict) or a.get("index") not in objects for a in anchors.values()):
                        code = "E_RELEASE_ANCHOR"
                    elif any(objects[a["index"]]["region"] != rid for a in anchors.values()):
                        code = "E_RELEASE_ANCHOR_REGION"   # outside the region, so outside its selection
                if code:
                    excluded[code] += 1
                else:
                    good.append(e)
            if not good:
                continue
            texts += 1
            kept_texts += 1
            entries += len(good)
            if keep_entries:
                kept_entries += [(rid, text, g["target_index"]) for g in good]
            multi += len(good) > 1
            e = good[0]
            rel_types[str(e.get("relation_type"))] += 1
            relations[str(e.get("relation"))] += 1
            anchors_hist[min(len(e["anchors"]), 2)] += 1
            distractors_hist[min(len(e["distractor_ids"]), 3)] += 1
            attributes["color"] += bool(e.get("target_color_used"))
            attributes["size"] += bool(e.get("target_size_used"))
            false_texts += bool(e.get("false_statements"))
            cats = frozenset([objects[e["target_index"]]["label"]] + [objects[a["index"]]["label"] for a in e["anchors"].values()])
            n = len(SEL.select(projected[rid], cats))
            fit["too_few" if n < 3 else ("fits" if n <= 10 else "over_budget")] += 1
        per_region[rid] = {"objects": len(projected[rid]), "texts": kept_texts}
    labels = Counter(v["label"] for v in objects.values())
    rec.update({
        "retained": True, "source_id": sid, "objects": len(objects), "regions": len(regions), "per_region": per_region,
        "texts": texts, "entries": entries, "texts_with_several_entries": multi,
        "objects_without_region": sum(1 for v in objects.values() if v["region"] == NO_REGION),
        "entry_exclusions": dict(sorted(excluded.items())),
        "compatibility_estimate": {k: fit.get(k, 0) for k in ("too_few", "fits", "over_budget")},
        "relation_types": dict(sorted(rel_types.items())), "relations": dict(sorted(relations.items())),
        "anchors": {str(k): anchors_hist[k] for k in sorted(anchors_hist)},
        "distractors": {("3+" if k == 3 else str(k)): distractors_hist[k] for k in sorted(distractors_hist)},
        "attributes_used": {"color": attributes["color"], "size": attributes["size"]},
        "texts_with_false_statements": false_texts,
        "labels_outside_vocabulary": sorted(l for l in labels if l not in vocabulary and l not in ("", V1.UNKNOWN_SENTINEL)),
        "frame_annotations": 0,
        "fingerprint": _sha(json.dumps(sorted([v["label"]] + [round(x, 2) for x in v["box"][3:]] for v in objects.values()),
                                       separators=(",", ":")).encode("utf-8")) if len(objects) >= FINGERPRINT_MIN_OBJECTS else None,
        "input_sha256": {k: _sha(files[k]) for k in KINDS},
    })
    if keep_entries:
        rec["entry_list"] = kept_entries
    return rec


def group_of(source, scene, ancestry=None):
    """(group ID, how it is known): by name, from supplied metadata, or unverified."""
    sid, prefix, pattern, rule, basis = SOURCES[source]
    known = (ancestry or {}).get(source, {})
    if scene in known:
        return f"{prefix}:{known[scene]}", "metadata"
    if rule == "scan_series":
        return f"{prefix}:scene{re.fullmatch(pattern, scene).group(1)}", "name"
    if rule == "scene":
        return f"{prefix}:{scene}", "name"
    return f"{prefix}:{scene}", "unverified"


# --------------------------------------------------------------------------------------------- the inventory
def build_inventory(sources, vocabulary, ancestry=None) -> dict:
    """sources: [(source name, path)]. The scene records, their groups (identical fingerprints merged), the official
    lists, and the counts by source."""
    scenes, lists, others = [], {}, []
    for name, path in sources:
        found, ls, other = read_source(path, name)
        for k, v in ls.items():
            lists[f"{name}/{k}"] = v
        others += [f"{name}/{x}" for x in other]
        for scene in sorted(found):
            rec = inspect_scene(name, scene, found[scene], vocabulary)
            if rec["retained"]:
                rec["group"], rec["ancestry"] = group_of(name, scene, ancestry)
            scenes.append(rec)
    kept = [r for r in scenes if r["retained"]]
    by_print = {}
    for r in kept:
        if r["fingerprint"]:
            by_print.setdefault(r["fingerprint"], []).append(r)
    merged = []
    for fp, rs in by_print.items():
        gs = sorted({r["group"] for r in rs})
        if len(gs) > 1:
            for r in rs:
                r["group"], r["merged_by_fingerprint"] = gs[0], gs
            merged.append({"fingerprint": fp, "groups": gs, "scenes": sorted(f"{r['source']}/{r['scene']}" for r in rs)})
    groups = {}
    for r in kept:
        g = groups.setdefault(r["group"], {"scenes": [], "ancestry": set(), "legacy": r["group"] in LEGACY_GROUPS})
        g["scenes"].append(f"{r['source']}/{r['scene']}")
        g["ancestry"].add(r["ancestry"])
    for g in groups.values():
        g["ancestry"] = "unverified" if "unverified" in g["ancestry"] else ("metadata" if "metadata" in g["ancestry"] else "name")
    by_source = {}
    for r in scenes:
        s = by_source.setdefault(r["source"], {"scenes_seen": 0, "retained": 0, "rejected": 0, "rejections": Counter(),
                                                "texts": 0, "entries": 0, "entry_exclusions": Counter(),
                                                "compatibility_estimate": Counter()})
        s["scenes_seen"] += 1
        if r["retained"]:
            s["retained"] += 1
            s["texts"] += r["texts"]
            s["entries"] += r["entries"]
            s["entry_exclusions"].update(r["entry_exclusions"])
            s["compatibility_estimate"].update(r["compatibility_estimate"])
        else:
            s["rejected"] += 1
            s["rejections"].update(x["code"] for x in r["rejections"])
    for s in by_source.values():
        for k in ("rejections", "entry_exclusions", "compatibility_estimate"):
            s[k] = dict(sorted(s[k].items()))
        s["groups"] = 0
    for g, v in groups.items():
        src = v["scenes"][0].split("/")[0]
        by_source[src]["groups"] += 1
    return {"format_version": 1, "record_type": "a26_release_inventory", "release_version": RELEASE_VERSION,
            "commit": COMMIT, "sources": by_source, "groups": {k: groups[k] for k in sorted(groups)},
            "merged_by_fingerprint": merged, "official_lists": {k: lists[k] for k in sorted(lists)},
            "other_text_files": others, "scenes": scenes}


def _share(salt, group, shares):
    x = int(hashlib.sha256(f"{salt}\n{group}".encode("utf-8")).hexdigest()[:16], 16) / float(1 << 64)
    total, acc = sum(w for _, w in shares), 0.0
    for name, w in shares:
        acc += w / total
        if x < acc:
            return name
    return shares[-1][0]


def propose_partitions(inventory, policy=POLICY, official=None) -> dict:
    """A proposal, never frozen: each group's partition and the counts per partition. Official lists are preserved: a
    group in a list named for validation or test never goes to training, and one in a training list goes to training.
    official adds lists published outside the release (load_official), with their coverage of the retained scenes."""
    lists = dict(inventory["official_lists"])
    lists.update(official or {})
    held = {s for k, v in lists.items() if re.search(r"val|test", k.split("/")[-1].lower()) for s in v}
    train = {s for k, v in lists.items() if re.search(r"train", k.split("/")[-1].lower()) for s in v}
    coverage = {}
    for k, v in lists.items():
        src = k.split("/")[0]
        c = coverage.setdefault(src, {"lists": [], "listed": set()})
        c["lists"].append(k)
        c["listed"].update(v)
    for src, c in coverage.items():
        kept = {r["scene"] for r in inventory["scenes"] if r["source"] == src and r["retained"]}
        not_listed, absent = sorted(kept - c["listed"]), sorted(c["listed"] - kept)
        coverage[src] = {"lists": sorted(c["lists"]), "retained": len(kept), "retained_not_listed": len(not_listed),
                         "retained_not_listed_examples": not_listed[:5], "listed_not_retained": len(absent),
                         "listed_not_retained_examples": absent[:5]}
    shares = policy["shares"]
    heldout_shares = [s for s in shares if s[0] != "training"]
    texts = {}
    for r in inventory["scenes"]:
        if r["retained"]:
            texts.setdefault(r["group"], [0, 0])
            texts[r["group"]][0] += r["texts"]
            texts[r["group"]][1] += r["compatibility_estimate"]["fits"]
    assignment, conflicts = {}, []
    for g, v in inventory["groups"].items():
        names = {s.split("/", 1)[1] for s in v["scenes"]}
        if v["legacy"]:
            part, why = policy["legacy_partition"], "the inspected legacy scene"
        elif v["ancestry"] == "unverified":
            part, why = policy["unverified_partition"], "ancestry unverified"
        elif names & held and names & train:
            part, why = policy["unverified_partition"], "listed as both training and held-out"
            conflicts.append(g)
        elif names & held:
            part, why = _share(policy["salt"], g, heldout_shares), "an official held-out list"
        elif names & train:
            part, why = "training", "an official training list"
        else:
            part, why = _share(policy["salt"], g, shares), "group hash"
        assignment[g] = {"partition": part, "basis": why}
    counts = {}
    for g, a in assignment.items():
        c = counts.setdefault(a["partition"], {"groups": 0, "scenes": 0, "parent_texts": 0, "fits_estimate": 0})
        c["groups"] += 1
        c["scenes"] += len(inventory["groups"][g]["scenes"])
        c["parent_texts"] += texts.get(g, [0, 0])[0]
        c["fits_estimate"] += texts.get(g, [0, 0])[1]
    return {"format_version": 1, "record_type": "a26_partition_proposal", "status": "proposed, not frozen",
            "policy": policy, "policy_sha256": _sha(json.dumps(policy, sort_keys=True).encode("utf-8")),
            "counts": {k: counts[k] for k in sorted(counts)}, "conflicts": conflicts, "official_coverage": coverage,
            "official_pins": {s: OFFICIAL_SPLITS[s] for s in coverage if s in OFFICIAL_SPLITS and official},
            "assignment": {k: assignment[k] for k in sorted(assignment)}}


# --------------------------------------------------------------------------------------------- writing
def _write(out, files: dict) -> Path:
    out = Path(out)
    if out.exists():
        raise ReleaseError(f"{out} exists; each run writes a new folder")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=str(out.parent)))
    try:
        for name, text in files.items():
            (tmp / name).write_text(text, encoding="utf-8")
        tmp.rename(out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return out


def _dumps(doc) -> str:
    return json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def write_inventory(inventory, out) -> Path:
    lines = [f"# IRef-VLA release inventory ({RELEASE_VERSION}, commit `{COMMIT[:7]}`)", "",
             "| Source | Scenes seen | Retained | Rejected | Groups | Parent texts | Entries | Fits 3-10 (estimate) |",
             "|---|---|---|---|---|---|---|---|"]
    for s, v in sorted(inventory["sources"].items()):
        lines.append(f"| {s} | {v['scenes_seen']} | {v['retained']} | {v['rejected']} | {v['groups']} | {v['texts']} | "
                     f"{v['entries']} | {v['compatibility_estimate'].get('fits', 0)} |")
    lines += ["", "Rejections and statement exclusions, by source:", ""]
    for s, v in sorted(inventory["sources"].items()):
        lines.append(f"- {s}: rejected {v['rejections'] or 'none'}; excluded statements {v['entry_exclusions'] or 'none'}")
    unv = sum(1 for g in inventory["groups"].values() if g["ancestry"] == "unverified")
    lines += ["", f"Groups: {len(inventory['groups'])}, of which {unv} have unverified ancestry; merged by fingerprint: "
              f"{len(inventory['merged_by_fingerprint'])}. Official lists found: {len(inventory['official_lists'])}.",
              "", "The compatibility figures are an estimate: A2.2c's selection over the statements' annotated categories, "
              "not the accepted parser's. No statement carries a frame or viewpoint (O27)."]
    return _write(out, {"inventory.json": _dumps(inventory), "report.md": "\n".join(lines) + "\n"})


def write_proposal(proposal, out) -> Path:
    lines = ["# Partition proposal (not frozen)", "", f"Policy `{proposal['policy']['policy_id']}`, salt "
             f"`{proposal['policy']['salt']}`, policy SHA-256 `{proposal['policy_sha256'][:12]}...`.", "",
             "| Partition | Groups | Scenes | Parent texts | Fits 3-10 (estimate) |", "|---|---|---|---|---|"]
    for k, c in proposal["counts"].items():
        lines.append(f"| {k} | {c['groups']} | {c['scenes']} | {c['parent_texts']} | {c['fits_estimate']} |")
    lines += ["", f"Conflicts between official lists: {proposal['conflicts'] or 'none'}."]
    for s, c in sorted(proposal.get("official_coverage", {}).items()):
        lines.append(f"Official lists for {s}: {', '.join(c['lists'])}; retained scenes in no list: {c['retained_not_listed']}; "
                     f"listed scenes not retained: {c['listed_not_retained']}.")
    return _write(out, {"proposal.json": _dumps(proposal), "report.md": "\n".join(lines) + "\n"})


# --------------------------------------------------------------------------------------------- conversion (A2.6b)
ROLE_KINDS = {"objects": "object_result.csv", "regions": "region_result.csv", "graph": "scene_graph.json",
              "statements": "referential_statements.json"}


def release_identity(source, scene, files, partition) -> object:
    """The conversion identity of one release scene: A2.2a's converter and validation, renamed for the scene. A2.6
    converts ScanNet scenes of the development partition and the legacy group only; closed partitions are never read."""
    if source != "Scannet" or not re.fullmatch(SOURCES[source][2], scene):
        raise ReleaseError(f"A2.6 converts ScanNet release scenes only, not {source}/{scene}")
    if partition not in ("development", "legacy-development"):
        raise ReleaseError(f"{scene} is in {partition!r}; only development and legacy scenes are converted")
    P = V1C.PINNED_IDENTITY
    fl = {role: {"name": f"{scene}_{kind}", "path": f"{SOURCE_ZIPS[source]}:{source}/{scene}/{scene}_{kind}",
                 "bytes": len(files[kind]), "sha256": _sha(files[kind])} for role, kind in ROLE_KINDS.items()}
    fl["vocabulary"] = dict(P.FILES["vocabulary"])
    values = dict(vars(P))
    values.update(
        FILES=fl, URL_PREFIX=zip_url(source) + "#", SCENE_NAME=scene, REGION_ID="0", SOURCE_ID="iref_scannet",
        SCENE_ID=f"iref.scannet.{scene}.full", FRAME_ID=f"iref.scannet.{scene}.native",
        MAP_ID=f"iref.scannet.nyu.{scene}.v1", COMMAND_PREFIX=f"iref.scannet.{scene}.r0.e.",
        PINNED={"repository": P.REPOSITORY, "commit": P.COMMIT, "url_prefix": zip_url(source) + "#", "files": fl},
        ORIGIN=f"the published IRef-VLA aligned-scene origin of ScanNet {scene} (ScanNet's axis alignment was applied "
               f"upstream; not recentred)",
        SOURCE_DESCRIPTION=f"IRef-VLA's ScanNet release scene {scene} ({SOURCE_ZIPS[source]}): published annotations of "
                           f"a ScanNet scan, used under the project lead's ScanNet Terms of Use. Development material "
                           f"({partition}), not held-out evaluation data",
        MAP_DESCRIPTION=f"IRef-VLA ScanNet release scene {scene}: raw source labels to NYU labels, used as both the "
                        f"standardized and the model-visible label. The model vocabulary is the whole pinned NYU "
                        f"vocabulary except its 'unknown' sentinel",
        EXPRESSION_DESCRIPTION=f"an expression of IRef-VLA's ScanNet release scene {scene}, unchanged; the source gives "
                               f"no user pose and no action",
        RELEASE_VERSION=RELEASE_VERSION)
    return V1C.Identity(**values)


def import_release_scene(source, scene, files, vocabulary_bytes, out, *, partition, expected_sha256) -> dict:
    """A2.2a's import of one release scene into a new folder `out`, under its release identity, with A2.2a's whole
    validation. Its input files must equal the hashes the inventory recorded (expected_sha256: kind -> SHA-256)."""
    if partition not in ("development", "legacy-development"):   # closed partitions are refused before any reading
        raise ReleaseError(f"{scene} is in {partition!r}; only development and legacy scenes are converted")
    got = {k: _sha(files[k]) for k in KINDS}
    if got != expected_sha256:
        raise ReleaseError(f"{scene}: its files differ from the inventory's recorded hashes")
    ident = release_identity(source, scene, files, partition)
    tmp = Path(tempfile.mkdtemp(prefix=f".{scene}-sources-"))
    try:
        paths = {}
        for role, kind in ROLE_KINDS.items():
            paths[role] = tmp / ident.FILES[role]["name"]
            paths[role].write_bytes(files[kind])
        paths["vocabulary"] = tmp / VOCABULARY["name"]
        paths["vocabulary"].write_bytes(vocabulary_bytes)
        return V1O.run_import(paths["objects"], paths["regions"], paths["vocabulary"], out, statements=paths["statements"],
                              graph=paths["graph"], pins=ident.PINNED, identity=ident)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------------------------- the summary
# A2.3d's measured medians on the RTX PC (float32, eager), per request, in ms, for coordinates and augmented in the full
# inventory, then the same in the source-known view (A2.3d_compare.md, section 10). A parent command costs four requests.
COSTS_MS = {"0.5B": (89.0, 143.2, 70.9, 92.4), "7B": (719.9, 1031.8, 574.6, 750.5)}


def _spread(values):
    v = sorted(values)
    if not v:
        return None
    n = len(v)
    return [v[0], v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2, v[-1]]


def summarize(inventory, proposal=None, caps=(16, 32, 64)) -> dict:
    """The coverage and partition summary, exact over the inventory's retained scenes; with a proposal, the development
    caps: per cap k, every development group contributes min(k, its fitting parent commands)."""
    kept = [r for r in inventory["scenes"] if r["retained"]]

    def total(key):
        c = Counter()
        for r in kept:
            c.update(r[key])
        return c
    s = {"format_version": 1, "record_type": "a26_release_summary", "release_version": RELEASE_VERSION,
         "sources": {k: {f: v[f] for f in ("scenes_seen", "retained", "rejected", "groups", "texts", "entries",
                                             "rejections", "entry_exclusions")} for k, v in inventory["sources"].items()},
         "ancestry": dict(Counter(g["ancestry"] for g in inventory["groups"].values())),
         "merged_by_fingerprint": len(inventory["merged_by_fingerprint"]),
         "objects_per_scene": _spread([r["objects"] for r in kept]),
         "objects_per_region": _spread([v["objects"] for r in kept for v in r["per_region"].values()]),
         "relation_types": dict(total("relation_types")), "relations": dict(total("relations").most_common()),
         "anchors": dict(total("anchors")), "distractors": dict(total("distractors")),
         "attributes_used": dict(total("attributes_used")),
         "texts_with_false_statements": sum(r["texts_with_false_statements"] for r in kept),
         "texts_with_several_entries": sum(r["texts_with_several_entries"] for r in kept),
         "frame_annotations": sum(r["frame_annotations"] for r in kept),
         "labels_outside_vocabulary": sorted({x for r in kept for x in r["labels_outside_vocabulary"]}),
         "compatibility_estimate": dict(total("compatibility_estimate"))}
    if proposal is not None:
        fits = Counter()
        for r in kept:
            fits[r["group"]] += r["compatibility_estimate"]["fits"]
        dev = sorted(g for g, a in proposal["assignment"].items() if a["partition"] == "development")
        s["partitions"] = proposal["counts"]
        s["official_coverage"] = proposal.get("official_coverage", {})
        s["development_fits_per_group"] = _spread([fits[g] for g in dev])
        s["development_caps"] = []
        for k in caps:
            parents = sum(min(k, fits[g]) for g in dev)
            s["development_caps"].append({"cap_per_group": k, "groups": len(dev), "parent_commands": parents,
                                          "requests_per_model": 4 * parents,
                                          "hours_0.5B": round(parents * sum(COSTS_MS["0.5B"]) / 3.6e6, 2),
                                          "hours_7B": round(parents * sum(COSTS_MS["7B"]) / 3.6e6, 2)})
    return s


def summary_markdown(s) -> str:
    L = ["# Release summary", ""]
    for k, v in sorted(s["sources"].items()):
        L.append(f"- {k}: {v['retained']} of {v['scenes_seen']} scenes retained ({v['rejected']} rejected {v['rejections'] or ''}), "
                 f"{v['groups']} groups, {v['texts']} parent texts, {v['entries']} entries; statement exclusions "
                 f"{v['entry_exclusions'] or 'none'}")
    L += [f"- Groups by ancestry: {s['ancestry']}; merged by fingerprint: {s['merged_by_fingerprint']}",
          f"- Objects per scene (min, median, max): {s['objects_per_scene']}; per region: {s['objects_per_region']}",
          f"- Fit 3 to 10 (estimate): {s['compatibility_estimate']}",
          f"- Relation types: {s['relation_types']}",
          f"- Relations: {s['relations']}",
          f"- Anchors: {s['anchors']}; distractors: {s['distractors']}; attributes used: {s['attributes_used']}",
          f"- Texts with false statements: {s['texts_with_false_statements']}; with several entries: "
          f"{s['texts_with_several_entries']}; frame annotations: {s['frame_annotations']}",
          f"- Labels outside the vocabulary: {len(s['labels_outside_vocabulary'])} {s['labels_outside_vocabulary'][:12]}"]
    if "partitions" in s:
        L += ["", "| Partition | Groups | Scenes | Parent texts | Fit 3-10 |", "|---|---|---|---|---|"]
        L += [f"| {k} | {c['groups']} | {c['scenes']} | {c['parent_texts']} | {c['fits_estimate']} |" for k, c in s["partitions"].items()]
        for src, c in sorted(s["official_coverage"].items()):
            L.append(f"\nOfficial lists ({src}): retained in no list {c['retained_not_listed']}, listed but not retained "
                     f"{c['listed_not_retained']}.")
        L += ["", f"Development fitting parents per group (min, median, max): {s['development_fits_per_group']}", "",
              "| Cap per group | Groups | Parent commands | Requests per model | 0.5B hours | 7B hours |", "|---|---|---|---|---|---|"]
        L += [f"| {c['cap_per_group']} | {c['groups']} | {c['parent_commands']} | {c['requests_per_model']} | {c['hours_0.5B']} | "
              f"{c['hours_7B']} |" for c in s["development_caps"]]
        L += ["", "Hours use A2.3d's median forward times on the RTX PC (four requests per parent: two formats, two views);",
              "p95 is higher, and the new D104 requests may differ."]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------------------------- command line
def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="python -m grounding.adapters.iref_vla.release")
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch-samples", help="download the six pinned sample scenes and the vocabulary, verified")
    f.add_argument("--out", required=True)
    sub.add_parser("probe", help="each source zip's size and identity, by header-only requests (nothing downloaded)")
    d = sub.add_parser("download", help="download one source zip, hashed, with a provenance receipt")
    d.add_argument("--source", required=True, choices=sorted(SOURCE_ZIPS))
    d.add_argument("--out", required=True)
    v = sub.add_parser("verify-samples", help="check a local copy of the pinned sample files")
    v.add_argument("--root", required=True)
    i = sub.add_parser("inventory", help="validate, count and group the scenes of one or more release sources")
    i.add_argument("--source", nargs=2, action="append", metavar=("NAME", "PATH"), required=True,
                   help="a release source name (Scannet, 3RScan, ARKitScenes, Matterport, HM3D, Unity) and its folder or zip")
    i.add_argument("--vocabulary", required=True, help="NYU_Object_Classes.csv at the pinned commit")
    i.add_argument("--ancestry", help="JSON: {source: {scene: environment key}} from a source's own metadata")
    i.add_argument("--out", required=True)
    q = sub.add_parser("partition", help="propose partitions from an inventory (never frozen)")
    q.add_argument("--inventory", required=True)
    q.add_argument("--official", help="a folder of pinned official lists (fetch-official-splits writes it)")
    q.add_argument("--out", required=True)
    o = sub.add_parser("fetch-official-splits", help="download the pinned official split lists published outside the release")
    o.add_argument("--out", required=True)
    m = sub.add_parser("summarize", help="the coverage and partition summary, with development caps and their cost")
    m.add_argument("--inventory", required=True)
    m.add_argument("--proposal")
    m.add_argument("--caps", type=int, nargs="+", default=[16, 32, 64])
    m.add_argument("--out", required=True)
    a = p.parse_args(argv)
    try:
        if a.command == "fetch-samples":
            print(fetch_samples(a.out))
        elif a.command == "probe":
            pr = probe()
            for s, v in pr["sources"].items():
                print(f"  {s:<12} " + (v["error"] if "error" in v else
                      f"{v['bytes'] / 1e9:8.2f} GB  ETag {v['etag']}  modified {v['last_modified']}" if v["bytes"] else str(v)))
            print(_dumps(pr), end="")
        elif a.command == "download":
            r = download(a.source, a.out)
            print(f"{a.source}: {r['bytes'] / 1e9:.2f} GB, SHA-256 {r['sha256']}; receipt beside the zip")
        elif a.command == "verify-samples":
            bad = verify_samples(a.root)
            print("\n".join(bad) if bad else "all pinned sample files verified")
            return 1 if bad else 0
        elif a.command == "inventory":
            vb = Path(a.vocabulary).read_bytes()
            if _sha(vb) != VOCABULARY["sha256"]:
                raise ReleaseError("the vocabulary is not the pinned NYU_Object_Classes.csv")
            anc = json.loads(Path(a.ancestry).read_text(encoding="utf-8")) if a.ancestry else None
            inv = build_inventory([(n, Path(x)) for n, x in a.source], read_vocabulary(vb), anc)
            out = write_inventory(inv, a.out)
            print((out / "report.md").read_text(encoding="utf-8"))
            print(f"written to {out}")
        elif a.command == "fetch-official-splits":
            print(fetch_official_splits(a.out))
        elif a.command == "summarize":
            inv = json.loads((Path(a.inventory) / "inventory.json").read_text(encoding="utf-8"))
            prop = json.loads((Path(a.proposal) / "proposal.json").read_text(encoding="utf-8")) if a.proposal else None
            s = summarize(inv, prop, tuple(a.caps))
            md = summary_markdown(s)
            out = _write(a.out, {"summary.json": _dumps(s), "summary.md": md})
            print(md)
            print(f"written to {out}")
        else:
            inv = json.loads((Path(a.inventory) / "inventory.json").read_text(encoding="utf-8"))
            off = load_official(a.official, sorted(inv["sources"])) if a.official else None
            out = write_proposal(propose_partitions(inv, official=off), a.out)
            print((out / "report.md").read_text(encoding="utf-8"))
            print(f"written to {out}")
    except ReleaseError as e:
        print(f"refused: {e}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
