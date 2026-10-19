"""A4 Delivery 1, stage 3: the perception folder, built from the repository and pushed to the headset over adb.

The headset's scan and the session's scanned rooms need files the accepted parser folder (A2.7, D115) doesn't hold: the
schemas of the capture record, scene v3, format-2 queries, results and commands, the size priors and the two
configurations; the scan's and the localization's configurations; the bedroom's category map and size priors. They
travel as their own folder, verified on the headset like the parser folder: file by file against manifest.json, whose own
SHA-256 identifies the folder in every capture record (docs/scan.md).

`build` checks every file first (the schemas are JSON Schemas; the configurations and the size priors pass their schemas;
the category map passes the contract), then copies each one into a new folder with Git's line endings (LF), so a checkout
with CRLF line endings builds the very same folder, and writes manifest.json last.
`push` verifies a built folder, removes the headset's old manifest so the app never loads a half-pushed folder, pushes
every file with its size checked on the headset, and the manifest last; a receipt with every file's SHA-256 is written
beside the folder. Neither installs, launches or reruns the app.

    python -m grounding.quest.perception_device build --out <new folder>
    python -m grounding.quest.perception_device push --folder <built folder>
    python -m grounding.quest.perception_device verify --folder <built folder>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "grounding.quest"

from ..evaluation.iref_vla.protocol import encode_json  # noqa: E402
from .replay_device import APP_DIR, _fail, _one_device, _remote_size, _stamp  # noqa: E402
from .runtime_identity import REPO, Adb  # noqa: E402

REMOTE = APP_DIR + "/perception"
SCHEMAS = ("auto-scan-config.v1.json", "auto-scan-config.v2.json", "auto-scan-config.v3.json", "auto-scan-config.v4.json", "auto-scan-config.v5.json", "auto-scan-config.v6.json", "auto-scan-config.v7.json", "auto-scan-config.v8.json", "capture.v1.json", "category-map.v1.json", "command-context.v2.json", "grounding-query.v2.json",
           "grounding-result.v2.json", "localization-config.v1.json", "scan-config.v1.json", "scene.v3.json",
           "size-priors.v1.json")
ROLES = {"scan_config": ("configs/scan.default.v3.json", "perception/scan/scan.default.v3.json", "scan-config.v1.json"),
         "localization_config": ("configs/localization.default.v2.json", "perception/localization/localization.default.v2.json",
                                 "localization-config.v1.json"),
         "size_priors": ("priors/size-priors.bedroom.v2.json", "perception/bedroom/size-priors.bedroom.v2.json", "size-priors.v1.json"),
         "category_map": ("maps/category-map.bedroom.v5.json", "perception/bedroom/category-map.bedroom.v5.json", None),
         "autoscan_config": ("configs/autoscan.default.v8.json", "perception/autoscan/autoscan.default.v8.json", "auto-scan-config.v8.json")}


def files(repo: Path = REPO) -> dict:
    """The folder's files: its path in the folder -> the repository file it copies."""
    out = {f"schemas/{n}": repo / "schemas" / n for n in SCHEMAS}
    out.update({dest: repo / src for dest, src, _ in ROLES.values()})
    return out


def check_sources(repo: Path = REPO) -> list:
    """Every reason the repository's files can't make a perception folder; empty when they can."""
    import jsonschema
    from grounding.contract import validate as CV
    problems = []
    for dest, src in files(repo).items():
        if not src.is_file():
            problems.append(f"{dest}: {src} is missing")
    if problems:
        return problems
    for n in SCHEMAS:
        try:
            s = json.loads((repo / "schemas" / n).read_text(encoding="utf-8"))
            jsonschema.validators.validator_for(s).check_schema(s)
        except Exception as e:  # noqa: BLE001
            problems.append(f"schemas/{n}: not a JSON Schema ({type(e).__name__}: {str(e)[:80]})")
    for role, (dest, src, schema) in ROLES.items():
        rec = json.loads((repo / src).read_text(encoding="utf-8"))
        if schema:
            s = json.loads((repo / "schemas" / schema).read_text(encoding="utf-8"))
            for e in jsonschema.validators.validator_for(s)(s).iter_errors(rec):
                problems.append(f"{dest}: {e.message[:100]}")
        else:
            problems += [f"{dest}: {i.code} {i.path}" for i in CV.check_record(dest, rec)]
    return problems


def build(*, out, repo=REPO) -> dict:
    out, repo = Path(out), Path(repo)
    if out.exists():
        _fail(f"{out} already exists; the folder is built new", "perception")
    problems = check_sources(repo)
    if problems:
        _fail("the repository's files can't make a perception folder: " + "; ".join(problems[:4]), "perception")
    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        _fail(f"{tmp} is left from an earlier build; remove it first", "perception")
    listed = {}
    for dest, src in sorted(files(repo).items()):
        b = src.read_bytes().replace(b"\r\n", b"\n")   # the same folder from any checkout
        (tmp / dest).parent.mkdir(parents=True, exist_ok=True)
        (tmp / dest).write_bytes(b)
        listed[dest] = {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
    manifest = {"format_version": 1, "record_type": "a4_perception_folder", "files": listed,
                **{role: dest for role, (dest, _, _) in ROLES.items()}}
    body = encode_json(manifest)
    (tmp / "manifest.json").write_bytes(body)
    tmp.rename(out)
    return {"folder": str(out), "manifest_sha256": hashlib.sha256(body).hexdigest(), "files": len(listed),
            "bytes": sum(f["bytes"] for f in listed.values())}


def verify(folder) -> list:
    """Every way a built folder differs from its manifest; empty when it reads back."""
    folder = Path(folder)
    mpath = folder / "manifest.json"
    if not mpath.is_file():
        return ["manifest.json is missing"]
    m = json.loads(mpath.read_text(encoding="utf-8"))
    if m.get("record_type") != "a4_perception_folder":
        return ["manifest.json is not a perception folder's manifest"]
    bad = []
    for name, f in m["files"].items():
        p = folder / name
        if not p.is_file():
            bad.append(f"{name} is missing")
        elif p.stat().st_size != f["bytes"] or hashlib.sha256(p.read_bytes()).hexdigest() != f["sha256"]:
            bad.append(f"{name} differs from the manifest")
    extra = sorted(str(p.relative_to(folder).as_posix()) for p in folder.rglob("*") if p.is_file()
                   and p.relative_to(folder).as_posix() not in m["files"] and p.name != "manifest.json")
    bad += [f"{x} is not in the manifest" for x in extra]
    for role in ROLES:
        if m.get(role) not in m["files"]:
            bad.append(f"the manifest's {role} names no listed file")
    return bad


def push(*, folder, adb=None, progress=print) -> dict:
    folder, adb = Path(folder), adb if adb is not None else Adb()
    bad = verify(folder)
    if bad:
        _fail("the folder doesn't read back: " + "; ".join(bad[:3]), "perception")
    serial = _one_device(adb)
    if adb.shell("ls", APP_DIR)[0] != 0:
        _fail(f"{APP_DIR} is not on the headset: start the app once (this helper never launches it)", "perception")
    adb.shell("rm", "-f", REMOTE + "/manifest.json")   # the app sees no folder until the new manifest arrives
    m = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    names = sorted(m["files"]) + ["manifest.json"]
    dirs = sorted({f"{REMOTE}/{n.rsplit('/', 1)[0]}" for n in names if "/" in n} | {REMOTE})
    if adb.shell("mkdir", "-p", *dirs)[0] != 0:
        _fail(f"cannot create {REMOTE}", "perception")
    pushed = []
    for name in names:   # the manifest last
        local, remote = folder / name, f"{REMOTE}/{name}"
        progress(f"pushing {name}")
        code, out, err = adb.run("push", str(local), remote)
        if code != 0:
            _fail(f"adb push {name} failed: {(err or out).strip()}", "perception")
        size, here = _remote_size(adb, remote), local.stat().st_size
        if size != here:
            _fail(f"{name}: {size} bytes on the headset, {here} here", "perception")
        pushed.append({"name": name, "bytes": here, "sha256": hashlib.sha256(local.read_bytes()).hexdigest()})
    receipt = {"format_version": 1, "record_type": "a4_perception_push", "pushed_utc": _stamp(), "device": serial,
               "folder": str(folder), "manifest_sha256": hashlib.sha256((folder / "manifest.json").read_bytes()).hexdigest(),
               "remote": REMOTE, "files": pushed}
    out = folder.parent / f"perception-push-{receipt['pushed_utc']}.json"
    out.write_bytes(encode_json(receipt))
    receipt["receipt"] = str(out)
    return receipt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m grounding.quest.perception_device", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="build a new perception folder from the repository")
    b.add_argument("--out", required=True)
    for name in ("push", "verify"):
        p = sub.add_parser(name, help=f"{name} a built perception folder")
        p.add_argument("--folder", required=True)
    a = ap.parse_args(argv)
    if a.command == "build":
        r = build(out=a.out)
        print(f"built {r['folder']}: {r['files']} files, {r['bytes']} bytes; manifest SHA-256 {r['manifest_sha256']}")
    elif a.command == "verify":
        bad = verify(a.folder)
        print("the folder reads back: every file matches its manifest" if not bad else "\n".join(bad))
        return 1 if bad else 0
    else:
        r = push(folder=a.folder)
        print(f"pushed {len(r['files'])} files to {r['remote']} (the manifest last); receipt {r['receipt']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
