"""A2.7 Delivery 3 (D115, D116): the laptop's helpers for the headset, over adb, in A2.5's manner (golden_device.py).

    python -m grounding.a27 parser-push --folder FOLDER --run RUN [--model GGUF] [--adb ADB]
    python -m grounding.a27 parser-pull --run RUN [--results STAMP] [--adb ADB]

parser-push verifies the parser folder here as the judge does (every file against its manifest, every count), removes the headset's old manifest first
so the app never takes a half-pushed folder for present, pushes every file into the app's files/parser with a size
check on the headset, and the manifest last. With --model, the GGUF goes to files/<its name>, checked here against the
folder's parser.json (SHA-256) and on the headset by size. A receipt goes to runs/RUN/raw/parser/push-<UTC>.json.
parser-pull copies one parser-check results folder (the newest with done.json, or the one named) into
runs/RUN/raw/parser/<results>/ and checks every file's size. It never writes over an existing folder. Neither helper
installs, launches or runs the app.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..evaluation.iref_vla.protocol import encode_json
from ..quest.replay_device import APP_DIR, _fail, _one_device, _remote_size, _run_folder, _stamp
from ..quest.runtime_identity import REPO, Adb, file_sha256

REMOTE_PARSER = APP_DIR + "/parser"
REMOTE_RESULTS = APP_DIR + "/parser-results"
RESULT_FILES = ("run.json", "decodes.jsonl", "masks.jsonl", "resolutions.jsonl", "readers.jsonl", "startup-log.txt", "done.json")


def verify_folder(folder: Path) -> dict:
    """The judge's own check (headset.verify_folder: every file by SHA-256 and size, every count), as a refusal here."""
    from .headset import HeadsetError
    from .headset import verify_folder as check
    try:
        return check(folder)
    except HeadsetError as e:
        _fail(str(e), "parser")


def push_parser(*, folder, run_id, model=None, adb=None, repo=None, progress=print) -> dict:
    adb, repo = adb if adb is not None else Adb(), Path(repo) if repo is not None else REPO
    run = _run_folder(repo, run_id)
    folder = Path(folder)
    man = verify_folder(folder)
    parser = json.loads((folder / "assets" / "parser.json").read_text(encoding="utf-8"))
    if model is not None and file_sha256(Path(model)) != parser["gguf_sha256"]:
        _fail(f"{model} is not the exported candidate named in the folder's parser.json", "parser")
    serial = _one_device(adb)
    if adb.shell("ls", APP_DIR)[0] != 0:
        _fail(f"{APP_DIR} is not on the headset: start the app once (this helper never launches it)", "parser")
    adb.shell("rm", "-f", f"{REMOTE_PARSER}/manifest.json")
    names = sorted(man["files"]) + ["manifest.json"]
    dirs = sorted({f"{REMOTE_PARSER}/{n.rsplit('/', 1)[0]}" for n in names if "/" in n} | {REMOTE_PARSER})
    if adb.shell("mkdir", "-p", *dirs)[0] != 0:
        _fail(f"cannot create {REMOTE_PARSER}", "parser")
    files = []
    for name in names:   # the manifest last
        local, remote = folder / name, f"{REMOTE_PARSER}/{name}"
        progress(f"pushing {name}")
        code, out, err = adb.run("push", str(local), remote)
        if code != 0:
            _fail(f"adb push {name} failed: {(err or out).strip()}", "parser")
        size, here = _remote_size(adb, remote), local.stat().st_size
        if size != here:
            _fail(f"{name}: {size} bytes on the headset, {here} here", "parser")
        files.append({"name": name, "bytes": here})
    model_rec = None
    if model is not None:
        remote = f"{APP_DIR}/{parser['gguf_file']}"
        here = Path(model).stat().st_size
        if _remote_size(adb, remote) == here:
            model_rec = {"file": parser["gguf_file"], "bytes": here, "pushed": False, "note": "already on the headset, same size"}
        else:
            progress(f"pushing {parser['gguf_file']} ({here / 2**20:.0f} MB)")
            code, out, err = adb.run("push", str(model), remote)
            if code != 0 or _remote_size(adb, remote) != here:
                _fail(f"the model push failed: {(err or out).strip()}", "parser")
            model_rec = {"file": parser["gguf_file"], "bytes": here, "pushed": True}
        model_rec["sha256"] = parser["gguf_sha256"]
    receipt = {"format_version": 1, "record_type": "a27_parser_push", "pushed_utc": _stamp(), "device": serial,
               "folder": str(folder), "manifest_sha256": file_sha256(folder / "manifest.json"), "remote": REMOTE_PARSER,
               "files": files, "model": model_rec}
    out = run / "raw" / "parser" / f"push-{receipt['pushed_utc']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(encode_json(receipt))
    receipt["receipt"] = str(out)
    return receipt


def pull_parser(*, run_id, results=None, adb=None, repo=None, progress=print) -> dict:
    adb, repo = adb if adb is not None else Adb(), Path(repo) if repo is not None else REPO
    run = _run_folder(repo, run_id)
    _one_device(adb)
    code, out, _ = adb.shell("ls", REMOTE_RESULTS)
    if code != 0:
        _fail(f"no {REMOTE_RESULTS} on the headset: run the Parser check first", "parser")
    stamps = sorted(x.strip() for x in out.split() if x.strip())
    if results is not None:
        if results not in stamps:
            _fail(f"no results folder {results} on the headset", "parser")
        pick = results
    else:
        done = [s for s in stamps if _remote_size(adb, f"{REMOTE_RESULTS}/{s}/done.json") is not None]
        if not done:
            _fail("no finished parser check (no results folder with done.json)", "parser")
        pick = done[-1]
    dest = run / "raw" / "parser" / pick
    if dest.exists():
        _fail(f"{dest} already exists", "parser")
    dest.mkdir(parents=True)
    pulled = []
    for name in RESULT_FILES:
        remote = f"{REMOTE_RESULTS}/{pick}/{name}"
        size = _remote_size(adb, remote)
        if size is None:
            continue
        progress(f"pulling {name}")
        code, o, e = adb.run("pull", remote, str(dest / name))
        if code != 0 or (dest / name).stat().st_size != size:
            _fail(f"adb pull {name} failed: {(e or o).strip()}", "parser")
        pulled.append({"name": name, "bytes": size})
    if not any(p["name"] == "done.json" for p in pulled) and results is None:
        _fail("the results folder lost its done.json while pulling", "parser")
    return {"results": pick, "folder": str(dest), "files": pulled}
