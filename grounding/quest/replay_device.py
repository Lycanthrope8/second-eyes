"""A2.5 delivery 1, step 2 (D103): the laptop's helpers for the headset replay, over adb.

`push` copies the bundle's headset part, and nothing else, into the app's data folder (`files/replay/bundle`). It first
reads the bundle back on its own, then checks each pushed file's size on the headset and writes a receipt with every
file's SHA-256 to `runs/<id>/raw/replay/push-<UTC time>.json`. The float32 references never leave the laptop.

`pull` copies one replay results folder from the headset into `runs/<id>/raw/replay/<results>/`: the newest one that
has `done.json`, or the one named. It also copies the session's event log, which `identity.json` names, as
`events.jsonl`. It never writes over an existing folder; files that are missing are recorded, not invented. The
receipt `pull.json` holds what arrived, with hashes and the done-record's counts.

Neither helper installs, launches or reruns the app. The adb commands used are `devices`, `shell ls`,
`shell mkdir -p`, `shell stat -c %s`, `push` and `pull`.
"""
from __future__ import annotations

import datetime
import json
import os
import re
import shutil
from pathlib import Path

from ..evaluation.iref_vla.protocol import EvaluationInputError, encode_json, issue
from .publish import publish
from .replay_bundle import verify_replay_bundle
from .runtime_identity import REPO, Adb, file_sha256

APP_DIR = "/sdcard/Android/data/com.secondeyes.quest/files"
REMOTE_BUNDLE = APP_DIR + "/replay/bundle"
REMOTE_RESULTS = APP_DIR + "/replay/results"
HEADSET_FILES = ("replay-manifest.json", "requests.jsonl", "fixtures-written.jsonl")
RESULT_FILES = ("identity.json", "startup-log.txt", "selfchecks.json", "fixtures.jsonl", "results.jsonl", "done.json")
RUN_ID = re.compile(r"\d{8}_[A-Z0-9]+_r\d{3}")
RESULTS_NAME = re.compile(r"\d{8}-\d{6}")


def _fail(message, where="replay"):
    raise EvaluationInputError([issue(where, "E_REPLAY_DEVICE", message)])


def _run_folder(repo, run_id) -> Path:
    if not RUN_ID.fullmatch(run_id or ""):
        _fail(f"{run_id!r} is not a run ID")
    folder = Path(repo) / "runs" / run_id
    if not (folder / "config.yaml").is_file():
        _fail(f"run {run_id} does not exist (no runs/{run_id}/config.yaml)")
    return folder


def _one_device(adb) -> str:
    code, out, err = adb.run("devices")
    serials = [line.split("\t")[0] for line in out.splitlines()[1:] if line.strip().endswith("\tdevice")]
    if code != 0 or len(serials) != 1:
        _fail(f"{len(serials)} devices ready; exactly one is needed" + (f" ({err.strip()})" if code != 0 else ""))
    return serials[0]


def _remote_size(adb, path):
    code, out, _ = adb.shell("stat", "-c", "%s", path)
    try:
        return int(out.strip()) if code == 0 else None
    except ValueError:
        return None


def _stamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")


def push(*, bundle, run_id, adb=None, repo=None, progress=print) -> dict:
    """Push the bundle's headset files; returns the receipt it wrote."""
    adb, repo = adb if adb is not None else Adb(), Path(repo) if repo is not None else REPO
    folder = _run_folder(repo, run_id)
    b = Path(bundle)
    bad = verify_replay_bundle(b)
    if bad:
        _fail("the bundle does not read back: " + "; ".join(bad[:3]), str(b))
    serial = _one_device(adb)
    if adb.shell("ls", APP_DIR)[0] != 0:
        _fail(f"{APP_DIR} is not on the headset: start the app once (this helper never launches it)")
    if adb.shell("mkdir", "-p", REMOTE_BUNDLE)[0] != 0:
        _fail(f"cannot create {REMOTE_BUNDLE}")
    files = []
    for name in HEADSET_FILES:
        local, remote = b / "headset" / name, f"{REMOTE_BUNDLE}/{name}"
        progress(f"pushing {name}")
        code, out, err = adb.run("push", str(local), remote)
        if code != 0:
            _fail(f"adb push {name} failed: {(err or out).strip()}")
        size, here = _remote_size(adb, remote), local.stat().st_size
        if size != here:
            _fail(f"{name}: {size} bytes on the headset, {here} here")
        files.append({"name": name, "remote": remote, "bytes": here, "sha256": file_sha256(local)})
    receipt = {"format_version": 1, "record_type": "a25_replay_push", "pushed_utc": _stamp(), "device": serial,
               "bundle": str(b), "bundle_manifest_sha256": file_sha256(b / "manifest.json"), "files": files,
               "not_pushed": "reference/ (the float32 references stay on the laptop)"}
    out = folder / "raw" / "replay" / f"push-{receipt['pushed_utc']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    k = 1
    while out.exists():   # never overwritten: a second receipt in the same second gets the next suffix
        k += 1
        out = folder / "raw" / "replay" / f"push-{receipt['pushed_utc']}-{k}.json"
    out.write_bytes(encode_json(receipt))
    receipt["receipt"] = str(out)
    return receipt


def pull(*, run_id, results=None, adb=None, repo=None, progress=print) -> dict:
    """Pull one results folder and its session's event log; returns the receipt it wrote."""
    adb, repo = adb if adb is not None else Adb(), Path(repo) if repo is not None else REPO
    folder = _run_folder(repo, run_id)
    serial = _one_device(adb)
    code, out, _ = adb.shell("ls", "-1", REMOTE_RESULTS)
    names = sorted(x.strip() for x in out.splitlines() if RESULTS_NAME.fullmatch(x.strip())) if code == 0 else []
    if results is not None:
        if results not in names:
            _fail(f"no results folder {results!r} on the headset; found {names}")
        chosen = results
    else:
        finished = [n for n in names if _remote_size(adb, f"{REMOTE_RESULTS}/{n}/done.json") is not None]
        if not finished:
            _fail(f"no finished results folder (with done.json) on the headset; found {names}")
        chosen = finished[-1]
    dest = folder / "raw" / "replay" / chosen
    if dest.exists():
        _fail(f"{dest} exists: already pulled (nothing is overwritten)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = dest.parent / f".{chosen}.partial"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    try:
        missing = []
        for name in RESULT_FILES:
            progress(f"pulling {name}")
            code, _, _ = adb.pull(f"{REMOTE_RESULTS}/{chosen}/{name}", staging / name)
            if code != 0 or not (staging / name).is_file():
                missing.append(name)
        ident = _read_json(staging / "identity.json")
        log = (ident or {}).get("app", {}).get("event_log") if isinstance(ident, dict) else None
        if log:
            progress("pulling the session's event log")
            code, _, _ = adb.pull(log, staging / "events.jsonl")
            if code != 0 or not (staging / "events.jsonl").is_file():
                missing.append("events.jsonl")
        else:
            missing.append("events.jsonl")
        done = _read_json(staging / "done.json")
        lines = (staging / "results.jsonl").read_text(encoding="utf-8").splitlines() if (staging / "results.jsonl").is_file() else []
        parsed = [_parse(x) for x in lines if x.strip()]
        ids = [p.get("request_id") for p in parsed if isinstance(p, dict)]
        checks = {"results_lines": len(parsed), "unparseable_lines": sum(p is None for p in parsed),
                  "distinct_request_ids": len(set(ids)),
                  "done_written": done.get("written") if isinstance(done, dict) else None,
                  "lines_match_done": isinstance(done, dict) and done.get("written") == len(parsed)}
        receipt = {"format_version": 1, "record_type": "a25_replay_pull", "pulled_utc": _stamp(), "device": serial,
                   "results": chosen, "complete": not missing and checks["lines_match_done"],
                   "files": {p.name: {"bytes": p.stat().st_size, "sha256": file_sha256(p)} for p in sorted(staging.iterdir())},
                   "missing": missing, "checks": checks,
                   "done": {k: done.get(k) for k in ("requests", "written", "fixtures", "fixtures_as_expected",
                                                     "self_checks_passed", "total_ms")} if isinstance(done, dict) else None}
        (staging / "pull.json").write_bytes(encode_json(receipt))
        publish(staging, dest)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    receipt["folder"] = str(dest)
    return receipt


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _parse(line):
    try:
        return json.loads(line)
    except ValueError:
        return None
