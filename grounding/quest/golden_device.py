"""A2.5 delivery 2: the laptop's helpers for the on-device golden self-check, over adb.

`push_goldens` (command `golden-push`) first reads the goldens back on their own. It then pushes the files their golden
manifest lists into the app's `files/prompting/goldens`, checking each size on the headset, and pushes the manifest itself
last, because the app treats goldens as present only once it is there. A receipt with every file's SHA-256 goes to
`runs/<id>/raw/goldens/push-<UTC time>.json`.

`pull_goldens` (command `golden-pull`) copies one results folder, the newest with `done.json` or the one named, and the
session's event log into `runs/<id>/raw/goldens/<results>/`. It never writes over an existing folder.

Neither helper installs, launches or reruns the app.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..evaluation.iref_vla.protocol import encode_json
from .prompt_goldens import verify_goldens
from .publish import publish
from .replay_device import APP_DIR, RESULTS_NAME, _fail, _one_device, _read_json, _remote_size, _run_folder, _stamp
from .runtime_identity import REPO, Adb, file_sha256

REMOTE_GOLDENS = APP_DIR + "/prompting/goldens"
REMOTE_RESULTS = APP_DIR + "/prompting/results"
RESULT_FILES = ("identity.json", "selfchecks.json", "results.jsonl", "done.json")
REMOTE_INTERACTIVE = APP_DIR + "/interactive/results"
INTERACTIVE_FILES = ("identity.json", "outcomes-off.jsonl", "outcomes-prefix.jsonl", "done.json")


def push_goldens(*, goldens, run_id, adb=None, repo=None, progress=print) -> dict:
    adb, repo = adb if adb is not None else Adb(), Path(repo) if repo is not None else REPO
    folder = _run_folder(repo, run_id)
    g = Path(goldens)
    bad = verify_goldens(g)
    if bad:
        _fail("the goldens do not read back: " + "; ".join(bad[:3]), str(g))
    serial = _one_device(adb)
    if adb.shell("ls", APP_DIR)[0] != 0:
        _fail(f"{APP_DIR} is not on the headset: start the app once (this helper never launches it)")
    head = g / "headset"
    names = sorted(json.loads((head / "golden-manifest.json").read_text(encoding="utf-8"))["files"]) + ["golden-manifest.json"]
    dirs = sorted({f"{REMOTE_GOLDENS}/{n.rsplit('/', 1)[0]}" for n in names if "/" in n} | {REMOTE_GOLDENS})
    if adb.shell("mkdir", "-p", *dirs)[0] != 0:
        _fail(f"cannot create {REMOTE_GOLDENS}")
    files = []
    for name in names:   # the manifest last
        local, remote = head / name, f"{REMOTE_GOLDENS}/{name}"
        progress(f"pushing {name}")
        code, out, err = adb.run("push", str(local), remote)
        if code != 0:
            _fail(f"adb push {name} failed: {(err or out).strip()}")
        size, here = _remote_size(adb, remote), local.stat().st_size
        if size != here:
            _fail(f"{name}: {size} bytes on the headset, {here} here")
        files.append({"name": name, "bytes": here, "sha256": file_sha256(local)})
    receipt = {"format_version": 1, "record_type": "a25_golden_push", "pushed_utc": _stamp(), "device": serial,
               "goldens": str(g), "goldens_manifest_sha256": file_sha256(g / "manifest.json"), "remote": REMOTE_GOLDENS,
               "files": files, "not_pushed": "manifest.json and selection.json (the PC's provenance)"}
    out = folder / "raw" / "goldens" / f"push-{receipt['pushed_utc']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    k = 1
    while out.exists():
        k += 1
        out = folder / "raw" / "goldens" / f"push-{receipt['pushed_utc']}-{k}.json"
    out.write_bytes(encode_json(receipt))
    receipt["receipt"] = str(out)
    return receipt


def _golden_checks(staging, done) -> dict:
    rows = [json.loads(x) for x in (staging / "results.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()] \
        if (staging / "results.jsonl").is_file() else []
    return {"result_lines": len(rows), "all_ok_lines": sum(r.get("all_ok") is True for r in rows),
            "lines_match_done": isinstance(done, dict) and done.get("checked") == len(rows)
            and done.get("all_ok") == sum(r.get("all_ok") is True for r in rows)}


def _interactive_checks(staging, done) -> dict:
    out = {}
    for mode in ("off", "prefix"):
        f = staging / f"outcomes-{mode}.jsonl"
        rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()] if f.is_file() else []
        statuses = {}
        for r in rows:
            statuses[r.get("status")] = statuses.get(r.get("status"), 0) + 1
        out[mode] = {"lines": len(rows), "statuses": dict(sorted(statuses.items(), key=lambda kv: str(kv[0]))),
                     "kept_tokens_total": sum(int(r.get("kept_tokens") or 0) for r in rows)}
    out["lines_match_done"] = isinstance(done, dict) and done.get("off") == out["off"]["lines"] and done.get("prefix") == out["prefix"]["lines"]
    return out


def pull_interactive(*, run_id, results=None, adb=None, repo=None, progress=print) -> dict:
    """The newest finished interactive-check folder (or the named one), into the run's raw/interactive."""
    return pull_goldens(run_id=run_id, results=results, adb=adb, repo=repo, progress=progress, remote=REMOTE_INTERACTIVE,
                        files=INTERACTIVE_FILES, kind="interactive", summarize=_interactive_checks)


def pull_goldens(*, run_id, results=None, adb=None, repo=None, progress=print, remote=REMOTE_RESULTS, files=RESULT_FILES,
                 kind="goldens", summarize=None) -> dict:
    adb, repo = adb if adb is not None else Adb(), Path(repo) if repo is not None else REPO
    folder = _run_folder(repo, run_id)
    serial = _one_device(adb)
    code, out, _ = adb.shell("ls", "-1", remote)
    names = sorted(x.strip() for x in out.splitlines() if RESULTS_NAME.fullmatch(x.strip())) if code == 0 else []
    if results is not None:
        if results not in names:
            _fail(f"no {kind} results folder {results!r} on the headset; found {names}")
        chosen = results
    else:
        finished = [n for n in names if _remote_size(adb, f"{remote}/{n}/done.json") is not None]
        if not finished:
            _fail(f"no finished {kind} results folder (with done.json) on the headset; found {names}")
        chosen = finished[-1]
    dest = folder / "raw" / kind / chosen
    if dest.exists():
        _fail(f"{dest} exists: already pulled (nothing is overwritten)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = dest.parent / f".{chosen}.partial"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    try:
        missing = []
        for name in files:
            progress(f"pulling {name}")
            code, _, _ = adb.pull(f"{remote}/{chosen}/{name}", staging / name)
            if code != 0 or not (staging / name).is_file():
                missing.append(name)
        ident = _read_json(staging / "identity.json")
        log = ident.get("event_log") if isinstance(ident, dict) else None
        if log:
            code, _, _ = adb.pull(log, staging / "events.jsonl")
            if code != 0 or not (staging / "events.jsonl").is_file():
                missing.append("events.jsonl")
        else:
            missing.append("events.jsonl")
        done = _read_json(staging / "done.json")
        checks = (summarize or _golden_checks)(staging, done)
        receipt = {"format_version": 1, "record_type": "a25_golden_pull" if kind == "goldens" else f"a25_{kind}_pull", "pulled_utc": _stamp(), "device": serial,
                   "results": chosen, "complete": not missing and checks["lines_match_done"],
                   "files": {p.name: {"bytes": p.stat().st_size, "sha256": file_sha256(p)} for p in sorted(staging.iterdir())},
                   "missing": missing, "checks": checks,
                   "done": ({k: done.get(k) for k in ("goldens", "checked", "all_ok", "total_ms")} if kind == "goldens" else done)
                   if isinstance(done, dict) else None}
        (staging / "pull.json").write_bytes(encode_json(receipt))
        publish(staging, dest)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    receipt["folder"] = str(dest)
    return receipt
