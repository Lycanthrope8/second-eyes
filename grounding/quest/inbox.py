"""A2.5 delivery 4: the laptop's side of the headset's ADB inbox (D99(1)-(3)). ADB is development transport only; all
grounding computation stays on the headset.

`inbox_send` builds one request from the goldens:

- the expected scene is a golden snapshot: its ID, scene ID, revision, and the SHA-256 of its file;
- the command is a golden dataset command or, given text, a written command, labelled `written` (D99(5)) with a
  suffixed command ID;
- it is pushed under a temporary name (`.json.tmp`) and renamed on the headset when complete (D99(2));
- the receipt records the PC's push-and-rename time on the PC clock only. It is never combined with the headset's
  stage times (D99(3)).

`outbox_pull` brings the acknowledgements, results and duplicate notices, and the newest session's log, into the run's
`raw/outbox/`, with a summary per request.
"""
from __future__ import annotations

import datetime
import json
import re
import shutil
import tempfile
import time
from pathlib import Path

from ..evaluation.iref_vla.protocol import encode_json, sha256
from .prompt_goldens import verify_goldens
from .publish import publish
from .replay_device import APP_DIR, _fail, _one_device, _run_folder, _stamp
from .runtime_identity import REPO, Adb, file_sha256

INBOX = APP_DIR + "/interactive/inbox"
OUTBOX = APP_DIR + "/interactive/outbox"
SESSIONS = APP_DIR + "/interactive/sessions"
REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def build_request(*, goldens, snapshot, request_id, command_file=None, text=None, dataset=None) -> dict:
    head = Path(goldens) / "headset"
    rows = [json.loads(x) for x in (head / "goldens.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    scenes = {}
    for r in rows:
        if r["snapshot_id"] not in scenes:
            data = (head / r["scene_file"]).read_bytes()
            scenes[r["snapshot_id"]] = (r["scene_file"], data, json.loads(data))
    pick = [sid for sid, (_, _, rec) in scenes.items() if sid == str(snapshot) or str(len(rec["objects"])) == str(snapshot)]
    if len(pick) != 1:
        counts = sorted(len(rec["objects"]) for _, _, rec in scenes.values())
        _fail(f"snapshot {snapshot!r} names {len(pick)} golden snapshots; give an object count ({counts}) or a snapshot ID")
    sid = pick[0]
    _, data, scene = scenes[sid]
    mine = [r for r in rows if r["snapshot_id"] == sid]
    if dataset is not None:
        base = [r for r in mine if r["kind"] == "dataset_command"]
        if not 1 <= int(dataset) <= len(base):
            _fail(f"snapshot {sid} has {len(base)} dataset commands; --dataset takes 1 to {len(base)}")
        command_file = base[int(dataset) - 1]["command_file"]
    if command_file is not None:
        hit = [r for r in mine if r["command_file"] == command_file]
        if not hit:
            _fail(f"{command_file} is not a golden command of snapshot {sid}")
        command, kind = json.loads((head / command_file).read_text(encoding="utf-8")), hit[0]["kind"]
    else:
        base = [r for r in mine if r["kind"] == "dataset_command"]
        if text is None or not text.strip() or not base:
            _fail("give a golden command file, or non-empty text on a snapshot with a dataset command")
        command = json.loads((head / base[0]["command_file"]).read_text(encoding="utf-8"))
        command = dict(command, text=text, command_id=f"{command['command_id']}-w{_stamp()}")
        kind = "written"
    return {"format_version": 1, "record_type": "a25_interactive_request", "request_id": request_id,
            "expected_scene": {"snapshot_id": sid, "scene_id": scene["scene_id"], "scene_revision": scene["scene_revision"],
                               "snapshot_sha256": sha256(data)},
            "command": command, "command_kind": kind}


def inbox_send(*, run_id, goldens, snapshot, command_file=None, text=None, dataset=None, request_id=None, resend=False,
               adb=None, repo=None, progress=print) -> dict:
    """resend delivers a request ID again on purpose, to check that the headset never processes it twice."""
    repo = Path(repo) if repo is not None else REPO
    adb = adb if adb is not None else Adb()
    folder = _run_folder(repo, run_id)
    request_id = request_id or f"pc-{_stamp()}"
    if not REQUEST_ID.match(request_id):
        _fail(f"request ID {request_id!r} must match {REQUEST_ID.pattern}")
    receipt_path = folder / "raw" / "inbox" / f"send-{request_id}.json"
    if receipt_path.exists() and not resend:
        _fail(f"request {request_id} was already sent in this run ({receipt_path}); request IDs are used once (--resend to test duplicates)")
    k = 2
    while receipt_path.exists():
        receipt_path = folder / "raw" / "inbox" / f"send-{request_id}-resend{k}.json"
        k += 1
    bad = verify_goldens(goldens)
    if bad:
        _fail("the goldens do not read back: " + "; ".join(bad[:3]), str(goldens))
    req = build_request(goldens=goldens, snapshot=snapshot, request_id=request_id, command_file=command_file, text=text, dataset=dataset)
    data = encode_json(req)
    _one_device(adb)
    if adb.shell("ls", INBOX)[0] != 0:
        _fail(f"{INBOX} is not on the headset: start a session on the panel first (this helper never launches it)")
    if not resend and adb.shell("stat", "-c", "%s", f"{OUTBOX}/{request_id}.result.json")[0] == 0:
        _fail(f"the headset already holds a result for request ID {request_id} from an earlier session (its outbox keeps them); "
              f"request IDs must be new: use another --request-id, or another --prefix for a batch")
    tmp_remote, final = f"{INBOX}/{request_id}.json.tmp", f"{INBOX}/{request_id}.json"
    with tempfile.TemporaryDirectory() as td:
        local = Path(td) / f"{request_id}.json"
        local.write_bytes(data)
        progress(f"pushing {request_id} under a temporary name")
        t0 = time.monotonic()
        if adb.run("push", str(local), tmp_remote)[0] != 0:
            _fail(f"pushing {tmp_remote} failed")
        if adb.shell("mv", tmp_remote, final)[0] != 0:
            _fail(f"renaming {tmp_remote} to {final} failed")
        seconds = time.monotonic() - t0
    receipt = {"format_version": 1, "record_type": "a25_inbox_send", "run_id": run_id, "request_id": request_id, "resend": bool(resend),
               "remote": final, "request_sha256": sha256(data), "request": req,
               "pc_push_and_rename_s": round(seconds, 4),
               "pc_clock_note": "the PC clock only; never combined with the headset's stage times (D99(3))",
               "sent_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_bytes(encode_json(receipt))
    return dict(receipt, receipt=str(receipt_path))


def outbox_pull(*, run_id, adb=None, repo=None, progress=print) -> dict:
    repo = Path(repo) if repo is not None else REPO
    adb = adb if adb is not None else Adb()
    folder = _run_folder(repo, run_id)
    _one_device(adb)
    code, out, _ = adb.shell("ls", "-1", OUTBOX)
    if code != 0:
        _fail(f"{OUTBOX} is not on the headset: no session has run")
    names = sorted(n.strip() for n in out.splitlines() if n.strip().endswith(".json") and not n.strip().endswith(".tmp"))
    code, out, _ = adb.shell("ls", "-1", SESSIONS)
    sessions = sorted(n.strip() for n in out.splitlines() if n.strip()) if code == 0 else []
    dest = folder / "raw" / "outbox" / _stamp()
    if dest.exists():
        _fail(f"{dest} already exists (nothing is overwritten)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{dest.name}.partial-", dir=str(dest.parent)))
    try:
        (staging / "outbox").mkdir()
        (staging / "session").mkdir()
        for n in names:
            progress(f"pulling {n}")
            if adb.pull(f"{OUTBOX}/{n}", staging / "outbox" / n)[0] != 0:
                _fail(f"pulling {n} failed")
        if sessions:
            for n in ("session.json", "outcomes.jsonl", "startup-log.txt"):
                progress(f"pulling the session {sessions[-1]}'s {n}")
                adb.pull(f"{SESSIONS}/{sessions[-1]}/{n}", staging / "session" / n)
        summary = {}
        for n in names:
            rec = json.loads((staging / "outbox" / n).read_text(encoding="utf-8"))
            s = summary.setdefault(rec.get("request_id"), {})
            if n.endswith(".ack.json"):
                s["ack"] = rec.get("status") + (f" ({rec['reason']})" if rec.get("reason") else "")
            elif n.endswith(".result.json"):
                o, t = rec.get("outcome", {}), rec.get("times", {})
                s.update(status=o.get("status"), reason=o.get("reason"), target=o.get("target_object_id"),
                         execution_path=o.get("execution_path"), app_observed_ms=t.get("app_observed_ms"))
            else:
                s["duplicates"] = s.get("duplicates", 0) + 1
        receipt = {"format_version": 1, "record_type": "a25_outbox_pull", "run_id": run_id, "session": sessions[-1] if sessions else None,
                   "files": {p.relative_to(staging).as_posix(): file_sha256(p) for p in sorted(staging.rglob("*")) if p.is_file()},
                   "requests": summary}
        (staging / "pull.json").write_bytes(encode_json(receipt))
        publish(staging, dest)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return dict(receipt, folder=str(dest))


def await_result(adb, request_id, *, timeout_s=180.0, poll_s=1.0, sleep=time.sleep):
    """The headset's result for one request, once it exists (results are written under a temporary name and renamed,
    so a file that exists is complete), or None after timeout_s."""
    path, t0 = f"{OUTBOX}/{request_id}.result.json", time.monotonic()
    while True:
        if adb.shell("stat", "-c", "%s", path)[0] == 0:
            with tempfile.TemporaryDirectory() as td:
                local = Path(td) / "result.json"
                if adb.pull(path, local)[0] == 0:
                    return json.loads(local.read_text(encoding="utf-8"))
        if time.monotonic() - t0 >= timeout_s:
            return None
        sleep(poll_s)


def inbox_batch(*, run_id, goldens, snapshot, prefix=None, paced=True, timeout_s=180.0, poll_s=1.0, sleep=time.sleep,
                adb=None, repo=None, progress=print) -> list:
    """Every golden dataset command of one snapshot, as separate requests (prefix + object count + "-" + index), in
    order. Paced (the default): each request is sent only after the previous one is answered, so no request waits in
    the headset's queue and the app-observed time is the command's own (r018 sent them all at once). The headset's
    session must be bound to that snapshot, or each is refused as a stale binding."""
    adb = adb if adb is not None else Adb()
    head = Path(goldens) / "headset"
    rows = [json.loads(x) for x in (head / "goldens.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    counts = {}
    for r in rows:
        if r["snapshot_id"] not in counts:
            counts[r["snapshot_id"]] = len(json.loads((head / r["scene_file"]).read_text(encoding="utf-8"))["objects"])
    pick = [sid for sid, n in counts.items() if sid == str(snapshot) or str(n) == str(snapshot)]
    if len(pick) != 1:
        _fail(f"snapshot {snapshot!r} names {len(pick)} golden snapshots; give an object count ({sorted(counts.values())}) or a snapshot ID")
    n = sum(r["snapshot_id"] == pick[0] and r["kind"] == "dataset_command" for r in rows)
    if prefix is None:   # unique per run: the headset's outbox keeps results across sessions (r019 met r018's)
        prefix = run_id.rsplit("_", 1)[-1] + "m"
    out = []
    for k in range(1, n + 1):
        rid = f"{prefix}{counts[pick[0]]}-{k:02d}"
        sent = inbox_send(run_id=run_id, goldens=goldens, snapshot=pick[0], dataset=k, request_id=rid, adb=adb, repo=repo,
                          progress=lambda m: None)
        if paced:
            res = await_result(adb, rid, timeout_s=timeout_s, poll_s=poll_s, sleep=sleep)
            if res is None:
                _fail(f"no result for {rid} within {timeout_s:.0f} s: is the session running on the headset, and showing the "
                      f"{counts[pick[0]]}-object scene? {k - 1} of {n} were answered; nothing more was sent")
            o, tm = res.get("outcome", {}), res.get("times", {})
            sent["result"] = {"status": o.get("status"), "reason": o.get("reason"), "target": o.get("target_object_id"),
                              "execution_path": o.get("execution_path"), "app_observed_ms": tm.get("app_observed_ms")}
            ms = tm.get("app_observed_ms")
            progress(f"{rid} ({k}/{n}): {o.get('status')}" + (f", target {o['target_object_id']}" if o.get("target_object_id") else "")
                     + (f" ({o.get('reason')})" if o.get("status") not in ("completed",) else "")
                     + (f", {ms / 1000:.1f} s app-observed" if isinstance(ms, (int, float)) else ""))
        else:
            progress(f"{rid} ({k}/{n}) sent")
        out.append(sent)
    return out
