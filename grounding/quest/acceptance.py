"""A2.5 correction acceptance: the checks of the raw records (ChatGPT's brief of 10 October, sections 5 and 6).

It reads the run's newest pulled outbox folder (`outbox-pull --sessions 2`), the run's inbox receipts, the newest D102
identity folder, the operator checklist and the recorded source identity. It writes `raw/acceptance/<UTC time>/`
(`summary.json`, `report.md`) with one row per requirement: the evidence, and pass, fail or incomplete. The helper
summary of `outbox-pull` is not relied on: every check reads the files themselves.

**Request IDs** are the run's number plus the case and step: `r023-A1`, `r023-B1`, `r023-B2`, `r023-C1`, `r023-C2`,
`r023-D1`. A repeated case uses a suffix (`r023-B1-2` with `r023-B2-2`). Each case is judged on its latest attempt;
earlier attempts are listed as incomplete, never counted as evidence.

**Presets** have no inbox acknowledgement, so they are taken from the session logs:

- case A's preset is the first preset of session 1;
- case C's preset is the one session 1 answered as `session_ended`.

**The cases:**

- **A:** session 1 starts; A1 and a preset complete (or ASK) on path U with nothing kept; the prompt and mapping
  identities equal the golden's.
- **B:** B1 is evaluated, then refused as `stale_result_scene_changed`, with no target. B2 is refused before
  evaluation (`scene_changed_before_evaluation`). The operator confirms that the current mapping is shown without the
  old target.
- **C:** C1 is cancelled during evaluation, with no target. C2 and the preset are answered `session_ended`. The
  session's closing record is clean and comes after its last result.
- **D:** session 2 starts, with its startup check passed again. D1 completes on path U. The closing record is clean,
  and no session-1 result appears in session 2's log.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from ..evaluation.iref_vla.protocol import encode_json
from .publish import publish
from .replay_device import _fail, _run_folder, _stamp
from .runtime_identity import REPO

ROW = ("requirement", "evidence", "result", "limitation")


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _jl(p) -> list:
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()] if Path(p).is_file() else []


def check_acceptance(*, run_id, goldens, repo=None, pull=None) -> dict:
    repo = Path(repo) if repo is not None else REPO
    folder = _run_folder(repo, run_id)
    prefix = run_id.rsplit("_", 1)[-1]
    pulls = sorted(p for p in (folder / "raw" / "outbox").glob("*") if (p / "pull.json").is_file()) if (folder / "raw" / "outbox").is_dir() else []
    src = Path(pull) if pull is not None else (pulls[-1] if pulls else None)
    if src is None:
        _fail(f"no pulled outbox folder under {folder / 'raw' / 'outbox'}; run outbox-pull --sessions 2 first")
    rows = []

    def row(req, evidence, ok, limitation="", incomplete=False):
        rows.append({"requirement": req, "evidence": evidence, "result": "incomplete" if incomplete else ("pass" if ok else "fail"),
                     "limitation": limitation})

    # the sessions
    sess_dirs = [d for d in [src / "session"] + sorted((src / "sessions").glob("*")) if (d / "session.json").is_file()]
    sessions = {}
    for d in sess_dirs:
        sj = json.loads((d / "session.json").read_text(encoding="utf-8"))
        sessions[sj["session"]] = {"dir": d, "json": sj, "log": _jl(d / "outcomes.jsonl"),
                                   "end": json.loads((d / "session-end.json").read_text(encoding="utf-8")) if (d / "session-end.json").is_file() else None,
                                   "events": _jl(d / "events.jsonl")}
    order = sorted(sessions)
    by_id = {}
    for s in order:
        for line in sessions[s]["log"]:
            by_id.setdefault(line["request_id"], []).append((s, line))
    first = lambda rid: (by_id.get(rid) or [(None, None)])[0]  # noqa: E731
    s1 = first(f"{prefix}-A1")[0]
    s2 = first(f"{prefix}-D1")[0]
    row("two intended sessions collected", f"{src.name}: sessions {', '.join(order)}", len(order) >= 2 and s1 and s2 and s1 != s2,
        "session 1 holds A1, session 2 holds D1")
    for s in order:
        sj, ev = sessions[s]["json"], sessions[s]["events"]
        guard = [e["seq"] for e in ev if e.get("ev") == "interactive.startup_check" and (e.get("data") or {}).get("session") == s]
        start = [e["seq"] for e in ev if e.get("ev") == "interactive.session.start" and (e.get("data") or {}).get("session") == s]
        row(f"startup check before serving (session {s})", f"{s}/session.json; events.jsonl seq {guard[:1]} < {start[:1]}",
            sj.get("startup_check_ok") is True and sj.get("goldens_equal") == 32 and sj.get("prompt_self_checks_passed") == 5
            and bool(guard) and bool(start) and guard[0] < start[0], "" if ev else "no event log pulled")
    # accounting of the run's inbox requests
    sent = sorted({json.loads(p.read_text(encoding="utf-8"))["request_id"] for p in (folder / "raw" / "inbox").glob("send-*.json")}) \
        if (folder / "raw" / "inbox").is_dir() else []
    problems = []
    results = {}
    for rid in sent:
        ack_p, res_p = src / "outbox" / f"{rid}.ack.json", src / "outbox" / f"{rid}.result.json"
        dups = list((src / "outbox").glob(f"{rid}.duplicate-*.json"))
        if not ack_p.is_file() or not res_p.is_file():
            problems.append(f"{rid}: acknowledgement or result missing")
            continue
        ack, res = json.loads(ack_p.read_text(encoding="utf-8")), json.loads(res_p.read_text(encoding="utf-8"))
        logged = by_id.get(rid, [])
        if dups:
            problems.append(f"{rid}: {len(dups)} duplicate notice(s)")
        if ack.get("session") != res.get("session") or res.get("session") not in sessions:
            problems.append(f"{rid}: acknowledged in {ack.get('session')}, answered in {res.get('session')} (sessions collected: {order})")
        if len(logged) != 1 or logged[0][0] != res.get("session"):
            problems.append(f"{rid}: logged {len(logged)} time(s) in {[x[0] for x in logged]}, expected once in {res.get('session')}")
        elif logged[0][1].get("outcome") != res.get("outcome"):
            problems.append(f"{rid}: the result file and its session-log line differ")
        results[rid] = res
    row("every sent inbox request: one acknowledgement, exactly one terminal result in the acknowledging session, the same in "
        "its session log", f"{len(sent)} receipts in raw/inbox; {src.name}/outbox; both outcomes.jsonl", not problems and len(sent) > 0,
        "; ".join(problems[:4]))
    stale_ids = [rid for rid in by_id if not rid.startswith("preset-") and rid not in sent]
    row("no other request (an old session's, or another run's) appears in the collected session logs", "both outcomes.jsonl",
        not stale_ids, ", ".join(stale_ids[:4]))
    # outcome rules, over every logged line of the two sessions (inbox and presets)
    bad = []
    for rid, items in by_id.items():
        for s, line in items:
            o = line["outcome"]
            if o["status"] in ("cancelled", "refused", "failed") and (o.get("target_object_id") or o.get("ask_basis")):
                bad.append(f"{rid}: {o['status']} with a usable target")
            if o.get("execution_path") == "U" and not (o.get("kept_tokens") == 0 and o.get("purpose") == "operational" and o.get("cache_mode") == "Off"):
                bad.append(f"{rid}: path U but kept {o.get('kept_tokens')}, {o.get('purpose')}, {o.get('cache_mode')}")
            if o.get("execution_path") == "P":
                bad.append(f"{rid}: prefix reuse in an operational session")
    row("cancelled and refused outcomes carry no target; every evaluated request is operational, uncached, path U, kept 0",
        "every line of both outcomes.jsonl", not bad, "; ".join(bad[:4]))
    # prompt and mapping identity of evaluated dataset commands against the goldens
    head = Path(goldens) / "headset"
    grows = _jl(head / "goldens.jsonl")
    by_cmd = {}
    for g in grows:
        by_cmd[(head / g["command_file"]).read_text(encoding="utf-8")] = g
    mism = []
    for rid, res in results.items():
        rc = json.loads(next((folder / "raw" / "inbox").glob(f"send-{rid}.json")).read_text(encoding="utf-8"))["request"]["command"]
        g = next((v for k, v in by_cmd.items() if json.loads(k) == rc), None)
        o = res["outcome"]
        if g is not None and o.get("prompt_sha256") and (o["prompt_sha256"] != g["prompt_sha256"] or o["mapping_sha256"] != g["mapping_sha256"]):
            mism.append(rid)
    row("evaluated dataset commands: prompt and mapping identities equal the goldens'", "results against goldens.jsonl", not mism, ", ".join(mism))
    # shutdown completeness per session
    for s in order:
        end, log = sessions[s]["end"], sessions[s]["log"]
        last = max((x["times"].get("outcome") or 0) for x in log) if log else 0
        row(f"shutdown after final publication (session {s})", f"{s}/session-end.json",
            end is not None and end.get("clean") is True and end.get("answered") == len(log) and end.get("written_ms", -1) >= last,
            "" if end else "no closing record")

    # the cases, each on its latest attempt
    def attempts(case, steps):
        out = []
        for k in range(1, 10):
            suffix = "" if k == 1 else f"-{k}"
            ids = [f"{prefix}-{case}{n}{suffix}" for n in steps]
            if any(i in by_id or i in sent for i in ids):
                out.append(ids)
        return out

    def outcome(rid):
        return (first(rid)[1] or {}).get("outcome") or {}

    presets1 = [x for x in sessions[s1]["log"] if x["request_id"].startswith("preset-")] if s1 else []
    a = attempts("A", [1])
    if a:
        o, p = outcome(a[-1][0]), (presets1[0]["outcome"] if presets1 else {})
        ok = o.get("status") in ("completed", "ask") and p.get("status") in ("completed", "ask") and o.get("execution_path") == "U" and p.get("execution_path") == "U"
        row("case A: startup, one inbox command and one preset complete on U", f"{a[-1][0]}; {presets1[0]['request_id'] if presets1 else 'no preset'}", ok)
    for case, steps, test, text in (
            ("B", [1, 2], lambda o1, o2: (o1.get("status") == "refused" and o1.get("reason") == "stale_result_scene_changed" and o1.get("execution_path") == "U"
                                          and o2.get("status") == "refused" and o2.get("reason") == "scene_changed_before_evaluation" and o2.get("execution_path") == "none"),
             "case B: the running request refused as stale after evaluation; the waiting one refused before evaluation"),
            ("C", [1, 2], lambda o1, o2: (o1.get("status") == "cancelled" and o1.get("reason") == "cancelled_during_evaluation"
                                          and o2.get("status") == "cancelled" and o2.get("reason") == "session_ended"),
             "case C: End session while evaluating: the running request cancelled, the waiting ones session_ended")):
        att = attempts(case, steps)
        for k, ids in enumerate(att):
            ok = test(outcome(ids[0]), outcome(ids[1]))
            if case == "C" and ok:   # its preset, answered session_ended, in the same session as this attempt
                sc = first(ids[0])[0]
                ok = any(x["request_id"].startswith("preset-") and x["outcome"].get("reason") == "session_ended"
                         for x in (sessions[sc]["log"] if sc in sessions else []))
            latest = k == len(att) - 1
            row(text + ("" if latest else f" (attempt {k + 1}, superseded)"), ", ".join(ids), ok,
                "" if ok else "the intended running or waiting condition did not occur; repeat this case with fresh IDs",
                incomplete=not ok or not latest)
    d = attempts("D", [1])
    if d:
        o = outcome(d[-1][0])
        s2log = [x["request_id"] for x in sessions[s2]["log"]] if s2 else []
        row("case D: a second session starts, D1 completes on U, and no session-1 result reaches session 2",
            f"{d[-1][0]}; {s2}/outcomes.jsonl", o.get("status") in ("completed", "ask") and o.get("execution_path") == "U"
            and s2 is not None and all(not by_id.get(r) or by_id[r][0][0] == s2 for r in s2log))
    # hashes, identity, source, operator checklist
    pj = json.loads((src / "pull.json").read_text(encoding="utf-8"))
    bad_hash = [k for k, v in pj.get("files", {}).items() if not (src / k).is_file() or _sha(src / k) != v]
    row("the pull manifest's hashes read back", f"{src.name}/pull.json", not bad_hash, ", ".join(bad_hash[:3]))
    ids = sorted(p for p in (folder / "raw" / "identity").glob("*") if (p / "manifest.json").is_file()) if (folder / "raw" / "identity").is_dir() else []
    if ids:
        man = json.loads((ids[-1] / "manifest.json").read_text(encoding="utf-8"))
        files = man.get("files", {})
        idj = json.loads((ids[-1] / "identity.json").read_text(encoding="utf-8")) if (ids[-1] / "identity.json").is_file() else {}
        okh = all((ids[-1] / k).is_file() and _sha(ids[-1] / k) == v for k, v in files.items())
        row("D102 after the installation: current identity verified, its evidence hashes read back",
            f"raw/identity/{ids[-1].name}", okh and (idj.get("status") or {}).get("current_identity") == "verified",
            "A1.8c continuity is recorded separately, under D102")
    else:
        row("D102 after the installation", "raw/identity", False, "no identity folder")
    srcid = folder / "raw" / "source"
    row("implementation identity recorded (commit and working tree)", "raw/source", srcid.is_dir() and any(srcid.iterdir()))
    uj = folder / "raw" / "operator" / "unity.json"
    u = json.loads(uj.read_text(encoding="utf-8")) if uj.is_file() else {}
    row("Unity: import and Build And Run finished with no Console errors", "raw/operator/unity.json",
        str(u.get("unity_console_zero_errors", "")).lower().startswith("y"), "an operator observation")
    ck = folder / "raw" / "operator" / "checklist.json"
    cj = json.loads(ck.read_text(encoding="utf-8")) if ck.is_file() else {}
    row("operator: the current scene's mapping shown, the old target not highlighted (cases B and D)", "raw/operator/checklist.json",
        all(str(cj.get(k, "")).lower().startswith("y") for k in ("B", "D")), "an operator observation, not a recorded measurement")
    overall = "pass" if all(r["result"] == "pass" or "superseded" in r["requirement"] for r in rows) else (
        "incomplete" if not any(r["result"] == "fail" for r in rows) else "fail")
    summary = {"format_version": 1, "record_type": "a25_correction_acceptance", "run_id": run_id, "pull": src.name, "sessions": order,
               "overall": overall, "rows": rows,
               "status_wording": "operator acceptance completed; independent review pending" if overall == "pass" else
                                 "operator acceptance not complete"}
    out = folder / "raw" / "acceptance" / _stamp()
    k = 2
    while out.exists():
        out = out.parent / f"{_stamp()}-{k}"
        k += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=str(out.parent)))
    try:
        (staging / "summary.json").write_bytes(encode_json(summary))
        lines = ["# A2.5 correction acceptance: " + run_id, "", f"Overall: **{overall}** ({summary['status_wording']}).", "",
                 "| Requirement | Evidence | Result | Limitation |", "|---|---|---|---|"]
        lines += [f"| {r['requirement']} | {r['evidence']} | {r['result']} | {r['limitation']} |" for r in rows]
        (staging / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        publish(staging, out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return dict(summary, folder=str(out))
