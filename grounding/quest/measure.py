"""A2.5 delivery 5: the measurement report (D98's measured resources; D99(3) timing and (7) memory as the runtime reports
it).

It reads a pulled session (`outbox-pull`: `session/session.json`, `outcomes.jsonl`, `startup-log.txt`) and the run's
inbox receipts, and writes `raw/measure/<UTC time>/summary.json` and `report.md`. It reports:

- per scene size and intake (`adb_inbox` or `preset`): status counts; app-observed latency (detection or button event to
  outcome, on the headset clock); queue wait; evaluation; the pipeline's stage times; tokens;
- memory as the runtime reports it: llama.cpp's buffer lines in the load log, and the wrapper's resident and peak memory
  around the load and after every answered request;
- the PC's push-and-rename times, separately. They are never combined with the headset's clock;
- descriptive only: for golden dataset commands, whether the choice equals the D2 CPU float32 baseline's. This is not an
  accuracy claim and not numerical acceptance.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
from pathlib import Path

from ..evaluation.iref_vla.protocol import encode_json
from .publish import publish
from .replay_device import _fail, _run_folder, _stamp
from .runtime_identity import REPO

BUFFER = re.compile(r"(\S+)\s+(model|KV|compute|output)\s+buffer size\s*=\s*([0-9.]+)\s*MiB")


def buffers(log_text) -> dict:
    """llama.cpp's own buffer report at load, in MiB, keyed by buffer type and kind."""
    out = {}
    for m in BUFFER.finditer(log_text):
        out[f"{m.group(1)} {m.group(2)}"] = float(m.group(3))
    return out


def stats(values) -> dict:
    v = sorted(x for x in values if isinstance(x, (int, float)) and not math.isnan(x))
    if not v:
        return {"n": 0}
    rank = lambda q: v[max(0, math.ceil(q * len(v)) - 1)]  # noqa: E731  nearest rank
    mid = len(v) // 2
    return {"n": len(v), "min": v[0], "median": v[mid] if len(v) % 2 else (v[mid - 1] + v[mid]) / 2, "p90": rank(0.9), "max": v[-1]}


def measure_report(*, run_id, goldens, references=None, pull=None, repo=None) -> dict:
    repo = Path(repo) if repo is not None else REPO
    folder = _run_folder(repo, run_id)
    pulls = sorted(p for p in (folder / "raw" / "outbox").iterdir() if p.is_dir()) if (folder / "raw" / "outbox").is_dir() else []
    src = Path(pull) if pull is not None else (pulls[-1] if pulls else None)
    if src is None or not (src / "session" / "outcomes.jsonl").is_file():
        _fail(f"no pulled session with outcomes.jsonl under {folder / 'raw' / 'outbox'}; run outbox-pull first")
    session = json.loads((src / "session" / "session.json").read_text(encoding="utf-8"))
    lines = [json.loads(x) for x in (src / "session" / "outcomes.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    log = (src / "session" / "startup-log.txt").read_text(encoding="utf-8", errors="replace") if (src / "session" / "startup-log.txt").is_file() else None
    head = Path(goldens) / "headset"
    grows = [json.loads(x) for x in (head / "goldens.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    objects = {}
    for g in grows:
        if g["snapshot_id"] not in objects:
            objects[g["snapshot_id"]] = len(json.loads((head / g["scene_file"]).read_text(encoding="utf-8"))["objects"])
    by_prompt = {g["prompt_sha256"]: g for g in grows}
    sends = {}
    for p in sorted((folder / "raw" / "inbox").glob("send-*.json")) if (folder / "raw" / "inbox").is_dir() else []:
        s = json.loads(p.read_text(encoding="utf-8"))
        sends.setdefault(s["request_id"], []).append(s)
    refs = {}
    if references is not None:
        for x in (Path(references) / "references.jsonl").read_text(encoding="utf-8").splitlines():
            if x.strip():
                r = json.loads(x)
                refs[r["request_id"]] = r["choice_code"]
    rows = []
    for r in lines:
        o, t = r["outcome"], r["times"]
        g = by_prompt.get(o.get("prompt_sha256"))
        sent = (sends.get(r["request_id"]) or [None])[0]
        sid = g["snapshot_id"] if g else (r.get("snapshot_id") or (sent["request"]["expected_scene"]["snapshot_id"] if sent else None))
        kind = g["kind"] if g else (sent["request"].get("command_kind") if sent else None)
        evaluated = o.get("execution_path") in ("U", "P")
        rows.append({"request_id": r["request_id"], "intake": t.get("intake") or o.get("source"), "status": o["status"], "reason": o.get("reason"),
                     "execution_path": o.get("execution_path"), "objects": objects.get(sid), "command_kind": kind,
                     "golden": g["request_id"] if g else None, "choice": o.get("choice_code"), "tokens": o.get("tokens"),
                     "app_observed_ms": t.get("app_observed_ms"),
                     "queue_ms": (t["inference_start"] - t["queued"]) if evaluated and t.get("inference_start") is not None and t.get("queued") is not None else None,
                     "inference_ms": (t["inference_end"] - t["inference_start"]) if evaluated and t.get("inference_end") is not None else None,
                     "stage_ms": o.get("ms") if evaluated else None, "memory_kb": r.get("memory_kb")})
    groups = {}
    for x in rows:
        scene = f"{x['objects']} objects" if x["objects"] is not None else "unknown scene"
        groups.setdefault(f"{scene}, {x['intake']}", []).append(x)
    table = {}
    for key, xs in sorted(groups.items(), key=lambda kv: (str(kv[0]))):
        ev = [x for x in xs if x["execution_path"] in ("U", "P") and x["status"] in ("completed", "ask")]
        table[key] = {"requests": len(xs), "statuses": {s: sum(x["status"] == s for x in xs) for s in sorted({x["status"] for x in xs})},
                      "answered_after_evaluation": len(ev), "app_observed_ms": stats([x["app_observed_ms"] for x in ev]),
                      "inference_ms": stats([x["inference_ms"] for x in ev]), "queue_ms": stats([x["queue_ms"] for x in ev]),
                      "tokens": stats([x["tokens"] for x in ev]),
                      "stage_ms": {k: stats([(x["stage_ms"] or {}).get(k) for x in ev]) for k in ("build", "tokenize", "eval", "score")}}
    mem = [x["memory_kb"] for x in rows if x["memory_kb"]]
    agree = [x for x in rows if x["golden"] and x["golden"] in refs and x["status"] in ("completed", "ask") and x["choice"]]
    summary = {"format_version": 1, "record_type": "a25_measure_summary", "run_id": run_id, "pull": src.name, "session": session.get("session"),
               "session_settings": {k: session.get(k) for k in ("purpose", "cache_mode", "execution_path", "context_tokens", "flags", "llama_cpp", "load_ms")},
               "clock": "headset session stopwatch (ms); the PC's push times are reported separately and never combined",
               "groups": table, "rows": rows,
               "memory": {"runtime_buffers_mib": buffers(log) if log is not None else None, "around_load_kb": session.get("memory_kb"),
                          "after_requests_rss_kb": stats([m.get("rss") for m in mem]), "after_requests_peak_kb": stats([m.get("peak") for m in mem])},
               "pc_push_and_rename_s": stats([s["pc_push_and_rename_s"] for v in sends.values() for s in v]),
               "descriptive_baseline_agreement": {"compared": len(agree), "same_choice": sum(refs[x["golden"]] == x["choice"] for x in agree),
                                                  "note": "golden dataset commands only; against the D2 CPU float32/eager baseline; descriptive, "
                                                          "not an accuracy claim or numerical acceptance"} if references is not None else None}
    out = folder / "raw" / "measure" / _stamp()
    if out.exists():
        _fail(f"{out} already exists (nothing is overwritten)")
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=str(out.parent)))
    try:
        (staging / "summary.json").write_bytes(encode_json(summary))
        (staging / "report.md").write_text(render(summary), encoding="utf-8")
        publish(staging, out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return dict(summary, folder=str(out))


def render(s) -> str:
    f = lambda st, k: "-" if not st.get("n") else f"{st[k] / 1000:.1f}"  # noqa: E731
    L = [f"# A2.5 measurement: run {s['run_id']}, session {s['session']}", "",
         f"Settings: {s['session_settings']}. Times on the {s['clock']}.", "",
         "| Scene, intake | Requests | Statuses | Evaluated | App-observed s (median / p90 / max) | Inference s (median) | Queue s (median) | Tokens (median) |",
         "|---|---|---|---|---|---|---|---|"]
    for k, g in s["groups"].items():
        a = g["app_observed_ms"]
        L.append(f"| {k} | {g['requests']} | {g['statuses']} | {g['answered_after_evaluation']} | {f(a, 'median')} / {f(a, 'p90')} / {f(a, 'max')} | "
                 f"{f(g['inference_ms'], 'median')} | {f(g['queue_ms'], 'median')} | {g['tokens'].get('median', '-')} |")
    m = s["memory"]
    L += ["", f"Memory as the runtime reports it: buffers (MiB) {m['runtime_buffers_mib']}; around the load (kB) {m['around_load_kb']}; "
              f"after requests, resident (kB) {m['after_requests_rss_kb']} and peak (kB) {m['after_requests_peak_kb']}.", "",
          f"PC push and rename (s, PC clock only): {s['pc_push_and_rename_s']}."]
    if s["descriptive_baseline_agreement"]:
        d = s["descriptive_baseline_agreement"]
        L += ["", f"Descriptive: {d['same_choice']} of {d['compared']} golden dataset commands chose as the CPU baseline did ({d['note']})."]
    return "\n".join(L) + "\n"
