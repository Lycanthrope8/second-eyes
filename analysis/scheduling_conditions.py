"""A1.10d: the detector against the language model, one recorded run per condition (D88, D90).

    python analysis/scheduling_conditions.py <run ID or raw folder> [more runs] [--json]

For each run (the language model alone; the model with detection at 1 Hz deliberately overlapping commands; or
sequential cycles of scan, release and commands) it reports:
- commands: the fixed schedule's dispatches (command.dispatch), skipped ones with their reasons, and for each sent
  command the time from dispatch to its answer (model.generate) and to its scored answer (model.scores), the panel's
  current endpoint; partial runtime evidence for Gate A, not end-of-speech latency (no speech yet). Commands whose
  interval overlaps a detector inference are reported apart, as is the first command after each release;
- stale frames (OVR Metrics, the clock calibrated as in analysis/detector_phases.py, judged at the least favourable
  offset) over the condition, over buckets overlapping commands and over buckets overlapping inference, and in
  sequential mode over the scans and the command windows; the bucket figures are a proxy (O26);
- residency: when the language model loaded, and each interval the detector was loaded;
- release: time, whether anything was in flight, results after release (must be 0), and per cycle the app memory
  before the load, at the scan's end and after the release, so retained allocations show across cycles;
- memory: peak app PSS and minimum headset available memory against D88's margins.
Answer correctness against the PC reference stays grounding/check_headset.py's job. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import detector_phases as DP  # noqa: E402
import profile as P  # noqa: E402


def pct(v, q):
    return DP.pct(v, q)


def commands(events) -> list:
    """Each dispatch with its request, answer and scores (matched by time, then request number)."""
    out, used = [], set()
    reqs = [e for e in events if e["ev"] == "model.request"]
    gens = {e["data"]["request"]: e for e in events if e["ev"] == "model.generate"}
    scores = {e["data"]["request"]: e for e in events if e["ev"] == "model.scores"}
    for e in events:
        if e["ev"] != "command.dispatch":
            continue
        d = e["data"]
        row = {"t": e["utc_us"] / 1e6, "seq": d["seq"], "prompt_id": d["prompt_id"], "dispatched": d["dispatched"],
               "reason": d["reason"], "request": None, "answer_s": None, "goal_s": None, "stopped": None}
        if d["dispatched"]:
            r = next((x for x in reqs if x["utc_us"] >= e["utc_us"] and x["data"]["request"] not in used
                      and x["data"]["prompt_id"] == d["prompt_id"]), None)
            if r is not None:
                n = r["data"]["request"]
                used.add(n)
                row["request"] = n
                if n in gens:
                    row["answer_s"] = gens[n]["utc_us"] / 1e6 - row["t"]
                    row["stopped"] = gens[n]["data"]["stopped"]
                if n in scores:
                    row["goal_s"] = scores[n]["utc_us"] / 1e6 - row["t"]
        out.append(row)
    return out


def inference_intervals(events) -> list:
    return [((e["data"].get("started_utc_us") or e["utc_us"]) / 1e6, e["utc_us"] / 1e6) for e in events
            if e["ev"] == "detector.result" and e["data"].get("path") == "camera"]


def windows(events, ev, stage_a, stage_b, key="cycle") -> list:
    out, open_ = [], {}
    for e in events:
        if e["ev"] != ev:
            continue
        d = e["data"]
        if d.get("stage") == stage_a:
            open_[d[key]] = e["utc_us"] / 1e6
        elif d.get("stage") == stage_b and d[key] in open_:
            out.append((d[key], open_.pop(d[key]), e["utc_us"] / 1e6))
    return out


def worst_rate(rows, offsets, intervals, refresh, any_overlap) -> dict:
    rates = [DP.union_rate(rows, o, intervals, refresh, any_overlap) for o in offsets]
    mid = rates[len(rates) // 2]
    valid = [r["rate"] for r in rates if r["rate"] is not None]
    mid["range"] = (min(valid), max(valid)) if valid else None
    return mid


def mem_at(rows, offset, t0, t1, column="app_pss_MB"):
    vals = [P.number(r, column) for r in rows if t0 <= r["_t"] + offset <= t1]
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def analyze(raw: Path, refresh: float = 72.0, limit_gib: float = 5.75) -> dict:
    header, samples = P.read_sampler(raw)
    rows, used = P.read_metrics(raw, header, samples)
    events = DP.read_events(raw)
    cal = DP.calibrate(rows, events)
    offs = DP.offsets_of(cal)
    mid = offs[len(offs) // 2]
    cmds = commands(events)
    sent = [c for c in cmds if c["dispatched"]]
    starts = [e["utc_us"] / 1e6 for e in events if e["ev"] == "command.schedule" and e["data"]["state"] == "start"]
    stops = [e["utc_us"] / 1e6 for e in events if e["ev"] == "command.schedule" and e["data"]["state"] == "stop"]
    infer = inference_intervals(events)
    cmd_iv = [(c["t"], c["t"] + c["goal_s"]) for c in sent if c["goal_s"] is not None]
    overl = [c for c in sent if c["goal_s"] is not None and any(a < c["t"] + c["goal_s"] and b > c["t"] for a, b in infer)]
    alone = [c for c in sent if c["goal_s"] is not None and c not in overl]
    first = events[0]["utc_us"] / 1e6
    t0 = min(starts + [x for x, _ in infer] or [first])
    t1 = max(stops + [y for _, y in infer] + [c["t"] + (c["goal_s"] or 0) for c in sent] or [events[-1]["utc_us"] / 1e6])
    cycles = windows(events, "detector.cycle", "scan_start", "scan_end")
    cmdwin = windows(events, "detector.cycle", "commands_start", "commands_end")
    released = {e["data"]["cycle"]: e["utc_us"] / 1e6 for e in events if e["ev"] == "detector.cycle" and e["data"]["stage"] == "released"}
    loads = [e["utc_us"] / 1e6 for e in events if e["ev"] == "detector.load"]
    unloads = [(e["utc_us"] / 1e6, e["data"]) for e in events if e["ev"] == "detector.unload"]
    per_cycle = []
    for c, a, b in cycles:
        load_t = max([x for x in loads if x <= a] or [a])
        rel = released.get(c)
        first_after = next((x for x in sent if rel is not None and x["t"] >= rel), None)
        per_cycle.append({"cycle": c, "scan_s": b - a,
                          "mem_before_load_mb": mem_at(rows, mid, load_t - 12, load_t - 2),
                          "mem_scan_end_mb": mem_at(rows, mid, b - 5, b),
                          "mem_after_release_mb": mem_at(rows, mid, rel + 3, rel + 13) if rel else None,
                          "first_command_goal_s": first_after["goal_s"] if first_after else None})
    after_release = 0
    for t, _ in unloads:
        nxt = min([x for x in loads if x > t] or [float("inf")])
        after_release += sum(1 for e in events if e["ev"] == "detector.result" and t < e["utc_us"] / 1e6 < nxt)
    llm = [e for e in events if e["ev"] == "model.load"]
    goal = [c["goal_s"] * 1000 for c in sent if c["goal_s"] is not None]
    mem = [r for r in rows if t0 <= r["_t"] + mid <= t1]
    return {"raw": str(raw), "calibration": cal, "window": (t0 - first, t1 - first),
            "schedule_events": sum(1 for e in events if e["ev"] == "command.schedule"),
            "commands": {"dispatched": len(sent), "skipped": len(cmds) - len(sent),
                         "skip_reasons": sorted({c["reason"] for c in cmds if not c["dispatched"]}),
                         "unfinished": sum(1 for c in sent if c["goal_s"] is None),
                         "goal_ms": {"median": pct(goal, .5), "p95": pct(goal, .95), "max": max(goal) if goal else None},
                         "goal_ms_overlapping": [round(c["goal_s"] * 1000) for c in overl],
                         "goal_ms_alone_median": pct([c["goal_s"] * 1000 for c in alone], .5),
                         "answer_ms_median": pct([c["answer_s"] * 1000 for c in sent if c["answer_s"] is not None], .5)},
            "stale_condition": worst_rate(rows, offs, [(t0, t1)], refresh, False),
            "stale_command_buckets": worst_rate(rows, offs, cmd_iv, refresh, True) if cmd_iv else None,
            "stale_inference_buckets": worst_rate(rows, offs, infer, refresh, True) if infer else None,
            "stale_scans": worst_rate(rows, offs, [(a, b) for _, a, b in cycles], refresh, False) if cycles else None,
            "stale_command_windows": worst_rate(rows, offs, [(a, b) for _, a, b in cmdwin], refresh, False) if cmdwin else None,
            "residency": {"language_model_loaded_s": [round(e["utc_us"] / 1e6 - first, 1) for e in llm],
                          "detector_loaded_s": [(round(a - first, 1), round(b - first, 1)) for a, b in
                                                zip(loads, [t for t, _ in unloads] + [None] * len(loads)) if b is not None]},
            "releases": [{"release_ms": d.get("release_ms"), "in_flight": d.get("in_flight")} for _, d in unloads],
            "results_after_release": after_release, "cycles": per_cycle,
            "keyframes": {"completed": sum(1 for e in events if e["ev"] == "detector.result" and e["data"].get("path") == "camera" and e["data"]["completed"]),
                          "rejected": sum(1 for e in events if e["ev"] == "detector.reject")},
            "memory": {"app_pss_max_mb": max(P.column(mem, "app_pss_MB"), default=None),
                       "available_min_mb": min(P.column(mem, "available_memory_MB"), default=None),
                       "limit_mb": limit_gib * 1024}}


def show(rep: dict) -> None:
    c, cal = rep["commands"], rep["calibration"]
    f = DP.fmt
    print(f"== {rep['raw']}: condition window {rep['window'][0]:.0f} to {rep['window'][1]:.0f} s after the log's start; "
          f"clock offset {cal['offset_s']:+.2f} s ({cal['method'] if cal['calibrated'] else 'not calibrated'})")
    print(f"  commands: {c['dispatched']} sent, {c['skipped']} skipped {c['skip_reasons'] or ''}, {c['unfinished']} without "
          f"scores; dispatch to scored answer median {f(c['goal_ms']['median'])} ms, p95 {f(c['goal_ms']['p95'])}, max "
          f"{f(c['goal_ms']['max'])} (to the answer {f(c['answer_ms_median'])} ms median); typed presets, partial "
          "runtime evidence, not end-of-speech")
    if not c["dispatched"] and not rep["schedule_events"]:
        print("  WARNING: no command.schedule event in this log: the command schedule was never started")
    if c["goal_ms_overlapping"]:
        print(f"  commands overlapping an inference: {len(c['goal_ms_overlapping'])}, {c['goal_ms_overlapping'][:12]} ms; "
              f"the others' median {f(c['goal_ms_alone_median'])} ms")
    for name, key in (("condition", "stale_condition"), ("buckets overlapping commands", "stale_command_buckets"),
                      ("buckets overlapping inference", "stale_inference_buckets"), ("scans", "stale_scans"),
                      ("command windows", "stale_command_windows")):
        if rep.get(key):
            print(f"  stale frames over the {name}: {DP.rate(rep[key])}")
    r = rep["residency"]
    print(f"  residency: language model loaded at {r['language_model_loaded_s']} s; detector loaded {r['detector_loaded_s']} s")
    if rep["releases"]:
        print(f"  releases: {[(x['release_ms'], x['in_flight']) for x in rep['releases']]} (ms, in flight); detector "
              f"results after a release: {rep['results_after_release']}; keyframes completed "
              f"{rep['keyframes']['completed']}, rejected {rep['keyframes']['rejected']}")
    for cy in rep["cycles"]:
        print(f"  cycle {cy['cycle']}: scan {cy['scan_s']:.0f} s; app PSS before load {f(cy['mem_before_load_mb'])} MB, at "
              f"scan end {f(cy['mem_scan_end_mb'])}, after release {f(cy['mem_after_release_mb'])}; first command after "
              f"release {f(None if cy['first_command_goal_s'] is None else cy['first_command_goal_s'] * 1000)} ms")
    m = rep["memory"]
    worst = DP.worst(rep["stale_condition"])
    print("  acceptance (D88):")
    print(f"    stale frames < 1% over the condition: {DP.verdict(worst is not None and worst < 0.01)} ({DP.rate(rep['stale_condition'])})")
    if rep.get("stale_inference_buckets"):
        w = DP.worst(rep["stale_inference_buckets"])
        print(f"    stale frames < 1% over buckets overlapping inference (proxy, O26): {DP.verdict(w is not None and w < 0.01)}")
    print(f"    peak app memory >= 1 GiB below the {m['limit_mb']:.0f} MB limit: "
          f"{DP.verdict(m['app_pss_max_mb'] is not None and m['app_pss_max_mb'] <= m['limit_mb'] - 1024)} ({f(m['app_pss_max_mb'])} MB)")
    print(f"    headset available memory >= 1 GiB: {DP.verdict(m['available_min_mb'] is not None and m['available_min_mb'] >= 1024)} "
          f"({f(m['available_min_mb'])} MB)")
    sent_ok = c["dispatched"] > 0 and c["skipped"] == 0 and c["unfinished"] == 0
    print(f"    every command sent and finished: {DP.verdict(sent_ok)}"
          + ("" if c["dispatched"] else " (no command was sent: the schedule never started; check its wiring)"))
    if rep["releases"]:
        print(f"    no detector result after a release, nothing in flight: "
              f"{DP.verdict(rep['results_after_release'] == 0 and not any(x['in_flight'] for x in rep['releases']))}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("targets", nargs="+", help="run IDs or raw folders, one per condition")
    ap.add_argument("--refresh-hz", type=float, default=72.0)
    ap.add_argument("--limit-gib", type=float, default=5.75)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    for t in a.targets:
        rep = analyze(DP.raw_folder(t), a.refresh_hz, a.limit_gib)
        print(json.dumps(rep, indent=1, default=str)) if a.json else show(rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
