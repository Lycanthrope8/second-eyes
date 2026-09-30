#!/usr/bin/env python3
"""Summarize and compare what tools/profile.py recorded.

    python analysis/profile.py summary 20261001_A1_r007
    python analysis/profile.py compare 20261001_A1_r007 20261001_A1_r008

A run's recording is runs/<run ID>/raw/sampler.jsonl (thermal, battery and processes every 30 s)
plus the OVR Metrics CSV(s) pulled next to it (frames, load and memory every second).
Only CSV rows inside the recording window count. Procedure and limits: docs/profiling.md.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
APP = "com.secondeyes.quest"
METRICS_TOOL = "com.oculus.ovrmonitormetricsservice"
JUMP_MB = 500     # app memory rising this much from one second to the next is flagged
GAP_S = 3.0       # seconds without an OVR Metrics row are flagged (app paused, headset off)
LIMITS = {"fps_mean": ("fps", 0.5), "app_mem_max_mb": ("%", 5.0), "cpu_mean": ("points", 3.0), "gpu_mean": ("points", 3.0)}


class ToolError(Exception):
    """A problem the user can fix. Printed without a traceback."""


# ---------------------------------------------------------------- reading

def read_sampler(raw: Path) -> tuple:
    path = raw / "sampler.jsonl"
    if not path.is_file():
        raise ToolError(f"No sampler.jsonl in {raw}. Record first with: python tools/profile.py record ...")
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ToolError(f"{path} is empty.")
    return lines[0], lines[1:]


def csv_start_epoch(path: Path, device_tz: str):
    """OVR Metrics names each CSV after the session's start in the headset's local time."""
    match = re.search(r"-(\d{8})_(\d{6})\.csv$", path.name)
    tz = re.fullmatch(r"([+-])(\d{2})(\d{2})", device_tz or "")
    if not match or not tz:
        return None
    offset = dt.timedelta(hours=int(tz.group(2)), minutes=int(tz.group(3)))
    if tz.group(1) == "-":
        offset = -offset
    local = dt.datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
    return local.replace(tzinfo=dt.timezone(offset)).timestamp()


def read_metrics(raw: Path, header: dict, samples: list) -> tuple:
    """OVR Metrics rows inside the recording window, oldest first, plus the files they came from."""
    start = header["device_start_s"]
    end = start + (samples[-1]["elapsed_s"] if samples else 0) + 1
    rows, used = [], []
    for path in sorted(raw.glob("*.csv")):
        base = csv_start_epoch(path, header.get("device_tz", ""))
        if base is None:
            continue
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            kept = 0
            for row in csv.DictReader(f):
                try:
                    t = base + float(row["Time Stamp"]) / 1000.0
                except (KeyError, ValueError):
                    continue
                if start <= t <= end:
                    row["_t"] = t
                    rows.append(row)
                    kept += 1
        if kept:
            used.append(path.name)
    rows.sort(key=lambda r: r["_t"])
    return rows, used


def number(row: dict, name: str):
    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError):
        return None


def column(rows: list, name: str) -> list:
    return [v for v in (number(r, name) for r in rows) if v is not None]


def parse_temps(thermal: str) -> dict:
    """Sensor name -> degrees C, from the fresh 'Current temperatures from HAL' section when present."""
    section = thermal
    start = thermal.find("Current temperatures from HAL:")
    if start >= 0:
        rest = thermal[start:].split("\n", 1)[-1]
        end = re.search(r"^\S.*:\s*$", rest, re.M)   # the next section heading
        section = rest[:end.start()] if end else rest
    return {name: float(value) for value, name in
            re.findall(r"Temperature\{mValue=([-\d.]+), mType=\d+, mName=([\w.-]+)", section)}


def parse_top(top: str) -> dict:
    """Process name -> %CPU of one core, from one 'top -b -n 1' listing."""
    cpu, in_table = {}, False
    for line in top.splitlines():
        if line.strip().startswith("PID USER"):
            in_table = True
            continue
        if not in_table:
            continue
        parts = line.split(None, 11)
        if len(parts) < 12:
            continue
        try:
            value = float(parts[8])
        except ValueError:
            continue
        name = parts[11].split()[0]
        if name == "top":
            continue  # the sampler's own top command
        cpu[name] = cpu.get(name, 0.0) + value
    return cpu


# ---------------------------------------------------------------- summarizing

def prompt_passes(raw: Path) -> list:
    """The run's sends from its event logs (not raw/before_run/): (request, Send, first token, end) in UTC seconds."""
    try:
        from eventlog import read_events
    except ImportError:
        return []
    sends = {}
    for log in sorted(raw.glob("*.jsonl")):
        if log.name == "sampler.jsonl":
            continue
        for e in read_events(log):
            d = e.get("data", {})
            if e.get("ev") in ("model.request", "model.token", "model.generate") and isinstance(d.get("request"), int):
                x = sends.setdefault((log.name, d["request"]), {"n": d["request"]})
                t = e.get("utc_us", 0) / 1e6
                if e["ev"] == "model.request":
                    x["send"] = t
                elif e["ev"] == "model.token":
                    x.setdefault("first", t)
                else:
                    x["end"] = t
    return [(x["n"], x["send"], x.get("first"), x.get("end")) for _, x in sorted(sends.items()) if "send" in x]


def frameless(before: dict, after: dict) -> int:
    """Seconds without app frames hidden in the interval between two rows (A1.7d): a gap over GAP_S, or a missing row
    followed by a second of mostly stale frames. A lone missing row with normal frames around it (2 s, 72 fps, no stale
    frames: seen in every run) is OVR Metrics skipping a row, not the app freezing."""
    gap = after["_t"] - before["_t"]
    if gap <= 1.5:
        return 0
    if gap > GAP_S or (number(after, "stale_frame_count") or 0) >= 30:
        return max(0, int(gap + 0.5) - 1)
    return 0


def phase_fps(rows: list, intervals: list):
    """Mean frame rate over the given time intervals, each row standing for the second before it and a gap in the
    rows for seconds without frames (A1.7d: the app freezes, and OVR Metrics writes nothing)."""
    frames = seconds = 0.0
    for before, after in zip(rows, rows[1:]):
        t0, t1, fps = before["_t"], after["_t"], number(after, "average_frame_rate") or 0.0
        pieces = [(max(t0, t1 - 1.0), t1, fps)] + ([(t0, t1 - 1.0, 0.0)] if frameless(before, after) else [])
        for a, b, rate in pieces:
            for x, y in intervals:
                overlap = min(b, y) - max(a, x)
                if overlap > 0:
                    frames += rate * overlap
                    seconds += overlap
    return frames / seconds if seconds else None


def summarize(run_id: str) -> dict:
    raw = RUNS / run_id / "raw"
    if not (RUNS / run_id).is_dir():
        raise ToolError(f"No run {run_id} in runs/.")
    header, samples = read_sampler(raw)
    rows, used = read_metrics(raw, header, samples)
    s = {"run_id": run_id, "samples": len(samples), "minutes": (samples[-1]["elapsed_s"] / 60 if samples else 0),
         "rows": len(rows), "csvs": used, "flags": []}

    if rows:
        fps, mem, gpu_mem = column(rows, "average_frame_rate"), column(rows, "app_pss_MB"), column(rows, "app_gpu_physical_MB")
        cpu, gpu = column(rows, "cpu_utilization_percentage"), column(rows, "gpu_utilization_percentage")
        # Seconds without a row count as seconds without frames: while the model's pass blocks the app, OVR Metrics
        # writes no row at all, so averaging only the rows would hide the worst seconds (A1.7d).
        missing = sum(frameless(a, b) for a, b in zip(rows, rows[1:]))
        s.update(fps_mean=sum(fps) / (len(fps) + missing), fps_mean_rows=statistics.mean(fps), missing_s=missing,
                 fps_min=0 if missing else min(fps), below_71_s=sum(1 for v in fps if v < 71) + missing,
                 stale=sum(column(rows, "stale_frame_count")),
                 cpu_mean=statistics.mean(cpu), cpu_max=max(cpu), gpu_mean=statistics.mean(gpu), gpu_max=max(gpu),
                 cpu_levels=sorted(set(int(v) for v in column(rows, "cpu_level"))),
                 gpu_levels=sorted(set(int(v) for v in column(rows, "gpu_level"))),
                 app_mem_start_mb=mem[0], app_mem_end_mb=mem[-1], app_mem_max_mb=max(mem),
                 gpu_mem_start_mb=gpu_mem[0] if gpu_mem else None, gpu_mem_max_mb=max(gpu_mem) if gpu_mem else None,
                 free_min_mb=min(column(rows, "available_memory_MB") or [0]),
                 power_mean_w=statistics.mean(column(rows, "power_wattage") or [0]) / 1000,
                 app_gpu_ms=statistics.mean(column(rows, "app_gpu_time_microseconds") or [0]) / 1000)
        sends = prompt_passes(raw)
        in_pass = []
        for before, after in zip(rows, rows[1:]):
            m0, m1 = number(before, "app_pss_MB"), number(after, "app_pss_MB")
            gap = after["_t"] - before["_t"]
            when = before["_t"] - header["device_start_s"]
            if gap > GAP_S:
                if any(t_send - 1.5 <= before["_t"] <= (first or end or t_send) + 0.5 for _, t_send, first, end in sends):
                    in_pass.append(gap)   # the model's prompt pass blocked the app: expected, reported below
                else:
                    s["flags"].append(f"no OVR Metrics rows for {gap:.0f} s at {when:.0f} s (app paused or headset off?)")
            if m0 is not None and m1 is not None and m1 - m0 > JUMP_MB:
                s["flags"].append(f"app memory jumped {m1 - m0:+.0f} MB at {when:.0f} s")
        if len(used) > 1:
            s["flags"].append(f"{len(used)} OVR Metrics sessions in one recording: the app restarted")
        s["pass_gaps"] = in_pass
        done = [(n, a, f, e) for n, a, f, e in sends if f and e and rows[0]["_t"] <= a and e <= rows[-1]["_t"]]
        if done:
            follow = [(e, nxt[1]) for (_, _, _, e), nxt in zip(done, done[1:])]
            s["model"] = {"sends": len(done), "prompt_pass": phase_fps(rows, [(a, f) for _, a, f, _ in done]),
                          "answer": phase_fps(rows, [(f, e) for _, _, f, e in done]), "pause": phase_fps(rows, follow)}
    else:
        s["flags"].append("no OVR Metrics rows inside the recording window (was its CSV recording on?)")

    temps = [parse_temps(x["thermal"]) for x in samples]
    statuses = [int(m.group(1)) for m in (re.search(r"Thermal Status: (\d+)", x["thermal"]) for x in samples) if m]
    soc = [t["soc-usr"] for t in temps if "soc-usr" in t]
    batt = [t["battery"] for t in temps if "battery" in t]
    s.update(thermal_max=max(statuses) if statuses else None,
             soc_start=soc[0] if soc else None, soc_end=soc[-1] if soc else None, soc_max=max(soc) if soc else None,
             batt_start=batt[0] if batt else None, batt_end=batt[-1] if batt else None)
    last = samples[-1]["battery"] if samples else ""
    level = re.search(r"level: (\d+)", last)
    ac = re.search(r"AC powered: (\w+)", last)
    usb = re.search(r"USB powered: (\w+)", last)
    s["charging"] = "plugged in" if (ac and ac.group(1) == "true") or (usb and usb.group(1) == "true") else "on battery"
    s["battery_level"] = int(level.group(1)) if level else None

    totals = {}
    for x in samples:
        for name, value in parse_top(x["top"]).items():
            totals[name] = totals.get(name, 0.0) + value
    averages = {name: total / max(1, len(samples)) for name, total in totals.items()}
    s["app_cpu_core"] = averages.get(APP, 0.0)
    s["tool_cpu_core"] = averages.get(METRICS_TOOL, 0.0)
    others = sorted(((v, n) for n, v in averages.items() if n not in (APP, METRICS_TOOL) and v >= 0.5), reverse=True)
    s["others"] = [(n, v) for v, n in others[:5]]
    if any(not x["app_running"] for x in samples):
        s["flags"].append("the app was not running for part of the recording")
    return s


def fmt(value, pattern="{:.0f}", missing="?"):
    return missing if value is None else pattern.format(value)


def print_summary(s: dict) -> None:
    print(f"Run {s['run_id']}: {s['minutes']:.1f} min recorded, {s['samples']} samples, "
          f"{s['rows']} OVR Metrics rows ({', '.join(s['csvs']) or 'none'})")
    if s["rows"]:
        print(f"  frames   fps mean {s['fps_mean']:.1f} (min {s['fps_min']:.0f}), stale frames {s['stale']:.0f}, "
              f"seconds below 71 fps {s['below_71_s']}" + (f", of them {s['missing_s']} without a frame (no row); "
              f"{s['fps_mean_rows']:.1f} fps over the rows alone" if s["missing_s"] else ""))
        if s.get("model"):
            m = s["model"]
            print(f"  model    {m['sends']} finished sends in the recording; fps mean in their prompt passes "
                  f"{fmt(m['prompt_pass'], '{:.1f}')}, answers {fmt(m['answer'], '{:.1f}')}, pauses {fmt(m['pause'], '{:.1f}')}"
                  + (f"; {len(s['pass_gaps'])} freeze(s) of {min(s['pass_gaps']):.0f}-{max(s['pass_gaps']):.0f} s inside "
                     "prompt passes" if s.get("pass_gaps") else ""))
        print(f"  load     CPU mean {s['cpu_mean']:.0f}% (max {s['cpu_max']:.0f}), GPU mean {s['gpu_mean']:.0f}% "
              f"(max {s['gpu_max']:.0f}), app GPU time {s['app_gpu_ms']:.1f} ms/frame, "
              f"levels CPU {s['cpu_levels']} GPU {s['gpu_levels']}")
        print(f"  memory   app {s['app_mem_start_mb']:.0f} -> {s['app_mem_end_mb']:.0f} MB (max {s['app_mem_max_mb']:.0f}), "
              f"of it graphics {fmt(s['gpu_mem_start_mb'])} -> max {fmt(s['gpu_mem_max_mb'])} MB, "
              f"headset free min {s['free_min_mb']:.0f} MB")
    print(f"  heat     SoC {fmt(s['soc_start'], '{:.1f}')} -> {fmt(s['soc_end'], '{:.1f}')} C "
          f"(max {fmt(s['soc_max'], '{:.1f}')}), battery {fmt(s['batt_start'], '{:.1f}')} -> {fmt(s['batt_end'], '{:.1f}')} C, "
          f"thermal status max {fmt(s['thermal_max'])}" + (f", power mean {s['power_mean_w']:.1f} W" if s["rows"] else ""))
    print(f"  battery  {s['charging']}, level {fmt(s['battery_level'])}%")
    others = ", ".join(f"{n} {v:.0f}%" for n, v in s["others"]) or "none"
    print(f"  CPU of one core: app {s['app_cpu_core']:.0f}%, OVR Metrics (measuring tool) {s['tool_cpu_core']:.0f}%; "
          f"top others: {others}")
    print("  flags    " + ("; ".join(s["flags"]) if s["flags"] else "none"))


def cmd_summary(args) -> int:
    for index, run_id in enumerate(args.run_ids):
        if index:
            print()
        print_summary(summarize(run_id))
    return 0


def cmd_compare(args) -> int:
    a, b = summarize(args.run_a), summarize(args.run_b)
    names = {"fps_mean": "fps mean", "app_mem_max_mb": "app memory max (MB)", "cpu_mean": "CPU mean (%)", "gpu_mean": "GPU mean (%)"}
    print(f"{'metric':22s} {args.run_a:>18s} {args.run_b:>18s} {'difference':>12s} {'limit':>10s}  agree")
    outside = []
    for key, (unit, limit) in LIMITS.items():
        va, vb = a.get(key), b.get(key)
        if va is None or vb is None:
            print(f"{names[key]:22s} {fmt(va, '{:.1f}'):>18s} {fmt(vb, '{:.1f}'):>18s} {'?':>12s} {limit:>8g} {unit:<2s}  ?")
            outside.append(names[key])
            continue
        diff = abs(va - vb) / max(va, vb) * 100 if unit == "%" else abs(va - vb)
        ok = diff <= limit
        if not ok:
            outside.append(names[key])
        print(f"{names[key]:22s} {va:18.1f} {vb:18.1f} {diff:12.1f} {limit:>8g} {unit:<2s}  {'yes' if ok else 'NO'}")
    for label, s in (("A", a), ("B", b)):
        if s["flags"]:
            print(f"flags in {label} ({s['run_id']}): " + "; ".join(s["flags"]))
    if outside:
        print("Not within the limits: " + ", ".join(outside) + ".")
        return 1
    print("The two runs agree within the limits.")
    return 0


def purpose_of(run_id: str) -> str:
    """The run's purpose from its config.yaml, for labeling a table row."""
    try:
        text = (RUNS / run_id / "config.yaml").read_text(encoding="utf-8")
    except OSError:
        return ""
    try:
        import yaml
        value = (yaml.safe_load(text) or {}).get("purpose")
    except Exception:   # no PyYAML, or a file it can't read: the quoted value, or the text before a comment
        m = re.search(r'^purpose:[ \t]*(?:"((?:[^"\\]|\\.)*)"|([^#\r\n]*))', text, re.M)
        value = (m.group(1) if m.group(1) is not None else m.group(2).strip()) if m else ""
    return str(value or "").replace("|", "/")


def cmd_table(args) -> int:
    """One Markdown row per run, for phase notes (A1.7d, D50); flags are counted here and listed by summary."""
    print("| Run | Purpose | Minutes | fps mean (min) | Seconds below 71 fps | Stale frames | CPU % mean (max) | "
          "GPU % mean (max) | App memory max, MB (graphics) | SoC °C start → max | Thermal status max | Flags |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for run_id in args.run_ids:
        s = summarize(run_id)
        if s["rows"]:
            frames, below, stale = f"{s['fps_mean']:.1f} ({s['fps_min']:.0f})", str(s["below_71_s"]), f"{s['stale']:.0f}"
            cpu, gpu = f"{s['cpu_mean']:.0f} ({s['cpu_max']:.0f})", f"{s['gpu_mean']:.0f} ({s['gpu_max']:.0f})"
            mem = f"{s['app_mem_max_mb']:.0f} ({fmt(s['gpu_mem_max_mb'])})"
        else:
            frames = below = stale = cpu = gpu = mem = "?"
        heat = f"{fmt(s['soc_start'], '{:.1f}')} → {fmt(s['soc_max'], '{:.1f}')}"
        print(f"| {run_id} | {purpose_of(run_id)} | {s['minutes']:.1f} | {frames} | {below} | {stale} | {cpu} | {gpu} | "
              f"{mem} | {heat} | {fmt(s['thermal_max'])} | {len(s['flags']) or 'none'} |")
    return 0


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except AttributeError:
            pass
    parser = argparse.ArgumentParser(prog="python analysis/profile.py",
                                     description="Summarize and compare recordings. See docs/profiling.md")
    sub = parser.add_subparsers(dest="command", required=True)
    summary = sub.add_parser("summary", help="summarize one or more recorded runs")
    summary.add_argument("run_ids", nargs="+", metavar="RUN_ID")
    table = sub.add_parser("table", help="one Markdown table row per run, for phase notes")
    table.add_argument("run_ids", nargs="+", metavar="RUN_ID")
    compare = sub.add_parser("compare", help="check two runs against the agreement limits")
    compare.add_argument("run_a", metavar="RUN_A")
    compare.add_argument("run_b", metavar="RUN_B")
    args = parser.parse_args(argv)
    try:
        return {"summary": cmd_summary, "table": cmd_table, "compare": cmd_compare}[args.command](args)
    except ToolError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
