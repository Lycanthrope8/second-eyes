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
        s.update(fps_mean=statistics.mean(fps), fps_min=min(fps), below_71_s=sum(1 for v in fps if v < 71),
                 stale=sum(column(rows, "stale_frame_count")),
                 cpu_mean=statistics.mean(cpu), cpu_max=max(cpu), gpu_mean=statistics.mean(gpu), gpu_max=max(gpu),
                 cpu_levels=sorted(set(int(v) for v in column(rows, "cpu_level"))),
                 gpu_levels=sorted(set(int(v) for v in column(rows, "gpu_level"))),
                 app_mem_start_mb=mem[0], app_mem_end_mb=mem[-1], app_mem_max_mb=max(mem),
                 gpu_mem_start_mb=gpu_mem[0] if gpu_mem else None, gpu_mem_max_mb=max(gpu_mem) if gpu_mem else None,
                 free_min_mb=min(column(rows, "available_memory_MB") or [0]),
                 power_mean_w=statistics.mean(column(rows, "power_wattage") or [0]) / 1000,
                 app_gpu_ms=statistics.mean(column(rows, "app_gpu_time_microseconds") or [0]) / 1000)
        for before, after in zip(rows, rows[1:]):
            m0, m1 = number(before, "app_pss_MB"), number(after, "app_pss_MB")
            gap = after["_t"] - before["_t"]
            when = before["_t"] - header["device_start_s"]
            if gap > GAP_S:
                s["flags"].append(f"no OVR Metrics rows for {gap:.0f} s at {when:.0f} s (app paused or headset off?)")
            if m0 is not None and m1 is not None and m1 - m0 > JUMP_MB:
                s["flags"].append(f"app memory jumped {m1 - m0:+.0f} MB at {when:.0f} s")
        if len(used) > 1:
            s["flags"].append(f"{len(used)} OVR Metrics sessions in one recording: the app restarted")
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
              f"seconds below 71 fps {s['below_71_s']}")
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
    compare = sub.add_parser("compare", help="check two runs against the agreement limits")
    compare.add_argument("run_a", metavar="RUN_A")
    compare.add_argument("run_b", metavar="RUN_B")
    args = parser.parse_args(argv)
    try:
        return cmd_summary(args) if args.command == "summary" else cmd_compare(args)
    except ToolError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
