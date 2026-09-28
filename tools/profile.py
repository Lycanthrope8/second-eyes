#!/usr/bin/env python3
"""Record the headset's state during a run, from the PC.

    python tools/profile.py record 20261001_A1_r007 --minutes 10

Every 30 seconds it saves the headset's thermal state, battery state and process list
(adb shell dumpsys thermalservice, dumpsys battery, top) to runs/<run ID>/raw/sampler.jsonl.
At the end it pulls the OVR Metrics CSV the headset recorded during the same time, which holds
frame rate, stale frames, CPU and GPU load and the app's memory, one row per second.
The headset stays connected by USB (D22). Summarize with: python analysis/profile.py summary <run ID>
Procedure: docs/profiling.md. Needs Python 3.9+ and adb on your PATH.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
PACKAGE = "com.secondeyes.quest"                          # the app's package name (D14)
METRICS_DIR = "/sdcard/Android/data/com.oculus.ovrmonitormetricsservice/files/CapturedMetrics"
METRICS_PREFIX = PACKAGE.replace(".", "_")               # OVR Metrics names its CSVs after the app


class ToolError(Exception):
    """A problem the user can fix. Printed without a traceback."""


def adb(*args: str, timeout: float = 30) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["adb", *args], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout)
    except FileNotFoundError:
        raise ToolError("adb was not found on your PATH.")
    except subprocess.TimeoutExpired:
        raise ToolError(f"adb did not answer within {timeout:.0f} s: adb {' '.join(args)}")


def text_of(result: subprocess.CompletedProcess) -> str:
    return (result.stdout or "").replace("\r", "")


def app_running() -> bool:
    result = adb("shell", "pidof", PACKAGE)
    return result.returncode == 0 and bool(text_of(result).strip())


def take_sample(started: float) -> dict:
    """One sample: the raw text of three commands. analysis/profile.py interprets it."""
    now = time.time()
    return {
        "pc_utc_s": round(now, 3),
        "elapsed_s": round(now - started, 1),
        "app_running": app_running(),
        "thermal": text_of(adb("shell", "dumpsys", "thermalservice")),
        "battery": text_of(adb("shell", "dumpsys", "battery")),
        "top": text_of(adb("shell", "top", "-b", "-n", "1", timeout=60)),
    }


def live_line(sample: dict, index: int, total: int) -> str:
    thermal = sample["thermal"]
    status = re.search(r"Thermal Status: (\d+)", thermal)
    current = thermal.find("Current temperatures from HAL:")   # fresher than the cached section
    soc = re.search(r"mValue=([-\d.]+), mType=\d+, mName=soc-usr", thermal[current:] if current >= 0 else thermal)
    app = "running" if sample["app_running"] else "NOT RUNNING"
    return (f"  [{sample['elapsed_s']:6.0f} s] sample {index}/{total}: app {app}, "
            f"thermal status {status.group(1) if status else '?'}, SoC {float(soc.group(1)):.1f} C" if soc else
            f"  [{sample['elapsed_s']:6.0f} s] sample {index}/{total}: app {app}, "
            f"thermal status {status.group(1) if status else '?'}")


def pull_metrics(raw: Path, since_device_s: int) -> list:
    """Copy the OVR Metrics CSVs of this app that were written during the recording."""
    listing = adb("shell", f"stat -c '%Y %n' {METRICS_DIR}/{METRICS_PREFIX}*.csv")
    pulled = []
    for line in text_of(listing).splitlines():
        parts = line.strip().split(" ", 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        modified, path = int(parts[0]), parts[1]
        if modified < since_device_s:
            continue  # last written before this recording started
        result = adb("pull", path, str(raw), timeout=300)
        if result.returncode == 0:
            pulled.append(path.rsplit("/", 1)[-1])
    return pulled


def cmd_record(args) -> int:
    run = RUNS / args.run_id
    if not (run / "config.yaml").is_file():
        raise ToolError(f"No run {args.run_id} in runs/. Create it first with: python tools/runs.py new ...")
    state = adb("get-state")
    if state.returncode != 0 or text_of(state).strip() != "device":
        raise ToolError("No headset ready over adb. Connect exactly one headset, awake, with USB debugging allowed.")
    if not app_running():
        raise ToolError("The app isn't running on the headset. Start it first, then record.")
    raw = run / "raw"
    raw.mkdir(exist_ok=True)
    out = raw / "sampler.jsonl"
    if out.exists():
        raise ToolError(f"runs/{args.run_id}/raw/sampler.jsonl already exists; one recording per run. "
                        "Create a new run for a new recording.")

    device_s = int(text_of(adb("shell", "date", "+%s")).strip() or 0)
    device_tz = text_of(adb("shell", "date", "+%z")).strip()
    total = int(args.minutes * 60 // args.interval_s) + 1
    started = time.time()
    header = {"format": 1, "run_id": args.run_id, "interval_s": args.interval_s, "minutes": args.minutes,
              "pc_utc_start_s": round(started, 3), "device_start_s": device_s, "device_tz": device_tz}
    print(f"Recording {args.minutes:g} min into runs/{args.run_id}/raw/sampler.jsonl "
          f"({total} samples, every {args.interval_s:g} s). Keep the headset on. Ctrl+C stops early.")

    taken = 0
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(header) + "\n")
        try:
            for index in range(1, total + 1):
                due = started + (index - 1) * args.interval_s
                time.sleep(max(0.0, due - time.time()))
                sample = take_sample(started)
                f.write(json.dumps(sample) + "\n")
                f.flush()
                taken += 1
                print(live_line(sample, index, total))
                if not sample["app_running"]:
                    print("  The app stopped, so the recording ends here.")
                    break
        except KeyboardInterrupt:
            print("  Stopped early.")

    pulled = pull_metrics(raw, device_s - 5)
    print(f"Saved {taken} sample(s).")
    if pulled:
        print("Pulled OVR Metrics recording(s): " + ", ".join(pulled))
    else:
        print("No OVR Metrics CSV from this recording was found on the headset: was its CSV recording on?\n"
              f"  (looked in {METRICS_DIR})")
    print(f"Next: python analysis/profile.py summary {args.run_id}")
    return 0


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except AttributeError:
            pass
    parser = argparse.ArgumentParser(prog="python tools/profile.py",
                                     description="Record the headset's state during a run. See docs/profiling.md")
    sub = parser.add_subparsers(dest="command", required=True)
    record = sub.add_parser("record", help="sample the headset for a number of minutes")
    record.add_argument("run_id", metavar="RUN_ID")
    record.add_argument("--minutes", type=float, required=True, help="how long to record (the reference uses 10)")
    record.add_argument("--interval-s", type=float, default=30.0, help="seconds between samples (keep 30 for measurements)")
    args = parser.parse_args(argv)
    try:
        return cmd_record(args)
    except ToolError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
