#!/usr/bin/env python3
"""Pull session logs from the headset into a run.

    python tools/logs.py pull 20261001_A1_r007

Quit the app first. The command copies every new session log from the headset into
runs/<run ID>/raw/, checks each copy's size, then moves the original into logs/pulled/
on the headset, so nothing is copied twice and nothing is deleted (D16).
Log format: docs/logging.md. Needs Python 3.9+ and adb on your PATH.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
PACKAGE = "com.secondeyes.quest"                         # the app's package name (D14)
LOG_DIR = f"/sdcard/Android/data/{PACKAGE}/files/logs"   # where EventLog.cs writes on the headset
PULLED_DIR = f"{LOG_DIR}/pulled"


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
    return (result.stdout or "").replace("\r", "").strip()


def error_of(result: subprocess.CompletedProcess) -> str:
    return ((result.stderr or "") + (result.stdout or "")).replace("\r", "").strip() or "no details"


def check_headset() -> None:
    """Exactly one headset must be ready, and the app must not be running."""
    state = adb("get-state")
    if state.returncode != 0 or text_of(state) != "device":
        raise ToolError("No headset ready over adb: " + error_of(state) + "\n"
                        "Connect exactly one headset, awake, with USB debugging allowed.")
    running = adb("shell", "pidof", PACKAGE)
    if running.returncode == 0 and text_of(running):
        raise ToolError("The app is still running on the headset. Quit it first (Meta button, then Quit),\n"
                        "so its log is complete and ends with session.end. Nothing was pulled.")


def new_logs() -> list:
    """Session logs on the headset that haven't been pulled yet, oldest first."""
    listing = adb("shell", "ls", "-1", LOG_DIR)
    if listing.returncode != 0:
        return []  # the folder doesn't exist yet: the app has never written a log
    return sorted(name for name in text_of(listing).splitlines() if name.endswith(".jsonl"))


def remote_size(path: str) -> int:
    result = adb("shell", "stat", "-c", "%s", path)
    size = text_of(result)
    if result.returncode != 0 or not size.isdigit():
        raise ToolError(f"Could not read the size of {path} on the headset: {error_of(result)}")
    return int(size)


def cmd_pull(args) -> int:
    run = RUNS / args.run_id
    if not (run / "config.yaml").is_file():
        raise ToolError(f"No run {args.run_id} in runs/. Create it first with: python tools/runs.py new ...")
    check_headset()
    names = new_logs()
    if not names:
        print(f"No new logs on the headset (looked in {LOG_DIR}).")
        return 0

    raw = run / "raw"
    raw.mkdir(exist_ok=True)
    if adb("shell", "mkdir", "-p", PULLED_DIR).returncode != 0:
        raise ToolError(f"Could not create {PULLED_DIR} on the headset. Nothing was pulled.")

    print(f"{len(names)} new log(s) on the headset:")
    for name in names:
        source = f"{LOG_DIR}/{name}"
        target = raw / name
        size = remote_size(source)
        note = ""
        if target.exists():
            if target.stat().st_size != size:
                raise ToolError(f"runs/{args.run_id}/raw/{name} already exists with a different size, so it was not "
                                "overwritten, and the original stays on the headset.")
            note = "  (already here from an earlier pull)"
        else:
            copy = adb("pull", source, str(raw), timeout=600)
            if copy.returncode != 0:
                target.unlink(missing_ok=True)
                raise ToolError(f"Copying {name} failed: {error_of(copy)}\n"
                                "The original stays on the headset; run the pull again.")
            copied = target.stat().st_size if target.is_file() else 0
            if copied != size:
                target.unlink(missing_ok=True)
                raise ToolError(f"The copy of {name} came out incomplete ({copied:,} of {size:,} bytes), so it was removed.\n"
                                "The original stays on the headset; run the pull again.")
        move = adb("shell", "mv", source, PULLED_DIR + "/")
        if move.returncode != 0:
            raise ToolError(f"{name} was copied, but moving it into pulled/ on the headset failed: {error_of(move)}\n"
                            "Run the pull again; it will recognize the copy.")
        print(f"  {name}  {size:,} bytes -> runs/{args.run_id}/raw/{note}")

    print(f"Pulled {len(names)} file(s); the originals are now in logs/pulled/ on the headset.")
    print(f"Next: python analysis/eventlog.py check {args.run_id}")
    return 0


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except AttributeError:
            pass
    parser = argparse.ArgumentParser(prog="python tools/logs.py",
                                     description="Pull session logs from the headset into a run. See docs/logging.md")
    sub = parser.add_subparsers(dest="command", required=True)
    pull = sub.add_parser("pull", help="copy new session logs into runs/<run ID>/raw/")
    pull.add_argument("run_id", metavar="RUN_ID")
    args = parser.parse_args(argv)
    try:
        return cmd_pull(args)
    except ToolError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
