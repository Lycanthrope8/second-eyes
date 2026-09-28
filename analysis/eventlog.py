#!/usr/bin/env python3
"""Read, check and summarize Second Eyes event logs (JSON Lines, one file per session).

    python analysis/eventlog.py check 20261001_A1_r007        every .jsonl in runs/<run ID>/raw/
    python analysis/eventlog.py check path/to/session.jsonl   one file
    python analysis/eventlog.py summary 20261001_A1_r007

Format: docs/logging.md and schemas/log-event.v1.json.
Scripts in analysis/ can reuse the reader:  from eventlog import read_events
Needs Python 3.9+ and: pip install -r analysis/requirements.txt
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path

try:
    import jsonschema
except ImportError:
    sys.exit("Missing package. Install it with: pip install -r analysis/requirements.txt")

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
SCHEMA = ROOT / "schemas" / "log-event.v1.json"
MAX_SHOWN = 20  # problems printed per file before "... and N more"


class ToolError(Exception):
    """A problem the user can fix. Printed without a traceback."""


# ---------------------------------------------------------------- reading

def read_events(path) -> list:
    """Return the events of one log file, in file order.

    Raises ValueError naming the line if a line is not a JSON object.
    Run `check` first if you need every problem in the file listed.
    """
    events = []
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"{path}, line {number}: not valid JSON ({err.msg})") from None
            if not isinstance(event, dict):
                raise ValueError(f"{path}, line {number}: not a JSON object")
            events.append(event)
    return events


# ---------------------------------------------------------------- checking

def load_validators() -> tuple:
    """Return (validator for every line, {event type: validator for its data})."""
    with open(SCHEMA, encoding="utf-8") as f:
        schema = json.load(f)
    cls = jsonschema.validators.validator_for(schema)
    return cls(schema), {name: cls(sub) for name, sub in schema.get("$defs", {}).items()}


def describe(err, prefix: str = "") -> str:
    path = ".".join(str(p) for p in err.absolute_path)
    where = prefix + path if path else prefix.rstrip(".")
    message = err.message
    hint = err.schema.get("description") if isinstance(err.schema, dict) else None
    if err.validator == "pattern" and hint:
        message = f"{err.instance!r} is not valid (expected: {hint})"
    elif hint and err.validator in ("type", "minLength", "minimum"):
        message += f" (expected: {hint})"
    return f"{where}: {message}" if where else message


def is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def check_file(path, validators=None) -> dict:
    """Check one log file. Returns problems, notes and the numbers `summary` prints."""
    envelope, data_validators = validators or load_validators()
    problems, notes = [], []
    counts, undocumented = Counter(), Counter()
    first = last = None
    ended = False
    prev_seq = prev_mono = None
    lines = 0
    try:
        with open(path, encoding="utf-8") as f:
            for number, line in enumerate(f, start=1):
                if not line.strip():
                    problems.append(f"line {number}: blank line")
                    continue
                lines += 1
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as err:
                    problems.append(f"line {number}: not valid JSON ({err.msg})")
                    continue
                if not isinstance(event, dict):
                    problems.append(f"line {number}: not a JSON object")
                    continue
                errors = list(envelope.iter_errors(event))
                for err in errors:
                    problems.append(f"line {number}: {describe(err)}")
                ev_valid = not any(list(err.absolute_path)[:1] == ["ev"] for err in errors)
                ev, seq, mono = event.get("ev"), event.get("seq"), event.get("mono_us")

                if is_int(seq):
                    if prev_seq is None and seq != 0:
                        problems.append(f"line {number}: seq starts at {seq}, expected 0")
                    elif prev_seq is not None and seq > prev_seq + 1:
                        problems.append(f"line {number}: seq jumps from {prev_seq} to {seq} "
                                        f"({seq - prev_seq - 1} line(s) missing)")
                    elif prev_seq is not None and seq <= prev_seq:
                        problems.append(f"line {number}: seq goes from {prev_seq} back to {seq}")
                    prev_seq = seq
                if is_int(mono):
                    if prev_mono is not None and mono < prev_mono:
                        problems.append(f"line {number}: mono_us goes backwards by {prev_mono - mono} us")
                    prev_mono = mono

                if isinstance(ev, str):
                    counts[ev] += 1
                    if ev in data_validators:
                        for err in data_validators[ev].iter_errors(event.get("data")):
                            problems.append(f"line {number} ({ev}): {describe(err, 'data.')}")
                    elif ev_valid:
                        undocumented[ev] += 1
                    if ev == "session.start" and first is not None:
                        problems.append(f"line {number}: another session.start; one file holds one session")
                if last is not None and last.get("ev") == "session.end":
                    problems.append(f"line {number}: event after session.end")
                ended = ended or ev == "session.end"
                if first is None:
                    first = event
                last = event
    except UnicodeDecodeError as err:
        problems.append(f"file is not UTF-8 text (byte {err.start})")

    if lines == 0 and not problems:
        problems.append("file is empty")
    elif lines:
        if first is None or first.get("ev") != "session.start":
            problems.append("first event is not session.start")
        if not ended:
            notes.append("no session.end: the app may have crashed, or the file was pulled mid-session")
    for ev, n in sorted(undocumented.items()):
        notes.append(f"undocumented event type '{ev}' ({n} line(s)): add it to docs/logging.md "
                     "and schemas/log-event.v1.json")

    info = {"path": Path(path), "lines": lines, "counts": counts,
            "problems": problems, "notes": notes, "session": {}}
    if first and last and is_int(first.get("mono_us")) and is_int(last.get("mono_us")):
        info["duration_s"] = (last["mono_us"] - first["mono_us"]) / 1e6
    if first and is_int(first.get("utc_us")):
        try:
            info["start_utc"] = dt.datetime.fromtimestamp(first["utc_us"] / 1e6, tz=dt.timezone.utc)
        except (OverflowError, OSError, ValueError):
            pass
    if first and first.get("ev") == "session.start" and isinstance(first.get("data"), dict):
        info["session"] = first["data"]
    return info


# ---------------------------------------------------------------- output

def display(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def targets_for(target: str) -> list:
    path = Path(target)
    if path.is_file():
        return [path]
    run = RUNS / target
    if run.is_dir():
        files = sorted((run / "raw").glob("*.jsonl"))
        if not files:
            raise ToolError(f"No .jsonl files in runs/{target}/raw/.")
        return files
    raise ToolError(f"'{target}' is neither a log file nor a run ID in runs/.")


def print_check(info: dict) -> None:
    problems, notes = info["problems"], info["notes"]
    length = f", {info['duration_s']:.1f} s" if "duration_s" in info else ""
    print(("FAIL  " if problems else "OK    ") + f"{display(info['path'])}  ({info['lines']} lines{length})")
    for problem in problems[:MAX_SHOWN]:
        print(f"      - {problem}")
    if len(problems) > MAX_SHOWN:
        print(f"      ... and {len(problems) - MAX_SHOWN} more")
    for note in notes:
        print(f"      note: {note}")


def print_summary(info: dict) -> None:
    session = info["session"]
    print(display(info["path"]))
    print(f"  session   {session.get('session_id', '?')}, app {session.get('app_version', '?')}")
    print(f"  os build  {session.get('os_build') or '-'}")
    if "start_utc" in info:
        print(f"  started   {info['start_utc']:%Y-%m-%d %H:%M:%S} UTC (device wall clock)")
    length = f"{info['duration_s']:.1f} s, " if "duration_s" in info else ""
    print(f"  length    {length}{info['lines']} lines")
    events = ", ".join(f"{ev} {n}" for ev, n in sorted(info["counts"].items())) or "-"
    print(f"  events    {events}")
    problems, notes = len(info["problems"]), len(info["notes"])
    verdict = "OK" if not problems else f"{problems} problem(s)"
    extra = f", {notes} note(s)" if notes else ""
    print(f"  check     {verdict}{extra}" + (" (run check for details)" if problems or notes else ""))


def cmd_check(args) -> int:
    validators = load_validators()
    files = [path for target in args.targets for path in targets_for(target)]
    failed = 0
    for path in files:
        info = check_file(path, validators)
        print_check(info)
        failed += bool(info["problems"])
    print(f"Checked {len(files)} file(s): {len(files) - failed} OK, {failed} with problems.")
    return 1 if failed else 0


def cmd_summary(args) -> int:
    validators = load_validators()
    files = [path for target in args.targets for path in targets_for(target)]
    for index, path in enumerate(files):
        if index:
            print()
        print_summary(check_file(path, validators))
    return 0


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except AttributeError:
            pass
    parser = argparse.ArgumentParser(
        prog="python analysis/eventlog.py",
        description="Read, check and summarize event logs. Format: docs/logging.md")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("check", "check logs against the format"),
                            ("summary", "print what each log contains")):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("targets", nargs="+", metavar="RUN_ID_OR_FILE")
    args = parser.parse_args(argv)
    try:
        return cmd_check(args) if args.command == "check" else cmd_summary(args)
    except ToolError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
