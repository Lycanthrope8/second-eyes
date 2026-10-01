#!/usr/bin/env python3
"""Create and check run folders for the Second Eyes run registry.

    python tools/runs.py new A1 --purpose "idle baseline" --operator AB
    python tools/runs.py check 20261001_A1_r001
    python tools/runs.py check --all

Rules and fields: runs/README.md. Format: schemas/run-config.v1.json.
Needs Python 3.9+ and: pip install -r tools/requirements.txt
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

try:
    import jsonschema
    import yaml
except ImportError:
    sys.exit("Missing packages. Install them with: pip install -r tools/requirements.txt")

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
TEMPLATE = RUNS / "_template.yaml"
SCHEMA = ROOT / "schemas" / "run-config.v1.json"

RUN_ID = re.compile(r"^(\d{8})_(.+)_r(\d{3,})$")
TOKEN = re.compile(r"__[A-Z0-9_]+__")


class ToolError(Exception):
    """A problem the user can fix. Printed without a traceback."""


# ---------------------------------------------------------------- helpers

def load_schema() -> dict:
    with open(SCHEMA, encoding="utf-8") as f:
        return json.load(f)


def allowed_phases(schema: dict) -> list:
    return schema["properties"]["phase"]["enum"]


def run_folders() -> list:
    """Every folder in runs/ except templates (_...) and hidden ones (.…)."""
    return sorted(p for p in RUNS.iterdir()
                  if p.is_dir() and not p.name.startswith(("_", ".")))


def next_number(phase: str) -> int:
    """Run numbers count per phase and never reset."""
    numbers = [int(m.group(3)) for p in run_folders()
               if (m := RUN_ID.match(p.name)) and m.group(2) == phase]
    return max(numbers, default=0) + 1


def git(*args: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, encoding="utf-8", errors="replace")
    except FileNotFoundError:
        raise ToolError("git was not found. Install Git and make sure it is on your PATH.")


def git_state() -> tuple:
    """Return (commit, dirty, changed paths). Refuse without a repository or a commit. Paths inside run folders
    (runs/<run ID>/) don't count: they record runs rather than change what a run executes, so several runs in one
    session needn't be committed one by one (D51)."""
    inside = git("rev-parse", "--is-inside-work-tree")
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        raise ToolError(
            "This folder is not inside a Git repository, so the run could not be traced\n"
            "to a commit. Run 'git init', commit, and try again.")
    head = git("rev-parse", "HEAD")
    if head.returncode != 0:
        raise ToolError(
            "The repository has no commits yet, so the run could not be traced to a commit.\n"
            "Commit first (git add . then git commit -m \"...\") and try again.")
    status = git("status", "--porcelain", "--", ".")
    if status.returncode != 0:
        raise ToolError("git status failed:\n" + status.stderr.strip())
    changed = [line[3:] for line in status.stdout.splitlines() if line.strip()]
    changed = [path for path in changed if not is_run_record(path)]
    return head.stdout.strip(), bool(changed), changed


def is_run_record(path: str) -> bool:
    """Whether a path from git status lies inside one run's folder, runs/<run ID>/ (D51)."""
    parts = path.strip().strip('"').split("/")
    return len(parts) >= 2 and parts[0] == "runs" and bool(RUN_ID.match(parts[1]))


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def headset_os_build() -> str:
    """Read ro.build.fingerprint from the connected headset over adb."""
    try:
        result = subprocess.run(["adb", "shell", "getprop", "ro.build.fingerprint"],
                                capture_output=True, text=True, timeout=20)
    except FileNotFoundError:
        raise ToolError("adb was not found on your PATH. Add it to your PATH, or run\n"
                        "without --headset and fill headset_os_build by hand.")
    except subprocess.TimeoutExpired:
        raise ToolError("adb did not answer within 20 s. Check the cable and that the headset is awake.")
    value = result.stdout.strip()
    if result.returncode != 0 or not value:
        detail = (result.stderr or result.stdout).strip() or "no output"
        raise ToolError("Could not read the headset's OS build over adb: " + detail + "\n"
                        "Check that the headset is connected, awake and allows USB debugging.")
    return value


def yaml_value(value) -> str:
    """Write a value into the template. Strings are quoted so YAML never reinterprets them."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return json.dumps(str(value), ensure_ascii=False)


def schema_errors(config, schema: dict) -> list:
    validator = jsonschema.validators.validator_for(schema)(schema)
    problems = []
    for err in sorted(validator.iter_errors(config),
                      key=lambda e: [str(p) for p in e.absolute_path]):
        where = ".".join(str(p) for p in err.absolute_path) or "(top level)"
        message = err.message
        hint = err.schema.get("description") if isinstance(err.schema, dict) else None
        if err.validator == "additionalProperties":
            message += "; new fields go into schemas/run-config.v1.json first"
        elif err.validator == "pattern" and hint:
            message = f"{err.instance!r} is not valid (expected: {hint})"
        elif hint and err.validator != "required":
            message += f" (expected: {hint})"
        problems.append(f"{where}: {message}")
    return problems


# ---------------------------------------------------------------- new

def cmd_new(args) -> int:
    schema = load_schema()
    phases = allowed_phases(schema)
    if args.phase not in phases:
        raise ToolError(f"Unknown phase '{args.phase}'. Allowed: {', '.join(phases)}")
    for name in ("purpose", "operator"):
        value = getattr(args, name)
        if not value.strip() or "\n" in value:
            raise ToolError(f"--{name} must be one non-empty line.")

    # Everything that can fail runs before anything is written.
    commit, dirty, changed = git_state()
    checkpoint = sha = None
    if args.checkpoint:
        path = Path(args.checkpoint)
        if not path.is_file():
            raise ToolError(f"Checkpoint file not found: {path}")
        print(f"Hashing {path.name} ...")
        checkpoint, sha = path.name, sha256_of(path)
    os_build = headset_os_build() if args.headset else None

    now = dt.datetime.now().astimezone().replace(microsecond=0)
    run_id = f"{now:%Y%m%d}_{args.phase}_r{next_number(args.phase):03d}"
    values = {
        "RUN_ID": run_id,
        "PHASE": args.phase,
        "CREATED": now.isoformat(),
        "PURPOSE": args.purpose.strip(),
        "OPERATOR": args.operator.strip(),
        "GIT_COMMIT": commit,
        "GIT_DIRTY": dirty,
        "MODEL_CHECKPOINT": checkpoint,
        "MODEL_SHA256": sha,
        "HEADSET_OS_BUILD": os_build,
    }

    template = TEMPLATE.read_text(encoding="utf-8")
    expected = {f"__{key}__" for key in values}
    found = set(TOKEN.findall(template))
    if found != expected:
        raise ToolError("runs/_template.yaml and tools/runs.py disagree on placeholders.\n"
                        f"  only in the template: {sorted(found - expected)}\n"
                        f"  only in the tool:     {sorted(expected - found)}")
    text = template
    for key, value in values.items():
        text = text.replace(f"__{key}__", yaml_value(value))
    problems = schema_errors(yaml.safe_load(text), schema)
    if problems:
        raise ToolError("The template and the schema disagree; fix one of them:\n  "
                        + "\n  ".join(problems))

    folder = RUNS / run_id
    folder.mkdir()
    (folder / "config.yaml").write_text(text, encoding="utf-8")

    print(f"Created run {run_id}")
    print(f"  config  {(folder / 'config.yaml').relative_to(ROOT).as_posix()}")
    print(f"  commit  {commit[:10]}")
    if dirty:
        print("Warning: the repository has uncommitted changes, so this run is marked git_dirty: true.")
        print("  Commit first if the run should be reproducible from the commit alone. Changed:")
        for line in changed[:10]:
            print(f"    {line}")
        if len(changed) > 10:
            print(f"    ... and {len(changed) - 10} more")
    print("Next: fill the optional fields that apply, then run")
    print(f"  python tools/runs.py check {run_id}")
    return 0


# ---------------------------------------------------------------- check

def check_one(folder: Path, schema: dict, used: dict) -> tuple:
    """Return (problems, notes) for one run folder."""
    problems, notes = [], []
    match = RUN_ID.match(folder.name)
    if not match:
        problems.append("folder name is not a run ID (YYYYMMDD_<phase>_r<NNN>)")
    else:
        key = (match.group(2), int(match.group(3)))
        others = [name for name in used[key] if name != folder.name]
        if others:
            problems.append(f"{key[0]} r{key[1]:03d} is also used by {', '.join(others)}")

    path = folder / "config.yaml"
    if not path.is_file():
        problems.append("config.yaml is missing")
        return problems, notes
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as err:
        mark = getattr(err, "problem_mark", None)
        where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        reason = getattr(err, "problem", None) or str(err).splitlines()[0]
        problems.append(f"config.yaml is not valid YAML{where}: {reason}")
        return problems, notes

    problems += schema_errors(config, schema)
    if not isinstance(config, dict):
        return problems, notes
    run_id, phase, created = config.get("run_id"), config.get("phase"), config.get("created")
    if isinstance(run_id, str) and run_id != folder.name:
        problems.append(f"run_id is {run_id} but the folder is named {folder.name}")
    if match and isinstance(phase, str) and phase != match.group(2):
        problems.append(f"phase is {phase} but the run ID says {match.group(2)}")
    if match and isinstance(created, str) and created[:10].replace("-", "") != match.group(1):
        problems.append(f"created is on {created[:10]} but the run ID says {match.group(1)}")
    if config.get("git_dirty") is True:
        notes.append("created with uncommitted changes (git_dirty: true)")
    return problems, notes


TEXT_FIELDS = ("purpose", "notes")   # the free-text fields `set` may change


def field_line(field: str):
    return re.compile(rf'^{field}:[ \t]*(?:"(?:[^"\\]|\\.)*"|[^#]*?)([ \t]*#.*)?$')


def cmd_set(args) -> int:
    """Set a run's purpose or notes, keeping the line's comment and the file's line endings, then read it back."""
    if args.field not in TEXT_FIELDS:
        raise ToolError(f"set changes only {' and '.join(TEXT_FIELDS)}; the other fields record facts.")
    path = RUNS / args.run_id / "config.yaml"
    if not path.is_file():
        raise ToolError(f"No run {args.run_id} in runs/.")
    text = path.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.split(newline)
    pattern = field_line(args.field)
    for i, line in enumerate(lines):
        m = pattern.match(line)
        if m:
            lines[i] = f"{args.field}: {yaml_value(args.text)}{m.group(1) or ''}"
            break
    else:
        raise ToolError(f"runs/{args.run_id}/config.yaml has no {args.field}: line to fill.")
    new_text = newline.join(lines)
    if (yaml.safe_load(new_text) or {}).get(args.field) != args.text:
        raise ToolError(f"The {args.field} didn't survive a read-back, so nothing was written.")
    path.write_bytes(new_text.encode("utf-8"))
    print(f"Wrote the {args.field} of {args.run_id} ({len(args.text)} characters). Next: python tools/runs.py check {args.run_id}")
    return 0


def cmd_note(args) -> int:
    """`note RUN TEXT` is `set RUN notes TEXT`."""
    args.field = "notes"
    return cmd_set(args)


def cmd_check(args) -> int:
    schema = load_schema()
    folders = run_folders()
    if args.all and args.run_ids:
        raise ToolError("Give run IDs or --all, not both.")
    if args.all:
        targets = folders
    elif args.run_ids:
        targets = []
        for run_id in args.run_ids:
            folder = RUNS / run_id
            if not folder.is_dir():
                raise ToolError(f"No run folder named {run_id} in runs/.")
            targets.append(folder)
    else:
        raise ToolError("Give one or more run IDs, or --all.")
    if not targets:
        print("No runs yet.")
        return 0

    used = {}  # (phase, number) -> folder names, to catch a number used twice
    for folder in folders:
        match = RUN_ID.match(folder.name)
        if match:
            used.setdefault((match.group(2), int(match.group(3))), []).append(folder.name)

    failed = 0
    for folder in targets:
        problems, notes = check_one(folder, schema, used)
        print(("FAIL  " if problems else "OK    ") + folder.name)
        for problem in problems:
            print(f"      - {problem}")
        for note in notes:
            print(f"      note: {note}")
        failed += bool(problems)
    print(f"Checked {len(targets)} run(s): {len(targets) - failed} OK, {failed} with problems.")
    return 1 if failed else 0


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except AttributeError:
            pass
    parser = argparse.ArgumentParser(
        prog="python tools/runs.py",
        description="Create and check run folders. Rules: runs/README.md")
    sub = parser.add_subparsers(dest="command", required=True)

    new = sub.add_parser("new", help="create the next run folder for a phase")
    new.add_argument("phase", help="phase ID from the proposal, e.g. A1")
    new.add_argument("--purpose", required=True, help="one line: why this run exists")
    new.add_argument("--operator", required=True, help="who runs it")
    new.add_argument("--checkpoint", metavar="FILE",
                     help="model file: records its name and SHA-256")
    new.add_argument("--headset", action="store_true",
                     help="read the OS build from the connected Quest over adb")

    check = sub.add_parser("check", help="check runs against the schema and the ID rules")
    check.add_argument("run_ids", nargs="*", metavar="RUN_ID")
    check.add_argument("--all", action="store_true", help="check every run folder")

    note = sub.add_parser("note", help="set a run's notes (the notes: line of its config.yaml)")
    note.add_argument("run_id", metavar="RUN_ID")
    note.add_argument("text", help="the notes, in quotes")

    setter = sub.add_parser("set", help="set a run's purpose or notes")
    setter.add_argument("run_id", metavar="RUN_ID")
    setter.add_argument("field", choices=TEXT_FIELDS)
    setter.add_argument("text", help="the new text, in quotes")

    args = parser.parse_args(argv)
    try:
        return {"new": cmd_new, "check": cmd_check, "note": cmd_note, "set": cmd_set}[args.command](args)
    except ToolError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
