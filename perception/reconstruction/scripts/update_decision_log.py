#!/usr/bin/env python3
"""Inserts this path's decision-log entries (docs/decision-record.md) into the repository's notes/decisions.md.

    python scripts/update_decision_log.py [--log PATH] [--dry-run]

Rows whose ID (D..., O...) the log already has are left alone and listed, so running it twice changes nothing. New
decisions go at the top of the log's Decisions table (newest first); new open items at the end of its Open table. If
either table can't be found, nothing is written and the rows to paste are printed instead. A copy of the old file is
kept as decisions.md.bak."""
import argparse, re, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ROW = re.compile(r"^\| ([DO]\d+) \|")


def rows_of(text, header):
    """The table rows under the first header line starting with `header`, as (id, line)."""
    lines = text.splitlines()
    for k, l in enumerate(lines):
        if l.startswith(header):
            out = []
            for m in lines[k + 2:]:
                if not m.startswith("|"):
                    break
                g = ROW.match(m)
                if g:
                    out.append((g.group(1), m))
            return out
    return None


def insert(log_text, new_dec, new_open):
    lines = log_text.splitlines()
    def find(header):
        return next((k for k, l in enumerate(lines) if l.startswith(header)), None)
    kd, ko = find("| ID | Date | Decision"), find("| ID | Question")
    if kd is None or ko is None:
        return None
    if new_open:                                   # at the end of the Open table (do this first: it is further down)
        end = ko + 2
        while end < len(lines) and lines[end].startswith("|"):
            end += 1
        lines[end:end] = new_open
    if new_dec:                                    # newest first, right under the header and separator
        kd = find("| ID | Date | Decision")
        lines[kd + 2:kd + 2] = new_dec
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default=str(HERE.parent.parent / "notes" / "decisions.md"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    rec = (HERE / "docs" / "decision-record.md").read_text(encoding="utf-8")
    dec, opn = rows_of(rec, "| ID | Date | Decision"), rows_of(rec, "| ID | Question")
    log_p = Path(a.log)
    if not log_p.is_file():
        sys.exit(f"{log_p} not found; give --log")
    log = log_p.read_text(encoding="utf-8")
    have = set(m.group(1) for m in re.finditer(r"^\| ([DO]\d+) \|", log, re.M))
    new_dec = [l for i, l in dec if i not in have]
    new_open = [l for i, l in opn if i not in have]
    skipped = [i for i, _ in dec + opn if i in have]
    if skipped:
        print("already in the log, left as is: " + ", ".join(skipped))
    if not new_dec and not new_open:
        print("nothing to add")
        return
    out = insert(log, new_dec, new_open)
    if out is None:
        print("the log's Decisions or Open table header was not found; paste these rows by hand:\n")
        print("\n".join(new_dec + [""] + new_open))
        sys.exit(1)
    print(f"adding {len(new_dec)} decision(s) and {len(new_open)} open item(s): "
          + ", ".join(ROW.match(l).group(1) for l in new_dec + new_open))
    if a.dry_run:
        return
    shutil.copy2(log_p, str(log_p) + ".bak")
    log_p.write_text(out, encoding="utf-8")
    print(f"-> {log_p} (previous version kept as {log_p.name}.bak)")


if __name__ == "__main__":
    main()
