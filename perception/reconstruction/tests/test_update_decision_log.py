import os, subprocess, sys

TOOL = os.path.join(os.path.dirname(__file__), "..", "scripts", "update_decision_log.py")
LOG = """# Decisions

## Decisions

| ID | Date | Decision | Alternatives | Reason or evidence | Who decided |
|---|---|---|---|---|---|
| D130 | 2026-10-19 | an older decision | - | - | project lead |

## Open

| ID | Question | Needed by | Options so far |
|---|---|---|---|
| O29 | an older question | - | - |

## Closed
"""


def test_rows_are_inserted_once_in_the_right_tables(tmp_path):
    log = tmp_path / "decisions.md"; log.write_text(LOG)
    r = subprocess.run([sys.executable, TOOL, "--log", str(log)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    text = log.read_text().splitlines()
    i130, i138, i131 = (next(k for k, l in enumerate(text) if l.startswith(f"| {x} |")) for x in ("D130", "D138", "D131"))
    assert i138 < i131 < i130                                               # newest first, above the old rows
    o29, o30, o43 = (next(k for k, l in enumerate(text) if l.startswith(f"| {x} |")) for x in ("O29", "O30", "O43"))
    assert o29 < o30 < o43 < text.index("## Closed")
    again = subprocess.run([sys.executable, TOOL, "--log", str(log)], capture_output=True, text=True)
    assert "nothing to add" in again.stdout and log.read_text().count("| D133 |") == 1
    assert (tmp_path / "decisions.md.bak").is_file()
