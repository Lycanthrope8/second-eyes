# S0 · Research infrastructure and logging

| | |
|---|---|
| Dates | 2026-09-27 – 2026-09-27 |
| Gate | none (S0 has no gate) |
| Author | project lead |

## 1. What was built

- **Repository skeleton (S0.1):** top level split by component (D2), a README map and status table, `docs/conventions.md`, the three running lists, the phase-note template, and the headset setup record `docs/setup/quest3.md`.
- **Run registry (S0.2):** run IDs `YYYYMMDD_<phase>_r<NNN>`, counted per phase and never reset (D3); config format v1 in `schemas/run-config.v1.json`, with `runs/_template.yaml`; `tools/runs.py new` and `check` (D4, D5). `git_dirty` counts every uncommitted change (D6).
- **Event log format v1 (S0.3):** JSON Lines, one file per session; every line has `seq`, `mono_us`, `utc_us`, `ev` and `data`; four event types defined (D9). The format is in `schemas/log-event.v1.json` and `docs/logging.md`, and `analysis/eventlog.py check` and `summary` read it. Logs live in `runs/<run ID>/raw/`, which Git ignores (D8).

## 2. Final configuration

- Formats: run config v1 and log event v1.
- The tools need Python 3.9 or newer, plus `tools/requirements.txt` (PyYAML, jsonschema) and `analysis/requirements.txt` (jsonschema).
- Tested only in throwaway copies of the repository in Claude's sandbox (Python 3.12.3, PyYAML 6.0.3, jsonschema 4.26.0, Git 2.43.0), not yet on the project's own machines.
- No runs: S0 built infrastructure and measured nothing, so there are no run IDs to cite.

## 3. Key numbers

None; S0 measured nothing.

## 4. Figures, photos and videos

None.

## 5. Decisions taken

All are logged in `notes/decisions.md`.

| ID | Decision |
|---|---|
| D1 | Storage and experiment tracking (S0 items 5–6) wait until Gate A |
| D2 | Top level split by component |
| D3 | Run numbers count per phase and never reset |
| D4 | Operator helpers live in `tools/` |
| D5 | Run registry v1 |
| D6 | `git_dirty` counts every uncommitted change |
| D7 | The C# logger and the log-pull command are built in A1 |
| D8 | Raw data lives in `runs/<run ID>/raw/`, ignored by Git, until storage is decided |
| D9 | Event log format v1 |
| D10 | Time sync (S0 item 4) waits until Vicon is actually used |

## 6. What failed and why

Nothing failed.

## 7. Deviations from the plan

- The S0 exit test (a dummy run logged, pulled, synchronized with Vicon and plotted by script) was not run. Pulling needs the A1 app (D7), and time sync waits for Vicon (D10).
- Item 3 is half done: the log format exists, but the logger inside the Quest app moved to A1 (D7).
- Item 4 is deferred until Vicon is used (D10), and items 5–6 until Gate A (D1).
- Added beyond the proposal's folder list: `docs/`, `runs/`, `schemas/` and `tools/` (D2, D4).

## 8. Open issues handed to the next phase

- O5: what to disable on the Quest 3, decided from the baseline measurement (headset prep, steps 1–2), before A1 profiling.
- A1 starts with the C# logger and the log-pull command (D7), which bring O6 (where C# code lives) and O12 (how a log is tied to its run).
- The tools have not yet run on the project's own machines. The first real use is a trial run with `tools/runs.py new` and `check`.
- Open for later: O3 (the first logged pose), O4 (when Vicon is used), O7 (B1), O8 (Gate A), O9 and O10 (after Gate A).

## 9. Paper hooks

- **Method:** every reported number traces to a run ID, a commit and a clean-or-dirty flag, and logs carry both a monotonic and a wall clock. Python package versions are in the two requirements files; the rest of the software stack (Unity, Meta XR SDK, inference runtime, drone SDK) is recorded in A1.
- **Appendix:** log format v1 (`docs/logging.md`, `schemas/log-event.v1.json`), which reviewers need to see how latency and error were measured. The synchronization procedure follows once time sync is built.
- **Ethics:** the data-handling plan comes with the IRB protocol (A0) and the storage decision (O9).
