# analysis

Offline analysis of logged runs: reading event logs, aligning the Quest, drone and Vicon clocks, and plots. Never part of the deployed system.

| Script | What it does | Since |
|---|---|---|
| `eventlog.py` | reads, checks and summarizes event logs (see `docs/logging.md`) | S0.3 |
| `profile.py` | summarizes and compares recordings from `tools/profile.py` (see `docs/profiling.md`) | A1.6a |

## Setup

Python 3.9 or newer, plus the packages in `requirements.txt`, installed into whichever Python environment you use for this repository:

```
pip install -r analysis/requirements.txt
```
