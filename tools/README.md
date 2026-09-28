# tools

Command-line helpers for the operator. They run on a PC, never on the headset.

| Tool | What it does | Since |
|---|---|---|
| `runs.py` | creates and checks run folders (see `runs/README.md`) | S0.2 |

## Setup

Python 3.9 or newer, plus two packages, installed into whichever Python environment you use for this repository:

```
pip install -r tools/requirements.txt
```

Run the tools from the repository root, for example `python tools/runs.py check --all`.
