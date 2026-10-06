# analysis

Offline analysis of logged runs: reading event logs, aligning the Quest, drone and Vicon clocks, and plots. Never part of the deployed system. It also keeps A2.1d's two token audits, which measure model inputs rather than runs, each in its own folder with a README.

| Script | What it does | Since |
|---|---|---|
| `eventlog.py` | reads, checks and summarizes event logs (see `docs/logging.md`) | S0.3 |
| `profile.py` | summarizes and compares recordings from `tools/profile.py` (see `docs/profiling.md`) | A1.6a |
| `camera_frames.py` | the passthrough camera's frames per second and its toggles, from a run's session logs (see `docs/logging.md`) | A1.10a |
| `A2.1d_token_sizing/` | the v0.2 token audit: renders the serializer proposal's two formats for four inputs and counts their tokens with the pinned tokenizer; kept unchanged as the indexed audit's reference | A2.1d |
| `A2.1d_indexed_token_sizing/` | the indexed relation-encoding audit, whose encoding the serializer adopted (D69): 64 verification checks, then token counts for eight inputs | A2.1d |

## Setup

Python 3.9 or newer, plus the packages in `requirements.txt`, installed into whichever Python environment you use for this repository:

```
pip install -r analysis/requirements.txt
```

The two token audits also need the pinned tokenizer and transformers; their READMEs say how to reproduce them.
