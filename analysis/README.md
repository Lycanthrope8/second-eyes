# analysis

Offline analysis of logged runs: reading event logs, aligning the Quest, drone and Vicon clocks, and plots. Never part of the deployed system. It also keeps A2.1d's two token audits, which measure model inputs rather than runs, each in its own folder with a README.

| Script | What it does | Since |
|---|---|---|
| `eventlog.py` | reads, checks and summarizes event logs (see `docs/logging.md`) | S0.3 |
| `profile.py` | summarizes and compares recordings from `tools/profile.py` (see `docs/profiling.md`) | A1.6a |
| `camera_frames.py` | the passthrough camera's frames per second and its toggles, from a run's session logs (see `docs/logging.md`) | A1.10a |
| `detector_runs.py` | the detector's loads, texture sources and per-path latency and rates, from a run's session logs; correctness is `python -m perception.detector parity` | A1.10c |
| `detector_phases.py` | OVR Metrics' stale frames per detector phase (loaded, scans, hover blocks, buckets overlapping inference), with the clock calibrated against the app's frame times, latency, rejections, memory and D88's acceptance lines | A1.10c (D88) |
| `scheduling_conditions.py` | A1.10d's conditions: command dispatch to scored answer, commands overlapping inference, stale frames per phase, residency, releases and memory per cycle | A1.10d |
| `A2.1d_token_sizing/` | the v0.2 token audit: renders the serializer proposal's two formats for four inputs and counts their tokens with the pinned tokenizer; kept unchanged as the indexed audit's reference | A2.1d |
| `A2.1d_indexed_token_sizing/` | the indexed relation-encoding audit, whose encoding the serializer adopted (D69): 64 verification checks, then token counts for eight inputs | A2.1d |

## Setup

Python 3.9 or newer, plus the packages in `requirements.txt`, installed into whichever Python environment you use for this repository:

```
pip install -r analysis/requirements.txt
```

The two token audits also need the pinned tokenizer and transformers; their READMEs say how to reproduce them.
