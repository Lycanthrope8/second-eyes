"""Summarize the detector's events in a run's session logs (A1.10c; events in docs/logging.md, detector.*).

    python analysis/detector_runs.py <run ID or log.jsonl> [--include-before-run]

Given a run ID, it reads the session logs pulled into runs/<run ID>/raw/ (files named <UTC start>_<session ID>.jsonl;
logs in raw/before_run/ only with --include-before-run). For each log it prints the loads (backend, load and warm-up
times), the texture sources (format, sRGB, whether the shader re-encodes), then per path and backend: inferences,
latency (median, 90th percentile, maximum), scheduling and post-processing time, frames waited, candidates and
detections per inference. For camera frames it also gives inferences and detections per second over the span of
camera results. Correctness against the PC is `python -m perception.detector parity`. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SESSION_LOG = re.compile(r"^[0-9]{8}T[0-9]{6}Z_[0-9A-Za-z]+[.]jsonl$")


def pct(values, q):
    v = sorted(values)
    if not v:
        return None
    k = max(0, min(len(v) - 1, int(round(q * (len(v) - 1)))))
    return v[k]


def fmt(x, d=1):
    return "-" if x is None else f"{x:.{d}f}"


def summarize(events) -> dict:
    loads = [e["data"] for e in events if e.get("ev") == "detector.load"]
    sources = [e["data"] for e in events if e.get("ev") == "detector.source"]
    groups = {}
    for e in events:
        if e.get("ev") == "detector.result":
            d = e["data"]
            groups.setdefault((d["path"], d["backend"]), []).append((e["mono_us"], d))
    out = {"loads": loads, "sources": sources, "groups": {}}
    for (path, backend), rows in sorted(groups.items()):
        done = [d for _, d in rows if d["completed"]]
        lat = [d["latency_ms"] for d in done if d["latency_ms"] is not None]
        g = {"inferences": len(rows), "completed": len(done),
             "latency_ms": {"median": pct(lat, .5), "p90": pct(lat, .9), "max": max(lat) if lat else None},
             "schedule_ms": {"median": pct([d["schedule_ms"] for d in done], .5),
                             "p90": pct([d["schedule_ms"] for d in done], .9)},
             "postprocess_ms": {"median": pct([d["postprocess_ms"] for d in done], .5)},
             "frames_waited": {"median": pct([d["frames_waited"] for d in done], .5)},
             "candidates_mean": statistics.fmean([d["candidates"] for d in done]) if done else None,
             "detections_mean": statistics.fmean([d["detections_total"] for d in done]) if done else None}
        if path == "camera" and len(rows) > 1:
            span = (rows[-1][0] - rows[0][0]) / 1e6
            g["span_s"] = span
            g["inferences_per_s"] = (len(rows) - 1) / span if span > 0 else None
            g["detections_per_s"] = sum(d["detections_total"] for _, d in rows[1:]) / span if span > 0 else None
        out["groups"][f"{path}/{backend}"] = g
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target", help="a run ID (its pulled session logs) or one log file")
    ap.add_argument("--include-before-run", action="store_true")
    ap.add_argument("--json", action="store_true", help="print the summary as JSON")
    a = ap.parse_args(argv)
    target = Path(a.target)
    if target.is_file():
        files = [target]
    else:
        raw = REPO / "runs" / a.target / "raw"
        if not raw.is_dir():
            print(f"no log file and no run folder {raw}", file=sys.stderr)
            return 2
        files = sorted(p for p in raw.glob("*.jsonl") if SESSION_LOG.match(p.name))
        if a.include_before_run:
            files += sorted(p for p in (raw / "before_run").glob("*.jsonl") if SESSION_LOG.match(p.name))
        if not files:
            print(f"no session logs in {raw}", file=sys.stderr)
            return 2
    for path in files:
        events = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        s = summarize(events)
        if a.json:
            print(json.dumps({"log": path.name, **s}, indent=1))
            continue
        print(f"== {path.name}")
        for d in s["loads"]:
            print(f"load: {d['backend']}, load {fmt(d['load_ms'])} ms, warm-up {fmt(d['warmup_ms'])} ms, "
                  f"{d['color_space']} colour space")
        for d in s["sources"]:
            print(f"source {d['path']}: {d['graphics_format']} ({'sRGB' if d['srgb'] else 'not sRGB'}), "
                  f"{d['width']}x{d['height']}, re-encode {d['encode_srgb']}, flip {d['flip']}")
        for k, g in s["groups"].items():
            line = (f"{k}: {g['inferences']} inferences ({g['completed']} completed); latency median "
                    f"{fmt(g['latency_ms']['median'])} ms, p90 {fmt(g['latency_ms']['p90'])}, max "
                    f"{fmt(g['latency_ms']['max'])}; scheduling median {fmt(g['schedule_ms']['median'], 2)} ms, "
                    f"p90 {fmt(g['schedule_ms']['p90'], 2)}; post-processing median "
                    f"{fmt(g['postprocess_ms']['median'], 3)} ms; frames waited median {fmt(g['frames_waited']['median'], 0)}; "
                    f"candidates {fmt(g['candidates_mean'])}, detections {fmt(g['detections_mean'], 2)} per inference")
            if "inferences_per_s" in g:
                line += (f"; {fmt(g['inferences_per_s'], 2)} inferences/s and {fmt(g['detections_per_s'], 2)} "
                         f"detections/s over {fmt(g['span_s'])} s")
            print(line)
        if not s["groups"]:
            print("no detector results in this log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
