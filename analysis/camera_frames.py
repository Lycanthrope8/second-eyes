"""Summarize the passthrough camera's events in a run's session logs (A1.10a; events in docs/logging.md).

    python analysis/camera_frames.py <run ID or log.jsonl> [--include-before-run] [--since-s S] [--until-s U]

Given a run ID, it reads the session logs pulled into runs/<run ID>/raw/: files named <UTC start>_<session ID>.jsonl.
Other files there, such as tools/profile.py's sampler.jsonl, are not event logs and are left alone. Logs in
raw/before_run/ started before the run was created; like the other checks, it skips them unless
--include-before-run is given. --since-s and --until-s limit a summary to seconds S..U after the log's first line.

For each log it prints the permission, toggle and state timeline, the resolution and frame signal, and the distinct
frames per second the camera delivered. The second in which the camera started is only partly a playing second, so
the rate is computed over full playing seconds and those start seconds are counted separately; the three lowest full
seconds are listed with their times. Frame counts are null when the camera component offered no frame signal; then
only the playing time is reported. Standard library only.
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
TIMELINE = ("camera.permission", "camera.resolutions", "camera.toggle", "camera.state", "camera.playing")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target", help="a run ID (its pulled session logs) or one log file")
    ap.add_argument("--include-before-run", action="store_true", help="also read logs in raw/before_run/")
    ap.add_argument("--since-s", type=float, default=None)
    ap.add_argument("--until-s", type=float, default=None)
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
        early = sorted(p for p in (raw / "before_run").glob("*.jsonl") if SESSION_LOG.match(p.name))
        if a.include_before_run:
            files += early
        elif early:
            print(f"skipped {len(early)} log(s) in raw/before_run/ (started before the run was created): "
                  f"{', '.join(p.name for p in early)}; add --include-before-run to read them")
        if not files:
            print(f"no session logs in {raw}", file=sys.stderr)
            return 2
    status = 0
    for path in files:
        print(f"== {path.name}" + ("  (raw/before_run/)" if path.parent.name == "before_run" else ""))
        status = max(status, summarize(path, a))
    return status


def summarize(path: Path, a) -> int:
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError as e:
                    print(f"line {n} is not JSON: {e}", file=sys.stderr)
                    return 2
    if not rows or not all(isinstance(r, dict) and "mono_us" in r and "ev" in r for r in rows):
        print("not a session event log; skipped")
        return 0
    t0 = rows[0]["mono_us"]

    def at(r):
        return (r["mono_us"] - t0) / 1e6

    def inside(t):
        return (a.since_s is None or t >= a.since_s) and (a.until_s is None or t <= a.until_s)

    print("timeline (seconds after the log's first line):")
    for r in rows:
        if r["ev"] in TIMELINE or (r["ev"] == "error" and r.get("data", {}).get("where") == "camera"):
            print(f"  {at(r):9.1f}  {r['ev']:20} {json.dumps(r['data'])}")
    print(f"  ended with session.end: {'yes' if rows[-1]['ev'] == 'session.end' else 'NO (crash, kill or pulled mid-session)'}")
    seconds = [(at(r), r["data"]) for r in rows if r["ev"] == "camera.second"]
    sec = [(t, d) for t, d in seconds if inside(t)]
    playing = [(t, d) for t, d in sec if d["playing"]]
    print(f"seconds summarized: {len(sec)}; camera wanted on: {sum(d['on'] for _, d in sec)}; playing: {len(playing)}; "
          f"wanted on but not playing: {sum(d['on'] and not d['playing'] for _, d in sec)}")
    sizes = sorted({(d["width"], d["height"]) for _, d in playing})
    print(f"resolutions while playing: {', '.join(f'{w}x{h}' for w, h in sizes) or 'none'}")
    if not playing:
        print("no playing seconds in this window")
        return 0
    starts = {i for i, (t, d) in enumerate(seconds) if d["playing"] and (i == 0 or not seconds[i - 1][1]["playing"])}
    full = [(t, d) for i, (t, d) in enumerate(seconds) if d["playing"] and i not in starts and inside(t)
            and d["frames"] is not None and d["window_ms"] > 0]
    partial = sum(1 for i in starts if inside(seconds[i][0]))
    if all(d["frames"] is None for _, d in playing):
        print("frames per second: not available (the camera component offered no frame signal)")
        return 0
    if not full:
        print(f"no full playing seconds (start seconds: {partial})")
        return 0
    fps = sorted((d["frames"] * 1000.0 / d["window_ms"], t) for t, d in full)
    rates = [x for x, _ in fps]
    print(f"frames per second over {len(rates)} full playing seconds: median {statistics.median(rates):.1f}, "
          f"min {rates[0]:.1f}, max {rates[-1]:.1f}; seconds with no new frame: {sum(x == 0 for x in rates)}; "
          f"start seconds excluded: {partial}")
    print("lowest full seconds: " + ", ".join(f"{x:.1f} at {t:.0f} s" for x, t in fps[:3]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
