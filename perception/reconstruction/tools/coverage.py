#!/usr/bin/env python3
"""Tracking coverage of a tracking run, in elapsed video time (D135).

    python tools/coverage.py --run RUN [--frames DIR]

Reports, from the tracker's per-frame log (tracking/frames.csv) and the poses it saved:
  - the tracked share of the whole processed recording;
  - the tracked share from the first successful map initialisation to the recording's end, with every later
    tracking-loss period in the denominator (it does not restart after a recovery or a new map);
  - the initialisation time and the longest tracking gap after it.

A frame counts as tracked when the tracker reported it Tracking and its pose was saved in a map (frames of a map that
stella reset by itself are reported separately as "tracked but discarded"). Each frame counts for the time until the
next processed frame. Times are the source video's own timestamps when the frame folder has source_times.csv (written
by `video_tools.py frames`, or added later with `video_tools.py source-times`); otherwise the frame list's nominal
times (frame index / frame rate), and the report says so. Keyframe counts play no part.
Writes runs/RUN/tracking/coverage.json and prints a summary."""
import argparse, csv, json, os, statistics, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def _key(t):
    return round(float(t), 4)


def read_frames_csv(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({"index": int(r["index"]), "t": float(r["t"]), "state": r["state"].strip(),
                         "map": int(r["map"]) if r.get("map") not in (None, "") else None})
    rows.sort(key=lambda r: r["index"])
    return rows


def _tum_times(path):
    out = set()
    if Path(path).is_file():
        for line in open(path):
            if line.strip() and not line.startswith("#"):
                out.add(_key(line.split()[0]))
    return out


def saved_pose_times(tracking_dir):
    """Times of frames whose pose was saved, over all saved maps (or the single trajectory of an older run)."""
    d = Path(tracking_dir)
    maps = []
    if (d / "maps.json").is_file():
        maps = json.loads((d / "maps.json").read_text())
        times = set()
        for m in maps:
            if m.get("status") == "saved" and m.get("dir"):
                times |= _tum_times(d / m["dir"] / "frame_trajectory.txt")
        return times, maps
    return _tum_times(d / "frame_trajectory.txt"), maps


def read_source_times(frames_dir):
    """{nominal time key: source time} from source_times.csv, or None."""
    p = Path(frames_dir) / "source_times.csv" if frames_dir else None
    if not p or not p.is_file():
        return None
    with open(p, newline="") as f:
        return {_key(r["nominal_s"]): float(r["source_s"]) for r in csv.DictReader(f)}


def compute(rows, saved, maps=(), source=None, rgb_count=None):
    if not rows:
        raise ValueError("no frames in the log")
    use_source = source is not None and all(_key(r["t"]) in source for r in rows)
    ts = [source[_key(r["t"])] if use_source else r["t"] for r in rows]
    gaps = [b - a for a, b in zip(ts, ts[1:]) if b > a]
    step = statistics.median(gaps) if gaps else 0.0
    dur = [max(0.0, b - a) for a, b in zip(ts, ts[1:])] + [step]
    start, end = ts[0], ts[-1] + step
    span = end - start
    tracked = [r["state"] == "Tracking" and _key(r["t"]) in saved for r in rows]
    by_state = {}
    for r, d in zip(rows, dur):
        by_state[r["state"]] = by_state.get(r["state"], 0.0) + d
    discarded = sum(d for r, d, k in zip(rows, dur, tracked) if r["state"] == "Tracking" and not k)
    tracked_s = sum(d for d, k in zip(dur, tracked) if k)
    out = {"timestamps": "source" if use_source else "nominal (frame index / frame rate)",
           "processed": {"first_index": rows[0]["index"], "last_index": rows[-1]["index"], "frames": len(rows),
                         "start_s": round(start, 3), "end_s": round(end, 3), "span_s": round(span, 3),
                         "whole_frame_list": rgb_count is not None and rows[0]["index"] == 0 and rows[-1]["index"] == rgb_count - 1},
           "tracked_s": round(tracked_s, 3), "tracked_share_whole": round(tracked_s / span, 4) if span > 0 else 0.0,
           "time_by_state_s": {k: round(v, 3) for k, v in sorted(by_state.items())},
           "tracked_but_discarded_s": round(discarded, 3),
           "initialization": None, "tracked_share_from_init": 0.0, "from_init_span_s": 0.0,
           "longest_gap_after_init": None, "gaps_after_init": 0, "untracked_after_init_s": 0.0}
    if any(tracked):
        i0 = tracked.index(True)
        t0 = ts[i0]
        after = sum(d for d, k in zip(dur[i0:], tracked[i0:]) if k)
        out["initialization"] = {"t_s": round(t0, 3), "index": rows[i0]["index"], "share_of_recording_before": round((t0 - start) / span, 4)}
        out["from_init_span_s"] = round(end - t0, 3)
        out["tracked_share_from_init"] = round(after / (end - t0), 4) if end > t0 else 0.0
        found, k = [], i0
        while k < len(rows):
            if tracked[k]:
                k += 1
                continue
            j = k
            while j < len(rows) and not tracked[j]:
                j += 1
            g_end = ts[j] if j < len(rows) else end
            found.append({"start_s": round(ts[k], 3), "end_s": round(g_end, 3), "duration_s": round(g_end - ts[k], 3),
                          "recovered": j < len(rows)})
            k = j
        out["gaps_after_init"] = len(found)
        out["untracked_after_init_s"] = round(sum(g["duration_s"] for g in found), 3)
        if found:
            out["longest_gap_after_init"] = max(found, key=lambda g: g["duration_s"])
    per_map = []
    for m in maps or []:
        mid = m.get("map")
        ms = sum(d for r, d, k in zip(rows, dur, tracked) if k and r.get("map") == mid)
        per_map.append(dict(m, tracked_s=round(ms, 3)))
    out["maps"] = per_map
    return out


def summary_line(c):
    p = c["processed"]
    s = (f"coverage ({c['timestamps']} timestamps): whole {'recording' if p['whole_frame_list'] else 'processed span'} "
         f"{c['tracked_share_whole']:.3f} ({c['tracked_s']:.1f} of {p['span_s']:.1f} s)")
    if c["initialization"]:
        s += f"; from initialisation at {c['initialization']['t_s']:.2f} s: {c['tracked_share_from_init']:.3f}"
        g = c["longest_gap_after_init"]
        s += (f"; longest gap after it {g['start_s']:.2f}-{g['end_s']:.2f} s ({g['duration_s']:.1f} s"
              f"{'' if g['recovered'] else ', not recovered'}); {c['gaps_after_init']} gap(s)") if g else "; no gap after it"
    else:
        s += "; no map was ever initialised"
    if c["tracked_but_discarded_s"] > 0:
        s += f"; {c['tracked_but_discarded_s']:.1f} s tracked in maps stella discarded"
    return s


def for_run(run, frames_dir=None):
    d = HERE / "runs" / run / "tracking"
    if not (d / "frames.csv").is_file():
        sys.exit(f"{d / 'frames.csv'} not found")
    if frames_dir is None:
        man = HERE / "runs" / run / "manifest.json"
        if man.is_file():
            fr = json.loads(man.read_text()).get("frames")
            frames_dir = (HERE / fr) if fr and not os.path.isabs(fr) else fr
    rows = read_frames_csv(d / "frames.csv")
    saved, maps = saved_pose_times(d)
    rgb = Path(frames_dir) / "rgb.txt" if frames_dir else None
    count = sum(1 for l in open(rgb) if l.strip() and not l.startswith("#")) if rgb and rgb.is_file() else None
    c = compute(rows, saved, maps, read_source_times(frames_dir), count)
    (d / "coverage.json").write_text(json.dumps(c, indent=1))
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True); ap.add_argument("--frames", help="frame folder (default: from the run's manifest)")
    a = ap.parse_args()
    c = for_run(a.run, a.frames)
    print(summary_line(c))
    for m in c["maps"]:
        print(f"  map {m['map']}: {m['status']}, {m.get('tracked_s', 0):.1f} s tracked"
              + (f", {m['first_t']:.2f}-{m['last_t']:.2f} s" if m.get("first_t", -1) >= 0 else ""))


if __name__ == "__main__":
    main()
