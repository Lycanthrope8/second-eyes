"""The detector's phases against OVR Metrics' stale frames (A1.10c tuning and confirmation, A1.10d; D88).

    python analysis/detector_phases.py <run ID or raw folder> [--json] [--refresh-hz 72] [--limit-gib 5.75]

OVR Metrics is the authoritative stale-frame measurement: one row per second with the compositor's stale_frame_count.
Its rows get absolute times from the recording's start (the CSV's name, to the second), as in analysis/profile.py.
This script then calibrates the offset between that clock and the event log's UTC clock: it shifts the one-second
buckets in 0.05 s steps and keeps the shift whose stale counts best match the app's own over-budget frames
(frame.second). The offset, the fit (Pearson r) and whether it was used are printed. With too little to match, the
file-name clock is used unchanged and the report says so.

Each time the detector was loaded (detector.load to detector.unload) is one segment, with its settings. For each
segment, and for its scan blocks (detector.scan hover start to end) and scans:
- stale-frame rate: stale frames / (refresh rate x one-second buckets), the denominator used in every condition;
- the same over the union of buckets that overlap an inference (started_utc_us to completion). This is a coarse
  proxy: one request per second does not mean one inference per bucket, an inference can cross a bucket boundary,
  and other app work also makes stale frames. It is not the exact inference-interval criterion (O26);
- stale frames per completed keyframe in those buckets: an aggregate association, not attribution;
- app frame times (frame.second), labelled as the app's: over-budget share, worst interval, longest over-budget run;
- keyframes requested, completed and rejected; capture-to-result latency (started to completion);
- peak app PSS and minimum available headset memory; detector results logged after its release (must be 0).
The acceptance lines apply D88's criteria. Standard library only; it reuses analysis/profile.py's readers.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import profile as P  # noqa: E402  (analysis/profile.py)

REPO = HERE.parent
sys.path.insert(1, str(REPO))
from perception.detector import gpu_frames  # noqa: E402
SESSION_LOG = re.compile(r"^[0-9]{8}T[0-9]{6}Z_[0-9A-Za-z]+[.]jsonl$")


def pct(values, q):
    v = sorted(values)
    if not v:
        return None
    return v[max(0, min(len(v) - 1, int(round(q * (len(v) - 1)))))]


def raw_folder(target: str) -> Path:
    p = Path(target)
    if p.is_dir():
        return p
    raw = REPO / "runs" / target / "raw"
    if not raw.is_dir():
        raise SystemExit(f"no raw folder {raw}")
    return raw


def read_events(raw: Path) -> list:
    events = []
    for path in sorted(p for p in raw.glob("*.jsonl") if SESSION_LOG.match(p.name)):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
    events.sort(key=lambda e: e["utc_us"])
    return events


def over_budget_times(events) -> list:
    out = []
    for e in events:
        if e["ev"] == "frame.second":
            end = e["utc_us"] / 1e6
            start = end - e["data"]["window_ms"] / 1000.0
            out += [start + ms / 1000.0 for ms in e["data"]["over_at_ms"]]
    return sorted(out)


def pearson(x, y):
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return statistics.correlation(x, y)


def calibrate(rows, events) -> dict:
    """The shift (seconds) to add to the OVR rows' times to put them on the event log's clock.

    With clock.sync pulses (deliberate stalls at logged times, phases spread across the second), only the rows around
    the pulses are fitted against the pulses' midpoints: a designed marker. Otherwise the app's over-budget frames are
    the predictor (incidental). Either way every offset whose fit is within 0.01 of the best is reported, and the
    phase figures are recomputed across that whole range (D89)."""
    pulses = [((e["data"]["start_utc_us"] + e["data"]["end_utc_us"]) / 2e6) for e in events if e["ev"] == "clock.sync"]
    if len(pulses) >= 4:
        lo, hi = min(pulses) - 3, max(pulses) + 3
        res = _fit([r for r in rows if lo <= r["_t"] <= hi + 2], pulses, "clock.sync pulses")
        if res["calibrated"]:
            return res
    over = over_budget_times(events)
    t0, t1 = events[0]["utc_us"] / 1e6, events[-1]["utc_us"] / 1e6
    return _fit([r for r in rows if t0 <= r["_t"] <= t1], over, "incidental over-budget app frames")


def _fit(inside, marks, method) -> dict:
    over = marks
    stale = [P.number(r, "stale_frame_count") or 0.0 for r in inside]
    best = {"offset_s": 0.0, "r": None, "calibrated": False, "over_budget_frames": len(over), "buckets": len(inside),
            "offset_range_s": None, "method": method}
    if len(over) < 5 or sum(stale) == 0:
        return best
    scores = []
    for k in range(-30, 31):
        d = round(k * 0.05, 2)
        counts = [sum(1 for t in over if r["_t"] - 1 + d <= t < r["_t"] + d) for r in inside]
        scores.append((d, pearson(counts, stale)))
    valid = [(d, r) for d, r in scores if r is not None]
    if not valid:
        return best
    top = max(r for _, r in valid)
    plateau = [d for d, r in valid if r >= top - 0.01]
    best.update(r=top, offset_range_s=(min(plateau), max(plateau)),
                offset_s=round((min(plateau) + max(plateau)) / 2, 3), calibrated=top >= 0.3)
    if not best["calibrated"]:
        best["offset_s"], best["offset_range_s"] = 0.0, None
    return best


def offsets_of(cal) -> list:
    """Every offset in the calibration's equally good range, 0.05 s apart (just the offset when there is none)."""
    if not cal.get("offset_range_s"):
        return [cal["offset_s"]]
    a, b = cal["offset_range_s"]
    n = int(round((b - a) / 0.05))
    return [round(a + k * 0.05, 3) for k in range(n + 1)]


def segments(events) -> list:
    """One per detector load: (start, end, settings), times in UTC seconds."""
    out, cur = [], None
    for e in events:
        t = e["utc_us"] / 1e6
        if e["ev"] == "detector.load":
            cur = {"start": t, "end": None, "settings": e["data"]}
        elif e["ev"] == "detector.unload" and cur is not None:
            cur["end"] = t
            cur["release"] = e["data"]
            out.append(cur)
            cur = None
    if cur is not None:
        cur["end"] = events[-1]["utc_us"] / 1e6
        cur["release"] = None
        out.append(cur)
    return out


def union_rate(rows, offset, intervals, refresh, any_overlap):
    """Stale frames and refreshes over buckets that overlap the intervals (any overlap, or at least half a second)."""
    stale = buckets = 0.0
    for r in rows:
        a, b = r["_t"] - 1 + offset, r["_t"] + offset
        ov = sum(max(0.0, min(b, y) - max(a, x)) for x, y in intervals)
        if (any_overlap and ov > 0) or (not any_overlap and ov >= 0.5):
            stale += P.number(r, "stale_frame_count") or 0.0
            buckets += 1
    return {"stale": stale, "buckets": int(buckets), "refreshes": refresh * buckets,
            "rate": (stale / (refresh * buckets)) if buckets else None}


def phase_report(rows, offsets, events, window, refresh, gpu=None) -> dict:
    """The phase's figures at the middle offset, with the stale rates' range across all equally good offsets."""
    offsets = offsets if isinstance(offsets, list) else [offsets]
    mid = offsets[len(offsets) // 2]
    rep = _phase_at(rows, mid, events, window, refresh)
    for key in ("stale_whole", "stale_inference_buckets"):
        rates = [_phase_at(rows, o, events, window, refresh)[key]["rate"] for o in offsets]
        rates = [x for x in rates if x is not None]
        rep[key]["range"] = (min(rates), max(rates)) if rates else None
    if gpu:
        rep["gpu_stages"] = gpu_stages(gpu, events, window, refresh)
    return rep


def gpu_stages(gpu, events, window, refresh) -> dict:
    """Frames whose app GPU time exceeds one refresh interval, by inference stage (frame-level, no clock alignment)."""
    a, b = window
    limit = 1000.0 / refresh
    stage = {}
    for e in events:
        d = e["data"]
        if e["ev"] != "detector.result" or d.get("path") != "camera" or not (a <= e["utc_us"] / 1e6 <= b):
            continue
        if not d.get("start_frame") or not d.get("end_frame"):
            continue
        stage[d["start_frame"]] = "preprocessing"
        for f in range(d["start_frame"] + 1, d["schedule_end_frame"] + 1):
            stage[f] = "scheduling"
        for f in range(d["schedule_end_frame"] + 1, d["end_frame"] + 1):
            stage[f] = "readback"
    frames = set()
    for e in events:
        if e["ev"] == "gpu.second" and a <= e["utc_us"] / 1e6 <= b:
            frames.update(e["data"]["frames"])
    out = {k: {"frames": 0, "over": 0, "max_ms": None} for k in ("preprocessing", "scheduling", "readback", "outside")}
    for f in frames:
        if f not in gpu:
            continue
        k = stage.get(f, "outside")
        g = gpu[f]
        out[k]["frames"] += 1
        out[k]["over"] += g > limit
        out[k]["max_ms"] = g if out[k]["max_ms"] is None else max(out[k]["max_ms"], g)
    return out


def _phase_at(rows, offset, events, window, refresh) -> dict:
    a, b = window
    res = [e for e in events if e["ev"] == "detector.result" and e["data"].get("path") == "camera"
           and a <= e["utc_us"] / 1e6 <= b]
    rejects = [e for e in events if e["ev"] == "detector.reject" and a <= e["utc_us"] / 1e6 <= b]
    infer = [((e["data"].get("started_utc_us") or e["utc_us"]) / 1e6, e["utc_us"] / 1e6) for e in res]
    done = [e for e in res if e["data"]["completed"]]
    lat = [(e["utc_us"] - e["data"]["started_utc_us"]) / 1000.0 for e in done if e["data"].get("started_utc_us")]
    whole = union_rate(rows, offset, [window], refresh, any_overlap=False)
    during = union_rate(rows, offset, infer, refresh, any_overlap=True)
    frames = [e["data"] for e in events if e["ev"] == "frame.second" and a <= e["utc_us"] / 1e6 <= b]
    nframes = sum(f["frames"] for f in frames)
    mem = [r for r in rows if a <= r["_t"] + offset <= b]
    return {"window_s": b - a, "stale_whole": whole, "stale_inference_buckets": during,
            "stale_per_keyframe": (during["stale"] / len(done)) if done else None,
            "keyframes": {"results": len(res), "completed": len(done), "rejected": len(rejects),
                          "requested": len(res) + len(rejects)},
            "latency_ms": {"median": pct(lat, .5), "p95": pct(lat, .95), "max": max(lat) if lat else None, "n": len(lat)},
            "app_frames": {"frames": nframes, "over_budget": sum(f["over_budget"] for f in frames),
                           "over_share": (sum(f["over_budget"] for f in frames) / nframes) if nframes else None,
                           "max_ms": max((f["max_ms"] or 0 for f in frames), default=None),
                           "longest_over_run": max((f["longest_over_run"] for f in frames), default=None)},
            "memory": {"app_pss_max_mb": max(P.column(mem, "app_pss_MB"), default=None),
                       "available_min_mb": min(P.column(mem, "available_memory_MB"), default=None)}}


def analyze(raw: Path, refresh: float, limit_gib: float) -> dict:
    header, samples = P.read_sampler(raw)
    rows, used = P.read_metrics(raw, header, samples)
    events = read_events(raw)
    if not events:
        raise SystemExit(f"no session logs in {raw}")
    cal = calibrate(rows, events)
    off = offsets_of(cal)
    gm = gpu_frames.frame_gpu_times(events)
    gpu = gm["by_frame"] or None
    segs = []
    for s in segments(events):
        window = (s["start"], s["end"])
        scans, hovers, starts = [], [], {}
        for e in events:
            if e["ev"] != "detector.scan" or not (window[0] <= e["utc_us"] / 1e6 <= window[1]):
                continue
            d, t = e["data"], e["utc_us"] / 1e6
            key = (d["scan"], d["hover"])
            if d["state"] == "start":
                starts[key] = t
            elif key in starts:
                (hovers if d["hover"] is not None else scans).append((starts.pop(key), t, d))
        after = [e for e in events if e["ev"] == "detector.result" and e["utc_us"] / 1e6 > window[1]
                 and s["release"] is not None]
        later_loads = [x["start"] for x in segments(events) if x["start"] > window[1]]
        if later_loads:
            after = [e for e in after if e["utc_us"] / 1e6 < min(later_loads)]
        seg = {"settings": s["settings"], "release": s["release"], "window": window,
               "loaded": phase_report(rows, off, events, window, refresh, gpu),
               "scans": [dict(phase_report(rows, off, events, (x, y), refresh), scan=d["scan"], counts=d)
                         for x, y, d in scans],
               "hover_blocks": [dict(phase_report(rows, off, events, (x, y), refresh), scan=d["scan"], hover=d["hover"],
                                     counts=d) for x, y, d in hovers],
               "results_after_release": len(after)}
        if hovers:
            seg["all_hover_blocks"] = union_rate(rows, off[len(off) // 2], [(x, y) for x, y, _ in hovers], refresh,
                                                 any_overlap=False)
        if scans:
            seg["all_scans"] = phase_report(rows, off, events, (scans[0][0], scans[-1][1]), refresh, gpu)
        segs.append(seg)
    limit_mb = limit_gib * 1024
    return {"raw": str(raw), "ovr_files": used, "ovr_rows": len(rows), "calibration": cal, "refresh_hz": refresh,
            "memory_limit_mb": limit_mb, "segments": segs, "gpu_timing": {k: gm[k] for k in ("lag", "spread_ms", "entries", "reason")}}


def rate(x):
    if x is None or x.get("rate") is None:
        return "-"
    s = f"{100 * x['rate']:.2f}% ({x['stale']:.0f} of {x['refreshes']:.0f})"
    r = x.get("range")
    if r and abs(r[1] - r[0]) > 1e-12:
        s += f", {100 * r[0]:.2f}-{100 * r[1]:.2f}% across the offset range"
    return s


def worst(x):
    r = x.get("range") if x else None
    return r[1] if r else (x or {}).get("rate")


def fmt(v, d=0):
    return "-" if v is None else f"{v:.{d}f}"


def print_report(rep: dict) -> None:
    cal = rep["calibration"]
    print(f"{rep['raw']}: {rep['ovr_rows']} OVR Metrics rows ({', '.join(rep['ovr_files']) or 'none'})")
    g = rep["gpu_timing"]
    print("per-frame GPU times: " + (f"{g['entries']} timings, matched with a lag of {g['lag']} frames (spread "
                                     f"{g['spread_ms']:.2f} ms)" if g["lag"] is not None else f"none ({g['reason']})"))
    print(f"clock: offset {cal['offset_s']:+.2f} s "
          + (f"calibrated by {cal['method']} (r = {cal['r']:.2f} over {cal['buckets']} buckets, {cal['over_budget_frames']} "
             f"marks; equally good from {cal['offset_range_s'][0]:+.2f} to {cal['offset_range_s'][1]:+.2f} s, and the "
             "overlap figures are given across that range)"
             if cal["calibrated"] else
             f"NOT calibrated (best r = {fmt(cal['r'], 2)}; {cal['over_budget_frames']} over-budget app frames): the "
             "file-name clock is used, so bucket alignment is uncertain by up to a second"))
    for k, s in enumerate(rep["segments"], 1):
        st, L = s["settings"], s["loaded"]
        print(f"== segment {k}: {st.get('backend')}, {st.get('steps_per_frame', '?')} steps/frame "
              f"({st.get('layer_count', '?')} layers), mode {st.get('mode', '?')}, {fmt(st.get('keyframe_rate_hz'), 1)} Hz; "
              f"loaded {L['window_s']:.0f} s; load {fmt(st.get('load_ms'))} ms, warm-up {fmt(st.get('warmup_ms'))} ms; "
              + (f"release {fmt(s['release'].get('release_ms'), 1)} ms, in flight {s['release'].get('in_flight')}"
                 if s["release"] else "not released in this log"))
        for name, ph in (("loaded (whole)", L), ("all scans", s.get("all_scans"))):
            if ph is None:
                continue
            kf, lat, af, mem = ph["keyframes"], ph["latency_ms"], ph["app_frames"], ph["memory"]
            print(f"  {name}: stale {rate(ph['stale_whole'])}; buckets overlapping inference {rate(ph['stale_inference_buckets'])}; "
                  f"stale per completed keyframe {fmt(ph['stale_per_keyframe'], 2)}")
            print(f"    keyframes requested {kf['requested']}, completed {kf['completed']}, rejected {kf['rejected']}; "
                  f"capture-to-result latency median {fmt(lat['median'])} ms, p95 {fmt(lat['p95'])}, max {fmt(lat['max'])}")
            print(f"    app frame times (the app's, not the compositor's): over budget {af['over_budget']} of {af['frames']}"
                  f" ({fmt(None if af['over_share'] is None else 100 * af['over_share'], 2)}%), worst {fmt(af['max_ms'], 1)} ms, "
                  f"longest over-budget run {af['longest_over_run']}")
            print(f"    memory: app PSS max {fmt(mem['app_pss_max_mb'])} MB, headset available min {fmt(mem['available_min_mb'])} MB")
            if ph.get("gpu_stages"):
                gs = ph["gpu_stages"]
                print("    frames with app GPU time over one refresh interval, by stage (frame-level): " + "; ".join(
                    f"{k} {v['over']} of {v['frames']} (worst {fmt(v['max_ms'], 1)} ms)" for k, v in gs.items()))
        if "all_hover_blocks" in s:
            print(f"  hover blocks only (inference active, excluding repositioning): stale {rate(s['all_hover_blocks'])}")
        for h in s["hover_blocks"]:
            print(f"    scan {h['scan']}.{h['hover']}: {h['window_s']:.1f} s, keyframes {h['counts']['completed']}/"
                  f"{h['counts']['requested']} (rejected {h['counts']['rejected']}), stale {rate(h['stale_whole'])}, "
                  f"p95 {fmt(h['latency_ms']['p95'])} ms")
        ph = s.get("all_scans") or L
        whole = worst(ph["stale_whole"])
        buckets = worst(ph["stale_inference_buckets"])
        p95 = ph["latency_ms"]["p95"]
        peak = ph["memory"]["app_pss_max_mb"]
        avail = ph["memory"]["available_min_mb"]
        print("  acceptance (D88; overlap figures judged at the least favourable offset in the range, D89):")
        print(f"    stale frames < 1% over the operational phase: {verdict(whole is not None and whole < 0.01)} ({rate(ph['stale_whole'])})")
        print(f"    stale frames < 1% over buckets overlapping inference (proxy, not the exact interval criterion; O26): "
              f"{verdict(buckets is not None and buckets < 0.01)} ({rate(ph['stale_inference_buckets'])})")
        print(f"    capture-to-result p95 <= 500 ms: {verdict(p95 is not None and p95 <= 500)} ({fmt(p95)} ms)")
        print(f"    peak app memory >= 1 GiB below the {rep['memory_limit_mb']:.0f} MB limit: "
              f"{verdict(peak is not None and peak <= rep['memory_limit_mb'] - 1024)} ({fmt(peak)} MB)")
        print(f"    headset available memory >= 1 GiB: {verdict(avail is not None and avail >= 1024)} ({fmt(avail)} MB)")
        print(f"    no detector result after release: {verdict(s['results_after_release'] == 0)} "
              f"({s['results_after_release']})")


def verdict(ok: bool) -> str:
    return "yes" if ok else "NO"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target", help="a run ID, or a raw folder holding sampler.jsonl, the OVR Metrics CSV and the logs")
    ap.add_argument("--refresh-hz", type=float, default=72.0)
    ap.add_argument("--limit-gib", type=float, default=5.75, help="the app memory limit to keep 1 GiB below")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    rep = analyze(raw_folder(a.target), a.refresh_hz, a.limit_gib)
    if a.json:
        print(json.dumps(rep, indent=1, default=str))
    else:
        print_report(rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
