"""Per-frame GPU times from the app's gpu.second events (D89), matched to the frames they belong to.

FrameTimingManager reports a frame a few frames after it ran. Each gpu.second entry pairs this frame's number and
Stopwatch timestamp with the latest completed frame's start timestamp (Unity's CPU timer) and GPU time. For each lag k,
`timing start - update time of frame (f - k)` is computed. When the two timers share an origin (both the monotonic
clock), the right lag is the one whose median lies within one frame before the update (a frame starts shortly before
its Update runs); otherwise the lag whose differences are most nearly constant is taken, which relies on frame-time
jitter, and the result says so. Each timing is then given to the frame whose update time is nearest. Repeated reports of
the same timing are counted once. Standard library only.
"""
from __future__ import annotations

import statistics


def frame_gpu_times(events) -> dict:
    """{'by_frame': {frame: gpu_ms}, 'lag': k, 'spread_ms': ..., 'entries': n, 'reason': None or why empty}."""
    timing = next((e["data"] for e in events if e["ev"] == "gpu.timing"), None)
    out = {"by_frame": {}, "lag": None, "spread_ms": None, "entries": 0, "reason": None}
    if timing is None:
        out["reason"] = "no gpu.timing event (GpuFrameTimes not in the scene)"
        return out
    fc, fs = timing["cpu_timer_frequency"], timing["stopwatch_frequency"]
    if not fc or not fs:
        out["reason"] = "the CPU timer frequency is 0"
        return out
    update, entries = {}, []
    for e in events:
        if e["ev"] != "gpu.second":
            continue
        d = e["data"]
        for f, u, t, g in zip(d["frames"], d["update_ticks"], d["timing_start"], d["gpu_ms"]):
            update[f] = u / fs
            if t and g is not None and g > 0:
                entries.append((f, t / fc, g))
    seen, unique = set(), []
    for f, t, g in entries:
        if t not in seen:
            seen.add(t)
            unique.append((f, t, g))
    out["entries"] = len(unique)
    if len(unique) < 10:
        out["reason"] = "fewer than 10 GPU timings (Frame Timing Stats off, or GPU times not reported on this platform)"
        return out
    fits = []
    for k in range(0, 9):
        res = [t - update[f - k] for f, t, _ in unique if (f - k) in update]
        if len(res) < 10:
            continue
        q = statistics.quantiles(res, n=4)
        fits.append((k, q[2] - q[0], statistics.median(res)))
    if not fits:
        out["reason"] = "no lag fits"
        return out
    frame = 1.0 / 72.0
    shared = [x for x in fits if -frame < x[2] <= 0.002]
    if shared:
        k, spread, offset = min(shared, key=lambda x: x[1])
        out["method"] = "shared clock: the frame starts within one frame before its update"
    else:
        k, spread, offset = min(fits, key=lambda x: x[1])
        out["method"] = "separate clocks: the lag with the most constant difference (relies on frame-time jitter)"
    frames = sorted(update)
    times = [update[f] for f in frames]
    import bisect
    by_frame = {}
    for _, t, g in unique:
        x = t - offset
        i = bisect.bisect_left(times, x)
        cands = [j for j in (i - 1, i) if 0 <= j < len(times)]
        j = min(cands, key=lambda j: abs(times[j] - x))
        by_frame[frames[j]] = g
    out.update(by_frame=by_frame, lag=k, spread_ms=spread * 1000.0)
    return out
