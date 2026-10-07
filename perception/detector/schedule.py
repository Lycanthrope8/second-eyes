"""Cost-balanced slice schedules from a profiling run (D89).

A profiling pass dispatches the preprocessing, then each of the runtime's scheduling steps alone, each followed by
empty frames, then the readbacks (detector.profile). With per-frame GPU times (gpu.second, matched by gpu_frames), a
step's cost is its frame's GPU time minus the median of the pass's empty frames; the median over passes is kept, with
its spread. Contiguous slices are then packed greedily against a per-frame budget (greedy packing gives the fewest
frames for a contiguous partition under a maximum per frame): the first frame carries the preprocessing, the last the
readback requests, graph order is kept, and a step that alone exceeds the budget gets a frame of its own and is
reported, never hidden. Costs are measured, not estimated; layer costs may not add exactly when grouped, so every
schedule is validated on the headset before one is frozen.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from . import gpu_frames
from . import package as P

FRAME_MS = 1000.0 / 72.0


def step_costs(events) -> dict:
    gpu = gpu_frames.frame_gpu_times(events)
    if not gpu["by_frame"]:
        raise SystemExit("no per-frame GPU times: " + (gpu["reason"] or "unknown") + ". Shape-based estimates are not "
                         "built; report this before designing a schedule (D89).")
    g = gpu["by_frame"]
    passes = [e["data"] for e in events if e["ev"] == "detector.profile" and e["data"]["completed"]]
    if not passes:
        raise SystemExit("no completed detector.profile passes in this run")
    n = len(passes[0]["step_frames"])
    per_step = [[] for _ in range(n)]
    pre, readback, base_all, waits = [], [], [], []
    for p in passes:
        marked = {p["pre_frame"], p["readback_frame"], *p["step_frames"]}
        empty = [g[f] for f in range(p["pre_frame"], p["done_frame"] + 1) if f not in marked and f in g]
        if len(empty) < 5:
            continue
        base = statistics.median(empty)
        base_all.append(base)
        waits.append(p["done_frame"] - p["readback_frame"])
        if p["pre_frame"] in g:
            pre.append(g[p["pre_frame"]] - base)
        if p["readback_frame"] in g:
            readback.append(g[p["readback_frame"]] - base)
        for i, f in enumerate(p["step_frames"][:n]):
            if f in g:
                per_step[i].append(g[f] - base)
    steps = []
    for i, xs in enumerate(per_step):
        if xs:
            q = statistics.quantiles(xs, n=4) if len(xs) >= 2 else [xs[0], xs[0], xs[0]]
            steps.append({"step": i, "cost_ms": max(0.0, statistics.median(xs)), "iqr_ms": q[2] - q[0], "n": len(xs)})
        else:
            steps.append({"step": i, "cost_ms": None, "iqr_ms": None, "n": 0})
    return {"passes": len(passes), "steps": steps, "baseline_ms": statistics.median(base_all),
            "pre_ms": max(0.0, statistics.median(pre)) if pre else 0.0,
            "readback_ms": max(0.0, statistics.median(readback)) if readback else 0.0,
            "readback_frames": statistics.median(waits) if waits else 3, "gpu_lag": gpu["lag"],
            "gpu_match_spread_ms": gpu["spread_ms"],
            "unmeasured": [s["step"] for s in steps if s["cost_ms"] is None]}


def pack(costs, budget, pre_ms=0.0, readback_ms=0.0) -> dict:
    """Contiguous slices (steps per frame) under `budget`, frame 0 carrying the preprocessing."""
    slices, loads, count, load = [], [], 0, pre_ms
    for c in costs:
        if load + c > budget and (count > 0 or load > 0):
            slices.append(count)
            loads.append(load)
            count, load = 0, 0.0
        count += 1
        load += c
    if load + readback_ms > budget and count > 1:
        slices.append(count - 1)
        loads.append(load - costs[-1])
        count, load = 1, costs[-1]
    slices.append(count)
    loads.append(load + readback_ms)
    return {"slices": slices, "frame_loads_ms": loads, "max_load_ms": max(loads),
            "over_budget_steps": [i for i, c in enumerate(costs) if c > budget]}


def build(events, budgets, profile_run: str) -> list:
    costs = step_costs(events)
    if costs["unmeasured"]:
        raise SystemExit(f"steps without a GPU time: {costs['unmeasured'][:20]}")
    layers = next((e["data"] for e in events if e["ev"] == "detector.layers"), None)
    load = next((e["data"] for e in events if e["ev"] == "detector.load"), None)
    if layers is None or load is None:
        raise SystemExit("the run has no detector.layers or detector.load event")
    if layers["count"] != len(costs["steps"]):
        raise SystemExit(f"{len(costs['steps'])} profiled steps for {layers['count']} runtime layers")
    c = [s["cost_ms"] for s in costs["steps"]]
    out = []
    for b in budgets:
        p = pack(c, b, costs["pre_ms"], costs["readback_ms"])
        frames = len(p["slices"])
        sid = f"yolox-nano-gpu-{b:.1f}ms-{profile_run}"
        out.append({"format_version": 1, "record_type": "detector_schedule", "schedule_id": sid,
                    "package_id": load["package_id"], "model_sha256": load["model_sha256"],
                    "runtime": "com.unity.ai.inference 2.2.1", "backend": load["backend"],
                    "layer_count": layers["count"], "layer_types_sha256": layers["types_sha256"],
                    "budget_ms": b, "slices": p["slices"],
                    "predicted": {"frames": frames, "readback_frames": costs["readback_frames"],
                                  "latency_ms": (frames + costs["readback_frames"]) * FRAME_MS,
                                  "max_frame_load_ms": p["max_load_ms"], "baseline_frame_ms": costs["baseline_ms"],
                                  "headroom_ms": FRAME_MS - costs["baseline_ms"]},
                    "over_budget_steps": [{"step": i, "cost_ms": c[i]} for i in p["over_budget_steps"]],
                    "source": {"profile_run": profile_run, "passes": costs["passes"],
                               "cost": "per-frame GPU time (FrameTimingManager) minus the pass's empty-frame median, "
                                       "median over passes", "pre_ms": costs["pre_ms"],
                               "readback_ms": costs["readback_ms"], "gpu_lag_frames": costs["gpu_lag"]}})
    return out, costs


def write(schedules, out_dir) -> list:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for s in schedules:
        path = out_dir / f"{s['schedule_id']}.json"
        path.write_text(json.dumps(s, indent=1) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def report(schedules, costs) -> str:
    top = sorted(costs["steps"], key=lambda s: -s["cost_ms"])[:8]
    lines = [f"{costs['passes']} passes; empty-frame GPU time {costs['baseline_ms']:.2f} ms (headroom "
             f"{FRAME_MS - costs['baseline_ms']:.2f} ms at 72 Hz); preprocessing {costs['pre_ms']:.2f} ms; readback "
             f"request {costs['readback_ms']:.2f} ms over {costs['readback_frames']} frames; total of all steps "
             f"{sum(s['cost_ms'] for s in costs['steps']):.1f} ms; GPU timings matched with lag "
             f"{costs['gpu_lag']} frames (spread {costs['gpu_match_spread_ms']:.2f} ms)",
             "costliest steps: " + ", ".join(f"#{s['step']} {s['cost_ms']:.2f} ms (IQR {s['iqr_ms']:.2f})" for s in top),
             "| budget per frame | frames | predicted latency | heaviest frame | steps over budget |",
             "|---|---|---|---|---|"]
    for s in schedules:
        pr = s["predicted"]
        lines.append(f"| {s['budget_ms']:.1f} ms | {pr['frames']} | {pr['latency_ms']:.0f} ms | "
                     f"{pr['max_frame_load_ms']:.2f} ms | {len(s['over_budget_steps'])} |")
    return "\n".join(lines)


SCHEDULE_DIR = P.REPO / "perception" / "detector" / "schedules"
