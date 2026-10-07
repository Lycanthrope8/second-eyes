"""Shape-based cost ESTIMATES for the runtime's scheduling steps (D89's branch for when GPU timing is unavailable).

FrameTimingManager reported no per-frame GPU times on the Quest in r044, so step costs cannot be measured. D89 allows
estimates in that case, labelled as such and validated on the headset. This module:

1. estimates each ONNX node's work from the graph's shapes (onnx shape inference): multiply-accumulates for
   convolutions, element counts for activations and arithmetic, moved elements for concatenation, slicing, resizing
   and reshaping, plus a fixed dispatch overhead per runtime layer (an assumption, stated in every report);
2. aligns the ONNX nodes to the runtime's layer list (`detector.layers`, in scheduling order): a node whose type
   matches the next runtime layer starts a new step; a node that does not (for example a Mul fused with the Sigmoid
   before it) is absorbed into the current step, so every node's work lands in exactly one step;
3. checks the estimates against what was observed: for fixed steps per frame, the number of frames whose estimated
   load exceeds a threshold H should track the stale frames per keyframe measured on the headset (r043, r044); H is
   fitted, and the fit is reported, not assumed;
4. partitions the steps into a given number of contiguous frames minimizing the heaviest frame (binary search on the
   load with greedy packing), the first frame carrying the preprocessing.
The units are arbitrary ("work"); only ratios matter. Every schedule written from here says it is estimated.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import package as P

OVERHEAD = 3.0e5          # assumed fixed work per runtime layer (dispatch and pipeline cost), in work units
PRE_WORK = 2 * 3 * 416 * 416 + 416 * 416 * 4     # letterbox pass and the texture-to-tensor conversion


def node_work(model_path=None) -> list:
    """[(op_type, work)] for every ONNX node in graph order, from inferred shapes."""
    import onnx
    from onnx import shape_inference
    m = onnx.load(str(model_path or (P.PACKAGE_DIR / P.MODEL_FILE)))
    m = shape_inference.infer_shapes(m)
    shapes = {}
    for v in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
        dims = [d.dim_value for d in v.type.tensor_type.shape.dim]
        if dims and all(d > 0 for d in dims):
            shapes[v.name] = dims
    for init in m.graph.initializer:
        shapes[init.name] = list(init.dims)
    size = lambda name: int(np.prod(shapes[name])) if name in shapes and shapes[name] else 0  # noqa: E731
    out = []
    for n in m.graph.node:
        o = size(n.output[0]) if n.output else 0
        ins = sum(size(i) for i in n.input if i)
        t = n.op_type
        if t == "Conv":
            w = shapes.get(n.input[1], [0, 0, 1, 1])
            group = next((a.i for a in n.attribute if a.name == "group"), 1)
            macs = o * (w[1] if len(w) > 1 else 1) * (w[2] if len(w) > 2 else 1) * (w[3] if len(w) > 3 else 1)
            work = macs + ins + o
            work = work if group == 1 else o * w[2] * w[3] + ins + o
        elif t in ("Sigmoid", "Exp"):
            work = 4 * o + ins
        elif t in ("Mul", "Add", "Sub"):
            work = o + ins
        elif t == "MaxPool":
            k = next((list(a.ints) for a in n.attribute if a.name == "kernel_shape"), [1, 1])
            work = o * k[0] * k[1] + ins
        elif t in ("ReduceMax", "ArgMax"):
            work = ins + o
        else:   # Concat, Slice, Gather, Reshape, Transpose, Resize: data movement
            work = 2 * max(o, 1)
        out.append((t, float(work)))
    return out


def align(nodes, runtime_types) -> dict:
    """Assign every ONNX node to a runtime step: a node matching the next step's type opens it; others are absorbed.

    Returns {'step_work': [...], 'absorbed': [(node index, op, step)], 'unmatched_steps': [...]}. A runtime type with no
    matching node is an error the caller reports; it never invents work."""
    work = [0.0] * len(runtime_types)
    absorbed, j = [], -1
    for i, (op, w) in enumerate(nodes):
        if j + 1 < len(runtime_types) and _same(op, runtime_types[j + 1]):
            j += 1
            work[j] += w
        elif j >= 0:
            work[j] += w
            absorbed.append((i, op, j))
        else:
            absorbed.append((i, op, 0))
            work[0] += w
    unmatched = list(range(j + 1, len(runtime_types)))
    return {"step_work": work, "absorbed": absorbed, "unmatched_steps": unmatched, "matched": j + 1}


def _same(op, layer_type) -> bool:
    a, b = op.lower(), layer_type.lower()
    if a == b:
        return True
    pairs = {("mul", "swish"), ("sigmoid", "swish"), ("reducemax", "reducemax"), ("argmax", "argmax"),
             ("conv", "conv2d"), ("maxpool", "maxpool2d"), ("resize", "upsample")}
    return (a, b) in pairs


def loads_fixed(step_work, steps_per_frame, pre=PRE_WORK) -> list:
    """Per-frame estimated loads for a fixed number of steps per frame (the first frame carries preprocessing)."""
    loads = []
    for k in range(0, len(step_work), steps_per_frame):
        loads.append(sum(step_work[k:k + steps_per_frame]) + (pre if k == 0 else 0.0))
    return loads


def partition(step_work, frames, pre=PRE_WORK) -> dict:
    """Contiguous slices over at most `frames` frames minimizing the heaviest frame's estimated load."""
    lo, hi = max(max(step_work), pre), sum(step_work) + pre
    best = None
    for _ in range(60):
        mid = (lo + hi) / 2
        s = _greedy(step_work, mid, pre)
        if s is not None and len(s["slices"]) <= frames:
            best, hi = s, mid
        else:
            lo = mid
    return best if best is not None else _greedy(step_work, hi, pre)


def _greedy(step_work, budget, pre):
    slices, loads, count, load = [], [], 0, pre
    for w in step_work:
        if load + w > budget and (count > 0 or load > 0):
            slices.append(count)
            loads.append(load)
            count, load = 0, 0.0
        if w > budget:
            return None
        count += 1
        load += w
    slices.append(count)
    loads.append(load)
    return {"slices": slices, "loads": loads, "max_load": max(loads)}


def fit_threshold(step_work, observed) -> dict:
    """Fit H so that frames over H per inference track observed stale frames per keyframe.

    observed: {steps_per_frame: stale per keyframe}. Returns H, the predicted counts and the worst error."""
    cand = sorted({x for s in observed for x in loads_fixed(step_work, s)})
    best = None
    for h in cand:
        pred = {s: sum(1 for x in loads_fixed(step_work, s) if x > h) for s in observed}
        err = sum((pred[s] - observed[s]) ** 2 for s in observed)
        if best is None or err < best[1]:
            best = (h, err, pred)
    h, err, pred = best
    return {"threshold": h, "predicted": pred, "observed": observed,
            "max_abs_error": max(abs(pred[s] - observed[s]) for s in observed)}


def report_lines(step_work, fit, schedules) -> list:
    total = sum(step_work) + PRE_WORK
    order = sorted(range(len(step_work)), key=lambda i: -step_work[i])[:8]
    lines = [f"ESTIMATES (shape-based, not measured; dispatch overhead assumed {OVERHEAD:.0e} work units per step): "
             f"{len(step_work)} steps, total work {total:.3g}; heaviest steps "
             + ", ".join(f"#{i} {100 * step_work[i] / total:.1f}%" for i in order),
             f"miss threshold fitted to the observed runs: frames above {100 * fit['threshold'] / total:.2f}% of an "
             f"inference's work; predicted stale per keyframe "
             + ", ".join(f"{s} steps {fit['predicted'][s]} (observed {fit['observed'][s]})" for s in sorted(fit["observed"]))
             + f"; worst error {fit['max_abs_error']:.2f}",
             "| frames | slices | heaviest frame (share of the inference) | frames above the fitted threshold |",
             "|---|---|---|---|"]
    for n, s in schedules:
        over = sum(1 for x in s["loads"] if x > fit["threshold"])
        lines.append(f"| {n} | {len(s['slices'])} | {100 * s['max_load'] / total:.2f}% | {over} |")
    return lines


def schedule_record(s, frames, load_event, layers_event, run, fit, total) -> dict:
    return {"format_version": 1, "record_type": "detector_schedule",
            "schedule_id": f"yolox-nano-gpu-est-{frames}f-{run}", "package_id": load_event["package_id"],
            "model_sha256": load_event["model_sha256"], "runtime": "com.unity.ai.inference 2.2.1",
            "backend": load_event["backend"], "layer_count": layers_event["count"],
            "layer_types_sha256": layers_event["types_sha256"], "budget_ms": 0.0, "slices": s["slices"],
            "estimated": True,
            "predicted": {"frames": len(s["slices"]), "readback_frames": 3,
                          "latency_ms": (len(s["slices"]) + 3) * 1000.0 / 72.0,
                          "heaviest_frame_share": s["max_load"] / total,
                          "frames_above_fitted_threshold": sum(1 for x in s["loads"] if x > fit["threshold"])},
            "over_budget_steps": [],
            "source": {"profile_run": run, "cost": "SHAPE-BASED ESTIMATE (D89): ONNX shape inference, multiply-"
                       "accumulates and moved elements per node, fixed dispatch overhead per step, nodes aligned to "
                       "the runtime layer list; miss threshold fitted to r043 and r044", "overhead_work": OVERHEAD,
                       "fit": {"threshold_share": fit["threshold"] / total, "predicted": fit["predicted"],
                               "observed": fit["observed"], "max_abs_error": fit["max_abs_error"]}}}


ESTIMATE_DIR = Path(__file__).resolve().parent / "schedules"
