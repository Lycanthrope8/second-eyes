"""Checks of the detector's PC side (A1.10b, A1.10c): python perception/detector/test_detector.py

Needs perception/detector/requirements.txt. It checks the package files, reproduces the committed PC reference, pins
the suppression to YOLOX's own, exercises the parity rules on hand-made cases, and runs a synthetic headset log built
from the reference through the log schema, the parity command and analysis/detector_runs.py.
"""
from __future__ import annotations

import copy
import io
import json
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))

import numpy as np  # noqa: E402

from perception.detector import package as P, parity as Q, reference as R, yolox_ref  # noqa: E402

FAILED, COUNT = [], [0]


def check(name, ok, detail=""):
    COUNT[0] += 1
    print(("PASS  " if ok else "FAIL  ") + name + ("" if ok or not detail else f": {detail}"))
    if not ok:
        FAILED.append(name)


def synthetic_log(ref, path: Path, tweak=None):
    lines, seq = [], 0

    def ev(name, data):
        nonlocal seq
        lines.append(json.dumps({"seq": seq, "mono_us": 1_000_000 + seq * 50_000, "utc_us": 1_700_000_000_000_000 + seq,
                                 "ev": name, "data": data}))
        seq += 1
    ev("session.start", {"format": 1, "session_id": "test0001", "app_version": "test", "os_build": None})
    ev("detector.toggle", {"on": True, "source": "button_y"})
    ev("detector.load", {"package_id": ref["package_id"], "model_sha256": ref["model_sha256"], "backend": "gpu_compute",
                         "input_size": 416, "load_ms": 120.5, "warmup_ms": 900.0, "warmup_completed": True,
                         "color_space": "linear"})
    base = {"backend": "gpu_compute", "readback": "blocking", "completed": True, "error": None, "schedule_ms": 2.5, "latency_ms": 40.0,
            "postprocess_ms": 0.05, "frames_waited": 3, "candidates": 4, "snapshot": None}
    for e in ref["images"]:
        dets = [d for d in e["canvas_detections"] if d[1] >= P.SCORE_THRESHOLD]
        if tweak:
            dets = tweak(e["id"], copy.deepcopy(dets))
        ev("detector.result", dict(base, path="tensor", image_id=e["id"], width=416, height=416, ratio=1.0,
                                   detections_total=len(dets), detections=dets))
    for e in ref["images"]:
        if "full_detections" in e:
            ev("detector.source", {"path": "texture_srgb", "graphics_format": "R8G8B8A8_SRGB", "srgb": True,
                                   "width": e["width"], "height": e["height"], "encode_srgb": True, "flip": False})
            dets = [d for d in e["full_detections"] if d[1] >= P.SCORE_THRESHOLD]
            ev("detector.result", dict(base, path="texture_srgb", image_id=e["id"], width=e["width"], height=e["height"],
                                       ratio=e["ratio"], detections_total=len(dets), detections=dets))
    for k in range(5):
        ev("detector.result", dict(base, path="camera", image_id=None, width=1280, height=960, ratio=0.325,
                                   detections_total=1, detections=[[56, 0.5, 10.0, 20.0, 110.0, 220.0]]))
    ev("detector.toggle", {"on": False, "source": "button_y"})
    ev("detector.unload", {"backend": "gpu_compute"})
    ev("session.end", {})
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_phases():
    """analysis/detector_phases.py on a synthetic run whose OVR clock runs 0.4 s behind the event log."""
    print("-- the phase analysis on a synthetic run (OVR Metrics clock 0.4 s behind the log)")
    import detector_phases as DP
    import eventlog
    T0, SHIFT = 1_760_000_000, 0.4
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp)
        (raw / "sampler.jsonl").write_text(json.dumps({"device_start_s": T0, "device_tz": "+0000"}) + "\n"
                                           + "".join(json.dumps({"elapsed_s": k}) + "\n" for k in range(0, 121, 30)))
        starts = [T0 + 20 + k + (0.05 + 0.137 * k) % 0.9 for k in range(12)]   # varied sub-second phases
        over = [s + 0.1 for s in starts]                       # one over-budget frame per inference, true times
        rows = []
        for k in range(1, 121):
            a, b = T0 + k - 1 + SHIFT, T0 + k + SHIFT
            stale = sum(1 for x in over if a <= x < b)
            rows.append({"Time Stamp": k * 1000, "stale_frame_count": stale, "average_frame_rate": 72 - stale,
                         "app_pss_MB": 2600 + k, "available_memory_MB": 1800})
        import csv
        with open(raw / "com.secondeyes.quest#UnityPlayerGameActivity-20251009_085320.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        import datetime as dt
        start = dt.datetime.strptime("20251009085320", "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc).timestamp()
        assert start == T0, start
        lines, seq = [], 0

        def ev(t, name, data):
            nonlocal seq
            lines.append({"seq": seq, "mono_us": int((t - T0) * 1e6), "utc_us": int(t * 1e6), "ev": name, "data": data})
            seq += 1
        ev(T0 + 0.2, "session.start", {"format": 1, "session_id": "synth001", "app_version": "test", "os_build": None})
        ev(T0 + 10, "detector.load", {"package_id": "yolox_nano_coco.se1", "model_sha256": "5" * 64, "backend": "gpu_compute",
                                      "input_size": 416, "load_ms": 60.0, "warmup_ms": 400.0, "warmup_completed": True,
                                      "color_space": "linear", "layer_count": 300, "steps_per_frame": 16, "floor": 0.25,
                                      "mode": "scan", "keyframe_rate_hz": 1.0})
        ev(T0 + 19.9, "detector.scan", {"scan": 1, "hover": None, "state": "start", "requested": 0, "completed": 0, "rejected": 0})
        ev(T0 + 19.95, "detector.scan", {"scan": 1, "hover": 1, "state": "start", "requested": 0, "completed": 0, "rejected": 0})
        base = {"path": "camera", "image_id": None, "backend": "gpu_compute", "readback": "async", "completed": True,
                "error": None, "width": 1280, "height": 960, "ratio": 0.325, "schedule_ms": 250.0, "latency_ms": 300.0,
                "postprocess_ms": 0.02, "frames_waited": 3, "steps_per_frame": 16, "schedule_steps": 300,
                "schedule_frames": 19, "floor": 0.25, "scan": 1, "hover": 1, "capture_stamp": None, "pose": None,
                "candidates": 5, "detections_total": 1, "detections": [[56, 0.6, 1.0, 2.0, 3.0, 4.0]], "snapshot": None}
        for k in range(12):
            s = starts[k]
            ev(s + 0.3, "detector.result", dict(base, keyframe=k + 1, requested_utc_us=int(s * 1e6), started_utc_us=int(s * 1e6)))
        ev(T0 + 31.7, "detector.reject", {"reason": "busy", "scan": 1, "hover": 1, "requested_utc_us": int((T0 + 31.7) * 1e6)})
        ev(T0 + 32.0, "detector.scan", {"scan": 1, "hover": 1, "state": "end", "requested": 13, "completed": 12, "rejected": 1})
        ev(T0 + 32.1, "detector.scan", {"scan": 1, "hover": None, "state": "end", "requested": 13, "completed": 12, "rejected": 1})
        for j in range(0, 119):
            a = T0 + j
            ats = [int(round((x - a) * 1000)) for x in over if a <= x < a + 1]
            ev(a + 1, "frame.second", {"window_ms": 1000, "frames": 72, "budget_ms": 13.89, "over_budget": len(ats),
                                       "max_ms": 40.0 if ats else 14.0, "longest_over_run": 1 if ats else 0, "over_at_ms": ats})
        ev(T0 + 50, "detector.unload", {"backend": "gpu_compute", "release_ms": 12.5, "in_flight": False})
        ev(T0 + 119.5, "session.end", {})
        lines.sort(key=lambda e: e["utc_us"])
        for k, e in enumerate(lines):
            e["seq"] = k
        log = raw / "20251009T085320Z_synth001.jsonl"
        log.write_text("".join(json.dumps(e) + "\n" for e in lines))
        info = eventlog.check_file(log)
        problems = info.get("problems") or info.get("errors") or []
        check("the synthetic log with the D88 events validates against the schema", not problems, str(problems)[:300])
        rep = DP.analyze(raw, 72.0, 5.75)
        cal = rep["calibration"]
        check("the clock calibration recovers the 0.4 s shift (within 0.05 s) with a strong fit",
              cal["calibrated"] and abs(cal["offset_s"] - SHIFT) <= 0.051 and cal["r"] > 0.9, str(cal))
        seg = rep["segments"][0]
        scans = seg["all_scans"]
        check("12 completed keyframes, 1 busy rejection, 13 requested", scans["keyframes"] ==
              {"results": 12, "completed": 12, "rejected": 1, "requested": 13}, str(scans["keyframes"]))
        check("capture-to-result latency 300 ms at the median and p95", scans["latency_ms"]["median"] == 300.0
              and scans["latency_ms"]["p95"] == 300.0, str(scans["latency_ms"]))
        d = scans["stale_inference_buckets"]
        want = sum(1 for k in range(1, 121) if any(min(T0 + k + SHIFT, s + 0.3) - max(T0 + k - 1 + SHIFT, s) > 0 for s in starts))
        check(f"12 stale frames in the {want} buckets overlapping inference (at the true shift): 12 / {72 * want} refreshes",
              d["stale"] == 12 and d["buckets"] == want and abs(d["rate"] - 12 / (72 * want)) < 1e-12, f"{d} want {want}")
        check("one stale frame per completed keyframe", abs(scans["stale_per_keyframe"] - 1.0) < 1e-12)
        check("the release is recorded and nothing ran after it", seg["release"]["release_ms"] == 12.5
              and seg["results_after_release"] == 0)
        out = io.StringIO()
        with redirect_stdout(out):
            DP.print_report(rep)
        text = out.getvalue()
        check("the report states the proxy limitation and the acceptance lines", "proxy, not the exact interval criterion"
              in text and "capture-to-result p95 <= 500 ms: yes" in text and "stale frames < 1% over buckets" in text,
              text[-400:])


def gpu_events(frame_ms, base_tick, lag, fc=10_000_000, fs=1_000_000_000, offset_s=-0.002):
    """gpu.timing and gpu.second events for frames {frame: gpu_ms}, reported `lag` frames late; frame starts precede
    their Update by 2 ms on the same clock, and frame intervals jitter by up to 1 ms."""
    import random
    jitter = random.Random(11)
    frames = sorted(frame_ms)
    ev = [{"ev": "gpu.timing", "utc_us": 1, "data": {"feature_enabled": True, "cpu_timer_frequency": fc,
                                                       "gpu_timer_frequency": fc, "stopwatch_frequency": fs}}]
    upd, t = {}, base_tick
    for f in frames:
        upd[f] = t
        t += 1 / 72.0 + jitter.uniform(-0.001, 0.001)
    batch = {"frames": [], "update_ticks": [], "timing_start": [], "gpu_ms": [], "cpu_ms": []}
    for f in frames:
        batch["frames"].append(f)
        batch["update_ticks"].append(int(round(upd[f] * fs)))
        g = f - lag
        if g in upd:
            batch["timing_start"].append(int(round((upd[g] + offset_s) * fc)))
            batch["gpu_ms"].append(frame_ms[g])
        else:
            batch["timing_start"].append(0)
            batch["gpu_ms"].append(-1)
        batch["cpu_ms"].append(5.0)
        if len(batch["frames"]) == 72:
            ev.append({"ev": "gpu.second", "utc_us": 2 + f, "data": batch})
            batch = {k: [] for k in batch}
    if batch["frames"]:
        ev.append({"ev": "gpu.second", "utc_us": 2 + frames[-1], "data": batch})
    return ev


def check_scheduling():
    """gpu_frames' frame matching, the schedule builder and sync-pulse calibration (D89), on synthetic data."""
    print("-- cost-balanced scheduling (D89) on synthetic profiling data")
    from perception.detector import gpu_frames as G, schedule as S
    import random
    rnd = random.Random(5)
    n = 40
    true = [4.0 if i < 3 else (1.2 if i < 10 else 0.2) for i in range(n)]   # heavy early steps, light late ones
    true[20] = 7.5                                                           # one indivisible heavy step
    base, spacing, frame, frame_ms, events = 8.0, 3, 1000, {}, []
    for pss in range(1, 4):
        pre = frame
        frame_ms[frame] = base + 0.6
        frame += 1
        for _ in range(spacing):
            frame_ms[frame] = base + rnd.uniform(-0.05, 0.05)
            frame += 1
        steps = []
        for i in range(n):
            steps.append(frame)
            frame_ms[frame] = base + true[i] + rnd.uniform(-0.05, 0.05)
            frame += 1
            for _ in range(spacing):
                frame_ms[frame] = base + rnd.uniform(-0.05, 0.05)
                frame += 1
        rb = frame
        frame_ms[frame] = base + 0.3
        frame += 1
        for _ in range(3):
            frame_ms[frame] = base
            frame += 1
        done = frame - 1
        events.append({"ev": "detector.profile", "utc_us": 10 + pss, "data": {"pass": pss, "spacing": spacing, "completed": True,
                       "error": None, "pre_frame": pre, "readback_frame": rb, "done_frame": done, "step_frames": steps}})
        for _ in range(2 * spacing):
            frame_ms[frame] = base
            frame += 1
    gev = gpu_events(frame_ms, 5000.0, lag=3)
    m = G.frame_gpu_times(gev)
    check("frame matching finds the 3-frame lag and assigns every timing to its own frame",
          m["lag"] == 3 and all(abs(m["by_frame"][f] - frame_ms[f]) < 1e-9 for f in list(m["by_frame"])[:500]), str(m["lag"]))
    events += gev
    events.append({"ev": "detector.layers", "utc_us": 3, "data": {"count": n, "types_sha256": "a" * 64, "types": ["Conv"] * n}})
    events.append({"ev": "detector.load", "utc_us": 4, "data": {"package_id": "yolox_nano_coco.se1", "model_sha256": "5" * 64,
                                                                 "backend": "gpu_compute"}})
    events.sort(key=lambda e: e["utc_us"])
    costs = S.step_costs(events)
    est = [s["cost_ms"] for s in costs["steps"]]
    check("measured step costs match the truth within 0.1 ms, baseline 8 ms, preprocessing 0.6 ms",
          max(abs(a - b) for a, b in zip(est, true)) < 0.1 and abs(costs["baseline_ms"] - 8.0) < 0.06
          and abs(costs["pre_ms"] - 0.6) < 0.1, f"{max(abs(a - b) for a, b in zip(est, true)):.3f}")
    schedules, _ = S.build(events, [5.0], "synthetic")
    s = schedules[0]
    loads = S.pack(est, 5.0, costs["pre_ms"], costs["readback_ms"])["frame_loads_ms"]
    check("the 5 ms schedule covers every step in graph order", sum(s["slices"]) == n and all(k >= 0 for k in s["slices"]))
    over_frames = [i for i, x in enumerate(loads) if x > 5.0 + 1e-9]
    check("every frame stays within 5 ms except one holding only the 7.5 ms step, which is reported",
          len(over_frames) == 1 and s["slices"][over_frames[0]] == 1
          and [o["step"] for o in s["over_budget_steps"]] == [20],
          f"slices {s['slices']} loads {['%.2f' % x for x in loads]} over {s['over_budget_steps']}")
    check("the heavy early steps get thin slices and the light late ones are packed",
          s["slices"][0] <= 1 and max(s["slices"]) >= 10, str(s["slices"]))
    check("the schedule carries its identity: model, backend, layer count and layer-type hash",
          s["model_sha256"] == "5" * 64 and s["backend"] == "gpu_compute" and s["layer_count"] == n
          and s["layer_types_sha256"] == "a" * 64)
    print("-- clock calibration from sync pulses")
    import detector_phases as DP
    T0, SHIFT = 1_760_000_000, 0.4
    pulses = [T0 + 10 + 1.37 * k for k in range(8)]
    rows = []
    for k in range(1, 60):
        a, b = T0 + k - 1 + SHIFT, T0 + k + SHIFT
        rows.append({"_t": T0 + k, "stale_frame_count": 4 * sum(1 for x in pulses if a <= x + 0.03 < b)})
    evs = [{"ev": "clock.sync", "utc_us": int((x + 0.06) * 1e6), "data": {"pulse": i + 1, "start_utc_us": int(x * 1e6),
            "end_utc_us": int((x + 0.06) * 1e6), "stall_ms": 60.0}} for i, x in enumerate(pulses)]
    evs = [{"ev": "session.start", "utc_us": int(T0 * 1e6), "data": {}}] + evs + [{"ev": "session.end", "utc_us": int((T0 + 59) * 1e6), "data": {}}]
    cal = DP.calibrate(rows, evs)
    lo, hi = cal["offset_range_s"]
    check("eight pulses at spread phases pin the offset within 0.05 s, with a range of at most 0.15 s",
          cal["method"] == "clock.sync pulses" and lo - 0.051 <= SHIFT <= hi + 0.051 and hi - lo <= 0.151, str(cal))
    print("-- stale frames during profiling passes, placed by step")
    passes, rows2 = [], []
    t_done = T0 + 100.0
    for k in range(3):
        start = T0 + 20 + 30 * k                     # pass k: pre at start, 280 steps 5 frames apart, readback, 3 frames
        pre_f = 10_000 * (k + 1)
        steps = [pre_f + 5 + 5 * i for i in range(280)]
        rb = steps[-1] + 5
        done = rb + 3
        t_done = start + (done - pre_f) / 72.0
        passes.append({"ev": "detector.profile", "utc_us": int(t_done * 1e6), "data": {"pass": k + 1, "spacing": 4,
                       "completed": True, "error": None, "pre_frame": pre_f, "readback_frame": rb, "done_frame": done,
                       "step_frames": steps}})
    heavy_t = [p["utc_us"] / 1e6 - (p["data"]["done_frame"] - p["data"]["step_frames"][11]) / 72.0 for p in passes]
    for j in range(1, 130):
        a, b = T0 + j - 1, T0 + j
        rows2.append({"_t": T0 + j, "stale_frame_count": sum(1 for x in heavy_t if a <= x < b)})
    pm = DP.profile_misses(rows2, [0.0], passes, 72.0)
    first = sum(pm["mid"]["steps"][0:20])
    check("a step that misses alone in every pass shows up in the first step range (3 of 3 stale frames there)",
          abs(first - 3.0) < 1e-9 and abs(pm["mid"]["stale"] - 3.0) < 1e-9, str(round(first, 3)))


def check_conditions():
    """analysis/scheduling_conditions.py on a synthetic sequential run: two cycles of scan, release and commands."""
    print("-- A1.10d conditions on a synthetic sequential run")
    import csv
    import scheduling_conditions as SC
    T0 = 1_760_000_000
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp)
        (raw / "sampler.jsonl").write_text(json.dumps({"device_start_s": T0, "device_tz": "+0000"}) + "\n"
                                           + "".join(json.dumps({"elapsed_s": k}) + "\n" for k in range(0, 241, 30)))
        lines = []

        def ev(t, name, data):
            lines.append({"seq": 0, "mono_us": int((t - T0) * 1e6), "utc_us": int(t * 1e6), "ev": name, "data": data})
        ev(T0 + 0.5, "session.start", {"format": 1, "session_id": "synth002", "app_version": "test", "os_build": None})
        ev(T0 + 5, "model.load", {"file": None, "copied": False, "ms": 900.0, "backend": "cpu", "execution_mode": "llama"})
        req, loaded = 0, []
        for cycle, base in ((1, T0 + 20), (2, T0 + 120)):
            if cycle > 1:
                ev(base - 0.5, "detector.cycle", {"cycle": cycle, "stage": "load"})
            ev(base, "detector.load", {"package_id": "p", "model_sha256": "5" * 64, "backend": "cpu", "input_size": 416,
                                       "load_ms": 40.0, "warmup_ms": 90.0, "warmup_completed": True, "color_space": "linear"})
            loaded.append((base, base + 21))
            ev(base + 2, "detector.cycle", {"cycle": cycle, "stage": "scan_start"})
            for k in range(12):
                s = base + 3 + k
                ev(s + 0.09, "detector.result", {"path": "camera", "image_id": None, "backend": "cpu", "completed": True,
                    "error": None, "width": 1280, "height": 960, "ratio": 0.325, "schedule_ms": 13.0, "latency_ms": 90.0,
                    "postprocess_ms": 0.02, "frames_waited": 6, "candidates": 3, "detections_total": 1,
                    "detections": [[56, 0.6, 1, 2, 3, 4]], "snapshot": None, "started_utc_us": int(s * 1e6)})
            ev(base + 20, "detector.cycle", {"cycle": cycle, "stage": "scan_end"})
            ev(base + 21, "detector.unload", {"backend": "cpu", "release_ms": 1.2, "in_flight": False})
            ev(base + 21.1, "detector.cycle", {"cycle": cycle, "stage": "released"})
            ev(base + 21.2, "command.schedule", {"state": "start", "source": f"sequential_cycle_{cycle}", "period_s": 12.0, "order": [1, 2]})
            ev(base + 21.3, "detector.cycle", {"cycle": cycle, "stage": "commands_start"})
            for j in range(3):
                t = base + 22 + 12 * j
                req += 1
                pid = "p1" if j % 2 == 0 else "p2"
                ev(t, "command.dispatch", {"seq": j + 1, "choice": 1 + j % 2, "prompt_id": pid, "dispatched": True,
                                           "reason": None, "late_ms": 3.0})
                ev(t + 0.01, "model.request", {"request": req, "prompt_id": pid, "prompt_tokens": 300, "prompt_token_ids": []})
                ev(t + 0.81, "model.generate", {"request": req, "prompt_id": pid, "answer": "x", "answer_tokens": 15,
                                                 "first_token_ms": 300.0, "total_ms": 800.0, "stopped": False})
                ev(t + 1.1 + 0.05 * j, "model.scores", {"request": req, "prompt_id": pid, "candidates": {"box_1": -1.2, "box_2": -0.3}, "best": "box_2", "ms": 300.0})
            ev(base + 70, "command.schedule", {"state": "stop", "source": f"sequential_cycle_{cycle}", "period_s": 12.0, "order": [1, 2]})
            ev(base + 70.1, "detector.cycle", {"cycle": cycle, "stage": "commands_end"})
        ev(T0 + 239, "session.end", {})
        lines.sort(key=lambda e: e["utc_us"])
        for k, e in enumerate(lines):
            e["seq"] = k
        (raw / "20251009T085320Z_synth002.jsonl").write_text("".join(json.dumps(e) + "\n" for e in lines))
        rows = []
        for k in range(1, 240):
            t = T0 + k
            pss = 2000 + (80 if any(a <= t <= b for a, b in loaded) else 0)
            stale = 1 if any(T0 + 20 + 3 + i <= t < T0 + 20 + 4 + i for i in (2, 7)) else 0
            rows.append({"Time Stamp": k * 1000, "stale_frame_count": stale, "average_frame_rate": 72, "app_pss_MB": pss,
                         "available_memory_MB": 2200})
        with open(raw / "com.secondeyes.quest#UnityPlayerGameActivity-20251009_085320.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        import eventlog
        info = eventlog.check_file(raw / "20251009T085320Z_synth002.jsonl")
        problems = info.get("problems") or info.get("errors") or []
        check("the synthetic A1.10d log validates against the schema (commands, cycles, model events)", not problems,
              str(problems)[:300])
        rep = SC.analyze(raw)
        c = rep["commands"]
        check("6 commands sent and matched to their scores; median dispatch-to-scored-answer 1,150 ms",
              c["dispatched"] == 6 and c["unfinished"] == 0 and abs(c["goal_ms"]["median"] - 1150.0) < 1.0, str(c))
        check("no command overlaps an inference (they run after each release)", c["goal_ms_overlapping"] == [])
        cy = rep["cycles"]
        check("two cycles; memory 2,000 MB before each load, 2,080 at the scan's end, 2,000 after release",
              len(cy) == 2 and all(x["mem_before_load_mb"] == 2000 and x["mem_scan_end_mb"] == 2080
                                   and x["mem_after_release_mb"] == 2000 for x in cy), str(cy))
        check("the first command after each release is measured (1.1 s)", all(abs(x["first_command_goal_s"] - 1.1) < 1e-6 for x in cy))
        check("two releases, nothing in flight, no detector result after a release",
              len(rep["releases"]) == 2 and rep["results_after_release"] == 0 and not any(x["in_flight"] for x in rep["releases"]))
        check("2 stale frames over the scans' buckets are counted", rep["stale_scans"]["stale"] == 2, str(rep["stale_scans"]))
        out = io.StringIO()
        with redirect_stdout(out):
            SC.show(rep)
        check("the report prints residency, cycles and the acceptance lines", "residency: language model loaded" in
              out.getvalue() and "cycle 2:" in out.getvalue() and "every command sent and finished: yes" in out.getvalue(),
              out.getvalue()[-500:])


def main() -> int:
    print("-- the detector package")
    man = P.load_manifest()
    check("the manifest's model hash matches the model file", True)
    check("YOLOX-Nano from release 0.1.1rc0, Apache-2.0, 80 COCO classes, chair and table mapped",
          man["source"]["release"] == "0.1.1rc0" and man["license"] == "Apache-2.0" and len(man["classes"]) == 80
          and man["project_categories"] == P.CATEGORIES)
    lic = (P.PACKAGE_DIR / "LICENSE.txt").read_text(encoding="utf-8")
    check("the Apache-2.0 license travels with the model", "Apache License" in lic and "Version 2.0" in lic)
    print("-- the PC reference")
    fresh = R.make_reference(P.PACKAGE_DIR / man["model_file"], R.IMAGES)
    old = json.loads(R.REFERENCE.read_text(encoding="utf-8"))
    same = all(len(o[k]) == len(n[k]) and all(a[0] == b[0] and abs(a[1] - b[1]) <= 1e-4 and Q.iou(a, b) >= 0.999
                                               for a, b in zip(o[k], n[k]))
               for o, n in zip(old["images"], fresh["images"]) for k in ("canvas_detections", "full_detections") if k in o)
    check("the committed reference reproduces here (scores within 1e-4, IoU at least 0.999)", same)
    check("every test canvas and full image has its recorded hash",
          all(P.sha256((P.REPO / e["canvas_file"]).read_bytes()) == e["canvas_sha256"] for e in old["images"])
          and all(P.sha256((P.REPO / e["full_file"]).read_bytes()) == e["full_sha256"] for e in old["images"] if "full_file" in e))
    check("the blank canvas gives no detection; the others give 3, 2, 2 and 1 at 0.3 or more",
          [sum(d[1] >= 0.3 for d in e["canvas_detections"]) for e in old["images"]] == [3, 2, 2, 1, 0])
    print("-- suppression pinned to YOLOX's own")
    rng = np.random.default_rng(7)
    ok = True
    for _ in range(200):
        n = int(rng.integers(2, 40))
        xy = rng.uniform(0, 400, (n, 2)).astype(np.float32)
        wh = rng.uniform(5, 120, (n, 2)).astype(np.float32)
        boxes = np.concatenate([xy, xy + wh], 1)
        scores = rng.permutation(n).astype(np.float32) / n + np.float32(0.01)
        theirs = [int(i) for i in yolox_ref.nms(boxes, scores, P.NMS_IOU)]
        order = np.argsort(-scores, kind="stable")
        ours = [int(order[k]) for k in R._nms_sorted(boxes[order], scores[order], P.NMS_IOU)]
        ok &= theirs == ours
    check("200 random sets with distinct scores: the package's suppression keeps exactly YOLOX's boxes", ok)
    print("-- the parity rules (D85)")
    a = [0, 0.80, 10, 10, 110, 110]
    check("same class, IoU 0.99 and a score 0.01 away match", Q.compare([a], [[0, 0.81, 10, 10, 110, 111]])["pass"])
    check("a score 0.03 away fails", not Q.compare([a], [[0, 0.83, 10, 10, 110, 110]])["pass"])
    check("an IoU of 0.90 fails", not Q.compare([a], [[0, 0.80, 10, 10, 110, 121]])["pass"])
    check("another class fails", not Q.compare([a], [[1, 0.80, 10, 10, 110, 110]])["pass"])
    check("an extra headset detection at 0.5 fails", not Q.compare([a], [a, [3, 0.5, 200, 200, 250, 250]])["pass"])
    r = Q.compare([a, [5, 0.305, 0, 0, 5, 5]], [a])
    check("a lone PC detection at 0.305 is borderline, not a failure", r["pass"] and len(r["borderline"]) == 1)
    r = Q.compare([a], [a, [5, 0.31, 0, 0, 5, 5]], got_floor=0.25)
    check("a lone headset detection at 0.31 fails when the PC's 0.25 floor shows no partner within 0.02 (D88)",
          not r["pass"] and len(r["failures"]) == 1)
    r = Q.compare([a, [5, 0.295, 0, 0, 5, 5]], [a, [5, 0.31, 0, 0, 5, 5]], got_floor=0.25)
    check("PC 0.295 against headset 0.31 is a verified threshold crossing, not strict and not a failure",
          r["pass"] and not r["strict"] and len(r["crossings"]) == 1 and not r["borderline"])
    r = Q.compare([a, [5, 0.305, 0, 0, 5, 5]], [a], got_floor=0.3)
    check("a lone PC detection at 0.305 stays an unresolved borderline when the headset logged only from 0.3",
          r["pass"] and not r["strict"] and len(r["borderline"]) == 1)
    r = Q.compare([a, [5, 0.305, 0, 0, 5, 5]], [a], got_floor=0.25)
    check("the same lone PC detection fails once the headset logs from 0.25 and still shows no partner",
          not r["pass"] and len(r["failures"]) == 1)
    check("identical lists are strict", Q.compare([a], [a])["strict"])
    print("-- a synthetic headset log through the schema, parity and the run summary")
    import eventlog  # analysis/eventlog.py
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "20261006T120000Z_test0001.jsonl"
        synthetic_log(old, log)
        info = eventlog.check_file(log)
        problems = info.get("problems") or info.get("errors") or []
        check("every synthetic detector event validates against schemas/log-event.v1.json", not problems, str(problems)[:300])
        rep = Q.evaluate(Q.results([log]), old)
        check("the reference's own detections pass parity", rep["parity_pass"], json.dumps(rep["rows"])[:300])
        def shift(i, d):
            if i == "dog":
                d[0][1] += 0.05
            return d
        bad = Path(tmp) / "20261006T120001Z_test0002.jsonl"
        synthetic_log(old, bad, shift)
        check("a dog score moved by 0.05 fails parity", not Q.evaluate(Q.results([bad]), old)["parity_pass"])
        def drop(i, d):
            return [x for x in d if x[1] >= 0.32] if i in ("fruits", "box_in_scene") else d
        border = Path(tmp) / "20261006T120002Z_test0003.jsonl"
        synthetic_log(old, border, drop)
        rep = Q.evaluate(Q.results([border]), old)
        check("dropping the 0.3006 orange and the 0.318 book still passes, both reported as borderline",
              rep["parity_pass"] and sum(len(x["borderline"]) for x in rep["rows"] if x["path"] == "tensor") == 2)
        import detector_runs
        s = detector_runs.summarize([json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()])
        g = s["groups"]["camera/gpu_compute"]
        check("the run summary counts 5 camera inferences at 20 per second", g["inferences"] == 5
              and abs(g["inferences_per_s"] - 20.0) < 1e-9, json.dumps(g)[:200])
        out = io.StringIO()
        with redirect_stdout(out):
            code = detector_runs.main([str(log)])
        check("analysis/detector_runs.py prints a summary for a log file", code == 0 and "camera/gpu_compute" in out.getvalue())
    check_phases()
    check_scheduling()
    check_conditions()
    print(f"{COUNT[0]} checks; {'FAILED: ' + ', '.join(FAILED) if FAILED else 'all checks passed'}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
