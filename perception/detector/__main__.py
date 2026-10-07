"""python -m perception.detector {build,images,reference,equivalence,parity,snapshot} (A1.10b and A1.10c; docs/detector.md)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m perception.detector")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("build", help="write the detector package from YOLOX's released yolox_nano.onnx")
    s.add_argument("--source", required=True)
    s = sub.add_parser("images", help="write the test images from the pinned originals")
    s.add_argument("--source-dir", required=True)
    s = sub.add_parser("reference", help="write fixtures/reference.v1.json, or --check it against a fresh run")
    s.add_argument("--check", action="store_true")
    s = sub.add_parser("equivalence", help="the wrapped graph against YOLOX's own pipeline on every test image")
    s.add_argument("--source", required=True)
    s = sub.add_parser("parity", help="a headset run's detections against the PC reference")
    s.add_argument("target", help="a run ID (its pulled session logs) or one log file")
    s.add_argument("--include-before-run", action="store_true")
    s.add_argument("--canvas-dir", help="folder of extra 416 x 416 canvases checked on the headset (for example the "
                                        "snapshots pushed to its parity/ folder), judged by a PC reference made now")
    s = sub.add_parser("schedule", help="cost-balanced slice schedules from a profiling run (D89)")
    s.add_argument("target", help="the profiling run's ID (its pulled session logs) or one log file")
    s.add_argument("--budgets", default="", help="per-frame GPU budgets in ms, comma-separated; default: 40, 55 "
                                                 "and 70 percent of the measured headroom")
    s.add_argument("--out", default=None, help="folder for the schedule files (default perception/detector/schedules)")
    s = sub.add_parser("estimate", help="shape-based cost ESTIMATES and balanced schedules when GPU timing is "
                                        "unavailable (D89)")
    s.add_argument("target", help="a run whose log has detector.layers and detector.load (the runtime layer list)")
    s.add_argument("--observed", default="8:1.44,16:2.56,32:5.55",
                   help="steps per frame:stale per keyframe measured on the headset, for fitting the miss threshold "
                        "(default: r043)")
    s.add_argument("--out", default=None)
    s = sub.add_parser("snapshot", help="a saved camera canvas against the headset's detections for it")
    s.add_argument("target")
    s.add_argument("--image", required=True)
    s.add_argument("--id", required=True)
    a = ap.parse_args(argv)
    from . import package as P
    from . import reference as R
    if a.cmd == "build":
        man = P.write_package(a.source)
        print(f"{P.PACKAGE_DIR.relative_to(P.REPO).as_posix()}/{man['model_file']}: {man['model_bytes']} bytes, "
              f"SHA-256 {man['model_sha256']}")
        return 0
    if a.cmd == "images":
        rows = R.make_images(a.source_dir)
        print(f"wrote {len(rows)} test images and the blank canvas to {R.TEST_DIR.relative_to(P.REPO).as_posix()}")
        return 0
    if a.cmd == "reference":
        man = P.load_manifest()
        ref = R.make_reference(P.PACKAGE_DIR / man["model_file"], R.IMAGES)
        if not a.check:
            R.REFERENCE.parent.mkdir(parents=True, exist_ok=True)
            R.REFERENCE.write_text(json.dumps(ref, indent=1) + "\n", encoding="utf-8")
            print(f"wrote {R.REFERENCE.relative_to(P.REPO).as_posix()}: "
                  + ", ".join(f"{e['id']} {sum(d[1] >= 0.3 for d in e['canvas_detections'])}" for e in ref["images"]))
            return 0
        from . import parity as Q
        old = json.loads(R.REFERENCE.read_text(encoding="utf-8"))
        worst, ok = 0.0, old["model_sha256"] == ref["model_sha256"]
        for o, n in zip(old["images"], ref["images"]):
            for key in ("canvas_detections", "full_detections"):
                if key in o:
                    ok &= o[f"{key.split('_')[0]}_sha256" if key == "full_detections" else "canvas_sha256"] == \
                          n[f"{key.split('_')[0]}_sha256" if key == "full_detections" else "canvas_sha256"]
                    ok &= len(o[key]) == len(n[key]) and all(
                        a_[0] == b_[0] and abs(a_[1] - b_[1]) <= 1e-4 and Q.iou(a_, b_) >= 0.999
                        for a_, b_ in zip(o[key], n[key]))
                    worst = max([worst] + [abs(a_[1] - b_[1]) for a_, b_ in zip(o[key], n[key])])
        print(f"reference check: {'reproduced' if ok else 'NOT reproduced'} (largest score difference {worst:.2e}; "
              f"tolerance 1e-4 and IoU 0.999)")
        return 0 if ok else 1
    if a.cmd == "equivalence":
        import numpy as np
        man = P.load_manifest()
        orig, wrap = R.session(a.source), R.session(P.PACKAGE_DIR / man["model_file"])
        worst, ok = 0.0, True
        for spec in R.IMAGES:
            for name in [R.canvas_name(spec["id"])] + ([R.full_name(spec["id"])] if spec["full"] else []):
                img = R.read_png(R.TEST_DIR / name)
                want = R.yolox_pipeline(orig, img, R.FLOOR)
                canvas, r = R.yolox_ref.preproc(img, (P.SIZE, P.SIZE))
                canvas = np.ascontiguousarray(canvas.transpose(1, 2, 0)).astype(np.uint8)
                got = R.detect_canvas(wrap, canvas, r, R.FLOOR)
                same = len(want) == len(got) and all(w[0] == g[0] and abs(w[1] - g[1]) <= 1e-5 and
                                                     max(abs(x - y) for x, y in zip(w[2:], g[2:])) <= 1e-3
                                                     for w, g in zip(want, got))
                worst = max([worst] + [abs(w[1] - g[1]) for w, g in zip(want, got)])
                ok &= same
                print(f"{name}: {len(want)} detections from YOLOX's pipeline, {len(got)} from the package: "
                      f"{'same' if same else 'DIFFERENT'}")
        print(f"equivalence: {'passed' if ok else 'FAILED'} (largest score difference {worst:.2e})")
        return 0 if ok else 1
    from . import parity as Q
    if a.cmd == "schedule":
        from . import schedule as S
        import json as _json
        events = []
        for path in Q.session_logs(a.target):
            events += [_json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
        events.sort(key=lambda e: e["utc_us"])
        costs = S.step_costs(events)
        head = 1000.0 / 72.0 - costs["baseline_ms"]
        budgets = [float(x) for x in a.budgets.split(",") if x.strip()] or [round(head * f, 1) for f in (0.4, 0.55, 0.7)]
        run = Path(a.target).name if Path(a.target).is_file() else a.target
        schedules, costs = S.build(events, budgets, run)
        for path in S.write(schedules, a.out or S.SCHEDULE_DIR):
            print(f"wrote {path}")
        print(S.report(schedules, costs))
        return 0
    if a.cmd == "estimate":
        from . import estimate as E
        import json as _json
        events = []
        for path in Q.session_logs(a.target):
            events += [_json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
        layers = next((e["data"] for e in events if e["ev"] == "detector.layers"), None)
        load = next((e["data"] for e in events if e["ev"] == "detector.load"), None)
        if layers is None or load is None:
            print("the run has no detector.layers or detector.load event")
            return 2
        al = E.align(E.node_work(), layers["types"])
        if al["unmatched_steps"]:
            print(f"alignment failed: runtime steps {al['unmatched_steps'][:10]} have no ONNX node; types "
                  f"{[layers['types'][i] for i in al['unmatched_steps'][:10]]}")
            return 1
        w = [x + E.OVERHEAD for x in al["step_work"]]
        total = sum(w) + E.PRE_WORK
        observed = {int(k): float(v) for k, v in (x.split(":") for x in a.observed.split(","))}
        fit = E.fit_threshold(w, observed)
        absorbed = sorted({op for _, op, _ in al["absorbed"]})
        print(f"aligned {len(E.node_work())} ONNX nodes to {len(w)} runtime steps; absorbed node types: {absorbed}")
        heavy = [i for i, x in enumerate(w) if x > fit["threshold"]]
        print("steps whose estimated work alone exceeds the fitted miss threshold: "
              + (", ".join(f"#{i} {layers['types'][i]} ({100 * w[i] / total:.1f}%)" for i in heavy) or "none"))
        plans = []
        budget = fit["threshold"] * 0.9
        g = E._greedy([min(x, budget) for x in w], budget, min(E.PRE_WORK, budget))
        for n in sorted({len(g["slices"]) if g else 0, 16, 20, 24, 28, 32} - {0}):
            p = E.partition(w, n)
            plans.append((n, p))
        for line in E.report_lines(w, fit, plans):
            print(line)
        run = Path(a.target).name if Path(a.target).is_file() else a.target
        out = Path(a.out) if a.out else E.ESTIMATE_DIR
        out.mkdir(parents=True, exist_ok=True)
        for n, p in plans:
            rec = E.schedule_record(p, n, load, layers, run, fit, total)
            path = out / f"{rec['schedule_id']}.json"
            path.write_text(_json.dumps(rec, indent=1) + "\n", encoding="utf-8")
            print(f"wrote {path}")
        return 0
    events = Q.results(Q.session_logs(a.target, getattr(a, "include_before_run", False)))
    if a.cmd == "snapshot":
        rep = Q.snapshot(events, a.image, a.id)
        print(f"snapshot {a.id} ({rep['backend']}): {'PASS' if rep['pass'] else 'FAIL'}; {len(rep['pairs'])} matched, "
              f"largest score difference {rep['max_score_diff']:.4f}, smallest IoU {rep['min_iou']:.4f}; "
              f"failures {rep['failures']}; borderline {rep['borderline']}")
        return 0 if rep["pass"] else 1
    ref = json.loads(R.REFERENCE.read_text(encoding="utf-8"))
    rep = Q.evaluate(events, ref, getattr(a, "canvas_dir", None))
    for x in rep["rows"]:
        tag = "parity" if x["path"] == "tensor" else "informational"
        steps = x["steps_per_frame"]
        verdict = "FAIL" if not x["pass"] else "strict" if x["strict"] else "pass with exceptions"
        print(f"{x['log']}:{x['line']} {x['backend']} {x['path']} {x['image']}"
              + (f" ({x['kind']})" if x["kind"] == "extra" else "")
              + (f" steps/frame {steps}" if steps is not None else "") + f" [{tag}]: {verdict}; {len(x['pairs'])} matched, "
              f"max score diff {x['max_score_diff']:.4f}, min IoU {x['min_iou']:.4f}"
              + (f"; verified threshold crossings {len(x['crossings'])}" if x["crossings"] else "")
              + (f"; unresolved borderlines {x['borderline']}" if x["borderline"] else "")
              + (f"; failures {x['failures']}" if x["failures"] else ""))
    code = 0
    for backend in ("gpu_compute", "cpu"):
        rows = [x for x in rep["rows"] if x["path"] == "tensor" and x["backend"] == backend and x["kind"] == "reference"]
        if not rows:
            continue
        missing = sorted({e["id"] for e in ref["images"]} - {x["image"] for x in rows})
        ok = not missing and all(x["pass"] for x in rows)
        strict = ok and all(x["strict"] for x in rows)
        extra = [x for x in rep["rows"] if x["kind"] == "extra" and x["backend"] == backend]
        ok = ok and all(x["pass"] for x in extra)
        code |= 0 if ok else 1
        print(f"PARITY {'PASSED' if ok else 'FAILED'} ({backend} backend, exact canvases"
              + (f" and {len(extra)} pushed canvas results" if extra else "") + "; criteria in D85)"
              + (" with strict agreement on every canvas" if strict else
                 (" with verified crossings or unresolved borderlines listed above" if ok else ""))
              + (f"; never run: {missing}" if missing else ""))
    if not any(x["path"] == "tensor" and x["kind"] == "reference" for x in rep["rows"]):
        print("PARITY FAILED: no exact-canvas results in these logs")
        return 1
    return code


if __name__ == "__main__":
    sys.exit(main())
