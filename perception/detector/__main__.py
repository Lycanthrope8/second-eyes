"""python -m perception.detector {build,images,reference,equivalence,parity,snapshot} (A1.10b and A1.10c; docs/detector.md)."""
from __future__ import annotations

import argparse
import json
import sys


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
    events = Q.results(Q.session_logs(a.target, getattr(a, "include_before_run", False)))
    if a.cmd == "snapshot":
        rep = Q.snapshot(events, a.image, a.id)
        print(f"snapshot {a.id} ({rep['backend']}): {'PASS' if rep['pass'] else 'FAIL'}; {len(rep['pairs'])} matched, "
              f"largest score difference {rep['max_score_diff']:.4f}, smallest IoU {rep['min_iou']:.4f}; "
              f"failures {rep['failures']}; borderline {rep['borderline']}")
        return 0 if rep["pass"] else 1
    ref = json.loads(R.REFERENCE.read_text(encoding="utf-8"))
    rep = Q.evaluate(events, ref)
    for x in rep["rows"]:
        tag = "parity" if x["path"] == "tensor" else "informational"
        print(f"{x['log']}:{x['line']} {x['backend']} {x['path']} {x['image']} [{tag}]: "
              f"{'pass' if x['pass'] else 'FAIL'}; {len(x['pairs'])} matched, max score diff {x['max_score_diff']:.4f}, "
              f"min IoU {x['min_iou']:.4f}" + (f"; failures {x['failures']}" if x["failures"] else "")
              + (f"; borderline {x['borderline']}" if x["borderline"] else ""))
    if rep["missing_gpu_images"]:
        print(f"parity images never run on the GPU backend: {rep['missing_gpu_images']}")
    print(f"PARITY {'PASSED' if rep['parity_pass'] else 'FAILED'} (GPU backend, exact canvases; criteria in D85)")
    return 0 if rep["parity_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
