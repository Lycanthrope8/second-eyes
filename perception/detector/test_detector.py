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
    r = Q.compare([a], [a, [5, 0.31, 0, 0, 5, 5]])
    check("a lone headset detection at 0.31 is borderline, not a failure", r["pass"] and len(r["borderline"]) == 1)
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
    print(f"{COUNT[0]} checks; {'FAILED: ' + ', '.join(FAILED) if FAILED else 'all checks passed'}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
