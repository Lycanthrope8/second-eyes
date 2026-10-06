"""The PC reference for the detector (A1.10b, D85): fixed test images, YOLOX's own pipeline and the wrapped graph.

Test images (no people) live in the Unity project as PNG bytes, so the headset and the PC read identical files:
`*_416.png.bytes` are 416 x 416 canvases already letterboxed by YOLOX's `preproc`, read on the headset as exact pixels
(the parity path); `*_full.png.bytes` are the originals as lossless PNGs, letterboxed on the headset's GPU (the
informational texture path). `reference` writes fixtures/reference.v1.json: every detection at or above the floor 0.25
after the package's non-maximum suppression, so detections near the 0.3 threshold can be matched either side.
"""
from __future__ import annotations

import platform
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort

from . import package as P
from . import yolox_ref

TEST_DIR = P.REPO / "quest-app" / "Assets" / "SecondEyes" / "Perception" / "DetectorTests"
REFERENCE = Path(__file__).resolve().parent / "fixtures" / "reference.v1.json"
FLOOR = 0.25
YOLOX_COMMIT = "6ddff4824372906469a7fae2dc3206c7aa4bbaee"
OPENCV_COMMIT = "3303de8ae0ec0567c50e334ca39cf4587d4abc46"
IMAGES = [
    {"id": "dog", "file": "dog.jpg", "full": True, "license": "Apache-2.0",
     "source": f"https://github.com/Megvii-BaseDetection/YOLOX/blob/{YOLOX_COMMIT}/assets/dog.jpg",
     "sha256": "5a9522051c3cec2bbd2f6323fccba32e8fbf3ddcc2b3e2fd46b04c720bc6f866"},
    {"id": "fruits", "file": "fruits.jpg", "full": True, "license": "Apache-2.0",
     "source": f"https://github.com/opencv/opencv/blob/{OPENCV_COMMIT}/samples/data/fruits.jpg", "sha256": "9c031d80a1c52da5eca790db896baffec6a7e52bf786cdb7bbfca5c7f880e6a1"},
    {"id": "box_in_scene", "file": "box_in_scene.png", "full": False, "license": "Apache-2.0",
     "source": f"https://github.com/opencv/opencv/blob/{OPENCV_COMMIT}/samples/data/box_in_scene.png", "sha256": "8b0225ff76244a42bd1400c0904f8b7afea7d97b7d8115495e42e98ad347bd51"},
    {"id": "apple", "file": "apple.jpg", "full": False, "license": "Apache-2.0",
     "source": f"https://github.com/opencv/opencv/blob/{OPENCV_COMMIT}/samples/data/apple.jpg", "sha256": "e86879de3d9a807dedc742a8f464f39e6b74bbb531a1fba3583155d94997d1cd"},
]


def canvas_name(i) -> str:
    return f"{i}_416.png.bytes"


def full_name(i) -> str:
    return f"{i}_full.png.bytes"


def make_images(source_dir) -> list:
    """Write the test images from the downloaded originals in `source_dir`, which must have their pinned hashes."""
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in IMAGES:
        raw = (Path(source_dir) / spec["file"]).read_bytes()
        if P.sha256(raw) != spec["sha256"]:
            raise SystemExit(f"{spec['file']}: not the pinned original")
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        canvas, _ = yolox_ref.preproc(img, (P.SIZE, P.SIZE))
        canvas = np.ascontiguousarray(canvas.transpose(1, 2, 0)).astype(np.uint8)
        _write_png(TEST_DIR / canvas_name(spec["id"]), canvas)
        if spec["full"]:
            _write_png(TEST_DIR / full_name(spec["id"]), img)
        rows.append(dict(spec, width=int(img.shape[1]), height=int(img.shape[0])))
    blank = np.full((P.SIZE, P.SIZE, 3), 114, np.uint8)
    _write_png(TEST_DIR / canvas_name("blank"), blank)
    return rows


def _write_png(path: Path, bgr: np.ndarray):
    ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise SystemExit(f"cannot encode {path.name}")
    path.write_bytes(buf.tobytes())


def read_png(path: Path) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_COLOR)


def session(model_path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])


def rgb01(bgr_canvas: np.ndarray) -> np.ndarray:
    """A BGR uint8 canvas as the wrapped model's input: RGB, [0, 1], NCHW float32 (k / 255 in float32)."""
    rgb = bgr_canvas[:, :, ::-1].astype(np.float32) / np.float32(255.0)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))[None]


def postprocess(boxes, scores, classes, ratio=1.0, floor=P.SCORE_THRESHOLD) -> list:
    """Candidates at or above `floor`, YOLOX's greedy class-agnostic NMS, then input pixels to source pixels."""
    keep_c = np.where(scores >= np.float32(floor))[0]
    if keep_c.size == 0:
        return []
    order = keep_c[np.lexsort((keep_c, -scores[keep_c]))]  # descending score, ties by ascending index
    b, s = boxes[order], scores[order]
    keep = _nms_sorted(b, s, P.NMS_IOU)
    out = []
    for k in keep:
        x1, y1, x2, y2 = (float(v) / ratio for v in b[k])
        out.append([int(classes[order[k]]), float(s[k]), x1, y1, x2, y2])
    return out


def _nms_sorted(boxes, scores, thr) -> list:
    """yolox_ref.nms on candidates already in the package's order (its argsort replaced by that order)."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.arange(len(scores))
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= thr)[0] + 1]
    return keep


def detect_canvas(sess, bgr_canvas, ratio=1.0, floor=P.SCORE_THRESHOLD) -> list:
    boxes, scores, classes = sess.run(["boxes", "scores", "classes"], {"image": rgb01(bgr_canvas)})
    return postprocess(boxes[0], scores[0], classes[0], ratio, floor)


def yolox_pipeline(original_model, bgr_image, floor) -> list:
    """YOLOX's own demo pipeline on the released model, for the wrapper's equivalence check."""
    canvas, r = yolox_ref.preproc(bgr_image, (P.SIZE, P.SIZE))
    out = original_model.run(None, {"images": canvas[None]})[0]
    pred = yolox_ref.demo_postprocess(out.copy(), (P.SIZE, P.SIZE))[0]
    boxes, scores = pred[:, :4], pred[:, 4:5] * pred[:, 5:]
    xyxy = np.ones_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
    xyxy /= r
    d = yolox_ref.multiclass_nms(xyxy, scores, nms_thr=P.NMS_IOU, score_thr=floor - 1e-9)
    if d is None:
        return []
    return sorted(([int(c), float(s), *map(float, b)] for *b, s, c in d), key=lambda x: -x[1])


def make_reference(model_path, images: list) -> dict:
    sess = session(model_path)
    entries = []
    for spec in images + [{"id": "blank", "full": False, "license": "generated", "source": "uniform pad value 114"}]:
        cpath = TEST_DIR / canvas_name(spec["id"])
        entry = {"id": spec["id"], "canvas_file": cpath.relative_to(P.REPO).as_posix(),
                 "canvas_sha256": P.sha256(cpath.read_bytes()),
                 "canvas_detections": detect_canvas(sess, read_png(cpath), 1.0, FLOOR)}
        if spec.get("full"):
            fpath = TEST_DIR / full_name(spec["id"])
            img = read_png(fpath)
            canvas, r = yolox_ref.preproc(img, (P.SIZE, P.SIZE))
            canvas = np.ascontiguousarray(canvas.transpose(1, 2, 0)).astype(np.uint8)
            entry.update(full_file=fpath.relative_to(P.REPO).as_posix(), full_sha256=P.sha256(fpath.read_bytes()),
                         width=int(img.shape[1]), height=int(img.shape[0]), ratio=float(r),
                         full_detections=detect_canvas(sess, canvas, r, FLOOR))
        entry["source"] = {k: spec.get(k) for k in ("source", "license", "sha256", "width", "height") if k in spec}
        entries.append(entry)
    man = P.load_manifest()
    return {"format_version": 1, "record_type": "detector_reference", "package_id": man["package_id"],
            "model_sha256": man["model_sha256"], "floor": FLOOR, "score_threshold": P.SCORE_THRESHOLD,
            "nms_iou_threshold": P.NMS_IOU, "detection_fields": ["class", "score", "x1", "y1", "x2", "y2"],
            "runtime": {"python": platform.python_version(), "onnxruntime": ort.__version__, "onnx": onnx.__version__,
                        "opencv": cv2.__version__, "numpy": np.__version__, "provider": "CPUExecutionProvider",
                        "threads": 1, "platform": platform.platform()},
            "images": entries}
