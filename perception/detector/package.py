"""The detector package (A1.10b, D84): YOLOX-Nano's released ONNX file, wrapped so the headset and the PC run one graph.

The wrapper adds two nodes before the released graph (scale [0, 1] to [0, 255]; reorder RGB to the BGR the model was
trained on) and YOLOX's own decoding after it (`demo_postprocess` and the score product of its ONNX demo), so the
outputs are `boxes` (x1, y1, x2, y2 in input pixels), `scores` (the best class's objectness x class probability) and
`classes` (that class's COCO index). Non-maximum suppression and the letterbox inverse stay outside the graph, in
`reference.py` here and `YoloxDetector.cs` on the headset, written to the same rules (`postprocess` in the manifest).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from . import yolox_ref

REPO = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO / "quest-app" / "Assets" / "SecondEyes" / "Models" / "yolox-nano"
MODEL_FILE = "yolox_nano_se.onnx"
MANIFEST_FILE = "detector-package.json"
PACKAGE_ID = "yolox_nano_coco.se1"
SIZE = 416
STRIDES = (8, 16, 32)
SCORE_THRESHOLD = 0.3
NMS_IOU = 0.45
SOURCE = {"repository": "https://github.com/Megvii-BaseDetection/YOLOX", "release": "0.1.1rc0",
          "asset": "yolox_nano.onnx",
          "url": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx",
          "sha256": "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d", "bytes": 3659407,
          "license": "Apache-2.0", "parameters": 904481, "published": "0.91M parameters, 1.08 GFLOPs at 416x416, "
                                                                 "COCO val mAP 25.8 (YOLOX's ONNX README)"}
CATEGORIES = [{"class": "chair", "category": "chair"}, {"class": "dining table", "category": "table"}]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def grids_and_strides():
    """YOLOX's demo_postprocess grid and stride tables for a 416 x 416 input, as float32 (exact small integers)."""
    grids, strides = [], []
    for s in STRIDES:
        n = SIZE // s
        xv, yv = np.meshgrid(np.arange(n), np.arange(n))
        grids.append(np.stack((xv, yv), 2).reshape(1, -1, 2))
        strides.append(np.full((1, n * n, 1), s))
    return np.concatenate(grids, 1).astype(np.float32), np.concatenate(strides, 1).astype(np.float32)


def build(source_path) -> bytes:
    """The wrapped model's bytes from the released file, which must have its pinned hash."""
    data = Path(source_path).read_bytes()
    if sha256(data) != SOURCE["sha256"]:
        raise SystemExit(f"{source_path}: SHA-256 {sha256(data)} is not the release's {SOURCE['sha256']}")
    m = onnx.load_from_string(data)
    g = m.graph
    if [i.name for i in g.input] != ["images"] or [o.name for o in g.output] != ["output"]:
        raise SystemExit("the released graph does not have the expected input 'images' and output 'output'")
    grids, strides = grids_and_strides()
    n = grids.shape[1]
    c = [numpy_helper.from_array(np.array(255.0, np.float32), "se_255"),
         numpy_helper.from_array(np.array([2, 1, 0], np.int64), "se_rgb_to_bgr"),
         numpy_helper.from_array(grids, "se_grids"), numpy_helper.from_array(strides, "se_strides"),
         numpy_helper.from_array(np.array(0.5, np.float32), "se_half"),
         numpy_helper.from_array(np.array([2], np.int64), "se_axis2")]
    spans = {"xy": (0, 2), "wh": (2, 4), "obj": (4, 5), "cls": (5, 85)}
    for k, (a, b) in spans.items():
        c += [numpy_helper.from_array(np.array([a], np.int64), f"se_{k}_start"),
              numpy_helper.from_array(np.array([b], np.int64), f"se_{k}_end")]
    pre = [helper.make_node("Mul", ["image", "se_255"], ["se_image_255"], name="se_scale_255"),
           helper.make_node("Gather", ["se_image_255", "se_rgb_to_bgr"], ["images"], axis=1, name="se_rgb_to_bgr")]
    post = [helper.make_node("Slice", ["output", f"se_{k}_start", f"se_{k}_end", "se_axis2"], [f"se_{k}"],
                             name=f"se_slice_{k}") for k in spans]
    post += [helper.make_node("Add", ["se_xy", "se_grids"], ["se_xy_grid"], name="se_add_grid"),
             helper.make_node("Mul", ["se_xy_grid", "se_strides"], ["se_centre"], name="se_centre"),
             helper.make_node("Exp", ["se_wh"], ["se_wh_exp"], name="se_exp"),
             helper.make_node("Mul", ["se_wh_exp", "se_strides"], ["se_size"], name="se_size"),
             helper.make_node("Mul", ["se_size", "se_half"], ["se_half_size"], name="se_half_size"),
             helper.make_node("Sub", ["se_centre", "se_half_size"], ["se_corner1"], name="se_corner1"),
             helper.make_node("Add", ["se_centre", "se_half_size"], ["se_corner2"], name="se_corner2"),
             helper.make_node("Concat", ["se_corner1", "se_corner2"], ["boxes"], axis=2, name="se_boxes"),
             helper.make_node("Mul", ["se_obj", "se_cls"], ["se_class_scores"], name="se_class_scores"),
             helper.make_node("ReduceMax", ["se_class_scores"], ["scores"], axes=[2], keepdims=0, name="se_scores"),
             helper.make_node("ArgMax", ["se_class_scores"], ["classes"], axis=2, keepdims=0, name="se_classes")]
    nodes = pre + list(g.node) + post
    del g.node[:]
    g.node.extend(nodes)
    g.initializer.extend(c)
    del g.input[:]
    g.input.append(helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, SIZE, SIZE]))
    del g.output[:]
    g.output.extend([helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, n, 4]),
                     helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, n]),
                     helper.make_tensor_value_info("classes", TensorProto.INT64, [1, n])])
    m.producer_name, m.producer_version = "second-eyes perception.detector", "1"
    del m.metadata_props[:]
    for k, v in (("second_eyes.package_id", PACKAGE_ID), ("second_eyes.source_sha256", SOURCE["sha256"])):
        m.metadata_props.add(key=k, value=v)
    onnx.checker.check_model(m, full_check=True)
    return m.SerializeToString()


def manifest(model_bytes: bytes) -> dict:
    return {"format_version": 1, "record_type": "detector_package", "package_id": PACKAGE_ID,
            "model_file": MODEL_FILE, "model_sha256": sha256(model_bytes), "model_bytes": len(model_bytes),
            "source": SOURCE,
            "build": {"command": "python -m perception.detector build --source yolox_nano.onnx",
                      "onnx": onnx.__version__, "added_before": ["Mul by 255", "Gather RGB to BGR on the channel axis"],
                      "added_after": ["YOLOX demo_postprocess decoding", "objectness x class probability",
                                      "ReduceMax and ArgMax over the 80 classes", "centre-size to corners"]},
            "input": {"name": "image", "size": SIZE, "shape": [1, 3, SIZE, SIZE], "channels": "RGB",
                      "range": [0.0, 1.0], "letterbox": "top_left", "pad_value": 114,
                      "resize": "bilinear with pixel-centre alignment (cv2.INTER_LINEAR)",
                      "ratio": "min(size / height, size / width); resized size int(width * ratio) x int(height * ratio)"},
            "outputs": {"boxes": "[1, 3549, 4] x1, y1, x2, y2 in input pixels",
                        "scores": "[1, 3549] the best class's objectness x class probability",
                        "classes": "[1, 3549] that class's COCO index"},
            "postprocess": {"score_threshold": SCORE_THRESHOLD, "nms": "class_agnostic_greedy",
                            "nms_iou_threshold": NMS_IOU, "nms_overlap": "yolox_plus_one",
                            "order": "descending score, ties by ascending index",
                            "coordinates": "source-image pixels: input-pixel boxes divided by the ratio, not clipped"},
            "classes": list(yolox_ref.COCO_CLASSES), "project_categories": CATEGORIES,
            "license": "Apache-2.0", "license_file": "LICENSE.txt"}


def write_package(source_path, out_dir=PACKAGE_DIR) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = build(source_path)
    (out / MODEL_FILE).write_bytes(model)
    man = manifest(model)
    (out / MANIFEST_FILE).write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    (out / "LICENSE.txt").write_bytes((Path(__file__).resolve().parent / "LICENSE-YOLOX.txt").read_bytes())
    return man


def load_manifest(package_dir=PACKAGE_DIR) -> dict:
    man = json.loads((Path(package_dir) / MANIFEST_FILE).read_text(encoding="utf-8"))
    data = (Path(package_dir) / man["model_file"]).read_bytes()
    if sha256(data) != man["model_sha256"]:
        raise SystemExit(f"{man['model_file']} does not have the manifest's SHA-256")
    return man
