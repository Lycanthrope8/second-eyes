#!/usr/bin/env python3
"""Export Depth Anything V2 Small to ONNX for the headset, at the fixed input size its own preprocessing gives for a
frame size, and check the result before it is used.

    python scripts/export_dav2_onnx.py --frame-size 960 540 --image some_frame.jpg --out models/dav2s_960x540.onnx

Checks, written to <out>.json:
  1. the ONNX model against PyTorch on the same input tensor;
  2. the whole headset path (the C++ program's preprocessing and its corner-aligned bilinear resize, re-implemented
     here in numpy, with the ONNX model) against Depth Anything V2's own infer_image on a real frame.
Run on the workstation, in the harness venv with the pinned repository and checkpoint (RUNBOOK). Needs onnx and
onnxruntime (pip install onnx onnxruntime==1.29.1)."""
import argparse, hashlib, json, math, os, sys
import numpy as np
import cv2

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def dav2_input_size(w, h, lower=518, mult=14):
    """Depth Anything V2's Resize(keep_aspect_ratio=True, ensure_multiple_of=14, resize_method='lower_bound')."""
    sh, sw = lower / h, lower / w
    if sw > sh:
        sh = sw
    else:
        sw = sh

    def constrain(x):
        y = int(round(x / mult) * mult)
        return int(math.ceil(x / mult) * mult) if y < lower else y
    return constrain(sh * h), constrain(sw * w)


def headset_preprocess(bgr, in_h, in_w):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rs = cv2.resize(rgb, (in_w, in_h), interpolation=cv2.INTER_CUBIC)
    rs = (rs - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    return rs.transpose(2, 0, 1)[None].astype(np.float32)


def resize_align_corners(o, h, w):
    oh, ow = o.shape
    ys = np.linspace(0, oh - 1, h); xs = np.linspace(0, ow - 1, w)
    y0 = np.floor(ys).astype(int); x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, oh - 1); x1 = np.minimum(x0 + 1, ow - 1)
    wy = (ys - y0)[:, None]; wx = (xs - x0)[None, :]
    return ((1 - wy) * ((1 - wx) * o[y0][:, x0] + wx * o[y0][:, x1]) + wy * ((1 - wx) * o[y1][:, x0] + wx * o[y1][:, x1])).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame-size", type=int, nargs=2, required=True, metavar=("W", "H"))
    ap.add_argument("--image", required=True, help="a real frame of that size, for the checks")
    ap.add_argument("--out", required=True)
    ap.add_argument("--short-side", type=int, default=518)
    ap.add_argument("--repo-dir", default=os.path.join(REPO, "third_party", "Depth-Anything-V2"))
    ap.add_argument("--checkpoint", default=os.path.join(REPO, "models", "depth_anything_v2_vits.pth"))
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()
    import torch
    import onnxruntime as ort
    sys.path.insert(0, a.repo_dir)
    from depth_anything_v2.dpt import DepthAnythingV2
    w, h = a.frame_size
    in_h, in_w = dav2_input_size(w, h, a.short_side)
    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(a.checkpoint, map_location="cpu"))
    model.eval()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(model, torch.zeros(1, 3, in_h, in_w), a.out, input_names=["image"], output_names=["depth"],
                          opset_version=a.opset, dynamic_axes=None)
    bgr = cv2.imread(a.image)
    if bgr is None or (bgr.shape[1], bgr.shape[0]) != (w, h):
        sys.exit(f"--image must be a {w}x{h} frame")
    x = headset_preprocess(bgr, in_h, in_w)
    sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
    o_onnx = sess.run(None, {"image": x})[0].reshape(in_h, in_w)
    with torch.no_grad():
        o_torch = model(torch.from_numpy(x)).numpy().reshape(in_h, in_w)
        ref = model.infer_image(bgr, a.short_side)                     # Depth Anything V2's own path, frame size
    head = resize_align_corners(o_onnx, h, w)
    rel = lambda p, q: float(np.median(np.abs(p - q) / (np.abs(q) + 1e-6)))
    report = {"frame_size": [w, h], "input_size": [in_h, in_w], "opset": a.opset,
              "checkpoint_sha256": hashlib.sha256(open(a.checkpoint, "rb").read()).hexdigest(),
              "onnx_sha256": hashlib.sha256(open(a.out, "rb").read()).hexdigest(),
              "onnx_vs_torch": {"max_abs": float(np.abs(o_onnx - o_torch).max()), "median_rel": rel(o_onnx, o_torch)},
              "headset_path_vs_infer_image": {"median_rel": rel(head, ref), "p99_rel": float(np.percentile(np.abs(head - ref) / (np.abs(ref) + 1e-6), 99))},
              "onnxruntime": ort.__version__, "torch": torch.__version__}
    report["ok"] = report["onnx_vs_torch"]["median_rel"] < 1e-3 and report["headset_path_vs_infer_image"]["median_rel"] < 1e-2
    json.dump(report, open(a.out + ".json", "w"), indent=2)
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
