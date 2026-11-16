"""Depth providers. Each returns per-keyframe output plus what kind of output it is:
  'inverse_relative' -> needs the inverse-depth scale-and-shift fit (Depth Anything V2 Small)
  'depth'            -> depth in the units of the supplied poses (DA3-Small with poses), checked by one scale
The model adapters are NOT tested in the delivery sandbox (no weights there); the oracle provider is."""
import os
import numpy as np


class OracleRelative:
    """Plumbing check: reference depth turned into an affine-distorted inverse depth with noise.
    If the pipeline is right, calibration recovers metric depth. Never a result."""
    kind = "inverse_relative"
    name = "oracle"

    def __init__(self, cfg):
        self.a, self.b, self.noise, self.rng = cfg["a"], cfg["b"], cfg["noise"], np.random.default_rng(cfg["seed"])

    def info(self):
        return {"name": self.name}

    def predict(self, frames, poses_wc=None, K=None):
        out = []
        for f in frames:
            d = f.depth.astype(np.float64)
            inv = np.where(d > 0, 1.0 / np.maximum(d, 1e-6), 0.0)
            rel = (inv - self.b) / self.a
            rel = rel + self.rng.normal(0, self.noise, rel.shape) * np.abs(rel).mean()
            out.append(rel.astype(np.float32))
        return out


class DepthAnythingV2Small:
    """Depth Anything V2 Small (relative). Needs the official repository on PYTHONPATH and its checkpoint."""
    kind = "inverse_relative"
    name = "dav2s"

    def __init__(self, cfg):
        import sys
        import torch
        sys.path.insert(0, cfg["repo_dir"])
        from depth_anything_v2.dpt import DepthAnythingV2
        self.torch = torch
        self.cfg = cfg
        self.model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
        self.model.load_state_dict(torch.load(cfg["checkpoint"], map_location="cpu"))
        self.model = self.model.to(cfg["device"]).eval()

    def info(self):
        from . import config as C
        return {"name": self.name, "checkpoint": os.path.basename(self.cfg["checkpoint"]),
                "checkpoint_sha256": C.file_sha256(self.cfg["checkpoint"]),
                "parameters": int(sum(p.numel() for p in self.model.parameters())),
                "input_size": self.cfg["input_size"]}

    def predict(self, frames, poses_wc=None, K=None):
        with self.torch.no_grad():
            return [self.model.infer_image(f.bgr, int(self.cfg["input_size"])).astype(np.float32) for f in frames]


class DepthAnything3Small:
    """DA3-Small with supplied poses, in overlapping windows. Extrinsics are world-to-camera (OpenCV), as the
    official API documents; the output depth is scaled by the API to the supplied poses."""
    kind = "depth"

    def __init__(self, cfg, window):
        import torch
        from depth_anything_3.api import DepthAnything3
        self.cfg, self.window = cfg, int(window)
        self.name = f"da3s_w{self.window}"
        self.model = DepthAnything3.from_pretrained(cfg["model_id"], revision=cfg.get("revision")).to(cfg["device"]).eval()
        self.torch = torch

    def info(self):
        return {"name": self.name, "model_id": self.cfg["model_id"], "revision": self.cfg.get("revision"),
                "parameters": int(sum(p.numel() for p in self.model.parameters())),
                "window": self.window, "stride": self.window // 2, "process_res": self.cfg["process_res"]}

    def windows(self, n):
        """Overlapping windows; each keyframe is taken from the window where it is nearest the centre."""
        w, s = self.window, max(1, self.window // 2)
        starts = list(range(0, max(n - w, 0) + 1, s))
        if not starts or starts[-1] + w < n:
            starts.append(max(n - w, 0))
        return [(a, min(a + w, n)) for a in sorted(set(starts))]

    def predict(self, frames, poses_wc, K):
        import cv2
        n = len(frames)
        best = [None] * n
        best_dist = [1e9] * n
        for a, b in self.windows(n):
            imgs = [cv2.cvtColor(frames[i].bgr, cv2.COLOR_BGR2RGB) for i in range(a, b)]
            ext = np.stack([np.linalg.inv(poses_wc[i]) for i in range(a, b)]).astype(np.float32)
            intr = np.stack([K for _ in range(a, b)]).astype(np.float32)
            pred = self.model.inference(imgs, extrinsics=ext, intrinsics=intr, align_to_input_ext_scale=True,
                                        process_res=int(self.cfg["process_res"]), export_format="mini_npz")
            centre = (a + b - 1) / 2.0
            for k, i in enumerate(range(a, b)):
                dist = abs(i - centre)
                if dist < best_dist[i]:
                    h, w = frames[i].bgr.shape[:2]
                    best[i] = cv2.resize(np.asarray(pred.depth[k], dtype=np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                    best_dist[i] = dist
        return best


def make(arm, cfg):
    if arm == "oracle":
        return OracleRelative(cfg["oracle"])
    if arm == "dav2s":
        return DepthAnythingV2Small(cfg["dav2s"])
    if arm.startswith("da3s_w"):
        return DepthAnything3Small(cfg["da3s"], int(arm.split("w")[-1]))
    raise ValueError(f"unknown arm {arm}")
