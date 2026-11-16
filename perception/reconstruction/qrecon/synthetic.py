"""A small synthetic room written in TUM RGB-D format. Used by the tests and by `qrecon.py selfcheck`
to prove the plumbing (conventions, calibration, fusion, metrics) before any real model or data is involved."""
import os
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

ROOM = np.array([[0.0, 0.0, 0.0], [4.0, 3.0, 2.5]])
BOX = np.array([[1.6, 1.0, 0.0], [2.3, 1.7, 0.8]])


def _texture(p):
    def h(c, k):
        return np.modf(np.sin(c @ np.array([12.9898, 78.233, 37.719]) + k) * 43758.5453)[0] % 1.0
    coarse = h(np.floor(p / 0.25), 0.0)
    fine = h(np.floor(p / 0.06), 1.7)
    return (40 + 150 * np.abs(coarse) + 60 * np.abs(fine)).clip(0, 255)


def _pose(pos, yaw, pitch=0.0):
    f = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)])
    up = np.array([0, 0, 1.0])
    r = np.cross(f, up); r /= np.linalg.norm(r)
    d = np.cross(f, r)
    T = np.eye(4)
    T[:3, :3] = np.stack([r, d, f], axis=1)
    T[:3, 3] = pos
    return T


def render(T_wc, K, w, h):
    v, u = np.mgrid[0:h, 0:w]
    dc = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, dtype=float)], axis=-1).reshape(-1, 3)
    dw = dc @ T_wc[:3, :3].T
    o = T_wc[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        bound = np.where(dw > 0, ROOM[1], ROOM[0])
        t_room = np.min(np.where(np.abs(dw) > 1e-12, (bound - o) / dw, np.inf), axis=1)
        t1, t2 = (BOX[0] - o) / dw, (BOX[1] - o) / dw
        tn, tf = np.max(np.minimum(t1, t2), axis=1), np.min(np.maximum(t1, t2), axis=1)
    hit = (tn < tf) & (tn > 0) & (tn < t_room)
    t = np.where(hit, tn, t_room)
    pts = o + t[:, None] * dw
    img = _texture(pts).reshape(h, w).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), t.reshape(h, w).astype(np.float32)


def write_tum(root, n=60, w=320, h=240, f=280.0, depth_factor=5000.0, hz=10.0):
    os.makedirs(os.path.join(root, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(root, "depth"), exist_ok=True)
    K = np.array([[f, 0, w / 2 - 0.5], [0, f, h / 2 - 0.5], [0, 0, 1.0]])
    rgb_l, dep_l, gt_l = [], [], []
    for i in range(n):
        a = i / (n - 1)
        pos = np.array([0.7 + 0.9 * a, 0.5 + 0.35 * np.sin(2 * np.pi * a), 1.2 + 0.1 * np.cos(2 * np.pi * a)])
        T = _pose(pos, yaw=np.radians(25 + 30 * a), pitch=np.radians(-12))
        bgr, depth = render(T, K, w, h)
        t = 1000.0 + i / hz
        cv2.imwrite(os.path.join(root, "rgb", f"{t:.6f}.png"), bgr)
        cv2.imwrite(os.path.join(root, "depth", f"{t:.6f}.png"), np.round(depth * depth_factor).astype(np.uint16))
        q = Rotation.from_matrix(T[:3, :3]).as_quat()
        rgb_l.append(f"{t:.6f} rgb/{t:.6f}.png"); dep_l.append(f"{t:.6f} depth/{t:.6f}.png")
        gt_l.append(f"{t:.6f} {T[0,3]:.6f} {T[1,3]:.6f} {T[2,3]:.6f} {q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f}")
    for name, lines in (("rgb.txt", rgb_l), ("depth.txt", dep_l), ("groundtruth.txt", gt_l)):
        with open(os.path.join(root, name), "w") as fh:
            fh.write("# synthetic\n" + "\n".join(lines) + "\n")
    return {"role": "development", "dir": root, "depth_factor": depth_factor,
            "camera": {"width": w, "height": h, "fx": f, "fy": f, "cx": w / 2 - 0.5, "cy": h / 2 - 0.5, "dist": [0, 0, 0, 0, 0]}}
