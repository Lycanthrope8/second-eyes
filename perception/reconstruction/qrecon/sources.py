"""Frame sources. Every source yields the same Frame, so reconstruction stages don't change when the
replayed file is replaced by the CM4 stream."""
import os
from dataclasses import dataclass
import numpy as np
import cv2
from .geometry import quat_to_T, K_matrix


@dataclass
class Frame:
    index: int
    t: float                       # seconds, source clock
    bgr: np.ndarray                # undistorted image
    K: np.ndarray                  # 3x3 intrinsics of the undistorted image
    T_wc: np.ndarray | None = None       # reference pose, if the source has one
    depth: np.ndarray | None = None      # reference depth in metres (0 = invalid), if the source has one


def _read_list(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.replace(",", " ").split()
            out.append((float(p[0]), p[1:]))
    return out


def associate(a, b, max_dt):
    """Greedy nearest-timestamp matching, as in the TUM associate tool. Returns index pairs."""
    cand = sorted((abs(ta - tb), i, j) for i, (ta, _) in enumerate(a) for j, (tb, _) in enumerate(b)
                  if abs(ta - tb) < max_dt) if len(a) * len(b) < 4_000_000 else None
    if cand is None:                       # long sequences: two-pointer nearest search
        tb = np.array([t for t, _ in b])
        pairs, used = [], set()
        for i, (ta, _) in enumerate(a):
            j = int(np.argmin(np.abs(tb - ta)))
            if abs(tb[j] - ta) < max_dt and j not in used:
                used.add(j)
                pairs.append((i, j))
        return pairs
    ua, ub, pairs = set(), set(), []
    for _, i, j in cand:
        if i not in ua and j not in ub:
            ua.add(i)
            ub.add(j)
            pairs.append((i, j))
    return sorted(pairs)


class TumSource:
    """TUM RGB-D sequence: rgb.txt, depth.txt, groundtruth.txt. RGB is the algorithm input; depth and the
    trajectory are reference data only."""

    def __init__(self, root, seq_cfg, max_dt=0.02):
        self.root = root
        self.cfg = seq_cfg
        rgb = _read_list(os.path.join(root, "rgb.txt"))
        dep = _read_list(os.path.join(root, "depth.txt"))
        gt = _read_list(os.path.join(root, "groundtruth.txt"))
        rd = associate(rgb, dep, max_dt)
        gt_t = np.array([t for t, _ in gt])
        self.items = []
        for i, j in rd:
            k = int(np.argmin(np.abs(gt_t - rgb[i][0])))
            if abs(gt_t[k] - rgb[i][0]) < max_dt:
                self.items.append((rgb[i][0], rgb[i][1][0], dep[j][1][0], [float(x) for x in gt[k][1][:7]]))
        c = seq_cfg["camera"]
        self.K_raw = K_matrix(c["fx"], c["fy"], c["cx"], c["cy"])
        self.dist = np.array(c.get("dist", [0, 0, 0, 0, 0]), dtype=np.float64)
        self.depth_factor = float(seq_cfg["depth_factor"])
        self.K = self.K_raw.copy()
        self._maps = None
        if np.any(self.dist != 0):
            w, h = c["width"], c["height"]
            self._maps = cv2.initUndistortRectifyMap(self.K_raw, self.dist, None, self.K, (w, h), cv2.CV_32FC1)

    def __len__(self):
        return len(self.items)

    def frame(self, idx):
        t, rgb_rel, dep_rel, pose = self.items[idx]
        bgr = cv2.imread(os.path.join(self.root, rgb_rel), cv2.IMREAD_COLOR)
        d16 = cv2.imread(os.path.join(self.root, dep_rel), cv2.IMREAD_UNCHANGED)
        depth = d16.astype(np.float32) / self.depth_factor
        if self._maps is not None:
            bgr = cv2.remap(bgr, self._maps[0], self._maps[1], cv2.INTER_LINEAR)
            depth = cv2.remap(depth, self._maps[0], self._maps[1], cv2.INTER_NEAREST)
        return Frame(idx, t, bgr, self.K, quat_to_T(*pose), depth)

    def poses(self):
        return [quat_to_T(*it[3]) for it in self.items]

    def times(self):
        return [it[0] for it in self.items]


class VideoFileSource:
    """Replayed video file (a phone clip now; a CM4 recording later). No reference pose or depth.
    Needs a calibration file written by scripts/calibrate_camera.py for this camera and video mode."""

    def __init__(self, path, calib, every=1):
        self.path = path
        self.every = int(every)
        self.K_raw = np.array(calib["K"], dtype=np.float64)
        self.dist = np.array(calib["dist"], dtype=np.float64)
        self.size = tuple(calib["size"])          # (width, height)
        self.K = self.K_raw.copy()
        self._maps = cv2.initUndistortRectifyMap(self.K_raw, self.dist, None, self.K, self.size, cv2.CV_32FC1)

    def frames(self):
        cap = cv2.VideoCapture(self.path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        i = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if i % self.every == 0:
                if (bgr.shape[1], bgr.shape[0]) != self.size:
                    raise ValueError(f"video is {bgr.shape[1]}x{bgr.shape[0]}, calibration is {self.size}")
                t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                t = t_ms / 1000.0 if t_ms > 0 else i / fps
                yield Frame(i, t, cv2.remap(bgr, self._maps[0], self._maps[1], cv2.INTER_LINEAR), self.K)
            i += 1
        cap.release()


def read_trajectory_tum(path):
    """Estimated trajectory in TUM format (timestamp tx ty tz qx qy qz qw), as stella_vslam writes it."""
    rows = _read_list(path)
    return [t for t, _ in rows], [quat_to_T(*[float(x) for x in p[:7]]) for _, p in rows]
