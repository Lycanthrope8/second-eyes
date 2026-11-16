"""Fusion into one bounded map. A plain voxel-hash TSDF written in numpy, so the same algorithm and
parameters can be ported to the headset; Open3D is not required."""
import numpy as np
from .geometry import inv


class VoxelMap:
    """Averages points per voxel and counts observations. Bounded: beyond max_voxels, new voxels are refused
    and counted, so memory cannot grow with every keyframe."""

    def __init__(self, voxel_m, max_voxels, min_observations):
        self.v = float(voxel_m)
        self.max_voxels = int(max_voxels)
        self.min_obs = int(min_observations)
        self.sum = {}
        self.refused = 0

    def integrate(self, points_w, colors=None):
        keys = np.floor(points_w / self.v).astype(np.int64)
        uniq, inv_idx = np.unique(keys, axis=0, return_inverse=True)
        inv_idx = inv_idx.ravel()
        psum = np.zeros((len(uniq), 3)); np.add.at(psum, inv_idx, points_w)
        cnt = np.bincount(inv_idx, minlength=len(uniq))
        csum = np.zeros((len(uniq), 3))
        if colors is not None:
            np.add.at(csum, inv_idx, colors)
        for k, p, c, n in zip(map(tuple, uniq), psum, csum, cnt):
            cur = self.sum.get(k)
            if cur is None:
                if len(self.sum) >= self.max_voxels:
                    self.refused += 1
                    continue
                self.sum[k] = [p, c, int(n), 1]
            else:
                cur[0] = cur[0] + p; cur[1] = cur[1] + c; cur[2] += int(n); cur[3] += 1

    def cloud(self):
        pts, cols = [], []
        for p, c, n, obs in self.sum.values():
            if obs >= self.min_obs:
                pts.append(p / n); cols.append(c / n)
        if not pts:
            return np.zeros((0, 3)), np.zeros((0, 3))
        return np.array(pts), np.array(cols)

    def stats(self):
        return {"voxels": len(self.sum), "refused_new_voxels": self.refused, "voxel_m": self.v}


def write_ply(path, pts, cols=None):
    pts = np.asarray(pts, dtype="<f4")
    with open(path, "wb") as f:
        head = "ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\n" % len(pts)
        if cols is not None and len(cols) == len(pts):
            head += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f.write((head + "end_header\n").encode())
        if cols is not None and len(cols) == len(pts):
            rec = np.zeros(len(pts), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
            rec["p"], rec["c"] = pts, np.clip(cols, 0, 255).astype("u1")
            f.write(rec.tobytes())
        else:
            f.write(pts.tobytes())


def read_ply_xyz(path):
    with open(path, "rb") as f:
        n, has_col = 0, False
        while True:
            line = f.readline().decode().strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            if "red" in line:
                has_col = True
            if line == "end_header":
                break
        if has_col:
            rec = np.frombuffer(f.read(), dtype=[("p", "<f4", 3), ("c", "u1", 3)], count=n)
            return rec["p"].astype(np.float64)
        return np.frombuffer(f.read(), dtype="<f4", count=n * 3).reshape(n, 3).astype(np.float64)
