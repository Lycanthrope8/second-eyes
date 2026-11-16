"""Sparse metric landmarks from RGB and known poses: ORB matches between keyframes, checked against the
epipolar geometry and triangulated. On the headset these come from the tracker; here they are rebuilt from
the poses in use, so the depth stage is tested the same way with reference or estimated poses."""
import numpy as np
import cv2
from .geometry import inv


def _proj(K, T_wc):
    return K @ inv(T_wc)[:3, :]


def triangulate(K, T_a, T_b, pa, pb):
    """pa, pb: (N,2) pixels in views a and b. Returns world points (N,3)."""
    X = cv2.triangulatePoints(_proj(K, T_a), _proj(K, T_b), pa.T.astype(np.float64), pb.T.astype(np.float64))
    return (X[:3] / X[3]).T


def _reproj_err(K, T_wc, Xw, px):
    Tcw = inv(T_wc)
    pc = Xw @ Tcw[:3, :3].T + Tcw[:3, 3]
    z = pc[:, 2]
    uv = np.stack([K[0, 0] * pc[:, 0] / z + K[0, 2], K[1, 1] * pc[:, 1] / z + K[1, 2]], axis=1)
    return np.linalg.norm(uv - px, axis=1), z


def landmarks_for(i, frames_gray, poses, K, cfg):
    """Landmarks seen in keyframe i: returns pixels (N,2), camera-space depth z (N,), world points (N,3)."""
    orb = cv2.ORB_create(nfeatures=int(cfg["orb_features"]))
    kp_i, de_i = orb.detectAndCompute(frames_gray[i], None)
    if de_i is None:
        return np.zeros((0, 2)), np.zeros(0), np.zeros((0, 3))
    pts_i = np.array([k.pt for k in kp_i])
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    px_all, z_all, X_all = [], [], []
    for off in cfg["neighbour_offsets"]:
        j = i + off
        if j < 0 or j >= len(frames_gray):
            continue
        kp_j, de_j = orb.detectAndCompute(frames_gray[j], None)
        if de_j is None or len(kp_j) < 2:
            continue
        pts_j = np.array([k.pt for k in kp_j])
        pairs = [m for m, n in (p for p in bf.knnMatch(de_i, de_j, k=2) if len(p) == 2)
                 if m.distance < cfg["ratio"] * n.distance]
        if len(pairs) < 8:
            continue
        a = pts_i[[m.queryIdx for m in pairs]]
        b = pts_j[[m.trainIdx for m in pairs]]
        X = triangulate(K, poses[i], poses[j], a, b)
        ea, za = _reproj_err(K, poses[i], X, a)
        eb, zb = _reproj_err(K, poses[j], X, b)
        ca, cb = poses[i][:3, 3], poses[j][:3, 3]
        ra, rb = X - ca, X - cb
        cosang = np.sum(ra * rb, axis=1) / (np.linalg.norm(ra, axis=1) * np.linalg.norm(rb, axis=1) + 1e-12)
        par = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        ok = ((za > cfg["min_depth_m"]) & (zb > cfg["min_depth_m"]) & (za < cfg["max_depth_m"]) &
              (ea < cfg["max_reproj_px"]) & (eb < cfg["max_reproj_px"]) & (par > cfg["min_parallax_deg"]))
        px_all.append(a[ok]); z_all.append(za[ok]); X_all.append(X[ok])
    if not px_all:
        return np.zeros((0, 2)), np.zeros(0), np.zeros((0, 3))
    return np.concatenate(px_all), np.concatenate(z_all), np.concatenate(X_all)
