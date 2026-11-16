"""Metric anchoring. A tracker with one camera returns poses in an arbitrary scale and frame.
(1) Room tag: its corners, seen from several translated views, are triangulated in the tracker's frame and
    fitted to the tag's measured size; this gives scale and the room frame.
(2) Diagnostics for public sequences without a tag: similarity fitted to reference positions, over the whole
    trajectory or over the first frames only. Both are labelled diagnostic in every output."""
import numpy as np
import cv2
from .geometry import umeyama, apply_sim3_to_pose
from .landmarks import triangulate


def tag_model_corners(size_m):
    """Tag frame: origin at the centre, x right, y up, z out of the tag towards the viewer.
    Corner order follows OpenCV's detector: top-left, top-right, bottom-right, bottom-left."""
    h = size_m / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


def detect_tag(bgr, tag_id, dictionary="DICT_APRILTAG_36h11"):
    """Returns the 4 corner pixels (4,2) of the tag with this id, or None."""
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    if ids is None:
        return None
    for c, i in zip(corners, ids.ravel()):
        if int(i) == int(tag_id):
            return c.reshape(4, 2).astype(np.float64)
    return None


def anchor_from_tag(observations, poses, K, size_m, cfg):
    """observations: {frame index: (4,2) corner pixels}; poses: tracker poses (arbitrary scale).
    Returns dict with scale s, R, t (tracker frame -> tag frame) and checks, or ok=False with the reason."""
    idx = sorted(observations)
    if len(idx) < cfg["min_views"]:
        return {"ok": False, "reason": "too few views of the tag", "views": len(idx)}
    centres = np.array([poses[i][:3, 3] for i in idx])
    base = float(np.max(np.linalg.norm(centres - centres[0], axis=1)))
    X = np.zeros((4, 3)); n = 0
    pairs = [(a, b) for ai, a in enumerate(idx) for b in idx[ai + 1:]
             if base > 0 and np.linalg.norm(poses[a][:3, 3] - poses[b][:3, 3]) >= cfg["min_pair_baseline_fraction"] * base]
    for a, b in pairs:
        Xi = triangulate(K, poses[a], poses[b], observations[a], observations[b])
        ra, rb = Xi.mean(0) - poses[a][:3, 3], Xi.mean(0) - poses[b][:3, 3]
        cosang = ra @ rb / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-12)
        if np.degrees(np.arccos(np.clip(cosang, -1, 1))) < cfg["min_parallax_deg"]:
            continue                                   # the two views see the tag from almost the same place
        X += Xi; n += 1
    if n == 0:
        return {"ok": False, "reason": "views of the tag are not translated enough", "views": len(idx)}
    X /= n
    s, R, t = umeyama(X, tag_model_corners(size_m))
    fitted = (s * (R @ X.T)).T + t
    resid = float(np.sqrt(np.mean(np.sum((fitted - tag_model_corners(size_m)) ** 2, axis=1))))
    edges = np.linalg.norm(X - np.roll(X, -1, axis=0), axis=1)
    out = {"ok": True, "scale": s, "R": R, "t": t, "views": len(idx), "pairs": n, "corner_fit_rms_m": resid,
           "edge_length_spread": float(edges.std() / edges.mean())}
    if resid > cfg["max_corner_fit_rms_m"]:
        out.update(ok=False, reason="tag corners do not fit a square of the measured size")
    return out


def to_anchor_frame(poses, s, R, t):
    return [apply_sim3_to_pose(T, s, R, t) for T in poses]


def diagnostic_alignment(est_poses, ref_poses, first_k=None):
    """DIAGNOSTIC ONLY. Similarity from estimated to reference camera positions; first_k limits the fit to the
    first frames, which shows drift after an anchor at the start."""
    e = np.array([T[:3, 3] for T in est_poses]); r = np.array([T[:3, 3] for T in ref_poses])
    n = len(e) if first_k is None else min(first_k, len(e))
    s, R, t = umeyama(e[:n], r[:n])
    return s, R, t
