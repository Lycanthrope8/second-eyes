"""Focal length from the straight lines of a room, without any board: the dominant vanishing points of a frame are
found in the image, and every pair of finite ones that belongs to perpendicular directions gives the focal length,
f^2 = -(v1 - c).(v2 - c), with c the image centre (square pixels, principal point at the centre, as for phone video).
One frame is noisy; the median over many frames of a walk through rooms with doors, floors and furniture is not."""
import numpy as np
import cv2


def segments(gray, min_len_frac=0.05):
    """Line segments (x1, y1, x2, y2) at least min_len_frac of the image width long."""
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(gray)[0]
    if lines is None:
        return np.zeros((0, 4))
    L = lines.reshape(-1, 4).astype(np.float64)
    return L[np.hypot(L[:, 2] - L[:, 0], L[:, 3] - L[:, 1]) >= min_len_frac * gray.shape[1]]


def _angles(L, V):
    """Angle in degrees between each segment and the line from its midpoint to each vanishing point (rows of V,
    homogeneous). Returns segments x points."""
    m = np.stack([(L[:, 0] + L[:, 2]) / 2, (L[:, 1] + L[:, 3]) / 2], 1)
    d = np.stack([L[:, 2] - L[:, 0], L[:, 3] - L[:, 1]], 1)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    fin = np.abs(V[:, 2]) > 1e-9
    P = np.where(fin[:, None], V[:, :2] / np.where(fin, V[:, 2], 1.0)[:, None], 0.0)
    tx = np.where(fin[None, :], P[None, :, 0] - m[:, None, 0], V[None, :, 0])
    ty = np.where(fin[None, :], P[None, :, 1] - m[:, None, 1], V[None, :, 1])
    tn = np.hypot(tx, ty) + 1e-12
    c = np.abs(d[:, None, 0] * tx + d[:, None, 1] * ty) / tn
    return np.degrees(np.arccos(np.clip(c, 0.0, 1.0)))


def vanishing_points(L, n_vp=3, n_hyp=2000, tol_deg=1.5, seed=0):
    """Up to n_vp dominant vanishing points, strongest first, as (point, supporting length, segments)."""
    rng = np.random.default_rng(seed)
    if len(L) < 4:
        return []
    ln = np.hypot(L[:, 2] - L[:, 0], L[:, 3] - L[:, 1])
    lines = np.cross(np.c_[L[:, :2], np.ones(len(L))], np.c_[L[:, 2:], np.ones(len(L))])
    alive = np.ones(len(L), bool)
    out = []
    for _ in range(n_vp):
        ids = np.nonzero(alive)[0]
        if len(ids) < 4:
            break
        ab = rng.integers(0, len(ids), size=(n_hyp, 2))
        ab = ab[ab[:, 0] != ab[:, 1]]
        V = np.cross(lines[ids[ab[:, 0]]], lines[ids[ab[:, 1]]])
        V = V[np.linalg.norm(V, axis=1) > 1e-12]
        if not len(V):
            break
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        score = ((_angles(L[ids], V) < tol_deg) * ln[ids][:, None]).sum(0)
        best = V[int(np.argmax(score))]
        inl = ids[_angles(L[ids], best[None])[:, 0] < tol_deg]
        if len(inl) < 2:
            break
        A = lines[inl] / np.linalg.norm(lines[inl][:, :2], axis=1, keepdims=True)
        _, _, vt = np.linalg.svd(A * ln[inl][:, None])
        out.append((vt[-1], float(ln[inl].sum()), int(len(inl))))
        alive[inl] = False
    return out


def focal_candidates(vps, size, min_support_frac=0.08, max_dist_frac=30.0):
    """Focal lengths (pixels) from each pair of finite, well-supported vanishing points of perpendicular directions."""
    w, h = size
    c = np.array([(w - 1) / 2.0, (h - 1) / 2.0])
    total = sum(s for _, s, _ in vps) or 1.0
    pts = [v[:2] / v[2] for v, s, _ in vps
           if s >= min_support_frac * total and abs(v[2]) > 1e-12 and np.linalg.norm(v[:2] / v[2] - c) <= max_dist_frac * w]
    out = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            f2 = -float(np.dot(pts[i] - c, pts[j] - c))
            if f2 > 0:
                out.append(np.sqrt(f2))
    return out


def frame_focals(gray, seed=0):
    L = segments(gray)
    return focal_candidates(vanishing_points(L, seed=seed), (gray.shape[1], gray.shape[0]))
