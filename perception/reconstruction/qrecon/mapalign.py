"""Placing a map that never saw a board, by the views it shares with maps that did (O42, option a).

A placed map has metric camera poses in a board's frame (its reconstruction's anchored_trajectory.txt). Frames of it are
sampled; the ORB features of each are triangulated against a neighbouring frame of the same map with those known poses,
which gives 3-D points in the board's frame. Frames of the unplaced map are compared with the placed frames (first by a
small bag of binary words, then by descriptor matching) and located by PnP on those 3-D points. The similarity that takes
the unplaced map's own tracker coordinates into the board's frame is then fitted to the located frames: rotation from
their orientations, scale and offset from their camera centres, with RANSAC over pairs of frames and a least-squares fit
on the inliers. Scale drift inside a long map is not modelled: the fit is an average over the located frames.
"""
import itertools
import numpy as np
import cv2


def quat_to_R(qx, qy, qz, qw):
    q = np.array([qw, qx, qy, qz], float)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def read_tum(path):
    """Times (N,) and camera-to-world poses (N, 4, 4) from a TUM trajectory file."""
    rows = [l.split() for l in open(path) if l.strip() and not l.startswith("#")]
    t = np.array([float(r[0]) for r in rows])
    T = np.tile(np.eye(4), (len(rows), 1, 1))
    for i, r in enumerate(rows):
        x, y, z, qx, qy, qz, qw = map(float, r[1:8])
        T[i, :3, :3] = quat_to_R(qx, qy, qz, qw)
        T[i, :3, 3] = [x, y, z]
    return t, T


def features(gray, n=1500, fast=12):
    orb = cv2.ORB_create(nfeatures=n, scaleFactor=1.2, nlevels=8, fastThreshold=fast)
    kp, d = orb.detectAndCompute(gray, None)
    if d is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 32), np.uint8)
    return np.array([k.pt for k in kp], np.float32), d


def match(d1, d2, ratio=0.8):
    if len(d1) < 2 or len(d2) < 2:
        return []
    out = []
    for m in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2):
        if len(m) == 2 and m[0].distance < ratio * m[1].distance:
            out.append((m[0].queryIdx, m[0].trainIdx))
    return out


class Vocab:
    """A small bag of binary words (k-means on the bits of ORB descriptors), to pick candidate frames quickly."""

    def __init__(self, descriptor_sets, k=128, seed=0, sample=30000):
        sets = [d for d in descriptor_sets if len(d)]
        bits = np.concatenate([np.unpackbits(d, axis=1) for d in sets]).astype(np.float32)
        rng = np.random.default_rng(seed)
        if len(bits) > sample:
            bits = bits[rng.choice(len(bits), sample, replace=False)]
        k = int(min(k, max(2, len(bits) // 10)))
        cv2.setRNGSeed(seed)
        _, _, self.C = cv2.kmeans(bits, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5), 2,
                                  cv2.KMEANS_PP_CENTERS)
        self.cc = (self.C ** 2).sum(1)
        df = np.zeros(len(self.C))
        for d in sets:
            df[np.unique(self.words(d))] += 1
        self.idf = np.log((len(sets) + 1.0) / (df + 1.0))

    def words(self, d):
        b = np.unpackbits(d, axis=1).astype(np.float32)
        return np.argmin(self.cc[None, :] - 2.0 * b @ self.C.T, axis=1)

    def vector(self, d):
        if not len(d):
            return np.zeros(len(self.C))
        h = np.bincount(self.words(d), minlength=len(self.C)).astype(np.float64) * self.idf
        n = np.linalg.norm(h)
        return h / n if n > 0 else h


def projection(K, T_wc):
    R = T_wc[:3, :3].T
    return K @ np.hstack([R, (-R @ T_wc[:3, 3])[:, None]])


def triangulate(kp1, d1, T1, kp2, d2, T2, K, max_px=2.0, min_parallax_deg=1.0):
    """3-D points (in the poses' frame) for features of frame 1, from its matches in frame 2: {feature index: xyz}."""
    m = match(d1, d2)
    if len(m) < 8:
        return {}
    i1 = np.array([a for a, _ in m])
    i2 = np.array([b for _, b in m])
    P1, P2 = projection(K, T1), projection(K, T2)
    X = cv2.triangulatePoints(P1, P2, kp1[i1].T.astype(np.float64), kp2[i2].T.astype(np.float64))
    X = (X[:3] / X[3]).T
    Xh = np.hstack([X, np.ones((len(X), 1))])
    ok = np.ones(len(X), bool)
    for P, kp, ii in ((P1, kp1, i1), (P2, kp2, i2)):
        h = Xh @ P.T
        ok &= h[:, 2] > 1e-6
        err = np.linalg.norm(h[:, :2] / np.where(np.abs(h[:, 2:]) < 1e-9, 1e-9, h[:, 2:]) - kp[ii], axis=1)
        ok &= err < max_px
    a, b = X - T1[:3, 3], X - T2[:3, 3]
    cosang = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    ok &= np.degrees(np.arccos(np.clip(cosang, -1, 1))) >= min_parallax_deg
    return {int(i1[j]): X[j] for j in np.flatnonzero(ok)}


def locate(kp_u, d_u, d_p, pts_p, K, min_inliers=25, px=3.0):
    """Camera-to-world pose of a frame in the placed map's frame, by PnP on that map's 3-D points; or None."""
    m = [(a, b) for a, b in match(d_u, d_p) if b in pts_p]
    if len(m) < min_inliers:
        return None
    obj = np.array([pts_p[b] for _, b in m], np.float64)
    img = np.array([kp_u[a] for a, _ in m], np.float64)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, K, None, iterationsCount=300, reprojectionError=px,
                                             confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inl is None or len(inl) < min_inliers:
        return None
    inl = inl.ravel()
    ok, rvec, tvec = cv2.solvePnP(obj[inl], img[inl], K, None, rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = (-R.T @ tvec).ravel()
    return T, int(len(inl)), len(m)


def rot_deg(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def chordal_mean(Rs):
    U, _, Vt = np.linalg.svd(np.sum(Rs, axis=0))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ D @ Vt


def fit_sim3(Tu, Tp, max_center_m=0.15, max_rot_deg=6.0, min_frames=4, min_baseline_m=0.3):
    """Similarity (s, R, t) with Tp ~ S(Tu): world_p = s R world_u + t. Hypotheses from pairs of located frames (rotation
    from one frame's orientations, scale from the pair's baseline), inliers by centre and orientation, then the rotation
    averaged and scale and offset fitted by least squares on the inliers. Returns a dict, or None with a reason."""
    n = len(Tu)
    if n < min_frames:
        return None, f"{n} located frames, fewer than {min_frames}"
    cu, cp = Tu[:, :3, 3], Tp[:, :3, 3]
    Rrel = np.array([Tp[i, :3, :3] @ Tu[i, :3, :3].T for i in range(n)])
    best = None
    for i, j in itertools.combinations(range(n), 2):
        bp, bu = np.linalg.norm(cp[j] - cp[i]), np.linalg.norm(cu[j] - cu[i])
        if bp < 0.1 or bu < 1e-9:
            continue
        for R in (Rrel[i], Rrel[j]):
            s = bp / bu
            t = cp[i] - s * R @ cu[i]
            ec = np.linalg.norm(s * cu @ R.T + t - cp, axis=1)
            er = np.array([rot_deg(R, Rrel[k]) for k in range(n)])
            inl = np.flatnonzero((ec < max_center_m) & (er < max_rot_deg))
            if best is None or len(inl) > len(best):
                best = inl
    if best is None or len(best) < min_frames:
        return None, f"no consistent placement: at most {0 if best is None else len(best)} of {n} located frames agree"
    R = chordal_mean(Rrel[best])
    a = cu[best] @ R.T
    am, bm = a.mean(0), cp[best].mean(0)
    s = float(((a - am) * (cp[best] - bm)).sum() / max(((a - am) ** 2).sum(), 1e-12))
    t = bm - s * am
    if s <= 0:
        return None, "the fitted scale is not positive"
    base = max(np.linalg.norm(cp[i] - cp[j]) for i, j in itertools.combinations(best, 2))
    if base < min_baseline_m:
        return None, f"located frames span only {base:.2f} m, under {min_baseline_m} m: the scale is not determined"
    ec = np.linalg.norm(s * cu[best] @ R.T + t - cp[best], axis=1)
    er = np.array([rot_deg(R, Rrel[k]) for k in best])
    return {"s": s, "R": R, "t": t, "inliers": [int(k) for k in best], "located": n,
            "center_rms_m": float(np.sqrt((ec ** 2).mean())), "rotation_rms_deg": float(np.sqrt((er ** 2).mean())),
            "baseline_m": float(base)}, None


def p_database(frames, K, step_s=1.0, n=1500, offsets_s=(3.0, 2.4, 1.8, 1.2, 0.9), min_baseline_m=0.25,
               min_points=30, agree_m=0.05, agree_rel=0.02):
    """Placed-map frames with 3-D points. frames: dicts {t, T (camera-to-world, metric), image (callable -> gray),
    frame, map}, in time order. Each sampled frame is triangulated against the two neighbours with the longest
    baselines (offsets tried longest first, both directions); a point is kept when both agree, which removes false
    matches that happen to lie on an epipolar line. With one usable neighbour only, its points are kept as they are."""
    times = np.array([f["t"] for f in frames])
    order = [o for off in offsets_s for o in (off, -off)]
    out, last, cache = [], -1e9, {}

    def feat(k):
        if k not in cache:
            cache[k] = features(frames[k]["image"](), n)
        return cache[k]
    for i, f in enumerate(frames):
        if f["t"] - last < step_s:
            continue
        kp, d = feat(i)
        sets = []
        for off in order:
            j = int(np.argmin(np.abs(times - (f["t"] + off))))
            if j == i or abs(times[j] - f["t"] - off) > 0.25:
                continue
            if np.linalg.norm(frames[j]["T"][:3, 3] - f["T"][:3, 3]) < min_baseline_m:
                continue
            kp2, d2 = feat(j)
            pts = triangulate(kp, d, f["T"], kp2, d2, frames[j]["T"], K)
            if len(pts) >= min_points:
                sets.append(pts)
                if len(sets) == 2:
                    break
        if len(sets) == 2:
            c = f["T"][:3, 3]
            pts = {}
            for k in set(sets[0]) & set(sets[1]):
                x0, x1 = sets[0][k], sets[1][k]
                if np.linalg.norm(x0 - x1) < max(agree_m, agree_rel * np.linalg.norm(x0 - c)):
                    pts[k] = 0.5 * (x0 + x1)
        else:
            pts = sets[0] if sets else {}
        if len(pts) >= min_points:
            out.append({"t": f["t"], "T": f["T"], "kp": kp, "d": d, "pts": pts, "frame": f["frame"], "map": f["map"]})
            last = f["t"]
        for k in [k for k in cache if times[k] < f["t"] - 4.0]:
            del cache[k]
    return out


def place(u_frames, pdb, K, vocab=None, step_s=0.5, top=5, n=1500, min_inliers=25, **fit_args):
    """Locates frames of an unplaced map (dicts {t, T (its own tracker coordinates), image}) against a placed-map
    database, and fits the similarity into the frame most of them were located in. Returns (result or None, report)."""
    report = {"sampled": 0, "located": [], "frame": None}
    if not pdb:
        return None, dict(report, reason="no placed frames to compare with")
    if vocab is None:
        vocab = Vocab([p["d"] for p in pdb])
    pv = np.array([vocab.vector(p["d"]) for p in pdb])
    last = -1e9
    for f in u_frames:
        if f["t"] - last < step_s:
            continue
        last = f["t"]
        report["sampled"] += 1
        kp, d = features(f["image"](), n)
        if len(d) < min_inliers:
            continue
        scores = pv @ vocab.vector(d)
        best = None
        for j in np.argsort(-scores)[:top]:
            r = locate(kp, d, pdb[j]["d"], pdb[j]["pts"], K, min_inliers=min_inliers)
            if r and (best is None or r[1] > best[1]):
                best = (int(j), r[1], r[0])
        if best:
            p = pdb[best[0]]
            report["located"].append({"t": f["t"], "T_u": f["T"], "T_p": best[2], "frame": p["frame"], "map": p["map"],
                                      "inliers": best[1]})
    if not report["located"]:
        return None, dict(report, reason=f"none of {report['sampled']} sampled frames matched a placed frame")
    frames = {}
    for L in report["located"]:
        frames.setdefault(L["frame"], []).append(L)
    F, Ls = max(frames.items(), key=lambda kv: len(kv[1]))
    report["frame"] = F
    res, why = fit_sim3(np.array([L["T_u"] for L in Ls]), np.array([L["T_p"] for L in Ls]), **fit_args)
    if res is None:
        return None, dict(report, reason=why)
    res["frame"] = F
    res["with_maps"] = sorted(set(Ls[k]["map"] for k in res["inliers"]))
    return res, report
