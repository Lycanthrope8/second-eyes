"""ChArUco reference boards: detection, camera calibration from views inside a recording, and metric anchoring.

Board frame (OpenCV's): origin at the board's top-left corner, x to the right along the top edge, y down along the
left edge, z into the board. Units are metres. Every physical value comes from configs/references.yaml."""
import numpy as np
import cv2
from .geometry import umeyama, inv

_UPRIGHT = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])   # board -> map with z up, for a level, upright board


def make_board(spec):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec["dictionary"]))
    n_markers = (spec["squares_x"] * spec["squares_y"]) // 2
    ids = np.arange(spec.get("first_marker_id", 0), spec.get("first_marker_id", 0) + n_markers, dtype=np.int32)
    board = cv2.aruco.CharucoBoard((spec["squares_x"], spec["squares_y"]), float(spec["square_length_m"]),
                                   float(spec["marker_length_m"]), d, ids)
    if spec.get("legacy_pattern", False):
        board.setLegacyPattern(True)
    return board


def detect(bgr, board, min_corners):
    """Returns (ids (N,), pixels (N,2)) of interpolated chessboard corners, or None."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    cc, ci, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
    if ci is None or len(ci) < min_corners:
        return None
    return ci.ravel().astype(int), cc.reshape(-1, 2).astype(np.float64)


def model_points(board, ids):
    return np.asarray(board.getChessboardCorners(), dtype=np.float64)[ids]


def calibrate(detections, board, image_size, cfg):
    """detections: list of (ids, pixels). Picks spread-out views, calibrates, reports error and image coverage."""
    return calibrate_views([(board, ids, px) for ids, px in detections], image_size, cfg)


def calibrate_views(views, image_size, cfg, fix_k1=False):
    """views: list of (board, ids, pixels), from one board or several. Intrinsics do not depend on a board's printed
    size (only the views' distances do), so a board whose print scale is unmeasured still constrains them.
    fix_k1 holds the radial distortion at zero, for comparing against the estimate that frees it."""
    usable = [v for v in views if len(v[1]) >= cfg["min_corners"]]
    if len(usable) < cfg["min_views"]:
        return {"ok": False, "reason": f"only {len(usable)} usable board views; need {cfg['min_views']}"}
    pick = np.linspace(0, len(usable) - 1, min(cfg["max_views"], len(usable))).round().astype(int)
    chosen = [usable[i] for i in sorted(set(pick))]
    obj = [model_points(b, ids).astype(np.float32) for b, ids, _ in chosen]
    img = [px.astype(np.float32).reshape(-1, 1, 2) for _, _, px in chosen]
    names = list(cfg["profiles"][cfg["profile"]] if "profiles" in cfg else cfg.get("flags", []))
    if fix_k1 and "CALIB_FIX_K1" not in names:
        names.append("CALIB_FIX_K1")
    flags = 0
    for name in names:
        flags |= getattr(cv2, name)
    w0, h0 = image_size
    K0 = np.array([[1.0, 0, (w0 - 1) / 2.0], [0, 1.0, (h0 - 1) / 2.0], [0, 0, 1]])   # aspect 1, centre principal point
    rms, K, dist, rvecs, tvecs, std_in, _, _ = cv2.calibrateCameraExtended(obj, img, tuple(image_size), K0, None, flags=flags)
    std_in = std_in.ravel()
    allpx = np.concatenate([px for _, _, px in chosen])
    g = cfg["coverage_grid"]
    w, h = image_size
    cells = set(zip((allpx[:, 0] * g[0] // w).astype(int).clip(0, g[0] - 1), (allpx[:, 1] * g[1] // h).astype(int).clip(0, g[1] - 1)))
    cov = len(cells) / float(g[0] * g[1])
    dists = [float(np.linalg.norm(t)) for t in tvecs]
    f_std = float(max(std_in[0], std_in[1]) / max(K[0, 0], 1e-9))
    out = {"ok": True, "K": K.tolist(), "dist": dist.ravel().tolist(), "size": list(image_size), "rms_px": float(rms),
           "views": len(chosen), "image_coverage": cov, "board_distance_m": [min(dists), max(dists)], "flags": names,
           "focal_std_fraction": f_std}
    if rms > cfg["max_rms_px"]:
        out.update(ok=False, reason=f"reprojection RMS {rms:.2f} px above {cfg['max_rms_px']}")
    elif f_std > cfg.get("max_focal_std_fraction", 1.0):
        out["warning"] = f"focal length uncertain: standard deviation {100 * f_std:.1f}% of its value"
    elif cov < cfg["min_image_coverage"]:
        out["warning"] = (f"board corners cover {cov:.0%} of the image; lens distortion is poorly determined outside that "
                          "area, so fewer distortion terms are estimated (see flags)")
    return out


def undistort_px(px, K, dist):
    return cv2.undistortPoints(px.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)


def triangulate_multi(K, poses_wc, pixels):
    """Linear triangulation of one point from several views (pixels already undistorted)."""
    A = []
    for T, (u, v) in zip(poses_wc, pixels):
        P = K @ inv(T)[:3, :]
        A.append(u * P[2] - P[0]); A.append(v * P[2] - P[1])
    X = np.linalg.svd(np.array(A))[2][-1]
    return X[:3] / X[3]


def _corners_in_tracker(obs, poses, K, dist, cfg):
    """obs: {frame: (ids, pixels)}. Triangulates each corner id seen in enough views with enough parallax."""
    seen = {}
    for f, (ids, px) in obs.items():
        upx = undistort_px(px, K, dist)
        for i, p in zip(ids, upx):
            seen.setdefault(int(i), []).append((f, p))
    pts, used = {}, 0
    for cid, lst in seen.items():
        if len(lst) < cfg["min_views_per_corner"]:
            continue
        centres = np.array([poses[f][:3, 3] for f, _ in lst])
        X = triangulate_multi(K, [poses[f] for f, _ in lst], [p for _, p in lst])
        rays = centres - X
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        spread = np.degrees(np.arccos(np.clip(np.min(rays @ rays.T), -1, 1)))
        if spread >= cfg["min_parallax_deg"]:
            pts[cid] = X
            used += 1
    return pts


def anchor(obs, poses, K, dist, board, cfg, holdout_every=None):
    """Similarity (s, R, t) taking tracker coordinates to the board frame, from corners triangulated with the
    tracker's poses and fitted to the measured board. Some views are held out to check it independently."""
    frames = sorted(obs)
    if len(frames) < cfg["min_views"]:
        return {"ok": False, "reason": f"board seen in {len(frames)} frames; need {cfg['min_views']}"}
    k = holdout_every or cfg["holdout_every"]
    held = set(frames[1::k]) if k and len(frames) >= 2 * cfg["min_views"] else set()
    fit_obs = {f: o for f, o in obs.items() if f not in held}
    pts = _corners_in_tracker(fit_obs, poses, K, dist, cfg)
    if len(pts) < cfg["min_corners"]:
        return {"ok": False, "reason": "board views are not translated enough to triangulate its corners",
                "frames": len(frames), "triangulated_corners": len(pts)}
    ids = np.array(sorted(pts))
    src = np.array([pts[i] for i in ids])
    dst = model_points(board, ids)
    s, R, t = umeyama(src, dst)
    resid = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
    out = {"ok": True, "scale": s, "R": R, "t": t, "frames": len(frames), "held_out_frames": len(held),
           "triangulated_corners": int(len(ids)), "corner_fit_rms_m": float(np.sqrt(np.mean(resid ** 2))),
           "corner_fit_max_m": float(resid.max())}
    def reproject(frame_ids):
        """Per frame: pixel errors of the measured board seen through the anchored path, and the distance at the
        board they amount to (pixels x depth / focal length), comparable with corner_fit_rms_m."""
        rows = []
        for f in frame_ids:
            fid, fpx = obs[f]
            T_b = poses[f].copy()
            T_b[:3, :3] = R @ T_b[:3, :3]
            T_b[:3, 3] = s * R @ T_b[:3, 3] + t
            P = model_points(board, fid)
            Tcb = inv(T_b)
            pc = P @ Tcb[:3, :3].T + Tcb[:3, 3]
            uv = np.stack([K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2], K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]], axis=1)
            e = np.linalg.norm(uv - undistort_px(fpx, K, dist), axis=1)
            rows.append((f, e, e * pc[:, 2] / K[0, 0], pc[:, 2]))
        return rows

    fit_rows = reproject(sorted(fit_obs))
    if fit_rows:
        e = np.concatenate([r[1] for r in fit_rows])
        out["fit_reprojection_px"] = {"median": float(np.median(e)), "p95": float(np.percentile(e, 95))}
    if held:
        rows = reproject(sorted(held))
        e = np.concatenate([r[1] for r in rows])
        em = np.concatenate([r[2] for r in rows])
        out["held_out_reprojection_px"] = {"median": float(np.median(e)), "p95": float(np.percentile(e, 95))}
        out["held_out_error_m"] = {"median": float(np.median(em)), "p95": float(np.percentile(em, 95))}
        out["held_out_detail"] = [{"frame": int(f), "corners": int(len(ep)), "distance_m": float(np.median(z)),
                                   "median_px": float(np.median(ep)), "median_m": float(np.median(m))}
                                  for f, ep, m, z in rows]
        metric = "max_held_out_m" in cfg              # a limit in metres at the board takes precedence over pixels
        value, limit = (np.median(em), cfg["max_held_out_m"]) if metric else (np.median(e), cfg["max_held_out_px"])
        out["held_out_limit"] = {"value": float(limit), "unit": "m" if metric else "px"}
        if value > limit:
            out.update(ok=False, reason="held-out views of the board disagree with the anchored trajectory")
    if out["ok"] and out["corner_fit_rms_m"] > cfg["max_corner_fit_rms_m"]:
        out.update(ok=False, reason="triangulated corners do not fit the measured board")
    return out


def second_board(obs2, poses_map, K, dist, board2, cfg):
    """Board 02 seen later in the same capture, with poses already in board 01's metric frame. Its fitted scale
    should be 1: the difference is the scale drift accumulated between the two boards."""
    pts = _corners_in_tracker(obs2, poses_map, K, dist, cfg)
    if len(pts) < cfg["min_corners"]:
        return {"ok": False, "reason": "board 02 not triangulated", "triangulated_corners": len(pts)}
    ids = np.array(sorted(pts))
    s, R, t = umeyama(np.array([pts[i] for i in ids]), model_points(board2, ids))
    T = np.eye(4); T[:3, :3] = R.T; T[:3, 3] = -R.T @ t / s           # board-02 frame expressed in board-01 frame
    return {"ok": True, "scale_ratio": float(s), "scale_drift": float(1.0 / s - 1.0), "triangulated_corners": int(len(ids)),
            "board02_in_board01": T}


def map_frame_rotation(mounting):
    """Rotation from the board frame to a z-up map frame. Valid only for a board mounted upright and level."""
    if mounting != "upright_level":
        raise ValueError("map_frame_rotation needs an upright, level board; otherwise take gravity from the floor plane")
    return _UPRIGHT.copy()
