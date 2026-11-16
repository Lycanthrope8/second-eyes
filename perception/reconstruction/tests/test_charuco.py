"""ChArUco tests on synthetic views: detection on rendered images, calibration, anchoring and drift."""
import os, sys
import numpy as np
import cv2
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qrecon import charuco as CH, config as C, geometry as G

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = C.load(os.path.join(REPO, "configs", "references.yaml"))
SPEC = {"dictionary": "DICT_5X5_100", "squares_x": 7, "squares_y": 5, "square_length_m": 0.04,
        "marker_length_m": 0.03, "first_marker_id": 0, "legacy_pattern": False}
K = G.K_matrix(900.0, 900.0, 640.0, 360.0)
SIZE = (1280, 720)


def _view(board, T_wc, K, dist=np.zeros(5), px_per_m=4000):
    """Render the board through camera T_wc (board at z=0 in world = board frame)."""
    W = int(SPEC["squares_x"] * SPEC["square_length_m"] * px_per_m)
    H = int(SPEC["squares_y"] * SPEC["square_length_m"] * px_per_m)
    tex = board.generateImage((W, H), marginSize=0)
    corners_m = np.array([[0, 0, 0], [W / px_per_m, 0, 0], [W / px_per_m, H / px_per_m, 0], [0, H / px_per_m, 0]])
    Tcw = G.inv(T_wc)
    rvec, _ = cv2.Rodrigues(Tcw[:3, :3])
    img_c, _ = cv2.projectPoints(corners_m, rvec, Tcw[:3, 3], K, dist)
    Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [W, 0], [W, H], [0, H]]), img_c.reshape(4, 2).astype(np.float32))
    out = cv2.warpPerspective(tex, Hm, SIZE, borderValue=200)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def _look_at(cam, target):
    f = target - cam; f /= np.linalg.norm(f)
    r = np.cross(f, np.array([0, -1.0, 0])); r /= np.linalg.norm(r)       # board y is down, so "up" is -y
    d = np.cross(f, r)
    T = np.eye(4); T[:3, :3] = np.stack([r, d, f], axis=1); T[:3, 3] = cam
    return T


def _poses(n, centre, radius=0.5, dist=0.9):
    out = []
    for k in range(n):
        a = np.radians(-35 + 70 * k / max(n - 1, 1))
        cam = centre + np.array([radius * np.sin(a), 0.05 * np.cos(3 * a), -dist])
        out.append(_look_at(cam, centre))
    return out


def test_detects_rendered_board():
    board = CH.make_board(SPEC)
    centre = np.array([0.14, 0.10, 0.0])
    det = CH.detect(_view(board, _poses(1, centre)[0], K), board, 6)
    assert det is not None and len(det[0]) >= 12


def test_calibration_from_rendered_views():
    board = CH.make_board(SPEC)
    centre = np.array([0.14, 0.10, 0.0])
    dets = []
    for k, T in enumerate(_poses(14, centre, radius=0.45, dist=0.55)):
        T = T.copy(); T[:3, 3] += [0.06 * np.sin(k), 0.05 * np.cos(2 * k), 0.08 * np.sin(0.7 * k)]
        T = _look_at(T[:3, 3], centre + [0.04 * np.cos(k), 0.03 * np.sin(k), 0])
        d = CH.detect(_view(board, T, K), board, 6)
        if d is not None:
            dets.append(d)
    out = CH.calibrate(dets, board, SIZE, dict(REF["calibration"], profile="full"))
    weak = CH.calibrate(dets, board, SIZE, REF["calibration"])
    assert abs(np.array(weak["K"])[0, 0] - 900) / 900 < 0.03 and "focal_std_fraction" in weak
    assert out["rms_px"] < 1.0
    f = np.array(out["K"])
    assert abs(f[0, 0] - 900) / 900 < 0.03 and abs(f[1, 1] - 900) / 900 < 0.03


def _observe(board, poses, K, rng, noise=0.3):
    obs = {}
    allp = np.asarray(board.getChessboardCorners(), dtype=np.float64)
    for f, T in enumerate(poses):
        u, v, z = G.project(allp, K, T)
        ok = (z > 0) & (u > 0) & (u < SIZE[0]) & (v > 0) & (v < SIZE[1])
        ids = np.nonzero(ok)[0]
        obs[f] = (ids, np.stack([u[ids], v[ids]], axis=1) + rng.normal(0, noise, (len(ids), 2)))
    return obs


def test_anchor_recovers_scale_and_checks_held_out_views():
    rng = np.random.default_rng(2)
    board = CH.make_board(SPEC)
    poses_metric = _poses(16, np.array([0.14, 0.10, 0.0]))
    true_scale = 3.1
    poses_tracker = []
    for T in poses_metric:
        Tt = T.copy(); Tt[:3, 3] /= true_scale; poses_tracker.append(Tt)
    out = CH.anchor(_observe(board, poses_metric, K, rng), poses_tracker, K, np.zeros(5), board, REF["anchor"])
    assert out["ok"], out
    assert abs(out["scale"] - true_scale) / true_scale < 0.01
    assert out["held_out_frames"] >= 3 and out["held_out_reprojection_px"]["median"] < 1.0


def test_anchor_refuses_rotation_only_views():
    rng = np.random.default_rng(3)
    board = CH.make_board(SPEC)
    cam = np.array([0.14, 0.10, -0.9])
    poses = [_look_at(cam, np.array([0.14 + 0.02 * k, 0.10, 0.0])) for k in range(10)]
    out = CH.anchor(_observe(board, poses, K, rng), poses, K, np.zeros(5), board, REF["anchor"])
    assert not out["ok"]


def test_second_board_measures_scale_drift():
    rng = np.random.default_rng(4)
    b1, b2 = CH.make_board(SPEC), CH.make_board(dict(SPEC, first_marker_id=20))
    T_b2_in_b1 = np.eye(4); T_b2_in_b1[:3, 3] = [3.0, 0.0, 1.5]           # board 02 three metres away
    drift = 0.04                                                          # tracker scale 4% too large by then
    poses_b2frame = _poses(12, np.array([0.14, 0.10, 0.0]))
    obs2 = _observe(b2, poses_b2frame, K, rng)
    poses_map = []
    for T in poses_b2frame:
        Tm = T_b2_in_b1 @ T
        Tm[:3, 3] *= (1 + drift)
        poses_map.append(Tm)
    out = CH.second_board(obs2, poses_map, K, np.zeros(5), b2, REF["anchor"])
    assert out["ok"] and abs(out["scale_drift"] - drift) < 0.005


def test_upright_board_gives_z_up_map_frame():
    R = CH.map_frame_rotation("upright_level")
    assert np.allclose(R @ np.array([0, -1.0, 0]), [0, 0, 1.0])             # board 'up' (-y) becomes +z
    assert abs(np.linalg.det(R) - 1) < 1e-12
    with pytest.raises(ValueError):
        CH.map_frame_rotation("other")
