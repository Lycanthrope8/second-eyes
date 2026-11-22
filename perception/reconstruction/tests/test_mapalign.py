import os, sys
import numpy as np
import cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qrecon import mapalign as MA  # noqa: E402

K = np.array([[420.0, 0, 319.5], [0, 420.0, 179.5], [0, 0, 1]])


def _texture(seed, h=700, w=1400):
    rng = np.random.default_rng(seed)
    t = sum(cv2.resize(rng.random((h // s, w // s)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC) * (s ** 0.5)
            for s in (2, 8, 32))
    return cv2.normalize(t, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)


PLANES = [(np.array([-3.0, -1.5, 4.0]), np.array([6.0, 0, 0]), np.array([0, 3.0, 0]), _texture(1)),      # wall ahead
          (np.array([3.0, -1.5, -1.0]), np.array([0, 0, 5.0]), np.array([0, 3.0, 0]), _texture(2)),      # wall to the right
          (np.array([-3.0, 1.5, -1.0]), np.array([6.0, 0, 0]), np.array([0, 0, 5.0]), _texture(3))]      # floor (y down)


def _render(T, size=(640, 360)):
    w, h = size
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1) @ T[:3, :3].T
    c = T[:3, 3]
    best, img = np.full((h, w), np.inf), np.zeros((h, w), np.float32)
    for o, e1, e2, tex in PLANES:
        n = np.cross(e1, e2)
        den = d @ n
        lam = ((o - c) @ n) / np.where(np.abs(den) < 1e-9, 1e-9, den)
        P = c + lam[..., None] * d
        a, b = (P - o) @ e1 / (e1 @ e1), (P - o) @ e2 / (e2 @ e2)
        ok = (lam > 0.05) & (a >= 0) & (a <= 1) & (b >= 0) & (b <= 1) & (lam < best)
        th, tw = tex.shape
        val = cv2.remap(tex, np.nan_to_num(a * (tw - 1)).astype(np.float32), np.nan_to_num(b * (th - 1)).astype(np.float32),
                        cv2.INTER_LINEAR)
        img[ok], best[ok] = val[ok], lam[ok]
    return img.clip(0, 255).astype(np.uint8)


def _pose(x, z, yaw_deg):
    a = np.radians(yaw_deg)
    T = np.eye(4)
    T[:3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
    T[:3, 3] = [x, 0.0, z]
    return T


def test_a_map_without_a_board_is_placed_by_the_views_it_shares():
    placed = [_pose(-1.5 + 0.1 * k, 0.0, -10 + 1.75 * k) for k in range(21)]          # metric, in the board's frame
    world_u = [_pose(-1.0 + 0.1 * k, 0.6, 0 + 1.75 * k) for k in range(21)]
    s, R = 2.0, cv2.Rodrigues(np.array([0.1, 0.5, -0.2]))[0]
    t = np.array([1.0, 2.0, 3.0])                                                     # tracker -> board frame (unknown)
    def to_tracker(T):
        o = np.eye(4); o[:3, :3] = R.T @ T[:3, :3]; o[:3, 3] = R.T @ (T[:3, 3] - t) / s
        return o
    pframes = [{"t": 0.3 * k, "T": T, "image": (lambda T=T: _render(T)), "frame": "ref02", "map": "P"} for k, T in enumerate(placed)]
    uframes = [{"t": 0.3 * k, "T": to_tracker(T), "image": (lambda T=T: _render(T))} for k, T in enumerate(world_u)]
    pdb = MA.p_database(pframes, K, step_s=0.5)
    assert len(pdb) >= 8 and all(len(p["pts"]) >= 30 for p in pdb)
    res, rep = MA.place(uframes, pdb, K, step_s=0.5)
    assert res is not None, rep.get("reason")
    assert res["frame"] == "ref02" and len(res["inliers"]) >= 6 and res["with_maps"] == ["P"]
    # 5%: ORB keypoints differ slightly between OpenCV builds (the laptop gave 3.0%, this sandbox under 3%)
    assert abs(res["s"] / s - 1) < 0.05 and MA.rot_deg(res["R"], R) < 1.5 and np.linalg.norm(res["t"] - t) < 0.15


def test_no_shared_views_means_no_placement():
    placed = [_pose(-1.5 + 0.1 * k, 0.0, -10 + 1.75 * k) for k in range(15)]
    pframes = [{"t": 0.3 * k, "T": T, "image": (lambda T=T: _render(T)), "frame": "ref01", "map": "P"} for k, T in enumerate(placed)]
    blank = [{"t": 0.3 * k, "T": np.eye(4), "image": (lambda: np.full((360, 640), 128, np.uint8))} for k in range(10)]
    res, rep = MA.place(blank, MA.p_database(pframes, K, step_s=0.5), K)
    assert res is None and "matched" in rep["reason"]


def test_fit_rejects_a_placement_whose_frames_disagree():
    rng = np.random.default_rng(0)
    Tu = np.tile(np.eye(4), (6, 1, 1)); Tp = np.tile(np.eye(4), (6, 1, 1))
    Tu[:, :3, 3] = rng.uniform(-1, 1, (6, 3)); Tp[:, :3, 3] = rng.uniform(-5, 5, (6, 3))           # unrelated
    res, why = MA.fit_sim3(Tu, Tp)
    assert res is None and "agree" in why


def test_a_placement_implying_impossible_speed_is_trimmed_or_refused():
    t = np.arange(0, 40, 0.5)
    c = np.zeros((len(t), 3))
    c[:, 0] = np.where(t < 20, 0.25 * t, 5.0 + 2.5 * (t - 20))      # tracker units; the second half drifted
    kind, span, speed = MA.choose_span(t, c, 2.0, [2.0, 5.0, 8.0], max_speed=2.0, pad_s=10.0)
    assert kind == "trimmed" and span == (0.0, 18.0) and abs(speed - 0.5) < 1e-6
    assert MA.choose_span(t[:40], c[:40], 2.0, [2.0, 8.0])[0] == "full"
    assert MA.choose_span(t, c, 20.0, [2.0, 8.0])[0] == "rejected"


def test_the_sharpest_frame_of_each_window_is_the_one_matched():
    import cv2
    rng = np.random.default_rng(4)
    sharp = (rng.random((240, 320)) * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 6)
    frames = [{"t": k / 30.0, "image": (lambda im=(sharp if k in (7, 22) else blurred): im)} for k in range(30)]
    picks = MA.sharp_picks(frames, 0.5, stride=1)
    assert picks == [7, 22]                                             # one per half second, the sharp one in each
    assert MA.sharpness(sharp) > 10 * MA.sharpness(blurred)
