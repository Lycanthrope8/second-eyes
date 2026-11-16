"""Focal length from a room's straight lines (vanishing points), on synthetic rooms and through video_tools."""
import json, os, subprocess, sys
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qrecon import vanishing as VP

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _room(f, w, h, yaw, pitch, seed, thickness=2):
    """Axis-aligned edges of a box room seen by a pinhole camera with focal length f (pixels), centre principal point."""
    rng = np.random.default_rng(seed)
    R = cv2.Rodrigues(np.array([np.radians(pitch), 0, 0]))[0] @ cv2.Rodrigues(np.array([0, np.radians(yaw), 0]))[0]
    K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]])
    img = np.full((h, w), 128, np.uint8)
    for _ in range(70):
        ax = rng.integers(0, 3)
        p = rng.uniform([-2, -1.3, 1.0], [2, 1.3, 6.0]); q = p.copy(); q[ax] += rng.uniform(0.3, 1.5)
        P = np.stack([p, q]) @ R.T
        if (P[:, 2] < 0.3).any():
            continue
        uv = P @ K.T; uv = uv[:, :2] / uv[:, 2:]
        cv2.line(img, tuple(int(x) for x in uv[0]), tuple(int(x) for x in uv[1]), int(rng.choice([20, 235])), thickness, cv2.LINE_AA)
    return img


def test_focal_from_vanishing_points_in_synthetic_rooms():
    for f in (609.0, 819.0):
        est = []
        for s in range(8):
            est += VP.frame_focals(_room(f, 960, 540, yaw=20 + 5 * s, pitch=-6 - s, seed=s), seed=s)
        assert abs(np.median(est) - f) / f < 0.04, (f, np.median(est))


def test_focal_from_lines_command(tmp_path):
    video = str(tmp_path / "room.avi")
    vw = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (1920, 1080))
    for k in range(40):
        vw.write(cv2.cvtColor(_room(1218.0, 1920, 1080, yaw=15 + 1.5 * k, pitch=-5 - 0.2 * k, seed=k, thickness=4), cv2.COLOR_GRAY2BGR))
    vw.release()
    out = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "video_tools.py"), "focal-from-lines", "--video", video,
                          "--every-s", "0.2", "--out", str(tmp_path / "f.json")], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    rep = json.load(open(tmp_path / "f.json"))
    assert rep["frames"] == 20 and abs(rep["focal_px_median"] - 1218) / 1218 < 0.04, rep
