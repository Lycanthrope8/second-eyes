"""End-to-end check of the video tools on a synthetic recording: a board segment, then a later second board."""
import json, os, subprocess, sys
import numpy as np
import cv2
import yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qrecon import charuco as CH, geometry as G
from test_charuco import SPEC, _view, _look_at

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = G.K_matrix(900.0, 900.0, 640.0, 360.0)


def test_video_tools_end_to_end(tmp_path, monkeypatch):
    b1, b2 = CH.make_board(SPEC), CH.make_board(dict(SPEC, first_marker_id=20))
    T_b2 = np.eye(4); T_b2[:3, 3] = [2.5, 0.0, 1.0]                    # board 02 in board 01's frame
    fps, frames, poses = 10.0, [], []
    centre = np.array([0.14, 0.10, 0.0])
    for k in range(60):                                                 # 0-6 s: board 01, translated arc
        a = np.radians(-35 + 70 * k / 59)
        T = _look_at(centre + [0.5 * np.sin(a), 0.04 * np.cos(2 * a), -0.85 + 0.15 * np.cos(a)], centre)
        frames.append(_view(b1, T, K)); poses.append(T)
    for k in range(40):                                                 # 6-10 s: board 02
        a = np.radians(-30 + 60 * k / 39)
        Tl = _look_at(centre + [0.45 * np.sin(a), 0.03 * np.cos(a), -0.8], centre)
        frames.append(_view(b2, Tl, K)); poses.append(T_b2 @ Tl)
    video = str(tmp_path / "house.avi")
    vw = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), fps, (1280, 720))
    for f in frames:
        vw.write(f)
    vw.release()
    ref = yaml.safe_load(open(os.path.join(REPO, "configs", "references.yaml")))
    ref["boards"]["ref01"] = dict(SPEC, mounting="upright_level")
    ref["boards"]["ref02"] = dict(SPEC, first_marker_id=20, mounting="upright_level")
    ref_path = tmp_path / "references.yaml"
    yaml.safe_dump(ref, open(ref_path, "w"))
    calib = {"K": K.tolist(), "dist": [0, 0, 0, 0, 0], "size": [1280, 720]}
    json.dump(calib, open(tmp_path / "calib.json", "w"))
    scale = 2.2                                                          # tracker trajectory in arbitrary units
    with open(tmp_path / "traj.txt", "w") as f:
        from scipy.spatial.transform import Rotation
        for k, T in enumerate(poses):
            q = Rotation.from_matrix(T[:3, :3]).as_quat()
            p = T[:3, 3] / scale
            f.write(f"{k / fps:.6f} {p[0]} {p[1]} {p[2]} {q[0]} {q[1]} {q[2]} {q[3]}\n")
    tool = os.path.join(REPO, "scripts", "video_tools.py")
    env = dict(os.environ)
    # point the tool at the test's references.yaml
    cfg_dir = os.path.join(REPO, "configs")
    orig = open(os.path.join(cfg_dir, "references.yaml")).read()
    try:
        open(os.path.join(cfg_dir, "references.yaml"), "w").write(open(ref_path).read())
        out = subprocess.run([sys.executable, tool, "frames", "--video", video, "--out", str(tmp_path / "fr"), "--every", "2", "--scale", "0.5"],
                             capture_output=True, text=True, env=env)
        assert out.returncode == 0, out.stderr
        lines = [l for l in open(tmp_path / "fr" / "rgb.txt") if not l.startswith("#")]
        assert len(lines) == 50 and os.path.exists(tmp_path / "fr" / lines[0].split()[1])
        out = subprocess.run([sys.executable, tool, "charuco-inspect", "--video", video, "--end", "9", "--every", "20"],
                             capture_output=True, text=True, env=env)
        assert "DICT_5X5_100" in out.stdout, out.stdout + out.stderr
        out = subprocess.run([sys.executable, tool, "charuco-anchor", "--video", video, "--trajectory", str(tmp_path / "traj.txt"),
                              "--calib", str(tmp_path / "calib.json"), "--ref01-end", "5.9", "--ref02-start", "6.0", "--ref02-end", "10",
                              "--every", "1", "--out", str(tmp_path / "anchored.txt")], capture_output=True, text=True, env=env)
        assert out.returncode == 0, out.stdout + out.stderr
        rep = json.load(open(str(tmp_path / "anchored.txt") + ".json"))
        assert abs(rep["ref01"]["scale"] - scale) / scale < 0.02
        assert rep["ref02"]["ok"] and abs(rep["ref02"]["scale_drift"]) < 0.02
        link = np.array(rep["ref02"]["board02_in_board01"])
        assert np.allclose(link[:3, 3], [2.5, 0.0, 1.0], atol=0.05)
    finally:
        open(os.path.join(cfg_dir, "references.yaml"), "w").write(orig)


def test_missing_video_is_an_error_not_zero_frames(tmp_path):
    tool = os.path.join(REPO, "scripts", "video_tools.py")
    out = subprocess.run([sys.executable, tool, "frames", "--video", str(tmp_path / "nope.mp4"), "--out", str(tmp_path / "fr")],
                         capture_output=True, text=True)
    assert out.returncode != 0 and "video not found" in (out.stderr + out.stdout)


def test_report_flags_a_blank_stretch_and_finds_the_board(tmp_path):
    """Synthetic recording: 0-3 s textured scene with board 01, 3-5 s a bare wall, 5-8 s textured again."""
    ref = yaml.safe_load(open(os.path.join(REPO, "configs", "references.yaml")))
    board = CH.make_board(ref["boards"]["ref01"])
    rng = np.random.default_rng(0)
    tex = cv2.GaussianBlur((rng.random((900, 3000)) * 255).astype(np.uint8), (0, 0), 3)
    tex = cv2.normalize(tex, None, 0, 255, cv2.NORM_MINMAX)
    tex = cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)
    btex = cv2.cvtColor(board.generateImage((500, 300), marginSize=20), cv2.COLOR_GRAY2BGR)
    fps, video = 30.0, str(tmp_path / "walk.avi")
    vw = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), fps, (1280, 720))
    for k in range(240):
        t = k / fps
        if 3.0 <= t < 5.0:
            f = np.full((720, 1280, 3), 180, np.uint8)                       # bare wall
            f = cv2.add(f, (rng.normal(0, 1.5, f.shape)).clip(0, 255).astype(np.uint8))
        else:
            x0 = 4 * k
            f = tex[90:810, x0:x0 + 1280].copy()
            if t < 3.0:
                f[200:500, 390:890] = btex
        vw.write(f)
    vw.release()
    tool = os.path.join(REPO, "scripts", "video_tools.py")
    out = subprocess.run([sys.executable, tool, "report", "--video", video, "--out", str(tmp_path / "rep"), "--sheet-every", "2"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    rep = json.load(open(tmp_path / "rep" / "report.json"))
    assert rep["frames"] == 240 and abs(rep["fps"] - 30) < 0.01 and rep["constant_rate"]
    spans = rep["tracking_screen"]["risky_spans_s"]
    assert any(s <= 3.2 and e >= 4.8 for s, e in spans), spans                # the bare wall is flagged
    assert not any(e <= 3.0 or s >= 5.2 for s, e in spans), spans             # and nothing else
    b = rep["boards"]["ref01"]
    assert b and b[0]["start_s"] <= 0.5 and 2.0 <= b[0]["end_s"] < 3.0 and not rep["boards"]["ref02"]
    assert rep["tracking_screen"]["map_start_estimate_s"] == 0.0
    assert rep["suggested_windows_s"]["ref01_start"][0] == 0.0 and rep["suggested_windows_s"]["calibration"][0] == 0.0
    assert os.path.exists(tmp_path / "rep" / "contact.jpg")
    lines = open(tmp_path / "rep" / "timeline.csv").read().splitlines()
    assert len(lines) == 1 + 120
