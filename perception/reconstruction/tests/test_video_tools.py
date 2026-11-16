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
