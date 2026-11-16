"""The headset program against the desktop harness, on the same synthetic input. Needs a desktop build of
questpipe/qrecon_quest (set QRECON_QUEST_BIN); skipped otherwise."""
import json, os, subprocess, sys
import numpy as np
import cv2
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qrecon import config as C, synthetic, pipeline as P, depth_models as DM, metrics as M, questconfig as QC, charuco as CH
from qrecon.sources import TumSource
from qrecon.fusion import read_ply_xyz

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.environ.get("QRECON_QUEST_BIN", "")
pytestmark = pytest.mark.skipif(not (BIN and os.path.exists(BIN)), reason="no desktop build of qrecon_quest")


def _run(cfg, out):
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, "recon.json")
    json.dump(cfg, open(p, "w"), indent=1)
    r = subprocess.run([BIN, "--config", p, "--out", out], capture_output=True, text=True)
    return r, json.load(open(os.path.join(out, "stats.json"))), json.load(open(os.path.join(out, "anchor.json")))


def test_dense_stages_match_the_desktop_harness(tmp_path):
    s = synthetic.write_tum(str(tmp_path / "seq"))
    cfg_y = C.load(os.path.join(REPO, "configs", "qrecon.yaml"))
    oracle = dict(cfg_y["arms"]["oracle"], noise=0.0)
    src = TumSource(s["dir"], s)
    kf = P.choose_keyframes(src, cfg_y)
    os.makedirs(tmp_path / "ref"); os.makedirs(tmp_path / "py")
    P.run_reference(src, kf, cfg_y, str(tmp_path / "ref"))
    poses = [src.frame(i).T_wc for i in kf]
    py_sum, _ = P.run_dense(src, kf, poses, DM.OracleRelative(oracle), cfg_y, str(tmp_path / "py"), log=lambda *_: None)
    cam = QC.camera_block({"K": src.K.tolist(), "dist": [0] * 5, "size": [s["camera"]["width"], s["camera"]["height"]]})
    depth = {"source": "oracle", "oracle": {"depth_list": os.path.join(s["dir"], "depth.txt"), "depth_factor": s["depth_factor"],
                                             "a": oracle["a"], "b": oracle["b"]}}
    cfg = QC.build(os.path.join(s["dir"], "rgb.txt"), os.path.join(s["dir"], "groundtruth.txt"), cam, depth)
    r, stats, rep = _run(cfg, str(tmp_path / "cpp"))
    assert r.returncode == 0, r.stderr
    assert stats["keyframes"] == len(kf) and stats["valid_keyframes"] == py_sum["valid_keyframes"]
    # Both fit scale and shift to landmarks triangulated from their own ORB matches, so their depths differ by the
    # landmark noise (about 1% here); the stages themselves must agree, and both must reach the same accuracy.
    cpp = read_ply_xyz(str(tmp_path / "cpp" / "cloud.ply"))
    py = read_ply_xyz(str(tmp_path / "py" / "predicted.ply"))
    ref = read_ply_xyz(str(tmp_path / "ref" / "reference.ply"))
    a_cpp = M.cloud_metrics(cpp, ref, [0.05], 0.02)
    a_py = M.cloud_metrics(py, ref, [0.05], 0.02)
    assert a_cpp["accuracy_median_m"] < 0.02 and a_cpp["fscore_5cm"] > 0.85
    assert abs(a_cpp["accuracy_median_m"] - a_py["accuracy_median_m"]) < 0.003
    # ORB differs between OpenCV versions, so the landmarks and the calibration fits differ, and with them how many
    # pixels pass the cross-view check. The Python harness alone gives 30.2k points under OpenCV 5.0 and 36.6k under
    # 4.10 on this room; the headset program built against 4.6 gives 29.3k. The bounds below are that spread.
    assert abs(a_cpp["fscore_5cm"] - a_py["fscore_5cm"]) < 0.05
    assert abs(len(cpp) - len(py)) / len(py) < 0.30


def _tiny_onnx(path, h, w):
    import onnx
    from onnx import helper, TensorProto
    x = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, h, w])
    y = helper.make_tensor_value_info("depth", TensorProto.FLOAT, [1, h, w])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
    g = helper.make_graph([helper.make_node("ReduceMean", ["image", "axes"], ["depth"], keepdims=0)], "tiny", [x], [y], [axes])
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)], ir_version=8), path)


def test_onnx_depth_path_runs(tmp_path):
    s = synthetic.write_tum(str(tmp_path / "seq"), n=20)
    _tiny_onnx(str(tmp_path / "tiny.onnx"), 112, 154)
    cam = QC.camera_block({"K": [[280, 0, 159.5], [0, 280, 119.5], [0, 0, 1]], "dist": [0] * 5, "size": [320, 240]})
    cfg = QC.build(os.path.join(s["dir"], "rgb.txt"), os.path.join(s["dir"], "groundtruth.txt"), cam,
                   {"source": "onnx", "model": str(tmp_path / "tiny.onnx"), "threads": 1})
    r, stats, rep = _run(cfg, str(tmp_path / "out"))
    assert r.returncode == 0, r.stderr
    assert rep["depth_model"]["input"] == [112, 154] and stats["stage_ms"]["depth"]["median"] > 0
    kfs = json.load(open(tmp_path / "out" / "keyframes.json"))
    assert all("reason" in k for k in kfs)          # a meaningless model is rejected with reasons, not fused


def test_board_anchoring_matches_python(tmp_path):
    from test_charuco import SPEC, _view, _look_at
    K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
    b1 = CH.make_board(SPEC)
    centre = np.array([0.14, 0.10, 0.0])
    (tmp_path / "bf").mkdir()
    lines, poses, scale = [], [], 2.2
    for k in range(40):
        a = np.radians(-35 + 70 * k / 39)
        T = _look_at(centre + [0.5 * np.sin(a), 0.04 * np.cos(2 * a), -0.85 + 0.15 * np.cos(a)], centre)
        cv2.imwrite(str(tmp_path / "bf" / f"{k:04d}.png"), _view(b1, T, K))
        lines.append(f"{k * 0.1:.6f} {k:04d}.png"); poses.append(T)
    (tmp_path / "bf" / "rgb.txt").write_text("\n".join(lines) + "\n")
    from scipy.spatial.transform import Rotation
    with open(tmp_path / "traj.txt", "w") as f:
        for k, T in enumerate(poses):
            q = Rotation.from_matrix(T[:3, :3]).as_quat(); p = T[:3, 3] / scale
            f.write(f"{k * 0.1:.6f} {p[0]} {p[1]} {p[2]} {q[0]} {q[1]} {q[2]} {q[3]}\n")
    ref = C.load(os.path.join(REPO, "configs", "references.yaml"))
    cam = {"K": K.tolist(), "dist": [0] * 5, "size": [1280, 720]}
    cfg = QC.build(str(tmp_path / "bf" / "rgb.txt"), str(tmp_path / "traj.txt"), cam, {"source": "oracle", "oracle": {}})
    cfg["anchor"] = {"enabled": True, "board_frames": str(tmp_path / "bf" / "rgb.txt"), "camera": cam,
                     "boards": {"ref01": dict(SPEC, mounting="unknown")}, "windows": {"ref01_start": [0.0, 4.0]},
                     "params": ref["anchor"]}
    cfg["anchor_only"] = True
    r, stats, rep = _run(cfg, str(tmp_path / "out"))
    assert r.returncode == 0, r.stderr
    assert rep["ref01"]["ok"], rep
    assert abs(rep["ref01"]["scale"] - scale) / scale < 0.02
    assert rep["ref01"]["held_out_reprojection_px"]["median"] < 1.5
    traj = np.loadtxt(tmp_path / "out" / "anchored_trajectory.txt")
    assert np.allclose(traj[:, 1:4], np.array([T[:3, 3] for T in poses]), atol=0.01)   # metric, in the board's frame


def test_anchored_reconstruction_in_metres(tmp_path, monkeypatch):
    """Tracker poses in arbitrary units, a board seen at the start, depth for every keyframe: the fused cloud must come
    out in metres in the board's frame and match the reference surface once moved back to the room's frame."""
    from test_charuco import SPEC, _view
    from qrecon.geometry import inv
    from scipy.spatial.transform import Rotation
    s = synthetic.write_tum(str(tmp_path / "seq"))
    src = TumSource(s["dir"], s)
    gt = src.poses(); times = src.times()
    T0 = gt[0]; r, d, f = T0[:3, 0], T0[:3, 1], T0[:3, 2]
    W, H = SPEC["squares_x"] * SPEC["square_length_m"], SPEC["squares_y"] * SPEC["square_length_m"]
    T_wb = np.eye(4); T_wb[:3, :3] = np.stack([r, d, f], axis=1)
    T_wb[:3, 3] = T0[:3, 3] + 0.7 * f - T_wb[:3, :3] @ np.array([W / 2, H / 2, 0])
    Kb = np.array([[1120.0, 0, 639.5], [0, 1120.0, 479.5], [0, 0, 1]])           # board frames at 4x resolution
    import test_charuco as TC
    monkeypatch.setattr(TC, "SIZE", (1280, 960))
    board = CH.make_board(SPEC)
    (tmp_path / "bf").mkdir(); lines = []
    for i in range(20):
        img = _view(board, inv(T_wb) @ gt[i], Kb)
        cv2.imwrite(str(tmp_path / "bf" / f"{i:04d}.png"), img); lines.append(f"{times[i]:.6f} {i:04d}.png")
    (tmp_path / "bf" / "rgb.txt").write_text("\n".join(lines) + "\n")
    scale = 2.5
    with open(tmp_path / "traj.txt", "w") as fh:
        for t, T in zip(times, gt):
            q = Rotation.from_matrix(T[:3, :3]).as_quat(); p = T[:3, 3] / scale
            fh.write(f"{t:.6f} {p[0]} {p[1]} {p[2]} {q[0]} {q[1]} {q[2]} {q[3]}\n")
    cfg_y = C.load(os.path.join(REPO, "configs", "qrecon.yaml"))
    ref = C.load(os.path.join(REPO, "configs", "references.yaml"))
    cam = QC.camera_block({"K": src.K.tolist(), "dist": [0] * 5, "size": [320, 240]})
    depth = {"source": "oracle", "oracle": {"depth_list": os.path.join(s["dir"], "depth.txt"), "depth_factor": s["depth_factor"],
                                             "a": cfg_y["arms"]["oracle"]["a"], "b": cfg_y["arms"]["oracle"]["b"]}}
    cfg = QC.build(os.path.join(s["dir"], "rgb.txt"), str(tmp_path / "traj.txt"), cam, depth)
    cfg["anchor"] = {"enabled": True, "board_frames": str(tmp_path / "bf" / "rgb.txt"),
                     "camera": {"K": Kb.tolist(), "dist": [0] * 5, "size": [1280, 960]},
                     "boards": {"ref01": dict(SPEC, mounting="unknown")}, "windows": {"ref01_start": [times[0], times[19]]},
                     "params": ref["anchor"]}
    r_, stats, rep = _run(cfg, str(tmp_path / "out"))
    assert r_.returncode == 0, r_.stderr
    assert rep["ref01"]["ok"] and abs(rep["ref01"]["scale"] - scale) / scale < 0.02, rep["ref01"]
    cloud_b = read_ply_xyz(str(tmp_path / "out" / "cloud.ply"))
    cloud_w = (T_wb[:3, :3] @ cloud_b.T).T + T_wb[:3, 3]
    kf = P.choose_keyframes(src, cfg_y)
    os.makedirs(tmp_path / "ref"); P.run_reference(src, kf, cfg_y, str(tmp_path / "ref"))
    m = M.cloud_metrics(cloud_w, read_ply_xyz(str(tmp_path / "ref" / "reference.ply")), [0.05], 0.02)
    assert m["accuracy_median_m"] < 0.025 and m["fscore_5cm"] > 0.8, m
