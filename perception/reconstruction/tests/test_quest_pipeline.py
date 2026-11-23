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
    assert rep["ref01"]["held_out_limit"]["unit"] == ("m" if "max_held_out_m" in ref["anchor"] else "px")
    assert "fit_reprojection_px" in rep["ref01"]
    assert rep["ref01"]["held_out_error_m"]["median"] < 0.005
    assert len(rep["ref01"]["held_out_detail"]) == rep["ref01"]["held_out_frames"]
    traj = np.loadtxt(tmp_path / "out" / "anchored_trajectory.txt")
    assert np.allclose(traj[:, 1:4], np.array([T[:3, 3] for T in poses]), atol=0.01)   # metric, in the board's frame
    pixels_only = {k: v for k, v in ref["anchor"].items() if k != "max_held_out_m"}
    cfg["anchor"]["params"] = dict(pixels_only, max_held_out_px=1e-6)                     # impossible pixel limit
    r, stats, rep = _run(cfg, str(tmp_path / "out_px"))
    assert not rep["ref01"]["ok"] and rep["ref01"]["held_out_limit"]["unit"] == "px"
    cfg["anchor"]["params"] = dict(pixels_only, max_held_out_px=1e-6, max_held_out_m=0.01)   # metres take precedence
    r, stats, rep = _run(cfg, str(tmp_path / "out_m"))
    assert r.returncode == 0, r.stderr
    assert rep["ref01"]["ok"] and rep["ref01"]["held_out_limit"] == {"value": 0.01, "unit": "m"}


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


def _two_board_walk(tmp_path, scale=2.2, still_at_board_02=False):
    """Board 01 at the world origin seen at 0-3.9 s, board 02 elsewhere seen at 4.0-7.9 s; tracker poses in arbitrary units."""
    from test_charuco import SPEC, _view, _look_at
    from scipy.spatial.transform import Rotation
    K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
    b1, b2 = CH.make_board(SPEC), CH.make_board(dict(SPEC, first_marker_id=20))
    T_wB = np.eye(4); T_wB[:3, :3] = Rotation.from_euler("y", 30, degrees=True).as_matrix(); T_wB[:3, 3] = [1.2, 0.0, 0.4]
    centre = np.array([0.14, 0.10, 0.0])
    (tmp_path / "bf").mkdir()
    lines, poses = [], []
    for k in range(80):
        a = np.radians(-35 + 70 * (k % 40) / 39) if not (still_at_board_02 and k >= 40) else np.radians(10.0)
        local = _look_at(centre + [0.5 * np.sin(a), 0.04 * np.cos(2 * a), -0.85 + 0.15 * np.cos(a)], centre)
        if k < 40:
            T, img = local, _view(b1, local, K)
        else:
            T = T_wB @ local                                   # the same kind of sweep, in front of board 02
            img = _view(b2, local, K)
        cv2.imwrite(str(tmp_path / "bf" / f"{k:04d}.png"), img)
        lines.append(f"{k * 0.1:.6f} {k:04d}.png"); poses.append(T)
    (tmp_path / "bf" / "rgb.txt").write_text("\n".join(lines) + "\n")

    def trajectory(path, keep):
        with open(path, "w") as f:
            for k in keep:
                q = Rotation.from_matrix(poses[k][:3, :3]).as_quat(); p = poses[k][:3, 3] / scale
                f.write(f"{k * 0.1:.6f} {p[0]} {p[1]} {p[2]} {q[0]} {q[1]} {q[2]} {q[3]}\n")
    ref = C.load(os.path.join(REPO, "configs", "references.yaml"))
    cam = {"K": K.tolist(), "dist": [0] * 5, "size": [1280, 720]}

    def config(traj, mode, params=None):
        cfg = QC.build(str(tmp_path / "bf" / "rgb.txt"), str(traj), cam, {"source": "oracle", "oracle": {}})
        cfg["anchor"] = {"enabled": True, "mode": mode, "board_frames": str(tmp_path / "bf" / "rgb.txt"), "camera": cam,
                         "boards": {"ref01": dict(SPEC, mounting="unknown"), "ref02": dict(SPEC, first_marker_id=20, mounting="unknown")},
                         "windows": {"ref01_start": [0.0, 3.95], "ref02": [[4.0, 7.95]]}, "params": params or ref["anchor"]}
        cfg["anchor_only"] = True
        return cfg
    return T_wB, poses, trajectory, config, ref


def test_best_effort_links_board_02_and_falls_back_to_it(tmp_path):
    T_wB, poses, trajectory, config, ref = _two_board_walk(tmp_path)
    trajectory(tmp_path / "all.txt", range(80))
    r, stats, rep = _run(config(tmp_path / "all.txt", "best_effort"), str(tmp_path / "link"))
    assert r.returncode == 0, r.stderr
    assert rep["frame"] == "ref01" and rep["anchored_by"]["window"] == "ref01_start"
    L = rep["ref02"]["sim3_board_in_frame"]
    assert abs(L["s"] - 1.0) < 0.02 and np.allclose(np.array(L["R"]), T_wB[:3, :3], atol=0.02)
    assert np.allclose(L["t"], T_wB[:3, 3], atol=0.02)                         # board 02's pose in board 01's frame
    trajectory(tmp_path / "late.txt", range(40, 80))                         # a stretch that never saw board 01
    r, stats, rep = _run(config(tmp_path / "late.txt", "best_effort"), str(tmp_path / "fallback"))
    assert r.returncode == 0, r.stderr
    assert rep["frame"] == "ref02" and stats["anchor_frame"] == "ref02"
    traj = np.loadtxt(tmp_path / "fallback" / "anchored_trajectory.txt")
    want = np.array([(np.linalg.inv(T_wB) @ poses[k])[:3, 3] for k in range(40, 80)])
    assert np.allclose(traj[:, 1:4], want, atol=0.01)                        # metric, in board 02's frame
    r, stats, rep = _run(config(tmp_path / "late.txt", "strict"), str(tmp_path / "strict"))
    assert r.returncode != 0 and "anchor refused" in rep["error"]           # strict still refuses without board 01


def test_best_effort_records_failed_checks_instead_of_refusing(tmp_path):
    T_wB, poses, trajectory, config, ref = _two_board_walk(tmp_path)
    trajectory(tmp_path / "all.txt", range(80))
    tight = dict(ref["anchor"], max_corner_fit_rms_m=1e-9)
    r, stats, rep = _run(config(tmp_path / "all.txt", "strict", tight), str(tmp_path / "strict"))
    assert r.returncode != 0 and "triangulated corners do not fit" in rep["error"]
    r, stats, rep = _run(config(tmp_path / "all.txt", "best_effort", tight), str(tmp_path / "best"))
    assert r.returncode == 0, r.stderr
    F = rep["frame"]                                                    # nothing passes: the tightest corner fit anchors
    fits = [x["corner_fit_rms_m"] for x in rep["anchor_attempts"] if "corner_fit_rms_m" in x]
    assert F in ("ref01", "ref02") and not rep[F]["ok"] and rep[F]["corner_fit_rms_m"] == min(fits)
    assert any("although triangulated corners do not fit" in w for w in rep["warnings"])
    assert "warning" in r.stderr


def test_best_effort_links_a_board_seen_only_from_a_still_camera(tmp_path):
    T_wB, poses, trajectory, config, ref = _two_board_walk(tmp_path, still_at_board_02=True)
    trajectory(tmp_path / "all.txt", range(80))
    r, stats, rep = _run(config(tmp_path / "all.txt", "best_effort"), str(tmp_path / "still"))
    assert r.returncode == 0, r.stderr
    assert rep["frame"] == "ref01" and rep["ref02"]["method"].startswith("single views")
    L = rep["ref02"]["sim3_board_in_frame"]
    assert L["s"] == 1.0 and np.allclose(np.array(L["R"]), T_wB[:3, :3], atol=0.02) and np.allclose(L["t"], T_wB[:3, 3], atol=0.02)
    assert any("single views" in w for w in rep["warnings"])


def test_board_windows_found_automatically_anchor_and_link_any_board(tmp_path):
    T_wB, poses, trajectory, config, ref = _two_board_walk(tmp_path)
    trajectory(tmp_path / "late.txt", range(40, 80))                    # this map only saw board 02
    cfg = config(tmp_path / "late.txt", "best_effort")
    del cfg["anchor"]["windows"]
    cfg["anchor"]["board_windows"] = {"ref01": [[0.0, 3.95]], "ref02": [[4.0, 7.95]]}
    cfg["anchor"]["board_order"] = ["ref01", "ref02"]
    r, stats, rep = _run(cfg, str(tmp_path / "auto"))
    assert r.returncode == 0, r.stderr
    assert rep["frame"] == "ref02" and rep["anchored_by"]["window"].startswith("ref02 4.0-")
    assert [x["board"] for x in rep["anchor_attempts"]] == ["ref01", "ref02"]
    assert "ref01" in rep["links"] and rep["links"]["ref01"]["ok"] is False


def test_best_effort_takes_a_boards_windows_together_when_each_is_too_short(tmp_path):
    T_wB, poses, trajectory, config, ref = _two_board_walk(tmp_path)
    trajectory(tmp_path / "all.txt", range(80))
    cfg = config(tmp_path / "all.txt", "best_effort")
    del cfg["anchor"]["windows"]
    cfg["anchor"]["board_windows"] = {"ref01": [[0.0, 0.45], [0.5, 0.95]], "ref02": [[4.0, 7.95]]}   # 5 views each: too few alone
    cfg["anchor"]["board_order"] = ["ref01", "ref02"]
    r, stats, rep = _run(cfg, str(tmp_path / "joint"))
    assert r.returncode == 0, r.stderr
    tried = [(x["board"], x["window"], x["ok"]) for x in rep["anchor_attempts"]]
    assert ("ref01", "ref01 all windows", True) in tried and not any(ok for b, w, ok in tried[:2])
    assert rep["frame"] == "ref01" and rep["anchored_by"]["window"] == "ref01 all windows" and "closure_ref01" not in rep
    assert "ref02" in rep["links"] and np.allclose(rep["links"]["ref02"]["sim3_board_in_frame"]["t"], T_wB[:3, 3], atol=0.03)


def test_a_fixed_similarity_puts_the_cloud_in_that_frame_without_a_board(tmp_path):
    T_wB, poses, trajectory, config, ref = _two_board_walk(tmp_path)
    trajectory(tmp_path / "all.txt", range(80))
    cfg = config(tmp_path / "all.txt", "best_effort")
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler("zyx", [20, -10, 5], degrees=True).as_matrix(); t = [0.4, -1.2, 2.0]; s = 1.7
    cfg["anchor"] = {"enabled": True, "fixed": {"frame": "ref02", "sim3": {"s": s, "R": R.tolist(), "t": t}, "by": "shared views"}}
    r, stats, rep = _run(cfg, str(tmp_path / "fixed"))
    assert r.returncode == 0, r.stderr
    assert rep["frame"] == "ref02" and rep["anchored_by"]["window"] == "shared views" and stats["anchor_frame"] == "ref02"
    raw = np.loadtxt(tmp_path / "all.txt"); out = np.loadtxt(tmp_path / "fixed" / "anchored_trajectory.txt")
    assert np.allclose(out[:, 1:4], s * raw[:, 1:4] @ R.T + np.array(t), atol=1e-5)


def test_the_trackers_points_calibrate_keyframes_that_have_nothing_to_triangulate(tmp_path):
    s = synthetic.write_tum(str(tmp_path / "seq"))
    cfg_y = C.load(os.path.join(REPO, "configs", "qrecon.yaml"))
    oracle = dict(cfg_y["arms"]["oracle"], noise=0.0)
    src = TumSource(s["dir"], s)
    cam = QC.camera_block({"K": src.K.tolist(), "dist": [0] * 5, "size": [s["camera"]["width"], s["camera"]["height"]]})
    depth = {"source": "oracle", "oracle": {"depth_list": os.path.join(s["dir"], "depth.txt"), "depth_factor": s["depth_factor"],
                                         "a": oracle["a"], "b": oracle["b"]}}
    cfg = QC.build(os.path.join(s["dir"], "rgb.txt"), os.path.join(s["dir"], "groundtruth.txt"), cam, depth)
    cfg["landmarks"]["neighbour_offsets"] = []                   # as in a turn on the spot: nothing to triangulate
    r, stats, _ = _run(cfg, str(tmp_path / "none"))
    assert r.returncode == 0, r.stderr
    assert stats["valid_keyframes"] == 0
    rng = np.random.default_rng(0)
    with open(tmp_path / "map_points.txt", "w") as f:            # points the tracker would hold, seen from every frame
        f.write("# x y z t...\n")
        for i in range(0, len(src), 3):
            fr = src.frame(i)
            v, u = np.nonzero(fr.depth > 0)
            pick = rng.choice(len(u), size=min(150, len(u)), replace=False)
            d = fr.depth[v[pick], u[pick]]
            Xc = np.c_[(u[pick] - src.K[0, 2]) * d / src.K[0, 0], (v[pick] - src.K[1, 2]) * d / src.K[1, 1], d]
            Xw = Xc @ fr.T_wc[:3, :3].T + fr.T_wc[:3, 3]
            for X in Xw:
                f.write(f"{X[0]:.5f} {X[1]:.5f} {X[2]:.5f} {fr.t:.4f} {fr.t + 0.001:.4f}\n")
    cfg["map_points"] = {"file": str(tmp_path / "map_points.txt"), "window_s": 0.5, "max_per_keyframe": 2000}
    r, stats, _ = _run(cfg, str(tmp_path / "with"))
    assert r.returncode == 0, r.stderr
    kfs = json.load(open(tmp_path / "with" / "keyframes.json"))
    assert stats["valid_keyframes"] >= 0.8 * stats["keyframes"] and stats["map_points_loaded"] > 0
    assert all(k["triangulated"] == 0 and k["map_points"] > 0 for k in kfs)
