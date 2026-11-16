"""Stage tests on synthetic data. Run:  python -m pytest -q tests"""
import os, sys, json, subprocess
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qrecon import geometry as G, calibrate as CAL, anchor as A, metrics as M, config as C, synthetic, pipeline as P, depth_models as DM
from qrecon.sources import TumSource, associate
from qrecon.fusion import VoxelMap, read_ply_xyz, write_ply

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = C.load(os.path.join(REPO, "configs", "qrecon.yaml"))


def test_pose_and_projection_round_trip():
    from scipy.spatial.transform import Rotation
    q = Rotation.from_euler("xyz", [20, -35, 50], degrees=True).as_quat()
    T = G.quat_to_T(0.3, -1.2, 2.0, *q)
    assert np.allclose(T[:3, :3], Rotation.from_quat(q).as_matrix(), atol=1e-9)
    K = G.K_matrix(500, 500, 320, 240)
    depth = np.full((480, 640), 2.0, dtype=np.float32)
    pts, uv = G.backproject(depth, K, T, stride=40)
    u, v, z = G.project(pts, K, T)
    assert np.allclose(u, uv[:, 0], atol=1e-6) and np.allclose(v, uv[:, 1], atol=1e-6) and np.allclose(z, 2.0, atol=1e-6)
    assert np.allclose(G.inv(T) @ T, np.eye(4), atol=1e-9)


def test_associate_matches_nearest_timestamps():
    a = [(1.00, ["a0"]), (1.10, ["a1"]), (1.20, ["a2"])]
    b = [(1.004, ["b0"]), (1.19, ["b1"]), (5.0, ["b2"])]
    assert associate(a, b, 0.02) == [(0, 0), (2, 1)]


def _landmarks(rng, n, shape, a, b, noise=0.0, outliers=0):
    h, w = shape
    px = np.stack([rng.uniform(0, w - 1, n), rng.uniform(0, h - 1, n)], axis=1)
    z_img = 1.0 + 3.0 * (np.mgrid[0:h, 0:w][1] / w)               # depth 1..4 m across the image
    rel = ((1.0 / z_img) - b) / a
    z = z_img[np.round(px[:, 1]).astype(int), np.round(px[:, 0]).astype(int)] * (1 + rng.normal(0, noise, n))
    z[:outliers] *= rng.uniform(1.6, 2.5, outliers)
    return rel.astype(np.float32), px, z, z_img


def test_inverse_depth_fit_recovers_metric_depth_despite_outliers():
    rng = np.random.default_rng(3)
    rel, px, z, z_img = _landmarks(rng, 200, (240, 320), a=0.37, b=0.05, noise=0.01, outliers=40)
    c = CAL.fit_inverse_depth(rel, px, z, CFG["calibration"], seed=1)
    assert c.ok, c.reason
    depth = CAL.apply_inverse_depth(rel, c, 10.0)
    assert np.median(np.abs(depth - z_img) / z_img) < 0.01


def test_fit_is_refused_without_support():
    rng = np.random.default_rng(4)
    rel, px, z, _ = _landmarks(rng, 200, (240, 320), 0.37, 0.05)
    few = CAL.fit_inverse_depth(rel, px[:10], z[:10], CFG["calibration"])
    assert not few.ok and "few" in few.reason
    corner = px[:, 0] < 40                                          # all landmarks in one image strip, one depth
    clustered = CAL.fit_inverse_depth(rel, px[corner], z[corner], CFG["calibration"]) if corner.sum() >= 30 else None
    assert clustered is None or not clustered.ok


def test_single_scale_is_not_enough_for_shifted_inverse_depth():
    """Why the fit has a shift: multiplying a disparity-like output by one number leaves a depth-dependent error."""
    rng = np.random.default_rng(5)
    rel, px, z, z_img = _landmarks(rng, 200, (240, 320), a=0.37, b=0.05)
    naive = 1.0 / np.maximum(rel, 1e-6)
    s = np.median(z_img / naive)
    assert np.percentile(np.abs(s * naive - z_img) / z_img, 90) > 0.05


def test_tag_anchor_recovers_scale_and_frame():
    rng = np.random.default_rng(6)
    K = G.K_matrix(600, 600, 320, 240)
    size = 0.16
    corners_tag = A.tag_model_corners(size)
    true_scale = 2.7                                                # tracker units are 1/2.7 m
    obs, poses = {}, {}
    for i, x in enumerate([-0.4, -0.2, 0.0, 0.25, 0.45]):
        T_metric = np.eye(4)
        T_metric[:3, :3] = np.diag([1.0, -1.0, -1.0])               # camera looks at the tag from +z
        T_metric[:3, 3] = [x, 0.05 * i, 1.2]
        u, v, z = G.project(corners_tag, K, T_metric)
        assert np.all(z > 0)
        obs[i] = np.stack([u, v], axis=1) + rng.normal(0, 0.2, (4, 2))
        T_slam = T_metric.copy(); T_slam[:3, 3] /= true_scale       # same path in arbitrary units
        poses[i] = T_slam
    out = A.anchor_from_tag(obs, poses, K, size, CFG["anchor"])
    assert out["ok"], out
    assert abs(out["scale"] - true_scale) / true_scale < 0.02
    anchored = A.to_anchor_frame([poses[i] for i in range(5)], out["scale"], out["R"], out["t"])
    assert np.allclose(anchored[2][:3, 3], [0.0, 0.10, 1.2], atol=0.02)


def test_tag_anchor_refuses_views_without_translation():
    K = G.K_matrix(600, 600, 320, 240)
    T = np.eye(4); T[:3, :3] = np.diag([1.0, -1.0, -1.0]); T[:3, 3] = [0, 0, 1.2]
    u, v, _ = G.project(A.tag_model_corners(0.16), K, T)
    obs = {i: np.stack([u, v], axis=1) for i in range(4)}
    out = A.anchor_from_tag(obs, {i: T for i in range(4)}, K, 0.16, CFG["anchor"])
    assert not out["ok"]


def test_cloud_metrics_and_bounded_map(tmp_path):
    rng = np.random.default_rng(7)
    ref = rng.uniform(0, 1, (20000, 3)) * [2, 2, 0]                 # a 2 m x 2 m plane
    pred = ref + [0, 0, 0.03]                                       # offset by 3 cm
    m = M.cloud_metrics(pred, ref, [0.05, 0.10], 0.02)
    assert 0.02 < m["accuracy_median_m"] < 0.04 and m["precision_5cm"] > 0.95 and m["fscore_10cm"] > 0.99
    vm = VoxelMap(0.05, max_voxels=100, min_observations=1)
    vm.integrate(ref)
    assert vm.stats()["voxels"] == 100 and vm.stats()["refused_new_voxels"] > 0
    p = str(tmp_path / "c.ply"); write_ply(p, ref, np.full((len(ref), 3), 128.0))
    assert np.allclose(read_ply_xyz(p), ref, atol=1e-5)


@pytest.fixture(scope="module")
def room(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("room"))
    s = synthetic.write_tum(os.path.join(root, "seq"))
    return root, s, TumSource(s["dir"], s)


def test_whole_pipeline_on_synthetic_room(room):
    root, s, src = room
    kf = P.choose_keyframes(src, CFG)
    os.makedirs(os.path.join(root, "ref"), exist_ok=True); os.makedirs(os.path.join(root, "run"), exist_ok=True)
    P.run_reference(src, kf, CFG, os.path.join(root, "ref"))
    poses = [src.frame(i).T_wc for i in kf]
    summary, per_kf = P.run_dense(src, kf, poses, DM.make("oracle", CFG["arms"]), CFG, os.path.join(root, "run"), log=lambda *_: None)
    cloud = M.cloud_metrics(read_ply_xyz(os.path.join(root, "run", "predicted.ply")), read_ply_xyz(os.path.join(root, "ref", "reference.ply")),
                            CFG["metrics"]["taus_m"], CFG["metrics"]["cloud_voxel_m"])
    assert summary["scale_error"] < 0.01 and cloud["accuracy_median_m"] < 0.02 and cloud["fscore_5cm"] > 0.85
    assert P.judge(summary, cloud, CFG)["passes_screening"]


def test_wrong_scale_poses_are_caught_not_hidden(room):
    """Poses 10% too large must show up as a scale error; nothing in the evaluation fits it away."""
    root, s, src = room
    kf = P.choose_keyframes(src, CFG)
    poses = []
    for i in kf:
        T = src.frame(i).T_wc.copy(); T[:3, 3] *= 1.10; poses.append(T)
    out = os.path.join(root, "bad"); os.makedirs(out, exist_ok=True)
    summary, _ = P.run_dense(src, kf, poses, DM.make("oracle", CFG["arms"]), CFG, out, log=lambda *_: None)
    assert summary["scale_error"] is None or summary["scale_error"] > 0.05
