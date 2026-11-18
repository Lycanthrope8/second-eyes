"""merge_clouds: clouds in board 02's frame are moved into board 01's with the link a run reports."""
import json, os, subprocess, sys
import numpy as np
from scipy.spatial.transform import Rotation
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import merge_clouds as MC


def _run(root, name, P, anchor):
    rec = root / "runs" / name / "recon"; rec.mkdir(parents=True)
    MC.write_ply(rec / "cloud.ply", P, np.full((len(P), 3), 200, np.uint8))
    (rec / "anchor.json").write_text(json.dumps(anchor))


def test_merge_places_board_02_clouds_with_the_link(tmp_path, monkeypatch):
    monkeypatch.setattr(MC, "HERE", tmp_path)
    R = Rotation.from_euler("y", 30, degrees=True).as_matrix(); t = np.array([1.2, 0.0, 0.4]); s = 1.1
    world = np.random.default_rng(0).uniform(-1, 1, (500, 3))                  # points in board 01's frame
    in_b2 = (world - t) @ R / s                                                  # the same points in board 02's frame
    link = {"s": s, "R": R.tolist(), "t": t.tolist()}
    _run(tmp_path, "a", world[:250], {"frame": "ref01", "ref01": {"ok": True}})
    _run(tmp_path, "b", in_b2[250:], {"frame": "ref02", "warnings": ["anchored by ref02 although ..."]})
    _run(tmp_path, "e", world[:10], {"frame": "ref01", "ref02": {"corner_fit_rms_m": 0.004, "sim3_board_in_frame": link}})
    _run(tmp_path, "t", world[:5], {"frame": "tracker"})
    sys.argv = ["merge_clouds.py", "--runs", "a", "b", "e", "t", "--out", "both"]
    MC.main()
    P, _ = MC.read_ply(tmp_path / "runs" / "both.ply")
    assert len(P) == 250 + 250 + 10
    assert np.allclose(np.sort(P[250:500], axis=0), np.sort(world[250:], axis=0), atol=1e-4)
    rep = json.load(open(tmp_path / "runs" / "both.json"))
    assert rep["link"]["from_run"] == "e" and [e["placed"] for e in rep["runs"]] == [True, True, True, False]


def test_merge_inverts_a_link_reported_by_a_board_02_run(tmp_path, monkeypatch):
    monkeypatch.setattr(MC, "HERE", tmp_path)
    R = Rotation.from_euler("x", -20, degrees=True).as_matrix(); t = np.array([-0.5, 0.2, 2.0]); s = 0.9
    world = np.random.default_rng(1).uniform(-1, 1, (100, 3))
    in_b2 = (world - t) @ R / s
    Ri = R.T; si = 1 / s; ti = -Ri @ t / s                                      # board 01's pose in board 02's frame
    _run(tmp_path, "b", in_b2, {"frame": "ref02", "ref01_link": {"corner_fit_rms_m": 0.006,
                                                                 "sim3_board_in_frame": {"s": si, "R": Ri.tolist(), "t": ti.tolist()}}})
    sys.argv = ["merge_clouds.py", "--runs", "b", "--out", "only_b"]
    MC.main()
    P, _ = MC.read_ply(tmp_path / "runs" / "only_b.ply")
    assert np.allclose(P, world, atol=1e-4)


def test_merge_takes_the_link_from_an_anchor_only_run(tmp_path, monkeypatch):
    monkeypatch.setattr(MC, "HERE", tmp_path)
    R = Rotation.from_euler("z", 15, degrees=True).as_matrix(); t = np.array([0.3, -1.0, 2.5])
    world = np.random.default_rng(2).uniform(-1, 1, (200, 3))
    _run(tmp_path, "a", world[:100], {"frame": "ref01"})
    _run(tmp_path, "b", (world[100:] - t) @ R, {"frame": "ref02"})
    rec = tmp_path / "runs" / "link" / "recon"; rec.mkdir(parents=True)          # anchor.json only, no cloud
    Ri, ti = R.T, -R.T @ t
    (rec / "anchor.json").write_text(json.dumps({"frame": "ref02", "ref01_link": {"corner_fit_rms_m": 0.02,
                                     "sim3_board_in_frame": {"s": 1.0, "R": Ri.tolist(), "t": ti.tolist()}}}))
    sys.argv = ["merge_clouds.py", "--runs", "a", "b", "link", "--out", "ab"]
    MC.main()
    P, _ = MC.read_ply(tmp_path / "runs" / "ab.ply")
    assert len(P) == 200 and np.allclose(np.sort(P[100:], axis=0), np.sort(world[100:], axis=0), atol=1e-4)
    rep = json.load(open(tmp_path / "runs" / "ab.json"))
    assert rep["link"]["from_run"] == "link" and rep["link_only_runs"] == ["link"]
