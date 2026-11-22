import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
sys.path.insert(0, os.path.dirname(__file__))
import one_map as OM  # noqa: E402
import place_maps as PM  # noqa: E402
import merge_clouds as MC  # noqa: E402
import process_recording as PR  # noqa: E402
from qrecon import mapalign as MA  # noqa: E402
import test_place_maps as TP  # noqa: E402
import test_mapalign as T  # noqa: E402


def test_a_video_s_pieces_become_one_trajectory_and_one_reconstruction(tmp_path, monkeypatch):
    monkeypatch.setattr(OM, "HERE", tmp_path); monkeypatch.setattr(PM, "HERE", tmp_path)
    summ, _ = TP._setup(tmp_path)
    info = OM.main(["--recording", str(summ), "--dry-run"])
    assert [p["how"] for p in info["pieces"]] == ["board", "shared views"] and info["frame"] == "ref02" and not info["left_out"]
    t, Tj = MA.read_tum(tmp_path / "runs" / "one_map" / "syn_trajectory.txt")
    assert len(t) == 42 and np.all(np.diff(t) > 0)
    truth = np.array([T._pose(-1.0 + 0.1 * k, 0.6, 1.75 * k)[:3, 3] for k in range(21)])
    assert np.median(np.linalg.norm(Tj[21:, :3, 3] - truth, axis=1)) < 0.10          # the joined piece lands where it was
    def fake(rec, traj, fixed, args):
        assert json.loads(open(fixed).read())["frame"] == "ref02"
        d = tmp_path / "runs" / "one1" / "recon"; d.mkdir(parents=True)
        (d / "stats.json").write_text(json.dumps({"keyframes": 30, "valid_keyframe_share": 0.9}))
        return "one1"
    monkeypatch.setattr(OM, "reconstruct", fake)
    OM.main(["--recording", str(summ)])
    S = json.loads(summ.read_text())
    assert S["recon_runs"] == ["one1"] and S["one_map"]["recon_run"] == "one1" and len(S["one_map"]["pieces"]) == 2
    assert (tmp_path / "runs" / "syn_recording.json.bak").is_file()


def test_pieces_in_the_other_board_s_frame_are_carried_through_the_link():
    L = {"frame": "ref02", "ref02": {"ok": True}, "links": {"ref01": {"position_spread_m": 0.001, "position_m": 5.0,
         "sim3_board_in_frame": {"s": 1.0, "R": np.eye(3).tolist(), "t": [5.0, 0.0, 0.0]}}}}
    paths = MC.best_paths(MC.run_links("L", L), "ref01")
    tf = {"ref01": (1.0, np.eye(3), np.zeros(3)), "ref02": paths["ref02"][1]}
    P = np.tile(np.eye(4), (2, 1, 1))
    a = {"frame": "ref01", "times": np.array([0.0, 1.0]), "poses": P.copy(), "map": 0}
    b = {"frame": "ref02", "times": np.array([2.0, 3.0]), "poses": P.copy(), "map": 1}
    b["poses"][:, :3, 3] = [5.0, 0.0, 0.0]
    frame, t, Tj, used, dropped = OM.combine([a, b], tf, "ref01")
    assert frame == "ref01" and not dropped and list(t) == [0.0, 1.0, 2.0, 3.0]
    assert np.allclose(Tj[2:, :3, 3], 0.0, atol=1e-9)                                  # board 02's origin maps to board 01's


def test_an_anchor_only_run_is_kept_whole_or_trimmed_by_its_speed(tmp_path, monkeypatch):
    monkeypatch.setattr(PR, "HERE", tmp_path)
    d = tmp_path / "runs" / "a1" / "recon"; d.mkdir(parents=True)
    (d / "anchor.json").write_text(json.dumps({"frame": "ref01", "ref01": {"ok": True}}))
    t = np.arange(240.0, 302.0, 0.5)
    P = np.tile(np.eye(4), (len(t), 1, 1)); P[:, 0, 3] = np.where(t < 285, 3.0 * (t - 240), 135.0 + 0.5 * (t - 285))
    TP._tum(d / "anchored_trajectory.txt", t, P)
    f = PR.anchor_entry("a1", 244.9, 301.4, {"ref01": [[297.5, 301.5]]}, 2.0, 10.0)
    assert f["anchor_ok"] and f["kept"] == "trimmed" and f["span_s"] == [287.5, 301.4]
    P[:, 0, 3] = 0.5 * (t - 240); TP._tum(d / "anchored_trajectory.txt", t, P)
    assert PR.anchor_entry("a1", 244.9, 301.4, {"ref01": [[297.5, 301.5]]}, 2.0, 10.0)["kept"] == "full"


def test_capped_keyframes_are_spread_over_the_whole_path(tmp_path):
    import reconstruct_on_quest as RQ
    t = np.arange(0.0, 300.0, 1 / 30)
    P = np.tile(np.eye(4), (len(t), 1, 1)); P[:, 0, 3] = 0.4 * t                    # walking 0.4 m/s for 5 minutes
    TP._tum(tmp_path / "path.txt", t, P)
    n_all, n_kept = RQ.spread_keyframes(tmp_path / "path.txt", tmp_path / "kf.txt", 200)
    tk, _ = MA.read_tum(tmp_path / "kf.txt")
    assert n_all > 400 and n_kept == 200 and tk[0] == 0.0 and tk[-1] > 299.0       # the end of the path is kept


def test_a_piece_nothing_else_can_place_joins_by_its_own_failed_anchor_flagged(tmp_path, monkeypatch):
    monkeypatch.setattr(OM, "HERE", tmp_path); monkeypatch.setattr(PM, "HERE", tmp_path)
    summ, _ = TP._setup(tmp_path)
    (tmp_path / "runs" / "rec0" / "recon" / "anchor.json").write_text(json.dumps({"frame": "ref02", "ref02": {"ok": False}}))
    info = OM.main(["--recording", str(summ), "--dry-run"])
    assert [(p["map"], p["how"]) for p in info["pieces"]] == [(0, "board, failed its checks")]
    assert [e["map"] for e in info["left_out"]] == [1]                    # a failed anchor never places another piece
