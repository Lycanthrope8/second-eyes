import json, os, sys
import numpy as np
import cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
sys.path.insert(0, os.path.dirname(__file__))
import place_maps as PM  # noqa: E402
import test_mapalign as T  # noqa: E402


def _tum(path, times, poses):
    from scipy.spatial.transform import Rotation
    with open(path, "w") as f:
        for t, P in zip(times, poses):
            q = Rotation.from_matrix(P[:3, :3]).as_quat()
            f.write(f"{t:.6f} {P[0, 3]} {P[1, 3]} {P[2, 3]} {q[0]} {q[1]} {q[2]} {q[3]}\n")


def _setup(tmp_path):
    placed = [T._pose(-1.5 + 0.1 * k, 0.0, -10 + 1.75 * k) for k in range(21)]
    world_u = [T._pose(-1.0 + 0.1 * k, 0.6, 1.75 * k) for k in range(21)]
    s, R, t = 2.0, cv2.Rodrigues(np.array([0.1, 0.5, -0.2]))[0], np.array([1.0, 2.0, 3.0])
    def to_tracker(P):
        o = np.eye(4); o[:3, :3] = R.T @ P[:3, :3]; o[:3, 3] = R.T @ (P[:3, 3] - t) / s
        return o
    fr = tmp_path / "data" / "syn_frames"; (fr / "rgb").mkdir(parents=True)
    tp, tu = [0.3 * k for k in range(21)], [10 + 0.3 * k for k in range(21)]
    lines = ["# syn every=1 scale=1.0 size=(640, 360) decoded=in-order"]
    for tt, P in list(zip(tp, placed)) + list(zip(tu, world_u)):
        name = f"rgb/{int(round(tt * 100)):06d}.png"
        cv2.imwrite(str(fr / name), T._render(P)); lines.append(f"{tt:.6f} {name}")
    (fr / "rgb.txt").write_text("\n".join(lines) + "\n")
    trk = tmp_path / "runs" / "trk"; (trk / "tracking" / "maps" / "map_01").mkdir(parents=True)
    (trk / "manifest.json").write_text(json.dumps({"frames": str(fr), "realtime_fps": 3.333}))
    (trk / "tracking" / "maps.json").write_text(json.dumps([{"map": 0, "status": "saved", "dir": "maps/map_00"},
                                                            {"map": 1, "status": "saved", "dir": "maps/map_01"}]))
    _tum(trk / "tracking" / "maps" / "map_01" / "frame_trajectory.txt", tu, [to_tracker(P) for P in world_u])
    rec = tmp_path / "runs" / "rec0" / "recon"; rec.mkdir(parents=True)
    _tum(rec / "anchored_trajectory.txt", tp, placed)
    (rec / "anchor.json").write_text(json.dumps({"frame": "ref02", "ref02": {"ok": True}}))
    (tmp_path / "calib.json").write_text(json.dumps({"K": T.K.tolist(), "dist": [0] * 5, "size": [640, 360]}))
    summ = tmp_path / "runs" / "syn_recording.json"
    summ.write_text(json.dumps({"name": "syn", "calib": "calib.json", "tracking_run": "trk", "recon_runs": ["rec0"],
                                "maps": [{"map": 0, "first_t": 0.0, "last_t": 6.0, "recon_run": "rec0", "frame": "ref02"},
                                         {"map": 1, "first_t": 10.0, "last_t": 16.0, "skipped": "no board seen"}]}))
    return summ, (s, R, t)


def test_place_maps_places_and_records_a_board_less_map(tmp_path, monkeypatch):
    monkeypatch.setattr(PM, "HERE", tmp_path)
    summ, (s, R, t) = _setup(tmp_path)
    sys.argv = ["place_maps.py", "--recordings", str(summ), "--dry-run"]
    PM.main()
    rep = json.loads(sorted((tmp_path / "runs").glob("placement_*.json"))[-1].read_text())
    e = rep["rounds"][0][0]
    assert e["placed"] and e["frame"] == "ref02" and abs(e["sim3"]["s"] / s - 1) < 0.05
    assert "skipped" in json.loads(summ.read_text())["maps"][1]                       # a dry run changes no summary

    made = []

    def fake_reconstruct(rec, m, fixed_path, args, trajectory=None):
        fixed = json.loads(open(fixed_path).read())
        assert fixed["frame"] == "ref02" and "shared views" in fixed["by"] and trajectory is None   # plausible speed: whole map
        name = f"rec{len(made) + 1}"
        made.append(name)
        d = tmp_path / "runs" / name / "recon"; d.mkdir(parents=True)
        (d / "stats.json").write_text(json.dumps({"keyframes": 20, "valid_keyframe_share": 0.8}))
        (d / "anchor.json").write_text(json.dumps({"frame": "ref02", "fixed": fixed}))
        return name
    monkeypatch.setattr(PM, "reconstruct", fake_reconstruct)
    sys.argv = ["place_maps.py", "--recordings", str(summ)]
    PM.main()
    m1 = json.loads(summ.read_text())["maps"][1]
    assert m1["recon_run"] == "rec1" and m1["frame"] == "ref02" and "skipped" not in m1 and m1["placed_by"]["inliers"] >= 4
    assert m1["placed_by"]["kept"] == "full" and m1["placed_by"]["speed_p95_mps"] < 1.0
    assert "rec1" in json.loads(summ.read_text())["recon_runs"] and (tmp_path / "runs" / "syn_recording.json.bak").is_file()
    sys.argv = ["place_maps.py", "--recordings", str(summ)]                     # nothing left to place: no change
    PM.main()
    assert made == ["rec1"]
    sys.argv = ["place_maps.py", "--recordings", str(summ), "--redo"]           # placed again by the current rules
    PM.main()
    S = json.loads(summ.read_text())
    assert S["maps"][1]["recon_run"] == "rec2" and S["maps"][1]["replaced_run"] == "rec1"
    assert "rec2" in S["recon_runs"] and "rec1" not in S["recon_runs"]
