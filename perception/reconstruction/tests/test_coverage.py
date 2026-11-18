import json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import coverage as CV  # noqa: E402


def _rows(states, fps=15.0, maps=None):
    return [{"index": i, "t": i / fps, "state": s, "map": (maps[i] if maps else (0 if s == "Tracking" else -1))} for i, s in enumerate(states)]


def test_whole_and_from_init_shares_count_every_later_gap():
    st = ["Initializing"] * 15 + ["Tracking"] * 30 + ["Lost"] * 45 + ["Tracking"] * 15     # 0-1 init, 1-3, 3-6 lost, 6-7
    rows = _rows(st)
    saved = {CV._key(r["t"]) for r in rows if r["state"] == "Tracking"}
    c = CV.compute(rows, saved, rgb_count=len(rows))
    assert abs(c["processed"]["span_s"] - 7.0) < 1e-6 and c["processed"]["whole_frame_list"]
    assert abs(c["tracked_share_whole"] - 3 / 7) < 1e-3
    assert c["initialization"]["t_s"] == 1.0 and abs(c["tracked_share_from_init"] - 0.5) < 1e-3
    g = c["longest_gap_after_init"]
    assert g["start_s"] == 3.0 and g["end_s"] == 6.0 and g["recovered"] and c["gaps_after_init"] == 1


def test_unsaved_tracking_is_not_counted_and_is_reported():
    st = ["Tracking"] * 30 + ["Initializing"] * 15 + ["Tracking"] * 30
    rows = _rows(st)
    saved = {CV._key(r["t"]) for r in rows[45:]}                     # the first map was discarded by stella
    c = CV.compute(rows, saved)
    assert c["initialization"]["t_s"] == 3.0 and abs(c["tracked_but_discarded_s"] - 2.0) < 1e-6
    assert abs(c["tracked_share_whole"] - 2.0 / 5.0) < 1e-3 and c["tracked_share_from_init"] == 1.0


def test_source_timestamps_replace_nominal_ones():
    st = ["Tracking"] * 10 + ["Lost"] * 10
    rows = _rows(st, fps=10.0)
    saved = {CV._key(r["t"]) for r in rows[:10]}
    src = {CV._key(r["t"]): (r["t"] if r["index"] < 10 else 1.0 + 2 * (r["t"] - 1.0)) for r in rows}   # second half slower; the last frame counts for the median step (0.1 s)
    c = CV.compute(rows, saved, source=src)
    assert c["timestamps"] == "source" and abs(c["processed"]["span_s"] - 2.9) < 1e-6
    assert abs(c["tracked_share_whole"] - 1.0 / 2.9) < 1e-3
    assert not c["longest_gap_after_init"]["recovered"]


def test_for_run_reads_maps_and_writes_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(CV, "HERE", tmp_path)
    d = tmp_path / "runs" / "r1" / "tracking"; (d / "maps" / "map_00").mkdir(parents=True); (d / "maps" / "map_01").mkdir()
    st = ["Tracking"] * 20 + ["Lost"] * 40 + ["Initializing"] * 10 + ["Tracking"] * 20
    mp = [0] * 20 + [-1] * 50 + [1] * 20
    with open(d / "frames.csv", "w") as f:
        f.write("index,t,load_ms,track_ms,state,pose,map\n")
        for i, (s, m) in enumerate(zip(st, mp)):
            f.write(f"{i},{i / 10:.6f},1,1,{s},{int(s == 'Tracking')},{m}\n")
    (d / "maps" / "map_00" / "frame_trajectory.txt").write_text("".join(f"{i / 10:.6f} 0 0 0 0 0 0 1\n" for i in range(20)))
    (d / "maps" / "map_01" / "frame_trajectory.txt").write_text("".join(f"{i / 10:.6f} 0 0 0 0 0 0 1\n" for i in range(70, 90)))
    (d / "maps.json").write_text(json.dumps([{"map": 0, "status": "saved", "first_t": 0.0, "last_t": 1.9, "tracked_frames": 20, "dir": "maps/map_00"},
                                             {"map": 1, "status": "saved", "first_t": 7.0, "last_t": 8.9, "tracked_frames": 20, "dir": "maps/map_01"}]))
    c = CV.for_run("r1", frames_dir=None)
    assert abs(c["tracked_share_whole"] - 4.0 / 9.0) < 1e-3 and [round(m["tracked_s"], 1) for m in c["maps"]] == [2.0, 2.0]
    assert json.loads((d / "coverage.json").read_text())["gaps_after_init"] == 1
    assert "from initialisation at 0.00 s" in CV.summary_line(c)
