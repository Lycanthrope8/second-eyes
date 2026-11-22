import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import process_recording as PR  # noqa: E402


def test_maps_are_reconstructed_only_when_long_enough_and_a_board_was_seen():
    maps = [{"map": 0, "status": "saved", "first_t": 8.0, "last_t": 75.0, "tracked_frames": 990},
            {"map": 1, "status": "discarded", "first_t": 77.0, "last_t": 78.0, "tracked_frames": 15},
            {"map": 2, "status": "saved", "first_t": 80.0, "last_t": 120.0, "tracked_frames": 600},
            {"map": 3, "status": "saved", "first_t": 150.0, "last_t": 170.0, "tracked_frames": 300},
            {"map": 4, "status": "saved", "first_t": 171.0, "last_t": 173.0, "tracked_frames": 30}]
    windows = {"ref01": [[0.0, 4.0], [46.0, 54.0]], "ref02": [[158.0, 166.5]]}
    plan = {m["map"]: (seen, why) for m, seen, why in PR.plan_maps(maps, windows, 5.0)}
    assert set(plan) == {0, 2, 3, 4}                                  # the discarded map is not considered
    assert plan[0] == (["ref01"], None) and plan[3] == (["ref02"], None)
    assert plan[2][1].startswith("no board seen") and "under --min-map-s" in plan[4][1]



def test_overlapping_board_windows_become_one():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from qrecon import questconfig as QC
    w = QC.merge_windows({"ref02": [[330.13, 332.63], [276.11, 280.61], [331.63, 338.13]], "ref01": [[46.0, 48.0], [49.0, 54.0]]})
    assert w["ref02"] == [[276.11, 280.61], [330.13, 338.13]] and w["ref01"] == [[46.0, 48.0], [49.0, 54.0]]


def test_a_board_anchor_is_trimmed_to_the_stretch_around_its_board():
    assert PR.board_span(244.9, 301.4, [[297.5, 301.5], [10.0, 20.0]], 10.0) == (287.5, 301.4)
    assert PR.board_span(10.0, 40.0, [[0.0, 5.0]], 10.0) is None
    assert PR.board_span(100.0, 200.0, [[120.0, 125.0], [150.0, 155.0]], 10.0) == (110.0, 165.0)
