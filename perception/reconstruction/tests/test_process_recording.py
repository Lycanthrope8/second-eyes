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
