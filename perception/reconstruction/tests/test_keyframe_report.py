import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import keyframe_report as KR  # noqa: E402
import reconstruct_on_quest as RQ  # noqa: E402


def test_rejections_are_counted_by_reason_and_by_piece():
    kfs = [{"t": 1.0, "calibrated": True, "reason": "", "landmarks": 600, "calibration": {"landmarks": 600, "cells": 9}},
           {"t": 2.0, "calibrated": False, "reason": "too few landmarks", "landmarks": 0, "calibration": {"landmarks": 0}},
           {"t": 30.0, "calibrated": False, "reason": "fit fails on held-out landmarks", "landmarks": 400,
            "calibration": {"landmarks": 400, "holdout_median_rel_err": 0.13}}]
    r = KR.report(kfs, [(0.0, 10.0, "map 0 (board)"), (25.0, 40.0, "map 4 (board)")])
    assert r["calibrated"] == 1 and r["reasons"]["too few landmarks"] == 1 and r["too_few_with_none"] == 1
    assert r["pieces"]["map 0 (board)"] == {"calibrated": 1, "keyframes": 2, "reasons": {"too few landmarks": 1}}
    assert r["by_20s"] == {"0-20 s": [1, 2], "20-40 s": [0, 1]}


def test_settings_can_be_changed_for_one_run_and_unknown_ones_are_refused():
    cfg = {"calibration": {"max_holdout_rel_err": 0.10, "min_cells": 6}, "landmarks": {"neighbour_offsets": [-2, -1, 1, 2]}}
    done = RQ.apply_settings(cfg, ["calibration.max_holdout_rel_err=0.15", "landmarks.neighbour_offsets=[-4,-3,-2,-1,1,2,3,4]"])
    assert cfg["calibration"]["max_holdout_rel_err"] == 0.15 and cfg["landmarks"]["neighbour_offsets"][0] == -4
    assert done == {"calibration.max_holdout_rel_err": 0.15, "landmarks.neighbour_offsets": [-4, -3, -2, -1, 1, 2, 3, 4]}
    import pytest
    with pytest.raises(SystemExit):
        RQ.apply_settings(cfg, ["calibration.no_such_setting=1"])
