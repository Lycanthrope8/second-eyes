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


def test_a_contact_sheet_shows_refused_keyframes_first(tmp_path):
    import cv2
    import numpy as np
    (tmp_path / "rgb").mkdir()
    with open(tmp_path / "rgb.txt", "w") as f:
        f.write("# frames\n")
        for i in range(8):
            cv2.imwrite(str(tmp_path / "rgb" / f"{i:06d}.jpg"), np.full((90, 160, 3), 30 * i, np.uint8))
            f.write(f"{i * 0.5:.3f} rgb/{i:06d}.jpg\n")
    kfs = [{"t": i * 0.5, "frame": i, "calibrated": i % 2 == 0, "reason": "" if i % 2 == 0 else "fit fails on held-out landmarks",
            "calibration": {"holdout_median_rel_err": 0.14}} for i in range(8)]
    out = KR.contact_sheet(kfs, tmp_path, tmp_path / "sheet.jpg", between=(0.0, 2.6), thumb_w=160, cols=3)
    img = cv2.imread(str(out))
    assert img is not None and img.shape[1] == 3 * 160 and img.shape[0] == 2 * 90     # 3 refused + 3 calibrated in 0-2.6 s
    assert KR.contact_sheet(kfs, tmp_path, tmp_path / "none.jpg", between=(50, 60)) is None


def test_left_out_parts_of_a_view_are_darkened_on_the_sheet(tmp_path):
    import cv2
    import numpy as np
    (tmp_path / "rgb").mkdir()
    cv2.imwrite(str(tmp_path / "rgb" / "000000.jpg"), np.full((100, 100, 3), 200, np.uint8))
    (tmp_path / "rgb.txt").write_text("0.000 rgb/000000.jpg\n")
    kfs = [{"t": 0.0, "frame": 0, "calibrated": True, "reason": "",
            "calibration": {"holdout_median_rel_err": 0.05, "region_mask": "0111"}}]     # top-left quarter left out
    img = cv2.imread(str(KR.contact_sheet(kfs, tmp_path, tmp_path / "s.jpg", thumb_w=100, cols=1)))
    assert img[40, 30].mean() < 100 and img[80, 80].mean() > 150
