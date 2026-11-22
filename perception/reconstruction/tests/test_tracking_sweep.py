import json, os, sys
import yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from qrecon import questconfig as QC  # noqa: E402
from qrecon.tracking_profiles import PROFILES  # noqa: E402
import tracking_sweep as TS  # noqa: E402

CALIB = {"K": [[1315.0, 0, 959.5], [0, 1315.0, 539.5], [0, 0, 1]], "dist": [0, 0, 0, 0, 0], "size": [1920, 1080]}


def test_profiles_change_only_known_stella_settings():
    known = {"Preprocessing": {"min_size"}, "Feature": {"ini_fast_threshold", "min_fast_threshold"},
             "Tracking": {"init_retry_threshold_time"}}
    for name, p in PROFILES.items():
        assert p["every"] in (1, 2) and p["restart_after"] > 0 and p["about"]
        for section, values in p["stella"].items():
            assert set(values) <= known[section], (name, section)


def test_camera_file_takes_the_overrides_and_keeps_the_camera():
    cfg = yaml.safe_load(QC.stella_camera_yaml(CALIB, 0.5, 30.0, overrides=PROFILES["fps30_features_keep"]["stella"]))
    assert cfg["Camera"]["fx"] == 657.5 and cfg["Camera"]["fps"] == 30.0 and cfg["Camera"]["cols"] == 960
    assert cfg["Feature"]["ini_fast_threshold"] == 12 and cfg["Feature"]["scale_factor"] == 1.2
    assert cfg["Preprocessing"]["min_size"] == 400 and cfg["Tracking"]["init_retry_threshold_time"] == 0.5
    assert "Tracking" not in yaml.safe_load(QC.stella_camera_yaml(CALIB, 0.5, 15.0))


def test_sweep_summary_counts_maps_breaks_and_the_longest_map(tmp_path):
    (tmp_path / "coverage.json").write_text(json.dumps({"processed": {"span_s": 300.0}, "tracked_share_whole": 0.8,
                                                        "tracked_share_from_init": 0.82}))
    (tmp_path / "maps.json").write_text(json.dumps([
        {"map": 0, "status": "saved", "first_t": 5.0, "last_t": 125.0}, {"map": 1, "status": "discarded", "first_t": 126, "last_t": 127},
        {"map": 2, "status": "saved", "first_t": 130.0, "last_t": 136.0}, {"map": 3, "status": "saved", "first_t": 140.0, "last_t": 299.0}]))
    (tmp_path / "stats.json").write_text(json.dumps({"map_restarts": 2, "track_ms_median": 19.0}))
    s = TS.summarize(tmp_path)
    assert (s["maps_saved"], s["maps_discarded"], s["breaks"], s["maps_10s_or_more"]) == (3, 1, 2, 2)
    assert s["longest_map_s"] == 159.0 and s["longest_map_share"] == 0.53
    assert "fps30" in TS.table([("fps30", s)])
