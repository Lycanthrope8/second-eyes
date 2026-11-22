"""Tracking presets, for tools/tracking_sweep.py and process_recording.py --profile.

Each preset says how often video frames are fed to the tracker (every Nth frame: 2 gives 15 frames a second from a
30 fps video, 1 gives all 30), how long tracking may stay lost before the map is saved and a new one started, and which
stella_vslam settings differ from their defaults. The aim is fewer breaks in tracking, so that one continuous video gives
as few maps as possible. These are method settings, the same for every recording.
"""

PROFILES = {
    "baseline": {
        "about": "15 frames a second, stella's defaults (what the recordings were processed with so far)",
        "every": 2, "restart_after": 2.0, "stella": {}},
    "fps30": {
        "about": "all 30 frames a second: half the motion between frames, so fast turns break tracking less",
        "every": 1, "restart_after": 2.0, "stella": {}},
    "fps30_features": {
        "about": "30 frames a second, more and weaker corners kept, for plain walls",
        "every": 1, "restart_after": 2.0,
        "stella": {"Preprocessing": {"min_size": 400}, "Feature": {"ini_fast_threshold": 12, "min_fast_threshold": 4}}},
    "fps30_features_keep": {
        "about": "as fps30_features, young maps kept (stella resets a map lost within 0.5 s of starting, not 5 s) and "
                 "4 s allowed for relocalising before a new map",
        "every": 1, "restart_after": 4.0,
        "stella": {"Preprocessing": {"min_size": 400}, "Feature": {"ini_fast_threshold": 12, "min_fast_threshold": 4},
                   "Tracking": {"init_retry_threshold_time": 0.5}}},
}
