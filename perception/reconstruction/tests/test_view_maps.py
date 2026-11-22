import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import view_maps as VM  # noqa: E402


def test_maps_are_described_from_the_recording_summaries(tmp_path):
    (tmp_path / "a_recording.json").write_text(json.dumps({"name": "a", "maps": [
        {"map": 0, "first_t": 10.0, "last_t": 54.0, "recon_run": "r1", "frame": "ref01", "valid_keyframe_share": 0.87},
        {"map": 5, "first_t": 128.0, "last_t": 176.0, "recon_run": "r2", "frame": "ref02", "valid_keyframe_share": 0.83,
         "placed_by": {"with_maps": ["b map 3"], "kept": "trimmed", "span_s": [159.3, 176.1]}},
        {"map": 1, "first_t": 72.0, "last_t": 92.0, "skipped": "no board seen"}]}))
    placed, missing = VM.map_info(tmp_path)
    assert placed["r1"]["how"] == "anchored by board 01" and placed["r1"]["usable"] == "87% of keyframes usable"
    assert placed["r2"]["how"] == "placed by shared views with b map 3" and placed["r2"]["span"].startswith("159-176 s kept")
    assert missing == ["a map 1 (72-92 s): no board seen"]
