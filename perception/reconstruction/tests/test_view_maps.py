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


def test_a_one_map_summary_labels_its_single_run_and_lists_what_it_left_out(tmp_path):
    (tmp_path / "v_recording.json").write_text(json.dumps({"name": "v", "maps": [{"map": 0, "anchor_run": "a0"}, {"map": 1}],
        "one_map": {"recon_run": "om2", "joined_s": 173.4, "share_of_video": 0.575, "valid_keyframe_share": 0.6, "frame": "ref01",
                    "pieces": [{"map": 0, "how": "board", "first_t": 5.1, "last_t": 91.9}],
                    "left_out": [{"map": 1, "seconds": 7.2, "reason": "no frame matched a placed piece"}]}}))
    (tmp_path / "v_recording.json.bak").write_text(json.dumps({"name": "v", "maps": [], "one_map": {
        "recon_run": "om1", "joined_s": 100.0, "share_of_video": 0.33, "pieces": [], "left_out": [{"map": 9, "seconds": 1, "reason": "x"}]}}))
    placed, missing = VM.map_info(tmp_path)
    assert placed["om2"]["title"] == "v one map" and "57% of the video" in placed["om2"]["span"]
    assert "map 0 5-92 s by board" in placed["om2"]["how"] and placed["om1"]["title"] == "v one map"
    assert missing == ["v map 1 (7.2 s): left out of the one map: no frame matched a placed piece"]     # only the current summary's
    (tmp_path / "om0" / "recon").mkdir(parents=True)
    (tmp_path / "om0" / "manifest.json").write_text(json.dumps({"name": "v"}))
    (tmp_path / "om0" / "recon" / "anchor.json").write_text(json.dumps({"frame": "ref01", "fixed": {"by": "one map of 4 piece(s)"}}))
    assert VM.run_label(tmp_path, "om0")["title"] == "v one map of 4 piece(s)"
