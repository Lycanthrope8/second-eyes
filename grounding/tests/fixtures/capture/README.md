# Capture record fixtures

Test data for the capture record's check (A4 Delivery 1; `docs/capture-record.md`), run by
`tools/csharp-tests/Contract`. Not training data, and not measurements: every number is an illustrative test value.

`valid/capture.bedroom.json` is a scan of two standing positions, three steps at the first and two at the second,
turning 30 degrees per step, with a few detections and one saved image. It passes.

`invalid/` holds one file per rule, each the valid scan with one change, named `<expected code>__<case>.json`; each
must fail with exactly that code.

| File | Change |
|---|---|
| `E_RECORD_TYPE__version_2.json` | schema_version is 2 |
| `E_SCHEMA_REQUIRED__keyframe_without_pose.json` | keyframe 1 has no camera_pose |
| `E_SCHEMA_ENUM__tracking_state.json` | keyframe 2's tracking state is "lost" |
| `E_SCHEMA_PATTERN__package_sha256.json` | the detector package's SHA-256 is "XYZ" |
| `E_SCHEMA_RANGE__score_above_one.json` | a detection's score is 1.2 |
| `E_CAPTURE_ORDER__index_gap.json` | keyframes 3 and 4 are numbered 4 and 5 |
| `E_CAPTURE_ORDER__time_backwards.json` | keyframe 2 is captured before keyframe 1 |
| `E_CAPTURE_ORDER__position_skipped.json` | the second position is numbered 2 |
| `E_CAPTURE_ORDER__step_not_restarted.json` | position 1 starts at step 1 |
| `E_CAPTURE_EPOCH__recenter_mid_scan.json` | keyframe 4 is in epoch 1, after a recenter |
| `E_CAPTURE_NOT_TRACKED__detections_without_tracking.json` | keyframe 3 lost tracking but keeps its detections |
| `E_CAPTURE_BOX__outside_image.json` | a box runs to x = 1300 in a 1280-pixel image |
| `E_CAPTURE_BOX__inverted_box.json` | a box's x1 is past its x2 |
| `E_CAPTURE_INTRINSICS__principal_point_outside.json` | cx is 2000 in a 1280-pixel image |
| `E_QUAT_NORM__scaled_rotation.json` | keyframe 1's rotation is [0, 0, 0, 1.01] |
