# Capture record fixtures

Test data for the capture record's check (A4 Delivery 1; `docs/capture-record.md`), run by
`tools/csharp-tests/Contract`. Not training data, and not measurements: every number is an illustrative test value.

`valid/capture.bedroom.json` is a scan of two standing positions, three steps at the first and two at the second,
turning 30 degrees per step, with a few detections and one saved image. It shows the conventions at work: the table's
box runs 11.5 pixels past the image's right edge, as the detector's unclipped boxes can, and the SDK times are
nanoseconds that map to the log's microseconds with a 5000 us offset, rounding down. It passes.

`invalid/` holds one file per rule, each the valid scan with one change, named `<expected code>__<case>.json`; each
must fail with exactly that code.

| File | Change |
|---|---|
| `E_RECORD_TYPE__version_2.json` | schema_version is 2 |
| `E_SCHEMA_REQUIRED__keyframe_without_pose.json` | keyframe 1 has no camera_pose |
| `E_SCHEMA_REQUIRED__no_clock.json` | the record has no clock |
| `E_SCHEMA_ENUM__tracking_state.json` | keyframe 2's tracking state is "lost" |
| `E_SCHEMA_PATTERN__package_sha256.json` | the detector package's SHA-256 is "XYZ" |
| `E_SCHEMA_RANGE__score_above_one.json` | a detection's score is 1.2 |
| `E_SCHEMA_RANGE__nms_iou_one.json` | the NMS IoU threshold is 1 |
| `E_SCHEMA_CONST__clipped_boxes.json` | the detector's boxes are said to be clipped |
| `E_SCHEMA_CONST__optical_pose_convention.json` | the poses are said to be in the optical frame |
| `E_CAPTURE_ORDER__index_gap.json` | keyframes 3 and 4 are numbered 4 and 5 |
| `E_CAPTURE_ORDER__time_backwards.json` | keyframe 2 is captured before keyframe 1 (its SDK time moved with it) |
| `E_CAPTURE_ORDER__position_skipped.json` | the second position is numbered 2 |
| `E_CAPTURE_ORDER__step_not_restarted.json` | position 1 starts at step 1 |
| `E_CAPTURE_FRAME_ID__repeated.json` | keyframe 4 reuses keyframe 3's frame ID |
| `E_CAPTURE_TIME__mapping_mismatch.json` | keyframe 1's log time is 1 us off its SDK time plus the offset |
| `E_CAPTURE_EPOCH__recenter_mid_scan.json` | keyframe 4 is in epoch 1, after a recenter |
| `E_CAPTURE_NOT_TRACKED__untracked_keyframe.json` | keyframe 2, which has no detections, lost tracking |
| `E_CAPTURE_NOT_TRACKED__detections_without_tracking.json` | keyframe 3 lost tracking but keeps its detections |
| `E_CAPTURE_BOX__off_the_image.json` | a box lies wholly right of a 1280-pixel image (x 1300 to 1350) |
| `E_CAPTURE_BOX__inverted_box.json` | a box's x1 is past its x2 |
| `E_CAPTURE_INTRINSICS__principal_point_outside.json` | cx is 2000 in a 1280-pixel image |
| `E_QUAT_NORM__scaled_rotation.json` | keyframe 1's rotation is [0, 0, 0, 1.01] |
