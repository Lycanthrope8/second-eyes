# The capture record (A4)

One scan on the headset: what the camera saw at each stop, where the camera was, and what the detector found there. A
perceived scene (`scene-contract.md`, format v3) is built from it, and localization can be run again on the PC from it.
Format: `schemas/capture.v1.json`. It is written on the headset and checked there by
`quest-app/Assets/SecondEyes/Perception/Capture/CaptureCheck.cs`; no Python validator exists yet.

## What it holds

- **The camera:** its ID, image size in pixels, intrinsics (`fx, fy, cx, cy` in OpenCV's pixel convention, `frames.md`)
  with where they came from, and whether the images carry lens distortion (`none`, as stated by that source, or
  `unknown`).
- **The app:** its build, and the detector package's ID and SHA-256.
- **The keyframes**, in order. Each has:
  - its place in the scan: `index` (0, 1, 2...), `position_index` (which standing position, from 0) and `step_index`
    (which stop at that position, from 0);
  - its capture time: the SDK's own timestamp, verbatim, and the same moment on the event log's monotonic clock
    (`headset_mono_us`);
  - the tracking epoch and state (`tracked` or `not_tracked`);
  - the camera's pose in `quest_tracking` at capture, exactly as Unity reports it: `translation_m` (p) and
    `rotation_xyzw` (q). `UnityBasis.CameraToScene` turns it into `T_scene_camera` (`frames.md`);
  - the detections: label, score and box (`x1, y1, x2, y2` in the image's pixels, top-left origin);
  - the image's file and SHA-256 when image saving is on, otherwise `not_saved`.

Images of a room can be private: saving is off by default, and saved images stay in `second-eyes-data`, never in Git.

## Rules

The check runs in layers, each only if the one before it passed: the record type and version, the JSON Schema, finite
numbers, then these rules.

| Code | Rule |
|---|---|
| `E_RECORD_TYPE` | not a capture record of format 1 |
| `E_SCHEMA_*` | the schema, by keyword, as in the scene contract |
| `E_NONFINITE` | a number is NaN or infinite |
| `E_CAPTURE_INTRINSICS` | the principal point is outside the image |
| `E_CAPTURE_ORDER` | keyframes aren't numbered 0, 1, 2... in order; capture times don't rise strictly; positions don't start at 0 and go up one at a time; or steps don't start at 0 at each position and go up one at a time |
| `E_CAPTURE_EPOCH` | a keyframe isn't in the first keyframe's tracking epoch: the tracking moved during the scan |
| `E_CAPTURE_NOT_TRACKED` | a keyframe without tracking carries detections; its pose isn't known |
| `E_QUAT_NORM` | a camera rotation isn't of unit length within the contract's tolerance (1e-6) |
| `E_CAPTURE_BOX` | a box isn't `x1 < x2` and `y1 < y2` inside the image |

The fixtures are in `grounding/tests/fixtures/capture/`, with a README listing each invalid case;
`tools/csharp-tests/Contract` checks every one.

## Not settled yet

- How the SDK's timestamp is mapped to `headset_mono_us`, and whether the pose the SDK gives belongs to the image's
  capture time, are checked on the Quest when capture is built (A4 Delivery 1, stage 3), not assumed.
- Lens distortion: the record states what its source says. How the localization treats `unknown` is decided with the
  localization core (A4 Delivery 1, stage 2).
