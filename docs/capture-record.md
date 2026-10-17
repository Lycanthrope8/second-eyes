# The capture record (A4)

One scan on the headset: what the camera saw at each stop, where the camera was, and what the detector found on that very
frame. A perceived scene (`scene-contract.md`, format v3) is built from it, and the localization can be run again on the PC
from it. Format: `schemas/capture.v1.json`. It is written on the headset and checked there by
`quest-app/Assets/SecondEyes/Perception/Capture/CaptureCheck.cs`; no Python validator exists yet.

## What it holds

- **The camera:** its ID, image size in pixels, intrinsics (`fx, fy, cx, cy` in OpenCV's pixel convention, `frames.md`)
  with where they came from (on the headset: fitted from the camera's own viewport rays, or the reported intrinsics
  carried to the texture by the centre-crop model; `scan.md`), and whether the images carry lens distortion (`none`, as stated by that source, or
  `unknown`).
- **The app's build.**
- **The detector**, as its manifest (`detector-package.json`) states it: package ID and SHA-256, input size, top-left
  letterbox, score threshold, class-agnostic greedy NMS and its IoU threshold, and the box convention below.
- **The configuration** that turned detections into a scene: the perception folder's manifest SHA-256, the category map's
  ID and the size priors' ID.
- **The clock:** the unit of the SDK's image timestamps (`ns`, `us`, `ms` or `100ns`) and `offset_us`, so that each
  keyframe's `headset_mono_us` is its SDK value in microseconds, rounded down, plus `offset_us`.
- **The pose convention:** `unity_camera_to_quest_tracking`. Each keyframe's `camera_pose` maps Unity's camera axes
  (`unity_camera`: x right, y up, z forward) into `quest_tracking`. Both are left-handed, so the rotation is proper; the
  optical frame and the scene are reached only through `UnityBasis` (`frames.md`).
- **The nominal step** in degrees: a keyframe's nominal turn at its position is `step_index × nominal_step_deg`.
- **The keyframes**, in order. Each has its `index`; a `frame_id`, unique in the capture, which ties the image, its pose
  and its detections together; `position_index` and `step_index`; its capture time (the SDK value and the log time); the
  tracking epoch and state; the camera's pose at capture, exactly as Unity reports it; the detections the detector made on
  this frame (label, score, box); and the image's file and SHA-256 when image saving is on, otherwise `not_saved`.

**Boxes** are `x1, y1, x2, y2` in the source image's pixels: the model's input-pixel boxes divided by the letterbox ratio,
**not clipped**, as the detector writes them (`source_pixels_unclipped`). A box may therefore run past the image's edge;
the localization reads such a box as cut off at that edge.

Images of a room can be private: saving is off by default, and saved images stay in `second-eyes-data`, never in Git.
Without them, a rerun on the PC repeats the localization from the recorded detections and poses only: it can't rerun the
detector or check the original pixels.

## Building one

`quest-app/Assets/SecondEyes/Perception/Capture/CaptureBuilder.cs`, Unity-free, builds the record keyframe by keyframe, so
the headset and the PC's checks build records the same way. It copies the detector block from the package's manifest,
never restating it, and refuses a package whose boxes are anything but source pixels, not clipped. Each keyframe's
`headset_mono_us` is derived from its SDK timestamp and the scan's one clock offset, so the clock rule below holds by
construction. `tools/csharp-tests/Contract` rebuilds the valid fixture from its own values and gets it back, value for
value, with the detector block from the real manifest.

## Rules

The check runs in layers, each only if the one before it passed: the record type and version, the JSON Schema, finite
numbers, then these rules.

| Code | Rule |
|---|---|
| `E_RECORD_TYPE` | not a capture record of format 1 |
| `E_SCHEMA_*` | the schema, by keyword, as in the scene contract (the detector's settings, the box and pose conventions, the clock's unit) |
| `E_NONFINITE` | a number is NaN or infinite |
| `E_CAPTURE_INTRINSICS` | the principal point is outside the image |
| `E_CAPTURE_ORDER` | keyframes aren't numbered 0, 1, 2... in order; capture times don't rise strictly; positions don't start at 0 and go up one at a time; or steps don't start at 0 at each position and go up one at a time |
| `E_CAPTURE_FRAME_ID` | a frame ID is used twice |
| `E_CAPTURE_TIME` | a keyframe's log time isn't its SDK time in microseconds, rounded down, plus `offset_us` |
| `E_CAPTURE_EPOCH` | a keyframe isn't in the first keyframe's tracking epoch: the tracking moved during the scan |
| `E_CAPTURE_NOT_TRACKED` | a keyframe isn't tracked. The app keeps only tracked keyframes; a later pose is never used in place of the frame's own |
| `E_QUAT_NORM` | a camera rotation isn't of unit length within the contract's tolerance (1e-6) |
| `E_CAPTURE_BOX` | a box isn't `x1 < x2` and `y1 < y2`, or doesn't overlap the image at all |

The fixtures are in `grounding/tests/fixtures/capture/`, with a README listing each invalid case;
`tools/csharp-tests/Contract` checks every one.

## Not settled yet

- The headset turns the camera's timestamp into a number and unit as it finds it (`Perception/Scan/ScanSession.cs`,
  `SdkStamp`): a `DateTime` becomes its 100 ns ticks, a whole number is read as nanoseconds, seconds become microseconds.
  The offset to the log's clock is set at the first keyframe kept, taking that frame's age as zero; every later frame's
  age against it is kept in the scan's `facts.json`, and the stage 3 trial checks both (`scan.md`).
- Lens distortion: the record states what its source says. The localization doesn't correct distortion, so `unknown`
  is localized as if it were none (`localization.md`, Known limits).
