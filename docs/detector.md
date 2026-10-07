# The on-device object detector (A1.10b, A1.10c)

The detector finds objects in camera frames on the Quest; in the full system it runs on drone keyframes during the
pre-scan. A1.10 shows that it runs beside mixed-reality rendering, what it costs, and that the headset's results match
the PC's. Accuracy on the lab props is A4's job, after fine-tuning: COCO has chairs and dining tables but no boxes.

## The detector package (D84)

`quest-app/Assets/SecondEyes/Models/yolox-nano/` holds one replaceable package:

- `yolox_nano_se.onnx`: YOLOX-Nano's released ONNX file (YOLOX release 0.1.1rc0, Apache-2.0, 0.91M parameters,
  416-pixel input, 80 COCO classes), wrapped so the headset and the PC run the same graph. Input `image`: RGB in [0, 1],
  [1, 3, 416, 416]. Outputs `boxes` (x1, y1, x2, y2 in input pixels), `scores` (the best class's objectness x class
  probability) and `classes` (its COCO index), one row per anchor point (3,549).
- `detector-package.json`: the manifest: the model's and the release's SHA-256, the input contract (top-left letterbox,
  pad value 114, the ratio rule), the post-processing (score threshold 0.3; greedy class-agnostic suppression at 0.45
  with YOLOX's +1 overlap rule; boxes divided by the ratio), the class list and the classes mapped to project
  categories (chair, and dining table as table).
- `LICENSE.txt`: YOLOX's Apache-2.0 license. D83 admits permissive licenses only.

Code reads detectors only through `IObjectDetector` (`Perception/IObjectDetector.cs`), beside `IFrameSource`. Another
YOLOX model with the same outputs is a new package and nothing else; another family is a new package plus one class
implementing the interface for its outputs.

## The PC side: `perception/detector`

Install `perception/detector/requirements.txt` (numpy, onnx, onnxruntime, OpenCV), then from the repository root:

```
python -m perception.detector build --source <path to yolox_nano.onnx>     # rebuild the package (byte-identical)
python -m perception.detector images --source-dir <folder with the originals>
python -m perception.detector reference [--check]                         # write, or reproduce, the PC reference
python -m perception.detector equivalence --source <path to yolox_nano.onnx>
python -m perception.detector parity <run ID or log.jsonl>
python -m perception.detector snapshot <run ID> --image <snapshot.png> --id <name>
python perception/detector/test_detector.py
```

`yolox_nano.onnx` comes from
https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx (SHA-256 `c789161e…`); `build`
refuses any other file. The test images live in `quest-app/Assets/SecondEyes/Perception/DetectorTests/` as PNG bytes,
so both sides read identical files: `*_416.png.bytes` are canvases already letterboxed by YOLOX's own `preproc`, and
`*_full.png.bytes` are full-size originals for the GPU letterbox. Their sources and hashes are in
`perception/detector/reference.py` and `fixtures/reference.v1.json`.

## Parity (D85)

On each test canvas the headset and the PC must have the same detections at or above 0.3: matched one-to-one by
class, with box IoU of at least 0.95 and scores within 0.02. A detection within 0.02 of the threshold that finds no
partner is reported as borderline, not as a failure (proposed reading). The canvases on the GPU backend decide; the
CPU backend, the GPU letterbox path (each full image as an sRGB and as a linear texture) and camera snapshots are
reported beside them. A snapshot saves a camera frame's letterboxed input exactly as the model saw it, so the PC can
judge real camera content by the same rule.

## Setting it up in Unity

1. Let Unity import the new files: the `.onnx` becomes a model asset, the `.json` and `.bytes` files text assets.
2. In `SampleScene`, add an empty object `Detector` with two components, `DetectorRunner` and `DetectorButtons`.
3. `DetectorRunner`: Model `Models/yolox-nano/yolox_nano_se`; Package Manifest `detector-package`; Letterbox Shader
   `Hidden/SecondEyes/Letterbox` (`Perception/Letterbox.shader`); Frame Source the `PassthroughFrameSource` on the
   `PassthroughCamera` object; Canvas Images the five `*_416.png` assets in `Perception/DetectorTests`; Full Images
   `dog_full.png` and `fruits_full.png`; Run Checks on; Use Gpu on; Max Rate Hz 0; Max Logged Detections 20; Flip
   Camera off.
4. A1.10d: add `CommandSchedule` (Grounding) to the `ChatPanel` object, with Panel the `ChatPanel`, Order 0, 1, 2,
   3 (the fixed prompt, then the presets) and Period 12 s; set `DetectorRunner`'s Commands and `DetectorButtons`'
   Commands to it. The right controller's grip starts and stops the schedule; sequential mode starts and stops it
   itself.
5. `DetectorButtons`: Detector, the `Detector` object. Add `FrameTimes` (App, D88) and `GpuFrameTimes` (App, D89)
   to the same object, enable Player Settings > Other Settings > Frame Timing Stats, and save the scene.
6. Optional, before building: in Play mode in the Editor, right-click `DetectorRunner` and choose "Detector on". This
   only works if Play mode is running frames: with OpenXR set to start in the Editor and no Quest Link, it may not be
   (the log then has no `camera.second` events). Start Quest Link, or untick Initialize XR on Startup on the PC tab of
   XR Plug-in Management, first. The self-checks run on the PC's GPU; check the session log (under the Editor's
   persistent data path) with `python -m perception.detector parity <log file>`. No camera frames arrive in the Editor.

## Running it on the headset (D88)

Each inference is dispatched over several frames: Worker.ScheduleIterable advances one layer per step, and the runner
takes a fixed number of steps per rendered frame (8, 16, 32 or 64; 0 dispatches the whole network in one frame, as
before D88). The model's layer count after import is logged at load (`layer_count`), and every result logs the steps it
took and the frames it spanned. One inference is in flight: its image is copied once, at its start, with the camera's
timestamp and pose (`capture_stamp`, `pose`), and its three outputs are read back together after the last step (D87).

Modes:
- **scan** (default): stop-and-shoot keyframes, 3 hover blocks of 12 keyframes at 1 Hz, 8 s repositioning between
  blocks and 15 s between scans, repeated until off. A keyframe that comes due while an inference is in flight is
  rejected and logged (`detector.reject`), never queued.
- **sequential** (A1.10d, D90): three cycles of load, one scan, release, then the fixed command schedule for 150 s;
  each stage is logged (`detector.cycle`), and the next cycle reloads the detector.
- **diagnostic**: one scan at 0.5 Hz, so even slow settings never overlap (D89); a diagnostic workload, not
  acceptance at 1 Hz.
- **profile**: five passes in which the preprocessing and then each scheduling step run alone, each followed by four
  empty frames, then the readbacks (`detector.profile`); with per-frame GPU times this measures every step's cost.
- **burst**: 2 Hz for 30 s, then idle (a diagnostic).
- **continuous**: back to back (stress evidence only).

**Cost-balanced schedules (D89).** `python -m perception.detector schedule <profiling run>` turns a profile run into
schedules: contiguous slices in graph order packed against a per-frame GPU budget, the first frame carrying the
preprocessing and the last the readback requests. Pushed to the headset's `files/schedules/`, each joins the
steps-per-frame choices (`schedule <id>` on the overlay). At load the app checks it against the model hash, backend
and the runtime's layer list, and refuses a mismatch. Nothing in the production loop measures or waits on the GPU: the
schedule is fixed.

**Measuring.** `GpuFrameTimes` (App) logs per-frame GPU times from FrameTimingManager (enable Player Settings > Other
Settings > Frame Timing Stats); `perception/detector/gpu_frames.py` assigns each timing to its frame. When a mode starts,
eight short deliberate stalls at logged times (`clock.sync`) let `analysis/detector_phases.py` align OVR Metrics'
one-second buckets with the log; it reports the overlap figures across every equally good offset and judges them at
the least favourable one, and lists frames over budget by inference stage (preprocessing, scheduling, readback).

Results keep detections down to a score of 0.25 so parity can verify threshold crossings; a detection is still one
at or above 0.3. Turning the detector off stops new keyframes, lets the inference in flight finish (up to 5 s) and then
releases it, logging the release time and whether anything was still running.

## How to tell that it works

The debug overlay's last line follows the detector, updated once a second: `detector starting`, `loading`,
`warming up (gpu_compute)`, `self-checks 3/9`, `on, waiting for camera frames (X)`, then for each camera inference
`GPU #123 21 ms, 2 found: chair 0.82, dining table 0.55` (backend, inference count, latency, detections and the two
best). Each stage is also written to Unity's console (`[SecondEyes] Detector: ...`, with the frame number) and to the
log as a `mark`, so in the Editor the console shows how far it got. The warm-up and the self-checks read their results
back synchronously, one inference per frame; camera frames read back asynchronously, all three outputs requested together (D87), unless Async
Camera Readback is off. If the detector makes no progress for 300 frames while the app keeps running, a watchdog reports
`stuck at <stage>` (console warning, `error` event). `failed at <stage>: <exception or reason>` means it stopped; the same text is in the log as an `error` event
and in Unity's console, the detector has been released, and Y starts it again. After a session, pull the log and run
`python -m perception.detector parity <run or log>` (the self-checks against the PC) and
`python analysis/detector_runs.py <run or log>` (latency and rates). Pointing at a chair or a table should show it on
the overlay; COCO calls many other things by their nearest class.

Controls on the left controller: Y turns the detector on and off (on: load, warm-up, self-checks, then the mode;
off: finish the inference in flight, then release the model). While it is off: press the thumbstick to switch GPU and
CPU, push it up or down to change the mode (scan, diagnostic, profile, burst, continuous), left or right to change the
steps per frame (4, 8, 16, 32, 64, whole network, then any pushed schedules); the overlay shows
`off | GPU | scan | 16 steps/frame`. While it is on, the left trigger saves the next keyframe's input as a snapshot.
Any 416 x 416 PNG in the app's `files/parity/` folder on the headset (pushed with adb, for example a saved snapshot) is
checked with the self-checks; `python -m perception.detector parity <run> --canvas-dir <folder with that PNG>` judges
it against a PC reference made from the same file. Parity reports each canvas as strict, as passing with verified
threshold crossings or unresolved borderlines (listed), or as failing. Events: `detector.*` in `docs/logging.md`. The measurement
procedure: `docs/profiling.md`, "A1.10c".
