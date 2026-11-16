# Capture interface and calibration procedure

Written before the CM4 and camera arrive, so the video source can be connected later without changing the
reconstruction stages.

## The interface

Every source delivers the same thing (`qrecon/sources.py`, `Frame`):

| Field | Meaning |
|---|---|
| `index`, `t` | Frame number and capture time in seconds on the source's clock |
| `bgr` | The image, already undistorted |
| `K` | Intrinsics of that undistorted image |
| `T_wc`, `depth` | Reference pose and depth. Present only on public sequences; never on a real camera |

Today: `TumSource` (public sequences) and `VideoFileSource` (a recorded file). The CM4 adds a third source with
the same output: frames over Wi-Fi, each with the CM4's capture timestamp. Telemetry travels beside it with its
own timestamps and is never mixed into the frame.

Tracking takes every frame. Mapping takes keyframes. A source that drops to one frame a second breaks tracking,
even if processing happens later.

## Calibration, per camera and per video mode

1. Fix the mode: resolution, frame rate, and **locked focus**. Camera Module 3 is autofocus; set a manual lens
   position and keep it. On a phone, lock focus and turn stabilisation off.
2. Record 30–60 s of a printed checkerboard in that exact mode, moving it through the whole image, tilted and at
   several distances. Measure one square with a ruler.
3. `python scripts/video_tools.py calibrate --video board.mp4 --cols 9 --rows 6 --square-m 0.025 --out calib.json`
   (`cols` and `rows` count inner corners).
4. Keep `calib.json` with the recordings it belongs to. A different mode or focus setting needs its own file.

## Recording a room

- Start and end on the room tag, seen from **several positions a few tens of centimetres apart**. Scale comes
  from those translated views; turning on the spot gives none.
- Move slowly, with short pauses. Prefer sideways and forward movement to rotation in place.
- Short exposure reduces blur. It doesn't remove rolling-shutter distortion, so slow turns matter.
- Note the tag's printed ID and its measured edge length.

## ChArUco references (the house video)

The house video shows ChArUco 01 at the start and ChArUco 02 on reaching the second room, each about 1.2 m above the
floor, filmed from about 1 m. A ChArUco board gives dozens of identified corners, so it serves three purposes:

| Purpose | From | Tool |
|---|---|---|
| Camera calibration (no separate calibration video exists) | the board 01 views | `video_tools.py charuco-calibrate` |
| Metric scale and the map frame | board 01, seen from translated positions | `video_tools.py charuco-anchor` |
| Scale drift across the house, and the link between the rooms | board 02, fitted in board 01's frame | the same command |

What the tools need, in `configs/references.yaml`, for each board:
- the dictionary and marker IDs (run `video_tools.py charuco-inspect --video house.mp4 --end 20` to see them);
- squares across and down;
- the printed square length and marker length, measured with a ruler;
- whether it hangs upright and level (then it also gives the up direction).

Limits of calibrating from these views: a board filmed from about 1 m covers part of the image only, so distortion at
the image edges is weakly determined. The calibration reports its image coverage, and with low coverage it estimates
fewer distortion terms (`profile: weak` in `configs/references.yaml`: focal length and k1 only) and reports how
uncertain the focal length is.

The house video's board windows are in `configs/references.yaml`. They come from sampled detection times, not verified
continuous visibility:
- board 01: 0–20 s, and again at about 5:00, which gives a closure check at the end;
- board 02: about 2:43 and 3:57.

Boards are detected on full-resolution frames, because at half resolution the markers are only a few pixels across.

## What a recording can show

| Recording | Tracking | Depth and fusion | Metric scale |
|---|---|---|---|
| With a tag or board of known size, or a tape-measured distance in view | Yes | Yes | Yes |
| With two references in one continuous recording (the house video) | Yes | Yes | Yes, and scale drift between them |
| Without either (possibly the existing house video) | Yes | Yes | **No** |

## Still to prepare when the hardware arrives

- The same calibration on the Pi Camera 3 in the mode the drone will stream.
- Timestamp relationship between the CM4's camera frames and the flight controller's messages: offset and drift.
- The camera's mounting position on the drone relative to the flight controller.
