# The automatic scan (A4)

Every time the app starts, the headset scans the room by itself: the on-device detector names what it sees, and Meta's
depth places each object in 3D from a single view. Nobody labels anything, walks between spots or holds still. What
the scan finds joins the session as a perceived scene, so typed commands can refer to it. The project lead chose this
on 17 October 2026, after three bedroom scans with the walking stand-in (`scan.md`) gave no usable scene: the headset
scans its side of the room once, accurately, and the drone will later scan the part of the room it can't see.

Code: the core is `quest-app/Assets/SecondEyes/Perception/AutoScan/AutoScan.cs` (free of Unity: the same code runs on the
headset and in the PC's replay); the Unity glue is `Perception/AutoScan/Glue/AutoScanner.cs`. Settings:
`schemas/auto-scan-config.v1.json`, defaults `perception/autoscan/autoscan.default.v1.json`, in the perception folder
(`perception-folder.md`).

## What happens

1. The scanner installs itself when the app starts and registers with the panel. It loads the perception folder,
   switches the detector to its on-request mode, measures the camera's model from the camera's own rays (`scan.md`), and
   starts Meta's depth: the spatial-data permission is asked for if it isn't granted, `EnvironmentDepthManager` runs
   with occlusion rendering off (depth for raycasts only) and hands removed, and `EnvironmentRaycastManager` casts rays
   against it (Meta XR SDK 207.0.0).
2. Every `interval_s`, while the head is tracked and no command is being processed, the detector runs on the current
   camera frame. Each detection whose label the category map knows, at or above `min_score`, is sampled with Meta's
   raycasts at five points of its box: its centre, and four points `sample_spread` of its width and height away. Each
   ray leaves the camera's position at the frame through the pixel, by the measured pinhole and the frame's pose.
3. The frame, every detection and sample included, is one line of `frames.jsonl`; `AutoScan` then processes exactly that
   line:
   - a label the map doesn't know, a score below `min_score`, a frame taken without tracking, or a frame from a later
     tracking epoch is left out;
   - the samples that hit a surface are sorted by distance from the camera, and the median one is the detection's
     point, so a ray slipping through a chair's mesh back to the wall doesn't move the chair; with no hit, it is left out;
   - a point beyond `max_range_m`, or lower than `floor_margin_m` below the floor (`floor_y_m`: the Quest's floor-level
     origin puts the floor at 0), is left out;
   - the point joins the nearest object of its category within reach, the larger of `min_reach_m` and `reach_factor`
     times the larger of the object's and the sighting's measured size (box size times distance over the focal
     length), or starts a new object. An object's centre is the per-axis median of its most recent
     `max_points_per_object` points; objects of a category that come within reach of each other merge.
4. An object seen in at least `min_frames` frames is in the scene. At most every `publish_interval_s`, a changed scene is
   published: `scene.json`, `trace.json` and `facts.json` are rewritten, the panel binds it to the session (replacing
   the scan's previous snapshot), and a label with its category floats over each object in view.
5. Sending a typed command binds the newest snapshot first, then holds the scan until the command's outcome is shown,
   so the parser has the headset's processors to itself. The panel's Scan button reads Pause or Resume.

Each object's centre names the `depth_surface` assumption: it is the median of points on the visible surface, so it lies
toward the camera from the true centre by up to half the object's depth. Its views count the frames it was seen in and
the viewpoints (camera positions `viewpoint_m` apart) it was seen from; seen from two or more, it is `multi_position`.
Labels are the detector's own: the bedroom's map v2 (`perception/bedroom/category-map.bedroom.v2.json`) keeps its couch
as a couch.

## The settings

| Setting | Default | What it does |
|---|---|---|
| `interval_s` | 0.5 | seconds between detector runs |
| `min_score` | 0.3 | detections below it are left out |
| `sample_spread` | 0.2 | where the four outer depth samples sit, as a fraction of the box |
| `max_range_m` | 6 | depth hits farther than this are left out |
| `floor_y_m` | 0 | the floor's height in `quest_tracking` |
| `floor_margin_m` | 0.15 | a point this far below the floor is impossible |
| `min_reach_m` | 0.3 | the smallest reach for joining an object |
| `reach_factor` | 0.5 | the reach grows to this fraction of the object's measured size |
| `min_frames` | 2 | frames an object needs before it is in the scene |
| `viewpoint_m` | 0.5 | camera positions closer than this are one viewpoint |
| `max_points_per_object` | 64 | the recent points an object's median uses |
| `publish_interval_s` | 3 | at most this often, a changed scene is published |
| `save_images_max` | 120 | frames saved as images, when `files/scans/save-images` exists |

The defaults are first values, set by reasoning, not tuned; a change is a new `config_id`.

## The files

Under `files/scans/(scan ID)/` on the headset: `frames.jsonl` (a header line, then one line per frame), `scene.json` and
`trace.json` (the latest snapshot; the trace gives every detection's fate and what was left out), and `facts.json` (the
camera's model, the counts, whether depth is supported). Frame images, when saved, are under `files/snapshots/`.
`grounding/quest/scan_device.py` pulls a scan (`pull`) and writes `view.html` (`view`), each box coloured by its fate.

## The replay on the PC

`dotnet run --project tools/localize -- --memory <frames.jsonl> <category-map> <size-priors> <autoscan-config> <out-dir>
[schemas] [--compare <scene.json>]` replays the headset's frames through the same `AutoScan` code. With `--compare`, it
stops at the frame count that the trace beside the compared scene recorded, and the scene must be the same: the same
objects, categories and placements, centres within 1 mm.

## Checks

`tools/csharp-tests/AutoScan` (32 checks) covers the settings, the sampling and the pinhole, every leave-out reason,
joining and merging (two chairs 1.2 m apart stay two; a bed seen at parts 0.72 m apart is one), the median's
robustness to a ray slipping through, viewpoints, the replay byte for byte, the order of detections and the contract.
`tools/csharp-tests/ScanGlue` compiles the scanner against stubs of the Unity, app and Meta SDK 207 members it calls,
the Meta signatures copied from the SDK's source.

## Known limits

- The room's walls aren't used yet. Meta's room model (MRUK) would let the scan drop objects beyond the walls, such as
  reflections in a mirror, but starting MRUK turns on its world lock, which moves the tracking space every position here
  is measured in. Until that is handled, mirrors are best covered while scanning.
- A position is a point on the visible surface, not the object's centre.
- Labels are what the detector says: a chair seen from behind can be called something else, and a laptop's own keyboard
  counts as a keyboard. Correcting a label by tapping it is for later.
- Meta's sample code is under the Oculus SDK License Agreement, not a licence D83 allows, so none of it is copied; the
  scan calls the SDK, which the app already depends on.
