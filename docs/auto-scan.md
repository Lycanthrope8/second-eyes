# The automatic scan (A4)

Every time the app starts, the headset scans the room by itself. Meta's Space Setup first builds the room: its walls
and floor, and a box around every obstacle, chairs included. Then the on-device detector names what it sees, and
Meta's depth places each sighting: furniture seen inside one of Meta's boxes names that box, and everything else (a
laptop, a mouse, a bottle, a keyboard, furniture outside every box) becomes an object of its own, placed by depth.
Nobody labels anything. What the scan finds joins the session as a perceived scene, so typed commands can refer to it.

The project lead chose this on 17 October 2026: the automatic scan after three bedroom scans with the walking stand-in
(`scan.md`) gave no usable scene; Meta's room after the first automatic scan, taking its logic from Space Setup (every
obstacle is a piece of furniture, a wall or something else, and must be identified), with Space Setup at every start and
Meta's name winning when Meta and the detector disagree (`notes/decisions.md`). The headset scans its side of the room;
the drone will later scan the part it can't see.

Code: the core is `quest-app/Assets/SecondEyes/Perception/AutoScan/AutoScan.cs` and `RoomModel.cs` (free of Unity: the
same code runs on the headset and in the PC's replay); the Unity glue is `Perception/AutoScan/Glue/AutoScanner.cs`.
Settings: `schemas/auto-scan-config.v2.json`, defaults `perception/autoscan/autoscan.default.v2.json`, in the perception
folder (`perception-folder.md`). The bedroom's map is `perception/bedroom/category-map.bedroom.v3.json`.

## What happens

1. The scanner installs itself when the app starts and loads the perception folder. It asks for the spatial data
   permission if it isn't granted: Meta's depth and Meta's room both need it.
2. **Space Setup** (`space_setup`: `every_start`). The app asks the headset for Space Setup (`OVRScene.RequestSpaceSetup`),
   which pauses the app: you look around while it builds the room and wraps the obstacles in boxes, then finish. The
   app resumes. (`if_missing` asks only when the headset has no room; `never` skips Meta's room.)
3. **Meta's room.** MRUK (Meta XR SDK 207) loads the room the headset holds. It is made inactive, given its settings (it
   reads them when it wakes; without them its `Update` throws every frame) and its world lock turned off, so every
   position stays in the tracking space the camera's poses and the depth's points are measured in. The room becomes the
   scan's room model: the floor's outline and height, the ceiling's height, and every anchor with a volume as a box (its
   centre, rotation, size and Meta's label). No room (Space Setup cancelled, no room found, or an error): the scan goes on
   by depth alone, and the log and facts say why.
4. The detector switches to its on-request mode (it refuses while an inference runs, so the scanner waits up to 30 s
   for it), the camera's model is measured from the camera's own rays (`scan.md`), and Meta's depth starts
   (`EnvironmentDepthManager` with occlusion rendering off and hands removed, `EnvironmentRaycastManager` for rays).
5. Every `interval_s`, while the head is tracked and no command is being processed, the detector runs on the current
   camera frame. Each detection whose label the map knows, at or above `min_score`, is sampled with Meta's raycasts at
   five points of its box; the median hit by distance is the detection's point.
6. The frame, every detection and sample included, is one line of `frames.jsonl`, and `AutoScan` processes exactly that
   line. A detection is left out when its label is unmapped, its score is low, its frame was untracked or from another
   tracking epoch, no sample hit, its point is beyond `max_range_m`, below the floor, or **outside the room**: farther
   than `room_margin_m` outside the floor's outline, or below the floor or above the ceiling by more. Reflections in a
   mirror on a wall land behind the wall, so they are left out.
7. A kept detection of a `box_categories` category (bed, chair, couch, table) whose point lies inside one of Meta's
   boxes, within `box_margin_m` (measured along the box's own axes; the smallest box when they overlap), **votes for
   that box**. Every other kept detection joins the nearest object of its category within reach, or starts one, as in
   the first automatic scan: small things are placed by depth even when they stand on a box.
8. **Every box is an object.** Its name is Meta's when the map has Meta's label (`meta:BED`, `meta:TABLE` and so on):
   Meta's name wins. A box Meta calls OTHER takes the detector's leading category once it has `min_box_votes` sightings
   of it. Otherwise its category is unknown, and it is still in the scene: an obstacle that is there, not yet identified.
   A depth object is in the scene once seen in `min_frames` frames.
9. At most every `publish_interval_s`, a changed scene is published: `scene.json`, `trace.json` and `facts.json` are
   rewritten, the panel binds it to the session (replacing the scan's previous snapshot), and a label floats over each
   object in view: over a depth object's centre, over a box's top, "?" for an unnamed box.
10. Sending a typed command binds the newest snapshot first, then holds the scan until the command's outcome is shown.
    The panel's Scan button reads Pause or Resume; the right thumbstick hides the panel when it is in view and brings it
    in front otherwise.

## What the scene says

- A box's object: Meta's room (`meta_room`, a sensor capture) as its source, Meta's anchor ID as its source object, and
  as its raw label the map entry that named it (`meta:BED`, the detector's `chair`, or `meta:OTHER` when unnamed). Its
  centre is the box's centre, naming the `meta_volume` assumption. Its views count the frames whose sightings voted for
  it and their viewpoints; a box no frame saw counts Space Setup's capture once, because a scene's views can't be zero.
- **The box's true size and rotation are in the trace (`room.boxes`), not the scene**: the scene contract's perceived
  profile allows a size only from the size priors and no rotation. Changing that is a separate contract decision.
- A depth object's centre is the median of its points on the visible surface (`depth_surface`), toward the camera from
  its true centre by up to half its depth.

## The settings

| Setting | Default (v2) | What it does |
|---|---|---|
| `space_setup` | `every_start` | when the app asks for Space Setup: `every_start`, `if_missing` or `never` |
| `interval_s` | 0.5 | seconds between detector runs |
| `min_score` | 0.3 | detections below it are left out |
| `sample_spread` | 0.2 | where the four outer depth samples sit, as a fraction of the box |
| `max_range_m` | 6 | depth hits farther than this are left out |
| `floor_y_m`, `floor_margin_m` | 0, 0.15 | with no room: the floor's height, and how far below it a point is impossible |
| `room_margin_m` | 0.25 | how far outside the room's outline, floor or ceiling a point may lie |
| `box_margin_m` | 0.1 | how close to one of Meta's boxes counts as inside it |
| `box_categories` | bed, chair, couch, table | the categories whose sightings vote for boxes |
| `min_box_votes` | 2 | sightings an OTHER box needs before the detector names it |
| `min_reach_m`, `reach_factor` | 0.3, 0.5 | joining a depth object: the larger of the two, times the larger measured size |
| `min_frames` | 3 | frames a depth object needs before it is in the scene |
| `viewpoint_m` | 0.5 | camera positions closer than this are one viewpoint |
| `max_points_per_object` | 64 | the recent points a depth object's median uses |
| `publish_interval_s` | 3 | at most this often, a changed scene is published |
| `save_images_max` | 120 | frames saved as images each session; the previous session's images are deleted when a scan starts |

The defaults are first values, set by reasoning, not tuned; a change is a new `config_id`. A v1 settings file (no
`space_setup`) still works: no Meta room, no box votes.

## The files

Under `files/scans/(scan ID)/` on the headset: `frames.jsonl` (a header line, Meta's room as an `auto_scan_room` line,
then one line per frame), `scene.json` and `trace.json` (the latest snapshot; the trace gives every detection's fate,
what was left out, and every box with its votes, size and rotation), and `facts.json` (the camera's model, the counts,
whether depth is supported, the room's status). Frame images are under `files/snapshots/`. The log has
`autoscan.space_setup`, `autoscan.room`, `autoscan.start` and `autoscan.publish` events.
`grounding/quest/scan_device.py` pulls a scan (`pull`) and writes `view.html` (`view`), each box coloured by its fate.

## The replay on the PC

`dotnet run --project tools/localize -- --memory <frames.jsonl> <category-map> <size-priors> <autoscan-config> <out-dir>
[schemas] [--compare <scene.json>]` replays the headset's frames through the same code, Meta's room included. With
`--compare`, it stops at the frame count the trace beside the compared scene recorded, and the scene must be the same:
the same objects from the same sources, categories and placements, centres within 1 mm.

## Checks

`tools/csharp-tests/AutoScan` (51 checks): the settings, sampling and the pinhole, every leave-out reason, joining and
merging, the median's robustness, viewpoints, the replay byte for byte, the contract, and Meta's room: a point in a
turned box, the room's outline with its margin, rotations into the scene's axes, Meta's name winning over the
detector's, an OTHER box named by the detector, one seen once staying unnamed, a laptop on a desk staying its own object,
a reflection outside the room, and the replay with the room line. `tools/csharp-tests/ScanGlue` compiles the scanner
against stubs of the Unity, app and Meta SDK 207 members it calls (Space Setup and MRUK included), every Meta signature
copied from the SDK's source.

## Known limits

- Space Setup at every start takes the time it takes you to look around the room, before the scan begins.
- A box's size and rotation aren't in the scene yet (see above).
- An unnamed box counts as seen once by convention; its trace entry shows it had no votes.
- Labels are what Meta and the detector say: Meta's name wins even where it is wrong, and the detector can still call a
  chair seen from behind something else. Correcting a label by tapping it is for later.
- Meta's sample code is under the Oculus SDK License Agreement, not a licence D83 allows, so none of it is copied; the
  scan calls the SDK, which the app already depends on.
