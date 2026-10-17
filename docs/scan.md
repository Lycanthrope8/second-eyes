# The stand-in scan (A4)

**No longer installed on the headset.** On 17 October 2026 the automatic scan (`auto-scan.md`) replaced it: the
detector names objects and Meta's depth places them, with no spots. This scan stays for the drone's camera-only scan
of the part of the room the headset can't see, since the drone has no depth sensor.

How the headset decides, on its own, when to take a keyframe during a scan (decision A of stage 3: automatic, so no
step can be forgotten or doubled). The rule is `quest-app/Assets/SecondEyes/Perception/Scan/KeyframeRule.cs`,
Unity-free; its settings are `schemas/scan-config.v1.json`, and the defaults are
`perception/scan/scan.default.v1.json`. Each keyframe it asks for becomes one detector inference on one camera frame,
copied with its own timestamp and pose, and one keyframe of the capture record (`capture-record.md`).

## The rule

Fed the head's pose once per rendered frame (position in `quest_tracking`, y up; yaw in degrees, 0 facing +z,
positive turning right; pitch in degrees, negative looking down), the rule asks for a keyframe:

- only while tracking holds, no inference is running, and the head has held still for `steady_s`: over that time
  its yaw varied by at most `steady_deg` (largest minus smallest) and its position stayed within `steady_m` of where
  it is now, so the image is sharp and the pose settled;
- the first as soon as the head is still;
- at the same spot, when the head faces a direction at least `step_deg - step_margin_deg` from every keyframe already
  taken there, so turning back adds nothing;
- at a new spot, when the head has moved `new_position_m` or more across the floor from the spot's first keyframe;
  that keyframe starts the next position.

With `down_pitch_deg` set (`scan.default.v3`), each spot has two rows of directions: level, and looking down by at
least that much. A direction is new when it is new in its own row, so looking down at a desk after a level view in the
same direction takes a keyframe; only the level row decides when a spot is done, and the head's pitch must hold still
too. A keyframe that couldn't be taken (the camera not delivering, say) leaves the rule as it was. The rule also gives the
panel a hint: hold still, turn about 30 degrees, move to a new spot, or the scan is complete.

## The settings

The defaults are A4's first values, set by reasoning, not tuned; a change is a new `config_id`. `scan.default.v2`
differs from v1 in one setting: a new spot counts from 1.5 m (`new_position_m`), after the first bedroom scan's two
spots were only about 1 m apart; the guidance asks for 2 to 3 m. The perception folder carries `scan.default.v3`: v2
plus a look-down row from 20 degrees (`down_pitch_deg`), after the second bedroom scan lost every desk object at the
bottom of the picture. The checks are worked out on v1, with v2 and v3 checked for what they change.

| Setting | Default | What it does |
|---|---|---|
| `step_deg` | 30 | the turn between keyframes at one spot (the capture record's nominal step) |
| `step_margin_deg` | 5 | how much short of a full step still counts as a new direction |
| `steady_s` | 0.5 | how long the head must hold still |
| `steady_deg` | 3 | how much its yaw may vary while still |
| `steady_m` | 0.05 | how far it may drift while still |
| `new_position_m` | 1.0 | how far across the floor makes a new spot |
| `max_keyframes_per_position` | 12 | keyframes at one spot at most (a full circle) |
| `max_positions` | 4 | spots in one scan at most |
| `down_pitch_deg` | none (v3: 20) | optional: how far down a view must look to join the look-down row; without it, one row |

## Checks

`tools/csharp-tests/Scan` runs the rule on simulated head movements at 72 Hz, holds, turns and walks, with
expectations worked out by hand:

1. A full circle in 30-degree turns gives twelve keyframes at one spot, each facing its own direction.
2. Turning back to the first direction adds nothing.
3. Turning back and forth between two directions gives two keyframes.
4. A slow continuous turn is never still.
5. A tremor of 0.5 degrees still counts as holding still; a sway of 2.5 degrees does not.
6. Walking 1.2 m and holding still starts a new position; walking 0.6 m does not.
7. A keyframe that comes due while the detector is busy waits, then comes once.
8. Nothing is taken while tracking is lost.
9. Directions are measured correctly across the +-180 degree seam.
10. Recording a keyframe that wasn't due is refused.
11. and 12. The per-position and per-scan limits hold, and the hint then says to move or to finish.

Ignoring a busy detector makes case 7 fail, which shows the cases bite.

## On the headset

The scan's logic is `Perception/Scan/ScanSession.cs`, free of Unity and checked in `tools/csharp-tests/Localization`
(case 10); `Perception/Scan/Glue/RoomScanner.cs` is its Unity side, compiled in `tools/csharp-tests/ScanGlue` against
stubs. Pressing **Scan** (`quest-panel.md`) with a session and a perception folder (`perception-folder.md`):

1. switches the detector into its on-request mode (`External`), waits until the camera and detector are ready, and
   measures the camera's model in the texture's pixels: it asks the camera component for its own rays at a 3x3 grid of
   viewport points (`ViewportPointToRay`), turns each into the camera's axes through the pose at the same moment, and
   fits a pinhole to them (`CameraFit`). The fit is used when it agrees with its rays to within 2 pixels; otherwise the
   reported intrinsics are carried from the sensor's pixels to the texture's by the centre-crop model of Meta's sample
   code. `facts.json` keeps the reported, cropped and fitted models, which was used, and the form of
   `ViewportPointToRay` it called, or every form it found and why none could be used (`viewport_rays`). A form whose
   other parameters are the camera's pose (given the pose read with the rays) or optional is used;
2. each frame, feeds the head's pose and tracking state to the rule; when it asks, starts one inference, which copies
   the camera's frame with its own timestamp and pose;
3. keeps each result through the capture builder, numbering positions and steps over the keyframes kept, and leaving
   out any inference that failed or any frame not tracked at capture;
4. stops with an error if tracking is reset (a new tracking epoch) or the timestamp changes kind;
5. at **Finish**: checks the capture, runs the localization, checks the scene against the contract, writes
   `facts.json`, `capture.json`, `scene.json` and `trace.json` under `files/scans/<capture ID>/`, puts the detector back
   as it found it, and binds the scanned room to the session as its current scene.

A keyframe's image (the model's 416-pixel input) is saved only when the file `files/scans/save-images` exists on the
headset: images of a room can be private, so saving is off unless chosen.

**Guidance in view.** While a scan runs, the head-locked overlay (A1's text with the frame rate) adds the scan's
guidance below it (`ScanGuide`), refreshed about ten times a second: a hint in words (hold still, turn about 30 degrees
and hold still, walk 2 to 3 m to another side of the room), the spot and keyframe counts, a bar of the 30-degree
directions already taken at this spot, and how to finish (the right thumbstick brings the panel; press Finish). Each
keyframe gives a short buzz on the right controller. With the look-down row, the bar has two rows, `level [...]` and
`down  [...]`, counted from the spot's first keyframe, and a line says to look down at desks and tables too and to skip
walls right beside you (15 of the second bedroom scan's 35 keyframes saw only a wall).

**Mirrors.** A mirror shows the room again, and a reflection is geometrically consistent: the localization places it
as an object behind the mirror (the second bedroom scan's two extra chairs). Cover mirrors while scanning. Later, the
headset's room layout (Space Setup) could leave out objects beyond the room's walls.

## On the PC

    python -m grounding.quest.scan_device pull --out <folder> [--capture <capture ID>]
    python -m grounding.quest.scan_device view --scan <pulled scan folder>
    dotnet run --project tools/localize -- <capture.json> <map> <priors> <config> <out> schemas --compare <scene.json>

`pull` copies a scan (unless named, the one the headset changed last: a capture ID's middle part is a session code,
not a time, so names can't say which is newest) and its keyframe images; `view` writes `view.html`, each keyframe's
image with its boxes drawn as the model saw them; the rerun with `--compare` must find the same objects, categories,
placements and possible duplicates as the headset, with centres within 1 mm.

## What the first bedroom scan showed (16 October 2026)

The scan worked end to end: the capture passed its check, the scene the contract, and the PC rerun matched the
headset's scene within 1 mm. The camera's timestamps are `DateTime` (100 ns ticks); every kept frame's age stayed
within 16 ms of the first; the headset reports its tracking state; the camera rig sits at the origin, so the tracking
frame is the scene's. The camera reports intrinsics for a 1280x1280 sensor while its texture is 1280x960; the first
build scaled them by 0.75 vertically, which is wrong if the texture is a centre crop (`notes/failures.md`), hence the
measurement above. Every object was placed from its size: keyframes came 50 to 90 degrees apart, as the hint was on
the panel, and the two spots were about 1 m apart, so no object was seen from both with a usable angle. The guidance in
view and `scan.default.v2` answer that.

## What the second bedroom scan showed (17 October 2026)

With the guidance in view, the scan came from 4 spots 2.1 to 3.6 m apart, and kept all 35 keyframes. Three objects
were placed from two spots, their rays meeting within 1 to 16 cm; the PC rerun matched the headset's scene within 1 mm.
The geometry confirms the centre-crop camera model: it puts the bed, a mattress on the floor, at 0.05 m and the monitor
at 1.13 m, where the first build's scaled model would put the bed below the floor. The viewport-ray fit didn't run: the
camera's `ViewportPointToRay` wasn't the one-parameter form the code accepted (`notes/failures.md`).

Of 40 detections, 24 were left out: 10 with labels the map doesn't use (people, a couch that was the bed, cups, an
oven that was the black desk), 6 below 0.3, and 8 cut by the picture's edge. These included all five desk detections
from the first spot (the laptop, its keyboard, the monitor read as a laptop): the camera sees 29 degrees below its
centre, the scan looked 6 to 13 degrees down, and the rule ignored pitch, so looking down never took a keyframe. The
scene had 9 objects for 4 real ones: three beds (two unplaced, seen from one spot each), two chairs that were
reflections in the closet mirror, and the laptop's screen read as a TV, placed 6 m away by the TV's size.

The answers: the look-down row (`scan.default.v3`), the ray step (`localization.md`, `localization.default.v2`, which
on this scan joins the two extra beds to the bed and marks the TV and the laptop screen as possible duplicates),
covered mirrors, each detection's fate in the trace and the scan view, and the measurement's method form in the facts.

## What the first scans must show

`facts.json` records what can only be learnt on the Quest, and the stage 3 trial checks it:

- the camera timestamp's kind and unit, and each kept frame's age against the first (it should stay within tens of
  milliseconds);
- the intrinsics as read, and from where, with the sensor's and the texture's sizes;
- the camera component's API, and whether the headset reports its tracking state;
- the tracking space's pose relative to the scene (the camera rig's), which should be the identity for the tracking
  frame to be the scene's;
- in `view.html`, whether the boxes sit on the objects and the images are upright (A1's `flipCamera` setting).

How still a real head holds, and so whether `steady_deg` and `steady_m` suit a person, shows in how readily keyframes
come during the trial.
