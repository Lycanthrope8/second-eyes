# The automatic scan (A4)

Every time the app starts, the headset scans the room by itself, and every object it finds is a 3D box: Meta's Space
Setup builds the room and puts a box around every obstacle, the on-device detector names what it sees, Meta's depth
places each sighting, and the detector's objects get boxes fitted to the points their depth samples hit. A sighting
inside an existing box joins it, so an object seen again from another side stays one object. The headset draws every
box as a see-through block with bright edges, its name floating over it. Nobody labels anything. What the scan finds
joins the session as a perceived scene, boxes included, so typed commands can refer to it.

The project lead chose this on 17 and 18 October 2026: the automatic scan, after three bedroom scans with the walking
stand-in (`scan.md`) gave no usable scene; Meta's room, taking its logic from Space Setup (every obstacle is a piece of
furniture, a wall or something else, and must be identified), with Space Setup at every start and Meta's name winning
when Meta and the detector disagree; then 3D boxes for every object in SpatialLM's form and Meta's look, carried in the
scene (`notes/decisions.md`). The headset scans its side of the room; the drone will later scan the part it can't see.

Code: the core is `quest-app/Assets/SecondEyes/Perception/AutoScan/AutoScan.cs`, `Boxes.cs` and `RoomModel.cs` (free of
Unity: the same code runs on the headset and in the PC's replay); the Unity glue is `Perception/AutoScan/Glue/AutoScanner.cs`;
the blocks' shader is `Assets/SecondEyes/Resources/SecondEyesSeeThrough.shader`. Settings:
`schemas/auto-scan-config.v5.json`, defaults `perception/autoscan/autoscan.default.v5.json`, in the perception folder
(`perception-folder.md`). The bedroom's map is `perception/bedroom/category-map.bedroom.v4.json`. Older settings files
keep their behaviour, so older scans replay exactly: v1 has no Meta room, v2 has Meta's room but no fitted boxes, v3 has
the boxes but neither support turns nor joining groups, v4 has those but no boxes from Meta's mesh.

## What happens

1. The scanner installs itself when the app starts, loads the perception folder, and asks for the spatial data
   permission if it isn't granted: Meta's depth and Meta's room both need it.
2. **Space Setup**, once per app launch (`space_setup`: `every_start`; `if_missing` asks only when the headset has no
   room; `never` skips Meta's room). The app asks the headset for Space Setup (`OVRScene.RequestSpaceSetup`), which
   pauses the app: you look around while it builds the room and wraps the obstacles in boxes, then finish. A scan
   restarted later in the same launch reloads the room without asking again.
3. **Meta's room.** MRUK (Meta XR SDK 207) loads the room, made inactive, given its settings and its world lock turned off,
   so every position stays in the tracking space the camera's poses and the depth's points are measured in. Its boxes
   stand on their own z axis; the scan uses each upright: centre, turn about the vertical, width, depth and height.
   No room: the scan goes on by depth alone, and the log and facts say why. With `use_mesh` (v5), MRUK's global mesh,
   Space Setup's fused depth of the whole room, is read too: its points in quest_tracking, thinned to one per
   `mesh_voxel_m`, are logged with the room, so the replay uses the same geometry.
4. The detector switches to its on-request mode (waiting up to 30 s while an inference runs), and the camera starts:
   after Space Setup it can take longer, so after 30 s the detector is restarted once, and the scan stops only at 60 s.
   The camera's model is measured from its own rays (`scan.md`), and Meta's depth starts.
5. Every `interval_s`, while the head is tracked and no command is being processed, the detector runs on the current
   camera frame. Each detection whose label the map knows, at or above `min_score`, is sampled with Meta's raycasts on a
   `sample_grid` by `sample_grid` grid across its box. The median hit by distance is the detection's point; the hits
   within `depth_window_m` of it are its surface, and the rest, the wall behind it, are dropped.
6. The frame, every detection and sample included, is one line of `frames.jsonl`, processed exactly as logged. A
   detection is left out for an unmapped label, a low score, an untracked frame or another tracking epoch, no hit, a
   point beyond `max_range_m`, below the floor, or **outside the room** (farther than `room_margin_m` outside the floor's
   outline, below the floor or above the ceiling): reflections in a mirror on a wall land behind the wall.
7. **Meta's boxes first.** A sighting belongs to one of Meta's boxes when it lies inside it within `box_margin_m`, or
   rests on its top within `top_allowance_m` (a pile on the bed), and either its category is furniture
   (`box_categories`: bed, chair, couch, table) or Meta named the box as its category (the monitor inside Meta's screen).
   It votes for that box: Meta's name wins; a box Meta calls OTHER takes the detector's leading category once it has
   `min_box_votes` sightings of it, else it stays unnamed, an obstacle still in the scene.
8. **Boxes from Meta's mesh** (v5). When the room is set, the mesh is cut into obstacles: points within
   `structure_margin_m` of the floor or the ceiling, within `wall_margin_m` of a wall, outside the room, or inside one of
   Meta's boxes (already objects) are taken off; the rest is split into clusters of points within `mesh_link_m` of each
   other, and each cluster of at least `cluster_min_points`, at least `cluster_min_size_m` some way and at most
   `cluster_max_size_m` across becomes a box fitted to it: every side Space Setup saw, measured, not assumed. A box
   resting on one of Meta's boxes (a laptop on a desk) reaches down to its top, since its lowest few centimetres went
   with the desk's surface. A sighting
   whose point lies inside one, of any category, votes for it, since the cluster is the object; it takes the leading
   category with `min_box_votes` sightings. A cluster nobody names stays in the scene as an obstacle of unknown
   category, as Space Setup boxes every obstacle.
9. **The scan's boxes.** Any other sighting joins the object whose box it lies inside (furniture joins furniture,
   whatever the label, and counts as inside when resting on it), else the nearest object of its category within reach,
   else it starts one. Its surface points join the object's, thinned to one per `point_voxel_m` (the voxel doubles past
   `box_points_max`), and the object's box is fitted again:
   - **the turn** in [0°, 90°) at which the points hug the footprint's edges most closely (the closeness criterion of
     L-shape fitting, used for vehicles in laser scans), in 10° steps and then 1° steps around the best: a camera mostly
     sees an object's front and one side, an L of points, whose smallest rectangle is ambiguous (aligned with the
     object or with the L's diagonal, the same area); each side is trimmed of `box_trim` of the points at either end so
     a stray ray doesn't stretch it; the box reaches the floor for `floor_categories`;
   - **completion** (`box_completion`): a side shorter than the category's closest size prior grows to it, away from the
     viewpoints that saw the object, since the hidden side is the far one, as SpatialLM's boxes are whole; a side longer
     than `box_max_over_prior` priors (a small thing's samples landing on the desk around it) shrinks to the prior,
     centred on the object's points;
   - **resting on a support** (v4, `align_to_support`): a small thing whose box rests on another (its bottom within
     `rest_gap_m` of that box's top, its centre over its footprint), one of Meta's or a piece of furniture, takes that
     box's turn and is fitted again at it: a keyboard lines up with its desk. A thin thing's outer samples land on the
     desk around it at its own height, so its fitted turn alone can be tens of degrees off;
   - the object's category is its leading vote. Objects whose boxes contain each other's centres merge, when they may be
     one object: the same category, both furniture, or both in one of `join_groups` (v4: a bottle and a cup, the tumbler
     the detector calls both; a keyboard and a laptop, the laptop's own keyboard, which then joins the laptop).
10. An object seen in at least `min_frames` frames is in the scene. At most every `publish_interval_s`, a changed scene is
   published: `scene.json`, `trace.json` and `facts.json` are rewritten, the panel binds it to the session, and every
   object is drawn: a see-through block (Meta's boxes blue, the scan's green, unnamed grey) with bright edges, its name
   over its top.
11. Sending a typed command binds the newest snapshot first and, once a room is scanned, goes to the room: if a dataset
    scene is showing, the panel switches back to the room and says so (the Scene button and presets are for the dataset
    scenes). The scan holds until the outcome is shown. The room's list gives each object's name ("?" for an unnamed
    box) and its box. The panel's
    Scan button reads Pause or Resume; the right thumbstick hides the panel when it is in view and brings it in front
    otherwise.

## What the scene says

Every object carries its 3D box (v3): its centre, its size (along its own width, depth, then height) and its turn about
+z (`yaw_only`), all `inferred` from a sensor capture: the scan's for fitted boxes, naming the `box_centre` assumption
and, when a side was completed, `box_completion`; Meta's room for Meta's boxes, naming `meta_volume`. The scene contract
allows this since 18 October 2026 (`scene-contract.md`). A box of Meta's carries as its raw label the map entry that
named it (`meta:BED`, the detector's `chair`, or `meta:OTHER` unnamed); its views count the frames that voted for it, and
one no frame saw counts Space Setup's capture once, because a scene's views can't be zero.

## The settings (v5)

| Setting | Default | What it does |
|---|---|---|
| `space_setup` | `every_start` | when Space Setup runs: once per app launch, only when no room, or never |
| `interval_s`, `min_score` | 0.5, 0.3 | seconds between detector runs; detections below the score are left out |
| `sample_grid`, `sample_spread` | 5, 0.2 | the depth grid in each detection's box (`sample_spread` is the older five-point pattern) |
| `depth_window_m` | 0.5 | a detection's hits this far beyond its median are the background |
| `max_range_m`, `floor_y_m`, `floor_margin_m` | 6, 0, 0.15 | farthest depth kept; with no room, the floor and the margin below it |
| `room_margin_m`, `box_margin_m`, `top_allowance_m` | 0.25, 0.1, 0.5 | outside the room; inside a box; resting on a box's top |
| `box_categories`, `min_box_votes` | bed, chair, couch, table; 2 | furniture; votes an OTHER box needs |
| `point_voxel_m`, `box_points_max`, `box_trim` | 0.03, 800, 0.03 | surface points kept per voxel; their cap; the share trimmed per side |
| `box_completion`, `box_max_over_prior`, `floor_categories` | true, 1.5; bed, chair, couch, table | completing to priors; shrinking far-too-long sides; boxes reaching the floor |
| `min_reach_m`, `reach_factor`, `viewpoint_m` | 0.3, 0.5, 0.5 | joining by distance when no box contains a sighting; one viewpoint |
| `min_frames`, `max_points_per_object`, `publish_interval_s` | 3, 64, 3 | frames before an object is in the scene; the older median's points; publishing |
| `save_images_max` | 120 | frames saved as images each session; the previous session's are deleted when a scan starts |
| `align_to_support`, `rest_gap_m` (v4) | bottle, cup, keyboard, laptop, mouse; 0.08 | small things that take their support's turn; how near counts as resting |
| `join_groups` (v4) | bottle and cup; keyboard and laptop | categories that may be one object |
| `use_mesh`, `mesh_voxel_m` (v5) | true, 0.03 | boxes from Meta's room mesh; the mesh thinned to one point per voxel |
| `structure_margin_m`, `wall_margin_m` (v5) | 0.05, 0.08 | mesh points this close to floor or ceiling or Meta's boxes, or to a wall, taken off |
| `mesh_link_m`, `cluster_min_points` (v5) | 0.06, 20 | points this close are one cluster; the points a cluster needs |
| `cluster_min_size_m`, `cluster_max_size_m` (v5) | 0.15, 3 | a cluster smaller every way is noise; wider, leftover structure |

The defaults are first values, set by reasoning, not tuned; a change is a new `config_id`.

## The files

Under `files/scans/(scan ID)/` on the headset: `frames.jsonl` (a header line, Meta's room as an `auto_scan_room` line,
then one line per frame), `scene.json` and `trace.json` (the latest snapshot; the trace gives every detection's fate and
every one of Meta's boxes with its votes, size and turn), and `facts.json` (the camera's model, the counts, the room's
status, whether boxes are drawn and the see-through shader loaded). Frame images are under `files/snapshots/`. The log
has `autoscan.space_setup`, `autoscan.room`, `autoscan.camera_retry`, `autoscan.start` and `autoscan.publish` events.
`grounding/quest/scan_device.py` pulls a scan (`pull`) and writes `view.html` (`view`), each box coloured by its fate.

## The replay on the PC

`dotnet run --project tools/localize -- --memory <frames.jsonl> <category-map> <size-priors> <autoscan-config> <out-dir>
[schemas] [--compare <scene.json>]` replays the headset's frames through the same code, Meta's room and the boxes
included. With `--compare`, it stops at the frame count the trace beside the compared scene recorded, and the scene must
be the same: the same objects from the same sources, categories and placements, centres within 1 mm.

## Checks

`tools/csharp-tests/AutoScan` (84 checks) covers the settings, sampling, every leave-out reason, joining and merging, the
replay byte for byte and the contract; Meta's room (a point in a turned box, the room's outline, rotations into the
scene's axes, Meta's name winning, OTHER boxes named or unnamed, a laptop on a desk, a reflection outside the room); and
the 3D boxes: the grid, a turned box fitted within a degree with stray rays trimmed, an L of points fitted as the object and not
along its diagonal, completion away from the viewer, a
stretched small thing shrunk, one of Meta's boxes standing on its z axis, a side view joining a chair's box (fitted at the chair's own turn), the monitor
going into Meta's screen box, a pile on the mattress counting as the bed, boxes in a scene that passes the contract, and
the replay; and v4: a keyboard whose outer samples hit the desk taking the desk's turn, a tumbler called a cup and a
bottle staying one cup, a laptop absorbing its own keyboard while the real keyboard stays, and the replay; and v5, on a
synthetic room mesh: the floor, walls, Meta's desk and a speck of noise taken off, a chair's box measured from the mesh
at its own size and turn (not its prior's), a laptop on the desk its own mesh box, a bin nobody named kept as an
obstacle, a mouse flattened into the desk keeping its live-depth box, the contract and the replay, the mesh included. `tools/csharp-tests/ScanGlue` compiles the scanner against stubs of the Unity, app and Meta SDK 207 members it
calls, every Meta signature copied from the SDK's source.

## Known limits

- How fine Meta's mesh is on the headset is unknown until the first scan with v5: small things (a mouse, a keyboard)
  probably flatten into the desk in it and keep their live-depth boxes, and things touching each other in the mesh (a
  chair against the bed) are one cluster, so one box. The mesh is Space Setup's: something moved after it isn't in it.

- A box is fitted to what was seen: until an object is seen from more sides, its hidden side comes from its size prior,
  and a category without one (the bed, the tables, a cup) keeps the visible extent: the cup's box takes in some of the desk
  around it.
- Meta's depth is coarse at a mouse's scale, so small things' boxes are approximate.
- A chair pushed fully under a desk's box is absorbed into the desk. Meta's name wins even where it is wrong, and the
  detector can still mislabel; the laptop's own keyboard counts as a keyboard.
- The see-through shader is shipped in a Resources folder so the build includes it; if it ever fails to load, the boxes
  aren't drawn and `facts.json` says `see_through_shader: false`. The scene's boxes don't depend on it.
- Meta's sample code is under the Oculus SDK License Agreement, not a licence D83 allows, and SpatialLM's released weights
  are non-commercial: neither is copied. The scan calls Meta's SDK, and its boxes take only SpatialLM's form.
