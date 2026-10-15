# Frames, transforms and units

How poses and positions are written, from A4 on (O3). The scene contract's own frame rules (`scene-contract.md`, S07)
are unchanged; this file names the frames around it and the one place where handedness changes.

## Units

Metres, radians in data (degrees only in names that say so, such as `yaw_deg`), quaternions as `x, y, z, w` with unit
length, and times in the log's clocks (`mono_us`, `utc_us`; `logging.md`). Units are part of field names
(`translation_m`, `rotation_xyzw`), as everywhere else in the project.

## Frames

| Frame | What it is | Axes |
|---|---|---|
| `quest_tracking` | the SDK's world space on the headset (Unity's world) | left-handed: x right, y up (gravity), z forward; metres |
| `camera:<id>` | a camera's optical frame, origin at its optical centre | right-handed: x right, y down, z forward (OpenCV's) |
| `scene` | the frame of scene records and of the semantic memory | right-handed, z up (gravity), metres (A2.1's, S07) |
| `pad`, `drone_body` | the drone's take-off pad and body | named now; defined in A6 |
| `vicon` | Vicon's world | evaluation only, never in an operational record; defined when Vicon is used (D10) |

`quest_tracking` has an **epoch**: an integer that starts at 0 when the app starts and goes up by one at every recenter
or tracking loss, since either can move the SDK's world. Anything computed in one epoch is valid only in that epoch.
A1.10c's logged camera poses (`detector.result`'s `pose`) are in `quest_tracking`, written before epochs existed.

In A4, `scene` is `quest_tracking` re-expressed through the basis change below: the same origin, the same epoch, with z up
and right-handed axes. The scan and the commands share one tracking session, so no other alignment is needed. A6
registers the drone's `pad` into this frame.

## Transforms

`T_dest_src` maps coordinates in frame `src` into frame `dest`: `x_dest = R x_src + t`. A stored transform names both
frames, the epoch where one applies, `translation_m` (t) and `rotation_xyzw` (R as a unit quaternion):

```json
{"dest": "scene", "src": "camera:quest_left", "epoch": 3,
 "translation_m": [0.12, -0.40, 1.55], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]}
```

Composition needs matching inner frames and the same epoch: `T_a_c = T_a_b · T_b_c`. The inverse swaps the names:
`T_src_dest = (Rᵀ, −Rᵀ t)`. Every `R` here is a proper rotation (determinant +1).

## The basis change: the one place where handedness changes

Unity gives a camera pose in `quest_tracking` as a position `p` and a quaternion `q`, meaning
`x_world = R(q) x_local + p` in Unity's own axes, where the camera's local axes are x right, y up, z forward. Two fixed
matrices carry this into the project's right-handed frames:

- `B = [[1,0,0],[0,0,1],[0,1,0]]` re-expresses a `quest_tracking` vector in `scene` (it swaps y and z);
- `F = diag(1, −1, 1)` re-expresses an OpenCV optical-frame vector in Unity's camera axes (it flips y).

So a camera's pose becomes `T_scene_camera = (B R(q) F, B p)`. `B` and `F` each have determinant −1, so their product
with `R(q)` is a proper rotation, which is converted back to a quaternion only after the matrix product. Quaternion
components are never swapped by hand. `Perception/Frames/UnityBasis.cs` is the only code that applies `B` and `F`;
tools/csharp-tests/Frames checks it on poses whose results are worked out by hand (translation, rotation, inverses,
composition and camera-ray projection).

## Pixels and rays

A camera's intrinsics are `fx, fy, cx, cy` in pixels for a stated resolution, following OpenCV: `u` runs right and `v`
down, and the top-left pixel's centre is `(0, 0)`. A point `(x, y, z)` in `camera:<id>` with `z > 0` projects to
`u = fx·x/z + cx`, `v = fy·y/z + cy`; a pixel's ray is the unit vector along `((u − cx)/fx, (v − cy)/fy, 1)`. How a
detector box's coordinates map onto these pixels (including any half-pixel offset) is stated by the localization code
that takes the rays. Lens distortion, where a camera has any, is stated with its calibration and removed before this
model applies.
