# A4: closing note (19 October 2026)

**The closure** (Astra's review of A4 Delivery 1, its condition met by the correction delivery and the project lead's
acceptance in stages 10 and 11):

> A4 implements a Quest-based room-memory and typed-grounding prototype using Meta scene geometry, depth and detector
> observations. The project lead accepted its usability within the documented scope. Drone perception transfer and
> formal Gate C evaluation remain pending.

The project lead accepted stage 11 on 19 October 2026 ("The recenter was good"), after stage 10's acceptance of the other
cases. No code changed after the accepted build (`3944dbb`); this closing adds records only.

## The scope

The 15 October planning report made A4 a usability milestone: Gate C deferred to a final evaluation, one shared C#
localizer with PC reruns. Three bedroom scans with the camera-only stand-in (walking to scan spots, ranging by known size,
triangulating) then gave no usable scene, and the project lead redirected A4 to the headset's own scan (D123 to D125).
The camera-only code stays, for the drone's path.

## What A4 delivered

- **The automatic scan** (`docs/auto-scan.md`): at every app start the headset scans by itself. Meta's platform
  perception services supply Space Setup's boxes (Meta's names win, D124) and room mesh, and the depth API; the app's
  detector names what it sees, and each object is a 3D box in SpatialLM's form (D125): Meta's box, a cluster of Meta's
  mesh named by the detector (D127), or a box fitted to live depth, with a size prior only for sides nothing measured. A
  box exists once it is detected (D128); the scene says it lists only recognized objects.
- **The scene contract** (`docs/scene-contract.md`): the perceived profile carries sensor-inferred sizes and turns, every
  assumption stated on the field it shaped; every scene valid before stays valid.
- **Typed commands on the scanned room** (D126), held apart from the scan's work; on the headset, see-through blocks.
- **Tracking**: MRUK's world lock holds the room through a recenter (D129); a real room shift makes the room stale, refuses
  commands and starts a fresh scan; frames in another world are dropped and counted.
- **The observation-log replay**: every scan logs its inputs after detection and depth sampling; the PC tool rebuilds the
  scene and compares the complete scene, every field.
- **The perception folder** and its checks, versioned settings (`autoscan.default.v1` to `v8`), maps (`v2` to `v5`) and
  size tables (`v1`, `v2`, D130).

## The acceptance

`docs/a4-usability.md`, "The acceptance": a normal command, an ambiguous one, two touching bottles, a turned keyboard, the
mouse, and a recenter, all accepted (the keyboard's turn as documented by D126). An engineering and usability check, not
Gate C.

## Measured, not judged

The bedroom trials, each scan replayed on the PC to the same scene:

| Stage | Settings | Objects | What it showed |
|---|---|---|---|
| 4 | v1 | 19 | the automatic scan works; many duplicates |
| 5 | v2 | 14 | Meta's boxes as objects; mirror reflections excluded |
| 6 | v3 | 14 | 3D boxes drawn; commands went to the wrong scene |
| 7 | v4 | 12 | commands on the room; small things on their desk's turn |
| 8 | v5 | 17 | Meta's mesh coarse: chairs in fragments; unnamed clusters held every command |
| 9 | v6 | 10 | every object named; every command answered |
| 10 | v7 | 12 | the acceptance cases; a recenter turned the room and the fresh scan never recovered |
| 11 | v8 | 11 | the recenter held (world lock); four commands answered |

Nothing is measured against ground truth. The bedroom is the development set: its trials shaped the settings (the 15 cm
mesh link is a development-set adjustment) and can't also be their evaluation.

## Known issues

- Small things (bottle, keyboard, mouse) are too small for Meta's mesh: their unseen sides come from the bedroom's size
  table, a development prior, and their turn from their desk (D126). The trace records each box as measured and as
  written.
- The scan depends on Meta's closed services (Space Setup, MRUK, the depth API, the global mesh): Quest only, and the
  logs keep the inputs to this app's pipeline, not Meta's reconstruction. The mesh is Space Setup's snapshot.
- The stock COCO detector names only its categories, and a box needs two sightings.
- The parser still misreads some inputs (stage 7: "hello" read as a command about a bottle): parser correctness, resolver
  correctness given the query, and physical target correctness are separate questions.
- The workload (depth, mesh, overlay, about 2 Hz scanning) differs from A1's accepted one; A1's runtime figures don't
  carry over.

## Identities

- Base: `833ef8d`. Accepted build: `3944dbb` (checkpoint `a4-delivery1-checkpoint-3944dbb.zip`, SHA-256
  `ae2069b7fe58cef522d625eba414ebfedf7ccfc2483a2b65c8878916c894dd7f`), the stage 11 APK SHA-256
  `78a5dcf5c6b7167d6151f1c9dd813425eefb6ef0e289ba3250c485a77e13d179`.
- The perception folder: 22 files, 153793 bytes, manifest SHA-256
  `3a4ce1a888fe59a0ce9a2a51ece03cab9d3b7527fd9f956b3982debcc6067fb5` (`autoscan.default.v8`, `map.bedroom.v5`,
  `sizes.bedroom.v2`).
- The accepted scans: `scan.468bfd88.1` and `.2` (stage 10), `scan.d421455f.1` (stage 11).
- Decisions D123 to D130 (`notes/decisions.md`); the failures of every stage and of Astra's review in
  `notes/failures.md`. The stage 9 version is kept on the tag `a4-d1-stage9-accepted` in the sandbox, with its zip.

## What this establishes, and what it doesn't

It establishes that the Quest can go from its own sensing, through a structured scene, to answered typed commands in one
room, accepted by the project lead for usability, and that every scan rebuilds on the PC to the complete, identical scene.

It doesn't establish that a drone can build the hidden part of the room's memory, that camera-only localization works or
fails in general (the stand-in's failures are this implementation's), how accurate the scene is against ground truth, or
that the settings carry beyond the bedroom.

## Next

- **Gate C** at the final evaluation: one framework, with results reported separately for each sensing path claimed
  (headset, drone, and their combination), against Vicon or a surveyed layout; its statistics, object matching and the
  treatment of misses, duplicates and unknowns settled before it is run.
- **The drone-side items** move to the drone phases: the drone camera's calibration, drone-height recordings, detector
  fine-tuning only if the intended objects need it, a video replay later. A second, Python localizer is not required.
- **The hidden-region design**: keep each observation's source and the region it covers, so that the headset-only
  condition doesn't already hold what the drone is meant to reveal.
- **The runtime** under the new workload is measured before it is relied on.
