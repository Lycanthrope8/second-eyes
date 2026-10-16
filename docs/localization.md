# The localization (A4)

How a scan becomes a perceived scene: the C# core in `quest-app/Assets/SecondEyes/Perception/Localization/`, Unity-free,
so the same code runs on the Quest and in the PC's rerun. Its inputs are a checked capture record (`capture-record.md`),
the category map, the size priors (`schemas/size-priors.v1.json`) and a configuration
(`schemas/localization-config.v1.json`; the default is `perception/localization/localization.default.v1.json`). Its
output is a perceived scene (`scene-contract.md`, format v3) and a trace of what it did, for reading and debugging; the
trace is not a contract record.

Rerun on the PC, from the repository root:

    dotnet run --project tools/localize -- <capture.json> <category-map.json> <size-priors.json> <config.json> <out-dir>

It checks every input first (the capture with its check, the priors and the configuration against their schemas, the
map against the contract), notes when the capture used a different map or priors than the rerun, writes
`scene.json` and `trace.json`, checks the scene against the contract, and lists the objects.

## The steps

1. **Rays.** Each detection whose label the category map knows, at or above `min_score`, becomes a ray in the scene
   frame: from the camera's position at capture through the box's centre, by the pinhole model and the capture pose
   (`frames.md`). A box within `edge_margin_px` of the image's edge is left out and counted: its centre and height are
   cut, and the scan's 30-degree steps usually show the same object whole in the next view.
2. **Distances from size.** An object whose category has an upright size prior gets a distance interval along its ray.
   At each candidate distance, the prior-sized box is projected through the actual camera pose, the camera's height and
   tilt included, and the distance is solved where the projected height meets the observed one. The object's yaw is
   unknown, so this is solved at six yaws; the interval is tight for a chair-like box and wide for a long, low one such
   as a bed. A category may have several sizes (the bedroom's two chairs, 88 and 115 cm tall): its interval is the union
   of theirs.
3. **Sightings.** At each standing position, rays of one category within `same_position_angle_deg` of each other whose
   intervals agree merge into one sighting. Turning in place adds no baseline, so these rays can't triangulate.
4. **Tracks across positions.** Position by position, sightings join tracks by minimum-cost assignment (the Hungarian
   algorithm). A link is valid only if the rays pass within `gap_max_m`, at least `min_parallax_deg` apart, in front of
   both cameras and within `max_range_m`, and if each distance falls inside its interval, widened by
   `range_ratio_tolerance`. This last test removes the ghost crossings that identical objects produce.
5. **Uncertain associations stay apart.** A link with a rival within `ambiguity_ratio` of its cost is not made: both
   stay separate objects, marked possible duplicates of each other, as is any near-tie left unmade. The resolver's
   association guard (`resolution.md`) then keeps a command from silently picking one of them.
6. **Centres.** A track seen from two or more positions gets the least-squares point nearest its rays; a ray farther
   than `gap_max_m` from it is split off as its own object, marked a possible duplicate (the third-view check). A track
   seen from one position gets the middle of its distance interval, or no centre without a size prior.

There is no prior on how many objects of a kind exist. Detections are taken in the order of their boxes, so the output
and the object IDs depend only on what was seen, not on the order the detector listed it. Numbers are written rounded
to 0.1 mm, so the PC and the headset write the same text.

## What the scene says

| Field | Evidence |
|---|---|
| category | inferred, from the scan, through the category map |
| centre, from two or more positions | inferred, from the scan, with no assumption |
| centre, from one position | inferred, from the scan, under the `size_prior` and `upright` assumptions |
| centre, from one position without a size prior | unknown (`not_estimated`) |
| size | assumed, from the size priors (`size_prior`), when the category has exactly one size; unknown (`no_permitted_prior`) without a prior, or with several, since no single size applies |
| rotation, semantic front, colours, geometry uncertainty | unknown (`not_estimated`) |
| capture time | measured: the latest keyframe that saw it, on the headset's monotonic clock |
| detector confidence | inferred: its best detection score |
| views and association | keyframes and positions it was seen in; `multi_position`, `single_position` or `possible_duplicate` with the objects it may be |

## The thresholds

The defaults are A4's first values, set by reasoning, not tuned; a change is a new `config_id`, made only after the
usability check shows a need.

| Threshold | Default | What it does |
|---|---|---|
| `min_score` | 0.3 | detections below it are left out (the detector's own threshold) |
| `edge_margin_px` | 4 | a box this close to the image's edge is cut and left out |
| `same_position_angle_deg` | 6 | rays at one position this close merge into one sighting |
| `range_ratio_tolerance` | 0.15 | how far outside its size interval a distance may fall |
| `gap_max_m` | 0.25 | how closely two rays, or a ray and a centre, must pass |
| `min_parallax_deg` | 4 | the smallest angle between two rays that triangulates |
| `ambiguity_ratio` | 1.5 | a rival within this factor of a link's cost makes it uncertain |
| `max_range_m` | 8 | nothing farther is placed |

## Checks

`tools/csharp-tests/Localization` runs the core on synthetic scans with exact ground truth. The tolerances come from the
geometry: a box's image centre drifts from its 3D centre, more for large boxes, so a triangulated chair must lie within
0.15 m of its box and a table or bed within 0.30 m; a size-based centre must lie within a quarter of its distance.

1. Three objects seen from two positions: one object per box, each from both positions, placed within tolerance.
2. Two identical chairs whose rays cross: the size intervals reject the ghost crossings, leaving exactly two chairs.
3. The same chairs without a chair prior: no ghost passes as a confident object; what isn't confident says so.
4. Turning in place: every object is `single_position`, and each centre names the `size_prior` and `upright` assumptions.
5. An object only ever cut by the image's edge is left out and counted, not guessed.
6. Listing the detections in reverse changes nothing, text for text.
7. Unmapped labels and low scores are left out and counted, and the scene is unchanged.
8. The capture fixture runs end to end.
9. The bedroom's two chairs, 88 and 115 cm tall, with both sizes in the priors: from two positions both are placed
   within 0.15 m and neither claims a size; from one position each centre comes from both sizes, within a quarter of
   its distance. The bedroom's size priors and category map pass their schema and the contract.

Every synthetic scan is built by `CaptureBuilder`, as the headset builds one.

Every scene must pass the contract (scene v3) with its category map, and every synthetic scan must pass the capture
check. Disabling the distance test makes case 2 fail, which shows the cases bite.

## Known limits

- Lens distortion is not corrected: a capture whose distortion is `unknown` is localized as if it had none. Whether
  the Quest's camera images need correction is checked when capture is built on the headset (stage 3).
- The distance from size assumes an upright box of the prior's size. A long, low object gets a wide interval, so its
  links are gated more loosely.
- An object seen only cut by the image's edge is missed rather than placed from a cut box.
- A pose error large enough to break every link makes a second object from the same thing, not marked a possible
  duplicate: with no valid link there is no near-tie to mark. A usability check would show it.
- The third-view split is a safety net that none of the synthetic scans triggers.
- An object of a category with several sizes, seen from one position, gets a wider distance interval: its size can't
  say which of the sizes it has. Scanning from a second position places it without sizes.
