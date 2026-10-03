# Directional relations, v1

How `grounding/relations/directions.py` decides left, right, in front of and behind (A2.1c, D68). The rules are the
approved A2.1c proposal as made concrete in ChatGPT's implementation brief of 2 October 2026. The caller selects the
frame. This increment includes no language parsing, automatic frame choice, frame-plausibility policy, candidate
filtering, ASK decision, resolver, serializer, dataset adapter, training, Unity or C# code, or headset execution.

## Interface

```
from grounding.relations.directions import DirectionalRelations
d = DirectionalRelations(scene_record, config=None)    # config: a DirectionConfig; default directions.v1.json
d.evaluate(relation, target_id, *, frame, anchor_id=None, command=None)   # -> DirectionResult
```

Relations: `right`, `left`, `in_front_of`, `behind`. Frames: `user_heading` (a command, no anchor),
`user_to_anchor` (a command and an anchor), `object_intrinsic` (an anchor; a command is optional and, if given, only
checked). Scene and command records must have passed the contract validator; nothing is schema-validated per call.

Raised as `ValueError`, never turned into UNKNOWN:
- unsupported relation or frame names;
- a missing or extra anchor or command;
- an object ID that isn't in the scene, including a repeated one;
- a command whose scene ID or revision differs from the scene's, or whose pose frame differs from the scene frame.

In the two anchor frames, a target equal to its anchor gives FALSE (`not_distinct`), after these checks and before
any evidence is read. These Python names are implementation interfaces, not a wire or model-facing format.

## Frames

All positions are first projected onto XY, in the right-handed, +Z-up scene frame, in metres. T is the target's
centre, A the anchor's, U the user's position.

| Frame | Needs | Construction |
|---|---|---|
| user_heading | T, U and the stored `heading_xy` H | h = H / norm(H); origin U; front = h; right = (h.y, −h.x); d = T − U |
| user_to_anchor | T, A and U; heading and user rotation are not read | w = A − U, L = norm(w); UNKNOWN if L ≤ `viewer_anchor_min_horizontal_m`; v = w / L; origin A; front = −v (toward the user); right = (v.y, −v.x), the right of a viewer looking at the anchor; d = T − A |
| object_intrinsic | T, A and the anchor's stored `semantic_front` F | p = XY(F), q = norm(p); UNKNOWN if q ≤ `semantic_front_min_horizontal_norm`, tested before normalizing; f = p / q; origin A; front = f; right = (f.y, −f.x); d = T − A |

In `user_to_anchor`, right comes from v, not from the front axis, unlike the other two frames. The library never
derives a heading from a quaternion, never replaces a semantic front with a box's rotation, and has no category gate
for fronts: eligibility belongs to the resolver. `quest_head` stays reserved by the contract.

## Scores and truth

right_score = dot(d, right) and front_score = dot(d, front). The requested score is right_score for `right`,
−right_score for `left`, front_score for `in_front_of` and −front_score for `behind`. With b = `direction_band_m`:

| Requested score | Value |
|---|---|
| > b | TRUE |
| < −b | FALSE |
| from −b to +b, both edges included | UNKNOWN |

This symmetric band is a separate rule from A2.1b's `le` and `ge`, which keep their asymmetric edges and are
unchanged. No numerical epsilon widens, narrows or snaps it; the arithmetic is plain float64. The two degeneracy
cutoffs have no band of their own. A target at the origin scores 0 and is UNKNOWN; a diagonal target can be both
right and in front. The relations use centres only: they don't need the boxes to be apart, and say nothing about
line of sight, occlusion, visibility or flight safety.

## Missing evidence

A required field that is unknown makes the result UNKNOWN, naming its owner. Fields a frame doesn't need never
block it: `user_to_anchor` ignores heading and user rotation, and `object_intrinsic` ignores the pose and the box's
size and rotation. When evidence is missing or a frame is degenerate, there is no requested score (`None`, never 0).

| Reason | Meaning |
|---|---|
| `missing_centre:<object>` | the target's or anchor's centre is unknown |
| `missing_user_position:<command>` | the command's user position is unknown |
| `missing_heading:<command>` | the command's heading is unknown; no quaternion fallback |
| `missing_semantic_front:<object>` | the anchor's semantic front is unknown |
| `degenerate_viewer_anchor` | the user is horizontally within the cutoff of the anchor |
| `vertical_semantic_front` | the semantic front's horizontal part is at or below the cutoff |
| `failed:<relation>` | the requested score is below −b |
| `boundary:<relation>` | the requested score is within the band |
| `not_distinct` | target and anchor are the same object |

## Results and provenance

A `DirectionResult` records:
- the relation, frame, target and anchor;
- the scene's ID, revision and profile;
- `command_id` for the two pose frames, `None` for `object_intrinsic`;
- the value and its reasons;
- `requested_score_m`;
- `measures`: right and front scores, and the viewer–anchor separation or the front's horizontal norm, whichever were computed;
- `inputs`, `assumptions`, `evidence` and the configuration's identity.

`inputs` lists only the fields the computation consulted, as `DirectionFieldRef`s. Each gives:
- the record type and identity: (scene ID, revision, profile) for a scene, (command ID,) for a command;
- the object ID for scene fields;
- the field path and state;
- the evidence kind or unknown reason, and the source;
- the assumption IDs;
- `heading_source` for a known heading.

Sources and assumption IDs are scoped to their record: `assumptions` holds (record type, record identity, ID)
triples, so a scene's `fx` and a command's `fx` stay distinct. A direction that reads no size never carries a size
prior.

## Configuration

`grounding/relations/directions.v1.json`, format `schemas/direction-config.v1.json`, status `provisional`:

| Parameter | Value | Meaning |
|---|---|---|
| `bands.direction_band_m` | 0.02 | symmetric band on the signed score, metres |
| `thresholds.viewer_anchor_min_horizontal_m` | 0.02 | separation at or below which `user_to_anchor` is UNKNOWN |
| `thresholds.semantic_front_min_horizontal_norm` | 0.10 | horizontal norm of the unit front at or below which `object_intrinsic` is UNKNOWN; below 1 |

These values are independent of each other and are starting points for later calibration, not measured perception
uncertainty. Loading uses A2.1b's strict parser and the same identity convention, `config_id` plus a hash of the
parsed content. A loaded `DirectionConfig` is immutable; a changed file loads as a new identity.
`relations.v1.json` is unchanged.

## Computation boundaries

The directional path reads centres, fronts and pose fields straight from the records. It never calls
`Scene.geometry()`, builds a box, or computes corners or footprints. Nothing pose-dependent is cached: each call uses
the command snapshot it is given, and only the object lookup and the configuration are kept per instance. That is
structural evidence of a light path, not a Quest timing; headset cost and Python–C# parity near boundaries belong
to A2.5.

## Tests

```
python grounding/tests/test_directions.py        # or: python -m grounding.tests.test_directions
```

`grounding/tests/fixtures/directions/cases.json` holds the brief's 43 fixed rows, as 90 predicate calls, transcribed
before the code existed; expected scores are the brief's own arithmetic expressions. The cross-cutting checks follow
Section 8 of the brief. Ten package-import checks, from the A2.1c correction, run fresh interpreters that import the
two modules through the package in both orders. They check that directional and relation results share one `Truth`,
and combine correctly under `AND` and `OR`. The accepted suites, `test_relations.py` and `test_contract.py`, also
run.

## Limitations

- The band and cutoffs are provisional and uncalibrated.
- The caller chooses the frame; frame choice, plausibility and the head, body or pointing convention are open.
- Compatibility with datasets' own directional labels is unchecked, and nothing here grounds natural language.
- Results are offline fixture evidence; nothing has run on the headset.
