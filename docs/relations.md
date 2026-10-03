# Relation library, v1

How `grounding/relations/` decides closest and farthest (ranks 1–3), near, above, below, on, inside and between,
as approved in the A2.1 specification v0.2 (§7.2–7.8) and in D67. This is the A2.1b increment. Directions (§7.9),
the resolver, serializers, adapters and headset code are not part of it.

## Inputs and configuration

The library reads scene records that have passed the contract validator (`docs/scene-contract.md`). Its parameters
are in `grounding/relations/relations.v1.json`, whose format is `schemas/relation-config.v1.json`. That file is marked
`provisional`: its values are documented starting points, to be compared on scene-disjoint development data and
frozen before any evaluation (§10.1). A configuration's identity is its `config_id` plus a hash of its parsed
content, so it is the same on every platform; every result records it. Category lists are by model-visible label and
cover only the fixtures' vocabulary (support and furniture: `table`; container: `box`) until the dataset map exists.

```
from grounding.relations.predicates import Relations
rel = Relations(scene_record)
rel.near("obj_005", "obj_001")       # -> RelationResult
rel.rank("closest", "obj_001", ["obj_004", "obj_005"], possible=["obj_006"], k=1)   # -> RankResult
```

The modules are imported through the package, with the repository root on the import path. They import each other
relatively, so there is one `Truth` for every module; importing one by its bare name, such as `import predicates`,
fails instead of loading a second copy (A2.1c correction).

## Results

| Field | Relation results | Ranking results |
|---|---|---|
| outcome | `value`: TRUE, FALSE or UNKNOWN | `status`: resolved, ambiguous, no_match or insufficient, with `object_ids` (one if resolved, the tie group if ambiguous) |
| why | `reasons`, empty when TRUE, e.g. `failed:lateral`, `boundary:distance`, `missing_rotation:obj_003` | `reasons`, e.g. `tie`, `missing_centre:obj_007`, `possible_competitors:obj_003,obj_004` |
| detail | `basis` for TRUE results of on (`top_face_support`), inside (`outer_box_containment`) and below (its branch); `measures`, the computed quantities | `distances_m`; `possible_count` and `combinations_checked` |
| provenance | `inputs`: every source field consulted, with its state, evidence kind or unknown reason, source and assumption IDs; `assumptions` and `evidence`, their unions; `config`, the configuration's identity | the same |

A relation's TRUE is geometric inference, not verified contact, a real cavity or a safe path (§7.1). A result
computed from an assumed category size stays conditional on that prior: its `assumptions` say so.

## Three-valued evaluation

Each relation is a set of conditions, each TRUE, FALSE or UNKNOWN. They combine by AND (FALSE if any is FALSE, TRUE if
all are TRUE, otherwise UNKNOWN) and OR (TRUE if any is TRUE, FALSE if all are FALSE, otherwise UNKNOWN). A result
is therefore FALSE only when a condition that could be evaluated is FALSE; missing geometry never becomes FALSE.
Gates are checked first and give UNKNOWN when they fail: on's support category and tilt, inside's container category,
between's anchor separation. Objects that repeat in one call make the result FALSE (`not_distinct`).

A box needs a known centre, size and rotation. A condition that needs a box with unknown rotation is UNKNOWN: no
missing-orientation bounds are approved.

## Comparisons

`le(x; t, b)` is TRUE if x ≤ t − b, FALSE if x > t + b, and UNKNOWN otherwise. `ge(x; t, b)` is TRUE if x ≥ t + b,
FALSE if x < t − b, and UNKNOWN otherwise. With b > 0, x = t is UNKNOWN; with b = 0, equality is TRUE. The bands are
operational boundary rules, not measured perception uncertainty. Values below are the provisional ones.

| Condition | Measure (unit) | Test |
|---|---|---|
| near | d, shortest box distance (m); 0 if the boxes intersect or touch | le(d; 0.50, 0.02) |
| above: vertical gap | g = zmin(T) − zmax(A) (m) | ge(g; 0.05, 0.01) |
| overlap, for above, on and below | o = footprint intersection area / smaller footprint area | ge(o; 0.5, 0.05) |
| on: gate | anchor category in the support list; tilt θ ≤ 5.0° (no band, 5.0° qualifies) | else UNKNOWN, before geometry |
| on: gap and penetration | g as above (m) | le(g; 0.05, 0.01) and ge(g; −0.05, 0.01) |
| below: separated branch | s = zmax(T) − zmin(A) (m) | le(s; 0, 0.01), and the overlap |
| below: furniture gate | anchor category in the furniture list | unknown category: branch UNKNOWN; not listed: branch FALSE |
| below: furniture branch | \|zmin(T) − zmin(A)\| and zmax(T) − zmax(A) (m) | le(·; 0.10, 0.02), le(·; 0, 0.01), and the overlap |
| inside: gate | anchor category in the container list | else UNKNOWN, before geometry |
| inside: containment | e = largest \|c′ᵢ\| − hᵢ over the target's 8 corners in the anchor's frame (m) | le(e; 0.02, 0.01) |
| between: separation gate | L = \|c_B − c_A\| (m) | evaluated only if L ≥ 0.22; else UNKNOWN, never FALSE |
| between: endpoints | a = (c_T − c_A)·(c_B − c_A)/L (m) | ge(a; 0, 0.02) and le(a; L, 0.02) |
| between: lateral | p, distance from c_T to the anchors' line (m); w = max(0.30, 0.20·L) | le(p; w, 0.02) |
| between: overlap with each anchor | o if the vertical extents overlap by more than 0 m, else 0 | le(o; 0.5, 0.05) |
| ranking ties | gap between neighbouring sorted distances (m) | linked if ≤ 0.05 |
| support tilt | angle between the anchor's rotated local +z and scene +z (°) | the on gate |

## Ranking

Closest and farthest rank candidates by 3D distance between centres (S17), for ranks 1–3. The anchor is never its own
candidate. Rank k is resolved only if it is linked to neither neighbour; otherwise the result is ambiguous, with the
run of linked neighbours as the tie group. Fewer than k candidates is no_match. If the anchor or any candidate lacks a
centre, the result is insufficient.

Candidates whose eligibility is unknown are passed separately as `possible`, and evaluated jointly: the result is the
outcome shared by every combination of them being eligible or not, and insufficient if two combinations differ.
Checking them one at a time is not enough; with one definite and two possible candidates, the third closest exists
only if both are eligible. The library evaluates all 2^m combinations for m possible candidates and records both
numbers. This is an exact offline reference for small fixtures, exponential in m, with no cap. Its cost on the
headset is not claimed, and a bounded or faster method would need its own verification before deployment. Who counts
as a possible candidate, for example an object of unknown category, is the resolver's decision.

## Geometry

`geometry.py` computes, exactly for axis-aligned, yaw-only and fully rotated boxes: corners from the centre, half
sizes and the normalized quaternion; vertical extents from the corners; the footprint as the convex hull of the
projected corners (§7.1); footprint intersection by polygon clipping; the shortest box distance as 0 if a
separating-axis test finds no separation, otherwise the smallest corner-to-box or edge-to-edge distance; and
containment from the target's corners in the anchor's frame. It is pure Python. Its cost on the Quest is measured in
A2.5, not claimed here.

## Tolerance kinds

| Kind | Where | Purpose |
|---|---|---|
| Validation tolerance, 1e-6 | contract, `checks.TOLERANCE` | stored data is well formed |
| Relation thresholds | the relation configuration | what each word means |
| Uncertainty bands | the relation configuration | which borderline cases stay undecided; kept apart from thresholds (§10.1) |
| Numerical epsilon, 1e-9 | `geometry.EPS` | floating-point guards only, such as parallel edges; no meaning |

## Caches

Per-object geometry is cached by (scene ID, scene revision, evidence profile, object ID). It doesn't depend on the
configuration. A key reused with different geometry raises an error rather than returning stale geometry. Relation
results are not cached; a cache of them would have to add the configuration's identity to its key.

## Tests

```
python grounding/tests/test_relations.py
```

The reviewed cases in `grounding/tests/fixtures/relations/cases.json` were calculated by hand from these definitions
and approved before the library existed; the library must reproduce their values within 1e-6. The test also checks
the configuration, the comparison rules, a tilted box's hand-calculated geometry, invariance under turning the scene
about +z and shifting it, swapped between-anchors, the restricted profile, provenance, the cache and the cost of joint
eligibility. Its sampling and grid checks are approximate diagnostics, not expected values.

## Limitations

- The parameters and category lists are provisional and uncalibrated; the lists cover only the fixtures' vocabulary.
- Centre distance can misrank candidates near a long anchor such as a partition (§9.6); S17 is unchanged, and the
  case isn't handled yet.
- Without approved missing-orientation bounds, box relations are UNKNOWN in the restricted profile.
- Compatibility with IRef-VLA's own relation labels is unchecked; it comes with the adapter.
- On and inside are approximations: a top face is not verified contact, and an outer box is not a cavity.
- Joint eligibility is exhaustive and exponential in the number of possible candidates.
- Nothing here has run on the headset.
