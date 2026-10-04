# Scene contract, v1

How scene, command-context and category-map records are written, and how they are checked offline. This is the
first A2.1 increment, A2.1a (D66), built from the A2.1 specification v0.2 and the contract proposal approved with
it. The relation library, resolver, serializer and dataset adapters come in later increments and read these records.

## Records and files

| Record | Schema | Holds |
|---|---|---|
| Scene | `schemas/scene.v1.json` | the frame, sources, assumptions and objects of one scene snapshot, in one evidence profile |
| Command context | `schemas/command-context.v1.json` | one typed command, the scene snapshot it refers to, and the user's pose at submission |
| Category map | `schemas/category-map.v1.json` | how one source vocabulary's raw labels map to standardized and model-visible labels |

One record per `.json` file, UTF-8 without a byte-order mark. Every record carries `schema_version` (1) and
`record_type`. Records are closed: every field the format defines is present, and no other field is allowed, so an
answer field such as a target ID can't slip into a scene or command. An incompatible change gets a new version
(`scene.v2.json`) and an entry in `notes/decisions.md`; files are regenerated, not migrated.

## Values and their evidence

Every field that carries evidence has one of two shapes:

```json
{"state": "known", "value": ..., "evidence": {"kind": "annotated", "source": "fx", "assumptions": []}}
{"state": "unknown", "reason": "not_in_source", "source": "fx"}
```

A known value always has its value and evidence; an unknown one never has a value. A known empty list (`"value": []`)
means known to be empty, which is different from unknown. There is no `null` and no sentinel such as `_`, NaN or 0
for missing data; an adapter turns a source's placeholder into an explicit unknown. Identity fields (`scene_id`,
`scene_revision`, `object_id`, `command_id`, `map_id`) are plain values and never unknown.

Evidence kinds:

| Kind | Meaning |
|---|---|
| `annotated` | a dataset's or a fixture's value, kept as that and never relabelled as a measurement |
| `measured` | a sensor measurement of ours; v1 has no source that can supply one (see Limitations) |
| `inferred` | computed from other evidence |
| `assumed` | taken from a prior, such as a category's size |

Unknown reasons: `not_in_source` (the source doesn't give it), `withheld_by_profile` (the restricted profile removed
it), `no_permitted_prior` (the restricted profile has no prior for it), `unmapped_label` (the raw label isn't in the
category map), `source_invalid` (the source's value failed validation and the adapter recorded it as unknown, and
counted it), `not_estimated` (no component estimates it).

Each record lists its `sources` (`source_id`, `kind`, `description`, `release`) and `assumptions` (`assumption_id`,
`description`); evidence and unknowns name them by ID. Source kinds are `dataset`, `fixture`, `prior_table` and
`profile`. Evidence is written for each field; nothing is inherited. A value that later code derives from others
carries the union of their assumptions, so a relation computed from a category-prior size stays conditional on that
prior.

Which source kinds may stand behind which evidence (v1):

| Evidence kind or reason | Source kinds |
|---|---|
| `annotated` | dataset, fixture |
| `inferred` | dataset, fixture, prior_table |
| `assumed` | prior_table |
| `measured` | none in v1 |
| `not_in_source`, `unmapped_label`, `source_invalid` | dataset, fixture |
| `withheld_by_profile` | profile |
| `no_permitted_prior` | prior_table, profile |
| `not_estimated` | any |

## Scene

| Field | Known value | Rules |
|---|---|---|
| `scene_id` | ID: `^[a-z0-9][a-z0-9_.-]{0,127}(?![\s\S])` | a scene's identity is its ID, revision and profile |
| `scene_revision` | integer ≥ 0 | |
| `evidence_profile` | `annotated` or `restricted` | see Evidence profiles |
| `category_map` | the `map_id` of its category map | |
| `coordinate_frame` | `frame_id`; `units` `m`; `handedness` `right`; `up_axis` `+z`; `origin` (how it was established); `conversion` | right-handed, z up, metres (S07); v1 accepts only `authored_in_scene_frame` |
| `objects` | a list of objects; may be empty | |

Each object:

| Field | Known value | Rules |
|---|---|---|
| `object_id` | `^obj_[0-9]{3,}(?![\s\S])` | unique in the scene; see Identities |
| `source_ref` | `source_id`, `source_object_id`, `source_label` (the raw label) | from a dataset or fixture source |
| `category` | `{standard, model}`: the source's standardized label and the model-visible label | may be unknown, for example `unmapped_label`; the object stays in the scene |
| `geometry.center_m` | 3 numbers: the bounding-box centre in the scene frame, metres | |
| `geometry.size_m` | 3 numbers > 0: full side lengths along the box's local x, y and z, metres | never sorted without the axes |
| `geometry.rotation_xyzw` | 4 numbers, a quaternion from the box's local axes to the scene axes, and `support` | see Geometry |
| `semantic_front` | a unit 3-vector in the scene frame | separate from the box's rotation |
| `attributes.colours` | a list of unique colour labels; `[]` means known to have none | v1 keeps labels only, not proportions |
| `observation.source` | the source that supplied the object (plain ID) | provenance, not what a user knows |
| `observation.capture_time` | `{clock, value_us}` | when it was observed, not when it was received |
| `observation.detector_confidence` | a number in [0, 1] | a category confidence, not positional accuracy |
| `observation.geometry_uncertainty` | unknown only in v1 | |

Times are `{"clock": "headset_mono" or "utc", "value_us": <integer>}`, in microseconds like the event log's `mono_us`
and `utc_us` (`docs/logging.md`). Units are part of each field's name or meaning; values are stored as produced, with
no rounding rule.

## Command context and user pose

| Field | Known value | Rules |
|---|---|---|
| `command_id` | ID | unique among the records validated together |
| `text` | the typed command, as given | |
| `scene_id`, `scene_revision` | the scene snapshot the command refers to | must match a given scene |
| `user_pose.pose_kind` | `none`, `authored`, `dataset_camera`, `dataset_viewpoint` or `quest_head` | `quest_head` is reserved: v1 rejects it |
| `user_pose.frame_id`, `user_pose.scene_revision` | the frame and snapshot the pose is expressed in | must match the scene's frame and the command's revision |
| `user_pose.position_m` | 3 numbers, scene frame, metres | |
| `user_pose.rotation_xyzw` | a quaternion and `support` (`yaw_only` or `full`) | |
| `user_pose.heading_xy` | a horizontal unit vector `[x, y]` in the scene frame, and `heading_source` | |
| `user_pose.snapshot_time` | `{clock, value_us}`: when the pose was taken | |

Pose rules in v1: with `pose_kind` `none`, every component is unknown. Authored and dataset poses are annotated
(`authored` by a fixture source, `dataset_camera` and `dataset_viewpoint` by a dataset source); a dataset camera pose
keeps that meaning and is never relabelled as a measured headset pose. Heading sources are `authored` for authored
poses and `camera_yaw` for dataset poses; `head_yaw`, `pointing` and `body` belong to `quest_head`, and which one the
headset uses is still open (A2.5).

## Category map

`map_id`, `description`, `release`, `model_vocabulary` (unique model-visible labels) and `entries`, each
`{source_label, standard, model}`. Each entry's model label is in the vocabulary, and each source label appears once.
An object's known category, together with its `source_label`, must be an entry of its scene's map. The raw label
stays in `source_ref`, so an unmapped object keeps it while its category is unknown. The real tables come with the
dataset adapters; v1 has only the fixture's map.

## Identities

`obj_` IDs are internal contract IDs. They are unique in a scene, don't encode the category, and are never reused for
another object in the same scene. How objects are spelled, tokenized and ordered for a model stays open; the
serializer decides that later, and it may map these IDs to other spellings. Source IDs stay in `source_ref`. A
single file can't show that an ID was never reused across revisions; that needs two revisions and isn't checked.

## Geometry

Rotations are normalized quaternions `[x, y, z, w]` from local to scene axes (S09). `support` says what the source
gave: `axis_aligned` (the identity, for boxes aligned with the scene axes), `yaw_only` (a rotation about +z; pitch and
roll were not observed) or `full`. Unknown orientation is an unknown value, never an assumed identity.

Validation tolerance: a quaternion's norm, a front's or heading's length, a yaw-only rotation's x and y, and an
axis-aligned rotation's distance from the identity must each be within 1e-6 (`checks.TOLERANCE`). This is a v1
validation choice for stored data. It is not a relation threshold or tolerance; those stay open in the
specification's parameter register (§10.2) and come with the relation library.

## Evidence profiles

Both profiles hold oracle identities, categories and centres (specification §9.5).

| Field | `annotated` | `restricted` |
|---|---|---|
| Size | annotated, or unknown | a category prior (`assumed`, from a `prior_table` source) or unknown (`no_permitted_prior`) |
| Rotation, semantic front, colours | annotated, or unknown | unknown (`withheld_by_profile`) |
| Anywhere | no `assumed` value, no `withheld_by_profile` or `no_permitted_prior` | |

The validator enforces these rules; building the restricted profile and its training-only priors is later work.

## Validation

```
python grounding/contract/validate.py FILE.json [FILE.json ...]
python grounding/tests/test_contract.py
```

The validator never writes. It prints one line per issue, `file: JSON path: code: message`, and exits 1 if any file
has an error. Layers run in order, and a record goes on to the next only if it passed:

1. **Parsing:** UTF-8 without a byte-order mark; no repeated keys, NaN, Infinity or number too large for a double.
   Python's `json` module would accept all four.
2. **Record type:** a JSON object with a supported `record_type` and `schema_version`.
3. **Schema:** the JSON Schema of that format: structure, closed records, the two value shapes, words, lengths,
   types, ID patterns and simple ranges.
4. **Semantic rules** (`checks.py`): finite numbers, unique IDs, resolvable sources and assumptions, provenance,
   geometry, the evidence profiles, the user pose and category maps.
5. **Cross-record rules,** for records given together: no two records with the same identity; a command's scene at
   its revision and in its frame; a scene's categories in its map. A referenced record that isn't given gives a
   warning, not an error.

IDs, labels and object IDs match as whole strings (D73). Their patterns end in `(?![\s\S])`, not `$`: jsonschema
applies a pattern with Python's `re.search`, where `$` also matches just before a final newline, so `fixture.a17\n`,
`chair\n` and `obj_001\n` (each ending in a real line feed) used to pass. A value with a final LF now fails as
`E_SCHEMA_PATTERN` at its path, as a final CR, tab or space already did, so the record goes no further than the schema
layer. Nothing is trimmed or rewritten. The generic ID is `^[a-z0-9][a-z0-9_.-]{0,127}(?![\s\S])` and a label
`^\S(.*\S)?(?![\s\S])`. Both are defined alike in `scene.v1.json`, `command-context.v1.json` and
`category-map.v1.json`, and copied into `grounding-query.v1.json`. They cover every field that refers to them,
nested ones included. Free text keeps its own rule: `command.text`, descriptions, `release`, `origin` and
`source_object_id` may contain line breaks.

| Code | Layer | Meaning |
|---|---|---|
| `E_PARSE_ENCODING`, `E_PARSE_JSON`, `E_PARSE_DUPLICATE_KEY` | parsing | not UTF-8 without a byte-order mark; not well-formed JSON; a repeated key |
| `E_NONFINITE` | parsing, semantic | NaN, infinity, or a number too large for a double |
| `E_RECORD_TYPE` | record type | not an object of a supported type and version |
| `E_SCHEMA_REQUIRED`, `_ADDITIONAL`, `_TYPE`, `_ENUM`, `_CONST`, `_PATTERN`, `_LENGTH`, `_RANGE`, `_UNIQUE`, `_OTHER` | schema | by JSON Schema keyword; `_REQUIRED` includes a known value without its value or evidence, `_ADDITIONAL` an unknown with a value |
| `E_DUPLICATE_ID`, `E_DUPLICATE_REGISTRY` | semantic | a repeated `object_id`; a repeated source or assumption ID |
| `E_UNRESOLVED_SOURCE`, `E_UNRESOLVED_ASSUMPTION` | semantic | an ID not in the record's lists |
| `E_EVIDENCE_SOURCE` | semantic | evidence or a reason that doesn't fit its source's kind |
| `E_QUAT_NORM`, `E_QUAT_YAW_TILT`, `E_QUAT_AXIS_ALIGNED`, `E_UNIT_VECTOR` | semantic | the geometry rules above |
| `E_PROFILE_WITHHELD`, `E_PROFILE_SIZE`, `E_PROFILE_ANNOTATED` | semantic | the profile rules above |
| `E_POSE_RESERVED`, `E_POSE_NONE_KNOWN`, `E_POSE_EVIDENCE`, `E_HEADING_SOURCE`, `E_POSE_REVISION` | semantic | the pose rules above |
| `E_MAP_VOCAB`, `E_MAP_DUPLICATE` | semantic | the category map rules above |
| `E_DUPLICATE_RECORD`, `E_SCENE_MISMATCH`, `E_FRAME_MISMATCH`, `E_CATEGORY_UNMAPPED` | cross-record | the cross-record rules above |
| `W_NOT_CROSS_CHECKED` | cross-record | a warning: a referenced scene or map wasn't given |

The tests' fixtures are in `grounding/tests/fixtures/contract/`, with a README listing each invalid case and what
it changes.

## On the headset (planned for A2.5, not built)

The app will check less than this validator: that the version and record type are supported, that every value's
state is `known` or `unknown` (a missing state rejects the file instead of reading as a default), that known values
are present, finite and of the right length, that IDs are unique, and that the command, scene and frame agree. This
reduced check assumes the files already passed the offline validator, which stays the only full check. Records that
live perception produces on the headset will need their own validation where they are produced; nothing here
provides it.

To verify in A2.5, not claimed now: whether Unity's JsonUtility reads these shapes correctly, how long parsing takes
and how much memory it uses on the Quest, and how Python's and C#'s arithmetic agree near relation boundaries. The
two don't compute identically.

## Limitations of v1

- Conversions: only frames authored in scene coordinates. Dataset conversions, with their numerical mapping, come
  in a later version alongside the adapter, defined before the code that applies them.
- Evidence: no source can supply `measured` values, and authored and dataset poses accept only `annotated` evidence.
  These are limits of this increment, not of the contract: a later version can let dataset transformations and
  derived headings, such as a heading computed from a dataset camera's rotation, carry `inferred` evidence. A
  dataset pose is never `measured`.
- Geometry uncertainty is unknown-only; its known form waits for an estimator.
- `quest_head` poses are reserved.
- No grounding-result or annotation schema yet; they come with the resolver and the adapters.
- One record per `.json` file.
- A label with an internal CR, U+2028 or U+2029 passes here, because Python's `.` matches them. It would fail in a
  validator using ECMAScript regular expressions, whose `.` doesn't (O24).
