# The IRef-VLA metadata adapter

A2.2a, D74. `grounding.adapters.iref_vla` imports one pinned public sample, IRef-VLA's ScanNet scene `scene0010_01`, into
the scene contract (`docs/scene-contract.md`). It keeps the complete object inventory, keeps each expression's text
unchanged and keeps the benchmark's answers apart from everything a model could see.

It does not:
- parse English or create resolver queries;
- assign actions;
- select sub-scenes;
- run a model.

This scene is development material, inspected in detail. Never present it, or another scan of its room, as held-out
evaluation data.

## Getting the pinned files

The adapter reads five local files from IRef-VLA commit `741cf0289a6ee7c8f0763058f40bb3788b76f8e0`. Each is downloaded
from `https://raw.githubusercontent.com/HaochenZ11/IRef-VLA/741cf0289a6ee7c8f0763058f40bb3788b76f8e0/` followed by its
path. Don't substitute `main`, a newer archive or the VLA-3D sample files.

| Role | Path in the IRef-VLA repository | Bytes | SHA-256 |
|---|---|---|---|
| objects | `sample_data/Scannet/scene0010_01/scene0010_01_object_result.csv` | 18,654 | `3bd5f282abf5cf5c6abe5d106a2fd87ad6090ec810d50b9e66e4d1e17aaf2996` |
| regions | `sample_data/Scannet/scene0010_01/scene0010_01_region_result.csv` | 224 | `1b77150f1e65bb5c377457bdfdb944575864bd04ab88fb94a24cc40510ee0091` |
| graph | `sample_data/Scannet/scene0010_01/scene0010_01_scene_graph.json` | 317,449 | `59eea627704d46b3028989c7eef418804f987955695697fa62c474dd53f27817` |
| statements | `sample_data/Scannet/scene0010_01/scene0010_01_referential_statements.json` | 4,716,682 | `0c077963bcdb0e7f19aa0885ddb547689dfe106da622d8d4b58d4e894d143d08` |
| vocabulary | `data/NYU_Object_Classes.csv` | 14,241 | `3c65ff2e4ad3a60a0a4ad87f01ef864f6848cdff8b8c7792e43e2389d9585f91` |

Keep them in one folder outside the repository, under their own file names, and never stage them. The upstream
downloader is not needed. Download all five as exact bytes; one way, in PowerShell:

```powershell
$base = "https://raw.githubusercontent.com/HaochenZ11/IRef-VLA/741cf0289a6ee7c8f0763058f40bb3788b76f8e0/"
$dir  = "$env:USERPROFILE\second-eyes-data\iref-vla\741cf0289a6e"
New-Item -ItemType Directory -Force $dir | Out-Null
foreach ($p in "sample_data/Scannet/scene0010_01/scene0010_01_object_result.csv",
               "sample_data/Scannet/scene0010_01/scene0010_01_region_result.csv",
               "sample_data/Scannet/scene0010_01/scene0010_01_scene_graph.json",
               "sample_data/Scannet/scene0010_01/scene0010_01_referential_statements.json",
               "data/NYU_Object_Classes.csv") {
    Invoke-WebRequest -UseBasicParsing -Uri ($base + $p) -OutFile (Join-Path $dir (Split-Path $p -Leaf))
}
Get-FileHash -Algorithm SHA256 (Join-Path $dir "*") | Format-Table Hash, Path -AutoSize
```

The adapter checks every supplied file's size and hash against these pins itself (`pinned.py`), before converting
anything.

## Running it

```
python -m grounding.adapters.iref_vla --objects DIR/scene0010_01_object_result.csv
    --regions DIR/scene0010_01_region_result.csv --vocabulary DIR/NYU_Object_Classes.csv
    [--statements DIR/scene0010_01_referential_statements.json] [--graph DIR/scene0010_01_scene_graph.json]
    --out NEW_FOLDER
```

- **Without `--statements`:** a metadata-only import, the scene, map and views alone. Their bytes are identical to a
  complete import's.
- **Without `--graph`:** the import has no graph cross-check.
- **The output folder must be new.** Nothing is ever overwritten.

| Exit | Meaning |
|---|---|
| 0 | imported |
| 2 | invalid, unsupported or pin-mismatched input, or an existing output folder: each issue is printed as `file: path: code: reason`, with no traceback |
| 3 | an internal error (traceback printed) or a failure to write the output |

Python's API: `convert_scene`, `convert_commands`, `convert_annotations`, `crosscheck_graph`, `run_import` and `encode`,
exported by `grounding.adapters.iref_vla`. `run_import(..., pins=None)` skips the pin check; it exists only for the
hand-written test fixtures.

## What an import writes

```
NEW_FOLDER/
  model_inputs/
    scene.annotated.json        scene v2, all 61 objects
    category-map.json           raw labels -> NYU labels; the 893-label vocabulary
    commands/<command_id>.json  one v1 command per distinct expression (1,936)
  reference_only/
    iref-annotations.json       every source annotation (1,951), with mapped references
    inventory-views.json        the two inventory views
    source-manifest.json        provenance, region, NYU/NYU40 metadata, conversion formulas, evidence boundary
    import-report.json          counts, short rows, relations, size words, the graph cross-check, the runtime
```

`model_inputs` holds what a serializer may later read. `reference_only` holds answers and provenance: never pass it to
a serializer, model, parser or prompt. The shared validator rejects an annotation bundle given as a scene, with
`E_RECORD_TYPE`.

## The scene

| Field | Value |
|---|---|
| format | scene v2 (`schemas/scene.v2.json`): v1 plus a dataset-identity frame |
| `scene_id`, `scene_revision`, `evidence_profile` | `iref.scannet.scene0010_01.full`, 0, `annotated` |
| frame | `iref.scannet.scene0010_01.native`: metres, right-handed, `+z` up; the published aligned origin, not recentred |
| conversion | `dataset_identity` from source `iref_scannet` (kind `dataset`), source frame = scene frame, identity matrix |
| `category_map` | `iref.scannet.nyu.sample.v1` |
| source release | the IRef-VLA commit and the hashes of the scene-side inputs only: objects, regions, vocabulary |

Each object is one CSV row, all 61, in numeric source-ID order. Integer `n` becomes `obj_` plus `n` padded to at least
three digits; nothing is renumbered.

| Contract field | From the CSV |
|---|---|
| `source_ref` | `iref_scannet`, the original ID string, the raw label |
| `category` | `nyu_label` as both standard and model label; NYU `unknown` (objects 55 and 58) as unknown `unmapped_label` |
| `center_m`, `size_m` | `object_bbox_cx/cy/cz`; `object_bbox_x/y/zlength` (full side lengths) |
| `rotation_xyzw` | `[0, 0, sin(h/2), cos(h/2)]` for `h = object_bbox_heading`: a positive rotation about `+z`, support `yaw_only` |
| `semantic_front` | `[cos(f), sin(f), 0]` for `f = object_front_heading`; `_` is unknown `not_in_source`. Never taken from the box's yaw |
| `colours` | colour-scheme labels in slot order, repeats removed keeping the first. `_` slots and omitted trailing groups are absent; with no label at all, unknown `not_in_source` (never `[]`) |
| `observation` | capture time, detector confidence and geometry uncertainty unknown `not_in_source` |

Every known value is `annotated` evidence from `iref_scannet`, with no assumption. Values are the formulas' direct
double-precision results, with no decimal rounding. They need no normalization: they are unit to within round-off.

The map has one entry per raw label with a known NYU label (24), and none for `object`. Its model vocabulary is the
whole pinned NYU vocabulary except the `unknown` sentinel: 893 labels, sorted by code point. RGB values, colour
shares and distances stay in the source; NYU and NYU40 IDs, NYU40 labels and region membership go to the manifest.

## Commands and annotations

Each distinct expression becomes a v1 command:
- **ID:** `iref.scannet.scene0010_01.r0.e.` plus the SHA-256 of its UTF-8 bytes.
- **Text:** the expression, unchanged.
- **Scene and pose:** the full scene's ID, revision and frame; pose kind `none`, with every pose component unknown
  `not_in_source`. The source gives no user pose.
- **No action:** the source gives none.

A command's source release names the IRef-VLA commit and the statements file's path, not its hash. So a command
depends only on its expression and the scene's identity: changing any answer field can't change it.

The annotation bundle (`schemas/iref-annotations.v1.json`) keeps every source annotation as one entry:
- **ID:** the command ID plus `.a` plus the annotation's source index, at least three digits.
- **Payload:** the original annotation object, unchanged, false-statement variants included.
- **Mapped references:** the target, the anchors under their original keys, and the distractors, as `obj_` IDs.

Entries sort by command ID, then source index. The 12 repeated expressions keep all their annotations, each with one
target; none is relabelled as ambiguity. Size words stay in the payloads: 703 entries with 706 occurrences, 584 big
and 122 small. The current query grammar has no size predicate, so their comparisons aren't expressible there.

Before an entry is kept, the adapter checks it:
- **Shape:** the pinned shape exactly. Any unknown key is reported as a source-format change.
- **References:** every target, anchor and distractor must exist in the CSV.
- **Scene-side agreement:** each target's and anchor's class must equal its CSV object's NYU label; its position its
  CSV centre, within 1e-9 m; and its size the box volume `lx*ly*lz`, within `abs(delta) <= 1e-12 * max(1.0, abs(source_volume))`: a relative tolerance
  with an absolute floor (D75 clarified this wording; the check is unchanged).

A disagreement fails the import; nothing is repaired.

## Inventory views

`inventory-views.json` names two views of the one canonical scene. Neither is a second scene file.

| View | Objects | Rule |
|---|---|---|
| `full_inventory` | 61 | every object row, in source-ID order |
| `source_known_nyu` | 59 | every object whose NYU label is a vocabulary label other than `unknown`; 55 and 58 excluded for `source_unknown_category` |

Both are built from the CSV and vocabulary only, before any statement is read. In the full inventory, the resolver
treats the two unknown-category objects as possible members of every class. The source's generator assumed the
59-object graph, so the two views answer different questions; later evaluation must state which one it uses.

## Separation of answers

- **The scene stage can't see answers.** `convert_scene` takes only the object CSV, the region CSV and the vocabulary.
- **Commands read only the expression keys.**
- **The tests check it:**
  - a metadata-only import writes byte-identical scene, map and view files;
  - with every target, anchor, distractor and position mutated, the scene stage and the commands stay byte-identical,
    and the annotation cross-check then rejects the mutated answers;
  - no annotation field name occurs as a key in any model-input record.

## Determinism and failure

- **Output format:** UTF-8, sorted keys, two-space indentation and one final LF, written as bytes. Records contain no
  timestamps, absolute paths or random IDs.
- **Determinism:** identical inputs give identical bytes under the same adapter version and Python runtime. The report
  names the runtime; across runtimes, compare geometry numerically.
- **Publishing:**
  1. `run_import` refuses an existing output folder.
  2. It converts and validates everything: the scene, map and commands together through the shared validator, with no
     issue allowed at all; the bundle against its schema; and the closed reference records and their cross-file
     references.
  3. It writes into a new sibling folder and publishes it with one rename. On any failure that folder is removed, so no
     apparently complete bundle remains.
- **Exceptions:** an unexpected exception is reported as an internal error (exit 3), never as an import.

| Code | Meaning | Examples |
|---|---|---|
| `E_IREF_SOURCE_SHAPE` | the file is not in the pinned format | a changed or repeated header, an incomplete colour group, an extra column, a byte-order mark, an unknown or missing annotation key, a repeated JSON key, an unreadable file |
| `E_IREF_SOURCE_VALUE` | a field's value is invalid | an empty field, a non-canonical or repeated ID, a non-finite or malformed number, a zero size, an invalid front, a label with whitespace at an end, a colour group mixing `_` and values |
| `E_IREF_REFERENCE` | a reference doesn't resolve | an object's region not in the region file; an annotation's target, anchor or distractor not in the CSV |
| `E_IREF_SOURCE_MISMATCH` | sources disagree | an NYU ID and label that disagree with the vocabulary, one raw label with two NYU labels, an annotation or graph object that disagrees with the CSV, a file that differs from its pin |
| `E_IREF_UNSUPPORTED_SOURCE` | valid, but outside this increment | a second region, another scene |
| `E_IREF_OUTPUT_EXISTS` | the output folder already exists | |
| `E_IREF_OUTPUT_IO` | the output could not be written; nothing was published | |

## Evidence boundary

The geometry is published IRef-VLA annotation of a ScanNet scan, in the aligned scene frame its preprocessing produced
(VLA-3D `a0c023c9662bd875f0b3aeff4ff60adc1991e449`). Its metric meaning follows from ScanNet's geometry and that
preprocessing path; it was not measured against the physical room.

The inspected code explains the conventions, and the tests confirm them on the sample: every graph box rebuilt from the
output quaternion matches the source's corners within 1e-9 m. The code doesn't certify which producer commit made the
released sample.

## Tests

```
python grounding/tests/test_iref_vla_adapter.py --sample DIR
```

`SECOND_EYES_IREF_VLA_SAMPLE` may name `DIR` instead. The expectations (`grounding/tests/fixtures/iref_vla/
expectations.json`) and the hand-written source-format fixtures beside them were fixed before the adapter was written.
Without the pinned files, the pinned-sample checks are reported as one FAIL, and the suite exits 1: acceptance can't
pass without its data, so they are never skipped.

## Choices where the brief was silent

1. **Error codes:** which failure raises each of the seven codes (the table above).
2. **Missing pinned files:** tests fail, never skip.
3. **Command provenance:** commands name the source commit and path but not the statements' hash.
4. **Unreadable inputs:** an unreadable input file is `E_IREF_SOURCE_SHAPE`.
5. **Unknown versus known labels:** a raw label that is NYU `unknown` in one row and known in another is a conflict.
6. **Tolerances:** 1e-9 m for coordinates; `abs(delta) <= 1e-12 * max(1.0, abs(source_volume))` for volumes, a relative tolerance with an absolute
   floor. The pinned sample agrees exactly.
7. **Graph cross-check scope:** it compares labels, IDs, centres, sizes, volumes and colour slots. Box corners are
   checked by the tests, with their own formula.
8. **Colour groups:** a placeholder group must be all `_`. A present group needs RGB values from 0 to 255, a share in
   [0, 1] and a distance of at least 0.

## Deferred

These belong to later approved increments:
- size predicates;
- assigning a protocol action;
- restricted-profile priors;
- sub-scene construction;
- the false-statement variants as negative examples;
- other IRef-VLA scenes or source versions;
- model evaluation and training;
- headset work.
