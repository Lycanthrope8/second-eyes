# The category-complete selection audit

A2.2c, D77. `grounding.subscenes.iref_vla` asks one question of the pinned A2.2a sample: can each command be paired
with a smaller inventory without discarding any object that could compete for its target or anchors?

For each parsed command, it keeps every object whose model category the accepted parse names on any node, and every
object whose category is unknown. It does this within each accepted inventory view, and records the result as a
selection plan over the unchanged scene.

The selection is:
- **Answer-independent**, but command-dependent and parser-dependent.
- **Not a spatial or visibility crop:** the selected objects may lie anywhere in the room, and this is not a
  field-of-view simulation.
- **Not a proven deployment selector**, and not an unbiased sample of natural commands.

The audit writes no scene, prompt, training example or model-facing document, and no new evaluation result.

## Why this rule

Every node of a query from the current grammar filters by an explicit category. An object of a known category outside
their union fails that filter for every node, so removing it can't remove a possible target or anchor.

Same-category objects stay even when their colour, distance or other properties would currently eliminate them.
Unknown-category objects stay because they could belong to any requested category.

This is about possible bindings under the accepted query semantics. It doesn't establish identical model predictions,
resolver work counts, metadata or execution under a finite work budget. Nor does it establish that the parser understood
the command correctly.

## Inputs

```
python -m grounding.subscenes.iref_vla --scene MODEL_INPUTS/scene.annotated.json --commands MODEL_INPUTS/commands
    --category-map MODEL_INPUTS/category-map.json --inventory-views REFERENCE_ONLY/inventory-views.json --out NEW_DIR
```

- **The inventory manifest** is permitted because its membership comes from scene categories alone. Nothing else in its
  folder is read.
- **Commands** are the declared folder's direct `.json` files, each named by its command ID; there is no recursive
  discovery.
- **No argument or function parameter** accepts annotations, statements, graphs, target positions, predictions or
  scores.
- **Validation is A2.2b's,** called unchanged: input, role and cross-record validation, the inventory-view policy, the
  parser with its full vocabulary, colour list and exact text handling, and the parse-record checks.
- **Each distinct text is parsed once.**
- **The command line is limited to the A2.2a sample.** The Python API also takes authored fixtures.

## The policy, `iref_category_complete.v1`

1. **Parse** each distinct command once, and validate its parse record.
2. **An unsupported parse** gets an `unassessed_parse` row in each view, with the parser's exact reason. No category is
   guessed from its words, and it isn't treated as needing zero objects.
3. **A parsed command's required categories** are the sorted unique labels on every node of its interpretation, both
   between anchors included. A parsed node without a vocabulary category is an internal failure (exit 3), never a
   fallback.
4. **In each view V,** the required set is `K(V, C) = {o in V : category(o) unknown, or category.value.model in C}`.
   Raw labels and ID spelling play no part, and no third exclusion rule is added.
5. **Order:** retained IDs are sorted lexicographically as whole strings, so `obj_1000` sorts before `obj_999`.
6. **The budget:** a set of at most 10 objects is `fits`; a larger one is `over_budget` and kept whole.

Colour, size, geometry, orientation, front, pose, confidence and relation results never affect membership. An anchor
isn't excluded from its own category's set. An absent known category contributes nothing. Zero-, one- and two-object
selections are valid and reported separately.

The core selector, `select(objects, categories)`, sees only object IDs with a model category or `None`, plus the
category set. It does no geometry, subset enumeration, inference or tokenization.

## Outputs

`selections.jsonl`, `summary.json`, `report.md` and `manifest.json`, published together through a new sibling folder
and one rename; an existing destination is refused.

Each row is a closed `iref_subscene_selection` record (`schemas/iref-subscene-audit.v1.json`), one per command and
view. Rows are ordered by command ID, then `full_inventory` before `source_known_nyu`. Each row records:
- `parse_status` and `parse_reason`, as the parser gave them;
- `status`: `fits`, `over_budget` or `unassessed_parse`;
- the selection fields: `required_categories`, `retained_object_ids`, `retained_unknown_category_ids`,
  `required_object_count` and `selection_id`. All five are null when unassessed; empty arrays are valid assessed
  outcomes;
- `base_object_count` and `primary_object_limit` (10).

`selection_id` is `iref.subset.` plus the SHA-256 of the canonical JSON payload (sorted keys, compact separators,
UTF-8, finite numbers, no trailing newline). It holds four fields: `policy_id`, `parent_scene_sha256`, `view_id` and
`retained_object_ids`. It contains no command, relation, colour or answer, so equal sets in one scene and view share
it.

The parent-scene hash is the accepted canonical JSON hash of the scene with its objects in lexicographic ID order, so
the object order of the file can't change an identity. For the pinned sample, already in that order, it equals the
hash of the file as given. A selection ID is an audit identity, not a contract scene ID. A changed source scene gets new
identities, while its memberships can stay the same.

**The summary** gives, per view:
- N, P and U, with `N = P + U`;
- the parser's unsupported reasons;
- the histogram of required sizes, and the small selections;
- `fits` and `over_budget`, with `P = fits + over_budget`;
- fits over N and over P;
- the count needing at most each limit from 3 to 10, over both denominators;
- distinct assessed and fitting selections;
- commands per selection;
- retained unknown-category counts.

It also pairs the two views' statuses. There is no accuracy, source-target retention or relation agreement: the audit
has no answers. The report is rendered from the summary and opens with the five required statements.

**The manifest** records the input file hashes, the policy, the parser and protocol, code hashes, the runtime and the
output hashes. Runtime and code provenance differ across machines, so the whole manifest isn't expected to match
elsewhere.

## Exit codes

| Exit | Meaning |
|---|---|
| 0 | the audit completed; over-budget and unassessed rows are normal outcomes |
| 2 | invalid input or an existing destination; nothing published |
| 3 | a write failure or an unexpected exception; nothing published |

Input errors use A2.2b's codes: `E_EVAL_INPUT`, `E_EVAL_VIEWS`, `E_EVAL_OUTPUT_EXISTS` and `E_EVAL_OUTPUT_IO`.

## The pinned sample

These figures are development measurements from one room. They were reproduced exactly as the brief predicted:
- **Population:** N 1,936, P 1,179 and U 757 (703 size comparisons, 42 coreferences, 12 plural references), giving
  3,872 rows.
- **At ten objects:** the full inventory has 1,004 fitting and 175 over budget; source-known has 1,095 and 84.
- **Paired:** 1,004 commands fit both views and 91 fit only source-known; none fits only the full inventory.
- **Distinct assessed memberships:** 123 per view.
- **The unknown pair:** the full inventory adds `obj_055` and `obj_058` to every assessed selection.

The source-known figure of 1,095 equals A2.2b's source-known target-selection count. They are different quantities, and
their equality says nothing about shared commands, accuracy or cause.

## Tests

```
python grounding/tests/test_iref_vla_subscenes.py --sample DIR    full acceptance (fixtures, then the pinned sample)
python grounding/tests/test_iref_vla_subscenes.py --unit-only     fixtures only; reported as NOT sample acceptance
```

The expectations (`grounding/tests/fixtures/iref_vla_subscenes/expectations.json`) were written by hand before the
selector existed. Without the pinned files the sample check fails, unless `--unit-only` is given.

## Not decided here

These remain open:
- deployment selection;
- prompt rendering;
- any comparison of formats on selected inventories. Both formats would need the identical preselected inventory, and
  a parser-conditioned subset is an extra pipeline assumption;
- reuse of A1's warm-cache timings, which don't transfer to per-command subsets;
- size predicates;
- parser expansion;
- scene splits;
- model, headset, detector, speech, Vicon and participant work;
- Gate B.
