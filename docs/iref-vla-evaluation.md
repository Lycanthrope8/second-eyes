# The IRef-VLA development evaluation: a text-only rules baseline

A2.2b, D75. `grounding.evaluation.iref_vla` asks one question of the pinned A2.2a sample (`scene0010_01`: 61 objects,
1,936 distinct command texts, 1,951 source annotations): what happens when a declared, text-only grammar builds a query
and the unchanged resolver evaluates it, with and without the two unknown-category objects?

This is a development diagnostic, from one inspected room with annotated geometry, a sample-specific rules parser and
provisional relation semantics. It is not:
- held-out grounding accuracy;
- a general English parser;
- a model comparison, a headset measurement or Gate B.

Agreement with source labels doesn't show that our relation semantics match the dataset's. Nothing here chooses between
the two eventual grounding designs, direct candidate scoring and parsing plus resolution.

## Three operations

| Operation | Reads | Never reads |
|---|---|---|
| parse | one command text, the category vocabulary and the ten fixed colours | a scene, object IDs, source relation labels, anchors, answers |
| predict | the validated scene, category map and commands; the inventory-view manifest; the relation and direction configurations; the tracked protocol | the annotation bundle, or any other reference file |
| score | a completed prediction folder and the separate annotation bundle | nothing it could change: it never reruns or alters a prediction |

The view manifest is the one reference-only file prediction may read: it lists membership, not answers. Prediction
recomputes membership from the scene and map, and rejects a manifest that disagrees. Prediction succeeds with the
annotations absent. Changing them can change scores, never a query or prediction byte.

## The frozen grammar

Text is matched exactly as given. Matching is case-sensitive, with single literal spaces. Nothing is trimmed, case-folded,
normalized, stripped of punctuation, singularized, repaired or expanded with synonyms. A valid command outside the grammar
is unsupported, never repaired.

- **Categories:** the 893-label `model_vocabulary` of the validated map, matched exactly, multiword labels included.
  `unknown` is not a category. A plural-looking label such as `shelves` names one object.
- **Colours,** fixed in `protocol.v1.json`: `black, blue, brown, gray, maroon, navy, orange, purple, white, yellow`.
  There's no `grey`, and no other colour.
- **Size words:** `big` and `small`, recognized but not evaluated.

A noun phrase is `[other] [SIZE] [COLOUR] CATEGORY`, with at most one size and one colour, in that order. Only
`CATEGORY` and `COLOUR CATEGORY` are supported; size and `other` are recognized for diagnosis. A sentence is wholly one
of:

```
the NP that is|are ALIAS the NP                 binary and ranking aliases
the NP that is|are ALIAS the NP and the NP      between aliases
the NP that is|are ALIAS both of the TAIL       between aliases: recognized as an unsupported plural
```

| Alias, with its preposition | Query relation | k |
|---|---|---|
| `closest to`, `nearest to` | closest | 1 |
| `second closest to`, `second nearest to` | closest | 2 |
| `third closest to`, `third nearest to` | closest | 3 |
| `farthest from`, `most distant from` | farthest | 1 |
| `second farthest from`, `second most distant from` | farthest | 2 |
| `third farthest from`, `third most distant from` | farthest | 3 |
| `below`, `under`, `beneath`, `underneath` | below | — |
| `near`, `next to`, `close to`, `adjacent to`, `beside` | near | — |
| `on` | on | — |
| `between`, `in between`, `in the middle of` | between | — |

These are operational aliases for this sample's language, not a claim that their meanings are interchangeable. The
existing libraries' `above`, `inside` and directional relations aren't added here.

**Enumeration.** Every complete analysis is enumerated:
- every boundary between target and anchors;
- every reading of each noun phrase: an exact label such as `white chair`, and `white` + `chair`;
- every alias.

Words inside a multiword label, such as the `and` in `mortar and pestle`, aren't delimiters by themselves. Identical
analyses are merged, and nothing else is chosen between. The decision order is fixed:

| Order | Condition | parse_status, parse_reason | Query reason |
|---|---|---|---|
| 1 | no complete analysis | unsupported, `unrecognized_text` | `unsupported_composition` |
| 2 | several distinct analyses | unsupported, `ambiguous_parse` | `unsupported_composition` |
| 3 | one, with `both of the` | unsupported, `plural_reference` | `unsupported_plural` |
| 4 | one, with `other` | unsupported, `coreference` | `unsupported_coreference` |
| 5 | one, with a size word | unsupported, `size_comparison` | `unsupported_size_comparison` |
| 6 | otherwise | parsed, null | — |

**Features** are the sorted set of `coreference`, `plural_reference` and `size_comparison` that the analyses contain,
taking their union when the parse is ambiguous. They're empty when nothing parsed. A parse record is closed:
`format_version`, `record_type: iref_parse`, `parent_command_id`, `text`, `parse_status`, `parse_reason`, `features`,
`analysis_count` (after merging) and `interpretation`.

## Queries and the protocol action

Every command, supported or not, gets a grounding-query v1 record and goes through the resolver. An unsupported
interpretation is `{interpretation_id: main, kind: unsupported, action: INSPECT, reason}`, and it is never turned into
`no_match`.

A parsed interpretation has:
- **Nodes:** root `target`, then `anchor_a` and, for between, `anchor_b`, in textual order.
- **Categories:** every one explicit.
- **Colours:** `colours_all` is empty or holds the one parsed colour.
- **Anchors:** no constraints and a null rank.
- **The target:** a ranking target has `rank = {relation, anchor: anchor_a, k}`; any other target has one constraint with
  a null frame.

No object ID is ever supplied. `INSPECT` is a fixed interface placeholder: not parsed, not a dataset label, not scored,
never executed. Its origin, `evaluation_protocol`, is recorded in the prediction manifest, not in the query.

## Inventory views and identities

Each view deep-copies the canonical scene and changes only `scene_id` and membership. Retained object records are
canonically byte-identical, and the frame, sources, revision and profile are kept.

| View | Scene ID (sample) | Command suffix | Keeps |
|---|---|---|---|
| `full_inventory` | `iref.scannet.scene0010_01.eval.full_inventory` | `.fi` | all 61 objects |
| `source_known_nyu` | `iref.scannet.scene0010_01.eval.source_known_nyu` | `.kn` | the 59 with a known category in the map's vocabulary |

This is an inventory policy, not a restricted evidence profile or a spatial crop. Neither view depends on text,
targets, anchors, distractors or outcomes.

The manifest must hold exactly the two views, with:
- the right parent;
- unique, disjoint lists that partition the scene;
- membership and exclusion reasons equal to the recomputed policy.

Anything else fails before the first resolver call.

**Derived records:**
- **Command contexts:** copies of the originals with the view's suffix on `command_id` and the view's `scene_id`;
  everything else is unchanged.
- **Query IDs:** the derived command ID plus `.q`.
- **Long identities:** one over 128 characters is rejected, never truncated.
- **Fixtures:** authored fixtures use their own scene ID plus `.fi` or `.kn`.

## Resolver configuration and execution

The accepted libraries are used unchanged, and their identities are checked before any work: `relations.v1@d922fe902669`
and `directions.v1@73590e939d4f`. A difference is a stop-and-report condition. Support and furniture stay `[table]` and
container `[box]`, with the 0.05 m rank tie band.

The resolver configuration is `resolver.iref.sample.v1`, vocabulary `iref.sample.nyu.colours.v1`, with the map's sorted
labels and the ten colours. Its offline allowances:

| Limit | Value |
|---|---:|
| interpretations | 1 |
| nodes | 3 |
| constraints | 1 |
| candidate checks | 10,000 |
| binding attempts | 1,000 |
| predicate calls | 10,000 |
| rank subset evaluations | 10,000 |
| trace events, trace bytes | 0 |

Each distinct text is parsed once and rebound to its parent. Each command is resolved once per view, 3,872 calls for the
sample, with `trace=False`. A budget failure is a recorded technical failure, never retried with higher limits.

## Prediction output

```
manifest.json              protocol and provenance: version, declared source, parent, action origin, views, vocabulary,
                           exact allowances, library identities, input/code hashes, runtime, semantic-hash rule
category-map.json          the validated map, unchanged
resolver-config.json       the exact expanded configuration
scenes/full_inventory.json, scenes/source_known_nyu.json
parses.jsonl               one parse record per parent command, by parent ID
command-contexts.jsonl     the derived contexts, by parent ID, then full before source-known
predictions.jsonl          one closed iref_prediction per command and view: parent, view, IDs, the query,
                           the complete resolution_run record
prediction-summary.json    totals, parser coverage, reasons and features, per-view statuses, budgets, work,
                           the semantic hash; nothing about answers
```

- **Format:** JSON is UTF-8 with sorted keys and a final LF. JSONL is one compact record per line.
- **Provenance:** source provenance copied from the import is labelled `declared`. Inputs are hashed as file bytes from
  the command line, and as canonical JSON in the Python API, each labelled. The command set is hashed as a sorted list of
  command ID plus file SHA-256.

**Determinism.** The resolver's elapsed times (`resolution.diagnostics.timings_s`) are kept as recorded. The semantic
hash removes exactly that field from a deep copy of each record and hashes the canonical JSONL stream. Every other field
is compared, including work counts, warnings, configurations and results.

Within one runtime and version, two runs are byte-identical apart from those timings:
- parses;
- derived scenes, contexts and configurations;
- all semantic counts.

The timings are laptop times; no headset latency may be inferred from them.

Identical bytes are not promised across platforms. The A2.2b Windows run recorded a different semantic hash from the
assistant's sandbox (`f0bbb5d0…` against `692c9a4d…`) with every outcome count identical. The cause is unexplained,
and the difference doesn't block the increment (D76).

## Scoring

Scoring first checks:
- **The prediction output:**
  - every record's shape, before any field is read;
  - every identity;
  - exactly one command context per parent and view, none extra, missing or duplicated, each with exactly the parse's
    text;
  - new records' versions, which must be the integer 1;
  - the semantic hash and summary, recomputed and never trusted (D76).
- **The annotations:** the schema; unique IDs and indices; one mapped target and one relation label per command;
  targets present in both views; exactly the predicted population.

Errors name the command, annotation or object.

| Term | Meaning, per view |
|---|---|
| N | every distinct command text (1,936) |
| P | commands the parser parsed |
| R | completed, `resolved` results |
| C | resolved results whose `target_id` is the mapped source target |

The four ratios are parser coverage P/N, resolved fraction R/N, all-command agreement C/N, and agreement among resolved
commands C/R. Each keeps its numerator and denominator, and C/R is null when R is 0.

An ambiguous result listing the target is not a correct selection, and `no_match` on this positive population is never
correct. Unsupported commands and technical failures stay in N.

**Bins.** Seven mutually exclusive outcomes sum to N in each view:
- five semantic statuses: `resolved`, `ambiguous`, `no_match`, `insufficient_information` and `unsupported`;
- two technical failures: `invalid_input` and `budget_exceeded`.

**Breakdowns:**
- by source relation label, by the presence of source size words, and by parser status;
- a paired transition table, full rows by source-known columns;
- selection changes between the views;
- annotation multiplicity, with the 12 repeated groups each counted once;
- parser-detected against source-labelled size words.

Reason codes and features are nonexclusive. There is no annotation-weighted headline and no average of the two views.

The output is `scores.jsonl` (one closed `iref_score` per command), `summary.json`, `report.md` (rendered from the
summary) and a scoring `manifest.json`. That manifest records both the semantic hash and the actual file hashes; across
otherwise equivalent reruns only the file hashes differ.

## Command line and errors

```
python -m grounding.evaluation.iref_vla predict --scene SCENE --commands COMMAND_DIR --category-map MAP
    --inventory-views VIEWS --relation-config REL --direction-config DIR --out NEW_DIR
python -m grounding.evaluation.iref_vla score --predictions PREDICTION_DIR --annotations ANNOTATIONS --out NEW_DIR
```

The prediction command line is limited to the A2.2a sample. The Python API also accepts authored fixtures.

| Exit | Meaning |
|---|---|
| 0 | complete; ordinary unsupported, ambiguous, insufficient or no_match results allowed |
| 1 | complete, but some records are resolver `invalid_input` or `budget_exceeded`; every command still represented |
| 2 | invalid batch input or reference data, or an existing destination; nothing published |
| 3 | an unexpected exception (traceback printed) or a write failure; nothing published |

Codes: `E_EVAL_INPUT`, `E_EVAL_VIEWS`, `E_EVAL_IDENTITY`, `E_EVAL_LIBRARY`, `E_EVAL_PROTOCOL`, `E_EVAL_PARSE`,
`E_EVAL_PREDICTIONS`, `E_EVAL_REFERENCE`, `E_EVAL_OUTPUT_EXISTS`, `E_EVAL_OUTPUT_IO`.

Each operation publishes through a new sibling folder and one final rename, and refuses to overwrite.

## Records and schema

`schemas/iref-evaluation.v1.json` holds closed definitions for the protocol, parse, prediction and score records. Their
version, count and limit fields must be plain integers, so `1.0` is rejected. Embedded queries and resolution runs are
checked by their accepted schemas and validators. The prediction manifest, both summaries and the scoring manifest are
checked as closed records in code.

## Tests

```
python grounding/tests/test_iref_vla_evaluation.py --sample DIR    full acceptance (fixtures, then the real sample)
python grounding/tests/test_iref_vla_evaluation.py --unit-only     fixtures only; reported as NOT sample acceptance
```

- **The pinned files:** `DIR` holds A2.2a's five pinned files; `SECOND_EYES_IREF_VLA_SAMPLE` may name it instead.
- **Without them:** the sample check fails unless `--unit-only` is given.
- **What the sample run does:** it re-imports the files through A2.2a's pin-verified adapter, then predicts and scores
  all 1,936 commands, which takes several minutes.
- **The expectations:** `grounding/tests/fixtures/iref_vla_evaluation/` holds them, fixed before implementation, with
  the hand-authored scenes F0, F1 and their variants.

## Choices where the brief was silent

1. **Parser input types:** text of another type is a `TypeError`, never converted with `str()`. Empty text is a
   `ValueError`, matching the command contract's rule.
2. **Object order:** derived view scenes list objects in object-ID order, so reordered input gives identical
   predictions. A2.2a's scene is already in that order.
3. **The parent link:** `parses.jsonl` and `predictions.jsonl` record `parent_command_id`. `command-contexts.jsonl` holds
   plain derived contract records, because the command record is closed.
4. **Selection changes** are reported as four counts: resolved in both views with the same target or with a different
   one, and resolved in one view only.
5. **The plural tail** is any nonempty text after `both of the`, read literally.
6. **Input hashes** are file bytes from the command line, and canonical JSON from the API, each labelled.
7. **Error codes:** the evaluation's own `E_EVAL_*` codes, listed above.
8. **Inputs:**
   - only annotated scenes are evaluated;
   - a commands folder may hold only `<command_id>.json` files;
   - the protocol file stores the aliases alongside the colours.

## Deferred

These belong to later approved increments:
- full datasets;
- false-statement negatives;
- size semantics;
- action inference;
- sub-scenes;
- model baselines and training;
- headset work;
- a new gate.
