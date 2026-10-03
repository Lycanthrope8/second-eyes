# The offline serializer (A2.1d, D69)

`grounding.serialization` renders a validated scene record and a compatible command-context record into one of two
model inputs:

| Format | Content |
|---|---|
| `coordinates_v2` | header, semantics, the object rows, the user's pose, the command |
| `coordinates_relations_v2` | the same, plus every relation state and centre distance in the indexed layout approved after the A2.1d audits |

Both formats remain candidates for A2.3's zero-shot screening. The serializer evaluates the accepted relation libraries
(`docs/relations.md`, `docs/directions.md`) on the records it is given. It never reads previously rendered documents.

The eight documents of the accepted indexed audit, plus two one-object documents from the brief, are its fixed
regression expectations (`grounding/tests/fixtures/serialization/`).

It is offline Python only. There is no resolver, language parsing, candidate filtering, action selection, ASK
policy, dataset adapter, model, tokenizer, cache, C# port or headset code.

## Calling it

```python
from grounding.serialization import serialize, TokenCheck

result = serialize(scene, command,
                   format="coordinates_relations_v2",
                   relation_config_path="grounding/relations/relations.v1.json",
                   direction_config_path="grounding/relations/directions.v1.json",
                   max_relation_work_units=1190,   # required for coordinates_relations_v2
                   category_maps=[category_map],   # optional; used only to validate
                   token_check=None)               # optional; see below
result.status        # "ok", "work_budget_exceeded" or "token_budget_exceeded"
result.document      # == result.static_prefix + result.dynamic_suffix when ok; all three None otherwise
result.metadata      # provenance, counts, hashes, evidence, timings
```

The records are decoded JSON objects of the A2.1a contract (`docs/scene-contract.md`). Nothing answer-bearing can be
passed: no target, anchor, frame, benchmark label or candidate list. A command carrying such a field fails validation.

**The boundary, in order.**
1. **Options.** The format must be one of the two names. The work cap must be a nonnegative integer for the augmented
   format, and `None` or a nonnegative integer for coordinates. A token check must be well formed.
2. **Records.** Private deep copies of the records are taken and validated together by the contract validator.
   - Its errors raise `SerializationInputError`. Each issue keeps its label, path, code and message.
   - Its warnings go to `metadata["validation"]["warnings"]`.
   - The scene and command must agree on scene ID, revision and frame, even for coordinates. The validator itself only
     warns when the scene IDs differ, so the serializer reports that as `E_SCENE_MISMATCH`.
   - A missing category map keeps the validator's `W_NOT_CROSS_CHECKED` warning. A supplied map with the scene's map ID
     that conflicts with it is an error.
   - Category maps never fill or replace a value.
3. **Configurations.** Both configuration files are loaded with the accepted loaders, for this call only. The same
   loaded values feed the semantics line, the computations and the metadata. A file that won't load, or that changes
   while it loads, raises `SerializationInputError` (`E_CONFIG`).
4. **Work.** The augmented work is counted from n, before any relation table, box, geometry lookup, tuple enumeration
   or evaluator exists. A cap below it returns `work_budget_exceeded` with no document. A cap equal to it passes.
5. **Rendering.** The document is rendered, then optionally token-checked.

Errors are never answers. A budget failure or an input error is not a no-match or an ASK. There is no fallback format
and no truncated document. An unexpected exception fails the call; nothing turns a library error into UNKNOWN.
Neither validation nor rendering changes the caller's records.

## The model text

UTF-8 JSON lines: compact, keys in a fixed order, each line ending in one LF, the last included. Line order:

| # | Line | Formats |
|---|---|---|
| 1 | header | both |
| 2 | semantics | both |
| 3 | objects | both |
| 4 | `near`, `above`, `below`, `on`, `inside`, `between`, intrinsic `right`, intrinsic `in_front_of`, `center_distance_m` | augmented |
| 5 | pose | both |
| 6 | user-heading `right` and `in_front_of`, user-to-anchor `right` and `in_front_of` | augmented |
| 7 | command | both |

`static_prefix` ends immediately before the pose line, and every viewer-dependent block comes after it. A stable prefix
allows later cache reuse, but doesn't establish it. Nothing is cached now.

**Header and semantics.** The header text is fixed per format (`grounding/serialization/constants.py`, copied from the
brief's Appendix A). The semantics line keeps its definitions verbatim. Its parameters and categories come from the
loaded configuration files on every call:
- thresholds, bands and direction values, each group sorted by key;
- the categories support, furniture and container, each sorted.

Configuration identities stay in metadata.

**Objects.** All objects are kept, sorted by complete object ID in lexicographic order. Indices in the derived blocks
are positions in that array, never numeric ID suffixes: `obj_001`, `obj_1000`, `obj_999` is indices 0, 1, 2. The
columns are `id, category, colours, center_m, size_m, rotation_xyzw, rotation_support, semantic_front,
conditional_fields`.
- **category** is the known `model` category or null; a raw or standard label never stands in.
- **colours** is a sorted list if known, `[]` if known empty, null if unknown.
- **Geometry** values appear as known, or null.
- **rotation_support** is the support of the known rotation wrapper. It belongs to that wrapper, so it is not named
  separately in `conditional_fields`.
- **semantic_front** comes only from its own field, never from the box rotation.
- **conditional_fields** names, in column order, each known wrapper that is assumed or carries assumptions.

**Pose.** `pose_kind, position_m, heading_xy, heading_source, conditional_fields`. `heading_source` is null unless the
heading is known. The pose rotation, timestamps and provenance stay out of the text.

**Derived blocks.** These use the indexed layout of the A2.1d indexed audit; the header's coverage legend states it
for the model:
- **Relation blocks** have the keys `relation, frame, domain, states, conditional`.
- **The distance block** has the keys `measure, domain, values, conditional`.
- **Domains:** `objects` (one character per target), `ordered_pairs` (target rows, anchor columns, `-` on the
  diagonal), `unordered_pairs` (upper-triangle rows) and `target_anchor_pairs` (`[a,b,row]` for every a<b).
- **Uniform shortcuts:** a nonempty uniform domain is one symbol, and a uniform membership is a JSON Boolean.
- **Empty domains** are `states: []` with `conditional: false`.
- **Distances** are exact numeric upper-triangle rows with null for unknown. There is no uniform shortcut for them.
- **Mirrors:** only `right` and `in_front_of` are stored. `left` and `behind` are their mirrors over legal cells: T and
  F exchanged, U kept, membership kept.
- **`[]`** means known-empty colours in object rows, and an empty domain in relation states.

**Numbers and text.**
- Integers are written in decimal.
- Integral finite floats use the equal integer, and both zeros are written `0`.
- Other floats use Python's shortest round-trip `repr`.
- Non-finite values are rejected.
- Geometry and distances are never rounded.
- The command text is kept exactly as decoded: JSON escaping only, no trimming, no Unicode normalization.

**Not model text.** Scene, profile, revision, command and source identities, observation times, confidence, view
histories and raw labels.

## Relation construction and work

Every predicate comes from `grounding.relations`, through its package imports, with one shared `Truth`. The one
computation done here is the approved centre distance, `norm(sub(center_b, center_a))`.

| Block | Domain | Work slots |
|---|---|---:|
| near | unordered distinct pairs | n(n−1)/2 |
| above, below, on, inside | ordered distinct pairs | 4n(n−1) |
| between | target, unordered pair of other anchors | n(n−1)(n−2)/2 |
| intrinsic right, in_front_of | ordered distinct pairs | 2n(n−1) |
| centre distance | unordered distinct pairs | n(n−1)/2 |
| user-heading right, in_front_of | each target | 2n |
| user-to-anchor right, in_front_of | ordered distinct pairs | 2n(n−1) |

The total is `9n(n−1) + n(n−1)(n−2)/2 + 2n`: 0, 2, 22, 63, 342 and 1,190 at n = 0, 1, 2, 3, 6 and 10. Distance slots
count even when a missing centre leaves the distance null.

Coordinates rendering does no relation work at all: no evaluator, box, tuple enumeration or geometry-cache lookup.
Rendering never calls ranking or eligibility enumeration, and never selects a frame from the command. Every frame is
provided as evidence.

A cell's conditional flag comes from the inputs its predicate actually consulted. It is true when a consulted known
input is assumed or carries assumptions, on FALSE and UNKNOWN cells too. A distance consults its two centre wrappers
only.

## Token check (optional)

```python
TokenCheck(wrap,               # document -> complete model input (task instructions, chat template, prefix)
           count_tokens,       # complete input -> nonnegative int, counted in one call
           max_input_tokens,   # already excludes any output or scoring reserve the caller keeps
           tokenizer_identity, wrapper_identity,
           measurement_kind)   # "exact" or "test_double"
```

**The callbacks.** They see only the finished document; they can't supply evidence or change the document. A non-string
wrapper output, a Boolean, negative or non-integer count, or a callback exception raises `TokenCheckError`. The budget
is never assumed to pass.

**Results.**
- **No check:** `token_count` null and `token_budget_status` `not_checked`.
- **Count equal to the limit:** passes.
- **Count above the limit:** returns `token_budget_exceeded`, with the count and limit in metadata and no document.

A `test_double` count is labelled as such; it cannot certify model fit. Tokenizers and models are never dependencies
of the serializer.

## Metadata

Ordinary dictionaries, written by the CLI as `metadata.json`:
- **Identity:** format and version; scene ID, revision, profile, frame and category-map reference; command ID; warnings.
- **Snapshot hashes:** SHA-256 of the private record copies, over JSON with sorted keys, compact separators, UTF-8 and
  finite numbers. These are provenance, not a cache key.
- **Configurations:** both configuration identities, their file SHA-256 values, and the effective thresholds, bands,
  categories and direction values.
- **Work and counts:** the ordered object IDs, n, planned work slots by block, actual calls, distances performed and
  missing, state counts and conditional counts.
- **Text:** byte counts and SHA-256 of the prefix, suffix and document.
- **Token check:** its fields, the work cap with required work and status.
- **Per-tuple evidence:** block, frame, object IDs, state or distance, conditional flag, input references with their
  consulted assumptions, and the predicates' reasons, basis and measures.
  - Scene references carry the scene key (scene ID, revision, profile); command references carry the command ID.
  - Assumption IDs are scoped by record, so equal names in a scene and a command stay distinct.
  - Registries are kept by reference to the snapshot hash, not copied.
- **Timings:** validation and configuration loading, static build, dynamic build and token check, on a monotonic
  clock. These are diagnostics, outside any equality or hash expectation.

## Command line

```
python -m grounding.serialization --scene PATH --command PATH --format coordinates_v2|coordinates_relations_v2
    --relation-config PATH --direction-config PATH [--category-map PATH ...]
    [--max-relation-work-units INTEGER] --out DIR
```

The records are parsed strictly by the contract's parser. The CLI never chooses a tokenizer, so its metadata always
says `not_checked`.

| Outcome | Files written | Exit |
|---|---|---|
| Success | `document.jsonl`, `static_prefix.jsonl`, `dynamic_suffix.jsonl`, `metadata.json` | 0 |
| Work-budget failure | `metadata.json` only | 1 |
| Rejected options, records or configuration, issues printed | none | 2 |
| `--out` already holds any of the four files | none, and the old files are untouched | 2 |
| Unexpected internal error, traceback printed | none | 3 |

## Error codes

The contract validator's codes pass through unchanged (`docs/scene-contract.md`). The serializer adds its own:

| Code | Meaning |
|---|---|
| `E_OPTION_FORMAT`, `E_OPTION_WORK_CAP`, `E_OPTION_TOKEN_CHECK`, `E_OPTION_CATEGORY_MAPS`, `E_OPTION_CONFIG_PATH` | an option is unsupported or malformed |
| `E_RECORD_ROLE` | a scene, command or category-map argument holds another record type |
| `E_SCENE_MISMATCH`, `E_FRAME_MISMATCH` | the existing codes, also raised when the scene IDs differ, which the validator alone only warns about |
| `E_CONFIG` | a configuration file is missing, invalid or fails its schema |
| `E_INPUT_FILE` | the CLI couldn't read a record file |

## Running the tests

```
python grounding/tests/test_serialization.py
python -m grounding.tests.test_serialization
```

Acceptance also needs the contract, relation and direction suites, with relations and directions module-style too
(see `notes/phases/A2.1d_serialization.md`).

## Limits

- **The two formats are candidates.** Byte-exact rendering says nothing about model comprehension, dataset semantics
  or Quest cost.
- **Model-fit claims need a real tokenizer.** Deployed-GGUF tokenizer parity and the non-NFC difference are still
  open.
- **Commands needing excluded attributes are unsupported.** Times, confidence or source identities are not model text,
  so no command can use them.
- **Per-tuple evidence makes metadata large.** Measured as `metadata.json`: 0.66 MB for a17 (n = 6, 342 entries)
  and 11.7 MB for scene.h (n = 19, 6,023 entries). Callers keep what they need.
- **Open questions flagged in D69:** whether a supplied category map with a different map ID should be an error rather
  than the validator's warning, and the CLI's exit code 3 for internal errors.
