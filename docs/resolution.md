# The offline resolver (A2.1e, D71)

`grounding.resolution` finds the scene objects that a structured grounding query describes. It executes the query's
complete alternative readings over every object of one validated scene, with the accepted relation and direction
libraries (`docs/relations.md`, `docs/directions.md`), and returns one `resolution_run` record.

It reads no English, no command text and no model probabilities. Its inputs are structured requests that someone
or something has already written. It is the semantic executor for the structured-query and structured-rule baselines.
It does not wrap the direct-scoring design: candidate scoring, model-side non-selection, action phrases and frame
plausibility are separate work.

## Calling it

```python
from grounding.resolution import resolve

record = resolve(scene, command, query,
                 resolver_config=config,
                 relation_config_path="grounding/relations/relations.v1.json",
                 direction_config_path="grounding/relations/directions.v1.json",
                 category_maps=[category_map],   # optional; see "Category maps"
                 trace=False)                    # True adds bounded detailed events
```

All inputs are decoded JSON records. The resolver evaluates private deep copies and never changes the caller's
records. It loads both relation configurations fresh through their accepted loaders for every call. It takes no
target ID, annotation, distractor list or scored shortlist: query nodes describe objects, and the resolver finds them.

| `processing_status` | Meaning | `result` |
|---|---|---|
| `completed` | evaluation finished; the result may be resolved, ambiguous, no_match, insufficient_information or unsupported | present |
| `invalid_input` | an input failed validation; `issues` says which | `null` |
| `budget_exceeded` | a caller limit would have been exceeded; `budget` says which | `null` |

Unexpected errors are not a fourth status: they raise. For example, two different scenes under one scene key make the
accepted geometry cache raise `ValueError`, and the resolver lets it propagate rather than clearing the cache.

## The query record

`schemas/grounding-query.v1.json`, JSON Schema 2020-12. Every object is closed, every listed key is required, and
nullable keys are present with `null`.

```json
{"schema_version": 1, "record_type": "grounding_query", "query_id": "fx.r08.query",
 "scene_id": "fixture.resolver.r", "scene_revision": 0, "evidence_profile": "annotated",
 "command_id": "fx.r.command", "interpretations": [...]}
```

The scene and command identities must match the supplied records. `query_id`, `scene_id` and `command_id` use the
contract's ID rule, and the revision and profile use the contract's definitions, copied unchanged from
`schemas/scene.v1.json`.

**Interpretations** are complete alternative readings of one request, never an OR to choose from. Each has a unique
`interpretation_id` (`^[a-z][a-z0-9_]{0,63}$`) and is one of three kinds:

| `kind` | Fields | Evaluates to |
|---|---|---|
| `query` | `action` (`GO_TO` or `INSPECT`), `root`, `nodes` | its branches' outcomes |
| `unavailable` | `action` or `null`, `reason`: `frame_interpretation_unavailable` | insufficient_information |
| `unsupported` | `action` or `null`, `reason`: one of ten `unsupported_*` codes | unsupported |

**A node** is a local reference variable, not an object identity: `{"node_id", "category", "colours_all",
"constraints", "rank"}`. Node IDs follow the interpretation-ID pattern, are unique within their interpretation and
must not begin `obj_`. `category` is a label or `null` (no category filter). `colours_all` holds unique labels, all
required. Labels match exactly and case-sensitively, with no trimming or fuzzy matching. A label with leading or
trailing whitespace is invalid input.

**A constraint** is `{"relation", "frame", "anchors"}`, and only these combinations are valid:

| relation | frame | anchors |
|---|---|---|
| `near`, `above`, `below`, `on`, `inside` | `null` | one node ID |
| `between` | `null` | two different node IDs |
| `left`, `right`, `in_front_of`, `behind` | `user_heading` | none |
| `left`, `right`, `in_front_of`, `behind` | `user_to_anchor` or `object_intrinsic` | one node ID |

A directional clause with no frame interpretation is an `unavailable` reading, never a direction with frame `null`.

**A rank** is `null` or `{"relation": "closest" | "farthest", "anchor": node ID, "k": 1 | 2 | 3}`. It applies after
the node's filters and constraints. `k` must be written as a JSON integer. JSON Schema's integer type admits a
whole-valued float such as `1.0`, but the resolver rejects it, with `E_QUERY_SCHEMA` at that `k`'s path in the input
(D72).

**Graph rules.** Dependencies are constraint anchors and the rank anchor. A query graph is rejected with
`E_QUERY_GRAPH` for any of these:
- a repeated or reserved node ID;
- a missing root or anchor;
- a self-dependency or a cycle;
- a node not reachable from the root.

These are rejected as structural errors (`E_QUERY_SCHEMA`):
- a repeated interpretation ID;
- a repeated constraint (compared after `between`'s anchor pair is sorted);
- a repeated colour;
- a wrong frame or arity;
- a Boolean `k`, or a `k` written as a float (D72);
- any unknown field, including answer fields.

Every ID and label is also checked as a full-string match. Since D73, the query's ID and label patterns reject a
final newline themselves, as the contract's do. The check stays as a safeguard. It still matters for the query's
`local_id` and the resolver configuration, whose patterns end in `$`, which Python's `re` also matches before a final
newline.

**Normalization.** Interpretations are sorted by ID and colours are sorted. Constraints are evaluated in the order of
their compact JSON `[relation, frame, anchors]`, with `between`'s anchors sorted and directed arguments never
reordered. Nodes are evaluated in a topological order: dependencies first, then the lexicographically smallest ready
node. The root comes last. Traversal is iterative and no depth limit is hard-coded; caller limits bound size and
work.

## The resolver configuration

`schemas/resolver-config.v1.json`: `config_id`, `vocabulary_id`, the permitted `category_labels` and `colour_labels`,
and nine required limits (nonnegative integers, never Booleans or floats, with no unlimited sentinel). JSON Schema's
integer type admits a whole-valued float such as `10000.0`, `0.0` or `1e4`. The resolver rejects one in any limit,
with `E_RESOLVER_CONFIG` at `$.limits.<name>`, and never converts or rounds it (D72). A query label outside
the vocabulary makes that reading `unsupported` (`unsupported_category_label` or `unsupported_colour_label`), not
`no_match`. A permitted label with no matching object gives an ordinary `no_match`. The fixture configuration's values
are unit-test settings, not deployment defaults or measured Quest allowances.

## Evaluation

**The universe** is every supplied object, in ascending object-ID order: no cropping and no pruning by score. A
sub-scene is a separate, validated scene the caller chose.

**One candidate, one node, one binding environment.** One candidate check is charged before the object is examined.
The category is compared with the known `category.value.model`, then the requested colours with the known colour set.
Every requested colour is required, so a known empty set fails a nonempty request. An unknown required value is
UNKNOWN. A FALSE attribute ends the check before any relation call. Otherwise the constraints run in normalized order
through the accepted predicates. They combine with the accepted three-valued `AND`: a FALSE ends the conjunction, but
an UNKNOWN never does, because a later FALSE still excludes the candidate. Matches are definite (D, TRUE) or possible
(P, UNKNOWN). The resolver infers nothing: no colour from a category, no direction from a box's yaw, no heading from a
quaternion, and no value from being the only candidate.

**Branches.** Each query interpretation keeps an explicit depth-first stack of binding environments, starting empty,
and visits alternatives in ascending object-ID order. A non-root node:

- with any possible match ends its branch as insufficient_information (`uncertain_anchor`) without enumerating
  completions: early termination is deliberately conservative;
- with no match ends as no_match (`anchor_not_found`);
- otherwise forks one environment per definite match: several anchors are alternatives, not an immediate ambiguity.

A ranked non-root node binds its resolved winner, forks over a tie, ends as no_match (`anchor_not_found`,
`too_few_rank_candidates`), or ends as insufficient_information (`uncertain_anchor` plus the rank's reasons).

One binding attempt is charged when each assignment is made, not when forks are queued. A variable keeps its object
through every later use. Different variables may bind the same object, except `between`'s two anchors. When both are
bound to one object, the environment is pruned as an illegal binding: it is counted, and it is not a no_match reading.
An interpretation whose every route is illegal ends as no_match (`no_legal_anchor_binding`), never as vacuous
consensus. A target equal to an anchor is left to the accepted predicates' repeated-object FALSE.

**The root**, after its dependencies:

| definite | possible | outcome |
|---|---|---|
| 1 | 0 | resolved |
| 0 | 0 | no_match (`target_not_found`) |
| 2 or more | any | ambiguous (`multiple_definite_matches`) |
| 0 or 1 | 1 or more | insufficient_information |

**Ranking** calls the accepted `Relations.rank(relation, anchor, candidates=D, possible=P, k=k)`. The library
excludes the anchor, ranks by 3D centre distance, links ties and evaluates possible eligibility jointly. There is no
second ranking implementation. Before the call, with `m` possible objects other than the anchor, the resolver checks
that `2 ** m` fits the remaining subset allowance by comparing `m` with the allowance's bit length. It never builds the
power or the subsets. If the bound does not fit, the result is `budget_exceeded` before `rank()` is called, with
`next_required` = remaining + 1, a bounded witness rather than the subset count. An admitted call is charged its
actual `combinations_checked`. A missing centre anywhere makes the library return insufficient after 0 subsets.

**Aggregation** over every terminal outcome of every reading, in this order:

1. any unsupported: unsupported;
2. otherwise any insufficient_information: insufficient_information;
3. otherwise all resolved with one target and one action: resolved;
4. otherwise all no_match: no_match;
5. otherwise: ambiguous.

The action is reported only when every outcome has the same non-null action. Different actions therefore prevent
`resolved`, and all-no_match readings with different actions remain no_match with action `null`.

## Work limits and caches

| Limit | Counts | Checked |
|---|---|---|
| `max_interpretations`, `max_nodes`, `max_constraints` | the request: every reading; nodes and constraints of query readings (ranks are not constraints) | after validation, before any relation work; `used` 0, `next_required` the full count |
| `max_candidate_checks` | each object visited per node and environment, even if its attributes reject it | before the visit |
| `max_binding_attempts` | each non-root assignment, including pruned ones | before the assignment |
| `max_predicate_calls` | each actual non-ranking predicate or direction call; cache hits are free | before the call |
| `max_rank_subset_evaluations` | each rank call's actual subsets, after the preflight above | before the call |
| `max_trace_events`, `max_trace_bytes` | diagnostic trace only | see "Trace" |

Equality with a limit is allowed and zero allowances are valid. A failure reports which limit, its value, the work
used so far and the next amount needed. It never returns a target, and the resolver never switches algorithm or
drops candidates to stay within a limit.

Results are memoized per call only. A predicate's key is its relation, frame and object arguments; only `near`'s pair
and `between`'s anchors are sorted, and directed arguments keep their order. A rank's key is its relation, `k`,
anchor, and the sorted definite and possible objects after anchor exclusion. A cache hit reuses its entry's consulted
evidence. Nothing survives the call.

## The result

| `status` | `reason_code` |
|---|---|
| resolved | `unique_target_consensus` |
| ambiguous | `nonunique_target_or_interpretation` |
| no_match | `no_matching_reference` |
| insufficient_information | `required_information_unavailable` |
| unsupported | `unsupported_interpretation` |

`target_id` is set only for `resolved`, and then `action` is not null. `reason_codes` is the sorted union of every
outcome's detail codes, plus `divergent_targets`, `divergent_actions` and `resolved_vs_no_match` where they apply, and
`conditional_evidence` when the result is conditional. Detail codes for UNKNOWN library results come from the
library's own reason strings:

| Library reason | Detail code |
|---|---|
| `missing_centre:*`, `missing_size:*`, `missing_rotation:*`, `missing_semantic_front:*` | `missing_centre`, `missing_size`, `missing_rotation`, `missing_semantic_front` |
| `missing_user_position:*`, `missing_heading:*` | `missing_pose` |
| `degenerate_viewer_anchor`, `vertical_semantic_front` | `degenerate_frame` |
| `boundary:*` | `boundary_relation` |
| any other cause (`degenerate_anchors`, `degenerate_footprint`, `support_not_represented:*`, ...) | `relation_unknown` |
| rank: `possible_competitors:*`, `tie`, `too_few_candidates` | `possible_rank_competitors`, `rank_tie`, `too_few_rank_candidates` |

Unknown attributes give `unknown_category` or `unknown_colour`. No missing-field cause is invented, and intrinsic
relations never report pose.

**Candidates** are unions over the evaluated branches, not sets true in every reading:

- `target_ids`: definite non-ranked root matches, and rank winners or tie groups;
- `uncertain_match_ids`: the possible matches of non-ranked roots;
- `rank_definite_eligible_ids` and `rank_possible_eligible_ids`: a ranked root's eligibility populations after
  anchor exclusion, including those of insufficient rank results. They are not claimed winners.

Anchor objects stay in the trace, not in these arrays. `coverage` is `incomplete_due_to_unresolved_reference` when any
branch ended at an uncertain or insufficiently ranked anchor, or any reading was unavailable or unsupported.
Otherwise it is `evaluated_branch_union`.

**Assumptions** are the scoped references actually consulted by completed evaluations, including those that excluded
competitors. Each is `{"record_type": "scene" | "command_context", "record_id": [scene_id, "revision", profile] or
[command_id], "assumption_id"}`, sorted and unique, so equal IDs in the scene and the command stay apart. `conditional`
is true when any consulted known value has assumptions or `assumed` evidence. This is conservative provenance, not a
confidence.

**Diagnostics** hold:

- the evaluator version (`resolver.v1`) and the context;
- canonical SHA-256 hashes of the scene, command, query and resolver-configuration snapshots, and the two accepted
  configuration identities;
- the validator's warnings;
- thirteen work counters and the outcome counts;
- timings for validation (including copying and configuration loading), evaluation and result validation;
- the trace.

Timings are excluded from deterministic comparisons. Before returning, every record is checked against
`schemas/grounding-result.v1.json` and the invariants the schema cannot express. A violation raises.

**Trace.** `trace=True` adds closed events in evaluation order:

| Event | When |
|---|---|
| `node` | after a node's evaluation; its definite matches, classified by the root table |
| `binding` | each assignment |
| `prune` | each illegal `between` binding |
| `predicate` | each actual call, with its arguments in role order and the library's reason strings |
| `rank` | each actual rank call |
| `terminal` | each outcome |

Cache hits are counted but not re-traced. Events stop, and `trace_truncated` becomes true, when the next event would
pass `max_trace_events` or `max_trace_bytes`, measured as the sum of the events' compact UTF-8 JSON lengths.
Evaluation carries on unchanged. With `trace=False` no event data is built.

## Category maps

D70's rule, applied locally:

- **No maps supplied:** valid, and the validator's `W_NOT_CROSS_CHECKED` warning is kept.
- **Maps supplied without the scene's map:** `invalid_input` with `E_CATEGORY_MAP_MISMATCH`.
- **The scene's map supplied:** it is checked by the contract, and other valid maps may accompany it.

## Command line

```text
python -m grounding.resolution --scene PATH --command PATH --query PATH --resolver-config PATH
    --relation-config PATH --direction-config PATH [--category-map PATH ...] [--trace] --out DIR
```

It parses every file strictly (repeated keys, non-finite numbers and byte-order marks are refused) and writes one
`DIR/resolution.json` in UTF-8 with a final LF. It never overwrites one, never reads the command text, and has no
execute or network option.

| Exit | Meaning |
|---|---|
| 0 | completed, whatever the semantic status |
| 1 | `budget_exceeded` (the record is written) |
| 2 | `invalid_input`, including parse and file errors (the record is written), or a refusal to overwrite |
| 3 | an unexpected exception (traceback printed, no file), or a failure to write the file (no file is left) |

## Error codes

| Code | Raised for |
|---|---|
| `E_QUERY_SCHEMA` | a structural query error, including unknown or answer fields, repeated interpretations, constraints or colours, and a rank `k` written as a float |
| `E_QUERY_GRAPH` | a repeated, missing or reserved node ID, an invalid dependency, a cycle or an unreachable node |
| `E_QUERY_CONTEXT` | a query, scene or command identity mismatch, including a command for another scene ID |
| `E_RESOLVER_CONFIG` | a malformed resolver configuration or option, including a limit written as a float |
| `E_CONFIG` | a relation or direction configuration that fails to load |
| `E_INPUT_FILE` | a file the CLI cannot read |
| `E_RECORD_ROLE` | a record in the wrong argument slot (the serializer's existing code) |
| contract and parser codes | unchanged, for example `E_SCENE_MISMATCH`, `E_FRAME_MISMATCH`, `E_CATEGORY_UNMAPPED`, `E_DUPLICATE_RECORD`, `E_CATEGORY_MAP_MISMATCH`, `E_PARSE_DUPLICATE_KEY` |

An `unsupported` reading is a valid query and completes; it is never invalid input.

## Choices made where the brief left a detail open

These are documented, flagged in D71 for confirmation, and pinned by tests.

1. A record in the wrong argument slot reuses the serializer's `E_RECORD_ROLE`.
2. `coverage` is incomplete only for anchor-level insufficiency, unavailable readings and unsupported readings. An
   anchor with no match, or a route with only illegal bindings, is a definite outcome.
3. Rank eligibility arrays are reported for the root only, after anchor exclusion.
4. Library UNKNOWN causes with no specific detail code map to `relation_unknown`, including `between`'s
   `degenerate_anchors`.
5. Repeated constraints are judged after `between`'s anchor pair is sorted.
6. Configuration labels follow the contract's label rule, which is stricter than "unique nonempty strings".
7. The CLI exits 3 when it cannot write `resolution.json`, and removes any partial file.
8. The category is checked before the colours, so a known category mismatch means the colours are not consulted.
9. A `node` trace event's outcome classifies the node's own matches by the root table. It is not the request's result.

## Running the tests

```text
python grounding/tests/test_resolution.py
python -m grounding.tests.test_resolution
```

The fixtures and their expectations (`grounding/tests/fixtures/resolution/`, see its `README.md`) were fixed before
the resolver existed. `notes/phases/A2.1e_resolution.md` records their hashes and the test evidence.

## Limits

- Hand-authored queries resolving correctly is not English grounding accuracy, and a supplied set of readings is not
  verified linguistic completeness.
- Bounded Python work counts are not headset feasibility or a CPU model; they say nothing about the Quest.
- Early anchor termination can return insufficient_information where a global proof could establish no_match.
- Centre-distance ranking and the libraries' conditional geometry are unchanged, and their limits carry over.
- The shared result envelope does not require the direct-scoring design to use query parsing.
