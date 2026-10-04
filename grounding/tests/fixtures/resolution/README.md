# Resolver fixtures (A2.1e, D71)

Fixed before the resolver existed, except where marked. `notes/phases/A2.1e_resolution.md` records their SHA-256
hashes; `.gitattributes` keeps their bytes exact.

| File | Content |
|---|---|
| `scene.r.json` | fixture R, the brief's section 12.1 table: seven objects in scene `fixture.resolver.r` |
| `command.r.json` | its command: authored pose at (-2, -2, 1.6), heading (0, 1); the text is a label, never parsed |
| `resolver-config.fixture.json` | the brief's section 4 configuration; generous unit-test limits, not deployment values |
| `cases.json` | 110 cases with their literal queries, variants and expected outcomes |
| `dispatch.json` | the 90 reviewed directional calls and 33 relation cases checked through queries |
| `query.r08.json` | added after the first run as a command-line example, not an expectation: R08's literal query over `scene.r.json` |

## Case families

- **R01-R36:** the brief's section 12.2 table, one case each; statuses and IDs are the brief's.
- **V01-V42:** section 12.3 validation: invalid queries (V01-V30) and records, maps and configurations (V31-V42).
- **B01a-B07d:** section 12.3 work limits; `base_case` names the case whose query is reused.
- **M01a-M05b:** order invariance, caches, trace, aggregation and command text or pose.
- **P01-P04:** provenance on accepted fixtures (`records`) and one valid variant.

Each R-fixture case gets its own IDs: scene `fixture.resolver.<id>`, command `fx.<id>.command`, query
`fx.<id>.query`. So no evidence-changing variant reuses a geometry-cache key. P cases keep the accepted records' IDs.

## Variant operations (applied by the test to a copy of fixture R)

| Key | Effect |
|---|---|
| `keep` | keep only the listed objects (a caller-chosen sub-scene; `[]` gives an empty scene) |
| `objects.<id>.colours` | a known colour set, or `"unknown"` for an unknown wrapper (`not_in_source`, source `fx`) |
| `objects.<id>.colours_evidence` | replace the known colour set's evidence record |
| `objects.<id>.category` | `"unknown"`: an unknown wrapper, keeping the source reference |
| `objects.<id>.front`, `.centre` | a known value, or `"unknown"` |
| `scene_assumptions` | assumption records added to the scene |
| `reverse_objects` | the same objects in reverse order |
| `command` | `heading: "unknown"`, `position`, `text`, `scene_id` or `frame_id` |
| `category_maps` | `omit`, `other_only`, `incompatible`, `matching_plus_other` or `duplicate` (default: the scene's map) |
| `limits`, `resolver_config` | configuration overrides (`limits.<name>` for one limit) |
| `relation_config` | `"missing"`: a configuration path that does not exist |
| `envelope_patch` | top-level query fields replaced as given |

## Expectation keys

`processing_status` always. For completed runs: `status`, `target_id`, `action`, `coverage` and `conditional` exactly;
the four candidate arrays exactly when given; `reason_codes_include` and `assumptions_include` as subsets;
`assumptions` exactly; `work` and `outcome_counts` exactly for the counters named; `warning_codes` exactly. For
invalid input, `issue_codes_include`. For budget failures, the `budget` record exactly. `same_as` compares a case's
semantic result and work counters with another case's; `same_trace` also compares their traces.

## Dispatch rule

The expected TRUE, FALSE or UNKNOWN comes from the reviewed tables (`../directions/cases.json`,
`../relations/cases.json`), never from the code under test. The test builds a query whose root has the target's
category and the call's constraint, with one anchor node per anchor object, over a sub-scene of the call's objects. It
then finds the call's predicate event in the trace. `between`'s anchors are compared as a set. After the first run,
`near` was also compared as a set: it is symmetric and memoized canonically, so only the first of `near(a, b)` and
`near(b, a)` is traced. Only that sentence of `dispatch.json` changed; its call lists did not.
