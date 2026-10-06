# IRef-VLA pilot scoring and the matched rules baseline (A2.3b)

A2.3b (D82) asks, on A2.3a's exact commands and frozen subscenes: which saved model choices match the published target,
how the two formats compare, and how the accepted rules system behaves. It is offline, CPU-only analysis of saved
outputs. It runs no language model, retokenizes nothing, rebuilds no prompt or subscene and changes no choice.

## Commands

```
python -m grounding.evaluation.iref_vla_pilot baseline --requests REQUESTS --bundle A22D_BUNDLE --relation-config grounding/relations/relations.v1.json --direction-config grounding/relations/directions.v1.json --out NEW_RULES
python -m grounding.evaluation.iref_vla_pilot score --pilot PILOT_RESULTS --requests REQUESTS --bundle A22D_BUNDLE --rules RULES --annotations ANNOTATIONS --out NEW_SCORES
```

Exit 0: published and technically complete, whatever the agreement. 1: published, but the analysis contains technical
failures (resolver `invalid_input` or `budget_exceeded`) or planned exclusions. 2: invalid input, a failed identity or
reference check, or an existing destination; nothing is published. 3: an unexpected or write failure; nothing is
published. Low agreement is a result, never an exit code. Issues print as `path: code: reason` under four codes:
`E_SCORING_INPUT` (malformed files, shapes, versions, plain integers, finite numbers), `E_SCORING_INTEGRITY` (stale or
disagreeing hashes, links, counts, order, mappings or saved decisions), `E_SCORING_REFERENCE` (the annotations' schema
or pinned identity, a missing or contradictory annotation, an absent target) and `E_SCORING_INTERNAL` (an output failed
its own readback). An existing destination keeps A2.2b's `E_EVAL_OUTPUT_EXISTS`. Both commands need only Python and
jsonschema. They read every input without changing it, refuse outputs that overlap an input, write into a temporary
sibling folder, read it back and only then rename it into place.

The command line evaluates only the pinned sample. Fixture mode, used by the tests, exists only as a function argument
and is labelled in every output. Tests: `python grounding/tests/test_iref_vla_pilot_scoring.py --unit-only`, or add
`--rules DIR --scores DIR --pilot DIR` to read back a real run.

## The baseline

For each selected parent, in selection order, and each inventory view, the A2.2d derived scene and command are used
unchanged, after A2.2d's bundle verifier, A2.3a's request verifier and the joins below. A2.2b's parser runs once per
exact text with the category map's whole vocabulary and A2.2b's colours. The query is A2.2b's: the parse's single
interpretation bound to the derived scene, its revision and evidence profile and the derived command, with the query
ID `<derived command ID>.q`. A2.2b's resolver configuration and limits are used unchanged, with `trace=False`. That is
one resolver call per parent and view (64 for the pilot); both formats share it. An unsupported parse stops the run:
the selection assumed parser support. Resolver outcomes, including technical failures, are kept whole. The command
accepts no annotation or model output.

## What scoring checks before it compares

- The pilot results pass A2.3a's result verifier; their manifest names exactly these requests (manifest hash), the
  same protocol, selection and source bundle.
- The requests pass A2.3a's request verifier, and for the pinned sample were prepared with the checked-in pilot
  protocol. The bundle passes A2.2d's verifier and is the one the requests name (manifest, artifacts, aggregates).
- Every result is its request: the same parent, view, format, derived IDs, document, prompt and token-ID hashes, token
  count and complete alias mapping, in canonical request order.
- Every parent and view joins through the preparation index: record hashes, IDs, revision 0, annotated evidence,
  ordered object IDs, category map and the shared contract. Both formats share one scene, command and mapping. The two
  views carry one identical text; for the pinned sample, the parent ID is the adapter's hash of that exact text, and
  the records name the pinned source, commit and statement file.
- Every saved decision is rechecked against its own offered scores (tolerance 1e-10): restricted shares, one shared
  normalizer for the log-probabilities, the largest offered logit, an exact maximal tie giving K even when K is not
  tied, the margin and the top share. A context exclusion carrying a choice is rejected: it never becomes ASK.
- The saved rules are read back in full and must name these requests, this bundle and exactly these parents and views.
- The whole annotation bundle is validated (schema, plain integers, unique IDs, the pinned scene, source, commit and
  statement hash) before it is filtered to the selected commands. Each needs at least one annotation, unique source
  indices, one mapped target and one source relation label, and its target must be present and offered in all four
  requests. Unselected annotations are not missing predictions.

## Measures

For each view and format: planned N, completed, object choices, ASK, planned exclusions, technical failures, the source
target matches C, and C/N, C/completed and C/object choices. A model request selects the source target exactly when
its completed object choice is the command's mapped target; ASK is false for this single-target measure, which does
not show that ASK was inappropriate. Unscored rows stay in N with a null value. Because C/object choices can rise by
abstaining, C/N stays beside it. Ratios keep their numerator and denominator; a zero denominator gives null.

The rules once per view: N, each resolver status, technical failures, resolved R, C, C/N and C/R. The rules select the
source target only when the resolver completed with `resolved` on it. Ambiguous, insufficient and no-match results may
be justified under our policy even though the source names one target.

Paired comparisons, per view: the two formats (both, coordinates only, augmented only, neither, unscored; same and
different saved choices; augmented minus coordinates in points of C/N), each format against the rules (with the rules'
not-selecting outcomes by status), and a cross-tab of model outcome against rules outcome, with same object, different
object and ASK where the rules resolved. An ASK against a non-resolved rules result is not called a correct
clarification.

Choice concentration, per view and format: chosen and source-target aliases, a source-by-chosen table, the always-B
diagnostic (agreement over N, with the number of requests offering B), the most chosen objects, and choices of objects
whose category is unknown in the scene record. Always B is a post-observation diagnostic on this sample, not a
competitive baseline or a trained rule. Breakdowns by source relation label and by subscene object count always state
N. The sample is 32 paired commands from one room, not 128 independent examples: no significance test, confidence
bound, format winner or Gate B decision is computed.

## Outputs

Rules folder: `parses.jsonl` (one A2.2b parse record per parent), `rules.jsonl` (one record per parent and view: the
complete query and resolution, the outcome, target, candidates and reasons), `summary.json`, `manifest.json` (input
identities, the protocol, configuration and library identities, code hashes, runtime). Its semantic hash removes
exactly `resolution.diagnostics.timings_s` from each record.

Score folder: `scores.jsonl` (one row per request, in request order), `comparisons.jsonl` (one row per parent and
view, with flags for format disagreements, unknown-category choices, unscored members and rules technical failures),
`summary.json` and `report.md`, both derived only from those rows, and `manifest.json` (input hashes, the reference
checks performed, code hashes, runtime). `summary.json` and `report.md` contain no derived IDs or hashes; `scores.jsonl`
and `comparisons.jsonl` do, so they differ between machines whose bundles differ. The report ends with a review table
of every parent and view.

## What the figures do not establish

This is one previously inspected IRef-VLA development room with annotated geometry. Its commands were selected through
the existing rules parser and the category-complete subscene policy, with a fixed alias order. Source agreement is a
development diagnostic, not held-out natural-language grounding accuracy. Conservative unknown handling and the
dataset's relation definitions may differ. No viewpoint-dependent language, actual perception noise or broad ASK
behaviour is tested. Restricted shares are not calibrated correctness probabilities. Nothing here reruns inference,
decides training, supports a deployment claim, changes a Paper 1 claim or concludes Gate B.
