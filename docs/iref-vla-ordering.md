# Crossed answer-code assignment and choices-order diagnostic (A2.3c, D94)

A development diagnostic on the A2.3a pilot. It asks whether changing the letter attached to an object, or the
position of that object in the prompt's `choices` list, changes the small model's selection. It is not training, not
a benchmark and not a Gate B measurement. Package: `grounding/inference/iref_vla_ordering/`; policy
`ordering-policy.v1.json` (`iref.order.crossed_cyclic.v1`); schema `schemas/iref-ordering.v1.json`.

## The design

For a base request with objects `O[0..n-1]` in scene order and letters `C[0..n-1]` (the first n of A..J):

```text
code assigned to object O[i]          = C[(i + a) mod n]
object at choices-list position j     = O[(j + p) mod n]
choices[j] = [C[(j + p + a) mod n], O[(j + p) mod n]]      then ["K", "ASK"], always last
```

Every pair `(a, p)` with `a, p` in `0..n-1` is one cell (zero-based; reports show positions as 1..n). `(0, 0)` is the
original request. Across the n² cells each object meets every (letter, list position) pair exactly once, so the two
factors are crossed without consulting any target. Views of the same grid: identity `(0, 0)`, assignment-only
`(a > 0, 0)`, list-order-only `(0, p > 0)`, crossed `(a > 0, p > 0)`, and all cells.

Only the choices line changes. The system message, wrapper and scene document (object order, IDs, relation tables)
are byte identical in every cell, so the design does not separate a preference for a scene position or an object ID
from correct reasoning; following one object across the grid shows invariance to these two transformations, not
correctness.

The pilot's 128 base requests give 4,536 cells: n = 3, 4, 5, 6, 7, 8, 9, 10 for 8, 22, 26, 34, 20, 14, 2, 2 bases;
3,004 cells in `full_inventory` and 1,532 in `source_known_nyu`, split equally between the two formats; 128 identity
cells (fresh repeat controls) and 4,408 others.

## Commands

```text
python -m grounding.inference.iref_vla_ordering prepare --base-requests DIR --bundle DIR --tokenizer-dir DIR
    --model-description FILE --out NEW_DIR
python -m grounding.inference.iref_vla_ordering run --requests DIR --base-pilot DIR --model-dir DIR --device cuda
    --tokenizer-dir DIR --model-description FILE --out NEW_DIR [--expected-weights-sha256 HEX]
python -m grounding.inference.iref_vla_ordering score --requests DIR --results DIR --base-pilot DIR
    --base-scores DIR --out NEW_DIR
```

`prepare` (laptop, pinned tokenizer environment, no model): verifies the A2.3a request folder with
`verify_request_dir`, its preparation bundle, the model description and the tokenizer identity; requires the pinned
selected-parent hash; rebuilds each original prompt from the bundle's document and requires byte equality; writes every
cell with the unchanged prompt builder, tokenizes it and checks every offered letter's one-token boundary (A..J are
tokens 32..41, K is 42); requires each `(0, 0)` cell to reproduce the original prompt bytes, token IDs and mapping;
checks the balance property; and compares the population with the policy's expected table. Cells over the 8,192-token
limit are recorded, never truncated: the bundle is then a verified audit (exit 1) that `run` refuses.

`run` (RTX PC, `--device cuda` only): verifies the bundle, run r003 (`verify_results`, the pinned `results.jsonl` hash,
its link to the base requests, and each identity cell against r003's completed request), the model description, the
checkpoint's file hashes against those r003 recorded, and the tokenizer; re-tokenizes every prompt. Then:

1. Canaries on the first base request's `(0, 0)`, `(1, 0)` and `(0, 1)` cells: the production forward against the
   independent last-hidden-state path, offered logits within atol = rtol = 1e-5 and the same code and object. Six model
   evaluations, counted separately from the 4,536.
2. The 128 identity cells, fresh, compared with r003: every offered logit within `1e-5 + 1e-5 * |old|`, the same code
   and the same decoded object or ASK.
3. Only if every control passes, the other 4,408 cells in the schedule's order.

The model path is A2.3a's unchanged `TorchModel`: float32, eager attention, batch 1, TF32 off, one final-position
forward with `use_cache=False`, no generation. Exact score ties go to K. A failed canary or control, an exception or an
interruption leaves `<out>.failed-<UTC time>/diagnostic.json` with the completed rows and identities and publishes no
result (exit 3). There is no automatic resume; a rerun starts with a new output folder.

`score` (laptop, no model): verifies the bundle, the results against it, r003 and the accepted A2.3b score folder
(`verify_score_dir`, the pinned summary and report hashes, and its links to r003 and the base requests); takes each
base request's source target and rules outcome from A2.3b's `scores.jsonl`; decodes every choice through its own cell's
mapping and finds the target's letter and list position in the same mapping. It never calls the parser, resolver,
relations, serializer or a model.

## Execution schedule

The identity cells run first, in base order. The rest follow by the SHA-256 of UTF-8(`second-eyes/a23c/order/v1` + LF
+ variant ID), ties by variant ID. This mixes the transformations over time without looking at any answer. Canonical
row order (base, then a, then p) does not depend on execution order. Variant IDs look like `q001.a00.p00`.

## Outputs

Request bundle: `manifest.json`, `policy.json` and `protocol.json` (copies), `request-index.jsonl` (one row per cell:
identities, n, a, p, the ordered mapping with token IDs, document, prompt and token hashes, token count, context
status, schedule position), `schedule.json`, and `variants/qNNN.jsonl` (each cell's exact prompt and token IDs).

Result folder: `results.jsonl` (canonical order; choice, decoded object, chosen list and scene positions with ASK as
null, offered logits, full-vocabulary log probabilities, restricted shares, ties, timing, execution index),
`controls.jsonl`, `canaries.json`, `summary.json`, `report.md`, `manifest.json`. No correctness is scored here.

Score folder: `scores.jsonl` (one row per cell), `commands.jsonl` (one row per base request with every subset, both
contrasts, stability and reference values), `summary.json`, `report.md`, `manifest.json`.

## Metrics

Per base request and per subset: planned, completed, object choices, ASK, unscored; source-target selections C,
C/planned and C/object choices (null for a zero denominator); the distributions of chosen letters, list positions,
objects and scene positions; agreement with the fresh identity cell (same object or ASK, same letter, and same list
position among cells where both are object choices).

One-factor contrasts: at each fixed p, `(0, p)` against `(a, p)` for a > 0 (the code assignment changed, the list order
fixed), counting object-or-ASK changes and same-letter choices; at each fixed a, `(a, 0)` against `(a, p)` for p > 0
(the list order changed, the codes fixed), counting changes and same-position choices. Rates are computed per command
first; each view and format then averages its 32 commands. Raw numerators and denominators are reported too; nothing
is pooled as if larger scenes mattered more. Full-grid agreement is the mean over commands of C/n², with the minimum
and maximum; the pooled C/cells figure is labelled cell-weighted.

Reference policies: always-B and always-second-list-position each select the source in exactly n of n² cells (1/n)
of every complete grid. A policy that always picks one scene position would keep whatever it scored in the original;
the report shows the source targets' scene positions so that this stays visible.

## Exit codes

0 a complete, technically valid result (whatever its accuracy); 1 a preparation audit with planned context
exclusions; 2 invalid input, a pin or identity mismatch, or an existing destination; 3 a failed canary or repeat
control, or an unexpected runtime or write failure. Error codes: `E_ORDER_POLICY`, `E_ORDER_PATH`, `E_ORDER_BASE`,
`E_ORDER_BUNDLE`, `E_ORDER_MODEL`, `E_ORDER_TOKENIZER`, `E_ORDER_TOKENS`, `E_ORDER_PIN`, `E_ORDER_POPULATION`,
`E_ORDER_REQUESTS`, `E_ORDER_EXCLUSIONS`, `E_ORDER_BASE_PILOT`, `E_ORDER_CHECKPOINT`, `E_ORDER_DEVICE`,
`E_ORDER_RESULTS`, `E_ORDER_BASE_SCORES`, `E_ORDER_REFERENCE`, `E_ORDER_INTERNAL`; `E_EVAL_OUTPUT_EXISTS` and
`E_EVAL_OUTPUT_IO` are the shared output codes.

## What the figures do not establish

One previously inspected development room with annotated geometry; parser-conditioned, category-complete subscenes; a
fixed scene document. No significance test, generalization interval, format winner, Gate B result, Quest measurement or
Paper 1 claim follows. Restricted shares are not calibrated probabilities. The grid is a diagnostic: voting over it is
not a deployable system, and no prompt, threshold or aggregation is tuned after seeing the answers.

## Windows line endings

The base requests pinned the model description's bytes as checked out on Windows (CRLF). Pass the laptop's
working-tree file to `prepare`, and give the RTX PC that same file for `run`; a copy with LF line endings has a
different hash and is refused (`E_ORDER_MODEL`).
