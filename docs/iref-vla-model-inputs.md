# IRef-VLA model inputs and token measurement (A2.2d)

A2.2d (D78) turns A2.2c's fitting selection plans into validated scene and command records, renders each through both
accepted serializers, wraps each document in a fixed sizing wrapper and measures it with the pinned Qwen tokenizer. It is
the bridge from imported data to A2.3's model comparison: reproducible model-input artifacts and their token costs. It
does not run a model, choose a format, create training labels, measure grounding accuracy or establish headset feasibility.

## Running it

The measurement needs its own pinned environment, kept apart from any training environment:

```
python -m venv .venv-token-audit
.venv-token-audit\Scripts\python -m pip install -r grounding\requirements-token-audit.txt
.venv-token-audit\Scripts\python -m grounding.preparation.iref_vla --scene ... --commands ... --category-map ...
    --inventory-views ... --selection-audit DIR --relation-config grounding\relations\relations.v1.json
    --direction-config grounding\relations\directions.v1.json
    --tokenizer-dir quest-app\Assets\SecondEyes\Models\qwen2.5-0.5b-instruct
    --model-description grounding\models\qwen2.5-0.5b-instruct.json --out NEW_FOLDER
```

Exit codes: 0, the bundle was published (input ceilings may be exceeded; those are reported outcomes); 2, invalid input;
3, an unexpected or write failure. An existing output folder is refused, never replaced, and nothing is published unless
the complete bundle passed readback. Tests: `python grounding/tests/test_iref_vla_preparation.py --unit-only` (fixtures
with labelled test doubles; not real-token acceptance), or with the pinned environment `--bundle DIR` (calibration, then
readback and the sample counts of a published bundle) or `--import DIR --selection-audit DIR` (also runs the CLI).

## Inputs, and trusting the selection audit

Exactly the four A2.2a originals (scene, commands folder, category map, inventory views), the A2.2c output folder, the
two library configurations, the three tokenizer files and the model description. No annotation, statement, graph, answer,
prediction or score is read. The audit folder must hold exactly its four artifacts. Its records are parsed strictly
(duplicate keys, truncated lines and non-objects are errors) and checked against A2.2c's closed schema; its manifest's
artifact hashes, its recorded input-file hashes (with A2.2c's command aggregate) and the D77 parent-scene hash must match
the supplied files; then the whole audit is recomputed from the originals with A2.2c's own function and compared with
type-aware equality, so a forged ID, membership, status, selection hash or row count fails even if an attacker
recomputes the audit's file hashes. All of this happens before any render or token count. The source audit's code and
runtime hashes are historical records and are not compared with the current tree.

## Population and identities

Every command/view pair gets one index row, in command order and then `full_inventory`, `source_known_nyu`. Fitting
pairs are materialized; over-limit pairs keep their required count and selection ID with no model input (never the first
ten objects); unassessed pairs keep their parser reason. An empty selection, when no requested category exists and no
unknown object remains, is materialized with zero objects; it is not an unassessed command.

A derived scene is the original annotated scene with `scene_id` set to the A2.2c selection ID, `scene_revision` 0 and
exactly the retained original object records in lexicographic ID order; everything else is unchanged, including the
coordinate frame (selecting objects does not transform coordinates), the full category map and every provenance entry.
One scene per selection ID is shared by its commands and both formats. A derived command is the original with
`command_id` = `iref.input.` + SHA-256 of the canonical JSON of `{preparation_policy, parent_command_id,
parent_command_sha256, selection_id}`, `scene_id` set to the selection, and both scene revisions 0; its text, timestamp,
pose, frame, sources and evidence are unchanged, and a missing pose stays missing. Each scene is validated with its map
and every command bound to it before rendering, and both records are checked independently against their parents.

## Rendering and the work allowance

The accepted `grounding.serialization.serialize` renders `coordinates_v2` and `coordinates_relations_v2` from the same
records, without its optional token checker (its own token status stays `not_checked`). Library identities must be
`relations.v1@d922fe902669` and `directions.v1@73590e939d4f`. The augmented work allowance is 1,190 slots, the accepted
`work_slots` formula at ten objects; coordinates need none. A work-cap failure records `work_budget_exceeded` with no
text; any other serializer failure aborts the batch. Rendered texts are stored under `diagnostic_inputs/` even when they
exceed every ceiling: they are sizing diagnostics, not accepted inference requests, and are never truncated.

## Tokenizer, wrapper and measurements

Qwen/Qwen2.5-0.5B-Instruct at revision `7ae557604adf67be50417f59c2c2f167def9a775`; `vocab.json`, `merges.txt` and
`tokenizer_config.json` must match their pinned SHA-256 values and are copied alone into a temporary folder before loading
with `AutoTokenizer` (local files only, no remote code, fast tokenizer; `Qwen2TokenizerFast` required) under transformers
4.57.6 and tokenizers 0.22.2. Before rendering, A1's four tracked prompts must count 230, 232, 230 and 227 tokens with a
191-token scene prefix (A1's own `grounding.scene` helpers), and the eight stored goldens must count as recorded in D78.
A1's raw token-ID references are not an input, so token-ID parity with A1 is reported as not claimed.

The wrapper is `<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{document}<|im_end|>\n<|im_start|>assistant\n`
with the accepted sizing audits' system message; it equals the model description's template through
`grounding.scene.formatted`. Text is tokenized as given, with no special tokens added. Label: exact for the pinned Hugging
Face tokenizer and this sizing wrapper; deployed-GGUF execution parity not established by this increment.

Each rendered input records its complete token count (one call), byte count, SHA-256 and token-ID digest; the document's
own count; static, dynamic and document hashes; work and per-block T/F/U counts from the serializer; and its result at
each input-only ceiling of 1,024, 2,048 and 4,096 tokens (equality passes). Prefix reuse compares token IDs: the wrapper
head plus the static prefix is tokenized, its longest common ID prefix with the complete input is recorded, and within
each (selection, format) group the minimum is the reusable prefix. These are potential reuse figures, conditional on that
selection's prefix being cached; no cross-selection hit is assumed. Per-block attribution tokenizes successive complete
prefixes (wrapper head, each document line, wrapper tail); its marginals telescope exactly to the full count and are
order-dependent, never clamped.

## Outputs

`preparation-index.jsonl`, `measurements.jsonl` (two rows per index row, in format order), `summary.json`, `report.md`,
`manifest.json`, `model_records/` (the category map, scenes and commands) and `diagnostic_inputs/` (document, static
prefix, dynamic suffix and prompt per rendered pair and format). Records follow `schemas/iref-model-input-audit.v1.json`.
The manifest records every source, audit and output hash, the pins, calibration, code hashes and runtime versions; runtime
timings and paths are not part of any record. Records and texts are deterministic across Windows and Linux.

Keep outputs outside the Git repository; the dataset and generated documents are never committed.

## Moving a bundle between machines

A bundle is an artifact: another machine uses a copy of its bytes, verified against its own manifest, and never
regenerates it and calls the result the same input. A2.2a's conversion computes quaternions and semantic fronts with
trigonometric functions, and a Linux and a Windows conversion of the same pinned sample have been seen to differ in the
last bit of three values. That changes the canonical scene hash and so every derived ID and file name, and those digits
in prompts, while the summary and report stay identical. A run on another machine is a new artifact with its own
provenance.
