# grounding

Language grounding (proposal §6, with A2 as the Revision 3.1 supplement plans it, D63): the scene contract and its validator, the relation library, the rule-based parser, and fine-tuning and evaluation of the language models, trained mainly on published datasets. Everything here runs on a PC; since D60 the headset runs the model with llama.cpp (On the headset, below).

| Path | What it is | Since |
|---|---|---|
| `models/<name>.json` | a model description: which model, its shape, chat template, length limits and the run settings of Meta's provider (`provider`), used only for A1.7's comparison since D60; Second Eyes → Fill chat provider copies them into the provider asset | A1.7b |
| `models/<name>/` | the downloaded and exported model files (ignored by Git; recreate with `export_onnx.py`) | A1.7b |
| `prompts/<id>.json` | a fixed prompt: system message, user text and the expected target. `a17-fixed` and the presets `a17-front`, `a17-left` and `a17-table` share one scene and system message and differ in their command (D49). The headset's test panel reads a copy in `quest-app/Assets/SecondEyes/Grounding/Prompts/`; `check_headset.py` compares token IDs, so a stale copy can't pass | A1.7b |
| `meta_runner.py` | a Python mirror of Meta's on-device runner: the model's input and output names and its generation loop | A1.7b |
| `export_onnx.py` | downloads a model and exports it to ONNX the way Meta's runner reads it, with one word table and the weights file below 2 GiB (D47) | A1.7b |
| `reference.py` | makes the PC reference answer that the headset must reproduce; with `--candidates`, each listed object's log-probability instead (PyTorch, 32-bit) | A1.7b, A1.8b |
| `scene.py` | what a prompt's scene and answer look like, shared by the references and the headset's jobs: the objects to score, the answer's start, where the cacheable scene ends | A1.8b |
| `export_gguf.py` | converts the downloaded model to GGUF for llama.cpp (8-bit Q8_0 by default) with llama.cpp's own converter, and fingerprints it | A1.8b |
| `llama_headset.py` | llama.cpp on the headset from the command line: prepares a job, pushes the test program and model, runs it, runs llama-bench, and checks the results against the PC references | A1.8b |
| `check_headset.py` | compares the sends in a run's logs (headset, or the editor's Play mode) with the PC reference: the prompt's token IDs, and the answer text token by token, even if the answer didn't finish | A1.7c |
| `debug_values.py` | saves onnxruntime's intermediate values for the fixed prompt, and a copy of the export that exposes them (`onnx/model_debug.onnx`), for check 7 of Second Eyes → Test model on the PC | A1.7c |
| `copy_to_unity.py` | copies a model's tokenizer files into the Unity app and writes `model.json`, which names the converted file and records the ONNX export for Second Eyes → Convert model (`docs/setup/quest-model.md`) | A1.7c |
| `contract/validate.py`, `contract/checks.py` | check scene, command-context and category-map records against the scene contract (`docs/scene-contract.md`): strict parsing, the schemas in `schemas/`, then semantic and cross-record rules; `python grounding/contract/validate.py FILE...` | A2.1a |
| `tests/test_contract.py`, `tests/fixtures/contract/` | the contract's tests and their fixtures: `python grounding/tests/test_contract.py` | A2.1a |
| `relations/geometry.py`, `relations/predicates.py` | the relation library (`docs/relations.md`): closest and farthest, near, above, below, on, inside and between over validated scene records, with three-valued results and field-level provenance | A2.1b |
| `relations/relations.v1.json` | the relation library's provisional parameters (format `schemas/relation-config.v1.json`) | A2.1b |
| `tests/test_relations.py`, `tests/fixtures/relations/` | the relation library's tests and reviewed cases: `python grounding/tests/test_relations.py` | A2.1b |
| `relations/directions.py` | the directional relations (`docs/directions.md`): left, right, in front of and behind in the user-heading, user-to-anchor and object-intrinsic frames | A2.1c |
| `relations/directions.v1.json` | their provisional band and cutoffs (format `schemas/direction-config.v1.json`) | A2.1c |
| `tests/test_directions.py`, `tests/fixtures/directions/` | their tests and the brief's fixed cases: `python grounding/tests/test_directions.py` | A2.1c |
| `serialization/` | the offline serializer (`docs/serialization.md`): `coordinates_v2` and `coordinates_relations_v2` from validated records, and its CLI, `python -m grounding.serialization` | A2.1d |
| `tests/test_serialization.py`, `tests/fixtures/serialization/` | its tests and fixed goldens: `python grounding/tests/test_serialization.py` | A2.1d |
| `resolution/` | the offline structured resolver (`docs/resolution.md`): structured grounding queries to scene-object IDs through the accepted relation libraries, and its CLI, `python -m grounding.resolution` | A2.1e |
| `tests/test_resolution.py`, `tests/fixtures/resolution/` | its tests and the cases fixed before code: `python grounding/tests/test_resolution.py` | A2.1e |
| `adapters/iref_vla/` | the IRef-VLA metadata adapter (`docs/iref-vla-adapter.md`): the pinned public ScanNet sample into a scene v2, commands and a reference-only annotation bundle, and its CLI, `python -m grounding.adapters.iref_vla` | A2.2a |
| `tests/test_iref_vla_adapter.py`, `tests/fixtures/iref_vla/` | its tests, the expectations fixed before code and hand-written source-format fixtures: `python grounding/tests/test_iref_vla_adapter.py --sample DIR` | A2.2a |
| `adapters/iref_vla/release.py` | the A2.6a multi-scene release reader (`iref_vla_release.v1`), a versioned extension that leaves the one-scene adapter and its pins unchanged: the six pinned sample scenes, header-only probes and receipted downloads of the release's source zips, validation and counts per scene, environment groups with their ancestry, A2.2c's selection as a compatibility estimate, and partition proposals (never frozen): `python -m grounding.adapters.iref_vla.release --help` | A2.6a |
| `tests/test_iref_vla_release.py` | its tests, on the pinned samples (fetch-samples writes them): `python grounding/tests/test_iref_vla_release.py --samples DIR` | A2.6a |
| `preparation/iref_vla_dev/` | the A2.6b development preparation: `freeze` (the partition, cap, sampling, desktop subset and analysis plan, hashed) and `prepare` (the accepted chain per scene in parallel, eligibility with each reason, multiple entries resolved, relation-stratified sampling with N and n, D104 requests in both formats and views, the desktop subset, a hashed manifest, answers kept reference-only): `python -m grounding.preparation.iref_vla_dev --help` | A2.6b |
| `tests/test_iref_vla_dev.py` | its tests, on a fixture release of real ScanNet list names (the pinned samples and the pinned official lists): `python grounding/tests/test_iref_vla_dev.py --samples DIR --official DIR` | A2.6b |
| `inference/iref_vla_dev/` | the A2.6c baseline execution and frozen analysis: `preflight` (read-only checks of the frozen requests), `smoke` and `run` (A2.3d's accepted runner, given the preflight's context), `rules` (one resolver result per parent and view), `desktop-export`, `desktop-run` and `desktop-compare` (the 64-request desktop Q8_0/U comparison), `score` (targets read here only) and `analyze` (the weights and the paired environment-cluster bootstrap): `python -m grounding.inference.iref_vla_dev --help` | A2.6c |
| `tests/test_iref_vla_baseline.py` | its tests: hand-calculable analysis fixtures, and a real fixture preparation through the preflight, the runner (labelled doubles), the rules, the desktop path, scoring and the analysis: `python grounding/tests/test_iref_vla_baseline.py --samples DIR --official DIR` | A2.6c |
| `evaluation/iref_vla/` | the A2.2b development evaluation (`docs/iref-vla-evaluation.md`): a declared text-only grammar, prediction through the accepted resolver in two inventory views, and separate scoring; its CLI, `python -m grounding.evaluation.iref_vla predict` and `score` | A2.2b |
| `tests/test_iref_vla_evaluation.py`, `tests/fixtures/iref_vla_evaluation/` | its tests and the cases fixed before implementation: `python grounding/tests/test_iref_vla_evaluation.py --sample DIR` (or `--unit-only`, which is not sample acceptance) | A2.2b |
| `subscenes/iref_vla/` | the A2.2c category-complete selection audit (`docs/iref-vla-subscenes.md`): selection plans per command and inventory view, from the accepted parser and validation; its CLI, `python -m grounding.subscenes.iref_vla` | A2.2c |
| `tests/test_iref_vla_subscenes.py`, `tests/fixtures/iref_vla_subscenes/` | its tests and the hand-written expectations: `python grounding/tests/test_iref_vla_subscenes.py --sample DIR` (or `--unit-only`, which is not sample acceptance) | A2.2c |
| `preparation/iref_vla/` | the A2.2d model-input preparation (`docs/iref-vla-model-inputs.md`): derived scenes and commands from A2.2c's fitting selections, both formats rendered and measured with the pinned tokenizer; its CLI, `python -m grounding.preparation.iref_vla` | A2.2d |
| `requirements-token-audit.txt` | the pinned token-measurement environment (transformers 4.57.6, tokenizers 0.22.2), installed apart from any training environment | A2.2d |
| `tests/test_iref_vla_preparation.py`, `tests/fixtures/iref_vla_preparation/` | its tests and the expectations written before implementation: `--unit-only` (test doubles; not real-token acceptance), `--bundle DIR` or `--import DIR --selection-audit DIR` | A2.2d |
| `inference/iref_vla/` | the A2.3a zero-shot direct-selection pilot (`docs/iref-vla-zero-shot-pilot.md`): `prepare` freezes exact requests from an A2.2d bundle; `run` verifies them and the checkpoint, runs the canary and one forward per request; its CLI, `python -m grounding.inference.iref_vla` | A2.3a |
| `tests/test_iref_vla_pilot.py`, `tests/fixtures/iref_vla_pilot/` | its tests and the expectations written before the scorer: `--unit-only` (fakes only), the pinned-tokenizer mode, `--requests DIR` | A2.3a |
| `evaluation/iref_vla_pilot/` | the A2.3b saved-pilot scoring and matched rules baseline (`docs/iref-vla-pilot-scoring.md`): `baseline` runs the accepted parser and resolver once per selected parent and view; `score` compares the saved model and rules results with the reference annotations; its CLI, `python -m grounding.evaluation.iref_vla_pilot` | A2.3b |
| `tests/test_iref_vla_pilot_scoring.py`, `tests/fixtures/iref_vla_pilot_scoring/` | its tests and the expectations written before the scoring layer: `--unit-only` (fixtures and labelled doubles), or `--rules DIR --scores DIR [--pilot DIR]` to read back a real run | A2.3b |
| `inference/iref_vla_ordering/` | the A2.3c crossed answer-code assignment and choices-order diagnostic (`docs/iref-vla-ordering.md`): `prepare` freezes the crossed-cyclic requests on the laptop, `run` executes them on the RTX PC after canaries and repeat controls, and `score` compares the choices with the accepted A2.3b targets; its CLI, `python -m grounding.inference.iref_vla_ordering` | A2.3c |
| `tests/test_iref_vla_ordering.py` | its focused checks: the crossed cyclic design, its metrics, the file chain, the gates and the boundaries: `python grounding/tests/test_iref_vla_ordering.py` | A2.3c |
| `models/acquire.py` | downloads a pinned Hugging Face model's files and proves they are the pinned ones (size, SHA-256, commit and ETag): `python -m grounding.models.acquire --description FILE --out DIR [--only tokenizer]` | A2.3d |
| `tests/test_model_acquire.py` | its checks, with a labelled fake downloader and no network: `python grounding/tests/test_model_acquire.py` | A2.3d |
| `inference/iref_vla_compare/` | the A2.3d comparison of rules, Qwen2.5-0.5B-Instruct and Qwen2.5-7B-Instruct, and the A2.3e checks (`docs/iref-vla-compare.md`): `prepare` and `rules` on the laptop, `smoke` and `run` on the RTX PC, `score` on the laptop; then `audit`, `costs` and `costs-regenerate` on the laptop and `cache` on the RTX PC; its CLI, `python -m grounding.inference.iref_vla_compare` | A2.3d, A2.3e |
| `tests/test_iref_vla_compare.py` | its tests, on A2.3b's fixture chain with labelled tokenizer and model doubles: `python grounding/tests/test_iref_vla_compare.py` | A2.3d, A2.3e |
| `quest/` | the PC side of A2.5's Quest integration (`docs/quest-replay.md`, `docs/quest-runtime-identity.md`): `replay-bundle` freezes the replay requests, their float32 references and the written integration fixtures, and `verify-replay-bundle` reads a bundle back; `replay_inputs.py` is the reference for the headset's input checks; `runtime-identity` checks the installed app's native library and model file with read-only adb; `replay-push` and `replay-pull` move the headset inputs to the app and the replay's results back; `replay-compare` makes the five D101 comparisons per request against the frozen bundle; `replay-desktop` runs the desktop diagnosis with the host build; `replay-repeat` and `repeat-compare` run and analyse the repeatability and context-history diagnostic (ChatGPT's proposal); `golden-build`, `golden-verify` and `golden-references` make delivery 2's goldens, and `golden-push` and `golden-pull` move them to the app and the results back; `interactive-pull` brings delivery 3's interactive-check results; `inbox-send` and `outbox-pull` are delivery 4's ADB inbox transport; `inbox-batch` and `measure-report` are delivery 5's measurement; `batch-diagnostic` and `batch-compare` run delivery 1's bounded batch-arithmetic diagnostic; `inbox-status` and `acceptance-check` serve the A2.5 correction acceptance; `provenance-collect` gathers the conversion and configuration provenance, read-only; its CLI, `python -m grounding.quest` | A2.5 |
| `tests/test_quest_replay_bundle.py` | its tests, on A2.3b's fixture chain with labelled doubles: `python grounding/tests/test_quest_replay_bundle.py [--tokenizer-dir DIR]` (the pinned-tokenizer mode is not acceptance) | A2.5 |
| `tests/test_quest_runtime_identity.py` | its tests for the identity check, with hand-built ARM64 ELF and GGUF files and a labelled fake adb: `python grounding/tests/test_quest_runtime_identity.py` | A2.5 |
| `tests/test_quest_replay_device.py` | its tests for replay-push and replay-pull, with a fixture bundle and a labelled fake device: `python grounding/tests/test_quest_replay_device.py` | A2.5 |
| `tests/test_quest_replay_compare.py` | its tests for replay-compare and replay-desktop, with a fixture bundle and a labelled fake native library: `python grounding/tests/test_quest_replay_compare.py` | A2.5 |
| `tests/test_quest_replay_repeat.py` | its tests for the repeatability and context-history diagnostic, with a fixture bundle and labelled fake natives: `python grounding/tests/test_quest_replay_repeat.py` | A2.5 |
| `tests/test_quest_prompt_goldens.py` | its tests for delivery 2's goldens, readback, references and device helpers, on A2.2d's fixture bundle: `python grounding/tests/test_quest_prompt_goldens.py` | A2.5 |
| `tests/test_quest_provenance.py` | its tests for the read-only provenance collector, with a fixture model folder, a hand-written GGUF header and a fake r006 run: `python grounding/tests/test_quest_provenance.py` | A2.5 |
| `tests/test_quest_inbox.py` | its tests for delivery 4's ADB inbox transport, on a labelled fake device with fixture goldens: `python grounding/tests/test_quest_inbox.py` | A2.5 |
| `tests/test_quest_measure.py` | its tests for delivery 5's inbox-batch and measurement report, on a labelled fake device and a synthetic pulled session: `python grounding/tests/test_quest_measure.py` | A2.5 |
| `tests/test_quest_batch_diagnostic.py` | its tests for the bounded batch-arithmetic diagnostic, on labelled fake natives: `python grounding/tests/test_quest_batch_diagnostic.py` | A2.5 |
| `tests/test_quest_acceptance.py` | its tests for the A2.5 correction-acceptance checks of the raw records (cases A to D), on a synthetic run: `python grounding/tests/test_quest_acceptance.py` | A2.5 |

## Setup

Python 3.9 or newer, and a few GB of downloads the first time:

```
python -m pip install -r grounding/requirements.txt
```

The contract's validator, the relation library, the directional relations, the serializer, the resolver, the IRef-VLA adapter, evaluation and selection audit and their tests need only `jsonschema` from that list.

The relation modules are imported through the package (`from grounding.relations import predicates`), with the repository root on the import path; they don't change the import path themselves. Their test scripts add the root, so from the repository root both `python grounding/tests/test_relations.py` and `python -m grounding.tests.test_relations` work, and likewise for `test_directions`.

## On the headset

Since D60 the headset runs the model with llama.cpp: `export_gguf.py` makes the 8-bit GGUF from the download
`export_onnx.py` made, and `llama_headset.py push-model` copies it into the app's data folder. The ONNX export, the
`.sentis` conversion and Meta's provider asset (`copy_to_unity.py`, Second Eyes → Convert model and Fill chat provider)
stay only for comparison with Meta's runner (A1.7).

The model's revision is pinned to a Hugging Face commit in its description (D61), so a download on another PC gets the
same weights as the headset's file and the references.

To use a different model, add a description in `models/` and export it; the tools read everything model-specific from
there.
