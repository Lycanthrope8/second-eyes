# docs

How things work now. Edited in place when something changes; log the change and its reason in `notes/decisions.md`.

- `conventions.md`: where things go and the rules
- `logging.md`: the event log format
- `meta-ai.md`: facts from Meta's on-device AI docs that the project relies on
- `profiling.md`: how the cost of a build is measured on the headset
- `scene-contract.md`: the scene, command-context and category-map records, and how they are checked
- `relations.md`: the relation library: what each relation means and how it is decided
- `directions.md`: the directional relations, their three frames and their boundary rule
- `serialization.md`: the offline serializer: both model-input formats, their exact layout, and how to call it
- `resolution.md`: the offline structured resolver: the query and result records, the fixed evaluation, its limits and how to call it
- `iref-vla-adapter.md`: the IRef-VLA metadata adapter: getting the pinned sample, the CLI, what an import writes and what it keeps apart
- `iref-vla-evaluation.md`: the text-only rules baseline on the pinned sample: the frozen grammar, the two inventory views, prediction kept apart from scoring, and the metrics
- `iref-vla-subscenes.md`: the category-complete selection audit: the policy, its rows and summary, what it reads and what it never claims
- `iref-vla-model-inputs.md`: the A2.2d preparation: materialized scenes and commands, both formats rendered and measured with the pinned tokenizer, and moving a bundle between machines
- `iref-vla-zero-shot-pilot.md`: the A2.3a zero-shot direct-selection pilot: the one-token choice interface, frozen requests, checkpoint identity, the canary and what the results do not establish
- `iref-vla-pilot-scoring.md`: the A2.3b scoring of the saved pilot against the source annotations, with the matched rules baseline: the two commands, their input checks, the metrics, the exit codes and what the figures do not establish
- `iref-vla-ordering.md`: the A2.3c crossed code-assignment and choices-order diagnostic: the design, the three commands (prepare, run, score), the canary and repeat-control gate, the metrics, the exit codes and what the figures do not establish
- `iref-vla-compare.md`: the A2.3d comparison of rules, a 0.5B and a 7B model: the model pins and acquisition, the selection and the frozen letter and list-order policy, and the prepare and rules commands
- `quest-replay.md`: A2.5's frozen-request replay (delivery 1): the fixture and reference bundle built on the PC, the headset's ordered input checks and written fixtures, the headset replay (paths U, R and P, the startup-log capture, the results files), the push and pull helpers, the D101 comparison (step 3), the desktop diagnosis, the read-only provenance collector and the bounded batch-arithmetic diagnostic
- `quest-runtime-identity.md`: A2.5's runtime identity check: the installed app's native library and model file against the build record and the accepted A1 artifacts, the two verdicts, the evidence and the command
- `quest-prompting.md`: A2.5 delivery 2: the headset's prompt builder, the PC's goldens (snapshots, prompt asset, token IDs, float32 references), the on-device golden check and the commands
- `quest-interactive.md`: A2.5 delivery 3: the interactive pipeline on the headset (one request type for every intake, the exact-prefix cache as a switch, structured outcomes), the Interactive check and `interactive-pull`; delivery 4's session (ADB inbox, presets, cancel, `inbox-send`, `outbox-pull`); delivery 5's measurement (`inbox-batch`, `measure-report`)
- `capture-record.md`: the capture record a scan writes on the headset, and its check (A4)
- `frames.md`: frames, transforms and units from A4 on (O3): the one place where handedness changes, and the pinhole model
- `quest-panel.md`: the headset panel: its two pages (main and More) and where each control is
- `detector.md`: the on-device object detector (A1.10b, A1.10c): the replaceable detector package, the PC reference and parity rule, the Unity setup and the headset checks
- `iref-vla-development.md`: the A2.6 modules: the release reader (with A2.4's training mode), the development preparation, the baseline execution and analysis
- `a24-training.md`: A2.4: the semantic language and constrained decoder, labels, preparation, training, the fine-tuned identity and evaluation
- `setup/quest3.md`: the headset's exact state and how to undo every change
- `setup/quest-model.md`: how a language model gets into the Quest app
- `plan/revision-3.1-supplement.md`: the Revision 3.1 supplement's text, verbatim (D63); unlike the rest of `docs/`, never edited in place (`conventions.md`, rule 11)
- `localization.md`: how a scan becomes a perceived scene (A4): the steps, the evidence each field carries, the thresholds, the checks and the known limits, and the PC rerun
- `auto-scan.md`: the automatic scan (A4): on entering the app, the detector names objects and Meta's depth places them; settings, files, the PC replay, checks and limits
- `scan.md`: the stand-in scan (A4): keyframes from spots, no longer installed on the headset, kept for the drone's camera-only scan
- `perception-folder.md`: the perception folder (A4): what the headset's scan and scanned rooms need, built, verified and pushed from the PC, loaded and checked on the headset
- `a4-usability.md`: the A4 usability check (D122): the room, the scan, the commands and what counts
