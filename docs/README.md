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
- `setup/quest3.md`: the headset's exact state and how to undo every change
- `setup/quest-model.md`: how a language model gets into the Quest app
- `plan/revision-3.1-supplement.md`: the Revision 3.1 supplement's text, verbatim (D63); unlike the rest of `docs/`, never edited in place (`conventions.md`, rule 11)
