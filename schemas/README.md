# schemas

File formats that every component follows, kept in one place so the headset app and the Python code cannot drift apart. To change a format, change its file here first and log the change in `notes/decisions.md`.

| File | Format of | Since |
|---|---|---|
| `run-config.v1.json` | `runs/<run ID>/config.yaml` | S0.2 |
| `log-event.v1.json` | one line of a session log (see `docs/logging.md`) | S0.3 |
| `scene.v1.json` | one scene record (`docs/scene-contract.md`) | A2.1a |
| `command-context.v1.json` | one typed command with the user's pose (`docs/scene-contract.md`) | A2.1a |
| `category-map.v1.json` | one category map (`docs/scene-contract.md`) | A2.1a |
| `scene.v2.json` | one scene record in format v2: v1 with a dataset-identity frame (`docs/scene-contract.md`, D74) | A2.2a |
| `iref-annotations.v1.json` | the IRef-VLA adapter's reference-only annotation bundle; validated by the adapter, not the contract (`docs/iref-vla-adapter.md`) | A2.2a |
| `iref-evaluation.v1.json` | the A2.2b evaluation's protocol, parse, prediction and score records; checked by the evaluation, not the contract (`docs/iref-vla-evaluation.md`) | A2.2b |
| `iref-subscene-audit.v1.json` | the A2.2c audit's selection rows, summary and manifest; checked by the audit, not the contract (`docs/iref-vla-subscenes.md`) | A2.2c |
| `iref-model-input-audit.v1.json` | the A2.2d preparation's index rows, measurement rows, summary and manifest; checked by the preparation and its readback, not the contract (`docs/iref-vla-model-inputs.md`) | A2.2d |
| `iref-zero-shot-pilot.v1.json` | the A2.3a pilot's request and result rows and manifests; checked by the pilot's verifiers, not the contract (`docs/iref-vla-zero-shot-pilot.md`) | A2.3a |
| `relation-config.v1.json` | the relation library's thresholds, bands and category lists (`docs/relations.md`) | A2.1b |
| `direction-config.v1.json` | the directional relations' band and cutoffs (`docs/directions.md`) | A2.1c |
